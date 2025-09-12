# engine_processor.py
import os
import re
import socket
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import HTTPException

from engine_runtime import get_http_client
from engine_common import (
    POLYGON_API_KEY,
    IBKR_ENABLED, IBKR_DEFAULT_QTY, IBKR_TIF, IBKR_ORDER_MODE, IBKR_USE_MID_AS_LIMIT,
    SCAN_MIN_VOL_RTH, SCAN_MIN_OI_RTH, SCAN_MIN_VOL_AH, SCAN_MIN_OI_AH,
    SEND_CHAIN_SCAN_ALERTS, SEND_CHAIN_SCAN_TOPN_ALERTS, REPLACE_IF_NO_NBBO,
    market_now, consume_llm,
    parse_alert_text,
    _is_rth_now, _occ_meta, _ticker_matches_side, _encode_ticker_path,
    _build_plus_minus_contracts, round_strike_to_common_increment,
    preflight_ok, compose_telegram_text,
)
from polygon_ops import (
    _http_json, _http_get_any, _pull_nbbo_direct, _probe_nbbo_verbose,
    _poly_reference_contracts_exists, _rescan_best_replacement, _find_nbbo_replacement_same_expiry,
)
from market_ops import (
    polygon_get_option_snapshot_export,
    poly_option_backfill,
    scan_for_best_contract_for_alert,
    scan_top_candidates_for_alert,
    ensure_nbbo,
)

from ibkr_client import place_recommended_option_order
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG

logger = logging.getLogger("trading_engine")

# =========================
# Recommendation helper
# =========================
async def _recommend_plusminus_from_top5(
    symbol: str,
    side: str,
    underlying_px: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    Pick +5% strike (CALL) or -5% strike (PUT) from the top-5 most liquid
    (ranked by OI then Vol) contracts of the given expiry and side.
    """
    client = get_http_client()
    if client is None:
        return None

    target_strike = round_strike_to_common_increment(
        underlying_px * (1.05 if side == "CALL" else 0.95)
    )

    # Pull a decent pool, then filter to side+expiry and rank by liquidity
    try:
        try:
            pool = await scan_top_candidates_for_alert(
                client,
                symbol,
                {"side": side, "symbol": symbol, "strike": target_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=30,
                restrict_expiries=[expiry_iso],  # type: ignore
            ) or []
        except TypeError:
            pool = await scan_top_candidates_for_alert(
                client,
                symbol,
                {"side": side, "symbol": symbol, "strike": target_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=40,
            ) or []
            pool = [it for it in pool if str(it.get("expiry")) == str(expiry_iso)]
    except Exception:
        pool = []

    # enforce side + valid strike
    pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side) and isinstance(it.get("strike"), (int, float))]

    if not pool:
        return None

    # rank by liquidity (OI desc, Vol desc), then take top 5
    def liq_key(it: Dict[str, Any]):
        return (-(it.get("oi") or 0), -(it.get("vol") or 0))
    pool.sort(key=liq_key)
    top5 = pool[:5]

    # choose within top5 the strike closest to target
    rec = min(top5, key=lambda it: abs(float(it["strike"]) - target_strike))

    # normalize fields
    occ = _occ_meta(rec.get("ticker") or "")
    rec_exp = occ["expiry"] if occ else str(rec.get("expiry") or expiry_iso)
    out = {
        "ticker": rec["ticker"],
        "strike": float(rec["strike"]),
        "expiry": rec_exp,
        "oi": rec.get("oi"),
        "vol": rec.get("vol"),
        "bid": rec.get("bid"),
        "ask": rec.get("ask"),
        "spread_pct": rec.get("spread_pct"),
        "target_strike": target_strike,
        "pool_count": len(pool),
        "considered_topn": len(top5),
    }
    return out

# =========================
# Core processing (called by runtime worker)
# =========================
async def process_tradingview_job(job: Dict[str, Any]) -> None:
    client = get_http_client()
    if client is None:
        logger.warning("[worker] HTTP client not ready")
        return

    selection_debug: Dict[str, Any] = {}
    replacement_note: Optional[Dict[str, Any]] = None
    option_ticker: Optional[str] = None
    recommendation: Optional[Dict[str, Any]] = None

    # 1) Parse
    try:
        alert = parse_alert_text(job["alert_text"])
        logger.info("parsed alert: side=%s symbol=%s strike=%s expiry=%s",
                    alert.get("side"), alert.get("symbol"), alert.get("strike"), alert.get("expiry"))
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    side = alert["side"]
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy  = bool(job["flags"].get("force_buy", False))
    qty        = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    # Preserve originals for messaging
    orig_strike = alert.get("strike")
    orig_expiry = alert.get("expiry")

    # 2) Expiry defaulting
    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    def _next_friday(d): return d + timedelta(days=(4 - d.weekday()) % 7)
    def same_week_friday(d): return (d - timedelta(days=d.weekday())) + timedelta(days=4)
    target_expiry_date = _next_friday(today_utc) + timedelta(days=7)
    swf = same_week_friday(today_utc)
    if (target_expiry_date - timedelta(days=target_expiry_date.weekday())) == (swf - timedelta(days=swf.weekday())):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]

    # 3) Chain scan thresholds
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) Selection via scan (strictly honor side)
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            client,
            alert["symbol"],
            {"side": side, "symbol": alert["symbol"], "strike": alert.get("strike"), "expiry": alert.get("expiry")},
            min_vol=scan_min_vol, min_oi=scan_min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
        )
    except Exception:
        best_from_scan = None

    candidate = None
    if best_from_scan and _ticker_matches_side(best_from_scan.get("ticker"), side):
        candidate = best_from_scan
    else:
        try:
            pool = await scan_top_candidates_for_alert(
                client,
                alert["symbol"],
                {"side": side, "symbol": alert["symbol"], "strike": alert.get("strike"), "expiry": alert.get("expiry")},
                min_vol=scan_min_vol, min_oi=scan_min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=6,
            ) or []
        except Exception:
            pool = []
        pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side)]
        candidate = pool[0] if pool else None

    if candidate:
        option_ticker = candidate["ticker"]
        if isinstance(candidate.get("strike"), (int, float)):
            desired_strike = float(candidate["strike"])
        occ = _occ_meta(option_ticker)
        chosen_expiry = occ["expiry"] if occ and occ.get("expiry") else str(candidate.get("expiry") or orig_expiry or target_expiry)
        selection_debug = {"selected_by": "chain_scan", "selected_ticker": option_ticker,
                           "best_item": candidate, "chosen_expiry": chosen_expiry}
        logger.info("chain_scan selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)
    else:
        fallback_exp = str(orig_expiry or target_expiry)
        option_ticker = pm["contract_call"] if side == "CALL" else pm["contract_put"]
        desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]
        chosen_expiry = fallback_exp
        selection_debug = {"selected_by": "
