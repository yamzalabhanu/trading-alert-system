# engine_processor.py
import os
import re
import math
import socket
import logging
from typing import Dict, Any, Optional, List, Tuple
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
    _build_plus_minus_contracts,
    preflight_ok, compose_telegram_text,
)
from polygon_ops import (
    _http_json, _http_get_any, _pull_nbbo_direct, _probe_nbbo_verbose,
    _poly_reference_contracts_exists, _rescan_best_replacement, _find_nbbo_replacement_same_expiry,
)

from ibkr_client import place_recommended_option_order
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG
from market_ops import (
    polygon_get_option_snapshot_export,
    poly_option_backfill,
    scan_for_best_contract_for_alert,
    scan_top_candidates_for_alert,
    ensure_nbbo,
)

logger = logging.getLogger("trading_engine")


# =============================================================================
# Synthetic NBBO helpers
# =============================================================================
def _spread_pct_heuristic(f: Dict[str, Any], rth: Optional[bool]) -> float:
    env_override = os.getenv("FALLBACK_SYNTH_SPREAD_PCT")
    if env_override:
        try:
            return float(env_override)
        except Exception:
            pass

    oi  = float(f.get("oi") or 0)
    vol = float(f.get("vol") or 0)
    delta = f.get("delta")
    mark_hint = f.get("mid", f.get("last", f.get("prev_close", 0.0)))

    # base by liquidity
    if oi >= 5000 or vol >= 5000: base = 3.0
    elif oi >= 1000 or vol >= 1000: base = 5.0
    elif oi >= 200 or vol >= 200: base = 8.0
    else: base = 12.0

    if isinstance(delta, (int, float)) and abs(delta) < 0.10:
        base += 2.0
    if isinstance(mark_hint, (int, float)) and mark_hint < 0.10:
        base += 5.0
    if rth is False:
        base *= 1.5

    return float(min(base, 25.0))


def _ensure_mid(f: Dict[str, Any]) -> Optional[float]:
    if isinstance(f.get("mid"), (int, float)):
        return float(f["mid"])
    if isinstance(f.get("last"), (int, float)):
        f["mid"] = float(f["last"])
        return f["mid"]
    if isinstance(f.get("prev_close"), (int, float)):
        f["mid"] = float(f["prev_close"])
        return f["mid"]
    return None


def synthesize_nbbo_if_missing(f: Dict[str, Any], rth: Optional[bool]) -> bool:
    has_nbbo = isinstance(f.get("bid"), (int, float)) and isinstance(f.get("ask"), (int, float))
    if has_nbbo:
        return False

    mid = _ensure_mid(f)
    if not isinstance(mid, (int, float)) or mid <= 0:
        return False

    spread_pct = _spread_pct_heuristic(f, rth)
    abs_spread = (spread_pct / 100.0) * mid
    half = abs_spread / 2.0

    bid = max(0.01, mid - half)
    ask = max(bid + 0.01, mid + half)

    f["bid"] = round(float(bid), 4)
    f["ask"] = round(float(ask), 4)
    f["option_spread_pct"] = round(float((ask - bid) / max(0.0001, (ask + bid) / 2.0) * 100.0), 3)
    f["nbbo_synthetic"] = True
    f.setdefault("nbbo_reason", "synth_from_last")
    f.setdefault("nbbo_http_status", 200)
    return True


# =============================================================================
# TA indicator helpers (RSI/SMA/EMA/MACD) with multi-tier Polygon aggregates
# =============================================================================
def _sma(vals: List[float], period: int) -> Optional[float]:
    if len(vals) < period: return None
    return float(sum(vals[-period:]) / period)

def _ema(vals: List[float], period: int) -> Optional[float]:
    if len(vals) < period: return None
    k = 2.0 / (period + 1.0)
    ema_val = sum(vals[:period]) / period  # seed with SMA of first 'period'
    for v in vals[period:]:
        ema_val = v * k + ema_val * (1.0 - k)
    return float(ema_val)

def _rsi(vals: List[float], period: int = 14) -> Optional[float]:
    if len(vals) <= period: return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(vals)):
        ch = vals[i] - vals[i-1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    if len(gains) < period or len(losses) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Wilder smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(max(0.0, min(100.0, rsi)))

def _macd(vals: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(vals) < max(slow, fast) + signal:
        return (None, None, None)
    ema_fast = []
    ema_slow = []

    def ema_series(x: List[float], period: int) -> List[float]:
        k = 2.0 / (period + 1.0)
        if len(x) < period:
            return []
        ema0 = sum(x[:period]) / period
        out = [ema0]
        for v in x[period:]:
            out.append(v * k + out[-1] * (1.0 - k))
        return out

    ef = ema_series(vals, fast)
    es = ema_series(vals, slow)
    if not ef or not es:
        return (None, None, None)

    # align by the tail of the shorter list
    min_len = min(len(ef), len(es))
    macd_line_series = [ef[-min_len + i] - es[-min_len + i] for i in range(min_len)]

    sig = _ema(macd_line_series, signal)
    macd_line = macd_line_series[-1] if macd_line_series else None
    hist = (macd_line - sig) if (macd_line is not None and sig is not None) else None
    return (float(macd_line) if macd_line is not None else None,
            float(sig) if sig is not None else None,
            float(hist) if hist is not None else None)

async def _polygon_aggs(underlying: str, mult: int, timespan: str, start_iso: str, end_iso: str) -> List[dict]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return []
    url = f"https://api.polygon.io/v2/aggs/ticker/{underlying}/range/{mult}/{timespan}/{start_iso}/{end_iso}"
    try:
        r = await client.get(url, params={"adjusted":"true","sort":"asc","limit":5000,"apiKey":POLYGON_API_KEY}, timeout=8.0)
        if r.status_code != 200:
            return []
        js = r.json() or {}
        res = js.get("results") or []
        return res if isinstance(res, list) else []
    except Exception:
        return []

def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat()

async def _attach_ta_indicators(underlying: str, f: Dict[str, Any]) -> None:
    """
    Populate RSI14/SMA20/EMA20/MACD for the UNDERLYING using multi-tier fallback:
      1m (≈ 6h) -> 5m (≈ 10d) -> 1d (≈ 300d)
    Also compute ta_pct_change (last vs prev bar).
    """
    if not POLYGON_API_KEY:
        return

    now = datetime.now(timezone.utc)

    tiers = [
        # mult, span, lookback window, min bars, tag
        (1,  "min",  now - timedelta(hours=6),  240, "1m"),
        (5,  "min",  now - timedelta(days=10),  240, "5m"),
        (1,  "day",  now - timedelta(days=300), 200, "1d"),
    ]

    closes: List[float] = []
    src = None

    for mult, span, start_dt, min_bars, tag in tiers:
        bars = await _polygon_aggs(underlying, mult, span, _iso(start_dt), _iso(now))
        cvals = [float(b.get("c")) for b in bars if isinstance(b.get("c"), (int, float))]
        if len(cvals) >= min_bars:
            closes = cvals
            src = tag
            break

    # If still insufficient, take whatever we have from the last tier
    if not closes:
        # Try grab at least whatever daily has
        bars = await _polygon_aggs(underlying, 1, "day", _iso(now - timedelta(days=365)), _iso(now))
        closes = [float(b.get("c")) for b in bars if isinstance(b.get("c"), (int, float))]
        src = src or "1d"

    if not closes or len(closes) < 2:
        # give at least a delta% if we have mid/last & prev_close at option level
        if isinstance(f.get("quote_change_pct"), (int, float)):
            f["ta_pct_change"] = float(f["quote_change_pct"])
            f["ta_src"] = src
        return

    rsi14 = _rsi(closes, 14)
    sma20 = _sma(closes, 20)
    ema20 = _ema(closes, 20)
    macd, macd_sig, macd_hist = _macd(closes, 12, 26, 9)
    last = closes[-1]
    prev = closes[-2]
    pct_change = (last - prev) / prev * 100.0 if prev != 0 else None

    # attach
    f["rsi14"] = rsi14
    f["sma20"] = sma20
    f["ema20"] = ema20
    f["macd"] = macd
    f["macd_signal"] = macd_sig
    f["macd_hist"] = macd_hist
    f["ta_pct_change"] = float(pct_change) if pct_change is not None else None
    f["ta_src"] = src


# =============================================================================
# Core processing
# =============================================================================
async def process_tradingview_job(job: Dict[str, Any]) -> None:
    client = get_http_client()
    if client is None:
        logger.warning("[worker] HTTP client not ready")
        return

    selection_debug: Dict[str, Any] = {}
    replacement_note: Optional[Dict[str, Any]] = None
    option_ticker: Optional[str] = None

    # 1) Parse
    try:
        alert = parse_alert_text(job["alert_text"])
        logger.info("parsed alert: side=%s symbol=%s strike=%s expiry=%s",
                    alert.get("side"), alert.get("symbol"), alert.get("strike"), alert.get("expiry"))
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    side = alert["side"]
    symbol = alert["symbol"]
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

    pm = _build_plus_minus_contracts(symbol, ul_px, target_expiry)
    desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]

    # 3) Chain scan thresholds
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) Selection via scan (strictly honor side)
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            client,
            symbol,
            {"side": side, "symbol": symbol, "strike": alert.get("strike"), "expiry": alert.get("expiry")},
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
                symbol,
                {"side": side, "symbol": symbol, "strike": alert.get("strike"), "expiry": alert.get("expiry")},
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
        selection_debug = {"selected_by": "fallback_pm", "reason": "scan_empty", "chosen_expiry": fallback_exp}
        logger.info("fallback selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)

    chosen_expiry = selection_debug.get("chosen_expiry", str(orig_expiry or target_expiry))

    # 4) Feature bundle + NBBO + TA indicators
    f: Dict[str, Any] = {}
    try:
        if not POLYGON_API_KEY:
            f = {
                "bid": None, "ask": None, "mid": None, "last": None,
                "option_spread_pct": None, "quote_age_sec": None,
                "oi": None, "vol": None,
                "delta": None, "gamma": None, "theta": None, "vega": None,
                "iv": None, "iv_rank": None, "rv20": None, "prev_close": None, "quote_change_pct": None,
                "dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days,
                "em_vs_be_ok": None, "mtf_align": None, "sr_headroom_ok": None, "regime_flag": "trending",
                "prev_day_high": None, "prev_day_low": None,
                "premarket_high": None, "premarket_low": None,
                "vwap": None, "vwap_dist": None,
                "above_pdh": None, "below_pdl": None, "above_pmh": None, "below_pml": None,
            }
        else:
            extra = await poly_option_backfill(client, symbol, option_ticker, datetime.now(timezone.utc).date())
            for k, v in (extra or {}).items():
                if v is not None:
                    f[k] = v

            snap = await polygon_get_option_snapshot_export(client, underlying=symbol, option_ticker=option_ticker)
            core = await build_features(client, alert={**alert, "strike": desired_strike, "expiry": chosen_expiry}, snapshot=snap)
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v

            # derive mid/spread if we already have NBBO
            try:
                bid = f.get("bid"); ask = f.get("ask"); mid = f.get("mid")
                if bid is not None and ask is not None:
                    if mid is None:
                        mid = (float(bid) + float(ask)) / 2.0
                        f["mid"] = round(mid, 4)
                    spread = float(ask) - float(bid)
                    if mid and mid > 0:
                        f["option_spread_pct"] = round((spread / mid) * 100.0, 3)
            except Exception:
                pass

            # ensure NBBO (best-effort real)
            try:
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo = await ensure_nbbo(client, option_ticker, tries=12, delay=0.35)
                    for k, v in (nbbo or {}).items():
                        if v is not None:
                            f[k] = v
            except Exception:
                pass

            if f.get("bid") is None or f.get("ask") is None:
                for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                    if v is not None:
                        f[k] = v

            if f.get("dte") is None:
                try:
                    f["dte"] = (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days
                except Exception:
                    pass

            if f.get("quote_change_pct") is None:
                try:
                    prev_close = f.get("prev_close")
                    mark = f.get("mid") if f.get("mid") is not None else f.get("last")
                    if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
                        f["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
                except Exception:
                    pass

            if (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("last"), (int, float)):
                f.setdefault("mid", float(f["last"]))

            if f.get("bid") is None or f.get("ask") is None:
                nbbo_dbg = await _probe_nbbo_verbose(option_ticker)
                for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
                    if nbbo_dbg.get(k) is not None:
                        f[k] = nbbo_dbg[k]
                f["nbbo_http_status"] = nbbo_dbg.get("nbbo_http_status")
                f["nbbo_reason"] = nbbo_dbg.get("nbbo_reason")
                f["nbbo_body_sample"] = nbbo_dbg.get("nbbo_body_sample")

            if (f.get("option_spread_pct") is None) and (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("mid"), (int, float)):
                f["option_spread_pct"] = float(os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "10.0"))

            # NEW: attach TA indicators for underlying with multi-tier fallback
            try:
                await _attach_ta_indicators(symbol, f)
            except Exception as e:
                logger.warning("TA attach failed: %r", e)

    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}

    # 4b) NBBO-driven replacement (listed but missing NBBO)
    if REPLACE_IF_NO_NBBO and (f.get("bid") is None or f.get("ask") is None or (f.get("nbbo_http_status") and f.get("nbbo_http_status") != 200)):
        try:
            alt = await _find_nbbo_replacement_same_expiry(
                symbol=symbol, side=side, desired_strike=desired_strike,
                expiry_iso=chosen_expiry, min_vol=scan_min_vol, min_oi=scan_min_oi,
            )
        except Exception:
            alt = None
        if alt and alt.get("ticker") and alt["ticker"] != option_ticker:
            old_tk = option_ticker
            option_ticker = alt["ticker"]
            desired_strike = float(alt.get("strike") or desired_strike)
            occ = _occ_meta(option_ticker)
            chosen_expiry = (occ["expiry"] if occ else str(alt.get("expiry") or chosen_expiry))
            try:
                extra2 = await poly_option_backfill(client, symbol, option_ticker, datetime.now(timezone.utc).date())
                for k, v in (extra2 or {}).items():
                    if v is not None:
                        f[k] = v
                for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                    if v is not None:
                        f[k] = v
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo_dbg2 = await _probe_nbbo_verbose(option_ticker)
                    for k in ("bid","ask","mid","option_spread_pct","quote_age_sec"):
                        if nbbo_dbg2.get(k) is not None:
                            f[k] = nbbo_dbg2[k]
                    f["nbbo_http_status"] = nbbo_dbg2.get("nbbo_http_status")
                    f["nbbo_reason"] = nbbo_dbg2.get("nbbo_reason")
                replacement_note = {"old": old_tk, "new": option_ticker, "why": "missing NBBO on initial pick"}
                logger.info("Replaced due to missing NBBO: %s → %s", old_tk, option_ticker)
            except Exception as e:
                logger.warning("NBBO replacement refresh failed: %r", e)

    # 5) 404 replacement if contract truly not listed
    if f.get("nbbo_http_status") == 404 and POLYGON_API_KEY:
        exist = await _poly_reference_contracts_exists(symbol, chosen_expiry, option_ticker)
        logger.info("NBBO 404 verification: listed=%s snapshot_ok=%s reason=%s",
                    exist.get("listed"), exist.get("snapshot_ok"), exist.get("reason"))
        if exist.get("listed") is False and not exist.get("snapshot_ok"):
            repl = await _rescan_best_replacement(
                symbol=symbol, side=side,
                desired_strike=desired_strike, expiry_iso=chosen_expiry,
                min_vol=scan_min_vol, min_oi=scan_min_oi,
            )
            if repl:
                old_tk = option_ticker
                option_ticker = repl["ticker"]
                desired_strike = float(repl.get("strike") or desired_strike)
                try:
                    occ = _occ_meta(option_ticker)
                    chosen_expiry = (occ["expiry"] if occ else str(repl.get("expiry") or chosen_expiry))
                    extra2 = await poly_option_backfill(client, symbol, option_ticker, datetime.now(timezone.utc).date())
                    for k, v in (extra2 or {}).items():
                        if v is not None:
                            f[k] = v
                    for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                        if v is not None:
                            f[k] = v
                    if f.get("bid") is None or f.get("ask") is None:
                        nbbo_dbg2 = await _probe_nbbo_verbose(option_ticker)
                        for k in ("bid","ask","mid","option_spread_pct","quote_age_sec"):
                            if nbbo_dbg2.get(k) is not None:
                                f[k] = nbbo_dbg2[k]
                        f["nbbo_http_status"] = nbbo_dbg2.get("nbbo_http_status")
                        f["nbbo_reason"] = nbbo_dbg2.get("nbbo_reason")
                    replacement_note = {
                        "old": old_tk, "new": option_ticker,
                        "why": "contract not listed in Polygon reference/snapshot",
                    }
                    logger.info("Replaced contract due to 404: %s → %s", old_tk, option_ticker)
                except Exception as e:
                    logger.warning("Replacement contract fetch failed: %r", e)
                    replacement_note = None

    # 5c) FINAL: Synthetic NBBO pass (only if still missing)
    try:
        synth_applied = synthesize_nbbo_if_missing(f, _is_rth_now())
        if synth_applied:
            logger.info("Applied synthetic NBBO around mid=%.4f (spread%%=%.2f)", f.get("mid", 0.0), f.get("option_spread_pct", -1))
    except Exception as e:
        logger.warning("Synthetic NBBO failed: %r", e)

    # 6) Optional Telegram pre-LLM
    if SEND_CHAIN_SCAN_ALERTS and selection_debug.get("selected_by", "").startswith("chain_scan"):
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                pre_text = (
                    "🔎 Chain-Scan Pick (from TradingView alert)\n"
                    f"{side} {symbol} | Strike {desired_strike} | Exp {chosen_expiry}\n"
                    f"Contract: {option_ticker}\n"
                    f"NBBO {f.get('bid')}/{f.get('ask')}  Mark={f.get('mid')}  Last={f.get('last')}\n"
                    f"Spread%={f.get('option_spread_pct')}  QuoteAge(s)={f.get('quote_age_sec')}\n"
                    f"OI={f.get('oi')}  Vol={f.get('vol')}  IV={f.get('iv')}  Δ={f.get('delta')} Γ={f.get('gamma')}\n"
                    f"DTE={f.get('dte')}  Regime={f.get('regime_flag')}  (pre-LLM)\n"
                    f"NBBO dbg: status={f.get('nbbo_http_status')} reason={f.get('nbbo_reason')}\n"
                )
                await send_telegram(pre_text)
        except Exception as e:
            logger.exception("[worker] Telegram pre-LLM chainscan error: %s", e)

    # 7) LLM
    pf_ok, pf_checks = preflight_ok(f)
    try:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
    except Exception as e:
        llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}", "checklist": {}, "ev_estimate": {}}
    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")

    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    if force_buy:
        decision_final = "buy"

    # Diff note (strike/expiry changes)
    diff_bits = []
    if isinstance(orig_strike, (int, float)) and isinstance(desired_strike, (int, float)) and float(orig_strike) != float(desired_strike):
        diff_bits.append(f"🎯 Selected strike {desired_strike} (alert was {orig_strike})")
    if orig_expiry and chosen_expiry and str(orig_expiry) != str(chosen_expiry):
        diff_bits.append(f"🗓 Selected expiry {chosen_expiry} (alert was {orig_expiry})")
    diff_note = "\n".join(diff_bits)

    # TA compact line for Telegram
    def _fmt(x, n=2):
        return ("None" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)))
    ta_line = f"📈 RSI14={_fmt(f.get('rsi14'))} SMA20≈{_fmt(f.get('sma20'))} EMA20≈{_fmt(f.get('ema20'))} Δ%≈{_fmt(f.get('ta_pct_change'))} src:{_fmt(f.get('ta_src'),0)}"

    # 8) Telegram final
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker, f=f, llm=llm, llm_ran=True, llm_reason="", score=score, rating=rating,
            diff_note=diff_note,
        )
        # append TA line
        tg_text += f"\n{ta_line}"
        if selection_debug.get("selected_by","").startswith("chain_scan"):
            tg_text += "\n🔎 Note: Contract selected via chain-scan (liquidity + strike/expiry fit)."
        if replacement_note is not None:
            tg_text += f"\n⚠️ Replacement: {replacement_note['old']} → {replacement_note['new']} ({replacement_note['why']})."
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await send_telegram(tg_text)
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    # 9) IBKR (optional)
    ib_attempted = False
    ib_result_obj: Optional[Any] = None
    try:
        if (decision_final == "buy") and ib_enabled and (pf_ok or force_buy):
            ib_attempted = True
            mode = IBKR_ORDER_MODE
            mid = f.get("mid")
            if mode == "market":
                use_market = True
            elif mode == "limit":
                use_market = (mid is None)
            else:
                use_market = not (IBKR_USE_MID_AS_LIMIT and (mid is not None))
            limit_px = None if use_market else float(mid) if mid is not None else None

            ib_result_obj = await place_recommended_option_order(
                symbol=symbol, side=side,
                strike=float(desired_strike), expiry_iso=chosen_expiry,
                quantity=int(qty),
                limit_price=limit_px, action="BUY", tif=IBKR_TIF,
            )
    except Exception as e:
        ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 10) Decision log
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": symbol,
        "side": side,
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": f"llm.{decision_final}",
        "prescore": None,
        "llm": {"ran": True, "decision": llm.get("decision"),
                "confidence": llm.get("confidence"), "reason": llm.get("reason")},
        "features": {
            "reco_expiry": chosen_expiry,
            "oi": f.get("oi"), "vol": f.get("vol"),
            "bid": f.get("bid"), "ask": f.get("ask"),
            "mark": f.get("mid"), "last": f.get("last"),
            "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
            "prev_close": f.get("prev_close"), "quote_change_pct": f.get("quote_change_pct"),
            "delta": f.get("delta"), "gamma": f.get("gamma"), "theta": f.get("theta"), "vega": f.get("vega"),
            "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
            "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag"),
            "prev_day_high": f.get("prev_day_high"), "prev_day_low": f.get("prev_day_low"),
            "premarket_high": f.get("premarket_high"), "premarket_low": f.get("premarket_low"),
            "vwap": f.get("vwap"), "vwap_dist": f.get("vwap_dist"),
            "above_pdh": f.get("above_pdh"), "below_pdl": f.get("below_pdl"),
            "above_pmh": f.get("above_pmh"), "below_pml": f.get("below_pml"),
            "rsi14": f.get("rsi14"), "sma20": f.get("sma20"), "ema20": f.get("ema20"),
            "macd": f.get("macd"), "macd_signal": f.get("macd_signal"), "macd_hist": f.get("macd_hist"),
            "ta_pct_change": f.get("ta_pct_change"), "ta_src": f.get("ta_src"),
            "nbbo_http_status": f.get("nbbo_http_status"), "nbbo_reason": f.get("nbbo_reason"),
            "nbbo_synthetic": f.get("nbbo_synthetic", False),
        },
        "pm_contracts": {
            "plus5_call": {"strike": pm["strike_call"], "contract": pm["contract_call"]},
            "minus5_put": {"strike": pm["strike_put"],  "contract": pm["contract_put"]},
        },
        "ibkr": {"enabled": ib_enabled, "attempted": ib_attempted, "result": ib_result_obj},
        "selection_debug": selection_debug,
        "alert_original": {"strike": orig_strike, "expiry": orig_expiry},
        "chosen": {"strike": desired_strike, "expiry": chosen_expiry},
        "replacement": replacement_note,
    })


# =============================================================================
# Diagnostics
# =============================================================================
async def diag_polygon_bundle(underlying: str, contract: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    enc = _encode_ticker_path(contract)
    out = {}

    m = re.search(r":([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$", contract)
    if m:
        yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
        expiry_iso = f"20{yy}-{mm}-{dd}"
        side = "call" if cp.upper() == "C" else "put"
        out["multi"] = await _http_json(
            client,
            f"https://api.polygon.io/v3/snapshot/options/{underlying}",
            {"apiKey": POLYGON_API_KEY, "contract_type": side, "expiration_date": expiry_iso, "limit": 5, "greeks": "true"},
            timeout=6.0
        )

    out["single"] = await _http_json(
        client,
        f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    out["last_quote"] = await _http_json(
        client,
        f"https://api.polygon.io/v3/quotes/options/{enc}/last",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    out["open_close"] = await _http_json(
        client,
        f"https://api.polygon.io/v1/open-close/options/{enc}/{yday}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    now_utc_dt = datetime.now(timezone.utc)
    frm_iso = datetime(now_utc_dt.year, now_utc_dt.month, now_utc_dt.day, 0,0,0,tzinfo=timezone.utc).isoformat()
    to_iso = now_utc_dt.isoformat()
    out["aggs"] = await _http_json(
        client,
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/min/{frm_iso}/{to_iso}?",
        {"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )

    def skim(d):
        if not isinstance(d, dict): return d
        res = d.get("results")
        return {
            "keys": list(d.keys())[:10],
            "sample": (res[:2] if isinstance(res, list) else (res if isinstance(res, dict) else d)),
            "status_hint": d.get("status"),
        }
    return {
        "multi": skim(out.get("multi")),
        "single": skim(out.get("single")),
        "last_quote": skim(out.get("last_quote")),
        "open_close": skim(out.get("open_close")),
        "aggs": skim(out.get("aggs")),
    }

async def net_debug_info() -> Dict[str, Any]:
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    out_ip = None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            out_ip = (await c.get("https://ifconfig.me/ip")).text.strip()
    except Exception as e:
        out_ip = f"fetch-failed: {e.__class__.__name__}"
    can_connect = None; err = None
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.close()
        can_connect = True
    except Exception as e:
        can_connect = False
        err = f"{e.__class__.__name__}: {e}"
    return {"ibkr_host": host, "ibkr_port": port, "egress_ip": out_ip, "connect_test": can_connect, "error": err}

__all__ = ["process_tradingview_job", "diag_polygon_bundle", "net_debug_info"]
