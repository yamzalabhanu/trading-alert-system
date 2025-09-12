# polygon_ops.py
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List

import httpx

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY

logger = logging.getLogger("trading_engine")

# --------------------
# Low-level HTTP helpers
# --------------------
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {"status": r.status_code, "error": True, "body": (r.json() if "application/json" in (r.headers.get("content-type") or "") else r.text)}
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

async def _http_get_any(url: str, params: dict | None = None, timeout: float = 6.0) -> Dict[str, Any]:
    client = get_http_client()
    if client is None:
        return {"status": None, "error": "HTTP client not ready"}
    try:
        r = await client.get(url, params=params or {}, timeout=timeout)
        ct = r.headers.get("content-type", "")
        try:
            payload = r.json() if "application/json" in ct else r.text
        except Exception:
            payload = r.text
        return {"status": r.status_code, "body": payload}
    except Exception as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}"}

# --------------------
# NBBO helpers (Options)
# --------------------
async def _pull_nbbo_direct(option_ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    client = get_http_client()
    if not POLYGON_API_KEY or client is None:
        return out
    try:
        enc = option_ticker.replace("/", "%2F")
        lastq = await _http_json(
            client,
            f"https://api.polygon.io/v3/quotes/options/{enc}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=4.0,
        )
        if not lastq or not isinstance(lastq, dict) or lastq.get("error"):
            return out
        res = lastq.get("results") or {}
        last = res.get("last") or res
        bid = last.get("bidPrice"); ask = last.get("askPrice")
        ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
            mid = (bid + ask) / 2.0
            out["bid"] = float(bid); out["ask"] = float(ask); out["mid"] = round(float(mid), 4)
            if mid > 0:
                out["option_spread_pct"] = round((ask - bid) / mid * 100.0, 3)
        if ts is not None:
            try:
                ns = int(ts)
                if   ns >= 10**14: sec = ns / 1e9
                elif ns >= 10**11: sec = ns / 1e6
                elif ns >= 10**8:  sec = ns / 1e3
                else:              sec = float(ns)
                out["quote_age_sec"] = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
            except Exception:
                pass
    except Exception:
        pass
    return out

async def _probe_nbbo_verbose(option_ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return {"nbbo_reason": "no POLYGON_API_KEY in env"}
    enc = option_ticker.replace("/", "%2F")
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    res = await _http_get_any(url, params={"apiKey": POLYGON_API_KEY}, timeout=6.0)
    out["nbbo_http_status"] = res.get("status")
    if res.get("status") != 200:
        body = res.get("body")
        if isinstance(body, dict):
            out["nbbo_reason"] = body.get("error") or body.get("message") or "non-200 from Polygon"
        else:
            out["nbbo_reason"] = "non-200 from Polygon"
        out["nbbo_body_sample"] = (body if isinstance(body, dict) else str(body))[:400]
        return out

    body = res.get("body") or {}
    last = (body.get("results") or {}).get("last") or body.get("results") or {}
    bid = last.get("bidPrice"); ask = last.get("askPrice")
    ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
        mid = (bid + ask) / 2.0
        out.update({
            "bid": float(bid), "ask": float(ask),
            "mid": round(mid, 4),
            "option_spread_pct": round(((ask - bid)/mid*100.0), 3) if mid > 0 else None,
        })
    else:
        out["nbbo_reason"] = "no bid/ask in response (thin or AH?)"
    try:
        if ts is not None:
            ns = int(ts)
            if   ns >= 10**14: sec = ns / 1e9
            elif ns >= 10**11: sec = ns / 1e6
            elif ns >= 10**8:  sec = ns / 1e3
            else:              sec = float(ns)
            out["quote_age_sec"] = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    except Exception:
        pass
    return out

# --------------------
# Listing checks & replacements
# --------------------
async def _poly_reference_contracts_exists(underlying: str, expiry_iso: str, ticker: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {"listed": None, "snapshot_ok": None, "reason": "no HTTP or API key"}
    try:
        base = "https://api.polygon.io/v3/reference/options/contracts"
        params = {"underlying_ticker": underlying, "expiration_date": expiry_iso, "limit": 1000, "apiKey": POLYGON_API_KEY}
        r = await client.get(base, params=params, timeout=8.0)
        listed = False
        if r.status_code == 200:
            js = r.json()
            for it in js.get("results", []):
                if it.get("ticker") == ticker:
                    listed = True
                    break
        elif r.status_code in (402, 403, 429):
            return {"listed": None, "snapshot_ok": None, "reason": f"ref-contracts {r.status_code}"}
        enc = ticker.replace("/", "%2F")
        s = await client.get(
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0,
        )
        snapshot_ok = (s.status_code == 200 and isinstance((s.json() or {}).get("results"), dict))
        return {"listed": listed, "snapshot_ok": snapshot_ok, "reason": None}
    except Exception as e:
        return {"listed": None, "snapshot_ok": None, "reason": f"error: {type(e).__name__}: {e}"}

async def _rescan_best_replacement(symbol: str, side: str, desired_strike: float, expiry_iso: str, min_vol: int, min_oi: int) -> Optional[Dict[str, Any]]:
    # Import here to avoid circular imports
    from market_ops import scan_top_candidates_for_alert
    try:
        try:
            pool = await scan_top_candidates_for_alert(
                get_http_client(), symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=15,
                restrict_expiries=[expiry_iso],  # type: ignore
            ) or []
        except TypeError:
            pool = await scan_top_candidates_for_alert(
                get_http_client(), symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_oi, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=20,
            ) or []
            pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []
    # correct side
    def _occ_side(tk: str) -> Optional[str]:
        m = re.search(r"([CP])\d{8,9}$", tk or "")
        if not m: return None
        return "CALL" if m.group(1).upper() == "C" else "PUT"
    try:
        import re
        pool = [it for it in pool if _occ_side(it.get("ticker")) == side]
    except Exception:
        pass
    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        nbbo_missing = 0 if (it.get("bid") is not None and it.get("ask") is not None) else 1
        return (sd, nbbo_missing, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))
    pool.sort(key=_rank)
    return pool[0] if pool else None

async def _find_nbbo_replacement_same_expiry(symbol: str, side: str, desired_strike: float, expiry_iso: str, min_vol: int, min_oi: int) -> Optional[Dict[str, Any]]:
    from market_ops import scan_top_candidates_for_alert
    try:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=12,
            restrict_expiries=[expiry_iso],  # type: ignore
        ) or []
    except TypeError:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=15,
        ) or []
        pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []
    # filter: correct side + has NBBO
    def _side_ok(tk: Optional[str]) -> bool:
        if not tk: return False
        c = tk[-10:-9].upper() if len(tk) >= 10 else ""
        return (c == "C" and side == "CALL") or (c == "P" and side == "PUT")
    pool = [it for it in pool if _side_ok(it.get("ticker")) and it.get("bid") is not None and it.get("ask") is not None]
    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        return (sd, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))
    pool.sort(key=_rank)
    return pool[0] if pool else None

# --------------------
# Enrichment: Reference, Corp Actions, Indicators, Aggregates, Snapshot
# --------------------
async def fetch_reference_overview(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}"
    js = await _http_json(client, url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    if not js or js.get("error"):
        return {}
    res = js.get("results") or {}
    return {
        "name": res.get("name"),
        "market_cap": res.get("market_cap"),
        "currency_name": res.get("currency_name"),
        "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
        "homepage_url": res.get("homepage_url"),
        "sic_description": res.get("sic_description"),
        "sic_code": res.get("sic_code"),
        "primary_exchange": res.get("primary_exchange")
    }

async def fetch_corporate_actions(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    divs = await _http_json(client, "https://api.polygon.io/v3/reference/dividends", {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY}, timeout=6.0)
    splits = await _http_json(client, "https://api.polygon.io/v3/reference/splits", {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY}, timeout=6.0)
    latest_div = (divs or {}).get("results", [{}])[:1]
    latest_spl = (splits or {}).get("results", [{}])[:1]
    return {
        "latest_dividend": latest_div[0] if latest_div else None,
        "latest_split": latest_spl[0] if latest_spl else None,
    }

async def fetch_technical_indicators(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    base = "https://api.polygon.io/v1/indicators"
    out: Dict[str, Any] = {}
    params_common = {"timespan": "day", "series_type": "close", "limit": 1, "apiKey": POLYGON_API_KEY}
    sma = await _http_json(client, f"{base}/sma/{symbol}", {**params_common, "window": 20}, timeout=6.0) or {}
    ema = await _http_json(client, f"{base}/ema/{symbol}", {**params_common, "window": 20}, timeout=6.0) or {}
    rsi = await _http_json(client, f"{base}/rsi/{symbol}", {**params_common, "window": 14}, timeout=6.0) or {}
    macd = await _http_json(client, f"{base}/macd/{symbol}", {"timespan": "day", "series_type": "close", "short_window": 12, "long_window": 26, "signal_window": 9, "limit": 1, "apiKey": POLYGON_API_KEY}, timeout=6.0) or {}
    def _val(js, key="value"):
        try:
            vals = (js.get("results") or {}).get("values") or []
            return vals[-1].get(key) if vals else None
        except Exception:
            return None
    out["sma20"] = _val(sma)
    out["ema20"] = _val(ema)
    out["rsi14"] = _val(rsi)
    try:
        vals = (macd.get("results") or {}).get("values") or []
        if vals:
            m = vals[-1]
            out["macd"] = {"macd": m.get("macd"), "signal": m.get("signal"), "hist": m.get("hist")}
    except Exception:
        pass
    return out

async def fetch_aggs_recent(symbol: str, timespan: str = "minute", limit: int = 50) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=3)).isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start}/{now.isoformat()}"
    js = await _http_json(client, url, {"adjusted": "true", "sort": "desc", "limit": max(1, limit), "apiKey": POLYGON_API_KEY}, timeout=8.0)
    if not js or js.get("error"):
        return {}
    results = js.get("results") or []
    last = results[0] if results else {}
    return {"aggs_last": last, "aggs_count": len(results)}

async def fetch_aggs_second_recent(symbol: str, limit: int = 120) -> Dict[str, Any]:
    return await fetch_aggs_recent(symbol, timespan="second", limit=limit)

async def fetch_stock_snapshot(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    url = f"https://api.polygon.io/v3/snapshot/stocks/{symbol.upper()}"
    js = await _http_json(client, url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    if not js or js.get("error"):
        return {}
    return js.get("results") or {}

async def enrich_underlying(symbol: str) -> Dict[str, Any]:
    """Convenience aggregator for enrichment payload used by LLM."""
    try:
        ref, ca, ind, snap = await fetch_reference_overview(symbol), await fetch_corporate_actions(symbol), await fetch_technical_indicators(symbol), await fetch_stock_snapshot(symbol)
        aggm = await fetch_aggs_recent(symbol, timespan="minute", limit=50)
        aggs = await fetch_aggs_second_recent(symbol, limit=120)
        return {
            "ref": ref,
            "corp_actions": ca,
            "indicators": ind,
            "snapshot": snap,
            "aggs_minute": aggm,
            "aggs_second": aggs,
        }
    except Exception as e:
        logger.warning("enrich_underlying failed: %r", e)
        return {}

__all__ = [
    # http helpers
    "_http_json", "_http_get_any",
    # nbbo helpers
    "_pull_nbbo_direct", "_probe_nbbo_verbose",
    # listing / replacement
    "_poly_reference_contracts_exists", "_rescan_best_replacement", "_find_nbbo_replacement_same_expiry",
    # enrichment
    "fetch_reference_overview", "fetch_corporate_actions", "fetch_technical_indicators", "fetch_aggs_recent", "fetch_aggs_second_recent", "fetch_stock_snapshot", "enrich_underlying",
]


# -------------------------------------------------
# engine_processor.py (updated to use enrichment)
# -------------------------------------------------
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
    _build_plus_minus_contracts,
    preflight_ok, compose_telegram_text,
    round_strike_to_common_increment,
)
from polygon_ops import (
    _http_json, _http_get_any, _pull_nbbo_direct, _probe_nbbo_verbose,
    _poly_reference_contracts_exists, _rescan_best_replacement, _find_nbbo_replacement_same_expiry,
    enrich_underlying,
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

async def _recommend_plusminus5_from_top5(symbol: str, side: str, ul_px: float, expiry_iso: str, min_vol: int, min_oi: int) -> Optional[Dict[str, Any]]:
    """Pick contract near +5% (call) / -5% (put) strike from top-5 by OI/Vol within same expiry."""
    try:
        target = round_strike_to_common_increment(ul_px * (1.05 if side == "CALL" else 0.95))
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": target, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=20,
            restrict_expiries=[expiry_iso],  # type: ignore
        ) or []
    except TypeError:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": target, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")), top_overall=20,
        ) or []
        pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []
    # closest to target, then sort by OI+Vol
    pool.sort(key=lambda it: (abs(float(it.get("strike") or target) - target)))
    top5 = pool[:5]
    top5.sort(key=lambda it: ((it.get("oi") or 0), (it.get("vol") or 0)), reverse=True)
    return top5[0] if top5 else None

# =========================
# Core processing
# =========================
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
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy  = bool(job["flags"].get("force_buy", False))
    qty        = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    # Preserve originals
    orig_strike = alert.get("strike")
    orig_expiry = alert.get("expiry")

    # 2) Expiry defaulting (two-weeks Friday, avoid same-week overlap)
    ul_px = float(alert["underlying_price_from_alert"])  # from TV
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

    # 3a) Selection via scan (strict side)
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
                top_overall=8,
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
        selection_debug = {"selected_by": "chain_scan", "selected_ticker": option_ticker, "best_item": candidate, "chosen_expiry": chosen_expiry}
        logger.info("chain_scan selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)
    else:
        fallback_exp = str(orig_expiry or target_expiry)
        option_ticker = pm["contract_call"] if side == "CALL" else pm["contract_put"]
        desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]
        chosen_expiry = fallback_exp
        selection_debug = {"selected_by": "fallback_pm", "reason": "scan_empty", "chosen_expiry": fallback_exp}
        logger.info("fallback selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)

    chosen_expiry = selection_debug.get("chosen_expiry", str(orig_expiry or target_expiry))

    # ✨ NEW: Recommendation block (±5% from UL, pick from top-5 by OI/Vol)
    recommended = await _recommend_plusminus5_from_top5(alert["symbol"], side, ul_px, chosen_expiry, scan_min_vol, scan_min_oi)

    # 4) Feature bundle + NBBO
    f: Dict[str, Any] = {}
    try:
        if not POLYGON_API_KEY:
            f = {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}
        else:
            extra = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
            for k, v in (extra or {}).items():
                if v is not None:
                    f[k] = v
            snap = await polygon_get_option_snapshot_export(get_http_client(), underlying=alert["symbol"], option_ticker=option_ticker)
            core = await build_features(get_http_client(), alert={**alert, "strike": desired_strike, "expiry": chosen_expiry}, snapshot=snap)
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v
            # derive mid/spread
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
            # ensure NBBO
            try:
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo = await ensure_nbbo(get_http_client(), option_ticker, tries=12, delay=0.35)
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
                    prev_close = f.get("prev_close"); mark = f.get("mid") if f.get("mid") is not None else f.get("last")
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
    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}

    # 4c) Enrich underlying context for the LLM
    try:
        f["underlying_enrichment"] = await enrich_underlying(alert["symbol"])  # reference, corp actions, indicators, snapshots, aggs
    except Exception as e:
        logger.warning("underlying enrichment failed: %r", e)

    # 4d) NBBO-driven replacement (listed but missing NBBO)
    if REPLACE_IF_NO_NBBO and (f.get("bid") is None or f.get("ask") is None or (f.get("nbbo_http_status") and f.get("nbbo_http_status") != 200)):
        try:
            alt = await _find_nbbo_replacement_same_expiry(
                symbol=alert["symbol"], side=side, desired_strike=desired_strike,
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
                extra2 = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
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
        exist = await _poly_reference_contracts_exists(alert["symbol"], chosen_expiry, option_ticker)
        logger.info("NBBO 404 verification: listed=%s snapshot_ok=%s reason=%s", exist.get("listed"), exist.get("snapshot_ok"), exist.get("reason"))
        if exist.get("listed") is False and not exist.get("snapshot_ok"):
            repl = await _rescan_best_replacement(
                symbol=alert["symbol"], side=side,
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
                    extra2 = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
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
                    replacement_note = {"old": old_tk, "new": option_ticker, "why": "contract not listed in Polygon reference/snapshot"}
                    logger.info("Replaced contract due to 404: %s → %s", old_tk, option_ticker)
                except Exception as e:
                    logger.warning("Replacement contract fetch failed: %r", e)
                    replacement_note = None

    # 6) Optional Telegram pre-LLM
    if SEND_CHAIN_SCAN_ALERTS and selection_debug.get("selected_by", "").startswith("chain_scan"):
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                rec_line = ""
                if recommended:
                    rec_line = (f"\n📌 Recommended (±5% from UL): {chosen_expiry} | {recommended['ticker']} | strike {recommended.get('strike')} | "
                                f"OI {recommended.get('oi')} Vol {recommended.get('vol')} | NBBO {recommended.get('bid')}/{recommended.get('ask')} mid={recommended.get('mid')} | "
                                f"spread%={recommended.get('spread_pct')}")
                pre_text = (
                    "🔎 Chain-Scan Pick (from TradingView alert)\n"
                    f"{side} {alert['symbol']} | Strike {desired_strike} | Exp {chosen_expiry}\n"
                    f"Contract: {option_ticker}{rec_line}\n"
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

    # Diff note
    diff_bits = []
    if isinstance(orig_strike, (int, float)) and isinstance(desired_strike, (int, float)) and float(orig_strike) != float(desired_strike):
        diff_bits.append(f"🎯 Selected strike {desired_strike} (alert was {orig_strike})")
    if orig_expiry and chosen_expiry and str(orig_expiry) != str(chosen_expiry):
        diff_bits.append(f"🗓 Selected expiry {chosen_expiry} (alert was {orig_expiry})")
    diff_note = "\n".join(diff_bits)

    # 8) Telegram final
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker, f=f, llm=llm, llm_ran=True, llm_reason="", score=score, rating=rating,
            diff_note=diff_note,
        )
        if selection_debug.get("selected_by","").startswith("chain_scan"):
            tg_text += "\n🔎 Note: Contract selected via chain-scan (liquidity + strike/expiry fit)."
        if recommended:
            tg_text += (f"\n📌 Recommended (±5% from UL): {chosen_expiry} | {recommended['ticker']} | strike {recommended.get('strike')} | "
                        f"OI {recommended.get('oi')} Vol {recommended.get('vol')} | NBBO {recommended.get('bid')}/{recommended.get('ask')} mid={recommended.get('mid')} | "
                        f"spread%={recommended.get('spread_pct')}")
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
                symbol=alert["symbol"], side=side,
                strike=float(desired_strike), expiry_iso=chosen_expiry,
                quantity=int(qty),
                limit_price=limit_px, action="BUY", tif=IBKR_TIF,
            )
    except Exception as e:
        ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 10) Decision log
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": alert["symbol"],
        "side": side,
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": f"llm.{decision_final}",
        "prescore": None,
        "llm": {"ran": True, "decision": llm.get("decision"), "confidence": llm.get("confidence"), "reason": llm.get("reason")},
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
            "nbbo_http_status": f.get("nbbo_http_status"), "nbbo_reason": f.get("nbbo_reason"),
            # enrichment snapshot
            "enrichment": f.get("underlying_enrichment"),
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
        "recommended": recommended,
    })

# Diagnostics left as-is (omitted here to keep file focused)
__all__ = ["process_tradingview_job"]
