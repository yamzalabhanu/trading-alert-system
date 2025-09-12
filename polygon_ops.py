# polygon_ops.py
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import httpx

from engine_runtime import get_http_client
from engine_common import _ticker_matches_side  # for side-safe filtering
from market_ops import scan_top_candidates_for_alert  # used by replacement finders

logger = logging.getLogger("trading_engine.polygon_ops")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLY_DEBUG_DUMP = int(os.getenv("POLY_DEBUG_DUMP", "0"))

# ---------------------------
# Low-level HTTP helpers
# ---------------------------
async def _http_get_any(url: str, params: dict | None = None, timeout: float = 6.0) -> Dict[str, Any]:
    client = get_http_client()
    if client is None:
        return {"status": None, "body": None, "error": "HTTP client not ready"}
    try:
        r = await client.get(url, params=params or {}, timeout=timeout)
        ct = (r.headers.get("content-type") or "").lower()
        try:
            body = r.json() if "application/json" in ct else r.text
        except Exception:
            body = r.text
        return {"status": r.status_code, "body": body}
    except Exception as e:
        return {"status": None, "body": None, "error": f"{type(e).__name__}: {e}"}

async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """Compatibility helper used by diagnostics."""
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

# ---------------------------
# NBBO extraction helpers
# ---------------------------
def _body_sample(b: Any, n: int = 400) -> str:
    try:
        if isinstance(b, (dict, list)):
            import json
            s = json.dumps(b)
        else:
            s = str(b)
        return s[:n]
    except Exception:
        return "<unprintable>"

def _extract_nbbo_from_snapshot(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    /v3/snapshot/options/{underlying}/{ticker}
    NBBO lives in results.last_quote if entitled.
    """
    out: Dict[str, Any] = {}
    try:
        res = js.get("results") or {}
        lq  = res.get("last_quote") or {}
        bid = lq.get("bid_price", lq.get("bid"))
        ask = lq.get("ask_price", lq.get("ask"))
        ts  = lq.get("t") or lq.get("timestamp") or lq.get("sip_timestamp")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
            out["bid"] = float(bid)
            out["ask"] = float(ask)
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            if mid > 0:
                out["option_spread_pct"] = round((out["ask"] - out["bid"]) / mid * 100.0, 3)
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

def _extract_nbbo_from_quotes(js: Dict[str, Any]) -> Dict[str, Any]:
    """
    /v3/quotes/{ticker}?order=desc&limit=1
    """
    out: Dict[str, Any] = {}
    try:
        arr = js.get("results") or []
        if not isinstance(arr, list) or not arr:
            return out
        q = arr[0]
        bid = q.get("bid_price")
        ask = q.get("ask_price")
        ts  = q.get("timestamp") or q.get("sip_timestamp")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
            out["bid"] = float(bid)
            out["ask"] = float(ask)
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            if mid > 0:
                out["option_spread_pct"] = round((out["ask"] - out["bid"]) / mid * 100.0, 3)
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

# ---------------------------
# Public NBBO helpers
# ---------------------------
async def _pull_nbbo_direct(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    Try snapshot first, then quotes endpoint. Attach http status fields and reason.
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    snap_status = None
    try:
        if underlying:
            enc = option_ticker
            snap = await _http_get_any(
                f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
                {"apiKey": POLYGON_API_KEY},
                timeout=6.0,
            )
            snap_status = snap.get("status")
            if POLY_DEBUG_DUMP:
                logger.info("[snapshot] status=%s sample=%s", snap_status, _body_sample(snap.get("body")))
            if snap_status == 200 and isinstance(snap.get("body"), dict):
                out |= _extract_nbbo_from_snapshot(snap["body"])
    except Exception as e:
        logger.debug("snapshot err: %r", e)

    quotes_status = None
    if out.get("bid") is None or out.get("ask") is None:
        try:
            enc = option_ticker
            q = await _http_get_any(
                f"https://api.polygon.io/v3/quotes/{enc}",
                {"apiKey": POLYGON_API_KEY, "limit": 1, "order": "desc", "sort": "timestamp"},
                timeout=6.0,
            )
            quotes_status = q.get("status")
            if POLY_DEBUG_DUMP:
                logger.info("[quotes] status=%s sample=%s", quotes_status, _body_sample(q.get("body")))
            if quotes_status == 200 and isinstance(q.get("body"), dict):
                out |= _extract_nbbo_from_quotes(q["body"])
        except Exception as e:
            logger.debug("quotes err: %r", e)

    out["snapshot_http_status"] = snap_status
    out["quotes_http_status"] = quotes_status
    if (snap_status in (200,) and not out.get("bid") and not out.get("ask")) or \
       (quotes_status == 200 and not out.get("bid") and not out.get("ask")):
        out["nbbo_reason"] = "no bid/ask fields returned; likely no options quotes entitlement on this API key"
    return out

async def _probe_nbbo_verbose(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper for detailed debug + parsed NBBO fields.
    """
    res = await _pull_nbbo_direct(option_ticker, underlying=underlying)
    out: Dict[str, Any] = {
        "nbbo_http_status": (res.get("snapshot_http_status") or res.get("quotes_http_status")),
        "snapshot_http_status": res.get("snapshot_http_status"),
        "quotes_http_status": res.get("quotes_http_status"),
        "nbbo_reason": res.get("nbbo_reason"),
    }
    for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
        if res.get(k) is not None:
            out[k] = res[k]
    return out

async def _poly_reference_contracts_exists(underlying: str, expiry_iso: str, ticker: str) -> Dict[str, Any]:
    """
    Confirm listing via reference endpoint and that snapshot returns results.
    """
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

        s = await client.get(
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{ticker}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        snapshot_ok = (s.status_code == 200 and isinstance((s.json() or {}).get("results"), dict))
        return {"listed": listed, "snapshot_ok": snapshot_ok, "reason": None}
    except Exception as e:
        return {"listed": None, "snapshot_ok": None, "reason": f"error: {type(e).__name__}: {e}"}

# ---------------------------
# Replacement helpers (missing before)
# ---------------------------
async def _rescan_best_replacement(
    symbol: str,
    side: str,
    desired_strike: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    If a contract 404s / isn't listed, pick a nearby replacement from the *same expiry*,
    ranking by: strike distance, NBBO presence, lower spread, higher OI/Vol.
    """
    client = get_http_client()
    if client is None:
        return None

    try:
        try:
            pool: List[Dict[str, Any]] = await scan_top_candidates_for_alert(
                client, symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=15,
                restrict_expiries=[expiry_iso],  # some versions support this
            ) or []
        except TypeError:
            # older market_ops without restrict_expiries
            pool = await scan_top_candidates_for_alert(
                client, symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=20,
            ) or []
            pool = [it for it in pool if str(it.get("expiry")) == str(expiry_iso)]
    except Exception:
        pool = []

    # side correctness
    pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side)]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        nbbo_missing = 0 if (it.get("bid") is not None and it.get("ask") is not None) else 1
        return (sd, nbbo_missing, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

async def _find_nbbo_replacement_same_expiry(
    symbol: str,
    side: str,
    desired_strike: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    Softer replacement when current pick is listed but has no NBBO.
    Look in the same expiry and require NBBO to be present.
    """
    client = get_http_client()
    if client is None:
        return None

    try:
        try:
            pool: List[Dict[str, Any]] = await scan_top_candidates_for_alert(
                client, symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=15,
                restrict_expiries=[expiry_iso],
            ) or []
        except TypeError:
            pool = await scan_top_candidates_for_alert(
                client, symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=20,
            ) or []
            pool = [it for it in pool if str(it.get("expiry")) == str(expiry_iso)]
    except Exception:
        pool = []

    # require correct side and NBBO present
    pool = [
        it for it in pool
        if _ticker_matches_side(it.get("ticker"), side)
        and it.get("bid") is not None and it.get("ask") is not None
    ]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        return (sd, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

# ---------------------------
# Exports
# ---------------------------
__all__ = [
    "_http_get_any",
    "_http_json",
    "_pull_nbbo_direct",
    "_probe_nbbo_verbose",
    "_poly_reference_contracts_exists",
    "_rescan_best_replacement",
    "_find_nbbo_replacement_same_expiry",
]
