# polygon_ops.py
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import httpx

from engine_runtime import get_http_client

logger = logging.getLogger("trading_engine.polygon_ops")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLY_DEBUG_DUMP = int(os.getenv("POLY_DEBUG_DUMP", "0"))

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

def _extract_nbbo_from_snapshot(js: Dict[str, Any]) -> Dict[str, Any]:
    """Polygon snapshot: results.last_quote may or may not be present depending on entitlement."""
    out: Dict[str, Any] = {}
    try:
        res = js.get("results") or {}
        lq  = res.get("last_quote") or {}
        # polygon sometimes uses bid_price/ask_price; keep fallbacks
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
    """Polygon quotes v3: results is a list, take newest [0]"""
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

async def _pull_nbbo_direct(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    Snapshot-first NBBO fetch; falls back to quotes endpoint.
    Returns any of: bid/ask/mid/option_spread_pct/quote_age_sec + debug statuses.
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    # A) snapshot
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

    # B) quotes (latest) if still missing
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

    # attach statuses for visibility
    out["snapshot_http_status"] = snap_status
    out["quotes_http_status"] = quotes_status
    # heuristic entitlement hint
    if (snap_status in (200, ) and not out.get("bid") and not out.get("ask")) or (quotes_status == 200 and not out.get("bid") and not out.get("ask")):
        out["nbbo_reason"] = "no bid/ask fields returned; likely no options quotes entitlement on this API key"
    return out

async def _probe_nbbo_verbose(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    Verbose probe that ALSO returns nbbo_http_status/nbbo_reason and copies parsed NBBO if found.
    """
    res = await _pull_nbbo_direct(option_ticker, underlying=underlying)
    out: Dict[str, Any] = {
        "nbbo_http_status": (res.get("snapshot_http_status") or res.get("quotes_http_status")),
        "snapshot_http_status": res.get("snapshot_http_status"),
        "quotes_http_status": res.get("quotes_http_status"),
        "nbbo_reason": res.get("nbbo_reason"),
    }
    for k in ("bid","ask","mid","option_spread_pct","quote_age_sec"):
        if res.get(k) is not None:
            out[k] = res[k]
    return out

async def _poly_reference_contracts_exists(underlying: str, expiry_iso: str, ticker: str) -> Dict[str, Any]:
    """
    Check if a contract appears in reference list and whether its snapshot returns results.
    Avoids false-alarms when reference index lags.
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

# Keep stubs for compatibility with imports from engine_processor
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402,403,404,429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

# These names are imported by engine_processor:
# _pull_nbbo_direct, _probe_nbbo_verbose, _poly_reference_contracts_exists
