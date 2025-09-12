# polygon_ops.py
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import httpx

from engine_runtime import get_http_client
from engine_common import (
    POLYGON_API_KEY,
    _encode_ticker_path, _ticker_matches_side,
)
from market_ops import scan_top_candidates_for_alert

logger = logging.getLogger("trading_engine")

async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
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

async def _pull_nbbo_direct(option_ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    client = get_http_client()
    if not POLYGON_API_KEY or client is None:
        return out
    try:
        enc = _encode_ticker_path(option_ticker)
        lastq = await _http_json(
            client,
            f"https://api.polygon.io/v3/quotes/options/{enc}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=4.0
        )
        if not lastq:
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

    enc = _encode_ticker_path(option_ticker)
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

        enc = _encode_ticker_path(ticker)
        s = await client.get(
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        snapshot_ok = (s.status_code == 200 and isinstance((s.json() or {}).get("results"), dict))
        return {"listed": listed, "snapshot_ok": snapshot_ok, "reason": None}
    except Exception as e:
        return {"listed": None, "snapshot_ok": None, "reason": f"error: {type(e).__name__}: {e}"}

async def _rescan_best_replacement(
    symbol: str, side: str, desired_strike: float, expiry_iso: str, min_vol: int, min_oi: int,
) -> Optional[Dict[str, Any]]:
    try:
        try:
            top_same = await scan_top_candidates_for_alert(
                get_http_client(), symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=10,
                restrict_expiries=[expiry_iso],  # type: ignore
            )
            pool = top_same or []
        except TypeError:
            top_any = await scan_top_candidates_for_alert(
                get_http_client(), symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol, min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=15,
            )
            pool = [it for it in (top_any or []) if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side)]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        nbbo_missing = 0 if (it.get("bid") is not None and it.get("ask") is not None) else 1
        return (sd, nbbo_missing, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

async def _find_nbbo_replacement_same_expiry(
    symbol: str, side: str, desired_strike: float, expiry_iso: str, min_vol: int, min_oi: int,
) -> Optional[Dict[str, Any]]:
    try:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=10,
            restrict_expiries=[expiry_iso],  # type: ignore
        ) or []
    except TypeError:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol, min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=12,
        ) or []
        pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side)
            and it.get("bid") is not None and it.get("ask") is not None]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        return (sd, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

__all__ = [
    "_http_json", "_http_get_any",
    "_pull_nbbo_direct", "_probe_nbbo_verbose",
    "_poly_reference_contracts_exists",
    "_rescan_best_replacement", "_find_nbbo_replacement_same_expiry",
]
