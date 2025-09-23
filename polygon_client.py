# polygon_client.py
import os
import asyncio
from typing import Dict, Any, List, Optional
from urllib.parse import quote
from datetime import datetime, timezone
import httpx

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# What this module publicly exports
__all__ = [
    "list_contracts_for_expiry",
    "get_option_snapshot",
    "build_option_contract",
    "get_aggs",
    "get_shared_client",
]

# ---------- HTTP helpers ----------

_DEFAULT_TIMEOUT = httpx.Timeout(read=6.0, write=6.0, connect=3.0, pool=3.0)
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

async def _sleep_backoff(attempt: int) -> None:
    await asyncio.sleep((0.1 * (2 ** attempt)) + (0.05 * attempt))

async def _poly_get(
    client: httpx.AsyncClient,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    max_retries: int = 3,
) -> Dict[str, Any]:
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY or ""
    for attempt in range(max_retries + 1):
        try:
            r = await client.get(f"https://api.polygon.io{path}", params=p, timeout=_DEFAULT_TIMEOUT)
            if r.status_code in (402, 403, 404):
                return {}
            if r.status_code in _RETRY_STATUSES and attempt < max_retries:
                await _sleep_backoff(attempt)
                continue
            r.raise_for_status()
            return r.json() or {}
        except httpx.HTTPError:
            if attempt < max_retries:
                await _sleep_backoff(attempt)
                continue
            return {}
        except Exception:
            return {}

# ---------- option helpers ----------

def build_option_contract(ticker: str, expiry_yyyy_mm_dd: str, side: str, strike: float) -> str:
    """
    OCC format with Polygon 'O:' prefix:
      O:<TICKER><YYMMDD><C/P><STRIKE*1000, zero-padded 8>
    Example: 12.5 -> 00012500
    """
    yy = expiry_yyyy_mm_dd[2:4]
    mm = expiry_yyyy_mm_dd[5:7]
    dd = expiry_yyyy_mm_dd[8:10]
    cp = "C" if side.upper().startswith("C") else "P"
    strike_int = int(round(float(strike) * 1000))
    strike_part = f"{strike_int:08d}"
    return f"O:{ticker.upper()}{yy}{mm}{dd}{cp}{strike_part}"

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

# ---------- client factory ----------

def get_shared_client() -> httpx.AsyncClient:
    """Create a reusable AsyncClient. (Call .aclose() on shutdown.)"""
    return httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

# ---------- PUBLIC API ----------

async def list_contracts_for_expiry(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    expiry: str,
    side: str,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    """List contracts for a given expiry (paginates by cursor)."""
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while len(out) < limit:
        params = {
            "underlying_ticker": symbol,
            "expiration_date": expiry,  # YYYY-MM-DD
            "contract_type": "call" if side.upper().startswith("C") else "put",
            "limit": min(1000, limit - len(out)),
        }
        if cursor:
            params["cursor"] = cursor

        js = await _poly_get(client, "/v3/reference/options/contracts", params)
        results = (js or {}).get("results") or []
        out.extend(results)
        # Polygon returns 'next_url' or 'cursor' depending on path; handle both
        cursor = (js or {}).get("cursor") or (js or {}).get("next_url")
        if not cursor or not results:
            break

    return out[:limit]

def _encode_ticker_path(t: str) -> str:
    return quote(t or "", safe="")

def _age_sec(ts_val: Any) -> Optional[float]:
    if ts_val is None:
        return None
    try:
        ns = int(ts_val)
        if   ns >= 10**14: sec = ns / 1e9
        elif ns >= 10**11: sec = ns / 1e6
        elif ns >= 10**8:  sec = ns / 1e3
        else:              sec = float(ns)
        return max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    except Exception:
        return None

async def get_option_snapshot(
    client: httpx.AsyncClient,
    symbol: str,
    contract: str,
) -> Dict[str, Any]:
    """
    Return the raw Polygon Options Advanced snapshot payload for a single contract.
    Do NOT reshape fields here; callers should read `results.last_quote`, `results.day`,
    `results.open_interest`, `results.greeks`, etc., as provided by Polygon.

    Endpoint:
      /v3/snapshot/options/{underlying}/{contract}
    """
    if client is None or not POLYGON_API_KEY:
        return {}

    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}/{contract}"
    r = await client.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=8.0)
    # Let 4xx/5xx raise; upstream will catch and handle
    r.raise_for_status()
    js = r.json()
    return js if isinstance(js, dict) else {}

        # Day + OI
        day = res.get("day") or {}
        vol = day.get("volume") or day.get("v")
        oi = res.get("open_interest")

        # Greeks / IV (varies by entitlement)
        greeks = res.get("greeks") or {}
        iv = res.get("implied_volatility") or greeks.get("iv")

        # Last quote: flat or nested
        lq = res.get("last_quote") or res.get("lastQuote") or {}
        # Try flat first
        bid = lq.get("bid")
        ask = lq.get("ask")
        last_updated = lq.get("last_updated")
        timeframe = lq.get("timeframe") or res.get("timeframe")

        # Fallback to nested shapes
        if bid is None:
            bid = (lq.get("bid_price") if isinstance(lq.get("bid_price"), (int,float)) else None)
        if ask is None:
            ask = (lq.get("ask_price") if isinstance(lq.get("ask_price"), (int,float)) else None)
        if bid is None and isinstance(lq.get("bid"), dict):
            bid = lq.get("bid", {}).get("price")
        if ask is None and isinstance(lq.get("ask"), dict):
            ask = lq.get("ask", {}).get("price")
        if last_updated is None:
            last_updated = lq.get("sip_timestamp") or lq.get("participant_timestamp") or lq.get("t")

        # Derive mid/spread/age
        mid = None
        spread_pct = None
        if isinstance(bid, (int,float)) and isinstance(ask, (int,float)) and ask >= 0 and bid >= 0:
            mid = (bid + ask) / 2.0
            if mid and mid > 0:
                spread_pct = (ask - bid) / mid * 100.0
        age = _age_sec(last_updated)

        out = {
            "results": res,  # keep raw for callers that expect it
            # normalized fields your engine expects
            "bid": float(bid) if isinstance(bid, (int,float)) else None,
            "ask": float(ask) if isinstance(ask, (int,float)) else None,
            "mid": round(float(mid), 4) if isinstance(mid, (int,float)) else None,
            "option_spread_pct": round(float(spread_pct), 3) if isinstance(spread_pct, (int,float)) else None,
            "quote_age_sec": float(age) if isinstance(age, (int,float)) else None,
            "nbbo_provider": "polygon:snapshot",
            "timeframe": timeframe or "UNKNOWN",
            "vol": int(vol) if isinstance(vol, (int,float)) else None,
            "oi": int(oi) if isinstance(oi, (int,float)) else None,
        }

        # Attach greeks/iv if present
        for k in ("delta","gamma","theta","vega"):
            if isinstance(greeks.get(k), (int,float)):
                out[k] = greeks[k]
        if isinstance(iv, (int,float)):
            out["iv"] = iv

        return out
    finally:
        if close_client:
            await client.aclose()


async def get_aggs(
    client: httpx.AsyncClient,
    *,
    ticker: str,
    multiplier: int,
    timespan: str,
    frm: str,
    to: str,
    limit: int = 50000,
    sort: str = "asc",
) -> List[Dict[str, Any]]:
    """Return a list of bars (empty list on error)."""
    js = await _poly_get(
        client,
        f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{frm}/{to}",
        {"adjusted": "true", "sort": sort, "limit": limit},
    )
    return (js or {}).get("results") or []
