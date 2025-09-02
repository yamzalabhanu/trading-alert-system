# polygon_client.py
import os
import asyncio
from typing import Dict, Any, List, Optional
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

async def get_option_snapshot(
    symbol: str,
    contract: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, Any]:
    """
    Snapshot with normalized fields. Signature kept backward-compatible:
    you can call with (symbol, contract) OR pass a shared client.
    Returns keys: iv, oi, volume, bid, ask, mid, quote_age_sec, delta, gamma, theta, vega, _raw
    """
    owned_client = False
    if client is None:
        client = get_shared_client()
        owned_client = True

    try:
        js = await _poly_get(client, f"/v3/snapshot/options/{symbol}/{contract}", {})
        res = (js or {}).get("results") or {}

        greeks = res.get("greeks") or {}
        last_quote = res.get("last_quote") or {}
        bid = (last_quote.get("bid") or {}).get("price")
        ask = (last_quote.get("ask") or {}).get("price")
        mid = round((bid + ask) / 2, 4) if (bid is not None and ask is not None) else None

        iv = _coalesce(res.get("implied_volatility"), res.get("iv"), default=None)
        oi = _coalesce(res.get("open_interest"), res.get("oi"), default=None)
        vol = _coalesce((res.get("day") or {}).get("volume"), res.get("volume"), default=None)
        quote_age = _coalesce(last_quote.get("last_updated"), last_quote.get("sip_timestamp"), default=None)

        quote_age_sec = None
        if isinstance(quote_age, (int, float)):
            q = float(quote_age)
            if q > 1e14:
                quote_age_sec = round(q / 1e9, 3)   # ns -> s
            elif q > 1e11:
                quote_age_sec = round(q / 1e6, 3)   # us -> s
            elif q > 1e8:
                quote_age_sec = round(q / 1e3, 3)   # ms -> s
            else:
                quote_age_sec = q

        return {
            "iv": iv,
            "oi": oi,
            "volume": vol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bid_size": (last_quote.get("bid") or {}).get("size"),
            "ask_size": (last_quote.get("ask") or {}).get("size"),
            "quote_age_sec": quote_age_sec,
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "vega": greeks.get("vega"),
            "_raw": res,
        }
    finally:
        if owned_client:
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
