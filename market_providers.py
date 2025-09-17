# market_providers.py
import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

import httpx

log = logging.getLogger("trading_engine.market_providers")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ------------ Env ------------
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLY_HTTP_TIMEOUT = float(os.getenv("POLY_HTTP_TIMEOUT", "8.0"))
POLY_MAX_RETRIES = int(os.getenv("POLY_MAX_RETRIES", "3"))
POLY_RETRY_BASE = float(os.getenv("POLY_RETRY_BASE", "0.20"))  # seconds
POLY_RETRY_JIT  = float(os.getenv("POLY_RETRY_JIT",  "0.20"))  # seconds

def _iso(dt: Union[str, datetime]) -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    raise TypeError(f"Unsupported date type: {type(dt)}")

def _normalize_aggs(results: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(results, list):
        return out
    for b in results:
        if not isinstance(b, dict):
            continue
        out.append({
            "t": b.get("t"),  # ms since epoch
            "o": b.get("o"),
            "h": b.get("h"),
            "l": b.get("l"),
            "c": b.get("c"),
            "v": b.get("v"),
        })
    return out

async def _poly_aggs(
    client: httpx.AsyncClient,
    symbol: str,
    mult: int,
    span: str,   # "minute" or "day"
    start: Union[str, datetime],
    end:   Union[str, datetime],
    limit: int = 50000,
) -> List[Dict[str, Any]]:
    """Fetch bars from Polygon v2 aggs; normalized to [{t,o,h,l,c,v}, ...]."""
    if not POLYGON_API_KEY:
        log.debug("Polygon disabled: no POLYGON_API_KEY")
        return []

    frm = _iso(start)
    to  = _iso(end)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{frm}/{to}"
    params = {"adjusted": "true", "sort": "asc", "limit": int(limit), "apiKey": POLYGON_API_KEY}

    last_err = None
    for i in range(POLY_MAX_RETRIES):
        try:
            r = await client.get(url, params=params, timeout=POLY_HTTP_TIMEOUT)
            # Surface common rate-limit & auth statuses so callers see why
            if r.status_code in (402, 403, 404, 429):
                body = (r.text or "")[:300]
                raise httpx.HTTPStatusError(
                    f"Polygon {r.status_code}: {body}", request=r.request, response=r
                )
            r.raise_for_status()
            js = r.json()
            return _normalize_aggs(js.get("results"))
        except Exception as e:
            last_err = e
            # Backoff on transient errors
            await asyncio.sleep(POLY_RETRY_BASE + (POLY_RETRY_JIT * i))
    log.warning("[market_providers] polygon aggs failed for %s %s/%s: %r", symbol, mult, span, last_err)
    return []

async def _ensure_client(client: Optional[httpx.AsyncClient]) -> (httpx.AsyncClient, bool):
    if client is not None:
        return client, False
    c = httpx.AsyncClient(timeout=POLY_HTTP_TIMEOUT)
    return c, True

# ------------ Public API (used by feature_engine.py) ------------

async def fetch_1m_bars_any(
    client: Optional[httpx.AsyncClient],
    symbol: str,
    start: Union[str, datetime],
    end:   Union[str, datetime],
    limit: int = 50000,
) -> List[Dict[str, Any]]:
    """
    Return 1-minute bars for equity symbol as [{t,o,h,l,c,v}, ...].
    Provider order: Polygon (more can be added later).
    """
    c, ephemeral = await _ensure_client(client)
    try:
        bars = await _poly_aggs(c, symbol, 1, "minute", start, end, limit)
        if bars:
            return bars
        # TODO: add Alpaca / other fallbacks here
        return []
    finally:
        if ephemeral:
            await c.aclose()

async def fetch_5m_bars_any(
    client: Optional[httpx.AsyncClient],
    symbol: str,
    start: Union[str, datetime],
    end:   Union[str, datetime],
    limit: int = 50000,
) -> List[Dict[str, Any]]:
    c, ephemeral = await _ensure_client(client)
    try:
        bars = await _poly_aggs(c, symbol, 5, "minute", start, end, limit)
        if bars:
            return bars
        # TODO: add Alpaca / other fallbacks here
        return []
    finally:
        if ephemeral:
            await c.aclose()

async def fetch_1d_bars_any(
    client: Optional[httpx.AsyncClient],
    symbol: str,
    start: Union[str, datetime],
    end:   Union[str, datetime],
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    c, ephemeral = await _ensure_client(client)
    try:
        bars = await _poly_aggs(c, symbol, 1, "day", start, end, limit)
        if bars:
            return bars
        # TODO: add Alpaca / other fallbacks here
        return []
    finally:
        if ephemeral:
            await c.aclose()

# Optional helpers so engine_processor._try_multi_provider_nbbo() can import safely.
def synthetic_from_last(last: Optional[float]) -> Dict[str, Any]:
    if not isinstance(last, (int, float)) or last <= 0:
        return {}
    # 1% synthetic spread around last
    half = 0.01 / 2.0
    bid = last * (1 - half)
    ask = last * (1 + half)
    mid = (bid + ask) / 2.0
    return {
        "provider": "synthetic(last)",
        "bid": round(bid, 4),
        "ask": round(ask, 4),
        "mid": round(mid, 4),
        "spread_pct": round((ask - bid) / mid * 100.0, 3),
        "synthetic_nbbo_used": True,
        "synthetic_nbbo_spread_est": 1.0,
    }

async def get_nbbo_any(option_ticker: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stub for compatibility. Return None so the caller can fall back.
    Implement real NBBO providers here (IBKR / Tradier / E*TRADE) if desired.
    """
    return None

__all__ = [
    "fetch_1m_bars_any",
    "fetch_5m_bars_any",
    "fetch_1d_bars_any",
    "synthetic_from_last",
    "get_nbbo_any",
]
