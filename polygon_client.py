# polygon_client.py
from __future__ import annotations

import os
import asyncio
from typing import Dict, Any, List, Optional
from urllib.parse import quote
from datetime import datetime, timezone
import httpx

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

__all__ = [
    "build_option_contract",
    "list_contracts_for_expiry",
    "get_option_snapshot",
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

def build_option_contract(
    symbol: str,
    expiry_iso: str,   # "YYYY-MM-DD"
    side: str,         # "CALL" | "PUT" or "C" | "P"
    strike: float,
) -> str:
    """
    Build OCC-style ticker used by Polygon:
      O:{SYMBOL}{YY}{MM}{DD}{C|P}{strike*1000:08d}
    Example: O:AAPL250926C00255000
    """
    symbol = (symbol or "").upper().strip()
    side   = (side or "").upper().strip()
    cpor   = "C" if side.startswith("C") else "P"

    # expiry -> YYMMDD
    y, m, d = expiry_iso.split("-")
    yy = y[-2:]
    yymmdd = f"{yy}{m}{d}"

    # strike -> *1000, zero-padded width 8
    k1000 = int(round(float(strike) * 1000))
    strike_part = f"{k1000:08d}"

    return f"O:{symbol}{yymmdd}{cpor}{strike_part}"

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
    client: httpx.AsyncClient,
    symbol: str,
    contract: str,
) -> Dict[str, Any]:
    """
    Return the RAW Polygon Options snapshot JSON:
      /v3/snapshot/options/{symbol}/{contract}

    We DO NOT reshape anything here. Callers should read:
      results.last_quote.bid / ask / last_updated / timeframe
      results.open_interest
      results.day.volume
      results.greeks (if present)
      results.underlying_asset (if present)
    """
    if client is None or not POLYGON_API_KEY:
        return {}

    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}/{contract}"
    r = await client.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=8.0)
    r.raise_for_status()
    js = r.json()
    return js if isinstance(js, dict) else {}

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
