# alert_server.py
"""
FastAPI + Polygon.io (Options) — single-file service

Features
- Async httpx client with keep-alive, connection limits, and timeouts
- Endpoints for options contracts, snapshots (all & one), unusual activity, latest trade/quote
- EMA indicator on the underlying
- Underlying last price fetch + ATM & near-expiry filter
- Simple in-memory TTL cache to reduce rate-limit pressure
- Clean JSON responses ready for your trading logic

ENV (set before running):
  POLYGON_API_KEY=...

Run:
  uvicorn alert_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query

# =========================
# Config & HTTP client
# =========================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("POLYGON_API_KEY is not set in environment.")

BASE_URL = "https://api.polygon.io"

# Create client on startup so we reuse connections
client: Optional[httpx.AsyncClient] = None


# =========================
# Tiny TTL Cache (in-memory)
# =========================
class TTLCache:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str) -> Optional[Any]:
        entry = self.store.get(key)
        if not entry:
            return None
        if self._now() >= entry["exp"]:
            self.store.pop(key, None)
            return None
        return entry["val"]

    def set(self, key: str, val: Any, ttl: int):
        self.store[key] = {"val": val, "exp": self._now() + ttl}


cache = TTLCache()


def cache_key(path: str, params: Dict[str, Any]) -> str:
    # Stable cache key for GET requests
    items = sorted((k, str(v)) for k, v in params.items())
    return f"{path}?{'&'.join([f'{k}={v}' for k,v in items])}"


async def cached_get(path: str, params: Dict[str, Any], ttl: int = 30) -> Dict[str, Any]:
    """GET with TTL cache + auth header attached automatically."""
    assert client is not None, "HTTP client not initialized"
    params = {k: v for k, v in params.items() if v is not None}
    key = cache_key(path, params)
    hit = cache.get(key)
    if hit is not None:
        return hit
    r = await client.get(f"{BASE_URL}{path}", params=params)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    data = r.json()
    cache.set(key, data, ttl)
    return data


# =========================
# Polygon wrappers (async)
# =========================
async def list_option_contracts(
    symbol: str,
    contract_type: Optional[str] = None,      # 'call' | 'put'
    expiration_date: Optional[str] = None,    # 'YYYY-MM-DD'
    strike_price: Optional[float] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    params = {
        "underlying_ticker": symbol.upper(),
        "contract_type": contract_type,
        "expiration_date": expiration_date,
        "strike_price": strike_price,
        "limit": limit,
        # Note: add more filters if needed (e.g., "expired": "false")
    }
    return await cached_get("/v3/reference/options/contracts", params, ttl=60)


async def snapshot_options_all(symbol: str) -> Dict[str, Any]:
    # All active options for underlying (includes greeks, oi, vol, last, etc.)
    return await cached_get(f"/v3/snapshot/options/{symbol.upper()}", {}, ttl=15)


async def snapshot_option_one(symbol: str, option_ticker: str) -> Dict[str, Any]:
    return await cached_get(f"/v3/snapshot/options/{symbol.upper()}/{option_ticker}", {}, ttl=10)


async def unusual_activity(symbol: Optional[str] = None) -> Dict[str, Any]:
    params = {"ticker": symbol.upper() if symbol else None}
    return await cached_get("/v3/unusual_activity/stocks", params, ttl=30)


async def latest_option_trade(option_ticker: str) -> Dict[str, Any]:
    return await cached_get(f"/v3/trades/{option_ticker}/last", {}, ttl=5)


async def latest_option_quote(option_ticker: str) -> Dict[str, Any]:
    return await cached_get(f"/v3/quotes/{option_ticker}/last", {}, ttl=5)


async def ema_indicator(symbol: str, window: int = 9, timespan: str = "minute") -> Dict[str, Any]:
    params = {"window": window, "timespan": timespan}
    return await cached_get(f"/v1/indicators/ema/{symbol.upper()}", params, ttl=15)


async def underlying_last_trade(symbol: str) -> float:
    """
    Get last trade price for the underlying stock.
    Uses /v2/last/trade/{ticker} (stock).
    """
    data = await cached_get(f"/v2/last/trade/{symbol.upper()}", {}, ttl=3)
    # Expected shape: {"results":{"p": <price>, ...}}
    try:
        return float(data["results"]["p"])
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected last trade payload: {data}")


# =========================
# Helpers
# =========================
def filter_near_atm_and_expiry(
    contracts: List[Dict[str, Any]],
    underlying_price: float,
    max_dte: int = 21,
    moneyness_band: float = 0.10,  # +/-10% of underlying
) -> List[Dict[str, Any]]:
    """
    Keep contracts within +/- moneyness_band of spot and expiring within max_dte days.
    contracts: list items shaped like Polygon "reference/options/contracts" results
    """
    today = datetime.utcnow().date()
    out: List[Dict[str, Any]] = []
    for c in contracts:
        try:
            details = c.get("details") or {}
            strike = float(details["strike_price"])
            expiry = datetime.strptime(details["expiration_date"], "%Y-%m-%d").date()
            dte = (expiry - today).days
            if dte <= 0 or dte > max_dte:
                continue
            if underlying_price <= 0:
                continue
            if abs(strike - underlying_price) / underlying_price <= moneyness_band:
                out.append(c)
        except Exception:
            # Skip malformed entries
            continue
    return out


# =========================
# FastAPI app & routes
# =========================
app = FastAPI(title="Polygon Options API (FastAPI single-file)")

@app.on_event("startup")
async def _startup():
    global client
    # Build a single shared HTTP client with auth header
    client = httpx.AsyncClient(
        timeout=10.0,
        headers={"Authorization": f"Bearer {POLYGON_API_KEY}"},
        limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
        http2=False,
    )


@app.on_event("shutdown")
async def _shutdown():
    global client
    if client:
        await client.aclose()
    client = None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "component": "polygon_options_service",
        "time": datetime.utcnow().isoformat() + "Z",
    }


# ---- Reference: list contracts
@app.get("/options/contracts/{symbol}")
async def api_contracts(
    symbol: str,
    contract_type: Optional[str] = Query(None, regex="^(call|put)$"),
    expiration_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    strike_price: Optional[float] = None,
    limit: int = 100,
):
    return await list_option_contracts(symbol, contract_type, expiration_date, strike_price, limit)


# ---- Snapshot: all options for underlying
@app.get("/options/snapshot/{symbol}")
async def api_snapshot_all(symbol: str):
    return await snapshot_options_all(symbol)


# ---- Snapshot: one contract
@app.get("/options/snapshot/{symbol}/{option_ticker}")
async def api_snapshot_one(symbol: str, option_ticker: str):
    return await snapshot_option_one(symbol, option_ticker)


# ---- Unusual activity
@app.get("/options/unusual")
async def api_unusual(symbol: Optional[str] = None):
    return await unusual_activity(symbol)


# ---- Latest trade/quote for a specific option
@app.get("/options/{option_ticker}/last-trade")
async def api_last_trade(option_ticker: str):
    return await latest_option_trade(option_ticker)


@app.get("/options/{option_ticker}/last-quote")
async def api_last_quote(option_ticker: str):
    return await latest_option_quote(option_ticker)


# ---- EMA on underlying
@app.get("/indicators/ema/{symbol}")
async def api_ema(symbol: str, window: int = 9, timespan: str = "minute"):
    return await ema_indicator(symbol, window, timespan)


# ---- Combined helper: fetch contracts → filter ATM & near-expiry
@app.get("/options/near-atm/{symbol}")
async def api_near_atm(
    symbol: str,
    max_dte: int = 21,
    moneyness_band: float = 0.10,
    contract_type: Optional[str] = Query(None, regex="^(call|put)$"),
    # Optional: if you already know spot, pass it to skip an extra request
    spot_hint: Optional[float] = None,
    limit: int = 500,
):
    """
    Returns:
      {
        "underlying_price": float,
        "count": int,
        "results": [contracts...]
      }
    """
    # 1) Spot
    spot = float(spot_hint) if spot_hint is not None else await underlying_last_trade(symbol)

    # 2) Contracts (reference endpoint is lighter than full snapshot, good for filtering)
    contracts_resp = await list_option_contracts(symbol, contract_type=contract_type, limit=limit)
    contracts = contracts_resp.get("results") or []

    # 3) Filter ATM & near-expiry
    filtered = filter_near_atm_and_expiry(contracts, spot, max_dte=max_dte, moneyness_band=moneyness_band)

    return {
        "underlying_price": spot,
        "count": len(filtered),
        "results": filtered,
    }
