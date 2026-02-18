# polygon_client.py
import os
import logging
from typing import Dict, Any, List, Optional
import httpx

log = logging.getLogger("trading_engine.data")

POLYGON_API_KEY = (os.getenv("POLYGON_API_KEY", "") or "").strip()
POLYGON_BASE_URL = (os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/"))

MASSIVE_API_KEY = (os.getenv("MASSIVE_API_KEY", "") or "").strip()
MASSIVE_BASE_URL = (os.getenv("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/"))

def polygon_enabled() -> bool:
    return bool(POLYGON_API_KEY)

def massive_enabled() -> bool:
    return bool(MASSIVE_API_KEY)

def _is_entitlement_error(js: Dict[str, Any]) -> bool:
    msg = str(js.get("message") or "").lower()
    status = str(js.get("status") or "").lower()
    return ("not_authorized" in status) or ("not entitled" in msg) or ("upgrade your plan" in msg)

class PolygonClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.enabled = polygon_enabled()

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 8.0) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        p = dict(params or {})
        p["apiKey"] = POLYGON_API_KEY
        url = f"{POLYGON_BASE_URL}{path}"
        r = await self.http.get(url, params=p, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # Keep your existing Polygon methods here (get_last_trade, get_last_quote, get_stock_snapshot, get_aggregates, etc)
    # If any endpoint is not entitled, it will throw HTTPStatusError before returning JSON.
    # That’s fine because engine_processor now falls back to Massive in hybrid mode.

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        return await self._get(f"/v2/last/trade/{symbol}")

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        return await self._get(f"/v2/last/nbbo/{symbol}")

    async def get_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        # This is the one you are NOT entitled to (403). Keep it but expect fallback.
        return await self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")

    async def get_aggregates(self, symbol: str, *, multiplier: int, timespan: str, limit: int = 500) -> List[Dict[str, Any]]:
        # Best-effort: for your usage you only rely on results list
        # Using a generic recent range is ok; your Massive client uses lookback_days explicitly.
        # Here keep it simple; you can restore your original date-window logic if already present.
        js = await self._get(f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/2020-01-01/2099-12-31", params={"limit": limit})
        return (js.get("results") or []) if isinstance(js, dict) else []

    async def get_technicals_bundle(self, symbol: str) -> Dict[str, Any]:
        # Keep your existing method (or return {} if you don’t have it implemented)
        return {}

    async def get_targeted_option_context(self, symbol: str, *, expiry_iso: str, side: str, strike: float) -> Dict[str, Any]:
        # Keep your existing method (or return {} if you don’t have it implemented)
        return {}


class MassiveClient:
    """
    Minimal Massive client for your use-cases:
      - last trade
      - last quote
      - aggregates (minute/hour/day)
      - targeted option context (best-effort; depends on your Massive plan)
    """
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        self.enabled = massive_enabled()

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 8.0) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        p = dict(params or {})
        p["apiKey"] = MASSIVE_API_KEY
        url = f"{MASSIVE_BASE_URL}{path}"
        r = await self.http.get(url, params=p, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else {}

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        # Example (you already tried):
        # GET /v2/last/trade/NVDA?apiKey=...
        js = await self._get(f"/v2/last/trade/{symbol}")
        # Normalize
        # Try common shapes:
        # { "symbol":"NVDA","price":..., "timestamp":... } OR { "results": { "p": ... } }
        if "price" in js:
            return {"price": js.get("price"), "ts": js.get("timestamp")}
        res = js.get("results") or js.get("result") or {}
        return {"price": res.get("p") or res.get("price"), "ts": res.get("t")}

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        # If Massive supports it; otherwise return {}
        js = await self._get(f"/v2/last/quote/{symbol}")
        if "bid" in js or "ask" in js:
            return {"bid": js.get("bid"), "ask": js.get("ask"), "ts": js.get("timestamp")}
        res = js.get("results") or js.get("result") or {}
        return {"bid": res.get("bp") or res.get("bid"), "ask": res.get("ap") or res.get("ask"), "ts": res.get("t")}

    async def get_aggs(self, symbol: str, *, multiplier: int, timespan: str, lookback_days: int, limit: int) -> List[Dict[str, Any]]:
        """
        Massive aggs endpoint differs by vendor. This is a best-effort adapter:
        - If Massive provides /v2/aggs/ticker/... similar to Polygon, great.
        - If not, replace this with Massive's documented bars endpoint.
        """
        # Try Polygon-like first:
        # /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}?limit=...
        # Use a simple rolling window.
        from_dt = (date.today() - timedelta(days=max(1, int(lookback_days)))).isoformat()
        to_dt = (date.today() + timedelta(days=1)).isoformat()

        try:
            js = await self._get(
                f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_dt}/{to_dt}",
                params={"limit": limit},
            )
            results = js.get("results") or []
            if isinstance(results, list):
                return results
        except Exception:
            pass

        # If Massive doesn't have that endpoint, return empty for now.
        return []

    async def get_targeted_option_context(self, symbol: str, *, expiry_iso: str, side: str, strike: float) -> Dict[str, Any]:
        """
        If your plan includes options snapshot/chain endpoint, implement it here.
        If not, return {} and system will treat as equity-only.
        """
        # Placeholder: return {}
        return {}
