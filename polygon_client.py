# polygon_client.py
import httpx
from typing import Dict, Any, List
from config import POLYGON_API_KEY

async def _poly_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY
    r = await client.get(f"https://api.polygon.io{path}", params=p, timeout=20.0)
    if r.status_code in (402, 403, 404, 429):
        return {}
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {}

async def list_contracts_for_expiry(client: httpx.AsyncClient, *, symbol: str, expiry: str, side: str, limit: int = 250) -> List[Dict[str, Any]]:
    js = await _poly_get(client, "/v3/reference/options/contracts", {
        "underlying_ticker": symbol,
        "expiration_date": expiry,
        "contract_type": "call" if side == "CALL" else "put",
        "limit": limit,
    })
    return (js or {}).get("results", []) or []



async def get_option_snapshot(symbol: str, contract: str):
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}/{contract}"
    params = {"apiKey": POLYGON_API_KEY}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("results", {})

async def get_aggs(client: httpx.AsyncClient, *, ticker: str, multiplier: int, timespan: str,
                   frm: str, to: str, limit: int = 50000, sort: str = "asc") -> List[Dict[str, Any]]:
    js = await _poly_get(client, f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{frm}/{to}", {
        "adjusted": "true",
        "sort": sort,
        "limit": limit,
    })
    return (js or {}).get("results", []) or []
