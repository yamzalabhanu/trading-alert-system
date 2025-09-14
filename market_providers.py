# market_providers.py
import os
import re
import math
import json
import pytz
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY

logger = logging.getLogger("trading_engine.providers")

# ---------- small utils ----------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

async def _http_json(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: float = 10.0):
    client = get_http_client()
    if client is None:
        return None
    params = params or {}
    headers = headers or {}
    try:
        r = await client.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        return js
    except Exception as e:
        logger.warning("[providers] http fail %s %r", url, e)
        return None

def _occ_parse(option_ticker: str) -> Dict[str, Any]:
    """
    Parse OCC ticker like O:AAPL250919C00185000 -> symbol=AAPL, expiry=20250919, right=C, strike=185.0
    """
    m = re.search(r"O:([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", option_ticker)
    if not m:
        return {}
    sym = m.group(1)
    yy, mm, dd = m.group(2), m.group(3), m.group(4)
    right = m.group(5)
    strike_int = m.group(6)
    # OCC strike has 3 decimals (e.g., 00185000 -> 185.000)
    strike = int(strike_int) / 1000.0
    expiry = f"20{yy}{mm}{dd}"  # yyyymmdd
    return {"symbol": sym, "expiry_yyyymmdd": expiry, "right": right, "strike": strike}

# =============================================================================
# NBBO providers
# =============================================================================

async def polygon_nbbo_last(option_ticker: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        return {}
    enc = option_ticker
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    js = await _http_json(url, {"apiKey": POLYGON_API_KEY})
    # Expected shape: { status, results: { last: { bidPrice, askPrice, price, sipTimestamp, ... } } }
    try:
        last = ((js or {}).get("results") or {}).get("last") or {}
        bid = last.get("bidPrice"); ask = last.get("askPrice"); last_px = last.get("price")
        out = {
            "bid": float(bid) if isinstance(bid, (int, float)) else None,
            "ask": float(ask) if isinstance(ask, (int, float)) else None,
            "last": float(last_px) if isinstance(last_px, (int, float)) else None,
            "provider": "polygon",
            "raw": js,
        }
        if out["bid"] is not None and out["ask"] is not None:
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            out["spread_pct"] = round(((out["ask"] - out["bid"]) / mid) * 100.0, 3) if mid > 0 else None
        return out
    except Exception:
        return {}

# --- Tradier NBBO (REST) ---
async def tradier_nbbo(option_ticker: str) -> Dict[str, Any]:
    token = os.getenv("TRADIER_TOKEN")
    if not token:
        return {}
    url = "https://api.tradier.com/v1/markets/options/quotes"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    js = await _http_json(url, params={"symbols": option_ticker, "greeks": "true"}, headers=headers)
    try:
        qt = (((js or {}).get("quotes") or {}).get("quote") or {})
        # If Tradier returns a list, pick first
        if isinstance(qt, list) and qt:
            qt = qt[0]
        bid = qt.get("bid"); ask = qt.get("ask"); last_px = qt.get("last")
        out = {
            "bid": float(bid) if isinstance(bid, (int, float)) else None,
            "ask": float(ask) if isinstance(ask, (int, float)) else None,
            "last": float(last_px) if isinstance(last_px, (int, float)) else None,
            "provider": "tradier",
            "raw": js,
        }
        if out["bid"] is not None and out["ask"] is not None:
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            out["spread_pct"] = round(((out["ask"] - out["bid"]) / mid) * 100.0, 3) if mid > 0 else None
        return out
    except Exception:
        return {}

# --- IBKR (ib_insync) snapshot NBBO ---
async def ibkr_nbbo(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    ctx needs: symbol, right ('C'/'P'), strike, expiry_yyyymmdd
    """
    try:
        from ib_insync import IB, Option
    except Exception:
        return {}
    symbol = ctx.get("symbol")
    right = ctx.get("right")
    strike = ctx.get("strike")
    expiry = ctx.get("expiry_yyyymmdd")
    if not all([symbol, right, strike, expiry]):
        return {}
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "17"))

    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=client_id, readonly=True)
        contract = Option(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                          strike=float(strike), right=right, exchange='SMART', currency='USD')
        ticker = ib.reqMktData(contract, genericTickList='', snapshot=True)
        await ib.sleep(1.5)
        bid = ticker.bid if ticker.bid is not None else None
        ask = ticker.ask if ticker.ask is not None else None
        last = ticker.last if ticker.last is not None else None
        out = {"bid": bid, "ask": ask, "last": last, "provider": "ibkr"}
        if out["bid"] is not None and out["ask"] is not None:
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            out["spread_pct"] = round(((out["ask"] - out["bid"]) / mid) * 100.0, 3) if mid > 0 else None
        return out
    except Exception as e:
        logger.warning("[providers] ibkr nbbo fail: %r", e)
        return {}
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

def synthetic_from_last(last: Optional[float], spread_pct: float = None) -> Dict[str, Any]:
    if not isinstance(last, (int, float)) or last <= 0:
        return {}
    sp = float(spread_pct if spread_pct is not None else os.getenv("SYNTH_SPREAD_PCT", "1.0"))
    half = sp / 200.0
    bid = last * (1 - half / 100.0)
    ask = last * (1 + half / 100.0)
    mid = (bid + ask) / 2.0
    return {
        "bid": round(bid, 4),
        "ask": round(ask, 4),
        "mid": round(mid, 4),
        "spread_pct": round(((ask - bid) / mid) * 100.0, 3) if mid > 0 else sp,
        "synthetic_nbbo_used": True,
        "synthetic_nbbo_spread_est": sp,
        "provider": "synthetic",
    }

async def get_nbbo_any(option_ticker: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try Polygon → IBKR → Tradier → Synthetic (from last or ctx['last'])
    """
    # 1) Polygon
    nb = await polygon_nbbo_last(option_ticker)
    if (nb.get("bid") is not None) or (nb.get("ask") is not None):
        return nb

    # fill ctx if missing using OCC
    occ = _occ_parse(option_ticker)
    merged = {**occ, **ctx}

    # 2) IBKR
    ib = await ibkr_nbbo(merged)
    if (ib.get("bid") is not None) or (ib.get("ask") is not None):
        return ib

    # 3) Tradier
    tr = await tradier_nbbo(option_ticker)
    if (tr.get("bid") is not None) or (tr.get("ask") is not None):
        return tr

    # 4) Synthetic
    last = nb.get("last") or ib.get("last") or tr.get("last") or merged.get("last")
    syn = synthetic_from_last(last)
    return syn or {"provider": "none"}

# =============================================================================
# Bars providers (for VWAP/ORB/RSI etc.)
# =============================================================================

# --- IEX Cloud intraday (minute) ---
async def iex_bars_1m(symbol: str) -> List[Dict[str, Any]]:
    token = os.getenv("IEX_TOKEN")
    if not token:
        return []
    url = f"https://cloud.iexapis.com/stable/stock/{symbol}/intraday-prices"
    js = await _http_json(url, {"token": token})
    if not isinstance(js, list):
        return []
    # Convert to our schema with epoch ms (IEX is US/Eastern)
    est = pytz.timezone("US/Eastern")
    out: List[Dict[str, Any]] = []
    for row in js:
        try:
            d = row.get("date")  # '2025-09-13'
            m = row.get("minute")  # '09:30'
            if not d or not m:
                continue
            dt_est = est.localize(datetime.fromisoformat(f"{d}T{m}:00"))
            t = int(dt_est.astimezone(timezone.utc).timestamp() * 1000)
            o = row.get("open"); h=row.get("high"); l=row.get("low"); c=row.get("close"); v=row.get("volume")
            vw = row.get("vwap")
            out.append({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "vw": vw})
        except Exception:
            continue
    return out

# --- Alpaca (minute or day) ---
async def alpaca_bars(symbol: str, timeframe: str = "1Min", start_iso: Optional[str] = None, end_iso: Optional[str] = None, limit: int = 1500) -> List[Dict[str, Any]]:
    key = os.getenv("ALPACA_KEY"); sec = os.getenv("ALPACA_SECRET")
    if not key or not sec:
        return []
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": str(limit)}
    if start_iso: params["start"] = start_iso
    if end_iso: params["end"] = end_iso
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    js = await _http_json(url, params, headers=headers)
    bars = (js or {}).get("bars") or []
    out: List[Dict[str, Any]] = []
    for b in bars:
        try:
            # b.t is ISO8601
            dt = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            t = int(dt.timestamp() * 1000)
            out.append({"t": t, "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"), "v": b.get("v"), "vw": b.get("vw")})
        except Exception:
            continue
    return out

# Unified helpers for feature_engine
async def fetch_1m_bars_any(symbol: str) -> List[Dict[str, Any]]:
    # Try IEX first (cheap), then Alpaca
    bars = await iex_bars_1m(symbol)
    if bars:
        return bars
    # last resort Alpaca (no explicit start -> vendor default recent window)
    bars = await alpaca_bars(symbol, timeframe="1Min", limit=1500)
    return bars or []

async def fetch_5m_bars_any(symbol: str) -> List[Dict[str, Any]]:
    # Alpaca supports 5Min directly
    bars = await alpaca_bars(symbol, timeframe="5Min", limit=1500)
    return bars or []

async def fetch_1d_bars_any(symbol: str) -> List[Dict[str, Any]]:
    bars = await alpaca_bars(symbol, timeframe="1Day", limit=1500)
    return bars or []

__all__ = [
    "get_nbbo_any", "synthetic_from_last",
    "fetch_1m_bars_any", "fetch_5m_bars_any", "fetch_1d_bars_any",
]
