import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from config import POLYGON_API_KEY
from engine_common import build_option_contract

logger = logging.getLogger("trading_engine")


class PolygonClient:
    """Thin async Polygon.io REST client for stocks, technicals, and options snapshots."""

    def __init__(self, http_client: httpx.AsyncClient, api_key: Optional[str] = None) -> None:
        self.http = http_client
        self.api_key = api_key or POLYGON_API_KEY
        self.base = "https://api.polygon.io"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 6.0) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("POLYGON_API_KEY is not set")
        q = dict(params or {})
        q["apiKey"] = self.api_key
        url = f"{self.base}{path}"
        try:
            r = await self.http.get(url, params=q, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            # Log response body (truncated) for fast debugging
            try:
                body = e.response.text
            except Exception:
                body = "<no body>"
            logger.warning("[polygon] %s %s -> %s %s", "GET", url, e.response.status_code, (body or "")[:500])
            raise
        except Exception as e:
            logger.warning("[polygon] GET failed %s: %r", url, e)
            raise

    async def get_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        js = await self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol.upper()}", timeout=6.0)
        return js.get("ticker") or {}

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        js = await self._get(f"/v2/last/nbbo/{symbol.upper()}", timeout=6.0)
        return js.get("results") or {}

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        js = await self._get(f"/v2/last/trade/{symbol.upper()}", timeout=6.0)
        return js.get("results") or {}

    # -------------------------------------------------------------------------
    # Aggregates
    # -------------------------------------------------------------------------
    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 5,
        timespan: str = "minute",
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible wide-window aggregate fetch.
        Kept for existing callers (e.g., 5m bars).
        """
        js = await self._get(
            f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/2020-01-01/2100-01-01",
            params={"adjusted": "true", "sort": "asc", "limit": limit},
            timeout=8.0,
        )
        return js.get("results") or []

    async def get_aggs_window(
        self,
        symbol: str,
        *,
        multiplier: int,
        timespan: str,
        from_: str,
        to: str,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Preferred aggregates fetch for bounded windows.
        Used for MTF context (daily/hourly/15m) so we don't pull huge ranges.
        from_/to are YYYY-MM-DD.
        """
        js = await self._get(
            f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_}/{to}",
            params={"adjusted": "true", "sort": "asc", "limit": limit},
            timeout=10.0,
        )
        return js.get("results") or []

    # -------------------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------------------
    async def get_indicator(
        self,
        kind: str,
        symbol: str,
        *,
        timespan: str = "minute",
        window: int = 14,
        series_type: str = "close",
        limit: int = 200,
        macd_short: int = 12,
        macd_long: int = 26,
        macd_signal: int = 9,
        field: str = "value",
    ) -> Optional[float]:
        path = f"/v1/indicators/{kind}/{symbol.upper()}"
        params: Dict[str, Any] = {
            "timespan": timespan,
            "window": window,
            "series_type": series_type,
            "order": "desc",
            "limit": limit,
            "adjusted": "true",
        }
        if kind == "macd":
            params.update(
                {
                    "short_window": macd_short,
                    "long_window": macd_long,
                    "signal_window": macd_signal,
                }
            )
        js = await self._get(path, params=params, timeout=8.0)
        values = (((js.get("results") or {}).get("values")) or [])
        if not values:
            return None
        item = values[0]
        val = item.get(field)
        return float(val) if isinstance(val, (int, float)) else None

    async def get_technicals_bundle(self, symbol: str) -> Dict[str, Optional[float]]:
        async def _safe(kind: str, **kwargs: Any) -> Optional[float]:
            try:
                return await self.get_indicator(kind, symbol, **kwargs)
            except Exception:
                return None

        sma20, ema20, ema50, rsi14, adx14, macd, macd_sig, macd_hist = await asyncio.gather(
            _safe("sma", window=20),
            _safe("ema", window=20),
            _safe("ema", window=50),
            _safe("rsi", window=14),
            _safe("adx", window=14),
            _safe("macd", field="value"),
            _safe("macd", field="signal"),
            _safe("macd", field="histogram"),
        )
        return {
            "sma20": sma20,
            "ema20": ema20,
            "ema50": ema50,
            "rsi14": rsi14,
            "adx14": adx14,
            "macd_line": macd,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
        }

    # -------------------------------------------------------------------------
    # Options snapshots
    # -------------------------------------------------------------------------
    async def get_options_chain_snapshot(self, symbol: str, limit: int = 250) -> List[Dict[str, Any]]:
        js = await self._get(
            "/v3/snapshot/options",
            params={"underlying_ticker": symbol.upper(), "limit": limit},
            timeout=10.0,
        )
        return js.get("results") or []

    async def get_option_snapshot(self, option_ticker: str) -> Dict[str, Any]:
        js = await self._get(f"/v3/snapshot/options/{option_ticker}", timeout=10.0)
        return js.get("results") or {}

    async def get_targeted_option_context(
        self,
        symbol: str,
        *,
        expiry_iso: Optional[str],
        side: Optional[str],
        strike: Optional[float],
    ) -> Dict[str, Any]:
        if not expiry_iso or strike is None:
            return {}
        try:
            opt = build_option_contract(symbol, expiry_iso, side or "CALL", float(strike))
            snap = await self.get_option_snapshot(opt)
            q = snap.get("last_quote") or {}
            details = snap.get("details") or {}
            greeks = snap.get("greeks") or {}
            day = snap.get("day") or {}

            bid = q.get("bid")
            ask = q.get("ask")
            mid = (
                ((bid + ask) / 2.0)
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and (bid + ask) > 0
                else None
            )
            spr = (
                ((ask - bid) / max(mid, 1e-9) * 100.0)
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and mid
                else None
            )

            return {
                "option_ticker": opt,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "option_spread_pct": spr,
                "oi": details.get("open_interest") or day.get("open_interest"),
                "vol": day.get("volume"),
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "iv": greeks.get("implied_volatility"),
            }
        except Exception as e:
            logger.debug("[polygon] targeted option context failed: %r", e)
            return {}


def polygon_enabled() -> bool:
    return bool(POLYGON_API_KEY)
