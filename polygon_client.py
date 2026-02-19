# polygon_client.py
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
        self.api_key = (api_key or POLYGON_API_KEY or "").strip()
        self.base = "https://api.polygon.io"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Hard-fail request (raise_for_status). Use _get_soft() for plan-blocked endpoints.
        """
        if not self.enabled:
            raise RuntimeError("POLYGON_API_KEY is not set")

        q = dict(params or {})
        q["apiKey"] = self.api_key
        url = f"{self.base}{path}"

        r = await self.http.get(url, params=q, timeout=timeout)
        r.raise_for_status()
        return r.json()

    async def _get_soft(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 6.0,
        *,
        log_prefix: str = "[polygon]",
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Soft-fail for endpoints that can be blocked by plan (403/401) or missing (404).
        Returns {} on those, re-raises on other unexpected errors.
        """
        try:
            return await self._get(path, params=params, timeout=timeout)
        except httpx.HTTPStatusError as e:
            code = getattr(e.response, "status_code", None)
            sym = (symbol or "").upper()
            # Common "not available / not entitled / not found" cases
            if code in (401, 403, 404, 429):
                if code in (401, 403):
                    logger.warning("%s blocked for %s (HTTP %s) path=%s", log_prefix, sym, code, path)
                elif code == 404:
                    logger.warning("%s not found for %s (HTTP 404) path=%s", log_prefix, sym, path)
                elif code == 429:
                    logger.warning("%s rate-limited for %s (HTTP 429) path=%s", log_prefix, sym, path)
                return {}
            logger.warning("%s failed for %s (HTTP %s) path=%s err=%r", log_prefix, sym, code, path, e)
            raise
        except Exception as e:
            logger.warning("%s exception for %s path=%s err=%r", log_prefix, (symbol or "").upper(), path, e)
            return {}

    # -----------------------------
    # Stocks
    # -----------------------------
    async def get_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        js = await self._get_soft(f"/v2/snapshot/locale/us/markets/stocks/tickers/{sym}", symbol=sym, log_prefix="[polygon] snapshot")
        return js.get("ticker") or {}

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        ✅ Correct endpoint for last quote is NBBO:
          /v2/last/nbbo/{ticker}
        (There is no /v2/last/quote/{ticker} — that caused your 404s.)
        """
        sym = symbol.upper()
        js = await self._get_soft(f"/v2/last/nbbo/{sym}", symbol=sym, log_prefix="[polygon] nbbo")
        return js.get("results") or {}

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        js = await self._get_soft(f"/v2/last/trade/{sym}", symbol=sym, log_prefix="[polygon] last/trade")
        return js.get("results") or {}

    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 5,
        timespan: str = "minute",
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        sym = symbol.upper()
        js = await self._get(
            f"/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/2020-01-01/2100-01-01",
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
        adjusted: bool = True,
        sort: str = "asc",
    ) -> List[Dict[str, Any]]:
        """
        More precise aggregate fetch for a window (helps daily/weekly context without the wide 2020-2100 range).
        """
        sym = symbol.upper()
        js = await self._get(
            f"/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/{from_}/{to}",
            params={"adjusted": "true" if adjusted else "false", "sort": sort, "limit": limit},
            timeout=10.0 if timespan == "day" else 8.0,
        )
        return js.get("results") or []

    # -----------------------------
    # Indicators / Technicals
    # -----------------------------
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
            "timespan": timespan,        # "minute" (intraday) or "day" (daily bias)
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

        js = await self._get(path, params=params)
        values = (((js.get("results") or {}).get("values")) or [])
        if not values:
            return None
        item = values[0]
        val = item.get(field)
        return float(val) if isinstance(val, (int, float)) else None

    async def get_technicals_bundle(
        self,
        symbol: str,
        *,
        timespan: str = "minute",
    ) -> Dict[str, Optional[float]]:
        """
        Returns a standard indicator bundle for the given timespan.
        - timespan="minute" => intraday signal context
        - timespan="day"    => daily-chart bias/context for higher precision
        """
        async def _safe(kind: str, **kwargs: Any) -> Optional[float]:
            try:
                return await self.get_indicator(kind, symbol, timespan=timespan, **kwargs)
            except Exception:
                return None

        sma20, ema20, ema50, rsi14, macd, macd_sig, macd_hist = await asyncio.gather(
            _safe("sma", window=20),
            _safe("ema", window=20),
            _safe("ema", window=50),
            _safe("rsi", window=14),
            _safe("macd", field="value"),
            _safe("macd", field="signal"),
            _safe("macd", field="histogram"),
        )
        return {
            "sma20": sma20,
            "ema20": ema20,
            "ema50": ema50,
            "rsi14": rsi14,
            "macd_line": macd,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
        }

    async def get_technicals_daily_bundle(self, symbol: str) -> Dict[str, Optional[float]]:
        """
        ✅ Daily chart indicators, names suffixed with _d to avoid collisions.
        """
        d = await self.get_technicals_bundle(symbol, timespan="day")
        return {
            "sma20_d": d.get("sma20"),
            "ema20_d": d.get("ema20"),
            "ema50_d": d.get("ema50"),
            "rsi14_d": d.get("rsi14"),
            "macd_line_d": d.get("macd_line"),
            "macd_signal_d": d.get("macd_signal"),
            "macd_hist_d": d.get("macd_hist"),
        }

    # -----------------------------
    # Options snapshots
    # -----------------------------
    async def get_options_chain_snapshot(self, symbol: str, limit: int = 250) -> List[Dict[str, Any]]:
        sym = symbol.upper()
        js = await self._get(
            "/v3/snapshot/options",
            params={"underlying_ticker": sym, "limit": limit},
            timeout=8.0,
        )
        return js.get("results") or []

    async def get_option_snapshot(self, option_ticker: str) -> Dict[str, Any]:
        js = await self._get(f"/v3/snapshot/options/{option_ticker}")
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

        sym = symbol.upper()
        try:
            opt = build_option_contract(sym, expiry_iso, side or "CALL", float(strike))
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
        except httpx.HTTPStatusError as e:
            code = getattr(e.response, "status_code", None)
            logger.debug("[polygon] option ctx blocked/failed for %s (HTTP %s): %r", sym, code, e)
            return {}
        except Exception as e:
            logger.debug("[polygon] targeted option context failed for %s: %r", sym, e)
            return {}


def polygon_enabled() -> bool:
    return bool((POLYGON_API_KEY or "").strip())
