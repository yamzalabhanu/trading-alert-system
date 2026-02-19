# polygon_client.py
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import POLYGON_API_KEY
from engine_common import build_option_contract

logger = logging.getLogger("trading_engine")


@dataclass
class _CacheItem:
    exp: float
    val: Any


class PolygonClient:
    """
    Thin async Polygon.io REST client with:
      - soft-fail for blocked/missing endpoints (401/403/404)
      - 429 retry w/ exponential backoff + jitter
      - in-memory TTL caching to reduce rate limit pressure
      - per-client concurrency limiting (semaphore)
      - "blocked endpoint cooldown" so we don't spam 403 endpoints
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: Optional[str] = None) -> None:
        self.http = http_client
        self.api_key = (api_key or POLYGON_API_KEY or "").strip()
        self.base = "https://api.polygon.io"

        # keep calls under control
        self._sem = asyncio.Semaphore(int((__import__("os").getenv("POLYGON_MAX_CONCURRENCY", "3")).strip() or 3))

        # simple in-memory cache (per process)
        self._cache: Dict[str, _CacheItem] = {}

        # cooldown for plan-blocked endpoints (403/401)
        self._blocked_until: Dict[str, float] = {}

        # retry config
        self._max_429_retries = int((__import__("os").getenv("POLYGON_429_RETRIES", "4")).strip() or 4)
        self._base_backoff = float((__import__("os").getenv("POLYGON_429_BACKOFF", "0.7")).strip() or 0.7)
        self._block_cooldown_s = float((__import__("os").getenv("POLYGON_403_COOLDOWN", "1800")).strip() or 1800.0)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    # -----------------------------
    # tiny cache helpers
    # -----------------------------
    def _cache_get(self, key: str) -> Optional[Any]:
        it = self._cache.get(key)
        if not it:
            return None
        if it.exp <= time.time():
            self._cache.pop(key, None)
            return None
        return it.val

    def _cache_set(self, key: str, val: Any, ttl_s: float) -> Any:
        self._cache[key] = _CacheItem(exp=time.time() + ttl_s, val=val)
        return val

    def _blocked_key(self, path: str, symbol: Optional[str]) -> str:
        sym = (symbol or "").upper()
        # group by endpoint type, not by full query string
        return f"{sym}:{path}"

    def _is_blocked(self, path: str, symbol: Optional[str]) -> bool:
        k = self._blocked_key(path, symbol)
        until = self._blocked_until.get(k)
        return bool(until and until > time.time())

    def _mark_blocked(self, path: str, symbol: Optional[str]) -> None:
        k = self._blocked_key(path, symbol)
        self._blocked_until[k] = time.time() + self._block_cooldown_s

    # -----------------------------
    # core request
    # -----------------------------
    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 6.0,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("POLYGON_API_KEY is not set")

        q = dict(params or {})
        q["apiKey"] = self.api_key
        url = f"{self.base}{path}"

        async with self._sem:
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
        cache_ttl_s: float = 0.0,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Soft-fail for common entitlement/missing/rate limit issues.
        Returns {} on 401/403/404 or final 429 failure.
        """
        sym = (symbol or "").upper()

        # blocked cooldown (avoid spamming known-403 endpoints)
        if self._is_blocked(path, sym):
            return {}

        ck = cache_key
        if cache_ttl_s > 0 and ck:
            hit = self._cache_get(ck)
            if hit is not None:
                return hit

        attempt = 0
        while True:
            attempt += 1
            try:
                js = await self._get(path, params=params, timeout=timeout)
                if cache_ttl_s > 0 and ck:
                    return self._cache_set(ck, js, cache_ttl_s)
                return js

            except httpx.HTTPStatusError as e:
                code = getattr(e.response, "status_code", None)

                # plan blocked / not entitled
                if code in (401, 403):
                    logger.warning("%s blocked for %s (HTTP %s) path=%s", log_prefix, sym, code, path)
                    self._mark_blocked(path, sym)
                    return {}

                # not found (bad endpoint / ticker not supported)
                if code == 404:
                    logger.warning("%s not found for %s (HTTP 404) path=%s", log_prefix, sym, path)
                    return {}

                # rate limit retry
                if code == 429:
                    if attempt >= self._max_429_retries:
                        logger.warning("%s 429 rate limited (attempt=%d/%d) path=%s", log_prefix, attempt, self._max_429_retries, path)
                        return {}
                    sleep_s = (self._base_backoff * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)
                    logger.warning("%s 429 rate limited (attempt=%d/%d) path=%s sleeping=%.2fs", log_prefix, attempt, self._max_429_retries, path, sleep_s)
                    await asyncio.sleep(sleep_s)
                    continue

                logger.warning("%s failed for %s (HTTP %s) path=%s err=%r", log_prefix, sym, code, path, e)
                return {}

            except Exception as e:
                logger.warning("%s exception for %s path=%s err=%r", log_prefix, sym, path, e)
                return {}

    # -----------------------------
    # Stocks
    # -----------------------------
    async def get_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        path = f"/v2/snapshot/locale/us/markets/stocks/tickers/{sym}"
        js = await self._get_soft(
            path,
            symbol=sym,
            log_prefix="[polygon] snapshot",
            cache_ttl_s=8.0,  # snapshot can be cached briefly
            cache_key=f"snap:{sym}",
        )
        return js.get("ticker") or {}

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Correct endpoint for last quote is NBBO:
          /v2/last/nbbo/{ticker}
        """
        sym = symbol.upper()
        path = f"/v2/last/nbbo/{sym}"
        js = await self._get_soft(
            path,
            symbol=sym,
            log_prefix="[polygon] nbbo",
            cache_ttl_s=3.0,
            cache_key=f"nbbo:{sym}",
        )
        return js.get("results") or {}

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        sym = symbol.upper()
        path = f"/v2/last/trade/{sym}"
        js = await self._get_soft(
            path,
            symbol=sym,
            log_prefix="[polygon] last/trade",
            cache_ttl_s=3.0,
            cache_key=f"trade:{sym}",
        )
        return js.get("results") or {}

    # -----------------------------
    # Aggregates
    # -----------------------------
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
        cache_ttl_s: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Windowed aggs fetch. Use this instead of the huge 2020-2100 range.
        """
        sym = symbol.upper()
        path = f"/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/{from_}/{to}"
        js = await self._get_soft(
            path,
            params={"adjusted": "true" if adjusted else "false", "sort": sort, "limit": limit},
            timeout=10.0 if timespan == "day" else 8.0,
            symbol=sym,
            log_prefix="[polygon] aggs",
            cache_ttl_s=cache_ttl_s,
            cache_key=f"aggs:{sym}:{multiplier}:{timespan}:{from_}:{to}:{limit}",
        )
        return js.get("results") or []

    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 5,
        timespan: str = "minute",
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Kept for compatibility, but internally should prefer get_aggs_window().
        """
        sym = symbol.upper()
        path = f"/v2/aggs/ticker/{sym}/range/{multiplier}/{timespan}/2020-01-01/2100-01-01"
        js = await self._get_soft(
            path,
            params={"adjusted": "true", "sort": "asc", "limit": limit},
            timeout=8.0,
            symbol=sym,
            log_prefix="[polygon] aggs_wide",
        )
        return js.get("results") or []

    # -----------------------------
    # Indicators / Technicals
    # -----------------------------
    async def _indicator_raw(
        self,
        kind: str,
        symbol: str,
        *,
        timespan: str,
        window: int = 14,
        series_type: str = "close",
        limit: int = 200,
        macd_short: int = 12,
        macd_long: int = 26,
        macd_signal: int = 9,
        cache_ttl_s: float = 0.0,
    ) -> Dict[str, Any]:
        sym = symbol.upper()
        path = f"/v1/indicators/{kind}/{sym}"
        params: Dict[str, Any] = {
            "timespan": timespan,
            "window": window,
            "series_type": series_type,
            "order": "desc",
            "limit": limit,
            "adjusted": "true",
        }
        if kind == "macd":
            params.update({"short_window": macd_short, "long_window": macd_long, "signal_window": macd_signal})

        return await self._get_soft(
            path,
            params=params,
            symbol=sym,
            log_prefix=f"[polygon] indicators/{kind}",
            cache_ttl_s=cache_ttl_s,
            cache_key=f"ind:{sym}:{kind}:{timespan}:{window}:{series_type}:{limit}:{macd_short}:{macd_long}:{macd_signal}",
        )

    @staticmethod
    def _first_value(js: Dict[str, Any], field: str = "value") -> Optional[float]:
        vals = (((js.get("results") or {}).get("values")) or [])
        if not vals:
            return None
        v = vals[0].get(field)
        return float(v) if isinstance(v, (int, float)) else None

    async def get_technicals_bundle(
        self,
        symbol: str,
        *,
        timespan: str = "minute",
    ) -> Dict[str, Optional[float]]:
        """
        Indicator bundle.
        ✅ MACD is fetched ONCE and we extract (value, signal, histogram) from the same response.
        Caching:
          - minute: ~60s
          - day:    ~15m
        """
        ttl = 60.0 if timespan == "minute" else 900.0

        sma_js, ema20_js, ema50_js, rsi_js, macd_js = await asyncio.gather(
            self._indicator_raw("sma", symbol, timespan=timespan, window=20, cache_ttl_s=ttl),
            self._indicator_raw("ema", symbol, timespan=timespan, window=20, cache_ttl_s=ttl),
            self._indicator_raw("ema", symbol, timespan=timespan, window=50, cache_ttl_s=ttl),
            self._indicator_raw("rsi", symbol, timespan=timespan, window=14, cache_ttl_s=ttl),
            self._indicator_raw("macd", symbol, timespan=timespan, cache_ttl_s=ttl),
        )

        return {
            "sma20": self._first_value(sma_js, "value"),
            "ema20": self._first_value(ema20_js, "value"),
            "ema50": self._first_value(ema50_js, "value"),
            "rsi14": self._first_value(rsi_js, "value"),
            "macd_line": self._first_value(macd_js, "value"),
            "macd_signal": self._first_value(macd_js, "signal"),
            "macd_hist": self._first_value(macd_js, "histogram"),
        }

    async def get_technicals_daily_bundle(self, symbol: str) -> Dict[str, Optional[float]]:
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
        js = await self._get_soft(
            "/v3/snapshot/options",
            params={"underlying_ticker": sym, "limit": limit},
            timeout=8.0,
            symbol=sym,
            log_prefix="[polygon] options/chain",
            cache_ttl_s=20.0,
            cache_key=f"optchain:{sym}:{limit}",
        )
        return js.get("results") or []

    async def get_option_snapshot(self, option_ticker: str) -> Dict[str, Any]:
        js = await self._get_soft(
            f"/v3/snapshot/options/{option_ticker}",
            timeout=8.0,
            symbol=None,
            log_prefix="[polygon] options/snap",
            cache_ttl_s=5.0,
            cache_key=f"optsnap:{option_ticker}",
        )
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
        except Exception as e:
            logger.debug("[polygon] targeted option context failed for %s: %r", sym, e)
            return {}


def polygon_enabled() -> bool:
    return bool((POLYGON_API_KEY or "").strip())
