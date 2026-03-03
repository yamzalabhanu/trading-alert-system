# polygon_client.py
import os
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable

import httpx

log = logging.getLogger("trading_engine.polygon")

POLYGON_API_KEY = (os.getenv("POLYGON_API_KEY", "") or "").strip()
POLYGON_BASE_URL = (os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/"))


# -----------------------------
# Endpoint toggles (env)
# -----------------------------
def _truthy(v: str) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


POLYGON_ENABLE_SNAPSHOT = _truthy(os.getenv("POLYGON_ENABLE_SNAPSHOT", "1"))
POLYGON_ENABLE_NBBO = _truthy(os.getenv("POLYGON_ENABLE_NBBO", "1"))
POLYGON_ENABLE_LAST_TRADE = _truthy(os.getenv("POLYGON_ENABLE_LAST_TRADE", "1"))
POLYGON_ENABLE_AGGS = _truthy(os.getenv("POLYGON_ENABLE_AGGS", "1"))
POLYGON_ENABLE_INDICATORS = _truthy(os.getenv("POLYGON_ENABLE_INDICATORS", "1"))
POLYGON_ENABLE_OPTIONS = _truthy(os.getenv("POLYGON_ENABLE_OPTIONS", "1"))

# If you’re on a plan that blocks snapshot/last/nbbo, set these to 0:
#   POLYGON_ENABLE_SNAPSHOT=0
#   POLYGON_ENABLE_NBBO=0
#   POLYGON_ENABLE_LAST_TRADE=0


@dataclass
class _CacheItem:
    exp: float
    val: Any


class PolygonClient:
    """
    Thin async Polygon.io REST client with:
      - soft-fail for blocked/missing endpoints (401/403/404)
      - 429 retry w/ exponential backoff + jitter
      - in-memory TTL caching
      - per-client concurrency limiting (semaphore)
      - "blocked endpoint cooldown" so we don't spam 403 endpoints
      - inflight de-dupe: same cache_key => await the same Task
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: Optional[str] = None) -> None:
        self.http = http_client
        self.api_key = (api_key or POLYGON_API_KEY or "").strip()
        self.base = POLYGON_BASE_URL

        max_conc = int((os.getenv("POLYGON_MAX_CONCURRENCY", "3") or "3").strip() or 3)
        self._sem = asyncio.Semaphore(max_conc)

        self._cache: Dict[str, _CacheItem] = {}
        self._blocked_until: Dict[str, float] = {}

        self._max_429_retries = int((os.getenv("POLYGON_429_RETRIES", "4") or "4").strip() or 4)
        self._base_backoff = float((os.getenv("POLYGON_429_BACKOFF", "0.7") or "0.7").strip() or 0.7)
        self._block_cooldown_s = float((os.getenv("POLYGON_403_COOLDOWN", "1800") or "1800").strip() or 1800.0)

        # inflight de-dupe (per client)
        self._inflight: Dict[str, asyncio.Task] = {}
        self._inflight_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    # -----------------------------
    # cache helpers
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

    # -----------------------------
    # block cooldown
    # -----------------------------
    def _blocked_key(self, path: str, symbol: Optional[str]) -> str:
        sym = (symbol or "").upper()
        return f"{sym}:{path}"

    def _is_blocked(self, path: str, symbol: Optional[str]) -> bool:
        k = self._blocked_key(path, symbol)
        until = self._blocked_until.get(k)
        return bool(until and until > time.time())

    def _mark_blocked(self, path: str, symbol: Optional[str]) -> None:
        k = self._blocked_key(path, symbol)
        self._blocked_until[k] = time.time() + self._block_cooldown_s

    # -----------------------------
    # inflight de-dupe
    # -----------------------------
    async def _dedupe(self, key: str, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        """
        If same key is already in-flight, await the existing task.
        Otherwise create a task, store it, await it, then remove.
        """
        async with self._inflight_lock:
            t = self._inflight.get(key)
            if t and not t.done():
                return await t

            task = asyncio.create_task(coro_factory())
            self._inflight[key] = task

        try:
            return await task
        finally:
            async with self._inflight_lock:
                cur = self._inflight.get(key)
                if cur is task:
                    self._inflight.pop(key, None)

    # -----------------------------
    # core request
    # -----------------------------
    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 6.0) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("POLYGON_API_KEY is not set")

        q = dict(params or {})
        q["apiKey"] = self.api_key
        url = f"{self.base}{path}"

        async with self._sem:
            r = await self.http.get(url, params=q, timeout=timeout)
        r.raise_for_status()

        js = r.json()
        return js if isinstance(js, dict) else {}

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
        Soft-fail on 401/403/404 and final 429 exhaustion.
        Returns {} on failure.
        """
        sym = (symbol or "").upper()

        # avoid spamming a known-blocked endpoint for this symbol
        if self._is_blocked(path, sym):
            return {}

        # cache hit
        if cache_ttl_s > 0 and cache_key:
            hit = self._cache_get(cache_key)
            if hit is not None:
                return hit

        async def _runner() -> Dict[str, Any]:
            attempt = 0
            while True:
                attempt += 1
                try:
                    js = await self._get(path, params=params, timeout=timeout)
                    if cache_ttl_s > 0 and cache_key:
                        return self._cache_set(cache_key, js, cache_ttl_s)
                    return js

                except httpx.HTTPStatusError as e:
                    code = getattr(e.response, "status_code", None)

                    # plan blocked / not entitled
                    if code in (401, 403):
                        log.warning("%s blocked for %s (HTTP %s) path=%s", log_prefix, sym, code, path)
                        self._mark_blocked(path, sym)
                        return {}

                    # missing endpoint / invalid ticker
                    if code == 404:
                        log.warning("%s not found for %s (HTTP 404) path=%s", log_prefix, sym, path)
                        return {}

                    # rate limit retry
                    if code == 429:
                        if attempt >= self._max_429_retries:
                            log.warning(
                                "%s 429 rate limited (attempt=%d/%d) path=%s",
                                log_prefix,
                                attempt,
                                self._max_429_retries,
                                path,
                            )
                            return {}
                        sleep_s = (self._base_backoff * (2 ** (attempt - 1))) + random.uniform(0.0, 0.35)
                        log.warning(
                            "%s 429 rate limited (attempt=%d/%d) path=%s sleeping=%.2fs",
                            log_prefix,
                            attempt,
                            self._max_429_retries,
                            path,
                            sleep_s,
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                    log.warning("%s failed for %s (HTTP %s) path=%s err=%r", log_prefix, sym, code, path, e)
                    return {}

                except Exception as e:
                    log.warning("%s exception for %s path=%s err=%r", log_prefix, sym, path, e)
                    return {}

        # inflight de-dupe if cache_key provided
        if cache_key:
            return await self._dedupe(f"inflight:{cache_key}", _runner)

        return await _runner()

    # -----------------------------
    # Stocks
    # -----------------------------
    async def get_stock_snapshot(self, symbol: str) -> Dict[str, Any]:
        if not POLYGON_ENABLE_SNAPSHOT:
            return {}
        sym = symbol.upper()
        path = f"/v2/snapshot/locale/us/markets/stocks/tickers/{sym}"
        js = await self._get_soft(
            path,
            symbol=sym,
            log_prefix="[polygon] snapshot",
            cache_ttl_s=8.0,
            cache_key=f"snap:{sym}",
        )
        return js.get("ticker") or {}

    async def get_last_quote(self, symbol: str) -> Dict[str, Any]:
        if not POLYGON_ENABLE_NBBO:
            return {}
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
        if not POLYGON_ENABLE_LAST_TRADE:
            return {}
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
        if not POLYGON_ENABLE_AGGS:
            return []
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
        if not POLYGON_ENABLE_INDICATORS:
            return {}
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

    async def get_technicals_bundle(self, symbol: str, *, timespan: str = "minute") -> Dict[str, Optional[float]]:
        if not POLYGON_ENABLE_INDICATORS:
            return {}
        ttl = 60.0 if timespan == "minute" else 900.0

        # NOTE: inflight de-dupe will collapse identical requests across workers using the same PolygonClient instance
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
        if not d:
            return {}
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
        if not POLYGON_ENABLE_OPTIONS:
            return []
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
        if not POLYGON_ENABLE_OPTIONS:
            return {}
        js = await self._get_soft(
            f"/v3/snapshot/options/{option_ticker}",
            timeout=8.0,
            symbol=None,
            log_prefix="[polygon] options/snap",
            cache_ttl_s=5.0,
            cache_key=f"optsnap:{option_ticker}",
        )
        return js.get("results") or {}


def polygon_enabled() -> bool:
    return bool((POLYGON_API_KEY or "").strip())
