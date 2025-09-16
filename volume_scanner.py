# volume_scanner.py
import os
import math
import time
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import httpx

from engine_runtime import get_http_client
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("trading_engine.uv_scan")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("UV_SCAN_LOG_LEVEL", "INFO"))

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# ---------- Config (env) ----------
UV_SCAN_TICKERS           = [s.strip().upper() for s in os.getenv("UV_SCAN_TICKERS", "").split(",") if s.strip()]
UV_SCAN_ENABLED           = os.getenv("UV_SCAN_ENABLED", "1") == "1"
UV_SCAN_SLEEP_SECONDS     = int(os.getenv("UV_SCAN_SLEEP_SECONDS", "300"))     # 5 min cycle
UV_SCAN_BATCH_SIZE        = int(os.getenv("UV_SCAN_BATCH_SIZE", "12"))        # scan subset per cycle
UV_SCAN_CONCURRENCY       = int(os.getenv("UV_SCAN_CONCURRENCY", "3"))        # parallel requests
UV_SCAN_QPS               = float(os.getenv("UV_SCAN_QPS", "4"))              # overall request budget
UV_SCAN_STOCK_WINDOW_MIN  = int(os.getenv("UV_SCAN_STOCK_WINDOW_MIN", "90"))  # aggs lookback
UV_SCAN_STOCK_BAR         = os.getenv("UV_SCAN_STOCK_BAR", "1")               # 1-minute bars
UV_SCAN_SPIKE_LOOKBACK    = int(os.getenv("UV_SCAN_SPIKE_LOOKBACK", "30"))    # median baseline window (min)
UV_SCAN_SPIKE_RECENT      = int(os.getenv("UV_SCAN_SPIKE_RECENT", "5"))       # recent minutes window
UV_SCAN_SPIKE_MULT        = float(os.getenv("UV_SCAN_SPIKE_MULT", "3.0"))     # spike threshold
UV_SCAN_MIN_ABS_VOL       = int(os.getenv("UV_SCAN_MIN_ABS_VOL", "150000"))   # absolute floor for stock vol

UV_SCAN_OPTIONS_ENABLED   = os.getenv("UV_SCAN_OPTIONS_ENABLED", "1") == "1"
UV_SCAN_OPT_LIMIT         = int(os.getenv("UV_SCAN_OPT_LIMIT", "60"))         # top N options by volume
UV_SCAN_OPT_MIN_DELTA     = int(os.getenv("UV_SCAN_OPT_MIN_DELTA", "500"))    # min 5m vol delta to alert
UV_SCAN_OPT_MULT          = float(os.getenv("UV_SCAN_OPT_MULT", "2.0"))       # 5m growth vs prior cached

# ---------- Token bucket for QPS ----------
class TokenBucket:
    def __init__(self, rate_per_sec: float, burst: Optional[int] = None):
        self.rate = max(0.1, rate_per_sec)
        self.capacity = max(1.0, float(burst if burst is not None else math.ceil(rate_per_sec * 2)))
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self.updated)
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            # need to wait
            need = 1.0 - self.tokens
            wait_s = need / self.rate
        await asyncio.sleep(wait_s)
        # recursive single retry (simple)
        await self.take()

_bucket = TokenBucket(UV_SCAN_QPS)
_sem = asyncio.Semaphore(UV_SCAN_CONCURRENCY)

# ---------- HTTP helpers ----------
def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat()

async def _safe_get(client: httpx.AsyncClient, url: str, params: Dict, timeout: float = 6.0):
    await _bucket.take()
    try:
        async with _sem:
            r = await client.get(url, params=params, timeout=timeout)
    except httpx.RemoteProtocolError as e:
        log.debug("[uv-scan] http proto error %s %s %r", url, params, e)
        # small backoff and one retry
        await asyncio.sleep(0.5)
        await _bucket.take()
        async with _sem:
            r = await client.get(url, params=params, timeout=timeout)
    if r.status_code == 429:
        retry_after = r.headers.get("Retry-After")
        sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else 1.5
        log.debug("[uv-scan] 429 %s; sleeping %.2fs", url, sleep_s)
        await asyncio.sleep(sleep_s)
        await _bucket.take()
        async with _sem:
            r = await client.get(url, params=params, timeout=timeout)
    if r.status_code in (400, 402, 403, 404, 422, 429, 500, 502, 503):
        try:
            body = r.text[:400]
        except Exception:
            body = "<no body>"
        raise httpx.HTTPStatusError(f"{r.status_code} for {url} :: {body}", request=r.request, response=r)
    r.raise_for_status()
    return r.json()

# ---------- Spike logic ----------
def _stock_volume_spike(bars: List[Dict]) -> Optional[Dict]:
    """
    bars: asc sorted 1m bars (Polygon v2 aggs results)
    Return spike dict or None.
    """
    if not bars or len(bars) < max(UV_SCAN_SPIKE_LOOKBACK, UV_SCAN_SPIKE_RECENT) + 2:
        return None

    # recent window and baseline
    recent = bars[-UV_SCAN_SPIKE_RECENT:]
    base   = bars[-(UV_SCAN_SPIKE_RECENT + UV_SCAN_SPIKE_LOOKBACK):-UV_SCAN_SPIKE_RECENT]
    if not base:
        return None

    sum_recent = sum(int(b.get("v") or 0) for b in recent)
    base_vols  = [int(b.get("v") or 0) for b in base]
    med_base   = sorted(base_vols)[len(base_vols)//2] if base_vols else 0

    if med_base <= 0:
        return None

    mult = (sum_recent / max(1, UV_SCAN_SPIKE_RECENT)) / med_base
    if mult >= UV_SCAN_SPIKE_MULT and sum_recent >= UV_SCAN_MIN_ABS_VOL:
        ts_last = int(recent[-1].get("t") or 0)
        return {
            "recent_avg": round(sum_recent / UV_SCAN_SPIKE_RECENT),
            "baseline_med": med_base,
            "mult": round(mult, 2),
            "last_bar_ts": ts_last,
            "recent_sum": sum_recent,
        }
    return None

# cache of option contract last seen cumulative volume to compute short-term deltas
_opt_last_vol: Dict[str, int] = {}

def _options_volume_bursts(results: List[Dict], underlying: str) -> List[Dict]:
    """
    Look at snapshot options list; compute per-contract delta of 'day.volume' since last scan.
    Alert if delta >= UV_SCAN_OPT_MIN_DELTA and delta ratio >= UV_SCAN_OPT_MULT.
    """
    bursts = []
    global _opt_last_vol
    for it in results or []:
        # structure: each 'it' is a contract snapshot object
        # attempt to get contract ticker and today's cumulative volume
        tk = (it.get("ticker") or it.get("contract") or it.get("details", {}).get("ticker") or "").strip()
        if not tk:
            continue
        day = it.get("day") or it.get("daily") or {}
        today_vol = int(day.get("volume") or day.get("v") or 0)
        if today_vol <= 0:
            continue
        prev = _opt_last_vol.get(tk, 0)
        delta = today_vol - prev
        if prev > 0 and delta >= UV_SCAN_OPT_MIN_DELTA:
            ratio = (delta / max(1, prev))
            if ratio >= UV_SCAN_OPT_MULT:
                bursts.append({"underlying": underlying, "ticker": tk, "delta": delta, "today": today_vol, "ratio": round(ratio, 2)})
        _opt_last_vol[tk] = today_vol
    # sort by delta desc
    bursts.sort(key=lambda x: x["delta"], reverse=True)
    return bursts[:8]

# ---------- Fetchers ----------
async def _fetch_stock_aggs(client: httpx.AsyncClient, symbol: str) -> Optional[List[Dict]]:
    # small, recent window only
    to_dt = datetime.now(timezone.utc)
    frm_dt = to_dt - timedelta(minutes=UV_SCAN_STOCK_WINDOW_MIN)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{UV_SCAN_STOCK_BAR}/minute/{_iso(frm_dt)}/{_iso(to_dt)}"
    params = {"adjusted": "true", "sort": "asc", "limit": 1800, "apiKey": POLYGON_API_KEY}
    try:
        js = await _safe_get(client, url, params, timeout=7.0)
        res = js.get("results") if isinstance(js, dict) else None
        if isinstance(res, list):
            return res
        return None
    except httpx.HTTPStatusError as e:
        # downgrade noisy logs to DEBUG
        log.debug("[uv-scan] stock aggs error %s %s -> %r", url, params, e)
        return None
    except Exception as e:
        log.debug("[uv-scan] stock aggs fail %s -> %r", symbol, e)
        return None

async def _fetch_options_snapshot(client: httpx.AsyncClient, underlying: str) -> Optional[List[Dict]]:
    # NOTE: do NOT pass greeks=true here; was causing 400s. Keep payload minimal.
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
    params = {
        "order": "desc",
        "sort": "volume",
        "limit": min(max(10, UV_SCAN_OPT_LIMIT), 120),  # server caps ~120
        "apiKey": POLYGON_API_KEY,
    }
    try:
        js = await _safe_get(client, url, params, timeout=7.0)
        # v3 returns {"results":[...], "next_url": ...}
        res = js.get("results") if isinstance(js, dict) else None
        if isinstance(res, list):
            return res
        return None
    except httpx.HTTPStatusError as e:
        log.debug("[uv-scan] options snapshot error %s %s -> %r", url, params, e)
        return None
    except Exception as e:
        log.debug("[uv-scan] options snapshot fail %s -> %r", underlying, e)
        return None

# ---------- Alert formatting ----------
def _fmt_int(n): 
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

async def _alert_stock_spike(symbol: str, spike: Dict]):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        txt = (
            f"⚡ Unusual Stock Volume\n"
            f"{symbol}: {_fmt_int(spike['recent_sum'])} in last {UV_SCAN_SPIKE_RECENT}m "
            f"(avg {_fmt_int(spike['recent_avg'])}/m vs med {_fmt_int(spike['baseline_med'])}/m) "
            f"→ x{spike['mult']}"
        )
        try:
            await send_telegram(txt)
        except Exception:
            pass

async def _alert_option_bursts(underlying: str, bursts: List[Dict]):
    if not bursts:
        return
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        lines = [f"⚡ Unusual Options Volume – {underlying}"]
        for b in bursts[:6]:
            lines.append(f"{b['ticker']}: +{_fmt_int(b['delta'])} since last scan (today {_fmt_int(b['today'])})")
        txt = "\n".join(lines)
        try:
            await send_telegram(txt)
        except Exception:
            pass

# ---------- Main loop ----------
async def run_scanner_loop():
    if not UV_SCAN_ENABLED:
        log.info("[uv-scan] disabled via UV_SCAN_ENABLED=0")
        return
    if not POLYGON_API_KEY:
        log.warning("[uv-scan] POLYGON_API_KEY missing; scanner idle")
        return
    if not UV_SCAN_TICKERS:
        log.info("[uv-scan] no UV_SCAN_TICKERS configured; scanner idle")
        return

    log.info("[uv-scan] starting; tickers=%s batch=%s qps=%.2f conc=%s", 
             len(UV_SCAN_TICKERS), UV_SCAN_BATCH_SIZE, UV_SCAN_QPS, UV_SCAN_CONCURRENCY)

    idx = 0
    client = get_http_client()
    if client is None:
        # create local client if engine HTTP client not yet started
        client = httpx.AsyncClient(http2=True, timeout=7.0)

    try:
        while True:
            # rotate batch
            batch = UV_SCAN_TICKERS[idx: idx + UV_SCAN_BATCH_SIZE]
            if not batch:
                idx = 0
                batch = UV_SCAN_TICKERS[:UV_SCAN_BATCH_SIZE]
            idx += len(batch)

            # STOCK SPIKES
            async def scan_stock(sym: str):
                bars = await _fetch_stock_aggs(client, sym)
                if not bars:
                    return
                spike = _stock_volume_spike(bars)
                if spike:
                    await _alert_stock_spike(sym, spike)

            # OPTIONS BURSTS (top by volume, delta since last)
            async def scan_options(sym: str):
                if not UV_SCAN_OPTIONS_ENABLED:
                    return
                snaps = await _fetch_options_snapshot(client, sym)
                if not snaps:
                    return
                bursts = _options_volume_bursts(snaps, sym)
                if bursts:
                    await _alert_option_bursts(sym, bursts)

            tasks = []
            for sym in batch:
                tasks.append(asyncio.create_task(scan_stock(sym)))
                tasks.append(asyncio.create_task(scan_options(sym)))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # sleep with a touch of jitter
            sleep_s = UV_SCAN_SLEEP_SECONDS + (0.15 * UV_SCAN_SLEEP_SECONDS) * (0.5 - (time.time() % 1))
            await asyncio.sleep(max(10, sleep_s))
    finally:
        try:
            if isinstance(client, httpx.AsyncClient):
                await client.aclose()
        except Exception:
            pass
