# fastapi_app.py
"""
🚀 FastAPI service with robust SSL & timezone fallbacks + asyncio-based scheduler (no multiprocessing),
scoring/confidence, success-rate tracking, self-tests, and operational enhancements.

This app does:
1) Fetch daily pre‑market active tickers from Polygon and refresh the list each morning.
2) Maintain a user‑provided ticker list.
3) 9:00 AM ET job: pull ~20 trading days of stock + options snapshot data, Finnhub news/sentiment,
   compute EMA9/EMA21 + volume, and analyze with OpenAI to rank tickers.
4) Send rankings to Telegram **with confidence scores**.
5) Every 30 minutes during market hours, fetch intraday stock volume + options snapshot, enrich with sentiment,
   analyze again, and alert.
6) Keep an in‑memory log of decisions and compute a rolling **success rate**.
7) Generate a **post‑market daily report** at 4:15 PM ET with win/loss stats and top movers (CSV attached).

Environment (.env):
- POLYGON_API_KEY=
- FINNHUB_API_KEY=
- OPENAI_API_KEY=
- TELEGRAM_BOT_TOKEN=
- TELEGRAM_CHAT_ID=
- (optional) REDIS_URL=
- (optional) SENTRY_DSN=
- (optional) APP_BASE_PATH=/api

Run: `uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload`

If you see `SSL module not available`: install a Python build with OpenSSL (e.g., official python:3.12).
If you see `No time zone found with key America/New_York`: `pip install tzdata` or install OS tzdata.
"""

from __future__ import annotations

import os
import sys
import io
import csv
import json
import math
import asyncio
from datetime import datetime, timedelta, date, timezone as dt_timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv

# --- SSL preflight -----------------------------------------------------------
try:
    import ssl  # noqa: F401
    SSL_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SSL_AVAILABLE = False

# ===== Config & Settings =====================================================
load_dotenv()

# Runtime‑tunable settings (also editable via /settings)
SETTINGS: Dict[str, Any] = {
    # Accuracy controls
    "options_dte_min": int(os.getenv("OPTIONS_DTE_MIN", 5)),
    "options_dte_max": int(os.getenv("OPTIONS_DTE_MAX", 21)),
    "moneyness_min": float(os.getenv("MONEYNESS_MIN", 0.9)),
    "moneyness_max": float(os.getenv("MONEYNESS_MAX", 1.1)),
    # Alert gating
    "alert_llm_conf_threshold": int(os.getenv("ALERT_LLM_CONF", 70)),
    "alert_score_threshold": int(os.getenv("ALERT_SCORE", 60)),
    "alert_cooldown_minutes": int(os.getenv("ALERT_COOLDOWN_MIN", 30)),
    "universe_max": int(os.getenv("UNIVERSE_MAX", 50)),
    # Scheduler toggles
    "SCHEDULER_ENABLED": os.getenv("SCHEDULER_ENABLED", "true").lower() == "true",
    "INTRADAY_ENABLED": os.getenv("INTRADAY_ENABLED", "true").lower() == "true",
    "CLOSE_REPORT_ENABLED": os.getenv("CLOSE_REPORT_ENABLED", "true").lower() == "true",
}

APP_BASE_PATH = os.getenv("APP_BASE_PATH", "").strip("/")

# --- Timezone resolver (America/New_York with fallbacks) --------------------
TZ_ET_OK = False
try:
    from zoneinfo import ZoneInfo
    try:
        NY = ZoneInfo("America/New_York")
        TZ_ET_OK = True
    except Exception:
        try:
            import tzdata  # noqa: F401
            NY = ZoneInfo("America/New_York")
            TZ_ET_OK = True
        except Exception:
            NY = dt_timezone.utc
            TZ_ET_OK = False
except Exception:  # pragma: no cover - very rare
    NY = dt_timezone.utc
    TZ_ET_OK = False

# ===== Globals (shared) ======================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
REDIS_URL = os.getenv("REDIS_URL", "")

premarket_tickers: Set[str] = set()
user_tickers: Set[str] = set()
last_daily_ranking: List[Dict[str, Any]] = []
last_intraday_ranking: List[Dict[str, Any]] = []

# Track background tasks for clean shutdown
_bg_tasks: List[asyncio.Task] = []

# Observability counters (very lightweight Prometheus text output)
METRICS: Dict[str, float] = {
    "jobs_daily_total": 0.0,
    "jobs_intraday_total": 0.0,
    "jobs_close_total": 0.0,
    "errors_total": 0.0,
}

# Cooldown tracker for alerts
_last_alert_at: Dict[str, datetime] = {}

# Optional Redis (cache + leader lock)
redis = None
try:  # pragma: no cover
    import redis.asyncio as redis_mod  # type: ignore
    if REDIS_URL:
        redis = redis_mod.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
except Exception:
    redis = None

# Optional Sentry
if SENTRY_DSN:  # pragma: no cover
    try:
        import sentry_sdk  # type: ignore
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.1)
    except Exception:
        pass

# ===== Utility (pure) =======================================================

def ema(series: List[float], length: int) -> List[float]:
    if not series:
        return []
    k = 2 / (length + 1)
    out: List[float] = []
    prev = series[0]
    for price in series:
        prev = price * k + prev * (1 - k)
        out.append(prev)
    return out


def atr14(daily: List[Dict[str, Any]]) -> float:
    if not daily:
        return 0.0
    trs: List[float] = []
    prev_close = None
    for b in daily[-15:]:
        h, l, c = b.get("h"), b.get("l"), b.get("c")
        if h is None or l is None or c is None:
            continue
        if prev_close is None:
            tr = float(h - l)
        else:
            tr = max(float(h - l), abs(float(h - prev_close)), abs(float(l - prev_close)))
        trs.append(tr)
        prev_close = float(c)
    return float(np.mean(trs)) if trs else 0.0


def summarize_options_snapshot(
    contracts: List[Dict[str, Any]], *,
    underlying_price: Optional[float] = None,
    dte_min: int = 0, dte_max: int = 10**6,
    moneyness_min: float = 0.0, moneyness_max: float = 10.0,
) -> Dict[str, float]:
    """Aggregate put/call volume & OI with optional DTE & moneyness filters.
    Also returns a crude IV rank proxy (cross-sectional) and call/put IV skew.
    """
    put_vol = call_vol = put_oi = call_oi = 0.0
    ivs: List[float] = []
    iv_calls: List[float] = []
    iv_puts: List[float] = []

    now_dt = datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)
    for c in contracts:
        details = c.get("details", {})
        o_type = (details.get("contract_type") or c.get("contract_type") or "").lower()
        strike = details.get("strike_price") or c.get("strike_price")
        exp = details.get("expiration_date") or c.get("expiration_date")
        if not strike or not exp:
            continue
        try:
            exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00")).date()
        except Exception:
            try:
                y, m, d = str(exp).split("-")
                exp_dt = date(int(y), int(m), int(d))
            except Exception:
                continue
        dte = (exp_dt - now_dt.date()).days
        if dte < dte_min or dte > dte_max:
            continue
        und = underlying_price or details.get("underlying_price") or c.get("underlying_price") or c.get("underlying", {}).get("price")
        if not und or float(und) <= 0:
            continue
        try:
            mny = float(und) / float(strike)
        except Exception:
            continue
        if not (moneyness_min <= mny <= moneyness_max):
            continue

        vol = (c.get("day", {}).get("volume") or c.get("last_quote", {}).get("volume") or c.get("volume") or 0)
        oi = (c.get("open_interest") or c.get("day", {}).get("open_interest") or 0)
        iv = (c.get("implied_volatility") or details.get("implied_volatility"))
        if isinstance(iv, (int, float)):
            ivs.append(float(iv))
            if o_type in ("call", "put"):
                (iv_calls if o_type == "call" else iv_puts).append(float(iv))

        if o_type == "put":
            put_vol += float(vol); put_oi += float(oi)
        elif o_type == "call":
            call_vol += float(vol); call_oi += float(oi)

    pcr = (put_vol / call_vol) if call_vol else math.inf
    oir = (put_oi / call_oi) if call_oi else math.inf

    # IV rank proxy from cross-section
    iv_rank = 50.0
    if ivs:
        cur_iv = float(np.median(ivs))
        pct = float(sum(1 for v in ivs if v <= cur_iv)) / len(ivs)
        iv_rank = pct * 100.0
    skew = (np.median(iv_calls) - np.median(iv_puts)) if (iv_calls and iv_puts) else 0.0

    return {
        "put_volume": float(put_vol),
        "call_volume": float(call_vol),
        "put_call_volume_ratio": float(pcr),
        "put_open_interest": float(put_oi),
        "call_open_interest": float(call_oi),
        "put_call_oi_ratio": float(oir),
        "iv_rank_proxy": float(iv_rank),
        "iv_call_put_skew": float(skew),
    }


def compute_stock_features(daily_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not daily_bars:
        return {}
    closes = [b.get("c") for b in daily_bars if b.get("c") is not None]
    volumes = [b.get("v") for b in daily_bars if b.get("v") is not None]
    highs = [b.get("h") for b in daily_bars if b.get("h") is not None]
    lows = [b.get("l") for b in daily_bars if b.get("l") is not None]
    opens = [b.get("o") for b in daily_bars if b.get("o") is not None]
    if not closes:
        return {}
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema_trend = (ema9[-1] - ema21[-1]) if (ema9 and ema21) else 0.0
    avg_vol = float(np.mean(volumes[-20:])) if volumes else 0.0
    stock_rv20 = float(np.std(np.diff(np.log(closes[-21:]))) * np.sqrt(252)) if len(closes) >= 21 else 0.0
    atr = atr14([{"h": h, "l": l, "c": c} for h, l, c in zip(highs, lows, closes)])
    gap = 0.0
    if len(closes) >= 2 and len(opens) >= 1:
        prev_close = closes[-2]
        today_open = opens[-1] if opens[-1] else prev_close
        if prev_close:
            gap = (today_open - prev_close) / prev_close
    return {
        "ema9": float(ema9[-1]) if ema9 else None,
        "ema21": float(ema21[-1]) if ema21 else None,
        "ema_trend": float(ema_trend),
        "avg_volume_20d": avg_vol,
        "rv20": stock_rv20,
        "atr14": float(atr),
        "gap_pct": float(gap),
    }


def zscore(value: float, series: List[float]) -> float:
    if not series:
        return 0.0
    mu = float(np.mean(series))
    sd = float(np.std(series))
    if sd == 0:
        return 0.0
    return float((value - mu) / sd)

# Simple in-memory TTL cache
class TTLCache:
    def __init__(self):
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < asyncio.get_event_loop().time():
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any, ttl: float) -> None:
        self._store[key] = (asyncio.get_event_loop().time() + ttl, val)

TTL = TTLCache()

# Market calendar utilities (basic NYSE closures)

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d

# Anonymous Gregorian algorithm for Easter

def _easter(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_market_holidays(year: int) -> Set[date]:
    hol: Set[date] = set()
    # New Year's (observed)
    nyd = date(year, 1, 1)
    hol.add(nyd if nyd.weekday() < 5 else (nyd + timedelta(days=1) if nyd.weekday() == 5 else nyd - timedelta(days=1)))
    # MLK Day (3rd Monday Jan)
    hol.add(_nth_weekday(year, 1, 0, 3))
    # Presidents' Day (3rd Monday Feb)
    hol.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday (2 days before Easter Sunday)
    hol.add(_easter(year) - timedelta(days=2))
    # Memorial Day (last Monday May)
    hol.add(_last_weekday(year, 5, 0))
    # Juneteenth (observed)
    j19 = date(year, 6, 19)
    hol.add(j19 if j19.weekday() < 5 else (j19 + timedelta(days=1) if j19.weekday() == 5 else j19 - timedelta(days=1)))
    # Independence Day (observed)
    i4 = date(year, 7, 4)
    hol.add(i4 if i4.weekday() < 5 else (i4 + timedelta(days=1) if i4.weekday() == 5 else i4 - timedelta(days=1)))
    # Labor Day (1st Monday Sep)
    hol.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving (4th Thursday Nov)
    hol.add(_nth_weekday(year, 11, 3, 4))
    # Christmas (observed)
    x25 = date(year, 12, 25)
    hol.add(x25 if x25.weekday() < 5 else (x25 + timedelta(days=1) if x25.weekday() == 5 else x25 - timedelta(days=1)))
    return hol


def is_market_open_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    return d not in us_market_holidays(d.year)

# Calibration history & helpers
_calib_hist: List[Tuple[float, int]] = []  # (raw_conf%, outcome 0/1)

def _compute_calibration() -> Tuple[float, float]:
    if len(_calib_hist) < 20:
        return 1.0, 0.0
    xs = np.array([x for x, _ in _calib_hist], dtype=float)
    ys = np.array([y * 100.0 for _, y in _calib_hist], dtype=float)
    a, b = np.polyfit(xs, ys, 1)
    return float(a), float(b)

# Leader lock using Redis (optional)
async def acquire_lock(name: str, ttl_seconds: int) -> bool:
    if not redis:
        return True
    try:
        return bool(await redis.set(name, "1", ex=ttl_seconds, nx=True))
    except Exception:
        return True

# Heuristic feature scoring (deterministic backup to LLM)

def score_features(stock_feats: Dict[str, Any], opt_sum: Dict[str, Any], news_sent: Dict[str, Any]) -> Tuple[float, float, str]:
    if not stock_feats or not opt_sum:
        return 50.0, 40.0, "Insufficient features"

    ema_trend = float(stock_feats.get("ema_trend", 0.0))
    pcr = float(opt_sum.get("put_call_volume_ratio", float("inf")))
    oir = float(opt_sum.get("put_call_oi_ratio", float("inf")))

    # Basic sentiment signal if present
    sent_raw = 0.0
    if isinstance(news_sent, dict):
        for k in ("companyNewsScore", "score", "sentiment", "buzz"):
            v = news_sent.get(k)
            if isinstance(v, (int, float)):
                sent_raw = float(v)
                break

    # Normalize components
    ema_component = float(np.tanh(ema_trend / (stock_feats.get("ema21", 1.0) or 1.0)) * 20.0)  # ±20
    pcr_capped = min(max(pcr, 0.1), 5.0)
    pcr_component = float((1.0 - (pcr_capped - 0.1) / (5.0 - 0.1)) * 25.0)  # 0..25
    oir_capped = min(max(oir, 0.1), 5.0)
    oir_component = float((1.0 - (oir_capped - 0.1) / (5.0 - 0.1)) * 15.0)  # 0..15
    sent_component = float(max(min(sent_raw, 1.0), -1.0) * 15.0)

    raw = 50.0 + ema_component + pcr_component + oir_component + sent_component
    score = float(max(min(raw, 100.0), 0.0))

    # Confidence: agreement between trend and options + magnitude
    trend_bias = 1.0 if ema_component >= 0 else -1.0
    options_bias = 1.0 if (pcr_component + oir_component) >= 0 else -1.0
    agreement = 1.0 if trend_bias == options_bias else 0.0
    conf = 40.0 + 25.0 * agreement + 0.2 * abs(ema_component) + 0.2 * (pcr_component + oir_component)
    conf = float(max(min(conf, 100.0), 0.0))

    reason = f"EMA trend {'>' if ema_trend > 0 else '<'} 0, PCR={pcr:.2f}, OIR={oir:.2f}, sentiment≈{sent_raw:.2f}"
    return score, conf, reason

# Async scheduler helpers (pure)

def next_time_at(hour: int, minute: int, tz: dt_timezone, now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(tz)
    now = now.astimezone(tz)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


def next_half_hour_boundary(now: Optional[datetime] = None, tz: Optional[dt_timezone] = None) -> datetime:
    tz = tz or dt_timezone.utc
    now = (now or datetime.now(tz)).astimezone(tz)
    minute = 30 if now.minute < 30 else 60
    next_dt = now.replace(minute=0 if minute == 60 else 30, second=0, microsecond=0)
    if minute == 60:
        next_dt = next_dt + timedelta(hours=1)
    if next_dt <= now:
        next_dt = now + timedelta(minutes=1)
        next_dt = next_dt.replace(second=0, microsecond=0)
    return next_dt

# ==============
# SSL fallback: minimal ASGI app when ssl is missing
# ==============
if not SSL_AVAILABLE:
    async def _fallback_app(scope, receive, send):
        if scope.get("type") != "http":
            return
        body = json.dumps({
            "error": "SSL module not available in this Python runtime.",
            "why": "FastAPI/AnyIO/httpx require the stdlib ssl module for HTTPS.",
            "timezone": "America/New_York" if TZ_ET_OK else "UTC (fallback; install tzdata)",
            "fix": [
                "Use a Python build with OpenSSL (official python:3.12 image or OS packages).",
                "Debian/Ubuntu: apt-get install -y libssl-dev libffi-dev then reinstall Python.",
                "Alpine: apk add openssl-dev libffi-dev then reinstall Python.",
            ],
            "note": "App started in fallback mode; network features disabled.",
        }).encode()
        headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]
        await send({"type": "http.response.start", "status": 503, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    app = _fallback_app  # type: ignore

else:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import httpx

    app = FastAPI(title="Premarket & Intraday Options Analyzer")

    class TickersIn(BaseModel):
        tickers: List[str]

    class Http:
        def __init__(self):
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
            headers = {"Accept-Encoding": "gzip, deflate"}
            self.client = httpx.AsyncClient(timeout=30, limits=limits, headers=headers)

        async def get(self, url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None, tries: int = 3) -> httpx.Response:
            backoff = 1.0
            for i in range(tries):
                try:
                    r = await self.client.get(url, params=params, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                    r.raise_for_status()
                    return r
                except httpx.HTTPError:
                    if i == tries - 1:
                        raise
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)

        async def post(self, url: str, json: Dict[str, Any] | None, headers: Dict[str, str] | None = None, tries: int = 3) -> httpx.Response:
            backoff = 1.0
            for i in range(tries):
                try:
                    r = await self.client.post(url, json=json, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 8)
                        continue
                    r.raise_for_status()
                    return r
                except httpx.HTTPError:
                    if i == tries - 1:
                        raise
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 8)

    http = Http()

    # ===== Polygon helpers =====
    async def polygon_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        base = "https://api.polygon.io"
        params = params or {}
        params["apiKey"] = POLYGON_API_KEY
        r = await http.get(base + path, params=params)
        return r.json()

    async def get_premarket_actives(limit: int = 25) -> List[str]:
        data = await polygon_get("/v2/snapshot/locale/us/markets/stocks/active")
        tickers = [it.get("ticker") for it in data.get("tickers", []) if it.get("ticker")]
        return tickers[:limit]

    async def get_stock_daily_agg(ticker: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {"adjusted": True, "sort": "asc", "limit": 5000}
        data = await polygon_get(path, params)
        return data.get("results", [])

    async def get_today_ohlc(ticker: str) -> Dict[str, Any]:
        today = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date().isoformat()
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{today}/{today}"
        params = {"adjusted": True, "sort": "asc", "limit": 1}
        data = await polygon_get(path, params)
        arr = data.get("results", [])
        return arr[0] if arr else {}

    async def get_minute_agg_today(ticker: str) -> List[Dict[str, Any]]:
        today = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date().isoformat()
        path = f"/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}"
        params = {"adjusted": True, "sort": "asc", "limit": 50000}
        data = await polygon_get(path, params)
        return data.get("results", [])

    async def get_options_snapshot_for_underlying(underlying: str, limit: int = 500) -> List[Dict[str, Any]]:
        path = f"/v3/snapshot/options/{underlying}"
        params = {"limit": limit}
        data = await polygon_get(path, params)
        return data.get("results", [])

    # ===== Finnhub helpers =====
    async def finnhub_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        base = "https://finnhub.io/api/v1"
        params = {**params, "token": FINNHUB_API_KEY}
        cache_key = f"finnhub:{path}:{json.dumps(params, sort_keys=True)}"
        cached = TTL.get(cache_key)
        if cached is not None:
            return cached
        try:
            r = await http.get(base + path, params=params)
            data = r.json()
        except Exception:
            data = {"_error": "finnhub_failed"}
        TTL.set(cache_key, data, ttl=120)
        return data

    async def get_company_news(ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        to = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date()
        frm = to - timedelta(days=days_back)
        data = await finnhub_get("/company-news", {"symbol": ticker, "from": str(frm), "to": str(to)})
        return data if isinstance(data, list) else []

    async def get_news_sentiment(ticker: str) -> Dict[str, Any]:
        data = await finnhub_get("/news-sentiment", {"symbol": ticker})
        return data if isinstance(data, dict) else {}

    # ===== OpenAI helper =====
    async def openai_rank_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        prompt = (
            "You are an equity & options analyst. Given per‑ticker features (put/call volume & OI, put/call ratios, "
            "stock average volume, EMA9 vs EMA21 trend, recent news sentiment), classify each ticker as Bullish/Bearish/Neutral. "
            "Then produce a ranked list from strongest Bullish to strongest Bearish with a confidence (0‑100). "
            "Prefer lower put/call ratios for bullish, higher for bearish, and EMA9>EMA21 as bullish. Consider positive/negative news. "
            "Return strict JSON: {\"ranked\":[{\"ticker\":...,\"stance\":...,\"confidence\":...,\"reason\":...}]}"
        )
        messages = [
            {"role": "system", "content": "You are a concise, quantitative analyst."},
            {"role": "user", "content": prompt + "\n\nDATA:\n" + json.dumps(payload, default=str)},
        ]
        body = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2}
        r = await http.post(url, json=body, headers=headers)
        out = r.json()
        try:
            text = out["choices"][0]["message"]["content"]
        except Exception:
            return {"ranked": []}
        try:
            return json.loads(text)
        except Exception:
            return {"ranked": []}

    async def merge_llm_with_scores(payload: Dict[str, Any], llm_out: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Combine heuristic score/confidence with LLM output per ticker and add regime awareness & calibration."""
        ranked = llm_out.get("ranked") if isinstance(llm_out, dict) else None
        merged: List[Dict[str, Any]] = []
        for tkr, feats in payload.items():
            sfeat = feats.get("stock_features", {})
            ofeat = feats.get("options_summary", {})
            nsent = feats.get("news_sentiment", {})
            base_score, base_conf, why = score_features(sfeat, ofeat, nsent)
            # Regime weighting
            gap = float(sfeat.get("gap_pct", 0.0))
            atr = float(sfeat.get("atr14", 0.0))
            ema_tr = float(sfeat.get("ema_trend", 0.0))
            regime_multi = 1.0 + 0.2 * np.tanh(ema_tr / (sfeat.get("ema21", 1.0) or 1.0)) + 0.1 * np.tanh(gap * 10) + 0.1 * np.tanh((atr / (sfeat.get("ema21", 1.0) or 1.0)) * 10)
            score = float(max(min(base_score * regime_multi, 100.0), 0.0))
            conf = float(max(min(base_conf * (0.9 + 0.1 * regime_multi), 100.0), 0.0))
            merged.append({
                "ticker": tkr,
                "heuristic_score": score,
                "heuristic_confidence": conf,
                "heuristic_reason": why,
                "iv_rank": ofeat.get("iv_rank_proxy"),
                "pcr": ofeat.get("put_call_volume_ratio"),
                "oir": ofeat.get("put_call_oi_ratio"),
            })
        if isinstance(ranked, list):
            llm_map = {str(it.get("ticker")).upper(): it for it in ranked if isinstance(it, dict)}
            for m in merged:
                it = llm_map.get(m["ticker"].upper(), {})
                if it:
                    m.update({"stance": it.get("stance"), "llm_confidence": it.get("confidence"), "llm_reason": it.get("reason")})
        # Calibration (simple linear fit from recent decisions if available)
        try:
            a, b = _compute_calibration()
            for m in merged:
                raw = float(m.get("llm_confidence") or m.get("heuristic_confidence") or 50)
                cal = a * raw + b
                m["calibrated_confidence"] = float(max(min(cal, 100.0), 0.0))
        except Exception:
            for m in merged:
                m["calibrated_confidence"] = float(m.get("llm_confidence") or m.get("heuristic_confidence") or 50)
        merged.sort(key=lambda x: (-(x.get("calibrated_confidence") or 0), -x["heuristic_score"]))
        return merged

    # ===== Telegram helpers =====
    async def send_telegram(text: str) -> None:
        if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await http.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

    async def send_telegram_file(filename: str, content: bytes, caption: str = "") -> None:
        if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = []
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TELEGRAM_CHAT_ID}\r\n")
        if caption:
            body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n")
        body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n")
        body_bytes = ("".join(body)).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        await http.client.post(url, content=body_bytes, headers=headers)

    # ===== Payload builders =====
    async def build_daily_payload(tickers: List[str]) -> Dict[str, Any]:
        to = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date()
        frm = to - timedelta(days=40)
        result: Dict[str, Any] = {}

        async def process(t: str):
            stock_bars = await get_stock_daily_agg(t, str(frm), str(to))
            stock_feats = compute_stock_features(stock_bars)
            # 20d z-score for stock volume
            vols = [b.get("v", 0) for b in stock_bars[-20:]]
            stock_feats["stock_vol_z20"] = zscore(vols[-1] if vols else 0.0, vols) if vols else 0.0

            opt_snapshot = await get_options_snapshot_for_underlying(t)
            und_px = (stock_bars[-1].get("c") if stock_bars else None)
            opt_summary = summarize_options_snapshot(
                opt_snapshot,
                underlying_price=und_px,
                dte_min=SETTINGS["options_dte_min"], dte_max=SETTINGS["options_dte_max"],
                moneyness_min=SETTINGS["moneyness_min"], moneyness_max=SETTINGS["moneyness_max"],
            )

            news = await get_company_news(t, days_back=7)
            sentiment = await get_news_sentiment(t)
            if sentiment.get("_error"):
                sentiment = {"missing": True}

            # RV vs IV spread proxy
            rv = stock_feats.get("rv20", 0.0)
            iv_rank = opt_summary.get("iv_rank_proxy", 50.0)
            stock_feats["rv_iv_spread"] = float(iv_rank / 100.0 - rv)

            result[t] = {
                "stock_features": stock_feats,
                "options_summary": opt_summary,
                "news_count": len(news) if isinstance(news, list) else 0,
                "news_sentiment": sentiment,
            }

        sem = asyncio.Semaphore(5)

        async def bounded(t: str):
            async with sem:
                try:
                    await process(t)
                except Exception as e:
                    result[t] = {"error": str(e)}

        await asyncio.gather(*(bounded(t) for t in tickers[:SETTINGS["universe_max"]]))
        return result

    async def build_intraday_payload(tickers: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        async def process(t: str):
            minutes = await get_minute_agg_today(t)
            cur_vol = sum([m.get("v", 0) for m in minutes])
            opt_snapshot = await get_options_snapshot_for_underlying(t, limit=300)
            und_px = (minutes[-1].get("c") if minutes else None)
            opt_summary = summarize_options_snapshot(
                opt_snapshot,
                underlying_price=und_px,
                dte_min=SETTINGS["options_dte_min"], dte_max=SETTINGS["options_dte_max"],
                moneyness_min=SETTINGS["moneyness_min"], moneyness_max=SETTINGS["moneyness_max"],
            )
            sentiment = await get_news_sentiment(t)
            if sentiment.get("_error"):
                sentiment = {"missing": True}
            result[t] = {
                "intraday_stock_volume_sum": float(cur_vol),
                "options_summary": opt_summary,
                "news_sentiment": sentiment,
            }

        sem = asyncio.Semaphore(5)

        async def bounded(t: str):
            async with sem:
                try:
                    await process(t)
                except Exception as e:
                    result[t] = {"error": str(e)}

        await asyncio.gather(*(bounded(t) for t in tickers[:SETTINGS["universe_max"]]))
        return result

    # ===== Simple in‑memory log for success tracking ========================
    decisions_log: List[Dict[str, Any]] = []  # each: {ts, ticker, stance, conf, ref_price}

    def record_decisions(now_ts: datetime, ranked_items: List[Dict[str, Any]]):
        for it in ranked_items:
            t = it.get("ticker")
            stance = (
                it.get("stance")
                or ("Bullish" if it.get("heuristic_score", 50) >= 55 else "Bearish" if it.get("heuristic_score", 50) <= 45 else "Neutral")
            )
            if not t or stance == "Neutral":
                continue
            conf_used = float(
                it.get("calibrated_confidence") or it.get("llm_confidence") or it.get("heuristic_confidence") or 50.0
            )
            decisions_log.append({
                "ts": now_ts.isoformat(),
                "ticker": t,
                "stance": stance,
                "conf": conf_used,
                "ref_price": None,
            })

    async def evaluate_today_success() -> Tuple[int, int, List[Dict[str, Any]]]:
        """Compare stance vs. same‑day price change (open→close)."""
        if not decisions_log:
            return 0, 0, []
        evals: List[Dict[str, Any]] = []
        wins = 0
        total = 0
        tickers = sorted({d["ticker"] for d in decisions_log})
        sem = asyncio.Semaphore(5)
        results: Dict[str, Dict[str, Any]] = {}

        async def fetch_one(t: str):
            async with sem:
                results[t] = await get_today_ohlc(t)

        await asyncio.gather(*(fetch_one(t) for t in tickers))
        for d in decisions_log:
            t = d["ticker"]
            stance = d["stance"]
            ohlc = results.get(t, {})
            if not ohlc:
                continue
            open_p = ohlc.get("o")
            close_p = ohlc.get("c")
            if open_p is None or close_p is None:
                continue
            move = (close_p - open_p)
            good = (move > 0 and stance == "Bullish") or (move < 0 and stance == "Bearish")
            wins += 1 if good else 0
            total += 1
            evals.append({"ticker": t, "stance": stance, "open": open_p, "close": close_p, "win": bool(good)})
        return wins, total, evals

    # ===== Orchestration jobs (asyncio loops, no multiprocessing) ==========
    async def refresh_premarket_list(limit: int = 25) -> List[str]:
        tickers = await get_premarket_actives(limit=limit)
        premarket_tickers.clear()
        premarket_tickers.update(tickers)
        return tickers

    async def daily_9am_job() -> None:
        today = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date()
        if not is_market_open_day(today):
            return
        actives = await refresh_premarket_list(limit=25)
        universe = sorted(set(actives) | set(user_tickers))[:SETTINGS["universe_max"]]
        if not universe:
            await send_telegram("No tickers in universe for daily job.")
            return
        payload = await build_daily_payload(universe)
        llm = await openai_rank_analysis(payload)
        merged = await merge_llm_with_scores(payload, llm)
        # Gating & cooldown
        kept: List[Dict[str, Any]] = []
        for it in merged:
            conf = float(it.get("calibrated_confidence") or it.get("llm_confidence") or it.get("heuristic_confidence") or 0)
            score = float(it.get("heuristic_score", 0))
            if _should_alert(it["ticker"], conf, score):
                kept.append(it)
        if not kept:
            kept = merged[:10]  # always show something at open
        warn = "_(Timezone fallback active: install tzdata for America/New_York scheduling.)_" if not TZ_ET_OK else ""
        lines: List[str] = ["*Daily 9:00 AM – Ranked Tickers*" + ("\n" + warn if warn else "")]
        for i, it in enumerate(kept[:15], 1):
            stance = it.get("stance") or ("Bullish" if it["heuristic_score"] >= 55 else "Bearish" if it["heuristic_score"] <= 45 else "Neutral")
            emoji = "🟢" if stance == "Bullish" else ("🔴" if stance == "Bearish" else "⚪️")
            conf = int(it.get("calibrated_confidence") or it.get("llm_confidence") or it.get("heuristic_confidence") or 0)
            reason = it.get("llm_reason") or it.get("heuristic_reason") or ""
            pcr = it.get("pcr"); oir = it.get("oir"); ivr = it.get("iv_rank")
            stats: List[str] = []
            if pcr is not None:
                stats.append(f"PCR {pcr:.2f}")
            if oir is not None:
                stats.append(f"OIR {oir:.2f}")
            if ivr is not None:
                stats.append(f"IVR {ivr:.0f}")
            stat_text = f"  _({' | '.join(stats)})_" if stats else ""
            lines.append(f"{i}. `{it['ticker']}` {emoji} *{stance}* — conf: {conf}%{stat_text}\n   _{reason}_")
        await send_telegram("\n".join(lines))
        record_decisions(datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc), kept[:15])
        METRICS["jobs_daily_total"] += 1

    async def intraday_30m_job() -> None:
        if not SETTINGS.get("INTRADAY_ENABLED", True):
            return
        if not TZ_ET_OK:
            return
        now = datetime.now(NY)
        if not is_market_open_day(now.date()):
            return
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if not (market_open <= now <= market_close):
            return
        universe = sorted(set(premarket_tickers) | set(user_tickers))[:SETTINGS["universe_max"]]
        if not universe:
            return
        payload = await build_intraday_payload(universe)
        llm = await openai_rank_analysis(payload)
        merged = await merge_llm_with_scores(payload, llm)
        kept: List[Dict[str, Any]] = []
        for it in merged:
            conf = float(it.get("calibrated_confidence") or it.get("llm_confidence") or it.get("heuristic_confidence") or 0)
            score = float(it.get("heuristic_score", 0))
            if _should_alert(it["ticker"], conf, score):
                kept.append(it)
        if not kept:
            return
        lines: List[str] = ["*Intraday Update (every 30m)*"]
        for i, it in enumerate(kept[:15], 1):
            stance = it.get("stance") or ("Bullish" if it["heuristic_score"] >= 55 else "Bearish" if it["heuristic_score"] <= 45 else "Neutral")
            emoji = "🟢" if stance == "Bullish" else ("🔴" if stance == "Bearish" else "⚪️")
            conf = int(it.get("calibrated_confidence") or it.get("llm_confidence") or it.get("heuristic_confidence") or 0)
            pcr = it.get("pcr"); ivr = it.get("iv_rank")
            stat = f" (PCR {pcr:.2f}, IVR {ivr:.0f})" if (pcr is not None and ivr is not None) else ""
            lines.append(f"{i}. `{it['ticker']}` {emoji} *{stance}* — conf: {conf}%{stat}")
        await send_telegram("\n".join(lines))
        METRICS["jobs_intraday_total"] += 1

    async def post_market_415pm_job() -> None:
        if not SETTINGS.get("CLOSE_REPORT_ENABLED", True):
            return
        if not TZ_ET_OK:
            return
        today = datetime.now(NY).date()
        if not is_market_open_day(today):
            return
        wins, total, evals = await evaluate_today_success()
        rate = (wins / total * 100.0) if total else 0.0
        # Update calibration history
        for d in decisions_log:
            t = d["ticker"]
            conf = float(d.get("conf", 50.0))
            match = next((e for e in evals if e["ticker"] == t), None)
            if match is not None:
                _calib_hist.append((conf, 1 if match["win"] else 0))
        # Build text
        lines: List[str] = ["*Daily Report – 4:15 PM ET*", f"Success rate today: *{rate:.1f}%* ({wins}/{total})"]
        for e in evals[:20]:
            emoji = "✅" if e["win"] else "❌"
            lines.append(f"{emoji} `{e['ticker']}` {e['stance']} — open: {e['open']:.2f}, close: {e['close']:.2f}")
        msg = "\n".join(lines) if len(lines) > 2 else "Daily report: no decisions to evaluate."
        await send_telegram(msg)
        # CSV artifact
        try:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["ticker", "stance", "open", "close", "win"])
            for e in evals:
                w.writerow([e["ticker"], e["stance"], e["open"], e["close"], int(e["win"])])
            await send_telegram_file("daily_report.csv", buf.getvalue().encode(), caption="Daily results CSV")
        except Exception:
            pass
        decisions_log.clear()
        METRICS["jobs_close_total"] += 1

    # ===== Asyncio scheduler loops =====
    async def _daily_loop():
        if not SETTINGS.get("SCHEDULER_ENABLED", True):
            return
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_time_at(9, 0, tz, now)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                if await acquire_lock("lock:daily", 30 * 60):
                    await daily_9am_job()
            except Exception as e:
                METRICS["errors_total"] += 1
                await send_telegram(f"Daily job failed: {e}")

    async def _intraday_loop():
        if not SETTINGS.get("SCHEDULER_ENABLED", True) or not SETTINGS.get("INTRADAY_ENABLED", True):
            return
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_half_hour_boundary(now, tz)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                if await acquire_lock("lock:intraday", 10 * 60):
                    await intraday_30m_job()
            except Exception as e:
                METRICS["errors_total"] += 1
                await send_telegram(f"Intraday job failed: {e}")

    async def _close_loop():
        if not SETTINGS.get("SCHEDULER_ENABLED", True) or not SETTINGS.get("CLOSE_REPORT_ENABLED", True):
            return
        if not TZ_ET_OK:
            return
        tz = NY
        while True:
            now = datetime.now(tz)
            nxt = next_time_at(16, 15, tz, now)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                if await acquire_lock("lock:close", 30 * 60):
                    await post_market_415pm_job()
            except Exception as e:
                METRICS["errors_total"] += 1
                await send_telegram(f"Close report failed: {e}")

    # ===== API =====
    @app.post("/test/telegram")
    async def test_telegram_post(payload: Dict[str, Any]):
        text = (payload or {}).get("text")
        if not text or not isinstance(text, str):
            raise HTTPException(400, "Provide JSON body with {'text': '<message>'}")
        await send_telegram(text)
        return {"ok": True, "sent": True, "len": len(text)}

    @app.get("/test/telegram")
    async def test_telegram_get(text: str = "Test alert: system is online ✅"):
        await send_telegram(text)
        return {"ok": True, "sent": True, "len": len(text)}

    @app.on_event("startup")
    async def on_startup():
        if SETTINGS.get("SCHEDULER_ENABLED", True):
            _bg_tasks.append(asyncio.create_task(_daily_loop()))
            _bg_tasks.append(asyncio.create_task(_intraday_loop()))
            _bg_tasks.append(asyncio.create_task(_close_loop()))
        if redis:
            try:
                await redis.ping()
            except Exception:
                pass

    @app.on_event("shutdown")
    async def on_shutdown():
        for t in _bg_tasks:
            t.cancel()
        try:
            await asyncio.gather(*_bg_tasks, return_exceptions=True)
        except Exception:
            pass
        try:
            await http.client.aclose()
        except Exception:
            pass
        if redis:
            try:
                await redis.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass

    @app.get("/healthz")
    async def healthz():
        tz_name = getattr(NY, "key", None) if TZ_ET_OK else "UTC (fallback)"
        return {"status": "ok", "ssl": True, "timezone": tz_name, "tz_et_ok": TZ_ET_OK}

    @app.get("/status")
    async def status():
        return {
            "premarket_tickers": sorted(premarket_tickers),
            "user_tickers": sorted(user_tickers),
            "universe": sorted(set(premarket_tickers) | set(user_tickers)),
            "last_daily_ranking_count": len(last_daily_ranking),
            "last_intraday_ranking_count": len(last_intraday_ranking),
            "tz_et_ok": TZ_ET_OK,
            "settings": {k: SETTINGS[k] for k in SETTINGS},
            "app_base_path": APP_BASE_PATH or "/",
        }

    @app.post("/tickers")
    async def set_user_tickers(body: "TickersIn"):
        cleaned = [t.upper().strip() for t in body.tickers if t and isinstance(t, str)]
        user_tickers.clear()
        user_tickers.update(cleaned)
        return {"ok": True, "count": len(user_tickers)}

    @app.post("/run/daily")
    async def run_daily_now():
        await daily_9am_job()
        return {"ok": True, "ran": "daily"}

    @app.post("/run/intraday")
    async def run_intraday_now():
        await intraday_30m_job()
        return {"ok": True, "ran": "intraday"}

    @app.get("/settings")
    async def get_settings():
        return {"settings": SETTINGS}

    @app.post("/settings")
    async def set_settings(body: Dict[str, Any]):
        allowed = {
            "options_dte_min",
            "options_dte_max",
            "moneyness_min",
            "moneyness_max",
            "alert_llm_conf_threshold",
            "alert_score_threshold",
            "alert_cooldown_minutes",
            "universe_max",
            "SCHEDULER_ENABLED",
            "INTRADAY_ENABLED",
            "CLOSE_REPORT_ENABLED",
        }
        changed: Dict[str, Any] = {}
        for k, v in (body or {}).items():
            if k in allowed:
                SETTINGS[k] = v
                changed[k] = v
        return {"ok": True, "changed": changed}

    @app.get("/metrics")
    async def metrics():
        lines = ["# HELP app_jobs_total Total jobs run", "# TYPE app_jobs_total counter"]
        lines.append(f"app_jobs_total{{type=\"daily\"}} {METRICS['jobs_daily_total']}")
        lines.append(f"app_jobs_total{{type=\"intraday\"}} {METRICS['jobs_intraday_total']}")
        lines.append(f"app_jobs_total{{type=\"close\"}} {METRICS['jobs_close_total']}")
        lines.append("# HELP app_errors_total Total errors")
        lines.append("# TYPE app_errors_total counter")
        lines.append(f"app_errors_total {METRICS['errors_total']}")
        return "\n".join(lines)

# ======================
# Self‑tests (no network)
# ======================

def _selftest() -> None:
    # EMA test: constant series should converge to the same constant
    s = [10.0] * 20
    e9 = ema(s, 9)
    assert abs(e9[-1] - 10.0) < 1e-6, "EMA should track constant series"

    # Options summary basic test
    snapshot = [
        {"details": {"contract_type": "call", "strike_price": 100, "expiration_date": "2099-01-01"}, "day": {"volume": 100}, "open_interest": 300},
        {"details": {"contract_type": "put",  "strike_price": 100, "expiration_date": "2099-01-01"}, "day": {"volume": 50},  "open_interest": 600},
    ]
    sm = summarize_options_snapshot(snapshot, underlying_price=100)
    assert sm["call_volume"] == 100 and sm["put_volume"] == 50, "Volume aggregation failed"
    assert sm["call_open_interest"] == 300 and sm["put_open_interest"] == 600, "OI aggregation failed"
    assert sm["put_call_volume_ratio"] == 0.5, "PCR calc failed"

    # Stock features test: upward prices => ema_trend > 0
    prices = [{"c": 1 + i, "v": 1000 + i * 10, "h": 1 + i + 0.2, "l": 1 + i - 0.2, "o": 1 + i - 0.1} for i in range(30)]
    feats = compute_stock_features(prices)
    assert feats["ema_trend"] > 0, "EMA trend should be positive for rising prices"

    # Timezone resolver test
    now = datetime.now(NY)
    assert hasattr(now, "tzinfo") and now.tzinfo is not None, "Timezone resolver returned naive datetime"

    # Scheduler helpers tests
    tz = dt_timezone.utc
    n1 = datetime(2024, 1, 1, 8, 59, tzinfo=tz)
    r1 = next_time_at(9, 0, tz, n1)
    assert r1.date() == n1.date() and r1.hour == 9 and r1.minute == 0, "next_time_at before boundary failed"

    n2 = datetime(2024, 1, 1, 9, 0, tzinfo=tz)
    r2 = next_time_at(9, 0, tz, n2)
    assert r2.date() == (n2 + timedelta(days=1)).date() and r2.hour == 9, "next_time_at at boundary failed"

    n3 = datetime(2024, 1, 1, 10, 15, tzinfo=tz)
    h1 = next_half_hour_boundary(n3, tz)
    assert h1.hour == 10 and h1.minute == 30, "next_half_hour_boundary 10:15 -> 10:30 failed"

    n4 = datetime(2024, 1, 1, 10, 30, tzinfo=tz)
    h2 = next_half_hour_boundary(n4, tz)
    assert h2.hour == 11 and h2.minute == 0, "next_half_hour_boundary 10:30 -> 11:00 failed"

    # Scoring logic tests (bullish bias)
    stock_feats = {"ema_trend": 1.0, "ema21": 100.0}
    opt_sum = {"put_call_volume_ratio": 0.8, "put_call_oi_ratio": 0.9}
    news_sent = {"companyNewsScore": 0.2}
    score, conf, _ = score_features(stock_feats, opt_sum, news_sent)
    assert 50 < score <= 100 and 40 <= conf <= 100, "Scoring should produce bullish bias with reasonable confidence"

    # PCR overflow/underflow safety
    sm_inf = {"put_call_volume_ratio": float("inf"), "put_call_oi_ratio": float("inf")}
    s2, c2, _ = score_features({"ema_trend": 0.0, "ema21": 100.0}, sm_inf, {})
    assert 0 <= s2 <= 100 and 0 <= c2 <= 100, "Scoring must clamp with inf ratios"

    # DST transition sanity (if tzdata available)
    if TZ_ET_OK:
        spring = datetime(2024, 3, 10, 8, 0, tzinfo=NY)
        r = next_time_at(9, 0, NY, spring)
        assert r.hour == 9, "next_time_at should target 9 ET even across DST"

    # Options filter keeps only intended DTE/moneyness
    nowd = datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)
    contracts = [
        {"details": {"contract_type": "call", "strike_price": 100, "expiration_date": (nowd + timedelta(days=7)).date().isoformat()}, "day": {"volume": 10}, "open_interest": 5},
        {"details": {"contract_type": "put",  "strike_price": 100, "expiration_date": (nowd + timedelta(days=30)).date().isoformat()}, "day": {"volume": 20}, "open_interest": 15},
    ]
    summ = summarize_options_snapshot(contracts, underlying_price=100, dte_min=5, dte_max=21, moneyness_min=0.9, moneyness_max=1.1)
    assert summ["call_volume"] == 10 and (summ["put_volume"] == 0 or summ["put_volume"] == 0.0), "Options filter by DTE failed"

    print("Selftest OK – core computations pass.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    print("Run with uvicorn: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload")
