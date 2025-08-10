# fastapi_app.py
"""
🚀 FastAPI service with robust SSL & timezone fallbacks + **asyncio-based scheduler** (no multiprocessing) and self-tests.

This app does:
1) Fetch daily pre‑market active tickers from Polygon and refresh the list each morning.
2) Maintain a user‑provided ticker list.
3) 9:00 AM ET job: pull ~20 trading days of stock + options snapshot data, Finnhub news/sentiment, compute EMA9/EMA21 + volume, and analyze with OpenAI to rank tickers.
4) Send rankings to Telegram.
5) Every 30 minutes during market hours, fetch intraday stock volume + options snapshot, enrich with sentiment, analyze again, and alert.

✅ New in this revision (to fix your errors):
- Handles environments missing the built‑in `ssl` module by exposing a fallback ASGI app (503 JSON) so the process starts instead of crashing.
- **Fixes `ZoneInfoNotFoundError` for `America/New_York`** by trying to load `tzdata` if available and falling back to UTC when the IANA database is absent.
- **Replaces APScheduler** (which indirectly imports `_multiprocessing`) with a **pure-asyncio scheduler**. This avoids `ModuleNotFoundError: No module named '_multiprocessing'` in sandboxed/minimal Python builds.
- Advertises timezone status via `/healthz`, and **skips intraday scheduling** when a proper NY timezone is unavailable (prevents wrong timing).
- Added self‑tests for EMA, options summary, stock features, timezone resolver, **and the new scheduling helpers**. Run: `python fastapi_app.py --selftest`.

⚙️ Environment (.env):
- POLYGON_API_KEY=
- FINNHUB_API_KEY=
- OPENAI_API_KEY=
- TELEGRAM_BOT_TOKEN=
- TELEGRAM_CHAT_ID=

Run: `uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload`

🛠️ If you see `SSL module not available`:
- Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y build-essential libssl-dev libffi-dev && pyenv install 3.12.4`
- Alpine: `apk add openssl-dev libffi-dev` then reinstall Python
- Prefer official `python:3.12` images or ensure OpenSSL headers are present during build.

🕒 If you see `No time zone found with key America/New_York`:
- `pip install tzdata` (Python package supplying the IANA database)
- Or install OS tzdata: Debian/Ubuntu `apt-get install -y tzdata`; Alpine `apk add tzdata`
- In Pyodide: `await pyodide.loadPackage('tzdata')` before importing `zoneinfo`.
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
from dotenv import load_dotenv

# --- SSL preflight -----------------------------------------------------------
try:
    import ssl  # noqa: F401
    SSL_AVAILABLE = True
except Exception:
    SSL_AVAILABLE = False

# ===== Config =====
load_dotenv()

# --- Timezone resolver (America/New_York with fallbacks) --------------------
TZ_ET_OK = False
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        NY = ZoneInfo("America/New_York")
        TZ_ET_OK = True
    except Exception:
        # Try to load tzdata dynamically if present
        try:
            import tzdata  # noqa: F401  # ensures IANA DB is available to zoneinfo
            NY = ZoneInfo("America/New_York")
            TZ_ET_OK = True
        except Exception:
            NY = dt_timezone.utc  # ultimate fallback
            TZ_ET_OK = False
except Exception:
    # zoneinfo entirely missing – use UTC
    NY = dt_timezone.utc
    TZ_ET_OK = False

# ===== Globals (shared) =====
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

premarket_tickers: Set[str] = set()       # refreshed every morning
user_tickers: Set[str] = set()            # user‑managed
last_daily_ranking: List[Dict[str, Any]] = []
last_intraday_ranking: List[Dict[str, Any]] = []

# Track background tasks for clean shutdown
_bg_tasks: List[asyncio.Task] = []

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


def summarize_options_snapshot(contracts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate put/call volume & OI from a Polygon options snapshot list."""
    put_vol = 0.0
    call_vol = 0.0
    put_oi = 0.0
    call_oi = 0.0
    for c in contracts:
        o_type = (c.get("details", {}).get("contract_type") or c.get("contract_type") or "").lower()
        vol = (
            c.get("day", {}).get("volume")
            or c.get("last_quote", {}).get("volume")
            or c.get("volume")
            or 0
        )
        oi = (
            c.get("open_interest")
            or c.get("day", {}).get("open_interest")
            or 0
        )
        if o_type == "put":
            put_vol += float(vol)
            put_oi += float(oi)
        elif o_type == "call":
            call_vol += float(vol)
            call_oi += float(oi)
    pcr = (put_vol / call_vol) if call_vol else math.inf
    oir = (put_oi / call_oi) if call_oi else math.inf
    return {
        "put_volume": float(put_vol),
        "call_volume": float(call_vol),
        "put_call_volume_ratio": float(pcr),
        "put_open_interest": float(put_oi),
        "call_open_interest": float(call_oi),
        "put_call_oi_ratio": float(oir),
    }


def compute_stock_features(daily_bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not daily_bars:
        return {}
    closes = [b.get("c") for b in daily_bars if b.get("c") is not None]
    volumes = [b.get("v") for b in daily_bars if b.get("v") is not None]
    if not closes:
        return {}
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema_trend = (ema9[-1] - ema21[-1]) if ema9 and ema21 else 0.0
    avg_vol = float(np.mean(volumes[-20:])) if volumes else 0.0
    return {
        "ema9": float(ema9[-1]) if ema9 else None,
        "ema21": float(ema21[-1]) if ema21 else None,
        "ema_trend": float(ema_trend),
        "avg_volume_20d": avg_vol,
    }

# ===== Async scheduler helpers (pure) =======================================

def next_time_at(hour: int, minute: int, tz: dt_timezone, now: Optional[datetime] = None) -> datetime:
    """Return the next datetime strictly after `now` occurring at hour:minute in tz."""
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
        # Safety net (should not happen, but keep monotonic guarantee)
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

    # Expose a valid ASGI callable for uvicorn
    app = _fallback_app  # type: ignore

else:
    # Normal path with full features (no APScheduler; pure asyncio tasks)
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import httpx

    app = FastAPI(title="Premarket & Intraday Options Analyzer")

    # ===== Models =====
    class TickersIn(BaseModel):
        tickers: List[str]

    # ===== HTTP helpers =====
    class Http:
        def __init__(self):
            self.client = httpx.AsyncClient(timeout=30)

        async def get(self, url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None, tries: int = 3) -> httpx.Response:
            backoff = 1.0
            for i in range(tries):
                try:
                    r = await self.client.get(url, params=params, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    r.raise_for_status()
                    return r
                except httpx.HTTPError:
                    if i == tries - 1:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= 2

        async def post(self, url: str, json: Dict[str, Any], headers: Dict[str, str] | None = None, tries: int = 3) -> httpx.Response:
            backoff = 1.0
            for i in range(tries):
                try:
                    r = await self.client.post(url, json=json, headers=headers)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    r.raise_for_status()
                    return r
                except httpx.HTTPError:
                    if i == tries - 1:
                        raise
                    await asyncio.sleep(backoff)
                    backoff *= 2

    http = Http()

    # ===== Polygon helpers =====
    async def polygon_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        base = "https://api.polygon.io"
        params = params or {}
        params["apiKey"] = POLYGON_API_KEY
        r = await http.get(base + path, params=params)
        return r.json()

    async def get_premarket_actives(limit: int = 25) -> List[str]:
        """Most‑active tickers snapshot. During pre‑market, this reflects pre‑market.
        Endpoint: /v2/snapshot/locale/us/markets/stocks/active
        """
        data = await polygon_get("/v2/snapshot/locale/us/markets/stocks/active")
        tickers = [it.get("ticker") for it in data.get("tickers", []) if it.get("ticker")]
        return tickers[:limit]

    async def get_stock_daily_agg(ticker: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {"adjusted": True, "sort": "asc", "limit": 5000}
        data = await polygon_get(path, params)
        return data.get("results", [])

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
        r = await http.get(base + path, params=params)
        return r.json()

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
            {"role": "user", "content": prompt + "\n\nDATA:\n" + str(payload)},
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

    # ===== Telegram helper =====
    async def send_telegram(text: str) -> None:
        if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await http.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

    # ===== Payload builders =====
    async def build_daily_payload(tickers: List[str]) -> Dict[str, Any]:
        to = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date()
        frm = to - timedelta(days=40)  # capture ~20 trading days
        result: Dict[str, Any] = {}

        async def process(t: str):
            stock_bars = await get_stock_daily_agg(t, str(frm), str(to))
            stock_feats = compute_stock_features(stock_bars)
            opt_snapshot = await get_options_snapshot_for_underlying(t)
            opt_summary = summarize_options_snapshot(opt_snapshot)
            news = await get_company_news(t, days_back=7)
            sentiment = await get_news_sentiment(t)
            result[t] = {
                "stock_features": stock_feats,
                "options_summary": opt_summary,
                "news_count": len(news),
                "news_sentiment": sentiment,
            }

        sem = asyncio.Semaphore(5)

        async def bounded(t: str):
            async with sem:
                try:
                    await process(t)
                except Exception as e:
                    result[t] = {"error": str(e)}

        await asyncio.gather(*(bounded(t) for t in tickers))
        return result

    async def build_intraday_payload(tickers: List[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        async def process(t: str):
            minutes = await get_minute_agg_today(t)
            cur_vol = sum([m.get("v", 0) for m in minutes])
            opt_snapshot = await get_options_snapshot_for_underlying(t, limit=300)
            opt_summary = summarize_options_snapshot(opt_snapshot)
            sentiment = await get_news_sentiment(t)
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

        await asyncio.gather(*(bounded(t) for t in tickers))
        return result

    # ===== Orchestration jobs (asyncio loops, no multiprocessing) ==========
    async def refresh_premarket_list(limit: int = 25) -> List[str]:
        tickers = await get_premarket_actives(limit=limit)
        premarket_tickers.clear()
        premarket_tickers.update(tickers)
        return tickers

    async def daily_9am_job() -> None:
        # If timezone is UTC fallback, still run daily job at the scheduler's 9:00 (UTC) but warn in Telegram
        actives = await refresh_premarket_list(limit=25)
        universe = sorted(set(actives) | set(user_tickers))
        if not universe:
            await send_telegram("No tickers in universe for daily job.")
            return
        payload = await build_daily_payload(universe)
        analysis = await openai_rank_analysis(payload)
        global last_daily_ranking
        last_daily_ranking = analysis.get("ranked", [])
        warn = "\n_(Timezone fallback active: install tzdata for America/New_York scheduling.)_" if not TZ_ET_OK else ""
        msg_lines = ["*Daily 9:00 AM – Ranked Tickers*" + warn + "\n"]
        for i, item in enumerate(last_daily_ranking, 1):
            msg_lines.append(f"{i}. `{item.get('ticker','?')}` — *{item.get('stance','?')}* ({item.get('confidence','?')}%)\n   _{item.get('reason','')}_")
        await send_telegram("\n".join(msg_lines) or "Daily job complete")

    async def intraday_30m_job() -> None:
        # Only run intraday windows when true NY timezone is available
        if not TZ_ET_OK:
            return
        now = datetime.now(NY)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if not (market_open <= now <= market_close):
            return
        universe = sorted(set(premarket_tickers) | set(user_tickers))
        if not universe:
            return
        payload = await build_intraday_payload(universe)
        analysis = await openai_rank_analysis(payload)
        global last_intraday_ranking
        last_intraday_ranking = analysis.get("ranked", [])
        lines = ["*Intraday Update (every 30m)*\n"]
        for i, item in enumerate(last_intraday_ranking[:15], 1):
            lines.append(f"{i}. `{item.get('ticker','?')}` — *{item.get('stance','?')}* ({item.get('confidence','?')}%)")
        await send_telegram("\n".join(lines))

    # ===== Asyncio scheduler loops =====
    async def _daily_loop():
        # Wait until the next 9:00 (NY if available, else UTC)
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_time_at(9, 0, tz, now)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                await daily_9am_job()
            except Exception as e:
                # Avoid crashing the loop; log to Telegram for visibility
                await send_telegram(f"Daily job failed: {e}")
            # Loop will compute the next run again

    async def _intraday_loop():
        # Tick on each half-hour boundary; run job only in market window
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_half_hour_boundary(now, tz)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                await intraday_30m_job()
            except Exception as e:
                await send_telegram(f"Intraday job failed: {e}")

    # ===== API =====
    @app.on_event("startup")
    async def on_startup():
        # Start background loops
        _bg_tasks.append(asyncio.create_task(_daily_loop()))
        _bg_tasks.append(asyncio.create_task(_intraday_loop()))

    @app.on_event("shutdown")
    async def on_shutdown():
        # Cancel background tasks and close http client
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

    @app.get("/healthz")
    async def healthz():
        tz_name = getattr(NY, "key", None) if TZ_ET_OK else "UTC (fallback)"
        return {"status": "ok", "ssl": True, "timezone": tz_name, "tz_et_ok": TZ_ET_OK}

    # --- Add test Telegram endpoints here ---
@app.post("/test/telegram")
async def test_telegram_post(payload: Dict[str, Any]):
    """Send a test alert to Telegram with provided text."""
    text = (payload or {}).get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(400, "Provide JSON body with {'text': '<message>'}")
    await send_telegram(text)
    return {"ok": True, "sent": True, "len": len(text)}

@app.get("/test/telegram")
async def test_telegram_get(text: str = "Test alert: system is online ✅"):
    await send_telegram(text)
    return {"ok": True, "sent": True, "len": len(text)}
    

    @app.get("/status")
    async def status():
        return {
            "premarket_tickers": sorted(premarket_tickers),
            "user_tickers": sorted(user_tickers),
            "universe": sorted(set(premarket_tickers) | set(user_tickers)),
            "last_daily_ranking_count": len(last_daily_ranking),
            "last_intraday_ranking_count": len(last_intraday_ranking),
            "tz_et_ok": TZ_ET_OK,
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

    @app.get("/validate/{ticker}")
    async def validate_ticker(ticker: str):
        try:
            data = await polygon_get("/v3/reference/tickers", {"ticker": ticker.upper(), "limit": 1})
            count = data.get("count", 0)
            return {"valid": count > 0}
        except Exception as e:
            raise HTTPException(502, f"validation failed: {e}")

# ======================
# Self‑tests (no network)
# ======================

def _selftest() -> None:
    # EMA test: constant series should converge to the same constant
    s = [10.0] * 20
    e9 = ema(s, 9)
    assert abs(e9[-1] - 10.0) < 1e-6, "EMA should track constant series"

    # Options summary test
    snapshot = [
        {"details": {"contract_type": "call"}, "day": {"volume": 100}, "open_interest": 300},
        {"details": {"contract_type": "put"},  "day": {"volume": 50},  "open_interest": 600},
    ]
    sm = summarize_options_snapshot(snapshot)
    assert sm["call_volume"] == 100 and sm["put_volume"] == 50, "Volume aggregation failed"
    assert sm["call_open_interest"] == 300 and sm["put_open_interest"] == 600, "OI aggregation failed"
    assert sm["put_call_volume_ratio"] == 0.5, "PCR calc failed"

    # Stock features test: upward prices => ema_trend > 0
    prices = [{"c": 1+i, "v": 1000+i*10} for i in range(30)]
    feats = compute_stock_features(prices)
    assert feats["ema_trend"] > 0, "EMA trend should be positive for rising prices"

    # Timezone resolver test
    now = datetime.now(NY)
    assert hasattr(now, "tzinfo") and now.tzinfo is not None, "Timezone resolver returned naive datetime"

    # Scheduler helpers tests (new)
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

    if not SSL_AVAILABLE:
        print("Selftest OK (SSL MISSING mode) – core computations pass.")
    else:
        tz_msg = "NY" if TZ_ET_OK else "UTC fallback"
        print(f"Selftest OK – core computations pass. TZ={tz_msg}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    print("Run with uvicorn: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload")
