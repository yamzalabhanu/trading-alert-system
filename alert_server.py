"""
FastAPI service: Polygon + Finnhub + OpenAI + Telegram
- Keep a user-managed ticker list
- 9:00 AM America/New_York (or UTC fallback): pull ~20 trading days option & stock context, fetch news, analyze via LLM, rank, send Telegram
- Every 30 minutes between 9:30 AM and 4:00 PM ET on trading days: intraday refresh (options + stock), indicators, news enrichment, LLM analysis, Telegram

Env (.env)
- POLYGON_API_KEY=
- FINNHUB_API_KEY=
- OPENAI_API_KEY=
- TELEGRAM_BOT_TOKEN=
- TELEGRAM_CHAT_ID=   # numeric chat id or @channelusername

Run
  uvicorn app:app --host 0.0.0.0 --port 8000

Notes
- This version guards against environments where Python was built **without** the `ssl` module.
  If `ssl` is missing, we still export a minimal ASGI app that returns a clear diagnostic instead of crashing.
- It also guards against missing IANA timezone data (tzdata). If `zoneinfo.ZoneInfo('America/New_York')`
  is unavailable, we fall back to UTC and surface a clear hint. On **Pyodide**, you must load tzdata first:
    >>> import pyodide
    >>> await pyodide.loadPackage('tzdata')
  then (re)import this module.
- When `ssl` and tzdata are available, the full FastAPI stack runs.
- Fixed premarket filter (proper `datetime.time(9, 30)` comparison).
- Added more self-tests: EMA/VWAP/breakout as before + timezone resolution + premarket guard.
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta, timezone, time as dt_time
from typing import Any, Dict, List, Optional, Tuple

# ---------------- Timezone guard (handles missing tzdata / Pyodide) ----------------
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:  # very unlikely on 3.9+
    ZoneInfo = None  # type: ignore
    ZoneInfoNotFoundError = Exception  # type: ignore


def _resolve_ny_tz() -> Tuple[object, str, Optional[str]]:
    """Return (tzinfo, source, warning). Fallback to UTC if tzdata is unavailable.
    If running on Pyodide (emscripten), include an explicit instruction to load tzdata.
    """
    # Prefer real zone if possible
    if ZoneInfo is not None:
        try:
            return ZoneInfo("America/New_York"), "zoneinfo", None
        except Exception as e:  # ZoneInfoNotFoundError
            pass
    warn = (
        "No time zone found with key America/New_York. "
        "If you're on Pyodide, run: `await pyodide.loadPackage('tzdata')` before importing. "
        "Falling back to UTC; schedules will use UTC times."
    )
    try:
        if sys.platform == "emscripten":  # Pyodide hint
            warn = (
                "Detected Pyodide. Load tzdata first:\n"
                "    import pyodide\n    await pyodide.loadPackage('tzdata')\n"
                "then re-import this module. Falling back to UTC for now."
            )
    except Exception:
        pass
    return timezone.utc, "utc-fallback", warn


NY, TZ_SOURCE, TZ_WARNING = _resolve_ny_tz()

# ---------------- SSL guard (critical for FastAPI/AnyIO) ----------------
try:
    import ssl  # noqa: F401
    SSL_AVAILABLE = True
except ModuleNotFoundError:
    ssl = None  # type: ignore
    SSL_AVAILABLE = False

# Keep these globally so both fast path and fallback can reference them
USER_TICKERS: List[str] = []

# ---------------- Minimal ASGI app used when ssl is NOT available ----------------
if not SSL_AVAILABLE:
    async def app(scope, receive, send):  # type: ignore[misc]
        if scope["type"] != "http":
            await send({"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Only HTTP supported in fallback."})
            return
        path = scope.get("path", "/")
        if path == "/health":
            payload = {
                "ok": False,
                "ssl_available": False,
                "tz_source": TZ_SOURCE,
                "tz_warning": TZ_WARNING,
                "reason": "Python was built without the 'ssl' module. FastAPI/AnyIO require it, and HTTPS calls to Polygon/Finnhub/Telegram/OpenAI cannot work.",
                "fix": [
                    "Install Python with OpenSSL support (e.g., apt-get install libssl-dev then reinstall Python).",
                    "If on Alpine, use a Python build linked with openssl (not LibreSSL) or install py3-openssl.",
                    "On Render/Docker, use an official python image (e.g., python:3.12-slim) which includes ssl.",
                ],
                "tickers": USER_TICKERS,
                "tz": str(NY),
                "now": datetime.now(tz=NY).isoformat(),
            }
            body = json.dumps(payload).encode()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
            })
            await send({"type": "http.response.body", "body": body})
            return
        # default response
        payload = {
            "error": "ssl module missing",
            "details": "Cannot start FastAPI stack. Hit /health for guidance.",
            "tz_source": TZ_SOURCE,
            "tz_warning": TZ_WARNING,
        }
        body = json.dumps(payload).encode()
        await send({
            "type": "http.response.start",
            "status": 503,
            "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")],
        })
        await send({"type": "http.response.body", "body": body})

# ---------------- Full implementation when ssl IS available ----------------
else:
    import math
    from cachetools import TTLCache
    import httpx
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from dotenv import load_dotenv
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    # ---------- Config ----------
    load_dotenv()
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    if not all([POLYGON_API_KEY, FINNHUB_API_KEY, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        print("[WARN] One or more API keys are missing. Set them in .env before deploying.")

    if TZ_WARNING:
        print(f"[WARN] {TZ_WARNING}")

    # ---------- App & State ----------
    app = FastAPI(title="Options LLM Analyst")

    # Light caches to avoid hammering providers
    news_cache = TTLCache(maxsize=512, ttl=60 * 15)       # 15 min
    ref_cache = TTLCache(maxsize=512, ttl=60 * 60 * 12)   # 12 hr

    # Shared async client
    client = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(20.0, read=30.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )

    # OpenAI client (async)
    openai_client: Optional[Any] = None
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[WARN] OpenAI client init failed: {e}")

    # ---------- Models ----------
    class TickerList(BaseModel):
        tickers: List[str] = Field(default_factory=list, description="Uppercase ticker symbols")

    class AnalysisItem(BaseModel):
        symbol: str
        score: float
        stance: str  # bullish/bearish/neutral
        rationale: str

    class RankedResult(BaseModel):
        as_of: str
        items: List[AnalysisItem]

    # ---------- Utilities: indicators ----------
    def ema(values: List[float], length: int) -> Optional[float]:
        if not values or len(values) < length:
            return None
        k = 2 / (length + 1)
        # seed with SMA
        sma = sum(values[:length]) / length
        e = sma
        for v in values[length:]:
            e = v * k + e * (1 - k)
        return e

    def vwap(high: List[float], low: List[float], close: List[float], volume: List[float]) -> Optional[float]:
        if not close or len(close) != len(volume):
            return None
        typical_price = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
        pv = sum(tp * v for tp, v in zip(typical_price, volume))
        vol_sum = sum(volume)
        return pv / vol_sum if vol_sum else None

    # ---------- Providers ----------
    async def polygon_stock_agg(symbol: str, start: datetime, end: datetime, timespan: str = "minute") -> Dict[str, Any]:
        # Example: /v2/aggs/ticker/AAPL/range/1/minute/2024-10-01/2024-10-10
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start.date()}/{end.date()}"
        params = {"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY}
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def polygon_options_snapshot_underlying(symbol: str) -> Dict[str, Any]:
        # Underlying snapshot (price/volume)
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        r = await client.get(url, params={"apiKey": POLYGON_API_KEY})
        r.raise_for_status()
        return r.json()

    async def polygon_options_activity(symbol: str, window_days: int = 20) -> Dict[str, Any]:
        """Indicative options activity fetcher. Adjust to your polygon plan/endpoints.
        Strategy: get daily aggregates for options volume/OI by filtering to near ATM and near expiry on client side if needed.
        """
        end = datetime.now(tz=NY).date()
        start = end - timedelta(days=window_days * 2)  # pad for weekends/holidays
        # Example: using stocks aggregates as proxy for volume trend + separate snapshot for options; replace with your options endpoint
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        r = await client.get(url, params={"adjusted": "true", "sort": "asc", "apiKey": POLYGON_API_KEY})
        r.raise_for_status()
        return r.json()

    async def finnhub_company_news(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        key = f"news:{symbol}:{start.date()}:{end.date()}"
        if key in news_cache:
            return news_cache[key]
        url = "https://finnhub.io/api/v1/company-news"
        params = {"symbol": symbol, "from": start.date().isoformat(), "to": end.date().isoformat(), "token": FINNHUB_API_KEY}
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        news_cache[key] = data
        return data

    async def telegram_send(message: str) -> None:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            print("[WARN] Telegram not configured; skipping send.")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            r = await client.post(url, data=payload)
            r.raise_for_status()
        except Exception as e:
            print(f"[ERR] Telegram send failed: {e}")

    # ---------- Feature engineering ----------
    def is_premarket_ms(ts_ms: int, tz: object) -> bool:
        """Return True if the timestamp (ms) is before 09:30 local time for the given tz."""
        t_local = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz).time()
        return t_local < dt_time(9, 30)

    async def compute_features(symbol: str, now_ny: datetime) -> Dict[str, Any]:
        # Pull ~20 trading days stock aggregates (daily) for trend + intraday minute bars for vwap and premkt levels
        start_daily = (now_ny - timedelta(days=40)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_daily = now_ny
        daily = await polygon_stock_agg(symbol, start_daily, end_daily, timespan="day")

        closes = [b.get("c") for b in daily.get("results", [])]
        highs = [b.get("h") for b in daily.get("results", [])]
        lows = [b.get("l") for b in daily.get("results", [])]
        vols = [b.get("v") for b in daily.get("results", [])]

        ema9_v = ema(closes, 9) if closes else None
        ema21_v = ema(closes, 21) if closes else None
        ema50_v = ema(closes, 50) if closes else None

        # Premarket high/low approximation: minute bars before 09:30 local today
        start_intraday = now_ny.replace(hour=4, minute=0, second=0, microsecond=0)
        end_intraday = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        intraday = await polygon_stock_agg(symbol, start_intraday, end_intraday, timespan="minute")
        ibars = intraday.get("results", [])

        premkt = [b for b in ibars if is_premarket_ms(b.get("t", 0), NY)]
        premkt_high = max((b.get("h", float("-inf")) for b in premkt), default=None)
        premkt_low = min((b.get("l", float("inf")) for b in premkt), default=None)

        # VWAP (session)
        v = None
        if ibars:
            v = vwap(
                [b.get("h", 0.0) for b in ibars],
                [b.get("l", 0.0) for b in ibars],
                [b.get("c", 0.0) for b in ibars],
                [b.get("v", 0.0) for b in ibars],
            )

        # Previous 5 days highs/lows
        last5_high = max(highs[-5:], default=None) if highs else None
        last5_low = min(lows[-5:], default=None) if lows else None

        # Options activity (indicative)
        options = await polygon_options_activity(symbol, window_days=20)

        return {
            "symbol": symbol,
            "ema9": ema9_v,
            "ema21": ema21_v,
            "ema50": ema50_v,
            "vwap": v,
            "premarket_high": premkt_high,
            "premarket_low": premkt_low,
            "last5_high": last5_high,
            "last5_low": last5_low,
            "daily": daily,
            "intraday": intraday,
            "options": options,
        }

    # ---------- LLM Analysis ----------
    async def llm_rank(tickers_data: List[Dict[str, Any]], news_map: Dict[str, List[Dict[str, Any]]], now_ny: datetime) -> "RankedResult":
        if openai_client is None:
            # Fallback: trivial scorer if no OpenAI client
            items = []
            for td in tickers_data:
                score = 0.0
                if td.get("ema9") and td.get("ema21") and td.get("ema9") > td.get("ema21"):
                    score += 0.5
                if td.get("vwap") and td.get("daily", {}).get("results"):
                    score += 0.2
                items.append(AnalysisItem(symbol=td["symbol"], score=round(score, 3), stance="neutral", rationale="LLM disabled; heuristic score").model_dump())
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            return RankedResult(as_of=now_ny.isoformat(), items=[AnalysisItem(**i) for i in items_sorted])

        # Compose compact JSON context per ticker to keep token cost sane
        def compact(td: Dict[str, Any]) -> Dict[str, Any]:
            d = {
                "symbol": td["symbol"],
                "ema9": td.get("ema9"),
                "ema21": td.get("ema21"),
                "ema50": td.get("ema50"),
                "vwap": td.get("vwap"),
                "premarket_high": td.get("premarket_high"),
                "premarket_low": td.get("premarket_low"),
                "last5_high": td.get("last5_high"),
                "last5_low": td.get("last5_low"),
            }
            # Add light options/volume summaries if present
            try:
                res = td.get("daily", {}).get("results", [])
                d["stock_avg_vol20"] = round(sum(b.get("v", 0) for b in res[-20:]) / max(1, len(res[-20:])), 2)
            except Exception:
                d["stock_avg_vol20"] = None
            d["news"] = [
                {"headline": n.get("headline"), "summary": n.get("summary"), "sentiment": n.get("sentiment", ""), "datetime": n.get("datetime")}
                for n in (news_map.get(td["symbol"], [])[:6])
            ]
            return d

        compact_payload = [compact(td) for td in tickers_data]

        system = (
            "You are a trading analyst. Given JSON per ticker containing indicators (EMA9/21/50, VWAP, premarket levels, last5 high/low), "
            "approximate stock average volume, and 3-6 recent news items with sentiment hints, identify stance per ticker: bullish, bearish, or neutral. "
            "Use: puts/calls (if available), open interest/volume cues (if provided), stock avg volume, EMA9/EMA21 cross/stack, and VWAP relation. "
            "Return a JSON array of {symbol, score, stance, rationale}. Score must be in [-1.0, 1.0]."
        )

        user = json.dumps(compact_payload)

        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o-mini",  # choose your deployed model
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            items = [AnalysisItem(**{
                "symbol": p.get("symbol"),
                "score": float(p.get("score", 0.0)),
                "stance": str(p.get("stance", "neutral")),
                "rationale": str(p.get("rationale", ""))[:400]
            }) for p in parsed]
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            return RankedResult(as_of=now_ny.isoformat(), items=items_sorted)
        except Exception as e:
            print(f"[ERR] LLM analysis failed; falling back. {e}")
            # Fallback to heuristic
            items = []
            for td in tickers_data:
                score = 0.0
                if td.get("ema9") and td.get("ema21") and td.get("ema9") > td.get("ema21"):
                    score += 0.5
                if td.get("vwap") and td.get("daily", {}).get("results"):
                    score += 0.2
                items.append(AnalysisItem(symbol=td["symbol"], score=round(score, 3), stance="neutral", rationale="LLM failed; heuristic").model_dump())
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            return RankedResult(as_of=now_ny.isoformat(), items=[AnalysisItem(**i) for i in items_sorted])

    def rank_to_telegram_msg(res: RankedResult) -> str:
        lines = [f"<b>LLM Ranking @ {res.as_of}</b>"]
        for i, it in enumerate(res.items, 1):
            lines.append(f"{i}. <b>{it.symbol}</b> | score: {it.score:.3f} | {it.stance}\n{it.rationale}")
        return "\n\n".join(lines)

    # ---------- Workflows ----------
    async def morning_job(tickers: List[str]) -> RankedResult:
        now_ny = datetime.now(tz=NY)
        if not tickers:
            raise ValueError("No tickers configured")

        # Fetch features + news concurrently
        async def one(symbol: str):
            feats = await compute_features(symbol, now_ny)
            # Last 7 days news
            news = await finnhub_company_news(symbol, now_ny - timedelta(days=7), now_ny)
            return symbol, feats, news

        tasks = [one(sym) for sym in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tickers_data: List[Dict[str, Any]] = []
        news_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            if isinstance(r, Exception):
                print(f"[WARN] morning item failed: {r}")
                continue
            sym, feats, news = r
            tickers_data.append(feats)
            news_map[sym] = news

        ranked = await llm_rank(tickers_data, news_map, now_ny)
        await telegram_send(rank_to_telegram_msg(ranked))
        return ranked

    async def intraday_job(tickers: List[str]) -> RankedResult:
        # Similar to morning, but emphasize intraday context and 30-min cadence
        now_ny = datetime.now(tz=NY)

        async def one(symbol: str):
            feats = await compute_features(symbol, now_ny)
            news = await finnhub_company_news(symbol, now_ny - timedelta(days=2), now_ny)
            return symbol, feats, news

        results = await asyncio.gather(*[one(t) for t in tickers], return_exceptions=True)
        tickers_data: List[Dict[str, Any]] = []
        news_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            if isinstance(r, Exception):
                print(f"[WARN] intraday item failed: {r}")
                continue
            sym, feats, news = r
            tickers_data.append(feats)
            news_map[sym] = news

        ranked = await llm_rank(tickers_data, news_map, now_ny)
        await telegram_send("<b>Intraday Refresh</b>\n\n" + rank_to_telegram_msg(ranked))
        return ranked

    # ---------- Scheduler ----------
    # NOTE: If TZ_SOURCE == "utc-fallback", these cron times are interpreted in UTC, not New York.
    scheduler = AsyncIOScheduler(timezone=str(NY))

    # 9:00 AM ET daily (weekdays)
    scheduler.add_job(lambda: asyncio.create_task(morning_job(USER_TICKERS.copy())),
                      CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone=NY))

    # Every 30 minutes between 9:30 and 16:00 ET (weekdays)
    for h in range(9, 16 + 1):
        for m in (0, 30):
            if h == 9 and m == 0:  # skip 9:00 (handled by morning job)
                continue
            if h == 9 and m < 30:
                continue
            scheduler.add_job(lambda: asyncio.create_task(intraday_job(USER_TICKERS.copy())),
                              CronTrigger(day_of_week="mon-fri", hour=h, minute=m, timezone=NY))

    @app.on_event("startup")
    async def on_startup():
        scheduler.start()

    @app.on_event("shutdown")
    async def on_shutdown():
        await client.aclose()
        scheduler.shutdown(wait=False)

    # ---------- API ----------
    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "ssl_available": True,
            "tz": str(NY),
            "tz_source": TZ_SOURCE,
            "tz_warning": TZ_WARNING,
            "tickers": USER_TICKERS,
            "now": datetime.now(tz=NY).isoformat(),
        }

    @app.get("/tz")
    async def tz_info():
        return {
            "tz": str(NY),
            "tz_source": TZ_SOURCE,
            "hint": TZ_WARNING,
        }

    @app.post("/tickers/set")
    async def set_tickers(payload: TickerList):
        global USER_TICKERS
        # Normalize and dedupe
        uniq = sorted({t.strip().upper() for t in payload.tickers if t.strip()})
        USER_TICKERS = uniq
        return {"message": "tickers updated", "tickers": USER_TICKERS}

    @app.get("/tickers")
    async def get_tickers():
        return {"tickers": USER_TICKERS}

    @app.post("/run/morning")
    async def run_morning_now():
        if not USER_TICKERS:
            raise HTTPException(400, "No tickers set")
        res = await morning_job(USER_TICKERS.copy())
        return res.model_dump()

    @app.post("/run/intraday")
    async def run_intraday_now():
        if not USER_TICKERS:
            raise HTTPException(400, "No tickers set")
        res = await intraday_job(USER_TICKERS.copy())
        return res.model_dump()

    # ---------- Helper: simplistic breakout/retest flags (optional extension) ----------
    def breakout_flags(ibars: List[Dict[str, Any]], last5_high: Optional[float], last5_low: Optional[float]) -> Tuple[Optional[bool], Optional[bool]]:
        if not ibars or last5_high is None or last5_low is None:
            return None, None
        last_close = ibars[-1].get("c")
        broke_up = last_close is not None and last5_high is not None and last_close > last5_high
        broke_dn = last_close is not None and last5_low is not None and last_close < last5_low
        return broke_up, broke_dn

    # ---------- Lightweight self-tests (no network) ----------
    def _self_tests() -> Dict[str, Any]:
        # EMA test: known sequence
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        e3 = ema(seq, 3)
        assert e3 is not None and isinstance(e3, float)
        # VWAP test: flat prices => vwap == price
        h = [10, 10, 10]
        l = [10, 10, 10]
        c = [10, 10, 10]
        v = [100, 200, 300]
        assert abs((vwap(h, l, c, v) or 0) - 10) < 1e-9
        # breakout_flags: trivial
        ibars = [{"c": 11}, {"c": 12}]
        assert breakout_flags(ibars, last5_high=10, last5_low=5) == (True, False)
        # timezone resolution test: ensure tz source flag is set and datetime is tz-aware
        now_local = datetime.now(tz=NY)
        assert getattr(now_local.tzinfo, "tzname", lambda *_: "?")(now_local) is not None
        assert TZ_SOURCE in {"zoneinfo", "utc-fallback"}
        # premarket checker test: 09:00 local is premarket, 09:45 is not (uses whatever tz NY currently represents)
        def _mk_ms(y, m, d, hh, mm):
            return int(datetime(y, m, d, hh, mm, tzinfo=NY).timestamp() * 1000)
        assert is_premarket_ms(_mk_ms(2024, 7, 1, 9, 0), NY) is True
        assert is_premarket_ms(_mk_ms(2024, 7, 1, 9, 45), NY) is False
        return {"ema3": e3, "vwap": vwap(h, l, c, v), "breakout_example": True, "tz_source": TZ_SOURCE}

    @app.get("/tests")
    async def tests():
        try:
            return {"ok": True, "results": _self_tests()}
        except AssertionError as e:
            return {"ok": False, "error": str(e)}

# ---------------- CLI self-test for fallback or quick check ----------------
if __name__ == "__main__":
    if not SSL_AVAILABLE:
        print("[SELFTEST] ssl is NOT available – FastAPI stack will not start. Hit /health for guidance.")
    else:
        if TZ_SOURCE != "zoneinfo":
            print("[SELFTEST] tzdata missing; running with UTC fallback. On Pyodide: `await pyodide.loadPackage('tzdata')`. ")
        print("[SELFTEST] You can run: uvicorn app:app --host 0.0.0.0 --port 8000")
