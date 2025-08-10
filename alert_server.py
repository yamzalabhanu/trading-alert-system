# fastapi_app.py
"""
🚀 FastAPI service with robust SSL & timezone fallbacks + **asyncio-based scheduler** (no multiprocessing), scoring/confidence, success-rate tracking, and self-tests.

This app does:
1) Fetch daily pre‑market active tickers from Polygon and refresh the list each morning.
2) Maintain a user‑provided ticker list.
3) 9:00 AM ET job: pull ~20 trading days of stock + options snapshot data, Finnhub news/sentiment, compute EMA9/EMA21 + volume, and analyze with OpenAI to rank tickers.
4) Send rankings to Telegram **with confidence scores**.
5) Every 30 minutes during market hours, fetch intraday stock volume + options snapshot, enrich with sentiment, analyze again, and alert.
6) **NEW:** Keep an in‑memory log of decisions and compute a rolling **success rate**.
7) **NEW:** Generate a **post‑market daily report** at 4:15 PM ET with win/loss stats and top movers.

✅ New in this revision:
- Kept prior fixes: SSL fallback, tzdata fallback, and swapped APScheduler for pure asyncio (no `_multiprocessing`).
- Added **feature scoring** + **confidence** computation as a deterministic backup to LLM output (and for additional signal).
- Enhanced Telegram messages to show **confidence** and **emoji**.
- Added a **daily close report** (4:15 PM ET) comparing stance vs same‑day price move to track a **success rate**; report is sent to Telegram.
- Added more **self‑tests** for scoring logic.

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
- `pip install tzdata`
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
        try:
            import tzdata  # noqa: F401
            NY = ZoneInfo("America/New_York")
            TZ_ET_OK = True
        except Exception:
            NY = dt_timezone.utc
            TZ_ET_OK = False
except Exception:
    NY = dt_timezone.utc
    TZ_ET_OK = False

# ===== Globals (shared) =====
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

premarket_tickers: Set[str] = set()
user_tickers: Set[str] = set()
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

# ===== Feature scoring (pure) ===============================================

def score_features(stock_feats: Dict[str, Any], opt_sum: Dict[str, Any], news_sent: Dict[str, Any]) -> Tuple[float, float, str]:
    """Return (score_0_100, confidence_0_100, reason). Higher is more bullish.
    Heuristic blend so you have a fast, deterministic backup to LLM.
    """
    if not stock_feats or not opt_sum:
        return 50.0, 40.0, "Insufficient features"

    ema_trend = float(stock_feats.get("ema_trend", 0.0))
    pcr = float(opt_sum.get("put_call_volume_ratio", float("inf")))
    oir = float(opt_sum.get("put_call_oi_ratio", float("inf")))

    # Sentiment: Finnhub news-sentiment may expose 'companyNewsScore' or 'sentiment'
    sent_raw = 0.0
    if isinstance(news_sent, dict):
        for k in ("companyNewsScore", "score", "sentiment", "buzz"):
            v = news_sent.get(k)
            if isinstance(v, (int, float)):
                sent_raw = float(v)
                break

    # Normalize pieces
    # EMA trend: tanh squashing for stability
    ema_component = np.tanh(ema_trend / (stock_feats.get("ema21", 1.0) or 1.0)) * 20.0  # ±20
    # PCR lower is bullish; cap extreme ratios
    pcr_capped = min(max(pcr, 0.1), 5.0)
    pcr_component = (1.0 - (pcr_capped - 0.1) / (5.0 - 0.1)) * 25.0  # 0..25
    # OIR similar intuition
    oir_capped = min(max(oir, 0.1), 5.0)
    oir_component = (1.0 - (oir_capped - 0.1) / (5.0 - 0.1)) * 15.0  # 0..15
    # sentiment (map roughly -1..+1 -> -15..+15)
    sent_component = max(min(sent_raw, 1.0), -1.0) * 15.0

    raw = 50.0 + ema_component + pcr_component + oir_component + sent_component
    score = float(max(min(raw, 100.0), 0.0))

    # Confidence: magnitude of components and consistency between ema vs options
    trend_bias = 1.0 if ema_component >= 0 else -1.0
    options_bias = 1.0 if (pcr_component + oir_component) >= 0 else -1.0
    agreement = 1.0 if trend_bias == options_bias else 0.0
    conf = 40.0 + 25.0 * agreement + 0.2 * abs(ema_component) + 0.2 * (pcr_component + oir_component)
    conf = float(max(min(conf, 100.0), 0.0))

    reason = (
        f"EMA trend {'>' if ema_trend>0 else '<'} 0, PCR={pcr:.2f}, OIR={oir:.2f}, sentiment≈{sent_raw:.2f}"
    )
    return score, conf, reason

# ===== Async scheduler helpers (pure) =======================================

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
        """Combine heuristic score/confidence with LLM output per ticker."""
        ranked = llm_out.get("ranked") if isinstance(llm_out, dict) else None
        merged: List[Dict[str, Any]] = []
        # Build deterministic map of features for scoring
        for tkr, feats in payload.items():
            sfeat = feats.get("stock_features", {})
            ofeat = feats.get("options_summary", {})
            nsent = feats.get("news_sentiment", {})
            s, c, why = score_features(sfeat, ofeat, nsent)
            merged.append({"ticker": tkr, "heuristic_score": s, "heuristic_confidence": c, "heuristic_reason": why})
        # If LLM present, enrich by ticker match
        if isinstance(ranked, list):
            llm_map = {str(it.get("ticker")).upper(): it for it in ranked if isinstance(it, dict)}
            for m in merged:
                it = llm_map.get(m["ticker"].upper(), {})
                if it:
                    m.update({
                        "stance": it.get("stance"),
                        "llm_confidence": it.get("confidence"),
                        "llm_reason": it.get("reason"),
                    })
        # Final ordering: prefer LLM if available, else heuristic score
        merged.sort(key=lambda x: (-(x.get("llm_confidence") or 0), -x["heuristic_score"]))
        return merged

    # ===== Telegram helper =====
    async def send_telegram(text: str) -> None:
        if not TELEGRAM_CHAT_ID or not TELEGRAM_BOT_TOKEN:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await http.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

    # ===== Payload builders =====
    async def build_daily_payload(tickers: List[str]) -> Dict[str, Any]:
        to = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date()
        frm = to - timedelta(days=40)
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

    # ===== Simple in‑memory log for success tracking ========================
    decisions_log: List[Dict[str, Any]] = []  # each: {ts, ticker, stance, reference_price}

    def record_decisions(now_ts: datetime, ranked_items: List[Dict[str, Any]]):
        for it in ranked_items:
            t = it.get("ticker")
            stance = (it.get("stance") or ("Bullish" if it.get("heuristic_score", 50) >= 55 else "Bearish" if it.get("heuristic_score", 50) <= 45 else "Neutral"))
            if not t or stance == "Neutral":
                continue
            decisions_log.append({
                "ts": now_ts.isoformat(),
                "ticker": t,
                "stance": stance,
                "ref_price": None,  # can be filled by fetching today's open later
            })

    async def evaluate_today_success() -> Tuple[int, int, List[Dict[str, Any]]]:
        """Compare stance vs. same‑day price change (open→close)."""
        if not decisions_log:
            return 0, 0, []
        today = (datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc)).date().isoformat()
        evals: List[Dict[str, Any]] = []
        wins = 0
        total = 0
        tickers = sorted({d["ticker"] for d in decisions_log})
        # Fetch OHLC for each ticker
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
        actives = await refresh_premarket_list(limit=25)
        universe = sorted(set(actives) | set(user_tickers))
        if not universe:
            await send_telegram("No tickers in universe for daily job.")
            return
        payload = await build_daily_payload(universe)
        llm = await openai_rank_analysis(payload)
        merged = await merge_llm_with_scores(payload, llm)
        global last_daily_ranking
        last_daily_ranking = merged
        record_decisions(datetime.now(NY) if TZ_ET_OK else datetime.now(dt_timezone.utc), merged[:15])
        warn = "\n_(Timezone fallback active: install tzdata for America/New_York scheduling.)_" if not TZ_ET_OK else ""
        lines = ["*Daily 9:00 AM – Ranked Tickers*" + warn + "\n"]
        for i, it in enumerate(merged[:15], 1):
            conf = it.get("llm_confidence") or it.get("heuristic_confidence")
            stance = it.get("stance") or ("Bullish" if it["heuristic_score"] >= 55 else "Bearish" if it["heuristic_score"] <= 45 else "Neutral")
            emoji = "🟢" if stance == "Bullish" else ("🔴" if stance == "Bearish" else "⚪️")
            reason = it.get("llm_reason") or it.get("heuristic_reason")
            lines.append(f"{i}. `{it['ticker']}` {emoji} *{stance}* — conf: {int(conf or 0)}%\n   _{reason}_")
        await send_telegram("\n".join(lines))

    async def intraday_30m_job() -> None:
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
        llm = await openai_rank_analysis(payload)
        merged = await merge_llm_with_scores(payload, llm)
        global last_intraday_ranking
        last_intraday_ranking = merged
        lines = ["*Intraday Update (every 30m)*\n"]
        for i, it in enumerate(merged[:15], 1):
            conf = it.get("llm_confidence") or it.get("heuristic_confidence")
            stance = it.get("stance") or ("Bullish" if it["heuristic_score"] >= 55 else "Bearish" if it["heuristic_score"] <= 45 else "Neutral")
            emoji = "🟢" if stance == "Bullish" else ("🔴" if stance == "Bearish" else "⚪️")
            lines.append(f"{i}. `{it['ticker']}` {emoji} *{stance}* — conf: {int(conf or 0)}%")
        await send_telegram("\n".join(lines))

    async def post_market_415pm_job() -> None:
        if not TZ_ET_OK:
            return
        wins, total, evals = await evaluate_today_success()
        rate = (wins / total * 100.0) if total else 0.0
        lines = ["*Daily Report – 4:15 PM ET*\n", f"Success rate today: *{rate:.1f}%* ({wins}/{total})\n"]
        for e in evals[:20]:
            emoji = "✅" if e["win"] else "❌"
            lines.append(f"{emoji} `{e['ticker']}` {e['stance']} — open: {e['open']:.2f}, close: {e['close']:.2f}")
        await send_telegram("\n".join(lines) if len(lines) > 2 else "Daily report: no decisions to evaluate.")
        # Reset log for the next day
        decisions_log.clear()

    # ===== Asyncio scheduler loops =====
    async def _daily_loop():
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_time_at(9, 0, tz, now)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                await daily_9am_job()
            except Exception as e:
                await send_telegram(f"Daily job failed: {e}")

    async def _intraday_loop():
        tz = NY if TZ_ET_OK else dt_timezone.utc
        while True:
            now = datetime.now(tz)
            nxt = next_half_hour_boundary(now, tz)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                await intraday_30m_job()
            except Exception as e:
                await send_telegram(f"Intraday job failed: {e}")

    async def _close_loop():
        if not TZ_ET_OK:
            return
        tz = NY
        while True:
            now = datetime.now(tz)
            nxt = next_time_at(16, 15, tz, now)
            await asyncio.sleep((nxt - now).total_seconds())
            try:
                await post_market_415pm_job()
            except Exception as e:
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
        _bg_tasks.append(asyncio.create_task(_daily_loop()))
        _bg_tasks.append(asyncio.create_task(_intraday_loop()))
        _bg_tasks.append(asyncio.create_task(_close_loop()))

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

    # Scoring logic tests (new)
    stock_feats = {"ema_trend": 1.0, "ema21": 100.0}
    opt_sum = {"put_call_volume_ratio": 0.8, "put_call_oi_ratio": 0.9}
    news_sent = {"companyNewsScore": 0.2}
    score, conf, _ = score_features(stock_feats, opt_sum, news_sent)
    assert 50 < score <= 100 and 40 <= conf <= 100, "Scoring should produce bullish bias with reasonable confidence"

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
