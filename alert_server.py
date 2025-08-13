import os
import re
import json
import asyncio
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, Optional, Tuple, List

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# === Environment ===
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

# Safety checks (validated at request time; we don't fail startup)
REQUIRED_KEYS = {
    "POLYGON_API_KEY": POLYGON_API_KEY,
    "FINNHUB_API_KEY": FINNHUB_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

# === App & HTTP client ===
app = FastAPI(title="TradingView → Polygon/Finnhub → OpenAI Decision API")

HTTP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
transport = httpx.AsyncHTTPTransport(retries=2)
client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=transport, headers={"Accept-Encoding": "gzip"})

# === Simple in-memory quota for LLM calls (20/day) ===
LLM_DAILY_LIMIT = 20
_llm_calls_today = 0
_llm_roll_date = datetime.now(timezone.utc).date()

def _reset_quota_if_needed():
    global _llm_calls_today, _llm_roll_date
    today = datetime.now(timezone.utc).date()
    if today != _llm_roll_date:
        _llm_roll_date = today
        _llm_calls_today = 0

def _llm_quota_ok() -> bool:
    _reset_quota_if_needed()
    return _llm_calls_today < LLM_DAILY_LIMIT

def _tick_llm_quota():
    global _llm_calls_today
    _llm_calls_today += 1

# === Regex for alert parsing ===
# Example: "PUT/CALL: AAPL at $230 Strike: $245 Expiry: 08-15-2025"
ALERT_RE = re.compile(
    r"""(?ix)          # ignore case, verbose
    ^\s*
    (PUT|CALL)         # 1: side
    \s*:\s*
    ([A-Z.\-]+)        # 2: ticker (AAPL, BRK.B, etc)
    \s+at\s+\$?([\d.]+)   # 3: underlying price (optional use)
    \s+Strike:\s+\$?([\d.]+) # 4: strike
    \s+Expiry:\s+(\d{2}-\d{2}-\d{4}) # 5: MM-DD-YYYY
    \s*$
    """
)

def parse_alert(text: str) -> Dict[str, Any]:
    m = ALERT_RE.match(text.strip())
    if not m:
        raise ValueError("Alert format invalid. Expected: 'PUT/CALL: TICKER at $PRICE Strike: $STRIKE Expiry: MM-DD-YYYY'")
    side = m.group(1).upper()       # PUT or CALL
    ticker = m.group(2).upper()
    price = float(m.group(3))
    strike = float(m.group(4))
    exp_str = m.group(5)            # MM-DD-YYYY
    exp_iso = datetime.strptime(exp_str, "%m-%d-%Y").date().isoformat()
    return {"side": side, "ticker": ticker, "price": price, "strike": strike, "expiry": exp_iso, "expiry_input": exp_str}

# === Nearby search knobs ===
NEARBY_DAY_WINDOW = 7           # +/- days around requested expiry
STRIKE_PCT_WINDOW = 0.10        # +/- percent around requested strike (10%)
MAX_CANDIDATES    = 5           # how many closest contracts to return

# === Helpers: Polygon ===

async def polygon_get(url: str, params: Dict[str, Any] = None) -> Any:
    if not POLYGON_API_KEY:
        raise HTTPException(500, detail="Missing POLYGON_API_KEY")
    params = params or {}
    params["apiKey"] = POLYGON_API_KEY
    r = await client.get(url, params=params)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, detail=f"Polygon error: {r.text}")
    return r.json()

async def get_intraday_volume_snapshot(ticker: str) -> Dict[str, Any]:
    prev = await polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev")
    today = datetime.now(timezone.utc).date().isoformat()
    aggs = await polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}",
                             params={"adjusted": "true", "sort": "asc", "limit": 1200})
    prev_vol = (prev.get("results") or [{}])[0].get("v", None)
    today_vol = sum(b.get("v", 0) for b in aggs.get("results", []) or [])
    return {"prev_volume": prev_vol, "today_volume": today_vol}

async def get_indicators(ticker: str) -> Dict[str, Any]:
    # Daily EMA9/EMA21/RSI14/ADX14; adjust timespan/window/limit if you prefer
    ema9  = await polygon_get(f"https://api.polygon.io/v1/indicators/ema/{ticker}",
                              params={"timespan": "day", "window": 9, "series_type": "close", "limit": 1})
    ema21 = await polygon_get(f"https://api.polygon.io/v1/indicators/ema/{ticker}",
                              params={"timespan": "day", "window": 21, "series_type": "close", "limit": 1})
    rsi14 = await polygon_get(f"https://api.polygon.io/v1/indicators/rsi/{ticker}",
                              params={"timespan": "day", "window": 14, "series_type": "close", "limit": 1})
    adx14 = await polygon_get(f"https://api.polygon.io/v1/indicators/adx/{ticker}",
                              params={"timespan": "day", "window": 14, "limit": 1})

    def last_val(obj, key="results"):
        try:
            return (obj.get(key) or [])[0].get("values", [])[0].get("value")
        except Exception:
            return None

    return {
        "ema9": last_val(ema9),
        "ema21": last_val(ema21),
        "rsi14": last_val(rsi14),
        "adx14": last_val(adx14),
    }

def _iso(d: date) -> str:
    return d.isoformat()

def _clamp_strike_bounds(strike: float, pct: float) -> Tuple[float, float]:
    lo = max(0.01, strike * (1.0 - pct))
    hi = strike * (1.0 + pct)
    return (round(lo, 2), round(hi, 2))

async def find_option_candidates_nearby(
    ticker: str,
    expiry_iso: str,
    side: str,
    strike: float,
    day_window: int = NEARBY_DAY_WINDOW,
    strike_pct_window: float = STRIKE_PCT_WINDOW,
    max_candidates: int = MAX_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Find nearby option contracts constrained by:
      - contract_type == side (CALL/PUT)
      - expiration_date within +/- day_window days of requested expiry
      - strike within +/- strike_pct_window of requested strike
    Rank by (days_from_target, abs(strike_pct_diff)).
    """
    ctype = "call" if side.upper() == "CALL" else "put"

    tgt_date = datetime.fromisoformat(expiry_iso).date()
    start = tgt_date - timedelta(days=day_window)
    end   = tgt_date + timedelta(days=day_window)

    lo_strike, hi_strike = _clamp_strike_bounds(strike, strike_pct_window)

    data = await polygon_get(
        "https://api.polygon.io/v3/reference/options/contracts",
        params={
            "underlying_ticker": ticker,
            "contract_type": ctype,
            "expiration_date.gte": _iso(start),
            "expiration_date.lte": _iso(end),
            "strike_price.gte": f"{lo_strike:.2f}",
            "strike_price.lte": f"{hi_strike:.2f}",
            "order": "asc",
            "sort": "expiration_date",
            "limit": 1000,
        }
    )
    results = data.get("results") or []
    if not results:
        return []

    def rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        # smaller is better
        try:
            exp = datetime.fromisoformat(row["expiration_date"]).date()
        except Exception:
            exp = tgt_date
        days_diff = abs((exp - tgt_date).days)
        k_strike = float(row.get("strike_price", 0.0)) or 0.0
        pct_diff = abs((k_strike - strike) / strike) if strike else 1.0
        return (days_diff, pct_diff, k_strike)

    ranked = sorted(results, key=rank_key)
    return ranked[:max_candidates]

async def get_option_snapshots_many(underlying: str, option_tickers: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch snapshots for multiple option tickers, one-by-one (parallelized).
    Uses /v3/snapshot/options/{underlying} filtered by ticker.
    """
    async def _one(tkr: str):
        try:
            snap = await polygon_get(
                f"https://api.polygon.io/v3/snapshot/options/{underlying}",
                params={"limit": 250, "order": "asc", "sort": "ticker", "ticker": tkr}
            )
            rows = snap.get("results") or []
            for r in rows:
                if r.get("ticker") == tkr:
                    return r
            return rows[0] if rows else None
        except Exception:
            return None

    tasks = [asyncio.create_task(_one(t)) for t in option_tickers]
    snaps = await asyncio.gather(*tasks)
    return [s for s in snaps if s]

# === Helpers: Finnhub ===

async def finnhub_get(url: str, params: Dict[str, Any]) -> Any:
    if not FINNHUB_API_KEY:
        raise HTTPException(500, detail="Missing FINNHUB_API_KEY")
    params = dict(params or {})
    params["token"] = FINNHUB_API_KEY
    r = await client.get(url, params=params)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, detail=f"Finnhub error: {r.text}")
    return r.json()

async def get_news_and_sentiment(ticker: str) -> Dict[str, Any]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=3)
    news = await finnhub_get("https://finnhub.io/api/v1/company-news", {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()})
    sent = await finnhub_get("https://finnhub.io/api/v1/news-sentiment", {"symbol": ticker})
    overall = sent.get("sentiment", {}) if isinstance(sent, dict) else {}
    score = overall.get("score", 0) if isinstance(overall, dict) else 0
    return {
        "news": news[:8] if isinstance(news, list) else [],
        "sentiment_score": score
    }

# === Helpers: OpenAI ===

async def openai_decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls OpenAI *only* if we are under the daily quota.
    Returns a decision dict: { "decision": "BUY/SELL/NEUTRAL", "confidence": 0-100, "rationale": "..." }
    """
    if not OPENAI_API_KEY:
        raise HTTPException(500, detail="Missing OPENAI_API_KEY")
    if not _llm_quota_ok():
        return {"decision": "NEUTRAL", "confidence": 0, "rationale": "LLM quota exceeded for today; skipping model."}

    sys_prompt = (
        "You are a disciplined trading assistant for options. "
        "Given technicals, volume context, option snapshot, and news sentiment, output a conservative decision:\n"
        "- 'BUY CALL', 'BUY PUT', 'SELL CALL', 'SELL PUT', or 'NEUTRAL'\n"
        "- A 0-100 confidence score\n"
        "- 2-4 bullet reasons that reference specific inputs. Avoid overconfidence."
    )

    user_prompt = (
        f"ALERT: {payload['side']} {payload['ticker']} strike {payload['strike']} exp {payload['expiry']} (spot ~{payload['price']}).\n"
        f"TECH: EMA9={payload['indicators'].get('ema9')}, EMA21={payload['indicators'].get('ema21')}, "
        f"RSI14={payload['indicators'].get('rsi14')}, ADX14={payload['indicators'].get('adx14')}.\n"
        f"VOL: prev_vol={payload['volume'].get('prev_volume')}, today_vol={payload['volume'].get('today_volume')}.\n"
        f"OPTION (top candidate snapshot): {payload['option_snapshot']}\n"
        f"NEWS_SENTIMENT: {payload['news'].get('sentiment_score')}; recent_headlines_count={len(payload['news'].get('news', []))}.\n"
        "Return JSON with keys: decision, confidence, rationale."
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "temperature": 0.2,
    }
    resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, detail=f"OpenAI error: {resp.text}")
    _tick_llm_quota()
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        decision = str(parsed.get("decision", "NEUTRAL")).upper().strip()
        confidence = int(parsed.get("confidence", 0))
        rationale = parsed.get("rationale", "")
        return {"decision": decision, "confidence": confidence, "rationale": rationale}
    except Exception:
        return {"decision": "NEUTRAL", "confidence": 0, "rationale": "Model response parse error"}

# === Simple heuristic score (pre-LLM) ===
def simple_score(tech: Dict[str, Any], vol: Dict[str, Any], news_score: float, side: str) -> float:
    score = 0.0
    ema9, ema21 = tech.get("ema9"), tech.get("ema21")
    rsi, adx = tech.get("rsi14"), tech.get("adx14")
    if ema9 is not None and ema21 is not None:
        trend_up = ema9 > ema21
        score += 10 if trend_up else -10
        score += (5 if (side == "CALL" and trend_up) else (-5 if side == "CALL" else (5 if not trend_up else -5)))
    if rsi is not None:
        score += (rsi - 50) / 2 if side == "CALL" else (50 - rsi) / 2
    if adx is not None:
        score += min(max((adx - 20), 0), 10)
    prev_v, today_v = vol.get("prev_volume") or 0, vol.get("today_volume") or 0
    if prev_v and today_v:
        rel = today_v / prev_v
        score += (rel - 1.0) * 10
    score += (news_score - 0.5) * 20
    return round(score, 2)

# === Telegram (optional) ===
async def send_telegram(msg: str) -> None:
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True})

# === Pydantic model (optional JSON webhook) ===
class AlertIn(BaseModel):
    text: str

# === Routes ===

@app.get("/health")
async def health():
    _reset_quota_if_needed()
    return {
        "ok": True,
        "llm_calls_today": _llm_calls_today,
        "llm_daily_limit": LLM_DAILY_LIMIT,
        "keys_loaded": {k: bool(v) for k, v in REQUIRED_KEYS.items()},
    }

@app.get("/quota")
async def quota():
    _reset_quota_if_needed()
    return {"remaining": max(LLM_DAILY_LIMIT - _llm_calls_today, 0), "limit": LLM_DAILY_LIMIT}

@app.post("/webhook/tradingview", response_class=JSONResponse)
async def tradingview_webhook(request: Request):
    """
    Accepts plain text in the TradingView alert body OR JSON { "text": "<alert>" }.
    """
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        body = await request.json()
        text = (body.get("text") or "").strip()
    else:
        text = (await request.body()).decode("utf-8").strip()

    if not text:
        raise HTTPException(400, detail="Empty alert body")

    # 1) Parse
    try:
        parsed = parse_alert(text)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # 2) Fetch data (Polygon + Finnhub) in parallel
    try:
        candidates_task = asyncio.create_task(find_option_candidates_nearby(
            parsed["ticker"], parsed["expiry"], parsed["side"], parsed["strike"]
        ))
        vol_task  = asyncio.create_task(get_intraday_volume_snapshot(parsed["ticker"]))
        ind_task  = asyncio.create_task(get_indicators(parsed["ticker"]))
        news_task = asyncio.create_task(get_news_and_sentiment(parsed["ticker"]))

        option_candidates = await candidates_task
        volume     = await vol_task
        indicators = await ind_task
        news       = await news_task

        option_snapshots: List[Dict[str, Any]] = []
        if option_candidates:
            tickers = [c["ticker"] for c in option_candidates if c.get("ticker")]
            option_snapshots = await get_option_snapshots_many(parsed["ticker"], tickers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, detail=f"Upstream data fetch failed: {e}")

    # 3) Pre-score
    pre_score = simple_score(indicators, volume, news.get("sentiment_score", 0), parsed["side"])

    # 4) LLM decision (respect daily quota) – feed the best snapshot if available
    payload = {
        **parsed,
        "indicators": indicators,
        "volume": volume,
        "news": news,
        "option_snapshot": (option_snapshots[0] if option_snapshots else None),
        "pre_score": pre_score,
        "option_candidates_count": len(option_candidates),
    }
    decision = await openai_decide(payload)

    # 5) Build final message + (optional) Telegram
    cand_line = f"CANDIDATES: {', '.join([c['ticker'] for c in option_candidates[:3]])}" if option_candidates else "CANDIDATES: none"
    summary_lines = [
        f"ALERT ➜ {parsed['side']} {parsed['ticker']}  strike {parsed['strike']}  exp {parsed['expiry_input']} (spot ~{parsed['price']})",
        cand_line,
        f"PRE-SCORE: {pre_score}",
        f"DECISION: {decision.get('decision')}  (confidence {decision.get('confidence')}%)",
        f"WHY: {decision.get('rationale')[:300]}",
    ]
    msg = "\n".join(summary_lines)
    await send_telegram(msg)

    # 6) Response
    return {
        "parsed": parsed,
        "pre_score": pre_score,
        "indicators": indicators,
        "volume": volume,
        "option_candidates": option_candidates,
        "option_snapshots": option_snapshots,
        "news_sentiment": news.get("sentiment_score"),
        "decision": decision,
        "llm_calls_used_today": _llm_calls_today,
        "llm_daily_limit": LLM_DAILY_LIMIT
    }

# Graceful shutdown: close the shared client
@app.on_event("shutdown")
async def _shutdown():
    await client.aclose()
