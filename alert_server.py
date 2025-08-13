import os
import re
import json
import asyncio
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, Optional, Tuple, List

# =============================================================
# Runs in sandboxes without ssl: avoids network data providers
# =============================================================
try:
    import ssl  # noqa: F401
    SSL_AVAILABLE = True
except Exception:
    SSL_AVAILABLE = False

# FastAPI import (with stubs if ssl isn't available)
try:
    if not SSL_AVAILABLE:
        raise ImportError("ssl not available; defer FastAPI import")
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
except Exception:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str = ""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class JSONResponse(dict):
        pass

    class _StubRequest:  # minimal placeholder
        headers: Dict[str, str] = {}
        async def body(self) -> bytes:  # pragma: no cover
            return b""

    class _StubApp:
        """Minimal stand-in for FastAPI, with route decorators and on_event."""
        def __init__(self, *_: Any, **__: Any):
            self.routes: Dict[Tuple[str, str], Any] = {}
            self.events: Dict[str, List[Any]] = {}
        def get(self, path: str, *_, **__):
            def deco(fn):
                self.routes[("GET", path)] = fn
                return fn
            return deco
        def post(self, path: str, *_, **__):
            def deco(fn):
                self.routes[("POST", path)] = fn
                return fn
            return deco
        def on_event(self, event: str):
            def deco(fn):
                self.events.setdefault(event, []).append(fn)
                return fn
            return deco
        async def __call__(self, scope, receive, send):  # pragma: no cover
            raise RuntimeError("FastAPI unavailable: ssl module missing in this environment.")
    FastAPI = _StubApp  # type: ignore
    Request = _StubRequest  # type: ignore

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- Environment --------------------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

REQUIRED_KEYS = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

# -------------------- HTTP client (lazy; used only for OpenAI/Telegram) --------------------
_http_client = None

def get_http_client():
    global _http_client
    if _http_client is not None:
        return _http_client
    if not SSL_AVAILABLE:
        raise RuntimeError("SSL not available; HTTP client disabled in this environment.")
    import httpx  # lazy import to avoid anyio->ssl on module import
    HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
    transport = httpx.AsyncHTTPTransport(retries=2)
    _http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=transport, headers={"Accept-Encoding": "gzip"})
    return _http_client

# -------------------- App --------------------
app = FastAPI(title="TradingView → LLM Decision API (no external market/news)")

# -------------------- LLM quota --------------------
LLM_DAILY_LIMIT = int(os.getenv("LLM_DAILY_LIMIT", "20"))
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

# -------------------- Alert parsing --------------------
ALERT_RE = re.compile(
    r"""(?ix)
    ^\s*
    (PUT|CALL)
    \s*:\s*
    ([A-Z.\-]+)
    \s+at\s+\$?([\d.]+)
    \s+Strike:\s+\$?([\d.]+)
    \s+Expiry:\s+(\n?\d{2}-\d{2}-\d{4})
    \s*$
    """
)

def parse_alert(text: str) -> Dict[str, Any]:
    logging.info("Processing alert: %s", text)
    m = ALERT_RE.match(text.strip())
    if not m:
        raise ValueError("Alert format invalid. Expected: 'PUT/CALL: TICKER at $PRICE Strike: $STRIKE Expiry: MM-DD-YYYY'")
    side = m.group(1).upper()
    ticker = m.group(2).upper()
    price = float(m.group(3))
    strike = float(m.group(4))
    exp_str = m.group(5).strip()
    exp_iso = datetime.strptime(exp_str, "%m-%d-%Y").date().isoformat()
    return {"side": side, "ticker": ticker, "price": price, "strike": strike, "expiry": exp_iso, "expiry_input": exp_str}

# -------------------- Config knobs (provider-free) --------------------
NEARBY_DAY_WINDOW = 7           # kept for API shape compatibility
STRIKE_PCT_WINDOW = 0.10
MAX_CANDIDATES    = 5

# -------------------- Provider-free stubs --------------------
async def get_intraday_volume_snapshot(_ticker: str, warnings: List[str]) -> Dict[str, Any]:
    """No external provider: return neutral/unknown volume snapshot."""
    logging.info("Volume snapshot disabled (no provider)")
    return {"prev_volume": None, "today_volume": None, "timespan": "disabled"}

async def get_indicators(_ticker: str) -> Dict[str, Any]:
    """No external provider: return indicators as None."""
    logging.info("Indicators disabled (no provider)")
    return {"ema9": None, "ema21": None, "rsi14": None, "adx14": None}

# Nearby option candidates: turned off when no provider

def _iso(d: date) -> str:
    return d.isoformat()

def _clamp_strike_bounds(strike: float, pct: float) -> Tuple[float, float]:
    lo = max(0.01, strike * (1.0 - pct))
    hi = strike * (1.0 + pct)
    return (round(lo, 2), round(hi, 2))

async def find_option_candidates_nearby(*_args, **_kwargs) -> List[Dict[str, Any]]:
    logging.info("Nearby option search disabled (no provider)")
    return []

async def get_option_snapshots_many(*_args, **_kwargs) -> List[Dict[str, Any]]:
    logging.info("Option snapshots disabled (no provider)")
    return []

# -------------------- Generic helpers (no provider usage) --------------------

def _build_headers_from_values(api_key: str, api_key_header: str, api_headers_json: str) -> Dict[str, str]:
    """Build headers from user-provided pieces (pure function used in tests)."""
    headers: Dict[str, str] = {}
    if api_headers_json:
        try:
            user = json.loads(api_headers_json)
            if isinstance(user, dict):
                headers.update({str(k): str(v) for k, v in user.items()})
        except Exception:
            pass
    if api_key:
        headers[str(api_key_header)] = str(api_key)
    return headers


def _parse_yahoo_rss(xml_text: str) -> List[Dict[str, Any]]:
    """Generic RSS parser used by tests (name kept to preserve existing tests)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        logging.warning("Failed to parse RSS: %s", e)
        return []
    items = []
    for item in root.findall('.//item'):
        title = (item.findtext('title') or '').strip()
        link = (item.findtext('link') or '').strip()
        pub = (item.findtext('pubDate') or '').strip()
        items.append({"title": title, "link": link, "pubDate": pub})
    return items


def calc_headline_sentiment(headlines: List[str]) -> float:
    """Lightweight lexicon sentiment 0..1 (0=neg, 0.5=neutral, 1=pos)."""
    if not headlines:
        return 0.5
    pos = {
        "surge","beat","beats","record","strong","growth","upgrade","upgraded","bullish","rally","profit",
        "soar","soars","gain","gains","optimistic","outperform","top","tops","exceed","exceeds","positive"
    }
    neg = {
        "plunge","miss","misses","lawsuit","probe","bearish","downgrade","downgraded","loss","decline","falls",
        "fall","drop","drops","warning","cuts","cut","negative","missed","disappoint","disappoints","bankrupt"
    }
    import re as _re
    score_acc = 0
    count = 0
    for t in headlines[:20]:
        words = set(_re.findall(r"[a-z']+", (t or '').lower()))
        p = len(words & pos)
        n = len(words & neg)
        if p == 0 and n == 0:
            continue
        raw = (p - n) / float(p + n)
        mapped = (raw + 1.0) / 2.0
        score_acc += mapped
        count += 1
    return round(score_acc / count, 3) if count else 0.5

async def get_news_and_sentiment(_ticker: str) -> Dict[str, Any]:
    """No external news: return neutral sentiment and empty headlines."""
    logging.info("News fetch disabled (no provider)")
    return {"news": [], "sentiment_score": 0.5}

# -------------------- OpenAI helper (UPDATED) --------------------
async def openai_decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask OpenAI for a conservative, options-specific decision.
    Returns a normalized dict even if the model response is malformed.
    """
    logging.info("LLM processing start for %s", payload["ticker"])
    if not OPENAI_API_KEY:
        raise HTTPException(500, detail="Missing OPENAI_API_KEY")
    if not _llm_quota_ok():
        return {
            "decision": "NEUTRAL",
            "confidence": 0,
            "rationale": ["LLM quota exceeded for today; skipping model."],
            "suggested_action": "SKIP",
            "risk_notes": ["Quota exhausted"],
            "time_horizon_days": 0,
            "alts": []
        }

    # Make the prompt robust to missing data.
    sys_prompt = (
        "You are a disciplined, risk-aware options trading assistant.\n"
        "Given technicals, volume context, an option snapshot (may be None), and news sentiment,\n"
        "return a JSON decision with these keys: decision (BUY CALL|BUY PUT|SELL CALL|SELL PUT|NEUTRAL),\n"
        "confidence (0-100 integer), rationale (array of 2-6 short bullets), suggested_action (BUY|PASS|HEDGE),\n"
        "risk_notes (array), time_horizon_days (integer), and alts (array of textual alternative ideas).\n"
        "Be conservative when inputs are missing; call out missing data explicitly in rationale."
    )

    # Try to echo user alert and context
    ind = payload.get('indicators') or {}
    vol = payload.get('volume') or {}
    news = payload.get('news') or {}
    opt = payload.get('option_snapshot')

    user_prompt = (
        f"ALERT: {payload['side']} {payload['ticker']} strike {payload['strike']} exp {payload['expiry']} (spot ~{payload['price']}).\n"
        f"TECH: EMA9={ind.get('ema9')}, EMA21={ind.get('ema21')}, RSI14={ind.get('rsi14')}, ADX14={ind.get('adx14')}.\n"
        f"VOL: prev_vol={vol.get('prev_volume')}, today_vol={vol.get('today_volume')} (timespan={vol.get('timespan')}).\n"
        f"OPTION (top candidate snapshot): {opt}.\n"
        f"NEWS_SENTIMENT: {news.get('sentiment_score')} (headlines may be empty).\n"
        f"PRE_SCORE: {payload.get('pre_score')} | candidates={payload.get('option_candidates_count')}.\n"
        "Return *only* JSON with: decision, confidence, rationale, suggested_action, risk_notes, time_horizon_days, alts."
    )

    cli = get_http_client()
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 350,
        "temperature": 0.2,
    }

    # Resilient call with a single retry on 5xx
    for attempt in range(2):
        resp = await cli.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
        if 500 <= resp.status_code < 600 and attempt == 0:
            await asyncio.sleep(0.6)
            continue
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, detail=f"OpenAI error: {resp.text}")
        break

    _tick_llm_quota()
    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "{}")

    # Normalize output
    def _norm(d: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(d.get("decision", "NEUTRAL")).upper().strip()
        if decision not in {"BUY CALL", "BUY PUT", "SELL CALL", "SELL PUT", "NEUTRAL"}:
            decision = "NEUTRAL"
        try:
            confidence = int(d.get("confidence", 0))
        except Exception:
            confidence = 0
        rationale = d.get("rationale")
        if not isinstance(rationale, list):
            rationale = [str(rationale) if rationale else "No rationale provided by model."]
        suggested_action = str(d.get("suggested_action", "SKIP"))[:16].upper()
        risk_notes = d.get("risk_notes")
        if not isinstance(risk_notes, list):
            risk_notes = [str(risk_notes) if risk_notes else "No explicit risks noted."]
        try:
            th = int(d.get("time_horizon_days", 0))
        except Exception:
            th = 0
        alts = d.get("alts")
        if not isinstance(alts, list):
            alts = []
        return {
            "decision": decision,
            "confidence": max(0, min(100, confidence)),
            "rationale": rationale,
            "suggested_action": suggested_action,
            "risk_notes": risk_notes,
            "time_horizon_days": max(0, th),
            "alts": alts,
        }

    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {}
    return _norm(parsed)

# -------------------- Simple heuristic score (pre-LLM) --------------------

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

# -------------------- Telegram (optional) --------------------
async def send_telegram(msg: str) -> None:
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    cli = get_http_client()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    await cli.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True})

# -------------------- Pydantic model (optional JSON webhook) --------------------
try:
    from pydantic import BaseModel  # noqa: F811
except Exception:  # pragma: no cover
    class BaseModel:  # minimal stub for tests if pydantic missing
        pass

class AlertIn(BaseModel):
    text: str

# -------------------- Health/Quota endpoints --------------------
@app.get("/health")
async def health():
    _reset_quota_if_needed()
    return {
        "ok": True,
        "ssl_available": SSL_AVAILABLE,
        "llm_calls_today": _llm_calls_today,
        "llm_daily_limit": LLM_DAILY_LIMIT,
        "keys_loaded": {k: bool(v) for k, v in REQUIRED_KEYS.items()},
        "providers": "disabled",
    }

@app.get("/quota")
async def quota():
    _reset_quota_if_needed()
    return {"remaining": max(LLM_DAILY_LIMIT - _llm_calls_today, 0), "limit": LLM_DAILY_LIMIT}

# -------------------- Main webhook --------------------
@app.post("/webhook/tradingview", response_class=JSONResponse)
async def tradingview_webhook(request: Request):
    content_type = getattr(request, 'headers', {}).get("content-type", "").lower()

    if "application/json" in content_type and hasattr(request, 'json'):
        body = await request.json()
        text = (body.get("text") or "").strip()
    else:
        raw = await request.body() if hasattr(request, 'body') else b""
        text = (raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)).strip()

    if not text:
        raise HTTPException(400, detail="Empty alert body")

    # 1) Parse alert
    try:
        parsed = parse_alert(text)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    warnings: List[str] = []

    # 2) Provider-free data: all stubs
    option_candidates: List[Dict[str, Any]] = await find_option_candidates_nearby(parsed["ticker"], parsed["expiry"], parsed["side"], parsed["strike"])
    volume     = await get_intraday_volume_snapshot(parsed["ticker"], warnings)
    indicators = await get_indicators(parsed["ticker"])
    news       = await get_news_and_sentiment(parsed["ticker"])
    option_snapshots: List[Dict[str, Any]] = await get_option_snapshots_many(parsed["ticker"], [c.get("ticker") for c in option_candidates], warnings)

    # 3) Pre-score
    pre_score = simple_score(indicators, volume, news.get("sentiment_score", 0.5), parsed["side"])

    # 4) LLM decision (respect daily quota)
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

    # 4.1) Apply a very light policy layer (optional)
    # If pre_score and confidence disagree strongly, nudge to NEUTRAL.
    if abs(pre_score) < 3 and decision.get("confidence", 0) < 55:
        decision = {**decision, "decision": "NEUTRAL", "suggested_action": "PASS"}

    # 5) Optional Telegram + response
    cand_line = f"CANDIDATES: {', '.join([c['ticker'] for c in option_candidates[:3] if c.get('ticker')])}" if option_candidates else "CANDIDATES: none"
    warn_line = f"WARNINGS: {len(warnings)}" if warnings else "WARNINGS: none"
    summary_lines = [
        f"ALERT ➜ {parsed['side']} {parsed['ticker']}  strike {parsed['strike']}  exp {parsed['expiry_input']} (spot ~{parsed['price']})",
        cand_line,
        warn_line,
        f"PRE-SCORE: {pre_score}",
        f"DECISION: {decision.get('decision')}  (confidence {decision.get('confidence')}%) | ACTION: {decision.get('suggested_action')}",
        "WHY:" , *[f"- {r}" for r in decision.get('rationale', [])][:4],
        "RISKS:", *[f"- {r}" for r in decision.get('risk_notes', [])][:3],
    ]
    try:
        await send_telegram("\n".join(summary_lines))
    except Exception:
        pass

    return {
        "parsed": parsed,
        "pre_score": pre_score,
        "indicators": indicators,
        "volume": volume,
        "option_candidates": option_candidates,
        "option_snapshots": option_snapshots,
        "news_sentiment": news.get("sentiment_score"),
        "decision": decision,
        "warnings": warnings,
        "llm_calls_used_today": _llm_calls_today,
        "llm_daily_limit": LLM_DAILY_LIMIT
    }

# -------------------- Shutdown hook --------------------
@app.on_event("shutdown")
async def _shutdown():
    try:
        cli = get_http_client()
        await cli.aclose()
    except Exception:
        pass

# =============================================================
# Minimal unit tests (run: `python fastapi_trading_alert_service.py`)
# These DO NOT make network calls and work without ssl/fastapi.
# =============================================================

def _assert(cond: bool, msg: str):
    if not cond:
        logging.error("TEST FAIL: %s", msg)
        raise AssertionError(msg)


def test_parse_alert_valid():
    s = "CALL: AAPL at $230 Strike: $245 Expiry: 08-15-2025"
    out = parse_alert(s)
    _assert(out["side"] == "CALL", "side parse")
    _assert(out["ticker"] == "AAPL", "ticker parse")
    _assert(abs(out["price"] - 230.0) < 1e-9, "price parse")
    _assert(abs(out["strike"] - 245.0) < 1e-9, "strike parse")
    _assert(out["expiry"] == "2025-08-15", "expiry parse")


def test_parse_alert_invalid():
    bad = "BUY: AAPL 230 245 08-15-2025"
    try:
        parse_alert(bad)
    except ValueError:
        return
    raise AssertionError("Invalid alert should raise ValueError")


def test_simple_score_direction():
    tech = {"ema9": 11, "ema21": 10, "rsi14": 60, "adx14": 25}
    vol = {"prev_volume": 1000, "today_volume": 1500}
    s_call = simple_score(tech, vol, news_score=0.6, side="CALL")
    s_put  = simple_score(tech, vol, news_score=0.6, side="PUT")
    _assert(s_call > s_put, "CALL should score higher when ema9>ema21 & RSI>50")

# --- Additional tests (no network / SSL required) ---

def test_parse_alert_case_insensitive_and_spaces():
    s = "  put: nvda at $183 Strike: $180 Expiry: 09-20-2025  "
    out = parse_alert(s)
    _assert(out["side"] == "PUT", "case-insensitive PUT parse")
    _assert(out["ticker"] == "NVDA", "ticker upper-cased")
    _assert(out["expiry"] == "2025-09-20", "expiry normalized to ISO")


def test_clamp_strike_bounds():
    lo, hi = _clamp_strike_bounds(100.0, 0.10)
    _assert(abs(lo - 90.0) < 1e-9 and abs(hi - 110.0) < 1e-9, "10% strike window bounds")


def test_stub_on_event_and_routes_registration():
    if SSL_AVAILABLE:
        return
    _assert(hasattr(app, 'on_event'), "Stub app should have on_event")
    called = {"shutdown": False}

    @app.on_event("shutdown")
    async def _test_shutdown():
        called["shutdown"] = True

    _assert("shutdown" in getattr(app, 'events', {}), "Shutdown event registered in stub")


def test_http_client_no_ssl_raises():
    if SSL_AVAILABLE:
        return
    try:
        get_http_client()
    except RuntimeError:
        return
    raise AssertionError("get_http_client should raise RuntimeError when SSL is unavailable")


def test_calc_headline_sentiment():
    pos_titles = [
        "Shares surge after strong earnings beat",
        "Company upgraded; outlook positive",
    ]
    neg_titles = [
        "Stock plunges after lawsuit",
        "Revenue misses estimates and guidance cut",
    ]
    s_pos = calc_headline_sentiment(pos_titles)
    s_neg = calc_headline_sentiment(neg_titles)
    _assert(s_pos > 0.55, "positive titles should score > 0.55")
    _assert(s_neg < 0.45, "negative titles should score < 0.45")


def test_calc_headline_sentiment_neutral_empty():
    _assert(abs(calc_headline_sentiment([]) - 0.5) < 1e-9, "empty list should be neutral 0.5")
    mixed = ["Company announces new product", "Analyst says outlook uncertain"]
    _assert(abs(calc_headline_sentiment(mixed) - 0.5) < 1e-6, "mixed/no-keywords should be ~neutral")


def test_build_headers_from_values_merges_and_overrides():
    base = _build_headers_from_values("k1", "X-Api-Key", json.dumps({"A":"1","B":"2"}))
    _assert(base.get("A") == "1" and base.get("B") == "2", "base headers merged")
    _assert(base.get("X-Api-Key") == "k1", "api key header set/overridden")
    over = _build_headers_from_values("k2", "X-Api-Key", json.dumps({"X-Api-Key":"OVERRIDE","C":"3"}))
    _assert(over.get("X-Api-Key") == "k2", "explicit api key wins over JSON override")
    _assert(over.get("C") == "3", "extra header kept")


def test_parse_yahoo_rss_basic():
    sample = """
    <rss><channel>
      <item><title>Apple shares surge on record revenue</title><link>http://example.com/a</link><pubDate>Mon, 12 Aug 2025 10:00:00 GMT</pubDate></item>
      <item><title>Apple downgraded by broker</title><link>http://example.com/b</link><pubDate>Mon, 12 Aug 2025 09:00:00 GMT</pubDate></item>
    </channel></rss>
    """
    items = _parse_yahoo_rss(sample)
    _assert(len(items) == 2, "should parse two items")
    _assert(items[0]["title"].startswith("Apple shares surge"), "first title parsed")
    _assert(items[1]["link"].endswith("/b"), "second link parsed")


if __name__ == "__main__":
    logging.info("Running self-tests...")
    test_parse_alert_valid()
    test_parse_alert_invalid()
    test_simple_score_direction()
    test_parse_alert_case_insensitive_and_spaces()
    test_clamp_strike_bounds()
    test_stub_on_event_and_routes_registration()
    test_http_client_no_ssl_raises()
    test_calc_headline_sentiment()
    test_calc_headline_sentiment_neutral_empty()
    test_build_headers_from_values_merges_and_overrides()
    test_parse_yahoo_rss_basic()
    logging.info("All tests passed. If you need the HTTP server, run with uvicorn in an environment with ssl support:")
    logging.info("uvicorn fastapi_trading_alert_service_llm_updated:app --host 0.0.0.0 --port 10000")
