import os
import re
import json
import asyncio
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, Optional, Tuple, List

# =============================================================
# Robust import strategy to work in sandboxes WITHOUT ssl module
# =============================================================
try:
    import ssl  # noqa: F401
    SSL_AVAILABLE = True
except Exception:
    SSL_AVAILABLE = False

# Try to import FastAPI only if ssl is available (Starlette/AnyIO import ssl)
try:
    if not SSL_AVAILABLE:
        raise ImportError("ssl not available; defer FastAPI import")
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
except Exception:
    # Lightweight stubs so the module can be imported and unit tests can run
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
        """A tiny stand-in for FastAPI used when ssl (and thus Starlette) isn't available.
        It supports .get/.post decorators and a no-op .on_event to satisfy code that
        registers lifecycle handlers.
        """
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
        def on_event(self, event: str):  # mock lifecycle hook
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
POLYGON_API_KEY   = os.getenv("POLYGON_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")

# Yahoo provider config (supports free RSS or paid API via custom endpoint)
NEWS_PROVIDER            = os.getenv("NEWS_PROVIDER", "yahoo_rss").lower()  # 'yahoo_rss' (default) or 'yahoo_premium'
YAHOO_NEWS_URL           = os.getenv("YAHOO_NEWS_URL", "")  # full URL for paid endpoint (preferred)
YAHOO_API_BASE           = os.getenv("YAHOO_API_BASE", "")  # base URL if not using full URL
YAHOO_NEWS_ENDPOINT      = os.getenv("YAHOO_NEWS_ENDPOINT", "news")  # relative path joined to base
YAHOO_SYMBOL_PARAM       = os.getenv("YAHOO_SYMBOL_PARAM", "symbol")  # some APIs use 'ticker'
YAHOO_EXTRA_PARAMS_JSON  = os.getenv("YAHOO_EXTRA_PARAMS", "")  # JSON string of extra query params
YAHOO_API_KEY            = os.getenv("YAHOO_API_KEY", "")
YAHOO_API_KEY_HEADER     = os.getenv("YAHOO_API_KEY_HEADER", "x-api-key")  # e.g., 'X-RapidAPI-Key'
YAHOO_API_HEADERS_JSON   = os.getenv("YAHOO_API_HEADERS", "")  # JSON string of additional headers (wins over key header)

REQUIRED_KEYS = {
    "POLYGON_API_KEY": POLYGON_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}

# -------------------- HTTP client (lazy, avoids importing httpx when ssl is missing) --------------------
_http_client = None

def get_http_client():
    global _http_client
    if _http_client is not None:
        return _http_client
    if not SSL_AVAILABLE:
        # In this sandbox there is no TLS; network calls shouldn't run.
        raise RuntimeError("SSL not available; HTTP client disabled in this environment.")
    # Lazy import httpx to avoid AnyIO->ssl import at module import time in restricted envs
    import httpx  # noqa: WPS433
    HTTP_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
    transport = httpx.AsyncHTTPTransport(retries=2)
    _http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, transport=transport, headers={"Accept-Encoding": "gzip"})
    return _http_client

# -------------------- App --------------------
app = FastAPI(title="TradingView → Polygon/Yahoo Finance → OpenAI Decision API")

# -------------------- LLM quota --------------------
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

# -------------------- Alert parsing --------------------
ALERT_RE = re.compile(
    r"""(?ix)
    ^\s*
    (PUT|CALL)
    \s*:\s*
    ([A-Z.\-]+)
    \s+at\s+\$?([\d.]+)
    \s+Strike:\s+\$?([\d.]+)
    \s+Expiry:\s+(\d{2}-\d{2}-\d{4})
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
    exp_str = m.group(5)
    exp_iso = datetime.strptime(exp_str, "%m-%d-%Y").date().isoformat()
    return {"side": side, "ticker": ticker, "price": price, "strike": strike, "expiry": exp_iso, "expiry_input": exp_str}

# -------------------- Nearby search knobs --------------------
NEARBY_DAY_WINDOW = 7           # +/- days around requested expiry
STRIKE_PCT_WINDOW = 0.10        # +/- percent around requested strike (10%)
MAX_CANDIDATES    = 5           # how many closest contracts to return

# -------------------- Polygon helpers (with graceful auth handling) --------------------
class PolygonAuthzError(Exception):
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(msg)

async def polygon_get(url: str, params: Dict[str, Any] = None) -> Any:
    if not POLYGON_API_KEY:
        raise HTTPException(500, detail="Missing POLYGON_API_KEY")
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    logging.info("Retrieving data from Polygon.io: %s", url)
    cli = get_http_client()
    r = await cli.get(url, params=params)
    if r.status_code >= 400:
        try:
            j = r.json()
            if isinstance(j, dict) and str(j.get("status")).upper() == "NOT_AUTHORIZED":
                raise PolygonAuthzError(j.get("message", "NOT_AUTHORIZED"))
        except Exception:
            pass
        raise HTTPException(r.status_code, detail=f"Polygon error: {r.text}")
    return r.json()

async def get_intraday_volume_snapshot(ticker: str, warnings: List[str]) -> Dict[str, Any]:
    logging.info("Retrieving Polygon indicators/volume for %s", ticker)
    prev = await polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev")
    prev_vol = (prev.get("results") or [{}])[0].get("v", None)

    today = datetime.now(timezone.utc).date().isoformat()

    # Try minute bars first
    try:
        aggs = await polygon_get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{today}/{today}",
            params={"adjusted": "true", "sort": "asc", "limit": 1200}
        )
        today_vol = sum(b.get("v", 0) for b in (aggs.get("results") or []))
        return {"prev_volume": prev_vol, "today_volume": today_vol, "timespan": "minute"}
    except PolygonAuthzError as e:
        warnings.append(f"Minute bars not available on current plan: {e.msg}. Fell back to daily bars.")

    # Fallback: daily bars
    try:
        daggs = await polygon_get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{today}/{today}",
            params={"adjusted": "true", "sort": "asc", "limit": 1}
        )
        rows = daggs.get("results") or []
        today_vol = rows[0].get("v") if rows else None
        return {"prev_volume": prev_vol, "today_volume": today_vol, "timespan": "day"}
    except PolygonAuthzError as e:
        warnings.append(f"Daily bars not available on current plan: {e.msg}. Today volume unavailable.")
        return {"prev_volume": prev_vol, "today_volume": None, "timespan": "unavailable"}

async def get_indicators(ticker: str) -> Dict[str, Any]:
    ema9  = await polygon_get(f"https://api.polygon.io/v1/indicators/ema/{ticker}",
                              params={"timespan": "day", "window": 9, "series_type": "close", "limit": 1})
    ema21 = await polygon_get(f"https://api.polygon.io/v1/indicators/ema/{ticker}",
                              params={"timespan": "day", "window": 21, "series_type": "close", "limit": 1})
    rsi14 = await polygon_get(f"https://api.polygon.io/v1/indicators/rsi/{ticker}",
                              params={"timespan": "day", "window": 14, "series_type": "close", "limit": 1})
    adx14 = await polygon_get(f"https://api.polygon.io/v1/indicators/adx/{ticker}",
                              params={"timespan": "day", "window": 14, "limit": 1})

    def last_val(obj):
        try:
            return (obj.get("results") or [])[0].get("values", [])[0].get("value")
        except Exception:
            return None

    return {
        "ema9": last_val(ema9),
        "ema21": last_val(ema21),
        "rsi14": last_val(rsi14),
        "adx14": last_val(adx14),
    }

# -------------------- Nearby contract search --------------------

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
    logging.info("Searching for nearby option contracts for %s", ticker)
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

    def rank_key(row: Dict[str, Any]):
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

async def get_option_snapshots_many(underlying: str, option_tickers: List[str], warnings: List[str]) -> List[Dict[str, Any]]:
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
        except PolygonAuthzError as e:
            warnings.append(f"Option snapshots not available on current plan: {e.msg}.")
            return None
        except Exception:
            return None

    snaps = await asyncio.gather(*(asyncio.create_task(_one(t)) for t in option_tickers))
    return [s for s in snaps if s]

# -------------------- Yahoo Finance helpers --------------------

def _build_headers_from_values(api_key: str, api_key_header: str, api_headers_json: str) -> Dict[str, str]:
    """Pure helper for tests: build headers from pieces.
    - If api_headers_json is valid JSON, merge it first.
    - Then, if api_key is non-empty, set/override api_key_header with the key.
    """
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


def _resolve_premium_news_url() -> str:
    if YAHOO_NEWS_URL:
        return YAHOO_NEWS_URL
    if YAHOO_API_BASE:
        base = YAHOO_API_BASE.rstrip("/")
        path = "/" + YAHOO_NEWS_ENDPOINT.lstrip("/")
        return base + path
    return ""


def _build_premium_params(ticker: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {YAHOO_SYMBOL_PARAM: ticker}
    if YAHOO_EXTRA_PARAMS_JSON:
        try:
            extra = json.loads(YAHOO_EXTRA_PARAMS_JSON)
            if isinstance(extra, dict):
                params.update(extra)
        except Exception:
            pass
    return params


def _premium_configured() -> bool:
    return NEWS_PROVIDER == "yahoo_premium" and bool(_resolve_premium_news_url())


async def _yahoo_fetch_rss_xml(ticker: str) -> str:
    """Fetch Yahoo Finance RSS XML for a ticker (uses Yahoo's public RSS)."""
    logging.info("Fetching Yahoo Finance RSS for %s", ticker)
    # Yahoo uses hyphen for class suffixes like BRK-B instead of BRK.B
    yf_symbol = ticker.replace(".", "-")
    url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    params = {"s": yf_symbol, "region": "US", "lang": "en-US"}
    cli = get_http_client()
    r = await cli.get(url, params=params)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, detail=f"Yahoo Finance RSS error: {r.text}")
    return r.text


def _parse_yahoo_rss(xml_text: str) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        logging.warning("Failed to parse Yahoo RSS: %s", e)
        return []
    # RSS structure: <rss><channel><item><title>..</title><link>..</link><pubDate>..</pubDate>
    items = []
    for item in root.findall('.//item'):
        title = (item.findtext('title') or '').strip()
        link = (item.findtext('link') or '').strip()
        pub = (item.findtext('pubDate') or '').strip()
        items.append({"title": title, "link": link, "pubDate": pub})
    return items


def calc_headline_sentiment(headlines: List[str]) -> float:
    """Very lightweight lexicon-based sentiment -> 0..1 (0=neg, 0.5=neutral, 1=pos)."""
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
    for t in headlines[:20]:  # cap to last 20 for speed
        words = set(_re.findall(r"[a-z']+", (t or '').lower()))
        p = len(words & pos)
        n = len(words & neg)
        if p == 0 and n == 0:
            continue
        raw = (p - n) / float(p + n)
        mapped = (raw + 1.0) / 2.0  # -1..1 -> 0..1
        score_acc += mapped
        count += 1
    return round(score_acc / count, 3) if count else 0.5


async def get_yahoo_news_and_sentiment_rss(ticker: str) -> Dict[str, Any]:
    xml_text = await _yahoo_fetch_rss_xml(ticker)
    items = _parse_yahoo_rss(xml_text)
    headlines = [it.get('title','') for it in items]
    score = calc_headline_sentiment(headlines)
    # Match previous shape: { "news": [...], "sentiment_score": <0..1> }
    return {"news": items[:8], "sentiment_score": score}


def _flex_articles_from_json(payload: Any) -> List[Dict[str, Any]]:
    """Attempt to extract a list of article-like dicts from unknown JSON shapes."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "news", "articles", "data", "results"):
            val = payload.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
        # last resort: return any first list of dicts
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _map_article_fields(it: Dict[str, Any]) -> Dict[str, Any]:
    title = it.get("title") or it.get("headline") or it.get("name") or ""
    link = it.get("link") or it.get("url") or it.get("article_url") or ""
    pub = it.get("pubDate") or it.get("published_at") or it.get("publishedAt") or it.get("providerPublishTime") or it.get("time") or ""
    return {"title": str(title), "link": str(link), "pubDate": str(pub)}


async def get_yahoo_news_and_sentiment_premium(ticker: str) -> Dict[str, Any]:
    """Fetch news from a paid Yahoo provider endpoint.
    Configure with env:
    - NEWS_PROVIDER=yahoo_premium
    - YAHOO_NEWS_URL=full endpoint URL  (or YAHOO_API_BASE + YAHOO_NEWS_ENDPOINT)
    - YAHOO_API_KEY=... and YAHOO_API_KEY_HEADER=... (or YAHOO_API_HEADERS JSON)
    - YAHOO_SYMBOL_PARAM=symbol|ticker and optional YAHOO_EXTRA_PARAMS JSON
    """
    url = _resolve_premium_news_url()
    if not url:
        raise HTTPException(500, detail="Yahoo premium not configured: set YAHOO_NEWS_URL or YAHOO_API_BASE/YAHOO_NEWS_ENDPOINT")
    headers = _build_headers_from_values(YAHOO_API_KEY, YAHOO_API_KEY_HEADER, YAHOO_API_HEADERS_JSON)
    params = _build_premium_params(ticker)
    logging.info("Fetching Yahoo Premium news for %s via %s", ticker, url)
    cli = get_http_client()
    r = await cli.get(url, params=params, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, detail=f"Yahoo Premium error: {r.text}")
    try:
        payload = r.json()
    except Exception:
        raise HTTPException(502, detail="Yahoo Premium returned non-JSON body")
    raw_items = _flex_articles_from_json(payload)
    items = [_map_article_fields(it) for it in raw_items]
    headlines = [it.get('title','') for it in items]
    score = calc_headline_sentiment(headlines)
    return {"news": items[:8], "sentiment_score": score}


async def get_news_and_sentiment(ticker: str) -> Dict[str, Any]:
    if NEWS_PROVIDER == "yahoo_premium":
        return await get_yahoo_news_and_sentiment_premium(ticker)
    # default path
    return await get_yahoo_news_and_sentiment_rss(ticker)

# -------------------- OpenAI helper --------------------
async def openai_decide(payload: Dict[str, Any]) -> Dict[str, Any]:
    logging.info("LLM processing start for %s", payload["ticker"])
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
        f"VOL: prev_vol={payload['volume'].get('prev_volume')}, today_vol={payload['volume'].get('today_volume')} (timespan={payload['volume'].get('timespan')}).\n"
        f"OPTION (top candidate snapshot): {payload['option_snapshot']}\n"
        f"NEWS_SENTIMENT: {payload['news'].get('sentiment_score')}; recent_headlines_count={len(payload['news'].get('news', []))}.\n"
        "Return JSON with keys: decision, confidence, rationale."
    )

    # Lazy import httpx to avoid ssl import at module import time
    cli = get_http_client()
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
    resp = await cli.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
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
        "news_provider": NEWS_PROVIDER,
        "yahoo_premium_configured": _premium_configured(),
    }

@app.get("/quota")
async def quota():
    _reset_quota_if_needed()
    return {"remaining": max(LLM_DAILY_LIMIT - _llm_calls_today, 0), "limit": LLM_DAILY_LIMIT}

# -------------------- Main webhook --------------------
@app.post("/webhook/tradingview", response_class=JSONResponse)
async def tradingview_webhook(request: Request):
    content_type = getattr(request, 'headers', {}).get("content-type", "").lower()

    if "application/json" in content_type:
        body = await request.json() if hasattr(request, 'json') else {}
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

    # 2) Fetch data in parallel (Polygon + Yahoo Finance), with graceful degradation
    try:
        candidates_task = asyncio.create_task(find_option_candidates_nearby(
            parsed["ticker"], parsed["expiry"], parsed["side"], parsed["strike"]
        ))
        vol_task  = asyncio.create_task(get_intraday_volume_snapshot(parsed["ticker"], warnings))
        ind_task  = asyncio.create_task(get_indicators(parsed["ticker"]))
        news_task = asyncio.create_task(get_news_and_sentiment(parsed["ticker"]))

        option_candidates = await candidates_task
        volume     = await vol_task
        indicators = await ind_task
        news       = await news_task

        option_snapshots: List[Dict[str, Any]] = []
        if option_candidates:
            tickers = [c["ticker"] for c in option_candidates if c.get("ticker")]
            option_snapshots = await get_option_snapshots_many(parsed["ticker"], tickers, warnings)

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

    # 5) Optional Telegram + response
    cand_line = f"CANDIDATES: {', '.join([c['ticker'] for c in option_candidates[:3]])}" if option_candidates else "CANDIDATES: none"
    warn_line = f"WARNINGS: {len(warnings)} (plan limits)" if warnings else "WARNINGS: none"
    summary_lines = [
        f"ALERT ➜ {parsed['side']} {parsed['ticker']}  strike {parsed['strike']}  exp {parsed['expiry_input']} (spot ~{parsed['price']})",
        cand_line,
        warn_line,
        f"PRE-SCORE: {pre_score}",
        f"DECISION: {decision.get('decision')}  (confidence {decision.get('confidence')}%)",
        f"WHY: {decision.get('rationale')[:300]}",
    ]
    try:
        await send_telegram("\n".join(summary_lines))
    except Exception:
        # Ignore Telegram errors in restricted environments
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

def _assert(cond: bool, msg: str):  # simple assert so script never exits abruptly in CI-like sandboxes
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
    # Only meaningful when FastAPI is stubbed (no ssl)
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
    # Mixed with no lexicon hits should map to neutral
    _assert(abs(calc_headline_sentiment(mixed) - 0.5) < 1e-6, "mixed/no-keywords should be ~neutral")


def test_build_headers_from_values_merges_and_overrides():
    base = _build_headers_from_values("k1", "X-Api-Key", json.dumps({"A":"1","B":"2"}))
    _assert(base.get("A") == "1" and base.get("B") == "2", "base headers merged")
    _assert(base.get("X-Api-Key") == "k1", "api key header set/overridden")
    # overrides via JSON
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
    logging.info("uvicorn fastapi_trading_alert_service:app --host 0.0.0.0 --port 10000")
