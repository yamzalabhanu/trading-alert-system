# main.py (Enhanced Trading System with Redis Logging)
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from collections import Counter, defaultdict
import asyncio

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTP may fail in this environment.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from cachetools import TTLCache
from zoneinfo import ZoneInfo
import httpx
import redis
import json

# === Load Config ===
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# === Redis Client ===
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# === Global Async Client ===
shared_client = httpx.AsyncClient()

# === Caching ===
cache: TTLCache = TTLCache(maxsize=200, ttl=300)
gpt_cache: TTLCache = TTLCache(maxsize=500, ttl=60)

# === Cooldown + Logs ===
cooldown_tracker: Dict[Tuple[str, str], datetime] = {}
COOLDOWN_WINDOW = timedelta(minutes=10)
signal_log: list = []
outcome_log: list = []

# === Models ===
class Alert(BaseModel):
    symbol: str
    price: float
    signal: str
    strike: int | None = None
    expiry: str | None = None
    triggers: str | None = None
    indicators: Dict[str, Any] | None = None

# === Cooldown ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

def log_signal(symbol: str, signal: str, gpt_decision: str, strike: int | None = None, expiry: str | None = None):
    entry = {
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt": gpt_decision,
        "strike": strike,
        "expiry": expiry,
        "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat()
    }
    signal_log.append(entry)
    try:
        redis_client.rpush("trade_logs", json.dumps(entry))
    except Exception as e:
        logging.warning(f"Redis logging failed: {e}")

async def call_openai_chat(prompt: str, cache_key: str) -> str:
    if cache_key in gpt_cache:
        return gpt_cache[cache_key]

    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = await shared_client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if resp.status_code == 429:
            return "⚠️ GPT rate limit hit."
        resp.raise_for_status()
        reply = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No content")
        gpt_cache[cache_key] = reply
        return reply
    except Exception as e:
        logging.error(f"OpenAI GPT error: {e}")
        return "⚠️ GPT error."

async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    key = f"market_status_{symbol}"
    if key in cache:
        return
    ref_url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    status_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
    ref_resp, status_resp = await asyncio.gather(
        shared_client.get(ref_url), shared_client.get(status_url)
    )
    if ref_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    if not allow_closed and status_resp.status_code == 200:
        if status_resp.json().get("market", "").lower() != "open":
            raise HTTPException(status_code=403, detail="Market closed")
    cache[key] = True

async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    key = f"greeks_{symbol}"
    if key in cache:
        return cache[key]
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        resp = await shared_client.get(url)
        data = resp.json().get("results", [])
        nearest = None
        current_price = None
        for option in data:
            if option.get("details", {}).get("contract_type") != "call": continue
            if not current_price:
                current_price = option.get("underlying_asset", {}).get("last", {}).get("price", 0)
            strike = option.get("details", {}).get("strike_price", 0)
            if current_price and abs(strike - current_price) < abs((nearest or {}).get("details", {}).get("strike_price", 0) - current_price):
                nearest = option
        greeks = nearest.get("greeks", {}) if nearest else {}
        iv = nearest.get("implied_volatility", {}).get("midpoint") if nearest else None
        result = {"delta": greeks.get("delta"), "gamma": greeks.get("gamma"), "theta": greeks.get("theta"), "iv": iv}
        cache[key] = result
        return result
    except Exception as e:
        logging.warning(f"Greeks error: {e}")
        return {}

@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")
    if is_in_cooldown(alert.symbol, alert.signal):
        return {"status": "cooldown"}
    try:
        await validate_symbol_and_market(alert.symbol, allow_closed=True)
        greeks = await get_option_greeks(alert.symbol)

        # Pre-GPT scoring
        score = 0
        if "VWAP Reclaim" in (alert.triggers or ""): score += 2
        if "EMA Crossover" in (alert.triggers or ""): score += 2
        if "ORB Breakout" in (alert.triggers or ""): score += 2
        if alert.indicators:
            if alert.indicators.get("ADX", 0) > 25: score += 1
            if alert.indicators.get("RSI", 0) > 60 and alert.indicators.get("Supertrend") == "bullish": score += 1
        if score < 5:
            return {"status": "filtered", "reason": "low local score"}

        cache_key = f"gpt_{alert.symbol}_{alert.signal}_{alert.strike}_{alert.expiry}"
        gpt_prompt = f"""
Evaluate this intraday options signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Price: {alert.price}

Triggers:
{alert.triggers}

Indicators:
{alert.indicators}

Option Details:
Strike: {alert.strike}
Expiry: {alert.expiry}

Option Greeks:
Delta: {greeks.get('delta')}
Gamma: {greeks.get('gamma')}
Theta: {greeks.get('theta')}
IV: {greeks.get('iv')}

Respond with:
- Trade decision (Yes/No)
- Confidence score (0–100)
- Reason (1 sentence)
"""
        gpt_reply = await call_openai_chat(gpt_prompt, cache_key)

        gpt_decision = "skip"
        confidence = 0
        if "yes" in gpt_reply.lower():
            gpt_decision = "buy"
            match = re.search(r"confidence[:=]?(\s*)(\d+)", gpt_reply.lower())
            if match: confidence = int(match.group(2))
        if gpt_decision != "buy" or confidence < 80:
            return {"status": "filtered", "reason": f"decision={gpt_decision}, confidence={confidence}"}

        option_info = f"\n🎯 Option: {'CALL' if alert.signal.lower() == 'buy' else 'PUT'} ${alert.strike} Exp: {alert.expiry}" if alert.strike else ""
        tg_msg = f"📈 *{alert.signal.upper()} ALERT* for `{alert.symbol}` @ `${alert.price}`{option_info}\n\n📊 GPT Review:\n{gpt_reply}"
        await shared_client.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg, "parse_mode": "Markdown"})

        update_cooldown(alert.symbol, alert.signal)
        log_signal(alert.symbol, alert.signal, gpt_decision, alert.strike, alert.expiry)

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))
