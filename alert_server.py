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

from fastapi import FastAPI, HTTPException, Request
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

# === Helper: Parse Alert String ===
def parse_tradingview_message(msg: str) -> Alert:
    try:
        pattern_legacy = r"Action:\s*(BUY|SELL).*?Symbol:\s*(\w+).*?Price:\s*([\d.]+).*?Option:\s*(CALL|PUT).*?Strike:\s*(\d+).*?Expiry:\s*(\d{4}-\d{2}-\d{2})"
        pattern_simple = r"(CALL|PUT) Signal:\s*(\w+) at ([\d.]+)\s*Strike:\s*(\d+)\s*Expiry:\s*(\d{4}-\d{2}-\d{2})"

        legacy = re.search(pattern_legacy, msg)
        if legacy:
            signal, symbol, price, option_type, strike, expiry = legacy.groups()
            return Alert(
                signal=signal.lower(),
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry
            )

        simple = re.search(pattern_simple, msg)
        if simple:
            option_type, symbol, price, strike, expiry = simple.groups()
            return Alert(
                signal="buy" if option_type.upper() == "CALL" else "sell",
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry
            )

        raise ValueError("Unrecognized alert message format")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse alert: {e}")

# === Ingest Raw Message ===
@app.post("/webhook/raw")
async def handle_raw_message(request: Request):
    body = await request.body()
    text = body.decode("utf-8")
    alert = parse_tradingview_message(text)
    return await handle_alert(alert)

# === Existing /webhook stays unchanged ===
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
