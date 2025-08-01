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

# === Cooldown Helpers ===
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

# === Symbol and Market Validation ===
async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    key = f"{symbol}_valid"
    if key in cache:
        return
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    market_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
    try:
        async with shared_client as client:
            ref_resp, market_resp = await asyncio.gather(
                client.get(url),
                client.get(market_url)
            )
        if ref_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Symbol not found on Polygon")
        if not allow_closed:
            market = market_resp.json()
            if market.get("market", {}).get("isOpen") is False:
                raise HTTPException(status_code=400, detail="Market is closed")

        cache[key] = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

# === Option Greeks from Polygon Snapshot ===
async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        resp = await shared_client.get(url)
        data = resp.json()
        options = data.get("results", {}).get("options", [])
        if not options:
            return {}

        # Filter to nearest expiry >= today
        today = datetime.utcnow().date()
        valid = [o for o in options if "details" in o and o["details"].get("expiration_date")]
        valid = [o for o in valid if datetime.strptime(o["details"]["expiration_date"], "%Y-%m-%d").date() >= today]

        # Find nearest ATM call
        underlying_price = data.get("results", {}).get("underlying_asset", {}).get("last", {}).get("price", 0)
        valid = sorted(valid, key=lambda o: abs(o["details"]["strike_price"] - underlying_price))

        for opt in valid:
            greeks = opt.get("greeks", {})
            return {
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "iv": greeks.get("iv")
            }

        return {}
    except Exception as e:
        logging.warning(f"Failed to fetch option greeks: {e}")
        return {}

# === Helper: Parse Alert String ===
def parse_tradingview_message(msg: str) -> Alert:
    try:
        pattern_legacy = r"Action:\s*(BUY|SELL).*?Symbol:\s*(\w+).*?Price:\s*([\d.]+).*?Option:\s*(CALL|PUT).*?Strike:\s*(\d+).*?Expiry:\s*(\d{4}-\d{2}-\d{2})"
        pattern_simple = r"(CALL|PUT) Signal:\s*(\w+) at ([\d.]+)\s*Strike:\s*(\d+)\s*Expiry:\s*(\d{4}-\d{2}-\d{2})"
        pattern_unix_expiry = r"(CALL|PUT) Signal:\s*(\w+) at ([\d.]+)\s*Strike:\s*(\d+)\s*Expiry:\s*(\d{10,})"

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

        simple = re.search(pattern_simple, msg.replace("\n", " "))
        if simple:
            option_type, symbol, price, strike, expiry = simple.groups()
            return Alert(
                signal="buy" if option_type.upper() == "CALL" else "sell",
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry
            )

        unix_expiry = re.search(pattern_unix_expiry, msg.replace("\n", " "))
        if unix_expiry:
            option_type, symbol, price, strike, expiry_unix = unix_expiry.groups()
            expiry_dt = datetime.utcfromtimestamp(int(expiry_unix) / 1000).strftime("%Y-%m-%d")
            return Alert(
                signal="buy" if option_type.upper() == "CALL" else "sell",
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry_dt
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

# === Webhook for JSON Alert Input ===
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
