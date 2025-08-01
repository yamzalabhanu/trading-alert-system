# main.py
import os
import logging
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

# === Load Config ===
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Caching ===
cache: TTLCache = TTLCache(maxsize=100, ttl=300)

# === Cooldown + Logs ===
cooldown_tracker: Dict[Tuple[str, str], datetime] = {}
COOLDOWN_WINDOW = timedelta(minutes=10)
signal_log: list = []

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
    signal_log.append({
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt": gpt_decision,
        "strike": strike,
        "expiry": expiry,
        "timestamp": datetime.now(ZoneInfo("America/New_York"))
    })

async def call_openai_chat(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 429:
                logging.warning("⚠️ OpenAI rate limit hit.")
                return "⚠️ GPT rate limit hit. No decision made."
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ GPT returned no content.")
    except Exception as e:
        logging.error(f"OpenAI GPT error: {e}")
        return "⚠️ GPT unavailable due to an internal error."

async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    async with httpx.AsyncClient() as client:
        ref_url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        ref_resp = await client.get(ref_url)
        if ref_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid symbol")

        status_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
        status_resp = await client.get(status_url)
        if not allow_closed and status_resp.status_code == 200:
            if status_resp.json().get("market", "").lower() != "open":
                raise HTTPException(status_code=403, detail="Market is closed")

async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json().get("results", [])

        nearest = None
        current_price = None

        for option in data:
            if option.get("details", {}).get("contract_type") != "call":
                continue
            if not current_price:
                current_price = option.get("underlying_asset", {}).get("last", {}).get("price", 0)
            strike = option.get("details", {}).get("strike_price", 0)
            if current_price and abs(strike - current_price) < abs((nearest or {}).get("details", {}).get("strike_price", 0) - current_price):
                nearest = option

        if not nearest:
            return {}

        greeks = nearest.get("greeks", {})
        iv = nearest.get("implied_volatility", {}).get("midpoint", None)
        return {
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "iv": iv,
        }
    except Exception as e:
        logging.warning(f"Failed to fetch option Greeks: {e}")
        return {}

@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")
    if is_in_cooldown(alert.symbol, alert.signal):
        logging.info(f"⏱️ Cooldown active for {alert.symbol} - {alert.signal}")
        return {"status": "cooldown"}

    try:
        await validate_symbol_and_market(alert.symbol, allow_closed=True)
        greeks = await get_option_greeks(alert.symbol)

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
- Brief reasoning (1 sentence)
"""
        gpt_reply = await call_openai_chat(gpt_prompt)

        gpt_decision = "unknown"
        if "yes" in gpt_reply.lower():
            gpt_decision = "buy"
        elif "no" in gpt_reply.lower():
            gpt_decision = "skip"

        option_info = ""
        if alert.strike and alert.expiry:
            option_info = f"\n🎯 Option: {'CALL' if alert.signal.lower() == 'buy' else 'PUT'} ${alert.strike} Exp: {alert.expiry}"

        tg_msg = (
            f"📈 *{alert.signal.upper()} ALERT* for `{alert.symbol}` @ `${alert.price}`"
            f"{option_info}\n\n📊 GPT Review:\n{gpt_reply}"
        )

        await httpx.AsyncClient().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg, "parse_mode": "Markdown"}
        )

        update_cooldown(alert.symbol, alert.signal)
        log_signal(alert.symbol, alert.signal, gpt_decision, alert.strike, alert.expiry)

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))
