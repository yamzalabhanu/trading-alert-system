# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import asyncio

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

# Validate environment variables
REQUIRED_ENV = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "POLYGON_API_KEY", "OPENAI_API_KEY"]
missing = [var for var in REQUIRED_ENV if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

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
class OptionData(BaseModel):
    strike: int
    expiry: str
    type: str

class IndicatorData(BaseModel):
    rsi: float
    adx: float
    supertrendDir: int
    squeezeOn: bool

class Alert(BaseModel):
    type: str
    symbol: str
    price: float
    option: OptionData
    indicators: IndicatorData

# === Cooldown & Logs ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

def log_signal(symbol: str, signal: str, gpt_decision: str, strike: Optional[int] = None, expiry: Optional[str] = None):
    signal_log.append({
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt": gpt_decision,
        "strike": strike,
        "expiry": expiry,
        "timestamp": datetime.now(ZoneInfo("America/New_York"))
    })

# === GPT Evaluation ===
async def call_openai_chat(prompt: str, model: str = "gpt-4") -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions", 
                headers=headers, 
                json=payload
            )
            
        if resp.status_code == 429:
            logging.warning("⚠️ OpenAI rate limit hit. Retrying with GPT-3.5")
            return await call_openai_chat(prompt, model="gpt-3.5-turbo")
            
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ GPT returned no content.")
    except Exception as e:
        logging.error(f"OpenAI GPT error: {e}")
        return "⚠️ GPT unavailable due to error."

# === Validation ===
async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    async with httpx.AsyncClient(timeout=15) as client:
        # Validate symbol
        ref_url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        ref_resp = await client.get(ref_url)
        if ref_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid symbol")

        # Check market status
        if not allow_closed:
            status_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
            status_resp = await client.get(status_url)
            status_data = status_resp.json()
            if status_resp.status_code != 200 or status_data.get("market", "").lower() != "open":
                raise HTTPException(status_code=403, detail="Market is closed")

# === Option Greeks ===
async def get_option_greeks(symbol: str, option_type: str, strike: int, expiry: str) -> Dict[str, Any]:
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {}
                
            data = resp.json().get("results", [])
            if not data:
                return {}

        # Find specific option contract
        target_option = None
        for option in data:
            details = option.get("details", {})
            if (details.get("contract_type", "").lower() == option_type.lower() and
                details.get("strike_price", 0) == strike and
                details.get("expiration_date") == expiry):
                target_option = option
                break

        if not target_option:
            return {}

        greeks = target_option.get("greeks", {})
        return {
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "iv": target_option.get("implied_volatility", {}).get("midpoint")
        }
    except Exception as e:
        logging.error(f"Failed to fetch option Greeks: {e}")
        return {}

# === Webhook Endpoint ===
@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"📥 Received alert: {alert.symbol} {alert.type.upper()} @ ${alert.price}")

    if is_in_cooldown(alert.symbol, alert.type):
        logging.info(f"⏱️ Cooldown active for {alert.symbol} - {alert.type}")
        return {"status": "cooldown"}

    try:
        # Market validation
        await validate_symbol_and_market(alert.symbol)
        
        # Get option Greeks
        greeks = await get_option_greeks(
            alert.symbol,
            alert.option.type,
            alert.option.strike,
            alert.option.expiry
        )

        # === Prompt Composition ===
        prompt = f"""
Intraday options trading alert received.

🔹 Symbol: {alert.symbol}
🔹 Price: ${alert.price}
🔹 Signal Type: {alert.type.upper()}
🔹 Suggested Option: {alert.option.type} ${alert.option.strike} expiring on {alert.option.expiry}

📊 Technical Indicators:
- RSI: {alert.indicators.rsi}
- ADX: {alert.indicators.adx}
- Supertrend Direction: {alert.indicators.supertrendDir}
- TTM Squeeze: {'ON' if alert.indicators.squeezeOn else 'OFF'}

📈 Option Greeks:
- Delta: {greeks.get('delta', 'N/A')}
- Gamma: {greeks.get('gamma', 'N/A')}
- Theta: {greeks.get('theta', 'N/A')}
- Implied Volatility: {greeks.get('iv', 'N/A')}

➡️ Evaluate this trade opportunity and provide:
1. Trade decision (Yes/No)
2. Confidence level (0-100)
3. One-sentence reasoning.
"""

        gpt_reply = await call_openai_chat(prompt)
        gpt_decision = "buy" if "yes" in gpt_reply.lower() else "skip"

        # === Telegram Notification ===
        msg = (
            f"🚨 *{alert.type.upper()}* alert for `{alert.symbol}` @ `${alert.price}`\n"
            f"💡 *Option:* {alert.option.type} ${alert.option.strike} exp {alert.option.expiry}\n"
            f"📊 *Indicators:* RSI: {alert.indicators.rsi} | ADX: {alert.indicators.adx} | "
            f"Squeeze: {'ON' if alert.indicators.squeezeOn else 'OFF'}\n"
            f"📈 *Greeks:* Δ {greeks.get('delta', 'N/A')} | Γ {greeks.get('gamma', 'N/A')} | "
            f"Θ {greeks.get('theta', 'N/A')} | IV {greeks.get('iv', 'N/A')}\n\n"
            f"🧠 *GPT Review:*\n{gpt_reply}"
        )
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "text": msg, 
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
            )

        update_cooldown(alert.symbol, alert.type)
        log_signal(alert.symbol, alert.type, gpt_decision, alert.option.strike, alert.option.expiry)

        return {"status": "ok", "gpt_review": gpt_reply}

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.exception("Webhook processing failed")
        return {"status": "error", "detail": str(e)}

# === Background Tasks ===
async def schedule_daily_summary():
    while True:
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.hour == 16 and now.minute == 15:
            await send_daily_summary()
            await asyncio.sleep(60)
        await asyncio.sleep(30)

async def send_daily_summary():
    try:
        if not signal_log:
            return
            
        now = datetime.now(ZoneInfo("America/New_York"))
        summary = f"📊 *Daily Summary Report* ({now.strftime('%Y-%m-%d')}):\n\n"
        
        # Signal statistics
        counter = Counter((log["symbol"], log["signal"]) for log in signal_log)
        gpt_counter = Counter(log["gpt"] for log in signal_log)
        
        summary += "🔝 *Top Signals:*\n"
        for (sym, sig), count in counter.most_common(5):
            summary += f"- `{sym}` ({sig.upper()}): {count} signals\n"
            
        summary += "\n🧠 *GPT Decisions:*\n"
        for decision, count in gpt_counter.most_common():
            summary += f"- {decision.title()}: {count}\n"
            
        # Send summary
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "text": summary, 
                    "parse_mode": "Markdown"
                }
            )
            
        # Clear logs
        signal_log.clear()
        
    except Exception as e:
        logging.error(f"Daily summary failed: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(schedule_daily_summary())
