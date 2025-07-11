import logging
import os
import json
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Environment
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TRADINGVIEW_TOKEN = os.getenv("TRADINGVIEW_TOKEN", "my-secret-token")

# FastAPI App
app = FastAPI()
security = HTTPBearer()

# Check for critical environment vars
for var in ["POLYGON_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY"]:
    if not os.getenv(var):
        logging.warning(f"Missing environment variable: {var}")

# Models
class TradingViewAlert(BaseModel):
    symbol: str
    price: float
    action: str
    volume: float
    time: str

    @validator("action")
    def validate_action(cls, v):
        if v.upper() not in ["CALL", "PUT"]:
            raise ValueError("Invalid action type")
        return v.upper()

# Health
@app.get("/")
async def root():
    return {"status": "running"}

# Auth dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != TRADINGVIEW_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

# Alert endpoint
@app.post("/alert")
async def receive_alert(alert: TradingViewAlert, credentials: HTTPAuthorizationCredentials = Depends(security)):
    logging.info(f"Received alert: {alert}")

    indicator_data = await fetch_market_indicators(alert.symbol)
    option_decision, decision_text = await analyze_with_llm(alert.symbol, alert.action, indicator_data)

    if option_decision:
        await send_telegram_alert(option_decision, decision_text, alert.price)
    else:
        await send_telegram_message(f"⚠️ <b>No actionable trade</b> for <b>{alert.symbol}</b>. LLM couldn't recommend a trade.")

    return {"status": "ok", "received": alert.dict()}

# Indicator fetch
async def fetch_market_indicators(symbol: str) -> str:
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        async with httpx.AsyncClient() as client:
            prev = await client.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_API_KEY}")
            prev.raise_for_status()
            prev_day = prev.json()["results"][0]

            intra = await client.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}?adjusted=true&limit=10&apiKey={POLYGON_API_KEY}")
            intraday_data = intra.json().get("results", [])

            return json.dumps({
                "prev_high": prev_day["h"],
                "prev_low": prev_day["l"],
                "close": prev_day["c"],
                "volume": prev_day["v"],
                "intraday_snap": intraday_data[:3]
            }, indent=2)

    except Exception as e:
        logging.exception("Failed to fetch market indicators")
        return ""

# LLM analysis
async def analyze_with_llm(symbol: str, direction: str, indicator_data: str) -> (Optional[dict], str):
    try:
        polygon_url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&limit=10&apiKey={POLYGON_API_KEY}"
        async with httpx.AsyncClient() as client:
            res = await client.get(polygon_url)
            res.raise_for_status()
            options = res.json().get("results", [])

        if not options:
            return None, "No options data returned."

        parsed_data = json.loads(indicator_data) if indicator_data else {}
        close_price = float(parsed_data.get("close", 0))
        filtered = [o for o in options if abs(float(o.get("strikePrice", 0)) - close_price) < 10][:5]

        prompt = f"""
You are a trading assistant.
Rules:
- Breakout + Retest = Call
- Breakdown + Retest = Put
- 9/20 EMA Breakout = Call
- 9/20 EMA Breakdown = Put
- Unusual volume = Trade in trend direction

INDICATORS:
{json.dumps(parsed_data, indent=2)}

OPTIONS:
{json.dumps(filtered, indent=2)}
"""

        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }, json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a financial trading assistant."},
                    {"role": "user", "content": prompt}
                ]
            })
            result = resp.json()
            explanation = result.get("choices", [{}])[0].get("message", {}).get("content", "LLM failed to return explanation.")

            return {
                "symbol": symbol,
                "type": direction.lower(),
                "strike": float(filtered[0].get("strikePrice", 100)),
                "expiry": filtered[0].get("expiration_date", "2024-12-20"),
                "quantity": 1
            }, explanation

    except Exception as e:
        logging.exception("LLM analysis failed")
        return None, "LLM request failed."

# Telegram messaging
async def send_telegram_alert(option: dict, explanation: str, current_price: float):
    chart_url = f"https://www.tradingview.com/chart/?symbol={option['symbol']}"
    message = (
        f"<b>📈 TRADE ALERT</b>\n"
        f"<b>Symbol:</b> {option['symbol']}\n"
        f"<b>Type:</b> {option['type']}\n"
        f"<b>Strike:</b> {option['strike']}\n"
        f"<b>Expiry:</b> {option['expiry']}\n"
        f"<b>Qty:</b> {option['quantity']}\n"
        f"<b>Current Price:</b> ${current_price}\n"
        f"<b>Chart:</b> <a href='{chart_url}'>View</a>\n\n"
        f"<b>🔍 Reasoning:</b>\n{explanation}"
    )
    await send_telegram_message(message)

async def send_telegram_message(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            })
        logging.info("Telegram message sent.")
    except Exception as e:
        logging.exception("Failed to send Telegram message")
