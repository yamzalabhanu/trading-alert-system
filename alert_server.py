import logging
import os
import json
from datetime import datetime, time
from typing import Optional
from pathlib import Path

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module not available. Secure HTTP connections may fail.")

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, validator
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

# Load environment variables from .env file
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Environment
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI App
app = FastAPI()

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

# Alert endpoint
@app.post("/alert")
async def receive_alert(alert: TradingViewAlert):
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
        today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        now = datetime.now(ZoneInfo("America/New_York")).time()
        market_open = time(9, 30)
        market_close = time(16, 0)

        async with httpx.AsyncClient() as client:
            prev = await client.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_API_KEY}")
            prev.raise_for_status()
            prev_day = prev.json()["results"][0]

            if not (market_open <= now <= market_close):
                logging.warning("Market is closed — skipping intraday fetch.")
                intraday_data = []
            else:
                intra = await client.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}?adjusted=true&limit=10&apiKey={POLYGON_API_KEY}")
                if intra.status_code == 403:
                    logging.warning("403: Skipping intraday data (requires premium Polygon access)")
                    intraday_data = []
                else:
                    intra.raise_for_status()
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
        polygon_url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&limit=50&apiKey={POLYGON_API_KEY}"
        async with httpx.AsyncClient() as client:
            res = await client.get(polygon_url)
            res.raise_for_status()
            options = res.json().get("results", [])

        if not options:
            return None, "No options data returned."

        parsed_data = json.loads(indicator_data) if indicator_data else {}
        close_price = float(parsed_data.get("close", 0))
        threshold = max(10, close_price * 0.10)
        filtered = [o for o in options if abs(float(o.get("strike_price", 0)) - close_price) <= threshold][:5]

        if not filtered:
            logging.warning(f"No filtered options found for {symbol}. Close={close_price}, Options={options}")
            filtered = options[:5]

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

        # Save for debug
        Path("logs").mkdir(exist_ok=True)
        with open(f"logs/{symbol}_prompt.json", "w") as f:
            json.dump({"prompt": prompt}, f, indent=2)

        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }, json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a financial trading assistant."},
                    {"role": "user", "content": prompt}
                ]
            })

            if resp.status_code != 200:
                logging.error(f"OpenAI error {resp.status_code}: {resp.text}")
                return None, f"LLM request failed with status {resp.status_code}"

            result = resp.json()
            with open(f"logs/{symbol}_response.json", "w") as f:
                json.dump(result, f, indent=2)

            explanation = result.get("choices", [{}])[0].get("message", {}).get("content", "LLM failed to return explanation.")

            return {
                "symbol": symbol,
                "type": direction.lower(),
                "strike": float(filtered[0].get("strike_price", 100)),
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
