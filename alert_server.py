
import os
import logging
from datetime import datetime, timedelta
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

cooldowns: Dict[str, datetime] = {}

class TradingViewAlert(BaseModel):
    symbol: str
    price: float
    action: str
    volume: int
    time: str

async def send_to_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

async def get_polygon_context(symbol: str) -> str:
    if not POLYGON_API_KEY:
        return "Polygon data unavailable."

    try:
        async with httpx.AsyncClient() as client:
            prev = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
                params={"adjusted": "true", "apiKey": POLYGON_API_KEY}
            )
            prev_data = prev.json()
            close = prev_data.get("results", [{}])[0].get("c", "N/A")

            snap = await client.get(
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
                params={"apiKey": POLYGON_API_KEY}
            )
            snap_data = snap.json()
            last_price = snap_data.get("ticker", {}).get("lastTrade", {}).get("p", "N/A")
            day_change = snap_data.get("ticker", {}).get("todaysChangePerc", "N/A")

            return f"Prev Close: {close}, Last: {last_price}, %Chg Today: {day_change}"
    except Exception as e:
        logging.error(f"Polygon error: {e}")
        return "Polygon fetch error."

async def get_gpt_summary(alert: TradingViewAlert, context: str) -> str:
    if not OPENAI_API_KEY:
        return "No GPT summary (API key missing)."

    prompt = (
        f"Summarize this trading alert and assess its value:\n"
        f"Symbol: {alert.symbol}\n"
        f"Price: {alert.price}\n"
        f"Action: {alert.action}\n"
        f"Volume: {alert.volume}\n"
        f"Time: {alert.time}\n"
        f"Context: {context}"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
            return summary
    except Exception as e:
        logging.error(f"GPT summary error: {e}")
        return "GPT summary error."

@app.post("/webhook/alerts")
async def receive_alert(alert: TradingViewAlert):
    try:
        alert_dt = datetime.strptime(alert.time, "%Y-%m-%d %H:%M")
        symbol_key = f"{alert.symbol}_{alert.action}"

        if symbol_key in cooldowns and datetime.utcnow() - cooldowns[symbol_key] < timedelta(minutes=5):
            msg = f"⚠️ Cooldown active for {alert.symbol} {alert.action}. Alert skipped."
            logging.info(msg)
            await send_to_telegram(msg)
            return {"status": "cooldown", "message": msg}

        cooldowns[symbol_key] = datetime.utcnow()
        context = await get_polygon_context(alert.symbol)
        summary = await get_gpt_summary(alert, context)

        message = f"""📊 *Trading Alert*
*{alert.symbol}* `{alert.action}` at `${alert.price}`
Volume: {alert.volume}
Time: {alert.time}
Context: {context}
*GPT:* {summary}"""

        await send_to_telegram(message)
        return {"status": "success", "summary": summary, "context": context}
    except Exception as e:
        logging.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=400, detail=str(e))
