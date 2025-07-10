### trading_alert_system/alert_server.py

import logging
import os
import json
import sys
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    import ssl
except ModuleNotFoundError:
    ssl = None
    logging.warning("SSL module not available. HTTPS requests may fail.")

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class TradingViewAlert(BaseModel):
    symbol: str
    price: float
    action: str
    volume: float
    time: str

@app.get("/")
async def root():
    return {"status": "running"}

@app.post("/alert")
async def receive_alert(alert: TradingViewAlert):
    logging.info(f"Received alert: {alert}")

    indicator_data = await fetch_market_indicators(alert.symbol)
    option_decision, decision_text = await analyze_with_llm(alert.symbol, alert.action, indicator_data)

    if option_decision:
        await send_telegram_alert(option_decision, decision_text, alert.price)
    else:
        await send_telegram_message(f"⚠️ <b>No actionable trade</b> recommended for <b>{alert.symbol}</b> based on current analysis.")

    return {"status": "ok", "received": alert.dict()}


async def fetch_market_indicators(symbol: str) -> str:
    if not POLYGON_API_KEY:
        logging.error("Missing POLYGON_API_KEY.")
        return ""

    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
            )
            response.raise_for_status()
            prev_day = response.json()["results"][0]

            intraday_response = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}?adjusted=true&limit=10&apiKey={POLYGON_API_KEY}"
            )
            intraday_data = intraday_response.json().get("results", [])

            indicators = {
                "prev_high": prev_day["h"],
                "prev_low": prev_day["l"],
                "close": prev_day["c"],
                "volume": prev_day["v"],
                "intraday_snap": intraday_data[:3]
            }
            return json.dumps(indicators, indent=2)

    except Exception as e:
        logging.exception("Error fetching market indicators")
        return ""


async def analyze_with_llm(symbol: str, direction: str, indicator_data: str) -> (Optional[dict], str):
    if not POLYGON_API_KEY:
        logging.error("Missing POLYGON_API_KEY.")
        return None, ""

    polygon_url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&limit=10&apiKey={POLYGON_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            polygon_resp = await client.get(polygon_url)
            polygon_resp.raise_for_status()
            polygon_data = polygon_resp.json()
    except Exception as e:
        logging.exception("Failed to fetch options data from Polygon")
        return None, ""

    if not polygon_data.get("results"):
        logging.warning("No options data from Polygon.")
        return None, ""

    prompt = f"""
You are a financial trading assistant. Based on the following market indicators and logic, make a decision:

- If stock breaks previous day high and premarket high and confirms second breakout after retest, suggest call.
- If stock breaks below previous day low and premarket low with second breakdown, suggest put.
- If breakout above 9 EMA & 20 EMA (second attempt), suggest call.
- If breakdown below 9 EMA & 20 EMA (second attempt), suggest put.
- If unusual intraday volume spike, suggest call/put based on overall trend.

INDICATORS:
{indicator_data}

OPTIONS:
{json.dumps(polygon_data['results'], indent=2)[:1500]}
"""

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a financial trading assistant."},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            result = resp.json()
            decision_text = result.get("choices", [{}])[0].get("message", {}).get("content", "LLM did not return any reasoning.")
            logging.info(f"LLM Decision: {decision_text}")

            return {
                "symbol": symbol,
                "type": direction.lower(),
                "strike": 100.0,
                "expiry": "2024-12-20",
                "quantity": 1
            }, decision_text

    except Exception as e:
        logging.exception("LLM analysis failed")
        return None, ""


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
        f"<b>Chart:</b> <a href='{chart_url}'>View Chart</a>\n\n"
        f"<b>🔍 Reasoning:</b>\n{explanation}"
    )
    await send_telegram_message(message)


async def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing Telegram bot token or chat ID.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
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

