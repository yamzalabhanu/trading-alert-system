import logging
import os
import json
from datetime import datetime, time
from typing import Optional
from pathlib import Path
from collections import defaultdict
import time as time_module
import html
import asyncio

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
DEEPAI_API_KEY = os.getenv("c8d42725-5ed7-44f2-8c82-29c4355a0430")

# FastAPI App
app = FastAPI()

# Check for critical environment vars
for var in ["POLYGON_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY"]:
    if not os.getenv(var):
        logging.warning(f"Missing environment variable: {var}")

# Cooldown tracker
last_alert_times = defaultdict(lambda: 0)
ALERT_COOLDOWN_MINUTES = 5
CONFIDENCE_THRESHOLD = 75

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

@app.get("/")
async def root():
    return {"status": "running"}

@app.get("/logs")
async def get_logs():
    log_path = "logs"
    files = os.listdir(log_path) if os.path.exists(log_path) else []
    return {"log_files": files}

@app.get("/dashboard")
async def dashboard():
    trades = []
    log_path = "logs"
    if os.path.exists(log_path):
        for fname in os.listdir(log_path):
            if fname.endswith("_trade_log.json"):
                with open(os.path.join(log_path, fname)) as f:
                    for line in f:
                        try:
                            trades.append(json.loads(line.strip()))
                        except:
                            continue
    return {"trade_summary": trades[-20:]}

@app.post("/alert")
async def receive_alert(alert: TradingViewAlert):
    logging.info(f"Received alert: {alert}")

    cooldown_key = f"{alert.symbol}_{alert.action}"
    now = time_module.time()
    if now - last_alert_times[cooldown_key] < ALERT_COOLDOWN_MINUTES * 60:
        logging.info(f"Skipping duplicate alert for {cooldown_key} due to cooldown")
        return {"status": "cooldown", "symbol": alert.symbol}

    last_alert_times[cooldown_key] = now

    indicator_data = await fetch_market_indicators(alert.symbol)
    sentiment = await get_polygon_news_sentiment(alert.symbol)

    option_decision, decision_text = await analyze_with_llm(alert.symbol, alert.action, indicator_data, sentiment)

    if option_decision and option_decision.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
        await send_telegram_alert(option_decision, decision_text, alert.price)
        return {"status": "sent", "symbol": alert.symbol, "confidence": option_decision.get("confidence")}
    else:
        await send_telegram_message(f"⚠️ <b>No high-confidence trade</b> for <b>{alert.symbol}</b>.<br>Reason: {html.escape(decision_text)}")
        return {"status": "skipped", "symbol": alert.symbol, "reason": decision_text}

# === Sentiment, Indicators, Market Fetchers ===
async def retry_request(session, method, url, **kwargs):
    for attempt in range(3):
        try:
            return await session.request(method, url, **kwargs)
        except httpx.RequestError as e:
            logging.warning(f"[{attempt+1}/3] Retry failed: {e}")
            await asyncio.sleep(2 ** attempt)
    raise Exception("Failed after retries")

async def get_polygon_news_sentiment(symbol: str) -> str:
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=5&apiKey={POLYGON_API_KEY}"
        async with httpx.AsyncClient() as client:
            res = await retry_request(client, "GET", url)
            if res.status_code == 200:
                data = res.json()
                headlines = [item.get("title", "") for item in data.get("results", [])]
                sentiment_scores = []
                for title in headlines:
                    senti = await client.post("https://api.deepai.org/api/sentiment-analysis",
                                              headers={"api-key": DEEPAI_API_KEY},
                                              data={"text": title})
                    result = senti.json()
                    sentiment_scores.extend(result.get("output", []))
                return ", ".join(sentiment_scores)
        return "neutral"
    except Exception as e:
        logging.warning("Polygon news sentiment fetch failed")
        return "neutral"

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
                intraday_data = []
            else:
                intra = await client.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}?adjusted=true&limit=10&apiKey={POLYGON_API_KEY}")
                if intra.status_code == 403:
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

# === LLM & Messaging ===
async def analyze_with_llm(symbol: str, direction: str, indicator_data: str, sentiment: str):
    try:
        prompt = f"""
You are an options trading assistant.
Make a decision to buy/sell based on market indicators, sentiment, and option data.
Include if sweep detected, best strike/expiry based on IV & OI.
Output must include: decision, reasoning, confidence score.
Sentiment: {sentiment}
Indicators:
{indicator_data}
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
            confidence = 80 if "high" in explanation.lower() else 65 if "medium" in explanation.lower() else 50
            return {
                "symbol": symbol,
                "type": direction.lower(),
                "strike": 100,
                "expiry": "2024-12-20",
                "quantity": 1,
                "confidence": confidence,
                "sentiment": sentiment
            }, explanation
    except Exception as e:
        logging.exception("LLM analysis failed")
        return None, "LLM request failed."

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
        f"<b>Confidence:</b> {option.get('confidence', 'N/A')}\n"
        f"<b>Sentiment:</b> {option.get('sentiment', 'N/A')}\n"
        f"<b>Chart:</b> <a href='{chart_url}'>View</a>\n\n"
        f"<b>🔍 Reasoning:</b>\n{html.escape(explanation)}"
    )
    log_entry = {
        "symbol": option['symbol'],
        "timestamp": datetime.now().isoformat(),
        "price": current_price,
        "decision": explanation,
        "confidence": option.get("confidence", "N/A"),
        "sentiment": option.get("sentiment", "N/A")
    }
    Path("logs").mkdir(exist_ok=True)
    with open(f"logs/{option['symbol']}_trade_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
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
