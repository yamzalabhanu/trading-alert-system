import os
import json
import logging
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List
from pathlib import Path
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, RootModel, validator
from dotenv import load_dotenv
import pandas as pd
import numpy as np

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure connections may fail.")

try:
    from zoneinfo import ZoneInfo
    TIMEZONE = ZoneInfo("America/New_York")
except Exception:
    from datetime import timezone, timedelta
    logging.warning("ZoneInfo not available, using fallback UTC-5")
    TIMEZONE = timezone(timedelta(hours=-5))

load_dotenv()

COOLDOWN_SECONDS = 300
CONFIDENCE_THRESHOLD = 75
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)

app = FastAPI()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")

cooldowns = {}
sentiment_logs = {}

class TradingViewAlert(BaseModel):
    symbol: str
    price: float
    action: str
    volume: float
    time: str

    @validator("action")
    def validate_action(cls, v):
        if v.upper() not in ("CALL", "PUT"):
            raise ValueError("action must be CALL or PUT")
        return v.upper()

class AlertList(RootModel[List[TradingViewAlert]]):
    pass

async def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
    except Exception as e:
        logging.warning(f"Telegram message failed: {e}")

async def call_openai(system_msg, user_msg):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
            res.raise_for_status()
            return res.json().get("choices", [{}])[0].get("message", {}).get("content")
    except Exception as e:
        logging.warning(f"OpenAI request failed: {e}")
        return None

async def analyze_with_llm(symbol, direction, indicators, sentiment):
    system_prompt = (
        "You are an options trading assistant. Given stock market indicators and sentiment, decide if a trade should be taken. "
        "Respond ONLY with a JSON object in this format: "
        "{\"symbol\": \"AAPL\", \"type\": \"call\", \"strike\": 210, \"expiry\": \"2025-07-26\", \"quantity\": 1, \"confidence\": 85}"
    )
    user_data = {
        "symbol": symbol,
        "direction": direction,
        "indicators": indicators,
        "sentiment": sentiment,
    }
    response = await call_openai(system_prompt, json.dumps(user_data))
    logging.info(f"LLM response: {response}")

    if not response:
        logging.warning("No response from OpenAI")
        return None

    try:
        parsed = json.loads(response)
        if parsed.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
            log_path = LOG_DIR / f"{symbol}_trade_log.json"
            with open(log_path, "a") as f:
                f.write(json.dumps(parsed) + "\n")
            return parsed
    except Exception as e:
        logging.warning(f"LLM parsing error: {e}")
        with open(LOG_DIR / f"{symbol}_llm_raw.txt", "a") as raw_log:
            raw_log.write(f"{datetime.utcnow()}\n{response}\n\n")

    return None

@app.post("/alert")
async def alert(alert: TradingViewAlert):
    symbol = alert.symbol.upper()
    now = datetime.now()
    if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < COOLDOWN_SECONDS:
        logging.info(f"{symbol} alert skipped due to cooldown")
        return {"status": "cooldown"}

    try:
        indicators = await fetch_market_indicators(symbol)
        sentiment = await get_combined_sentiment(symbol)
        decision = await analyze_with_llm(symbol, alert.action.lower(), indicators, sentiment)

        if decision:
            await send_telegram_alert(decision, indicators)
            cooldowns[symbol] = now
            return {"status": "alert_sent", "decision": decision}
        else:
            logging.warning(f"⚠️ Trade not taken for {symbol}")
            return {"status": "rejected"}

    except Exception as e:
        error_message = f"Error handling alert for {symbol}: {e}"
        logging.error(error_message)
        await send_telegram_message(f"<b>Error:</b> {error_message}")
        return {"status": "error", "message": str(e)}

@app.post("/backtest")
async def backtest(alerts: AlertList):
    results = []
    for alert in alerts:
        try:
            indicators = await fetch_market_indicators(alert.symbol)
            sentiment = await get_combined_sentiment(alert.symbol)
            decision = await analyze_with_llm(alert.symbol, alert.action.lower(), indicators, sentiment)
            results.append({
                "symbol": alert.symbol,
                "decision": decision
            })
        except Exception as e:
            logging.error(f"Backtest error for {alert.symbol}: {e}")
            results.append({
                "symbol": alert.symbol,
                "error": str(e)
            })
    return results

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/healthcheck")
def health():
    return {
        "polygon": bool(POLYGON_API_KEY),
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "openai": bool(OPENAI_API_KEY),
        "deepai": bool(DEEPAI_API_KEY),
        "ssl_available": ssl is not None,
        "timezone": str(TIMEZONE),
    }

@app.get("/logs")
def logs():
    return [f.name for f in LOG_DIR.glob("*_trade_log.json")]

@app.get("/dashboard")
def dashboard():
    entries = []
    for f in sorted(LOG_DIR.glob("*_trade_log.json"), key=os.path.getmtime, reverse=True):
        with open(f) as file:
            lines = file.readlines()[-20:]
            entries.extend([json.loads(l) for l in lines])
        if len(entries) >= 20:
            break
    return entries[:20]
