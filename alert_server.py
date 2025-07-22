
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

async def get_technical_patterns(symbol: str) -> str:
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/5/minute"
        params = {
            "adjusted": "true",
            "sort": "desc",
            "limit": 10,
            "apiKey": POLYGON_API_KEY
        }
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            bars = r.json().get("results", [])
            if not bars or len(bars) < 2:
                return "Insufficient bar data."

            ema9 = sum(b["c"] for b in bars[-9:]) / 9
            ema20 = sum(b["c"] for b in bars[-20:]) / 20 if len(bars) >= 20 else ema9

            pattern = []

            if ema9 > ema20:
                pattern.append("EMA9 > EMA20 (Uptrend)")
            if bars[-1]["c"] > bars[-2]["h"]:
                pattern.append("Breakout candle (Close > Prev High)")
            if bars[-1]["v"] > sum(b["v"] for b in bars) / len(bars) * 1.5:
                pattern.append("Unusual volume spike")

            return ", ".join(pattern) if pattern else "No strong technical pattern detected."
    except Exception as e:
        logging.error(f"Technical pattern error: {e}")
        return "Pattern analysis failed."

async def get_news_sentiment(symbol: str) -> str:
    try:
        url = f"https://api.polygon.io/v2/reference/news"
        params = {"ticker": symbol, "limit": 3, "apiKey": POLYGON_API_KEY}
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            news_items = r.json().get("results", [])
            summaries = [f"- {n['title']}" for n in news_items]
            return "\n".join(summaries) if summaries else "No recent headlines."
    except Exception as e:
        logging.error(f"News sentiment error: {e}")
        return "News fetch failed."

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

async def get_gpt_summary(alert: TradingViewAlert, context_text: str) -> str:
    if not OPENAI_API_KEY:
        return "No GPT summary (API key missing)."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": context_text}],
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
        patterns = await get_technical_patterns(alert.symbol)
        news = await get_news_sentiment(alert.symbol)

        context_text = f"""
        Market Context:
        {context}

        Technical Patterns:
        {patterns}

        News Headlines:
        {news}

        Alert Details:
        Symbol: {alert.symbol}
        Price: {alert.price}
        Action: {alert.action}
        Volume: {alert.volume}
        Time: {alert.time}

        Provide a brief summary and score the alert strength (0–100).
        """

        summary = await get_gpt_summary(alert, context_text)

        message = (
            f"📊 *Trading Alert*
"
            f"*{alert.symbol}* `{alert.action}` at `${alert.price}`
"
            f"Volume: {alert.volume}\n"
            f"Time: {alert.time}\n"
            f"Context: {context}\n"
            f"Patterns: {patterns}\n"
            f"News: {news}\n"
            f"*GPT:* {summary}"
        )

        alert_log.append(alert.dict())
        log_to_google_sheets(alert, context, summary)
        await send_to_telegram(message)
        return {"status": "success", "summary": summary, "context": context}
    except Exception as e:
        logging.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=400, detail=str(e))

from fastapi_utils.tasks import repeat_every

# Store recent alerts
alert_log = []

@app.on_event("startup")
@repeat_every(seconds=86400, time=datetime.utcnow().replace(hour=21, minute=0, second=0, microsecond=0))  # Run daily at 9PM UTC
async def send_daily_summary():
    if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN:
        return

    if not alert_log:
        await send_to_telegram("📊 No alerts received today.")
        return

    content = "\n".join([
        f"{a['symbol']} {a['action']} @ {a['price']} ({a['volume']})"
        for a in alert_log
    ])
    prompt = f"Summarize these trading alerts and generate key insights or patterns from today:\n{content}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            summary = response.json()["choices"][0]["message"]["content"].strip()
            await send_to_telegram(f"📈 *Daily LLM Summary*\n{summary}")
    except Exception as e:
        logging.error(f"Daily summary error: {e}")

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Setup Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("gspread_creds.json", scope)
sheet = gspread.authorize(creds).open("Trading Alerts").sheet1

# Logging alert to Google Sheets
def log_to_google_sheets(alert: TradingViewAlert, context: str, summary: str):
    try:
        sheet.append_row([
            alert.time,
            alert.symbol,
            alert.action,
            alert.price,
            alert.volume,
            context,
            summary
        ])
    except Exception as e:
        logging.error(f"Google Sheets logging failed: {e}")

# Enhanced summary includes basic metrics
@app.on_event("startup")
@repeat_every(seconds=86400, time=datetime.utcnow().replace(hour=21, minute=0, second=0, microsecond=0))  # Run daily at 9PM UTC
async def send_daily_summary():
    if not OPENAI_API_KEY or not TELEGRAM_BOT_TOKEN:
        return

    if not alert_log:
        await send_to_telegram("📊 No alerts received today.")
        return

    total = len(alert_log)
    calls = len([a for a in alert_log if a['action'].upper() == 'CALL'])
    puts = len([a for a in alert_log if a['action'].upper() == 'PUT'])

    call_win = sum(1 for a in alert_log if a['action'].upper() == 'CALL' and a['price'] < a['price'] * 1.05)
    put_win = sum(1 for a in alert_log if a['action'].upper() == 'PUT' and a['price'] > a['price'] * 0.95)

    win_rate = (call_win + put_win) / total * 100 if total > 0 else 0
    rr_ratio = "N/A (mocked)"

    content = "\n".join([
        f"{a['symbol']} {a['action']} @ {a['price']} ({a['volume']})"
        for a in alert_log
    ])
    prompt = f"Summarize these trading alerts and generate insights:\n{content}"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            summary = response.json()["choices"][0]["message"]["content"].strip()
            msg = (
                f"📈 *Daily LLM Summary*
"
                f"Total Alerts: {total}
Calls: {calls}, Puts: {puts}
"
                f"Win Rate (TP est.): {win_rate:.1f}%
"
                f"Risk/Reward: {rr_ratio}

"
                f"{summary}"
            )
            await send_to_telegram(msg)
    except Exception as e:
        logging.error(f"Daily summary error: {e}")
