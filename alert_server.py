# option_flow_alert/main.py
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
import openai

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required but could not be imported. Ensure the 'ssl' module is available in your Python environment.")

# === Load Env ===
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === App Setup ===
app = FastAPI()
logging.basicConfig(level=logging.INFO)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# === Predefined Tickers ===
TICKERS = ["AAPL", "TSLA", "AMD", "NVDA", "MSFT"]

# === Fetch from Polygon ===
async def fetch_unusual_activity(symbol: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v3/unusual_options_activity/stocks/{symbol}?apiKey={POLYGON_API_KEY}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch unusual activity for {symbol}: {response.text}")
            return pd.DataFrame()
        data = response.json().get("results", [])
        return pd.DataFrame(data)

# === Prompt Formatter ===
def format_prompt(data: Dict[str, pd.DataFrame]) -> str:
    prompt = """
You are an expert options trader.
Based on the following unusual options activity data, suggest 3–5 high-conviction call/put trades for intraday or swing setups.
Respond in this format:
- Ticker
- Trade type (Call/Put)
- Timeframe (Intraday/Swing)
- Strike/Expiry (if mentioned)
- Reason (1–2 lines)
- Confidence Score (0–100)
"""
    for symbol, df in data.items():
        if df.empty:
            continue
        prompt += f"\n--- {symbol} Unusual Activity ---\n"
        prompt += df.head(10).to_string(index=False) + "\n"
    return prompt.strip()

# === GPT Analysis ===
async def ask_openai(prompt: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a professional options trader."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# === Telegram Alert ===
async def send_to_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],
        "parse_mode": "Markdown"
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

# === Route ===
@app.get("/scan")
async def scan_and_alert():
    results = {}
    for ticker in TICKERS:
        unusual_df = await fetch_unusual_activity(ticker)
        results[ticker] = unusual_df

    prompt = format_prompt(results)
    summary = await ask_openai(prompt)
    await send_to_telegram(f"\ud83d\udcca *Options Trade Ideas*\n\n{summary}")
    return {"status": "sent", "tickers_scanned": TICKERS}
