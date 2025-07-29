# option_flow_alert/main.py
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
import httpx

# === Load Env ===
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === App Setup ===
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Predefined Tickers ===
TICKERS = ["AAPL", "TSLA", "AMD", "NVDA", "MSFT"]

# === Fetch from Polygon (mock unusual activity using options snapshot filter) ===
async def fetch_unusual_activity(symbol: str) -> pd.DataFrame:
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json().get("results", {}).get("options", [])
            df = pd.DataFrame(data)
            if df.empty:
                return df

            # Filter: high volume and open interest (mocking 'unusual')
            df = df[df.get("volume", 0) > 500]
            df = df[df.get("open_interest", 0) > 1000]
            return df.sort_values(by=["volume"], ascending=False).head(10)
    except httpx.HTTPError as e:
        logging.warning(f"HTTP error fetching {symbol}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error fetching {symbol}: {str(e)}")
        return pd.DataFrame()

# === Prompt Formatter ===
def format_prompt(data: Dict[str, pd.DataFrame]) -> str:
    prompt = """
You are an expert options trader.
Based on the following options activity data, suggest 3–5 high-conviction call/put trades for intraday or swing setups.
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
        prompt += f"\n--- {symbol} Snapshot Options ---\n"
        prompt += df.to_string(index=False) + "\n"
    return prompt.strip()

# === GPT Analysis (mocked if openai is unavailable) ===
async def ask_openai(prompt: str) -> str:
    try:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
    except ModuleNotFoundError:
        logging.error("OpenAI module not installed. Returning mock summary.")
        return "Mock GPT response: Based on the data, consider bullish trades on high-volume, high-OI tickers."
    except Exception as e:
        logging.error(f"OpenAI request failed: {str(e)}")
        return "OpenAI API failed to return a response."

# === Telegram Alert ===
async def send_to_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],
        "parse_mode": "Markdown"
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload)
    except Exception as e:
        logging.error(f"Telegram message failed: {str(e)}")

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
