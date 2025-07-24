import os
import logging
from datetime import datetime, timedelta
from typing import List

import httpx
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

async def send_telegram(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await httpx.AsyncClient().post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

async def get_polygon_data(endpoint: str, params: dict) -> dict:
    params["apiKey"] = POLYGON_API_KEY
    async with httpx.AsyncClient() as client:
        res = await client.get(f"https://api.polygon.io{endpoint}", params=params)
        return res.json()

async def gpt_summary(prompt: str) -> str:
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }
    async with httpx.AsyncClient() as client:
        res = await client.post("https://api.openai.com/v1/chat/completions", headers=BASE_HEADERS, json=payload)
        return res.json()["choices"][0]["message"]["content"].strip()

@app.get("/scan")
async def run_all_scans():
    now = datetime.utcnow()
    today = now.strftime('%Y-%m-%d')
    results = []

    # 1. High momentum/active stocks
    momentum = await get_polygon_data("/v2/snapshot/locale/us/markets/stocks/gainers", {})
    results.append("Top Gainers: " + ", ".join([t['ticker'] for t in momentum.get("tickers", [])[:5]]))

    # 2. High intraday volume spikes > 50000
    actives = await get_polygon_data("/v2/snapshot/locale/us/markets/stocks/most_active", {})
    spikes = [t for t in actives.get("tickers", []) if t.get("volume", 0) > 50000]
    results.append("High Volume Spikes: " + ", ".join([s['ticker'] for s in spikes]))

    # 3. High options flow (calls/puts)
    high_flow = await get_polygon_data("/v3/reference/options/contracts", {
        "underlying_ticker": "AAPL",  # placeholder for looped tickers
        "limit": 20
    })
    results.append("Options Flow Example: " + ", ".join([c['ticker'] for c in high_flow.get("results", [])]))

    # 4. Sentiment analysis using placeholder DeepAI or Polygon news (simulated)
    sentiment_bullish = ["AAPL", "NVDA"]
    sentiment_bearish = ["TSLA", "AMD"]
    results.append(f"Bullish Sentiment: {', '.join(sentiment_bullish)}")
    results.append(f"Bearish Sentiment: {', '.join(sentiment_bearish)}")

    # 5. Upgrades/Downgrades
    news = await get_polygon_data("/v2/reference/news", {"limit": 20})
    upgrade_downgrade = [n for n in news.get("results", []) if "upgrade" in n['title'].lower() or "downgrade" in n['title'].lower()]
    results.append("Upgrades/Downgrades: " + ", ".join([n['title'] for n in upgrade_downgrade[:5]]))

    full_summary = "\n".join(results)
    gpt_msg = await gpt_summary(f"Analyze and summarize these trading signals:\n{full_summary}")

    await send_telegram("\n".join(results) + f"\n\n📊 AI Summary:\n{gpt_msg}")

    return {"message": "Alert sent", "details": results}
