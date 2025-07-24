import os
import httpx
import asyncio
import logging
from datetime import datetime
from typing import List

from fastapi import FastAPI
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
l
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === App & Logging ===
app = FastAPI()
logging.basicConfig(level=logging.INFO)
BASE_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# === Telegram ===
async def send_telegram(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"})
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

# === Polygon API Wrapper ===
async def get_polygon_data(endpoint: str, params: dict) -> dict:
    params["apiKey"] = POLYGON_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"https://api.polygon.io{endpoint}", params=params)
            return res.json()
    except Exception as e:
        logging.warning(f"Polygon API error: {endpoint} | {e}")
        return {}

# === GPT Summary ===
async def gpt_summary(prompt: str) -> str:
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post("https://api.openai.com/v1/chat/completions", headers=BASE_HEADERS, json=payload)
            return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"GPT summary error: {e}")
        return "GPT analysis unavailable."

# === Sentiment Analysis ===
async def get_sentiment_analysis(tickers: List[str]) -> dict:
    bullish, bearish = [], []

    async with httpx.AsyncClient(timeout=10) as client:
        for ticker in tickers:
            try:
                news_url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_API_KEY}"
                news_resp = await client.get(news_url)
                news_data = news_resp.json()
                headlines = [item["title"] for item in news_data.get("results", [])]

                sentiment_scores = []
                for headline in headlines:
                    deepai_url = "https://api.deepai.org/api/sentiment-analysis"
                    deepai_resp = await client.post(
                        deepai_url,
                        data={"text": headline},
                        headers={"api-key": DEEPAI_API_KEY}
                    )
                    deepai_result = deepai_resp.json()
                    if "output" in deepai_result:
                        sentiments = deepai_result["output"]
                        score = sentiments.count("Positive") - sentiments.count("Negative")
                        sentiment_scores.append(score)

                net_sentiment = sum(sentiment_scores)
                if net_sentiment > 1:
                    bullish.append(ticker)
                elif net_sentiment < -1:
                    bearish.append(ticker)

            except Exception as e:
                logging.warning(f"Sentiment error for {ticker}: {e}")

    return {"bullish": bullish, "bearish": bearish}

# === /sentiment endpoint ===
@app.get("/sentiment")
async def sentiment_summary():
    tickers = [
        "AAPL", "TSLA", "AMZN", "GOOG", "META", "CRCL", "PLTR", "CRWV", "NVDA", "AMD",
        "AVGO", "MSFT", "BABA", "UBER", "MSTR", "COIN", "HOOD", "CLSK", "MARA", "CORZ",
        "IONQ", "SOUN", "RGTI", "QBTS", "UNH", "XYZ", "PYPL", "XOM", "CVX"
    ]
    return await get_sentiment_analysis(tickers)

# === /scan endpoint (also used by background worker) ===
@app.get("/scan")
async def run_all_scans():
    start_time = datetime.utcnow()
    now_str = start_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    results = [f"🕒 *Scan Time:* `{now_str}`\n"]

    # 1. Top Gainers
    gainers = await get_polygon_data("/v2/snapshot/locale/us/markets/stocks/gainers", {})
    gainers_list = [t['ticker'] for t in gainers.get("tickers", [])[:30]]
    results.append(f"📈 *Top Gainers*: {', '.join(gainers_list)}")

    # 2. High Volume Spikes
    actives = await get_polygon_data("/v2/snapshot/locale/us/markets/stocks/most_active", {})
    spikes = [t for t in actives.get("tickers", []) if t.get("volume", 0) > 50000]
    results.append(f"🔊 *High Volume Spikes*: {', '.join([s['ticker'] for s in spikes])}")

    # 3. Option Flow
    option_tickers = gainers_list
    flow_results = []
    for ticker in option_tickers[:20]:
        opt_flow = await get_polygon_data("/v3/reference/options/contracts", {"underlying_ticker": ticker, "limit": 2})
        for item in opt_flow.get("results", []):
            flow_results.append(item.get("ticker", "N/A"))
    results.append(f"🧾 *Options Flow Examples*: {', '.join(flow_results)}")

    # 4. Sentiment
    sentiment = await get_sentiment_analysis(option_tickers[:20])
    results.append(f"🐂 *Bullish Sentiment*: {', '.join(sentiment['bullish'])}")
    results.append(f"🐻 *Bearish Sentiment*: {', '.join(sentiment['bearish'])}")

    # 5. Upgrades / Downgrades
    news = await get_polygon_data("/v2/reference/news", {"limit": 30})
    upgrades = [n['title'] for n in news.get("results", []) if "upgrade" in n['title'].lower() or "downgrade" in n['title'].lower()]
    results.append(f"📢 *Upgrades/Downgrades*: {', '.join(upgrades[:10])}")

    # 6. GPT Summary
    full_summary = "\n".join(results)
    gpt_msg = await gpt_summary(f"Analyze and summarize these trading signals:\n{full_summary}")

    combined_message = f"{full_summary}\n\n📊 *AI Summary*:\n{gpt_msg}"
    if len(combined_message) > 4000:
        combined_message = f"{full_summary[:3900]}...\n\n📊 *AI Summary*:\n{gpt_msg[:500]}..."

    await send_telegram(combined_message)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logging.info(f"Scan completed in {duration:.2f} seconds.")
    return {"message": "Scan and summary sent", "duration": duration, "results": results}

# === Background Worker Loop (Render-compatible) ===
@app.on_event("startup")
async def start_background_loop():
    async def loop():
        while True:
            try:
                await run_all_scans()
                await asyncio.sleep(300)  # every 5 minutes
            except Exception as e:
                logging.error(f"Scan loop error: {e}")
                await asyncio.sleep(60)

    asyncio.create_task(loop())
