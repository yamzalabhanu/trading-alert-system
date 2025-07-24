import os
import httpx
import asyncio
import logging
from datetime import datetime, time
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv

# Fallback to no SSL in limited environments
try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTP may fail in this environment.")

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

# === Load environment variables ===
load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")
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

# === Finnhub Wrappers ===
async def get_finnhub(endpoint: str, params: dict) -> dict:
    base_url = f"https://finnhub.io/api/v1{endpoint}"
    params["token"] = FINNHUB_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(base_url, params=params)
            return res.json()
    except Exception as e:
        logging.error(f"Finnhub API error ({endpoint}): {e}")
        return {}

# === Polygon Option Flow ===
async def get_polygon(endpoint: str, params: dict) -> dict:
    params["apiKey"] = POLYGON_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"https://api.polygon.io{endpoint}", params=params)
            return res.json()
    except Exception as e:
        logging.warning(f"Polygon API error: {endpoint} | {e}")
        return {}

# === Market News ===
async def get_top_news(limit: int = 10) -> List[str]:
    news = await get_finnhub("/news", {"category": "general"})
    return [item['headline'] for item in news[:limit]]

# === Sentiment Analysis ===
async def get_sentiment_analysis(tickers: List[str]) -> dict:
    bullish, bearish = [], []

    async with httpx.AsyncClient(timeout=10) as client:
        for ticker in tickers:
            try:
                news = await get_finnhub("/company-news", {"symbol": ticker, "from": datetime.now().strftime('%Y-%m-%d'), "to": datetime.now().strftime('%Y-%m-%d')})
                headlines = [item["headline"] for item in news[:5]]

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

                net = sum(sentiment_scores)
                if net > 1:
                    bullish.append(ticker)
                elif net < -1:
                    bearish.append(ticker)

            except Exception as e:
                logging.warning(f"Sentiment error for {ticker}: {e}")

    return {"bullish": bullish, "bearish": bearish}

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

# === Main Scan Logic ===
@app.get("/scan")
async def run_all_scans():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    results = [f"🕒 *Scan Time:* `{now}`\n"]

    gainers = await get_finnhub("/stock/symbol", {"exchange": "US"})
    top_tickers = [s['symbol'] for s in gainers[:20]] if gainers else []
    results.append(f"📈 *Active Symbols*: {', '.join(top_tickers)}")

    sentiment = await get_sentiment_analysis(top_tickers[:15])
    results.append(f"🐂 *Bullish*: {', '.join(sentiment['bullish'])}")
    results.append(f"🐻 *Bearish*: {', '.join(sentiment['bearish'])}")

    headlines = await get_top_news()
    results.append("📰 *Top Headlines:*\n" + '\n'.join([f"- {h}" for h in headlines]))

    # === Polygon Options Flow ===
    option_flow = []
    for symbol in top_tickers[:10]:
        opt = await get_polygon("/v3/reference/options/contracts", {"underlying_ticker": symbol, "limit": 2})
        for o in opt.get("results", []):
            option_flow.append(o.get("ticker"))
    if option_flow:
        results.append("🧾 *Options Flow:* " + ', '.join(option_flow))

    summary_input = "\n".join(results)
    gpt_msg = await gpt_summary(f"Summarize today's market scan:\n{summary_input}")

    combined_message = f"{summary_input}\n\n📊 *AI Summary:*\n{gpt_msg}"
    await send_telegram(combined_message)

    return {"status": "complete", "summary": gpt_msg}

# === Background Loop: Only During U.S. Market Hours ===
@app.on_event("startup")
async def start_background_loop():
    async def loop():
        eastern = ZoneInfo("America/New_York")
        market_open = time(9, 30)
        market_close = time(16, 0)

        while True:
            now = datetime.now(tz=eastern)
            if now.weekday() < 5 and market_open <= now.time() <= market_close:
                logging.info(f"Market open — running scan at {now}")
                try:
                    await run_all_scans()
                except Exception as e:
                    logging.error(f"Error in scan loop: {e}")
            else:
                logging.info(f"Outside market hours ({now.time()}) — skipping scan.")

            await asyncio.sleep(300)

    asyncio.create_task(loop())
