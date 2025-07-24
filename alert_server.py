import os
import httpx
import asyncio
import logging
import re
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
def escape_markdown(text: str) -> str:
    return re.sub(r'([_\*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)

async def send_telegram(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        if len(message) > 4000:
            message = message[:3990] + "\n...truncated"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": escape_markdown(message),
                    "parse_mode": "MarkdownV2"
                })
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
                news = await get_finnhub("/company-news", {
                    "symbol": ticker,
                    "from": datetime.now().strftime('%Y-%m-%d'),
                    "to": datetime.now().strftime('%Y-%m-%d')
                })

                if not news:
                    continue

                headlines = [item.get("headline", "") for item in news[:5] if item.get("headline")]
                if not headlines:
                    continue

                sentiment_scores = []
                for headline in headlines:
                    try:
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
                    except Exception as de:
                        logging.warning(f"DeepAI error for headline '{headline[:50]}...': {de}")

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
            if res.status_code != 200:
                logging.error(f"GPT API error {res.status_code}: {res.text}")
            data = res.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "GPT response incomplete.").strip()
    except Exception as e:
        logging.error(f"GPT summary error: {e}")
        return "GPT analysis unavailable."

# === Main Scan Logic ===
@app.get("/scan")
async def run_all_scans():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    results = [f"🕒 *Scan Time:* `{now}`\n"]

    tickers = [
        "AAPL", "TSLA", "AMZN", "GOOG", "META", "CRCL", "PLTR", "CRWV", "NVDA", "AMD",
        "AVGO", "MSFT", "BABA", "UBER", "MSTR", "COIN", "HOOD", "CLSK", "MARA", "CORZ",
        "IONQ", "SOUN", "RGTI", "QBTS", "UNH", "XYZ", "PYPL", "XOM", "CVX"
    ]

    results.append(f"📈 *Watchlist Symbols*: {', '.join(tickers)}")

    sentiment = await get_sentiment_analysis(tickers)
    results.append(f"🐂 *Bullish*: {', '.join(sentiment['bullish']) or 'None'}")
    results.append(f"🐻 *Bearish*: {', '.join(sentiment['bearish']) or 'None'}")

    headlines = await get_top_news()
    results.append("📰 *Top Headlines:*\n" + '\n'.join([f"- {h}" for h in headlines]))

    # === Polygon Options Flow ===
    option_flow = []
    unusual = []
    for symbol in tickers:
        opt = await get_polygon("/v3/reference/options/contracts", {"underlying_ticker": symbol, "limit": 30})
        for o in opt.get("results", []):
            option_flow.append(o.get("ticker"))
            if o.get("implied_volatility", 0) and o.get("open_interest", 0):
                score = o.get("implied_volatility") * o.get("open_interest")
                if score > 5000:
                    unusual.append(f"{o.get('ticker')} (IVxOI={score:.0f})")

    if option_flow:
        results.append("🧾 *Options Flow:* " + ', '.join(option_flow))
    if unusual:
        results.append("⚠️ *Unusual Activity:* " + ', '.join(unusual))

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
