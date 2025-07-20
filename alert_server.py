import os
import json
import logging
from datetime import datetime, timedelta, time as dtime
from typing import Optional
from pathlib import Path
import asyncio
import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, validator
from dotenv import load_dotenv

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
        if v not in ("CALL", "PUT"):
            raise ValueError("action must be CALL or PUT")
        return v.upper()

def is_market_open():
    now = datetime.now(TIMEZONE).time()
    return dtime(9, 30) <= now <= dtime(16, 0)

async def get_combined_sentiment(symbol: str):
    if not DEEPAI_API_KEY:
        logging.warning("DEEPAI_API_KEY is not set. Skipping sentiment analysis.")
        return 0

    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=5&apiKey={POLYGON_API_KEY}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        headlines = [item["title"] for item in res.json().get("results", [])]

    scores = []
    sentiment_logs[symbol] = []

    for title in headlines:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepai.org/api/sentiment-analysis",
                    data={"text": title},
                    headers={"api-key": str(DEEPAI_API_KEY)},
                )
                json_response = response.json()
                output = json_response.get("output")
                if not output or not isinstance(output, list):
                    raise ValueError("Invalid sentiment response structure")
                result = output[0]
                score = {"positive": 1, "neutral": 0, "negative": -1}.get(result, 0)
                scores.append(score)
                sentiment_logs[symbol].append({"headline": title, "score": score})
        except Exception as e:
            logging.warning(f"Sentiment fetch error: {e}")
            sentiment_logs[symbol].append({"headline": title, "score": 0})
    return sum(scores) / len(scores) if scores else 0

async def fetch_market_indicators(symbol):
    async with httpx.AsyncClient() as client:
        prev_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
        prev_data = await client.get(prev_url)
        prev_json = prev_data.json().get("results", [{}])[0]

        snapshot = {}
        if is_market_open():
            snap_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}?apiKey={POLYGON_API_KEY}"
            snap_data = await client.get(snap_url)
            snapshot = snap_data.json().get("ticker", {})

        sweep_url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
        sweep_data = await client.get(sweep_url)
        try:
            sweeps = sweep_data.json().get("results", {})
        except Exception as e:
            logging.error(f"Failed to parse sweep/flow data: {e} - Content: {sweep_data.text}")
            sweeps = {}

        options_url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&limit=20&apiKey={POLYGON_API_KEY}"
        option_data = await client.get(options_url)
        contracts = option_data.json().get("results", [])

        best_option = max(contracts, key=lambda x: (x.get("implied_volatility", 0) * x.get("open_interest", 0)), default={})

    return {
        "previous": prev_json,
        "snapshot": snapshot,
        "sweeps": sweeps,
        "best_option": best_option,
        "ema_pattern": {},
        "rsi_signal": {},
        "macd_signal": {}
    }

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
            return res.json().get("choices", [{}])[0].get("message", {}).get("content")
    except httpx.HTTPStatusError as e:
        logging.warning(f"OpenAI error: {e.response.status_code}")
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

async def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

async def send_telegram_alert(decision, indicators):
    iv = indicators.get("best_option", {}).get("implied_volatility", "N/A")
    oi = indicators.get("best_option", {}).get("open_interest", "N/A")
    sweeps = indicators.get("sweeps", [])
    if isinstance(sweeps, list) and sweeps:
        sweep_info = sweeps[0].get("underlying_asset", {}).get("trend", "N/A")
    else:
        sweep_info = "N/A"
    pattern = indicators.get("ema_pattern", {}).get("pattern") or indicators.get("rsi_signal", {}).get("pattern") or indicators.get("macd_signal", {}).get("pattern")

    text = f"<b>{decision['symbol']} {decision['type'].upper()}</b>\n" \
           f"Strike: {decision['strike']} | Expiry: {decision['expiry']}\n" \
           f"Quantity: {decision['quantity']} | Confidence: {decision['confidence']}%\n" \
           f"IV: {iv} | OI: {oi} | Sweep: {sweep_info}\n"
    if pattern:
        text += f"Pattern: {pattern}\n"

    await send_telegram_message(text)

@app.post("/alert")
async def alert(alert: TradingViewAlert):
    symbol = alert.symbol.upper()
    now = datetime.now()
    if symbol in cooldowns and (now - cooldowns[symbol]).total_seconds() < COOLDOWN_SECONDS:
        logging.info(f"{symbol} alert skipped due to cooldown")
        return {"status": "cooldown"}

    indicators = await fetch_market_indicators(symbol)
    sentiment = await get_combined_sentiment(symbol)
    decision = await analyze_with_llm(symbol, alert.action.lower(), indicators, sentiment)

    if decision:
        await send_telegram_alert(decision, indicators)
        cooldowns[symbol] = now
        return {"status": "alert_sent", "decision": decision}
    else:
        logging.warning(f"⚠️ Trade not taaken for {symbol}")
        return {"status": "rejected"}

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
