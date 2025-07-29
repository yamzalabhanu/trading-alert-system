import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# === Config ===
REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "POLYGON_API_KEY",
    "OPENAI_API_KEY",
    "API_SECRET_KEY"
]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        logging.error(f"Missing required environment variable: {var}")
        raise RuntimeError(f"Environment variable {var} not set")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# === Security ===
api_key_header = APIKeyHeader(name="X-API-Key")

async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# === Caching Layer ===
cache: TTLCache = TTLCache(maxsize=100, ttl=300)  # 5 min TTL

# === Models ===
class Alert(BaseModel):
    symbol: str
    price: float
    signal: str
    
    @field_validator('symbol')
    def symbol_uppercase(cls, v):
        return v.upper()

# === Polygon API Helper ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_polygon_data(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict:
    try:
        timeout = httpx.Timeout(10.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logging.error(f"Polygon API error: {e.response.status_code} - {e.response.text}")
        return {}
    except httpx.RequestError as e:
        logging.error(f"Polygon network error: {str(e)}")
        return {}

# === Validation Helpers ===
async def validate_symbol_and_market(symbol: str):
    # Validate symbol
    ref_url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    ref_data = await fetch_polygon_data(
        ref_url, 
        params={"apiKey": POLYGON_API_KEY}
    )
    if not ref_data.get('results'):
        raise HTTPException(status_code=400, detail="Invalid symbol")

    # Check market status
    status_url = "https://api.polygon.io/v1/marketstatus/now"
    status_data = await fetch_polygon_data(
        status_url,
        params={"apiKey": POLYGON_API_KEY}
    )
    if not status_data.get("market", "").lower() == "open":
        raise HTTPException(status_code=403, detail="Market is closed")

# === Polygon Data Fetch with Cache ===
async def get_polygon_data(symbol: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    if cached := cache.get(symbol):
        logging.info(f"[CACHE HIT] {symbol}")
        return cached

    base = "https://api.polygon.io"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    
    endpoints = {
        "unusual": f"{base}/v3/unusual_activity/stocks/{symbol}",
        "ema": f"{base}/v1/indicators/ema/{symbol}",
        "rsi": f"{base}/v1/indicators/rsi/{symbol}",
        "macd": f"{base}/v1/indicators/macd/{symbol}"
    }
    
    params = {
        "ema": {"timespan": "minute", "window": "14", "adjusted": "true", "series_type": "close"},
        "rsi": {"timespan": "minute", "window": "14", "adjusted": "true", "series_type": "close"},
        "macd": {"timespan": "minute", "adjusted": "true", "series_type": "close"}
    }
    
    tasks = {
        "unusual": fetch_polygon_data(endpoints["unusual"], headers=headers),
        "ema": fetch_polygon_data(endpoints["ema"], params={**params["ema"], "apiKey": POLYGON_API_KEY}),
        "rsi": fetch_polygon_data(endpoints["rsi"], params={**params["rsi"], "apiKey": POLYGON_API_KEY}),
        "macd": fetch_polygon_data(endpoints["macd"], params={**params["macd"], "apiKey": POLYGON_API_KEY})
    }
    
    results = await asyncio.gather(*tasks.values())
    result = dict(zip(tasks.keys(), results))
    
    cache[symbol] = result
    return result

# === OpenAI Helper ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_gpt_evaluation(prompt: str) -> str:
    try:
        timeout = httpx.Timeout(30.0, connect=40.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 256
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return "⚠️ GPT evaluation failed. Check logs for details."

# === Webhook Endpoint ===
@app.post("/webhook")
async def handle_alert(
    alert: Alert,
    api_key: str = Depends(validate_api_key)
):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")

    try:
        await validate_symbol_and_market(alert.symbol)
        polygon_data = await get_polygon_data(alert.symbol)

        gpt_prompt = f"""
Evaluate this intraday options signal (be concise):
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Triggered Price: ${alert.price:.2f}

Unusual Options Flow (latest 3):
{str(polygon_data['unusual'].get('results', [])[:3])}

Technical Indicators:
- EMA: {polygon_data['ema'].get('results', {}).get('values', [])[:1] or 'N/A'}
- RSI: {polygon_data['rsi'].get('results', {}).get('values', [])[:1] or 'N/A'}
- MACD: {polygon_data['macd'].get('results', {}).get('values', [])[:1] or 'N/A'}

Respond in this format:
Decision: [Yes/No]
Confidence: [0-100%]
Reason: [1-2 sentences]
"""

        gpt_reply = await get_gpt_evaluation(gpt_prompt)

        # Format Telegram message
        tg_msg = (
            f"🚨 *{alert.signal.upper()} ALERT* for `{alert.symbol}`\n"
            f"💵 Price: ${alert.price:.2f}\n\n"
            f"📊 *GPT Analysis*\n{gpt_reply}"
        )
        
        # Send to Telegram
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(
                telegram_url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": tg_msg,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
            )

        return {"status": "success", "symbol": alert.symbol}

    except HTTPException as e:
        logging.error(f"Validation failed: {e.detail}")
        raise
    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail="Internal server error")
