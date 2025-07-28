# main.py
import os
import logging

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTPS requests may fail.")

try:
    import httpx
except ImportError as e:
    raise ImportError("httpx module is required but could not be loaded. Check if ssl support is available.") from e

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Alert(BaseModel):
    symbol: str
    price: float
    signal: str

async def get_polygon_data(symbol: str):
    base = f"https://api.polygon.io"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    try:
        async with httpx.AsyncClient() as client:
            options_url = f"{base}/v3/universal/snapshot/options/{symbol.upper()}"
            options_resp = await client.get(options_url, headers=headers)
            options_data = options_resp.json() if options_resp.status_code == 200 else {}

            ema_url = f"{base}/v1/indicators/ema/{symbol.upper()}?timespan=minute&window=14&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"
            ema_resp = await client.get(ema_url)
            ema_data = ema_resp.json() if ema_resp.status_code == 200 else {}

            rsi_url = f"{base}/v1/indicators/rsi/{symbol.upper()}?timespan=minute&window=14&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"
            rsi_resp = await client.get(rsi_url)
            rsi_data = rsi_resp.json() if rsi_resp.status_code == 200 else {}

            macd_url = f"{base}/v1/indicators/macd/{symbol.upper()}?timespan=minute&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"
            macd_resp = await client.get(macd_url)
            macd_data = macd_resp.json() if macd_resp.status_code == 200 else {}
    except Exception as e:
        logging.warning(f"Polygon fetch failed: {e}")
        return {"options": {}, "ema": {}, "rsi": {}, "macd": {}}

    return {
        "options": options_data,
        "ema": ema_data,
        "rsi": rsi_data,
        "macd": macd_data
    }

@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")

    try:
        polygon_data = await get_polygon_data(alert.symbol)

        gpt_prompt = f"""
Evaluate this trading signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Triggered Price: {alert.price}

Options Flow Snapshot:
{polygon_data['options']}

Technical Indicators:
EMA: {polygon_data['ema']}
RSI: {polygon_data['rsi']}
MACD: {polygon_data['macd']}

Respond with:
- Trade decision (Yes/No)
- Confidence score (0–100)
- 1-line reasoning
        """

        async with httpx.AsyncClient() as client:
            gpt_resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": gpt_prompt}],
                    "temperature": 0.3
                }
            )
            gpt_reply = gpt_resp.json()["choices"][0]["message"]["content"]

        tg_msg = f"📈 *{alert.signal.upper()} ALERT* for `{alert.symbol}` @ `${alert.price}`\n\n📊 GPT Review:\n{gpt_reply}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg, "parse_mode": "Markdown"}
            )

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))
