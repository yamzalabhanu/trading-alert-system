# main.py
import os
import httpx
import logging
from fastapi import FastAPI, Request
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

@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")

    try:
        # === Get Polygon snapshot ===
        async with httpx.AsyncClient() as client:
            polygon_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{alert.symbol.upper()}?apiKey={POLYGON_API_KEY}"
            snapshot_resp = await client.get(polygon_url)
            if snapshot_resp.status_code != 200:
                logging.warning(f"Polygon snapshot error: {snapshot_resp.status_code}")
                snapshot_data = {"error": f"Polygon returned {snapshot_resp.status_code}"}
            else:
                snapshot_data = snapshot_resp.json()

        # === Compose OpenAI prompt ===
        gpt_prompt = f"""
Evaluate this trading signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Triggered Price: {alert.price}

Market Snapshot: {snapshot_data}

Respond with: 
- Trade decision (Yes/No)
- Confidence score (0–100)
- 1-line reasoning
        """

        # === Call OpenAI GPT ===
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

        # === Send to Telegram ===
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
