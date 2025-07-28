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

    # Fetch real-time Polygon data
    async with httpx.AsyncClient() as client:
        snapshot_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{alert.symbol.upper()}?apiKey={POLYGON_API_KEY}"
        snapshot = (await client.get(snapshot_url)).json()

    # Compose GPT input
    gpt_prompt = f"""
    Evaluate the following signal for {alert.symbol}:
    Signal: {alert.signal.upper()}
    Price: {alert.price}
    Current Snapshot: {snapshot}
    Should the trade be taken? Respond with reason and confidence.
    """

    # OpenAI Evaluation
    openai_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    gpt_response = await client.post(openai_url, json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": gpt_prompt}],
        "temperature": 0.3
    }, headers=headers)

    reply = gpt_response.json()["choices"][0]["message"]["content"]

    # Send to Telegram
    tg_msg = f"📈 {alert.signal.upper()} Alert: {alert.symbol} @ ${alert.price:.2f}\n\nGPT Review:\n{reply}"
    tg_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    await client.post(tg_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg})

    return {"status": "ok", "review": reply}
