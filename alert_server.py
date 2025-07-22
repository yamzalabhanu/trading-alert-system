import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import re

import httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# === Load environment ===
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# === FastAPI app ===
app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

cooldowns: Dict[str, datetime] = {}

# === Alert schema ===
class TradingViewAlert(BaseModel):
    symbol: str
    price: float
    action: str
    volume: int
    time: str

# === Telegram ===
async def send_to_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

# === Pre-market & Previous Day Levels ===
async def get_levels(symbol: str) -> Dict[str, float]:
    try:
        async with httpx.AsyncClient() as client:
            prev_resp = await client.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
                params={"adjusted": "true", "apiKey": POLYGON_API_KEY}
            )
            prev_data = prev_resp.json().get("results", [{}])[0]
            prev_high = prev_data.get("h", 0)
            prev_low = prev_data.get("l", 0)

            today = datetime.utcnow().strftime('%Y-%m-%d')
            bars_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}"
            bars_resp = await client.get(
                bars_url,
                params={"adjusted": "true", "limit": 5000, "apiKey": POLYGON_API_KEY}
            )
            bars = bars_resp.json().get("results", [])
            pm_bars = [
                b for b in bars
                if 4 <= datetime.utcfromtimestamp(b['t'] / 1000).hour < 9 or (
                    datetime.utcfromtimestamp(b['t'] / 1000).hour == 9 and
                    datetime.utcfromtimestamp(b['t'] / 1000).minute < 30
                )
            ]
            pm_high = max((b['h'] for b in pm_bars), default=0)
            pm_low = min((b['l'] for b in pm_bars), default=0)

            return {
                "prev_high": prev_high,
                "prev_low": prev_low,
                "pm_high": pm_high,
                "pm_low": pm_low
            }
    except Exception as e:
        logging.error(f"Level fetch error: {e}")
        return {}

# === EMA/VWAP logic ===
def calculate_indicators(df: pd.DataFrame) -> Dict[str, bool]:
    df['ema9'] = df['c'].ewm(span=9).mean()
    df['ema20'] = df['c'].ewm(span=20).mean()
    df['vwap'] = (df['c'] * df['v']).cumsum() / df['v'].cumsum()

    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['c'].ewm(span=12, adjust=False).mean()
    exp2 = df['c'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    latest = df.iloc[-1]
    return {
        "ema_bullish": latest['ema9'] > latest['ema20'],
        "ema_bearish": latest['ema9'] < latest['ema20'],
        "above_vwap": latest['c'] > latest['vwap'],
        "below_vwap": latest['c'] < latest['vwap'],
        "rsi": latest['rsi'],
        "macd": latest['macd'],
        "macd_signal": latest['macd_signal']
    }

# === Polygon context ===
async def get_polygon_context(symbol: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            snap_resp = await client.get(
                f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
                params={"apiKey": POLYGON_API_KEY}
            )
            snap = snap_resp.json().get("ticker", {})
            last_price = snap.get("lastTrade", {}).get("p", "N/A")
            day_change = snap.get("todaysChangePerc", "N/A")
            today_volume = snap.get("day", {}).get("volume", 0)

            hist_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{(datetime.utcnow() - timedelta(days=40)).strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%Y-%m-%d')}"
            hist_resp = await client.get(hist_url, params={"adjusted": "true", "limit": 40, "apiKey": POLYGON_API_KEY})
            hist = hist_resp.json()
            volumes = [d['v'] for d in hist.get('results', [])]
            avg_vol = sum(volumes[-30:]) / 30 if volumes else 0
            unusual_vol = today_volume > 2 * avg_vol
            volume_status = "🧨 Unusual Volume" if unusual_vol else "Normal"

            options_url = f"https://api.polygon.io/v3/reference/options/contracts"
            params = {"underlying_ticker": symbol, "limit": 10, "apiKey": POLYGON_API_KEY}
            opt_resp = await client.get(options_url, params=params)
            opt_data = opt_resp.json().get("results", [])
            unusual_flow = any(o.get("implied_volatility", 0) > 1.0 and o.get("open_interest", 0) > 5000 for o in opt_data)
            flow_status = "🔥 Unusual Options Flow" if unusual_flow else "-"

            return f"Last: {last_price}, %Chg: {day_change}, Vol: {today_volume} ({volume_status}), {flow_status}"
    except Exception as e:
        logging.error(f"Polygon context error: {e}")
        return "Polygon context unavailable"

# === GPT Summary + Confidence Score ===
async def get_gpt_summary(alert: TradingViewAlert, context: str) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        return {"summary": "No GPT summary (missing key)", "confidence": "N/A"}

    prompt = (
        f"Summarize this trading alert and assess its significance. Provide a confidence score (0-100):\n"
        f"Symbol: {alert.symbol}\nPrice: {alert.price}\nAction: {alert.action}\n"
        f"Volume: {alert.volume}\nTime: {alert.time}\nContext: {context}"
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            content = resp.json()["choices"][0]["message"]["content"].strip()
            score_match = re.search(r"confidence.*?(\d{1,3})", content, re.IGNORECASE)
            score = score_match.group(1) if score_match else "N/A"
            return {"summary": content, "confidence": score}
    except Exception as e:
        logging.error(f"GPT error: {e}")
        return {"summary": "GPT summary error", "confidence": "N/A"}

# === Webhook ===
@app.post("/webhook/alerts")
async def receive_alert(alert: TradingViewAlert):
    try:
        alert_dt = datetime.strptime(alert.time, "%Y-%m-%d %H:%M")
        key = f"{alert.symbol}_{alert.action}"
        if key in cooldowns and datetime.utcnow() - cooldowns[key] < timedelta(minutes=5):
            msg = f"⚠️ Cooldown active for {alert.symbol} {alert.action}."
            await send_to_telegram(msg)
            return {"status": "cooldown"}

        cooldowns[key] = datetime.utcnow()

        async with httpx.AsyncClient() as client:
            bars_url = f"https://api.polygon.io/v2/aggs/ticker/{alert.symbol}/range/1/minute/{(datetime.utcnow()-timedelta(days=1)).strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%Y-%m-%d')}"
            resp = await client.get(bars_url, params={"adjusted": "true", "limit": 500, "apiKey": POLYGON_API_KEY})
            bars = resp.json().get("results", [])
            df = pd.DataFrame(bars)

        indicators = calculate_indicators(df) if not df.empty else {}
        context = await get_polygon_context(alert.symbol)
        levels = await get_levels(alert.symbol)
        gpt = await get_gpt_summary(alert, context)
        summary = gpt["summary"]
        confidence = gpt["confidence"]

        confirm = False
        if alert.action == "CALL":
            confirm = indicators.get("ema_bullish") and indicators.get("above_vwap") and indicators.get("rsi", 50) < 70 and indicators.get("macd", 0) > indicators.get("macd_signal", 0)
        elif alert.action == "PUT":
            confirm = indicators.get("ema_bearish") and indicators.get("below_vwap") and indicators.get("rsi", 50) > 30 and indicators.get("macd", 0) < indicators.get("macd_signal", 0)

        status_emoji = "✅ Confirmed" if confirm else "⚠️ Unconfirmed"

        msg = f"""📊 *Trading Alert*
*{alert.symbol}* `{alert.action}` at `${alert.price}`
Volume: {alert.volume} | Time: {alert.time}
Prev High: {levels.get('prev_high', 0)} | Prev Low: {levels.get('prev_low', 0)}
PM High: {levels.get('pm_high', 0)} | PM Low: {levels.get('pm_low', 0)}
Context: {context}
{status_emoji}
*Confidence:* {confidence}
*GPT:* {summary}
"""
        await send_to_telegram(msg)

        os.makedirs("logs", exist_ok=True)
        with open("logs/alerts.log", "a") as f:
            f.write(json.dumps({
                **alert.dict(),
                "confirmed": confirm,
                "confidence": confidence,
                "summary": summary
            }) + "\n")

        return {"status": "success", "confirmed": confirm, "confidence": confidence, "summary": summary}
    except Exception as e:
        logging.error(f"Alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Health ===
@app.get("/status")
def status():
    return {"status": "running", "alerts_count": len(cooldowns)}
