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
        "ema_bullish": bool(latest['ema9'] > latest['ema20']),
        "ema_bearish": bool(latest['ema9'] < latest['ema20']),
        "above_vwap": bool(latest['c'] > latest['vwap']),
        "below_vwap": bool(latest['c'] < latest['vwap']),
        "rsi": float(latest['rsi']),
        "macd": float(latest['macd']),
        "macd_signal": float(latest['macd_signal'])
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
        log_data = {
            **alert.dict(),
            "confirmed": bool(confirm),
            "confidence": str(confidence),
            "summary": str(summary)
        }
        with open("logs/alerts.log", "a") as f:
            f.write(json.dumps(log_data) + "\n")

        return {"status": "success", "confirmed": confirm, "confidence": confidence, "summary": summary}
    except Exception as e:
        logging.error(f"Alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Health ===
@app.get("/status")
def status():
    return {"status": "running", "alerts_count": len(cooldowns)}

# === Back Test ===
@app.get("/backtest")
async def backtest():
    try:
        from_zone = datetime.utcnow() - timedelta(days=30)
        results = []
        total_return = 0
        win_count = 0
        loss_count = 0
        tested = 0
        confidences = []

        if not os.path.exists("logs/alerts.log"):
            return {"error": "No alerts log found."}

        with open("logs/alerts.log", "r") as f:
            lines = f.readlines()

        for line in lines:
            alert = json.loads(line.strip())
            if "symbol" not in alert or "price" not in alert or "time" not in alert:
                continue

            alert_time = datetime.strptime(alert["time"], "%Y-%m-%d %H:%M")
            symbol = alert["symbol"]
            entry_price = alert["price"]
            action = alert["action"]
            confidence = int(alert.get("confidence", 50))
            tested += 1
            confidences.append(confidence)

            # Fetch next 20 minutes of 1-min bars
            end_time = alert_time + timedelta(minutes=20)
            from_str = alert_time.strftime("%Y-%m-%d")
            to_str = (end_time + timedelta(days=1)).strftime("%Y-%m-%d")

            try:
                async with httpx.AsyncClient() as client:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{from_str}/{to_str}"
                    resp = await client.get(url, params={"adjusted": "true", "limit": 1000, "apiKey": POLYGON_API_KEY})
                    data = resp.json().get("results", [])
                    bars = [
                        b for b in data if alert_time.timestamp()*1000 <= b["t"] <= end_time.timestamp()*1000
                    ]
            except Exception as e:
                logging.warning(f"Polygon data fetch failed: {e}")
                continue

            if not bars:
                continue

            prices = [b["c"] for b in bars]
            max_price = max(prices)
            min_price = min(prices)

            gain_pct = ((max_price - entry_price) / entry_price) * 100 if action == "CALL" else ((entry_price - min_price) / entry_price) * 100
            loss_pct = ((entry_price - min_price) / entry_price) * 100 if action == "CALL" else ((max_price - entry_price) / entry_price) * 100

            if gain_pct >= 10:
                win_count += 1
                total_return += 10
            elif loss_pct >= 5:
                loss_count += 1
                total_return -= 5
            else:
                total_return += (gain_pct if action == "CALL" else -gain_pct)

            results.append({
                "symbol": symbol,
                "time": alert["time"],
                "action": action,
                "confidence": confidence,
                "entry": entry_price,
                "max": max_price,
                "min": min_price,
                "gain_pct": round(gain_pct, 2),
                "loss_pct": round(loss_pct, 2)
            })

        win_rate = (win_count / tested) * 100 if tested else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        avg_return = total_return / tested if tested else 0

        return {
            "alerts_tested": tested,
            "win_rate": round(win_rate, 2),
            "avg_return_per_alert": round(avg_return, 2),
            "avg_confidence": round(avg_conf, 2),
            "total_return": round(total_return, 2),
            "backtest_log": results[-10:]  # last 10 entries
        }
    except Exception as e:
        logging.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

