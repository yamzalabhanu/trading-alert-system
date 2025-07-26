import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    import ssl
except ImportError:
    ssl = None
    import warnings
    warnings.warn("SSL module not found. HTTPS requests may fail.")

try:
    import httpx
except ImportError as e:
    raise ImportError("httpx module is required but failed to import. Ensure 'ssl' is available in your environment.") from e

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI()
alerts_db = []  # In-memory DB (use SQLite or Postgres in prod)

class TradingSignal(BaseModel):
    symbol: str
    price: float
    signal: str
    strike: int
    confidence: int
    expiry: str
    time: str
    gptReady: Optional[bool] = False

# === Helpers ===
async def fetch_options_data(symbol, strike, expiry, direction):
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}"
    params = {"apiKey": POLYGON_API_KEY}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()

    contracts = [c for c in data.get("results", []) if (
        c["details"]["strike_price"] == strike and
        expiry in c["details"]["expiration_date"] and
        c["details"]["contract_type"].lower() == ("call" if direction == "BUY" else "put")
    )]

    if not contracts:
        return None

    best = sorted(contracts, key=lambda x: abs(x["greeks"]["delta"] - 0.5))[0]
    return {
        "bid": best["last_quote"]["bid"],
        "ask": best["last_quote"]["ask"],
        "iv": best["greeks"]["implied_volatility"],
        "delta": best["greeks"]["delta"],
        "gamma": best["greeks"]["gamma"],
        "theta": best["greeks"]["theta"],
        "vega": best["greeks"]["vega"],
        "oi": best["open_interest"],
        "volume": best["volume"]
    }

async def gpt_score(signal: dict, options: dict) -> str:
    prompt = f"""
The user received a trading signal to {signal['signal']} {signal['symbol']} at ${signal['price']}.
Options Contract Strike: {signal['strike']}, Expiry: {signal['expiry']}
Greeks: Delta={options['delta']}, Gamma={options['gamma']}, IV={options['iv']}, Volume={options['volume']}, OI={options['oi']}

Is this a safe entry for an intraday options trade? Respond with SAFE, NEUTRAL, or RISKY.
"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content'].strip().upper()

async def send_telegram(msg: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        await httpx.AsyncClient().post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# === Routes ===
@app.post("/webhook")
async def receive_alert(alert: TradingSignal):
    alert_dict = alert.dict()

    options_data = await fetch_options_data(
        symbol=alert.symbol,
        strike=alert.strike,
        expiry=alert.expiry,
        direction=alert.signal
    ) if alert.gptReady else None

    gpt_rating = await gpt_score(alert_dict, options_data) if options_data else "UNKNOWN"
    alert_dict.update(options_data or {})
    alert_dict["gpt_rating"] = gpt_rating

    alerts_db.append(alert_dict)

    message = f"{alert.signal} ALERT: {alert.symbol} @ {alert.price}\nStrike: {alert.strike} Exp: {alert.expiry}\nDelta: {alert_dict.get('delta', '?')} IV: {alert_dict.get('iv', '?')}\nGPT: {gpt_rating}"
    await send_telegram(message)
    return {"status": "ok", "gpt_rating": gpt_rating}

@app.get("/dashboard")
async def dashboard():
    return alerts_db
