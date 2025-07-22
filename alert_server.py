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

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure connections may fail.")

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

# === GPT Prompt Builder ===
def build_gpt_prompt(alert: TradingViewAlert, enriched: Dict[str, str]) -> str:
    return f"""
Based on the following trading alert and technical/enriched data, provide a structured recommendation including:
- Recommendation (CALL, PUT, or AVOID)
- Confidence score (0-100)
- Reason for the recommendation

### Alert ###
Symbol: {alert.symbol}
Price: {alert.price}
Action: {alert.action}
Volume: {alert.volume}
Time: {alert.time}

### Enriched Data ###
EMA Trend: {enriched.get("ema_trend")}
VWAP Position: {enriched.get("vwap_position")}
MACD: {enriched.get("macd")}
MACD Signal: {enriched.get("macd_signal")}
RSI: {enriched.get("rsi")}
ADX: {enriched.get("adx")}
Breakout Pattern: {enriched.get("breakout_pattern")}
Bid: {enriched.get("bid")}
Ask: {enriched.get("ask")}
Z-Score Volume: {enriched.get("zscore_volume")}
Option Greeks: {enriched.get("option_greeks")}
Context: {enriched.get("context")}
"""

# === GPT Summary + Confidence Score ===
async def get_gpt_summary(alert: TradingViewAlert, enriched: Dict[str, str]) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        return {"summary": "No GPT summary (missing key)", "confidence": "N/A", "recommendation": "N/A"}

    prompt = build_gpt_prompt(alert, enriched)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            content = resp.json()["choices"][0]["message"]["content"].strip()

            # Try extracting from structured GPT output
            recommendation = re.search(r"(?i)recommendation\s*[:\-]\s*(CALL|PUT|AVOID)", content)
            confidence = re.search(r"(?i)confidence.*?(\d{1,3})", content)
            reason = re.search(r"(?i)reason\s*[:\-]\s*(.*)", content)

            return {
                "summary": content,
                "confidence": confidence.group(1) if confidence else "N/A",
                "recommendation": recommendation.group(1).upper() if recommendation else "N/A",
                "reason": reason.group(1) if reason else "N/A"
            }
    except Exception as e:
        logging.error(f"GPT error: {e}")
        return {"summary": "GPT summary error", "confidence": "N/A", "recommendation": "N/A"}
