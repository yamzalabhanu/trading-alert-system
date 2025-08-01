# main.py (Enhanced Trading System with Redis Logging)
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from collections import Counter, defaultdict
import asyncio

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTP may fail in this environment.")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from cachetools import TTLCache
from zoneinfo import ZoneInfo
import httpx
import redis
import json

# === Load Config ===
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# === Redis Client ===
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# === Global Async Client ===
shared_client = httpx.AsyncClient()

# === Caching ===
cache: TTLCache = TTLCache(maxsize=200, ttl=300)
gpt_cache: TTLCache = TTLCache(maxsize=500, ttl=60)

# === Cooldown + Logs ===
cooldown_tracker: Dict[Tuple[str, str], datetime] = {}
COOLDOWN_WINDOW = timedelta(minutes=10)
signal_log: list = []
outcome_log: list = []

# === Models ===
class Alert(BaseModel):
    symbol: str
    price: float
    signal: str
    strike: int | None = None
    expiry: str | None = None
    triggers: str | None = None
    indicators: Dict[str, Any] | None = None

# === Cooldown Helpers ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

def log_signal(symbol: str, signal: str, gpt_decision: str, strike: int | None = None, expiry: str | None = None):
    entry = {
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt": gpt_decision,
        "strike": strike,
        "expiry": expiry,
        "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat()
    }
    signal_log.append(entry)
    try:
        redis_client.rpush("trade_logs", json.dumps(entry))
    except Exception as e:
        logging.warning(f"Redis logging failed: {e}")

# === Helper: Parse Alert String ===
def parse_tradingview_message(msg: str) -> Alert:
    try:
        pattern_legacy = r"Action:\s*(BUY|SELL).*?Symbol:\s*(\w+).*?Price:\s*([\d.]+).*?Option:\s*(CALL|PUT).*?Strike:\s*(\d+).*?Expiry:\s*(\d{4}-\d{2}-\d{2})"
        pattern_simple = r"(CALL|PUT) Signal:\s*(\w+) at ([\d.]+)\s*Strike:\s*(\d+)\s*Expiry:\s*(\d{4}-\d{2}-\d{2})"

        legacy = re.search(pattern_legacy, msg)
        if legacy:
            signal, symbol, price, option_type, strike, expiry = legacy.groups()
            return Alert(
                signal=signal.lower(),
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry
            )

        simple = re.search(pattern_simple, msg.replace("\n", " "))
        if simple:
            option_type, symbol, price, strike, expiry = simple.groups()
            return Alert(
                signal="buy" if option_type.upper() == "CALL" else "sell",
                symbol=symbol.upper(),
                price=float(price),
                strike=int(strike),
                expiry=expiry
            )

        raise ValueError("Unrecognized alert message format")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse alert: {e}")
