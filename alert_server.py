# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List
from collections import Counter, defaultdict
import asyncio

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTP may fail in this environment.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from cachetools import TTLCache
from zoneinfo import ZoneInfo
import httpx

# === Load Config ===
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Caching ===
cache: TTLCache = TTLCache(maxsize=100, ttl=300)

# === Cooldown + Logs ===
cooldown_tracker: Dict[Tuple[str, str], datetime] = {}
COOLDOWN_WINDOW = timedelta(minutes=10)
signal_log: list = []

# === Models ===
class Alert(BaseModel):
    symbol: str
    price: float
    signal: str

# === Cooldown & Logs ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

def log_signal(symbol: str, signal: str, gpt_decision: str):
    signal_log.append({
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt": gpt_decision,
        "timestamp": datetime.now(ZoneInfo("America/New_York"))
    })

# === Validation ===

# 🔼 Trendline Breakout Detection (Linear Regression)
from numpy.polynomial.polynomial import Polynomial
import numpy as np

async def detect_trendline_breakout(symbol: str, price: float) -> str:
    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=10)
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{start}/{end}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json().get("results", [])[-50:]  # Last 50 points

        if len(data) < 20:
            return "insufficient data"

        closes = np.array([bar["c"] for bar in data])
        x = np.arange(len(closes))

        # Fit linear trendline (degree=1)
        p = Polynomial.fit(x, closes, 1)
        trend = p(x)

        # Check if the latest price is a breakout (above trend + std)
        std_dev = np.std(closes - trend)
        latest = closes[-1]
        upper = trend[-1] + std_dev
        lower = trend[-1] - std_dev

        if latest > upper:
            return "breakout"
        elif latest < lower:
            return "breakdown"
        else:
            return "neutral"
    except Exception as e:
        logging.warning(f"Trendline breakout detection failed: {e}")
        return "error"
async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    async with httpx.AsyncClient() as client:
        ref_url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        ref_resp = await client.get(ref_url)
        if ref_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid symbol")

        status_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
        status_resp = await client.get(status_url)
        if not allow_closed and (status_resp.status_code != 200 or not status_resp.json().get("market", "").lower() == "open"):
            raise HTTPException(status_code=403, detail="Market is closed")

# === Fetch Full Options Chain ===
async def fetch_filtered_options(symbol: str, current_price: float) -> List[Dict[str, Any]]:
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            options = resp.json().get("results", [])

        this_week = datetime.utcnow().isocalendar().week
        next_week = this_week + 1
        current_year = datetime.utcnow().year

        filtered = []
        for opt in options:
            details = opt.get("details", {})
            contract_type = details.get("contract_type")
            expiry_date = details.get("expiration_date")
            strike = details.get("strike_price")

            if not expiry_date or not strike:
                continue

            expiry_week = datetime.fromisoformat(expiry_date).isocalendar().week
            expiry_year = datetime.fromisoformat(expiry_date).year

            if expiry_year != current_year or expiry_week not in (this_week, next_week):
                continue

            if contract_type == "call" and strike >= current_price * 1.05:
                opt["bid"] = opt.get("last_quote", {}).get("bid")
                opt["ask"] = opt.get("last_quote", {}).get("ask")
                opt["oi"] = opt.get("open_interest")
                opt["volume"] = opt.get("volume")
                opt["delta"] = opt.get("greeks", {}).get("delta")
                if opt["volume"] and opt["volume"] >= 100 and opt["oi"] and opt["oi"] >= 500 and opt["delta"] and 0.4 <= abs(opt["delta"]) <= 0.8:
                    filtered.append(opt)
            elif contract_type == "put" and strike <= current_price * 0.95:
                opt["bid"] = opt.get("last_quote", {}).get("bid")
                opt["ask"] = opt.get("last_quote", {}).get("ask")
                opt["oi"] = opt.get("open_interest")
                opt["volume"] = opt.get("volume")
                opt["delta"] = opt.get("greeks", {}).get("delta")
                if opt["volume"] and opt["volume"] >= 100 and opt["oi"] and opt["oi"] >= 500 and opt["delta"] and abs(opt["delta"]) >= 0.7:
                    filtered.append(opt)

        return filtered
    except Exception as e:
        logging.warning(f"Failed to fetch full options chain: {e}")
        return []

# === Webhook Endpoint ===

@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")

    if is_in_cooldown(alert.symbol, alert.signal):
        logging.info(f"⏱️ Cooldown active for {alert.symbol} - {alert.signal}")
        return {"status": "cooldown"}

    try:
        await validate_symbol_and_market(alert.symbol, allow_closed=True)
        polygon_data = await get_polygon_data(alert.symbol)
        trend_status = await detect_trendline_breakout(alert.symbol, alert.price)
        greeks = await get_option_greeks(alert.symbol)
        enriched_options = await fetch_filtered_options(alert.symbol, alert.price)

        gpt_prompt = f"""
Evaluate this intraday options signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()} ({trend_status})
Triggered Price: {alert.price}

Filtered Option Contracts (ATM calls > 5% strike, ITM puts < 95%):
{enriched_options}

Unusual Options Flow:
{polygon_data['unusual']}

Technical Indicators:
EMA: {polygon_data['ema']}
RSI: {polygon_data['rsi']}
MACD: {polygon_data['macd']}

Option Greeks:
Delta: {greeks.get('delta')}
Gamma: {greeks.get('gamma')}
Theta: {greeks.get('theta')}
IV: {greeks.get('iv')}

Respond with:
- Trade decision (Yes/No)
- Confidence score (0–100)
- Brief reasoning (1 sentence)
- If Yes, suggest top 1–2 contracts (strike, expiry, volume, delta, bid/ask)
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

        gpt_decision = "unknown"
        if "yes" in gpt_reply.lower():
            gpt_decision = "buy"
        elif "no" in gpt_reply.lower():
            gpt_decision = "skip"

        tg_msg = ("📈 *{} ALERT* for `{}` @ `${}`\n\n📊 GPT Review:\n{}"
                  .format(alert.signal.upper(), alert.symbol, alert.price, gpt_reply))
        await httpx.AsyncClient().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_msg, "parse_mode": "Markdown"}
        )

        update_cooldown(alert.symbol, alert.signal)
        log_signal(alert.symbol, alert.signal, gpt_decision)
        await track_backtest_outcome(alert.symbol, alert.price, datetime.utcnow(), gpt_decision)

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Background Cron & Summary ===

# 🔍 Real-time Unusual Options Activity Scanner
async def scan_unusual_activity():
    watchlist = ["AAPL", "TSLA", "AMD", "MSFT", "NVDA", "GOOG", "PLTR", "CRCL", "CRWV", "AMZN", "HOOD", "IONQ", "OKLO", "COIN", "MSTR", "UNH", "PDD", "BABA", "XOM", "CVX"]
    while True:
        try:
            now = datetime.now(ZoneInfo("America/New_York"))
            if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
                await asyncio.sleep(60)
                continue

            async with httpx.AsyncClient() as client:
                for symbol in watchlist:
                    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
                    r = await client.get(url)
                    data = r.json().get("results", [])
                    spikes = [opt for opt in data if opt.get("volume", 0) > 5000 and opt.get("open_interest", 0) > 5000]
                    if spikes:
                        logging.info(f"Unusual option volume detected for {symbol} ({len(spikes)} contracts)")
                        await handle_alert(Alert(symbol=symbol, price=0, signal="flow"))
        except Exception as e:
            logging.warning(f"Unusual activity scan error: {e}")
        await asyncio.sleep(300)

# 📉 Backtesting Tracker
backtest_log: List[Dict[str, Any]] = []

async def track_backtest_outcome(symbol: str, entry_price: float, signal_time: datetime, decision: str):
    try:
        async with httpx.AsyncClient() as client:
            end_time = signal_time + timedelta(days=1)
            date_str = signal_time.strftime("%Y-%m-%d")
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{date_str}/{date_str}?apiKey={POLYGON_API_KEY}"
            r = await client.get(url)
            data = r.json().get("results", [])
            next_open = next((bar["o"] for bar in data if bar["t"] > signal_time.timestamp() * 1000), None)

        pnl = round(((next_open - entry_price) / entry_price) * 100, 2) if next_open else None
        backtest_log.append({
            "symbol": symbol,
            "decision": decision,
            "entry": entry_price,
            "exit": next_open,
            "pnl%": pnl,
            "timestamp": signal_time.isoformat()
        })
    except Exception as e:
        logging.warning(f"Backtest tracking failed: {e}")

async def schedule_daily_summary():
    while True:
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.hour == 16 and now.minute == 15:
            await send_daily_summary()
            await asyncio.sleep(60)
        await asyncio.sleep(30)

async def send_daily_summary():
    try:
        now = datetime.now(ZoneInfo("America/New_York"))
        summary = f"📊 *Daily Summary Report* ({now.strftime('%Y-%m-%d')}):\n\n"

        if not signal_log:
            summary += "_No trading signals today._"
        else:
            counter = Counter((log["symbol"], log["signal"]) for log in signal_log)
            gpt_counter = defaultdict(int)
            for log in signal_log:
                gpt_counter[log["gpt"].lower()] += 1

            summary += "🔝 *Top Symbols:*\n"
            for (sym, sig), count in counter.most_common(5):
                summary += f"- `{sym}` ({sig.upper()}): {count} signals\n"

            summary += "\n🧠 *GPT Decisions:*\n"
            for decision, count in gpt_counter.items():
                summary += f"- {decision.title()}: {count}\n"

        await httpx.AsyncClient().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": summary, "parse_mode": "Markdown"}
        )
        signal_log.clear()
    except Exception as e:
        logging.warning(f"Daily summary failed: {e}")

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(schedule_daily_summary())
    loop.create_task(scan_unusual_activity())
    loop = asyncio.get_event_loop()
    loop.create_task(schedule_daily_summary())
