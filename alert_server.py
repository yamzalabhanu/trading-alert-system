# main.py
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
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

async def call_openai_chat(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if resp.status_code == 429:
                logging.warning("⚠️ OpenAI rate limit hit.")
                return "⚠️ GPT rate limit hit. No decision made."

            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ GPT returned no content.")
    except Exception as e:
        logging.error(f"OpenAI GPT error: {e}")
        return "⚠️ GPT unavailable due to an internal error."


# === Validation ===
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

# === Polygon Data Fetch ===
async def get_polygon_data(symbol: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    if symbol in cache:
        logging.info(f"[CACHE HIT] {symbol}")
        return cache[symbol]

    base = "https://api.polygon.io"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}

    try:
        async with httpx.AsyncClient() as client:
            unusual_url = f"{base}/v3/unusual_activity/stocks/{symbol}"
            ema_url = f"{base}/v1/indicators/ema/{symbol}?timespan=minute&window=14&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"
            rsi_url = f"{base}/v1/indicators/rsi/{symbol}?timespan=minute&window=14&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"
            macd_url = f"{base}/v1/indicators/macd/{symbol}?timespan=minute&adjusted=true&series_type=close&apiKey={POLYGON_API_KEY}"

            r = await client.get(unusual_url, headers=headers)
            unusual = r.json() if r.status_code == 200 else {}

            r = await client.get(ema_url)
            ema = r.json() if r.status_code == 200 else {}

            r = await client.get(rsi_url)
            rsi = r.json() if r.status_code == 200 else {}

            r = await client.get(macd_url)
            macd = r.json() if r.status_code == 200 else {}

        result = {"unusual": unusual, "ema": ema, "rsi": rsi, "macd": macd}
        cache[symbol] = result
        return result

    except Exception as e:
        logging.warning(f"Polygon fetch failed: {e}")
        return {"unusual": {}, "ema": {}, "rsi": {}, "macd": {}}

# === Option Greeks ===
async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json().get("results", [])

        nearest = None
        current_price = None

        for option in data:
            if option.get("details", {}).get("contract_type") != "call":
                continue
            if not current_price:
                current_price = option.get("underlying_asset", {}).get("last", {}).get("price", 0)
            strike = option.get("details", {}).get("strike_price", 0)
            if current_price and abs(strike - current_price) < abs((nearest or {}).get("details", {}).get("strike_price", 0) - current_price):
                nearest = option

        if not nearest:
            return {}

        greeks = nearest.get("greeks", {})
        iv = nearest.get("implied_volatility", {}).get("midpoint", None)
        return {
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "theta": greeks.get("theta"),
            "iv": iv,
        }
    except Exception as e:
        logging.warning(f"Failed to fetch option Greeks: {e}")
        return {}

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
        greeks = await get_option_greeks(alert.symbol)

        gpt_prompt = f"""
Evaluate this intraday options signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Triggered Price: {alert.price}

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
            gpt_reply = await call_openai_chat(gpt_prompt)


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

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Webhook processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Background Cron & Summary ===

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
    loop.create_task(scan_unusual_activity())
    loop.create_task(schedule_daily_summary())

async def scan_unusual_activity():
    symbols_to_check = ["AAPL", "TSLA", "AMD", "MSFT", "NVDA", "GOOG", "PLTR", "CRCL", "CRWV", "AMZN", "HOOD", "IONQ", "OKLO", "COIN", "MSTR", "UNH", "PDD", "BABA", "XOM", "CVX"]
    logging.info("📡 Simulating unusual activity based on volume and OI spikes...")
    
    while True:
        try:
            async with httpx.AsyncClient() as client:
                for symbol in symbols_to_check:
                    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}?apiKey={POLYGON_API_KEY}"
                    try:
                        resp = await client.get(url)
                        data = resp.json().get("results", [])
                        high_volume_contracts = [opt for opt in data if opt.get("volume", 0) > 500 and opt.get("open_interest", 0) > 1000]

                        if high_volume_contracts:
                            logging.info(f"🔥 Unusual contracts found for {symbol}: {len(high_volume_contracts)}")
                            await client.post(
                                "http://localhost:8000/webhook",
                                json={"symbol": symbol, "price": 0, "signal": "flow", "allow_closed": True}
                            )
                    except Exception as e:
                        logging.warning(f"Error checking {symbol}: {e}")
        except Exception as e:
            logging.warning(f"Unusual scan error: {e}")

        await asyncio.sleep(300)
