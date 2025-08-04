# main.py (Enhanced Trading System with Redis Logging)
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
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
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

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

class Outcome(BaseModel):
    symbol: str
    signal: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    profit: float | None = None
    profit_pct: float | None = None
    outcome: str  # "win" or "loss"

# === Initialize Scheduler ===
scheduler = AsyncIOScheduler(timezone=ZoneInfo("America/New_York"))

# === Cooldown Helpers ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

# === Enhanced Signal Logging ===
def log_signal(symbol: str, signal: str, gpt_reply: str, strike: int | None = None, expiry: str | None = None):
    # Parse confidence from GPT reply
    confidence = 0
    match = re.search(r"confidence[:=]?\s*(\d+)", gpt_reply.lower())
    if match:
        confidence = int(match.group(1))
    
    # Extract reasoning
    reason = "No reason provided"
    if "reason:" in gpt_reply.lower():
        reason = gpt_reply.split("Reason:")[-1].strip()
    elif "-" in gpt_reply:
        reason = gpt_reply.split("-")[-1].strip()
    
    entry = {
        "symbol": symbol.upper(),
        "signal": signal.lower(),
        "gpt_reply": gpt_reply,
        "confidence": confidence,
        "reason": reason,
        "strike": strike,
        "expiry": expiry,
        "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat()
    }
    signal_log.append(entry)
    try:
        redis_client.rpush("trade_logs", json.dumps(entry))
    except Exception as e:
        logging.warning(f"Redis logging failed: {e}")

# === Initialize Logs from Redis ===
try:
    # Load trade signals
    redis_signals = redis_client.lrange("trade_logs", 0, -1)
    for entry in redis_signals:
        signal_log.append(json.loads(entry))
    logging.info(f"Loaded {len(signal_log)} historical signals from Redis")
    
    # Load outcomes
    redis_outcomes = redis_client.lrange("outcome_logs", 0, -1)
    for entry in redis_outcomes:
        outcome_log.append(json.loads(entry))
    logging.info(f"Loaded {len(outcome_log)} historical outcomes from Redis")
except Exception as e:
    logging.error(f"Failed loading logs from Redis: {e}")

# === Symbol and Market Validation ===
async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    key = f"{symbol}_valid"
    if key in cache:
        return
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    market_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
    try:
        async with shared_client as client:
            ref_resp, market_resp = await asyncio.gather(
                client.get(url),
                client.get(market_url)
            )
        if ref_resp.status_code != 200:
            raise HTTPException(status_code=404, detail="Symbol not found on Polygon")
        if not allow_closed:
            market = market_resp.json()
            if market.get("market", {}).get("isOpen") is False:
                raise HTTPException(status_code=400, detail="Market is closed")

        cache[key] = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

# === Option Greeks from Polygon Snapshot ===
async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        resp = await shared_client.get(url)
        data = resp.json()
        options = data.get("results", {}).get("options", [])
        if not options:
            return {}

        # Filter to nearest expiry >= today
        today = datetime.utcnow().date()
        valid = [o for o in options if "details" in o and o["details"].get("expiration_date")]
        valid = [o for o in valid if datetime.strptime(o["details"]["expiration_date"], "%Y-%m-%d").date() >= today]

        # Find nearest ATM call
        underlying_price = data.get("results", {}).get("underlying_asset", {}).get("last", {}).get("price", 0)
        valid = sorted(valid, key=lambda o: abs(o["details"]["strike_price"] - underlying_price))

        for opt in valid:
            greeks = opt.get("greeks", {})
            return {
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "iv": greeks.get("iv")
            }

        return {}
    except Exception as e:
        logging.warning(f"Failed to fetch option greeks: {e}")
        return {}

# === Helper: Parse TradingView Alert String ===
def parse_tradingview_message(msg: str) -> Alert:
    try:
        # New format: "CALL Signal: {{ticker}} at {{close}} Strike: {{plot_0}} Expiry: {{plot_1}}"
        pattern = r"(CALL|PUT)\s*Signal:\s*([A-Z]+)\s*at\s*([\d.]+)\s*Strike:\s*([\d.]+)\s*Expiry:\s*([\d-]+|\d{10,})"
        match = re.search(pattern, msg.replace("\n", " ").strip())
        
        if not match:
            raise ValueError("Unrecognized alert message format")
            
        option_type, symbol, price, strike, expiry = match.groups()
        
        # Handle Unix timestamp expiry if provided
        if expiry.isdigit() and len(expiry) >= 10:
            expiry_dt = datetime.utcfromtimestamp(int(expiry) / 1000)
            expiry = expiry_dt.strftime("%Y-%m-%d")
        
        return Alert(
            signal="buy" if option_type.upper() == "CALL" else "sell",
            symbol=symbol.upper(),
            price=float(price),
            strike=int(float(strike)),  # Handle decimal strikes
            expiry=expiry
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse alert: {e}")

# === OpenAI API Call ===
async def call_openai_chat(prompt: str, cache_key: str) -> str:
    if cache_key in gpt_cache:
        return gpt_cache[cache_key]
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.3
        }
        
        async with shared_client as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()
            gpt_cache[cache_key] = reply
            return reply
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "Error: Unable to evaluate signal"

# === Process Alert (Common Logic) ===
async def process_alert(alert: Alert):
    if is_in_cooldown(alert.symbol, alert.signal):
        return {"status": "cooldown"}
    
    try:
        await validate_symbol_and_market(alert.symbol, allow_closed=True)
        greeks = await get_option_greeks(alert.symbol)

        # Pre-GPT scoring
        score = 0
        if alert.indicators:
            if alert.indicators.get("ADX", 0) > 25: score += 1
            if alert.indicators.get("RSI", 0) > 60 and alert.signal == "buy": score += 1
            if alert.indicators.get("RSI", 0) < 40 and alert.signal == "sell": score += 1
        
        # Minimum score requirement
        if score < 1:
            return {"status": "filtered", "reason": "low local score"}

        cache_key = f"gpt_{alert.symbol}_{alert.signal}_{alert.strike}_{alert.expiry}"
        gpt_prompt = f"""
Evaluate this options signal:
Symbol: {alert.symbol}
Signal: {alert.signal.upper()}
Price: {alert.price}
Strike: {alert.strike}
Expiry: {alert.expiry}

Option Greeks:
Delta: {greeks.get('delta')}
Gamma: {greeks.get('gamma')}
Theta: {greeks.get('theta')}
IV: {greeks.get('iv')}

Respond with:
- Trade decision (Yes/No)
- Confidence score (0-100)
- Reason (1 sentence)
"""
        gpt_reply = await call_openai_chat(gpt_prompt, cache_key)

        gpt_decision = "skip"
        confidence = 0
        if "yes" in gpt_reply.lower():
            gpt_decision = "buy"
            match = re.search(r"confidence[:=]?(\s*)(\d+)", gpt_reply.lower())
            if match: confidence = int(match.group(2))
        
        if gpt_decision != "buy" or confidence < 70:
            return {"status": "filtered", "reason": f"decision={gpt_decision}, confidence={confidence}"}

        # Format Telegram message
        option_type = "CALL" if alert.signal.lower() == "buy" else "PUT"
        option_info = f"\n🎯 {option_type} ${alert.strike} Exp: {alert.expiry}" if alert.strike else ""
        
        tg_msg = (
            f"📈 *{alert.signal.upper()} ALERT* for `{alert.symbol}` @ `${alert.price}`"
            f"{option_info}\n\n"
            f"📊 GPT Review:\n{gpt_reply}"
        )
        
        await shared_client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": tg_msg,
                "parse_mode": "Markdown"
            }
        )

        update_cooldown(alert.symbol, alert.signal)
        log_signal(alert.symbol, alert.signal, gpt_reply, alert.strike, alert.expiry)

        return {"status": "ok", "gpt_review": gpt_reply}

    except Exception as e:
        logging.exception("Alert processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Outcome Logging Endpoint ===
@app.post("/log_outcome")
async def log_outcome(outcome: Outcome):
    # Calculate profit metrics
    outcome.profit = outcome.exit_price - outcome.entry_price
    outcome.profit_pct = (outcome.profit / outcome.entry_price) * 100
    
    # Prepare log entry
    log_entry = outcome.dict()
    log_entry["timestamp"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    
    # Store in Redis and in-memory log
    outcome_log.append(log_entry)
    try:
        redis_client.rpush("outcome_logs", json.dumps(log_entry))
    except Exception as e:
        logging.warning(f"Redis outcome logging failed: {e}")
    
    return {"status": "logged"}

# === Filtered Signal Retrieval ===
@app.get("/signals")
async def get_filtered_signals(
    symbol: Optional[str] = None, 
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None,
    limit: int = 50
) -> List[Dict]:
    filtered = signal_log
    
    if symbol:
        filtered = [s for s in filtered if s["symbol"] == symbol.upper()]
    
    if min_confidence is not None:
        filtered = [s for s in filtered if s.get("confidence", 0) >= min_confidence]
    
    if max_confidence is not None:
        filtered = [s for s in filtered if s.get("confidence", 0) <= max_confidence]
    
    return filtered[-limit:]

# === Filtered Outcome Retrieval ===
@app.get("/outcomes")
async def get_filtered_outcomes(
    symbol: Optional[str] = None,
    min_profit_pct: Optional[float] = None,
    max_profit_pct: Optional[float] = None,
    outcome_type: Optional[str] = None,
    limit: int = 50
) -> List[Dict]:
    filtered = outcome_log
    
    if symbol:
        filtered = [o for o in filtered if o["symbol"] == symbol.upper()]
    
    if min_profit_pct is not None:
        filtered = [o for o in filtered if o.get("profit_pct", 0) >= min_profit_pct]
    
    if max_profit_pct is not None:
        filtered = [o for o in filtered if o.get("profit_pct", 0) <= max_profit_pct]
    
    if outcome_type:
        filtered = [o for o in filtered if o.get("outcome", "").lower() == outcome_type.lower()]
    
    return filtered[-limit:]

# === Performance Statistics ===
@app.get("/stats")
async def get_performance_stats():
    if not outcome_log:
        return {"message": "No outcomes logged yet"}
    
    # Calculate performance metrics
    wins = [o for o in outcome_log if o["outcome"] == "win"]
    losses = [o for o in outcome_log if o["outcome"] == "loss"]
    
    win_rate = len(wins) / len(outcome_log) * 100
    avg_profit = sum(o["profit"] for o in outcome_log) / len(outcome_log)
    avg_win = sum(o["profit"] for o in wins) / len(wins) if wins else 0
    avg_loss = sum(o["profit"] for o in losses) / len(losses) if losses else 0
    
    # Calculate best/worst performers
    symbol_perf = defaultdict(list)
    for o in outcome_log:
        symbol_perf[o["symbol"]].append(o["profit_pct"])
    
    best_symbol = ""
    worst_symbol = ""
    if symbol_perf:
        symbol_avg = {s: sum(p) / len(p) for s, p in symbol_perf.items()}
        best_symbol = max(symbol_avg, key=symbol_avg.get)
        worst_symbol = min(symbol_avg, key=symbol_avg.get)
    
    # Confidence performance correlation
    conf_perf = []
    for o in outcome_log:
        # Find matching signal
        signal = next((s for s in signal_log 
                      if s["symbol"] == o["symbol"] 
                      and s["timestamp"] == o["entry_time"]), None)
        if signal:
            conf_perf.append({
                "confidence": signal.get("confidence", 0),
                "profit_pct": o.get("profit_pct", 0)
            })
    
    avg_conf_profit = sum(cp["profit_pct"] for cp in conf_perf) / len(conf_perf) if conf_perf else 0
    
    return {
        "total_trades": len(outcome_log),
        "win_rate": round(win_rate, 2),
        "avg_profit": round(avg_profit, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "best_symbol": best_symbol,
        "worst_symbol": worst_symbol,
        "confidence_correlation": {
            "trades_with_confidence": len(conf_perf),
            "avg_confidence": round(sum(cp["confidence"] for cp in conf_perf) / len(conf_perf), 1) if conf_perf else 0,
            "avg_profit_at_confidence": round(avg_conf_profit, 2)
        }
    }

# === Daily Summary Task ===
async def send_daily_summary():
    try:
        today = datetime.now(ZoneInfo("America/New_York")).date()
        today_str = today.isoformat()
        
        # Filter today's signals
        today_signals = [s for s in signal_log if s["timestamp"].startswith(today_str)]
        
        # Filter today's outcomes
        today_outcomes = [o for o in outcome_log if o["exit_time"].startswith(today_str)]
        
        # Generate summary
        if not today_signals and not today_outcomes:
            return
        
        summary = f"📊 *Daily Summary* - {today_str}\n\n"
        summary += f"📨 Signals Received: *{len(today_signals)}*\n"
        
        if today_signals:
            signal_counter = Counter(s["symbol"] for s in today_signals)
            top_symbols = ", ".join([f"{sym} ({count})" for sym, count in signal_counter.most_common(3)])
            summary += f"🏆 Top Symbols: {top_symbols}\n"
            
            avg_confidence = sum(s.get("confidence", 0) for s in today_signals) / len(today_signals)
            summary += f"🎯 Avg Confidence: *{avg_confidence:.1f}%*\n"
            
            # Find highest confidence signal
            if today_signals:
                best_signal = max(today_signals, key=lambda x: x.get("confidence", 0))
                summary += f"💎 Best Signal: {best_signal['symbol']} ({best_signal['confidence']}%)\n"
        
        summary += f"\n💼 Trades Executed: *{len(today_outcomes)}*\n"
        
        if today_outcomes:
            wins = [o for o in today_outcomes if o["outcome"] == "win"]
            losses = [o for o in today_outcomes if o["outcome"] == "loss"]
            
            win_rate = len(wins) / len(today_outcomes) * 100 if today_outcomes else 0
            summary += f"✅ Wins: *{len(wins)}* | ❌ Losses: *{len(losses)}*\n"
            summary += f"📈 Win Rate: *{win_rate:.1f}%*\n"
            
            total_profit = sum(o["profit"] for o in today_outcomes)
            summary += f"💰 Net P&L: *${total_profit:.2f}*\n"
            
            best_trade = max(today_outcomes, key=lambda x: x["profit_pct"], default=None)
            if best_trade:
                summary += f"🚀 Best Trade: {best_trade['symbol']} +{best_trade['profit_pct']:.1f}%\n"
        
        await shared_client.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": summary,
                "parse_mode": "Markdown"
            }
        )
        
    except Exception as e:
        logging.error(f"Failed to send daily summary: {e}")

# === Webhook for TradingView Alert ===
@app.post("/webhook/tradingview")
async def handle_tradingview_alert(request: Request):
    try:
        body = await request.body()
        text = body.decode("utf-8")
        alert = parse_tradingview_message(text)
        return await process_alert(alert)
    except Exception as e:
        logging.error(f"Error processing TradingView alert: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# === Webhook for JSON Alert Input ===
@app.post("/webhook")
async def handle_alert(alert: Alert):
    logging.info(f"Received alert: {alert.symbol} @ {alert.price}")
    return await process_alert(alert)

# === Start Scheduler ===
@scheduler.scheduled_job(CronTrigger(hour=16, minute=15, timezone="America/New_York"))
def scheduled_daily_summary():
    asyncio.create_task(send_daily_summary())

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logging.info("Scheduler started - Daily summaries scheduled at 4:15 PM ET")
    logging.info(f"System initialized with {len(signal_log)} signals and {len(outcome_log)} outcomes")

# === Health Check ===
@app.get("/")
async def health_check():
    return {
        "status": "running",
        "signals": len(signal_log),
        "outcomes": len(outcome_log),
        "cooldown_entries": len(cooldown_tracker)
    }
