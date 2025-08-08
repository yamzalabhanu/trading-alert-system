# main.py - Enhanced Trading System with Debug Logging
import os
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter, defaultdict, deque
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
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# === Configure Advanced Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_system.log")
    ]
)
logger = logging.getLogger(__name__)

# === Load Config ===
load_dotenv()
app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOG_PERSISTENCE_FILE = os.getenv("LOG_PERSISTENCE_FILE", "trading_logs.json")

# === Validate Critical Environment Variables ===
MISSING_ENV = []
if not TELEGRAM_TOKEN: MISSING_ENV.append("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_CHAT_ID: MISSING_ENV.append("TELEGRAM_CHAT_ID")
if not POLYGON_API_KEY: MISSING_ENV.append("POLYGON_API_KEY")
if not OPENAI_API_KEY: MISSING_ENV.append("OPENAI_API_KEY")

if MISSING_ENV:
    logger.critical(f"CRITICAL: Missing environment variables: {', '.join(MISSING_ENV)}")
    # We'll continue to let the app start for debugging purposes
    # but will disable Telegram functionality

# === Constants ===
MAX_LOG_ENTRIES = 1000
RENDER_PORT = int(os.getenv("PORT", 8000))

# === Caching ===
cache: TTLCache = TTLCache(maxsize=200, ttl=300)
gpt_cache: TTLCache = TTLCache(maxsize=500, ttl=60)

# === Cooldown + Logs ===
cooldown_tracker: Dict[Tuple[str, str], datetime] = {}
COOLDOWN_WINDOW = timedelta(minutes=10)
signal_log = deque(maxlen=MAX_LOG_ENTRIES)
outcome_log = deque(maxlen=MAX_LOG_ENTRIES)

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

# === Telegram Helper ===
async def send_telegram_message(message: str, disable_preview=True) -> bool:
    """Send message to Telegram with comprehensive error handling"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials missing - message not sent")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": disable_preview
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            # Check Telegram API response
            resp_data = response.json()
            if not resp_data.get("ok"):
                logger.error(f"Telegram API error: {resp_data.get('description')}")
                return False
                
            logger.info(f"Telegram message sent successfully to chat {TELEGRAM_CHAT_ID}")
            return True
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Telegram HTTP error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Telegram connection error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected Telegram error")
    
    return False

# === Cooldown Helpers ===
def is_in_cooldown(symbol: str, signal: str) -> bool:
    key = (symbol.upper(), signal.lower())
    last_alert = cooldown_tracker.get(key)
    return last_alert and datetime.utcnow() - last_alert < COOLDOWN_WINDOW

def update_cooldown(symbol: str, signal: str):
    cooldown_tracker[(symbol.upper(), signal.lower())] = datetime.utcnow()

# === Enhanced Signal Logging ===
def log_signal(symbol: str, signal: str, gpt_reply: str, strike: int | None = None, expiry: str | None = None):
    confidence = 0
    match = re.search(r"confidence[:=]?\s*(\d+)", gpt_reply.lower())
    if match:
        confidence = int(match.group(1))
    
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
    return entry

# === Initialize Logs from File ===
def load_logs_from_file():
    global signal_log, outcome_log
    
    try:
        if os.path.exists(LOG_PERSISTENCE_FILE):
            with open(LOG_PERSISTENCE_FILE, 'r') as f:
                logs = json.load(f)
                signal_log = deque(logs.get('signals', []), maxlen=MAX_LOG_ENTRIES)
                outcome_log = deque(logs.get('outcomes', []), maxlen=MAX_LOG_ENTRIES)
            logger.info(f"Loaded {len(signal_log)} signals and {len(outcome_log)} outcomes from file")
        else:
            logger.info("No log file found - starting with empty logs")
    except Exception as e:
        logger.error(f"Failed loading logs from file: {str(e)}")
        signal_log = deque(maxlen=MAX_LOG_ENTRIES)
        outcome_log = deque(maxlen=MAX_LOG_ENTRIES)

# === Save Logs to File ===
def save_logs_to_file():
    try:
        logs = {
            'signals': list(signal_log),
            'outcomes': list(outcome_log),
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(LOG_PERSISTENCE_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
        logger.info(f"Saved {len(signal_log)} signals and {len(outcome_log)} outcomes to file")
    except Exception as e:
        logger.error(f"Failed to save logs: {str(e)}")

# === Symbol and Market Validation ===
async def validate_symbol_and_market(symbol: str, allow_closed: bool = False):
    key = f"{symbol}_valid"
    if key in cache:
        return
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
    market_url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={POLYGON_API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            ref_resp, market_resp = await asyncio.gather(
                client.get(url),
                client.get(market_url)
            )
            ref_resp.raise_for_status()
            market_resp.raise_for_status()

            ref_data = ref_resp.json()
            market_data = market_resp.json()

            if not isinstance(ref_data, dict) or not ref_data.get("results"):
                logger.warning(f"Symbol not found: {symbol}")
                raise HTTPException(status_code=404, detail="Symbol not found on Polygon")
            
            if not allow_closed:
                if not isinstance(market_data, dict) or not market_data.get("market", {}).get("isOpen", True):
                    logger.info(f"Market closed - rejecting {symbol}")
                    raise HTTPException(status_code=400, detail="Market is closed")

        cache[key] = True
    except httpx.HTTPStatusError as e:
        logger.error(f"Polygon API error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Polygon API error: {str(e)}")
    except Exception as e:
        logger.exception("Symbol validation failed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# === Option Greeks from Polygon Snapshot ===
async def get_option_greeks(symbol: str) -> Dict[str, Any]:
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper()}?apiKey={POLYGON_API_KEY}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            
            if not data.get("results") or not isinstance(data["results"], dict):
                logger.warning(f"No results for {symbol} options")
                return {}
                
            options = data["results"].get("options", [])
            if not options:
                logger.info(f"No options data for {symbol}")
                return {}

            today = datetime.utcnow().date()
            valid_options = []
            for o in options:
                if not isinstance(o, dict):
                    continue
                details = o.get("details", {})
                if not details or not isinstance(details, dict):
                    continue
                expiry_date = details.get("expiration_date")
                if not expiry_date:
                    continue
                    
                try:
                    expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    if expiry_dt >= today:
                        valid_options.append(o)
                except ValueError:
                    logger.warning(f"Invalid expiry date: {expiry_date} for {symbol}")
                    continue

            if not valid_options:
                logger.info(f"No valid options contracts for {symbol}")
                return {}

            underlying_asset = data["results"].get("underlying_asset", {})
            if isinstance(underlying_asset, dict):
                last_data = underlying_asset.get("last", {})
                if isinstance(last_data, dict):
                    underlying_price = last_data.get("price", 0)
                else:
                    underlying_price = 0
            else:
                underlying_price = 0

            def get_abs_diff(option):
                details = option.get("details", {})
                strike_price = details.get("strike_price", 0)
                return abs(strike_price - underlying_price)
                
            valid_options.sort(key=get_abs_diff)
            
            for opt in valid_options:
                greeks = opt.get("greeks", {})
                if isinstance(greeks, dict):
                    logger.info(f"Found greeks for {symbol}: {greeks}")
                    return {
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "iv": greeks.get("iv")
                    }

        return {}
    except httpx.HTTPStatusError as e:
        logger.error(f"Polygon API error: {e.response.status_code} - {e.response.text}")
        return {}
    except Exception as e:
        logger.exception(f"Failed to fetch option greeks for {symbol}")
        return {}

# === Helper: Parse TradingView Alert String ===
def parse_tradingview_message(msg: str) -> Alert:
    try:
        pattern = r"(CALL|PUT)\s*Signal:\s*([A-Z]+)\s*at\s*([\d.]+)\s*Strike:\s*([\d.]+)\s*Expiry:\s*([\d-]+|\d{10,})"
        match = re.search(pattern, msg.replace("\n", " ").strip())
        
        if not match:
            raise ValueError("Unrecognized alert message format")
            
        option_type, symbol, price, strike, expiry = match.groups()
        
        if expiry.isdigit() and len(expiry) >= 10:
            expiry_dt = datetime.utcfromtimestamp(int(expiry) / 1000)
            expiry = expiry_dt.strftime("%Y-%m-%d")
        
        return Alert(
            signal="buy" if option_type.upper() == "CALL" else "sell",
            symbol=symbol.upper(),
            price=float(price),
            strike=int(float(strike)),
            expiry=expiry
        )
    except Exception as e:
        logger.error(f"Alert parsing failed: {msg} - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse alert: {str(e)}")

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
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=15.0
            )
            response.raise_for_status()
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()
            gpt_cache[cache_key] = reply
            return reply
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
        return "Error: Unable to evaluate signal (API error)"
    except Exception as e:
        logger.exception("OpenAI request failed")
        return "Error: Unable to evaluate signal"

# === Process Alert (Common Logic) ===
async def process_alert(alert: Alert):
    if is_in_cooldown(alert.symbol, alert.signal):
        logger.info(f"Cooldown active for {alert.symbol} {alert.signal}")
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
        
        if score < 1:
            logger.info(f"Signal filtered for {alert.symbol}: low score {score}")
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
            logger.info(f"GPT rejected {alert.symbol}: decision={gpt_decision}, confidence={confidence}")
            return {"status": "filtered", "reason": f"decision={gpt_decision}, confidence={confidence}"}

        # Format Telegram message
        option_type = "CALL" if alert.signal.lower() == "buy" else "PUT"
        option_info = f"\n🎯 {option_type} ${alert.strike} Exp: {alert.expiry}" if alert.strike else ""
        
        tg_msg = (
            f"📈 *{alert.signal.upper()} ALERT* for `{alert.symbol}` @ `${alert.price}`"
            f"{option_info}\n\n"
            f"📊 GPT Review:\n{gpt_reply}"
        )
        
        # Send Telegram message
        success = await send_telegram_message(tg_msg)
        if not success:
            logger.error(f"Failed to send Telegram alert for {alert.symbol}")

        update_cooldown(alert.symbol, alert.signal)
        log_signal(alert.symbol, alert.signal, gpt_reply, alert.strike, alert.expiry)

        return {"status": "ok", "gpt_review": gpt_reply}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Alert processing failed")
        raise HTTPException(status_code=500, detail=str(e))

# === Improved Outcome Logging Endpoint ===
@app.post("/log_outcome")
async def log_outcome(outcome: Outcome):
    try:
        outcome.profit = outcome.exit_price - outcome.entry_price
        outcome.profit_pct = (outcome.profit / outcome.entry_price) * 100 if outcome.entry_price != 0 else 0
        
        log_entry = outcome.dict()
        log_entry["timestamp"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
        outcome_log.append(log_entry)
        
        logger.info(f"Logged outcome for {outcome.symbol}: {outcome.outcome} ({outcome.profit_pct:.2f}%)")
        return {"status": "logged", "data": log_entry}
    except Exception as e:
        logger.exception("Outcome logging failed")
        raise HTTPException(status_code=500, detail="Outcome logging error")

# === Filtered Signal Retrieval ===
@app.get("/signals")
async def get_filtered_signals(
    symbol: Optional[str] = None, 
    min_confidence: Optional[int] = None,
    max_confidence: Optional[int] = None,
    limit: int = 50
) -> List[Dict]:
    filtered = list(signal_log)
    
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
    filtered = list(outcome_log)
    
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
    
    outcomes = list(outcome_log)
    wins = [o for o in outcomes if o["outcome"] == "win"]
    losses = [o for o in outcomes if o["outcome"] == "loss"]
    
    win_rate = len(wins) / len(outcomes) * 100
    avg_profit = sum(o["profit"] for o in outcomes) / len(outcomes)
    avg_win = sum(o["profit"] for o in wins) / len(wins) if wins else 0
    avg_loss = sum(o["profit"] for o in losses) / len(losses) if losses else 0
    
    symbol_perf = defaultdict(list)
    for o in outcomes:
        symbol_perf[o["symbol"]].append(o["profit_pct"])
    
    best_symbol = ""
    worst_symbol = ""
    if symbol_perf:
        symbol_avg = {s: sum(p) / len(p) for s, p in symbol_perf.items()}
        best_symbol = max(symbol_avg, key=symbol_avg.get)
        worst_symbol = min(symbol_avg, key=symbol_avg.get)
    
    conf_perf = []
    signals = list(signal_log)
    for o in outcomes:
        signal = next((s for s in signals 
                      if s["symbol"] == o["symbol"] 
                      and s["timestamp"] == o["entry_time"]), None)
        if signal:
            conf_perf.append({
                "confidence": signal.get("confidence", 0),
                "profit_pct": o.get("profit_pct", 0)
            })
    
    avg_conf_profit = sum(cp["profit_pct"] for cp in conf_perf) / len(conf_perf) if conf_perf else 0
    
    return {
        "total_trades": len(outcomes),
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
        
        today_signals = [s for s in signal_log if s["timestamp"].startswith(today_str)]
        today_outcomes = [o for o in outcome_log if o["exit_time"].startswith(today_str)]
        
        if not today_signals and not today_outcomes:
            logger.info("No signals or outcomes for daily summary")
            return
        
        summary = f"📊 *Daily Summary* - {today_str}\n\n"
        summary += f"📨 Signals Received: *{len(today_signals)}*\n"
        
        if today_signals:
            signal_counter = Counter(s["symbol"] for s in today_signals)
            top_symbols = ", ".join([f"{sym} ({count})" for sym, count in signal_counter.most_common(3)])
            summary += f"🏆 Top Symbols: {top_symbols}\n"
            
            avg_confidence = sum(s.get("confidence", 0) for s in today_signals) / len(today_signals)
            summary += f"🎯 Avg Confidence: *{avg_confidence:.1f}%*\n"
            
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
        
        await send_telegram_message(summary)
        logger.info("Daily summary sent")
        
    except Exception as e:
        logger.exception("Failed to send daily summary")

# === Webhook for TradingView Alert ===
@app.post("/webhook/tradingview")
async def handle_tradingview_alert(request: Request):
    try:
        body = await request.body()
        text = body.decode("utf-8")
        logger.info(f"Received TradingView alert: {text}")
        alert = parse_tradingview_message(text)
        return await process_alert(alert)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("TradingView alert processing failed")
        raise HTTPException(status_code=400, detail=str(e))

# === Webhook for JSON Alert Input ===
@app.post("/webhook")
async def handle_alert(alert: Alert):
    logger.info(f"Received JSON alert: {alert.symbol} @ {alert.price}")
    return await process_alert(alert)

# === Telegram Test Endpoint ===
@app.get("/test-telegram")
async def test_telegram():
    """Test Telegram connectivity"""
    test_msg = (
        "🚀 *SYSTEM TEST* - Telegram is working!\n"
        f"• Time: `{datetime.now().isoformat()}`\n"
        f"• Environment: `{os.getenv('ENVIRONMENT', 'development')}`\n"
        "✅ All systems operational"
    )
    
    success = await send_telegram_message(test_msg)
    return {
        "status": "success" if success else "failed",
        "message": test_msg,
        "credentials_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    }

# === Verify Telegram Credentials ===
async def verify_telegram_credentials():
    """Check Telegram credentials at startup"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram credentials not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get("ok"):
                logger.info(f"Telegram connected: @{data['result']['username']}")
                return True
            else:
                logger.error(f"Telegram credential check failed: {data.get('description')}")
                return False
    except Exception as e:
        logger.exception("Telegram credential verification failed")
        return False

# === Health Check ===
@app.get("/")
async def health_check():
    return {
        "status": "running",
        "signals": len(signal_log),
        "outcomes": len(outcome_log),
        "telegram_configured": bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "log_retention": f"{len(signal_log)}/{MAX_LOG_ENTRIES} signals, {len(outcome_log)}/{MAX_LOG_ENTRIES} outcomes"
    }

# === Startup Event ===
@app.on_event("startup")
async def startup_event():
    load_logs_from_file()
    
    # Verify Telegram credentials
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        await verify_telegram_credentials()
    else:
        logger.error("Telegram credentials missing - alerts will not be sent")
    
    # Start scheduler
    scheduler.start()
    
    # Schedule periodic log saving
    scheduler.add_job(
        save_logs_to_file,
        'interval',
        minutes=15,
        timezone=ZoneInfo("America/New_York")
    )
    
    logger.info("Scheduler started - Daily summaries scheduled at 4:15 PM ET")
    logger.info(f"Server running on port {RENDER_PORT}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"System initialized with {len(signal_log)} signals and {len(outcome_log)} outcomes")

# === Shutdown Event ===
@app.on_event("shutdown")
async def shutdown_event():
    save_logs_to_file()
    logger.info("Application shutting down - logs saved")

# === Run Server for Render ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=RENDER_PORT,
        reload=False,
        workers=1,
        timeout_keep_alive=60,
        log_config=None
    )
