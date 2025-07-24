import os
import httpx
import asyncio
import logging
import re
from datetime import datetime, time, timedelta
from typing import List
from fastapi import FastAPI
from dotenv import load_dotenv

try:
    import ssl
except ImportError:
    ssl = None
    logging.warning("SSL module is not available. Secure HTTP may fail in this environment.")

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()
logging.basicConfig(level=logging.INFO)
BASE_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

MIN_OI = 1000
MIN_VOLUME = 500
IV_PERCENTILE_THRESHOLD = 0.75
ATM_PROXIMITY_PCT = 0.02  # Within 2% of price


def escape_markdown(text: str) -> str:
    return re.sub(r'([_\*\[\]()~`>#+\-=|{}.!])', r'\\\1', text)


async def send_telegram(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        if len(message) > 4000:
            message = message[:3990] + "\n...truncated"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": escape_markdown(message),
                    "parse_mode": "MarkdownV2"
                })
        except Exception as e:
            logging.error(f"Telegram send error: {e}")


async def get_polygon(endpoint: str, params: dict) -> dict:
    params["apiKey"] = POLYGON_API_KEY
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"https://api.polygon.io{endpoint}", params=params)
            return res.json()
    except Exception as e:
        logging.warning(f"Polygon API error: {endpoint} | {e}")
        return {}


async def gpt_summary(prompt: str) -> str:
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post("https://api.openai.com/v1/chat/completions", headers=BASE_HEADERS, json=payload)
            if res.status_code != 200:
                logging.error(f"GPT API error {res.status_code}: {res.text}")
            data = res.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "GPT response incomplete.").strip()
    except Exception as e:
        logging.error(f"GPT summary error: {e}")
        return "GPT analysis unavailable."


@app.get("/scan")
async def run_all_scans():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    results = [f"🕒 *Scan Time:* `{now}`\n"]

    tickers = [
        "AAPL", "TSLA", "AMZN", "GOOG", "META", "CRCL", "PLTR"
    ]

    results.append(f"📈 *Watchlist Symbols*: {', '.join(tickers)}")

    option_flow = []
    today = datetime.utcnow().date()
    week_ahead = today + timedelta(days=7)

    for symbol in tickers:
        opt = await get_polygon("/v3/reference/options/contracts", {"underlying_ticker": symbol, "limit": 100})
        calls = []
        puts = []
        for o in opt.get("results", []):
            expiry = o.get("expiration_date")
            if not expiry:
                continue
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            if expiry_date > week_ahead:
                continue

            contract = o.get("ticker")
            strike = o.get("strike_price")
            iv = o.get("implied_volatility", 0)
            oi = o.get("open_interest", 0)
            vol = o.get("volume", 0)
            side = o.get("contract_type")
            price = o.get("underlying_price", 0)
            score = iv * oi if iv and oi else 0
            atm = abs(strike - price) / price < ATM_PROXIMITY_PCT if price else False

            if oi < MIN_OI or vol < MIN_VOLUME:
                continue

            entry = {
                "symbol": symbol,
                "contract": contract,
                "side": side,
                "iv": iv,
                "oi": oi,
                "vol": vol,
                "score": score,
                "strike": strike,
                "expiry": expiry,
                "atm": atm
            }
            if side == "call":
                calls.append(entry)
            elif side == "put":
                puts.append(entry)

        option_flow.extend(sorted(calls, key=lambda x: x["score"], reverse=True)[:2])
        option_flow.extend(sorted(puts, key=lambda x: x["score"], reverse=True)[:2])

    detailed_lines = [
        f"{x['symbol']} | {x['contract']} | {x['side']} | Exp: {x['expiry']} | Strike: {x['strike']} | IVxOI: {x['score']:.0f} | Vol: {x['vol']} | OI: {x['oi']}"
        for x in option_flow
    ]
    results.append("🧾 *Top Weekly Options Activity (Top 2 Calls and Puts per Ticker):*\n" + "\n".join(detailed_lines))

    prompt = (
        "You are a professional options trader. Analyze the following list of weekly options contracts for unusual activity. "
        "Identify top candidates for BUY or SELL, based on high IV×OI, IV percentile, ATM proximity, strike positioning, and near expiration. "
        "Consider volume/open interest trends and breakout potential. Only include contracts with open interest ≥ 1000 and volume ≥ 500. "
        "Prioritize ATM or slightly ITM contracts. Format the response as clear, actionable bullet points for a trading alert with reasoning.\n\n"
        "Format example:\n"
        "- [SYMBOL] [CALL/PUT] expiring on [DATE] @ [STRIKE] → Buy/Sell. Reason: [e.g., high IV, strong OI, ATM, breakout, volume spike]\n\n"
        "Options Flow Data:\n"
        + "\n".join([
            f"{x['symbol']} {x['side'].upper()} expiring on {x['expiry']} @ {x['strike']} → Buy/Sell. Reason: IVxOI: {x['score']:.0f}, IV: {x['iv']}, Vol: {x['vol']}, OI: {x['oi']}, ATM: {x['atm']}"
            for x in option_flow
        ])
    )

    gpt_msg = await gpt_summary(prompt)
    combined_message = f"{chr(10).join(results)}\n\n📊 *LLM Recommendation:*\n{gpt_msg}"
    await send_telegram(combined_message)

    return {"status": "complete", "summary": gpt_msg}


@app.on_event("startup")
async def start_background_loop():
    async def loop():
        eastern = ZoneInfo("America/New_York")
        market_open = time(9, 30)
        market_close = time(16, 0)

        while True:
            now = datetime.now(tz=eastern)
            if now.weekday() < 5 and market_open <= now.time() <= market_close:
                logging.info(f"Market open — running scan at {now}")
                try:
                    await run_all_scans()
                except Exception as e:
                    logging.error(f"Error in scan loop: {e}")
            else:
                logging.info(f"Outside market hours ({now.time()}) — skipping scan.")

            await asyncio.sleep(300)

    asyncio.create_task(loop())
