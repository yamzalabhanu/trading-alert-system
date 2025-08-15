# alert_server.py
import os
import re
import json
from datetime import datetime, timezone, date
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import httpx
from openai import OpenAI

# --- Configuration ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # chat id or channel @handle (numeric id preferred)
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")  # optional, for forum topics

if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

oai_client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="TradingView Options Alert Ingestor + Telegram")

ALERT_RE = re.compile(
    r"^\s*(CALL|PUT)\s*Signal:\s*([A-Z][A-Z0-9\.\-]*)\s*at\s*([0-9]*\.?[0-9]+)\s*"
    r"Strike:\s*([0-9]*\.?[0-9]+)\s*Expiry:\s*(\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE,
)

# ---------- Helpers ----------

def parse_alert(text: str) -> Dict[str, Any]:
    m = ALERT_RE.match(text.strip())
    if not m:
        raise HTTPException(
            status_code=400,
            detail='Alert must look like: "CALL Signal: TICKER at 123.45 Strike: 123 Expiry: YYYY-MM-DD"',
        )
    side, symbol, underlying_px, strike, expiry = m.groups()
    return {
        "side": side.upper(),
        "symbol": symbol.upper(),
        "underlying_price_from_alert": float(underlying_px),
        "strike": float(strike),
        "expiry": expiry,
    }

async def polygon_get_option_ticker(client: httpx.AsyncClient, *, symbol: str, strike: float,
                                    expiry: str, side: str) -> Optional[str]:
    params = {
        "underlying_ticker": symbol,
        "strike_price": strike,
        "expiration_date": expiry,
        "contract_type": "call" if side == "CALL" else "put",
        "limit": 5,
        "apiKey": POLYGON_API_KEY,
    }
    r = await client.get("https://api.polygon.io/v3/reference/options/contracts", params=params, timeout=20.0)
    r.raise_for_status()
    results = (r.json() or {}).get("results", []) or []
    if not results:
        return None
    results.sort(key=lambda x: (x.get("correction", 0), x.get("ticker", "")))
    return results[0].get("ticker")

async def polygon_get_option_snapshot(client: httpx.AsyncClient, *, underlying: str, option_ticker: str) -> Dict[str, Any]:
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}/{option_ticker}"
    r = await client.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=20.0)
    r.raise_for_status()
    return r.json()

def dte(expiry: str) -> int:
    y, m, d = map(int, expiry.split("-"))
    return (date(y, m, d) - datetime.now(timezone.utc).date()).days

def safe_get(d: Dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if cur is None:
            return default
        cur = cur.get(key)
    return cur if cur is not None else default

def build_llm_prompt(context: Dict[str, Any]) -> str:
    lines = [
        f"Alert: {context['side']} {context['symbol']} strike {context['strike']} exp {context['expiry']} (~{context['dte']} DTE) at underlying ≈ {context['underlying_price_from_alert']}",
        f"Option ticker: {context['option_ticker']}",
        "Snapshot:",
        f"  IV: {context['implied_volatility']}",
        f"  OI: {context['open_interest']}",
        f"  Volume (today): {context['volume']}",
    ]
    greeks = context.get("greeks") or {}
    if greeks:
        lines.append("  Greeks: " + ", ".join(f"{k}={v}" for k, v in greeks.items() if v is not None))
    ua = context.get("underlying_asset") or {}
    upx = ua.get("price")
    if upx:
        lines.append(f"  Underlying last (snapshot): {upx}")
    return "\n".join(lines)

async def analyze_with_openai(context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_llm_prompt(context)
    system = (
        "You are a disciplined options trading analyst. Evaluate IV level, liquidity (OI/volume), DTE, and greeks. "
        "Return STRICT JSON: "
        '{"decision":"buy|skip","confidence":0..1,"reason":"one or two sentences",'
        '"factors":{"iv":"low|medium|high","oi_ok":true|false,"volume_ok":true|false,"greeks_hint":"text"}}'
    )
    try:
        resp = oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {
            "decision": "skip",
            "confidence": 0.0,
            "reason": f"LLM call failed: {type(e).__name__}.",
            "factors": {"iv": "unknown", "oi_ok": False, "volume_ok": False, "greeks_hint": "n/a"},
        }

def _fmt(val):
    return "—" if val is None else str(val)

def compose_telegram_text(alert: Dict[str, Any], option_ticker: str, metrics: Dict[str, Any], llm: Dict[str, Any]) -> str:
    # Plain text keeps things simple across chat types
    g = metrics.get("greeks", {})
    lines = [
        f"📣 Options Alert",
        f"{alert['side']} {alert['symbol']} | Strike {alert['strike']} | Exp {alert['expiry']} (~{dte(alert['expiry'])} DTE)",
        f"Underlying (alert): {alert['underlying_price_from_alert']}",
        f"Contract: {option_ticker}",
        "",
        "Snapshot:",
        f"  IV: {_fmt(metrics.get('implied_volatility'))}",
        f"  OI: {_fmt(metrics.get('open_interest'))}",
        f"  Vol: {_fmt(metrics.get('volume'))}",
        f"  Greeks: Δ={_fmt(g.get('delta'))}, Γ={_fmt(g.get('gamma'))}, Θ={_fmt(g.get('theta'))}, Vega={_fmt(g.get('vega'))}",
        "",
        f"LLM Decision: {llm.get('decision', 'skip').upper()}  (confidence: {llm.get('confidence', 0):.2f})",
        f"Reason: {llm.get('reason', '')}",
        f"Factors: {json.dumps(llm.get('factors', {}))}",
        "",
        "⚠️ Educational demo; not financial advice."
    ]
    return "\n".join(lines)

async def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    """Send a message to Telegram if token/chat are configured. Returns API response JSON or None."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if TELEGRAM_THREAD_ID:
        payload["message_thread_id"] = int(TELEGRAM_THREAD_ID)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        # Do not raise on status; just return what Telegram said for troubleshooting
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}

# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@app.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(payload: str = Body(..., media_type="text/plain")):
    """
    Accepts raw TradingView alert text like:
    'CALL Signal: GOOGL at 206.13 Strike: 205 Expiry: 2025-08-21'
    """
    alert = parse_alert(payload)

    async with httpx.AsyncClient(http2=True, timeout=20.0) as client:
        option_ticker = await polygon_get_option_ticker(
            client, symbol=alert["symbol"], strike=alert["strike"], expiry=alert["expiry"], side=alert["side"]
        )
        if not option_ticker:
            raise HTTPException(status_code=404, detail="No matching option contract found on Polygon.")

        snap = await polygon_get_option_snapshot(client, underlying=alert["symbol"], option_ticker=option_ticker)

    res = snap.get("results") or {}
    iv = res.get("implied_volatility")
    oi = res.get("open_interest")
    day = res.get("day") or {}
    vol = day.get("volume")
    greeks = res.get("greeks") or {}
    ua = res.get("underlying_asset") or {}

    context = {
        **alert,
        "option_ticker": option_ticker,
        "dte": dte(alert["expiry"]),
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": vol,
        "greeks": {k: greeks.get(k) for k in ("delta", "gamma", "theta", "vega", "rho")},
        "underlying_asset": {"price": safe_get(ua, "price")},
    }

    llm = await analyze_with_openai(context)

    metrics = {
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": vol,
        "greeks": context["greeks"],
    }

    # --- Telegram push (if configured) ---
    tg_text = compose_telegram_text(alert, option_ticker, metrics, llm)
    tg_result = await send_telegram(tg_text)

    return {
        "ok": True,
        "parsed_alert": alert,
        "option_ticker": option_ticker,
        "metrics": metrics,
        "llm_decision": llm,
        "telegram": {"sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False, "result": tg_result},
        "notes": "Educational demo; not financial advice.",
    }

if __name__ == "__main__":
    uvicorn.run("alert_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
