# alert_server.py
import os
import re
import json
import math
from datetime import datetime, timezone, date
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import httpx

# --- Configuration ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # small, fast default
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# OpenAI SDK (official library)
from openai import OpenAI
oai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI app ---
app = FastAPI(title="TradingView Options Alert Ingestor")

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
            detail="Alert text did not match expected pattern: "
                   '"CALL Signal: TICKER at 123.45 Strike: 123 Expiry: YYYY-MM-DD"',
        )
    side, symbol, underlying_px, strike, expiry = m.groups()
    side = side.upper()
    return {
        "side": side,  # CALL / PUT
        "symbol": symbol.upper(),
        "underlying_price_from_alert": float(underlying_px),
        "strike": float(strike),
        "expiry": expiry,  # YYYY-MM-DD
    }

async def polygon_get_option_ticker(client: httpx.AsyncClient, *, symbol: str, strike: float,
                                    expiry: str, side: str) -> Optional[str]:
    """
    Use Polygon 'All Contracts' search to find exact contract ticker.
    """
    params = {
        "underlying_ticker": symbol,
        "strike_price": strike,
        "expiration_date": expiry,
        "contract_type": "call" if side == "CALL" else "put",
        "limit": 5,
        "apiKey": POLYGON_API_KEY,
    }
    r = await client.get(
        "https://api.polygon.io/v3/reference/options/contracts", params=params, timeout=20.0
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []
    if not results:
        return None
    # If multiple, take the one with exact strike and earliest correction (most common)
    results.sort(key=lambda x: (x.get("correction", 0), x.get("ticker", "")))
    return results[0].get("ticker")  # e.g., "O:GOOGL250821C00205000"

async def polygon_get_option_snapshot(client: httpx.AsyncClient, *, underlying: str, option_ticker: str) -> Dict[str, Any]:
    """
    Fetch Option Contract Snapshot (IV, OI, volume, greeks, etc.)
    """
    # Endpoint path: /v3/snapshot/options/{underlyingAsset}/{optionContract}
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
    """
    Provide a compact, structured briefing for the model.
    """
    lines = [
        f"Alert: {context['side']} {context['symbol']} strike {context['strike']} exp {context['expiry']} "
        f"(~{context['dte']} DTE) at underlying ≈ {context['underlying_price_from_alert']}",
        f"Option ticker: {context['option_ticker']}",
        f"Snapshot:",
        f"  IV: {context['implied_volatility']}",
        f"  Open Interest: {context['open_interest']}",
        f"  Volume (today): {context['volume']}",
    ]
    greeks = context.get("greeks") or {}
    if greeks:
        lines.append(
            "  Greeks: "
            + ", ".join(f"{k}={v}" for k, v in greeks.items() if isinstance(v, (int, float)))
        )
    ua = context.get("underlying_asset") or {}
    if ua:
        upx = ua.get("price")
        if upx:
            lines.append(f"  Underlying last price (snapshot): {upx}")
    return "\n".join(lines)

async def analyze_with_openai(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the model for a clear buy/skip decision with rationale.
    """
    prompt = build_llm_prompt(context)
    system = (
        "You are a disciplined options trading analyst. "
        "Evaluate the single contract with risk awareness (IV level, OI/volume context, DTE). "
        "Consider liquidity (OI & volume), greeks sign & magnitude, and alignment with CALL/PUT. "
        "Return STRICT JSON with fields: "
        '{"decision":"buy|skip","confidence":0..1,"reason":"one or two sentences",'
        '"factors":{"iv":"low|medium|high","oi_ok":true|false,"volume_ok":true|false,"greeks_hint":"text"}} '
        "No extra text."
    )
    try:
        # Using Chat Completions (supported indefinitely)
        resp = oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception as e:
        # Fallback: try Responses API; if still fails, return a minimal skip.
        try:
            resp = oai_client.responses.create(
                model=OPENAI_MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                text_format={"type": "json_object"},
                temperature=0.2,
            )
            return json.loads(resp.output_text)
        except Exception:
            return {
                "decision": "skip",
                "confidence": 0.0,
                "reason": f"LLM call failed: {type(e).__name__}.",
                "factors": {"iv": "unknown", "oi_ok": False, "volume_ok": False, "greeks_hint": "n/a"},
            }

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
            client,
            symbol=alert["symbol"],
            strike=alert["strike"],
            expiry=alert["expiry"],
            side=alert["side"],
        )
        if not option_ticker:
            raise HTTPException(
                status_code=404,
                detail="No matching option contract found on Polygon for the given symbol/strike/expiry.",
            )

        snap = await polygon_get_option_snapshot(
            client,
            underlying=alert["symbol"],
            option_ticker=option_ticker,
        )

    res = snap.get("results") or {}
    # Extract key fields safely
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
        "greeks": {k: greeks.get(k) for k in ("delta", "gamma", "theta", "vega", "rho") if k in greeks},
        "underlying_asset": {"price": safe_get(ua, "price")},
    }

    llm = await analyze_with_openai(context)

    return {
        "ok": True,
        "parsed_alert": alert,
        "option_ticker": option_ticker,
        "metrics": {
            "implied_volatility": iv,
            "open_interest": oi,
            "volume": vol,
            "greeks": context["greeks"],
        },
        "llm_decision": llm,
        "notes": "Educational demo; not financial advice.",
    }

if __name__ == "__main__":
    uvicorn.run("alert_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
