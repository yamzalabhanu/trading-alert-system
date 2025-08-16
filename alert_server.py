# alert_server.py
import os
import re
import json
from datetime import datetime, timezone, date, timedelta, time as dt_time
from typing import Any, Dict, Optional, List, Tuple
from zoneinfo import ZoneInfo
import asyncio
import collections

import uvicorn
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from openai import OpenAI

# --- Configuration ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM budget + prefilter thresholds (configurable)
MAX_LLM_PER_DAY = int(os.getenv("MAX_LLM_PER_DAY", "50"))
VOLUME_MIN_FOR_LLM = int(os.getenv("VOLUME_MIN_FOR_LLM", "5000"))
OI_MIN_FOR_LLM = int(os.getenv("OI_MIN_FOR_LLM", "10000"))

# NEW: cooldown + daily report schedule
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "600"))  # 10 minutes default
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")
REPORT_HHMM = os.getenv("REPORT_HHMM", "16:15")  # 4:15 PM local market tz

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")

if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

oai_client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="TradingView Options Alert Ingestor + Telegram (Same-Week Expiry)")

# ---------- Alert parsing patterns ----------

# Accepts with expiry:
#   CALL Signal: GOOGL at 206.13 Strike: 205 Expiry: 2025-08-21
# Or without expiry (we'll compute same-week Friday):
#   CALL Signal: GOOGL at 206.13 Strike: 205
ALERT_RE_WITH_EXP = re.compile(
    r"^\s*(CALL|PUT)\s*Signal:\s*([A-Z][A-Z0-9\.\-]*)\s*at\s*([0-9]*\.?[0-9]+)\s*"
    r"Strike:\s*([0-9]*\.?[0-9]+)\s*Expiry:\s*(\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE,
)
ALERT_RE_NO_EXP = re.compile(
    r"^\s*(CALL|PUT)\s*Signal:\s*([A-Z][A-Z0-9\.\-]*)\s*at\s*([0-9]*\.?[0-9]+)\s*"
    r"Strike:\s*([0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE,
)

# ---------- LLM daily quota (UTC) ----------

# In-memory daily counter: {"date": "YYYY-MM-DD", "used": int}
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}

def _utc_date_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def _maybe_reset_quota() -> None:
    today = _utc_date_str()
    if _llm_quota["date"] != today:
        _llm_quota["date"] = today
        _llm_quota["used"] = 0

def llm_quota_snapshot() -> Dict[str, Any]:
    _maybe_reset_quota()
    used = int(_llm_quota["used"])
    return {
        "date_utc": _llm_quota["date"],
        "used": used,
        "max": MAX_LLM_PER_DAY,
        "remaining": max(0, MAX_LLM_PER_DAY - used),
    }

def can_consume_llm() -> bool:
    snap = llm_quota_snapshot()
    return snap["used"] < snap["max"]

def consume_llm() -> None:
    _maybe_reset_quota()
    _llm_quota["used"] += 1

# ---------- Cooldown state & daily log (market TZ) ----------

_COOLDOWN: Dict[Tuple[str, str], datetime] = {}  # key: (symbol, side) -> last_processed_utc
_DECISIONS_LOG: List[Dict[str, Any]] = []        # per-alert log for daily report

def market_now() -> datetime:
    return datetime.now(ZoneInfo(MARKET_TZ))

def same_week_friday(today: date) -> date:
    # Monday=0 ... Friday=4
    offset = 4 - today.weekday()
    if offset < 0:  # Sat/Sun -> next Friday
        offset += 7
    return today + timedelta(days=offset)

def round_strike_to_common_increment(strike: float) -> float:
    # Simple, practical rounding: nearest 0.5
    return round(strike * 2) / 2.0

def parse_alert_text(text: str) -> Dict[str, Any]:
    s = text.strip()
    m = ALERT_RE_WITH_EXP.match(s)
    if m:
        side, symbol, underlying_px, strike, expiry = m.groups()
        return {
            "side": side.upper(),
            "symbol": symbol.upper(),
            "underlying_price_from_alert": float(underlying_px),
            "strike": float(strike),
            "expiry": expiry,
            "expiry_source": "alert",
        }
    m = ALERT_RE_NO_EXP.match(s)
    if m:
        side, symbol, underlying_px, strike = m.groups()
        # compute same-week Friday (server local date in UTC for determinism)
        swf = same_week_friday(datetime.now(timezone.utc).date()).isoformat()
        return {
            "side": side.upper(),
            "symbol": symbol.upper(),
            "underlying_price_from_alert": float(underlying_px),
            "strike": float(strike),
            "expiry": swf,
            "expiry_source": "computed_same_week_friday",
        }
    raise HTTPException(
        status_code=400,
        detail='Alert must be like: "CALL Signal: TICKER at 123.45 Strike: 123" '
               'or with expiry: "... Expiry: YYYY-MM-DD"',
    )

async def polygon_list_contracts_for_expiry(
    client: httpx.AsyncClient, *, symbol: str, expiry: str, side: str, limit: int = 250
) -> List[Dict[str, Any]]:
    params = {
        "underlying_ticker": symbol,
        "expiration_date": expiry,
        "contract_type": "call" if side == "CALL" else "put",
        "limit": limit,
        "apiKey": POLYGON_API_KEY,
    }
    r = await client.get("https://api.polygon.io/v3/reference/options/contracts", params=params, timeout=20.0)
    r.raise_for_status()
    return (r.json() or {}).get("results", []) or []

def pick_nearest_strike(contracts: List[Dict[str, Any]], desired_strike: float) -> Optional[Dict[str, Any]]:
    best = None
    best_diff = float("inf")
    for c in contracts:
        sp = c.get("strike_price")
        if sp is None:
            continue
        diff = abs(float(sp) - desired_strike)
        if diff < best_diff:
            best_diff = diff
            best = c
    return best

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
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default

def build_llm_prompt(context: Dict[str, Any]) -> str:
    lines = [
        f"Alert: {context['side']} {context['symbol']} strike {context['strike']} "
        f"exp {context['expiry']} (~{context['dte']} DTE) at underlying ≈ {context['underlying_price_from_alert']}",
        f"Option ticker: {context['option_ticker']}",
        "Snapshot:",
        f"  IV: {context['implied_volatility']}",
        f"  Open Interest: {context['open_interest']}",
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
    system = (
        "You are a disciplined options trading analyst. Evaluate IV level, liquidity (OI/volume), DTE, and greeks.\n"
        'Return STRICT JSON: {"decision":"buy|skip","confidence":0..1,"reason":"one or two sentences",'
        '"factors":{"iv":"low|medium|high","oi_ok":true|false,"volume_ok":true|false,"greeks_hint":"text"}}'
    )
    prompt = build_llm_prompt(context)
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

def compose_telegram_text(alert: Dict[str, Any], option_ticker: str, metrics: Dict[str, Any],
                          llm: Dict[str, Any], *, llm_ran: bool, llm_reason: str) -> str:
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
    ]
    if llm_ran:
        lines += [
            f"LLM Decision: {llm.get('decision', 'skip').upper()}  (confidence: {llm.get('confidence', 0):.2f})",
            f"Reason: {llm.get('reason', '')}",
            f"Factors: {json.dumps(llm.get('factors', {}))}",
        ]
    else:
        lines += [
            "LLM: Skipped",
            f"Reason: {llm_reason}",
        ]
    lines += [
        "",
        "⚠️ Educational demo; not financial advice."
    ]
    return "\n".join(lines)

async def send_telegram(text: str) -> Optional[Dict[str, Any]]:
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
        try:
            return r.json()
        except Exception:
            return {"status_code": r.status_code, "text": r.text}

# ---------- Daily report helpers ----------

def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

def _next_report_dt_utc(now_utc: datetime) -> datetime:
    """Compute next report datetime in UTC corresponding to REPORT_HHMM in MARKET_TZ."""
    tz = ZoneInfo(MARKET_TZ)
    hh, mm = _parse_hhmm(REPORT_HHMM)
    # Current market-day local time
    now_local = now_utc.astimezone(tz)
    target_local = datetime.combine(now_local.date(), dt_time(hour=hh, minute=mm), tzinfo=tz)
    if target_local <= now_local:
        target_local = target_local + timedelta(days=1)
    return target_local.astimezone(timezone.utc)

def _summarize_day_for_report(local_date: date) -> str:
    # Filter log entries for this local_date in market TZ
    entries = [e for e in _DECISIONS_LOG if e["timestamp_local"].date() == local_date]
    total_alerts = len(entries)
    llm_runs = sum(1 for e in entries if e["llm"]["ran"])
    llm_skips = total_alerts - llm_runs
    buys = sum(1 for e in entries if e["llm"]["decision"] == "buy")
    skips = sum(1 for e in entries if e["llm"]["decision"] != "buy")
    avg_conf = (
        sum(float(e["llm"]["confidence"] or 0.0) for e in entries if e["llm"]["ran"]) / llm_runs
        if llm_runs else 0.0
    )
    by_symbol = collections.Counter((e["symbol"] for e in entries))
    top = ", ".join(f"{sym}({cnt})" for sym, cnt in by_symbol.most_common(5)) or "—"

    quota = llm_quota_snapshot()
    header = f"📊 Daily Report — {local_date.isoformat()} ({MARKET_TZ})"
    body = [
        f"Alerts handled: {total_alerts}",
        f"LLM ran: {llm_runs}, skipped: {llm_skips}",
        f"Decisions — BUY: {buys}, SKIP: {skips}",
        f"Avg confidence (when ran): {avg_conf:.2f}",
        f"Top tickers: {top}",
        "",
        f"Quota used: {quota['used']}/{quota['max']} (remaining {quota['remaining']})",
        "",
        "⚠️ Educational demo; not financial advice."
    ]
    return header + "\n" + "\n".join(body)

async def _send_daily_report_now() -> Dict[str, Any]:
    today_local = market_now().date()
    text = _summarize_day_for_report(today_local)
    tg_result = await send_telegram(text)
    return {"ok": True, "sent": bool(tg_result), "result": tg_result}

async def _daily_report_scheduler():
    """Background task: waits until next scheduled 4:15 PM market time and sends report each day."""
    while True:
        now_utc = datetime.now(timezone.utc)
        next_utc = _next_report_dt_utc(now_utc)
        sleep_s = max(1, int((next_utc - now_utc).total_seconds()))
        try:
            await asyncio.sleep(sleep_s)
            await _send_daily_report_now()
        except asyncio.CancelledError:
            raise
        except Exception:
            # swallow and schedule next
            await asyncio.sleep(1)

# ---------- Lifespan events ----------

@app.on_event("startup")
async def on_startup():
    # Kick off the scheduler in background
    app.state.report_task = asyncio.create_task(_daily_report_scheduler())

@app.on_event("shutdown")
async def on_shutdown():
    task = getattr(app.state, "report_task", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(Exception):
            await task

# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@app.get("/quota")
def quota():
    """Check current LLM quota snapshot (UTC day bucket)."""
    return {"ok": True, "quota": llm_quota_snapshot()}

@app.post("/run/daily_report")
async def run_daily_report():
    """Manual trigger to test the daily report immediately."""
    res = await _send_daily_report_now()
    return {"ok": True, "trigger": "manual", **res}

async def _get_alert_text(request: Request) -> str:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        data = await request.json()
        return data.get("message", "")
    return (await request.body()).decode("utf-8", errors="ignore")

@app.post("/webhook", response_class=JSONResponse)
@app.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(request: Request):
    payload = await _get_alert_text(request)
    alert = parse_alert_text(payload)

    # Round strike to common increment and search by same-week expiry, pick nearest strike if exact missing
    desired_strike = round_strike_to_common_increment(alert["strike"])

    async with httpx.AsyncClient(http2=True, timeout=20.0) as client:
        # Pull all contracts for that expiry and side, then pick closest strike.
        contracts = await polygon_list_contracts_for_expiry(
            client, symbol=alert["symbol"], expiry=alert["expiry"], side=alert["side"], limit=250
        )
        if not contracts:
            raise HTTPException(
                status_code=404,
                detail=f"No contracts found for {alert['symbol']} {alert['side']} exp {alert['expiry']} (same-week Friday).",
            )
        best = pick_nearest_strike(contracts, desired_strike)
        if not best:
            raise HTTPException(
                status_code=404,
                detail=f"No strikes available near {desired_strike} for {alert['symbol']} on {alert['expiry']}.",
            )
        option_ticker = best.get("ticker")

        snap = await polygon_get_option_snapshot(client, underlying=alert["symbol"], option_ticker=option_ticker)

    res = snap.get("results") or {}
    iv = res.get("implied_volatility")
    oi = res.get("open_interest") or 0
    day = res.get("day") or {}
    vol = day.get("volume") or 0
    greeks = res.get("greeks") or {}
    ua = res.get("underlying_asset") or {}

    context = {
        **alert,
        "strike": desired_strike,  # normalized
        "option_ticker": option_ticker,
        "dte": dte(alert["expiry"]),
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": vol,
        "greeks": {k: greeks.get(k) for k in ("delta", "gamma", "theta", "vega", "rho")},
        "underlying_asset": {"price": safe_get(ua, "price")},
    }

    # ---------- Cooldown check ----------
    key = (alert["symbol"], alert["side"])
    now_utc = datetime.now(timezone.utc)
    cooldown_reason = ""
    in_cooldown = False
    last_ts = _COOLDOWN.get(key)
    if last_ts is not None:
        elapsed = (now_utc - last_ts).total_seconds()
        if elapsed < COOLDOWN_SECONDS:
            in_cooldown = True
            cooldown_reason = f"In cooldown: {int(COOLDOWN_SECONDS - elapsed)}s remaining."

    # ---------- Prefilter + daily budget gate for LLM ----------
    llm_should_run = (vol >= VOLUME_MIN_FOR_LLM) or (oi >= OI_MIN_FOR_LLM)
    llm_reason = ""
    llm_ran = False
    llm = {
        "decision": "skip",
        "confidence": 0.0,
        "reason": "Skipped.",
        "factors": {
            "iv": "unknown",
            "oi_ok": oi >= OI_MIN_FOR_LLM,
            "volume_ok": vol >= VOLUME_MIN_FOR_LLM,
            "greeks_hint": "n/a"
        },
        "ran": False,
    }

    if in_cooldown:
        llm_reason = cooldown_reason or "In cooldown."
    elif not llm_should_run:
        llm_reason = (
            f"Below thresholds: vol {vol} < {VOLUME_MIN_FOR_LLM} AND "
            f"OI {oi} < {OI_MIN_FOR_LLM}."
        )
    elif not can_consume_llm():
        snap_quota = llm_quota_snapshot()
        llm_reason = f"Daily LLM quota reached ({snap_quota['used']}/{snap_quota['max']})."
    else:
        # Run the LLM analysis and consume budget.
        llm = await analyze_with_openai(context)
        consume_llm()
        llm_ran = True
        llm["ran"] = True

    metrics = {
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": vol,
        "greeks": context["greeks"],
    }

    tg_text = compose_telegram_text(alert, option_ticker, metrics, llm,
                                    llm_ran=llm_ran, llm_reason=llm_reason)
    tg_result = await send_telegram(tg_text)

    # update cooldown timestamp after processing (even if LLM skipped; we still saw an alert)
    _COOLDOWN[key] = now_utc

    # log entry for daily report (in market tz)
    entry = {
        "timestamp_local": market_now(),
        "symbol": alert["symbol"],
        "side": alert["side"],
        "option_ticker": option_ticker,
        "volume": vol,
        "open_interest": oi,
        "llm": {
            "ran": llm_ran,
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "reason": llm.get("reason") if llm_ran else llm_reason,
        },
    }
    _DECISIONS_LOG.append(entry)

    return {
        "ok": True,
        "parsed_alert": alert,
        "option_ticker": option_ticker,
        "metrics": metrics,
        "llm": {
            "ran": llm_ran,
            "reason": llm_reason,
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "factors": llm.get("factors"),
        },
        "cooldown": {
            "seconds": COOLDOWN_SECONDS,
            "active": in_cooldown,
            "reason": cooldown_reason if in_cooldown else "",
        },
        "quota": llm_quota_snapshot(),
        "thresholds": {
            "volume_min_for_llm": VOLUME_MIN_FOR_LLM,
            "oi_min_for_llm": OI_MIN_FOR_LLM,
        },
        "telegram": {"sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False, "result": tg_result},
        "notes": "LLM runs only when volume/OI thresholds pass, cooldown allows, and daily budget permits. Expiry defaults to same-week Friday when not provided. Educational demo; not financial advice.",
    }

if __name__ == "__main__":
    uvicorn.run("alert_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
