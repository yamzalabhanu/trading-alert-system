# routes.py
# at top
import os
from fastapi import Query
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import re
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta, date, time as dt_time



POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # optional, for checks


from config import (
    COOLDOWN_SECONDS,
    CDT_TZ,
    WINDOWS_CDT,          # e.g. [(8,30,11,30), (14,0,15,0)] in local CDT
    MAX_LLM_PER_DAY,      # make sure this exists in config; default in your app
)
from models import Alert, WebhookResponse
from polygon_client import list_contracts_for_expiry, get_option_snapshot
from llm_client import analyze_with_openai, build_llm_prompt
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG, _send_daily_report_now, _summarize_day_for_report, _chunk_lines_for_telegram

router = APIRouter()

# ========== Global state ==========
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

# ========== Regex patterns ==========
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

# ========== Small utilities used here ==========

def market_now() -> datetime:
    """Current time localized to market (CDT)."""
    return datetime.now(CDT_TZ)

def _reset_quota_if_new_day() -> None:
    today = market_now().date()
    if _llm_quota["date"] != today:
        _llm_quota["date"] = today
        _llm_quota["used"] = 0

def llm_quota_snapshot() -> Dict[str, Any]:
    _reset_quota_if_new_day()
    used = int(_llm_quota.get("used", 0))
    limit = int(MAX_LLM_PER_DAY)
    return {"limit": limit, "used": used, "remaining": max(0, limit - used), "date": str(_llm_quota["date"])}

def consume_llm(n: int = 1) -> None:
    _reset_quota_if_new_day()
    _llm_quota["used"] = int(_llm_quota.get("used", 0)) + n

def allowed_now_cdt() -> bool:
    """
    Check if current CDT time is inside any configured window.
    WINDOWS_CDT is expected like [(8,30,11,30), (14,0,15,0)] -> 08:30–11:30, 14:00–15:00.
    """
    now = market_now().time()
    for (sh, sm, eh, em) in WINDOWS_CDT:
        if (dt_time(sh, sm) <= now <= dt_time(eh, em)):
            return True
    return False

def round_strike_to_common_increment(val: float) -> float:
    """
    Round to a 'common' equity option increment.
    - Under $25: 0.5
    - $25 to <$200: 1
    - $200 to <$1000: 5
    - >= $1000: 10
    """
    if val < 25:
        step = 0.5
    elif val < 200:
        step = 1
    elif val < 1000:
        step = 5
    else:
        step = 10
    return round(round(val / step) * step, 2)

def _next_friday(d: date) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7)

def same_week_friday(d: date) -> date:
    """Friday (week ending) for date d."""
    base_monday = d - timedelta(days=d.weekday())
    return base_monday + timedelta(days=4)

def two_weeks_friday(d: date) -> date:
    """Friday two weeks out from date d (same week’s Friday + 14 days)."""
    return _next_friday(d) + timedelta(days=7)  # next Friday + 1 week

def is_same_week(a: date, b: date) -> bool:
    am = a - timedelta(days=a.weekday())
    bm = b - timedelta(days=b.weekday())
    return am == bm

async def _get_alert_text(request: Request) -> str:
    """
    Accepts TradingView alerts in either raw text/plain, JSON {"message": "..."} or {"alert": "..."}.
    """
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            return str(data.get("message") or data.get("alert") or data.get("text") or "").strip()
        body = await request.body()
        return body.decode("utf-8").strip()
    except Exception:
        return ""

def parse_alert_text(text: str) -> Dict[str, Any]:
    """
    Parses alert lines like:
      'CALL Signal: AAPL at 208.15 Strike: 210 Expiry: 2025-09-12'
      'PUT Signal: NVDA at 120.50 Strike: 115'
    Returns dict with keys: side, symbol, underlying_price_from_alert, strike (if present), expiry (if present).
    """
    m = ALERT_RE_WITH_EXP.match(text)
    if m:
        side, symbol, ul, strike, exp = m.groups()
        return {
            "side": side.upper(),
            "symbol": symbol.upper(),
            "underlying_price_from_alert": float(ul),
            "strike": float(strike),
            "expiry": exp,
        }
    m = ALERT_RE_NO_EXP.match(text)
    if m:
        side, symbol, ul, strike = m.groups()
        return {
            "side": side.upper(),
            "symbol": symbol.upper(),
            "underlying_price_from_alert": float(ul),
            "strike": float(strike),
        }
    raise HTTPException(status_code=400, detail="Unrecognized alert format")

def compose_telegram_text(
    alert: Dict[str, Any],
    option_ticker: str,
    f: Dict[str, Any],
    llm: Dict[str, Any],
    llm_ran: bool,
    llm_reason: str,
    score: Optional[float],
    rating: Optional[str],
) -> str:
    header = f"📣 Options Alert\n{alert['side']} {alert['symbol']} | Strike {alert.get('strike')} | Exp {alert.get('expiry')}\nUnderlying (alert): {alert.get('underlying_price_from_alert')}"
    contract = f"Contract: {option_ticker}"
    snap = (
        f"Snapshot:\n"
        f"  IV: {f.get('iv')}  (IV rank: {f.get('iv_rank')})\n"
        f"  OI: {f.get('oi')}  Vol: {f.get('vol')}\n"
        f"  NBBO: bid={f.get('bid')} ask={f.get('ask')} mid={f.get('mid')}\n"
        f"  Spread%: {f.get('option_spread_pct')}  QuoteAge(s): {f.get('quote_age_sec')}\n"
        f"  Greeks: Δ={f.get('delta')} Γ={f.get('gamma')} Θ={f.get('theta')} ν={f.get('vega')}\n"
        f"  EM_vs_BE_ok: {f.get('em_vs_be_ok')}  MTF align: {f.get('mtf_align')}\n"
        f"  S/R ok: {f.get('sr_headroom_ok')}  Regime: {f.get('regime_flag')}  DTE: {f.get('dte')}\n"
    )
    if llm_ran:
        decision = f"LLM Decision: {llm.get('decision','WAIT').upper()}  (conf: {llm.get('confidence')})"
        reason = f"Reason: {llm.get('reason','')}"
        scoreline = f"Score: {score}  Rating: {rating}"
    else:
        decision = "LLM Decision: SKIPPED (outside window)"
        reason = f"Note: {llm_reason}"
        scoreline = ""
    return "\n".join([header, contract, "", snap, decision, reason, scoreline]).strip()

# Async wrappers for polygon_client (keeping your current call style)
async def polygon_list_contracts_for_expiry(
    client: httpx.AsyncClient,
    symbol: str,
    expiry: str,
    side: str,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    return await list_contracts_for_expiry(client, symbol=symbol, expiry=expiry, side=side, limit=limit)

async def polygon_get_option_snapshot(
    client: httpx.AsyncClient,
    underlying: str,
    option_ticker: str,
) -> Dict[str, Any]:
    return await get_option_snapshot(client, underlying=underlying, option_ticker=option_ticker)

# ========== Routes ==========

@router.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@router.get("/quota")
def quota():
    return {"ok": True, "quota": llm_quota_snapshot()}

@router.get("/config")
def get_config():
    # If you have an _active_config_dict elsewhere, import and return it here.
    # To avoid a NameError, we return a minimal snapshot from config.
    cfg = {
        "COOLDOWN_SECONDS": COOLDOWN_SECONDS,
        "WINDOWS_CDT": WINDOWS_CDT,
        "MAX_LLM_PER_DAY": MAX_LLM_PER_DAY,
        "CDT_TZ": str(CDT_TZ),
    }
    return {"ok": True, "config": cfg}

@router.get("/logs/today")
def logs_today(limit: int = 50):
    limit = max(1, min(int(limit), 500))
    today_local = market_now().date()
    # Filter today's logs (assumes _DECISIONS_LOG entries have 'timestamp_local' as datetime)
    todays = [x for x in reversed(_DECISIONS_LOG) if isinstance(x.get("timestamp_local"), datetime) and x["timestamp_local"].date() == today_local]
    return {"ok": True, "count": len(todays[:limit]), "items": todays[:limit]}

@router.post("/run/daily_report")
async def run_daily_report():
    res = await _send_daily_report_now()
    return {"ok": True, "trigger": "manual", **res}

@router.get("/report/preview")
def report_preview():
    today_local = market_now().date()
    rep = _summarize_day_for_report(today_local)
    chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")
    return {"ok": True, "header": rep["header"], "contract_chunks": chunks, "count": rep["count"]}

@router.post("/webhook", response_class=JSONResponse)
@router.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(request: Request, offline: int = Query(default=0)):
    payload = await _get_alert_text(request)
    if not payload:
        raise HTTPException(status_code=400, detail="Empty alert payload")
    alert = parse_alert_text(payload)

    # === Recommendation policy (unchanged) ===
    ul_px = float(alert["underlying_price_from_alert"])
    raw_reco_strike = ul_px * (1.05 if alert["side"] == "CALL" else 0.95)
    desired_strike = round_strike_to_common_increment(raw_reco_strike)
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    option_ticker = "OFFLINE-TICKER"
    f = {}

    try:
        if offline or not POLYGON_API_KEY:
            # ---- OFFLINE MODE: skip Polygon, fabricate minimal features ----
            f = {
                "bid": None, "ask": None, "mid": None,
                "option_spread_pct": None, "quote_age_sec": None,
                "oi": None, "vol": None,
                "delta": None, "gamma": None, "theta": None, "vega": None,
                "iv": None, "iv_rank": None, "rv20": None,
                "dte": (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days,
                "em_vs_be_ok": None,
                "mtf_align": None, "sr_headroom_ok": None, "regime_flag": "trending",
                "prev_day_high": None, "prev_day_low": None,
                "premarket_high": None, "premarket_low": None,
                "vwap": None, "vwap_dist": None,
                "above_pdh": None, "below_pdl": None, "above_pmh": None, "below_pml": None,
            }
        else:
            # ---- ONLINE MODE: real Polygon calls ----
            async with httpx.AsyncClient(http2=True, timeout=20.0) as client:
                contracts = await polygon_list_contracts_for_expiry(
                    client, symbol=alert["symbol"], expiry=target_expiry, side=alert["side"], limit=250
                )
                if not contracts:
                    raise HTTPException(status_code=404, detail=f"No contracts found for {alert['symbol']} {alert['side']} exp {target_expiry}.")
                best = min(contracts, key=lambda c: abs(float(c.get("strike", 0)) - desired_strike)) if contracts else None
                if not best:
                    raise HTTPException(status_code=404, detail=f"No strikes near {desired_strike} for {alert['symbol']} on {target_expiry}.")
                option_ticker = best.get("ticker") or best.get("symbol")
                if not option_ticker:
                    raise HTTPException(status_code=500, detail="Polygon returned contract without ticker")

                snap = await polygon_get_option_snapshot(client, underlying=alert["symbol"], option_ticker=option_ticker)
                f = await build_features(
                    client,
                    alert={**alert, "strike": desired_strike, "expiry": target_expiry},
                    snapshot=snap
                )
    except HTTPException:
        raise
    except Exception as e:
        # Make it a clear 502 instead of a generic 500
        raise HTTPException(status_code=502, detail=f"Upstream error in Polygon/features: {type(e).__name__}: {e}")

    # ... keep the rest of the function exactly as we already have ...


    in_window = allowed_now_cdt()
    llm_ran = False
    llm_reason = ""
    tg_result = None
    decision_final = "skip"
    decision_path = "window.skip"
    score: Optional[float] = None
    rating: Optional[str] = None
    llm: Dict[str, Any] = {
        "decision": "wait",
        "confidence": 0.0,
        "reason": "",
        "checklist": {},
        "ev_estimate": {}
    }

    if in_window:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
        llm_ran = True
        decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")
        # Compute score + grade for buys
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
        decision_path = f"llm.{decision_final}"

        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": target_expiry},
            option_ticker=option_ticker,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",
            score=score,
            rating=rating
        )
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            tg_result = await send_telegram(tg_text)
    else:
        llm_reason = "Outside processing windows (Allowed: 08:30–11:30 & 14:00–15:00 CDT). LLM + Telegram skipped."

    # Cooldown timestamp (not a gate)
    _COOLDOWN[(alert["symbol"], alert["side"])] = datetime.now(timezone.utc)

    # Log (even if skipped, so you can audit)
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": alert["symbol"],
        "side": alert["side"],
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": decision_path,
        "prescore": None,
        "llm": {
            "ran": llm_ran,
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "reason": llm.get("reason")
        },
        "features": {
            "reco_expiry": target_expiry,
            "oi": f.get("oi"), "vol": f.get("vol"),
            "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
            "delta": f.get("delta"), "gamma": f.get("gamma"), "theta": f.get("theta"), "vega": f.get("vega"),
            "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
            "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag"),
            "prev_day_high": f.get("prev_day_high"), "prev_day_low": f.get("prev_day_low"),
            "premarket_high": f.get("premarket_high"), "premarket_low": f.get("premarket_low"),
            "vwap": f.get("vwap"), "vwap_dist": f.get("vwap_dist"),
            "above_pdh": f.get("above_pdh"), "below_pdl": f.get("below_pdl"),
            "above_pmh": f.get("above_pmh"), "below_pml": f.get("below_pml"),
        }
    })

    return {
        "ok": True,
        "parsed_alert": alert,
        "option_ticker": option_ticker,
        "features": f,
        "prescore": None,
        "recommendation": {
            "side": alert["side"],
            "underlying_from_alert": ul_px,
            "strike_policy": "+5% for CALL / -5% for PUT (rounded)",
            "strike_recommended": desired_strike,
            "expiry_policy": "Friday two weeks out (never same-week)",
            "expiry_recommended": target_expiry,
        },
        "decision": {
            "final": decision_final,
            "path": decision_path,
            "score": (score if in_window else None),
            "rating": (rating if in_window else None)
        },
        "llm": {
            "ran": llm_ran,
            "reason": llm_reason,
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "checklist": llm.get("checklist"),
            "ev_estimate": llm.get("ev_estimate"),
        },
        "cooldown": {"seconds": COOLDOWN_SECONDS, "active": False},
        "quota": llm_quota_snapshot(),
        "telegram": {
            "sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False,
            "result": tg_result
        },
        "notes": "LLM/Telegram run only during 08:30–11:30 & 14:00–15:00 CDT. Polygon-enhanced features for LLM context. Buys are graded into Strong/Moderate/Cautious.",
    }
