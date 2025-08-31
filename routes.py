# routes.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import re
from typing import Dict, Any
from datetime import datetime, timezone
from config import COOLDOWN_SECONDS, CDT_TZ, WINDOWS_CDT
from models import Alert, WebhookResponse
from polygon_client import list_contracts_for_expiry, get_option_snapshot
from llm_client import analyze_with_openai, build_llm_prompt
from telegram_client import send_telegram
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG

router = APIRouter()

# Global state
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

# Regex patterns
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

@router.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@router.get("/quota")
def quota():
    return {"ok": True, "quota": llm_quota_snapshot()}
    pass

@router.get("/config")
def get_config():
     return {"ok": True, "config": _active_config_dict()}
    pass

@router.get("/logs/today")
def logs_today(limit: int = 50):
       limit = max(1, min(int(limit), 500))
    today_local = market_now().date()
    pass

@router.post("/run/daily_report")
async def run_daily_report():
     res = await _send_daily_report_now()
    return {"ok": True, "trigger": "manual", **res}
    pass

@router.get("/report/preview")
def report_preview():
      today_local = market_now().date()
    rep = _summarize_day_for_report(today_local)
    chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")
    return {"ok": True, "header": rep["header"], "contract_chunks": chunks, "count": rep["count"]}
    pass

@router.post("/webhook", response_class=JSONResponse)
@router.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(request: Request):
       payload = await _get_alert_text(request)
    alert = parse_alert_text(payload)

    # === Recommendation policy ===
    # Strike: +5% (CALL) / -5% (PUT) of underlying (from alert)
    # Expiry: Friday two weeks out (ignores alert-provided expiry)
    ul_px = float(alert["underlying_price_from_alert"])
    raw_reco_strike = ul_px * (1.05 if alert["side"] == "CALL" else 0.95)
    desired_strike = round_strike_to_common_increment(raw_reco_strike)
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    # Hard guard: never allow same-week expiry
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)  # push to next week if ever equal
    target_expiry = target_expiry_date.isoformat()

    async with httpx.AsyncClient(http2=True, timeout=20.0) as client:
        contracts = await polygon_list_contracts_for_expiry(client,
            symbol=alert["symbol"], expiry=target_expiry, side=alert["side"], limit=250)
        if not contracts:
            raise HTTPException(status_code=404, detail=f"No contracts found for {alert['symbol']} {alert['side']} exp {target_expiry}.")
        best = pick_nearest_strike(contracts, desired_strike)
        if not best:
            raise HTTPException(status_code=404, detail=f"No strikes near {desired_strike} for {alert['symbol']} on {target_expiry}.")
        option_ticker = best.get("ticker")
        snap = await polygon_get_option_snapshot(client, underlying=alert["symbol"], option_ticker=option_ticker)
        # Build features with recommended strike/expiry baked into alert context
        f = await build_features(client, alert={**alert, "strike": desired_strike, "expiry": target_expiry}, snapshot=snap)

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
            # Ensure the message shows the recommendation (strike/expiry)
            alert={**alert, "strike": desired_strike, "expiry": target_expiry},
            option_ticker=option_ticker,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",
            score=score,
            rating=rating
        )
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
        "llm": {"ran": llm_ran, "decision": llm.get("decision"), "confidence": llm.get("confidence"), "reason": llm.get("reason")},
        "features": {
            # expose recommendation context for audit
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
        "telegram": {"sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False, "result": tg_result},
        "notes": "LLM/Telegram run only during 08:30–11:30 & 14:00–15:00 CDT. Polygon-enhanced features for LLM context. Buys are graded into Strong/Moderate/Cautious.",
    }
    pass