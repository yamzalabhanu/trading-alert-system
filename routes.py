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

from ibkr_client import place_recommended_option_order

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # optional, for checks

from config import (
    COOLDOWN_SECONDS,
    CDT_TZ,
    WINDOWS_CDT,          # e.g. [(dt_time, dt_time)] or [(8,30,11,30)]
    MAX_LLM_PER_DAY,
)
from models import Alert, WebhookResponse
from polygon_client import list_contracts_for_expiry, get_option_snapshot
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG, _send_daily_report_now, _summarize_day_for_report, _chunk_lines_for_telegram

router = APIRouter()

# ========== Global state ==========
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

# --- IBKR toggles (env defaults; can be overridden via query params) ---
_IBKR_ENV_DEFAULT = os.getenv("IBKR_ENABLED", "0")
def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

IBKR_ENABLED = _env_truthy(_IBKR_ENV_DEFAULT)           # default behavior from env
IBKR_DEFAULT_QTY = int(os.getenv("IBKR_DEFAULT_QTY", "1"))
IBKR_TIF = os.getenv("IBKR_TIF", "DAY").upper()         # e.g., DAY, GTC
IBKR_ORDER_MODE = os.getenv("IBKR_ORDER_MODE", "auto").lower()   # auto | market | limit
IBKR_USE_MID_AS_LIMIT = os.getenv("IBKR_USE_MID_AS_LIMIT", "1") == "1"

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

def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def flag_from(req: Request, name: str, env_name: str, default: bool = False) -> bool:
    """
    Read a boolean flag with priority: query param -> env -> default.
    """
    qv = req.query_params.get(name)
    if qv is not None:
        return _truthy(qv)
    return _truthy(os.getenv(env_name, "1" if default else "0"))

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
    Accepts WINDOWS_CDT as either:
      - [(dt_time(..., tzinfo=CDT_TZ), dt_time(..., tzinfo=CDT_TZ)), ...]  OR
      - [(8,30,11,30), (14,0,15,0)]
    """
    now = market_now().time()  # aware time in CDT
    for win in WINDOWS_CDT:
        # Case 1: (start_time, end_time) as datetime.time objects
        if isinstance(win, (tuple, list)) and len(win) == 2 \
           and isinstance(win[0], dt_time) and isinstance(win[1], dt_time):
            start, end = win
            if start <= now <= end:
                return True
        # Case 2: (sh, sm, eh, em) as integers
        elif isinstance(win, (tuple, list)) and len(win) == 4 and all(isinstance(x, int) for x in win):
            sh, sm, eh, em = win
            if dt_time(sh, sm, tzinfo=CDT_TZ) <= now <= dt_time(eh, em, tzinfo=CDT_TZ):
                return True
        # Unknown format → ignore
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

def _ibkr_result_to_dict(res: Any) -> Dict[str, Any]:
    # Nothing came back
    if res is None:
        return {"ok": False, "error": "ibkr_client returned None", "raw": None}

    # If ibkr_client returned a dict, keep it intact
    if isinstance(res, dict):
        return {
            "ok": bool(res.get("ok", False)),
            "order_id": res.get("order_id"),
            "status": res.get("status"),
            "filled": res.get("filled"),
            "remaining": res.get("remaining"),
            "avg_fill_price": res.get("avg_fill_price"),
            "error": res.get("error"),
            "raw": res.get("raw", res),  # keep full dict in raw
        }

    # If it returned a plain string
    if isinstance(res, str):
        return {"ok": False, "error": res, "raw": res}

    # Fallback for object-like responses
    try:
        payload = {
            "ok": getattr(res, "ok", False),
            "order_id": getattr(res, "order_id", None),
            "status": getattr(res, "status", None),
            "filled": getattr(res, "filled", None),
            "remaining": getattr(res, "remaining", None),
            "avg_fill_price": getattr(res, "avg_fill_price", None),
            "error": getattr(res, "error", None),
        }
        # Preserve a raw representation
        try:
            payload["raw"] = getattr(res, "raw", None) or repr(res)
        except Exception:
            payload["raw"] = repr(res)
        return payload
    except Exception as e:
        return {"ok": False, "error": f"serialize-failed: {type(e).__name__}: {e}", "raw": repr(res)}


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
    todays = [x for x in reversed(_DECISIONS_LOG)
              if isinstance(x.get("timestamp_local"), datetime) and x["timestamp_local"].date() == today_local]
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
async def webhook_tradingview(request: Request, offline: int = Query(default=0), ib: int = Query(default=0)):
    payload = await _get_alert_text(request)
    if not payload:
        raise HTTPException(status_code=400, detail="Empty alert payload")
    alert = parse_alert_text(payload)

    # === Flags (query param overrides env) ===
    effective_ib_enabled = flag_from(request, "ib", "IBKR_ENABLED", default=IBKR_ENABLED)
    force = 1 #flag_from(request, "force", "FORCE", default=False)
    force_buy = 1 #flag_from(request, "force_buy", "FORCE_BUY", default=False)

    # === Recommendation policy ===
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
    f: Dict[str, Any] = {}

    # ---------- Data prep (Polygon/features) ----------
    try:
        if offline or not POLYGON_API_KEY:
            # OFFLINE MODE: fabricate minimal features
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
            # ONLINE MODE: real Polygon calls
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
        raise HTTPException(status_code=502, detail=f"Upstream error in Polygon/features: {type(e).__name__}: {e}")

    # ---------- Decisioning / LLM / Telegram / IBKR ----------
    in_window = allowed_now_cdt() or force

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

    ib_attempted = False
    ib_result_obj: Optional[Any] = None

    if in_window:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
        llm_ran = True
        decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")
        # Compute score + grade for buys
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
        decision_path = f"llm.{decision_final}"

        # Optional decision override for deterministic testing
        if force_buy:
            decision_final = "buy"
            decision_path = "force.buy"

        # Telegram
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

        # IBKR optional placement (paper)
        try:
            if decision_final == "buy" and effective_ib_enabled:
                ib_attempted = True

                # Decide order style
                #   auto: use limit @ mid if available (and IBKR_USE_MID_AS_LIMIT=1), else market
                #   limit: always limit; if mid missing -> market
                #   market: always market
                use_market = False
                mode = IBKR_ORDER_MODE
                mid = f.get("mid")

                if mode == "market":
                    use_market = True
                elif mode == "limit":
                    use_market = (mid is None)
                else:  # auto
                    use_market = not (IBKR_USE_MID_AS_LIMIT and (mid is not None))

                limit_px = None if use_market else float(mid)

                ib_result_obj = await place_recommended_option_order(
                    symbol=alert["symbol"],
                    side=alert["side"],                # "CALL" or "PUT" (your ibkr_client should handle this)
                    strike=float(desired_strike),     # recommended strike
                    expiry_iso=target_expiry,         # two-weeks Friday rule, ISO "YYYY-MM-DD"
                    quantity=int(os.getenv("IBKR_DEFAULT_QTY", str(IBKR_DEFAULT_QTY))),
                    limit_price=limit_px,             # None => market
                    action="BUY",
                    tif=IBKR_TIF,
                )
        except Exception as e:
            ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        llm_reason = "Outside processing windows (Allowed: 08:30–11:30 & 14:00–15:00 CDT). LLM + Telegram skipped."
        ib_attempted = False
        ib_result_obj = None

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
        "ibkr": {
            "enabled": effective_ib_enabled,
            "attempted": ib_attempted,
            "result": (_ibkr_result_to_dict(ib_result_obj) if ib_result_obj is not None else None),
        },
        "notes": "LLM/Telegram run only during 08:30–11:30 & 14:00–15:00 CDT. Polygon-enhanced features for LLM context. Buys are graded into Strong/Moderate/Cautious.",
    }
