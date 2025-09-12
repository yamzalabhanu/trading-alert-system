# engine_common.py
import os
import re
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta, date

from fastapi import HTTPException, Request
from config import CDT_TZ, MAX_LLM_PER_DAY
from polygon_client import build_option_contract

logger = logging.getLogger("trading_engine")

# =========================
# Env / knobs
# =========================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

IBKR_ENABLED = _env_truthy(os.getenv("IBKR_ENABLED", "0"))
IBKR_DEFAULT_QTY = int(os.getenv("IBKR_DEFAULT_QTY", "1"))
IBKR_TIF = os.getenv("IBKR_TIF", "DAY").upper()
IBKR_ORDER_MODE = os.getenv("IBKR_ORDER_MODE", "auto").lower()   # auto | market | limit
IBKR_USE_MID_AS_LIMIT = os.getenv("IBKR_USE_MID_AS_LIMIT", "1") == "1"

# Trading thresholds (tunable)
TARGET_DELTA_CALL = float(os.getenv("TARGET_DELTA_CALL", "0.35"))
TARGET_DELTA_PUT  = float(os.getenv("TARGET_DELTA_PUT", "-0.35"))
MAX_SPREAD_PCT    = float(os.getenv("MAX_SPREAD_PCT", "6.0"))
MAX_QUOTE_AGE_S   = float(os.getenv("MAX_QUOTE_AGE_S", "30"))
MIN_VOL_TODAY     = int(os.getenv("MIN_VOL_TODAY", "100"))
MIN_OI            = int(os.getenv("MIN_OI", "200"))
MIN_DTE           = int(os.getenv("MIN_DTE", "3"))
MAX_DTE           = int(os.getenv("MAX_DTE", "45"))

# Optional scan knobs (RTH vs AH)
SCAN_MIN_VOL_RTH = int(os.getenv("SCAN_MIN_VOL_RTH", os.getenv("SCAN_MIN_VOL", "500")))
SCAN_MIN_OI_RTH  = int(os.getenv("SCAN_MIN_OI_RTH",  os.getenv("SCAN_MIN_OI",  "500")))
SCAN_MIN_VOL_AH  = int(os.getenv("SCAN_MIN_VOL_AH", "0"))
SCAN_MIN_OI_AH   = int(os.getenv("SCAN_MIN_OI_AH",  "100"))

SEND_CHAIN_SCAN_ALERTS      = _env_truthy(os.getenv("SEND_CHAIN_SCAN_ALERTS", "1"))
SEND_CHAIN_SCAN_TOPN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_TOPN_ALERTS", "1"))
REPLACE_IF_NO_NBBO          = _env_truthy(os.getenv("REPLACE_IF_NO_NBBO", "1"))

# =========================
# Small helpers / time & quota
# =========================
def market_now() -> datetime:
    return datetime.now(CDT_TZ)

_llm_quota: Dict[str, Any] = {"date": None, "used": 0}

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

# =========================
# Parsing / regex
# =========================
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

async def get_alert_text_from_request(request: Request) -> str:
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

# =========================
# Misc utils
# =========================
def round_strike_to_common_increment(val: float) -> float:
    if val < 25: step = 0.5
    elif val < 200: step = 1
    elif val < 1000: step = 5
    else: step = 10
    return round(round(val / step) * step, 2)

def _next_friday(d: date) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7)

def same_week_friday(d: date) -> date:
    base_monday = d - timedelta(days=d.weekday())
    return base_monday + timedelta(days=4)

def two_weeks_friday(d: date) -> date:
    return _next_friday(d) + timedelta(days=7)

def is_same_week(a: date, b: date) -> bool:
    am = a - timedelta(days=a.weekday()); bm = b - timedelta(days=b.weekday())
    return am == bm

def _encode_ticker_path(t: str) -> str:
    return quote(t or "", safe="")

def _is_rth_now() -> bool:
    now = datetime.now(CDT_TZ)
    if now.weekday() > 4:
        return False
    start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    end   = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return start <= now <= end

def _occ_meta(ticker: str) -> Optional[Dict[str, str]]:
    m = re.search(r":([A-Z0-9\.\-]+)(\d{2})(\d{2})(\d{2})([CP])\d{8,9}$", ticker or "")
    if not m:
        return None
    yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
    return {"side": ("CALL" if cp.upper() == "C" else "PUT"), "expiry": f"20{yy}-{mm}-{dd}"}

def _ticker_matches_side(ticker: Optional[str], side: str) -> bool:
    if not ticker:
        return False
    meta = _occ_meta(ticker)
    return bool(meta and meta["side"] == side)

# =========================
# Preflight
# =========================
def preflight_ok(f: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    checks: Dict[str, bool] = {}
    rth = _is_rth_now()

    quote_age = f.get("quote_age_sec")
    has_nbbo  = f.get("bid") is not None and f.get("ask") is not None
    has_last  = isinstance(f.get("last"), (int, float)) or isinstance(f.get("mid"), (int, float))

    if rth:
        checks["quote_fresh"] = (quote_age is not None and quote_age <= MAX_QUOTE_AGE_S and has_nbbo)
        checks["spread_ok"]   = (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= MAX_SPREAD_PCT)
    else:
        checks["quote_fresh"] = bool(has_last)
        checks["spread_ok"]   = True

    require_liquidity = os.getenv("REQUIRE_LIQUIDITY_FIELDS", "0") == "1"
    if rth and require_liquidity:
        checks["vol_ok"] = (f.get("vol") or 0) >= MIN_VOL_TODAY
        checks["oi_ok"]  = (f.get("oi")  or 0) >= MIN_OI
    else:
        checks["vol_ok"] = True
        checks["oi_ok"]  = True

    dte_val = f.get("dte")
    checks["dte_ok"] = (dte_val is not None) and (MIN_DTE <= dte_val <= MAX_DTE)

    ok = all(checks.values())
    return ok, checks

# =========================
# Telegram composition
# =========================
def compose_telegram_text(
    alert: Dict[str, Any],
    option_ticker: str,
    f: Dict[str, Any],
    llm: Dict[str, Any],
    llm_ran: bool,
    llm_reason: str,
    score: Optional[float],
    rating: Optional[str],
    diff_note: str = "",
) -> str:
    header = (
        f"📣 Options Alert\n"
        f"{alert['side']} {alert['symbol']} | Strike {alert.get('strike')} | Exp {alert.get('expiry')}\n"
        f"Underlying (alert): {alert.get('underlying_price_from_alert')}"
    )
    contract = f"Contract: {option_ticker}"
    snap = (
        "Snapshot:\n"
        f"  NBBO: bid={f.get('bid')} ask={f.get('ask')}  Mark={f.get('mid')}  Last={f.get('last')}\n"
        f"  Spread%: {f.get('option_spread_pct')}  QuoteAge(s): {f.get('quote_age_sec')}\n"
        f"  PrevClose: {f.get('prev_close')}  Chg% vs PrevClose: {f.get('quote_change_pct')}\n"
        f"  OI: {f.get('oi')}  Vol: {f.get('vol')}  IV: {f.get('iv')}  (IV rank: {f.get('iv_rank')})\n"
        f"  Greeks: Δ={f.get('delta')} Γ={f.get('gamma')} Θ={f.get('theta')} ν={f.get('vega')}\n"
        f"  EM_vs_BE_ok: {f.get('em_vs_be_ok')}  MTF align: {f.get('mtf_align')}\n"
        f"  S/R ok: {f.get('sr_headroom_ok')}  Regime: {f.get('regime_flag')}  DTE: {f.get('dte')}\n"
        f"  NBBO debug: status={f.get('nbbo_http_status')} reason={f.get('nbbo_reason')}\n"
    )
    if llm_ran:
        decision = f"LLM Decision: {llm.get('decision','WAIT').upper()}  (conf: {llm.get('confidence')})"
        reason = f"Reason: {llm.get('reason','')}"
        scoreline = f"Score: {score}  Rating: {rating}"
    else:
        decision = "LLM Decision: SKIPPED"
        reason = f"Note: {llm_reason or 'LLM not executed'}"
        scoreline = ""
    parts = [header, contract, "", snap, diff_note.strip(), decision, reason, scoreline]
    return "\n".join([p for p in parts if p]).strip()

def _ibkr_result_to_dict(res: Any) -> Dict[str, Any]:
    if res is None:
        return {"ok": False, "error": "ibkr_client returned None", "raw": None}
    if isinstance(res, dict):
        return {
            "ok": bool(res.get("ok", False)),
            "order_id": res.get("order_id"),
            "status": res.get("status"),
            "filled": res.get("filled"),
            "remaining": res.get("remaining"),
            "avg_fill_price": res.get("avg_fill_price"),
            "error": res.get("error"),
            "raw": res.get("raw", res),
        }
    if isinstance(res, str):
        return {"ok": False, "error": res, "raw": res}
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
        try:
            payload["raw"] = getattr(res, "raw", None) or repr(res)
        except Exception:
            payload["raw"] = repr(res)
        return payload
    except Exception as e:
        return {"ok": False, "error": f"serialize-failed: {type(e).__name__}: {e}", "raw": repr(res)}

# strike builder (local)
def _build_plus_minus_contracts(symbol: str, ul_px: float, expiry_iso: str) -> Dict[str, Any]:
    call_strike = round_strike_to_common_increment(ul_px * 1.05)
    put_strike  = round_strike_to_common_increment(ul_px * 0.95)
    return {
        "strike_call": call_strike,
        "strike_put":  put_strike,
        "contract_call": build_option_contract(symbol, expiry_iso, "CALL", call_strike),
        "contract_put":  build_option_contract(symbol, expiry_iso, "PUT",  put_strike),
    }

__all__ = [
    "POLYGON_API_KEY",
    "IBKR_ENABLED","IBKR_DEFAULT_QTY","IBKR_TIF","IBKR_ORDER_MODE","IBKR_USE_MID_AS_LIMIT",
    "TARGET_DELTA_CALL","TARGET_DELTA_PUT","MAX_SPREAD_PCT","MAX_QUOTE_AGE_S",
    "MIN_VOL_TODAY","MIN_OI","MIN_DTE","MAX_DTE",
    "SCAN_MIN_VOL_RTH","SCAN_MIN_OI_RTH","SCAN_MIN_VOL_AH","SCAN_MIN_OI_AH",
    "SEND_CHAIN_SCAN_ALERTS","SEND_CHAIN_SCAN_TOPN_ALERTS","REPLACE_IF_NO_NBBO",
    "market_now","llm_quota_snapshot","consume_llm",
    "get_alert_text_from_request","parse_alert_text",
    "round_strike_to_common_increment","_next_friday","same_week_friday","two_weeks_friday","is_same_week",
    "_encode_ticker_path","_is_rth_now","_occ_meta","_ticker_matches_side",
    "preflight_ok","compose_telegram_text","_ibkr_result_to_dict","_build_plus_minus_contracts",
]
