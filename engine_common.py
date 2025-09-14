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
# Telegram composition (compact human-friendly)
# =========================
def _fmt_num(n, nd=2, dash="None"):
    try:
        if n is None:
            return dash
        n = float(n)
        if abs(n - round(n)) < 1e-9:
            return str(int(round(n)))
        return f"{n:.{nd}f}"
    except Exception:
        return dash

def _fmt_pct(n, nd=1, dash="None"):
    try:
        if n is None:
            return dash
        return f"{float(n):.{nd}f}%"
    except Exception:
        return dash

def _first_sentence(s: str, max_len: int = 120) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    for stop in [". ", "! ", "? "]:
        p = s.find(stop)
        if 0 < p < max_len:
            return s[:p + 1]
    return s[:max_len].rstrip()

def _pick_factor_lines_from_reason(reason: str, wanted=("RSI", "EMA", "MACD", "VWAP", "Bollinger", "ORB")):
    if not reason:
        return []
    lines = [ln.strip() for ln in reason.splitlines() if ln.strip()]
    body = []
    for ln in lines:
        if ln.startswith("🕒 Suggested trade"):
            continue
        body.append(ln)
    picked = []
    for tag in wanted:
        for ln in body:
            if tag.lower() in ln.lower():
                if ln not in picked:
                    picked.append(ln)
                    break
    return picked[:3]

def compose_telegram_text(
    alert: Dict[str, Any],
    option_ticker: Optional[str],
    f: Dict[str, Any],
    llm: Dict[str, Any],
    llm_ran: bool = False,
    llm_reason: str = "",
    score: Optional[float] = None,
    rating: Optional[str] = None,
    diff_note: str = "",
) -> str:
    """
    Pretty Telegram formatter (concise summary style).
    Keeps the old signature so engine_processor doesn't need to change.
    """
    side = (alert.get("side") or "").upper()
    sym = alert.get("symbol") or "?"
    strike = alert.get("strike")
    right = "C" if side == "CALL" else "P"
    expiry = alert.get("expiry") or ""

    # DTE
    dte = f.get("dte")
    if dte is None and expiry:
        try:
            dte = (datetime.fromisoformat(expiry).date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    # LLM bits
    hdr_decision = (llm.get("decision") or "wait").upper() if isinstance(llm, dict) else "WAIT"
    horizon = (llm.get("horizon") or "").upper() if isinstance(llm, dict) else ""
    horizon_reason = llm.get("horizon_reason") if isinstance(llm, dict) else None
    conf = llm.get("confidence")
    reason_text = (llm.get("reason") or llm_reason or "").strip()

    # Header
    s_fmt = _fmt_num(strike, 0)
    header = f"{hdr_decision} – {sym} {s_fmt}{right} ({expiry}, { _fmt_num(dte, 0) } DTE)"

    # Bias / Summary
    bias_line = ""
    if horizon:
        bias_short = _first_sentence(horizon_reason or "", max_len=80)
        bias_line = f"Bias: {horizon}" + (f" ({bias_short})" if bias_short else "")

    summary_line = ""
    if reason_text:
        lines = [ln for ln in reason_text.splitlines() if ln.strip()]
        narrative = ""
        for i, ln in enumerate(lines):
            if ln.startswith("🕒 Suggested trade"):
                if i + 1 < len(lines):
                    narrative = lines[i + 1].strip()
                break
        if not narrative:
            narrative = _first_sentence(reason_text, max_len=180)
        if narrative:
            summary_line = _first_sentence(narrative, max_len=180)

    # Setup (pick a few key factor lines)
    setup_bits = _pick_factor_lines_from_reason(reason_text)
    setup_line = ""
    if setup_bits:
        cleaned = []
        for s in setup_bits:
            s2 = s
            if s2.startswith("[") and "] " in s2:
                s2 = s2.split("] ", 1)[1]
            cleaned.append(s2)
        setup_line = "Setup: " + "; ".join(cleaned)

    # Liquidity
    oi = f.get("oi"); vol = f.get("vol")
    liq_line = f"Liquidity: OI { _fmt_num(oi, 0, dash='?') } / Vol { _fmt_num(vol, 0, dash='?') }"

    # Context (short-volume ratio)
    svr = f.get("short_volume_ratio")
    if svr is None:
        svr_disp = "—"
    else:
        try:
            val = float(svr)
            if 0 <= val <= 1.0:
                val *= 100.0
            svr_disp = _fmt_pct(val, 0)
        except Exception:
            svr_disp = "—"
    ctx_line = f"Context: Short-vol ratio {svr_disp}"

    # Quotes
    spread_pct = f.get("option_spread_pct")
    qage = f.get("quote_age_sec")
    syn = bool(f.get("synthetic_nbbo_used"))
    quotes_bits = []
    if qage is not None:
        label = "stale" if (isinstance(qage, (int, float)) and qage > 90) else "age"
        try:
            qage_i = int(float(qage))
        except Exception:
            qage_i = None
        quotes_bits.append(f"NBBO {label}" + (f" (~{qage_i}s)" if qage_i is not None else ""))
    else:
        quotes_bits.append("NBBO age n/a")
    if syn:
        est = f.get("synthetic_nbbo_spread_est") if f.get("synthetic_nbbo_spread_est") is not None else spread_pct
        quotes_bits.append(f"synthetic spread est. { _fmt_pct(est, 1, dash='?') }")
    else:
        if spread_pct is not None:
            quotes_bits.append(f"spread { _fmt_pct(spread_pct, 1) }")
    quotes_line = "Quotes: " + "; ".join(quotes_bits)

    # Confidence & Score/Rating
    conf_line = f"Confidence: { _fmt_num(conf, 2, dash='—') }"
    if syn and (qage is None or (isinstance(qage, (int, float)) and qage > 120)):
        conf_line += " (verify live quote)"
    sr_line = ""
    if score is not None or rating is not None:
        sr_line = f"Score: { _fmt_num(score, 2, dash='—') }"
        if rating:
            sr_line += f"  Rating: {rating}"

    appendix = diff_note.strip()
    if appendix:
        appendix = "\n" + appendix

    out_lines = [
        header,
        bias_line or None,
        summary_line or None,
        setup_line or None,
        liq_line,
        ctx_line,
        quotes_line,
        conf_line,
        sr_line or None,
        appendix or None,
    ]
    return "\n".join([ln for ln in out_lines if ln])

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
