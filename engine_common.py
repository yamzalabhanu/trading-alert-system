# engine_common.py
import os
import re
import math
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

# Keep these env toggles defined (even if you’ve disabled messages elsewhere)
SEND_CHAIN_SCAN_ALERTS      = _env_truthy(os.getenv("SEND_CHAIN_SCAN_ALERTS", "0"))
SEND_CHAIN_SCAN_TOPN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_TOPN_ALERTS", "0"))
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
# Telegram composition (compact)
# =========================
def _fmt_num(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "-"
        v = float(x)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return "-"

def _fmt_int(x):
    try:
        if x is None:
            return "-"
        return f"{int(x):,}"
    except Exception:
        return "-"

def _fmt_pct(x, nd=2):
    try:
        if x is None:
            return "-"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "-"

def _dte_label(expiry_iso: str, now_dt: Optional[datetime] = None) -> str:
    now_dt = now_dt or market_now()
    try:
        d = datetime.fromisoformat(expiry_iso).date()
        today = now_dt.date()
        dte = (d - today).days
        if dte <= 0:
            return "0DTE"
        if dte == 1:
            return "1DTE"
        return f"{dte} DTE"
    except Exception:
        return ""

def _side_letter(side: str) -> str:
    return "C" if (side or "").upper().startswith("C") else "P"

def _compact_adjustments(diff_note: str) -> str:
    if not diff_note:
        return ""
    parts = []
    for raw in diff_note.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("🎯"):
            s = s.replace("🎯 ", "").replace("Selected ", "").replace(" (alert was", " (was")
        elif s.startswith("🗓"):
            s = s.replace("🗓 ", "").replace("Selected ", "").replace(" (alert was", " (was")
        elif s.startswith("📡"):
            s = s.replace("📡 ", "").replace("NBBO via ", "NBBO ")
        elif s.startswith("🧪"):
            s = s.replace("🧪 ", "").replace("Synthetic NBBO used", "Synthetic")
            s = s.replace(" spread est.", "")
        parts.append(s)
    return "Adj: " + " • ".join(parts)

def compose_telegram_text(
    alert: Dict[str, Any],
    option_ticker: Optional[str],
    f: Dict[str, Any],
    llm: Dict[str, Any],
    llm_ran: bool = True,
    llm_reason: str = "",
    score: Optional[float] = None,
    rating: Optional[str] = None,
    diff_note: str = "",
) -> str:
    """
    Compact, skimmable Telegram message.
    """
    sym = alert.get("symbol") or "-"
    side = (alert.get("side") or "").upper()
    strike = alert.get("strike")
    exp = alert.get("expiry") or "-"
    dte_lbl = _dte_label(exp)
    sideL = _side_letter(side)

    decision = (llm.get("decision") or "wait").upper()
    conf_s = _fmt_num(llm.get("confidence"), 2)
    score_s = _fmt_num(score, 2) if score is not None else "-"
    rating_s = rating or "-"

    # Liquidity
    oi_s = _fmt_int(f.get("oi"))
    vol_s = _fmt_int(f.get("vol"))

    # Tech snapshot
    rsi = _fmt_num(f.get("rsi14"), 1)
    ema20 = f.get("ema20"); ema50 = f.get("ema50"); ema200 = f.get("ema200")
    ema_stack = None
    try:
        if all(isinstance(x, (int, float)) for x in (ema20, ema50, ema200)):
            if ema20 > ema50 > ema200:
                ema_stack = "EMA20>50>200 (bullish)"
            elif ema20 < ema50 < ema200:
                ema_stack = "EMA20<50<200 (bearish)"
    except Exception:
        pass
    macd_hist = f.get("macd_hist")
    macd_tag = "MACD hist +" if isinstance(macd_hist, (int,float)) and macd_hist > 0 else ("MACD hist -" if isinstance(macd_hist, (int,float)) else None)

    # Context
    svr = llm.get("short_vol_ratio") or f.get("short_vol_ratio") or None
    svr_s = _fmt_pct(float(svr)*100 if (isinstance(svr,(int,float)) and svr<=1) else svr, 0) if svr is not None else "-"

    # NBBO line
    bid = _fmt_num(f.get("bid"), 4)
    ask = _fmt_num(f.get("ask"), 4)
    mid = _fmt_num(f.get("mid"), 4)
    spread = _fmt_pct(f.get("option_spread_pct"), 2)
    qage = f.get("quote_age_sec")
    age_s = (f"{int(qage)}s" if isinstance(qage, (int, float)) and qage >= 0 else "n/a")
    provider = f.get("nbbo_provider")
    synth = bool(f.get("synthetic_nbbo_used"))
    if synth and not provider:
        provider = "synthetic"
    nbbo_hdr = f"NBBO: {bid}/{ask} mid {mid} • spread {spread} • age {age_s}"
    if provider:
        nbbo_hdr += f" • {provider}"

    # Bias
    bias = (llm.get("bias") or ("INTRADAY" if isinstance(f.get("dte"), (int,float)) and f.get("dte") <= 5 else "SWING")).upper()

    tech_bits = []
    if ema_stack: tech_bits.append(ema_stack)
    if rsi != "-": tech_bits.append(f"RSI {rsi}")
    if macd_tag: tech_bits.append(macd_tag)
    tech_line = " | ".join(tech_bits) if tech_bits else "—"

    # VWAP / ORB hints (only show if present)
    vwap_s = _fmt_num(f.get("vwap"), 2)
    orb_hi = _fmt_num(f.get("orb15_high"), 2)
    orb_lo = _fmt_num(f.get("orb15_low"), 2)
    vwap_part = f"VWAP {vwap_s}" if vwap_s != "-" else "VWAP —"
    orb_part = f"ORB {orb_lo}-{orb_hi}" if (orb_lo != "-" and orb_hi != "-") else "ORB —"

    # Adjustments
    adj_line = _compact_adjustments(diff_note)

    # Header
    header = f"{decision} – {sym} {_fmt_num(strike, 2)}{sideL} ({dte_lbl}, {exp})"

    # Shortened reason
    reason = (llm.get("reason") or llm_reason or "").strip().replace("\n", " ")
    if len(reason) > 220:
        reason = reason[:217] + "…"

    lines = [
        header,
        f"Bias: {bias}" + (" • ⚠️ 0DTE" if dte_lbl == "0DTE" else ""),
        (reason if reason else None),
        f"Tech: {tech_line}",
        f"Context: SVR {svr_s} • {vwap_part} • {orb_part}",
        f"Liquidity: OI {oi_s} / Vol {vol_s}",
        f"{nbbo_hdr}",
        f"Confidence: {conf_s} | Score: {score_s} | Rating: {rating_s}",
        (adj_line if adj_line else None),
    ]
    return "\n".join([ln for ln in lines if ln and ln.strip()])

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
