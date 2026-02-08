# engine_common.py
import os
import re
import math
from typing import List
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta, date

from fastapi import HTTPException, Request
from config import CDT_TZ, MAX_LLM_PER_DAY

logger = logging.getLogger("trading_engine")


def build_option_contract(symbol: str, expiry_iso: str, side: str, strike: float) -> str:
    dt_ = datetime.fromisoformat(expiry_iso).date()
    yy = dt_.year % 100
    cp = "C" if str(side).upper().startswith("C") else "P"
    strike_int = int(round(float(strike) * 1000))
    return f"O:{symbol.upper()}{yy:02d}{dt_.month:02d}{dt_.day:02d}{cp}{strike_int:08d}"

# =========================
# Env / knobs
# =========================
POLYGON_API_KEY = None  # Polygon integration removed

def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

IBKR_ENABLED = False  # IBKR integration removed
IBKR_DEFAULT_QTY = 0
IBKR_TIF = "DAY"
IBKR_ORDER_MODE = "disabled"
IBKR_USE_MID_AS_LIMIT = False

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
def _fmt_num(x, nd: int = 2, dash: str = "-"):
    """
    Format number with nd decimals. If x is None/NaN/inf, return `dash`.
    Accepts a custom dash to support typographic '—' or '?' placeholders.
    """
    try:
        if x is None:
            return dash
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return dash
        if abs(v - round(v)) < 1e-9 and nd == 0:
            return str(int(round(v)))
        return f"{v:.{nd}f}"
    except Exception:
        return dash

def _fmt_int(x, dash: str = "-"):
    try:
        if x is None:
            return dash
        return f"{int(x):,}"
    except Exception:
        return dash

def _fmt_pct(x, nd=2, dash: str = "-"):
    try:
        if x is None:
            return dash
        return f"{float(x):.{nd}f}%"
    except Exception:
        return dash

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

def _first_sentence(text: str, max_len: int = 80) -> str:
    """
    Return the first sentence-ish fragment (up to max_len).
    Falls back to truncation with ellipsis.
    """
    s = (text or "").strip()
    if not s:
        return ""
    for sep in [". ", "? ", "! "]:
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s

def _pick_factor_lines_from_reason(reason_text: str, max_lines: int = 3) -> List[str]:
    """
    Extract up to N short bullet lines from a multi-line reason/analysis blob.
    Looks for dash/emoji bullets first; otherwise splits by sentences.
    """
    s = (reason_text or "").strip()
    if not s:
        return []
    lines: List[str] = []

    # Prefer existing bullet-like lines
    for line in s.splitlines():
        t = line.strip(" •-—*·\t")
        if not t:
            continue
        if line.lstrip().startswith(("•", "-", "—", "*", "·")):
            lines.append(t)
        if len(lines) >= max_lines:
            break

    if not lines:
        # Fall back to sentences
        tmp = s.replace("! ", ". ").replace("? ", ". ").split(". ")
        for frag in tmp:
            frag = frag.strip()
            if frag:
                lines.append(frag)
            if len(lines) >= max_lines:
                break

    # Trim each to a reasonable length
    out: List[str] = []
    for l in lines[:max_lines]:
        out.append(l if len(l) <= 96 else (l[:95].rstrip() + "…"))
    return out

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

    s_fmt = _fmt_num(strike, 0)
    header = f"{hdr_decision} – {sym} {s_fmt}{right} ({_fmt_num(dte, 0)} DTE, {expiry})"

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

    # ===== Technicals line (RSI, MACD, EMAs, VWAP, ORB) =====
    def _fmt_opt(n, nd=2):
        return _fmt_num(n, nd=nd, dash="—")

    rsi = f.get("rsi14")
    ema20 = f.get("ema20"); ema50 = f.get("ema50"); ema200 = f.get("ema200")
    macd_l = f.get("macd_line"); macd_s = f.get("macd_signal"); macd_h = f.get("macd_hist")
    vwap_val = f.get("vwap_rth") if f.get("vwap_rth") is not None else f.get("vwap")
    orb_hi = f.get("orb15_high"); orb_lo = f.get("orb15_low")

    tech_chunks = []
    if isinstance(rsi, (int, float)):
        tech_chunks.append(f"RSI {round(rsi):d}")
    if isinstance(macd_l, (int, float)) and isinstance(macd_s, (int, float)):
        cross = "L>S" if macd_l > macd_s else ("L<S" if macd_l < macd_s else "L=S")
        hist_tag = None
        if isinstance(macd_h, (int, float)):
            hist_tag = f"{'+' if macd_h>0 else ''}{_fmt_opt(macd_h, 2)}"
        tech_chunks.append(f"MACD {cross}" + (f" ({hist_tag})" if hist_tag is not None else ""))
    if any(isinstance(x, (int,float)) for x in (ema20, ema50, ema200)):
        tech_chunks.append(f"EMA20/50/200 {_fmt_opt(ema20,2)}/{_fmt_opt(ema50,2)}/{_fmt_opt(ema200,2)}")
    if isinstance(vwap_val, (int, float)):
        tech_chunks.append(f"VWAP {_fmt_opt(vwap_val,2)}")
    if isinstance(orb_hi, (int,float)) or isinstance(orb_lo, (int,float)):
        tech_chunks.append(f"ORB {_fmt_opt(orb_hi,2)}/{_fmt_opt(orb_lo,2)}")

    tech_line = "Tech: " + (" • ".join(tech_chunks) if tech_chunks else "—")

    # Liquidity & context
    oi = f.get("oi"); vol = f.get("vol")
    liq_line = f"Liquidity: OI { _fmt_num(oi, 0, dash='?') } / Vol { _fmt_num(vol, 0, dash='?') }"

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
    ctx_line = f"Context: SVR {svr_disp}"

    # Quotes (NBBO)
    spread_pct = f.get("option_spread_pct")
    qage = f.get("quote_age_sec")
    provider = f.get("nbbo_provider") or ("polygon:snapshot" if f.get("bid") is not None or f.get("ask") is not None else None)
    quotes_bits = []
    if f.get("bid") is not None and f.get("ask") is not None:
        quotes_bits.append(f"{_fmt_num(f.get('bid'))}/{_fmt_num(f.get('ask'))} mid {_fmt_num(f.get('mid'))}")
    else:
        quotes_bits.append("-/- mid " + _fmt_num(f.get('mid')))
    if spread_pct is not None:
        quotes_bits.append(f"spread {_fmt_pct(spread_pct, 2)}")
    if isinstance(qage, (int, float)):
        quotes_bits.append(f"age {int(qage)}s")
    if provider:
        quotes_bits.append(provider)
    quotes_line = "NBBO: " + " • ".join(quotes_bits)

    conf_line = f"Confidence: { _fmt_num(conf, 2, dash='—') }"
    sr_line = ""
    if score is not None or rating is not None:
        sr_line = f"Score: { _fmt_num(score, 2, dash='—') }"
        if rating:
            sr_line += f" | Rating: {rating}"

    appendix = diff_note.strip()
    if appendix:
        appendix = "\n" + appendix

    out_lines = [
        header,
        bias_line or None,
        "🕒 Suggested trade: " + (horizon or "—") + (f" — {summary_line}" if summary_line else ""),
        tech_line,
        liq_line,
        ctx_line,
        quotes_line,
        conf_line + ((" | " + sr_line) if sr_line else ""),
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
