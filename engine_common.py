# engine_common.py
import os
import re
import math
import json
from typing import List
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta, date

from fastapi import HTTPException, Request
from config import CDT_TZ, MAX_LLM_PER_DAY, POLYGON_API_KEY as CFG_POLYGON_API_KEY

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
POLYGON_API_KEY = CFG_POLYGON_API_KEY


def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


IBKR_ENABLED = False  # IBKR integration removed
IBKR_DEFAULT_QTY = 0
IBKR_TIF = "DAY"
IBKR_ORDER_MODE = "disabled"
IBKR_USE_MID_AS_LIMIT = False

# Trading thresholds (tunable)
TARGET_DELTA_CALL = float(os.getenv("TARGET_DELTA_CALL", "0.35"))
TARGET_DELTA_PUT = float(os.getenv("TARGET_DELTA_PUT", "-0.35"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "6.0"))
MAX_QUOTE_AGE_S = float(os.getenv("MAX_QUOTE_AGE_S", "30"))
MIN_VOL_TODAY = int(os.getenv("MIN_VOL_TODAY", "100"))
MIN_OI = int(os.getenv("MIN_OI", "200"))
MIN_DTE = int(os.getenv("MIN_DTE", "3"))
MAX_DTE = int(os.getenv("MAX_DTE", "45"))

# Optional scan knobs (RTH vs AH)
SCAN_MIN_VOL_RTH = int(os.getenv("SCAN_MIN_VOL_RTH", os.getenv("SCAN_MIN_VOL", "500")))
SCAN_MIN_OI_RTH = int(os.getenv("SCAN_MIN_OI_RTH", os.getenv("SCAN_MIN_OI", "500")))
SCAN_MIN_VOL_AH = int(os.getenv("SCAN_MIN_VOL_AH", "0"))
SCAN_MIN_OI_AH = int(os.getenv("SCAN_MIN_OI_AH", "100"))

# Keep these env toggles defined
SEND_CHAIN_SCAN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_ALERTS", "0"))
SEND_CHAIN_SCAN_TOPN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_TOPN_ALERTS", "0"))
REPLACE_IF_NO_NBBO = _env_truthy(os.getenv("REPLACE_IF_NO_NBBO", "1"))

# Allow enriched JSON alerts without strike/expiry (equity-only ok)
ALLOW_NO_STRIKE_JSON = _env_truthy(os.getenv("ALLOW_NO_STRIKE_JSON", "1"))

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

# -----------------------------
# Robust JSON salvage helpers
# -----------------------------
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_NA_TOKEN_RE = re.compile(r":\s*na(\s*[,}])", re.IGNORECASE)  # :na, or :na}
_DOUBLE_COMMA_RE = re.compile(r",\s*,+")  # ",," or ", ,"
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # invalid JSON control chars


def _extract_json_object(s: str) -> Optional[str]:
    """Return the outermost {...} slice if present, else None."""
    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a : b + 1]


def _maybe_unwrap_json_string(text: str) -> str:
    """
    If the payload is a JSON string containing JSON (double-encoded),
    decode it once:
      "\"{\\\"a\\\":1}\""  ->  "{\"a\":1}"
    """
    t = (text or "").strip()
    if (len(t) >= 2) and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
        try:
            inner = json.loads(t)
            if isinstance(inner, str):
                return inner.strip()
        except Exception:
            pass
    return t


def _balance_braces(s: str) -> str:
    """
    Best-effort fix for truncated JSON missing one or more closing braces.
    Only used as a last-resort before json.loads().
    """
    s = (s or "").strip()
    o = s.count("{")
    c = s.count("}")
    if o > c:
        s = s + ("}" * (o - c))
    return s


def _salvage_json_text(s: str) -> str:
    """
    Fix common TradingView/Pine JSON issues:
      - extra prefix/suffix text around JSON
      - Pine 'na' token => JSON null
      - trailing commas before } or ]
      - accidental double commas:  "tp1":null,, "x":1
      - stray control chars
    """
    s = (s or "").strip()
    obj = _extract_json_object(s)
    if obj is not None:
        s = obj

    s = _CONTROL_CHARS_RE.sub("", s)
    s = _NA_TOKEN_RE.sub(r": null\1", s)

    # Fix double commas anywhere
    while True:
        s2 = _DOUBLE_COMMA_RE.sub(",", s)
        if s2 == s:
            break
        s = s2

    # Remove trailing commas before closing braces/brackets
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


async def get_alert_text_from_request(request: Request) -> str:
    """
    Returns:
      - plain text if client posted text/plain
      - a JSON string if client posted application/json
    """
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            msg = str(data.get("message") or data.get("alert") or data.get("text") or "").strip()
            if msg:
                return msg
            return json.dumps(data, separators=(",", ":"))
        body = await request.body()
        return body.decode("utf-8").strip()
    except Exception:
        return ""


def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _norm_side(side_raw: str) -> Optional[str]:
    s = (side_raw or "").upper().strip()
    if s in ("CALL", "BUY", "LONG", "BULL"):
        return "CALL"
    if s in ("PUT", "SELL", "SHORT", "BEAR"):
        return "PUT"
    return None


def _parse_alert_json_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Support: original schema + scalper_v4_2 schema
    side = _norm_side(str(payload.get("side") or payload.get("type") or payload.get("signal") or ""))
    symbol = str(payload.get("symbol") or payload.get("ticker") or payload.get("underlying") or "").upper().strip()

    px = _as_float(payload.get("price") if payload.get("price") is not None else payload.get("last"))
    strike = _as_float(payload.get("strike"))
    expiry = str(payload.get("expiry") or "").strip()

    # must have at least side+symbol+price
    if not side or not symbol or px is None:
        return None

    out: Dict[str, Any] = {
        "side": side,
        "symbol": symbol,
        "underlying_price_from_alert": px,
    }

    # Optional options fields
    if strike is not None:
        out["strike"] = strike
    if re.match(r"^\d{4}-\d{2}-\d{2}$", expiry):
        out["expiry"] = expiry

    # IMPORTANT: keep meta keys in the same names your processor/llm expect
    for k in ("source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level"):
        if payload.get(k) is not None:
            out[k] = payload.get(k)

    # pass-through other useful fields (if present)
    for k in ("adx", "relVol", "relvol", "chop", "tp1", "tp2", "tp3", "trail", "fast_stop", "ason", "bp", "ats"):
        if k in payload:
            out[k] = payload.get(k)

    if out.get("strike") is None and not ALLOW_NO_STRIKE_JSON:
        return None

    return out


def parse_alert_text(text: str) -> Dict[str, Any]:
    # 0) unwrap if the payload is a JSON string containing JSON (double-encoded)
    text = _maybe_unwrap_json_string(text)
    text = (text or "").strip()

    # 1) Plain-text patterns first
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

    # 2) Accept JSON payloads from TradingView/Pine webhook directly (with salvage).
    # NOTE: don't require '}' because some payloads are double-encoded or truncated in logs.
    if "{" in text:
        try:
            fixed = _salvage_json_text(_balance_braces(text))
            payload = json.loads(fixed)

            # If we got a JSON string (double-encoded) even after unwrap, decode once more.
            if isinstance(payload, str):
                payload2 = json.loads(_salvage_json_text(_balance_braces(payload)))
                payload = payload2

            if isinstance(payload, dict):
                parsed = _parse_alert_json_payload(payload)
                if parsed:
                    return parsed
        except Exception as e:
            logger.debug("JSON parse failed: %s", e)

    raise HTTPException(status_code=400, detail="Unrecognized alert format")


# =========================
# Misc utils
# =========================
def round_strike_to_common_increment(val: float) -> float:
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
    base_monday = d - timedelta(days=d.weekday())
    return base_monday + timedelta(days=4)


def two_weeks_friday(d: date) -> date:
    return _next_friday(d) + timedelta(days=7)


def is_same_week(a: date, b: date) -> bool:
    am = a - timedelta(days=a.weekday())
    bm = b - timedelta(days=b.weekday())
    return am == bm


def _encode_ticker_path(t: str) -> str:
    return quote(t or "", safe="")


def _is_rth_now() -> bool:
    now = datetime.now(CDT_TZ)
    if now.weekday() > 4:
        return False
    start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    end = now.replace(hour=15, minute=0, second=0, microsecond=0)
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
    has_nbbo = f.get("bid") is not None and f.get("ask") is not None
    has_last = isinstance(f.get("last"), (int, float)) or isinstance(f.get("mid"), (int, float))

    if rth:
        checks["quote_fresh"] = (quote_age is not None and quote_age <= MAX_QUOTE_AGE_S and has_nbbo)
        checks["spread_ok"] = (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= MAX_SPREAD_PCT)
    else:
        checks["quote_fresh"] = bool(has_last)
        checks["spread_ok"] = True

    require_liquidity = os.getenv("REQUIRE_LIQUIDITY_FIELDS", "0") == "1"
    if rth and require_liquidity:
        checks["vol_ok"] = (f.get("vol") or 0) >= MIN_VOL_TODAY
        checks["oi_ok"] = (f.get("oi") or 0) >= MIN_OI
    else:
        checks["vol_ok"] = True
        checks["oi_ok"] = True

    dte_val = f.get("dte")
    checks["dte_ok"] = (dte_val is not None) and (MIN_DTE <= dte_val <= MAX_DTE)

    ok = all(checks.values())
    return ok, checks


# =========================
# Telegram composition (restored)
# =========================
def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)
    except Exception:
        return str(v)


def _fmt_pct(v: Any, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}%"
    except Exception:
        return "n/a"


def compose_telegram_text(
    *,
    alert: Dict[str, Any],
    option_ticker: Optional[str],
    f: Dict[str, Any],
    llm: Dict[str, Any],
    llm_ran: bool,
    llm_reason: str,
    score: Optional[float] = None,
    rating: Optional[str] = None,
    diff_note: str = "",
) -> str:
    """
    Backward-compatible Telegram composer expected by engine_logic/engine_processor.

    Works for:
      - option alerts (strike/expiry present)
      - equity-only alerts (strike/expiry missing)
      - enriched TV JSON (event/model/reason/etc.)
    """
    sym = str(alert.get("symbol") or alert.get("ticker") or "").upper()
    side = str(alert.get("side") or "").upper()
    event = str(alert.get("event") or alert.get("alert_event") or "").strip().lower()
    model = str(alert.get("model") or alert.get("alert_model") or "").strip()

    ul_px = alert.get("underlying_price_from_alert") or alert.get("price") or f.get("last") or f.get("mid")
    strike = alert.get("strike")
    expiry = alert.get("expiry")

    decision = str(llm.get("decision") or "wait").upper()
    conf = llm.get("confidence")
    reason = (llm.get("reason") or llm_reason or "").strip()

    # Core header
    hdr_bits = [decision, sym, side]
    if event:
        hdr_bits.append(f"({event})")
    if rating:
        hdr_bits.append(f"[{rating}]")
    header = " ".join([x for x in hdr_bits if x])

    # Price/contract line
    contract_bits = [f"Price: {_fmt(ul_px, 2)}"]
    if strike is not None:
        contract_bits.append(f"Strike: {_fmt(strike, 2)}")
    if expiry:
        contract_bits.append(f"Exp: {expiry}")
    if option_ticker:
        contract_bits.append(f"OCC: {option_ticker}")
    contract_line = " | ".join(contract_bits)

    # Market/TA snippets (best effort)
    last_ = f.get("last", ul_px)
    bid = f.get("bid")
    ask = f.get("ask")
    mid = f.get("mid")
    spr = f.get("option_spread_pct")
    chg = f.get("quote_change_pct")
    rsi = f.get("rsi14")
    ema20, ema50 = f.get("ema20"), f.get("ema50")
    vwap = f.get("vwap")
    vwap_dist = f.get("vwap_dist")
    orb_h, orb_l = f.get("orb15_high"), f.get("orb15_low")
    mtf = f.get("mtf_align")
    regime = f.get("regime_flag")

    mkt_parts = []
    if bid is not None and ask is not None:
        mkt_parts.append(f"Bid/Ask: {_fmt(bid,2)}/{_fmt(ask,2)}")
    if mid is not None:
        mkt_parts.append(f"Mid: {_fmt(mid,2)}")
    if spr is not None:
        mkt_parts.append(f"Spr: {_fmt_pct(spr,2)}")
    if chg is not None:
        mkt_parts.append(f"Δ%: {_fmt(chg,2)}")
    mkt_line = " | ".join(mkt_parts) if mkt_parts else ""

    ta_parts = []
    if rsi is not None:
        ta_parts.append(f"RSI14: {_fmt(rsi,1)}")
    if ema20 is not None and ema50 is not None and last_ is not None:
        ta_parts.append(f"EMA20/50: {_fmt(ema20,2)}/{_fmt(ema50,2)}")
    if vwap is not None:
        ta_parts.append(f"VWAP: {_fmt(vwap,2)}")
    if vwap_dist is not None:
        ta_parts.append(f"VWAPΔ: {_fmt(vwap_dist,2)}%")
    if orb_h is not None or orb_l is not None:
        ta_parts.append(f"ORB15 H/L: {_fmt(orb_h,2)}/{_fmt(orb_l,2)}")
    if isinstance(mtf, bool):
        ta_parts.append(f"MTF: {'OK' if mtf else 'NO'}")
    if regime:
        ta_parts.append(f"Regime: {regime}")
    ta_line = " | ".join(ta_parts) if ta_parts else ""

    # Score line (optional)
    score_line = ""
    if score is not None or conf is not None or model:
        bits = []
        if score is not None:
            bits.append(f"Score: {_fmt(score,1)}")
        if conf is not None:
            bits.append(f"Conf: {_fmt(conf,2)}")
        if model:
            bits.append(f"Model: {model}")
        score_line = " | ".join(bits)

    note = diff_note.strip()
    if note:
        note = f"\n{note}"

    lines = [header, contract_line]
    if score_line:
        lines.append(score_line)
    if mkt_line:
        lines.append(mkt_line)
    if ta_line:
        lines.append(ta_line)
    if reason:
        lines.append(reason)

    return ("\n".join([ln for ln in lines if ln.strip()])).strip() + note


__all__ = [
    "POLYGON_API_KEY",
    "IBKR_ENABLED",
    "IBKR_DEFAULT_QTY",
    "IBKR_TIF",
    "IBKR_ORDER_MODE",
    "IBKR_USE_MID_AS_LIMIT",
    "TARGET_DELTA_CALL",
    "TARGET_DELTA_PUT",
    "MAX_SPREAD_PCT",
    "MAX_QUOTE_AGE_S",
    "MIN_VOL_TODAY",
    "MIN_OI",
    "MIN_DTE",
    "MAX_DTE",
    "SCAN_MIN_VOL_RTH",
    "SCAN_MIN_OI_RTH",
    "SCAN_MIN_VOL_AH",
    "SCAN_MIN_OI_AH",
    "SEND_CHAIN_SCAN_ALERTS",
    "SEND_CHAIN_SCAN_TOPN_ALERTS",
    "REPLACE_IF_NO_NBBO",
    "ALLOW_NO_STRIKE_JSON",
    "market_now",
    "llm_quota_snapshot",
    "consume_llm",
    "get_alert_text_from_request",
    "parse_alert_text",
    "round_strike_to_common_increment",
    "_next_friday",
    "same_week_friday",
    "two_weeks_friday",
    "is_same_week",
    "_encode_ticker_path",
    "_is_rth_now",
    "_occ_meta",
    "_ticker_matches_side",
    "preflight_ok",
    "compose_telegram_text",  
]
