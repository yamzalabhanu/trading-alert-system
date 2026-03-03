# engine_common.py
import os
import re
import json
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, date

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


IBKR_ENABLED = False
IBKR_DEFAULT_QTY = 0
IBKR_TIF = "DAY"
IBKR_ORDER_MODE = "disabled"
IBKR_USE_MID_AS_LIMIT = False

TARGET_DELTA_CALL = float(os.getenv("TARGET_DELTA_CALL", "0.35"))
TARGET_DELTA_PUT = float(os.getenv("TARGET_DELTA_PUT", "-0.35"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "6.0"))
MAX_QUOTE_AGE_S = float(os.getenv("MAX_QUOTE_AGE_S", "30"))
MIN_VOL_TODAY = int(os.getenv("MIN_VOL_TODAY", "100"))
MIN_OI = int(os.getenv("MIN_OI", "200"))
MIN_DTE = int(os.getenv("MIN_DTE", "3"))
MAX_DTE = int(os.getenv("MAX_DTE", "45"))

SCAN_MIN_VOL_RTH = int(os.getenv("SCAN_MIN_VOL_RTH", os.getenv("SCAN_MIN_VOL", "500")))
SCAN_MIN_OI_RTH = int(os.getenv("SCAN_MIN_OI_RTH", os.getenv("SCAN_MIN_OI", "500")))
SCAN_MIN_VOL_AH = int(os.getenv("SCAN_MIN_VOL_AH", "0"))
SCAN_MIN_OI_AH = int(os.getenv("SCAN_MIN_OI_AH", "100"))

SEND_CHAIN_SCAN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_ALERTS", "0"))
SEND_CHAIN_SCAN_TOPN_ALERTS = _env_truthy(os.getenv("SEND_CHAIN_SCAN_TOPN_ALERTS", "0"))
REPLACE_IF_NO_NBBO = _env_truthy(os.getenv("REPLACE_IF_NO_NBBO", "1"))

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

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_NA_TOKEN_RE = re.compile(r":\s*na(\s*[,}])", re.IGNORECASE)
_DOUBLE_COMMA_RE = re.compile(r",\s*,+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _extract_json_object(s: str) -> Optional[str]:
    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a : b + 1]


def _maybe_unwrap_json_string(text: str) -> str:
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
    s = (s or "").strip()
    o = s.count("{")
    c = s.count("}")
    if o > c:
        s = s + ("}" * (o - c))
    return s


def _salvage_json_text(s: str) -> str:
    s = (s or "").strip()
    obj = _extract_json_object(s)
    if obj is not None:
        s = obj

    s = _CONTROL_CHARS_RE.sub("", s)
    s = _NA_TOKEN_RE.sub(r": null\1", s)

    while True:
        s2 = _DOUBLE_COMMA_RE.sub(",", s)
        if s2 == s:
            break
        s = s2

    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


async def get_alert_text_from_request(request: Request) -> str:
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


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except Exception:
        return None


def _norm_side(side_raw: str) -> Optional[str]:
    s = (side_raw or "").upper().strip()
    if s in ("CALL", "CALLS", "BUY", "LONG", "BULL"):
        return "CALL"
    if s in ("PUT", "PUTS", "SELL", "SHORT", "BEAR"):
        return "PUT"
    return None


def _next_friday(d: date) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7)


def same_week_friday(d: date) -> date:
    base_monday = d - timedelta(days=d.weekday())
    return base_monday + timedelta(days=4)


def two_weeks_friday(d: date) -> date:
    return _next_friday(d) + timedelta(days=7)


def _expiry_from_mode(mode: str, *, ref_dt: Optional[datetime] = None) -> Optional[str]:
    m = (mode or "").strip().lower()
    now = (ref_dt or market_now()).date()

    if not m:
        return None

    if "next friday" in m:
        return _next_friday(now).isoformat()
    if "same week" in m and "friday" in m:
        return same_week_friday(now).isoformat()
    if "two weeks" in m and "friday" in m:
        return two_weeks_friday(now).isoformat()

    # direct ISO date?
    if re.match(r"^\d{4}-\d{2}-\d{2}$", mode.strip()):
        return mode.strip()

    return None


def _time_to_iso(payload_time: Any) -> Optional[str]:
    """
    Accept TradingView 'time' as:
      - epoch millis string/number: "1709857200000"
      - epoch seconds: 1709857200
      - already ISO string
    """
    if payload_time is None:
        return None

    if isinstance(payload_time, str) and re.match(r"^\d{10,13}$", payload_time.strip()):
        n = int(payload_time.strip())
        if n > 10_000_000_000:  # millis
            dt = datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        else:  # seconds
            dt = datetime.fromtimestamp(n, tz=timezone.utc)
        return dt.isoformat()

    if isinstance(payload_time, (int, float)):
        n = int(payload_time)
        if n > 10_000_000_000:
            dt = datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(n, tz=timezone.utc)
        return dt.isoformat()

    if isinstance(payload_time, str):
        s = payload_time.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
            return s

    return None


def _parse_alert_json_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalizes multiple webhook schemas into the engine's internal alert shape.

    Supports your new TradingView JSON:
      {
        "src":"tv_pine",
        "symbol":"SPY",
        "optionType":"PUTS",
        "price":512.34,
        "time":"1709857200000",
        "optionsHint": { "strike":512, "expiryMode":"Next Friday" },
        ...
      }
    """
    symbol = str(payload.get("symbol") or payload.get("ticker") or payload.get("underlying") or "").upper().strip()
    if not symbol:
        return None

    # side from either old keys or your new key "optionType"
    side = _norm_side(str(payload.get("side") or payload.get("type") or payload.get("signal") or payload.get("optionType") or ""))
    px = _as_float(payload.get("price") if payload.get("price") is not None else payload.get("last"))

    if not side or px is None:
        return None

    out: Dict[str, Any] = {
        "side": side,
        "symbol": symbol,
        "underlying_price_from_alert": px,
        "src": payload.get("src") or "webhook",
    }

    # --- strike/expiry: support direct + optionsHint ---
    strike = _as_float(payload.get("strike"))
    expiry = str(payload.get("expiry") or "").strip()

    options_hint = payload.get("optionsHint") if isinstance(payload.get("optionsHint"), dict) else {}
    if strike is None and isinstance(options_hint, dict):
        strike = _as_float(options_hint.get("strike"))
    if (not expiry) and isinstance(options_hint, dict):
        expiry = str(_expiry_from_mode(str(options_hint.get("expiryMode") or "")) or "").strip()

    if strike is not None:
        out["strike"] = strike
    if re.match(r"^\d{4}-\d{2}-\d{2}$", expiry):
        out["expiry"] = expiry

    # --- normalize TV time ---
    t_iso = _time_to_iso(payload.get("time"))
    if t_iso:
        out["time_iso"] = t_iso
    if payload.get("time") is not None:
        out["time"] = payload.get("time")

    # --- keep important TV fields for LLM + Telegram ---
    passthru_keys = [
        "bias", "tf", "score", "tier",
        "vwap", "ema9", "ema21", "adx", "volSpike",
        "orbH", "orbL", "orbRange", "atr",
        "sl", "tp", "rr",
        "mtfMode", "mtfExecTF", "mtfConfTF",
        "mtfExecLong", "mtfConfLong", "mtfExecShort", "mtfConfShort",
        "optionsHint",
    ]
    for k in passthru_keys:
        if k in payload:
            out[k] = payload.get(k)

    # Back-compat meta keys your engine might already use
    for k in ("source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level"):
        if payload.get(k) is not None:
            out[k] = payload.get(k)

    if out.get("strike") is None and not ALLOW_NO_STRIKE_JSON:
        return None

    return out


def parse_alert_text(text: str) -> Dict[str, Any]:
    text = _maybe_unwrap_json_string(text)
    text = (text or "").strip()

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

    if "{" in text:
        try:
            fixed = _salvage_json_text(_balance_braces(text))
            payload = json.loads(fixed)

            if isinstance(payload, str):
                payload = json.loads(_salvage_json_text(_balance_braces(payload)))

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
# Telegram composition
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
    sym = str(alert.get("symbol") or alert.get("ticker") or "").upper()
    side = str(alert.get("side") or "").upper()

    # Prefer TV schema fields if present
    bias = str(alert.get("bias") or "").upper().strip()
    tier = str(alert.get("tier") or "").upper().strip()
    tv_score = alert.get("score")

    decision = str(llm.get("decision") or "wait").upper()
    conf = llm.get("confidence")
    reason = (llm.get("reason") or llm_reason or "").strip()

    ul_px = alert.get("underlying_price_from_alert") or alert.get("price") or f.get("last") or f.get("mid")
    strike = alert.get("strike")
    expiry = alert.get("expiry")

    hdr_bits = [decision, sym, side]
    if bias:
        hdr_bits.append(f"({bias})")
    if tier:
        hdr_bits.append(f"[{tier}]")
    if rating:
        hdr_bits.append(f"[{rating}]")
    header = " ".join([x for x in hdr_bits if x])

    contract_bits = [f"Price: {_fmt(ul_px, 2)}"]
    if strike is not None:
        contract_bits.append(f"Strike: {_fmt(strike, 2)}")
    if expiry:
        contract_bits.append(f"Exp: {expiry}")
    if option_ticker:
        contract_bits.append(f"OCC: {option_ticker}")
    contract_line = " | ".join(contract_bits)

    score_line_bits = []
    if tv_score is not None:
        score_line_bits.append(f"TVScore: {_fmt(tv_score, 0)}")
    if score is not None:
        score_line_bits.append(f"Score: {_fmt(score, 1)}")
    if conf is not None:
        score_line_bits.append(f"Conf: {_fmt(conf, 2)}")
    score_line = " | ".join(score_line_bits) if score_line_bits else ""

    # add TV indicators if present
    tv_bits = []
    for k, label, nd in [
        ("vwap", "VWAP", 2),
        ("ema9", "EMA9", 2),
        ("ema21", "EMA21", 2),
        ("adx", "ADX", 1),
        ("atr", "ATR", 2),
    ]:
        if alert.get(k) is not None:
            tv_bits.append(f"{label}: {_fmt(float(alert.get(k)), nd)}")
    if isinstance(alert.get("volSpike"), bool):
        tv_bits.append(f"VolSpike: {'YES' if alert.get('volSpike') else 'NO'}")
    tv_line = " | ".join(tv_bits) if tv_bits else ""

    risk_bits = []
    for k, label in [("sl", "SL"), ("tp", "TP"), ("rr", "RR")]:
        if alert.get(k) is not None:
            risk_bits.append(f"{label}: {_fmt(alert.get(k), 2)}")
    risk_line = " | ".join(risk_bits) if risk_bits else ""

    note = diff_note.strip()
    if note:
        note = f"\n{note}"

    lines = [header, contract_line]
    if score_line:
        lines.append(score_line)
    if tv_line:
        lines.append(tv_line)
    if risk_line:
        lines.append(risk_line)
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
