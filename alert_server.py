# alert_server.py-backup 08/19
import os
import re
import json
import math
import contextlib
from collections import deque, Counter
from datetime import datetime, timezone, date, timedelta, time as dt_time
from typing import Any, Dict, Optional, List, Tuple
from zoneinfo import ZoneInfo
import asyncio

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from openai import OpenAI

# =======================
# Configuration / Knobs
# =======================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Existing LLM budget & volume/OI gate
MAX_LLM_PER_DAY     = int(os.getenv("MAX_LLM_PER_DAY", "50"))
VOLUME_MIN_FOR_LLM  = int(os.getenv("VOLUME_MIN_FOR_LLM", "500"))
OI_MIN_FOR_LLM      = int(os.getenv("OI_MIN_FOR_LLM", "1000"))

# Cooldown + daily report
COOLDOWN_SECONDS    = int(os.getenv("COOLDOWN_SECONDS", "600"))  # 10m
MARKET_TZ           = os.getenv("MARKET_TZ", "America/New_York")
REPORT_HHMM         = os.getenv("REPORT_HHMM", "16:15")          # 4:15 PM ET

# -------- New enhancement knobs --------
# Liquidity/microstructure gates
OPT_SPREAD_MAX_PCT  = float(os.getenv("OPT_SPREAD_MAX_PCT", "0.12"))  # 12%
QUOTE_AGE_MAX_SEC   = int(os.getenv("QUOTE_AGE_MAX_SEC", "10"))       # 10 seconds

# Trade style (affects bands & risk plan)
TRADE_STYLE         = os.getenv("TRADE_STYLE", "intraday").lower()    # intraday|swing
# Delta/DTE bands
DELTA_MIN_INTRADAY  = float(os.getenv("DELTA_MIN_INTRADAY", "0.35"))
DELTA_MAX_INTRADAY  = float(os.getenv("DELTA_MAX_INTRADAY", "0.55"))
DTE_MIN_INTRADAY    = int(os.getenv("DTE_MIN_INTRADAY",   "3"))
DTE_MAX_INTRADAY    = int(os.getenv("DTE_MAX_INTRADAY",   "10"))

DELTA_MIN_SWING     = float(os.getenv("DELTA_MIN_SWING",  "0.25"))
DELTA_MAX_SWING     = float(os.getenv("DELTA_MAX_SWING",  "0.45"))
DTE_MIN_SWING       = int(os.getenv("DTE_MIN_SWING",      "7"))
DTE_MAX_SWING       = int(os.getenv("DTE_MAX_SWING",      "21"))

# EM vs Break-even
EM_VS_BE_RATIO_MIN  = float(os.getenv("EM_VS_BE_RATIO_MIN", "0.80"))

# S/R headroom in ATR units (daily ATR(14))
HEADROOM_MIN_R      = float(os.getenv("HEADROOM_MIN_R", "1.0"))

# Hybrid pre-score thresholds
PRESCORE_AUTO_BUY   = int(os.getenv("PRESCORE_AUTO_BUY", "80"))
PRESCORE_AUTO_SKIP  = int(os.getenv("PRESCORE_AUTO_SKIP","50"))

# LLM morning window (default: allow until 11:30 AM CST)
LLM_CUTOFF_TZ    = os.getenv("LLM_CUTOFF_TZ", "America/Chicago")
LLM_CUTOFF_HHMM  = os.getenv("LLM_CUTOFF_HHMM", "10:30")  # inclusive cutoff

# Market regime thresholds
SPY_ATR_PCT_DAYS    = int(os.getenv("SPY_ATR_PCT_DAYS", "14"))
REGIME_TREND_EMA_FAST = int(os.getenv("REGIME_TREND_EMA_FAST", "9"))
REGIME_TREND_EMA_SLOW = int(os.getenv("REGIME_TREND_EMA_SLOW", "21"))

# IV rank history (per symbol) kept in-memory
IV_HISTORY_LEN      = int(os.getenv("IV_HISTORY_LEN", "120"))  # ~6 months of trading days

# Optional simple macro date blocklist (comma-separated YYYY-MM-DD)
MACRO_EVENT_DATES   = [d.strip() for d in os.getenv("MACRO_EVENT_DATES", "").split(",") if d.strip()]

# =================================
# Safety: Required API keys present
# =================================
if not POLYGON_API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# =================================
# App & API Clients
# =================================
oai_client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="TradingView Options Alert Ingestor + Telegram (Enhanced)")

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")

# ==========
# Regex I/O
# ==========
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

# =======================
# State (in-memory)
# =======================
# LLM quota per UTC day
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}

# cooldown: (symbol, side) -> last_processed_utc
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

# daily decisions log (for report & future calibration)
_DECISIONS_LOG: List[Dict[str, Any]] = []

# IV history per symbol (for IV rank)
_IV_HISTORY: Dict[str, deque] = {}

# =======================
# Small utilities
# =======================
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

def market_now() -> datetime:
    return datetime.now(ZoneInfo(MARKET_TZ))

def same_week_friday(today: date) -> date:
    # Monday=0 ... Friday=4
    offset = 4 - today.weekday()
    if offset < 0:  # Sat/Sun -> next Friday
        offset += 7
    return today + timedelta(days=offset)

def round_strike_to_common_increment(strike: float) -> float:
    return round(strike * 2) / 2.0  # nearest 0.5

def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

def _is_within_llm_window(now_utc: datetime) -> bool:
    """Return True if current local time in LLM_CUTOFF_TZ is <= LLM_CUTOFF_HHMM."""
    tz = ZoneInfo(LLM_CUTOFF_TZ)
    hh, mm = _parse_hhmm(LLM_CUTOFF_HHMM)
    now_local = now_utc.astimezone(tz)
    cutoff_local = datetime.combine(now_local.date(), dt_time(hour=hh, minute=mm), tzinfo=tz)
    # "till 11:30" => inclusive
    return now_local <= cutoff_local

def _next_report_dt_utc(now_utc: datetime) -> datetime:
    tz = ZoneInfo(MARKET_TZ)
    hh, mm = _parse_hhmm(REPORT_HHMM)
    now_local = now_utc.astimezone(tz)
    target_local = datetime.combine(now_local.date(), dt_time(hour=hh, minute=mm), tzinfo=tz)
    if target_local <= now_local:
        target_local += timedelta(days=1)
    return target_local.astimezone(timezone.utc)

def _fmt(val):
    return "—" if val is None else (f"{val:.4f}" if isinstance(val, float) else str(val))

# =======================
# Polygon API helpers
# =======================
async def _poly_get(client: httpx.AsyncClient, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Tolerant GET: soft-handle 402/403/404/429 to avoid 500s."""
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY
    r = await client.get(f"https://api.polygon.io{path}", params=p, timeout=20.0)
    if r.status_code in (402, 403, 404, 429):
        return {}
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {}

async def polygon_list_contracts_for_expiry(client: httpx.AsyncClient, *, symbol: str, expiry: str, side: str, limit: int = 250) -> List[Dict[str, Any]]:
    js = await _poly_get(client, "/v3/reference/options/contracts", {
        "underlying_ticker": symbol,
        "expiration_date": expiry,
        "contract_type": "call" if side == "CALL" else "put",
        "limit": limit,
    })
    return (js or {}).get("results", []) or []

async def polygon_get_option_snapshot(client: httpx.AsyncClient, *, underlying: str, option_ticker: str) -> Dict[str, Any]:
    return await _poly_get(client, f"/v3/snapshot/options/{underlying}/{option_ticker}", {})

async def polygon_aggs(client: httpx.AsyncClient, *, ticker: str, multiplier: int, timespan: str,
                       frm: str, to: str, limit: int = 50000, sort: str = "asc") -> List[Dict[str, Any]]:
    js = await _poly_get(client, f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{frm}/{to}", {
        "adjusted": "true",
        "sort": sort,
        "limit": limit,
    })
    return (js or {}).get("results", []) or []

# =======================
# Technical calcs
# =======================
def ema(values: List[float], period: int) -> Optional[float]:
    if not values or period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e

def atr14(daily: List[Dict[str, Any]]) -> Optional[float]:
    if len(daily) < 15:
        return None
    bars = sorted(daily, key=lambda x: x["t"])  # ascending
    trs = []
    prev_close = bars[0]["c"]
    for b in bars[1:]:
        tr = max(b["h"] - b["l"], abs(b["h"] - prev_close), abs(b["l"] - prev_close))
        trs.append(tr)
        prev_close = b["c"]
    if len(trs) < 14:
        return None
    return sum(trs[-14:]) / 14.0

def realized_vol_annualized(daily_closes: List[float], window: int = 20) -> Optional[float]:
    if len(daily_closes) < window + 1:
        return None
    import statistics
    rets = []
    for i in range(1, window + 1):
        r = math.log(daily_closes[-i] / daily_closes[-i - 1])
        rets.append(r)
    stdev = statistics.pstdev(rets)
    return stdev * math.sqrt(252)

def iv_rank_from_history(hist: deque, current_iv: Optional[float]) -> Optional[float]:
    if current_iv is None or not hist:
        return None
    xs = list(hist)
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return 0.0
    return (current_iv - lo) / (hi - lo)

def pick_nearest_strike(contracts: List[Dict[str, Any]], desired_strike: float) -> Optional[Dict[str, Any]]:
    best = None
    best_diff = float("inf")
    for c in contracts:
        sp = c.get("strike_price")
        if sp is None:
            continue
        diff = abs(float(sp) - desired_strike)
        if diff < best_diff:
            best = c
            best_diff = diff
    return best

def dte(expiry: str) -> int:
    y, m, d = map(int, expiry.split("-"))
    return (date(y, m, d) - datetime.now(timezone.utc).date()).days

# =======================
# Feature engineering
# =======================
async def build_features(client: httpx.AsyncClient, *, alert: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Builds all features needed for pre-score & LLM context."""
    res = snapshot.get("results") or {}
    greeks = res.get("greeks") or {}
    ua = res.get("underlying_asset") or {}
    day = res.get("day") or {}
    last_quote = res.get("last_quote") or {}
    last_trade = res.get("last_trade") or {}

    S = ua.get("price") or alert["underlying_price_from_alert"]
    cur_iv = res.get("implied_volatility")  # annualized (fraction)
    oi = int(res.get("open_interest") or 0)
    vol = int(day.get("volume") or 0)

    # NBBO spread% & quote age
    bid = last_quote.get("bid_price")
    ask = last_quote.get("ask_price")
    mid = None
    opt_spread_pct = None
    if (bid is not None) and (ask is not None) and (bid > 0) and (ask > 0):
        mid = (bid + ask) / 2.0
        if mid > 0:
            opt_spread_pct = (ask - bid) / mid
    if mid is None:
        mid = last_trade.get("price")

    quote_ts_ns = last_quote.get("sip_timestamp") or last_quote.get("last_updated")
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    quote_age_sec = None
    if isinstance(quote_ts_ns, (int, float)) and quote_ts_ns > 0:
        quote_age_sec = max(0, (now_ns - int(quote_ts_ns)) / 1e9)

    # DTE and delta
    days_to_exp = dte(alert["expiry"])
    delta = greeks.get("delta")

    # Daily context: ~70 calendar days for ATR, RV, 20d hi/lo
    to_day = datetime.now(timezone.utc).date()
    frm_day = (to_day - timedelta(days=70)).isoformat()
    to_iso = to_day.isoformat()
    daily = await polygon_aggs(client, ticker=alert["symbol"], multiplier=1, timespan="day",
                               frm=frm_day, to=to_iso, sort="asc")
    atr = atr14(daily) if daily else None
    closes = [b["c"] for b in daily] if daily else []
    rv20 = realized_vol_annualized(closes, window=20) if closes else None
    high20 = max((b["h"] for b in daily[-20:]), default=None) if daily else None
    low20  = min((b["l"] for b in daily[-20:]), default=None) if daily else None

    # Headroom: use 20d extremes as S/R approximation
    sr_headroom_ok = None
    if atr and S and high20 and low20:
        if alert["side"] == "CALL":
            sr_headroom_ok = (high20 - S) >= HEADROOM_MIN_R * atr
        else:
            sr_headroom_ok = (S - low20) >= HEADROOM_MIN_R * atr

    # MTF trend alignment (5m + 15m EMA9/21) — anchor to MARKET_TZ day
    local_today = market_now().date()
    frm_intraday = local_today.isoformat()
    to_intraday  = (local_today + timedelta(days=1)).isoformat()

    def _closes(bars: List[Dict[str, Any]]) -> List[float]:
        return [b["c"] for b in bars]

    fivem   = await polygon_aggs(client, ticker=alert["symbol"], multiplier=5,  timespan="minute",
                                 frm=frm_intraday, to=to_intraday, sort="asc", limit=5000)
    fifteen = await polygon_aggs(client, ticker=alert["symbol"], multiplier=15, timespan="minute",
                                 frm=frm_intraday, to=to_intraday, sort="asc", limit=5000)

    ema9_5  = ema(_closes(fivem), 9) if fivem else None
    ema21_5 = ema(_closes(fivem), 21) if fivem else None
    ema9_15 = ema(_closes(fifteen), 9) if fifteen else None
    ema21_15= ema(_closes(fifteen), 21) if fifteen else None

    mtf_align = None
    if all(v is not None for v in [ema9_5, ema21_5, ema9_15, ema21_15]):
        if alert["side"] == "CALL":
            mtf_align = (ema9_5 > ema21_5) and (ema9_15 > ema21_15)
        else:
            mtf_align = (ema9_5 < ema21_5) and (ema9_15 < ema21_15)

    # EM vs Break-even
    # EM(1σ) ≈ S * IV * sqrt(DTE/365)
    em_1s = None
    em_vs_be_ok = None
    if S and cur_iv and days_to_exp is not None and days_to_exp >= 0:
        em_1s = S * float(cur_iv) * math.sqrt(max(0.0, days_to_exp) / 365.0)
        premium = mid or 0.0
        if alert["side"] == "CALL":
            be_price = alert["strike"] + premium
            be_dist  = max(0.0, be_price - S)
        else:
            be_price = alert["strike"] - premium
            be_dist  = max(0.0, S - be_price)
        if be_dist == 0.0:
            em_vs_be_ok = True
        elif em_1s is not None:
            em_vs_be_ok = (em_1s >= EM_VS_BE_RATIO_MIN * be_dist)

    # IV rank from in-memory history
    sym = alert["symbol"]
    if sym not in _IV_HISTORY:
        _IV_HISTORY[sym] = deque(maxlen=IV_HISTORY_LEN)
    if cur_iv is not None:
        _IV_HISTORY[sym].append(float(cur_iv))
    iv_rank = iv_rank_from_history(_IV_HISTORY[sym], cur_iv)

    # Market regime via SPY daily EMA trend
    spy_daily = await polygon_aggs(client, ticker="SPY", multiplier=1, timespan="day",
                                   frm=frm_day, to=to_iso, sort="asc")
    spy_close = [b["c"] for b in spy_daily] if spy_daily else []
    spy_ema_fast = ema(spy_close, REGIME_TREND_EMA_FAST) if spy_close else None
    spy_ema_slow = ema(spy_close, REGIME_TREND_EMA_SLOW) if spy_close else None
    regime_flag = None
    if (spy_ema_fast is not None) and (spy_ema_slow is not None):
        regime_flag = "trending" if spy_ema_fast > spy_ema_slow else "choppy"

    # Trade style bands
    if TRADE_STYLE == "swing":
        delta_min, delta_max = DELTA_MIN_SWING, DELTA_MAX_SWING
        dte_min, dte_max     = DTE_MIN_SWING, DTE_MAX_SWING
        risk_style           = "swing"
        initial_stop_pct     = 0.30
        tp_pct               = 0.60
        trail_after_pct      = 0.40
    else:
        delta_min, delta_max = DELTA_MIN_INTRADAY, DELTA_MAX_INTRADAY
        dte_min, dte_max     = DTE_MIN_INTRADAY, DTE_MAX_INTRADAY
        risk_style           = "intraday"
        initial_stop_pct     = 0.30
        tp_pct               = 0.60
        trail_after_pct      = 0.40

    # Band checks
    delta_band_ok = None if delta is None else (delta_min <= abs(float(delta)) <= delta_max)
    dte_band_ok   = (dte_min <= days_to_exp <= dte_max)

    # Macro blocklist (simple)
    today_local_str = market_now().date().isoformat()
    no_event_risk = today_local_str not in MACRO_EVENT_DATES

    features = {
        "S": S,
        "oi": oi,
        "vol": vol,
        "iv": cur_iv,
        "iv_rank": iv_rank,                 # 0..1 or None
        "rv20": realized_vol_annualized(closes, 20) if closes else None,  # annualized realized vol
        "opt_bid": bid,
        "opt_ask": ask,
        "opt_mid": mid,
        "option_spread_pct": opt_spread_pct,
        "quote_age_sec": quote_age_sec,
        "delta": delta,
        "dte": days_to_exp,
        "atr": atr,
        "high20": high20,
        "low20": low20,
        "sr_headroom_ok": sr_headroom_ok,
        "mtf_align": mtf_align,
        "em_1s": em_1s,
        "em_vs_be_ok": em_vs_be_ok,
        "regime_flag": regime_flag,
        "no_event_risk": no_event_risk,
        "risk_plan": {
            "style": risk_style,
            "initial_stop_pct": initial_stop_pct,
            "take_profit_pct": tp_pct,
            "trail_after_pct": trail_after_pct,
        },
        "bands": {
            "delta_min": delta_min, "delta_max": delta_max,
            "dte_min": dte_min, "dte_max": dte_max,
        }
    }
    return features

def numerical_pre_score(side: str, f: Dict[str, Any]) -> int:
    """0..100 pre-score to gate LLM (hybrid)."""
    score = 0

    # Liquidity
    if (f.get("oi", 0) >= 10000): score += 10
    if (f.get("vol", 0) >= 5000): score += 10
    sp = f.get("option_spread_pct")
    if (sp is not None) and (sp <= OPT_SPREAD_MAX_PCT): score += 5
    qa = f.get("quote_age_sec")
    if (qa is not None) and (qa <= QUOTE_AGE_MAX_SEC): score += 5

    # Structure
    if f.get("mtf_align") is True: score += 15
    if f.get("sr_headroom_ok") is True: score += 10

    # Vol context
    rv = f.get("rv20"); iv = f.get("iv")
    if rv is not None and iv is not None:
        if rv > iv: score += 10
        elif abs(rv - iv) / max(1e-9, iv) <= 0.1: score += 6  # approx equal

    # Geometry
    delta = f.get("delta")
    bands = f.get("bands", {})
    if (delta is not None) and (bands):
        if bands["delta_min"] <= abs(float(delta)) <= bands["delta_max"]:
            score += 15
    if f.get("em_vs_be_ok") is True: score += 15

    # Regime
    if f.get("regime_flag") == "trending": score += 10

    return int(score)

# =======================
# LLM
# =======================
def build_llm_prompt(alert: Dict[str, Any], f: Dict[str, Any]) -> str:
    iv_rank = f.get("iv_rank")
    iv_ctx = "low" if (iv_rank is not None and iv_rank < 0.33) else "high" if (iv_rank is not None and iv_rank > 0.66) else "medium"
    rv_iv_spread = (
        "rv>iv" if (f.get("rv20") and f.get("iv") and f["rv20"] > f["iv"]) else
        "rv≈iv" if (f.get("rv20") and f.get("iv") and abs(f["rv20"] - f["iv"]) / max(1e-9, f["iv"]) <= 0.1) else
        "rv<iv"
    )
    checklist_hint = {
        "liquidity_ok": (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= OPT_SPREAD_MAX_PCT)
                        and (f.get("quote_age_sec") is not None and f["quote_age_sec"] <= QUOTE_AGE_MAX_SEC),
        "spread_ok": (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= OPT_SPREAD_MAX_PCT),
        "delta_band_ok": f.get("bands") and (f["bands"]["delta_min"] <= abs(float(f.get("delta") or 0)) <= f["bands"]["delta_max"]),
        "dte_band_ok": (f["bands"]["dte_min"] <= f.get("dte", 0) <= f["bands"]["dte_max"]) if f.get("dte") is not None else False,
        "iv_context": iv_ctx,
        "rv_iv_spread": rv_iv_spread,
        "em_vs_breakeven_ok": f.get("em_vs_be_ok") is True,
        "mtf_trend_alignment": f.get("mtf_align") is True,
        "sr_headroom_ok": f.get("sr_headroom_ok") is True,
        "no_event_risk": f.get("no_event_risk") is True
    }
    lines = [
        f"Alert: {alert['side']} {alert['symbol']} strike {alert['strike']} "
        f"exp {alert['expiry']} (~{f.get('dte')} DTE) at underlying ≈ {f.get('S')}",
        f"Trade style: {f['risk_plan']['style']}",
        "Snapshot:",
        f"  IV: {f.get('iv')}",
        f"  IV_rank: {iv_rank}",
        f"  OI: {f.get('oi')}  Vol: {f.get('vol')}",
        f"  NBBO: bid={f.get('opt_bid')} ask={f.get('opt_ask')} mid={f.get('opt_mid')} spread%={f.get('option_spread_pct')}",
        f"  Quote age (s): {f.get('quote_age_sec')}",
        f"  Greeks: delta={f.get('delta')}",
        f"  EM_1s: {f.get('em_1s')}  EM_vs_BE_ok: {f.get('em_vs_be_ok')}",
        f"  MTF align: {f.get('mtf_align')}  S/R headroom ok: {f.get('sr_headroom_ok')}",
        f"  Regime: {f.get('regime_flag')}  ATR(14): {f.get('atr')}",
        f"Checklist: {json.dumps(checklist_hint)}",
        "Return strict JSON per schema."
    ]
    return "\n".join(lines)

async def analyze_with_openai(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are a disciplined options trading analyst. Use the provided snapshot & checklist.\n"
        "Respond with STRICT JSON:\n"
        "{\n"
        '  "decision": "buy|wait|skip",\n'
        '  "confidence": <0..1>,\n'
        '  "reason": "<=2 sentences>",\n'
        '  "checklist": {\n'
        '    "liquidity_ok": true/false,\n'
        '    "spread_ok": true/false,\n'
        '    "delta_band_ok": true/false,\n'
        '    "dte_band_ok": true/false,\n'
        '    "iv_context": "low|medium|high",\n'
        '    "rv_iv_spread": "rv>iv|rv≈iv|rv<iv",\n'
        '    "em_vs_breakeven_ok": true/false,\n'
        '    "mtf_trend_alignment": true/false,\n'
        '    "sr_headroom_ok": true/false,\n'
        '    "no_event_risk": true/false\n'
        "  },\n"
        '  "risk_plan": {"style":"intraday|swing","initial_stop_pct":0..1,"take_profit_pct":0..1,"trail_after_pct":0..1},\n'
        '  "ev_estimate": {"win_prob":0..1,"avg_win_pct":0..5,"avg_loss_pct":0..5,"expected_value_pct":-5..5}\n'
        "}\n"
        "If any checklist *_ok is false, decision must be 'skip'. Keep temperature low, be consistent."
    )
    prompt = build_llm_prompt(alert, f)
    try:
        resp = oai_client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
        )
        out = json.loads(resp.choices[0].message.content)
    except Exception as e:
        out = {
            "decision": "skip",
            "confidence": 0.0,
            "reason": f"LLM call failed: {type(e).__name__}.",
            "checklist": {
                "liquidity_ok": False, "spread_ok": False, "delta_band_ok": False, "dte_band_ok": False,
                "iv_context": "medium", "rv_iv_spread": "rv≈iv", "em_vs_breakeven_ok": False,
                "mtf_trend_alignment": False, "sr_headroom_ok": False, "no_event_risk": True
            },
            "risk_plan": f.get("risk_plan", {}),
            "ev_estimate": {"win_prob": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "expected_value_pct": 0.0}
        }

    # Guardrails: enforce skip if any *_ok false
    try:
        chk = out.get("checklist", {})
        gates = ["liquidity_ok", "spread_ok", "delta_band_ok", "dte_band_ok",
                 "em_vs_breakeven_ok", "mtf_trend_alignment", "sr_headroom_ok", "no_event_risk"]
        if any(chk.get(k) is False for k in gates):
            out["decision"] = "skip"
    except Exception:
        out["decision"] = "skip"

    return out

# =======================
# Output formatting
# =======================
def compose_telegram_text(alert: Dict[str, Any], option_ticker: str, f: Dict[str, Any],
                          prescore: int, llm: Dict[str, Any], *, llm_ran: bool, llm_reason: str) -> str:
    lines = [
        "📣 Options Alert",
        f"{alert['side']} {alert['symbol']} | Strike {alert['strike']} | Exp {alert['expiry']} (~{f.get('dte')} DTE)",
        f"Underlying (alert): {_fmt(f.get('S'))}",
        f"Contract: {option_ticker}",
        "",
        "Snapshot:",
        f"  IV: {_fmt(f.get('iv'))}  (IV rank: {_fmt(f.get('iv_rank'))})",
        f"  OI: {_fmt(f.get('oi'))}  Vol: {_fmt(f.get('vol'))}",
        f"  NBBO: bid={_fmt(f.get('opt_bid'))} ask={_fmt(f.get('opt_ask'))} mid={_fmt(f.get('opt_mid'))}",
        f"  Spread%: {_fmt(f.get('option_spread_pct'))}  QuoteAge(s): {_fmt(f.get('quote_age_sec'))}",
        f"  Greeks: Δ={_fmt(f.get('delta'))}",
        f"  EM_1σ: {_fmt(f.get('em_1s'))}  EM_vs_BE_ok: {f.get('em_vs_be_ok')}",
        f"  MTF align: {f.get('mtf_align')}  S/R ok: {f.get('sr_headroom_ok')}",
        f"  Regime: {f.get('regime_flag')}  ATR(14): {_fmt(f.get('atr'))}",
        "",
        f"Pre-Score: {prescore}/100 (auto-buy>={PRESCORE_AUTO_BUY}, auto-skip<={PRESCORE_AUTO_SKIP})",
    ]
    if llm_ran:
        lines += [
            f"LLM Decision: {llm.get('decision','skip').upper()}  (conf: {llm.get('confidence',0):.2f})",
            f"Reason: {llm.get('reason','')}",
            f"Checklist: {json.dumps(llm.get('checklist', {}))}",
            f"EV: {json.dumps(llm.get('ev_estimate', {}))}",
        ]
    else:
        lines += ["LLM: Skipped", f"Reason: {llm_reason}"]
    lines += ["", "⚠️ Educational demo; not financial advice."]
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
        with contextlib.suppress(Exception):
            return r.json()
        return {"status_code": r.status_code, "text": r.text}

# =======================
# Daily report
# =======================
def _summarize_day_for_report(local_date: date) -> str:
    entries = [e for e in _DECISIONS_LOG if e["timestamp_local"].date() == local_date]
    total_alerts = len(entries)
    llm_runs = sum(1 for e in entries if e["llm"]["ran"])
    llm_skips = total_alerts - llm_runs
    buys = sum(1 for e in entries if e["decision_final"] == "buy")
    skips = total_alerts - buys
    avg_conf = (sum(float(e["llm"].get("confidence") or 0.0) for e in entries if e["llm"]["ran"]) / llm_runs) if llm_runs else 0.0
    by_symbol = Counter((e["symbol"] for e in entries))
    top = ", ".join(f"{sym}({cnt})" for sym, cnt in by_symbol.most_common(5)) or "—"
    by_outcome = Counter((e["decision_path"] for e in entries))
    quota = llm_quota_snapshot()
    header = f"📊 Daily Report — {local_date.isoformat()} ({MARKET_TZ})"
    body = [
        f"Alerts: {total_alerts} | LLM ran: {llm_runs} | skips: {llm_skips}",
        f"Decisions — BUY: {buys} | SKIP: {skips}",
        f"Avg LLM confidence (when ran): {avg_conf:.2f}",
        f"Top tickers: {top}",
        f"Paths: {dict(by_outcome)}",
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
            await asyncio.sleep(1)

# =======================
# FastAPI lifecycle
# =======================
@app.on_event("startup")
async def on_startup():
    app.state.report_task = asyncio.create_task(_daily_report_scheduler())

@app.on_event("shutdown")
async def on_shutdown():
    task = getattr(app.state, "report_task", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(Exception):
            await task

# =======================
# Routes
# =======================
@app.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@app.get("/quota")
def quota():
    return {"ok": True, "quota": llm_quota_snapshot()}

# ---- Config & Logs Endpoints ----
def _active_config_dict() -> Dict[str, Any]:
    return {
        "model": OPENAI_MODEL,
        "market_tz": MARKET_TZ,
        "report_hhmm": REPORT_HHMM,
        "trade_style": TRADE_STYLE,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "iv_history_len": IV_HISTORY_LEN,
        "macro_event_dates": MACRO_EVENT_DATES,
        "llm_budget": llm_quota_snapshot(),
        "thresholds": {
            "volume_min_for_llm": VOLUME_MIN_FOR_LLM,
            "oi_min_for_llm": OI_MIN_FOR_LLM,
            "opt_spread_max_pct": OPT_SPREAD_MAX_PCT,
            "quote_age_max_sec":   QUOTE_AGE_MAX_SEC,
            "em_vs_be_ratio_min":  EM_VS_BE_RATIO_MIN,
            "headroom_min_r":      HEADROOM_MIN_R,
            "prescore": {
                "auto_buy":  PRESCORE_AUTO_BUY,
                "auto_skip": PRESCORE_AUTO_SKIP,
            },
            "delta_bands": {
                "intraday": {"min": DELTA_MIN_INTRADAY, "max": DELTA_MAX_INTRADAY},
                "swing":    {"min": DELTA_MIN_SWING,    "max": DELTA_MAX_SWING},
            },
            "dte_bands": {
                "intraday": {"min": DTE_MIN_INTRADAY, "max": DTE_MAX_INTRADAY},
                "swing":    {"min": DTE_MIN_SWING,    "max": DTE_MAX_SWING},
            },

            "llm_window": {
            "cutoff_tz": LLM_CUTOFF_TZ,
            "cutoff_hhmm": LLM_CUTOFF_HHMM,
            "inclusive": True
        },
        },
    }

@app.get("/config")
def get_config():
    """Dump active thresholds/knobs (sanitized; no API keys)."""
    return {"ok": True, "config": _active_config_dict()}

@app.get("/logs/today")
def logs_today(limit: int = 50):
    """Return last N decisions for today's market date (in MARKET_TZ)."""
    limit = max(1, min(int(limit), 500))
    today_local = market_now().date()

    def _serialize(e: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp_local": e["timestamp_local"].isoformat(),
            "symbol": e["symbol"],
            "side": e["side"],
            "option_ticker": e.get("option_ticker"),
            "prescore": e.get("prescore"),
            "decision_final": e.get("decision_final"),
            "decision_path": e.get("decision_path"),
            "llm": {
                "ran": e["llm"].get("ran"),
                "decision": e["llm"].get("decision"),
                "confidence": e["llm"].get("confidence"),
                "reason": e["llm"].get("reason"),
            },
            "features": {
                "oi":  e["features"].get("oi"),
                "vol": e["features"].get("vol"),
                "spread_pct": e["features"].get("spread_pct"),
                "quote_age_sec": e["features"].get("quote_age_sec"),
                "delta": e["features"].get("delta"),
                "dte":   e["features"].get("dte"),
                "em_vs_be_ok": e["features"].get("em_vs_be_ok"),
                "mtf_align":   e["features"].get("mtf_align"),
                "sr_ok":       e["features"].get("sr_ok"),
                "iv":          e["features"].get("iv"),
                "iv_rank":     e["features"].get("iv_rank"),
                "rv20":        e["features"].get("rv20"),
                "regime":      e["features"].get("regime"),
            },
        }

    todays = [e for e in _DECISIONS_LOG if e["timestamp_local"].date() == today_local]
    out = [_serialize(e) for e in todays[-limit:]]
    return {"ok": True, "count": len(out), "limit": limit, "date": today_local.isoformat(), "entries": out}

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
        swf = same_week_friday(datetime.now(timezone.utc).date()).isoformat()
        return {
            "side": side.upper(),
            "symbol": symbol.upper(),
            "underlying_price_from_alert": float(underlying_px),
            "strike": float(strike),
            "expiry": swf,
            "expiry_source": "computed_same_week_friday",
        }
    raise HTTPException(status_code=400, detail='Alert must be like: "CALL Signal: TICKER at 123.45 Strike: 123" or with expiry: "... Expiry: YYYY-MM-DD"')

@app.post("/webhook", response_class=JSONResponse)
@app.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(request: Request):
    payload = await _get_alert_text(request)
    alert = parse_alert_text(payload)
    desired_strike = round_strike_to_common_increment(alert["strike"])

    async with httpx.AsyncClient(http2=True, timeout=20.0) as client:
        contracts = await polygon_list_contracts_for_expiry(
            client, symbol=alert["symbol"], expiry=alert["expiry"], side=alert["side"], limit=250
        )
        if not contracts:
            raise HTTPException(status_code=404, detail=f"No contracts found for {alert['symbol']} {alert['side']} exp {alert['expiry']}.")
        best = pick_nearest_strike(contracts, desired_strike)
        if not best:
            raise HTTPException(status_code=404, detail=f"No strikes near {desired_strike} for {alert['symbol']} on {alert['expiry']}.")
        option_ticker = best.get("ticker")
        snap = await polygon_get_option_snapshot(client, underlying=alert["symbol"], option_ticker=option_ticker)

        # Build features
        f = await build_features(client, alert={**alert, "strike": desired_strike}, snapshot=snap)

    # ---------- Existing gates: vol/OI + cooldown + quota ----------
    oi = int(f.get("oi") or 0)
    vol = int(f.get("vol") or 0)

    key = (alert["symbol"], alert["side"])
    now_utc = datetime.now(timezone.utc)
    last_ts = _COOLDOWN.get(key)
    in_cooldown = False
    cooldown_reason = ""
    if last_ts is not None:
        elapsed = (now_utc - last_ts).total_seconds()
        if elapsed < COOLDOWN_SECONDS:
            in_cooldown = True
            cooldown_reason = f"In cooldown: {int(COOLDOWN_SECONDS - elapsed)}s remaining."

    base_gate_ok = (vol >= VOLUME_MIN_FOR_LLM) or (oi >= OI_MIN_FOR_LLM)

    # ---------- New hard liquidity gates ----------
    sp_ok = (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= OPT_SPREAD_MAX_PCT)
    qa_ok = (f.get("quote_age_sec") is not None and f["quote_age_sec"] <= QUOTE_AGE_MAX_SEC)

    # ---------- Band checks ----------
    delta = f.get("delta")
    delta_band_ok = (delta is not None and f["bands"]["delta_min"] <= abs(float(delta)) <= f["bands"]["delta_max"])
    dte_band_ok = (f["bands"]["dte_min"] <= f.get("dte", 0) <= f["bands"]["dte_max"]) if f.get("dte") is not None else False

    # ---------- Event risk prefilter ----------
    if not f.get("no_event_risk", True):
        llm_reason = "Blocked by macro/event risk for today."
        decision_path = "skip.event"
        decision_final = "skip"
        llm_ran = False
        # After processing, set cooldown timestamp and log + return
        _COOLDOWN[key] = now_utc
        llm = {
            "decision": "skip",
            "confidence": 0.0,
            "reason": llm_reason,
            "checklist": {"no_event_risk": False},
            "ev_estimate": {}
        }
        prescore = numerical_pre_score(alert["side"], f)
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike},
            option_ticker=option_ticker, f=f, prescore=prescore,
            llm=llm, llm_ran=False, llm_reason=llm_reason
        )
        tg_result = await send_telegram(tg_text)
        _DECISIONS_LOG.append({
            "timestamp_local": market_now(),
            "symbol": alert["symbol"],
            "side": alert["side"],
            "option_ticker": option_ticker,
            "decision_final": decision_final,
            "decision_path": decision_path,
            "prescore": prescore,
            "llm": {"ran": False, "decision": "skip", "confidence": 0.0, "reason": llm_reason},
            "features": {
                "oi": oi, "vol": vol, "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
                "delta": f.get("delta"), "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
                "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
                "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag")
            }
        })
        return {
            "ok": True,
            "parsed_alert": alert,
            "option_ticker": option_ticker,
            "features": f,
            "prescore": prescore,
            "decision": {"final": decision_final, "path": decision_path},
            "llm": {"ran": False, "reason": llm_reason},
            "cooldown": {"seconds": COOLDOWN_SECONDS, "active": in_cooldown},
            "quota": llm_quota_snapshot(),
            "telegram": {"sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False, "result": tg_result},
            "notes": "Event-risk prefilter blocked this alert."
        }

    # ---------- Pre-score ----------
    prescore = numerical_pre_score(alert["side"], f)

    llm_ran = False
    llm_reason = ""
    decision_final = "skip"
    decision_path = ""

    # Auto-skip ladders
    if in_cooldown:
        llm_reason = cooldown_reason
        decision_path = "skip.cooldown"
    elif not base_gate_ok:
        llm_reason = f"Below thresholds: vol {vol} < {VOLUME_MIN_FOR_LLM} AND OI {oi} < {OI_MIN_FOR_LLM}."
        decision_path = "skip.base_gate"
    elif not sp_ok or not qa_ok:
      llm_reason = f"Liquidity gate failed: spread_ok={sp_ok}, quote_age_ok={qa_ok}"
       decision_path = "skip.liquidity" 
    elif not delta_band_ok or not dte_band_ok:
        llm_reason = f"Band gate failed: delta_band_ok={delta_band_ok}, dte_band_ok={dte_band_ok}"
        decision_path = "skip.bands"
    elif prescore >= PRESCORE_AUTO_BUY:
        decision_final = "buy"
        decision_path = "auto.buy"
    elif prescore <= PRESCORE_AUTO_SKIP:
        decision_final = "skip"
        decision_path = "auto.skip"
    elif not can_consume_llm():
        llm_reason = f"Daily LLM quota reached ({llm_quota_snapshot()['used']}/{llm_quota_snapshot()['max']})."
        decision_path = "skip.quota"
    elif not _is_within_llm_window(datetime.now(timezone.utc)):
        # After 11:30 AM CST (by default), do NOT run LLM. Autos still applied above.
        llm_reason = f"LLM window closed (allowed until {LLM_CUTOFF_HHMM} {LLM_CUTOFF_TZ})."
        decision_path = "skip.llm_window"
    else:
        # Gray zone -> LLM (morning only)
        llm = await analyze_with_openai(alert, f)
        consume_llm()
        llm_ran = True
        decision_final = "buy" if llm.get("decision") == "buy" else "skip"
        decision_path = "llm.buy" if decision_final == "buy" else "llm.skip"

    # After processing, set cooldown timestamp
    _COOLDOWN[key] = now_utc

    # Prepare LLM section for outputs (even if not run)
    if not llm_ran:
        llm = {
            "decision": "skip",
            "confidence": 0.0,
            "reason": llm_reason,
            "checklist": {},
            "ev_estimate": {}
        }

    tg_text = compose_telegram_text(
        alert={**alert, "strike": desired_strike},
        option_ticker=option_ticker,
        f=f,
        prescore=prescore,
        llm=llm,
        llm_ran=llm_ran,
        llm_reason=llm_reason
    )
    tg_result = await send_telegram(tg_text)

    # Log for report/calibration
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": alert["symbol"],
        "side": alert["side"],
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": decision_path,
        "prescore": prescore,
        "llm": {"ran": llm_ran, "decision": llm.get("decision"), "confidence": llm.get("confidence"), "reason": llm.get("reason")},
        "features": {
            "oi": oi, "vol": vol, "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
            "delta": f.get("delta"), "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
            "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag")
        }
    })

    return {
        "ok": True,
        "parsed_alert": alert,
        "option_ticker": option_ticker,
        "features": f,
        "prescore": prescore,
        "decision": {
            "final": decision_final,
            "path": decision_path,
        },
        "llm": {
            "ran": llm_ran,
            "reason": llm_reason if not llm_ran else "",
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "checklist": llm.get("checklist"),
            "ev_estimate": llm.get("ev_estimate"),
        },
        "cooldown": {
            "seconds": COOLDOWN_SECONDS,
            "active": in_cooldown,
        },
        "quota": llm_quota_snapshot(),
        "thresholds": {
            "volume_min_for_llm": VOLUME_MIN_FOR_LLM,
            "oi_min_for_llm": OI_MIN_FOR_LLM,
            "opt_spread_max_pct": OPT_SPREAD_MAX_PCT,
            "quote_age_max_sec": QUOTE_AGE_MAX_SEC,
            "prescore_auto_buy": PRESCORE_AUTO_BUY,
            "prescore_auto_skip": PRESCORE_AUTO_SKIP,
        },
        "telegram": {"sent": bool(tg_result) if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else False, "result": tg_result},
        "notes": "Hybrid gating: liquidity/bands -> pre-score -> LLM for gray zone. Educational demo; not financial advice.",
    }

if __name__ == "__main__":
    uvicorn.run("alert_server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
