# trading_engine.py
import os
import re
import asyncio
import socket
import logging
from urllib.parse import quote
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta, date

import httpx
from fastapi import HTTPException, Request

from ibkr_client import place_recommended_option_order
from config import (
    COOLDOWN_SECONDS,
    CDT_TZ,
    WINDOWS_CDT,
    MAX_LLM_PER_DAY,
)
from polygon_client import (
    build_option_contract,   # OCC builder: O:<SYM><YYMMDD><C/P><STRIKE*1000>
)
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG

# === Polygon/market helpers (split module) ===
from market_ops import (
    polygon_get_option_snapshot_export,
    poly_option_backfill,
    choose_best_contract,
    scan_for_best_contract_for_alert,
    scan_top_candidates_for_alert,
    ensure_nbbo,
)

# ------------- Logger -------------
logger = logging.getLogger("trading_engine")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# =========================
# Global state / resources
# =========================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # optional, for checks

# Force chain scan at all times
ALWAYS_CHAIN_SCAN = os.getenv("ALWAYS_CHAIN_SCAN", "1") == "1"

# Telegram toggles
SEND_CHAIN_SCAN_ALERTS = os.getenv("SEND_CHAIN_SCAN_ALERTS", "1") == "1"
SEND_CHAIN_SCAN_TOPN_ALERTS = os.getenv("SEND_CHAIN_SCAN_TOPN_ALERTS", "1") == "1"

_llm_quota: Dict[str, Any] = {"date": None, "used": 0}
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

HTTP: httpx.AsyncClient | None = None
WORK_Q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
WORKER_COUNT = 0

def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

# --- IBKR toggles ---
IBKR_ENABLED = _env_truthy(os.getenv("IBKR_ENABLED", "0"))
IBKR_DEFAULT_QTY = int(os.getenv("IBKR_DEFAULT_QTY", "1"))
IBKR_TIF = os.getenv("IBKR_TIF", "DAY").upper()
IBKR_ORDER_MODE = os.getenv("IBKR_ORDER_MODE", "auto").lower()   # auto | market | limit
IBKR_USE_MID_AS_LIMIT = os.getenv("IBKR_USE_MID_AS_LIMIT", "1") == "1"

# NOTE: kept for compatibility, but ALWAYS_CHAIN_SCAN supersedes this
PREFER_CHAIN_SCAN = True

# =========================
# Trading thresholds (tunable)
# =========================
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

# ========== Regex ==========
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

# ========== Exported small helpers ==========
def market_now() -> datetime:
    return datetime.now(CDT_TZ)

def llm_quota_snapshot() -> Dict[str, Any]:
    _reset_quota_if_new_day()
    used = int(_llm_quota.get("used", 0))
    limit = int(MAX_LLM_PER_DAY)
    return {"limit": limit, "used": used, "remaining": max(0, limit - used), "date": str(_llm_quota["date"])}

def get_worker_stats() -> Dict[str, Any]:
    return {"queue_size": WORK_Q.qsize(), "queue_maxsize": WORK_Q.maxsize, "workers": WORKER_COUNT}

def get_http_client() -> httpx.AsyncClient | None:
    return HTTP

# ========== Quota ==========
def _reset_quota_if_new_day() -> None:
    today = market_now().date()
    if _llm_quota["date"] != today:
        _llm_quota["date"] = today
        _llm_quota["used"] = 0

def consume_llm(n: int = 1) -> None:
    _reset_quota_if_new_day()
    _llm_quota["used"] = int(_llm_quota.get("used", 0)) + n

# ========== Parsing ==========
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

# ========== Misc utils ==========
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
    end   = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return start <= now <= end

# =========================
# Strike helpers
# =========================
def _build_plus_minus_contracts(symbol: str, ul_px: float, expiry_iso: str) -> Dict[str, Any]:
    call_strike = round_strike_to_common_increment(ul_px * 1.05)
    put_strike  = round_strike_to_common_increment(ul_px * 0.95)
    return {
        "strike_call": call_strike,
        "strike_put":  put_strike,
        "contract_call": build_option_contract(symbol, expiry_iso, "CALL", call_strike),
        "contract_put":  build_option_contract(symbol, expiry_iso, "PUT",  put_strike),
    }

# =========================
# Preflight
# =========================
def preflight_ok(f: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    checks: Dict[str, bool] = {}
    rth = _is_rth_now()

    quote_age = f.get("quote_age_sec")
    has_nbbo  = f.get("bid") is not None and f.get("ask") is not None
    has_last  = isinstance(f.get("last"), (int, float)) or isinstance(f.get("mid"), (int, float))

    if rth:
        checks["quote_fresh"] = (quote_age is not None and quote_age <= MAX_QUOTE_AGE_S and has_nbbo)
        checks["spread_ok"]   = (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= MAX_SPREAD_PCT)
    else:
        # After hours: accept if we have any tradable price proxy (last or synthetic mid)
        checks["quote_fresh"] = bool(has_last)
        checks["spread_ok"]   = True

    # After-hours: ignore day vol/oi gates (can be zero)
    require_liquidity = os.getenv("REQUIRE_LIQUIDITY_FIELDS", "0") == "1"
    if rth and require_liquidity:
        checks["vol_ok"] = (f.get("vol") or 0) >= MIN_VOL_TODAY
        checks["oi_ok"]  = (f.get("oi") or 0)  >= MIN_OI
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
    return "\n".join([header, contract, "", snap, decision, reason, scoreline]).strip()

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

# =========================
# Worker lifecycle
# =========================
async def startup():
    global HTTP, WORKER_COUNT
    HTTP = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(read=6.0, write=6.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    WORKER_COUNT = int(os.getenv("WORKERS", "3"))
    for i in range(WORKER_COUNT):
        asyncio.create_task(_worker())
    logger.info("startup complete; HTTP ready; workers=%d", WORKER_COUNT)

async def shutdown():
    global HTTP
    if HTTP:
        await HTTP.aclose()
        logger.info("shutdown complete; HTTP closed")

async def _worker():
    logger.info("worker task started")
    while True:
        job = await WORK_Q.get()
        try:
            logger.info("processing alert job: %s", (job.get("alert_text") or "")[:200])
            await _process_tradingview_job(job)
            logger.info("job processed")
        except Exception as e:
            logger.exception("[worker] error: %r", e)
        finally:
            WORK_Q.task_done()

# =========================
# Public job enqueue
# =========================
def enqueue_webhook_job(alert_text: str, flags: Dict[str, Any]) -> bool:
    job = {"alert_text": alert_text, "flags": flags}
    try:
        WORK_Q.put_nowait(job)
        logger.info("enqueue ok; flags=%s", flags)
        return True
    except asyncio.QueueFull:
        logger.warning("enqueue failed: queue full")
        return False

# =========================
# HTTP helpers (verbose)
# =========================
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

# Return status + json/text; do NOT swallow 402/403/429 so we can see entitlement/rate-limits.
async def _http_get_any(url: str, params: dict | None = None, timeout: float = 6.0) -> Dict[str, Any]:
    if HTTP is None:
        return {"status": None, "error": "HTTP client not ready"}
    try:
        r = await HTTP.get(url, params=params or {}, timeout=timeout)
        ct = r.headers.get("content-type", "")
        try:
            payload = r.json() if "application/json" in ct else r.text
        except Exception:
            payload = r.text
        return {"status": r.status_code, "body": payload}
    except Exception as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}"}

async def _pull_nbbo_direct(option_ticker: str) -> Dict[str, Any]:
    """
    Direct Polygon last-quote probe to derive bid/ask/mid/spread/age.
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY or HTTP is None:
        return out
    try:
        enc = _encode_ticker_path(option_ticker)
        lastq = await _http_json(
            HTTP,
            f"https://api.polygon.io/v3/quotes/options/{enc}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=4.0
        )
        if not lastq:
            return out
        res = lastq.get("results") or {}
        last = res.get("last") or res
        bid = last.get("bidPrice")
        ask = last.get("askPrice")
        ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
            mid = (bid + ask) / 2.0
            out["bid"] = float(bid)
            out["ask"] = float(ask)
            out["mid"] = round(float(mid), 4)
            if mid > 0:
                out["option_spread_pct"] = round((ask - bid) / mid * 100.0, 3)
        if ts is not None:
            try:
                ns = int(ts)
                if ns >= 10**14: sec = ns / 1e9
                elif ns >= 10**11: sec = ns / 1e6
                elif ns >= 10**8: sec = ns / 1e3
                else: sec = float(ns)
                out["quote_age_sec"] = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
            except Exception:
                pass
    except Exception:
        pass
    return out

async def _probe_nbbo_verbose(option_ticker: str) -> Dict[str, Any]:
    """
    Hit Polygon v3 last-quote endpoint and report status + minimal derived fields.
    Surfaces 402/403/429 so you know if it’s entitlement/rate-limit.
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return {"nbbo_reason": "no POLYGON_API_KEY in env"}

    enc = _encode_ticker_path(option_ticker)
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    res = await _http_get_any(url, params={"apiKey": POLYGON_API_KEY}, timeout=6.0)

    out["nbbo_http_status"] = res.get("status")
    if res.get("status") != 200:
        body = res.get("body")
        if isinstance(body, dict):
            out["nbbo_reason"] = body.get("error") or body.get("message") or "non-200 from Polygon"
        else:
            out["nbbo_reason"] = "non-200 from Polygon"
        out["nbbo_body_sample"] = (body if isinstance(body, dict) else str(body))[:400]
        return out

    body = res.get("body") or {}
    last = (body.get("results") or {}).get("last") or body.get("results") or {}
    bid = last.get("bidPrice"); ask = last.get("askPrice")
    ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")

    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
        mid = (bid + ask) / 2.0
        out.update({
            "bid": float(bid), "ask": float(ask),
            "mid": round(mid, 4),
            "option_spread_pct": round(((ask - bid)/mid*100.0), 3) if mid > 0 else None,
        })
    else:
        out["nbbo_reason"] = "no bid/ask in response (thin or AH?)"

    try:
        if ts is not None:
            ns = int(ts)
            if ns >= 10**14: sec = ns / 1e9
            elif ns >= 10**11: sec = ns / 1e6
            elif ns >= 10**8: sec = ns / 1e3
            else: sec = float(ns)
            out["quote_age_sec"] = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    except Exception:
        pass

    return out

# =========================
# Core processing
# =========================
async def _process_tradingview_job(job: Dict[str, Any]) -> None:
    global HTTP
    if HTTP is None:
        logger.warning("[worker] HTTP client not ready")
        return

    # 1) Parse TradingView alert text
    try:
        alert = parse_alert_text(job["alert_text"])
        logger.info("parsed alert: side=%s symbol=%s strike=%s expiry=%s",
                    alert.get("side"), alert.get("symbol"), alert.get("strike"), alert.get("expiry"))
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy = bool(job["flags"].get("force_buy", False))
    qty = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    # 2) Compute default expiry if missing (2w Friday)
    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    desired_strike = pm["strike_call"] if alert["side"] == "CALL" else pm["strike_put"]

    # 3) CHAIN SCAN — thresholds (relax after-hours)
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) primary contract selection (ALWAYS chain scan; broaden if needed)
    selection_debug: Dict[str, Any] = {}
    option_ticker = None

    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            HTTP,
            alert["symbol"],
            {"side": alert["side"], "symbol": alert["symbol"],
             "strike": alert.get("strike"), "expiry": alert.get("expiry")},
            min_vol=scan_min_vol,
            min_oi=scan_min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
        )
    except Exception:
        best_from_scan = None

    if best_from_scan:
        option_ticker = best_from_scan["ticker"]
        if isinstance(best_from_scan.get("strike"), (int, float)):
            desired_strike = float(best_from_scan["strike"])
        chosen_expiry = str(alert.get("expiry") or target_expiry)
        selection_debug = {
            "selected_by": "chain_scan",
            "selected_ticker": option_ticker,
            "best_item": best_from_scan,
            "chosen_expiry": chosen_expiry,
        }
        logger.info("chain_scan selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)
    else:
        # Broaden: pick the single best across the next 3 weeks as a backstop
        try:
            one_best = await scan_top_candidates_for_alert(
                HTTP,
                alert["symbol"],
                {"side": alert["side"], "symbol": alert["symbol"],
                 "strike": alert.get("strike"), "expiry": alert.get("expiry")},
                min_vol=scan_min_vol,
                min_oi=scan_min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=1,
            )
        except Exception:
            one_best = None

        if one_best:
            it = one_best[0]
            option_ticker = it["ticker"]
            desired_strike = float(it.get("strike") or desired_strike)
            chosen_expiry = str(it.get("expiry") or alert.get("expiry") or target_expiry)
            selection_debug = {
                "selected_by": "chain_scan_broadened",
                "selected_ticker": option_ticker,
                "best_item": it,
                "chosen_expiry": chosen_expiry,
            }
            logger.info("chain_scan (broadened) selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)
        else:
            # Last resort: fallback but respect the alert expiry if provided
            fallback_exp = str(alert.get("expiry") or target_expiry)
            pm_fallback = _build_plus_minus_contracts(alert["symbol"], ul_px, fallback_exp)
            option_ticker = pm_fallback["contract_call"] if alert["side"] == "CALL" else pm_fallback["contract_put"]
            desired_strike = pm_fallback["strike_call"] if alert["side"] == "CALL" else pm_fallback["strike_put"]
            chosen_expiry = fallback_exp
            selection_debug = {"selected_by": "fallback_pm", "reason": "scan_empty", "chosen_expiry": fallback_exp}
            logger.info("fallback selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)

    # Use the chosen expiry everywhere downstream
    chosen_expiry = selection_debug.get("chosen_expiry", str(alert.get("expiry") or target_expiry))

    # 4) Build feature bundle + aggressive NBBO backfills + verbose reasons
    f: Dict[str, Any] = {}
    try:
        if not POLYGON_API_KEY:
            f = {
                "bid": None, "ask": None, "mid": None, "last": None,
                "option_spread_pct": None, "quote_age_sec": None,
                "oi": None, "vol": None,
                "delta": None, "gamma": None, "theta": None, "vega": None,
                "iv": None, "iv_rank": None, "rv20": None, "prev_close": None, "quote_change_pct": None,
                "dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days,
                "em_vs_be_ok": None, "mtf_align": None, "sr_headroom_ok": None, "regime_flag": "trending",
                "prev_day_high": None, "prev_day_low": None,
                "premarket_high": None, "premarket_low": None,
                "vwap": None, "vwap_dist": None,
                "above_pdh": None, "below_pdl": None, "above_pmh": None, "below_pml": None,
            }
        else:
            # A) rich backfill (OI/Vol/Greeks/IV + last + sometimes NBBO)
            extra_from_snap = await poly_option_backfill(HTTP, alert["symbol"], option_ticker, today_utc)
            for k, v in (extra_from_snap or {}).items():
                if v is not None:
                    f[k] = v

            # B) snapshot-consistent features (VWAP, regime, headroom, etc.)
            snap = None
            try:
                snap = await polygon_get_option_snapshot_export(HTTP, underlying=alert["symbol"], option_ticker=option_ticker)
            except Exception:
                snap = None

            core = await build_features(
                HTTP,
                alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
                snapshot=snap
            )
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v

            # C) derive mid/spread if we already have NBBO
            try:
                bid = f.get("bid"); ask = f.get("ask"); mid = f.get("mid")
                if bid is not None and ask is not None:
                    if mid is None:
                        mid = (float(bid) + float(ask)) / 2.0
                        f["mid"] = round(mid, 4)
                    spread = float(ask) - float(bid)
                    if mid and mid > 0:
                        f["option_spread_pct"] = round((spread / mid) * 100.0, 3)
            except Exception:
                pass

            # D) aggressively ensure NBBO if missing
            try:
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo = await ensure_nbbo(HTTP, option_ticker, tries=12, delay=0.35)
                    for k, v in (nbbo or {}).items():
                        if v is not None:
                            f[k] = v
            except Exception:
                pass

            # E) direct last-quote probe to fill NBBO/age/spread
            if f.get("bid") is None or f.get("ask") is None:
                probe = await _pull_nbbo_direct(option_ticker)
                for k, v in (probe or {}).items():
                    if v is not None:
                        f[k] = v

            # F) ensure DTE
            try:
                if f.get("dte") is None:
                    f["dte"] = (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days
            except Exception:
                pass

            # G) compute change% vs prev close (use mid, fallback last)
            try:
                if f.get("quote_change_pct") is None:
                    prev_close = f.get("prev_close")
                    mark = f.get("mid") if f.get("mid") is not None else f.get("last")
                    if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
                        f["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
            except Exception:
                pass

            # H) if still no NBBO but we have last, synthesize a mark (AH convenience)
            if (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("last"), (int, float)):
                f.setdefault("mid", float(f["last"]))

            # I) final verbose NBBO probe to attach reason/status when still missing
            if f.get("bid") is None or f.get("ask") is None:
                nbbo_dbg = await _probe_nbbo_verbose(option_ticker)
                for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
                    if nbbo_dbg.get(k) is not None:
                        f[k] = nbbo_dbg[k]
                f["nbbo_http_status"] = nbbo_dbg.get("nbbo_http_status")
                f["nbbo_reason"] = nbbo_dbg.get("nbbo_reason")
                f["nbbo_body_sample"] = nbbo_dbg.get("nbbo_body_sample")
                logger.info("NBBO verbose: status=%s reason=%s",
                            f.get("nbbo_http_status"), f.get("nbbo_reason"))

            # J) optional: synthesize a soft spread for analytics when we only have mark/last
            if (f.get("option_spread_pct") is None) and (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("mid"), (int, float)):
                f["option_spread_pct"] = float(os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "10.0"))

    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}

    # 5) PRE-LLM CHAIN-SCAN ALERT (send only if scan picked the contract)
    if SEND_CHAIN_SCAN_ALERTS and selection_debug.get("selected_by", "").startswith("chain_scan"):
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                pre_text = (
                    "🔎 Chain-Scan Pick (from TradingView alert)\n"
                    f"{alert['side']} {alert['symbol']} | Strike {desired_strike} | Exp {chosen_expiry}\n"
                    f"Contract: {option_ticker}\n"
                    f"NBBO {f.get('bid')}/{f.get('ask')}  Mark={f.get('mid')}  Last={f.get('last')}\n"
                    f"Spread%={f.get('option_spread_pct')}  QuoteAge(s)={f.get('quote_age_sec')}\n"
                    f"OI={f.get('oi')}  Vol={f.get('vol')}  IV={f.get('iv')}  Δ={f.get('delta')} Γ={f.get('gamma')}\n"
                    f"DTE={f.get('dte')}  Regime={f.get('regime_flag')}  (pre-LLM)\n"
                    f"NBBO dbg: status={f.get('nbbo_http_status')} reason={f.get('nbbo_reason')}\n"
                )
                logger.info("Telegram pre-LLM chainscan: sending…")
                await send_telegram(pre_text)
                logger.info("Telegram pre-LLM chainscan: sent")
            else:
                logger.info("Telegram pre-LLM chainscan: skipped (no bot/chat env)")
        except Exception as e:
            logger.exception("[worker] Telegram pre-LLM chainscan error: %s", e)

    # 5b) SECOND ALERT — Top 3 across next 3 weeks (always message if Telegram configured)
    if SEND_CHAIN_SCAN_TOPN_ALERTS:
        try:
            logger.info("computing chain top3 across next 3 weeks…")
            top3 = await scan_top_candidates_for_alert(
                HTTP,
                alert["symbol"],
                {"side": alert["side"], "symbol": alert["symbol"],
                 "strike": alert.get("strike"), "expiry": alert.get("expiry")},
                min_vol=scan_min_vol,
                min_oi=scan_min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=3,
            )
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                if top3:
                    lines = [
                        f"📊 Chain Top 3 (next 3 weeks) for {alert['symbol']} {alert['side']}",
                        "(ranked by strike fit + spread + liquidity)"
                    ]
                    for i, it in enumerate(top3, 1):
                        lines.append(
                            f"{i}. {it['expiry']} | {it['ticker']} | strike {it.get('strike')} | "
                            f"OI {it.get('oi')} Vol {it.get('vol')} | "
                            f"NBBO {it.get('bid')}/{it.get('ask')} mid={it.get('mid')} | spread%={it.get('spread_pct')}"
                        )
                else:
                    lines = [
                        f"📊 Chain Top 3 (next 3 weeks) for {alert['symbol']} {alert['side']}",
                        "No candidates found. Likely reasons:",
                        f"• After-hours={not rth} (day vol=0 until RTH)",
                        f"• Thresholds in effect (min_vol={scan_min_vol}, min_oi={scan_min_oi})",
                        "• Illiquid expiry/strike for now"
                    ]
                logger.info("Telegram top3 chainscan: sending (%d items)…", len(top3 or []))
                await send_telegram("\n".join(lines))
                logger.info("Telegram top3 chainscan: sent")
            else:
                logger.info("Telegram top3 chainscan: skipped (no bot/chat env)")
        except Exception as e:
            logger.exception("[worker] Telegram top3 chainscan error: %s", e)

    # 6) PRE-FLIGHT + LLM
    pf_ok, pf_checks = preflight_ok(f)

    try:
        logger.info("running LLM analysis…")
        llm = await analyze_with_openai(alert, f)
        consume_llm()
    except Exception as e:
        llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}", "checklist": {}, "ev_estimate": {}}

    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")

    score: Optional[float] = None
    rating: Optional[str] = None
    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    if force_buy:
        decision_final = "buy"

    logger.info("LLM decision=%s conf=%s pf_ok=%s", decision_final, llm.get("confidence"), pf_ok)

    # 7) Telegram final
    tg_result = None
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",
            score=score,
            rating=rating
        )
        if selection_debug.get("selected_by", "").startswith("chain_scan"):
            tg_text += "\n🔎 Note: Contract selected via chain-scan (liquidity + strike/expiry fit)."
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            logger.info("Telegram final alert: sending…")
            tg_result = await send_telegram(tg_text)
            logger.info("Telegram final alert: sent")
        else:
            logger.info("Telegram final alert: skipped (no bot/chat env)")
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    # 8) IBKR order (optional)
    ib_attempted = False
    ib_result_obj: Optional[Any] = None
    try:
        if (decision_final == "buy") and ib_enabled and (pf_ok or force_buy):
            ib_attempted = True
            mode = IBKR_ORDER_MODE
            mid = f.get("mid")
            if mode == "market":
                use_market = True
            elif mode == "limit":
                use_market = (mid is None)
            else:
                use_market = not (IBKR_USE_MID_AS_LIMIT and (mid is not None))
            limit_px = None if use_market else float(mid) if mid is not None else None

            logger.info("IBKR placing order qty=%s mode=%s", qty, IBKR_ORDER_MODE)
            ib_result_obj = await place_recommended_option_order(
                symbol=alert["symbol"],
                side=alert["side"],
                strike=float(desired_strike),
                expiry_iso=chosen_expiry,
                quantity=int(qty),
                limit_price=limit_px,
                action="BUY",
                tif=IBKR_TIF,
            )
        else:
            logger.info("IBKR not attempted (decision=%s ib_enabled=%s pf_ok=%s force_buy=%s)",
                        decision_final, ib_enabled, pf_ok, force_buy)
    except Exception as e:
        ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    _COOLDOWN[(alert["symbol"], alert["side"])] = datetime.now(timezone.utc)

    # 9) Decision log
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": alert["symbol"],
        "side": alert["side"],
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": f"llm.{decision_final}",
        "prescore": None,
        "llm": {
            "ran": True,
            "decision": llm.get("decision"),
            "confidence": llm.get("confidence"),
            "reason": llm.get("reason"),
        },
        "features": {
            "reco_expiry": chosen_expiry,
            "oi": f.get("oi"), "vol": f.get("vol"),
            "bid": f.get("bid"), "ask": f.get("ask"),
            "mark": f.get("mid"), "last": f.get("last"),
            "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
            "prev_close": f.get("prev_close"), "quote_change_pct": f.get("quote_change_pct"),
            "delta": f.get("delta"), "gamma": f.get("gamma"), "theta": f.get("theta"), "vega": f.get("vega"),
            "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
            "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag"),
            "prev_day_high": f.get("prev_day_high"), "prev_day_low": f.get("prev_day_low"),
            "premarket_high": f.get("premarket_high"), "premarket_low": f.get("premarket_low"),
            "vwap": f.get("vwap"), "vwap_dist": f.get("vwap_dist"),
            "above_pdh": f.get("above_pdh"), "below_pdl": f.get("below_pdl"),
            "above_pmh": f.get("above_pmh"), "below_pml": f.get("below_pml"),
            "nbbo_http_status": f.get("nbbo_http_status"),
            "nbbo_reason": f.get("nbbo_reason"),
        },
        "pm_contracts": {
            "plus5_call": {"strike": pm["strike_call"], "contract": pm["contract_call"]},
            "minus5_put": {"strike": pm["strike_put"],  "contract": pm["contract_put"]},
        },
        "preflight": pf_checks,
        "selection_debug": selection_debug,
        "telegram": {
            "configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "result": tg_result
        },
        "ibkr": {
            "enabled": ib_enabled,
            "attempted": ib_attempted,
            "result": (_ibkr_result_to_dict(ib_result_obj) if ib_result_obj is not None else None),
        },
    })

# =========================
# Diagnostics helpers (routes will call)
# =========================
async def diag_polygon_bundle(underlying: str, contract: str) -> Dict[str, Any]:
    if HTTP is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    enc = _encode_ticker_path(contract)
    out = {}

    m = re.search(r":([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$", contract)
    if m:
        yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
        expiry_iso = f"20{yy}-{mm}-{dd}"
        side = "call" if cp.upper() == "C" else "put"
        out["multi"] = await _http_json(
            HTTP,
            f"https://api.polygon.io/v3/snapshot/options/{underlying}",
            {"apiKey": POLYGON_API_KEY, "contract_type": side, "expiration_date": expiry_iso, "limit": 5, "greeks": "true"},
            timeout=6.0
        )

    out["single"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    out["last_quote"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v3/quotes/options/{enc}/last",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    out["open_close"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v1/open-close/options/{enc}/{yday}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    now_utc_dt = datetime.now(timezone.utc)
    frm_iso = datetime(now_utc_dt.year, now_utc_dt.month, now_utc_dt.day, 0,0,0,tzinfo=timezone.utc).isoformat()
    to_iso = now_utc_dt.isoformat()
    out["aggs"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/min/{frm_iso}/{to_iso}?",
        {"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )

    def skim(d):
        if not isinstance(d, dict): return d
        res = d.get("results")
        return {
            "keys": list(d.keys())[:10],
            "sample": (res[:2] if isinstance(res, list) else (res if isinstance(res, dict) else d)),
            "status_hint": d.get("status"),
        }
    return {
        "multi": skim(out.get("multi")),
        "single": skim(out.get("single")),
        "last_quote": skim(out.get("last_quote")),
        "open_close": skim(out.get("open_close")),
        "aggs": skim(out.get("aggs")),
    }

async def net_debug_info() -> Dict[str, Any]:
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    out_ip = None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            out_ip = (await c.get("https://ifconfig.me/ip")).text.strip()
    except Exception as e:
        out_ip = f"fetch-failed: {e.__class__.__name__}"
    can_connect = None
    err = None
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.close()
        can_connect = True
    except Exception as e:
        can_connect = False
        err = f"{e.__class__.__name__}: {e}"
    return {"ibkr_host": host, "ibkr_port": port, "egress_ip": out_ip, "connect_test": can_connect, "error": err}
