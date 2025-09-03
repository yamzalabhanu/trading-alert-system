# routes.py
import os
import asyncio
import socket
import json
import re
from fastapi import Query
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
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
from polygon_client import (
    list_contracts_for_expiry,
    get_option_snapshot,     # may have either (client, symbol, contract) or (symbol, contract)
    build_option_contract,   # OCC builder (O:<TICKER><YYMMDD><C/P><STRIKE*1000, 8 digits>)
)
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG, _send_daily_report_now, _summarize_day_for_report, _chunk_lines_for_telegram

router = APIRouter()

# =========================
# Global state / resources
# =========================
_llm_quota: Dict[str, Any] = {"date": None, "used": 0}
_COOLDOWN: Dict[Tuple[str, str], datetime] = {}

# Shared HTTP client + background queue/worker
HTTP: httpx.AsyncClient | None = None
WORK_Q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
WORKER_COUNT = 0  # for stats

# --- IBKR toggles (env defaults; can be overridden via query params) ---
def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

IBKR_ENABLED = _env_truthy(os.getenv("IBKR_ENABLED", "0"))           # default behavior from env
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
        try:
            payload["raw"] = getattr(res, "raw", None) or repr(res)
        except Exception:
            payload["raw"] = repr(res)
        return payload
    except Exception as e:
        return {"ok": False, "error": f"serialize-failed: {type(e).__name__}: {e}", "raw": repr(res)}

# =========================
# Strike normalization helpers
# =========================
def _normalize_poly_strike(raw) -> Optional[float]:
    """
    Polygon v3 sometimes reports strike as:
      - float (e.g., 155.0) OR
      - int like 155000 (strike * 1000)
    Normalize to price dollars.
    """
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    # Heuristic: treat larger numbers as *1000 (covers 2000+ safely)
    return v / 1000.0 if v >= 2000 else v

def _within_reasonable_strike(diff: float, step: float) -> bool:
    """
    Accept a fallback only if it's reasonably close.
    step = common increment at this underlying (0.5, 1, 5, 10)
    Rule: within <= 3 increments OR <= 10 dollars, whichever is larger.
    """
    return diff <= max(3 * step, 10.0)

# =========================
# Polygon wrappers (robust)
# =========================
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
    """
    Robustly call polygon_client.get_option_snapshot regardless of whether it expects:
      - (client, *, symbol=..., contract=...)  OR
      - (symbol, contract)
    """
    try:
        return await get_option_snapshot(client, symbol=underlying, contract=option_ticker)
    except TypeError as e1:
        try:
            return await get_option_snapshot(underlying, option_ticker)
        except TypeError as e2:
            raise RuntimeError(
                f"get_option_snapshot signature mismatch: "
                f"tried (client,symbol,contract): {e1}; "
                f"tried (symbol,contract): {e2}"
            )

# =========================
# Lifespan: startup/shutdown
# =========================

@router.on_event("startup")
async def _startup():
    global HTTP, WORKER_COUNT
    HTTP = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(read=6.0, write=6.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    WORKER_COUNT = int(os.getenv("WORKERS", "3"))
    for _ in range(WORKER_COUNT):
        asyncio.create_task(_worker())

@router.on_event("shutdown")
async def _shutdown():
    global HTTP
    if HTTP:
        await HTTP.aclose()

async def _worker():
    """Background worker that processes TradingView jobs."""
    while True:
        job = await WORK_Q.get()
        try:
            await _process_tradingview_job(job)
        except Exception as e:
            print(f"[worker] error: {e!r}")
        finally:
            WORK_Q.task_done()

# =========================
# Helpers for ±5% contracts
# =========================
def _build_plus_minus_contracts(symbol: str, ul_px: float, expiry_iso: str) -> Dict[str, Any]:
    """
    Returns dict with:
      strike_call (+5%), strike_put (-5%),
      contract_call, contract_put
    """
    call_strike = round_strike_to_common_increment(ul_px * 1.05)
    put_strike  = round_strike_to_common_increment(ul_px * 0.95)

    contract_call = build_option_contract(symbol, expiry_iso, "CALL", call_strike)
    contract_put  = build_option_contract(symbol, expiry_iso, "PUT",  put_strike)

    return {
        "strike_call": call_strike,
        "strike_put":  put_strike,
        "contract_call": contract_call,
        "contract_put":  contract_put,
    }

# =========================
# Time helpers
# =========================
def _quote_age_from_ts(ts_val: Any) -> Optional[float]:
    """
    Try to compute age (seconds) from polygon timestamps (ns/us/ms/s).
    """
    if ts_val is None:
        return None
    try:
        ns = int(ts_val)
    except Exception:
        return None
    # Scale guess
    if ns >= 10**14:
        sec = ns / 1e9      # ns
    elif ns >= 10**11:
        sec = ns / 1e6      # us
    elif ns >= 10**8:
        sec = ns / 1e3      # ms
    else:
        sec = float(ns)     # s
    age = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    return round(age, 3)

# =========================
# OPTIONAL: snapshot backfill helpers
# =========================
async def _poly_option_backfill(
    client: httpx.AsyncClient,
    symbol: str,
    option_ticker: str,
    today_utc: date,
) -> Dict[str, Any]:
    """
    Try to backfill missing fields using:
      1) Snapshot again (parse nested fields into flat values: OI, Vol, NBBO, Greeks, quote_age)
      2) Previous day open/close (OI/Vol)
      3) Last quote (NBBO + quote age)
    Returns only keys that are discovered (non-None).
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    # 1) Snapshot parse (preferred)
    try:
        snap = await polygon_get_option_snapshot(client, underlying=symbol, option_ticker=option_ticker)
        if isinstance(snap, dict):
            res = snap.get("results") or {}
            if isinstance(res, dict):
                # OI
                oi = res.get("open_interest")
                if oi is not None:
                    out["oi"] = oi
                # Volume
                day_block = res.get("day") or {}
                vol = day_block.get("volume")
                if vol is not None:
                    out["vol"] = vol
                # NBBO + quote age
                lq = res.get("last_quote") or {}
                bid_px = lq.get("bid_price")
                ask_px = lq.get("ask_price")
                if bid_px is not None:
                    out["bid"] = float(bid_px)
                if ask_px is not None:
                    out["ask"] = float(ask_px)
                if out.get("bid") is not None and out.get("ask") is not None:
                    out["mid"] = round((out["bid"] + out["ask"]) / 2.0, 4)
                ts = lq.get("sip_timestamp") or lq.get("participant_timestamp") or lq.get("trf_timestamp")
                age = _quote_age_from_ts(ts)
                if age is not None:
                    out["quote_age_sec"] = age
                # Greeks
                greeks = res.get("greeks") or {}
                for k_src, k_dst in [("delta", "delta"), ("gamma", "gamma"), ("theta", "theta"), ("vega", "vega")]:
                    v = greeks.get(k_src)
                    if v is not None:
                        out[k_dst] = v
                # IV (per-snapshot)
                iv = res.get("implied_volatility") or greeks.get("iv")
                if iv is not None:
                    out["iv"] = iv
    except Exception:
        pass

    # 2) Previous day open/close for OI / Volume
    try:
        yday = (today_utc - timedelta(days=1)).isoformat()
        r1 = await client.get(
            f"https://api.polygon.io/v1/open-close/options/{option_ticker}/{yday}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        if r1.status_code == 200:
            js = r1.json() or {}
            oi = js.get("open_interest")
            vol = js.get("volume")
            if (out.get("oi") is None) and (oi is not None):
                out["oi"] = oi
            if (out.get("vol") is None) and (vol is not None):
                out["vol"] = vol
    except Exception:
        pass

    # 3) Last quote for NBBO (if still missing)
    try:
        r2 = await client.get(
            f"https://api.polygon.io/v3/quotes/options/{option_ticker}/last",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        if r2.status_code == 200:
            js = r2.json() or {}
            res = js.get("results") or {}
            last = res.get("last") or res
            bid_px = last.get("bidPrice")
            ask_px = last.get("askPrice")
            if out.get("bid") is None and bid_px is not None:
                out["bid"] = float(bid_px)
            if out.get("ask") is None and ask_px is not None:
                out["ask"] = float(ask_px)
            if out.get("mid") is None and out.get("bid") is not None and out.get("ask") is not None:
                out["mid"] = round((out["bid"] + out["ask"]) / 2.0, 4)
            ts = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
            age = _quote_age_from_ts(ts)
            if age is not None and out.get("quote_age_sec") is None:
                out["quote_age_sec"] = age
    except Exception:
        pass

    return out

# =========================
# Core processing (moved off-path)
# =========================
async def _process_tradingview_job(job: Dict[str, Any]) -> None:
    """
    job keys:
      - alert_text (str)
      - flags: dict(ib_enabled, force, force_buy, qty, debug)
    """
    global HTTP
    if HTTP is None:
        print("[worker] HTTP client not ready")
        return

    try:
        alert = parse_alert_text(job["alert_text"])
    except Exception as e:
        print(f"[worker] bad alert payload: {e}")
        return

    # Flags
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force = bool(job["flags"].get("force", False))
    force_buy = bool(job["flags"].get("force_buy", False))
    qty = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))
    debug_mode = bool(job["flags"].get("debug", False))

    # ===== Recommendation policy — compute two-week Friday expiry (never same-week) =====
    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    # Build ±5% strikes and OCC contracts deterministically
    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    if alert["side"] == "CALL":
        desired_strike = pm["strike_call"]
        option_ticker  = pm["contract_call"]
    else:
        desired_strike = pm["strike_put"]
        option_ticker  = pm["contract_put"]

    f: Dict[str, Any] = {}
    selection_debug: Dict[str, Any] = {
        "desired_strike": desired_strike,
        "selected_ticker_initial": option_ticker,
        "fallback_considered": [],
        "fallback_applied": False,
    }

    # ===== Data prep (Polygon/features) =====
    try:
        if not POLYGON_API_KEY:
            # OFFLINE MODE (fabricate minimal set)
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
            # ONLINE MODE — first try direct snapshot for our OCC contract
            snap = await polygon_get_option_snapshot(HTTP, underlying=alert["symbol"], option_ticker=option_ticker)

            # Fallback: if snapshot looks empty (illiquid/missing), search the expiry and pick nearest strike (normalized)
            if not snap or not snap.get("results"):
                contracts = await polygon_list_contracts_for_expiry(
                    HTTP, symbol=alert["symbol"], expiry=target_expiry, side=alert["side"], limit=250
                )
                if contracts:
                    candidates = []
                    for c in contracts:
                        cs = _normalize_poly_strike(c.get("strike"))
                        if cs is None:
                            tk = c.get("ticker") or c.get("symbol") or c.get("contract") or ""
                            m = re.search(r"[CP](\d{8,9})$", tk or "")
                            if m:
                                try:
                                    cs = int(m.group(1)) / 1000.0
                                except Exception:
                                    cs = None
                        if cs is not None:
                            diff = abs(cs - desired_strike)
                            candidates.append((diff, cs, c))

                    # sort by closeness
                    candidates.sort(key=lambda x: x[0])
                    if debug_mode:
                        selection_debug["fallback_considered"] = [
                            {
                                "diff": d,
                                "strike_norm": s,
                                "ticker": cand.get("ticker") or cand.get("symbol") or cand.get("contract")
                            }
                            for d, s, cand in candidates[:5]
                        ]

                    if candidates:
                        best_diff, best_strike_norm, best_contract = candidates[0]
                        # sanity gate: only replace if reasonably close
                        step = 0.5 if ul_px < 25 else (1 if ul_px < 200 else (5 if ul_px < 1000 else 10))
                        if _within_reasonable_strike(best_diff, step):
                            fallback_ticker = (
                                best_contract.get("ticker")
                                or best_contract.get("symbol")
                                or best_contract.get("contract")
                            )
                            if fallback_ticker:
                                option_ticker = fallback_ticker
                                selection_debug["fallback_applied"] = True
                                selection_debug["selected_ticker_fallback"] = option_ticker
                                selection_debug["selected_strike_norm"] = best_strike_norm
                                # fetch snapshot for the fallback
                                snap = await polygon_get_option_snapshot(
                                    HTTP, underlying=alert["symbol"], option_ticker=option_ticker
                                )

            f = await build_features(
                HTTP,
                alert={**alert, "strike": desired_strike, "expiry": target_expiry},
                snapshot=snap
            )

            # ---- Backfill: parse snapshot + other endpoints to fill sparse fields ----
            extra_from_snap = await _poly_option_backfill(HTTP, alert["symbol"], option_ticker, today_utc)
            if extra_from_snap:
                for k, v in extra_from_snap.items():
                    if v is not None and (f.get(k) is None):
                        f[k] = v
                if f.get("mid") is None and f.get("bid") is not None and f.get("ask") is not None:
                    f["mid"] = round((float(f["bid"]) + float(f["ask"])) / 2.0, 4)

    except Exception as e:
        print(f"[worker] Polygon/features error: {e}")
        f = f or {"dte": (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days}

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
        try:
            llm = await analyze_with_openai(alert, f)
            consume_llm()
        except Exception as e:
            llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}", "checklist": {}, "ev_estimate": {}}

        llm_ran = True
        decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")
        try:
            score = compute_decision_score(f, llm)
            rating = map_score_to_rating(score, llm.get("decision"))
        except Exception:
            score, rating = None, None
        decision_path = f"llm.{decision_final}"

        if force_buy:
            decision_final = "buy"
            decision_path = "force.buy"

        try:
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
        except Exception as e:
            print(f"[worker] Telegram error: {e}")

        try:
            if decision_final == "buy" and ib_enabled:
                ib_attempted = True
                mode = IBKR_ORDER_MODE
                mid = f.get("mid")
                if mode == "market":
                    use_market = True
                elif mode == "limit":
                    use_market = (mid is None)
                else:  # auto
                    use_market = not (IBKR_USE_MID_AS_LIMIT and (mid is not None))
                limit_px = None if use_market else float(mid) if mid is not None else None

                ib_result_obj = await place_recommended_option_order(
                    symbol=alert["symbol"],
                    side=alert["side"],
                    strike=float(desired_strike),
                    expiry_iso=target_expiry,
                    quantity=int(qty),
                    limit_price=limit_px,
                    action="BUY",
                    tif=IBKR_TIF,
                )
        except Exception as e:
            ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    else:
        llm_reason = "Outside processing windows (Allowed: 08:30–11:30 & 14:00–15:00 CDT). LLM + Telegram skipped."
        ib_attempted = False
        ib_result_obj = None

    _COOLDOWN[(alert["symbol"], alert["side"])] = datetime.now(timezone.utc)

    log_entry = {
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
        },
        "pm_contracts": {
            "plus5_call": {"strike": pm["strike_call"], "contract": pm["contract_call"]},
            "minus5_put": {"strike": pm["strike_put"],  "contract": pm["contract_put"]},
        }
    }
    if debug_mode:
        log_entry["selection_debug"] = selection_debug

    _DECISIONS_LOG.append(log_entry)

# =========================
# Routes
# =========================

@router.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@router.get("/healthz")
def healthz():
    return {"ok": True}

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

@router.get("/worker/stats")
def worker_stats():
    return {
        "ok": True,
        "queue_size": WORK_Q.qsize(),
        "queue_maxsize": WORK_Q.maxsize,
        "workers": WORKER_COUNT,
    }

@router.post("/run/daily_report")
async def run_daily_report():
    res = await _send_daily_report_now()
    return {"ok": True, "trigger": "manual", **res}

@router.get("/net/debug")
async def net_debug():
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

@router.get("/report/preview")
def report_preview():
    today_local = market_now().date()
    rep = _summarize_day_for_report(today_local)
    chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")
    return {"ok": True, "header": rep["header"], "contract_chunks": chunks, "count": rep["count"]}

# --- Non-blocking webhook: ACKs immediately and enqueues work ---
@router.post("/webhook", response_class=JSONResponse)
@router.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(
    request: Request,
    offline: int = Query(default=0),
    ib: int = Query(default=0),
    qty: int = Query(default=IBKR_DEFAULT_QTY),
    force: int = Query(default=0),
    force_buy: int = Query(default=0),
    debug: int = Query(default=0),
):
    payload_text = await _get_alert_text(request)
    if not payload_text:
        raise HTTPException(status_code=400, detail="Empty alert payload")

    try:
        _ = parse_alert_text(payload_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid alert: {e}")

    effective_ib_enabled = bool(ib) if request.query_params.get("ib") is not None else IBKR_ENABLED

    job = {
        "alert_text": payload_text,
        "flags": {
            "ib_enabled": effective_ib_enabled and (not offline),
            "force": bool(force),
            "force_buy": bool(force_buy),
            "qty": int(qty),
            "debug": bool(debug),
        }
    }

    try:
        WORK_Q.put_nowait(job)
    except asyncio.QueueFull:
        return JSONResponse({"status": "busy", "detail": "queue full"}, status_code=429)

    return JSONResponse({"status": "accepted"}, status_code=202)
