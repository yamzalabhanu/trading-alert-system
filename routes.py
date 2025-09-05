# routes.py
import os
import asyncio
import socket
import re
from urllib.parse import quote
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
    WINDOWS_CDT,
    MAX_LLM_PER_DAY,
)
from models import Alert, WebhookResponse
from polygon_client import (
    list_contracts_for_expiry,
    get_option_snapshot,     # may be (client, symbol, contract) or (symbol, contract)
    build_option_contract,   # OCC builder: O:<SYM><YYMMDD><C/P><STRIKE*1000>
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

HTTP: httpx.AsyncClient | None = None
WORK_Q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
WORKER_COUNT = 0

# --- IBKR toggles ---
def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

IBKR_ENABLED = _env_truthy(os.getenv("IBKR_ENABLED", "0"))
IBKR_DEFAULT_QTY = int(os.getenv("IBKR_DEFAULT_QTY", "1"))
IBKR_TIF = os.getenv("IBKR_TIF", "DAY").upper()
IBKR_ORDER_MODE = os.getenv("IBKR_ORDER_MODE", "auto").lower()   # auto | market | limit
IBKR_USE_MID_AS_LIMIT = os.getenv("IBKR_USE_MID_AS_LIMIT", "1") == "1"

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

# ========== Utils ==========
def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def flag_from(req: Request, name: str, env_name: str, default: bool = False) -> bool:
    qv = req.query_params.get(name)
    if qv is not None:
        return _truthy(qv)
    return _truthy(os.getenv(env_name, "1" if default else "0"))

def market_now() -> datetime:
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

async def _get_alert_text(request: Request) -> str:
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
# HTTP helpers
# =========================
def _encode_ticker_path(t: str) -> str:
    return quote(t or "", safe="")

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

async def _http_json_url(client: httpx.AsyncClient, url: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """GET a fully-formed URL (used for Polygon next_url pages)."""
    try:
        r = await client.get(url, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

# =========================
# Strike helpers
# =========================
def _normalize_poly_strike(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    return v / 1000.0 if v >= 2000 else v

def _within_reasonable_strike(diff: float, step: float) -> bool:
    return diff <= max(3 * step, 10.0)

# =========================
# Polygon wrappers
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
# Time helpers
# =========================
def _quote_age_from_ts(ts_val: Any) -> Optional[float]:
    if ts_val is None:
        return None
    try:
        ns = int(ts_val)
    except Exception:
        return None
    if ns >= 10**14:
        sec = ns / 1e9
    elif ns >= 10**11:
        sec = ns / 1e6
    elif ns >= 10**8:
        sec = ns / 1e3
    else:
        sec = float(ns)
    age = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    return round(age, 3)

def _today_utc_range_for_aggs(now_utc: datetime) -> Tuple[str, str]:
    start = datetime(now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return start, now_utc.isoformat()

# =========================
# Quote sampler
# =========================
async def _sample_best_quote(client, enc_opt, tries=5, delay=0.6) -> Optional[Dict[str, Any]]:
    best = {}
    for _ in range(tries):
        lastq = await _http_json(client, f"https://api.polygon.io/v3/quotes/options/{enc_opt}/last",
                                 {"apiKey": POLYGON_API_KEY}, timeout=3.0)
        if lastq:
            res = lastq.get("results") or {}
            last = res.get("last") or res
            bid = last.get("bidPrice")
            ask = last.get("askPrice")
            ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
            if bid is not None and ask is not None and ask >= bid:
                mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
                spread_pct = ((ask - bid) / mid * 100) if (mid and mid > 0) else None
                age = _quote_age_from_ts(ts)
                cand = {
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "mid": float(mid) if mid is not None else None,
                    "quote_age_sec": age,
                    "option_spread_pct": round(spread_pct, 3) if spread_pct is not None else None
                }
                if not best or \
                   (cand["option_spread_pct"] or 1e9) < (best.get("option_spread_pct") or 1e9) or \
                   (cand["quote_age_sec"] or 1e9) < (best.get("quote_age_sec") or 1e9):
                    best = cand
        await asyncio.sleep(delay)
    return best or None

# =========================
# Backfill layers (multi-snapshot FIRST)
# =========================
async def _poly_option_backfill(
    client: httpx.AsyncClient,
    symbol: str,
    option_ticker: str,
    today_utc: date,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    def _apply_from_results(res: Dict[str, Any], out_dict: Dict[str, Any]) -> None:
        if not isinstance(res, dict):
            return

        # OI (top-level)
        oi = res.get("open_interest")
        if oi is not None:
            out_dict["oi"] = oi

        # Day -> volume
        day_block = res.get("day") or {}
        vol = day_block.get("volume", day_block.get("v"))
        if vol is not None:
            out_dict["vol"] = vol

        # Prefer last_quote
        lq = res.get("last_quote") or {}
        bid_px = lq.get("bid_price")
        ask_px = lq.get("ask_price")
        if bid_px is not None:
            out_dict["bid"] = float(bid_px)
        if ask_px is not None:
            out_dict["ask"] = float(ask_px)
        if out_dict.get("mid") is None and out_dict.get("bid") is not None and out_dict.get("ask") is not None:
            out_dict["mid"] = round((out_dict["bid"] + out_dict["ask"]) / 2.0, 4)

        # Age (prefer quote ts)
        ts = (
            lq.get("sip_timestamp")
            or lq.get("participant_timestamp")
            or lq.get("trf_timestamp")
            or lq.get("t")
        )
        if ts is None:
            lt = res.get("last_trade") or {}
            lt_px = lt.get("price")
            if isinstance(lt_px, (int, float)) and out_dict.get("mid") is None:
                out_dict["mid"] = float(lt_px)
            ts = (
                lt.get("sip_timestamp")
                or lt.get("participant_timestamp")
                or lt.get("trf_timestamp")
                or lt.get("t")
            )
        age = _quote_age_from_ts(ts)
        if age is not None:
            out_dict["quote_age_sec"] = age

        # Greeks & IV
        greeks = res.get("greeks") or {}
        for k_src in ("delta", "gamma", "theta", "vega"):
            v = greeks.get(k_src)
            if v is not None:
                out_dict[k_src] = v
        iv = res.get("implied_volatility") or greeks.get("iv")
        if iv is not None:
            out_dict["iv"] = iv

    # 1) Multi-snapshot (list)
    try:
        m = re.search(r":([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$", option_ticker)
        if m:
            yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
            expiry_iso = f"20{yy}-{mm}-{dd}"
            side = "call" if cp.upper() == "C" else "put"
            rlist = await _http_json(
                client,
                f"https://api.polygon.io/v3/snapshot/options/{symbol}",
                {
                    "apiKey": POLYGON_API_KEY,
                    "contract_type": side,
                    "expiration_date": expiry_iso,
                    "limit": 1000,
                    "greeks": "true",
                    "include_greeks": "true",
                },
                timeout=8.0
            )
            if rlist:
                items = rlist.get("results") or []
                chosen = None
                for it in items:
                    tkr = (it.get("details") or {}).get("ticker") or it.get("ticker")
                    if tkr == option_ticker:
                        chosen = it
                        break
                if not chosen and items:
                    # fallback by strike proximity
                    desired = None
                    mm3 = re.search(r"[CP](\d{8,9})$", option_ticker)
                    if mm3:
                        try:
                            desired = int(mm3.group(1)) / 1000.0
                        except Exception:
                            desired = None
                    cands = []
                    for it in items:
                        det = it.get("details") or {}
                        s_norm = _normalize_poly_strike(det.get("strike"))
                        if s_norm is None:
                            tk = det.get("ticker") or it.get("ticker") or ""
                            mm2 = re.search(r"[CP](\d{8,9})$", tk)
                            if mm2:
                                try:
                                    s_norm = int(mm2.group(1)) / 1000.0
                                except Exception:
                                    s_norm = None
                        if s_norm is not None and desired is not None:
                            cands.append((abs(s_norm - desired), it))
                    if cands:
                        cands.sort(key=lambda x: x[0])
                        chosen = cands[0][1]
                if chosen:
                    _apply_from_results(chosen, out)
    except Exception:
        pass

    # 2) Single-contract snapshot
    if not out:
        try:
            enc_opt = _encode_ticker_path(option_ticker)
            snap = await _http_json(
                client,
                f"https://api.polygon.io/v3/snapshot/options/{symbol}/{enc_opt}",
                {"apiKey": POLYGON_API_KEY},
                timeout=8.0
            )
            if snap:
                _apply_from_results(snap.get("results") or {}, out)
        except Exception:
            pass

    # 3) Previous-day open/close (T+1 OI/Vol)
    try:
        yday = (today_utc - timedelta(days=1)).isoformat()
        enc_opt = _encode_ticker_path(option_ticker)
        oc = await _http_json(
            client,
            f"https://api.polygon.io/v1/open-close/options/{enc_opt}/{yday}",
            {"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        if oc:
            oi = oc.get("open_interest")
            vol = oc.get("volume")
            if out.get("oi") is None and oi is not None:
                out["oi"] = oi
            if out.get("vol") is None and vol is not None:
                out["vol"] = vol
    except Exception:
        pass

    # 4) Sample best quote (tightest/freshest)
    try:
        enc_opt = _encode_ticker_path(option_ticker)
        sampled = await _sample_best_quote(client, enc_opt, tries=5, delay=0.6)
        if sampled:
            for k, v in sampled.items():
                if v is not None:
                    out[k] = v
    except Exception:
        pass

    # 5) Minute aggregates (intraday volume sum if missing)
    try:
        if out.get("vol") is None:
            enc_opt = _encode_ticker_path(option_ticker)
            now_utc_dt = datetime.now(timezone.utc)
            frm_iso, to_iso = _today_utc_range_for_aggs(now_utc_dt)
            aggs = await _http_json(
                client,
                f"https://api.polygon.io/v2/aggs/ticker/{enc_opt}/range/1/min/{frm_iso}/{to_iso}",
                {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY},
                timeout=8.0
            )
            if aggs:
                results = aggs.get("results") or []
                if results:
                    vol_sum = 0
                    for bar in results:
                        v = bar.get("v")
                        if isinstance(v, (int, float)):
                            vol_sum += v
                    if vol_sum > 0:
                        out["vol"] = int(vol_sum)
    except Exception:
        pass

    return out

# =========================
# ±5% helpers
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
# Delta-targeted selector
# =========================
async def _choose_best_contract(
    client: httpx.AsyncClient,
    symbol: str,
    expiry_iso: str,
    side: str,
    ul_px: float,
    desired_strike: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (ticker, debug_info)
    Ranking:
      1) |delta - target|  (0.35 for calls, -0.35 for puts)
      2) lower spread%
      3) higher intraday volume
      4) higher OI
      5) |strike - desired_strike|
    """
    debug = {"candidates": []}
    if not POLYGON_API_KEY:
        base = _build_plus_minus_contracts(symbol, ul_px, expiry_iso)
        t = base["contract_call"] if side == "CALL" else base["contract_put"]
        return t, {"reason": "offline/no key"}

    chain = await polygon_list_contracts_for_expiry(client, symbol=symbol, expiry=expiry_iso, side=side, limit=1000)
    if not chain:
        base = _build_plus_minus_contracts(symbol, ul_px, expiry_iso)
        t = base["contract_call"] if side == "CALL" else base["contract_put"]
        return t, {"reason": "no_chain"}

    try:
        side_poly = "call" if side.upper() == "CALL" else "put"
        mlist = await _http_json(
            client,
            f"https://api.polygon.io/v3/snapshot/options/{symbol}",
            {
                "apiKey": POLYGON_API_KEY,
                "contract_type": side_poly,
                "expiration_date": expiry_iso,
                "limit": 1000,
                "greeks": "true",
            },
            timeout=8.0
        )
    except Exception:
        mlist = None

    index_by_ticker = {}
    if mlist:
        for it in (mlist.get("results") or []):
            tk = (it.get("details") or {}).get("ticker") or it.get("ticker")
            if tk:
                index_by_ticker[tk] = it

    cands: List[Tuple] = []
    for c in chain:
        tk = c.get("ticker") or c.get("symbol") or c.get("contract")
        if not tk:
            continue
        det = index_by_ticker.get(tk) or {}
        greeks = det.get("greeks") or {}
        delta = greeks.get("delta")
        oi = det.get("open_interest")
        day_block = det.get("day") or {}
        vol = day_block.get("volume")
        enc = _encode_ticker_path(tk)
        q = await _http_json(client, f"https://api.polygon.io/v3/quotes/options/{enc}/last",
                             {"apiKey": POLYGON_API_KEY}, timeout=3.0)
        spread_pct = None
        if q:
            res = q.get("results") or {}
            last = res.get("last") or res
            b, a = last.get("bidPrice"), last.get("askPrice")
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0 and b >= 0:
                mid = (a + b)/2.0
                if mid > 0:
                    spread_pct = (a - b)/mid*100.0

        tgt = TARGET_DELTA_CALL if side == "CALL" else TARGET_DELTA_PUT
        delta_miss = abs((delta if isinstance(delta, (int, float)) else tgt) - tgt)

        s_norm = _normalize_poly_strike(c.get("strike"))
        if s_norm is None:
            m2 = re.search(r"[CP](\d{8,9})$", tk)
            if m2:
                try:
                    s_norm = int(m2.group(1)) / 1000.0
                except Exception:
                    s_norm = None
        strike_miss = abs((s_norm if s_norm is not None else desired_strike) - desired_strike)

        rank_tuple = (
            delta_miss,
            (spread_pct if spread_pct is not None else 1e9),
            -(vol or 0),
            -(oi or 0),
            strike_miss,
        )
        cands.append((rank_tuple, tk, delta, spread_pct, vol, oi, s_norm))

    if not cands:
        base = _build_plus_minus_contracts(symbol, ul_px, expiry_iso)
        t = base["contract_call"] if side == "CALL" else base["contract_put"]
        return t, {"reason": "no_candidates"}

    cands.sort(key=lambda x: x[0])
    top = cands[0]
    debug["candidates"] = [
        {
            "ticker": tk,
            "rank": r[0],
            "delta": d,
            "spread_pct": sp,
            "vol": v,
            "oi": oi,
            "strike": s,
        }
        for r, tk, d, sp, v, oi, s in cands[:5]
    ]
    return top[1], debug

# =========================
# Preflight (non-blocking)
# =========================
def preflight_ok(f: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    checks = {}
    checks["quote_fresh"] = (f.get("quote_age_sec") is not None and f["quote_age_sec"] <= MAX_QUOTE_AGE_S)
    checks["spread_ok"]   = (f.get("option_spread_pct") is not None and f["option_spread_pct"] <= MAX_SPREAD_PCT)
    checks["vol_ok"]      = (f.get("vol") or 0) >= MIN_VOL_TODAY
    checks["oi_ok"]       = (f.get("oi") or 0) >= MIN_OI
    dte_val = f.get("dte")
    checks["dte_ok"]      = (dte_val is not None) and (MIN_DTE <= dte_val <= MAX_DTE)
    ok = all(checks.values())
    return ok, checks

# =========================
# Lifespan
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
    while True:
        job = await WORK_Q.get()
        try:
            await _process_tradingview_job(job)
        except Exception as e:
            print(f"[worker] error: {e!r}")
        finally:
            WORK_Q.task_done()

# =========================
# Core processing
# =========================
async def _process_tradingview_job(job: Dict[str, Any]) -> None:
    global HTTP
    if HTTP is None:
        print("[worker] HTTP client not ready")
        return

    try:
        alert = parse_alert_text(job["alert_text"])
    except Exception as e:
        print(f"[worker] bad alert payload: {e}")
        return

    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force = bool(job["flags"].get("force", False))
    force_buy = bool(job["flags"].get("force_buy", False))
    qty = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))
    debug_mode = bool(job["flags"].get("debug", False))

    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    # ±5% contracts for reference/logs
    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    desired_strike = pm["strike_call"] if alert["side"] == "CALL" else pm["strike_put"]

    # Delta-targeted selection with liquidity tie-breakers
    try:
        option_ticker, sel_dbg = await _choose_best_contract(
            HTTP, alert["symbol"], target_expiry, alert["side"], ul_px, desired_strike
        )
    except Exception:
        option_ticker, sel_dbg = (pm["contract_call"] if alert["side"] == "CALL" else pm["contract_put"], {"reason": "selector_error"})

    f: Dict[str, Any] = {}
    selection_debug: Dict[str, Any] = {"desired_strike": desired_strike, "selected_ticker": option_ticker, **(sel_dbg or {})}

    # --- Enrich features (Polygon + internal) ---
    try:
        if not POLYGON_API_KEY:
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
            extra_from_snap = await _poly_option_backfill(HTTP, alert["symbol"], option_ticker, today_utc)
            for k, v in (extra_from_snap or {}).items():
                if v is not None:
                    f[k] = v

            snap = None
            try:
                snap = await polygon_get_option_snapshot(HTTP, underlying=alert["symbol"], option_ticker=option_ticker)
            except Exception:
                snap = None

            core = await build_features(
                HTTP,
                alert={**alert, "strike": desired_strike, "expiry": target_expiry},
                snapshot=snap
            )
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v

            if f.get("mid") is None and f.get("bid") is not None and f.get("ask") is not None:
                f["mid"] = round((float(f["bid"]) + float(f["ask"])) / 2.0, 4)

            try:
                if f.get("dte") is None:
                    f["dte"] = (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days
            except Exception:
                pass

            try:
                bid = f.get("bid")
                ask = f.get("ask")
                mid = f.get("mid")
                if bid is not None and ask is not None:
                    if mid is None:
                        mid = (float(bid) + float(ask)) / 2.0
                        f["mid"] = round(mid, 4)
                    spread = float(ask) - float(bid)
                    if mid and mid > 0:
                        f["option_spread_pct"] = round((spread / mid) * 100.0, 3)
            except Exception:
                pass

    except Exception as e:
        print(f"[worker] Polygon/features error: {e}")
        f = f or {"dte": (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days}

    # ---------- Preflight (do NOT block LLM) ----------
    pf_ok, pf_checks = preflight_ok(f)

    # ---------- Always run LLM ----------
    llm_ran = True
    try:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
    except Exception as e:
        llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}", "checklist": {}, "ev_estimate": {}}

    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")
    decision_path = f"llm.{decision_final}"
    score: Optional[float] = None
    rating: Optional[str] = None
    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    if force_buy:
        decision_final = "buy"
        decision_path = "force.buy"

    # ---------- Telegram ----------
    tg_result = None
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": target_expiry},
            option_ticker=option_ticker,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",  # no “outside window” messages anymore
            score=score,
            rating=rating
        )
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            tg_result = await send_telegram(tg_text)
    except Exception as e:
        print(f"[worker] Telegram error: {e}")

    # ---------- IBKR placement (still gated by pf_ok unless force_buy) ----------
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
            "reason": llm.get("reason"),
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
    }

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
    return {"ok": True, "queue_size": WORK_Q.qsize(), "queue_maxsize": WORK_Q.maxsize, "workers": WORKER_COUNT}

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

# --- Non-blocking webhook: ACK + enqueue ---
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

# ================
# Diagnostics
# ================
@router.get("/diag/polygon")
async def diag_polygon(underlying: str, contract: str):
    if HTTP is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    enc = _encode_ticker_path(contract)
    out = {}

    # multi-snapshot
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

    # single
    out["single"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )

    # last quote
    out["last_quote"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v3/quotes/options/{enc}/last",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )

    # open/close yday
    yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    out["open_close"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v1/open-close/options/{enc}/{yday}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )

    # minute aggs
    now_utc_dt = datetime.now(timezone.utc)
    frm_iso = datetime(now_utc_dt.year, now_utc_dt.month, now_utc_dt.day, 0,0,0,tzinfo=timezone.utc).isoformat()
    to_iso = now_utc_dt.isoformat()
    out["aggs"] = await _http_json(
        HTTP,
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/min/{frm_iso}/{to_iso}",
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

@router.post("/diag/telegram")
async def diag_telegram(payload: dict | None = None):
    """
    Send a quick test message to your Telegram chat to verify BOT_TOKEN/CHAT_ID.
    POST JSON: {"text": "hello"}  (text is optional)
    """
    text = ((payload or {}).get("text")) or "🚨 Telegram test from alert server"
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "error": "Missing TELEGRAM_BOT_TOKEN env var"}
    if not TELEGRAM_CHAT_ID:
        return {"ok": False, "error": "Missing TELEGRAM_CHAT_ID env var"}
    try:
        res = await send_telegram(text)
        return {"ok": True, "sent": True, "text": text, "result": res}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ======================================================
# Chain Scanner (weeks 1/2/3) – calls + puts (pagination)
# ======================================================
async def _poly_paginated_snapshot_list(
    client: httpx.AsyncClient,
    underlying: str,
    expiry_iso: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch snapshot list for BOTH calls and puts for a given expiry, with pagination.
    """
    if not POLYGON_API_KEY:
        return []

    all_rows: List[Dict[str, Any]] = []
    for side in ("call", "put"):
        base = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
        params = {
            "apiKey": POLYGON_API_KEY,
            "expiration_date": expiry_iso,
            "contract_type": side,
            "limit": limit,
            "greeks": "true",
            "include_greeks": "true",
        }
        first = await _http_json(client=client, url=base, params=params, timeout=10.0)
        if first and isinstance(first.get("results"), list):
            all_rows.extend(first["results"])
            nxt = first.get("next_url")
        else:
            nxt = None

        while nxt:
            # ensure apiKey is on the next_url
            if "apiKey=" not in nxt and POLYGON_API_KEY:
                sep = "&" if "?" in nxt else "?"
                nxt = f"{nxt}{sep}apiKey={POLYGON_API_KEY}"
            page = await _http_json_url(client, nxt, timeout=10.0)
            if not page or not isinstance(page.get("results"), list):
                break
            all_rows.extend(page["results"])
            nxt = page.get("next_url")

    return all_rows

@router.post("/scan/options")
async def scan_and_alert_top_liquidity(payload: Dict[str, Any]):
    """
    Body:
    {
      "symbol": "AAPL",
      "minVol": 2000,
      "minOI":  2000,
      "sort":   "oi" | "vol" | "comb" | "comb/oi",
      "top":    8
    }
    """
    if HTTP is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    symbol = str(payload.get("symbol", "")).upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
    if not POLYGON_API_KEY:
        raise HTTPException(status_code=400, detail="POLYGON_API_KEY missing")

    min_vol = int(payload.get("minVol", 2000))
    min_oi  = int(payload.get("minOI",  2000))
    top_n   = int(payload.get("top", 6))
    sort_by = str(payload.get("sort", "oi")).lower()

    # Next 3 Friday expiries starting >= upcoming Friday
    today_utc = datetime.now(timezone.utc).date()
    wk1 = same_week_friday(today_utc)
    if wk1 <= today_utc:
        wk1 = wk1 + timedelta(days=7)
    wk2 = wk1 + timedelta(days=7)
    wk3 = wk2 + timedelta(days=7)
    weeks = [wk1.isoformat(), wk2.isoformat(), wk3.isoformat()]

    buckets = []
    for i, exp in enumerate(weeks, start=1):
        rows = await _poly_paginated_snapshot_list(HTTP, symbol, exp, limit=1000)
        scanned = len(rows)

        items = []
        for r in rows:
            det = r.get("details") or {}
            tk  = det.get("ticker") or r.get("ticker")
            if not tk:
                continue

            # liquidity
            day = r.get("day") or {}
            vol = int(day.get("volume") or day.get("v") or 0)
            oi  = int(r.get("open_interest") or 0)

            # NBBO + spread
            lq = r.get("last_quote") or {}
            b = lq.get("bid_price"); a = lq.get("ask_price")
            mid = None; spread_pct = None
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0:
                mid = (a + b) / 2.0
                if mid > 0:
                    spread_pct = (a - b) / mid * 100.0

            if (vol >= min_vol) or (oi >= min_oi):
                if sort_by == "vol":
                    key = (-vol, -oi)
                elif sort_by == "oi":
                    key = (-oi, -vol)
                elif sort_by == "comb":
                    key = (-(vol + oi), -oi, -vol)
                else:  # "comb/oi"
                    key = (-oi, -(vol + oi), -vol)

                items.append({
                    "ticker": tk,
                    "expiry": exp,
                    "strike": _normalize_poly_strike(det.get("strike")),
                    "type": det.get("contract_type"),
                    "vol": vol,
                    "oi": oi,
                    "bid": b, "ask": a, "mid": mid,
                    "spread_pct": (round(spread_pct, 3) if spread_pct is not None else None),
                    "_rank_key": key
                })

        items.sort(key=lambda x: x["_rank_key"])
        for it in items:
            it.pop("_rank_key", None)
        kept = items[:top_n]
        buckets.append({
            "week": i,
            "expiry": exp,
            "scanned": scanned,
            "kept": len(kept),
            "items": kept,
        })

    # Telegram summary (optional)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        lines = [f"🔎 Chain Scan — {symbol}\n",
                 f"Filters: minVol≥{min_vol} or minOI≥{min_oi} · sort={sort_by} · top={top_n}\n"]
        for b in buckets:
            lines.append(f"📅 Week {b['week']} — {b['expiry']} (scanned {b['scanned']}, kept {b['kept']})")
            if b["kept"] == 0:
                lines.append("(no contracts meeting thresholds)\n")
                continue
            for it in b["items"]:
                sp = f" · spr={it['spread_pct']}%" if it.get("spread_pct") is not None else ""
                lines.append(f"{(it.get('type') or '?')[0].upper()} {it['ticker']}  vol={it['vol']} oi={it['oi']}{sp}")
            lines.append("")
        try:
            await send_telegram("\n".join(lines).strip())
        except Exception:
            pass

    return {"ok": True, "symbol": symbol, "weeks": buckets}
