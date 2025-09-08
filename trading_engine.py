# trading_engine.py
import os
import re
import asyncio
import socket
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
    list_contracts_for_expiry,
    get_option_snapshot,     # may be (client, symbol, contract) or (symbol, contract)
    build_option_contract,   # OCC builder: O:<SYM><YYMMDD><C/P><STRIKE*1000>
)
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG

# =========================
# Global state / resources
# =========================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # optional, for checks

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

# Prefer contract picked by chain scan when available
PREFER_CHAIN_SCAN = _env_truthy(os.getenv("PREFER_CHAIN_SCAN", "1"))

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

# =========================
# HTTP helpers (shared)
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

# =========================
# Polygon wrappers
# =========================
async def polygon_list_contracts_for_expiry_export(
    client: httpx.AsyncClient,
    symbol: str,
    expiry: str,
    side: str,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    return await list_contracts_for_expiry(client, symbol=symbol, expiry=expiry, side=side, limit=limit)

async def polygon_get_option_snapshot_export(
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
                    "mid": float(mid) if mid is not None else None,   # mark
                    "quote_age_sec": age,
                    "option_spread_pct": round(spread_pct, 3) if spread_pct is not None else None
                }
                if not best or \
                   (cand["option_spread_pct"] or 1e9) < (best.get("option_spread_pct") or 1e9) or \
                   (cand["quote_age_sec"] or 1e9) < (best.get("quote_age_sec") or 1e9):
                    best = cand
        await asyncio.sleep(delay)
    return best or None

async def _poly_option_backfill(
    client: httpx.AsyncClient,
    symbol: str,
    option_ticker: str,
    today_utc: date,
) -> Dict[str, Any]:
    """
    Collects:
      - bid/ask/mid(mark), last, spread%, quote_age_sec
      - OI, Vol (day), IV, greeks
      - prev_close (prior day), quote_change_pct vs prev_close
      - fallback intraday volume via 1m aggs
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    def _apply_from_results(res: Dict[str, Any], out_dict: Dict[str, Any]) -> None:
        if not isinstance(res, dict):
            return
        oi = res.get("open_interest")
        if oi is not None:
            out_dict["oi"] = oi
        day_block = res.get("day") or {}
        vol = day_block.get("volume", day_block.get("v"))
        if vol is not None:
            out_dict["vol"] = vol

        # quote + last + derived mark/spread/age
        lq = res.get("last_quote") or {}
        bid_px = lq.get("bid_price")
        ask_px = lq.get("ask_price")
        if bid_px is not None:
            out_dict["bid"] = float(bid_px)
        if ask_px is not None:
            out_dict["ask"] = float(ask_px)

        lt = res.get("last_trade") or {}
        lt_px = lt.get("price")
        if isinstance(lt_px, (int, float)):
            out_dict["last"] = float(lt_px)

        if out_dict.get("mid") is None and out_dict.get("bid") is not None and out_dict.get("ask") is not None:
            out_dict["mid"] = round((out_dict["bid"] + out_dict["ask"]) / 2.0, 4)

        ts = (
            lq.get("sip_timestamp")
            or lq.get("participant_timestamp")
            or lq.get("trf_timestamp")
            or lq.get("t")
            or lt.get("sip_timestamp")
            or lt.get("participant_timestamp")
            or lt.get("trf_timestamp")
            or lt.get("t")
        )
        age = _quote_age_from_ts(ts)
        if age is not None:
            out_dict["quote_age_sec"] = age

        # greeks & IV
        greeks = res.get("greeks") or {}
        for k_src in ("delta", "gamma", "theta", "vega"):
            v = greeks.get(k_src)
            if v is not None:
                out_dict[k_src] = v
        iv = res.get("implied_volatility") or greeks.get("iv")
        if iv is not None:
            out_dict["iv"] = iv

    # 1) multi-snapshot list -> exact match if present
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
                if chosen:
                    _apply_from_results(chosen, out)
    except Exception:
        pass

    # 2) single-contract snapshot
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

    # 3) previous-day open/close (for prev_close and later change%)
    prev_close = None
    try:
        yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
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
            pc = oc.get("close")
            if isinstance(pc, (int, float)):
                prev_close = float(pc)
                out["prev_close"] = prev_close
    except Exception:
        pass

    # 4) sample best quote (robust NBBO + mark + spread + age)
    try:
        enc_opt = _encode_ticker_path(option_ticker)
        sampled = await _sample_best_quote(client, enc_opt, tries=5, delay=0.6)
        if sampled:
            for k, v in sampled.items():
                if v is not None:
                    out[k] = v
    except Exception:
        pass

    # 5) minute aggregates as fallback volume
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

    # 6) derive quote_change_pct if possible (mark vs prev_close)
    try:
        mark = out.get("mid")
        if mark is None:
            # fall back to last if present
            mark = out.get("last")
        if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
            out["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
    except Exception:
        pass

    return out

# =========================
# Contract pickers / scanners
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

async def _choose_best_contract(
    client: httpx.AsyncClient,
    symbol: str,
    expiry_iso: str,
    side: str,
    ul_px: float,
    desired_strike: float,
) -> Tuple[str, Dict[str, Any]]:
    if not POLYGON_API_KEY:
        base = _build_plus_minus_contracts(symbol, ul_px, expiry_iso)
        t = base["contract_call"] if side == "CALL" else base["contract_put"]
        return t, {"reason": "offline/no key"}

    chain = await polygon_list_contracts_for_expiry_export(client, symbol=symbol, expiry=expiry_iso, side=side, limit=1000)
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

    def _normalize_from_tk(tk: str, default: Optional[float]) -> Optional[float]:
        s_norm = default
        if s_norm is None:
            m2 = re.search(r"[CP](\d{8,9})$", tk)
            if m2:
                try:
                    s_norm = int(m2.group(1)) / 1000.0
                except Exception:
                    s_norm = None
        return s_norm

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
        s_norm = _normalize_from_tk(tk, s_norm)
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
    debug = {
        "selected_by": "delta_selector",
        "candidates": [
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
    }
    return top[1], debug

# ======================================================
# Chain Scanner helpers (weeks 1/2/3) – calls + puts
# ======================================================
async def _poly_paginated_snapshot_list(
    client: httpx.AsyncClient,
    underlying: str,
    expiry_iso: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
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
            if "apiKey=" not in nxt and POLYGON_API_KEY:
                sep = "&" if "?" in nxt else "?"
                nxt = f"{nxt}{sep}apiKey={POLYGON_API_KEY}"
            page = await _http_json_url(client, nxt, timeout=10.0)
            if not page or not isinstance(page.get("results"), list):
                break
            all_rows.extend(page["results"])
            nxt = page.get("next_url")

    return all_rows

async def _poly_reference_contracts(
    client: httpx.AsyncClient,
    underlying: str,
    expiry_iso: str,
    max_contracts: int = 1200,
) -> List[str]:
    if not POLYGON_API_KEY:
        return []
    tickers: List[str] = []
    base = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": underlying,
        "expiration_date": expiry_iso,
        "limit": 1000,
        "apiKey": POLYGON_API_KEY,
    }
    first = await _http_json(client, base, params, timeout=10.0)
    if first and isinstance(first.get("results"), list):
        for it in first["results"]:
            t = it.get("ticker")
            if t:
                tickers.append(t)
        nxt = first.get("next_url")
    else:
        nxt = None

    while nxt and len(tickers) < max_contracts:
        if "apiKey=" not in nxt and POLYGON_API_KEY:
            sep = "&" if "?" in nxt else "?"
            nxt = f"{nxt}{sep}apiKey={POLYGON_API_KEY}"
        page = await _http_json_url(client, nxt, timeout=10.0)
        if not page or not isinstance(page.get("results"), list):
            break
        for it in page["results"]:
            t = it.get("ticker")
            if t:
                tickers.append(t)
        nxt = page.get("next_url")

    return tickers[:max_contracts]

async def _poly_snapshot_for_ticker(client: httpx.AsyncClient, underlying: str, ticker: str) -> Optional[Dict[str, Any]]:
    enc = _encode_ticker_path(ticker)
    snap = await _http_json(client, f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
                            {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    return snap.get("results") if snap else None

def _score_chain_item_for_alert(alert: Dict[str, Any], it: Dict[str, Any]) -> float:
    strike_alert = float(alert.get("strike") or 0.0)
    strike_item  = float(it.get("strike") or 0.0)
    expiry_alert = str(alert.get("expiry") or "")

    strike_dist = abs(strike_item - strike_alert)
    expiry_penalty = 0.0 if it.get("expiry") == expiry_alert else 10.0
    oi = int(it.get("oi") or 0); vol = int(it.get("vol") or 0)
    liq_score = -(oi * 2 + vol) / 1000.0
    spread_pct = it.get("spread_pct")
    nbbo_pen = 0.5 if (it.get("bid") is None or it.get("ask") is None) else 0.0
    spr_pen  = 0.0
    if isinstance(spread_pct, (int, float)):
        if spread_pct > 15: spr_pen = 2.0
        elif spread_pct > 10: spr_pen = 1.0
        elif spread_pct > 5:  spr_pen = 0.3
    else:
        spr_pen = 0.7
    return strike_dist + expiry_penalty + nbbo_pen + spr_pen + liq_score

async def scan_for_best_contract_for_alert(
    client: httpx.AsyncClient,
    symbol: str,
    alert: Dict[str, Any],
    min_vol: int = 500,
    min_oi: int  = 500,
    top_n_each_week: int = 12,
) -> Optional[Dict[str, Any]]:
    today_utc = datetime.now(timezone.utc).date()
    wk1 = _next_friday(today_utc)
    if wk1 <= today_utc:
        wk1 = wk1 + timedelta(days=7)
    wk2 = wk1 + timedelta(days=7)
    wk3 = wk2 + timedelta(days=7)
    default_weeks = [wk1.isoformat(), wk2.isoformat(), wk3.isoformat()]

    expiry_alert = str(alert.get("expiry") or "")
    weeks = []
    if expiry_alert:
        seen = set()
        for e in [expiry_alert] + default_weeks:
            if e and e not in seen:
                weeks.append(e); seen.add(e)
    else:
        weeks = default_weeks

    candidates: List[Dict[str, Any]] = []
    for exp in weeks:
        rows = await _poly_paginated_snapshot_list(client, symbol, exp, limit=1000)
        if not rows:
            tickers = await _poly_reference_contracts(client, symbol, exp, max_contracts=1200)
            sem = asyncio.Semaphore(12)
            async def fetch_one(tk: str):
                async with sem:
                    try:
                        return await _poly_snapshot_for_ticker(client, symbol, tk)
                    except Exception:
                        return None
            snaps = await asyncio.gather(*[fetch_one(t) for t in tickers])
            rows = [r for r in snaps if isinstance(r, dict)]

        items = []
        for r in rows:
            det = r.get("details") or {}
            tk  = det.get("ticker") or r.get("ticker")
            if not tk:
                continue
            day = r.get("day") or {}
            vol = int(day.get("volume") or day.get("v") or 0)
            oi  = int(r.get("open_interest") or 0)
            if vol < min_vol and oi < min_oi:
                continue

            lq = r.get("last_quote") or {}
            b = lq.get("bid_price"); a = lq.get("ask_price")
            mid = None; spread_pct = None
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0:
                mid = (a + b) / 2.0
                if mid and mid > 0:
                    spread_pct = (a - b) / mid * 100.0

            items.append({
                "ticker": tk,
                "expiry": exp,
                "strike": _normalize_poly_strike(det.get("strike")),
                "type": det.get("contract_type"),
                "vol": vol,
                "oi": oi,
                "bid": b, "ask": a, "mid": mid,
                "spread_pct": round(spread_pct, 3) if isinstance(spread_pct, (int, float)) else None,
            })

        items.sort(key=lambda x: (-(x["oi"] or 0), -(x["vol"] or 0)))
        candidates.extend(items[:top_n_each_week])

    if not candidates:
        return None

    scored = [( _score_chain_item_for_alert(alert, it), it ) for it in candidates]
    scored.sort(key=lambda z: z[0])
    return scored[0][1] if scored else None

# =========================
# Preflight
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
    for _ in range(WORKER_COUNT):
        asyncio.create_task(_worker())

async def shutdown():
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
# Public job enqueue
# =========================
def enqueue_webhook_job(alert_text: str, flags: Dict[str, Any]) -> bool:
    job = {"alert_text": alert_text, "flags": flags}
    try:
        WORK_Q.put_nowait(job)
        return True
    except asyncio.QueueFull:
        return False

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
    force_buy = bool(job["flags"].get("force_buy", False))
    qty = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    target_expiry_date = two_weeks_friday(today_utc)
    swf = same_week_friday(today_utc)
    if is_same_week(target_expiry_date, swf):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    desired_strike = pm["strike_call"] if alert["side"] == "CALL" else pm["strike_put"]

    # --- Chain scan selection (triggers alerts using chain scan) ---
    selection_debug: Dict[str, Any] = {}
    option_ticker = None
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            HTTP,
            alert["symbol"],
            {"side": alert["side"], "symbol": alert["symbol"], "strike": alert.get("strike"), "expiry": alert.get("expiry")},
            min_vol=int(os.getenv("SCAN_MIN_VOL", "500")),
            min_oi=int(os.getenv("SCAN_MIN_OI", "500")),
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
        )
    except Exception:
        best_from_scan = None

    if PREFER_CHAIN_SCAN and best_from_scan:
        option_ticker = best_from_scan["ticker"]
        if isinstance(best_from_scan.get("strike"), (int, float)):
            desired_strike = float(best_from_scan["strike"])
        selection_debug = {"selected_by": "chain_scan", "selected_ticker": option_ticker, "best_item": best_from_scan}
    else:
        try:
            option_ticker, sel_dbg = await _choose_best_contract(
                HTTP, alert["symbol"], target_expiry, alert["side"], ul_px, desired_strike
            )
            selection_debug = {"selected_by": "delta_selector", **(sel_dbg or {})}
        except Exception:
            option_ticker = pm["contract_call"] if alert["side"] == "CALL" else pm["contract_put"]
            selection_debug = {"selected_by": "fallback_pm", "reason": "selector_error"}

    # --- Build feature bundle, backfill missing quote/iv/greeks and compute change% ---
    f: Dict[str, Any] = {}
    try:
        if not POLYGON_API_KEY:
            f = {
                "bid": None, "ask": None, "mid": None, "last": None,
                "option_spread_pct": None, "quote_age_sec": None,
                "oi": None, "vol": None,
                "delta": None, "gamma": None, "theta": None, "vega": None,
                "iv": None, "iv_rank": None, "rv20": None, "prev_close": None, "quote_change_pct": None,
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
                snap = await polygon_get_option_snapshot_export(HTTP, underlying=alert["symbol"], option_ticker=option_ticker)
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

            # derive mark & spread if still missing
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

            # ensure DTE present
            try:
                if f.get("dte") is None:
                    f["dte"] = (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days
            except Exception:
                pass

            # compute change% vs prev close if not set already
            try:
                if f.get("quote_change_pct") is None:
                    prev_close = f.get("prev_close")
                    mark = f.get("mid") if f.get("mid") is not None else f.get("last")
                    if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
                        f["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
            except Exception:
                pass

    except Exception as e:
        print(f"[worker] Polygon/features error: {e}")
        f = f or {"dte": (datetime.fromisoformat(target_expiry).date() - datetime.now(timezone.utc).date()).days}

    pf_ok, pf_checks = preflight_ok(f)

    # --- LLM decision ---
    try:
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

    if job["flags"].get("force_buy", False):
        decision_final = "buy"

    # --- Telegram alert (explicitly notes chain-scan) ---
    tg_result = None
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
        if selection_debug and selection_debug.get("selected_by") == "chain_scan":
            tg_text += "\n🔎 Note: Contract selected via chain-scan (liquidity + strike/expiry fit)."
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            tg_result = await send_telegram(tg_text)
    except Exception as e:
        print(f"[worker] Telegram error: {e}")

    # --- IBKR order (optional) ---
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

    # --- Decision log with new fields ---
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
            "reco_expiry": target_expiry,
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
