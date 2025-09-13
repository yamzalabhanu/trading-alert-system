# feature_engine.py
import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import httpx

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY
from config import CDT_TZ

logger = logging.getLogger("trading_engine.feature_engine")

# =========================
# HTTP helper
# =========================
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            logger.info("[feature] aggs status=%s for %s", r.status_code, url)
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception as e:
        logger.warning("[feature] _http_json error: %r", e)
        return None

# =========================
# NumPy TA primitives
# =========================
def _ema(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return np.array([])
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

def _sma(x: np.ndarray, n: int) -> np.ndarray:
    if x.size < n:
        return np.array([np.nan] * x.size)
    csum = np.cumsum(x, dtype=float)
    csum[n:] = csum[n:] - csum[:-n]
    sma = csum[n - 1:] / n
    return np.concatenate([np.full(n - 1, np.nan), sma])

def _std(x: np.ndarray, n: int) -> np.ndarray:
    if x.size < n:
        return np.array([np.nan] * x.size)
    out = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        window = x[i - n + 1:i + 1]
        out[i] = np.std(window, ddof=0)
    return out

def _rsi_wilder(close: np.ndarray, n: int = 14) -> np.ndarray:
    if close.size < n + 1:
        return np.array([np.nan] * close.size)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty_like(close); avg_loss = np.empty_like(close)
    avg_gain[:n] = np.nan; avg_loss[:n] = np.nan
    avg_gain[n] = np.mean(gain[1:n+1]); avg_loss[n] = np.mean(loss[1:n+1])
    for i in range(n + 1, close.size):
        avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gain[i]) / n
        avg_loss[i] = (avg_loss[i - 1] * (n - 1) + loss[i]) / n
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:n] = np.nan
    return rsi

def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def _vwap_from_bars(prices: np.ndarray, volumes: np.ndarray) -> Optional[float]:
    if prices.size == 0 or volumes.size == 0 or np.sum(volumes) <= 0:
        return None
    return float(np.sum(prices * volumes) / np.sum(volumes))

# =========================
# Multi-tier aggregates
# =========================
async def _fetch_aggs_multi_tier(symbol: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (src, bars) where src in {"1m","5m","1d","none"}.
    """
    if not POLYGON_API_KEY:
        return "none", []
    client = get_http_client()
    if client is None:
        return "none", []

    now_utc = datetime.now(timezone.utc)

    # Try 1m (3 days)
    frm = now_utc - timedelta(days=3)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/min/{frm.isoformat()}/{now_utc.isoformat()}"
    js = await _http_json(client, url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY})
    bars_1m = (js or {}).get("results") or []
    if len(bars_1m) >= 120:
        return "1m", bars_1m

    # Try 5m (14 days)
    frm = now_utc - timedelta(days=14)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/min/{frm.isoformat()}/{now_utc.isoformat()}"
    js = await _http_json(client, url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY})
    bars_5m = (js or {}).get("results") or []
    if len(bars_5m) >= 120:
        return "5m", bars_5m

    # Try daily (2y)
    frm = now_utc - timedelta(days=365 * 2)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{frm.isoformat()}/{now_utc.isoformat()}"
    js = await _http_json(client, url, {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY})
    bars_1d = (js or {}).get("results") or []
    if len(bars_1d) >= 60:
        return "1d", bars_1d

    return "none", []

# =========================
# ORB-15 (intraday)
# =========================
def _orb15_from_intraday(bars: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not bars:
        return None, None
    today_local = datetime.now(CDT_TZ).date()
    session_open_local = datetime.now(CDT_TZ).replace(hour=8, minute=30, second=0, microsecond=0)
    session_open_local = session_open_local.replace(year=today_local.year, month=today_local.month, day=today_local.day)
    first_15_end = session_open_local + timedelta(minutes=15)
    hi = None; lo = None
    for b in bars:
        t_ms = int(b.get("t", 0))
        t = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc).astimezone(CDT_TZ)
        if t.date() != today_local:
            continue
        if session_open_local <= t < first_15_end:
            h = float(b.get("h", np.nan)); l = float(b.get("l", np.nan))
            if not math.isnan(h):
                hi = h if hi is None else max(hi, h)
            if not math.isnan(l):
                lo = l if lo is None else min(lo, l)
    return hi, lo

# =========================
# TA from bars
# =========================
def _build_ta_from_bars(symbol: str, side: str, ul_price: Optional[float], src: str, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ta_src": src}
    if not bars:
        return out

    closes = np.array([float(b.get("c", np.nan)) for b in bars], dtype=float)
    vols   = np.array([float(b.get("v", 0.0)) for b in bars], dtype=float)
    if closes.size == 0 or np.all(np.isnan(closes)):
        return out

    rsi_series = _rsi_wilder(closes, 14)
    rsi14 = float(rsi_series[-1]) if not np.isnan(rsi_series[-1]) else None

    ema20_series = _ema(closes, 20) if closes.size >= 1 else np.array([np.nan])
    ema50_series = _ema(closes, 50) if closes.size >= 1 else np.array([np.nan])
    ema200_series = _ema(closes, 200) if closes.size >= 1 else np.array([np.nan])
    ema20 = float(ema20_series[-1]) if not np.isnan(ema20_series[-1]) else None
    ema50 = float(ema50_series[-1]) if closes.size >= 50 and not np.isnan(ema50_series[-1]) else None
    ema200 = float(ema200_series[-1]) if closes.size >= 200 and not np.isnan(ema200_series[-1]) else None

    sma20_series = _sma(closes, 20)
    sma20 = float(sma20_series[-1]) if not np.isnan(sma20_series[-1]) else None

    macd, macd_signal, macd_hist = _macd(closes, 12, 26, 9)
    macd_v = float(macd[-1]) if not np.isnan(macd[-1]) else None
    macd_sig_v = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None
    macd_hist_v = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None

    std20 = _std(closes, 20)
    if not np.isnan(sma20_series[-1]) and not np.isnan(std20[-1]):
        bb_upper = float(sma20_series[-1] + 2.0 * std20[-1])
        bb_lower = float(sma20_series[-1] - 2.0 * std20[-1])
        bb_width = float((bb_upper - bb_lower) / sma20_series[-1]) if sma20_series[-1] != 0 else None
    else:
        bb_upper = bb_lower = bb_width = None

    vwap = None
    if src in ("1m", "5m"):
        today_local = datetime.now(CDT_TZ).date()
        px = []; vv = []
        for b in bars:
            t_ms = int(b.get("t", 0))
            t = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc).astimezone(CDT_TZ)
            if t.date() == today_local:
                px.append(float(b.get("c", np.nan))); vv.append(float(b.get("v", 0.0)))
        if px and vv:
            vwap = _vwap_from_bars(np.array(px, dtype=float), np.array(vv, dtype=float))
    if vwap is None:
        vwap = _vwap_from_bars(closes, vols)

    vwap_dist = None
    if vwap is not None and ul_price is not None and vwap != 0:
        vwap_dist = float((ul_price - vwap) / vwap * 100.0)

    if src in ("1m", "5m"):
        orb_hi, orb_lo = _orb15_from_intraday(bars)
    else:
        orb_hi, orb_lo = (None, None)

    out.update({
        "rsi14": rsi14,
        "sma20": sma20,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "macd": macd_v,
        "macd_signal": macd_sig_v,
        "macd_hist": macd_hist_v,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_width": bb_width,
        "vwap": vwap,
        "vwap_dist": vwap_dist,
        "orb15_high": orb_hi,
        "orb15_low": orb_lo,
    })
    return out

# =========================
# Reference / Corporate actions
# =========================
async def _enrich_reference_and_corp(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out
    client = get_http_client()
    if client is None:
        return out

    try:
        ref = await _http_json(client, f"https://api.polygon.io/v3/reference/tickers/{symbol}",
                               {"apiKey": POLYGON_API_KEY}, timeout=6.0)
        if isinstance(ref, dict):
            res = ref.get("results") or {}
            out["ref_primary_exchange"] = (res.get("primary_exchange") or {}).get("mic")
            out["ref_market_cap"] = res.get("market_cap")
            out["ref_name"] = res.get("name")
    except Exception:
        pass

    try:
        dvd = await _http_json(client, "https://api.polygon.io/v3/reference/dividends",
                               {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY}, timeout=6.0)
        if isinstance(dvd, dict):
            rs = (dvd.get("results") or [])
            if rs:
                r0 = rs[0]
                out["dividend_ex_date"] = r0.get("ex_dividend_date")
                out["dividend_amount"] = r0.get("cash_amount")
    except Exception:
        pass

    try:
        spl = await _http_json(client, "https://api.polygon.io/v3/reference/splits",
                               {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY}, timeout=6.0)
        if isinstance(spl, dict):
            rs = (spl.get("results") or [])
            if rs:
                r0 = rs[0]
                out["last_split_date"] = r0.get("execution_date")
                out["last_split_ratio"] = (r0.get("split_from"), r0.get("split_to"))
    except Exception:
        pass

    return out

# =========================
# Synthetic NBBO
# =========================
def _convert_poly_ts_to_sec(ts_val: float) -> Optional[float]:
    try:
        ns = int(ts_val)
        if ns >= 10**14: return ns / 1e9
        if ns >= 10**11: return ns / 1e6
        if ns >= 10**8:  return ns / 1e3
        return float(ns)
    except Exception:
        return None

def _extract_last_price_and_ts(snapshot: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    Robustly walk the snapshot dict to find a last trade price and timestamp.
    Looks for keys like 'price', 'p', 'last', 'last_price' and ts keys like 't', 'timestamp', 'sip_timestamp'.
    """
    last_price: Optional[float] = None
    last_ts_sec: Optional[float] = None

    def walk(obj: Any):
        nonlocal last_price, last_ts_sec
        if isinstance(obj, dict):
            price_keys = ("price", "p", "last", "last_price", "lastPrice")
            ts_keys = ("t", "timestamp", "sip_timestamp", "ts", "time")
            found_price = None
            for k in price_keys:
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    found_price = float(v)
                    break
            if found_price is not None and last_price is None:
                last_price = found_price
                # try to grab a sibling timestamp
                for kt in ts_keys:
                    tv = obj.get(kt)
                    if isinstance(tv, (int, float)):
                        last_ts_sec = _convert_poly_ts_to_sec(tv)
                        break
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    try:
        walk(snapshot or {})
    except Exception:
        pass
    return last_price, last_ts_sec

def synthesize_nbbo_from_last(last_price: Optional[float], spread_pct: Optional[float] = None,
                              last_ts_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Build a synthetic NBBO envelope around a last price.
    - spread_pct: total % width (ask-bid)/mid*100 (defaults from env: SYNTH_SPREAD_PCT or FALLBACK_SYNTH_SPREAD_PCT)
    """
    out: Dict[str, Any] = {}
    if last_price is None or not isinstance(last_price, (int, float)) or last_price <= 0:
        return out

    try:
        sp = float(spread_pct) if spread_pct is not None else float(
            os.getenv("SYNTH_SPREAD_PCT", os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "12.0"))
        )
    except Exception:
        sp = 12.0

    half = sp / 200.0
    bid = round(float(last_price) * (1.0 - half), 4)
    ask = round(float(last_price) * (1.0 + half), 4)
    mid = round((bid + ask) / 2.0, 4)

    out.update({
        "synthetic_bid": bid,
        "synthetic_ask": ask,
        "synthetic_mid": mid,
        "synthetic_spread_pct": float(sp),
    })

    if last_ts_sec is not None:
        try:
            now_sec = datetime.now(timezone.utc).timestamp()
            out["synthetic_quote_age_sec"] = max(0.0, now_sec - float(last_ts_sec))
        except Exception:
            pass

    return out

def attach_synthetic_nbbo_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract last price + ts from Polygon snapshot and return synthetic_* NBBO fields.
    """
    if not snapshot:
        return {}
    last_price, last_ts = _extract_last_price_and_ts(snapshot)
    return synthesize_nbbo_from_last(last_price, None, last_ts)

# =========================
# Public API
# =========================
async def build_features(
    client: httpx.AsyncClient,
    alert: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a rich feature bundle:
      • TA (RSI14, SMA20, EMA20/50/200, MACD, Bollinger, VWAP, ORB-15) with multi-tier fallback (1m→5m→1d)
      • Reference / Corporate actions (best-effort)
      • Synthetic NBBO (returned as synthetic_* keys; does NOT overwrite real NBBO)
    """
    symbol = str(alert.get("symbol") or "").upper()
    side = str(alert.get("side") or "").upper()
    try:
        ul_price = float(alert.get("underlying_price_from_alert"))
    except Exception:
        ul_price = None

    out: Dict[str, Any] = {}

    # --- TA via multi-tier bars ---
    try:
        src, bars = await _fetch_aggs_multi_tier(symbol)
        ta = _build_ta_from_bars(symbol, side, ul_price, src, bars)
        out.update(ta)
    except Exception as e:
        logger.warning("[feature] TA build failed: %r", e)
        out.setdefault("ta_src", "none")

    # Placeholders to keep downstream compatible
    out.setdefault("regime_flag", "trending")
    out.setdefault("mtf_align", False)

    # --- Reference / Corp actions ---
    try:
        out.update(await _enrich_reference_and_corp(symbol))
    except Exception as e:
        logger.info("[feature] ref/corp enrich failed: %r", e)

    # --- Synthetic NBBO from snapshot (safe: synthetic_* keys only) ---
    try:
        if snapshot:
            out.update(attach_synthetic_nbbo_from_snapshot(snapshot))
    except Exception as e:
        logger.info("[feature] synthetic NBBO attach failed: %r", e)

    return out

__all__ = [
    "build_features",
    # optional helpers if you want to use them in the processor:
    "synthesize_nbbo_from_last",
    "attach_synthetic_nbbo_from_snapshot",
]
