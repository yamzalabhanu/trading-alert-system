# feature_engine.py
import os
import math
import random
import asyncio 
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY, CDT_TZ

# Extra providers (IEX/Alpaca) as fallbacks: expect (symbol, start_dt, end_dt)
from market_providers import fetch_1m_bars_any, fetch_5m_bars_any, fetch_1d_bars_any

logger = logging.getLogger("trading_engine.feature_engine")

# ---------- Small HTTP helper ----------
async def _http_json(url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    client = get_http_client()
    if client is None:
        return None
    try:
        r = await client.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception as e:
        logger.warning("[feature] _http_json error: %r", e)
        return None

# ---------- Time helpers ----------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _today_rth_utc_window() -> Tuple[datetime, datetime]:
    """Return today's RTH open/close in UTC using the configured CDT_TZ (09:30–16:00 ET == 08:30–15:00 CT)."""
    now_cdt = datetime.now(CDT_TZ)
    open_cdt = now_cdt.replace(hour=8, minute=30, second=0, microsecond=0)
    close_cdt = now_cdt.replace(hour=15, minute=0, second=0, microsecond=0)
    return open_cdt.astimezone(timezone.utc), close_cdt.astimezone(timezone.utc)

# ---------- Basic TA ----------
def _sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def _ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val

def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _stddev(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    window = values[-period:]
    mean = sum(window) / period
    var = sum((x - mean) ** 2 for x in window) / period
    return math.sqrt(var)

def _macd(values: List[float], f=12, s=26, sig=9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(values) < (s + sig):
        return (None, None, None)

    def ema_series(vals: List[float], period: int) -> List[Optional[float]]:
        if len(vals) < period:
            return [None] * len(vals)
        k = 2 / (period + 1)
        ema_v = sum(vals[:period]) / period
        out: List[Optional[float]] = [None]*(period-1) + [ema_v]
        for v in vals[period:]:
            ema_v = v * k + ema_v * (1 - k)
            out.append(ema_v)
        return out

    ef = ema_series(values, f)
    es = ema_series(values, s)
    macd_line_series = [(a - b) if (a is not None and b is not None) else None for a, b in zip(ef, es)]
    macd_vals = [m for m in macd_line_series if m is not None]
    if len(macd_vals) < sig:
        return (None, None, None)

    def ema_over_series(series: List[Optional[float]], period: int) -> List[Optional[float]]:
        vals = [x for x in series if x is not None]
        if len(vals) < period:
            return [None] * len(series)
        k2 = 2 / (period + 1)
        ema_v2 = sum(vals[:period]) / period
        res: List[Optional[float]] = []
        lead_nones = series.index(vals[0])
        res.extend([None] * lead_nones)
        res.append(ema_v2)
        for i in range(lead_nones + 1, len(series)):
            cur = series[i]
            if cur is None:
                res.append(res[-1])
            else:
                ema_v2 = cur * k2 + res[-1] * (1 - k2)
                res.append(ema_v2)
        return res[:len(series)]

    sig_series = ema_over_series(macd_line_series, sig)
    macd_line = macd_line_series[-1]
    macd_signal = sig_series[-1]
    if macd_line is None or macd_signal is None:
        return (None, None, None)
    macd_hist = macd_line - macd_signal
    return (macd_line, macd_signal, macd_hist)

# ---------- Bars fetch (Polygon + fallbacks) ----------
async def _fetch_aggs_polygon(symbol: str, multiplier: int, timespan: str, start: datetime, end: datetime, limit: int = 50000) -> List[Dict[str, Any]]:
    """Polygon /v2/aggs with basic retry/backoff for 429."""
    if not POLYGON_API_KEY:
        return []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{_to_ms(start)}/{_to_ms(end)}"
    params = {"adjusted": "true", "sort": "asc", "limit": str(limit), "apiKey": POLYGON_API_KEY}
    client = get_http_client()
    if client is None:
        return []
    attempts = int(os.getenv("POLY_AGGS_RETRIES", "3"))
    for i in range(attempts):
        try:
            r = await client.get(url, params=params, timeout=10.0)
            if r.status_code == 429:
                # backoff + jitter
                sleep_s = min(3.0, 0.6 * (i + 1)) + random.uniform(0, 0.25)
                logger.warning("[feature] Polygon 429 on aggs (%s %s) attempt %d/%d; backing off %.2fs",
                               symbol, timespan, i+1, attempts, sleep_s)
                await asyncio.sleep(sleep_s)  # type: ignore  # asyncio is available in runtime
                # lower limit a bit on retry
                params["limit"] = str(max(1000, int(params["limit"]) // 2))
                continue
            r.raise_for_status()
            js = r.json()
            return js.get("results") or []
        except Exception as e:
            if i == attempts - 1:
                logger.warning("[feature] aggs fetch failed (final): %r", e)
            else:
                await asyncio.sleep(0.3 + random.uniform(0, 0.2))  # type: ignore
    return []

def _pick_close_series(bars: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for b in bars:
        c = b.get("c")
        if isinstance(c, (int, float)):
            out.append(float(c))
    return out

def _intraday_vwap(bars: List[Dict[str, Any]]) -> Optional[float]:
    tot_pv = 0.0
    tot_v = 0.0
    for b in bars:
        v = float(b.get("v") or 0.0)
        if v <= 0:
            continue
        price = b.get("vw")
        if not isinstance(price, (int, float)):
            c = b.get("c")
            if not isinstance(c, (int, float)):
                continue
            price = float(c)
        tot_pv += float(price) * v
        tot_v += v
    return (tot_pv / tot_v) if tot_v > 0 else None

def _orb_15(bars_1m_today: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Opening 15-minute range using CDT RTH mapped to UTC."""
    if not bars_1m_today:
        return (None, None)
    rth_open_utc, _ = _today_rth_utc_window()
    start_ms = _to_ms(rth_open_utc)
    end_ms   = start_ms + 15 * 60 * 1000
    hi = None; lo = None
    for b in bars_1m_today:
        t = b.get("t")
        if not isinstance(t, (int, float)):
            continue
        if start_ms <= t < end_ms:
            h = b.get("h"); l = b.get("l")
            if isinstance(h, (int, float)): hi = h if hi is None else max(hi, h)
            if isinstance(l, (int, float)): lo = l if lo is None else min(lo, l)
    return (hi, lo)

# Fallbacks expect (symbol, start_dt, end_dt)
async def _fallback_1m(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    try:
        return await fetch_1m_bars_any(symbol, start, end) or []
    except TypeError:
        # Some envs still expose old signature; try without range
        return await fetch_1m_bars_any(symbol) or []

async def _fallback_5m(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    try:
        return await fetch_5m_bars_any(symbol, start, end) or []
    except TypeError:
        return await fetch_5m_bars_any(symbol) or []

async def _fallback_1d(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    try:
        return await fetch_1d_bars_any(symbol, start, end) or []
    except TypeError:
        return await fetch_1d_bars_any(symbol) or []

# ---------- TA Orchestrator ----------
async def _compute_ta_bundle(symbol: str) -> Dict[str, Any]:
    now = _now_utc()
    out: Dict[str, Any] = {"ta_src": None}

    # 1) 1-minute (Polygon → fallbacks). Reduce lookback to 1 day to lower rate use.
    try:
        start_1m_hist = now - timedelta(days=1)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start_1m_hist, now, limit=20000)
        if not bars_1m:
            bars_1m = await _fallback_1m(symbol, start_1m_hist, now)
    except Exception as e:
        logger.warning("[feature] 1m fetch failed: %r", e)
        bars_1m = await _fallback_1m(symbol, now - timedelta(days=1), now)

    # Today subset for VWAP/ORB
    bars_1m_today: List[Dict[str, Any]] = []
    if bars_1m:
        today_date = now.date()
        for b in bars_1m:
            ts = b.get("t")
            if not isinstance(ts, (int, float)):
                continue
            dt = datetime.fromtimestamp(ts/1000.0, tz=timezone.utc)
            if dt.date() == today_date:
                bars_1m_today.append(b)

    vwap = _intraday_vwap(bars_1m_today) if bars_1m_today else None
    orb_hi, orb_lo = _orb_15(bars_1m_today) if bars_1m_today else (None, None)

    closes_1m = _pick_close_series(bars_1m)
    rsi14_1m = _rsi(closes_1m, 14) if len(closes_1m) >= 50 else None
    sma20_1m = _sma(closes_1m, 20)
    ema20_1m = _ema(closes_1m, 20)
    ema50_1m = _ema(closes_1m, 50)
    ema200_1m = _ema(closes_1m, 200)
    macd_line_1m, macd_signal_1m, macd_hist_1m = _macd(closes_1m)

    # 2) 5-minute (14 days)
    try:
        start_5m = now - timedelta(days=14)
        bars_5m = await _fetch_aggs_polygon(symbol, 5, "minute", start_5m, now, limit=50000)
        if not bars_5m:
            bars_5m = await _fallback_5m(symbol, start_5m, now)
    except Exception as e:
        logger.warning("[feature] 5m fetch failed: %r", e)
        bars_5m = await _fallback_5m(symbol, now - timedelta(days=14), now)

    closes_5m = _pick_close_series(bars_5m)
    rsi14_5m = _rsi(closes_5m, 14) if len(closes_5m) >= 50 else None
    sma20_5m = _sma(closes_5m, 20)
    ema20_5m = _ema(closes_5m, 20)
    ema50_5m = _ema(closes_5m, 50)
    ema200_5m = _ema(closes_5m, 200)
    macd_line_5m, macd_signal_5m, macd_hist_5m = _macd(closes_5m)

    # 3) Daily (750 days)
    try:
        start_1d = now - timedelta(days=750)
        bars_1d = await _fetch_aggs_polygon(symbol, 1, "day", start_1d, now, limit=50000)
        if not bars_1d:
            bars_1d = await _fallback_1d(symbol, start_1d, now)
    except Exception as e:
        logger.warning("[feature] 1d fetch failed: %r", e)
        bars_1d = await _fallback_1d(symbol, now - timedelta(days=750), now)

    closes_1d = _pick_close_series(bars_1d)
    rsi14_1d = _rsi(closes_1d, 14) if len(closes_1d) >= 30 else None
    sma20_1d = _sma(closes_1d, 20)
    ema20_1d = _ema(closes_1d, 20)
    ema50_1d = _ema(closes_1d, 50)
    ema200_1d = _ema(closes_1d, 200)
    macd_line_1d, macd_signal_1d, macd_hist_1d = _macd(closes_1d)

    bb_mid_1d = sma20_1d
    bb_std_1d = _stddev(closes_1d, 20) if len(closes_1d) >= 20 else None
    bb_upper_1d = (bb_mid_1d + 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None
    bb_lower_1d = (bb_mid_1d - 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None

    def _pick(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    out.update({
        "rsi14":    _pick(rsi14_1m, rsi14_5m, rsi14_1d),
        "sma20":    _pick(sma20_1m, sma20_5m, sma20_1d),
        "ema20":    _pick(ema20_1m, ema20_5m, ema20_1d),
        "ema50":    _pick(ema50_1m, ema50_5m, ema50_1d),
        "ema200":   _pick(ema200_1m, ema200_5m, ema200_1d),
        "macd_line":   _pick(macd_line_1m, macd_line_5m, macd_line_1d),
        "macd_signal": _pick(macd_signal_1m, macd_signal_5m, macd_signal_1d),
        "macd_hist":   _pick(macd_hist_1m, macd_hist_5m, macd_hist_1d),
        "bb_upper": bb_upper_1d,
        "bb_lower": bb_lower_1d,
        "bb_mid":   bb_mid_1d,
        "vwap": vwap,
        "orb15_high": orb_hi,
        "orb15_low":  orb_lo,
        "ta_src": "1m" if rsi14_1m is not None else ("5m" if rsi14_5m is not None else ("1d" if rsi14_1d is not None else None)),
    })
    return out

# ---------- Synthetic NBBO seed ----------
def _synthetic_nbbo_from_last(last: Optional[float], base_spread_pct: float = None) -> Dict[str, Any]:
    if not isinstance(last, (int, float)) or last <= 0:
        return {}
    spread_pct = float(base_spread_pct if base_spread_pct is not None else os.getenv("SYNTH_SPREAD_PCT", "1.0"))
    half = spread_pct / 200.0
    bid = last * (1 - half / 100.0)
    ask = last * (1 + half / 100.0)
    mid = (bid + ask) / 2.0
    return {
        "synthetic_bid": round(bid, 4),
        "synthetic_ask": round(ask, 4),
        "synthetic_mid": round(mid, 4),
        "synthetic_spread_pct": round((ask - bid) / mid * 100.0, 3) if mid > 0 else spread_pct,
        "synthetic_quote_age_sec": None,
    }

# ---------- Context (prev day, 5d avg, PM high/low, short vol/interest) ----------
async def _prev_day_ohlc(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out
    try:
        yday = (_now_utc().date() - timedelta(days=1)).isoformat()
        url = f"https://api.polygon.io/v1/open-close/{symbol}/{yday}"
        js = await _http_json(url, {"adjusted": "true", "apiKey": POLYGON_API_KEY})
        if js:
            out["prev_open"] = js.get("open")
            out["prev_high"] = js.get("high")
            out["prev_low"]  = js.get("low")
            out["prev_close"]= js.get("close")
    except Exception as e:
        logger.warning("[feature] prev day ohlc fail: %r", e)
    return out

async def _prev5_avg_hilo(symbol: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        now = _now_utc()
        bars = await _fetch_aggs_polygon(symbol, 1, "day", now - timedelta(days=10), now) or await _fallback_1d(symbol, now - timedelta(days=10), now)
        highs, lows = [], []
        for b in bars[-5:]:
            if isinstance(b.get("h"), (int,float)): highs.append(float(b["h"]))
            if isinstance(b.get("l"), (int,float)): lows.append(float(b["l"]))
        if highs and lows:
            out["prev5_avg_high"] = sum(highs)/len(highs)
            out["prev5_avg_low"] = sum(lows)/len(lows)
    except Exception as e:
        logger.warning("[feature] prev5 avg hilo fail: %r", e)
    return out

async def _premarket_hilo(symbol: str) -> Dict[str, Any]:
    """Best-effort: use today's minute bars before CDT RTH open as 'premarket'."""
    out: Dict[str, Any] = {}
    try:
        now = _now_utc()
        start = now - timedelta(days=1)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start, now, limit=20000) or await _fallback_1m(symbol, start, now)
        pm_hi = None; pm_lo = None
        rth_open_utc, _ = _today_rth_utc_window()
        cutoff_ms = _to_ms(rth_open_utc)
        for b in bars_1m:
            t = b.get("t")
            if not isinstance(t, (int, float)):
                continue
            if t < cutoff_ms:
                h = b.get("h"); l = b.get("l")
                if isinstance(h,(int,float)): pm_hi = h if pm_hi is None else max(pm_hi,h)
                if isinstance(l,(int,float)): pm_lo = l if pm_lo is None else min(pm_lo,l)
        out["premarket_high"] = pm_hi
        out["premarket_low"] = pm_lo
    except Exception as e:
        logger.warning("[feature] premarket H/L fail: %r", e)
    return out

async def _short_volume_ratio(symbol: str) -> Dict[str, Any]:
    """Legacy v1 short-volume/interest (optional). Set POLY_SHORT_V1=0 to disable."""
    if os.getenv("POLY_SHORT_V1", "1") != "1":
        return {}
    out: Dict[str, Any] = {}
    base_qs = {"ticker": symbol, "limit": "1", "sort": "date.desc", "apiKey": os.getenv("POLYGON_API_KEY","")}
    # Short volume
    try:
        url_sv = "https://api.polygon.io/stocks/v1/short-volume"
        js_sv = await _http_json(url_sv, base_qs)
        total = None; short = None
        try:
            results = (js_sv or {}).get("results") or []
            if results:
                rec = results[0]
                short = rec.get("short_volume")
                total = rec.get("total_volume")
        except Exception:
            pass
        if isinstance(short, (int,float)) and isinstance(total, (int,float)) and total > 0:
            out["short_volume"] = float(short)
            out["short_volume_total"] = float(total)
            out["short_volume_ratio"] = round(float(short)/float(total), 4)
    except Exception as e:
        logger.warning("[feature] short volume fail: %r", e)
    # Short interest (coverage varies)
    try:
        url_si = "https://api.polygon.io/stocks/v1/short-interest"
        js_si = await _http_json(url_si, base_qs)
        if js_si and isinstance(js_si.get("results"), list) and js_si["results"]:
            out["short_interest"] = js_si["results"][0]
    except Exception as e:
        # 400s are common; don't spam logs
        logger.info("[feature] short interest unavailable for %s (%r)", symbol, e)
    return out

# ---------- Public builder ----------
async def build_features(client, alert: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    symbol = alert.get("symbol")
    out: Dict[str, Any] = {}

    # Carry snapshot extras if present
    if snapshot and isinstance(snapshot.get("results"), dict):
        res = snapshot["results"]
        for k in ("day", "underlying_asset", "greeks", "implied_volatility"):
            if k in res and out.get(k) is None:
                out[k] = res[k]

    # TA on underlying
    try:
        ta = await _compute_ta_bundle(symbol)
        out.update(ta)
    except Exception as e:
        logger.warning("[feature] TA compute failed: %r", e)

    # Prev day OHLC + change%
    try:
        prev = await _prev_day_ohlc(symbol)
        out.update(prev)
        out.setdefault("quote_change_pct", None)
    except Exception as e:
        logger.warning("[feature] prev day bundle fail: %r", e)

    # 5-day avg PDH/PDL
    out.update(await _prev5_avg_hilo(symbol))

    # Premarket H/L
    out.update(await _premarket_hilo(symbol))

    # Short volume/interest (optional)
    out.update(await _short_volume_ratio(symbol))

    # Provide synthetic NBBO scaffold (downstream may adopt if needed)
    out.update(_synthetic_nbbo_from_last(None, None))

    return out
