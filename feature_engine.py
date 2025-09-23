# feature_engine.py
import os
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY, CDT_TZ

# Extra providers (IEX/Alpaca) as fallbacks
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
    """Return today's RTH open/close in UTC using the configured CDT_TZ."""
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

# ---------- RTH VWAP & ORB helpers ----------
def _today_rth_bounds_utc(now: datetime) -> Tuple[int, int]:
    """
    Regular Trading Hours in UTC for US equities:
    8:30–15:00 CDT => 13:30–20:00 UTC (intraday end capped to 'now').
    """
    start = now.replace(hour=13, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
    end   = now.replace(hour=20, minute=0,  second=0, microsecond=0, tzinfo=timezone.utc)
    end = min(end, now)
    return _to_ms(start), _to_ms(end)

def _slice_bars_by_ms(bars: List[Dict[str, Any]], start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in bars:
        t = b.get("t")
        if isinstance(t, (int, float)) and start_ms <= t < end_ms:
            out.append(b)
    return out

def _rth_vwap(bars_in_window: List[Dict[str, Any]]) -> Optional[float]:
    """VWAP computed over provided RTH window (uses 'vw' if present, else close)."""
    tot_pv = 0.0
    tot_v = 0.0
    for b in bars_in_window:
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

def _orb_15_from_bars(bars_1m: List[Dict[str, Any]], now: datetime) -> Tuple[Optional[float], Optional[float]]:
    """
    Opening 15-minute range is strictly 13:30–13:45 UTC (first 15 minutes of RTH).
    """
    start_open = now.replace(hour=13, minute=30, second=0, microsecond=0, tzinfo=timezone.utc)
    end_open   = now.replace(hour=13, minute=45, second=0, microsecond=0, tzinfo=timezone.utc)
    start_ms, end_ms = _to_ms(start_open), _to_ms(end_open)
    hi = None; lo = None
    for b in bars_1m:
        t = b.get("t")
        if not isinstance(t, (int, float)):
            continue
        if start_ms <= t < end_ms:
            h = b.get("h"); l = b.get("l")
            if isinstance(h, (int, float)): hi = h if hi is None else max(hi, h)
            if isinstance(l, (int, float)): lo = l if lo is None else min(lo, l)
    return (hi, lo)

# ---------- Bars fetch (Polygon + fallbacks) ----------
async def _fetch_aggs_polygon(symbol: str, multiplier: int, timespan: str, start: datetime, end: datetime, limit: int = 50000) -> List[Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{_to_ms(start)}/{_to_ms(end)}"
    js = await _http_json(url, {"adjusted": "true", "sort": "asc", "limit": str(limit), "apiKey": POLYGON_API_KEY}, timeout=10.0)
    if not js or not isinstance(js.get("results"), list):
        return []
    return js["results"]

def _pick_close_series(bars: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for b in bars:
        c = b.get("c")
        if isinstance(c, (int, float)):
            out.append(float(c))
    return out

# ---------- TA Orchestrator ----------
async def _compute_ta_bundle(symbol: str) -> Dict[str, Any]:
    now = _now_utc()
    out: Dict[str, Any] = {"ta_src": None}

    # 1) 1-minute (Polygon → fallbacks)
    try:
        start_1m_hist = now - timedelta(days=3)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start_1m_hist, now, limit=50000)
        if not bars_1m:
            bars_1m = await fetch_1m_bars_any(symbol) or []
    except Exception as e:
        logger.warning("[feature] 1m fetch failed: %r", e)
        bars_1m = await fetch_1m_bars_any(symbol) or []

    # RTH VWAP & ORB (13:30–20:00 UTC; ORB = 13:30–13:45 UTC)
    vwap_rth: Optional[float] = None
    orb_hi: Optional[float] = None
    orb_lo: Optional[float] = None
    if bars_1m:
        rth_start_ms, rth_end_ms = _today_rth_bounds_utc(now)
        bars_1m_rth = _slice_bars_by_ms(bars_1m, rth_start_ms, rth_end_ms)
        if bars_1m_rth:
            vwap_rth = _rth_vwap(bars_1m_rth)
        orb_hi, orb_lo = _orb_15_from_bars(bars_1m, now)

    # Intraday TA using full 1m history we fetched
    closes_1m = _pick_close_series(bars_1m)
    rsi14_1m = _rsi(closes_1m, 14) if len(closes_1m) >= 50 else None
    sma20_1m = _sma(closes_1m, 20)
    ema20_1m = _ema(closes_1m, 20)
    ema50_1m = _ema(closes_1m, 50)
    ema200_1m = _ema(closes_1m, 200)
    macd_line_1m, macd_signal_1m, macd_hist_1m = _macd(closes_1m)

    # 2) 5-minute
    try:
        start_5m = now - timedelta(days=14)
        bars_5m = await _fetch_aggs_polygon(symbol, 5, "minute", start_5m, now, limit=50000)
        if not bars_5m:
            bars_5m = await fetch_5m_bars_any(symbol) or []
    except Exception as e:
        logger.warning("[feature] 5m fetch failed: %r", e)
        bars_5m = await fetch_5m_bars_any(symbol) or []

    closes_5m = _pick_close_series(bars_5m)
    rsi14_5m = _rsi(closes_5m, 14) if len(closes_5m) >= 50 else None
    sma20_5m = _sma(closes_5m, 20)
    ema20_5m = _ema(closes_5m, 20)
    ema50_5m = _ema(closes_5m, 50)
    ema200_5m = _ema(closes_5m, 200)
    macd_line_5m, macd_signal_5m, macd_hist_5m = _macd(closes_5m)

    # 3) Daily
    try:
        start_1d = now - timedelta(days=750)
        bars_1d = await _fetch_aggs_polygon(symbol, 1, "day", start_1d, now, limit=50000)
        if not bars_1d:
            bars_1d = await fetch_1d_bars_any(symbol) or []
    except Exception as e:
        logger.warning("[feature] 1d fetch failed: %r", e)
        bars_1d = await fetch_1d_bars_any(symbol) or []

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
        "rsi14":       _pick(rsi14_1m, rsi14_5m, rsi14_1d),
        "sma20":       _pick(sma20_1m, sma20_5m, sma20_1d),
        "ema20":       _pick(ema20_1m, ema20_5m, ema20_1d),
        "ema50":       _pick(ema50_1m, ema50_5m, ema50_1d),
        "ema200":      _pick(ema200_1m, ema200_5m, ema200_1d),
        "macd_line":   _pick(macd_line_1m, macd_line_5m, macd_line_1d),
        "macd_signal": _pick(macd_signal_1m, macd_signal_5m, macd_signal_1d),
        "macd_hist":   _pick(macd_hist_1m, macd_hist_5m, macd_hist_1d),
        "bb_upper": bb_upper_1d,
        "bb_lower": bb_lower_1d,
        "bb_mid":   bb_mid_1d,
        "vwap_rth": vwap_rth,
        "vwap":     vwap_rth,   # backward-compat
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
        bars = await _fetch_aggs_polygon(symbol, 1, "day", now - timedelta(days=10), now) or await fetch_1d_bars_any(symbol) or []
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
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", now - timedelta(days=1), now) or await fetch_1m_bars_any(symbol) or []
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
    """Use v1 endpoints (may be delayed/limited)."""
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
        out["short_interest"] = ((js_si or {}).get("results") or [{}])[0] if (js_si and js_si.get("results")) else None
    except Exception as e:
        logger.warning("[feature] short interest fail: %r", e)

    return out

async def _fallback_1m(symbol: str) -> List[Dict[str, Any]]:
    from datetime import timedelta
    now = _now_utc()
    start = now - timedelta(days=3)
    try:
        return await fetch_1m_bars_any(symbol, start, now)  # signature (symbol, start, end)
    except TypeError:
        try:
            return await fetch_1m_bars_any(symbol)  # signature (symbol)
        except Exception:
            return []

async def _fallback_5m(symbol: str) -> List[Dict[str, Any]]:
    from datetime import timedelta
    now = _now_utc()
    start = now - timedelta(days=14)
    try:
        return await fetch_5m_bars_any(symbol, start, now)
    except TypeError:
        try:
            return await fetch_5m_bars_any(symbol)
        except Exception:
            return []

async def _fallback_1d(symbol: str) -> List[Dict[str, Any]]:
    from datetime import timedelta
    now = _now_utc()
    start = now - timedelta(days=750)
    try:
        return await fetch_1d_bars_any(symbol, start, now)
    except TypeError:
        try:
            return await fetch_1d_bars_any(symbol)
        except Exception:
            return []

# ---------- Public builder ----------
async def build_features(client, alert: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    symbol = alert.get("symbol")
    out: Dict[str, Any] = {}

    # Carry snapshot extras if any (Options Advanced snapshot often includes greeks/iv)
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
        # Leave quote_change_pct for downstream to compute from live mark vs prev_close
        out.setdefault("quote_change_pct", None)
    except Exception as e:
        logger.warning("[feature] prev day bundle fail: %r", e)

    # 5-day avg PDH/PDL
    out.update(await _prev5_avg_hilo(symbol))

    # Premarket H/L
    out.update(await _premarket_hilo(symbol))

    # Short volume/interest
    out.update(await _short_volume_ratio(symbol))

    # Provide synthetic NBBO scaffold (downstream can adopt if needed)
    out.update(_synthetic_nbbo_from_last(None, None))

    return out
