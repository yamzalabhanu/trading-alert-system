# feature_engine v1.0.py
import os
import math
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY

logger = logging.getLogger("trading_engine.feature_engine")

# ==============================================================================
# Small HTTP helpers
# ==============================================================================

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

async def _rate_limit_json(url: str, params: Dict[str, Any], timeout: float = 8.0, retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Like _http_json but tolerates 429 with short backoff; on other client errors, aborts.
    """
    client = get_http_client()
    if client is None:
        return None
    last_exc = None
    for i in range(retries + 1):
        try:
            r = await client.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, dict) else None
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                await asyncio.sleep(0.4 * (i + 1))
                continue
            break
    logger.warning("[feature] _rate_limit_json failed: %r (url=%s)", last_exc, url)
    return None

# ==============================================================================
# Time helpers
# ==============================================================================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_ms(dt: datetime) -> int:
    # Polygon v2/aggs accepts epoch ms; use precise intraday windows
    return int(dt.timestamp() * 1000)

# ==============================================================================
# Basic TA
# ==============================================================================

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
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        if chg >= 0:
            gains.append(chg); losses.append(0.0)
        else:
            gains.append(0.0); losses.append(-chg)
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

def _macd(values: List[float], f: int = 12, s: int = 26, sig: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(values) < (s + sig):
        return (None, None, None)

    def ema_series(vals: List[float], period: int) -> List[Optional[float]]:
        k = 2 / (period + 1)
        out: List[Optional[float]] = []
        if len(vals) < period:
            return [None] * len(vals)
        ema_v = sum(vals[:period]) / period
        out.extend([None] * (period - 1))
        out.append(ema_v)
        for v in vals[period:]:
            ema_v = v * k + ema_v * (1 - k)
            out.append(ema_v)
        return out

    ema_f_series = ema_series(values, f)
    ema_s_series = ema_series(values, s)
    macd_line_series: List[Optional[float]] = []
    for ef, es in zip(ema_f_series, ema_s_series):
        macd_line_series.append((ef - es) if (ef is not None and es is not None) else None)

    macd_clean = [m for m in macd_line_series if m is not None]
    if len(macd_clean) < sig:
        return (None, None, None)

    def ema_over_series(series: List[Optional[float]], period: int) -> List[Optional[float]]:
        vals = [x for x in series if x is not None]
        if len(vals) < period:
            return [None] * len(series)
        k2 = 2 / (period + 1)
        ema_v2 = sum(vals[:period]) / period
        result: List[Optional[float]] = []
        lead_nones = series.index(vals[0])
        result.extend([None] * lead_nones)
        result.append(ema_v2)
        for i in range(lead_nones + 1, len(series)):
            cur = series[i]
            if cur is None:
                result.append(result[-1])
            else:
                ema_v2 = cur * k2 + result[-1] * (1 - k2)
                result.append(ema_v2)
        result = result[:len(series)]
        if len(result) < len(series):
            result.extend([None] * (len(series) - len(result)))
        return result

    macd_signal_series = ema_over_series(macd_line_series, sig)
    macd_line = macd_line_series[-1]
    macd_signal = macd_signal_series[-1]
    if macd_line is None or macd_signal is None:
        return (None, None, None)
    macd_hist = macd_line - macd_signal
    return (macd_line, macd_signal, macd_hist)

# ==============================================================================
# Polygon bars fetch
# ==============================================================================

async def _fetch_aggs_polygon(
    symbol: str,
    multiplier: int,
    timespan: str,  # "minute" or "day"
    start: datetime,
    end: datetime,
    limit: int = 50000,
) -> List[Dict[str, Any]]:
    """
    Returns list of bars with keys (t, o, h, l, c, v, vw) when available.
    Uses epoch ms for from/to to satisfy Polygon v2 aggs format.
    """
    if not POLYGON_API_KEY:
        return []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{_to_ms(start)}/{_to_ms(end)}"
    js = await _rate_limit_json(
        url,
        {"adjusted": "true", "sort": "asc", "limit": str(limit), "apiKey": POLYGON_API_KEY},
        timeout=10.0,
    )
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

def _intraday_vwap(bars: List[Dict[str, Any]]) -> Optional[float]:
    """
    Session VWAP via sum(vw_i * v_i) / sum(v_i) if 'vw' present; else sum(c_i * v_i)/sum(v_i).
    """
    tot_pv = 0.0
    tot_v = 0.0
    for b in bars:
        v = float(b.get("v") or 0.0)
        if v <= 0:
            continue
        if isinstance(b.get("vw"), (int, float)):
            price = float(b["vw"])
        else:
            c = b.get("c")
            if not isinstance(c, (int, float)):
                continue
            price = float(c)
        tot_pv += price * v
        tot_v += v
    return (tot_pv / tot_v) if tot_v > 0 else None

def _orb_15(bars_today_1m: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Opening Range Breakout (first 15 mins) for US equities:
    RTH open ~ 13:30:00 UTC. We take bars where t in [13:30, 13:45) UTC.
    """
    if not bars_today_1m:
        return (None, None)
    open_utc = datetime.combine(datetime.utcnow().date(), datetime.min.time()).replace(tzinfo=timezone.utc) \
               .replace(hour=13, minute=30, second=0, microsecond=0)
    end_utc = open_utc + timedelta(minutes=15)
    start_ms = _to_ms(open_utc)
    end_ms = _to_ms(end_utc)
    lo: Optional[float] = None
    hi: Optional[float] = None
    for b in bars_today_1m:
        ts = b.get("t")
        if not isinstance(ts, (int, float)):
            continue
        if start_ms <= ts < end_ms:
            low = b.get("l"); high = b.get("h")
            if isinstance(low, (int, float)):
                lo = low if lo is None else min(lo, low)
            if isinstance(high, (int, float)):
                hi = high if hi is None else max(hi, high)
    return (hi, lo)

# ==============================================================================
# Short Volume / Short Interest (v1 endpoints, tolerant)
# ==============================================================================

async def _pull_short_flow_polygon(symbol: str) -> Dict[str, Any]:
    """
    Use Polygon legacy v1 short endpoints. Avoid unsupported server-side sorts.
    Fallback: fetch a page without ticker and filter locally.
    Returns: {
      short_volume, short_volume_total, short_volume_ratio, short_interest
    }  (keys only when available)
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    base = "https://api.polygon.io/stocks/v1"
    pkey = {"apiKey": POLYGON_API_KEY}

    # Short Volume
    js = await _rate_limit_json(f"{base}/short-volume", {"ticker": symbol, "limit": 5, **pkey})
    if not js or not isinstance(js.get("results"), list):
        js = await _rate_limit_json(f"{base}/short-volume", {"limit": 100, **pkey})
        results = [r for r in (js.get("results") or []) if r.get("ticker") == symbol] if js else []
    else:
        results = js["results"]

    row = None
    if results:
        def _row_dt(x):
            d = x.get("date") or x.get("t") or x.get("timestamp")
            try:
                return str(d)
            except Exception:
                return ""
        row = sorted(results, key=_row_dt, reverse=True)[0]

    if row:
        sv  = row.get("short_volume") or row.get("shortVolume") or row.get("short_vol")
        tv  = row.get("total_volume")  or row.get("totalVolume")  or row.get("volume")
        try:
            svf = float(sv) if sv is not None else None
            tvf = float(tv) if tv is not None and float(tv) > 0 else None
        except Exception:
            svf, tvf = None, None
        if svf is not None:
            out["short_volume"] = svf
        if tvf is not None:
            out["short_volume_total"] = tvf
        if svf is not None and tvf:
            out["short_volume_ratio"] = round(svf / tvf, 4)

    # Short Interest
    js2 = await _rate_limit_json(f"{base}/short-interest", {"ticker": symbol, "limit": 5, **pkey})
    if not js2 or not isinstance(js2.get("results"), list):
        js2 = await _rate_limit_json(f"{base}/short-interest", {"limit": 100, **pkey})
        si_results = [r for r in (js2.get("results") or []) if r.get("ticker") == symbol] if js2 else []
    else:
        si_results = js2["results"]

    if si_results:
        def _row_dt2(x):
            d = x.get("date") or x.get("t") or x.get("timestamp")
            try:
                return str(d)
            except Exception:
                return ""
        srow = sorted(si_results, key=_row_dt2, reverse=True)[0]
        si_val = srow.get("short_interest") or srow.get("shortInterest") or srow.get("si") or srow.get("value")
        try:
            if si_val is not None:
                out["short_interest"] = float(si_val)
        except Exception:
            pass

    return out

# ==============================================================================
# TA Orchestrator
# ==============================================================================

async def _compute_ta_bundle(symbol: str) -> Dict[str, Any]:
    """
    Multi-tier fetch: 1m (2-3 days) -> 5m (~2 weeks) -> 1d (~2 years).
    Combine to get best-available RSI/SMA/EMA/MACD/Bollinger/VWAP/ORB.
    """
    now = _now_utc()
    out: Dict[str, Any] = {"ta_src": None}

    # 1) 1-minute bars (for VWAP/ORB + high-res TA if enough history)
    try:
        start_1m_hist = now - timedelta(days=3)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start_1m_hist, now, limit=50000)
    except Exception as e:
        logger.warning("[feature] 1m fetch failed: %r", e)
        bars_1m = []

    # Today subset for VWAP/ORB and premarket H/L
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

    # premarket high/low (before 13:30 UTC)
    pm_high = None; pm_low = None
    if bars_1m_today:
        rth_open = datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=timezone.utc, hour=13, minute=30)
        cutoff = _to_ms(rth_open)
        for b in bars_1m_today:
            ts = b.get("t"); h = b.get("h"); l = b.get("l")
            if isinstance(ts, (int, float)) and ts < cutoff:
                if isinstance(h, (int, float)):
                    pm_high = h if pm_high is None else max(pm_high, h)
                if isinstance(l, (int, float)):
                    pm_low = l if pm_low is None else min(pm_low, l)

    closes_1m = _pick_close_series(bars_1m)
    rsi14_1m = _rsi(closes_1m, 14) if len(closes_1m) >= 50 else None
    sma20_1m = _sma(closes_1m, 20)
    ema20_1m = _ema(closes_1m, 20)
    ema50_1m = _ema(closes_1m, 50)
    ema200_1m = _ema(closes_1m, 200)
    macd_line_1m, macd_signal_1m, macd_hist_1m = _macd(closes_1m)

    # 2) 5-minute bars as fallback
    try:
        start_5m = now - timedelta(days=14)
        bars_5m = await _fetch_aggs_polygon(symbol, 5, "minute", start_5m, now, limit=50000)
    except Exception as e:
        logger.warning("[feature] 5m fetch failed: %r", e)
        bars_5m = []
    closes_5m = _pick_close_series(bars_5m)
    rsi14_5m = _rsi(closes_5m, 14) if len(closes_5m) >= 50 else None
    sma20_5m = _sma(closes_5m, 20)
    ema20_5m = _ema(closes_5m, 20)
    ema50_5m = _ema(closes_5m, 50)
    ema200_5m = _ema(closes_5m, 200)
    macd_line_5m, macd_signal_5m, macd_hist_5m = _macd(closes_5m)

    # 3) Daily bars for longer trends / Bollinger / context
    try:
        start_1d = now - timedelta(days=750)  # ~3y to compute EMA200 safely
        bars_1d = await _fetch_aggs_polygon(symbol, 1, "day", start_1d, now, limit=50000)
    except Exception as e:
        logger.warning("[feature] 1d fetch failed: %r", e)
        bars_1d = []
    closes_1d = _pick_close_series(bars_1d)
    rsi14_1d = _rsi(closes_1d, 14) if len(closes_1d) >= 30 else None
    sma20_1d = _sma(closes_1d, 20)
    ema20_1d = _ema(closes_1d, 20)
    ema50_1d = _ema(closes_1d, 50)
    ema200_1d = _ema(closes_1d, 200)
    macd_line_1d, macd_signal_1d, macd_hist_1d = _macd(closes_1d)

    # Bollinger from daily closes (20, 2σ)
    bb_mid_1d = sma20_1d
    bb_std_1d = _stddev(closes_1d, 20) if len(closes_1d) >= 20 else None
    bb_upper_1d = (bb_mid_1d + 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None
    bb_lower_1d = (bb_mid_1d - 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None
    bb_width = None
    if bb_mid_1d and bb_upper_1d and bb_lower_1d and bb_mid_1d != 0:
        bb_width = (bb_upper_1d - bb_lower_1d) / bb_mid_1d * 100.0

    # Choose best available granularity (prefer higher resolution)
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
        "bb_width": bb_width,
        "vwap": vwap,
        "orb15_high": orb_hi,
        "orb15_low":  orb_lo,
        "premarket_high": pm_high,
        "premarket_low":  pm_low,
    })

    # Mark primary TA source
    ta_src = None
    if out["rsi14"] is not None and out["ema20"] is not None and out["sma20"] is not None:
        if rsi14_1m == out["rsi14"] or ema20_1m == out["ema20"] or sma20_1m == out["sma20"]:
            ta_src = "1m"
        elif rsi14_5m == out["rsi14"] or ema20_5m == out["ema20"] or sma20_5m == out["sma20"]:
            ta_src = "5m"
        else:
            ta_src = "1d"
    else:
        ta_src = "1m" if closes_1m else ("5m" if closes_5m else ("1d" if closes_1d else None))
    out["ta_src"] = ta_src

    # From last ~6 daily bars compute prev-day OHLC and 5-day avg highs/lows
    try:
        now2 = _now_utc()
        bars_y = await _fetch_aggs_polygon(symbol, 1, "day", now2 - timedelta(days=7), now2)
        if isinstance(bars_y, list) and len(bars_y) >= 2:
            # last bar is "current"; prev is yesterday
            prev_bar = bars_y[-2]
            out["prev_open"]  = float(prev_bar.get("o")) if isinstance(prev_bar.get("o"), (int, float)) else None
            out["prev_high"]  = float(prev_bar.get("h")) if isinstance(prev_bar.get("h"), (int, float)) else None
            out["prev_low"]   = float(prev_bar.get("l")) if isinstance(prev_bar.get("l"), (int, float)) else None
            out["prev_close"] = float(prev_bar.get("c")) if isinstance(prev_bar.get("c"), (int, float)) else None

            # 5-day avg previous highs/lows (exclude current bar)
            highs: List[float] = []
            lows: List[float] = []
            history = bars_y[:-1]  # drop current bar
            for b in history[-5:]:
                if isinstance(b.get("h"), (int, float)):
                    highs.append(float(b["h"]))
                if isinstance(b.get("l"), (int, float)):
                    lows.append(float(b["l"]))
            if highs:
                out["prev5_avg_high"] = sum(highs) / len(highs)
            if lows:
                out["prev5_avg_low"] = sum(lows) / len(lows)

            # %Δ vs prev close using current close (if reported) or last c
            last_c = bars_y[-1].get("c")
            prev_c = prev_bar.get("c")
            if isinstance(last_c, (int, float)) and isinstance(prev_c, (int, float)) and prev_c > 0:
                out["quote_change_pct"] = round((float(last_c) - float(prev_c)) / float(prev_c) * 100.0, 3)
    except Exception as e:
        logger.warning("[feature] prev-day / 5d context failed: %r", e)

    return out

# ==============================================================================
# Synthetic NBBO
# ==============================================================================

def _synthetic_nbbo_from_last(last: Optional[float], base_spread_pct: Optional[float] = None) -> Dict[str, Any]:
    """
    Build synthetic bid/ask/mid around 'last' with configurable spread pct (default 1.0%).
    """
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
        "synthetic_quote_age_sec": None,  # set by caller if they have a timestamp
    }

# ==============================================================================
# Public builder
# ==============================================================================

async def build_features(
    client,  # kept for backward-compatibility with caller; not used directly
    alert: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a unified feature dict combining:
      - TA on the UNDERLYING (RSI/SMA/EMA/MACD/BBands/VWAP/ORB + premarket H/L)
      - Context: prev-day OHLC, %Δ vs prev close, 5d avg PDH/PDL, short volume/interest
      - Synthetic NBBO fields (if option 'last' is available in snapshot)
    """
    symbol = alert.get("symbol")
    out: Dict[str, Any] = {}

    # --- TA bundle on underlying
    try:
        ta = await _compute_ta_bundle(symbol)
        out.update(ta)
    except Exception as e:
        logger.warning("[feature] TA compute failed: %r", e)

    # --- Short volume / interest
    try:
        short_ctx = await _pull_short_flow_polygon(symbol)
        out.update(short_ctx)
    except Exception as e:
        logger.warning("[feature] short-flow fetch failed: %r", e)

    # --- Synthetic NBBO seed from LAST if present in snapshot
    try:
        opt_last = None
        if snapshot and isinstance(snapshot.get("results"), dict):
            res = snapshot["results"]
            # try a few plausible locations
            # v3 snapshot often has last_trade or last_quote for options
            last_trade = res.get("last_trade") or res.get("last_quote") or {}
            lp = last_trade.get("p") or last_trade.get("price") or res.get("day", {}).get("close")
            if isinstance(lp, (int, float)):
                opt_last = float(lp)
        synth = _synthetic_nbbo_from_last(opt_last, None)
        if synth:
            out.update(synth)
    except Exception as e:
        logger.warning("[feature] synthetic NBBO seed failed: %r", e)

    # --- VWAP distance vs underlying price from alert (if available)
    try:
        ul_px = alert.get("underlying_price_from_alert")
        if isinstance(ul_px, (int, float)) and isinstance(out.get("vwap"), (int, float)) and out["vwap"] != 0:
            out["vwap_dist"] = (float(ul_px) - float(out["vwap"])) / float(out["vwap"]) * 100.0
    except Exception:
        pass

    return out

__all__ = [
    "build_features",
    "_synthetic_nbbo_from_last",
]
