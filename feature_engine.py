import os
import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY, CDT_TZ

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
    # Polygon v2/aggs accepts epoch ms or YYYY-MM-DD.
    # We use epoch ms to get precise intraday windows.
    return int(dt.timestamp() * 1000)

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
    gains = []
    losses = []
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        if chg >= 0:
            gains.append(chg)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-chg)
    # Wilder's smoothing
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
    ema_fast = _ema(values, f)
    ema_slow = _ema(values, s)
    if ema_fast is None or ema_slow is None:
        return (None, None, None)
    macd_line_series: List[float] = []
    # Build EMA series to get proper macd signal
    # We'll approximate by recomputing EMA across the whole list for robustness
    # (Lightweight and sufficient here)
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
    for ef, es in zip(ema_f_series, ema_s_series):
        if ef is None or es is None:
            macd_line_series.append(None)
        else:
            macd_line_series.append(ef - es)

    # signal line EMA on macd_line
    macd_clean = [m for m in macd_line_series if m is not None]
    if len(macd_clean) < sig:
        return (None, None, None)

    # compute EMA over macd_line_series (ignoring Nones)
    def ema_over_series(series: List[Optional[float]], period: int) -> List[Optional[float]]:
        vals = [x for x in series if x is not None]
        if len(vals) < period:
            return [None] * len(series)
        k2 = 2 / (period + 1)
        ema_v2 = sum(vals[:period]) / period
        result: List[Optional[float]] = []
        # placeholders for the leading Nones
        lead_nones = series.index(vals[0])
        result.extend([None] * lead_nones)
        # first EMA
        result.append(ema_v2)
        idx_vals = period
        for i in range(lead_nones + 1, len(series)):
            cur = series[i]
            if cur is None:
                result.append(result[-1])
            else:
                if idx_vals < len(vals):
                    # should never happen in this flow
                    pass
                ema_v2 = cur * k2 + result[-1] * (1 - k2)
                result.append(ema_v2)
        # fix length
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

# ---------- Polygon bars fetch ----------
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
        vw = b.get("vw")
        if isinstance(vw, (int, float)):
            price = float(vw)
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
    # Polygon bar 't' is epoch ms UTC
    open_utc = datetime.combine(datetime.utcnow().date(), datetime.min.time()).replace(tzinfo=timezone.utc) \
               .replace(hour=13, minute=30, second=0, microsecond=0)
    end_utc = open_utc + timedelta(minutes=15)
    lo = None
    hi = None
    start_ms = _to_ms(open_utc)
    end_ms = _to_ms(end_utc)
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

# ---------- TA Orchestrator ----------
async def _compute_ta_bundle(symbol: str) -> Dict[str, Any]:
    """
    Multi-tier fetch: 1m (2-3 days) -> 5m (~2 weeks) -> 1d (~2 years).
    We combine to get best-available RSI/SMA/EMA/MACD/Bollinger/VWAP/ORB.
    """
    now = _now_utc()
    out: Dict[str, Any] = {"ta_src": None}

    # 1) 1-minute bars for today (VWAP & ORB) and recent history (RSI/EMA/SMA if enough)
    try:
        start_1m_hist = now - timedelta(days=3)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start_1m_hist, now, limit=50000)
    except Exception as e:
        logger.warning("[feature] 1m fetch failed: %r", e)
        bars_1m = []

    # Today subset for VWAP/ORB
    bars_1m_today: List[Dict[str, Any]] = []
    if bars_1m:
        # Keep bars with date == today
        today_date = now.date()
        for b in bars_1m:
            ts = b.get("t")
            if not isinstance(ts, (int, float)):
                continue
            dt = datetime.fromtimestamp(ts/1000.0, tz=timezone.utc).astimezone(timezone.utc)
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

    # 3) Daily bars for longer trends / Bollinger
    try:
        start_1d = now - timedelta(days=750)  # ~3 years to get EMA200 safely
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

    # Bollinger (20, 2σ) from daily closes
    bb_mid_1d = sma20_1d
    bb_std_1d = _stddev(closes_1d, 20) if len(closes_1d) >= 20 else None
    bb_upper_1d = (bb_mid_1d + 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None
    bb_lower_1d = (bb_mid_1d - 2 * bb_std_1d) if (bb_mid_1d is not None and bb_std_1d is not None) else None

    # Choose best available granularity for each TA
    # Preference: 1m (richest) -> 5m -> 1d
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
    })

    # Mark the primary source we managed to use for RSI/EMA20/SMA20
    ta_src = None
    if out["rsi14"] is not None and out["ema20"] is not None and out["sma20"] is not None:
        if rsi14_1m == out["rsi14"] or ema20_1m == out["ema20"] or sma20_1m == out["sma20"]:
            ta_src = "1m"
        elif rsi14_5m == out["rsi14"] or ema20_5m == out["ema20"] or sma20_5m == out["sma20"]:
            ta_src = "5m"
        else:
            ta_src = "1d"
    else:
        # pick the best available hint
        ta_src = "1m" if closes_1m else ("5m" if closes_5m else ("1d" if closes_1d else None))
    out["ta_src"] = ta_src
    return out

# ---------- Synthetic NBBO ----------
def _synthetic_nbbo_from_last(last: Optional[float], base_spread_pct: float = None) -> Dict[str, Any]:
    if not isinstance(last, (int, float)) or last <= 0:
        return {}
    # Wider synthetic after-hours, tighter in RTH via env knob
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

# ---------- Public builder ----------
async def build_features(
    client,
    alert: Dict[str, Any],
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a unified feature dict combining:
      - Option snapshot enrichments (when provided)
      - TA on the UNDERLYING (RSI/SMA/EMA/MACD/BBands/VWAP/ORB)
      - Synthetic NBBO fields (if needed) based on option 'last'
    """
    symbol = alert.get("symbol")
    out: Dict[str, Any] = {}

    # Carry over some snapshot enrichments if provided by caller (market_ops)
    if snapshot and isinstance(snapshot.get("results"), dict):
        res = snapshot["results"]
        # best-effort extra fields (these may already be incorporated by caller)
        for k in ("day", "underlying_asset", "greeks", "implied_volatility"):
            if k in res and out.get(k) is None:
                out[k] = res[k]

    # Technicals on the underlying
    try:
        ta = await _compute_ta_bundle(symbol)
        out.update(ta)
    except Exception as e:
        logger.warning("[feature] TA compute failed: %r", e)

    # % change vs prior close on underlying (daily bars)
    try:
        now = _now_utc()
        # Small 5d range to catch yesterday
        bars_y = await _fetch_aggs_polygon(symbol, 1, "day", now - timedelta(days=5), now)
        if isinstance(bars_y, list) and len(bars_y) >= 2:
            last_c = bars_y[-1].get("c")
            prev_c = bars_y[-2].get("c")
            if isinstance(last_c, (int, float)) and isinstance(prev_c, (int, float)) and prev_c > 0:
                out["ul_change_pct"] = round((float(last_c) - float(prev_c)) / float(prev_c) * 100.0, 3)
    except Exception as e:
        logger.warning("[feature] UL change pct failed: %r", e)

    # Synthetic NBBO seed from LAST if option NBBO missing downstream
    # Caller will adopt synthetic_* if real NBBO still missing.
    option_last = None
    # If caller passed an option-level snapshot dict, try to extract last trade price
    try:
        if snapshot and isinstance(snapshot.get("results"), dict):
            opt = snapshot["results"].get("option", {})
            last_trade = opt.get("last_quote") or {}
            # Not always present; we rely on market_ops.poly_option_backfill to set f['last'] too.
    except Exception:
        pass

    # The caller (engine_processor) merges our fields into 'f' and already fetched f['last']
    # We don't see that here. Provide a synthetic pack that engine can overlay later.
    synth = _synthetic_nbbo_from_last(option_last, None)  # likely None here; engine will call this again if needed
    out.update(synth)
    return out
