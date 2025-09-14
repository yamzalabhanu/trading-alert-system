# feature_engine.py
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
    # We use epoch ms to get precise intraday windows and avoid 400s.
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

    # Build EMA series so we can generate a proper MACD signal EMA
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
        if ef is None or es is None:
            macd_line_series.append(None)
        else:
            macd_line_series.append(ef - es)

    # signal line EMA on macd_line
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

async def _prev_day_bar(symbol: str) -> Optional[Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return None
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
    js = await _http_json(url, {"adjusted": "true", "apiKey": POLYGON_API_KEY}, timeout=6.0)
    return js

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

# ---------- Short interest / short volume (best-effort) ----------
async def _try_short_metrics(symbol: str) -> Dict[str, Any]:
    """
    Best-effort short interest / short volume (plan-dependent).
    Returns empty dict if not available on your plan.
    """
    if not POLYGON_API_KEY:
        return {}
    out: Dict[str, Any] = {}
    candidates = [
        ("v3_shorts", f"https://api.polygon.io/v3/reference/shorts"),
        ("v2_shorts", f"https://api.polygon.io/v2/reference/shorts/{symbol}"),
        ("v2_shorts_vol", f"https://api.polygon.io/v2/reference/shorts/{symbol}/volume"),
    ]
    for name, url in candidates:
        js = await _http_json(url, {"ticker": symbol, "apiKey": POLYGON_API_KEY}, timeout=6.0)
        if js and isinstance(js, dict):
            out[name] = js

    # Try to normalize
    def _extract(j: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        sv = ti = tv = None
        try:
            rows = j.get("results") or j.get("data") or []
            if isinstance(rows, dict):
                rows = [rows]
            latest = rows[-1] if rows else {}
            sv = latest.get("short_volume") or latest.get("shortVolume")
            tv = latest.get("total_volume") or latest.get("totalVolume")
            ti = latest.get("short_interest") or latest.get("shortInterest")
            return (float(sv) if sv is not None else None,
                    float(ti) if ti is not None else None,
                    float(tv) if tv is not None else None)
        except Exception:
            return (None, None, None)

    sv = si = tv = None
    for _, v in out.items():
        _sv, _si, _tv = _extract(v)
        sv = sv or _sv; si = si or _si; tv = tv or _tv
    ratio = (sv / tv) if (sv and tv and tv > 0) else None
    return {
        "short_volume": sv,
        "short_interest": si,
        "short_volume_total": tv,
        "short_volume_ratio": ratio,
    }

# ---------- TA Orchestrator ----------
async def _compute_ta_bundle(symbol: str) -> Dict[str, Any]:
    """
    Multi-tier fetch: 1m (2-3 days) -> 5m (~2 weeks) -> 1d (~2 years).
    We combine to get best-available RSI/SMA/EMA/MACD/Bollinger/VWAP/ORB.
    """
    now = _now_utc()
    out: Dict[str, Any] = {"ta_src": None}

    # 1) 1-minute bars (3d) for VWAP/ORB + TA if enough history
    try:
        start_1m_hist = now - timedelta(days=3)
        bars_1m = await _fetch_aggs_polygon(symbol, 1, "minute", start_1m_hist, now, limit=50000)
    except Exception as e:
        logger.warning("[feature] 1m fetch failed: %r", e)
        bars_1m = []

    # Today subset for VWAP/ORB
    bars_1m_today: List[Dict[str, Any]] = []
    if bars_1m:
        today_date = now.date()
        for b in bars_1m:
            ts = b.get("t")
            if not isinstance(ts, (int, float)):
                continue
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
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

    # 2) 5-minute bars (~14d) as fallback
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

    # 3) Daily bars (~750d) for long trend + Bollinger
    try:
        start_1d = now - timedelta(days=750)
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
        "vwap": vwap,
        "orb15_high": orb_hi,
        "orb15_low":  orb_lo,
    })

    # Tag TA source
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

    # Also map macd_line → macd (for consumers expecting 'macd')
    if out.get("macd_line") is not None and "macd" not in out:
        out["macd"] = out["macd_line"]

    return out

# ---------- Premarket & previous-5 averages ----------
async def _prev5_avg_hilo(symbol: str, now_utc: datetime) -> Tuple[Optional[float], Optional[float]]:
    end = (now_utc - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=10)
    bars = await _fetch_aggs_polygon(symbol, 1, "day", start, end, limit=50000)
    if not bars:
        return (None, None)
    last5 = bars[-5:] if len(bars) >= 5 else bars
    highs = [b.get("h") for b in last5 if isinstance(b.get("h"), (int, float))]
    lows  = [b.get("l") for b in last5 if isinstance(b.get("l"), (int, float))]
    if not highs or not lows:
        return (None, None)
    return (sum(highs) / len(highs), sum(lows) / len(lows))

async def _premarket_high_low(symbol: str, today_local_dt: datetime) -> Tuple[Optional[float], Optional[float]]:
    """
    Premarket window 4:00–9:30 ET → 3:00–8:30 CT. Use 1m bars.
    """
    d = today_local_dt.date()
    start_ct = datetime(d.year, d.month, d.day, 3, 0, 0, tzinfo=CDT_TZ).astimezone(timezone.utc)
    end_ct   = datetime(d.year, d.month, d.day, 8, 30, 0, tzinfo=CDT_TZ).astimezone(timezone.utc)
    bars = await _fetch_aggs_polygon(symbol, 1, "minute", start_ct, end_ct, limit=50000)
    if not bars:
        return (None, None)
    highs = [b.get("h") for b in bars if isinstance(b.get("h"), (int, float))]
    lows  = [b.get("l") for b in bars if isinstance(b.get("l"), (int, float))]
    return (max(highs) if highs else None, min(lows) if lows else None)

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
        "synthetic_spread_pct": round((ask - bid) / mid * 100.0, 3) if mid > 0 else round(spread_pct, 3),
        "synthetic_quote_age_sec": None,  # engine may set this if it has a last timestamp
        "synthetic_nbbo_spread_est": round(spread_pct, 3),
    }

def _extract_option_last_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Try hard to find a usable 'last' from Polygon option snapshot.
    Tolerates various shapes; falls back to mid if bid/ask present.
    """
    try:
        if not snapshot or not isinstance(snapshot.get("results"), dict):
            return None
        res = snapshot["results"]

        # Common places
        for key in ("last", "mark", "mid", "price", "p"):
            v = res.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)

        # nested option -> last_trade/last_quote
        opt = res.get("option") or {}
        last_trade = opt.get("last_trade") or opt.get("last") or {}
        for k in ("price", "p", "last", "c"):
            v = last_trade.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)

        # derive from bid/ask
        bid = res.get("bid")
        ask = res.get("ask")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid and (ask + bid) > 0:
            return (float(bid) + float(ask)) / 2.0
    except Exception:
        pass
    return None

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
      - Previous-day OHLC, previous-5 avg high/low, pre-market H/L
      - Short volume / short interest (best-effort)
      - Synthetic NBBO fields (if needed) based on option 'last'
      - Underlying Δ% vs prior close
    """
    symbol = alert.get("symbol")
    out: Dict[str, Any] = {}

    # ---------- Technicals on underlying ----------
    try:
        ta = await _compute_ta_bundle(symbol)
        out.update(ta)
    except Exception as e:
        logger.warning("[feature] TA compute failed: %r", e)

    # ---------- Previous day bar & Δ% vs prior close ----------
    try:
        prev = await _prev_day_bar(symbol)
        pr = (prev or {}).get("results") or {}
        if isinstance(pr, list):
            pr = pr[-1] if pr else {}
        if isinstance(pr, dict):
            out["prev_close"] = pr.get("c")
            out["prev_open"]  = pr.get("o")
            out["prev_high"]  = pr.get("h")
            out["prev_low"]   = pr.get("l")

        # Compute change vs previous close using alert UL price if available; else latest daily close
        ul_price = alert.get("underlying_price_from_alert")
        base = out.get("prev_close")
        ref_price = float(ul_price) if isinstance(ul_price, (int, float)) else None
        if base is not None and isinstance(base, (int, float)) and base > 0:
            if ref_price is None:
                # fallback to most recent daily close
                now = _now_utc()
                bars = await _fetch_aggs_polygon(symbol, 1, "day", now - timedelta(days=5), now, limit=50000)
                if bars:
                    last_c = bars[-1].get("c")
                    if isinstance(last_c, (int, float)):
                        ref_price = float(last_c)
            if ref_price is not None:
                out["quote_change_pct"] = round((ref_price - float(base)) / float(base) * 100.0, 3)
    except Exception as e:
        logger.warning("[feature] prev-day & Δ%% failed: %r", e)

    # ---------- Previous 5 days avg PDH/PDL ----------
    try:
        avg_hi, avg_lo = await _prev5_avg_hilo(symbol, _now_utc())
        out["prev5_avg_high"] = avg_hi
        out["prev5_avg_low"]  = avg_lo
    except Exception as e:
        logger.warning("[feature] prev5 avg H/L failed: %r", e)

    # ---------- Pre-market high/low (today) ----------
    try:
        pm_hi, pm_lo = await _premarket_high_low(symbol, datetime.now(CDT_TZ))
        out["premarket_high"] = pm_hi
        out["premarket_low"]  = pm_lo
    except Exception as e:
        logger.warning("[feature] premarket H/L failed: %r", e)

    # ---------- Short interest / short volume (best-effort) ----------
    try:
        out.update(await _try_short_metrics(symbol))
    except Exception as e:
        logger.warning("[feature] short metrics failed: %r", e)

    # ---------- VWAP distance vs alert price ----------
    try:
        vwap = out.get("vwap")
        price = float(alert.get("underlying_price_from_alert")) if alert.get("underlying_price_from_alert") is not None else None
        if vwap is not None and isinstance(price, (int, float)) and vwap != 0:
            out["vwap_dist"] = (price - vwap) / vwap * 100.0
    except Exception:
        pass

    # ---------- Synthetic NBBO seed from LAST if option NBBO is missing ----------
    try:
        option_last = _extract_option_last_from_snapshot(snapshot)
        if option_last is not None and option_last > 0:
            out.update(_synthetic_nbbo_from_last(option_last, None))
    except Exception as e:
        logger.warning("[feature] synthetic NBBO seed failed: %r", e)

    # Tag source if not set
    out.setdefault("ta_src", out.get("ta_src") or "indicators+aggs")

    # Ensure fields LLM looks for exist (when we only computed daily MACD as macd_line)
    if out.get("macd_line") is not None and "macd" not in out:
        out["macd"] = out["macd_line"]

    return out
