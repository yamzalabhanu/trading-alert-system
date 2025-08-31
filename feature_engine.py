# feature_engine.py
import math
from typing import List, Dict, Any, Optional
from collections import deque
from config import IV_HISTORY_LEN, REGIME_TREND_EMA_FAST, REGIME_TREND_EMA_SLOW
from polygon_client import get_aggs

# Global state for IV history
_IV_HISTORY: Dict[str, deque] = {}

def ema(values: List[float], period: int) -> Optional[float]:
        if not values or period <= 0 or len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e
    pass

def atr14(daily: List[Dict[str, Any]]) -> Optional[float]:
        if len(daily) < 15:
        return None
    bars = sorted(daily, key=lambda x: x["t"])  # ascending
    trs = []
    prev_close = bars[0]["c"]
    for b in bars[1:]:
        tr = max(b["h"] - b["l"], abs(b["h"] - prev_close), abs(b["l"] - prev_close))
        trs.append(tr)
        prev_close = b["c"]
    if len(trs) < 14:
        return None
    return sum(trs[-14:]) / 14.0
    pass

def realized_vol_annualized(daily_closes: List[float], window: int = 20) -> Optional[float]:
        if len(daily_closes) < window + 1:
        return None
    import statistics
    rets = []
    for i in range(1, window + 1):
        r = math.log(daily_closes[-i] / daily_closes[-i - 1])
        rets.append(r)
    stdev = statistics.pstdev(rets)
    return stdev * math.sqrt(252)
    pass

def iv_rank_from_history(hist: deque, current_iv: Optional[float]) -> Optional[float]:
        if current_iv is None or not hist:
        return None
    xs = list(hist)
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return 0.0
    return (current_iv - lo) / (hi - lo)
    pass

async def build_features(client: httpx.AsyncClient, *, alert: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
     res = snapshot.get("results") or {}
    greeks = res.get("greeks") or {}
    ua = res.get("underlying_asset") or {}
    day = res.get("day") or {}
    last_quote = res.get("last_quote") or {}
    last_trade = res.get("last_trade") or {}

    S = ua.get("price") or alert["underlying_price_from_alert"]
    cur_iv = res.get("implied_volatility")
    oi = int(res.get("open_interest") or 0)
    vol = int(day.get("volume") or 0)

    # NBBO & spread
    bid = last_quote.get("bid_price")
    ask = last_quote.get("ask_price")
    mid = None
    opt_spread_pct = None
    if (bid is not None) and (ask is not None) and (bid > 0) and (ask > 0):
        mid = (bid + ask) / 2.0
        if mid > 0:
            opt_spread_pct = (ask - bid) / mid
    if mid is None:
        mid = last_trade.get("price")

    # Quote age
    quote_ts_ns = last_quote.get("sip_timestamp") or last_quote.get("last_updated")
    now_ns = int(datetime.now(timezone.utc).timestamp() * 1e9)
    quote_age_sec = None
    if isinstance(quote_ts_ns, (int, float)) and quote_ts_ns > 0:
        quote_age_sec = max(0, (now_ns - int(quote_ts_ns)) / 1e9)

    # DTE + greeks
    days_to_exp = dte(alert["expiry"])
    delta = greeks.get("delta")
    gamma = greeks.get("gamma")
    theta = greeks.get("theta")
    vega  = greeks.get("vega")

    # Daily context (70 cal days)
    to_day = datetime.now(timezone.utc).date()
    frm_day = (to_day - timedelta(days=70)).isoformat()
    to_iso = to_day.isoformat()
    daily = await polygon_aggs(client, ticker=alert["symbol"], multiplier=1, timespan="day", frm=frm_day, to=to_iso, sort="asc")
    atr = atr14(daily) if daily else None
    closes = [b["c"] for b in daily] if daily else []
    rv20 = realized_vol_annualized(closes, window=20) if closes else None

    # Previous day H/L
    prev_day_high = prev_day_low = None
    if daily and len(daily) >= 2:
        prev = daily[-2]
        prev_day_high = prev.get("h")
        prev_day_low  = prev.get("l")

    # Intraday minute bars today (ET)
    et_open = _et_time(9, 30)
    et_close = _et_time(16, 0)
    et_from = _et_time(4, 0)      # earliest premarket slice
    et_to   = _et_time(23, 59)    # safety

    intraday = await polygon_aggs(client, ticker=alert["symbol"], multiplier=1, timespan="minute",
                                  frm=et_from.date().isoformat(), to=et_to.date().isoformat(), sort="asc", limit=5000)

    # Premarket H/L (before 09:30 ET)
    pm_high = pm_low = None
    if intraday:
        pm = [b for b in intraday if b.get("t") and (datetime.fromtimestamp(b["t"]/1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York")) < et_open)]
        if pm:
            pm_high = max(b["h"] for b in pm)
            pm_low  = min(b["l"] for b in pm)

    # VWAP since regular session open (simple cumulative TP*V/ΣV)
    vwap = None
    if intraday:
        reg = [b for b in intraday if b.get("t") and (et_open <= datetime.fromtimestamp(b["t"]/1000, tz=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York")) <= et_close)]
        if reg:
            num = 0.0
            den = 0.0
            for b in reg:
                tp = (b["h"] + b["l"] + b["c"]) / 3.0
                v  = float(b.get("v") or 0.0)
                num += tp * v
                den += v
            if den > 0:
                vwap = num / den

    # 20d extremes for S/R headroom
    high20 = max((b["h"] for b in daily[-20:]), default=None) if daily else None
    low20  = min((b["l"] for b in daily[-20:]), default=None) if daily else None

    sr_headroom_ok = None
    if atr and S and high20 and low20:
        if alert["side"] == "CALL":
            sr_headroom_ok = (high20 - S) >= HEADROOM_MIN_R * atr
        else:
            sr_headroom_ok = (S - low20) >= HEADROOM_MIN_R * atr

    # MTF trend alignment (5m + 15m EMA9/21)
    local_today = market_now().date()
    frm_intraday = local_today.isoformat()
    to_intraday  = (local_today + timedelta(days=1)).isoformat()

    def _closes(bars: List[Dict[str, Any]]) -> List[float]:
        return [b["c"] for b in bars]

    fivem   = await polygon_aggs(client, ticker=alert["symbol"], multiplier=5,  timespan="minute", frm=frm_intraday, to=to_intraday, sort="asc", limit=5000)
    fifteen = await polygon_aggs(client, ticker=alert["symbol"], multiplier=15, timespan="minute", frm=frm_intraday, to=to_intraday, sort="asc", limit=5000)

    ema9_5  = ema(_closes(fivem), 9) if fivem else None
    ema21_5 = ema(_closes(fivem), 21) if fivem else None
    ema9_15 = ema(_closes(fifteen), 9) if fifteen else None
    ema21_15= ema(_closes(fifteen), 21) if fifteen else None

    mtf_align = None
    if all(v is not None for v in [ema9_5, ema21_5, ema9_15, ema21_15]):
        if alert["side"] == "CALL":
            mtf_align = (ema9_5 > ema21_5) and (ema9_15 > ema21_15)
        else:
            mtf_align = (ema9_5 < ema21_5) and (ema9_15 < ema21_15)

    # EM vs Break-even (context)
    em_1s = None
    em_vs_be_ok = None
    if S and cur_iv is not None and days_to_exp is not None and days_to_exp >= 0:
        em_1s = S * float(cur_iv) * math.sqrt(max(0.0, days_to_exp) / 365.0)
        premium = mid or 0.0
        if alert["side"] == "CALL":
            be_price = alert["strike"] + premium
            be_dist  = max(0.0, be_price - S)
        else:
            be_price = alert["strike"] - premium
            be_dist  = max(0.0, S - be_price)
        if be_dist == 0.0:
            em_vs_be_ok = True
        elif em_1s is not None:
            em_vs_be_ok = (em_1s >= EM_VS_BE_RATIO_MIN * be_dist)

    # IV rank memory
    sym = alert["symbol"]
    if sym not in _IV_HISTORY:
        _IV_HISTORY[sym] = deque(maxlen=IV_HISTORY_LEN)
    if cur_iv is not None:
        _IV_HISTORY[sym].append(float(cur_iv))
    iv_rank = iv_rank_from_history(_IV_HISTORY[sym], cur_iv)

    # Market regime via SPY EMA trend
    spy_daily = await polygon_aggs(client, ticker="SPY", multiplier=1, timespan="day", frm=frm_day, to=to_iso, sort="asc")
    spy_close = [b["c"] for b in spy_daily] if spy_daily else []
    spy_ema_fast = ema(spy_close, REGIME_TREND_EMA_FAST) if spy_close else None
    spy_ema_slow = ema(spy_close, REGIME_TREND_EMA_SLOW) if spy_close else None
    regime_flag = None
    if (spy_ema_fast is not None) and (spy_ema_slow is not None):
        regime_flag = "trending" if spy_ema_fast > spy_ema_slow else "choppy"

    # Style presets
    if TRADE_STYLE == "swing":
        delta_min, delta_max = DELTA_MIN_SWING, DELTA_MAX_SWING
        dte_min, dte_max     = DTE_MIN_SWING, DTE_MAX_SWING
        risk_style           = "swing"
        initial_stop_pct     = 0.30
        tp_pct               = 0.60
        trail_after_pct      = 0.40
    else:
        delta_min, delta_max = DELTA_MIN_INTRADAY, DELTA_MAX_INTRADAY
        dte_min, dte_max     = DTE_MIN_INTRADAY, DTE_MAX_INTRADAY
        risk_style           = "intraday"
        initial_stop_pct     = 0.30
        tp_pct               = 0.60
        trail_after_pct      = 0.40

    # Relative position flags (helps LLM)
    above_pdh = (S is not None and prev_day_high is not None and S > prev_day_high)
    below_pdl = (S is not None and prev_day_low  is not None and S < prev_day_low)
    above_pmh = (S is not None and pm_high is not None and S > pm_high) if pm_high is not None else None
    below_pml = (S is not None and pm_low  is not None and S < pm_low)  if pm_low  is not None else None
    vwap_dist = (S - vwap) if (S is not None and vwap is not None) else None

    features = {
        "S": S,
        "oi": oi,
        "vol": vol,
        "iv": cur_iv,
        "iv_rank": iv_rank,
        "rv20": rv20,
        "opt_bid": bid,
        "opt_ask": ask,
        "opt_mid": mid,
        "option_spread_pct": opt_spread_pct,
        "quote_age_sec": quote_age_sec,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega":  vega,
        "dte": days_to_exp,
        "atr": atr,
        "high20": high20,
        "low20": low20,
        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
        "premarket_high": pm_high,
        "premarket_low": pm_low,
        "vwap": vwap,
        "vwap_dist": vwap_dist,
        "above_pdh": above_pdh,
        "below_pdl": below_pdl,
        "above_pmh": above_pmh,
        "below_pml": below_pml,
        "sr_headroom_ok": sr_headroom_ok,
        "mtf_align": mtf_align,
        "em_1s": em_1s,
        "em_vs_be_ok": em_vs_be_ok,
        "regime_flag": regime_flag,
        "no_event_risk": True,
        "risk_plan": {
            "style": risk_style,
            "initial_stop_pct": initial_stop_pct,
            "take_profit_pct": tp_pct,
            "trail_after_pct": trail_after_pct,
        },
        "bands": {
            "delta_min": delta_min, "delta_max": delta_max,
            "dte_min": dte_min, "dte_max": dte_max,
        }
    }
    return features
    pass