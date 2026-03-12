# engine_processor.py
import os
import asyncio
import logging
import math
import time
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple, Awaitable, Callable

from engine_runtime import get_http_client
from engine_common import (
    market_now,
    consume_llm,
    parse_alert_text,
    preflight_ok,
    compose_telegram_text,
)
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from scoring import compute_decision_score, map_score_to_rating
from daily_reporter import log_alert_snapshot
from polygon_client import PolygonClient, polygon_enabled

try:
    from reporting import _DECISIONS_LOG
except Exception:
    _DECISIONS_LOG = []

logger = logging.getLogger("trading_engine")
NY_TZ = ZoneInfo("America/New_York")

DATA_PROVIDER = os.getenv("DATA_PROVIDER", "polygon").strip().lower()  # polygon|hybrid
USE_YAHOO_FALLBACK = str(os.getenv("USE_YAHOO_FALLBACK", "0")).strip().lower() in ("1", "true", "yes", "on")
MIN_TV_SCORE_TO_SEND = float(os.getenv("MIN_TV_SCORE_TO_SEND", "75"))


# ---------------------------------------------------------------------
# Engine-level micro cache (reduces repeated calls per symbol)
# ---------------------------------------------------------------------
_ENGINE_CACHE: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    it = _ENGINE_CACHE.get(key)
    if not it:
        return None
    exp, val = it
    if exp <= time.time():
        _ENGINE_CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: Any, ttl_s: float) -> Any:
    _ENGINE_CACHE[key] = (time.time() + ttl_s, val)
    return val


async def _cached(key: str, ttl_s: float, coro_factory: Callable[[], Awaitable[Any]]) -> Any:
    hit = _cache_get(key)
    if hit is not None:
        return hit
    val = await coro_factory()
    return _cache_set(key, val, ttl_s)


# ---------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------
def _ema_last(vals: List[float], period: int) -> Optional[float]:
    if len(vals) < period:
        return None
    k = 2.0 / (period + 1.0)
    ema = sum(vals[:period]) / period
    for v in vals[period:]:
        ema = (v * k) + (ema * (1.0 - k))
    return float(ema)


def _ema_series(vals: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(vals)
    if len(vals) < period:
        return out
    k = 2.0 / (period + 1.0)
    ema = sum(vals[:period]) / period
    out[period - 1] = float(ema)
    for i in range(period, len(vals)):
        ema = (vals[i] * k) + (ema * (1.0 - k))
        out[i] = float(ema)
    return out


def _rsi_last(vals: List[float], period: int = 14) -> Optional[float]:
    if len(vals) <= period:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        d = vals[i] - vals[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(vals)):
        d = vals[i] - vals[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _stddev(vals: List[float]) -> Optional[float]:
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return math.sqrt(var)


def _safe_pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None or b == 0) else ((a - b) / b) * 100.0


def _last_close(bars: List[Dict[str, Any]]) -> Optional[float]:
    for b in reversed(bars or []):
        c = b.get("c")
        if isinstance(c, (int, float)):
            return float(c)
    return None


def _atr14_from_daily(daily_bars: List[Dict[str, Any]]) -> Optional[float]:
    if not daily_bars or len(daily_bars) < 16:
        return None
    trs: List[float] = []
    prev_c: Optional[float] = None
    for b in daily_bars[-16:]:
        h = b.get("h")
        l = b.get("l")
        c = b.get("c")
        if not all(isinstance(x, (int, float)) for x in (h, l, c)):
            continue
        h = float(h)
        l = float(l)
        c = float(c)
        if prev_c is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        prev_c = c
    if len(trs) < 15:
        return None
    return sum(trs[-14:]) / 14.0


def _extract_tv_score(alert: Dict[str, Any], llm: Dict[str, Any]) -> float:
    candidates = [
        alert.get("tvScore"),
        alert.get("tv_score"),
        alert.get("TVScore"),
        alert.get("score_tv"),
        alert.get("tvscore"),
        llm.get("tvScore") if isinstance(llm, dict) else None,
        llm.get("tv_score") if isinstance(llm, dict) else None,
    ]
    for val in candidates:
        try:
            if val is not None:
                return float(val)
        except Exception:
            pass
    return 0.0


def _should_send_telegram_alert(*, live_data_available: bool, tv_score: float) -> Tuple[bool, str]:
    if not live_data_available:
        return False, "Live Polygon data unavailable"
    if tv_score < MIN_TV_SCORE_TO_SEND:
        return False, f"TV score {tv_score:.2f} below minimum {MIN_TV_SCORE_TO_SEND:.2f}"
    return True, "Eligible to send"


# ---------------------------------------------------------------------
# Yahoo chart helpers (NEW)
# ---------------------------------------------------------------------
async def _fetch_yahoo_chart_payload(symbol: str) -> Dict[str, Any]:
    cli = get_http_client()
    if cli is None:
        return {}

    sym = (symbol or "").upper().strip()
    if not sym:
        return {}

    async def _do_fetch() -> Dict[str, Any]:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        params = {
            "range": "5d",
            "interval": "5m",
            "includePrePost": "true",
            "events": "div,splits",
        }
        try:
            r = await cli.get(url, params=params, timeout=8.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("[yahoo] chart fetch failed for %s: %r", sym, e)
            return {}

    return await _cached(f"yahoo:chart_payload:{sym}", 30.0, _do_fetch)


def _parse_yahoo_rows(js: Dict[str, Any]) -> List[Tuple[datetime, float, float, float, float, float]]:
    try:
        res = (js.get("chart") or {}).get("result") or []
        if not res:
            return []

        node = res[0]
        ts = node.get("timestamp") or []
        q = ((node.get("indicators") or {}).get("quote") or [{}])[0]
        closes_raw = q.get("close") or []
        highs_raw = q.get("high") or []
        lows_raw = q.get("low") or []
        opens_raw = q.get("open") or []
        vols_raw = q.get("volume") or []

        rows: List[Tuple[datetime, float, float, float, float, float]] = []
        for t, o, h, l, c, v in zip(ts, opens_raw, highs_raw, lows_raw, closes_raw, vols_raw):
            if c is None or h is None or l is None or o is None:
                continue
            dt = datetime.fromtimestamp(int(t), tz=timezone.utc).astimezone(NY_TZ)
            rows.append((dt, float(o), float(h), float(l), float(c), float(v or 0.0)))
        return rows
    except Exception as e:
        logger.warning("[yahoo] row parse failed: %r", e)
        return []


def _build_yahoo_intraday_context(
    symbol: str,
    rows: List[Tuple[datetime, float, float, float, float, float]],
) -> Dict[str, Any]:
    if len(rows) < 30:
        return {}

    closes = [r[4] for r in rows]
    highs = [r[2] for r in rows]
    lows = [r[3] for r in rows]
    vols = [r[5] for r in rows]

    day_key = rows[-1][0].date()
    day_rows = [r for r in rows if r[0].date() == day_key]
    prev_rows = [r for r in rows if r[0].date() < day_key]

    if not day_rows:
        return {}

    last_px = day_rows[-1][4]
    day_open = day_rows[0][1]
    day_high = max(r[2] for r in day_rows)
    day_low = min(r[3] for r in day_rows)
    day_vol = sum(r[5] for r in day_rows)

    rth = [
        r for r in day_rows
        if (r[0].hour > 9 or (r[0].hour == 9 and r[0].minute >= 30))
        and (r[0].hour < 16)
    ]
    pm = [
        r for r in day_rows
        if (r[0].hour > 4 or (r[0].hour == 4 and r[0].minute >= 0))
        and (r[0].hour < 9 or (r[0].hour == 9 and r[0].minute < 30))
    ]

    prev_day = prev_rows[-78:] if prev_rows else []
    prev_close = prev_day[-1][4] if prev_day else None
    prev_high = max((r[2] for r in prev_day), default=None)
    prev_low = min((r[3] for r in prev_day), default=None)

    ema9 = _ema_last(closes, 9)
    ema20 = _ema_last(closes, 20)
    ema50 = _ema_last(closes, 50)
    rsi14 = _rsi_last(closes, 14)

    vwap = None
    if rth:
        try:
            vwap_num = sum((((r[2] + r[3] + r[4]) / 3.0) * r[5]) for r in rth)
            vwap_den = sum(r[5] for r in rth)
            if vwap_den > 0:
                vwap = vwap_num / vwap_den
        except Exception:
            vwap = None

    last3 = rth[-3:] if len(rth) >= 3 else day_rows[-3:]
    last3_dir = []
    for r in last3:
        o, c = r[1], r[4]
        if c > o:
            last3_dir.append("up")
        elif c < o:
            last3_dir.append("down")
        else:
            last3_dir.append("flat")

    green_count = sum(1 for d in last3_dir if d == "up")
    red_count = sum(1 for d in last3_dir if d == "down")
    if green_count >= 2:
        last3_bias = "bullish"
    elif red_count >= 2:
        last3_bias = "bearish"
    else:
        last3_bias = "mixed"

    pm_high = max((r[2] for r in pm), default=None)
    pm_low = min((r[3] for r in pm), default=None)

    if len(rth) >= 3:
        orb15 = rth[:3]
        orb15_high = max(r[2] for r in orb15)
        orb15_low = min(r[3] for r in orb15)
    else:
        orb15_high = None
        orb15_low = None

    five_day_high = max(highs) if highs else None
    five_day_low = min(lows) if lows else None
    five_day_return_pct = _safe_pct(closes[-1], closes[0]) if len(closes) >= 2 else None
    day_change_pct = _safe_pct(last_px, day_open)
    gap_pct = _safe_pct(day_open, prev_close) if prev_close is not None else None
    vwap_dist_pct = _safe_pct(last_px, vwap) if vwap is not None else None

    intraday_bias = "neutral"
    if vwap is not None and ema9 is not None and ema20 is not None:
        if last_px > vwap and ema9 > ema20:
            intraday_bias = "bullish"
        elif last_px < vwap and ema9 < ema20:
            intraday_bias = "bearish"

    breakout_state = "inside"
    if orb15_high is not None and last_px > orb15_high:
        breakout_state = "above_orb15"
    elif orb15_low is not None and last_px < orb15_low:
        breakout_state = "below_orb15"

    sr_state = "inside_prev_day"
    if prev_high is not None and last_px > prev_high:
        sr_state = "above_prev_high"
    elif prev_low is not None and last_px < prev_low:
        sr_state = "below_prev_low"

    summary_parts = [
        f"{symbol} Yahoo 5d/5m context",
        f"intraday_bias={intraday_bias}",
        f"last3_candles={last3_bias}",
    ]
    if vwap is not None:
        summary_parts.append(f"vs_vwap={vwap_dist_pct:.2f}%")
    if day_change_pct is not None:
        summary_parts.append(f"day_change={day_change_pct:.2f}%")
    if gap_pct is not None:
        summary_parts.append(f"gap={gap_pct:.2f}%")
    if five_day_return_pct is not None:
        summary_parts.append(f"5d_return={five_day_return_pct:.2f}%")
    summary_parts.append(f"orb_state={breakout_state}")
    summary_parts.append(f"sr_state={sr_state}")

    return {
        "yahoo_ctx_available": True,
        "yahoo_symbol": symbol,
        "yahoo_chart_range": "5d",
        "yahoo_chart_interval": "5m",
        "yahoo_last": last_px,
        "yahoo_day_open": day_open,
        "yahoo_day_high": day_high,
        "yahoo_day_low": day_low,
        "yahoo_day_volume": day_vol,
        "yahoo_prev_close": prev_close,
        "yahoo_prev_high": prev_high,
        "yahoo_prev_low": prev_low,
        "yahoo_premarket_high": pm_high,
        "yahoo_premarket_low": pm_low,
        "yahoo_orb15_high": orb15_high,
        "yahoo_orb15_low": orb15_low,
        "yahoo_ema9": ema9,
        "yahoo_ema20": ema20,
        "yahoo_ema50": ema50,
        "yahoo_rsi14": rsi14,
        "yahoo_vwap": vwap,
        "yahoo_vwap_dist_pct": vwap_dist_pct,
        "yahoo_intraday_bias": intraday_bias,
        "yahoo_last3_candle_bias": last3_bias,
        "yahoo_last3_sequence": ",".join(last3_dir) if last3_dir else None,
        "yahoo_day_change_pct": day_change_pct,
        "yahoo_gap_pct": gap_pct,
        "yahoo_5d_high": five_day_high,
        "yahoo_5d_low": five_day_low,
        "yahoo_5d_return_pct": five_day_return_pct,
        "yahoo_orb_break_state": breakout_state,
        "yahoo_prev_day_sr_state": sr_state,
        "yahoo_llm_summary": " | ".join(summary_parts),
    }


async def _fetch_yahoo_chart_context(symbol: str) -> Dict[str, Any]:
    js = await _fetch_yahoo_chart_payload(symbol)
    if not js:
        return {}
    rows = _parse_yahoo_rows(js)
    if not rows:
        return {}
    return _build_yahoo_intraday_context(symbol, rows)


# ---------------------------------------------------------------------
# Yahoo fallback (generic market features)
# ---------------------------------------------------------------------
async def _fetch_yahoo_features(symbol: str) -> Dict[str, Any]:
    js = await _fetch_yahoo_chart_payload(symbol)
    if not js:
        return {}

    rows = _parse_yahoo_rows(js)
    if len(rows) < 30:
        return {}

    try:
        closes = [rr[4] for rr in rows]
        vols = [rr[5] for rr in rows]
        last_px = closes[-1]

        ema20 = _ema_last(closes, 20)
        ema50 = _ema_last(closes, 50)
        ema200 = _ema_last(closes, 200)
        rsi14 = _rsi_last(closes, 14)

        ema12_s = _ema_series(closes, 12)
        ema26_s = _ema_series(closes, 26)
        macd_series: List[float] = []
        for a, b in zip(ema12_s, ema26_s):
            if a is not None and b is not None:
                macd_series.append(a - b)
        macd_line = macd_series[-1] if macd_series else None
        macd_signal = _ema_last(macd_series, 9) if macd_series else None
        macd_hist = (macd_line - macd_signal) if (macd_line is not None and macd_signal is not None) else None

        sma20 = sum(closes[-20:]) / 20.0 if len(closes) >= 20 else None
        st = _stddev(closes[-20:]) if len(closes) >= 20 else None
        bb_upper = (sma20 + 2 * st) if (sma20 is not None and st is not None) else None
        bb_lower = (sma20 - 2 * st) if (sma20 is not None and st is not None) else None

        day_key = rows[-1][0].date()
        day_rows = [rr for rr in rows if rr[0].date() == day_key]
        prev_rows = [rr for rr in rows if rr[0].date() < day_key]
        prev_day = prev_rows[-78:] if prev_rows else []

        prev_high = max((rr[2] for rr in prev_day), default=None)
        prev_low = min((rr[3] for rr in prev_day), default=None)
        prev_close = prev_day[-1][4] if prev_day else None
        prev_open = prev_day[0][1] if prev_day else None

        pm = [
            rr
            for rr in day_rows
            if (rr[0].hour > 4 or (rr[0].hour == 4 and rr[0].minute >= 0))
            and (rr[0].hour < 9 or (rr[0].hour == 9 and rr[0].minute < 30))
        ]
        pm_high = max((rr[2] for rr in pm), default=None)
        pm_low = min((rr[3] for rr in pm), default=None)

        rth = [
            rr
            for rr in day_rows
            if (rr[0].hour > 9 or (rr[0].hour == 9 and rr[0].minute >= 30)) and (rr[0].hour < 16)
        ]
        orb15 = [rr for rr in rth if (rr[0].hour == 9 and rr[0].minute < 45)]
        orb15_high = max((rr[2] for rr in orb15), default=None)
        orb15_low = min((rr[3] for rr in orb15), default=None)

        vwap_num = sum((((rr[2] + rr[3] + rr[4]) / 3.0) * rr[5]) for rr in rth)
        vwap_den = sum(rr[5] for rr in rth)
        vwap = (vwap_num / vwap_den) if vwap_den > 0 else None

        vol_now = day_rows[-1][5] if day_rows else vols[-1]
        qchg = _safe_pct(last_px, prev_close)

        regime_flag = (
            "trending"
            if (ema20 is not None and ema50 is not None and abs(ema20 - ema50) / max(last_px, 1e-9) > 0.002)
            else "choppy"
        )

        mtf_align = bool(
            ema20 is not None and ema50 is not None and ((last_px > ema20 > ema50) or (last_px < ema20 < ema50))
        )

        return {
            "last": last_px,
            "mid": last_px,
            "rsi14": rsi14,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "sma20": sma20,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "vwap": vwap,
            "vwap_dist": _safe_pct(last_px, vwap),
            "orb15_high": orb15_high,
            "orb15_low": orb15_low,
            "prev_open": prev_open,
            "prev_high": prev_high,
            "prev_low": prev_low,
            "prev_close": prev_close,
            "premarket_high": pm_high,
            "premarket_low": pm_low,
            "quote_change_pct": qchg,
            "vol": vol_now,
            "oi": None,
            "option_spread_pct": None,
            "quote_age_sec": 0,
            "nbbo_provider": "yahoo:equity",
            "em_vs_be_ok": True,
            "mtf_align": mtf_align,
            "sr_headroom_ok": True,
            "regime_flag": regime_flag,
            "ta_src": "yahoo_chart_5m",
            "synthetic_nbbo_used": True,
            "data_provider": "yahoo",
        }
    except Exception as e:
        logger.warning("[features] yahoo parse error for %s: %r", symbol, e)
        return {}


# ---------------------------------------------------------------------
# Polygon features
# ---------------------------------------------------------------------
async def _fetch_polygon_features(
    symbol: str,
    *,
    expiry_iso: Optional[str],
    side: Optional[str],
    strike: Optional[float],
) -> Dict[str, Any]:
    cli = get_http_client()
    if cli is None:
        return {}

    pc = PolygonClient(cli)
    if not pc.enabled:
        return {}

    sym = (symbol or "").upper()

    async def _empty_dict() -> Dict[str, Any]:
        return {}

    async def _safe_call(name: str, coro: Awaitable[Any], default: Any) -> Any:
        try:
            return await coro
        except Exception as e:
            logger.warning("[features] polygon subcall failed (%s) for %s: %r", name, sym, e)
            return default

    today = date.today()
    to_iso = (today + timedelta(days=2)).isoformat()

    from_5m = (today - timedelta(days=10)).isoformat()
    from_15m = (today - timedelta(days=18)).isoformat()
    from_h1 = (today - timedelta(days=40)).isoformat()
    from_d = (today - timedelta(days=320)).isoformat()

    k_snap = f"poly:snap:{sym}"
    k_quote = f"poly:quote:{sym}"
    k_trade = f"poly:trade:{sym}"
    k_aggs5m = f"poly:aggs5m:{sym}:{from_5m}:{to_iso}"
    k_aggs15m = f"poly:aggs15m:{sym}:{from_15m}:{to_iso}"
    k_aggsh1 = f"poly:aggsh1:{sym}:{from_h1}:{to_iso}"
    k_aggsd = f"poly:aggsd:{sym}:{from_d}:{to_iso}"
    k_techm = f"poly:techm:{sym}"
    k_techd = f"poly:techd:{sym}"

    TTL_SNAP = 10.0
    TTL_QUOTE = 5.0
    TTL_TRADE = 5.0
    TTL_AGGS_5M = 20.0
    TTL_AGGS_15M = 60.0
    TTL_AGGS_H1 = 300.0
    TTL_AGGS_D = 900.0
    TTL_TECH_M = 60.0
    TTL_TECH_D = 900.0

    snap_task = _safe_call("snapshot", _cached(k_snap, TTL_SNAP, lambda: pc.get_stock_snapshot(sym)), {})
    quote_task = _safe_call("last_quote", _cached(k_quote, TTL_QUOTE, lambda: pc.get_last_quote(sym)), {})
    trade_task = _safe_call("last_trade", _cached(k_trade, TTL_TRADE, lambda: pc.get_last_trade(sym)), {})

    agg5m_task = _safe_call(
        "aggs_5m",
        _cached(
            k_aggs5m,
            TTL_AGGS_5M,
            lambda: pc.get_aggs_window(
                sym,
                multiplier=5,
                timespan="minute",
                from_=from_5m,
                to=to_iso,
                limit=600,
                cache_ttl_s=20.0,
            ),
        ),
        [],
    )
    m15_task = _safe_call(
        "aggs_m15",
        _cached(
            k_aggs15m,
            TTL_AGGS_15M,
            lambda: pc.get_aggs_window(
                sym,
                multiplier=15,
                timespan="minute",
                from_=from_15m,
                to=to_iso,
                limit=1200,
                cache_ttl_s=60.0,
            ),
        ),
        [],
    )
    h1_task = _safe_call(
        "aggs_h1",
        _cached(
            k_aggsh1,
            TTL_AGGS_H1,
            lambda: pc.get_aggs_window(
                sym,
                multiplier=1,
                timespan="hour",
                from_=from_h1,
                to=to_iso,
                limit=800,
                cache_ttl_s=300.0,
            ),
        ),
        [],
    )
    daily_task = _safe_call(
        "aggs_daily",
        _cached(
            k_aggsd,
            TTL_AGGS_D,
            lambda: pc.get_aggs_window(
                sym,
                multiplier=1,
                timespan="day",
                from_=from_d,
                to=to_iso,
                limit=260,
                cache_ttl_s=900.0,
            ),
        ),
        [],
    )

    tech_task = _safe_call(
        "techs_m",
        _cached(k_techm, TTL_TECH_M, lambda: pc.get_technicals_bundle(sym, timespan="minute")),
        {},
    )
    tech_d_task = _safe_call(
        "techs_d",
        _cached(k_techd, TTL_TECH_D, lambda: pc.get_technicals_daily_bundle(sym)),
        {},
    )

    if expiry_iso and side and (strike is not None):
        k_opt = f"poly:optctx:{sym}:{expiry_iso}:{side}:{strike}"
        opt_task: Awaitable[Dict[str, Any]] = _safe_call(
            "opt_ctx",
            _cached(
                k_opt,
                10.0,
                lambda: pc.get_targeted_option_context(sym, expiry_iso=expiry_iso, side=side, strike=strike),
            ),
            {},
        )
    else:
        opt_task = _empty_dict()

    stock_snap, last_quote, last_trade, aggs5m, techs, techs_d, opt_ctx, daily_bars, h1_bars, m15_bars = await asyncio.gather(
        snap_task,
        quote_task,
        trade_task,
        agg5m_task,
        tech_task,
        tech_d_task,
        opt_task,
        daily_task,
        h1_task,
        m15_task,
    )

    if not any([stock_snap, last_quote, last_trade, aggs5m, techs, techs_d, opt_ctx, daily_bars, h1_bars, m15_bars]):
        return {}

    out: Dict[str, Any] = {
        "ta_src": "polygon_rest",
        "data_provider": "polygon",
        "nbbo_provider": "polygon:stocks",
        "synthetic_nbbo_used": False,
        "em_vs_be_ok": True,
        "sr_headroom_ok": True,
        "mtf_align": True,
    }

    q = (stock_snap or {}).get("lastQuote") or {}
    d = (stock_snap or {}).get("day") or {}
    pd = (stock_snap or {}).get("prevDay") or {}

    bid = q.get("p") or (last_quote or {}).get("p")
    ask = q.get("P") or (last_quote or {}).get("P")

    if isinstance(bid, (int, float)):
        out["bid"] = float(bid)
    if isinstance(ask, (int, float)):
        out["ask"] = float(ask)
    if isinstance(out.get("bid"), float) and isinstance(out.get("ask"), float):
        out["mid"] = (out["bid"] + out["ask"]) / 2.0

    last_trade_px = ((stock_snap or {}).get("lastTrade") or {}).get("p") or (last_trade or {}).get("p")
    if isinstance(last_trade_px, (int, float)):
        out["last"] = float(last_trade_px)

    if isinstance(d.get("v"), (int, float)):
        out["vol"] = float(d.get("v"))
    if isinstance(pd.get("h"), (int, float)):
        out["prev_high"] = float(pd.get("h"))
    if isinstance(pd.get("l"), (int, float)):
        out["prev_low"] = float(pd.get("l"))
    if isinstance(pd.get("c"), (int, float)):
        out["prev_close"] = float(pd.get("c"))

    bars = aggs5m or []
    if bars:
        closes = [float(x.get("c")) for x in bars if isinstance(x.get("c"), (int, float))]
        highs = [float(x.get("h")) for x in bars if isinstance(x.get("h"), (int, float))]
        lows = [float(x.get("l")) for x in bars if isinstance(x.get("l"), (int, float))]
        vols = [float(x.get("v") or 0.0) for x in bars]

        if closes:
            out.setdefault("last", closes[-1])
            if len(closes) >= 15:
                out["orb15_high"] = max(highs[:3]) if len(highs) >= 3 else None
                out["orb15_low"] = min(lows[:3]) if len(lows) >= 3 else None

            if out.get("prev_close"):
                out["quote_change_pct"] = _safe_pct(out.get("last"), out.get("prev_close"))

            if len(vols) >= 20:
                out["vol_avg20"] = sum(vols[-20:]) / 20.0

            try:
                tps: List[Tuple[float, float]] = [
                    (
                        (float(x.get("h")) + float(x.get("l")) + float(x.get("c"))) / 3.0,
                        float(x.get("v") or 0.0),
                    )
                    for x in bars
                    if all(isinstance(x.get(k), (int, float)) for k in ("h", "l", "c"))
                ]
                den = sum(v for _, v in tps)
                if den > 0:
                    out["vwap"] = sum(tp * v for tp, v in tps) / den
                    out["vwap_dist"] = _safe_pct(out.get("last"), out.get("vwap"))
            except Exception:
                pass

    if isinstance(techs, dict) and techs:
        out.update({k: v for k, v in techs.items() if v is not None})

    if isinstance(techs_d, dict) and techs_d:
        out.update({k: v for k, v in techs_d.items() if v is not None})

    try:
        lp = float(out["last"]) if out.get("last") is not None else None
        ema20_d = out.get("ema20_d")
        ema50_d = out.get("ema50_d")
        if lp is not None and isinstance(ema20_d, (int, float)) and isinstance(ema50_d, (int, float)):
            if lp > float(ema20_d) > float(ema50_d):
                out["daily_trend_bias"] = "bull"
            elif lp < float(ema20_d) < float(ema50_d):
                out["daily_trend_bias"] = "bear"
            else:
                out["daily_trend_bias"] = "neutral"
    except Exception:
        pass

    atr14 = _atr14_from_daily(daily_bars or [])
    if atr14 is not None:
        out["atr14_daily"] = atr14
        if out.get("last") is not None:
            out["atr14_pct"] = (atr14 / max(float(out["last"]), 1e-9)) * 100.0

    out["bars_meta"] = {
        "daily_n": len(daily_bars or []),
        "h1_n": len(h1_bars or []),
        "m15_n": len(m15_bars or []),
        "m5_n": len(bars or []),
    }
    out["mtf"] = {
        "daily_last": _last_close(daily_bars or []),
        "h1_last": _last_close(h1_bars or []),
        "m15_last": _last_close(m15_bars or []),
        "m5_last": _last_close(bars or []),
    }

    out["regime_flag"] = (
        "trending"
        if (
            out.get("ema20") is not None
            and out.get("ema50") is not None
            and out.get("last") is not None
            and abs(out["ema20"] - out["ema50"]) / max(float(out["last"]), 1e-9) > 0.002
        )
        else "choppy"
    )
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp2 = float(out["last"])
        out["mtf_align"] = bool((lp2 > out["ema20"] > out["ema50"]) or (lp2 < out["ema20"] < out["ema50"]))

    if opt_ctx and isinstance(opt_ctx, dict) and opt_ctx:
        out.update({k: v for k, v in opt_ctx.items() if v is not None})
        out["nbbo_provider"] = "polygon:options_snapshot"

    return out


async def _fetch_equity_features(
    symbol: str,
    *,
    expiry_iso: Optional[str] = None,
    side: Optional[str] = None,
    strike: Optional[float] = None,
) -> Dict[str, Any]:
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    if DATA_PROVIDER == "hybrid" and USE_YAHOO_FALLBACK:
        return await _fetch_yahoo_features(symbol)
    return {}


async def process_tradingview_job(job: Dict[str, Any]) -> None:
    client = get_http_client()
    if client is None:
        logger.warning("[worker] HTTP client not ready")
        return

    try:
        alert = parse_alert_text(job["alert_text"])
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    symbol = str(alert.get("symbol") or "").upper()

    tv_meta_keys = [
        "source",
        "model",
        "confirm_tf",
        "chart_tf",
        "event",
        "reason",
        "exchange",
        "level",
        "ats",
        "bp",
        "tp1",
        "tp2",
        "tp3",
        "trail",
        "relvol",
        "relVol",
        "chop",
        "fast_stop",
        "ason",
        "adx",
        "tvScore",
        "tv_score",
        "TVScore",
    ]
    tv_meta: Dict[str, Any] = {k: alert.get(k) for k in tv_meta_keys if alert.get(k) is not None}

    expiry = alert.get("expiry")
    dte: Optional[int] = None
    if expiry:
        try:
            dte = (datetime.fromisoformat(expiry).date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    f: Dict[str, Any] = {
        "dte": dte,
        "last": alert.get("underlying_price_from_alert"),
        "mid": alert.get("underlying_price_from_alert"),
        "option_spread_pct": None,
        "quote_age_sec": 0,
        "vol": None,
        "oi": None,
        "em_vs_be_ok": True,
        "mtf_align": True,
        "sr_headroom_ok": True,
        "nbbo_provider": "disabled",
        "tv_meta": tv_meta,
    }

    live, yahoo_ctx = await asyncio.gather(
        _fetch_equity_features(
            symbol,
            expiry_iso=alert.get("expiry"),
            side=alert.get("side"),
            strike=alert.get("strike"),
        ),
        _fetch_yahoo_chart_context(symbol),
    )

    for k, v in live.items():
        if v is not None:
            f[k] = v

    for k, v in yahoo_ctx.items():
        if v is not None:
            f[k] = v

    pf_ok, pf_checks = preflight_ok(f)

    try:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
    except Exception as e:
        llm = {
            "decision": "wait",
            "confidence": 0.0,
            "reason": f"LLM error: {e}",
            "checklist": {"preflight": pf_checks},
            "ev_estimate": {},
        }

    event = str(alert.get("event") or "").lower().strip()
    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")
    if event == "exit":
        decision_final = f"{decision_final}_exit"

    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    tv_score = _extract_tv_score(alert, llm)
    live_data_available = bool(live) and str(f.get("data_provider") or "").lower() == "polygon"

    try:
        src = str(f.get("ta_src") or "unknown")
        provider = str(f.get("data_provider") or "unknown")

        meta_bits = []
        if tv_meta.get("event"):
            meta_bits.append(f"event={tv_meta.get('event')}")
        if tv_meta.get("confirm_tf"):
            meta_bits.append(f"confirm={tv_meta.get('confirm_tf')}")
        if tv_meta.get("model"):
            meta_bits.append(f"model={tv_meta.get('model')}")
        if tv_meta.get("reason"):
            meta_bits.append(f"reason={tv_meta.get('reason')}")
        if tv_score > 0:
            meta_bits.append(f"tvScore={tv_score:.2f}")
        if f.get("yahoo_intraday_bias"):
            meta_bits.append(f"yahooBias={f.get('yahoo_intraday_bias')}")
        if f.get("yahoo_last3_candle_bias"):
            meta_bits.append(f"yahooLast3={f.get('yahoo_last3_candle_bias')}")

        meta_note = (" | " + ", ".join(meta_bits)) if meta_bits else ""

        if live_data_available:
            data_note = f"📊 Data: {provider} ({src}) + Yahoo chart context{meta_note}"
        else:
            data_note = f"⚠️ Live Polygon data unavailable; using alert baseline + Yahoo chart context{meta_note}"

        tg_text = compose_telegram_text(
            alert=alert,
            option_ticker=None,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",
            score=score,
            rating=rating,
            diff_note=data_note,
        )

        send_ok, skip_reason = _should_send_telegram_alert(
            live_data_available=live_data_available,
            tv_score=tv_score,
        )

        if not send_ok:
            logger.info(
                "[tg] skip: symbol=%s reason=%s live_data=%s provider=%s tv_score=%.2f threshold=%.2f",
                symbol,
                skip_reason,
                live_data_available,
                provider,
                tv_score,
                MIN_TV_SCORE_TO_SEND,
            )
        elif TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            logger.info(
                "[tg] send attempt: chat_id=%s len=%d decision=%s tv_score=%.2f yahoo_ctx=%s",
                str(TELEGRAM_CHAT_ID),
                len(tg_text or ""),
                str(llm.get("decision")),
                tv_score,
                bool(f.get("yahoo_ctx_available")),
            )
            await send_telegram(tg_text)
            logger.info("[tg] send done")
        else:
            logger.warning(
                "[tg] missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID (token=%s chat_id=%s)",
                bool(TELEGRAM_BOT_TOKEN),
                bool(TELEGRAM_CHAT_ID),
            )
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    try:
        log_alert_snapshot(alert, None, f)
    except Exception as e:
        logger.warning("[daily-report] log snapshot failed: %r", e)

    _append_decision_log(
        {
            "timestamp_local": market_now(),
            "symbol": alert.get("symbol"),
            "side": alert.get("side"),
            "event": alert.get("event"),
            "decision_final": decision_final,
            "llm": llm,
            "features": f,
            "preflight_ok": pf_ok,
            "preflight_checks": pf_checks,
            "ibkr": {"enabled": False, "attempted": False, "result": None},
            "tv_meta": tv_meta,
            "telegram_gate": {
                "live_data_available": live_data_available,
                "tv_score": tv_score,
                "min_tv_score_to_send": MIN_TV_SCORE_TO_SEND,
            },
        }
    )


def _append_decision_log(entry: Dict[str, Any]) -> None:
    try:
        _DECISIONS_LOG.append(entry)
    except Exception as e:
        logger.warning("decision log append failed: %r", e)


async def net_debug_info() -> Dict[str, Any]:
    md = "polygon_only" if (DATA_PROVIDER == "polygon") else f"hybrid(yahoo_fallback={USE_YAHOO_FALLBACK})"
    return {
        "integrations": {
            "ibkr": "disabled",
            "polygon": "enabled" if polygon_enabled() else "disabled",
            "perplexity": "disabled",
            "market_data": md,
        }
    }


__all__ = ["process_tradingview_job", "net_debug_info"]
