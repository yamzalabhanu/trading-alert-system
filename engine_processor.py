# engine_processor.py
import os
import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple

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

# -----------------------------------------------------------------------------
# Provider policy (Polygon primary, Yahoo optional fallback)
# -----------------------------------------------------------------------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "polygon").strip().lower()  # polygon|hybrid
USE_YAHOO_FALLBACK = str(os.getenv("USE_YAHOO_FALLBACK", "0")).strip().lower() in ("1", "true", "yes", "on")


# =============================================================================
# Small TA helpers
# =============================================================================
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
    """
    Simple ATR(14) using daily bars with keys h/l/c, bars sorted asc.
    Returns None if insufficient data.
    """
    if not daily_bars or len(daily_bars) < 16:
        return None

    trs: List[float] = []
    prev_c: Optional[float] = None
    # Use last ~16 to compute 14 TRs safely
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
    # last 14 TRs
    return sum(trs[-14:]) / 14.0


# =============================================================================
# Yahoo fallback features (unchanged)
# =============================================================================
async def _fetch_yahoo_features(symbol: str) -> Dict[str, Any]:
    """Fallback market-data enricher using Yahoo 5m chart API."""
    cli = get_http_client()
    if cli is None:
        return {}

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": "5d",
        "interval": "5m",
        "includePrePost": "true",
        "events": "div,splits",
    }
    try:
        r = await cli.get(url, params=params, timeout=8.0)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        logger.warning("[features] yahoo fetch failed for %s: %r", symbol, e)
        return {}

    try:
        res = (js.get("chart") or {}).get("result") or []
        if not res:
            return {}

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

        if len(rows) < 30:
            return {}

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
            if (rr[0].hour > 9 or (rr[0].hour == 9 and rr[0].minute >= 30))
            and (rr[0].hour < 16)
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
            if (
                ema20 is not None
                and ema50 is not None
                and abs(ema20 - ema50) / max(last_px, 1e-9) > 0.002
            )
            else "choppy"
        )
        mtf_align = bool(
            ema20 is not None
            and ema50 is not None
            and ((last_px > ema20 > ema50) or (last_px < ema20 < ema50))
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
        logger.warning("[features] parse error for %s: %r", symbol, e)
        return {}


# =============================================================================
# Polygon primary features (+ MTF summaries)
# =============================================================================
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

    # Helper: if PolygonClient doesn't yet have get_aggs_window(), fallback to get_aggregates for minute/hour/day.
    async def _aggs_window(sym: str, multiplier: int, timespan: str, from_: str, to: str, limit: int) -> List[Dict[str, Any]]:
        # Prefer native method if available
        if hasattr(pc, "get_aggs_window"):
            return await pc.get_aggs_window(sym, multiplier=multiplier, timespan=timespan, from_=from_, to=to, limit=limit)
        # Fallback: polygon_client.get_aggregates likely uses a wide range internally; keep as backup
        # Only safe for minute timespans in your current implementation.
        if timespan in ("minute", "hour", "day"):
            # Best-effort: minute/hour/day with a limit (may not honor from/to)
            return await pc.get_aggregates(sym, multiplier=multiplier, timespan=timespan, limit=limit)
        return []

    try:
        snap_task = pc.get_stock_snapshot(symbol)
        quote_task = pc.get_last_quote(symbol)
        trade_task = pc.get_last_trade(symbol)
        agg5m_task = pc.get_aggregates(symbol, multiplier=5, timespan="minute", limit=600)
        tech_task = pc.get_technicals_bundle(symbol)
        opt_task = pc.get_targeted_option_context(symbol, expiry_iso=expiry_iso, side=side, strike=strike)

        # Multi-timeframe context
        today = date.today()
        to_iso = (today + timedelta(days=2)).isoformat()
        daily_task = _aggs_window(symbol, 1, "day", (today - timedelta(days=220)).isoformat(), to_iso, limit=260)
        h1_task = _aggs_window(symbol, 1, "hour", (today - timedelta(days=30)).isoformat(), to_iso, limit=800)
        m15_task = _aggs_window(symbol, 15, "minute", (today - timedelta(days=10)).isoformat(), to_iso, limit=1200)

        stock_snap, last_quote, last_trade, aggs5m, techs, opt_ctx, daily_bars, h1_bars, m15_bars = await asyncio.gather(
            snap_task,
            quote_task,
            trade_task,
            agg5m_task,
            tech_task,
            opt_task,
            daily_task,
            h1_task,
            m15_task,
        )
    except Exception as e:
        logger.warning("[features] polygon fetch failed for %s: %r", symbol, e)
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

    q = stock_snap.get("lastQuote") or {}
    d = stock_snap.get("day") or {}
    pd = stock_snap.get("prevDay") or {}

    bid = q.get("p") or last_quote.get("p")
    ask = q.get("P") or last_quote.get("P")

    if isinstance(bid, (int, float)):
        out["bid"] = float(bid)
    if isinstance(ask, (int, float)):
        out["ask"] = float(ask)
    if isinstance(out.get("bid"), float) and isinstance(out.get("ask"), float):
        out["mid"] = (out["bid"] + out["ask"]) / 2.0

    last_trade_px = (stock_snap.get("lastTrade") or {}).get("p") or last_trade.get("p")
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

    # 5m bars (execution context + VWAP/ORB)
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
                tps = [
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

    # Indicator bundle
    if isinstance(techs, dict):
        out.update({k: v for k, v in techs.items() if v is not None})

    # Daily ATR/regime context (high-value for scalping risk + expectation)
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
    # Keep MTF payload light: last close only
    out["mtf"] = {
        "daily_last": _last_close(daily_bars or []),
        "h1_last": _last_close(h1_bars or []),
        "m15_last": _last_close(m15_bars or []),
        "m5_last": _last_close(bars or []),
    }

    # Regime + alignment
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
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    # Options context (if provided)
    if opt_ctx and isinstance(opt_ctx, dict):
        out.update({k: v for k, v in opt_ctx.items() if v is not None})
        out["nbbo_provider"] = "polygon:options_snapshot"

    return out


# =============================================================================
# Unified features fetch (Polygon → optional Yahoo fallback)
# =============================================================================
async def _fetch_equity_features(
    symbol: str,
    *,
    expiry_iso: Optional[str] = None,
    side: Optional[str] = None,
    strike: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Provider policy:
      - polygon (default): require Polygon; no silent Yahoo drift
      - hybrid: polygon first, optional Yahoo fallback if USE_YAHOO_FALLBACK=1
    """
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly

    if DATA_PROVIDER == "hybrid" and USE_YAHOO_FALLBACK:
        return await _fetch_yahoo_features(symbol)

    return {}


# =============================================================================
# Worker entrypoint
# =============================================================================
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

    expiry = alert.get("expiry")
    dte: Optional[int] = None
    if expiry:
        try:
            dte = (datetime.fromisoformat(expiry).date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    # Baseline from alert, then enrich
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
    }

    live = await _fetch_equity_features(
        str(alert.get("symbol") or "").upper(),
        expiry_iso=alert.get("expiry"),
        side=alert.get("side"),
        strike=alert.get("strike"),
    )
    for k, v in live.items():
        if v is not None:
            f[k] = v

    pf_ok, pf_checks = preflight_ok(f)

    # LLM analysis (include MTF summaries in f; already included)
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

    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")

    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    # Telegram
    try:
        src = str(f.get("ta_src") or "unknown")
        provider = str(f.get("data_provider") or "unknown")
        if live:
            data_note = f"📊 Data: {provider} ({src})"
        else:
            data_note = "⚠️ Live data unavailable; using alert baseline"

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

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            logger.info("[tg] send attempt: chat_id=%s len=%d decision=%s",
                        str(TELEGRAM_CHAT_ID), len(tg_text or ""), str(llm.get("decision")))
            await send_telegram(tg_text)
            logger.info("[tg] send done")
        else:
            logger.warning("[tg] missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID (token=%s chat_id=%s)",
                           bool(TELEGRAM_BOT_TOKEN), bool(TELEGRAM_CHAT_ID))
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
            "decision_final": decision_final,
            "llm": llm,
            "features": f,
            "preflight_ok": pf_ok,
            "preflight_checks": pf_checks,
            "ibkr": {"enabled": False, "attempted": False, "result": None},
        }
    )


def _append_decision_log(entry: Dict[str, Any]) -> None:
    """Append decision entry to in-memory log with local safety guard."""
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
