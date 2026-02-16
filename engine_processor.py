# engine_processor.py
import logging
import math
from datetime import datetime, timezone
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

        closes = [r[4] for r in rows]
        vols = [r[5] for r in rows]
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
        day_rows = [r for r in rows if r[0].date() == day_key]
        prev_rows = [r for r in rows if r[0].date() < day_key]
        prev_day = prev_rows[-78:] if prev_rows else []

        prev_high = max((r[2] for r in prev_day), default=None)
        prev_low = min((r[3] for r in prev_day), default=None)
        prev_close = prev_day[-1][4] if prev_day else None
        prev_open = prev_day[0][1] if prev_day else None

        pm = [r for r in day_rows if (r[0].hour > 4 or (r[0].hour == 4 and r[0].minute >= 0)) and (r[0].hour < 9 or (r[0].hour == 9 and r[0].minute < 30))]
        pm_high = max((r[2] for r in pm), default=None)
        pm_low = min((r[3] for r in pm), default=None)

        rth = [r for r in day_rows if (r[0].hour > 9 or (r[0].hour == 9 and r[0].minute >= 30)) and (r[0].hour < 16)]
        orb15 = [r for r in rth if (r[0].hour == 9 and r[0].minute < 45)]
        orb15_high = max((r[2] for r in orb15), default=None)
        orb15_low = min((r[3] for r in orb15), default=None)

        vwap_num = sum((((r[2] + r[3] + r[4]) / 3.0) * r[5]) for r in rth)
        vwap_den = sum(r[5] for r in rth)
        vwap = (vwap_num / vwap_den) if vwap_den > 0 else None

        vol_now = day_rows[-1][5] if day_rows else vols[-1]
        qchg = _safe_pct(last_px, prev_close)

        regime_flag = "trending" if (ema20 is not None and ema50 is not None and abs(ema20 - ema50) / max(last_px, 1e-9) > 0.002) else "choppy"
        mtf_align = bool(ema20 is not None and ema50 is not None and ((last_px > ema20 > ema50) or (last_px < ema20 < ema50)))

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


async def _fetch_polygon_features(symbol: str, *, expiry_iso: Optional[str], side: Optional[str], strike: Optional[float]) -> Dict[str, Any]:
    cli = get_http_client()
    if cli is None:
        return {}
    pc = PolygonClient(cli)
    if not pc.enabled:
        return {}
    try:
        snap_task = pc.get_stock_snapshot(symbol)
        quote_task = pc.get_last_quote(symbol)
        trade_task = pc.get_last_trade(symbol)
        agg_task = pc.get_aggregates(symbol, multiplier=5, timespan="minute", limit=600)
        tech_task = pc.get_technicals_bundle(symbol)
        opt_task = pc.get_targeted_option_context(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
        stock_snap, last_quote, last_trade, aggs, techs, opt_ctx = await asyncio.gather(
            snap_task, quote_task, trade_task, agg_task, tech_task, opt_task
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

    # Aggregates context (ORB + VWAP estimate)
    bars = aggs or []
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
                tps = [((float(x.get("h"))+float(x.get("l"))+float(x.get("c")))/3.0, float(x.get("v") or 0.0)) for x in bars if all(isinstance(x.get(k), (int, float)) for k in ("h","l","c"))]
                den = sum(v for _, v in tps)
                if den > 0:
                    out["vwap"] = sum(tp*v for tp, v in tps) / den
                    out["vwap_dist"] = _safe_pct(out.get("last"), out.get("vwap"))
            except Exception:
                pass

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

    out.update({k: v for k, v in techs.items() if v is not None})
    out["regime_flag"] = "trending" if (out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None and abs(out["ema20"]-out["ema50"])/max(float(out["last"]),1e-9) > 0.002) else "choppy"
    if out.get("ema20") is not None and out.get("ema50") is not None and out.get("last") is not None:
        lp = float(out["last"])
        out["mtf_align"] = bool((lp > out["ema20"] > out["ema50"]) or (lp < out["ema20"] < out["ema50"]))

    if opt_ctx:
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
    """Fetch live market features using Polygon first, with Yahoo fallback."""
    poly = await _fetch_polygon_features(symbol, expiry_iso=expiry_iso, side=side, strike=strike)
    if poly:
        return poly
    return await _fetch_yahoo_features(symbol)

        day_key = rows[-1][0].date()
        day_rows = [r for r in rows if r[0].date() == day_key]
        prev_rows = [r for r in rows if r[0].date() < day_key]
        prev_day = prev_rows[-78:] if prev_rows else []

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
    dte = None
    if expiry:
        try:
            dte = (datetime.fromisoformat(expiry).date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    # Start with alert baseline, then enrich with live ticker technicals.
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

    try:
        src = str(f.get("ta_src") or "unknown")
        data_note = f"📊 TA source: {src}" if live else "⚠️ Live data unavailable; using alert baseline"
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
            await send_telegram(tg_text)
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    try:
        log_alert_snapshot(alert, None, f)
    except Exception as e:
        logger.warning("[daily-report] log snapshot failed: %r", e)

    _append_decision_log({
        "timestamp_local": market_now(),
        "symbol": alert.get("symbol"),
        "side": alert.get("side"),
        "decision_final": decision_final,
        "llm": llm,
        "features": f,
        "preflight_ok": pf_ok,
        "preflight_checks": pf_checks,
        "ibkr": {"enabled": False, "attempted": False, "result": None},
    })


def _append_decision_log(entry: Dict[str, Any]) -> None:
    """Append decision entry to in-memory log with local safety guard."""
    try:
        _DECISIONS_LOG.append(entry)
    except Exception as e:
        logger.warning("decision log append failed: %r", e)

async def net_debug_info() -> Dict[str, Any]:
    return {
        "integrations": {
            "ibkr": "disabled",
            "polygon": "enabled" if polygon_enabled() else "disabled",
            "perplexity": "disabled",
            "market_data": "polygon_rest_with_yahoo_fallback",
        }
    }


__all__ = ["process_tradingview_job", "net_debug_info"]
