# engine_processor.py
import os
import re
import socket
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import HTTPException

from engine_runtime import get_http_client
from engine_common import (
    POLYGON_API_KEY,
    IBKR_ENABLED, IBKR_DEFAULT_QTY, IBKR_TIF, IBKR_ORDER_MODE, IBKR_USE_MID_AS_LIMIT,
    SCAN_MIN_VOL_RTH, SCAN_MIN_OI_RTH, SCAN_MIN_VOL_AH, SCAN_MIN_OI_AH,
    SEND_CHAIN_SCAN_ALERTS, SEND_CHAIN_SCAN_TOPN_ALERTS, REPLACE_IF_NO_NBBO,
    market_now, consume_llm,
    parse_alert_text,
    _is_rth_now, _occ_meta, _ticker_matches_side, _encode_ticker_path,
    _build_plus_minus_contracts,
    preflight_ok, compose_telegram_text,
)
from polygon_ops import (
    _http_json, _http_get_any, _pull_nbbo_direct, _probe_nbbo_verbose,
    _poly_reference_contracts_exists, _rescan_best_replacement, _find_nbbo_replacement_same_expiry,
)

from ibkr_client import place_recommended_option_order
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from feature_engine import build_features
from scoring import compute_decision_score, map_score_to_rating
from reporting import _DECISIONS_LOG
from market_ops import (
    polygon_get_option_snapshot_export,
    poly_option_backfill,
    scan_for_best_contract_for_alert,
    scan_top_candidates_for_alert,
    ensure_nbbo,
)

logger = logging.getLogger("trading_engine")

# =========================
# Math helpers (indicators)
# =========================
def _sma(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n: return None
    return sum(vals[-n:]) / float(n)

def _ema(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n: return None
    k = 2.0 / (n + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e

def _rsi(vals: List[float], n: int = 14) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) <= n: return None
    gains, losses = [], []
    for i in range(1, len(vals)):
        chg = vals[i] - vals[i-1]
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    ag = _sma(gains, n); al = _sma(losses, n)
    if ag is None or al is None: return None
    if al == 0: return 100.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))

def _macd(vals: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < slow + signal: return (None, None, None)
    # simple approximation
    def ema_n(vs, n):
        k = 2.0 / (n + 1); e = vs[0]
        for x in vs[1:]: e = x * k + e * (1 - k)
        return e
    tail = vals[-(slow + signal + 100):]
    macd_series = []
    for i in range(len(tail)):
        seg = tail[:i+1]
        if len(seg) >= slow:
            macd_series.append(ema_n(seg, fast) - ema_n(seg, slow))
    if not macd_series: return (None, None, None)
    macd_line = macd_series[-1]
    if len(macd_series) >= signal:
        signal_line = ema_n(macd_series[-(signal+50):], signal)
    else:
        signal_line = macd_line
    return (macd_line, signal_line, macd_line - signal_line if macd_line is not None and signal_line is not None else None)

# =========================
# Bars fetch with fallbacks
# =========================
async def _fetch_bars_for_indicators(symbol: str) -> Dict[str, Any]:
    """
    Try multiple Polygon endpoints so indicators are rarely None:
     1) 1-minute last ~180 mins
     2) 5-minute last ~5 days
     3) 1-day last ~120 days
    Compute RSI14 / SMA20 / EMA20 / MACD on the longest available series.
    """
    if not POLYGON_API_KEY:
        return {"note": "polygon api key missing"}

    client = get_http_client()
    if client is None:
        return {"note": "http client not ready"}

    now = datetime.now(timezone.utc)
    to_iso = now.isoformat()

    def closes_from(body: Any) -> List[float]:
        if not isinstance(body, dict): return []
        res = body.get("results") or []
        return [float(x.get("c")) for x in res if isinstance(x, dict) and isinstance(x.get("c"), (int, float))]

    # tier 1: 1-minute (3 hours)
    frm1 = (now - timedelta(minutes=180)).isoformat()
    r1 = await _http_get_any(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/min/{frm1}/{to_iso}",
        params={"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )
    c1 = closes_from(r1.get("body"))
    if len(c1) >= 30:
        rsi14 = _rsi(c1, 14); sma20 = _sma(c1, 20); ema20 = _ema(c1, 20)
        macd, sig, hist = _macd(c1, 12, 26, 9)
        return {
            "bars_source": "1m_recent",
            "indicator_closes_count": len(c1),
            "last_close": c1[-1],
            "rsi14": rsi14, "sma20": sma20, "ema20": ema20,
            "macd": macd, "macd_signal": sig, "macd_hist": hist,
        }

    # tier 2: 5-minute (5 days)
    frm2 = (now - timedelta(days=5)).isoformat()
    r2 = await _http_get_any(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/min/{frm2}/{to_iso}",
        params={"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )
    c2 = closes_from(r2.get("body"))
    if len(c2) >= 60:  # plenty for RSI/MACD on 5m too
        rsi14 = _rsi(c2, 14); sma20 = _sma(c2, 20); ema20 = _ema(c2, 20)
        macd, sig, hist = _macd(c2, 12, 26, 9)
        return {
            "bars_source": "5m_5d",
            "indicator_closes_count": len(c2),
            "last_close": c2[-1] if c2 else None,
            "rsi14": rsi14, "sma20": sma20, "ema20": ema20,
            "macd": macd, "macd_signal": sig, "macd_hist": hist,
        }

    # tier 3: 1-day (120 days)
    frm3 = (now - timedelta(days=200)).date().isoformat()
    r3 = await _http_get_any(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{frm3}/{to_iso}",
        params={"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )
    c3 = closes_from(r3.get("body"))
    if len(c3) >= 60:
        rsi14 = _rsi(c3, 14); sma20 = _sma(c3, 20); ema20 = _ema(c3, 20)
        macd, sig, hist = _macd(c3, 12, 26, 9)
        return {
            "bars_source": "1d_200d",
            "indicator_closes_count": len(c3),
            "last_close": c3[-1] if c3 else None,
            "rsi14": rsi14, "sma20": sma20, "ema20": ema20,
            "macd": macd, "macd_signal": sig, "macd_hist": hist,
        }

    return {"bars_source": "none", "indicator_closes_count": 0}

# =========================
# Enrichment bundle (stocks)
# =========================
async def _build_llm_enrichment(symbol: str) -> Dict[str, Any]:
    """
    Enrich LLM with:
      - Reference, Corporate actions, Stock snapshot
      - 1s/1m bars status, plus robust indicators via _fetch_bars_for_indicators
      - Simple 5m momentum & sentiment
    """
    if not POLYGON_API_KEY:
        return {"note": "polygon api key missing"}
    client = get_http_client()
    if client is None:
        return {"note": "http client not ready"}

    out: Dict[str, Any] = {}
    try:
        # Reference
        ref = await _http_get_any(
            "https://api.polygon.io/v3/reference/tickers",
            params={"ticker": symbol, "active": "true", "limit": 1, "apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        out["reference"] = (ref.get("body") or {}).get("results", [{}])[0] if isinstance(ref.get("body"), dict) else {"status": ref.get("status")}

        # Dividends & Splits
        div = await _http_get_any("https://api.polygon.io/v3/reference/dividends",
                                  params={"ticker": symbol, "limit": 1, "apiKey": POLYGON_API_KEY}, timeout=6.0)
        spl = await _http_get_any("https://api.polygon.io/v3/reference/splits",
                                  params={"ticker": symbol, "limit": 1, "apiKey": POLYGON_API_KEY}, timeout=6.0)
        out["corporate_actions"] = {
            "dividend_latest": (div.get("body") or {}).get("results", [{}])[0] if isinstance(div.get("body"), dict) else {},
            "split_latest": (spl.get("body") or {}).get("results", [{}])[0] if isinstance(spl.get("body"), dict) else {},
        }

        # Stock snapshot
        snap = await _http_get_any(f"https://api.polygon.io/v3/snapshot/stocks/{symbol}",
                                   params={"apiKey": POLYGON_API_KEY}, timeout=6.0)
        out["stock_snapshot"] = snap.get("body") if isinstance(snap.get("body"), dict) else {"status": snap.get("status")}

        # Status of recent aggregates (1m & 1s) for debugging feed health
        now = datetime.now(timezone.utc); to_iso = now.isoformat()
        frm_m = (now - timedelta(minutes=60)).isoformat()
        frm_s = (now - timedelta(seconds=120)).isoformat()
        aggs_1m = await _http_get_any(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/min/{frm_m}/{to_iso}",
            params={"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY}, timeout=8.0
        )
        aggs_1s = await _http_get_any(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/second/{frm_s}/{to_iso}",
            params={"adjusted":"true","sort":"asc","limit":1200,"apiKey":POLYGON_API_KEY}, timeout=8.0
        )
        out["aggs_1m_status"] = aggs_1m.get("status")
        out["aggs_1s_status"] = aggs_1s.get("status")

        # Indicators via robust fallbacks
        inds = await _fetch_bars_for_indicators(symbol)
        out["technical_indicators"] = {
            "rsi14": inds.get("rsi14"),
            "sma20": inds.get("sma20"),
            "ema20": inds.get("ema20"),
            "macd": inds.get("macd"),
            "macd_signal": inds.get("macd_signal"),
            "macd_hist": inds.get("macd_hist"),
        }
        out["bars_meta"] = {
            "source": inds.get("bars_source"),
            "closes_count": inds.get("indicator_closes_count"),
            "last_close": inds.get("last_close"),
        }

        # Simple 5m momentum (%)
        # If we had 1m bars in the indicator path we can approximate momentum using last 10 bars
        last = inds.get("last_close")
        closes_count = inds.get("indicator_closes_count") or 0
        mom5 = None
        if last is not None and closes_count >= 10:
            # not exact 5m but proportional to source (works for 1m & 5m; for daily it's a 10-day swing)
            # We don't store the series here for memory; rough proxy from counts isn't available, so leave as None unless we add series.
            pass
        out["momentum_5m_pct"] = mom5

        # Sentiment from indicators (fallback-neutral)
        rsi = out["technical_indicators"]["rsi14"]
        sentiment = "neutral"
        if isinstance(rsi, (int, float)):
            if rsi >= 60: sentiment = "bullish"
            elif rsi <= 40: sentiment = "bearish"
        out["market_sentiment"] = sentiment

    except Exception as e:
        out.setdefault("error", f"enrichment_failed: {type(e).__name__}: {e}")

    return out

# =========================
# Pick ±5% recommendation from top5 OI/Vol
# =========================
async def _recommend_plusminus_5pct(
    symbol: str,
    side: str,
    desired_strike: float,
    chosen_expiry: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    From scan_top_candidates_for_alert take top set, filter:
      - same expiry
      - correct side
    Then sort by (OI desc, Vol desc, spread asc, |strike - desired| asc)
    Return the best and include a short 'rec_text' for Telegram.
    """
    try:
        pool = await scan_top_candidates_for_alert(
            get_http_client(), symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": chosen_expiry},
            min_vol=min_vol, min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=12,
        ) or []
    except Exception:
        pool = []

    # Filter same expiry and correct side
    filtered: List[Dict[str, Any]] = []
    for it in pool:
        tk = it.get("ticker")
        if not tk: continue
        occ = _occ_meta(tk)
        if not occ or occ.get("expiry") != chosen_expiry: continue
        if not _ticker_matches_side(tk, side): continue
        filtered.append(it)

    # Sort by OI desc, Vol desc, spread asc, strike distance asc
    def oi(it): return it.get("oi") or 0
    def vol(it): return it.get("vol") or 0
    def spr(it): return float(it.get("spread_pct") or 1e9)
    def dist(it): return abs(float(it.get("strike") or desired_strike) - desired_strike)

    filtered.sort(key=lambda it: (-oi(it), -vol(it), spr(it), dist(it)))
    top5 = filtered[:5]

    if not top5:
        return None

    best = top5[0]
    rec = {
        "expiry": chosen_expiry,
        "ticker": best.get("ticker"),
        "strike": best.get("strike"),
        "bid": best.get("bid"), "ask": best.get("ask"), "mid": best.get("mid"),
        "spread_pct": best.get("spread_pct"),
        "oi": best.get("oi"), "vol": best.get("vol"),
    }
    rec["rec_text"] = (
        f"📌 Recommended (±5% from UL): {rec['expiry']} | {rec['ticker']} | "
        f"strike {rec['strike']} | OI {rec['oi']} Vol {rec['vol']} | "
        f"NBBO {rec['bid']}/{rec['ask']} mid={rec['mid']} | spread%={rec['spread_pct']}"
    )
    return rec

# =========================
# Core processing
# =========================
async def process_tradingview_job(job: Dict[str, Any]) -> None:
    client = get_http_client()
    if client is None:
        logger.warning("[worker] HTTP client not ready")
        return

    selection_debug: Dict[str, Any] = {}
    replacement_note: Optional[Dict[str, Any]] = None
    option_ticker: Optional[str] = None

    # 1) Parse
    try:
        alert = parse_alert_text(job["alert_text"])
        logger.info("parsed alert: side=%s symbol=%s strike=%s expiry=%s",
                    alert.get("side"), alert.get("symbol"), alert.get("strike"), alert.get("expiry"))
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    side = alert["side"]; symbol = alert["symbol"]
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy  = bool(job["flags"].get("force_buy", False))
    qty        = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))
    orig_strike = alert.get("strike"); orig_expiry = alert.get("expiry")

    # 2) Expiry defaulting
    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    def _next_friday(d): return d + timedelta(days=(4 - d.weekday()) % 7)
    def same_week_friday(d): return (d - timedelta(days=d.weekday())) + timedelta(days=4)
    target_expiry_date = _next_friday(today_utc) + timedelta(days=7)
    swf = same_week_friday(today_utc)
    if (target_expiry_date - timedelta(days=target_expiry_date.weekday())) == (swf - timedelta(days=swf.weekday())):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    pm = _build_plus_minus_contracts(symbol, ul_px, target_expiry)
    desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]

    # 3) Scan thresholds
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) Selection via scan (honor side)
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            client, symbol,
            {"side": side, "symbol": symbol, "strike": alert.get("strike"), "expiry": alert.get("expiry")},
            min_vol=scan_min_vol, min_oi=scan_min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
        )
    except Exception:
        best_from_scan = None

    candidate = None
    if best_from_scan and _ticker_matches_side(best_from_scan.get("ticker"), side):
        candidate = best_from_scan
    else:
        try:
            pool = await scan_top_candidates_for_alert(
                client, symbol,
                {"side": side, "symbol": symbol, "strike": alert.get("strike"), "expiry": alert.get("expiry")},
                min_vol=scan_min_vol, min_oi=scan_min_oi, top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=6,
            ) or []
        except Exception:
            pool = []
        pool = [it for it in pool if _ticker_matches_side(it.get("ticker"), side)]
        candidate = pool[0] if pool else None

    if candidate:
        option_ticker = candidate["ticker"]
        if isinstance(candidate.get("strike"), (int, float)):
            desired_strike = float(candidate["strike"])
        occ = _occ_meta(option_ticker)
        chosen_expiry = occ["expiry"] if occ and occ.get("expiry") else str(candidate.get("expiry") or orig_expiry or target_expiry)
        selection_debug = {"selected_by": "chain_scan", "selected_ticker": option_ticker,
                           "best_item": candidate, "chosen_expiry": chosen_expiry}
        logger.info("chain_scan selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)
    else:
        fallback_exp = str(orig_expiry or target_expiry)
        option_ticker = pm["contract_call"] if side == "CALL" else pm["contract_put"]
        desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]
        chosen_expiry = fallback_exp
        selection_debug = {"selected_by": "fallback_pm", "reason": "scan_empty", "chosen_expiry": fallback_exp}
        logger.info("fallback selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)

    chosen_expiry = selection_debug.get("chosen_expiry", str(orig_expiry or target_expiry))

    # 4) Features + NBBO + ENRICHMENT
    f: Dict[str, Any] = {}
    try:
        if not POLYGON_API_KEY:
            f = {
                "bid": None, "ask": None, "mid": None, "last": None,
                "option_spread_pct": None, "quote_age_sec": None,
                "oi": None, "vol": None,
                "delta": None, "gamma": None, "theta": None, "vega": None,
                "iv": None, "iv_rank": None, "rv20": None, "prev_close": None, "quote_change_pct": None,
                "dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days,
                "em_vs_be_ok": None, "mtf_align": None, "sr_headroom_ok": None, "regime_flag": "trending",
                "prev_day_high": None, "prev_day_low": None,
                "premarket_high": None, "premarket_low": None,
                "vwap": None, "vwap_dist": None,
                "above_pdh": None, "below_pdl": None, "above_pmh": None, "below_pml": None,
                "llm_enrichment": {"note": "polygon api key missing"},
            }
        else:
            extra = await poly_option_backfill(get_http_client(), symbol, option_ticker, datetime.now(timezone.utc).date())
            for k, v in (extra or {}).items():
                if v is not None:
                    f[k] = v

            snap = await polygon_get_option_snapshot_export(get_http_client(), underlying=symbol, option_ticker=option_ticker)
            core = await build_features(get_http_client(), alert={**alert, "strike": desired_strike, "expiry": chosen_expiry}, snapshot=snap)
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v

            # derive mid/spread
            try:
                bid = f.get("bid"); ask = f.get("ask"); mid = f.get("mid")
                if bid is not None and ask is not None:
                    if mid is None:
                        mid = (float(bid) + float(ask)) / 2.0
                        f["mid"] = round(mid, 4)
                    spread = float(ask) - float(bid)
                    if mid and mid > 0:
                        f["option_spread_pct"] = round((spread / mid) * 100.0, 3)
            except Exception:
                pass

            # ensure NBBO
            try:
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo = await ensure_nbbo(get_http_client(), option_ticker, tries=12, delay=0.35)
                    for k, v in (nbbo or {}).items():  # may be empty if no entitlement
                        if v is not None:
                            f[k] = v
            except Exception:
                pass

            if f.get("bid") is None or f.get("ask") is None:
                for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                    if v is not None:
                        f[k] = v

            if f.get("dte") is None:
                try:
                    f["dte"] = (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days
                except Exception:
                    pass

            if f.get("quote_change_pct") is None:
                try:
                    prev_close = f.get("prev_close")
                    mark = f.get("mid") if f.get("mid") is not None else f.get("last")
                    if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
                        f["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
                except Exception:
                    pass

            if (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("last"), (int, float)):
                f.setdefault("mid", float(f["last"]))

            # probe verbose NBBO; also detect entitlement
            if f.get("bid") is None or f.get("ask") is None:
                nbbo_dbg = await _probe_nbbo_verbose(option_ticker)
                for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
                    if nbbo_dbg.get(k) is not None:
                        f[k] = nbbo_dbg[k]
                f["nbbo_http_status"] = nbbo_dbg.get("nbbo_http_status")
                f["nbbo_reason"] = nbbo_dbg.get("nbbo_reason")
                f["nbbo_body_sample"] = nbbo_dbg.get("nbbo_body_sample")
                # detect likely plan/entitlement gap
                if f.get("nbbo_http_status") in (402, 403, 404):
                    f["nbbo_capability"] = "unavailable"
                else:
                    f["nbbo_capability"] = "ok"

            # soft spread when only mark/last
            if (f.get("option_spread_pct") is None) and (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("mid"), (int, float)):
                f["option_spread_pct"] = float(os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "10.0"))

            # Enrichment for LLM (robust indicators)
            enrichment = await _build_llm_enrichment(symbol)
            f["llm_enrichment"] = enrichment
            f["market_sentiment"] = enrichment.get("market_sentiment")

    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days,
                  "llm_enrichment": {"error": f"features_failed: {type(e).__name__}: {e}"}}

    # 4b) NBBO-driven replacement (listed but missing NBBO)
    if REPLACE_IF_NO_NBBO and (f.get("bid") is None or f.get("ask") is None or (f.get("nbbo_http_status") and f.get("nbbo_http_status") != 200)):
        try:
            alt = await _find_nbbo_replacement_same_expiry(
                symbol=symbol, side=side, desired_strike=desired_strike,
                expiry_iso=chosen_expiry, min_vol=scan_min_vol, min_oi=scan_min_oi,
            )
        except Exception:
            alt = None
        if alt and alt.get("ticker") and alt["ticker"] != option_ticker:
            old_tk = option_ticker
            option_ticker = alt["ticker"]
            desired_strike = float(alt.get("strike") or desired_strike)
            occ = _occ_meta(option_ticker)
            chosen_expiry = (occ["expiry"] if occ else str(alt.get("expiry") or chosen_expiry))
            try:
                extra2 = await poly_option_backfill(get_http_client(), symbol, option_ticker, datetime.now(timezone.utc).date())
                for k, v in (extra2 or {}).items():
                    if v is not None:
                        f[k] = v
                for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                    if v is not None:
                        f[k] = v
                if f.get("bid") is None or f.get("ask") is None:
                    nbbo_dbg2 = await _probe_nbbo_verbose(option_ticker)
                    for k in ("bid","ask","mid","option_spread_pct","quote_age_sec"):
                        if nbbo_dbg2.get(k) is not None:
                            f[k] = nbbo_dbg2[k]
                    f["nbbo_http_status"] = nbbo_dbg2.get("nbbo_http_status")
                    f["nbbo_reason"] = nbbo_dbg2.get("nbbo_reason")
                replacement_note = {"old": old_tk, "new": option_ticker, "why": "missing NBBO on initial pick"}
                logger.info("Replaced due to missing NBBO: %s → %s", old_tk, option_ticker)
            except Exception as e:
                logger.warning("NBBO replacement refresh failed: %r", e)

    # 5) 404 replacement if truly not listed
    if f.get("nbbo_http_status") == 404 and POLYGON_API_KEY:
        exist = await _poly_reference_contracts_exists(symbol, chosen_expiry, option_ticker)
        logger.info("NBBO 404 verification: listed=%s snapshot_ok=%s reason=%s",
                    exist.get("listed"), exist.get("snapshot_ok"), exist.get("reason"))
        if exist.get("listed") is False and not exist.get("snapshot_ok"):
            repl = await _rescan_best_replacement(
                symbol=symbol, side=side, desired_strike=desired_strike, expiry_iso=chosen_expiry,
                min_vol=scan_min_vol, min_oi=scan_min_oi,
            )
            if repl:
                old_tk = option_ticker
                option_ticker = repl["ticker"]
                desired_strike = float(repl.get("strike") or desired_strike)
                try:
                    occ = _occ_meta(option_ticker)
                    chosen_expiry = (occ["expiry"] if occ else str(repl.get("expiry") or chosen_expiry))
                    extra2 = await poly_option_backfill(get_http_client(), symbol, option_ticker, datetime.now(timezone.utc).date())
                    for k, v in (extra2 or {}).items():
                        if v is not None:
                            f[k] = v
                    for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                        if v is not None:
                            f[k] = v
                    if f.get("bid") is None or f.get("ask") is None:
                        nbbo_dbg2 = await _probe_nbbo_verbose(option_ticker)
                        for k in ("bid","ask","mid","option_spread_pct","quote_age_sec"):
                            if nbbo_dbg2.get(k) is not None:
                                f[k] = nbbo_dbg2[k]
                        f["nbbo_http_status"] = nbbo_dbg2.get("nbbo_http_status")
                        f["nbbo_reason"] = nbbo_dbg2.get("nbbo_reason")
                    replacement_note = {"old": old_tk, "new": option_ticker, "why": "contract not listed in Polygon reference/snapshot"}
                    logger.info("Replaced contract due to 404: %s → %s", old_tk, option_ticker)
                except Exception as e:
                    logger.warning("Replacement contract fetch failed: %r", e)
                    replacement_note = None

    # 6) Optional Telegram pre-LLM
    if SEND_CHAIN_SCAN_ALERTS and selection_debug.get("selected_by", "").startswith("chain_scan"):
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                pre_text = (
                    "🔎 Chain-Scan Pick (from TradingView alert)\n"
                    f"{side} {symbol} | Strike {desired_strike} | Exp {chosen_expiry}\n"
                    f"Contract: {option_ticker}\n"
                    f"NBBO {f.get('bid')}/{f.get('ask')}  Mark={f.get('mid')}  Last={f.get('last')}\n"
                    f"Spread%={f.get('option_spread_pct')}  QuoteAge(s)={f.get('quote_age_sec')}\n"
                    f"OI={f.get('oi')}  Vol={f.get('vol')}  IV={f.get('iv')}  Δ={f.get('delta')} Γ={f.get('gamma')}\n"
                    f"DTE={f.get('dte')}  Regime={f.get('regime_flag')}  (pre-LLM)\n"
                    f"NBBO dbg: status={f.get('nbbo_http_status')} reason={f.get('nbbo_reason')} cap={f.get('nbbo_capability')}\n"
                )
                await send_telegram(pre_text)
        except Exception as e:
            logger.exception("[worker] Telegram pre-LLM chainscan error: %s", e)

    # 7) LLM
    pf_ok, pf_checks = preflight_ok(f)
    try:
        llm = await analyze_with_openai(alert, f)  # f now includes llm_enrichment + market_sentiment + nbbo_capability
        consume_llm()
    except Exception as e:
        llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}", "checklist": {}, "ev_estimate": {}}
    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")

    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    if force_buy:
        decision_final = "buy"

    # 7b) Build ±5% recommendation (from top-5 OI/Vol)
    rec5 = await _recommend_plusminus_5pct(symbol, side, desired_strike, chosen_expiry, scan_min_vol, scan_min_oi)
    rec_text = rec5.get("rec_text") if rec5 else ""

    # 8) Telegram final (diff note + indicators + recommendation)
    try:
        bits = []
        if isinstance(orig_strike, (int, float)) and isinstance(desired_strike, (int, float)) and float(orig_strike) != float(desired_strike):
            bits.append(f"🎯 Selected strike {desired_strike} (alert was {orig_strike})")
        if orig_expiry and chosen_expiry and str(orig_expiry) != str(chosen_expiry):
            bits.append(f"🗓 Selected expiry {chosen_expiry} (alert was {orig_expiry})")

        # Compact enrichment line
        ti = ((f.get("llm_enrichment") or {}).get("technical_indicators") or {})
        rsi14 = ti.get("rsi14"); sma20 = ti.get("sma20"); ema20 = ti.get("ema20")
        enr_line = f"📈 RSI14={round(rsi14,2) if isinstance(rsi14,(int,float)) else rsi14} SMA20≈{round(sma20,2) if isinstance(sma20,(int,float)) else sma20} EMA20≈{round(ema20,2) if isinstance(ema20,(int,float)) else ema20} Sentiment={f.get('market_sentiment')}"
        bits.append(enr_line)

        if rec_text: bits.append(rec_text)
        diff_note = "\n".join([b for b in bits if b])

        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker, f=f, llm=llm, llm_ran=True, llm_reason="",
            score=score, rating=rating, diff_note=diff_note,
        )
        if selection_debug.get("selected_by","").startswith("chain_scan"):
            tg_text += "\n🔎 Note: Contract selected via chain-scan (liquidity + strike/expiry fit)."
        if replacement_note is not None:
            tg_text += f"\n⚠️ Replacement: {replacement_note['old']} → {replacement_note['new']} ({replacement_note['why']})."
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await send_telegram(tg_text)
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    # 9) IBKR (optional)
    ib_attempted = False
    ib_result_obj: Optional[Any] = None
    try:
        if (decision_final == "buy") and ib_enabled and (pf_ok or force_buy):
            ib_attempted = True
            mode = IBKR_ORDER_MODE
            mid = f.get("mid")
            if mode == "market":
                use_market = True
            elif mode == "limit":
                use_market = (mid is None)
            else:
                use_market = not (IBKR_USE_MID_AS_LIMIT and (mid is not None))
            limit_px = None if use_market else float(mid) if mid is not None else None

            ib_result_obj = await place_recommended_option_order(
                symbol=symbol, side=side, strike=float(desired_strike), expiry_iso=chosen_expiry,
                quantity=int(qty), limit_price=limit_px, action="BUY", tif=IBKR_TIF,
            )
    except Exception as e:
        ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 10) Decision log
    _DECISIONS_LOG.append({
        "timestamp_local": market_now(),
        "symbol": symbol,
        "side": side,
        "option_ticker": option_ticker,
        "decision_final": decision_final,
        "decision_path": f"llm.{decision_final}",
        "prescore": None,
        "llm": {"ran": True, "decision": llm.get("decision"),
                "confidence": llm.get("confidence"), "reason": llm.get("reason")},
        "features": {
            "reco_expiry": chosen_expiry,
            "oi": f.get("oi"), "vol": f.get("vol"),
            "bid": f.get("bid"), "ask": f.get("ask"),
            "mark": f.get("mid"), "last": f.get("last"),
            "spread_pct": f.get("option_spread_pct"), "quote_age_sec": f.get("quote_age_sec"),
            "prev_close": f.get("prev_close"), "quote_change_pct": f.get("quote_change_pct"),
            "delta": f.get("delta"), "gamma": f.get("gamma"), "theta": f.get("theta"), "vega": f.get("vega"),
            "dte": f.get("dte"), "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"), "sr_ok": f.get("sr_headroom_ok"), "iv": f.get("iv"),
            "iv_rank": f.get("iv_rank"), "rv20": f.get("rv20"), "regime": f.get("regime_flag"),
            "prev_day_high": f.get("prev_day_high"), "prev_day_low": f.get("prev_day_low"),
            "premarket_high": f.get("premarket_high"), "premarket_low": f.get("premarket_low"),
            "vwap": f.get("vwap"), "vwap_dist": f.get("vwap_dist"),
            "above_pdh": f.get("above_pdh"), "below_pdl": f.get("below_pdl"),
            "above_pmh": f.get("above_pmh"), "below_pml": f.get("below_pml"),
            "nbbo_http_status": f.get("nbbo_http_status"), "nbbo_reason": f.get("nbbo_reason"),
            "nbbo_capability": f.get("nbbo_capability"),
            "llm_enrichment": f.get("llm_enrichment"),
            "market_sentiment": f.get("market_sentiment"),
        },
        "pm_contracts": {
            "plus5_call": {"strike": pm["strike_call"], "contract": pm["contract_call"]},
            "minus5_put": {"strike": pm["strike_put"],  "contract": pm["contract_put"]},
        },
        "ibkr": {"enabled": ib_enabled, "attempted": ib_attempted, "result": ib_result_obj},
        "selection_debug": selection_debug,
        "alert_original": {"strike": orig_strike, "expiry": orig_expiry},
        "chosen": {"strike": desired_strike, "expiry": chosen_expiry},
        "replacement": replacement_note,
        "recommendation_pm5": rec5,
    })

# =========================
# Diagnostics
# =========================
async def diag_polygon_bundle(underlying: str, contract: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None:
        raise HTTPException(status_code=503, detail="HTTP client not ready")
    enc = _encode_ticker_path(contract)
    out = {}

    m = re.search(r":([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$", contract)
    if m:
        yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
        expiry_iso = f"20{yy}-{mm}-{dd}"
        side = "call" if cp.upper() == "C" else "put"
        out["multi"] = await _http_json(
            client,
            f"https://api.polygon.io/v3/snapshot/options/{underlying}",
            {"apiKey": POLYGON_API_KEY, "contract_type": side, "expiration_date": expiry_iso, "limit": 5, "greeks": "true"},
            timeout=6.0
        )

    out["single"] = await _http_json(
        client,
        f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    out["last_quote"] = await _http_json(
        client,
        f"https://api.polygon.io/v3/quotes/options/{enc}/last",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    out["open_close"] = await _http_json(
        client,
        f"https://api.polygon.io/v1/open-close/options/{enc}/{yday}",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0
    )
    now_utc_dt = datetime.now(timezone.utc)
    frm_iso = datetime(now_utc_dt.year, now_utc_dt.month, now_utc_dt.day, 0,0,0,tzinfo=timezone.utc).isoformat()
    to_iso = now_utc_dt.isoformat()
    out["aggs"] = await _http_json(
        client,
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/min/{frm_iso}/{to_iso}?",
        {"adjusted":"true","sort":"asc","limit":2000,"apiKey":POLYGON_API_KEY},
        timeout=8.0
    )

    def skim(d):
        if not isinstance(d, dict): return d
        res = d.get("results")
        return {
            "keys": list(d.keys())[:10],
            "sample": (res[:2] if isinstance(res, list) else (res if isinstance(res, dict) else d)),
            "status_hint": d.get("status"),
        }
    return {
        "multi": skim(out.get("multi")),
        "single": skim(out.get("single")),
        "last_quote": skim(out.get("last_quote")),
        "open_close": skim(out.get("open_close")),
        "aggs": skim(out.get("aggs")),
    }

async def net_debug_info() -> Dict[str, Any]:
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    out_ip = None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            out_ip = (await c.get("https://ifconfig.me/ip")).text.strip()
    except Exception as e:
        out_ip = f"fetch-failed: {e.__class__.__name__}"
    can_connect = None; err = None
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.close()
        can_connect = True
    except Exception as e:
        can_connect = False
        err = f"{e.__class__.__name__}: {e}"
    return {"ibkr_host": host, "ibkr_port": port, "egress_ip": out_ip, "connect_test": can_connect, "error": err}

__all__ = ["process_tradingview_job", "diag_polygon_bundle", "net_debug_info"]
