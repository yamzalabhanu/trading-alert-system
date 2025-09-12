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
# Indicator helpers (per-metric fallbacks 1m→5m→1d)
# =========================
def _sma(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / float(n)

def _ema(vals: List[float], n: int) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < max(5, n // 2):  # require a few points for stability
        return None
    k = 2.0 / (n + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e

def _rsi(vals: List[float], n: int = 14) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < n + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(vals)):
        chg = vals[i] - vals[i-1]
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    ag = _sma(gains, n)
    al = _sma(losses, n)
    if ag is None or al is None:
        return None
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))

def _macd(vals: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < slow + 1:
        return (None, None, None)

    def ema_series(seq: List[float], n: int) -> List[float]:
        if not seq:
            return []
        k = 2.0 / (n + 1)
        out = [seq[0]]
        for x in seq[1:]:
            out.append(x * k + out[-1] * (1 - k))
        return out

    ema_fast = ema_series(vals, fast)
    ema_slow = ema_series(vals, slow)
    cut = min(len(ema_fast), len(ema_slow))
    if cut == 0:
        return (None, None, None)
    macd_series = [ema_fast[-cut+i] - ema_slow[-cut+i] for i in range(cut)]
    if not macd_series:
        return (None, None, None)

    macd_line = macd_series[-1]
    if len(macd_series) >= signal:
        sig_series = ema_series(macd_series, signal)
        signal_line = sig_series[-1]
        hist = macd_line - signal_line
        return (macd_line, signal_line, hist)
    else:
        return (macd_line, None, None)

def _fmt2(v: Optional[float]) -> str:
    return "None" if v is None else f"{float(v):.2f}"

async def _fetch_bars_for_indicators_best_effort(symbol: str) -> Dict[str, Any]:
    """
    Pull 1m/5m/1d bars with generous windows.
    Compute each indicator independently with per-metric fallbacks:
      RSI needs >=15 bars, SMA20 >=20, EMA20 >=~10, MACD needs >=27, Δ% >=2.
    Returns values + which timeframe was used for each metric.
    """
    if not POLYGON_API_KEY:
        return {"note": "polygon api key missing"}

    client = get_http_client()
    if client is None:
        return {"note": "http client not ready"}

    def _closes_from(body: Any) -> List[float]:
        if not isinstance(body, dict):
            return []
        res = body.get("results") or []
        return [float(x.get("c")) for x in res if isinstance(x, dict) and isinstance(x.get("c"), (int, float))]

    async def _get_aggs(mult: int, unit: str, start: str, end: str, timeout: float = 9.0) -> Dict[str, Any]:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{mult}/{unit}/{start}/{end}"
        resp = await _http_get_any(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 2000, "apiKey": POLYGON_API_KEY},
            timeout=timeout,
        )
        body = resp.get("body")
        closes = _closes_from(body)
        return {"status": resp.get("status"), "closes": closes, "count": len(closes)}

    now = datetime.now(timezone.utc)
    to_date = now.date().isoformat()
    to_iso  = now.isoformat()

    # Windows generous enough to work AH / weekends
    r_1m = await _get_aggs(1, "min", (now - timedelta(hours=24)).isoformat(), to_iso)
    r_5m = await _get_aggs(5, "min", (now - timedelta(days=30)).isoformat(), to_iso)
    r_1d = await _get_aggs(1, "day", (now - timedelta(days=720)).date().isoformat(), to_date)

    series_map = {
        "1m": r_1m.get("closes", []),
        "5m": r_5m.get("closes", []),
        "1d": r_1d.get("closes", []),
    }
    counts_map = {k: len(v) for k, v in series_map.items()}
    status_map = {"1m": r_1m.get("status"), "5m": r_5m.get("status"), "1d": r_1d.get("status")}

    def pick(min_needed: int, order=("1m", "5m", "1d")) -> Tuple[Optional[List[float]], Optional[str]]:
        for tf in order:
            seq = series_map.get(tf, [])
            if len(seq) >= min_needed:
                return seq, tf
        # fall back to longest available even if short (will yield None for some)
        tf_longest = max(series_map.keys(), key=lambda k: len(series_map[k]))
        return series_map.get(tf_longest, []), tf_longest

    # Per-metric picks
    closes_rsi, rsi_src = pick(15)          # needs n+1 bars (n=14)
    closes_sma, sma_src = pick(20)
    closes_ema, ema_src = pick(10)          # let EMA run with ~10+ bars for stability
    closes_mac, mac_src = pick(27)          # 26 slow + 1
    closes_dp,  dp_src  = pick(2)

    rsi14 = _rsi(closes_rsi, 14) if closes_rsi else None
    sma20 = _sma(closes_sma, 20) if closes_sma else None
    ema20 = _ema(closes_ema, 20) if closes_ema else None
    macd_line, macd_sig, macd_hist = _macd(closes_mac, 12, 26, 9) if closes_mac else (None, None, None)
    delta_pct = None
    if closes_dp and len(closes_dp) >= 2:
        prev = closes_dp[-2]
        if prev:
            delta_pct = (closes_dp[-1] - prev) / prev * 100.0

    # Choose the overall source as the TF with most bars (for logging only)
    overall_src = max(series_map.keys(), key=lambda k: len(series_map[k])) if series_map else None

    return {
        "rsi14": rsi14, "rsi_source": rsi_src,
        "sma20": sma20, "sma_source": sma_src,
        "ema20": ema20, "ema_source": ema_src,
        "macd": macd_line, "macd_signal": macd_sig, "macd_hist": macd_hist, "macd_source": mac_src,
        "delta_pct": delta_pct, "delta_pct_source": dp_src,
        "bars_source": overall_src,
        "indicator_closes_count": counts_map.get(overall_src) if overall_src else 0,
        "aggs_status": status_map, "aggs_counts": counts_map,
        "last_close": {tf: (series_map[tf][-1] if series_map[tf] else None) for tf in ("1m", "5m", "1d")},
    }

# =========================
# Core processing (called by runtime worker)
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

    side = alert["side"]
    symbol = alert["symbol"]
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy  = bool(job["flags"].get("force_buy", False))
    qty        = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    # Preserve originals for messaging
    orig_strike = alert.get("strike")
    orig_expiry = alert.get("expiry")

    # 2) Expiry defaulting
    ul_px = float(alert["underlying_price_from_alert"])
    today_utc = datetime.now(timezone.utc).date()
    # compute two-weeks Friday, avoiding same-week overlap
    def _next_friday(d): return d + timedelta(days=(4 - d.weekday()) % 7)
    def same_week_friday(d): return (d - timedelta(days=d.weekday())) + timedelta(days=4)
    target_expiry_date = _next_friday(today_utc) + timedelta(days=7)
    swf = same_week_friday(today_utc)
    if (target_expiry_date - timedelta(days=target_expiry_date.weekday())) == (swf - timedelta(days=swf.weekday())):
        target_expiry_date = swf + timedelta(days=7)
    target_expiry = target_expiry_date.isoformat()

    pm = _build_plus_minus_contracts(symbol, ul_px, target_expiry)
    desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]

    # 3) Chain scan thresholds
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) Selection via scan (strictly honor side)
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            client,
            symbol,
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
                client,
                symbol,
                {"side": side, "symbol": symbol, "strike": alert.get("strike"), "expiry": alert.get("expiry")},  # noqa
                min_vol=scan_min_vol, min_oi=scan_min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
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
        # build fallback +/− strikes, then pick correct side
        option_ticker = pm["contract_call"] if side == "CALL" else pm["contract_put"]
        desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]
        chosen_expiry = fallback_exp
        selection_debug = {"selected_by": "fallback_pm", "reason": "scan_empty", "chosen_expiry": fallback_exp}
        logger.info("fallback selected: %s (strike=%s exp=%s)", option_ticker, desired_strike, chosen_expiry)

    chosen_expiry = selection_debug.get("chosen_expiry", str(orig_expiry or target_expiry))

    # 4) Feature bundle + NBBO
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
                    for k, v in (nbbo or {}).items():
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

            if f.get("bid") is None or f.get("ask") is None:
                nbbo_dbg = await _probe_nbbo_verbose(option_ticker)
                for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
                    if nbbo_dbg.get(k) is not None:
                        f[k] = nbbo_dbg[k]
                f["nbbo_http_status"] = nbbo_dbg.get("nbbo_http_status")
                f["nbbo_reason"] = nbbo_dbg.get("nbbo_reason")
                f["nbbo_body_sample"] = nbbo_dbg.get("nbbo_body_sample")

            if (f.get("option_spread_pct") is None) and (f.get("bid") is None or f.get("ask") is None) and isinstance(f.get("mid"), (int, float)):
                f["option_spread_pct"] = float(os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "10.0"))

    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}

    # === 4a) Indicators (per-metric fallbacks; never all-or-nothing) ===
    try:
        ind = await _fetch_bars_for_indicators_best_effort(symbol)
    except Exception as e:
        ind = {"note": f"indicator fetch error: {type(e).__name__}: {e}"}

    # Attach to features for LLM
    for k in (
        "sma20","ema20","rsi14","macd","macd_signal","macd_hist","delta_pct",
        "sma_source","ema_source","rsi_source","macd_source","delta_pct_source",
        "bars_source","indicator_closes_count","aggs_status","aggs_counts","last_close"
    ):
        if k in ind:
            f[k] = ind[k]

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

    # 5) 404 replacement if contract truly not listed
    if f.get("nbbo_http_status") == 404 and POLYGON_API_KEY:
        exist = await _poly_reference_contracts_exists(symbol, chosen_expiry, option_ticker)
        logger.info("NBBO 404 verification: listed=%s snapshot_ok=%s reason=%s",
                    exist.get("listed"), exist.get("snapshot_ok"), exist.get("reason"))
        if exist.get("listed") is False and not exist.get("snapshot_ok"):
            repl = await _rescan_best_replacement(
                symbol=symbol, side=side,
                desired_strike=desired_strike, expiry_iso=chosen_expiry,
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
                    replacement_note = {
                        "old": old_tk, "new": option_ticker,
                        "why": "contract not listed in Polygon reference/snapshot",
                    }
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
                    f"NBBO dbg: status={f.get('nbbo_http_status')} reason={f.get('nbbo_reason')}\n"
                )
                await send_telegram(pre_text)
        except Exception as e:
            logger.exception("[worker] Telegram pre-LLM chainscan error: %s", e)

    # 7) LLM
    pf_ok, pf_checks = preflight_ok(f)
    try:
        llm = await analyze_with_openai(alert, f)
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

    # Diff note (strike/expiry changes)
    diff_bits = []
    if isinstance(orig_strike, (int, float)) and isinstance(desired_strike, (int, float)) and float(orig_strike) != float(desired_strike):
        diff_bits.append(f"🎯 Selected strike {desired_strike} (alert was {orig_strike})")
    if orig_expiry and chosen_expiry and str(orig_expiry) != str(chosen_expiry):
        diff_bits.append(f"🗓 Selected expiry {chosen_expiry} (alert was {orig_expiry})")
    diff_note = "\n".join(diff_bits)

    # Indicator summary line (shows values + source TF for each)
    ind_line = (
        "📈 "
        f"RSI14={_fmt2(f.get('rsi14'))}({f.get('rsi_source')}) "
        f"SMA20={_fmt2(f.get('sma20'))}({f.get('sma_source')}) "
        f"EMA20={_fmt2(f.get('ema20'))}({f.get('ema_source')}) "
        f"Δ%={_fmt2(f.get('delta_pct'))}({f.get('delta_pct_source')})"
    )
    combined_note = "\n".join([x for x in [diff_note, ind_line] if x])

    # 8) Telegram final
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker, f=f, llm=llm, llm_ran=True, llm_reason="", score=score, rating=rating,
            diff_note=combined_note,
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
                symbol=symbol, side=side,
                strike=float(desired_strike), expiry_iso=chosen_expiry,
                quantity=int(qty),
                limit_price=limit_px, action="BUY", tif=IBKR_TIF,
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
            # indicators
            "rsi14": f.get("rsi14"), "sma20": f.get("sma20"), "ema20": f.get("ema20"),
            "macd": f.get("macd"), "macd_signal": f.get("macd_signal"), "macd_hist": f.get("macd_hist"),
            "delta_pct": f.get("delta_pct"),
            "rsi_source": f.get("rsi_source"), "sma_source": f.get("sma_source"),
            "ema_source": f.get("ema_source"), "macd_source": f.get("macd_source"),
            "delta_pct_source": f.get("delta_pct_source"),
            "bars_source": f.get("bars_source"),
            "indicator_closes_count": f.get("indicator_closes_count"),
            "aggs_status": f.get("aggs_status"), "aggs_counts": f.get("aggs_counts"),
            "last_close": f.get("last_close"),
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
