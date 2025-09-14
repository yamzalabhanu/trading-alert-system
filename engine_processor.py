# engine_processor_v1.0.py  (updated)
import os
import re
import socket
import logging
import asyncio
import random
from typing import Dict, Any, Optional
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
from market_ops import (
    polygon_get_option_snapshot_export,
    poly_option_backfill,
    scan_for_best_contract_for_alert,
    scan_top_candidates_for_alert,
    ensure_nbbo,
)

# ---- Safe decisions log import ----
try:
    from reporting import _DECISIONS_LOG
except Exception:
    _DECISIONS_LOG = []

logger = logging.getLogger("trading_engine")


# ---------- Provider gating & rate-limit helpers ----------

def _provider_tokens_present() -> bool:
    """Return True only if at least one external provider looks properly configured."""
    has_iex = bool(os.getenv("IEX_TOKEN") or os.getenv("IEX_CLOUD_TOKEN") or os.getenv("IEX_PUB_TOKEN"))
    has_alpaca = bool(os.getenv("APCA_API_KEY_ID") and (os.getenv("APCA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET")))
    has_tradier = bool(os.getenv("TRADIER_TOKEN") or os.getenv("TRADIER_ACCESS_TOKEN"))
    has_ibkr = bool(IBKR_ENABLED)
    return has_iex or has_alpaca or has_tradier or has_ibkr

def _should_use_multi_providers() -> bool:
    """
    Multi-provider NBBO is enabled if:
      - MULTI_PROVIDER_NBBO != "0" (default "1"), AND
      - At least one provider credential is present.
    """
    flag = os.getenv("MULTI_PROVIDER_NBBO", "1") != "0"
    return flag and _provider_tokens_present()

def _rl_jitter_seconds() -> float:
    """Tiny random sleep to de-synchronize retry bursts; overridable via env."""
    base = float(os.getenv("POLY_RL_BASE_JITTER", "0.05"))
    spread = float(os.getenv("POLY_RL_SPREAD_JITTER", "0.15"))
    return max(0.0, base + random.random() * spread)


# ---------- Synthetic NBBO helpers ----------
def _synth_spread_pct_default() -> float:
    """
    Spread estimate depending on RTH/AH (overridable via env):
      - SYNTH_SPREAD_PCT_RTH (default 1.0)
      - SYNTH_SPREAD_PCT_AH  (default 2.0)
      - fallback SYNTH_SPREAD_PCT (applies to both if specific not set)
    """
    fallback = os.getenv("SYNTH_SPREAD_PCT")
    if fallback is not None:
        try:
            return float(fallback)
        except Exception:
            pass
    rth = _is_rth_now()
    return float(os.getenv("SYNTH_SPREAD_PCT_RTH", "1.0") if rth else os.getenv("SYNTH_SPREAD_PCT_AH", "2.0"))


def _make_synthetic_nbbo_from_base(base_px: Optional[float], spread_pct: Optional[float] = None) -> Dict[str, Any]:
    """
    Build synthetic NBBO around base price (last or mid). Returns keys:
      synthetic_bid, synthetic_ask, synthetic_mid, synthetic_spread_pct
    """
    if not isinstance(base_px, (int, float)) or base_px <= 0:
        return {}
    spct = float(spread_pct if spread_pct is not None else _synth_spread_pct_default())
    # Convert % to +/- half around base
    half = spct / 200.0
    bid = base_px * (1 - half / 100.0)
    ask = base_px * (1 + half / 100.0)
    mid = (bid + ask) / 2.0
    return {
        "synthetic_bid": round(bid, 4),
        "synthetic_ask": round(ask, 4),
        "synthetic_mid": round(mid, 4),
        "synthetic_spread_pct": round((ask - bid) / mid * 100.0, 3) if mid > 0 else spct,
    }


def _adopt_synthetic_nbbo_if_missing(f: Dict[str, Any]) -> None:
    """
    If real NBBO is missing, adopt synthetic fields using best available base:
    mid → last → prev_close → theo_price. Records base source & provider.
    """
    # Already have NBBO? nothing to do
    if isinstance(f.get("bid"), (int, float)) and isinstance(f.get("ask"), (int, float)):
        return

    base = None
    base_src = None

    for key in ("mid", "last", "prev_close", "theo_price"):
        v = f.get(key)
        if isinstance(v, (int, float)) and v > 0:
            base = float(v)
            base_src = key
            break

    if base is None:
        f["synthetic_nbbo_attempted"] = True
        f.setdefault("nbbo_provider", "unavailable")
        return

    try:
        synth_min = float(os.getenv("SYNTH_MIN_BASE", "0.0"))
        base = max(base, synth_min)
    except Exception:
        pass

    synth = _make_synthetic_nbbo_from_base(base)
    for k, v in synth.items():
        f[k] = v

    f.setdefault("bid", f.get("synthetic_bid"))
    f.setdefault("ask", f.get("synthetic_ask"))
    f.setdefault("mid", f.get("synthetic_mid"))
    f.setdefault("option_spread_pct", f.get("synthetic_spread_pct"))

    f["synthetic_nbbo_used"] = True
    f["synthetic_nbbo_spread_est"] = synth.get("synthetic_spread_pct")
    f["synthetic_nbbo_base_src"] = base_src
    f.setdefault("nbbo_provider", f"synthetic(base={base_src})")


async def _try_multi_provider_nbbo(option_ticker: str, alert: Dict[str, Any], chosen_strike: float, chosen_expiry: str, f: Dict[str, Any]) -> None:
    """
    Optional multi-provider fallback:
      market_providers.get_nbbo_any -> (Polygon/IBKR/Tradier/Other) -> apply to f
      Only invoked if _should_use_multi_providers() is True.
    """
    if not _should_use_multi_providers():
        logger.info("multi-provider NBBO skipped (disabled or no provider creds found)")
        return

    try:
        from market_providers import get_nbbo_any, synthetic_from_last  # optional module
    except Exception:
        return

    if isinstance(f.get("bid"), (int, float)) and isinstance(f.get("ask"), (int, float)):
        return  # already have NBBO

    # Build context for the provider (OCC meta helps)
    occ = _occ_meta(option_ticker) or {}
    ctx = {
        "symbol": alert.get("symbol"),
        "right": "C" if (alert.get("side") or "").upper() == "CALL" else "P",
        "strike": float(chosen_strike),
        "expiry_yyyymmdd": (occ.get("expiry") or chosen_expiry.replace("-", "")),
        "last": f.get("last"),
    }

    try:
        alt = await get_nbbo_any(option_ticker, ctx)
    except Exception as e:
        logger.warning("get_nbbo_any failed: %r", e)
        alt = None

    if not alt:
        # Courtesy: try their synthetic helper if available
        try:
            syn = synthetic_from_last(f.get("last"))
        except Exception:
            syn = None
        if syn:
            f["nbbo_provider"] = syn.get("provider", "synthetic")
            f.setdefault("bid", syn.get("bid"))
            f.setdefault("ask", syn.get("ask"))
            f.setdefault("mid", syn.get("mid"))
            if f.get("option_spread_pct") is None and syn.get("spread_pct") is not None:
                f["option_spread_pct"] = syn.get("spread_pct")
            f["synthetic_nbbo_used"] = True
            f["synthetic_nbbo_spread_est"] = syn.get("synthetic_nbbo_spread_est")
        return

    # Apply provider NBBO
    f["nbbo_provider"] = alt.get("provider")
    if alt.get("bid") is not None: f["bid"] = float(alt["bid"])
    if alt.get("ask") is not None: f["ask"] = float(alt["ask"])
    if alt.get("mid") is not None: f["mid"] = float(alt["mid"])
    if alt.get("spread_pct") is not None: f["option_spread_pct"] = float(alt["spread_pct"])
    if alt.get("synthetic_nbbo_used"):
        f["synthetic_nbbo_used"] = True
        f["synthetic_nbbo_spread_est"] = alt.get("synthetic_nbbo_spread_est")


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
    ib_enabled = bool(job["flags"].get("ib_enabled", IBKR_ENABLED))
    force_buy  = bool(job["flags"].get("force_buy", False))
    qty        = int(job["flags"].get("qty", IBKR_DEFAULT_QTY))

    # Preserve originals for messaging
    orig_strike = alert.get("strike")
    orig_expiry = alert.get("expiry")

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

    pm = _build_plus_minus_contracts(alert["symbol"], ul_px, target_expiry)
    desired_strike = pm["strike_call"] if side == "CALL" else pm["strike_put"]

    # 3) Chain scan thresholds
    rth = _is_rth_now()
    scan_min_vol = SCAN_MIN_VOL_RTH if rth else SCAN_MIN_VOL_AH
    scan_min_oi  = SCAN_MIN_OI_RTH  if rth else SCAN_MIN_OI_AH

    # 3a) Selection via scan (strictly honor side)
    try:
        best_from_scan = await scan_for_best_contract_for_alert(
            client,
            alert["symbol"],
            {"side": side, "symbol": alert["symbol"], "strike": alert.get("strike"), "expiry": alert.get("expiry")},
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
                alert["symbol"],
                {"side": side, "symbol": alert["symbol"], "strike": alert.get("strike"), "expiry": alert.get("expiry")},
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
            # A) enrich
            extra = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
            for k, v in (extra or {}).items():
                if v is not None:
                    f[k] = v

            snap = await polygon_get_option_snapshot_export(get_http_client(), underlying=alert["symbol"], option_ticker=option_ticker)
            core = await build_features(get_http_client(), alert={**alert, "strike": desired_strike, "expiry": chosen_expiry}, snapshot=snap)
            for k, v in (core or {}).items():
                if v is not None or k not in f:
                    f[k] = v

            # B) derive mid/spread if NBBO present
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

            # C) aggressively ensure NBBO via Polygon (retry spinner + jitter)
            try:
                if f.get("bid") is None or f.get("ask") is None:
                    # jitter once before spinning
                    await asyncio.sleep(_rl_jitter_seconds())
                    tries = int(os.getenv("POLY_NBBO_TRIES", "12"))
                    delay = float(os.getenv("POLY_NBBO_DELAY", "0.35"))
                    nbbo = await ensure_nbbo(get_http_client(), option_ticker, tries=tries, delay=delay)
                    for k, v in (nbbo or {}).items():
                        if v is not None:
                            f[k] = v
            except Exception:
                pass

            # D) direct last-quote probe (Polygon)
            if f.get("bid") is None or f.get("ask") is None:
                try:
                    for k, v in (await _pull_nbbo_direct(option_ticker)).items():
                        if v is not None:
                            f[k] = v
                except Exception:
                    pass

            # D2) Multi-provider NBBO fallback (IBKR/Tradier/etc) if available & allowed
            if f.get("bid") is None or f.get("ask") is None:
                await _try_multi_provider_nbbo(option_ticker, alert, desired_strike, chosen_expiry, f)

            # E) ensure DTE
            if f.get("dte") is None:
                try:
                    f["dte"] = (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days
                except Exception:
                    pass

            # F) change vs prev close (option quote)
            if f.get("quote_change_pct") is None:
                try:
                    prev_close = f.get("prev_close")
                    mark = f.get("mid") if f.get("mid") is not None else f.get("last")
                    if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
                        f["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
                except Exception:
                    pass

            # G) if NBBO still missing adopt synthetic from best base
            _adopt_synthetic_nbbo_if_missing(f)

            # H) if still no spread but we have mid, set a soft spread (legacy fallback)
            if (f.get("option_spread_pct") is None) and isinstance(f.get("mid"), (int, float)):
                f["option_spread_pct"] = float(os.getenv("FALLBACK_SYNTH_SPREAD_PCT", "10.0"))

            # I) attach verbose NBBO status for debugging (Polygon probe)
            if f.get("bid") is None or f.get("ask") is None:
                try:
                    nbbo_dbg = await _probe_nbbo_verbose(option_ticker)
                    for k in ("bid", "ask", "mid", "option_spread_pct", "quote_age_sec"):
                        if nbbo_dbg.get(k) is not None:
                            f[k] = nbbo_dbg[k]
                    f["nbbo_http_status"] = nbbo_dbg.get("nbbo_http_status")
                    f["nbbo_reason"] = nbbo_dbg.get("nbbo_reason")
                    f["nbbo_body_sample"] = nbbo_dbg.get("nbbo_body_sample")
                except Exception:
                    pass

    except Exception as e:
        logger.exception("[worker] Polygon/features error: %s", e)
        f = f or {"dte": (datetime.fromisoformat(chosen_expiry).date() - datetime.now(timezone.utc).date()).days}

    # 4b) NBBO-driven replacement (listed but missing NBBO)
    if REPLACE_IF_NO_NBBO and (f.get("bid") is None or f.get("ask") is None or (f.get("nbbo_http_status") and f.get("nbbo_http_status") != 200)):
        try:
            alt = await _find_nbbo_replacement_same_expiry(
                symbol=alert["symbol"], side=side, desired_strike=desired_strike,
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
                extra2 = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
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
        exist = await _poly_reference_contracts_exists(alert["symbol"], chosen_expiry, option_ticker)
        logger.info("NBBO 404 verification: listed=%s snapshot_ok=%s reason=%s",
                    exist.get("listed"), exist.get("snapshot_ok"), exist.get("reason"))
        if exist.get("listed") is False and not exist.get("snapshot_ok"):
            repl = await _rescan_best_replacement(
                symbol=alert["symbol"], side=side,
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
                    extra2 = await poly_option_backfill(get_http_client(), alert["symbol"], option_ticker, datetime.now(timezone.utc).date())
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

    # Diff note (strike/expiry changes + provider + synthetic NBBO note)
    diff_bits = []
    if isinstance(orig_strike, (int, float)) and isinstance(desired_strike, (int, float)) and float(orig_strike) != float(desired_strike):
        diff_bits.append(f"🎯 Selected strike {desired_strike} (alert was {orig_strike})")
    if orig_expiry and chosen_expiry and str(orig_expiry) != str(chosen_expiry):
        diff_bits.append(f"🗓 Selected expiry {chosen_expiry} (alert was {orig_expiry})")
    if f.get("nbbo_provider") and not f.get("synthetic_nbbo_used"):
        diff_bits.append(f"📡 NBBO via {f.get('nbbo_provider')}")
    if f.get("synthetic_nbbo_used"):
        msg = f"🧪 Synthetic NBBO used ({f.get('synthetic_nbbo_spread_est')}% spread est.)"
        if f.get("synthetic_nbbo_base_src"):
            msg += f" [base={f.get('synthetic_nbbo_base_src')}]"
        diff_bits.append(msg)
    diff_note = "\n".join(diff_bits)

    # 8) Telegram final
    try:
        tg_text = compose_telegram_text(
            alert={**alert, "strike": desired_strike, "expiry": chosen_expiry},
            option_ticker=option_ticker, f=f, llm=llm, llm_ran=True, llm_reason="", score=score, rating=rating,
            diff_note=diff_note,
        )
      
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
                symbol=alert["symbol"], side=side,
                strike=float(desired_strike), expiry_iso=chosen_expiry,
                quantity=int(qty),
                limit_price=limit_px, action="BUY", tif=IBKR_TIF,
            )
    except Exception as e:
        ib_result_obj = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # 10) Decision log (guarded)
    try:
        _DECISIONS_LOG.append({
            "timestamp_local": market_now(),
            "symbol": alert["symbol"],
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
                "rsi14": f.get("rsi14"), "sma20": f.get("sma20"), "ema20": f.get("ema20"),
                "ema50": f.get("ema50"), "ema200": f.get("ema200"),
                "macd_line": f.get("macd_line"), "macd_signal": f.get("macd_signal"), "macd_hist": f.get("macd_hist"),
                "bb_upper": f.get("bb_upper"), "bb_lower": f.get("bb_lower"), "bb_mid": f.get("bb_mid"),
                "vwap": f.get("vwap"), "orb15_high": f.get("orb15_high"), "orb15_low": f.get("orb15_low"),
                "synthetic_nbbo_used": f.get("synthetic_nbbo_used"),
                "synthetic_nbbo_spread_est": f.get("synthetic_nbbo_spread_est"),
                "nbbo_provider": f.get("nbbo_provider"),
                "nbbo_http_status": f.get("nbbo_http_status"), "nbbo_reason": f.get("nbbo_reason"),
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
    except Exception as e:
        logger.warning("decision log append failed: %r", e)


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
