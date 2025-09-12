# llm_client.py
import os
import json
import math
from typing import Dict, Any, Tuple

from openai import AsyncOpenAI

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_client = AsyncOpenAI(api_key=_OPENAI_API_KEY)

# ---- helpers ---------------------------------------------------------------
def _clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def _nz(x, default=None):
    return x if x is not None else default

def _mk_bucket_scores(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, float]:
    """
    Lightweight numeric priors used as hints to the LLM (not hard gating).
    Just to seed a baseline and make outputs more stable.
    """
    scores = {
        "trend_technicals": 0.0,   # RSI/EMA/SMA/MACD/Δ%
        "volatility_iv": 0.0,      # IV, IV rank, RV
        "liquidity_exec": 0.0,     # NBBO, spread, quote age (NOT hard fail)
        "flow_chain": 0.0,         # OI/Volume on contract
        "time_decay": 0.0,         # DTE, Theta
        "price_value": 0.0,        # mark vs prev_close, greeks magnitude consistency
        "risk_regime": 0.0,        # regime_flag, MTF align, S/R headroom, VWAP distance
        "events_actions": 0.0,     # corporate actions / reference risk
        "microstructure": 0.0,     # minute/second aggs coverage
    }

    # Trend / Technicals
    rsi = f.get("rsi14"); ema20 = f.get("ema20"); sma20 = f.get("sma20")
    macd = f.get("macd"); macd_sig = f.get("macd_signal")
    delta_pct = f.get("delta_pct")
    t_score = 0.0
    if isinstance(rsi, (int, float)):
        # favor healthy momentum; avoid extreme overbought/oversold unless side fits
        if 45 <= rsi <= 65: t_score += 0.4
        if alert["side"] == "CALL" and rsi > 50: t_score += 0.2
        if alert["side"] == "PUT" and rsi < 50:  t_score += 0.2
    if isinstance(macd, (int, float)) and isinstance(macd_sig, (int, float)):
        if (macd - macd_sig) > 0: t_score += 0.2
    if isinstance(delta_pct, (int, float)):
        # small positive Δ% helps calls; small negative helps puts
        if alert["side"] == "CALL" and delta_pct > 0: t_score += 0.1
        if alert["side"] == "PUT"  and delta_pct < 0: t_score += 0.1
    scores["trend_technicals"] = _clip01(t_score)

    # Volatility / IV
    iv = f.get("iv"); iv_rank = f.get("iv_rank")
    v_score = 0.0
    if isinstance(iv, (int, float)):
        # moderate IV preferred for buying premium; too high hurts
        if 0.15 <= iv <= 0.45: v_score += 0.4
        elif iv < 0.15:        v_score += 0.2
        else:                  v_score += 0.1
    if isinstance(iv_rank, (int, float)):
        # medium IV rank better for buying
        if 0.2 <= iv_rank <= 0.6: v_score += 0.3
        elif iv_rank < 0.2:       v_score += 0.2
        else:                     v_score += 0.1
    scores["volatility_iv"] = _clip01(v_score)

    # Liquidity / Execution (soft influence)
    spread = f.get("option_spread_pct"); age = f.get("quote_age_sec")
    l_score = 0.3  # default soft baseline so we DON'T auto-fail on NBBO
    if isinstance(spread, (int, float)):
        if spread <= 5:   l_score += 0.4
        elif spread <=10: l_score += 0.2
    if isinstance(age, (int, float)):
        if age <= 60:     l_score += 0.2
        elif age <= 300:  l_score += 0.1
    scores["liquidity_exec"] = _clip01(l_score)

    # Flow / Chain (OI & Volume)
    oi = f.get("oi"); vol = f.get("vol")
    c_score = 0.0
    if isinstance(oi, (int, float)) and oi > 0:
        if oi >= 1000: c_score += 0.4
        elif oi >= 200: c_score += 0.2
    if isinstance(vol, (int, float)) and vol > 0:
        if vol >= 1000: c_score += 0.4
        elif vol >= 200: c_score += 0.2
    scores["flow_chain"] = _clip01(c_score)

    # Time / Decay (NOT a hard “no”)
    dte = f.get("dte"); theta = f.get("theta")
    td_score = 0.3  # baseline
    if isinstance(dte, (int, float)):
        if 5 <= dte <= 30: td_score += 0.4
        elif dte > 30:     td_score += 0.2
        else:              td_score += 0.1   # very short still possible for momentum/lotto
    if isinstance(theta, (int, float)):
        # less negative theta is “less decay”
        if theta > -0.2:   td_score += 0.2
    scores["time_decay"] = _clip01(td_score)

    # Price / Value vs previous close & greeks magnitude
    mark = f.get("mid", f.get("last"))
    prev = f.get("prev_close")
    pv_score = 0.3
    if isinstance(mark, (int, float)) and isinstance(prev, (int, float)) and prev > 0:
        chg = (mark - prev) / prev
        # modest green helps calls; modest red helps puts
        if alert["side"] == "CALL" and chg > 0: pv_score += 0.2
        if alert["side"] == "PUT"  and chg < 0: pv_score += 0.2
    # delta (option sensitivity) — not too tiny
    delta = f.get("delta")
    if isinstance(delta, (int, float)):
        if alert["side"] == "CALL" and delta >= 0.25: pv_score += 0.2
        if alert["side"] == "PUT"  and delta <= -0.25: pv_score += 0.2
    scores["price_value"] = _clip01(pv_score)

    # Risk / Regime
    rr_score = 0.3
    if f.get("regime_flag") == "trending": rr_score += 0.2
    if f.get("mtf_align"): rr_score += 0.2
    if f.get("sr_headroom_ok"): rr_score += 0.2
    scores["risk_regime"] = _clip01(rr_score)

    # Events & Corporate Actions (neutral if unknown)
    ev_score = 0.5
    actions = f.get("corporate_actions") or {}
    # if known dilutive / splits / earnings today AH → nudge down
    if actions.get("earnings_soon"): ev_score -= 0.1
    if actions.get("split_soon"):    ev_score -= 0.05
    if actions.get("dividend_soon"): ev_score -= 0.05
    scores["events_actions"] = _clip01(ev_score)

    # Microstructure coverage (minute/second aggregates presence)
    micro = 0.3
    counts = f.get("aggs_counts") or {}
    if counts.get("1m", 0) >= 50: micro += 0.2
    if counts.get("5m", 0) >= 20: micro += 0.2
    # if you add 1s/5s counts later, bump here
    scores["microstructure"] = _clip01(micro)

    return scores

def _prep_llm_payload(alert: Dict[str, Any], f: Dict[str, Any]) -> Tuple[str, str]:
    """
    Create a compact JSON payload + a rubric that *forces* consideration of
    EMA/RSI/MACD, IV/IV Rank, Greeks/OI, aggregates, snapshot, trades & events.
    """
    buckets = _mk_bucket_scores(alert, f)

    forced_considerations = {
        "trend_technicals": {
            "RSI14": f.get("rsi14"),
            "EMA20": f.get("ema20"),
            "SMA20": f.get("sma20"),
            "MACD": f.get("macd"),
            "MACD_signal": f.get("macd_signal"),
            "MACD_hist": f.get("macd_hist"),
            "delta_pct": f.get("delta_pct"),
            "sources": {
                "rsi": f.get("rsi_source"),
                "ema": f.get("ema_source"),
                "sma": f.get("sma_source"),
                "macd": f.get("macd_source"),
                "delta_pct": f.get("delta_pct_source"),
            },
        },
        "volatility_iv": {
            "IV": f.get("iv"),
            "IV_rank": f.get("iv_rank"),
            "RV20": f.get("rv20"),
        },
        "liquidity_exec": {
            "bid": f.get("bid"), "ask": f.get("ask"),
            "mark": f.get("mid", f.get("last")),
            "spread_pct": f.get("option_spread_pct"),
            "quote_age_sec": f.get("quote_age_sec"),
            "nbbo_http_status": f.get("nbbo_http_status"),
            "nbbo_reason": f.get("nbbo_reason"),
        },
        "flow_chain": {
            "OI": f.get("oi"),
            "Vol": f.get("vol"),
        },
        "time_decay": {
            "DTE": f.get("dte"),
            "Theta": f.get("theta"),
        },
        "price_value": {
            "mark": f.get("mid", f.get("last")),
            "prev_close": f.get("prev_close"),
            "delta": f.get("delta"),
            "gamma": f.get("gamma"),
            "vega": f.get("vega"),
        },
        "risk_regime": {
            "regime_flag": f.get("regime_flag"),
            "mtf_align": f.get("mtf_align"),
            "sr_headroom_ok": f.get("sr_headroom_ok"),
            "vwap": f.get("vwap"),
            "vwap_dist": f.get("vwap_dist"),
        },
        "events_actions": {
            # populate from your polygon corporate-actions fetcher when available
            "corporate_actions": f.get("corporate_actions"),
            "reference_data_notes": f.get("reference_data"),
        },
        "microstructure": {
            "aggs_status": f.get("aggs_status"),
            "aggs_counts": f.get("aggs_counts"),
            "bars_source": f.get("bars_source"),
            "last_close": f.get("last_close"),
            # add second-aggregates coverage if you gather it:
            "second_aggs": f.get("second_aggs_meta"),
            # and trades (if you fetch): last trade price/size/conditions
            "trades_meta": f.get("trades_meta"),
        },
    }

    user_payload = {
        "alert": alert,
        "forced_considerations": forced_considerations,
        "soft_priors_bucket_scores": buckets,
        "side": alert.get("side"),
        "symbol": alert.get("symbol"),
    }

    rubric = (
        "You are an options trading analyst. Produce a JSON decision using the rubric:\n"
        "Buckets & nominal weights (sum ~100):\n"
        "  - trend_technicals (20)\n"
        "  - volatility_iv (15)\n"
        "  - liquidity_exec (15)  # DO NOT auto-reject purely for NBBO/DTE\n"
        "  - flow_chain (15)\n"
        "  - time_decay (10)      # DTE is NOT a hard fail; short-dated momentum can justify buys\n"
        "  - price_value (10)\n"
        "  - risk_regime (10)\n"
        "  - events_actions (3)\n"
        "  - microstructure (2)\n"
        "MANDATORY: Deliberately reference EMA/RSI/MACD, IV & IV_rank, Greeks (delta/gamma/theta/vega), "
        "OI & volume, aggregates (minute/second) coverage, snapshot/last, trades (if present), "
        "and corporate/reference actions (if present). If any are missing, treat as neutral (not negative).\n"
        "Output JSON ONLY with keys: decision ('buy'|'wait'|'skip'), confidence [0,1], reason, "
        "scores (per-bucket 0..1), factors (brief dict of key inputs you used). No prose outside JSON."
    )
    return json.dumps(user_payload, separators=(",", ":")), rubric

# ---- public API ------------------------------------------------------------
async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the LLM with a rubric that forces consideration of technicals, IV, greeks/OI,
    aggregates & snapshot/trades/events. No hard gating on liquidity or DTE.
    """
    if not _OPENAI_API_KEY:
        # Fallback if no key: very conservative heuristic
        buckets = _mk_bucket_scores(alert, features)
        decision = "wait"
        conf = 0.35
        reason = "OPENAI_API_KEY missing; returned heuristic WAIT using multi-bucket hints (no hard NBBO/DTE gating)."
        return {
            "decision": decision,
            "confidence": conf,
            "reason": reason,
            "scores": buckets,
            "factors": {}
        }

    user_json, rubric = _prep_llm_payload(alert, features)

    msgs = [
        {
            "role": "system",
            "content": (
                "You are a meticulous options trading analyst. Use ALL provided dimensions. "
                "Do NOT auto-reject solely due to missing NBBO or short DTE; weigh everything."
            ),
        },
        {
            "role": "user",
            "content": rubric + "\n\nDATA:\n" + user_json,
        },
    ]

    try:
        resp = await _client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=msgs,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "800")),
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        # normalize
        dec = (data.get("decision") or "").lower()
        if dec not in ("buy", "wait", "skip"):
            dec = "wait"
        conf = _clip01(float(_nz(data.get("confidence"), 0.5)))
        reason = str(_nz(data.get("reason"), ""))
        scores = data.get("scores") or {}
        factors = data.get("factors") or {}
        return {
            "decision": dec,
            "confidence": conf,
            "reason": reason,
            "scores": scores,
            "factors": factors,
        }
    except Exception as e:
        # robust fallback: still *multi-bucket*, not liquidity/DTE only
        buckets = _mk_bucket_scores(alert, features)
        return {
            "decision": "wait",
            "confidence": 0.4,
            "reason": f"LLM error fallback: {type(e).__name__}: {e}",
            "scores": buckets,
            "factors": {},
        }
