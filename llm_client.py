# llm_client.py
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List

try:
    from openai import AsyncOpenAI  # openai>=1.x
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger("trading_engine.llm")

# --- thresholds (env overridable) --------------------------------------------
BUY_THRESHOLD = float(os.getenv("LLM_BUY_THRESHOLD", "60"))
WAIT_THRESHOLD = float(os.getenv("LLM_WAIT_THRESHOLD", "45"))

# primary weights (sum ≈ 100 when fully normalized)
W_TECH   = float(os.getenv("LLM_W_TECH",   "60"))   # RSI/EMA/MACD/VWAP/Bollinger
W_STRUCT = float(os.getenv("LLM_W_STRUCT", "15"))   # ORB15 + regime/MTF
W_OPTION = float(os.getenv("LLM_W_OPTION", "15"))   # Delta + IV + OI/Vol tilt
W_EXEC   = float(os.getenv("LLM_W_EXEC",   "10"))   # Spread + synthetic NBBO relief

# model choice: stronger default (overridable)
DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini"))

# --- tiny helpers -------------------------------------------------------------

def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0

def _mk_reason(name: str, score_delta: float, why: str) -> str:
    s = f"+{score_delta:.1f}" if score_delta >= 0 else f"{score_delta:.1f}"
    return f"[{name} {s}] {why}"

def _multiline(reasons: List[str]) -> str:
    return "\n".join(reasons)

# --- indicator scoring --------------------------------------------------------

def _score_rsi(side: str, rsi14: Optional[float]) -> Tuple[float, str]:
    if rsi14 is None:
        return 0.0, _mk_reason("RSI", 0, "missing")
    if side == "CALL":
        if 40 <= rsi14 <= 65:
            return 7.0, _mk_reason("RSI", +7, f"constructive zone {rsi14:.1f}")
        if rsi14 < 30:
            return 4.0, _mk_reason("RSI", +4, "oversold (bounce potential)")
        if rsi14 > 70:
            return -4.0, _mk_reason("RSI", -4, "overbought (late risk)")
    else:  # PUT
        if 35 <= rsi14 <= 55:
            return 7.0, _mk_reason("RSI", +7, f"bearish/neutral zone {rsi14:.1f}")
        if rsi14 > 70:
            return 5.0, _mk_reason("RSI", +5, "overbought (mean reversion down)")
        if rsi14 < 30:
            return -4.0, _mk_reason("RSI", -4, "oversold (bounce risk)")
    return 0.0, _mk_reason("RSI", 0, f"uninformative {rsi14:.1f}")

def _price_vs(val: Optional[float], ref: Optional[float]) -> Optional[int]:
    if val is None or ref is None:
        return None
    return 1 if val > ref else (-1 if val < ref else 0)

def _score_ema_stack(side: str, price: Optional[float], ema20: Optional[float], ema50: Optional[float], ema200: Optional[float]) -> Tuple[float, str]:
    pts = 0.0
    notes: List[str] = []
    if price is not None:
        for name, ema in (("EMA20", ema20), ("EMA50", ema50), ("EMA200", ema200)):
            rel = _price_vs(price, ema)
            if rel is None:
                continue
            if side == "CALL":
                if rel > 0: pts += 3.3; notes.append(f"{name}↑")
                else:       pts -= 2.0; notes.append(f"{name}↓")
            else:
                if rel < 0: pts += 3.3; notes.append(f"{name}↓")
                else:       pts -= 2.0; notes.append(f"{name}↑")
    else:
        if ema20 and ema50 and ema200:
            if ema20 > ema50 > ema200:
                pts += 6.0; notes.append("20>50>200")
            elif ema20 < ema50 < ema200:
                pts += 6.0 if side == "PUT" else -3.0; notes.append("20<50<200")
    if not notes:
        return 0.0, _mk_reason("EMA stack", 0, "missing")
    return pts, _mk_reason("EMA stack", pts, ",".join(notes))

def _score_macd(side: str, macd_line: Optional[float], macd_signal: Optional[float], macd_hist: Optional[float]) -> Tuple[float, str]:
    if macd_line is None or macd_signal is None:
        return 0.0, _mk_reason("MACD", 0, "missing")
    pts = 0.0
    cross = macd_line - macd_signal
    if side == "CALL":
        if macd_line > 0 and cross > 0: pts += 6.0
        elif cross > 0:                  pts += 3.0
        elif macd_line < 0 and cross<0:  pts -= 3.0
    else:
        if macd_line < 0 and cross < 0:  pts += 6.0
        elif cross < 0:                  pts += 3.0
        elif macd_line > 0 and cross>0:  pts -= 3.0
    if macd_hist is not None:
        pts += _clip((macd_hist * (1 if side == "CALL" else -1)) * 4.0, -2.0, 2.0)
    return pts, _mk_reason("MACD", pts, f"macd={macd_line:.3f},sig={macd_signal:.3f}")

def _score_vwap(side: str, price: Optional[float], vwap: Optional[float], vwap_dist: Optional[float]) -> Tuple[float, str]:
    if vwap is None and vwap_dist is None:
        return 0.0, _mk_reason("VWAP", 0, "missing")
    dist = vwap_dist
    if dist is None and vwap is not None and price is not None:
        dist = _pct(price, vwap)
    if dist is None:
        return 0.0, _mk_reason("VWAP", 0, "missing")
    pts = _clip((dist if side == "CALL" else -dist) * 0.6, -3.0, 3.0)
    return pts, _mk_reason("VWAP", pts, f"dist≈{dist:.2f}%")

def _score_bollinger(side: str, price: Optional[float], bb_upper: Optional[float], bb_lower: Optional[float], sma20: Optional[float]) -> Tuple[float, str]:
    if price is None and sma20 is None and bb_upper is None and bb_lower is None:
        return 0.0, _mk_reason("Bollinger", 0, "missing")
    pts = 0.0
    notes: List[str] = []
    if price is not None and bb_upper is not None and price >= bb_upper:
        pts += (5.0 if side == "PUT" else -3.0); notes.append("touch U")
    if price is not None and bb_lower is not None and price <= bb_lower:
        pts += (5.0 if side == "CALL" else -3.0); notes.append("touch L")
    if price is not None and sma20 is not None:
        rel = _price_vs(price, sma20)
        if rel is not None:
            pts += (1.5 if (rel > 0 and side == "CALL") or (rel < 0 and side == "PUT") else -1.5)
    if not notes and sma20 is None:
        return 0.0, _mk_reason("Bollinger", 0, "uninformative")
    return pts, _mk_reason("Bollinger", pts, ",".join(notes) or "centerline")

def _score_orb15(side: str, price: Optional[float], orb_high: Optional[float], orb_low: Optional[float]) -> Tuple[float, str]:
    if price is None or (orb_high is None and orb_low is None):
        return 0.0, _mk_reason("ORB15", 0, "missing")
    pts = 0.0
    if orb_high is not None and price > orb_high: pts += (6.0 if side == "CALL" else -4.0)
    if orb_low  is not None and price < orb_low:  pts += (6.0 if side == "PUT"  else -4.0)
    if pts == 0.0:
        return 0.0, _mk_reason("ORB15", 0, "inside range")
    return pts, _mk_reason("ORB15", pts, "range break")

def _score_delta(side: str, delta: Optional[float]) -> Tuple[float, str]:
    if delta is None:
        return 0.0, _mk_reason("Delta", 0, "missing")
    pts = 6.0 - (abs(delta - (0.45 if side == "CALL" else -0.45)) / 0.15) * 3.0
    pts = _clip(pts, -3.0, 6.0)
    return pts, _mk_reason("Delta", pts, f"Δ={delta:.3f}")

def _score_iv(side: str, iv: Optional[float], iv_rank: Optional[float]) -> Tuple[float, str]:
    if iv is None and iv_rank is None:
        return 0.0, _mk_reason("IV", 0, "missing")
    pts = 0.0
    if iv_rank is not None:
        if 0.3 <= iv_rank <= 0.7: pts += 4.0
        elif iv_rank > 0.85:      pts -= 3.0
        elif iv_rank < 0.15:      pts -= 2.0
    return pts, _mk_reason("IV", pts, f"rank={iv_rank!r}")

def _score_liquidity(spread_pct: Optional[float], oi: Optional[float], vol: Optional[float], synthetic_nbbo_used: bool) -> Tuple[float, str]:
    pts = 0.0
    notes: List[str] = []
    if spread_pct is not None:
        if spread_pct <= 5:   pts += 5.0; notes.append("tight")
        elif spread_pct <=12: pts += 2.0; notes.append("ok")
        else:                 pts -= 3.0; notes.append("wide")
    if oi is not None:
        if oi >= 5000: pts += 2.0
        elif oi >= 500: pts += 1.0
        else: pts -= 1.0
    if vol is not None and vol <= 10:
        pts -= 1.0
    if synthetic_nbbo_used:
        pts += 1.5; notes.append("synthetic-ok")
    return pts, _mk_reason("Liquidity", pts, "/".join(notes) if notes else "n/a")

def _score_structure(mtf_align: Any, regime_flag: Any) -> Tuple[float, str]:
    pts = 0.0
    if isinstance(mtf_align, bool):
        pts += (4.0 if mtf_align else -2.0)
    if isinstance(regime_flag, str):
        rf = regime_flag.lower()
        if rf.startswith("trending"): pts += 3.0
        elif rf.startswith("choppy"): pts -= 2.0
    return pts, _mk_reason("Structure", pts, f"mtf={mtf_align},regime={regime_flag}")

def _score_dte(dte: Optional[float]) -> Tuple[float, str]:
    if dte is None:
        return 0.0, _mk_reason("DTE", 0, "missing")
    if dte <= 0:
        return -10.0, _mk_reason("DTE", -10.0, "0DTE high risk")
    pts = 0.0
    if dte < 2:            pts -= 3.0
    elif 5 <= dte <= 35:   pts += 3.0
    elif dte > 45:         pts -= 2.0
    return pts, _mk_reason("DTE", pts, f"{dte} days")

# --- rule-based blend (guardrails) -------------------------------------------

def _rule_blend(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
    side = (alert.get("side") or "").upper()
    ul_price = _f(alert.get("underlying_price_from_alert"))

    rsi_pts,   rsi_note   = _score_rsi(side, _f(f.get("rsi14")))
    ema_pts,   ema_note   = _score_ema_stack(side, ul_price, _f(f.get("ema20")), _f(f.get("ema50")), _f(f.get("ema200")))
    macd_line = _f(f.get("macd_line")) if f.get("macd_line") is not None else _f(f.get("macd"))
    macd_pts,  macd_note  = _score_macd(side, macd_line, _f(f.get("macd_signal")), _f(f.get("macd_hist")))
    vwap_pts,  vwap_note  = _score_vwap(side, ul_price, _f(f.get("vwap")), _f(f.get("vwap_dist")))
    boll_pts,  boll_note  = _score_bollinger(side, ul_price, _f(f.get("bb_upper")), _f(f.get("bb_lower")), _f(f.get("sma20")))

    tech_raw    = rsi_pts + ema_pts + macd_pts + vwap_pts + boll_pts
    tech_scaled = _clip(tech_raw / 25.0 * W_TECH, -W_TECH, W_TECH)

    orb_pts,    orb_note  = _score_orb15(side, ul_price, _f(f.get("orb15_high")), _f(f.get("orb15_low")))
    struct_pts, struct_note = _score_structure(f.get("mtf_align"), f.get("regime_flag"))
    struct_raw    = orb_pts + struct_pts
    struct_scaled = _clip(struct_raw / 10.0 * W_STRUCT, -W_STRUCT, W_STRUCT)

    delta_pts,   delta_note = _score_delta(side, _f(f.get("delta")))
    iv_pts,      iv_note    = _score_iv(side, _f(f.get("iv")), _f(f.get("iv_rank")))
    opt_raw    = delta_pts + iv_pts
    opt_scaled = _clip(opt_raw / 10.0 * W_OPTION, -W_OPTION, W_OPTION)

    liq_pts,     liq_note   = _score_liquidity(_f(f.get("option_spread_pct")), _f(f.get("oi")), _f(f.get("vol")), bool(f.get("synthetic_nbbo_used")))
    exec_scaled = _clip(liq_pts / 6.0 * W_EXEC, -W_EXEC, W_EXEC)

    dte_pts, dte_note = _score_dte(_f(f.get("dte")))

    score = 50.0 + tech_scaled + struct_scaled + opt_scaled + exec_scaled + dte_pts
    score = _clip(score, 0.0, 100.0)

    if score >= BUY_THRESHOLD:      decision = "buy"
    elif score >= WAIT_THRESHOLD:   decision = "wait"
    else:                           decision = "skip"

    conf = 0.5 + abs(score - 50.0) / 100.0
    conf = round(_clip(conf, 0.5, 0.95), 2)

    factor_lines = [rsi_note, ema_note, macd_note, vwap_note, boll_note, orb_note, struct_note, delta_note, iv_note, dte_note, liq_note]

    # quick human summary (baseline) if LLM is offline
    positives = [ln for ln in factor_lines if " +" in ln]
    negatives = [ln for ln in factor_lines if " -" in ln]
    base_summary = (
        f"{side} setup: blended score {score:.1f} → {decision.upper()} "
        f"with confidence {conf:.2f}. "
        f"Positives: {', '.join(p.split('] ')[1] for p in positives[:3]) or 'none'}. "
        f"Watchouts: {', '.join(n.split('] ')[1] for n in negatives[:3]) or 'none'}."
    )

    checklist = {
        "technicals": {
            "rsi14": _f(f.get("rsi14")),
            "ema20": _f(f.get("ema20")), "ema50": _f(f.get("ema50")), "ema200": _f(f.get("ema200")),
            "macd_line": macd_line, "macd_signal": _f(f.get("macd_signal")), "macd_hist": _f(f.get("macd_hist")),
            "vwap": _f(f.get("vwap")), "vwap_dist": _f(f.get("vwap_dist")),
            "bb_upper": _f(f.get("bb_upper")), "bb_lower": _f(f.get("bb_lower")), "sma20": _f(f.get("sma20")),
            "orb15_high": _f(f.get("orb15_high")), "orb15_low": _f(f.get("orb15_low")),
            "ta_src": f.get("ta_src"),
        },
        "options": {
            "delta": _f(f.get("delta")), "gamma": _f(f.get("gamma")), "theta": _f(f.get("theta")), "vega": _f(f.get("vega")),
            "iv": _f(f.get("iv")), "iv_rank": _f(f.get("iv_rank")), "oi": _f(f.get("oi")), "vol": _f(f.get("vol")),
            "spread_pct": _f(f.get("option_spread_pct")),
            "synthetic_nbbo_used": bool(f.get("synthetic_nbbo_used", False)),
        },
        "structure": {
            "dte": _f(f.get("dte")), "mtf_align": f.get("mtf_align"), "regime_flag": f.get("regime_flag"),
        }
    }

    return {
        "decision": decision,
        "confidence": conf,
        "reason": base_summary + "\n" + _multiline(factor_lines),
        "checklist": checklist,
        "ev_estimate": {"edge_pct": round((score - 50.0) / 1.5, 2), "score": round(score, 2)},
        "factor_lines": factor_lines,     # expose for LLM refine prompt
        "base_summary": base_summary,     # expose for LLM refine prompt
    }

# --- OpenAI wrapper (refine within guardrails) --------------------------------

def _slim_features(f: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "rsi14","sma20","ema20","ema50","ema200",
        "macd_line","macd","macd_signal","macd_hist",
        "vwap","vwap_dist","bb_upper","bb_lower","bb_width",
        "orb15_high","orb15_low","mtf_align","regime_flag","dte",
        "delta","gamma","theta","vega","iv","iv_rank","oi","vol",
        "bid","ask","mid","option_spread_pct",
        "synthetic_nbbo_used","synthetic_nbbo_spread_est",
        "prev_close","quote_change_pct","nbbo_http_status","nbbo_reason","ta_src"
    ]
    return {k: f[k] for k in keys if k in f}

async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: { decision, confidence, reason, checklist, ev_estimate }
    - Deterministic weighted blend (RSI/EMA(20/50/200)/MACD/VWAP/Bollinger/ORB15 + Greeks/IV + structure + execution).
    - Stronger default model (gpt-4.1) produces: human 'summary' (2–4 sentences) + factor bullets; total adjustment ≤ ±10 pts.
    - Missing NBBO never auto-skips; if synthetic_nbbo_used=true, apply cautious-but-acceptable execution treatment.
    """
    baseline = _rule_blend(alert, features)

    # If OpenAI missing/offline, return baseline (already human-friendly)
    if AsyncOpenAI is None or not os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OFFLINE", "0") == "1":
        if "(offline rule-based)" not in baseline["reason"]:
            baseline["reason"] += "\n(offline rule-based)"
        return baseline

    try:
        client = AsyncOpenAI()
        model = DEFAULT_MODEL

        system_msg = (
            "You are an options-trading analyst. Produce a clear, human summary (2–4 sentences) explaining the trade call. "
            "Weigh: RSI(14), EMA(20/50/200) stack, MACD (line/signal/hist), VWAP distance, Bollinger(20,2σ), 15m ORB, "
            "Greeks (Δ/Θ/ν) & IV/IV Rank, plus execution (spread, OI/Vol). "
            "Use the provided baseline weighted score; you may adjust modestly (<= ±10 absolute points) only with strong justification. "
            "Do NOT auto-skip just because NBBO is missing; if synthetic_nbbo_used=true, treat execution as cautious but acceptable. "
            "Return JSON with: decision, confidence (0..1), summary (string), factors (array of short bullet strings), checklist (object), ev_estimate (object)."
        )

        # Build a compact payload the model can reason about
        user_payload = {
            "alert": {
                "side": alert.get("side"),
                "symbol": alert.get("symbol"),
                "underlying_price_from_alert": alert.get("underlying_price_from_alert"),
                "strike": alert.get("strike"),
                "expiry": alert.get("expiry"),
            },
            "features": _slim_features(features),
            "baseline": {
                "decision": baseline["decision"],
                "confidence": baseline["confidence"],
                "score": baseline.get("ev_estimate", {}).get("score"),
                "summary": baseline.get("base_summary"),
                "factors": baseline.get("factor_lines"),
            },
            "boundaries": {"buy_threshold": BUY_THRESHOLD, "wait_threshold": WAIT_THRESHOLD, "max_adjust_abs_points": 10}
        }

        resp = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Refine the baseline. Keep adjustments ≤ ±10 pts. "
                        "Include a concise 'summary' and 'factors' list (bullets like '[RSI +3.0] ...'). "
                        "Be conservative with liquidity penalties if synthetic_nbbo_used=true.\n\n"
                        + json.dumps(user_payload, ensure_ascii=False)
                    ),
                },
            ],
        )

        txt = resp.choices[0].message.content
        parsed = json.loads(txt)

        # sanitize
        decision = str(parsed.get("decision") or baseline["decision"]).lower()
        if decision not in ("buy", "wait", "skip"):
            decision = baseline["decision"]

        confidence = parsed.get("confidence")
        try:
            confidence = float(confidence)
            confidence = _clip(confidence, 0.0, 1.0)
        except Exception:
            confidence = baseline["confidence"]

        # combine human summary + bullets into the single 'reason' string our app displays
        summary = parsed.get("summary") or baseline.get("base_summary") or ""
        factors = parsed.get("factors") or baseline.get("factor_lines") or []
        if isinstance(factors, list):
            factors_block = "\n".join(factors)
        else:
            factors_block = str(factors)
        reason = (summary.strip() + ("\n" if summary else "") + factors_block).strip()

        checklist = parsed.get("checklist") or baseline.get("checklist") or {}
        ev_estimate = parsed.get("ev_estimate") or baseline.get("ev_estimate") or {}

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "checklist": checklist,
            "ev_estimate": ev_estimate,
        }

    except Exception as e:
        logger.exception("LLM refine failed: %s", e)
        out = dict(baseline)
        add = f"(LLM refine error: {type(e).__name__}: {e})"
        out["reason"] = (out.get("reason","") + ("\n" if out.get("reason") else "") + add).strip()
        return out
