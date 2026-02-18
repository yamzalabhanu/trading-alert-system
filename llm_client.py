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

BUY_THRESHOLD = float(os.getenv("LLM_BUY_THRESHOLD", "60"))
WAIT_THRESHOLD = float(os.getenv("LLM_WAIT_THRESHOLD", "45"))

W_TECH    = float(os.getenv("LLM_W_TECH",    "60"))
W_STRUCT  = float(os.getenv("LLM_W_STRUCT",  "15"))
W_OPTION  = float(os.getenv("LLM_W_OPTION",  "15"))
W_EXEC    = float(os.getenv("LLM_W_EXEC",    "10"))
W_CONTEXT = float(os.getenv("LLM_W_CONTEXT", "10"))

SHORT_VOL_PUT_BONUS_HI   = float(os.getenv("LLM_SV_PUT_BONUS_HI",   "1.5"))
SHORT_VOL_PUT_BONUS_MED  = float(os.getenv("LLM_SV_PUT_BONUS_MED",  "1.0"))
SHORT_VOL_PUT_BONUS_LO   = float(os.getenv("LLM_SV_PUT_BONUS_LO",   "0.5"))
SHORT_VOL_CALL_PEN_HI    = float(os.getenv("LLM_SV_CALL_PEN_HI",    "-1.0"))
SHORT_VOL_CALL_PEN_MED   = float(os.getenv("LLM_SV_CALL_PEN_MED",   "-0.6"))
SHORT_VOL_CALL_PEN_LO    = float(os.getenv("LLM_SV_CALL_PEN_LO",    "-0.3"))
SHORT_VOL_CALL_BONUS_LOW = float(os.getenv("LLM_SV_CALL_BONUS_LOW", "0.5"))
SHORT_VOL_PUT_PEN_LOW    = float(os.getenv("LLM_SV_PUT_PEN_LOW",    "-0.3"))

SHORT_INT_CALL_BONUS = float(os.getenv("LLM_SI_CALL_BONUS", "0.3"))
SHORT_INT_PUT_PEN    = float(os.getenv("LLM_SI_PUT_PEN",   "-0.2"))

DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1"))

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

# --- scoring pieces (kept from your version) ---------------------------------
# (I’m keeping them as-is except: DTE now neutral if missing, and new TV meta scorer.)

def _score_dte(dte: Optional[float]) -> Tuple[float, str]:
    if dte is None:
        return 0.0, _mk_reason("DTE", 0, "missing (equity-only ok)")
    if dte <= 0:
        return -10.0, _mk_reason("DTE", -10.0, "0DTE high risk")
    pts = 0.0
    if dte < 2:
        pts -= 3.0
    elif 5 <= dte <= 35:
        pts += 3.0
    elif dte > 45:
        pts -= 2.0
    return pts, _mk_reason("DTE", pts, f"{dte} days")

def _score_tv_meta(side: str, tv_meta: Dict[str, Any]) -> Tuple[float, str]:
    pts = 0.0
    notes: List[str] = []

    adx = _f(tv_meta.get("adx"))
    relvol = _f(tv_meta.get("relVol") if tv_meta.get("relVol") is not None else tv_meta.get("relvol"))
    chop = _f(tv_meta.get("chop"))
    event = str(tv_meta.get("event") or "").lower().strip()

    ADX_STRONG = float(os.getenv("LLM_META_ADX_STRONG", "22"))
    RELVOL_STRONG = float(os.getenv("LLM_META_RELVOL_STRONG", "1.3"))
    CHOP_MAX = float(os.getenv("LLM_META_CHOP_MAX", "55"))

    if adx is not None:
        if adx >= ADX_STRONG:
            pts += 2.0; notes.append(f"ADX {adx:.1f} trending")
        elif adx <= 15:
            pts -= 1.0; notes.append(f"ADX {adx:.1f} weak")

    if relvol is not None:
        if relvol >= RELVOL_STRONG:
            pts += 2.0; notes.append(f"RelVol {relvol:.2f} strong")
        elif relvol < 1.0:
            pts -= 0.5; notes.append(f"RelVol {relvol:.2f} soft")

    if chop is not None:
        if chop >= CHOP_MAX:
            pts -= 2.0; notes.append(f"CHOP {chop:.1f} choppy")
        elif chop <= 45:
            pts += 0.5; notes.append(f"CHOP {chop:.1f} ok")

    if event == "exit":
        pts -= 3.0; notes.append("EXIT event")

    return pts, _mk_reason("TV Meta", pts, ", ".join(notes) if notes else "neutral")


# NOTE: For brevity I’m not repeating every single helper from your long file here.
# Keep ALL your existing functions unchanged:
# _score_rsi, _score_ema_stack, _score_macd, _score_vwap, _score_bollinger, _score_orb15,
# _score_delta, _score_iv, _score_liquidity, _score_structure, _score_short_flow, _score_context, _infer_horizon
#
# The only REQUIRED edits inside _rule_blend / _slim_features / _alert_payload / analyze_with_openai are below.

# --- rule-based blend ---------------------------------------------------------
def _rule_blend(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
    # Import your existing scoring functions from your current file (keep them)
    # Here we assume they exist in this module exactly like your version.
    side = (alert.get("side") or "").upper()
    ul_price = _f(alert.get("underlying_price_from_alert"))

    # --- KEEP your existing scoring calls exactly as you have them ---
    rsi_pts,   rsi_note   = _score_rsi(side, _f(f.get("rsi14")))
    ema_pts,   ema_note   = _score_ema_stack(side, ul_price, _f(f.get("ema20")), _f(f.get("ema50")), _f(f.get("ema200")))
    macd_line = _f(f.get("macd_line")) if f.get("macd_line") is not None else _f(f.get("macd"))
    macd_pts,  macd_note  = _score_macd(side, macd_line, _f(f.get("macd_signal")), _f(f.get("macd_hist")))
    vwap_pts,  vwap_note  = _score_vwap(side, ul_price, _f(f.get("vwap")), _f(f.get("vwap_dist")))
    boll_pts,  boll_note  = _score_bollinger(side, ul_price, _f(f.get("bb_upper")), _f(f.get("bb_lower")), _f(f.get("sma20")))
    tech_raw    = rsi_pts + ema_pts + macd_pts + vwap_pts + boll_pts
    tech_scaled = _clip(tech_raw / 25.0 * W_TECH, -W_TECH, W_TECH)

    orb_pts,      orb_note    = _score_orb15(side, ul_price, _f(f.get("orb15_high")), _f(f.get("orb15_low")))
    struct_pts,   struct_note = _score_structure(f.get("mtf_align"), f.get("regime_flag"))
    struct_raw    = orb_pts + struct_pts
    struct_scaled = _clip(struct_raw / 10.0 * W_STRUCT, -W_STRUCT, W_STRUCT)

    delta_pts,   delta_note = _score_delta(side, _f(f.get("delta")))
    iv_pts,      iv_note    = _score_iv(side, _f(f.get("iv")), _f(f.get("iv_rank")))
    opt_raw    = delta_pts + iv_pts
    opt_scaled = _clip(opt_raw / 10.0 * W_OPTION, -W_OPTION, W_OPTION)

    liq_pts,     liq_note   = _score_liquidity(_f(f.get("option_spread_pct")), _f(f.get("oi")), _f(f.get("vol")), bool(f.get("synthetic_nbbo_used")))
    exec_scaled = _clip(liq_pts / 6.0 * W_EXEC, -W_EXEC, W_EXEC)

    ctx_pts, ctx_notes = _score_context(side, ul_price, f)
    ctx_scaled = _clip(ctx_pts / 4.0 * W_CONTEXT, -W_CONTEXT, W_CONTEXT)

    dte_pts, dte_note = _score_dte(_f(f.get("dte")))

    # NEW: tv_meta light scoring lane
    tv_meta = f.get("tv_meta") if isinstance(f.get("tv_meta"), dict) else {}
    tv_pts, tv_note = _score_tv_meta(side, tv_meta)
    tv_scaled = _clip(tv_pts / 4.0 * 6.0, -6.0, 6.0)

    score = 50.0 + tech_scaled + struct_scaled + opt_scaled + exec_scaled + ctx_scaled + dte_pts + tv_scaled
    score = _clip(score, 0.0, 100.0)

    event = str(tv_meta.get("event") or alert.get("event") or "").lower().strip()
    if event == "exit" and score >= BUY_THRESHOLD:
        score = min(score, BUY_THRESHOLD - 0.1)

    decision = "buy" if score >= BUY_THRESHOLD else ("wait" if score >= WAIT_THRESHOLD else "skip")
    if event == "exit" and decision == "buy":
        decision = "wait"

    conf = 0.5 + abs(score - 50.0) / 100.0
    conf = round(_clip(conf, 0.5, 0.95), 2)

    factor_lines = [
        rsi_note, ema_note, macd_note, vwap_note, boll_note,
        orb_note, struct_note, delta_note, iv_note, dte_note, liq_note,
        tv_note,
    ] + ctx_notes

    positives = [ln for ln in factor_lines if " +" in ln]
    negatives = [ln for ln in factor_lines if " -" in ln]
    base_summary = (
        f"{side} setup: blended score {score:.1f} → {decision.upper()} "
        f"with confidence {conf:.2f}. "
        f"Positives: {', '.join(p.split('] ')[1] for p in positives[:3]) or 'none'}. "
        f"Watchouts: {', '.join(n.split('] ')[1] for n in negatives[:3]) or 'none'}."
    )

    horizon, horizon_score, horizon_reason = _infer_horizon(alert, f)

    return {
        "decision": decision,
        "confidence": conf,
        "reason": base_summary + "\n" + _multiline(factor_lines),
        "checklist": baseline_checklist(alert, f, macd_line, horizon, horizon_score, horizon_reason),
        "ev_estimate": {"edge_pct": round((score - 50.0) / 1.5, 2), "score": round(score, 2)},
        "factor_lines": factor_lines,
        "base_summary": base_summary,
        "horizon": horizon,
        "horizon_reason": horizon_reason,
    }


def _slim_features(f: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "rsi14","sma20","ema20","ema50","ema200",
        "macd_line","macd","macd_signal","macd_hist",
        "vwap","vwap_dist","bb_upper","bb_lower","bb_width",
        "orb15_high","orb15_low","mtf_align","regime_flag","dte",
        "delta","gamma","theta","vega","iv","iv_rank","oi","vol",
        "bid","ask","mid","option_spread_pct",
        "synthetic_nbbo_used","synthetic_nbbo_spread_est",
        "prev_open","prev_high","prev_low","prev_close",
        "quote_change_pct","prev5_avg_high","prev5_avg_low",
        "premarket_high","premarket_low",
        "short_volume","short_interest","short_volume_total","short_volume_ratio",
        "ta_src","data_provider","nbbo_provider",
        "tv_meta","mtf","bars_meta","atr14_daily","atr14_pct",
    ]
    return {k: f[k] for k in keys if k in f}


def _alert_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "side": alert.get("side"),
        "symbol": alert.get("symbol"),
        "underlying_price_from_alert": alert.get("underlying_price_from_alert"),
    }
    if alert.get("strike") is not None:
        out["strike"] = alert.get("strike")
    if alert.get("expiry"):
        out["expiry"] = alert.get("expiry")
    for k in ("source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level", "adx", "relVol", "chop"):
        if alert.get(k) is not None:
            out[k] = alert.get(k)
    return out


async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    baseline = _rule_blend(alert, features)

    if AsyncOpenAI is None or not os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OFFLINE", "0") == "1":
        if "(offline rule-based)" not in baseline["reason"]:
            baseline["reason"] = (
                f"🕒 Suggested trade: {baseline['horizon'].upper()} — {baseline['horizon_reason']}\n"
                + baseline["reason"]
                + "\n(offline rule-based)"
            ).strip()
        return baseline

    try:
        client = AsyncOpenAI()
        model = DEFAULT_MODEL

        system_msg = (
            "You are an options/intraday trading analyst. Produce a clear, human summary (2–4 sentences). "
            "If strike/expiry/Greeks are missing, treat as equity-only. "
            "Use TradingView meta (ADX/RelVol/CHOP/event/reason/confirm_tf) when present. "
            "You may adjust baseline score by <= ±10 points only with strong justification. "
            "If event='exit', you MUST NOT recommend 'buy'. "
            "Return JSON: decision, confidence(0..1), summary, factors[], horizon(intraday|swing), horizon_reason, checklist, ev_estimate."
        )

        user_payload = {
            "alert": _alert_payload(alert),
            "features": _slim_features(features),
            "baseline": {
                "decision": baseline["decision"],
                "confidence": baseline["confidence"],
                "score": baseline.get("ev_estimate", {}).get("score"),
                "summary": baseline.get("base_summary"),
                "factors": baseline.get("factor_lines"),
                "horizon": baseline.get("horizon"),
                "horizon_reason": baseline.get("horizon_reason"),
            },
            "boundaries": {"buy_threshold": BUY_THRESHOLD, "wait_threshold": WAIT_THRESHOLD, "max_adjust_abs_points": 10},
        }

        resp = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )

        txt = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(txt) if txt else {}
        except Exception:
            logger.warning("LLM returned non-JSON; falling back to baseline. txt=%r", txt[:500])
            parsed = {}

        decision = str(parsed.get("decision") or baseline["decision"]).lower()
        if decision not in ("buy", "wait", "skip"):
            decision = baseline["decision"]

        # exit safety
        event = str((features.get("tv_meta") or {}).get("event") or alert.get("event") or "").lower().strip()
        if event == "exit" and decision == "buy":
            decision = "wait"

        confidence = parsed.get("confidence")
        try:
            confidence = float(confidence)
            confidence = _clip(confidence, 0.0, 1.0)
        except Exception:
            confidence = baseline["confidence"]

        summary = (parsed.get("summary") or baseline.get("base_summary") or "").strip()
        factors = parsed.get("factors") or baseline.get("factor_lines") or []
        if isinstance(factors, list):
            factors_block = "\n".join(str(x) for x in factors)
        else:
            factors_block = str(factors)

        horizon = str(parsed.get("horizon") or baseline.get("horizon") or "intraday").lower()
        if horizon not in ("intraday", "swing"):
            horizon = baseline.get("horizon", "intraday")

        horizon_reason = (parsed.get("horizon_reason") or baseline.get("horizon_reason") or "neutral mix").strip()

        reason = (
            f"🕒 Suggested trade: {horizon.upper()} — {horizon_reason}\n"
            + (summary + ("\n" if summary else ""))
            + factors_block
        ).strip()

        checklist = parsed.get("checklist") or baseline.get("checklist") or {}
        ev_estimate = parsed.get("ev_estimate") or baseline.get("ev_estimate") or {}

        return {
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "checklist": checklist,
            "ev_estimate": ev_estimate,
            "horizon": horizon,
            "horizon_reason": horizon_reason,
        }

    except Exception as e:
        logger.exception("LLM refine failed: %s", e)
        out = dict(baseline)
        add = f"(LLM refine error: {type(e).__name__}: {e})"
        out["reason"] = (
            f"🕒 Suggested trade: {out['horizon'].upper()} — {out['horizon_reason']}\n"
            + out.get("reason", "")
            + ("\n" if out.get("reason") else "")
            + add
        ).strip()
        return out


# --- You already have this checklist builder in your file. Keep yours.
def baseline_checklist(alert: Dict[str, Any], f: Dict[str, Any], macd_line: Optional[float],
                       horizon: str, horizon_score: float, horizon_reason: str) -> Dict[str, Any]:
    # Use your existing checklist section (unchanged) or keep it minimal:
    return {
        "technicals": {
            "rsi14": _f(f.get("rsi14")),
            "ema20": _f(f.get("ema20")), "ema50": _f(f.get("ema50")), "ema200": _f(f.get("ema200")),
            "macd_line": macd_line, "macd_signal": _f(f.get("macd_signal")), "macd_hist": _f(f.get("macd_hist")),
            "vwap": _f(f.get("vwap")), "vwap_dist": _f(f.get("vwap_dist")),
            "bb_upper": _f(f.get("bb_upper")), "bb_lower": _f(f.get("bb_lower")), "sma20": _f(f.get("sma20")),
            "orb15_high": _f(f.get("orb15_high")), "orb15_low": _f(f.get("orb15_low")),
            "ta_src": f.get("ta_src"),
        },
        "structure": {
            "dte": _f(f.get("dte")), "mtf_align": f.get("mtf_align"), "regime_flag": f.get("regime_flag"),
            "horizon": horizon, "horizon_score": horizon_score, "horizon_reason": horizon_reason,
        },
        "context": {
            "quote_change_pct": _f(f.get("quote_change_pct")),
            "tv_meta": f.get("tv_meta") if isinstance(f.get("tv_meta"), dict) else {},
        },
    }
