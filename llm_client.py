# llm_client.py
import os
import json
import math
import logging
from typing import Dict, Any, Optional, Tuple, List

try:
    # openai>=1.x
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger("trading_engine.llm")

# --- knobs (overridable by env) ---
BUY_THRESHOLD = float(os.getenv("LLM_BUY_THRESHOLD", "60"))
WAIT_THRESHOLD = float(os.getenv("LLM_WAIT_THRESHOLD", "45"))

W_TECH = float(os.getenv("LLM_W_TECH", "60"))          # RSI/EMA/MACD/VWAP/Bollinger
W_STRUCT = float(os.getenv("LLM_W_STRUCT", "15"))       # ORB15 + regime/MTF
W_OPTION = float(os.getenv("LLM_W_OPTION", "15"))       # Delta + IV + OI/Vol tilt
W_EXEC = float(os.getenv("LLM_W_EXEC", "10"))           # Spread + synthetic NBBO penalty relief

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

# --- indicator scoring --------------------------------------------------------

def _score_rsi(side: str, rsi14: Optional[float]) -> Tuple[float, str]:
    if rsi14 is None:
        return 0.0, _mk_reason("RSI", 0, "missing")
    # Balanced: avoid buying calls at >70 or puts at <30 unless reversal
    if side == "CALL":
        if 40 <= rsi14 <= 65:
            return 7.0, _mk_reason("RSI", +7, f"constructive zone {rsi14:.1f}")
        if rsi14 < 30:
            return 4.0, _mk_reason("RSI", +4, "oversold (potential bounce)")
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
    # each alignment worth ~3–4 points (total ~10)
    for name, ema in (("EMA20", ema20), ("EMA50", ema50), ("EMA200", ema200)):
        rel = _price_vs(price, ema)
        if rel is None:
            continue
        if side == "CALL":
            if rel > 0:
                pts += 3.3; notes.append(f"{name}↑")
            else:
                pts -= 2.0; notes.append(f"{name}↓")
        else:
            if rel < 0:
                pts += 3.3; notes.append(f"{name}↓")
            else:
                pts -= 2.0; notes.append(f"{name}↑")
    if not notes:
        return 0.0, _mk_reason("EMA stack", 0, "missing")
    return pts, _mk_reason("EMA stack", pts, ",".join(notes))

def _score_macd(side: str, macd: Optional[float], macd_signal: Optional[float], macd_hist: Optional[float]) -> Tuple[float, str]:
    if macd is None or macd_signal is None:
        return 0.0, _mk_reason("MACD", 0, "missing")
    pts = 0.0
    cross = macd - macd_signal
    if side == "CALL":
        if macd > 0 and cross > 0:
            pts += 6.0
        elif cross > 0:
            pts += 3.0
        elif macd < 0 and cross < 0:
            pts -= 3.0
    else:
        if macd < 0 and cross < 0:
            pts += 6.0
        elif cross < 0:
            pts += 3.0
        elif macd > 0 and cross > 0:
            pts -= 3.0
    if macd_hist is not None:
        pts += _clip((macd_hist * (1 if side == "CALL" else -1)) * 4.0, -2.0, 2.0)
    return pts, _mk_reason("MACD", pts, f"macd={macd:.3f},sig={macd_signal:.3f}")

def _score_vwap(side: str, price: Optional[float], vwap: Optional[float], vwap_dist: Optional[float]) -> Tuple[float, str]:
    # prefer above VWAP for CALLs, below for PUTs; use either dist or raw price-vwap
    if vwap is None and vwap_dist is None:
        return 0.0, _mk_reason("VWAP", 0, "missing")
    pts = 0.0
    dist = vwap_dist
    if dist is None and vwap is not None and price is not None:
        dist = _pct(price, vwap)
    if dist is None:
        return 0.0, _mk_reason("VWAP", 0, "missing")
    if side == "CALL":
        pts = _clip(dist * 0.6, -3.0, 3.0)
    else:
        pts = _clip(-dist * 0.6, -3.0, 3.0)
    return pts, _mk_reason("VWAP", pts, f"dist≈{dist:.2f}%")

def _score_bollinger(side: str, price: Optional[float], bb_upper: Optional[float], bb_lower: Optional[float], sma20: Optional[float]) -> Tuple[float, str]:
    # If near upper band -> favor PUTs; near lower band -> favor CALLs.
    if price is None or (bb_upper is None and bb_lower is None and sma20 is None):
        return 0.0, _mk_reason("Bollinger", 0, "missing")
    pts = 0.0
    notes: List[str] = []
    if bb_upper is not None and price >= bb_upper:
        pts += (5.0 if side == "PUT" else -3.0); notes.append("touch U")
    if bb_lower is not None and price <= bb_lower:
        pts += (5.0 if side == "CALL" else -3.0); notes.append("touch L")
    if sma20 is not None:
        rel = _price_vs(price, sma20)
        if rel is not None:
            if side == "CALL":
                pts += (1.5 if rel > 0 else -1.5)
            else:
                pts += (1.5 if rel < 0 else -1.5)
    if not notes and sma20 is None:
        return 0.0, _mk_reason("Bollinger", 0, "uninformative")
    return pts, _mk_reason("Bollinger", pts, ",".join(notes) or "centerline")

def _score_orb15(side: str, price: Optional[float], orb_high: Optional[float], orb_low: Optional[float]) -> Tuple[float, str]:
    if price is None or (orb_high is None and orb_low is None):
        return 0.0, _mk_reason("ORB15", 0, "missing")
    pts = 0.0
    if orb_high is not None and price > orb_high:
        pts += (6.0 if side == "CALL" else -4.0)
    if orb_low is not None and price < orb_low:
        pts += (6.0 if side == "PUT" else -4.0)
    if pts == 0.0:
        return 0.0, _mk_reason("ORB15", 0, "inside range")
    return pts, _mk_reason("ORB15", pts, "range break")

def _score_delta(side: str, delta: Optional[float]) -> Tuple[float, str]:
    if delta is None:
        return 0.0, _mk_reason("Delta", 0, "missing")
    pts = 0.0
    if side == "CALL":
        # sweet spot ~0.3-0.6
        pts = 6.0 - (abs(delta - 0.45) / 0.15) * 3.0  # ~[−? , +6]
    else:
        pts = 6.0 - (abs(delta + 0.45) / 0.15) * 3.0
    pts = _clip(pts, -3.0, 6.0)
    return pts, _mk_reason("Delta", pts, f"Δ={delta:.3f}")

def _score_iv(side: str, iv: Optional[float], iv_rank: Optional[float]) -> Tuple[float, str]:
    # For option BUY, too-high IV hurts due to premium; moderate ranks best.
    if iv is None and iv_rank is None:
        return 0.0, _mk_reason("IV", 0, "missing")
    pts = 0.0
    if iv_rank is not None:
        # favor mid ranks (0.3-0.7), penalize extremes
        if 0.3 <= iv_rank <= 0.7:
            pts += 4.0
        elif iv_rank > 0.85:
            pts -= 3.0
        elif iv_rank < 0.15:
            pts -= 2.0
    return pts, _mk_reason("IV", pts, f"rank={iv_rank!r}")

def _score_liquidity(spread_pct: Optional[float], oi: Optional[float], vol: Optional[float], synthetic_nbbo_used: bool) -> Tuple[float, str]:
    pts = 0.0
    notes: List[str] = []
    if spread_pct is not None:
        if spread_pct <= 5:
            pts += 5.0; notes.append("tight")
        elif spread_pct <= 12:
            pts += 2.0; notes.append("ok")
        else:
            pts -= 3.0; notes.append("wide")
    if oi is not None:
        if oi >= 5000: pts += 2.0
        elif oi >= 500: pts += 1.0
        else: pts -= 1.0
    if vol is not None and vol <= 10:
        pts -= 1.0
    if synthetic_nbbo_used:
        # reduce penalty sensitivity when synthetic used (we don't want LLM to auto-skip)
        pts += 1.5
        notes.append("synthetic-ok")
    return pts, _mk_reason("Liquidity", pts, "/".join(notes) if notes else "n/a")

def _score_structure(mtf_align: Any, regime_flag: Any) -> Tuple[float, str]:
    pts = 0.0
    if isinstance(mtf_align, bool):
        pts += (4.0 if mtf_align else -2.0)
    if isinstance(regime_flag, str):
        if regime_flag.lower().startswith("trending"):
            pts += 3.0
        elif regime_flag.lower().startswith("choppy"):
            pts -= 2.0
    return pts, _mk_reason("Structure", pts, f"mtf={mtf_align},regime={regime_flag}")

def _score_dte(dte: Optional[float]) -> Tuple[float, str]:
    if dte is None:
        return 0.0, _mk_reason("DTE", 0, "missing")
    pts = 0.0
    # Light touch: avoid zero DTE unless forced; mild penalty for very long
    if dte <= 0:
        pts -= 10.0
        return pts, _mk_reason("DTE", pts, "0DTE high risk")
    if dte < 2:
        pts -= 3.0
    elif 5 <= dte <= 35:
        pts += 3.0
    elif dte > 45:
        pts -= 2.0
    return pts, _mk_reason("DTE", pts, f"{dte} days")

# --- rule based blend ---------------------------------------------------------

def _rule_blend(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
    side = (alert.get("side") or "").upper()
    ul_price = _f(alert.get("underlying_price_from_alert"))

    # technicals
    rsi_pts, rsi_note = _score_rsi(side, _f(f.get("rsi14")))
    ema_pts, ema_note = _score_ema_stack(side, ul_price, _f(f.get("ema20")), _f(f.get("ema50")), _f(f.get("ema200")))
    macd_pts, macd_note = _score_macd(side, _f(f.get("macd")), _f(f.get("macd_signal")), _f(f.get("macd_hist")))
    vwap_pts, vwap_note = _score_vwap(side, ul_price, _f(f.get("vwap")), _f(f.get("vwap_dist")))
    boll_pts, boll_note = _score_bollinger(side, ul_price, _f(f.get("bb_upper")), _f(f.get("bb_lower")), _f(f.get("sma20")))
    tech_raw = rsi_pts + ema_pts + macd_pts + vwap_pts + boll_pts
    tech_scaled = _clip(tech_raw / 25.0 * W_TECH, -W_TECH, W_TECH)

    # structure (ORB15 + regime/MTF)
    orb_pts, orb_note = _score_orb15(side, ul_price, _f(f.get("orb15_high")), _f(f.get("orb15_low")))
    struct_pts, struct_note = _score_structure(f.get("mtf_align"), f.get("regime_flag"))
    struct_raw = orb_pts + struct_pts
    struct_scaled = _clip(struct_raw / 10.0 * W_STRUCT, -W_STRUCT, W_STRUCT)

    # options / vol
    delta_pts, delta_note = _score_delta(side, _f(f.get("delta")))
    iv_pts, iv_note = _score_iv(side, _f(f.get("iv")), _f(f.get("iv_rank")))
    opt_raw = delta_pts + iv_pts
    opt_scaled = _clip(opt_raw / 10.0 * W_OPTION, -W_OPTION, W_OPTION)

    # execution
    liq_pts, liq_note = _score_liquidity(_f(f.get("option_spread_pct")), _f(f.get("oi")), _f(f.get("vol")), bool(f.get("synthetic_nbbo_used")))
    exec_scaled = _clip(liq_pts / 6.0 * W_EXEC, -W_EXEC, W_EXEC)

    # dte as soft governor (not the main factor)
    dte_pts, dte_note = _score_dte(_f(f.get("dte")))

    # combine (base 50 + scaled sums + dte)
    score = 50.0 + tech_scaled + struct_scaled + opt_scaled + exec_scaled + dte_pts
    score = _clip(score, 0.0, 100.0)

    if score >= BUY_THRESHOLD:
        decision = "buy"
    elif score >= WAIT_THRESHOLD:
        decision = "wait"
    else:
        decision = "skip"

    # confidence proportional to distance from boundary
    conf = 0.5 + abs(score - 50.0) / 100.0  # 0.5..1.0
    conf = round(_clip(conf, 0.5, 0.95), 2)

    reasons = [rsi_note, ema_note, macd_note, vwap_note, boll_note, orb_note, struct_note, delta_note, iv_note, dte_note, liq_note]
    reason_text = "; ".join(reasons)

    checklist = {
        "technicals": {
            "rsi14": _f(f.get("rsi14")),
            "ema20": _f(f.get("ema20")),
            "ema50": _f(f.get("ema50")),
            "ema200": _f(f.get("ema200")),
            "macd": _f(f.get("macd")),
            "macd_signal": _f(f.get("macd_signal")),
            "macd_hist": _f(f.get("macd_hist")),
            "vwap": _f(f.get("vwap")),
            "vwap_dist": _f(f.get("vwap_dist")),
            "bb_upper": _f(f.get("bb_upper")),
            "bb_lower": _f(f.get("bb_lower")),
            "sma20": _f(f.get("sma20")),
            "orb15_high": _f(f.get("orb15_high")),
            "orb15_low": _f(f.get("orb15_low")),
            "ta_src": f.get("ta_src"),
        },
        "options": {
            "delta": _f(f.get("delta")),
            "gamma": _f(f.get("gamma")),
            "theta": _f(f.get("theta")),
            "vega": _f(f.get("vega")),
            "iv": _f(f.get("iv")),
            "iv_rank": _f(f.get("iv_rank")),
            "oi": _f(f.get("oi")),
            "vol": _f(f.get("vol")),
            "spread_pct": _f(f.get("option_spread_pct")),
            "synthetic_nbbo_used": bool(f.get("synthetic_nbbo_used", False)),
        },
        "structure": {
            "dte": _f(f.get("dte")),
            "mtf_align": f.get("mtf_align"),
            "regime_flag": f.get("regime_flag"),
        }
    }

    return {
        "decision": decision,
        "confidence": conf,
        "reason": reason_text,
        "checklist": checklist,
        "ev_estimate": {
            # very rough directional "edge" proxy mapped from score
            "edge_pct": round((score - 50.0) / 1.5, 2),  # -33..+33 approx
            "score": round(score, 2),
        },
    }

# --- OpenAI wrapper -----------------------------------------------------------

def _slim_features(f: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "rsi14","sma20","ema20","ema50","ema200","macd","macd_signal","macd_hist",
        "vwap","vwap_dist","bb_upper","bb_lower","bb_width",
        "orb15_high","orb15_low","mtf_align","regime_flag","dte",
        "delta","gamma","theta","vega","iv","iv_rank","oi","vol",
        "bid","ask","mid","option_spread_pct","synthetic_nbbo_used",
        "prev_close","quote_change_pct","nbbo_http_status","nbbo_reason","ta_src"
    ]
    out = {}
    for k in keys:
        if k in f:
            out[k] = f[k]
    return out

async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns dict: {decision: buy|wait|skip, confidence: float, reason: str, checklist: {}, ev_estimate: {}}
    - Explicit weights for RSI/EMA20/50/200/MACD/VWAP/Bollinger/ORB15 are applied in a deterministic rule blend.
    - If OPENAI_API_KEY is present, we prompt a model to refine/justify, but we keep the structure & guardrails.
    """
    # 1) Rule-based baseline
    baseline = _rule_blend(alert, features)

    # 2) If OpenAI not configured, return baseline
    if AsyncOpenAI is None or not os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OFFLINE", "0") == "1":
        baseline.setdefault("reason", "(offline rule-based)")
        return baseline

    # 3) Ask model to refine *within bounds* and produce JSON
    try:
        client = AsyncOpenAI()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        system_msg = (
            "You are an options-trading analyst. You must weigh technical indicators explicitly:\n"
            "- RSI(14), EMA(20/50/200) stack, MACD (line/signal/hist), VWAP distance,\n"
            "- Bollinger (upper/lower/centerline), and 15-min ORB breakout/inside.\n"
            "Blend with Greeks (Δ, Θ, ν) and IV/IV rank, plus execution (spread, OI/Vol).\n"
            "Given a baseline weighted score, you may adjust modestly (<= ±10 absolute points) only if the narrative strongly supports it.\n"
            "Never auto-skip solely from missing NBBO; if synthetic_nbbo_used is true, reduce execution penalties.\n"
            "Return ONLY valid JSON with fields: decision, confidence, reason, checklist, ev_estimate."
        )

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
                "reason": baseline["reason"],
                "ev_estimate": baseline.get("ev_estimate"),
                "score": baseline.get("ev_estimate", {}).get("score"),
            },
            "boundaries": {
                "buy_threshold": BUY_THRESHOLD,
                "wait_threshold": WAIT_THRESHOLD,
                "max_adjust_abs_points": 10
            }
        }

        resp = await client.chat.completions.create(
            model=model,
            temperature=0.15,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Refine the baseline decision using the explicit weights. "
                        "Be conservative with liquidity penalties if synthetic_nbbo_used=true. "
                        "If key TA is missing, state that and do NOT over-weight liquidity/DTE. "
                        "Here is the data:\n\n" + json.dumps(user_payload)
                    ),
                },
            ],
        )

        txt = resp.choices[0].message.content
        parsed = json.loads(txt)

        # minimal sanitation and fallback to baseline fields if missing
        decision = str(parsed.get("decision") or baseline["decision"]).lower()
        if decision not in ("buy", "wait", "skip"):
            decision = baseline["decision"]

        confidence = parsed.get("confidence")
        try:
            confidence = float(confidence)
            confidence = _clip(confidence, 0.0, 1.0)
        except Exception:
            confidence = baseline["confidence"]

        reason = parsed.get("reason") or baseline["reason"]
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
        # Fall back to baseline, but annotate
        out = dict(baseline)
        out["reason"] = f"{baseline['reason']} | LLM refine error: {type(e).__name__}: {e}"
        return out
