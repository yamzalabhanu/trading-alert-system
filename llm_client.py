# llm_client.py
import os
import json
import logging
from typing import Dict, Any, Optional, List

try:
    from openai import AsyncOpenAI  # openai>=1.x
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger("trading_engine.llm")

# -----------------------------
# knobs / weights (unchanged)
# -----------------------------
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

# -----------------------------
# tiny helpers (unchanged)
# -----------------------------
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

# -------------------------------------------------------------------
# KEEP your scoring helpers here as-is (not shown in your paste).
# -------------------------------------------------------------------

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
        "nbbo_http_status","nbbo_reason","ta_src",
        "sma20_d","ema20_d","ema50_d","rsi14_d","macd_line_d","macd_signal_d","macd_hist_d",
        "tv_meta",
    ]
    return {k: f[k] for k in keys if k in f}

def _alert_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "side": alert.get("side"),
        "symbol": alert.get("symbol"),
        "underlying_price_from_alert": alert.get("underlying_price_from_alert"),
    }
    if alert.get("strike") is not None:
        out["strike"] = alert.get("strike")
    if alert.get("expiry"):
        out["expiry"] = alert.get("expiry")

    # passthrough meta keys aligned with engine_common + engine_processor
    for k in (
        "source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level",
        "tp1", "tp2", "tp3", "trail", "relVol", "relvol", "chop", "adx"
    ):
        if alert.get(k) is not None:
            out[k] = alert.get(k)
    return out

# =============================
# SINGLE SOURCE OF TRUTH
# =============================
async def _analyze_impl(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    The ONLY implementation. Both public functions call this.
    This prevents recursion forever.
    """
    # If OpenAI is not configured, return safe fallback (no exceptions).
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or AsyncOpenAI is None:
        return {
            "decision": "wait",
            "confidence": 0.0,
            "reason": "LLM disabled (missing OPENAI_API_KEY or openai library)",
            "checklist": {},
            "ev_estimate": {},
        }

    model = DEFAULT_MODEL

    # Build minimal prompt context (safe + small)
    ap = _alert_payload(alert)
    sf = _slim_features(features or {})

    system = (
        "You are a trading assistant. Return STRICT JSON only with keys: "
        "decision (buy|wait|skip), confidence (0-1), reason (string), checklist (object), ev_estimate (object)."
    )
    user = {
        "alert": ap,
        "features": sf,
        "policy": {
            "buy_threshold": BUY_THRESHOLD,
            "wait_threshold": WAIT_THRESHOLD,
        }
    }

    try:
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
            ],
        )

        # Try to extract text
        text = None
        try:
            # openai responses API typically provides output_text convenience
            text = getattr(resp, "output_text", None)
        except Exception:
            text = None

        if not text:
            # fallback: try to reconstruct from output content
            try:
                # resp.output is a list; each item has content blocks
                blocks = []
                for out in getattr(resp, "output", []) or []:
                    for c in getattr(out, "content", []) or []:
                        if getattr(c, "type", "") in ("output_text", "text"):
                            blocks.append(getattr(c, "text", ""))
                text = "\n".join([b for b in blocks if b]).strip()
            except Exception:
                text = None

        if not text:
            return {
                "decision": "wait",
                "confidence": 0.0,
                "reason": "LLM returned empty response",
                "checklist": {},
                "ev_estimate": {},
            }

        # Parse JSON safely
        try:
            data = json.loads(text)
        except Exception:
            # salvage: try to slice outermost object
            a = text.find("{")
            b = text.rfind("}")
            if a != -1 and b != -1 and b > a:
                data = json.loads(text[a:b+1])
            else:
                raise

        decision = str(data.get("decision") or "wait").lower().strip()
        if decision not in ("buy", "wait", "skip"):
            decision = "wait"

        conf = data.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = _clip(conf_f, 0.0, 1.0)

        return {
            "decision": decision,
            "confidence": conf_f,
            "reason": str(data.get("reason") or "").strip()[:2000],
            "checklist": data.get("checklist") if isinstance(data.get("checklist"), dict) else {},
            "ev_estimate": data.get("ev_estimate") if isinstance(data.get("ev_estimate"), dict) else {},
            "model": model,
        }

    except Exception as e:
        logger.warning("LLM call failed: %r", e)
        return {
            "decision": "wait",
            "confidence": 0.0,
            "reason": f"LLM error: {e}",
            "checklist": {},
            "ev_estimate": {},
        }

# =============================
# Public API (NO recursion)
# =============================
async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    return await _analyze_impl(alert, features)

async def analyze_alert(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    # Back-compat alias ONLY (one-way)
    return await _analyze_impl(alert, features)

__all__ = ["analyze_with_openai", "analyze_alert"]
