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


# --- (all your scoring helpers unchanged) ---
# keep everything you pasted above unchanged up to _alert_payload(),
# except the _alert_payload() passthrough list (updated below).


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
        "tv_meta",
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

    # ✅ passthrough meta keys aligned with engine_common + engine_processor
    for k in ("source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level",
              "tp1", "tp2", "tp3", "trail", "relVol", "relvol", "chop", "adx"):
        if alert.get(k) is not None:
            out[k] = alert.get(k)
    return out


# llm_client.py

# If older code refers to analyze_alert, keep a compatibility alias.
# Make sure ONE of these exists (analyze_alert or analyze_with_openai).

async def analyze_with_openai(alert: dict, features: dict) -> dict:
    # If you already have an implementation named analyze_alert(), delegate to it.
    if "analyze_alert" in globals() and callable(globals()["analyze_alert"]):
        return await globals()["analyze_alert"](alert, features)

    # Otherwise: return a safe fallback so the system never crashes
    return {
        "decision": "wait",
        "confidence": 0.0,
        "reason": "LLM disabled/misconfigured: analyze_alert not found",
        "checklist": {},
        "ev_estimate": {},
    }

# Backward compat: if some modules call analyze_alert() directly
async def analyze_alert(alert: dict, features: dict) -> dict:
    return await analyze_with_openai(alert, features)

# keep analyze_with_openai() exactly as you pasted, no logic changes needed
# (it already supports equity-only alerts and missing options fields)
# ---------------------------------------------------------------

# NOTE: paste your existing analyze_with_openai implementation here unchanged.
