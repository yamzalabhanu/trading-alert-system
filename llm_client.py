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
# knobs / weights
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

DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.2"))


# -----------------------------
# tiny helpers
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
        # core technicals
        "last",
        "mid",
        "rsi14", "sma20", "ema20", "ema50", "ema200",
        "macd_line", "macd", "macd_signal", "macd_hist",
        "vwap", "vwap_dist", "bb_upper", "bb_lower", "bb_width",
        "orb15_high", "orb15_low", "mtf_align", "regime_flag", "dte",

        # options / execution
        "delta", "gamma", "theta", "vega", "iv", "iv_rank", "oi", "vol",
        "bid", "ask", "mid", "option_spread_pct",
        "synthetic_nbbo_used", "synthetic_nbbo_spread_est",

        # structure / prior day / premarket
        "prev_open", "prev_high", "prev_low", "prev_close",
        "quote_change_pct", "prev5_avg_high", "prev5_avg_low",
        "premarket_high", "premarket_low",

        # short metrics / misc
        "short_volume", "short_interest", "short_volume_total", "short_volume_ratio",
        "nbbo_http_status", "nbbo_reason", "ta_src",

        # daily technicals
        "sma20_d", "ema20_d", "ema50_d", "rsi14_d", "macd_line_d", "macd_signal_d", "macd_hist_d",
        "daily_trend_bias", "atr14_daily", "atr14_pct",

        # bars metadata
        "bars_meta", "mtf",

        # yahoo intraday / 5d context
        "yahoo_ctx_available",
        "yahoo_symbol",
        "yahoo_chart_range",
        "yahoo_chart_interval",
        "yahoo_last",
        "yahoo_day_open",
        "yahoo_day_high",
        "yahoo_day_low",
        "yahoo_day_volume",
        "yahoo_prev_close",
        "yahoo_prev_high",
        "yahoo_prev_low",
        "yahoo_premarket_high",
        "yahoo_premarket_low",
        "yahoo_orb15_high",
        "yahoo_orb15_low",
        "yahoo_ema9",
        "yahoo_ema20",
        "yahoo_ema50",
        "yahoo_rsi14",
        "yahoo_vwap",
        "yahoo_vwap_dist_pct",
        "yahoo_intraday_bias",
        "yahoo_last3_candle_bias",
        "yahoo_last3_sequence",
        "yahoo_day_change_pct",
        "yahoo_gap_pct",
        "yahoo_5d_high",
        "yahoo_5d_low",
        "yahoo_5d_return_pct",
        "yahoo_orb_break_state",
        "yahoo_prev_day_sr_state",
        "yahoo_llm_summary",

        # data provider info / metadata
        "data_provider",
        "nbbo_provider",
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

    for k in (
        "source", "model", "confirm_tf", "chart_tf", "event", "reason", "exchange", "level",
        "tp1", "tp2", "tp3", "trail", "relVol", "relvol", "chop", "adx",
        "tvScore", "tv_score", "TVScore"
    ):
        if alert.get(k) is not None:
            out[k] = alert.get(k)

    return out


def _build_analysis_policy() -> Dict[str, Any]:
    return {
        "buy_threshold": BUY_THRESHOLD,
        "wait_threshold": WAIT_THRESHOLD,
        "weights": {
            "tech": W_TECH,
            "structure": W_STRUCT,
            "option": W_OPTION,
            "execution": W_EXEC,
            "context": W_CONTEXT,
        },
        "instructions": [
            "Use TradingView alert context plus feature payload to judge intraday option tradeability.",
            "Prefer current live market structure when Polygon live data exists.",
            "Use Yahoo 5d/5m intraday candle structure as supplemental chart context, not as a replacement for live execution quality.",
            "Pay close attention to candle direction, VWAP relationship, EMA alignment, ORB state, prior-day support/resistance, premarket levels, and 5-day directional context.",
            "If setup quality is mixed, fragile, extended, illiquid, or missing critical tradeability inputs, prefer wait or skip over buy.",
            "Treat last-3-candle bias and Yahoo intraday bias as short-term structure clues, not standalone signals.",
            "If technicals conflict across timeframes, mention the conflict clearly in the reason.",
        ],
    }


def _build_focus_hints(alert: Dict[str, Any], f: Dict[str, Any]) -> Dict[str, Any]:
    side = str(alert.get("side") or "").lower()
    tv_meta = f.get("tv_meta") if isinstance(f.get("tv_meta"), dict) else {}

    hints: Dict[str, Any] = {
        "intended_trade_side": side,
        "chart_focus": [
            "intraday trend",
            "VWAP position",
            "EMA alignment",
            "opening range breakout / failure",
            "premarket interaction",
            "prior-day support resistance",
            "5-day context",
            "recent candle structure",
        ],
        "known_metadata": {
            "event": tv_meta.get("event"),
            "confirm_tf": tv_meta.get("confirm_tf"),
            "chart_tf": tv_meta.get("chart_tf"),
            "model": tv_meta.get("model"),
            "reason": tv_meta.get("reason"),
        },
    }

    if side == "calls":
        hints["side_specific_checks"] = [
            "Prefer bullish intraday structure",
            "Check whether price is above VWAP or reclaiming it with support",
            "Check whether recent candles show continuation rather than exhaustion",
            "Penalize if near resistance or failing ORB / prior-day levels",
        ]
    elif side == "puts":
        hints["side_specific_checks"] = [
            "Prefer bearish intraday structure",
            "Check whether price is below VWAP or rejecting it",
            "Check whether recent candles show downside continuation rather than weak drift",
            "Penalize if near support or failing to break ORB / prior-day levels",
        ]

    yahoo_summary = f.get("yahoo_llm_summary")
    if yahoo_summary:
        hints["yahoo_summary"] = yahoo_summary

    return hints


# =============================
# SINGLE SOURCE OF TRUTH
# =============================
async def _analyze_impl(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    The ONLY implementation. Both public functions call this.
    This prevents recursion forever.
    """
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

    ap = _alert_payload(alert)
    sf = _slim_features(features or {})
    policy = _build_analysis_policy()
    focus_hints = _build_focus_hints(alert, features or {})

    system = (
        "You are a disciplined intraday options trading assistant. "
        "Return STRICT JSON only with keys: "
        "decision (buy|wait|skip), confidence (0-1), reason (string), checklist (object), ev_estimate (object). "
        "Base your judgment on tradeability, structure quality, execution quality, and consistency across signals. "
        "Use Yahoo intraday/5-day chart context when present to improve chart reading. "
        "Do not invent missing values. "
        "If evidence is mixed or important execution inputs are missing, prefer wait or skip."
    )

    user = {
        "alert": ap,
        "features": sf,
        "policy": policy,
        "focus_hints": focus_hints,
        "output_schema": {
            "decision": "buy|wait|skip",
            "confidence": "float between 0 and 1",
            "reason": "brief but concrete explanation grounded in supplied data",
            "checklist": {
                "trend": "pass|mixed|fail",
                "structure": "pass|mixed|fail",
                "execution": "pass|mixed|fail",
                "options": "pass|mixed|fail",
                "context": "pass|mixed|fail",
                "yahoo_intraday": "pass|mixed|fail"
            },
            "ev_estimate": {
                "score_0_to_100": "optional numeric estimate",
                "notes": "optional short note"
            }
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

        text = None
        try:
            text = getattr(resp, "output_text", None)
        except Exception:
            text = None

        if not text:
            try:
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

        try:
            data = json.loads(text)
        except Exception:
            a = text.find("{")
            b = text.rfind("}")
            if a != -1 and b != -1 and b > a:
                data = json.loads(text[a:b + 1])
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

        checklist = data.get("checklist")
        if not isinstance(checklist, dict):
            checklist = {}

        ev_estimate = data.get("ev_estimate")
        if not isinstance(ev_estimate, dict):
            ev_estimate = {}

        return {
            "decision": decision,
            "confidence": conf_f,
            "reason": str(data.get("reason") or "").strip()[:2000],
            "checklist": checklist,
            "ev_estimate": ev_estimate,
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
# Public API
# =============================
async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    return await _analyze_impl(alert, features)


async def analyze_alert(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    return await _analyze_impl(alert, features)


__all__ = ["analyze_with_openai", "analyze_alert"]
