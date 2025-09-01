# llm_client.py
# Compatible with openai>=1.0 (Responses API or Chat Completions).
# Uses AsyncOpenAI so routes.py can `await analyze_with_openai(...)`.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from openai import AsyncOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _iv_ctx(iv: float | None, iv_rank: float | None) -> str:
    """
    Classify IV context into 'low' / 'medium' / 'high' based on IV rank.
    """
    if iv_rank is None:
        return "medium"
    if iv_rank < 0.33:
        return "low"
    if iv_rank > 0.66:
        return "high"
    return "medium"


def build_llm_prompt(alert: Dict[str, Any], f: Dict[str, Any]) -> str:
    """
    Build a compact, JSON-friendly instruction for the LLM.
    The model must choose one of: BUY / SKIP / WAIT.
    """
    # derive a few readable flags
    iv_rank = f.get("iv_rank")
    iv = f.get("iv")
    iv_context = _iv_ctx(iv, iv_rank)

    # short, clear facts block
    facts = {
        "symbol": alert.get("symbol"),
        "side": alert.get("side"),
        "underlying_from_alert": alert.get("underlying_price_from_alert"),
        "reco_strike": alert.get("strike"),
        "reco_expiry": alert.get("expiry"),
        "dte": f.get("dte"),
        "greeks": {
            "delta": f.get("delta"),
            "gamma": f.get("gamma"),
            "theta": f.get("theta"),
            "vega": f.get("vega"),
        },
        "nbbo": {
            "bid": f.get("bid"),
            "ask": f.get("ask"),
            "mid": f.get("mid"),
            "spread_pct": f.get("option_spread_pct"),
            "quote_age_sec": f.get("quote_age_sec"),
        },
        "liquidity": {"oi": f.get("oi"), "vol": f.get("vol")},
        "volatility": {"iv": iv, "iv_rank": iv_rank, "iv_context": iv_context, "rv20": f.get("rv20")},
        "edges": {
            "em_vs_be_ok": f.get("em_vs_be_ok"),
            "mtf_align": f.get("mtf_align"),
            "sr_headroom_ok": f.get("sr_headroom_ok"),
            "regime": f.get("regime_flag"),
        },
        "levels": {
            "prev_day_high": f.get("prev_day_high"),
            "prev_day_low": f.get("prev_day_low"),
            "premarket_high": f.get("premarket_high"),
            "premarket_low": f.get("premarket_low"),
            "above_pdh": f.get("above_pdh"),
            "below_pdl": f.get("below_pdl"),
            "above_pmh": f.get("above_pmh"),
            "below_pml": f.get("below_pml"),
            "vwap": f.get("vwap"),
            "vwap_dist": f.get("vwap_dist"),
        },
    }

    # instruction: **must** return strict JSON so our parser is reliable
    instruction = (
        "You are an options-trading assistant. Decide whether to BUY, SKIP, or WAIT on the suggested contract.\n"
        "- BUY only if the setup quality is good (liquidity ok, spread reasonable, alignment ok, edge present).\n"
        "- SKIP if risk is poor or data quality/liquidity is bad.\n"
        "- WAIT if unclear and more confirmation is needed.\n\n"
        "Return strict JSON ONLY with keys: decision, confidence, reason, checklist, ev_estimate.\n"
        "decision ∈ {buy, skip, wait}; confidence ∈ [0,1].\n"
        "checklist should include booleans like delta_band_ok, dte_band_ok, iv_context, rv_iv_spread, em_vs_breakeven_ok,\n"
        "mtf_trend_alignment, sr_headroom_ok, no_event_risk.\n"
        "ev_estimate should include win_prob, avg_win_pct, avg_loss_pct, expected_value_pct.\n"
    )

    prompt = {
        "role": "user",
        "content": instruction + "\nFacts:\n" + json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
    }
    return json.dumps(prompt, ensure_ascii=False)


async def _call_openai_for_json(prompt_payload: str) -> Dict[str, Any]:
    """
    Calls OpenAI; returns a parsed JSON dict or raises.
    Uses Chat Completions for maximum compatibility with openai>=1.0.
    """
    # Convert back to the messages list Chat Completions expects.
    user_msg = json.loads(prompt_payload)
    messages = [user_msg] if isinstance(user_msg, dict) else [{"role": "user", "content": prompt_payload}]

    resp = await _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def _fallback_wait(reason: str) -> Dict[str, Any]:
    return {
        "decision": "wait",
        "confidence": 0.0,
        "reason": reason,
        "checklist": {},
        "ev_estimate": {},
    }


async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates prompt build + OpenAI call and returns normalized structure.
    """
    try:
        prompt_payload = build_llm_prompt(alert, features)
    except Exception as e:
        return _fallback_wait(f"Prompt build failed: {e}")

    try:
        out = await _call_openai_for_json(prompt_payload)
        # Normalize & guard fields
        decision = str(out.get("decision", "wait")).lower()
        if decision not in {"buy", "skip", "wait"}:
            decision = "wait"

        confidence = out.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        return {
            "decision": decision,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": out.get("reason", ""),
            "checklist": out.get("checklist", {}),
            "ev_estimate": out.get("ev_estimate", {}),
        }
    except Exception as e:
        # Keep the app resilient if OpenAI is unreachable or responds with a schema error.
        return _fallback_wait(f"LLM call failed: {e}")
