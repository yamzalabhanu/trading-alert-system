# llm_client.py
import os
import json
from typing import Dict, Any
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # any Responses-capable model

# Tunables
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
# Safe cap; Responses API expects max_completion_tokens (NOT max_tokens)
LLM_MAX_TOK = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "700"))

SYSTEM_PROMPT = """You are an options-trading assistant. 
Given an alert (side, symbol, strike policy) and computed features (liquidity, greeks, IV context, preflight),
decide one of: "buy", "skip", or "wait".
Return a short reason, confidence in [0,1], and a brief checklist dict of key signals.
Keep answers concise and structured JSON only.
"""

def _make_user_payload(alert: Dict[str, Any], features: Dict[str, Any]) -> str:
    # Send the raw data the model needs, but compactly
    payload = {
        "alert": alert,
        "features": features,
        "instruction": (
            "Decide: buy/skip/wait. Consider liquidity (bid/ask/mid/spread%, vol, oi, quote_age), "
            "greeks (delta/gamma/theta/vega), IV & IV rank, DTE, and any regime/MTF alignment. "
            "If liquidity is poor or data is stale, prefer wait/skip. Output strictly JSON with keys: "
            "{decision, confidence, reason, checklist}."
        ),
    }
    return json.dumps(payload, separators=(",", ":"))

async def _responses_call(client: httpx.AsyncClient, system: str, user: str) -> Dict[str, Any]:
    """
    Calls the Responses API. Uses max_completion_tokens (correct field).
    If the server still complains about the field name, we retry without it.
    """
    url = f"{OPENAI_BASE_URL}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    base = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        # The correct limit param for Responses API:
        "max_completion_tokens": LLM_MAX_TOK,
        "response_format": {"type": "json_object"},
    }

    # First attempt with max_completion_tokens
    r = await client.post(url, headers=headers, json=base, timeout=30.0)
    if r.status_code == 400 and "max_tokens" in (r.text or ""):
        # Some deployments surface a confusing error; drop the cap and retry
        base.pop("max_completion_tokens", None)
        r = await client.post(url, headers=headers, json=base, timeout=30.0)

    r.raise_for_status()
    return r.json()

def _extract_text_from_responses(js: Dict[str, Any]) -> str:
    """
    Responses API returns content under output_text (SDKs) or in output[].content[] blocks (raw REST).
    Handle both defensively.
    """
    # SDK-compatible shape
    txt = js.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt

    # Raw REST shape
    out = js.get("output") or js.get("choices") or []
    if isinstance(out, list) and out:
        # Try to find a text segment
        first = out[0]
        parts = first.get("content") if isinstance(first, dict) else None
        if isinstance(parts, list) and parts:
            for p in parts:
                if p.get("type") in (None, "output_text", "text"):
                    val = p.get("text") or p.get("value")
                    if isinstance(val, str) and val.strip():
                        return val
    # Fallback: stringify
    return ""

def _safe_parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}

async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns: dict(decision, confidence, reason, checklist, ev_estimate?)
    """
    if not OPENAI_API_KEY:
        return {
            "decision": "wait",
            "confidence": 0.0,
            "reason": "LLM disabled: missing OPENAI_API_KEY",
            "checklist": {},
            "ev_estimate": {},
        }

    user_payload = _make_user_payload(alert, features)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            js = await _responses_call(client, SYSTEM_PROMPT, user_payload)
        except httpx.HTTPStatusError as e:
            # Surface server’s message for your logs
            return {
                "decision": "wait",
                "confidence": 0.0,
                "reason": f"LLM call failed: {e.response.status_code} - {e.response.text}",
                "checklist": {},
                "ev_estimate": {},
            }
        except Exception as e:
            return {
                "decision": "wait",
                "confidence": 0.0,
                "reason": f"LLM error: {type(e).__name__}: {e}",
                "checklist": {},
                "ev_estimate": {},
            }

    text = _extract_text_from_responses(js) or "{}"
    parsed = _safe_parse_json(text)

    decision = str(parsed.get("decision", "wait")).lower()
    if decision not in ("buy", "skip", "wait"):
        decision = "wait"

    conf = parsed.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {
        "decision": decision,
        "confidence": conf,
        "reason": parsed.get("reason", ""),
        "checklist": parsed.get("checklist", {}) or {},
        "ev_estimate": parsed.get("ev_estimate", {}) or {},
    }
