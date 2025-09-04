# llm_client.py
import os, json
from typing import Dict, Any
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_MAX_TOK = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "700"))

SYSTEM_PROMPT = """You are an options-trading assistant.
Given an alert and computed features, decide one of: "buy", "skip", or "wait".
Return concise JSON: {"decision": "...", "confidence": 0-1, "reason": "...", "checklist": {...}}.
"""

def _make_user_payload(alert: Dict[str, Any], features: Dict[str, Any]) -> str:
    return json.dumps({
        "alert": alert,
        "features": features,
        "instruction": (
            "Decide: buy/skip/wait. Consider liquidity (bid/ask/mid/spread%, vol, oi, quote_age), "
            "greeks, IV & IV rank, DTE, regime/MTF alignment. Prefer wait/skip if liquidity/staleness is poor. "
            "Output strictly JSON with keys: decision, confidence, reason, checklist."
        ),
    }, separators=(",", ":"))

async def _responses_call(client: httpx.AsyncClient, system: str, user: str) -> Dict[str, Any]:
    url = f"{OPENAI_BASE_URL}/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    body = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_completion_tokens": LLM_MAX_TOK,
        "text.format": "json",   # <-- correct param for JSON output
    }

    r = await client.post(url, headers=headers, json=body, timeout=30.0)

    if r.status_code == 400 and ("max_tokens" in r.text or "max_completion_tokens" in r.text):
        body.pop("max_completion_tokens", None)
        r = await client.post(url, headers=headers, json=body, timeout=30.0)

    r.raise_for_status()
    return r.json()


def _extract_text(js: Dict[str, Any]) -> str:
    # Prefer output_text when present
    t = js.get("output_text")
    if isinstance(t, str) and t.strip():
        return t
    # Fall back to raw blocks
    out = js.get("output") or js.get("choices") or []
    if isinstance(out, list) and out:
        parts = out[0].get("content") if isinstance(out[0], dict) else None
        if isinstance(parts, list):
            for p in parts:
                val = p.get("text") or p.get("value")
                if isinstance(val, str) and val.strip():
                    return val
    return ""

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}

async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"decision": "wait", "confidence": 0.0, "reason": "LLM disabled: missing OPENAI_API_KEY", "checklist": {}, "ev_estimate": {}}

    user_payload = _make_user_payload(alert, features)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await _responses_call(client, SYSTEM_PROMPT, user_payload)
        except httpx.HTTPStatusError as e:
            return {"decision": "wait", "confidence": 0.0, "reason": f"LLM call failed: {e.response.status_code} - {e.response.text}", "checklist": {}, "ev_estimate": {}}
        except Exception as e:
            return {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {type(e).__name__}: {e}", "checklist": {}, "ev_estimate": {}}

    text = _extract_text(resp) or "{}"
    parsed = _safe_json(text)

    decision = str(parsed.get("decision", "wait")).lower()
    if decision not in ("buy", "skip", "wait"):
        decision = "wait"

    try:
        conf = float(parsed.get("confidence", 0.0))
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
