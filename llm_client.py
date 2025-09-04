# llm_client.py
import os, json
from typing import Dict, Any
import httpx

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P        = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_MAX_TOK      = int(os.getenv("LLM_MAX_TOKENS", "700"))  # used for Chat fallback
OPENAI_API_STYLE = os.getenv("OPENAI_API_STYLE", "").lower()  # "responses" | "chat" | ""

SYSTEM_PROMPT = (
    "You are an options-trading assistant. Decide one of: buy, skip, wait. "
    "Consider liquidity (bid/ask/mid/spread%, vol, oi, quote_age), greeks, IV & IV rank, DTE, regime/MTF. "
    "Output STRICT JSON: {\"decision\":\"...\",\"confidence\":0-1,\"reason\":\"...\",\"checklist\":{}}"
)

def _user_payload(alert: Dict[str, Any], features: Dict[str, Any]) -> str:
    return json.dumps({
        "alert": alert,
        "features": features,
        "instruction": (
            "Decide buy/skip/wait. Prefer wait/skip if liquidity is poor or quotes are stale. "
            "Return ONLY JSON with keys: decision, confidence, reason, checklist."
        )
    }, separators=(",", ":"))

# ----------------------
# Responses API (minimal)
# ----------------------
async def _try_responses(client: httpx.AsyncClient, system: str, user: str) -> Dict[str, Any]:
    """Use /v1/responses with a minimal, broadly compatible payload."""
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
        # Deliberately omit modalities/text.format/max_* due to mixed gateway support
    }
    r = await client.post(url, headers=headers, json=body, timeout=30.0)
    r.raise_for_status()
    return r.json()

def _responses_text(js: Dict[str, Any]) -> str:
    # Prefer SDK-style
    t = js.get("output_text")
    if isinstance(t, str) and t.strip():
        return t
    # Raw REST variants
    out = js.get("output") or js.get("choices") or []
    if isinstance(out, list) and out:
        parts = out[0].get("content") if isinstance(out[0], dict) else None
        if isinstance(parts, list):
            for p in parts:
                val = p.get("text") or p.get("value")
                if isinstance(val, str) and val.strip():
                    return val
    return ""

# ----------------------
# Chat Completions fallback
# ----------------------
async def _try_chat(client: httpx.AsyncClient, system: str, user: str) -> Dict[str, Any]:
    """Use /v1/chat/completions with json_object response_format."""
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOK,
        "response_format": {"type": "json_object"},
    }
    r = await client.post(url, headers=headers, json=body, timeout=30.0)
    r.raise_for_status()
    return r.json()

def _chat_text(js: Dict[str, Any]) -> str:
    try:
        return js["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""

# ----------------------
# Common helpers
# ----------------------
def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}

def _normalize(parsed: Dict[str, Any]) -> Dict[str, Any]:
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

# ----------------------
# Public entry
# ----------------------
async def analyze_with_openai(alert: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"decision": "wait", "confidence": 0.0, "reason": "LLM disabled: missing OPENAI_API_KEY", "checklist": {}, "ev_estimate": {}}

    system = SYSTEM_PROMPT
    user = _user_payload(alert, features)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Allow forcing style via env
        if OPENAI_API_STYLE == "chat":
            try:
                js = await _try_chat(client, system, user)
                text = _chat_text(js)
                return _normalize(_safe_json(text))
            except httpx.HTTPStatusError as e:
                return {"decision": "wait", "confidence": 0.0, "reason": f"LLM chat failed: {e.response.status_code} - {e.response.text}", "checklist": {}, "ev_estimate": {}}

        if OPENAI_API_STYLE == "responses":
            try:
                js = await _try_responses(client, system, user)
                text = _responses_text(js)
                return _normalize(_safe_json(text))
            except httpx.HTTPStatusError as e:
                # Hard fail → fall back to chat automatically
                try:
                    js_c = await _try_chat(client, system, user)
                    text_c = _chat_text(js_c)
                    return _normalize(_safe_json(text_c))
                except httpx.HTTPStatusError as e2:
                    return {"decision": "wait", "confidence": 0.0, "reason": f"LLM resp+chat failed: {e.response.status_code}/{e2.response.status_code}", "checklist": {}, "ev_estimate": {}}

        # Auto-detect (try Responses first, then Chat)
        try:
            js = await _try_responses(client, system, user)
            text = _responses_text(js)
            return _normalize(_safe_json(text))
        except httpx.HTTPStatusError:
            try:
                js_c = await _try_chat(client, system, user)
                text_c = _chat_text(js_c)
                return _normalize(_safe_json(text_c))
            except httpx.HTTPStatusError as e2:
                return {"decision": "wait", "confidence": 0.0, "reason": f"LLM all failed: {e2.response.status_code} - {e2.response.text}", "checklist": {}, "ev_estimate": {}}
        except Exception as e:
            # As a last resort
            return {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {type(e).__name__}: {e}", "checklist": {}, "ev_estimate": {}}
