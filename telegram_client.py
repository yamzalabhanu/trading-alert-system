from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

import httpx

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    """
    Sends a message to Telegram using the Bot API.
    Returns the JSON response dict on success, None if not configured,
    or {"error": "..."} on failure.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Not configured; let caller treat as "not sent"
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",  # safe markup for bold/links
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ==========================
# Perplexity-aware formatter
# ==========================

def append_perplexity_context(base_text: str, px_ctx: Optional[Dict[str, Any]]) -> str:
    """
    Append Perplexity (Sonar/Search) insights to a Telegram alert body.
    - px_ctx["sonar_iv_verdict"]: True/False/None
    - px_ctx["sonar_iv_view"]: short text answer
    - px_ctx["sonar_citations"]: list of URLs
    - px_ctx["news_catalysts"]: list of dicts with title/url
    """
    if not px_ctx:
        return base_text

    verdict = px_ctx.get("sonar_iv_verdict")
    if verdict is True:
        iv_txt = "IV↑ likely"
    elif verdict is False:
        iv_txt = "IV↓ likely"
    else:
        iv_txt = "IV neutral/unclear"

    cite = None
    if isinstance(px_ctx.get("sonar_citations"), list) and px_ctx["sonar_citations"]:
        cite = px_ctx["sonar_citations"][0]
    src = f' <a href="{cite}">[src]</a>' if cite else ""

    body = base_text + f"\nNews: {iv_txt} (Sonar){src}"

    catalysts: List[Dict[str, Any]] = px_ctx.get("news_catalysts") or []
    if catalysts:
        top = catalysts[0]
        title = top.get("title") or ""
        url = top.get("url") or ""
        if title:
            if url:
                body += f'\nTop catalyst: <a href="{url}">{title}</a>'
            else:
                body += f"\nTop catalyst: {title}"

    return body
