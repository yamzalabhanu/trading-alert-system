from __future__ import annotations

import os
import html
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
# Formatting helpers
# ==========================

def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "—"


def _escape(s: Optional[str]) -> str:
    return html.escape(s or "")


def _gate_line(gates: Optional[Dict[str, Optional[bool]]]) -> Optional[str]:
    """
    Turn a dict of gate_name -> bool/None into a compact line of checkmarks.
    Example input:
      {
        "NBBO": True, "Spread": True, "OI": True, "MTF": False,
        "S/R": True, "EM/BE": True, "DTE": True, "Δ-band": True
      }
    """
    if not gates:
        return None
    parts: List[str] = []
    for k, v in gates.items():
        mark = "✅" if v is True else ("⚠️" if v is False else "➖")
        parts.append(f"{_escape(k)}{mark}")
    return "Gates: " + "  ".join(parts)


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
    src = f' <a href="{_escape(cite)}">[src]</a>' if cite else ""

    body = base_text + f"\nNews: {iv_txt} (Sonar){src}"

    catalysts: List[Dict[str, Any]] = px_ctx.get("news_catalysts") or []
    if catalysts:
        top = catalysts[0]
        title = _escape(top.get("title") or "")
        url = top.get("url") or ""
        if title:
            if url:
                body += f'\nTop catalyst: <a href="{_escape(url)}">{title}</a>'
            else:
                body += f"\nTop catalyst: {title}"

    return body


# =====================================
# Scores block (base → boost → final)
# =====================================

def append_scores_block(
    base_text: str,
    *,
    base_score: Optional[float],
    final_score: Optional[float],
    news_boost: Optional[float] = None,
    gates: Optional[Dict[str, Optional[bool]]] = None,
) -> str:
    """
    Appends a compact scores section to the alert body.

    Args:
        base_text: existing alert text (HTML parse_mode).
        base_score: score before Perplexity/news boost.
        final_score: score after boost (or same as base if no boost).
        news_boost: optional additive boost applied (e.g., +0.05).
        gates: optional dict of gate_name -> bool/None to render checkmarks.

    Returns:
        Updated alert text with a "Scores" section appended.
    """
    body = base_text

    # Scores header
    b = _fmt(base_score)
    f = _fmt(final_score)
    if news_boost is None:
        body += f"\nScores: Base <b>{b}</b> → Final <b>{f}</b>"
    else:
        nb = _fmt(news_boost)
        body += f"\nScores: Base <b>{b}</b>  |  News +<b>{nb}</b>  →  Final <b>{f}</b>"

    # Optional gates line
    gline = _gate_line(gates)
    if gline:
        body += f"\n{gline}"

    return body


# =====================================
# One-call convenience combiner (optional)
# =====================================

def append_scores_and_perplexity(
    base_text: str,
    *,
    base_score: Optional[float],
    final_score: Optional[float],
    px_ctx: Optional[Dict[str, Any]] = None,
    news_boost: Optional[float] = None,
    gates: Optional[Dict[str, Optional[bool]]] = None,
) -> str:
    """
    Appends the scores block, then any Perplexity insights.
    """
    text = append_scores_block(
        base_text,
        base_score=base_score,
        final_score=final_score,
        news_boost=news_boost,
        gates=gates,
    )
    return append_perplexity_context(text, px_ctx)
