# reporting.py
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
from config import CDT_TZ
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Public log the rest of the app can append to
_DECISIONS_LOG: List[Dict[str, Any]] = []


def _format_contract_line(e: Dict[str, Any]) -> str:
    sym = e.get("symbol", "?")
    side = e.get("side", "?")
    otkr = e.get("option_ticker") or "—"
    path = e.get("decision_path") or "n/a"
    fin = e.get("decision_final") or "n/a"
    fz = e.get("features") or {}
    ivr = fz.get("iv_rank")
    oi = fz.get("oi")
    vol = fz.get("vol")
    spr = fz.get("spread_pct") or fz.get("option_spread_pct")
    dte = fz.get("dte")

    bits = [f"{sym} {side}", f"{otkr}", f"→ {fin} [{path}]"]
    extras = []
    if dte is not None:
        extras.append(f"DTE:{dte}")
    if ivr is not None:
        extras.append(f"IVr:{round(ivr,3)}")
    if oi is not None:
        extras.append(f"OI:{oi}")
    if vol is not None:
        extras.append(f"Vol:{vol}")
    if spr is not None:
        extras.append(f"Spr%:{round(spr,3)}")
    if extras:
        bits.append("(" + ", ".join(map(str, extras)) + ")")
    return " ".join(map(str, bits))


def _chunk_lines_for_telegram(lines: List[str], prefix: str = "", max_chars: int = 3500) -> List[str]:
    """
    Telegram has a ~4096 char message limit. We keep a safe margin.
    Returns a list of message chunks (each is a string).
    """
    chunks: List[str] = []
    current = prefix.strip()
    if current:
        current += "\n"

    for line in lines:
        line = str(line)
        # +1 for newline if we add this line
        if len(current) + len(line) + 1 > max_chars:
            if current.strip():
                chunks.append(current.rstrip())
            current = ""
        current += line + "\n"

    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def _summarize_day_for_report(day_date) -> Dict[str, Any]:
    """
    Build a simple daily summary from _DECISIONS_LOG for the given local date (CDT).
    Returns: {header, count, contracts:[...]}
    """
    day_items = []
    for e in _DECISIONS_LOG:
        ts = e.get("timestamp_local")
        if isinstance(ts, datetime) and ts.astimezone(CDT_TZ).date() == day_date:
            day_items.append(e)

    # Sort newest first
    day_items.sort(key=lambda x: x.get("timestamp_local") or datetime.now(CDT_TZ), reverse=True)

    count = len(day_items)
    header = f"📊 Daily Report — {day_date.isoformat()} (CDT) — {count} decisions"
    contracts = [_format_contract_line(e) for e in day_items]
    return {"header": header, "count": count, "contracts": contracts}


async def _send_daily_report_now() -> Dict[str, Any]:
    """
    Compose today's report (CDT) and send it to Telegram if configured.
    Returns metadata about what would/were sent.
    """
    today_local = datetime.now(CDT_TZ).date()
    rep = _summarize_day_for_report(today_local)
    chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")

    sent = 0
    results: List[Dict[str, Any]] = []

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        # Send header first, then the chunks
        hres = await send_telegram(rep["header"])
        results.append(hres or {"note": "not_configured"})
        sent += 1 if hres and not hres.get("error") else 0

        for c in chunks:
            r = await send_telegram(c)
            results.append(r or {"note": "not_configured"})
            sent += 1 if r and not r.get("error") else 0
    else:
        # Not configured — return what we'd send
        results.append({"note": "telegram_not_configured"})

    return {
        "date": str(today_local),
        "count": rep["count"],
        "chunks": len(chunks),
        "sent": sent,
        "preview_header": rep["header"],
        "preview_first_chunk": (chunks[0] if chunks else ""),
        "results": results,
    }
