# reporting.py
import asyncio
import contextlib
from datetime import datetime, timezone, date
from typing import List, Dict, Any
from collections import Counter
from config import MARKET_TZ
from telegram_client import send_telegram

# Global state for decisions log
_DECISIONS_LOG: List[Dict[str, Any]] = []

def _format_contract_lines(entries: List[Dict[str, Any]]) -> List[str]:
        """One line per processed alert with contract + key features and LLM outcome."""
    lines = []
    for e in entries:
        sym   = e.get("symbol")
        side  = e.get("side")
        ct    = e.get("option_ticker") or "—"
        f     = e.get("features", {})
        llm   = e.get("llm", {})
        dte_v = f.get("dte")
        delta = f.get("delta")
        iv    = f.get("iv")
        mid   = f.get("opt_mid")
        spr   = f.get("spread_pct")   # fraction, e.g., 0.12
        oi    = f.get("oi")
        vol   = f.get("vol")
        dec   = e.get("decision_final")
        conf  = llm.get("confidence")

        line = (
            f"- {sym} {side} | {ct} | "
            f"DTE={dte_v if dte_v is not None else '—'}  "
            f"Δ={_fmt(delta)}  IV={_fmt(iv)}  mid={_fmt(mid)}  "
            f"spread={_fmt_pct(spr)}  OI={_fmt(oi)}  Vol={_fmt(vol)}  "
            f"→ {dec.upper() if dec else '—'} (conf={_fmt(conf)})"
        )
        lines.append(line)
    return lines

    pass

def _chunk_lines_for_telegram(lines: List[str], prefix: str = "", max_chars: int = 3500) -> List[str]:
      """Chunk many lines into multiple messages under max_chars (Telegram hard limit ~4096)."""
    chunks = []
    cur = prefix.strip() + ("\n" if prefix else "")
    for ln in lines:
        add_len = (1 if cur else 0) + len(ln)
        if len(cur) + add_len > max_chars:
            if cur:
                chunks.append(cur)
            cur = ln
        else:
            cur = (cur + ("\n" if cur and not cur.endswith("\n") else "")) + ln
    if cur:
        chunks.append(cur)
    return chunks
    pass

def _summarize_day_for_report(local_date: date) -> Dict[str, Any]:
        entries = [e for e in _DECISIONS_LOG if e["timestamp_local"].date() == local_date]
    total_alerts = len(entries)
    llm_runs = sum(1 for e in entries if e["llm"]["ran"])
    llm_skips = total_alerts - llm_runs
    buys = sum(1 for e in entries if e["decision_final"] == "buy")
    skips = total_alerts - buys
    avg_conf = (sum(float(e["llm"].get("confidence") or 0.0) for e in entries if e["llm"]["ran"]) / llm_runs) if llm_runs else 0.0
    by_symbol = Counter((e["symbol"] for e in entries))
    top = ", ".join(f"{sym}({cnt})" for sym, cnt in by_symbol.most_common(5)) or "—"
    by_outcome = Counter((e["decision_path"] for e in entries))
    quota = llm_quota_snapshot()

    header = f"📊 Daily Report — {local_date.isoformat()} ({MARKET_TZ})"
    body = [
        f"Alerts: {total_alerts} | LLM ran: {llm_runs} | skips: {llm_skips}",
        f"Decisions — BUY: {buys} | SKIP: {skips}",
        f"Avg LLM confidence (when ran): {avg_conf:.2f}",
        f"Top tickers: {top}",
        f"Paths: {dict(by_outcome)}",
        "",
        f"Quota used (tracked only): {quota['used']}/{quota['max']} (remaining {quota['remaining']})",
        "",
        "⚠️ Educational demo; not financial advice."
    ]
    header_text = header + "\n" + "\n".join(body)

    contract_lines = _format_contract_lines(entries)
    return {"header": header_text, "contracts": contract_lines, "count": total_alerts}
    pass

async def _send_daily_report_now() -> Dict[str, Any]:
    today_local = market_now().date()
    rep = _summarize_day_for_report(today_local)

    sent = []
    first = await send_telegram(rep["header"])
    sent.append(first)

    if rep["contracts"]:
        chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")
        for msg in chunks:
            sent.append(await send_telegram(msg))

    return {"ok": True, "sent": any(bool(x) for x in sent), "result": sent}
    pass

async def _daily_report_scheduler():
      while True:
        now_utc = datetime.now(timezone.utc)
        next_utc = _next_report_dt_utc(now_utc)
        sleep_s = max(1, int((next_utc - now_utc).total_seconds()))
        try:
            await asyncio.sleep(sleep_s)
            await _send_daily_report_now()
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1)
    pass

async def send_daily_report():
    return await _send_daily_report_now()