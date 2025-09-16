# daily_reporter.py
import os
import json
import math
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date, timezone, timedelta
from urllib.parse import quote

import httpx

from config import CDT_TZ
from engine_runtime import get_http_client
from telegram_client import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, send_telegram
from engine_common import POLYGON_API_KEY

log = logging.getLogger("trading_engine.daily_reporter")

DATA_DIR = os.getenv("DATA_DIR", "./data")
ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)

# -------------- Logging alerts (called at alert time) --------------
def _safe_float(x):
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def _today_cdt() -> date:
    return datetime.now(CDT_TZ).date()

def _alerts_path(d: date) -> str:
    return os.path.join(ALERTS_DIR, f"{d.isoformat()}.jsonl")

def log_alert_snapshot(alert: Dict[str, Any], option_ticker: str, f: Dict[str, Any]) -> None:
    """
    Append a JSONL line with the at-alert snapshot used for daily reporting.
    This is intentionally sync & tiny (one write) to avoid adding await points in the hot path.
    """
    try:
        d = _today_cdt()
        path = _alerts_path(d)
        rec = {
            "ts_local": datetime.now(CDT_TZ).isoformat(),
            "date": d.isoformat(),
            "symbol": alert.get("symbol"),
            "side": (alert.get("side") or "").upper(),
            "contract": option_ticker,
            "strike": _safe_float(alert.get("strike")),
            "expiry": alert.get("expiry"),
            "bid_alert": _safe_float(f.get("bid")),
            "ask_alert": _safe_float(f.get("ask")),
            "last_alert": _safe_float(f.get("last") if f.get("last") is not None else f.get("mid")),
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[daily-report] failed to log snapshot: %r", e)

# -------------- EOD fetch helpers --------------
async def _http_json(url: str, params: Dict[str, Any], timeout: float = 8.0) -> Tuple[int, Any]:
    client = get_http_client()
    close_after = False
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        close_after = True
    try:
        r = await client.get(url, params=params, timeout=timeout)
        status = r.status_code
        js = None
        try:
            js = r.json()
        except Exception:
            js = r.text
        return status, js
    finally:
        if close_after:
            await client.aclose()

async def _polygon_option_openclose(contract: str, d: date) -> Optional[Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return None
    enc = quote(contract, safe="")
    url = f"https://api.polygon.io/v1/open-close/options/{enc}/{d.isoformat()}"
    status, body = await _http_json(url, {"apiKey": POLYGON_API_KEY}, timeout=7.0)
    if status == 200 and isinstance(body, dict):
        return body
    return None

async def _polygon_option_aggs_close(contract: str, d: date) -> Optional[float]:
    """Fallback: get last 'c' from intraday aggs as EOD close."""
    if not POLYGON_API_KEY:
        return None
    enc = quote(contract, safe="")
    start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    end   = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/5/min/{start}/{end}"
    status, body = await _http_json(url, {"adjusted": "true", "sort": "asc", "limit": 2000, "apiKey": POLYGON_API_KEY}, timeout=8.5)
    if status == 200 and isinstance(body, dict) and isinstance(body.get("results"), list) and body["results"]:
        # Take the last candle close
        try:
            return float(body["results"][-1].get("c"))
        except Exception:
            return None
    return None

async def _polygon_option_last_quote(contract: str) -> Tuple[Optional[float], Optional[float]]:
    """Attempt to fetch a 'bid/ask' snapshot near now; acceptable proxy for EOD for options (no AH)."""
    if not POLYGON_API_KEY:
        return None, None
    enc = quote(contract, safe="")
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    status, body = await _http_json(url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    if status == 200 and isinstance(body, dict):
        res = body.get("results") or body.get("result")
        if isinstance(res, dict):
            # Polygon formats vary; try best-effort keys
            b = res.get("bidPrice") or res.get("bid_price") or res.get("p") or res.get("bP")
            a = res.get("askPrice") or res.get("ask_price") or res.get("p") or res.get("aP")
            try:
                bid = float(b) if b is not None else None
            except Exception:
                bid = None
            try:
                ask = float(a) if a is not None else None
            except Exception:
                ask = None
            return bid, ask
    return None, None

async def fetch_eod_for_contract(contract: str, d: date) -> Dict[str, Optional[float]]:
    """
    Returns: {bid_eod, ask_eod, last_eod}
    Strategy:
      1) try v1 open-close for options to get 'close'
      2) fallback to last aggs close
      3) bid/ask via last-quote (proxy)
    """
    last_eod: Optional[float] = None
    oc = await _polygon_option_openclose(contract, d)
    if oc is not None:
        # Try common fields
        for k in ("close", "closePrice", "close_price", "closing_price"):
            if oc.get(k) is not None:
                try:
                    last_eod = float(oc[k])
                    break
                except Exception:
                    pass
        # Some payloads embed it in "results"
        if last_eod is None and isinstance(oc.get("results"), dict):
            try:
                last_eod = float(oc["results"].get("close"))
            except Exception:
                pass

    if last_eod is None:
        last_eod = await _polygon_option_aggs_close(contract, d)

    bid_eod, ask_eod = await _polygon_option_last_quote(contract)

    return {"bid_eod": bid_eod, "ask_eod": ask_eod, "last_eod": last_eod}

# -------------- Report builder --------------
def _load_alerts(d: date) -> List[Dict[str, Any]]:
    path = _alerts_path(d)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _result_label(last_alert: Optional[float], last_eod: Optional[float]) -> str:
    if last_alert is None or last_eod is None:
        return "UNKNOWN"
    if last_eod > last_alert:
        return "SUCCESS"
    if last_eod < last_alert:
        return "FAILED"
    return "FLAT"

def _fmt(v: Any, nd: int = 2) -> str:
    try:
        if v is None: return "—"
        x = float(v)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.{nd}f}"
    except Exception:
        return "—"

def _make_table(rows: List[Dict[str, Any]]) -> str:
    """
    Build monospaced table suitable for Telegram (<= ~4000 chars per message; chunk if needed by caller).
    Columns:
      Contract | Dir | Strike | Exp | Bid/Ask/Last@Alert | Bid/Ask/Last@EOD | Result
    """
    headers = ["Contract", "Dir", "Strike", "Exp", "At Alert (B/A/L)", "EOD (B/A/L)", "Result"]
    data = []
    for r in rows:
        at_alert = f"{_fmt(r.get('bid_alert'))}/{_fmt(r.get('ask_alert'))}/{_fmt(r.get('last_alert'))}"
        at_eod   = f"{_fmt(r.get('bid_eod'))}/{_fmt(r.get('ask_eod'))}/{_fmt(r.get('last_eod'))}"
        data.append([
            r.get("contract") or "?",
            (r.get("side") or "?")[0],
            _fmt(r.get("strike"), 2),
            r.get("expiry") or "?",
            at_alert,
            at_eod,
            _result_label(r.get('last_alert'), r.get('last_eod')),
        ])
    # calc widths
    cols = list(zip(*([headers] + data))) if data else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    def fmt_row(row):
        return " | ".join(str(v).ljust(w) for v, w in zip(row, widths))
    lines = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in data]
    return "```\n" + "\n".join(lines) + "\n```"

async def build_daily_report(d: Optional[date] = None) -> Dict[str, Any]:
    target = d or _today_cdt()
    alerts = _load_alerts(target)
    if not alerts:
        return {"date": target.isoformat(), "rows": [], "table": "No alerts logged."}

    # Cache EOD per contract
    eod_cache: Dict[str, Dict[str, Optional[float]]] = {}
    rows: List[Dict[str, Any]] = []

    for rec in alerts:
        ct = rec.get("contract")
        if not ct:
            continue
        if ct not in eod_cache:
            try:
                eod_cache[ct] = await fetch_eod_for_contract(ct, target)
            except Exception as e:
                log.warning("[daily-report] eod fetch fail for %s: %r", ct, e)
                eod_cache[ct] = {"bid_eod": None, "ask_eod": None, "last_eod": None}

        out = {
            "contract": ct,
            "side": rec.get("side"),
            "strike": rec.get("strike"),
            "expiry": rec.get("expiry"),
            "bid_alert": rec.get("bid_alert"),
            "ask_alert": rec.get("ask_alert"),
            "last_alert": rec.get("last_alert"),
            "bid_eod": eod_cache[ct].get("bid_eod"),
            "ask_eod": eod_cache[ct].get("ask_eod"),
            "last_eod": eod_cache[ct].get("last_eod"),
        }
        rows.append(out)

    table = _make_table(rows)
    return {"date": target.isoformat(), "rows": rows, "table": table}

# -------------- Telegram sender --------------
async def send_daily_report_to_telegram(d: Optional[date] = None) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"ok": False, "error": "TELEGRAM not configured"}

    rep = await build_daily_report(d)
    text = f"📊 Daily Options Report — {rep['date']}\n{rep['table']}"
    # Telegram message length guard; split if too long
    if len(text) <= 3800:
        await send_telegram(text)
        return {"ok": True, "count": len(rep["rows"])}
    # chunk by lines
    header = f"📊 Daily Options Report — {rep['date']}\n"
    body = rep["table"]
    # crude split at code fences if needed
    chunks: List[str] = []
    cur = ""
    for line in body.splitlines(True):
        if len(header) + len(cur) + len(line) > 3800:
            chunks.append(cur)
            cur = ""
        cur += line
    if cur:
        chunks.append(cur)
    for i, part in enumerate(chunks, 1):
        await send_telegram(f"{header}```\n{part.strip('`')}\n``` (part {i}/{len(chunks)})")
    return {"ok": True, "count": len(rep["rows"]), "parts": len(chunks)}
