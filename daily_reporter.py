# daily_reporter.py
import os
import json
import math
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date as dt_date, timezone, timedelta

import httpx

from engine_runtime import get_http_client
from engine_common import CDT_TZ, market_now, POLYGON_API_KEY
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Storage location (simple JSONL). Works on ephemeral fs too.
DATA_DIR = os.getenv("DATA_DIR", "/tmp")
ALERTS_PATH = os.path.join(DATA_DIR, "daily_alerts.jsonl")

# ------------ Utilities ------------

def _ensure_dirs() -> None:
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass

def _to_cdt(d: datetime) -> datetime:
    return d.astimezone(CDT_TZ)

def _cdt_date_of(dt: datetime) -> dt_date:
    return _to_cdt(dt).date()

def _fmt(n: Optional[float], nd=2) -> str:
    try:
        if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
            return "-"
        if abs(float(n) - round(float(n))) < 1e-9:
            return str(int(round(float(n))))
        return f"{float(n):.{nd}f}"
    except Exception:
        return "-"

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

# ------------ Public API ------------

def log_alert_snapshot(alert: Dict[str, Any], option_ticker: Optional[str], f: Dict[str, Any]) -> None:
    """
    Called at alert-time from engine_processor.
    Stores a one-line JSON record with alert-time prices for later EOD comparison.
    """
    _ensure_dirs()
    ts = market_now()
    record = {
        "ts": ts.isoformat(),
        "date_cdt": _cdt_date_of(ts).isoformat(),
        "symbol": alert.get("symbol"),
        "side": (alert.get("side") or "").upper(),
        "option_ticker": option_ticker,
        "strike": alert.get("strike"),
        "expiry": alert.get("expiry"),
        # prices at alert time
        "bid": _safe_float(f.get("bid")),
        "ask": _safe_float(f.get("ask")),
        "mid": _safe_float(f.get("mid")),
        "last": _safe_float(f.get("last")),
        # extra context
        "nbbo_provider": f.get("nbbo_provider"),
        "synthetic_nbbo_used": bool(f.get("synthetic_nbbo_used")),
        "spread_pct": _safe_float(f.get("option_spread_pct")),
    }
    try:
        with open(ALERTS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        # swallow; reporting is best-effort
        pass


async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {"status": r.status_code, "body": r.text[:400]}
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else {"status": 500, "error": "non-json"}
    except Exception as e:
        return {"status": 500, "error": str(e)}


async def _fetch_option_eod_last(option_ticker: str, day: dt_date) -> Optional[float]:
    """
    Get end-of-day 'last' for the option.
    Priority:
      1) polygon v1 open-close (close)
      2) polygon last-quote (if same day after close; best-effort)
      3) polygon v2 aggs 1-min up to 15:00 CDT (close of last bar)
    Returns None if all attempts fail or API not configured.
    """
    if not POLYGON_API_KEY or not option_ticker:
        return None

    client = get_http_client()
    if client is None:
        return None

    enc = option_ticker.replace("/", "%2F")  # safe encode (OCC contains ':', digits ok)
    # 1) open-close
    oc = await _http_json(
        client,
        f"https://api.polygon.io/v1/open-close/options/{enc}/{day.isoformat()}",
        {"apiKey": POLYGON_API_KEY},
        timeout=8.0,
    )
    close_px = None
    if isinstance(oc, dict) and oc.get("status") not in (400, 402, 403, 404, 429, 500):
        # Known shape: { "close": 0.52, ... }
        close_px = _safe_float(oc.get("close"))
        if close_px is not None:
            return close_px

    # 2) last-quote (best-effort; not strictly EOD)
    lq = await _http_json(
        client,
        f"https://api.polygon.io/v3/quotes/options/{enc}/last",
        {"apiKey": POLYGON_API_KEY},
        timeout=6.0,
    )
    if isinstance(lq, dict) and lq.get("status") not in (400, 402, 403, 404, 429, 500):
        # Shape: { "results": { "P": { "ask":..., "bid":..., "price":... } } }
        try:
            res = lq.get("results") or {}
            # 'price' may be trade price (if last trade), otherwise we fallback to mid
            price = None
            if isinstance(res, dict):
                # some payloads nest differently; try common keys
                price = res.get("price") or res.get("p") or res.get("last", {}).get("price")
            price = _safe_float(price)
            if price is not None:
                return price
        except Exception:
            pass

    # 3) minute aggs up to market close (15:00 CDT)
    # Build UTC window that covers CDT day 08:30->15:00
    start_cdt = datetime(day.year, day.month, day.day, 8, 30, tzinfo=CDT_TZ)
    end_cdt = datetime(day.year, day.month, day.day, 15, 0, tzinfo=CDT_TZ)
    frm = start_cdt.astimezone(timezone.utc).isoformat()
    to = (end_cdt.astimezone(timezone.utc) + timedelta(minutes=1)).isoformat()

    aggs = await _http_json(
        client,
        f"https://api.polygon.io/v2/aggs/ticker/{enc}/range/1/minute/{frm}/{to}",
        {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY},
        timeout=10.0,
    )
    try:
        arr = aggs.get("results") if isinstance(aggs, dict) else None
        if isinstance(arr, list) and arr:
            # last bar close
            return _safe_float(arr[-1].get("c"))
    except Exception:
        pass

    return None


def _load_alerts_for_date(target: Optional[dt_date]) -> List[Dict[str, Any]]:
    _ensure_dirs()
    if not os.path.exists(ALERTS_PATH):
        return []
    out: List[Dict[str, Any]] = []
    with open(ALERTS_PATH, "r", encoding="utf-8") as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if target:
                if rec.get("date_cdt") == target.isoformat():
                    out.append(rec)
            else:
                # default to today (CDT)
                if rec.get("date_cdt") == _cdt_date_of(market_now()).isoformat():
                    out.append(rec)
    return out


async def build_daily_report(target_date: Optional[dt_date] = None) -> Dict[str, Any]:
    """
    Build the JSON report and a monospace table for Telegram.
    """
    rows = _load_alerts_for_date(target_date)
    target = target_date or _cdt_date_of(market_now())

    # Pre-fetch unique contracts' EOD last (avoid N calls per alert)
    uniq = sorted({r.get("option_ticker") for r in rows if r.get("option_ticker")})
    eod_cache: Dict[str, Optional[float]] = {}
    for tk in uniq:
        try:
            eod_cache[tk] = await _fetch_option_eod_last(tk, target)
        except Exception:
            eod_cache[tk] = None
        await asyncio.sleep(0.05)  # tiny jitter to be polite

    # Compose table rows
    table_rows: List[List[str]] = []
    header = ["Contract", "Side", "Strike", "Exp", "Alert Last", "EOD Last", "Bid/Ask@Alert", "Result"]
    for r in rows:
        tk = r.get("option_ticker") or "-"
        side = r.get("side") or "-"
        strike = r.get("strike")
        exp = r.get("expiry") or "-"
        last_alert = _safe_float(r.get("last")) or _safe_float(r.get("mid"))
        eod_last = eod_cache.get(tk)
        if last_alert is None or eod_last is None:
            result = "UNKNOWN"
        else:
            if eod_last > last_alert:
                result = "SUCCESS"
            elif eod_last < last_alert:
                result = "FAIL"
            else:
                result = "FLAT"
        table_rows.append([
            tk,
            side,
            _fmt(strike, 2),
            str(exp),
            _fmt(last_alert, 4),
            _fmt(eod_last, 4),
            f"{_fmt(r.get('bid'),4)}/{_fmt(r.get('ask'),4)}",
            result
        ])

    # Fit into a monospace table
    cols = list(zip(*( [header] + table_rows ))) if table_rows else []
    widths = [max(len(str(x)) for x in col) for col in cols] if cols else [9,4,6,8,10,9,14,6]

    def fmt_row(row: List[str]) -> str:
        parts = []
        for i, cell in enumerate(row):
            w = widths[i]
            parts.append(str(cell).ljust(w))
        return "  ".join(parts)

    lines = []
    lines.append(fmt_row(header))
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table_rows:
        lines.append(fmt_row(row))

    table_text = "Daily Options Alert Report — " + target.isoformat() + "\n```\n" + ("\n".join(lines) if lines else "(no alerts)") + "\n```"

    return {
        "date": target.isoformat(),
        "count": len(table_rows),
        "rows": table_rows,
        "table": table_text,
    }


async def send_daily_report_to_telegram(target_date: Optional[dt_date] = None) -> Dict[str, Any]:
    rep = await build_daily_report(target_date)
    ok = False; err = None
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set", "preview": rep.get("table")}

    msg = rep.get("table") or "(empty report)"
    try:
        # Telegram has a 4096 char limit per message; chunk if needed
        max_len = 4000
        if len(msg) <= max_len:
            await send_telegram(msg)
        else:
            head = "Daily Options Alert Report (part 1)\n"
            await send_telegram(head + msg[:max_len])
            idx = 2
            p = max_len
            while p < len(msg):
                await asyncio.sleep(0.4)
                await send_telegram(f"(part {idx})\n" + msg[p:p+max_len])
                p += max_len
                idx += 1
        ok = True
    except Exception as e:
        ok = False
        err = str(e)
    return {"ok": ok, "error": err or "", "count": rep.get("count", 0)}
