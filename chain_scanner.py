# chain_scanner.py
import os
import re
from datetime import datetime, timedelta, timezone, date
from typing import Dict, Any, List, Optional, Tuple

import httpx

from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# ---- Config knobs (env) ----
def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

SCAN_ENABLED            = _truthy(os.getenv("SCAN_CHAIN_ALERTS", "1"))
SCAN_TOP_N              = int(os.getenv("SCAN_TOP_N", "6"))          # per bucket (week)
SCAN_MIN_VOL            = int(os.getenv("SCAN_MIN_VOL", "200"))      # min day volume
SCAN_MIN_OI             = int(os.getenv("SCAN_MIN_OI", "300"))       # min OI
SCAN_INCLUDE_CALLS      = _truthy(os.getenv("SCAN_INCLUDE_CALLS", "1"))
SCAN_INCLUDE_PUTS       = _truthy(os.getenv("SCAN_INCLUDE_PUTS", "1"))
SCAN_SORT_PRIMARY       = os.getenv("SCAN_SORT_PRIMARY", "vol").lower()    # vol | oi | combo
SCAN_SORT_SECONDARY     = os.getenv("SCAN_SORT_SECONDARY", "oi").lower()   # oi | vol
SCAN_INCLUDE_SPREAD     = _truthy(os.getenv("SCAN_INCLUDE_SPREAD", "1"))    # compute spread% if NBBO present
SCAN_MAX_PER_SIDE       = int(os.getenv("SCAN_MAX_PER_SIDE", "999999"))     # optional cap per side within bucket

# ============ Helpers (time) ============
def _next_friday(d: date) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7)

def _same_week_friday(d: date) -> date:
    base_monday = d - timedelta(days=d.weekday())
    return base_monday + timedelta(days=4)

def _week_fridays(now_local: date) -> Tuple[date, date, date]:
    """Return (week1, week2, week3) Friday dates, where week1 is this week's Friday (or today if already Friday)."""
    w1 = _same_week_friday(now_local)
    # if already past Friday (weekend), bump forward to next real trading week
    if now_local > w1:
        w1 = _next_friday(now_local)
    w2 = w1 + timedelta(days=7)
    w3 = w2 + timedelta(days=7)
    return w1, w2, w3

# ============ Helpers (HTTP/Polygon) ============
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

def _encode_ticker_path(t: str) -> str:
    from urllib.parse import quote
    return quote(t or "", safe="")

def _quote_age_from_ts(ts_val: Any) -> Optional[float]:
    if ts_val is None: return None
    try:
        ns = int(ts_val)
    except Exception:
        return None
    if ns >= 10**14:    sec = ns / 1e9
    elif ns >= 10**11: sec = ns / 1e6
    elif ns >= 10**8:  sec = ns / 1e3
    else:              sec = float(ns)
    return max(0.0, datetime.now(timezone.utc).timestamp() - sec)

async def _snapshot_slice_for_expiry(
    client: httpx.AsyncClient,
    symbol: str,
    expiry_iso: str,
    side: str,  # "call" | "put"
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Uses Polygon v3 snapshot slice to pull day volume, OI, last_quote, greeks for all contracts
    of given side & expiry.
    """
    if not POLYGON_API_KEY:
        return []
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
    params = {
        "apiKey": POLYGON_API_KEY,
        "contract_type": side,
        "expiration_date": expiry_iso,
        "limit": limit,
        "greeks": "true",
        "include_greeks": "true",
    }
    js = await _http_json(client, url, params, timeout=8.0)
    if not js:
        return []
    return js.get("results") or []

def _extract_metrics(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn a snapshot item into a flat metrics dict."""
    try:
        details = item.get("details") or {}
        day     = item.get("day") or {}
        greeks  = item.get("greeks") or {}
        lq      = item.get("last_quote") or {}

        ticker  = details.get("ticker") or item.get("ticker")
        strike  = details.get("strike")
        # normalize strike if encoded
        if strike is None and ticker:
            m = re.search(r"[CP](\d{8,9})$", ticker)
            if m:
                try: strike = int(m.group(1)) / 1000.0
                except: strike = None

        vol     = day.get("volume") or day.get("v")
        oi      = item.get("open_interest")
        bid     = lq.get("bid_price")
        ask     = lq.get("ask_price")
        ts      = lq.get("sip_timestamp") or lq.get("participant_timestamp") or lq.get("trf_timestamp") or lq.get("t")
        age     = _quote_age_from_ts(ts)

        mid = None
        spread_pct = None
        if bid is not None and ask is not None and isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid and ask > 0:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_pct = (ask - bid) / mid * 100.0

        return {
            "ticker": ticker,
            "strike": strike,
            "vol": int(vol) if isinstance(vol, (int, float)) else None,
            "oi": int(oi) if isinstance(oi, (int, float)) else None,
            "bid": float(bid) if isinstance(bid, (int, float)) else None,
            "ask": float(ask) if isinstance(ask, (int, float)) else None,
            "mid": float(mid) if isinstance(mid, (int, float)) else None,
            "spread_pct": float(spread_pct) if isinstance(spread_pct, (int, float)) else None,
            "quote_age_sec": float(age) if isinstance(age, (int, float)) else None,
            "delta": greeks.get("delta"),
            "iv": item.get("implied_volatility") or greeks.get("iv"),
        }
    except Exception:
        return None

def _rank_key(m: Dict[str, Any]) -> Tuple:
    """Primary/secondary sorting logic."""
    primary = SCAN_SORT_PRIMARY
    secondary = SCAN_SORT_SECONDARY

    def _neg(x): return -(x or 0)
    p = _neg(m.get(primary)) if primary in ("vol", "oi") else -((m.get("vol") or 0) + (m.get("oi") or 0))
    s = _neg(m.get(secondary)) if secondary in ("vol", "oi") else -((m.get("vol") or 0) + (m.get("oi") or 0))
    # Use tighter spread as tiebreaker, then fresher quotes, then lower strike distance from ATM (if mid present)
    sp = (m.get("spread_pct") if m.get("spread_pct") is not None else 1e9)
    age = (m.get("quote_age_sec") if m.get("quote_age_sec") is not None else 1e9)
    return (p, s, sp, age)

def _fmt_line(m: Dict[str, Any]) -> str:
    parts = [
        m.get("ticker", ""),
        f"strike {m.get('strike')}",
        f"vol {m.get('vol')}",
        f"oi {m.get('oi')}",
    ]
    if m.get("mid") is not None:
        parts.append(f"mid {round(m['mid'],2)}")
    if SCAN_INCLUDE_SPREAD and m.get("spread_pct") is not None:
        parts.append(f"spr {round(m['spread_pct'],1)}%")
    if m.get("quote_age_sec") is not None:
        parts.append(f"age {int(m['quote_age_sec'])}s")
    return "  • " + " | ".join(parts)

async def _scan_one_bucket(client: httpx.AsyncClient, symbol: str, expiry_iso: str) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    if SCAN_INCLUDE_CALLS:
        calls = await _snapshot_slice_for_expiry(client, symbol, expiry_iso, "call")
        for it in (calls or []):
            m = _extract_metrics(it)
            if m: 
                m["side"] = "CALL"
                results.append(m)
    if SCAN_INCLUDE_PUTS:
        puts = await _snapshot_slice_for_expiry(client, symbol, expiry_iso, "put")
        for it in (puts or []):
            m = _extract_metrics(it)
            if m:
                m["side"] = "PUT"
                results.append(m)

    # Basic filters
    filt = [
        m for m in results
        if (m.get("vol") or 0) >= SCAN_MIN_VOL or (m.get("oi") or 0) >= SCAN_MIN_OI
    ]

    # Optional cap per side before global sort (keeps variety)
    if SCAN_MAX_PER_SIDE < 999999:
        side_groups = {"CALL": [], "PUT": []}
        for m in filt:
            side_groups.setdefault(m["side"], []).append(m)
        filt = []
        for side in ("CALL", "PUT"):
            g = side_groups.get(side, [])
            g.sort(key=_rank_key)
            filt.extend(g[:SCAN_MAX_PER_SIDE])

    # Final sort + trim
    filt.sort(key=_rank_key)
    top = filt[:SCAN_TOP_N]
    return {"expiry": expiry_iso, "items": top}

def _compose_telegram(symbol: str, wk1: Dict[str, Any], wk2: Dict[str, Any], wk3: Dict[str, Any]) -> str:
    def _sec(b):
        if not b or not b.get("items"):
            return "  (no contracts meeting thresholds)"
    # Header
    lines = [f"🔎 Chain Scan — {symbol}", "", f"Filters: minVol≥{SCAN_MIN_VOL} or minOI≥{SCAN_MIN_OI} · sort={SCAN_SORT_PRIMARY}/{SCAN_SORT_SECONDARY} · top={SCAN_TOP_N}"]
    # Buckets
    for label, bucket in (("Week 1", wk1), ("Week 2", wk2), ("Week 3", wk3)):
        lines.append(f"\n📅 {label} — {bucket.get('expiry')}")
        if not bucket.get("items"):
            lines.append("  (no contracts meeting thresholds)")
            continue
        for m in bucket["items"]:
            lines.append(_fmt_line(m))
    return "\n".join(lines).strip()

# ============ Public entrypoint ============
async def scan_and_alert_options_chain(client: httpx.AsyncClient, symbol: str, now_local_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Scans week1/week2/week3 expirations for high VOL/OI contracts and sends one Telegram alert.
    Returns a dict summary (for logs).
    """
    if not SCAN_ENABLED:
        return {"ok": False, "sent": False, "error": "scanner disabled"}

    if not POLYGON_API_KEY:
        return {"ok": False, "sent": False, "error": "missing POLYGON_API_KEY"}

    today = now_local_date or datetime.now().date()
    w1, w2, w3 = _week_fridays(today)

    wk1 = await _scan_one_bucket(client, symbol, w1.isoformat())
    wk2 = await _scan_one_bucket(client, symbol, w2.isoformat())
    wk3 = await _scan_one_bucket(client, symbol, w3.isoformat())

    text = _compose_telegram(symbol, wk1, wk2, wk3)

    sent = False
    err = None
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            await send_telegram(text)
            sent = True
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    else:
        err = "telegram not configured"

    return {
        "ok": sent,
        "sent": sent,
        "error": err,
        "symbol": symbol,
        "week1": wk1,
        "week2": wk2,
        "week3": wk3,
    }
