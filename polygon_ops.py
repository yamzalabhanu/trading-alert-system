# polygon_ops.py
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import httpx

from engine_runtime import get_http_client

logger = logging.getLogger("trading_engine")

# ========= ENV =========
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# ========= HTTP helpers (shared) =========
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any], timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """
    GET JSON and swallow 402/403/404/429 by returning None.
    For NBBO probing we usually want to *see* non-200s; use _http_get_any for that.
    """
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

async def _http_get_any(url: str, params: dict | None = None, timeout: float = 8.0) -> Dict[str, Any]:
    """
    GET and return {"status": int|None, "body": json|text|None, "error": "..."} without raising.
    """
    client = get_http_client()
    if client is None:
        return {"status": None, "error": "HTTP client not ready"}
    try:
        r = await client.get(url, params=params or {}, timeout=timeout)
        ct = r.headers.get("content-type", "")
        try:
            payload = r.json() if "application/json" in ct else r.text
        except Exception:
            payload = r.text
        return {"status": r.status_code, "body": payload}
    except Exception as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}"}

# ========= Small utils =========
def _norm_ts_to_seconds(ts: Any) -> Optional[float]:
    """
    Normalize Polygon timestamps (ns/us/ms/s) into seconds (float).
    """
    try:
        if ts is None:
            return None
        ns = int(ts)
        # Heuristics for unit
        if ns >= 10**14:   sec = ns / 1e9      # ns
        elif ns >= 10**11: sec = ns / 1e6      # us
        elif ns >= 10**8:  sec = ns / 1e3      # ms
        else:              sec = float(ns)     # s
        return float(sec)
    except Exception:
        try:
            return float(ts)
        except Exception:
            return None

def _age_from_ts(ts: Any) -> Optional[float]:
    sec = _norm_ts_to_seconds(ts)
    if sec is None:
        return None
    try:
        return max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    except Exception:
        return None

def _mk_spread_pct(bid: Optional[float], ask: Optional[float], mid: Optional[float]) -> Optional[float]:
    try:
        if bid is None or ask is None:
            return None
        if mid is None:
            mid = (float(bid) + float(ask)) / 2.0
        if mid <= 0:
            return None
        return round(((float(ask) - float(bid)) / float(mid)) * 100.0, 3)
    except Exception:
        return None

def _extract_bid_ask_ts_from_snapshot_last_quote(last_quote: Any) -> Dict[str, Any]:
    """
    Polygon snapshot `results.last_quote` for options can present bid/ask under a few different keys
    depending on response version. We try several.
    """
    if not isinstance(last_quote, dict):
        return {}
    # Candidate key sets (ordered)
    key_sets = [
        ("p", "P", "t"),                    # common in snapshot: p=bid, P=ask, t=ts
        ("bid_price", "ask_price", "t"),
        ("bid", "ask", "t"),
        ("bidPrice", "askPrice", "t"),
        ("bp", "ap", "t"),
    ]
    for kb, ka, kt in key_sets:
        bid = last_quote.get(kb)
        ask = last_quote.get(ka)
        ts  = last_quote.get(kt)
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            return {"bid": float(bid), "ask": float(ask), "ts": ts}
    # As a last attempt, scan for any numeric bid*/ask* keys
    num_bid = None
    num_ask = None
    tstamp  = last_quote.get("t")
    for k, v in last_quote.items():
        lk = k.lower()
        if "bid" in lk and isinstance(v, (int, float)) and num_bid is None:
            num_bid = float(v)
        if "ask" in lk and isinstance(v, (int, float)) and num_ask is None:
            num_ask = float(v)
    if num_bid is not None and num_ask is not None:
        return {"bid": num_bid, "ask": num_ask, "ts": tstamp}
    return {}

def _extract_bid_ask_ts_from_quotes_item(q: Any) -> Dict[str, Any]:
    """
    For /v3/quotes/{TK} results[0]
    Expected keys: bid_price, ask_price, sip_timestamp
    """
    if not isinstance(q, dict):
        return {}
    # Preferred keys
    bid = q.get("bid_price"); ask = q.get("ask_price"); ts = q.get("sip_timestamp")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        return {"bid": float(bid), "ask": float(ask), "ts": ts}
    # Fallback variants (be forgiving)
    for kb, ka, kt in [
        ("bidPrice", "askPrice", "sipTimestamp"),
        ("bp", "ap", "t"),
        ("bid", "ask", "t"),
    ]:
        bid = q.get(kb); ask = q.get(ka); ts = q.get(kt)
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            return {"bid": float(bid), "ask": float(ask), "ts": ts}
    # Last attempt: scan for numeric bid/ask
    num_bid = None
    num_ask = None
    tstamp  = q.get("sip_timestamp") or q.get("t")
    for k, v in q.items():
        lk = k.lower()
        if "bid" in lk and isinstance(v, (int, float)) and num_bid is None:
            num_bid = float(v)
        if "ask" in lk and isinstance(v, (int, float)) and num_ask is None:
            num_ask = float(v)
    if num_bid is not None and num_ask is not None:
        return {"bid": num_bid, "ask": num_ask, "ts": tstamp}
    return {}

# ========= NBBO probes (UPDATED) =========
async def _pull_nbbo_direct(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    UPDATED: Snapshot-first NBBO probe with quotes fallback.
      1) GET /v3/snapshot/options/{UL}/{TK}
      2) If absent, GET /v3/quotes/{TK}?limit=1&order=desc&sort=timestamp
    Returns a partial feature dict: bid, ask, mid, option_spread_pct, quote_age_sec.
    """
    out: Dict[str, Any] = {}
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return out

    # Try snapshot (requires underlying)
    if underlying:
        try:
            s_url = f"https://api.polygon.io/v3/snapshot/options/{underlying}/{option_ticker}"
            s_js = await _http_json(client, s_url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
            if isinstance(s_js, dict):
                last_quote = (s_js.get("results") or {}).get("last_quote")
                got = _extract_bid_ask_ts_from_snapshot_last_quote(last_quote)
                if got.get("bid") is not None and got.get("ask") is not None:
                    bid = got["bid"]; ask = got["ask"]
                    mid = (bid + ask) / 2.0
                    out["bid"] = bid
                    out["ask"] = ask
                    out["mid"] = round(mid, 4)
                    out["option_spread_pct"] = _mk_spread_pct(bid, ask, mid)
                    out["quote_age_sec"] = _age_from_ts(got.get("ts"))
                    return out
        except Exception:
            pass

    # Fallback: newest historical quote
    try:
        q_url = f"https://api.polygon.io/v3/quotes/{option_ticker}"
        q_js = await _http_json(
            client, q_url,
            {"apiKey": POLYGON_API_KEY, "limit": 1, "order": "desc", "sort": "timestamp"},
            timeout=6.0
        )
        if isinstance(q_js, dict):
            items: List[dict] = q_js.get("results") or []
            if items:
                got = _extract_bid_ask_ts_from_quotes_item(items[0])
                if got.get("bid") is not None and got.get("ask") is not None:
                    bid = got["bid"]; ask = got["ask"]
                    mid = (bid + ask) / 2.0
                    out["bid"] = bid
                    out["ask"] = ask
                    out["mid"] = round(mid, 4)
                    out["option_spread_pct"] = _mk_spread_pct(bid, ask, mid)
                    out["quote_age_sec"] = _age_from_ts(got.get("ts"))
                    return out
    except Exception:
        pass

    return out

async def _probe_nbbo_verbose(option_ticker: str, underlying: Optional[str] = None) -> Dict[str, Any]:
    """
    UPDATED: Verbose probe that surfaces HTTP status + reason for both snapshot and quotes endpoints.
    Fills NBBO fields if found anywhere.
    """
    out: Dict[str, Any] = {}

    if not POLYGON_API_KEY:
        return {"nbbo_reason": "no POLYGON_API_KEY in env"}

    # A) Snapshot first (only if we have underlying)
    snap_status = None
    if underlying:
        try:
            s_url = f"https://api.polygon.io/v3/snapshot/options/{underlying}/{option_ticker}"
            s_res = await _http_get_any(s_url, params={"apiKey": POLYGON_API_KEY}, timeout=8.0)
            snap_status = s_res.get("status")
            out["snapshot_http_status"] = snap_status
            body = s_res.get("body")
            if snap_status == 200 and isinstance(body, dict):
                last_quote = (body.get("results") or {}).get("last_quote")
                got = _extract_bid_ask_ts_from_snapshot_last_quote(last_quote)
                if got.get("bid") is not None and got.get("ask") is not None:
                    bid = got["bid"]; ask = got["ask"]
                    mid = (bid + ask) / 2.0
                    out.update({
                        "bid": bid, "ask": ask,
                        "mid": round(mid, 4),
                        "option_spread_pct": _mk_spread_pct(bid, ask, mid),
                        "quote_age_sec": _age_from_ts(got.get("ts")),
                    })
                else:
                    out["nbbo_reason"] = "snapshot has no last_quote bid/ask"
            else:
                if isinstance(body, dict):
                    out["nbbo_reason"] = body.get("error") or body.get("message") or "snapshot non-200"
                else:
                    out["nbbo_reason"] = "snapshot non-200"
        except Exception as e:
            out["snapshot_error"] = f"{type(e).__name__}: {e}"

    # If already filled by snapshot, return now
    if out.get("bid") is not None and out.get("ask") is not None:
        out["nbbo_http_status"] = out.get("snapshot_http_status")
        return out

    # B) Quotes fallback
    try:
        q_url = f"https://api.polygon.io/v3/quotes/{option_ticker}"
        q_res = await _http_get_any(q_url, params={"apiKey": POLYGON_API_KEY, "limit": 1, "order": "desc", "sort": "timestamp"}, timeout=8.0)
        q_status = q_res.get("status")
        out["quotes_http_status"] = q_status
        body = q_res.get("body")
        if q_status == 200 and isinstance(body, dict):
            items = body.get("results") or []
            if items:
                got = _extract_bid_ask_ts_from_quotes_item(items[0])
                if got.get("bid") is not None and got.get("ask") is not None:
                    bid = got["bid"]; ask = got["ask"]
                    mid = (bid + ask) / 2.0
                    out.update({
                        "bid": bid, "ask": ask,
                        "mid": round(mid, 4),
                        "option_spread_pct": _mk_spread_pct(bid, ask, mid),
                        "quote_age_sec": _age_from_ts(got.get("ts")),
                    })
                else:
                    out["nbbo_reason"] = "quotes[0] missing bid/ask"
            else:
                out["nbbo_reason"] = "no quotes returned"
        else:
            if isinstance(body, dict):
                out["nbbo_reason"] = body.get("error") or body.get("message") or "quotes non-200"
            else:
                out["nbbo_reason"] = "quotes non-200"
    except Exception as e:
        out["quotes_error"] = f"{type(e).__name__}: {e}"

    # For compatibility with previous logs
    out["nbbo_http_status"] = out.get("snapshot_http_status") or out.get("quotes_http_status")
    return out

# ========= Listing & replacement helpers (unchanged interfaces) =========
async def _poly_reference_contracts_exists(underlying: str, expiry_iso: str, ticker: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {"listed": None, "snapshot_ok": None, "reason": "no HTTP or API key"}
    try:
        # A) reference list
        base = "https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "expiration_date": expiry_iso,
            "limit": 1000,
            "apiKey": POLYGON_API_KEY,
        }
        r = await client.get(base, params=params, timeout=8.0)
        listed = False
        if r.status_code == 200:
            js = r.json()
            for it in js.get("results", []):
                if it.get("ticker") == ticker:
                    listed = True
                    break
        elif r.status_code in (402, 403, 429):
            return {"listed": None, "snapshot_ok": None, "reason": f"ref-contracts {r.status_code}"}

        # B) single snapshot check (sometimes ref list lags)
        s = await client.get(
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{ticker}",
            params={"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        snapshot_ok = (s.status_code == 200 and isinstance((s.json() or {}).get("results"), dict))
        return {"listed": listed, "snapshot_ok": snapshot_ok, "reason": None}
    except Exception as e:
        return {"listed": None, "snapshot_ok": None, "reason": f"error: {type(e).__name__}: {e}"}

# These two rely on market_ops.scan_top_candidates_for_alert in your codebase.
# Keep signatures stable; the ranking logic remains the same.
async def _rescan_best_replacement(
    symbol: str,
    side: str,
    desired_strike: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    If the chosen contract truly isn't listed, pick a nearby *listed* contract in the same expiry.
    This function is called by your processor when 404 is confirmed.
    """
    from market_ops import scan_top_candidates_for_alert  # late import to avoid cycles
    try:
        try:
            pool = await scan_top_candidates_for_alert(
                get_http_client(),
                symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol,
                min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=12,
                restrict_expiries=[expiry_iso],  # type: ignore
            ) or []
        except TypeError:
            pool_any = await scan_top_candidates_for_alert(
                get_http_client(),
                symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol,
                min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=15,
            ) or []
            pool = [it for it in pool_any if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        nbbo_missing = 0 if (it.get("bid") is not None and it.get("ask") is not None) else 1
        return (sd, nbbo_missing, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

async def _find_nbbo_replacement_same_expiry(
    symbol: str,
    side: str,
    desired_strike: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int,
) -> Optional[Dict[str, Any]]:
    """
    Softer replacement when the current pick is listed but has no NBBO:
    choose another contract in the same expiry that *does* have NBBO.
    """
    from market_ops import scan_top_candidates_for_alert  # late import to avoid cycles
    try:
        try:
            pool = await scan_top_candidates_for_alert(
                get_http_client(),
                symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol,
                min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=14,
                restrict_expiries=[expiry_iso],  # type: ignore
            ) or []
        except TypeError:
            pool = await scan_top_candidates_for_alert(
                get_http_client(),
                symbol,
                {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
                min_vol=min_vol,
                min_oi=min_oi,
                top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
                top_overall=16,
            ) or []
            pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    # Require NBBO present
    pool = [it for it in pool if it.get("bid") is not None and it.get("ask") is not None]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        return (sd, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

__all__ = [
    # http helpers
    "_http_json", "_http_get_any",
    # nbbo helpers (updated)
    "_pull_nbbo_direct", "_probe_nbbo_verbose",
    # listing / replacements
    "_poly_reference_contracts_exists", "_rescan_best_replacement", "_find_nbbo_replacement_same_expiry",
]
