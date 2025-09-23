# polygon_ops.py
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx

from engine_runtime import get_http_client
from engine_common import POLYGON_API_KEY

logger = logging.getLogger("trading_engine")

# ---------------------------------
# Env: enable Polygon Options Advanced fast-path
# ---------------------------------
def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

POLY_ADVANCED = _truthy(os.getenv("POLY_ADVANCED", "1"))

# --------------------
# Low-level HTTP helpers
# --------------------
async def _http_json(
    client: httpx.AsyncClient,
    url: str,
    params: Dict[str, Any],
    timeout: float = 8.0
) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {
                "status": r.status_code,
                "error": True,
                "body": (r.json() if "application/json" in (r.headers.get("content-type") or "") else r.text),
            }
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None


async def _http_get_any(url: str, params: dict | None = None, timeout: float = 6.0) -> Dict[str, Any]:
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

# --------------------
# Timestamp helper (ns/ms/s -> age seconds)
# --------------------
def _poly_ts_to_age_sec(ts_ns_or_ms: Any) -> Optional[float]:
    try:
        ts = int(ts_ns_or_ms)
        # Heuristic: 19 digits ~ ns, 13 ~ ms, 10 ~ s
        if ts >= 10**15:
            t = ts / 1_000_000_000  # ns -> s
        elif ts >= 10**12:
            t = ts / 1_000          # ms -> s
        else:
            t = float(ts)           # already seconds
        now = datetime.now(timezone.utc).timestamp()
        return max(0.0, now - t)
    except Exception:
        return None

# --------------------
# Options Advanced NBBO (preferred when available)
# --------------------
async def fetch_option_quote_adv(client: httpx.AsyncClient, option_ticker: str) -> Dict[str, Any]:
    """
    Preferred NBBO fetch for Polygon Options Advanced:
      GET /v3/quotes/options/{ticker}/last
    Returns fields:
      bid, ask, mid, size_bid, size_ask, quote_age_sec, option_spread_pct, nbbo_provider, nbbo_http_status
    """
    if not POLYGON_API_KEY or not POLY_ADVANCED:
        return {}

    enc = option_ticker.replace("/", "%2F")
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    try:
        r = await client.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=6.0)
        if r.status_code in (402, 403, 404, 429):
            return {"nbbo_http_status": r.status_code, "nbbo_reason": str(r.text)[:300]}
        r.raise_for_status()
        js = r.json()
        res = js.get("results") if isinstance(js, dict) else None
        if not isinstance(res, dict):
            return {"nbbo_http_status": 200}

        # v3 payload uses snake_case (bid_price/ask_price); some SDKs alias to 'last'
        last = res.get("last") or res

        bid = last.get("bid_price") or last.get("bidPrice")
        ask = last.get("ask_price") or last.get("askPrice")
        bid_sz = last.get("bid_size") or last.get("bidSize")
        ask_sz = last.get("ask_size") or last.get("askSize")
        ts = (
            last.get("sip_timestamp")
            or last.get("tape_timestamp")
            or last.get("timestamp")
            or last.get("t")
        )

        out: Dict[str, Any] = {"nbbo_provider": "polygon_v3", "nbbo_http_status": 200}

        if isinstance(bid, (int, float)):
            out["bid"] = float(bid)
        if isinstance(ask, (int, float)):
            out["ask"] = float(ask)
        if out.get("bid") is not None and out.get("ask") is not None:
            mid = (out["bid"] + out["ask"]) / 2.0
            out["mid"] = round(mid, 4)
            if mid > 0:
                out["option_spread_pct"] = round((out["ask"] - out["bid"]) / mid * 100.0, 3)

        if isinstance(bid_sz, (int, float)):
            out["size_bid"] = int(bid_sz)
        if isinstance(ask_sz, (int, float)):
            out["size_ask"] = int(ask_sz)

        age = _poly_ts_to_age_sec(ts) if ts is not None else None
        if age is not None:
            out["quote_age_sec"] = age

        return out
    except Exception as e:
        return {"nbbo_http_status": 500, "nbbo_reason": f"{type(e).__name__}: {e}"}

# --------------------
# NBBO helpers (legacy / generic) — kept for fallback
# --------------------
async def _pull_nbbo_direct(option_ticker: str) -> Dict[str, Any]:
    """
    Generic last-quote pull that tolerates both old and new shapes.
    Used as a fallback if advanced call didn't return NBBO.
    """
    out: Dict[str, Any] = {}
    client = get_http_client()
    if not POLYGON_API_KEY or client is None:
        return out
    try:
        enc = option_ticker.replace("/", "%2F")
        lastq = await _http_json(
            client,
            f"https://api.polygon.io/v3/quotes/options/{enc}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=4.0,
        )
        if not lastq or not isinstance(lastq, dict) or lastq.get("error"):
            return out
        res = lastq.get("results") or {}
        last = res.get("last") or res

        # handle both snake_case (v3) and camelCase (older samples)
        bid = last.get("bid_price") or last.get("bidPrice")
        ask = last.get("ask_price") or last.get("askPrice")
        ts = last.get("sip_timestamp") or last.get("tape_timestamp") or last.get("timestamp") or last.get("t")

        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
            mid = (float(bid) + float(ask)) / 2.0
            out["bid"] = float(bid)
            out["ask"] = float(ask)
            out["mid"] = round(mid, 4)
            if mid > 0:
                out["option_spread_pct"] = round((float(ask) - float(bid)) / mid * 100.0, 3)

        if ts is not None:
            age = _poly_ts_to_age_sec(ts)
            if age is not None:
                out["quote_age_sec"] = age
    except Exception:
        pass
    return out


async def _probe_nbbo_verbose(option_ticker: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return {"nbbo_reason": "no POLYGON_API_KEY in env"}
    enc = option_ticker.replace("/", "%2F")
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    res = await _http_get_any(url, params={"apiKey": POLYGON_API_KEY}, timeout=6.0)
    out["nbbo_http_status"] = res.get("status")
    if res.get("status") != 200:
        body = res.get("body")
        if isinstance(body, dict):
            out["nbbo_reason"] = body.get("error") or body.get("message") or "non-200 from Polygon"
        else:
            out["nbbo_reason"] = "non-200 from Polygon"
        out["nbbo_body_sample"] = (body if isinstance(body, dict) else str(body))[:400]
        return out

    body = res.get("body") or {}
    results = body.get("results") if isinstance(body, dict) else None
    last = (results or {}).get("last") or (results or {})
    bid = last.get("bid_price") or last.get("bidPrice")
    ask = last.get("ask_price") or last.get("askPrice")
    ts  = last.get("sip_timestamp") or last.get("tape_timestamp") or last.get("timestamp") or last.get("t")

    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
        mid = (float(bid) + float(ask)) / 2.0
        out.update({
            "bid": float(bid),
            "ask": float(ask),
            "mid": round(mid, 4),
            "option_spread_pct": round(((float(ask) - float(bid)) / mid * 100.0), 3) if mid > 0 else None,
        })
    else:
        out["nbbo_reason"] = "no bid/ask in response (thin or AH?)"

    age = _poly_ts_to_age_sec(ts) if ts is not None else None
    if age is not None:
        out["quote_age_sec"] = age
    return out

# --------------------
# Listing checks & replacements
# --------------------
async def _poly_reference_contracts_exists(underlying: str, expiry_iso: str, ticker: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {"listed": None, "snapshot_ok": None, "reason": "no HTTP or API key"}
    try:
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

        enc = ticker.replace("/", "%2F")
        s = await client.get(
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
            params={"apiKey": POLYGON_API_KEY, "greeks": "true"},
            timeout=6.0,
        )
        snapshot_ok = (s.status_code == 200 and isinstance((s.json() or {}).get("results"), dict))
        return {"listed": listed, "snapshot_ok": snapshot_ok, "reason": None}
    except Exception as e:
        return {"listed": None, "snapshot_ok": None, "reason": f"error: {type(e).__name__}: {e}"}


async def _rescan_best_replacement(
    symbol: str,
    side: str,
    desired_strike: float,
    expiry_iso: str,
    min_vol: int,
    min_oi: int
) -> Optional[Dict[str, Any]]:
    # Import here to avoid circular imports
    from market_ops import scan_top_candidates_for_alert
    import re  # local use

    try:
        pool = await scan_top_candidates_for_alert(
            get_http_client(),
            symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol,
            min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=15,
            restrict_expiries=[expiry_iso],  # type: ignore
        ) or []
    except TypeError:
        pool = await scan_top_candidates_for_alert(
            get_http_client(),
            symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_oi,
            min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=20,
        ) or []
        pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    def _occ_side(tk: str) -> Optional[str]:
        m = re.search(r"([CP])\d{8,9}$", tk or "")
        if not m:
            return None
        return "CALL" if m.group(1).upper() == "C" else "PUT"

    try:
        pool = [it for it in pool if _occ_side(it.get("ticker")) == side]
    except Exception:
        pass

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
    min_oi: int
) -> Optional[Dict[str, Any]]:
    from market_ops import scan_top_candidates_for_alert

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
        pool = await scan_top_candidates_for_alert(
            get_http_client(),
            symbol,
            {"side": side, "symbol": symbol, "strike": desired_strike, "expiry": expiry_iso},
            min_vol=min_vol,
            min_oi=min_oi,
            top_n_each_week=int(os.getenv("SCAN_TOPN_WEEK", "12")),
            top_overall=15,
        ) or []
        pool = [it for it in pool if it.get("expiry") == expiry_iso]
    except Exception:
        pool = []

    def _side_ok(tk: Optional[str]) -> bool:
        if not tk:
            return False
        c = tk[-10:-9].upper() if len(tk) >= 10 else ""
        return (c == "C" and side == "CALL") or (c == "P" and side == "PUT")

    pool = [it for it in pool if _side_ok(it.get("ticker")) and it.get("bid") is not None and it.get("ask") is not None]

    def _rank(it: Dict[str, Any]):
        sd = abs(float(it.get("strike") or desired_strike) - desired_strike)
        sp = float(it.get("spread_pct") or 1e9)
        return (sd, sp, -(it.get("oi") or 0), -(it.get("vol") or 0))

    pool.sort(key=_rank)
    return pool[0] if pool else None

# --------------------
# Enrichment: Reference, Corp Actions, Indicators, Aggregates, Snapshot
# --------------------
async def fetch_reference_overview(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol.upper()}"
    js = await _http_json(client, url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    if not js or js.get("error"):
        return {}
    res = js.get("results") or {}
    return {
        "name": res.get("name"),
        "market_cap": res.get("market_cap"),
        "currency_name": res.get("currency_name"),
        "share_class_shares_outstanding": res.get("share_class_shares_outstanding"),
        "homepage_url": res.get("homepage_url"),
        "sic_description": res.get("sic_description"),
        "sic_code": res.get("sic_code"),
        "primary_exchange": res.get("primary_exchange"),
    }


async def fetch_corporate_actions(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    divs = await _http_json(
        client,
        "https://api.polygon.io/v3/reference/dividends",
        {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY},
        timeout=6.0,
    ) or {}
    splits = await _http_json(
        client,
        "https://api.polygon.io/v3/reference/splits",
        {"ticker": symbol, "limit": 1, "order": "desc", "apiKey": POLYGON_API_KEY},
        timeout=6.0,
    ) or {}
    latest_div = (divs or {}).get("results", [{}])[:1]
    latest_spl = (splits or {}).get("results", [{}])[:1]
    return {
        "latest_dividend": latest_div[0] if latest_div else None,
        "latest_split": latest_spl[0] if latest_spl else None,
    }


async def fetch_technical_indicators(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    base = "https://api.polygon.io/v1/indicators"
    out: Dict[str, Any] = {}
    params_common = {"timespan": "day", "series_type": "close", "limit": 1, "apiKey": POLYGON_API_KEY}

    sma = await _http_json(client, f"{base}/sma/{symbol}", {**params_common, "window": 20}, timeout=6.0) or {}
    ema = await _http_json(client, f"{base}/ema/{symbol}", {**params_common, "window": 20}, timeout=6.0) or {}
    rsi = await _http_json(client, f"{base}/rsi/{symbol}", {**params_common, "window": 14}, timeout=6.0) or {}
    macd = await _http_json(
        client,
        f"{base}/macd/{symbol}",
        {"timespan": "day", "series_type": "close", "short_window": 12, "long_window": 26, "signal_window": 9, "limit": 1, "apiKey": POLYGON_API_KEY},
        timeout=6.0,
    ) or {}

    def _val(js, key="value"):
        try:
            vals = (js.get("results") or {}).get("values") or []
            return vals[-1].get(key) if vals else None
        except Exception:
            return None

    out["sma20"] = _val(sma)
    out["ema20"] = _val(ema)
    out["rsi14"] = _val(rsi)
    try:
        vals = (macd.get("results") or {}).get("values") or []
        if vals:
            m = vals[-1]
            out["macd"] = {"macd": m.get("macd"), "signal": m.get("signal"), "hist": m.get("hist")}
    except Exception:
        pass
    return out


async def fetch_aggs_recent(symbol: str, timespan: str = "minute", limit: int = 50) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=3)).isoformat()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start}/{now.isoformat()}"
    js = await _http_json(
        client,
        url,
        {"adjusted": "true", "sort": "desc", "limit": max(1, limit), "apiKey": POLYGON_API_KEY},
        timeout=8.0,
    )
    if not js or js.get("error"):
        return {}
    results = js.get("results") or []
    last = results[0] if results else {}
    return {"aggs_last": last, "aggs_count": len(results)}


async def fetch_aggs_second_recent(symbol: str, limit: int = 120) -> Dict[str, Any]:
    return await fetch_aggs_recent(symbol, timespan="second", limit=limit)


async def fetch_stock_snapshot(symbol: str) -> Dict[str, Any]:
    client = get_http_client()
    if client is None or not POLYGON_API_KEY:
        return {}
    url = f"https://api.polygon.io/v3/snapshot/stocks/{symbol.upper()}"
    js = await _http_json(client, url, {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    if not js or js.get("error"):
        return {}
    return js.get("results") or {}


async def enrich_underlying(symbol: str) -> Dict[str, Any]:
    """Convenience aggregator for enrichment payload used by LLM."""
    try:
        ref, ca, ind, snap = (
            await fetch_reference_overview(symbol),
            await fetch_corporate_actions(symbol),
            await fetch_technical_indicators(symbol),
            await fetch_stock_snapshot(symbol),
        )
        aggm = await fetch_aggs_recent(symbol, timespan="minute", limit=50)
        aggs = await fetch_aggs_second_recent(symbol, limit=120)
        return {
            "ref": ref,
            "corp_actions": ca,
            "indicators": ind,
            "snapshot": snap,
            "aggs_minute": aggm,
            "aggs_second": aggs,
        }
    except Exception as e:
        logger.warning("enrich_underlying failed: %r", e)
        return {}

__all__ = [
    # http helpers
    "_http_json",
    "_http_get_any",
    # nbbo helpers
    "fetch_option_quote_adv",
    "_pull_nbbo_direct",
    "_probe_nbbo_verbose",
    # listing / replacement
    "_poly_reference_contracts_exists",
    "_rescan_best_replacement",
    "_find_nbbo_replacement_same_expiry",
    # enrichment
    "fetch_reference_overview",
    "fetch_corporate_actions",
    "fetch_technical_indicators",
    "fetch_aggs_recent",
    "fetch_aggs_second_recent",
    "fetch_stock_snapshot",
    "enrich_underlying",
]
