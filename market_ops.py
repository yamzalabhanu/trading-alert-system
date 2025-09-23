# market_ops.py
# Polygon + market helpers split from trading_engine.py

import os
import re
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta, date
from urllib.parse import quote

import httpx

from polygon_ops import fetch_option_quote_adv  # optional (kept if you use it elsewhere)

from polygon_client import (
    list_contracts_for_expiry,
    get_option_snapshot,
)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# -------------------------
# Generic HTTP helpers
# -------------------------
async def _http_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] | None, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, params=params or {}, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {"status": r.status_code, "body": (r.text or "")[:400]}
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

async def _http_json_url(client: httpx.AsyncClient, url: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(url, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {"status": r.status_code, "body": (r.text or "")[:400]}
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

def _encode_ticker_path(t: str) -> str:
    return quote(t or "", safe="")

# -------------------------
# Small helpers
# -------------------------
def _normalize_poly_strike(raw) -> Optional[float]:
    if raw is None:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    return v / 1000.0 if v >= 2000 else v

def _quote_age_from_ts(ts_val: Any) -> Optional[float]:
    if ts_val is None:
        return None
    try:
        ns = int(ts_val)
    except Exception:
        return None
    if ns >= 10**14:
        sec = ns / 1e9
    elif ns >= 10**11:
        sec = ns / 1e6
    elif ns >= 10**8:
        sec = ns / 1e3
    else:
        sec = float(ns)
    age = max(0.0, datetime.now(timezone.utc).timestamp() - sec)
    return round(age, 3)

def _today_utc_range_for_aggs(now_utc: datetime) -> Tuple[str, str]:
    start = datetime(now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, tzinfo=timezone.utc).isoformat()
    return start, now_utc.isoformat()

# -------------------------
# Public wrappers
# -------------------------
async def polygon_list_contracts_for_expiry_export(
    client: httpx.AsyncClient,
    symbol: str,
    expiry: str,
    side: str,
    limit: int = 250,
) -> List[Dict[str, Any]]:
    return await list_contracts_for_expiry(client, symbol=symbol, expiry=expiry, side=side, limit=limit)

async def polygon_get_option_snapshot_export(
    client: httpx.AsyncClient,
    underlying: str,
    option_ticker: str,
) -> Dict[str, Any]:
    """Supports both signatures used in your project."""
    try:
        return await get_option_snapshot(client, symbol=underlying, contract=option_ticker)
    except TypeError as e1:
        try:
            return await get_option_snapshot(underlying, option_ticker)
        except TypeError as e2:
            raise RuntimeError(
                f"get_option_snapshot signature mismatch: "
                f"tried (client,symbol,contract): {e1}; "
                f"tried (symbol,contract): {e2}"
            )

# -------------------------
# Quote samplers / probes
# -------------------------
async def _sample_best_quote(client, enc_opt, tries=5, delay=0.6) -> Optional[Dict[str, Any]]:
    best: Dict[str, Any] = {}
    for _ in range(tries):
        lastq = await _http_json(
            client,
            f"https://api.polygon.io/v3/quotes/options/{enc_opt}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=3.0,
        )
        if lastq and not isinstance(lastq.get("status"), int):
            res = lastq.get("results") or {}
            last = res.get("last") or res
            bid = last.get("bidPrice")
            ask = last.get("askPrice")
            ts  = last.get("t") or last.get("sip_timestamp") or last.get("timestamp")
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
                spread_pct = ((ask - bid) / mid * 100.0) if (mid and mid > 0) else None
                age = _quote_age_from_ts(ts)
                cand = {
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "mid": float(mid) if mid is not None else None,
                    "quote_age_sec": age,
                    "option_spread_pct": round(spread_pct, 3) if spread_pct is not None else None,
                    "nbbo_provider": "polygon:last",
                }
                if (
                    not best
                    or (cand.get("option_spread_pct") or 1e9) < (best.get("option_spread_pct") or 1e9)
                    or (cand.get("quote_age_sec") or 1e9) < (best.get("quote_age_sec") or 1e9)
                ):
                    best = cand
        await asyncio.sleep(delay)
    return best or None

async def ensure_nbbo(
    client: httpx.AsyncClient,
    option_ticker: str,
    tries: int = 6,
    delay: float = 0.6,
) -> Dict[str, Any]:
    """
    Prefer NBBO from the *snapshot* endpoint (Options Advanced):
      - bid/ask + last_updated + timeframe
      - compute mid, spread%, quote_age_sec
    Fall back to /v3/quotes/options/{ticker}/last if snapshot is missing NBBO.
    Finally, fall back to last trade for a 'mid'/last + age only.

    Returns:
      {
        bid, ask, mid, option_spread_pct, quote_age_sec,
        nbbo_provider: "polygon:snapshot" | "polygon:last" | "polygon:last_trade",
        nbbo_timeframe: "DELAYED" | "REALTIME" | "UNKNOWN" (if snapshot path used)
      }
    """
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY or client is None:
        return out

    # Try snapshot first: it works on your plan even if quotes API is unauthorized.
    # We need the underlying symbol to call the snapshot endpoint; infer from OCC ticker.
    # OCC: O:SYMBOL YYMMDD C/P STRIKE (8–9 digits)
    m = re.search(r"^O:([A-Z0-9\.\-]+)\d{6}[CP]\d{8,9}$", option_ticker or "")
    underlying = m.group(1) if m else None
    enc_opt = _encode_ticker_path(option_ticker)

    if underlying:
        snap = await _http_json(
            client,
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc_opt}",
            {"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        if snap and not isinstance(snap.get("status"), int):
            res = snap.get("results") or {}
            lq = res.get("last_quote") or res.get("lastQuote") or {}
            bid = lq.get("bid") or lq.get("bid_price") or lq.get("bidPrice")
            ask = lq.get("ask") or lq.get("ask_price") or lq.get("askPrice")
            ts  = lq.get("last_updated") or lq.get("sip_timestamp") or lq.get("participant_timestamp") or lq.get("trf_timestamp") or lq.get("t")
            timeframe = lq.get("timeframe") or res.get("timeframe") or "UNKNOWN"

            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid:
                mid = (bid + ask) / 2.0
                spread_pct = ((ask - bid) / mid * 100.0) if (mid and mid > 0) else None
                age = _quote_age_from_ts(ts)
                out.update({
                    "bid": float(bid),
                    "ask": float(ask),
                    "mid": round(float(mid), 4) if mid is not None else None,
                    "option_spread_pct": round(float(spread_pct), 3) if spread_pct is not None else None,
                    "quote_age_sec": float(age) if age is not None else None,
                    "nbbo_provider": "polygon:snapshot",
                    "nbbo_timeframe": str(timeframe),
                })

    # If snapshot didn’t give us a proper NBBO, try quotes/last (may be unauthorized on some plans)
    if out.get("bid") is None or out.get("ask") is None:
        best = await _sample_best_quote(client, enc_opt, tries=max(1, tries), delay=delay)
        if best:
            out.update(best)

    # Last-trade fallback for a mark + age (still useful when there’s no NBBO)
    if out.get("bid") is None and out.get("ask") is None:
        t = await _http_json(
            client,
            f"https://api.polygon.io/v3/trades/options/{enc_opt}/last",
            {"apiKey": POLYGON_API_KEY},
            timeout=4.0,
        )
        if t and not isinstance(t.get("status"), int):
            res = t.get("results") or {}
            last_px = res.get("price")
            ts = res.get("sip_timestamp") or res.get("participant_timestamp") or res.get("t")
            if isinstance(last_px, (int, float)):
                out["last"] = float(last_px)
                out["mid"] = float(last_px)
            age = _quote_age_from_ts(ts)
            if age is not None:
                out["quote_age_sec"] = age
            out.setdefault("nbbo_provider", "polygon:last_trade")

    return out

# -------------------------
# Backfill bundle
# -------------------------
async def poly_option_backfill(
    client: httpx.AsyncClient,
    symbol: str,
    option_ticker: str,
    today_utc: date,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not POLYGON_API_KEY:
        return out

    def _apply_from_results(res: Dict[str, Any], out_dict: Dict[str, Any]) -> None:
        if not isinstance(res, dict):
            return

        # OI / Vol (from snapshot)
        oi = res.get("open_interest")
        if oi is not None:
            out_dict["oi"] = oi
        day_block = res.get("day") or {}
        vol = day_block.get("volume", day_block.get("v"))
        if vol is not None:
            out_dict["vol"] = vol

        # Snapshot NBBO + timeframe
        lq = res.get("last_quote") or res.get("lastQuote") or {}
        bid_px = lq.get("bid") or lq.get("bid_price") or lq.get("bidPrice")
        ask_px = lq.get("ask") or lq.get("ask_price") or lq.get("askPrice")
        timeframe = lq.get("timeframe") or res.get("timeframe") or "UNKNOWN"
        if bid_px is not None:
            out_dict["bid"] = float(bid_px)
        if ask_px is not None:
            out_dict["ask"] = float(ask_px)
        if bid_px is not None or ask_px is not None:
            out_dict["nbbo_provider"] = "polygon:snapshot"
            out_dict["nbbo_timeframe"] = str(timeframe)

        # Derive mid/spread from snapshot if possible
        if out_dict.get("mid") is None and out_dict.get("bid") is not None and out_dict.get("ask") is not None:
            out_dict["mid"] = round((out_dict["bid"] + out_dict["ask"]) / 2.0, 4)
        if out_dict.get("option_spread_pct") is None and out_dict.get("mid") not in (None, 0):
            try:
                out_dict["option_spread_pct"] = round((out_dict["ask"] - out_dict["bid"]) / out_dict["mid"] * 100.0, 3)
            except Exception:
                pass

        # Quote timestamp → age
        ts = (
            lq.get("last_updated")
            or lq.get("sip_timestamp")
            or lq.get("participant_timestamp")
            or lq.get("trf_timestamp")
            or lq.get("t")
        )
        age = _quote_age_from_ts(ts)
        if age is not None:
            out_dict["quote_age_sec"] = age

        # Last trade (for 'last')
        lt = res.get("last_trade") or res.get("lastTrade") or {}
        lt_px = lt.get("price")
        if isinstance(lt_px, (int, float)):
            out_dict["last"] = float(lt_px)

        # Greeks & IV
        greeks = res.get("greeks") or {}
        for k_src in ("delta", "gamma", "theta", "vega"):
            v = greeks.get(k_src)
            if v is not None:
                out_dict[k_src] = v
        iv = res.get("implied_volatility") or greeks.get("iv")
        if iv is not None:
            out_dict["iv"] = iv

    # 1) multi-snapshot list -> exact match if present
    try:
        m = re.search(r":([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$", option_ticker)
        if m:
            yy, mm, dd, cp = m.group(2), m.group(3), m.group(4), m.group(5)
            expiry_iso = f"20{yy}-{mm}-{dd}"
            side = "call" if cp.upper() == "C" else "put"
            rlist = await _http_json(
                client,
                f"https://api.polygon.io/v3/snapshot/options/{symbol}",
                {
                    "apiKey": POLYGON_API_KEY,
                    "contract_type": side,
                    "expiration_date": expiry_iso,
                    "limit": 1000,
                    "greeks": "true",
                    "include_greeks": "true",
                },
                timeout=8.0
            )
            if rlist and not isinstance(rlist.get("status"), int):
                items = rlist.get("results") or []
                chosen = None
                for it in items:
                    tkr = (it.get("details") or {}).get("ticker") or it.get("ticker")
                    if tkr == option_ticker:
                        chosen = it
                        break
                if chosen:
                    _apply_from_results(chosen, out)
    except Exception:
        pass

    # 2) single-contract snapshot
    if not out:
        try:
            enc_opt = _encode_ticker_path(option_ticker)
            snap = await _http_json(
                client,
                f"https://api.polygon.io/v3/snapshot/options/{symbol}/{enc_opt}",
                {"apiKey": POLYGON_API_KEY},
                timeout=8.0
            )
            if snap and not isinstance(snap.get("status"), int):
                _apply_from_results(snap.get("results") or {}, out)
        except Exception:
            pass

    # 3) previous-day open/close
    try:
        yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        enc_opt = _encode_ticker_path(option_ticker)
        oc = await _http_json(
            client,
            f"https://api.polygon.io/v1/open-close/options/{enc_opt}/{yday}",
            {"apiKey": POLYGON_API_KEY},
            timeout=6.0
        )
        if oc and not isinstance(oc.get("status"), int):
            oi = oc.get("open_interest")
            vol = oc.get("volume")
            if out.get("oi") is None and oi is not None:
                out["oi"] = oi
            if out.get("vol") is None and vol is not None:
                out["vol"] = vol
            pc = oc.get("close")
            if isinstance(pc, (int, float)):
                out["prev_close"] = float(pc)
    except Exception:
        pass

    # 4) If still no NBBO, sample quotes/last
    try:
        if out.get("bid") is None or out.get("ask") is None:
            enc_opt = _encode_ticker_path(option_ticker)
            sampled = await _sample_best_quote(client, enc_opt, tries=5, delay=0.6)
            if sampled:
                for k, v in sampled.items():
                    if v is not None:
                        out[k] = v
    except Exception:
        pass

    # 4b) Last-trade fallback if NBBO missing
    try:
        if out.get("bid") is None and out.get("ask") is None:
            enc_opt = _encode_ticker_path(option_ticker)
            t = await _http_json(
                client,
                f"https://api.polygon.io/v3/trades/options/{enc_opt}/last",
                {"apiKey": POLYGON_API_KEY},
                timeout=4.0,
            )
            if t and not isinstance(t.get("status"), int):
                res = t.get("results") or {}
                last_px = res.get("price")
                ts = res.get("sip_timestamp") or res.get("participant_timestamp") or res.get("t")
                if isinstance(last_px, (int, float)):
                    out["last"] = float(last_px)
                    out.setdefault("mid", float(last_px))
                age = _quote_age_from_ts(ts)
                if age is not None and out.get("quote_age_sec") is None:
                    out["quote_age_sec"] = age
                out.setdefault("nbbo_provider", "polygon:last_trade")
    except Exception:
        pass

    # 5) minute aggregates as fallback volume
    try:
        if out.get("vol") is None:
            enc_opt = _encode_ticker_path(option_ticker)
            now_utc_dt = datetime.now(timezone.utc)
            frm_iso, to_iso = _today_utc_range_for_aggs(now_utc_dt)
            aggs = await _http_json(
                client,
                f"https://api.polygon.io/v2/aggs/ticker/{enc_opt}/range/1/min/{frm_iso}/{to_iso}",
                {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY},
                timeout=8.0
            )
            if aggs and not isinstance(aggs.get("status"), int):
                results = aggs.get("results") or []
                if results:
                    vol_sum = 0
                    for bar in results:
                        v = bar.get("v")
                        if isinstance(v, (int, float)):
                            vol_sum += v
                    if vol_sum > 0:
                        out["vol"] = int(vol_sum)
    except Exception:
        pass

    # 6) derive change% if possible (mark vs prev_close)
    try:
        mark = out.get("mid") if out.get("mid") is not None else out.get("last")
        prev_close = out.get("prev_close")
        if isinstance(mark, (int, float)) and isinstance(prev_close, (int, float)) and prev_close > 0:
            out["quote_change_pct"] = round((float(mark) - float(prev_close)) / float(prev_close) * 100.0, 3)
    except Exception:
        pass

    return out

# -------------------------
# Contract chooser + chain scan
# -------------------------
async def choose_best_contract(
    client: httpx.AsyncClient,
    symbol: str,
    expiry_iso: str,
    side: str,
    ul_px: float,
    desired_strike: float,
) -> Tuple[str, Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return "", {"reason": "offline/no key"}

    chain = await polygon_list_contracts_for_expiry_export(client, symbol=symbol, expiry=expiry_iso, side=side, limit=1000)
    if not chain:
        return "", {"reason": "no_chain"}

    try:
        side_poly = "call" if side.upper() == "CALL" else "put"
        mlist = await _http_json(
            client,
            f"https://api.polygon.io/v3/snapshot/options/{symbol}",
            {
                "apiKey": POLYGON_API_KEY,
                "contract_type": side_poly,
                "expiration_date": expiry_iso,
                "limit": 1000,
                "greeks": "true",
            },
            timeout=8.0
        )
    except Exception:
        mlist = None

    index_by_ticker = {}
    if mlist and not isinstance(mlist.get("status"), int):
        for it in (mlist.get("results") or []):
            tk = (it.get("details") or {}).get("ticker") or it.get("ticker")
            if tk:
                index_by_ticker[tk] = it

    def _normalize_from_tk(tk: str, default: Optional[float]) -> Optional[float]:
        s_norm = default
        if s_norm is None:
            m2 = re.search(r"[CP](\d{8,9})$", tk)
            if m2:
                try:
                    s_norm = int(m2.group(1)) / 1000.0
                except Exception:
                    s_norm = None
        return s_norm

    TARGET_DELTA_CALL = float(os.getenv("TARGET_DELTA_CALL", "0.35"))
    TARGET_DELTA_PUT  = float(os.getenv("TARGET_DELTA_PUT", "-0.35"))

    cands: List[Tuple] = []
    for c in chain:
        tk = c.get("ticker") or c.get("symbol") or c.get("contract")
        if not tk:
            continue
        det = index_by_ticker.get(tk) or {}
        greeks = det.get("greeks") or {}
        delta = greeks.get("delta")
        oi = det.get("open_interest")
        day_block = det.get("day") or {}
        vol = day_block.get("volume")
        enc = _encode_ticker_path(tk)

        spread_pct = None
        # Try snapshot NBBO
        lq = det.get("last_quote") or {}
        b = lq.get("bid") or lq.get("bid_price") or lq.get("bidPrice")
        a = lq.get("ask") or lq.get("ask_price") or lq.get("askPrice")
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0:
            mid = (a + b) / 2.0
            if mid > 0:
                spread_pct = (a - b)/mid*100.0
        else:
            # Fallback to quotes/last
            q = await _http_json(client, f"https://api.polygon.io/v3/quotes/options/{enc}/last",
                                 {"apiKey": POLYGON_API_KEY}, timeout=3.0)
            if q and not isinstance(q.get("status"), int):
                res = q.get("results") or {}
                last = res.get("last") or res
                b2, a2 = last.get("bidPrice"), last.get("askPrice")
                if isinstance(b2, (int, float)) and isinstance(a2, (int, float)) and a2 >= b2 and a2 > 0:
                    mid = (a2 + b2)/2.0
                    if mid > 0:
                        spread_pct = (a2 - b2)/mid*100.0

        tgt = TARGET_DELTA_CALL if side == "CALL" else TARGET_DELTA_PUT
        delta_miss = abs((delta if isinstance(delta, (int, float)) else tgt) - tgt)

        s_norm = _normalize_poly_strike(c.get("strike"))
        s_norm = _normalize_from_tk(tk, s_norm)
        strike_miss = abs((s_norm if s_norm is not None else desired_strike) - desired_strike)

        rank_tuple = (
            delta_miss,
            (spread_pct if spread_pct is not None else 1e9),
            -(vol or 0),
            -(oi or 0),
            strike_miss,
        )
        cands.append((rank_tuple, tk, delta, spread_pct, vol, oi, s_norm))

    if not cands:
        return "", {"reason": "no_candidates"}

    cands.sort(key=lambda x: x[0])
    top = cands[0]
    debug = {
        "selected_by": "delta_selector",
        "candidates": [
            {
                "ticker": tk,
                "rank": r[0],
                "delta": d,
                "spread_pct": sp,
                "vol": v,
                "oi": oi,
                "strike": s,
            }
            for r, tk, d, sp, v, oi, s in cands[:5]
        ]
    }
    return top[1], debug

async def _poly_paginated_snapshot_list(
    client: httpx.AsyncClient,
    underlying: str,
    expiry_iso: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    if not POLYGON_API_KEY:
        return []

    all_rows: List[Dict[str, Any]] = []
    for side in ("call", "put"):
        base = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
        params = {
            "apiKey": POLYGON_API_KEY,
            "expiration_date": expiry_iso,
            "contract_type": side,
            "limit": limit,
            "greeks": "true",
            "include_greeks": "true",
        }
        first = await _http_json(client=client, url=base, params=params, timeout=10.0)
        if first and isinstance(first.get("results"), list):
            all_rows.extend(first["results"])
            nxt = first.get("next_url")
        else:
            nxt = None

        while nxt:
            if "apiKey=" not in nxt and POLYGON_API_KEY:
                sep = "&" if "?" in nxt else "?"
                nxt = f"{nxt}{sep}apiKey={POLYGON_API_KEY}"
            page = await _http_json_url(client, nxt, timeout=10.0)
            if not page or not isinstance(page.get("results"), list):
                break
            all_rows.extend(page["results"])
            nxt = page.get("next_url")

    return all_rows

async def _poly_reference_contracts(
    client: httpx.AsyncClient,
    underlying: str,
    expiry_iso: str,
    max_contracts: int = 1200,
) -> List[str]:
    if not POLYGON_API_KEY:
        return []
    tickers: List[str] = []
    base = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": underlying,
        "expiration_date": expiry_iso,
        "limit": 1000,
        "apiKey": POLYGON_API_KEY,
    }
    first = await _http_json(client, base, params, timeout=10.0)
    if first and isinstance(first.get("results"), list):
        for it in first["results"]:
            t = it.get("ticker")
            if t:
                tickers.append(t)
        nxt = first.get("next_url")
    else:
        nxt = None

    while nxt and len(tickers) < max_contracts:
        if "apiKey=" not in nxt and POLYGON_API_KEY:
            sep = "&" if "?" in nxt else "?"
            nxt = f"{nxt}{sep}apiKey={POLYGON_API_KEY}"
        page = await _http_json_url(client, nxt, timeout=10.0)
        if not page or not isinstance(page.get("results"), list):
            break
        for it in page.get("results", []):
            t = it.get("ticker")
            if t:
                tickers.append(t)
        nxt = page.get("next_url")

    return tickers[:max_contracts]

async def _poly_snapshot_for_ticker(client: httpx.AsyncClient, underlying: str, ticker: str) -> Optional[Dict[str, Any]]:
    enc = _encode_ticker_path(ticker)
    snap = await _http_json(client, f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
                            {"apiKey": POLYGON_API_KEY}, timeout=6.0)
    return snap.get("results") if snap else None

def _score_chain_item_for_alert(alert: Dict[str, Any], it: Dict[str, Any]) -> float:
    strike_alert = float(alert.get("strike") or 0.0)
    strike_item  = float(it.get("strike") or 0.0)
    expiry_alert = str(alert.get("expiry") or "")

    strike_dist = abs(strike_item - strike_alert)
    expiry_penalty = 0.0 if it.get("expiry") == expiry_alert else 10.0
    oi = int(it.get("oi") or 0); vol = int(it.get("vol") or 0)
    liq_score = -(oi * 2 + vol) / 1000.0
    spread_pct = it.get("spread_pct")
    nbbo_pen = 0.5 if (it.get("bid") is None or it.get("ask") is None) else 0.0
    spr_pen  = 0.0
    if isinstance(spread_pct, (int, float)):
        if spread_pct > 15: spr_pen = 2.0
        elif spread_pct > 10: spr_pen = 1.0
        elif spread_pct > 5:  spr_pen = 0.3
    else:
        spr_pen = 0.7
    return strike_dist + expiry_penalty + nbbo_pen + spr_pen + liq_score

async def scan_for_best_contract_for_alert(
    client: httpx.AsyncClient,
    symbol: str,
    alert: Dict[str, Any],
    min_vol: int = 500,
    min_oi: int  = 500,
    top_n_each_week: int = 12,
) -> Optional[Dict[str, Any]]:
    today_utc = datetime.now(timezone.utc).date()
    def _next_friday(d: date) -> date:
        return d + timedelta(days=(4 - d.weekday()) % 7)
    wk1 = _next_friday(today_utc)
    if wk1 <= today_utc:
        wk1 = wk1 + timedelta(days=7)
    wk2 = wk1 + timedelta(days=7)
    wk3 = wk2 + timedelta(days=7)
    default_weeks = [wk1.isoformat(), wk2.isoformat(), wk3.isoformat()]

    expiry_alert = str(alert.get("expiry") or "")
    weeks = []
    if expiry_alert:
        seen = set()
        for e in [expiry_alert] + default_weeks:
            if e and e not in seen:
                weeks.append(e); seen.add(e)
    else:
        weeks = default_weeks

    candidates: List[Dict[str, Any]] = []
    for exp in weeks:
        rows = await _poly_paginated_snapshot_list(client, symbol, exp, limit=1000)
        if not rows:
            tickers = await _poly_reference_contracts(client, symbol, exp, max_contracts=1200)
            sem = asyncio.Semaphore(12)
            async def fetch_one(tk: str):
                async with sem:
                    try:
                        return await _poly_snapshot_for_ticker(client, symbol, tk)
                    except Exception:
                        return None
            snaps = await asyncio.gather(*[fetch_one(t) for t in tickers])
            rows = [r for r in snaps if isinstance(r, dict)]

        items = []
        for r in rows:
            det = r.get("details") or {}
            tk  = det.get("ticker") or r.get("ticker")
            if not tk:
                continue
            day = r.get("day") or {}
            vol = int(day.get("volume") or day.get("v") or 0)
            oi  = int(r.get("open_interest") or 0)
            if vol < min_vol and oi < min_oi:
                continue

            lq = r.get("last_quote") or {}
            b = lq.get("bid") or lq.get("bid_price") or lq.get("bidPrice")
            a = lq.get("ask") or lq.get("ask_price") or lq.get("askPrice")
            mid = None; spread_pct = None
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0:
                mid = (a + b) / 2.0
                if mid and mid > 0:
                    spread_pct = (a - b) / mid * 100.0

            items.append({
                "ticker": tk,
                "expiry": exp,
                "strike": _normalize_poly_strike(det.get("strike")),
                "type": det.get("contract_type"),
                "vol": vol,
                "oi": oi,
                "bid": b, "ask": a, "mid": mid,
                "spread_pct": round(spread_pct, 3) if isinstance(spread_pct, (int, float)) else None,
            })

        items.sort(key=lambda x: (-(x["oi"] or 0), -(x["vol"] or 0)))
        candidates.extend(items[:top_n_each_week])

    if not candidates:
        return None

    scored = [( _score_chain_item_for_alert(alert, it), it ) for it in candidates]
    scored.sort(key=lambda z: z[0])
    return scored[0][1] if scored else None

# NEW: top-N (e.g., 3) across next 3 weeks
async def scan_top_candidates_for_alert(
    client: httpx.AsyncClient,
    symbol: str,
    alert: Dict[str, Any],
    min_vol: int = 500,
    min_oi: int  = 500,
    top_n_each_week: int = 12,
    top_overall: int = 3,
) -> List[Dict[str, Any]]:
    today_utc = datetime.now(timezone.utc).date()
    def _next_friday(d: date) -> date:
        return d + timedelta(days=(4 - d.weekday()) % 7)
    wk1 = _next_friday(today_utc)
    if wk1 <= today_utc:
        wk1 = wk1 + timedelta(days=7)
    wk2 = wk1 + timedelta(days=7)
    wk3 = wk2 + timedelta(days=7)
    default_weeks = [wk1.isoformat(), wk2.isoformat(), wk3.isoformat()]

    expiry_alert = str(alert.get("expiry") or "")
    weeks = []
    if expiry_alert:
        seen = set()
        for e in [expiry_alert] + default_weeks:
            if e and e not in seen:
                weeks.append(e); seen.add(e)
    else:
        weeks = default_weeks

    candidates: List[Dict[str, Any]] = []
    for exp in weeks:
        rows = await _poly_paginated_snapshot_list(client, symbol, exp, limit=1000)
        if not rows:
            tickers = await _poly_reference_contracts(client, symbol, exp, max_contracts=1200)
            sem = asyncio.Semaphore(12)
            async def fetch_one(tk: str):
                async with sem:
                    try:
                        return await _poly_snapshot_for_ticker(client, symbol, tk)
                    except Exception:
                        return None
            snaps = await asyncio.gather(*[fetch_one(t) for t in tickers])
            rows = [r for r in snaps if isinstance(r, dict)]

        for r in rows:
            det = r.get("details") or {}
            tk  = det.get("ticker") or r.get("ticker")
            if not tk:
                continue
            day = r.get("day") or {}
            vol = int(day.get("volume") or day.get("v") or 0)
            oi  = int(r.get("open_interest") or 0)
            if vol < min_vol and oi < min_oi:
                continue

            lq = r.get("last_quote") or {}
            b = lq.get("bid") or lq.get("bid_price") or lq.get("bidPrice")
            a = lq.get("ask") or lq.get("ask_price") or lq.get("askPrice")
            mid = None; spread_pct = None
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and a >= b and a > 0:
                mid = (a + b) / 2.0
                if mid and mid > 0:
                    spread_pct = (a - b) / mid * 100.0

            it = {
                "ticker": tk,
                "expiry": exp,
                "strike": _normalize_poly_strike(det.get("strike")),
                "type": det.get("contract_type"),
                "vol": vol,
                "oi": oi,
                "bid": b, "ask": a, "mid": mid,
                "spread_pct": round(spread_pct, 3) if isinstance(spread_pct, (int, float)) else None,
            }
            candidates.append(it)

    if not candidates:
        return []

    scored = [( _score_chain_item_for_alert(alert, it), it ) for it in candidates]
    scored.sort(key=lambda z: z[0])
    return [it for _, it in scored[:max(1, top_overall)]]
