# market_providers.py
import os
import re
import math
import logging
from typing import Optional, Dict, Any, Tuple

import httpx
from oauthlib.oauth1 import Client as OAuth1Client
from oauthlib.oauth1.rfc5849 import SIGNATURE_HMAC, SIGNATURE_TYPE_AUTH_HEADER

log = logging.getLogger("market_providers")

# ---------- Env ----------
ETRADE_ENABLED = os.getenv("ENABLE_ETRADE_PROVIDER", "1") != "0"
ETRADE_BASE = os.getenv("ETRADE_BASE_URL", "https://api.etrade.com")  # sandbox: https://apisb.etrade.com
ETRADE_CK = os.getenv("ETRADE_CONSUMER_KEY", "")
ETRADE_CS = os.getenv("ETRADE_CONSUMER_SECRET", "")
ETRADE_AT = os.getenv("ETRADE_ACCESS_TOKEN", "")
ETRADE_AS = os.getenv("ETRADE_ACCESS_SECRET", "")

def _etrade_config_ok() -> bool:
    return ETRADE_ENABLED and all([ETRADE_CK, ETRADE_CS, ETRADE_AT, ETRADE_AS])

# ---------- Helpers ----------
_RE_OCC = re.compile(r"^(?:O:)?([A-Z0-9\.\-]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8,9})$")

def _occ_to_etrade_symbol(occ: str) -> Optional[str]:
    """
    OCC -> E*TRADE path symbol: UNDERLIER:YYYY:M:D:CALL|PUT:STRIKE
    e.g. O:AAPL250919C00200000 -> AAPL:2025:9:19:CALL:200
    """
    m = _RE_OCC.match(occ.strip().upper())
    if not m:
        return None
    under, yy, mm, dd, cp, strike_raw = m.groups()
    year = 2000 + int(yy)
    month = int(mm)
    day = int(dd)

    # OCC strike is integer with 3 implied decimals (commonly 8 digits)
    try:
        strike_val = int(strike_raw) / 1000.0
    except Exception:
        return None

    # drop trailing .0 for clean path
    if math.isclose(strike_val, round(strike_val)):
        strike_txt = str(int(round(strike_val)))
    else:
        strike_txt = str(strike_val).rstrip("0").rstrip(".")

    right = "CALL" if cp == "C" else "PUT"
    return f"{under}:{year}:{month}:{day}:{right}:{strike_txt}"

def _from_ctx_to_etrade_symbol(ctx: Dict[str, Any]) -> Optional[str]:
    """
    Fallback builder using ctx when OCC not available.
    Expects: ctx['symbol'], ctx['right'] ('C'/'P' or 'CALL'/'PUT'),
             ctx['strike'] (float), ctx['expiry_yyyymmdd'] or ctx['expiry_iso']
    """
    sym = (ctx.get("symbol") or "").upper()
    right_in = (ctx.get("right") or "").upper()
    right = "CALL" if right_in in ("C", "CALL") else ("PUT" if right_in in ("P", "PUT") else None)

    ymd = ctx.get("expiry_yyyymmdd")
    if not ymd and ctx.get("expiry_iso"):
        iso = str(ctx["expiry_iso"])
        ymd = iso.replace("-", "")
    if not (sym and right and ymd and ctx.get("strike")):
        return None

    try:
        year = int(ymd[0:4]); month = int(ymd[4:6]); day = int(ymd[6:8])
        strike_val = float(ctx["strike"])
        strike_txt = (str(int(strike_val)) if math.isclose(strike_val, round(strike_val))
                      else str(strike_val).rstrip("0").rstrip("."))
        return f"{sym}:{year}:{month}:{day}:{right}:{strike_txt}"
    except Exception:
        return None

def _sign_oauth1(url: str, params: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict[str, str]:
    client = OAuth1Client(
        client_key=ETRADE_CK,
        client_secret=ETRADE_CS,
        resource_owner_key=ETRADE_AT,
        resource_owner_secret=ETRADE_AS,
        signature_method=SIGNATURE_HMAC,
        signature_type=SIGNATURE_TYPE_AUTH_HEADER,
    )
    uri, headers, _ = client.sign(
        url,
        http_method=method,
        headers={"Accept": "application/json"},
        body=None,
        params=params or {},
    )
    return headers  # includes Authorization

async def _etrade_quote_by_path_symbol(path_symbol: str) -> Optional[Dict[str, Any]]:
    """
    Calls: GET /v1/market/quote/{path_symbol}?detailFlag=OPTIONS
    Returns normalized dict {bid, ask, last, provider, ...}
    """
    base = ETRADE_BASE.rstrip("/")
    url = f"{base}/v1/market/quote/{path_symbol}"
    params = {"detailFlag": "OPTIONS"}
    headers = _sign_oauth1(url, params=params, method="GET")

    async with httpx.AsyncClient(timeout=8.0) as http:
        r = await http.get(url, params=params, headers=headers)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()

    # E*TRADE payloads vary; search recursively for keys we care about
    def _find_num(d: Any, cand_keys: Tuple[str, ...]) -> Optional[float]:
        if isinstance(d, dict):
            # prefer inner 'quoteData' → 'all'/'product' shapes, but be robust
            for k, v in d.items():
                lk = k.lower()
                if lk in [x.lower() for x in cand_keys]:
                    try:
                        return float(v)
                    except Exception:
                        pass
            for v in d.values():
                res = _find_num(v, cand_keys)
                if res is not None:
                    return res
        elif isinstance(d, list):
            for it in d:
                res = _find_num(it, cand_keys)
                if res is not None:
                    return res
        return None

    bid = _find_num(data, ("bid", "bidPrice"))
    ask = _find_num(data, ("ask", "askPrice"))
    last = _find_num(data, ("lastTrade", "lastPrice", "last"))

    out: Dict[str, Any] = {"provider": "etrade"}
    if bid is not None: out["bid"] = bid
    if ask is not None: out["ask"] = ask
    if last is not None: out["last"] = last
    if "bid" in out and "ask" in out:
        mid = (out["bid"] + out["ask"]) / 2.0
        out["mid"] = round(mid, 6)
        if mid > 0:
            out["spread_pct"] = round((out["ask"] - out["bid"]) / mid * 100.0, 3)
    return out if any(k in out for k in ("bid", "ask", "last")) else None

# ---------- Public API used by engine_processor ----------
def synthetic_from_last(last: Optional[float], spread_pct: float = 1.2) -> Optional[Dict[str, Any]]:
    if last is None or not isinstance(last, (int, float)) or last <= 0:
        return None
    half = spread_pct / 200.0
    bid = last * (1 - half / 100.0)
    ask = last * (1 + half / 100.0)
    mid = (bid + ask) / 2.0
    return {
        "provider": "synthetic(last)",
        "bid": round(bid, 6),
        "ask": round(ask, 6),
        "mid": round(mid, 6),
        "spread_pct": round((ask - bid) / mid * 100.0, 3) if mid > 0 else spread_pct,
        "synthetic_nbbo_used": True,
        "synthetic_nbbo_spread_est": spread_pct,
    }

async def get_nbbo_any(option_ticker: str, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try E*TRADE first (if configured). Return normalized NBBO dict or None.
    ctx is the same object engine passes (symbol/right/strike/expiry/etc).
    """
    # 1) E*TRADE
    if _etrade_config_ok():
        try:
            path_symbol = _occ_to_etrade_symbol(option_ticker) or _from_ctx_to_etrade_symbol(ctx)
            if path_symbol:
                res = await _etrade_quote_by_path_symbol(path_symbol)
                if res:
                    return res
        except Exception as e:
            log.warning("[etrade] quote failed for %s: %r", option_ticker, e)

    # 2) (Placeholder) Add more providers here (Tradier, IBKR, etc) if desired.

    # 3) Nothing found
    return None
