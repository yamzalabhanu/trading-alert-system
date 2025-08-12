from __future__ import annotations
"""
alert_server.py — Single-file app with Intraday Options Alerts

Purpose
- Run as an ASGI FastAPI app in production: `uvicorn alert_server:app --host 0.0.0.0 --port $PORT`.
- Still import and run tests cleanly **even if FastAPI is not installed** (e.g., sandbox).

What's new in this fix
- Removed the hard `RuntimeError` raised when FastAPI isn't present.
- Added a **graceful FastAPI shim**: when FastAPI is missing, minimal dummy types are used so the module
  imports and offline tests run. (Uvicorn won't serve in that mode, which is expected.)
- Kept all existing tests; added a small test to assert the app/router presence.

"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, time as dt_time
import time
import math

# ------------------------------
# FastAPI / ASGI bootstrap (with shim fallback)
# ------------------------------
try:
    from fastapi import FastAPI, APIRouter, HTTPException  # type: ignore
    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - sandbox without FastAPI
    FASTAPI_AVAILABLE = False

    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code: int, detail: str):
            super().__init__(f"HTTP {status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class _DummyDecorator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __call__(self, func):
            return func

    class APIRouter:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def post(self, *args: Any, **kwargs: Any):
            return _DummyDecorator()
        def get(self, *args: Any, **kwargs: Any):
            return _DummyDecorator()

    class FastAPI:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def include_router(self, *args: Any, **kwargs: Any) -> None:
            pass
        def get(self, *args: Any, **kwargs: Any):
            return _DummyDecorator()
        def post(self, *args: Any, **kwargs: Any):
            return _DummyDecorator()

# Create the ASGI app that Uvicorn expects (real FastAPI if available; shim otherwise)
app = FastAPI(title="Options Alerts Server", version="1.0.0")
router = APIRouter(prefix="/alerts", tags=["options-alerts"])  # type: ignore

# Optional dependency for live HTTP calls (we keep code import-safe without network)
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

# ==============================================================================
# Providers (Polygon) — network calls are guarded and return None/[] on failure
# ==============================================================================
async def fetch_underlying_snapshot(client: "httpx.AsyncClient", api_key: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Polygon stock snapshot for underlying price & recent trade/quote info."""
    if httpx is None:
        return None
    try:
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        r = await client.get(url, params={"apiKey": api_key})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

async def fetch_options_chain_top(client: "httpx.AsyncClient", api_key: str, symbol: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch a lightweight options snapshot list (plan-dependent). Returns [] on failure."""
    if httpx is None:
        return []
    try:
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
        r = await client.get(url, params={"limit": limit, "apiKey": api_key})
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "results" in data:
            return data["results"] or []
        return []
    except Exception:
        return []

# ==============================================================================
# Helpers: DTE, contract filtering (near-ATM, near-dated) and formatting
# ==============================================================================

def _parse_exp_to_utc(exp: Any) -> Optional[datetime]:
    """Parse expiration to a timezone-aware UTC datetime (00:00:00 of the date if date-only)."""
    try:
        if isinstance(exp, datetime):
            return exp if exp.tzinfo else exp.replace(tzinfo=timezone.utc)
        d = datetime.fromisoformat(str(exp))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _calc_dte_inclusive(exp_utc: datetime, now_utc: datetime) -> int:
    """Return inclusive Days-To-Expiry (ceil + include the expiration day)."""
    secs = (exp_utc - now_utc).total_seconds()
    days = math.ceil(secs / 86400.0)
    return max(0, days + 1)


def pick_near_atm_near_dated(
    chain: List[Dict[str, Any]],
    underlying_price: Optional[float],
    *,
    min_dte: int = 5,
    max_dte: int = 21,
    moneyness_band: Tuple[float, float] = (0.95, 1.05),
    top_n: int = 4,
    now_utc: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Filter to near-ATM and near-dated. Expects Polygon snapshot-like dicts."""
    if not chain or not underlying_price or underlying_price <= 0:
        return []

    now = now_utc or datetime.now(timezone.utc)
    sel: List[Tuple[float, Dict[str, Any]]] = []

    for o in chain:
        sym = o.get("ticker") or o.get("option", {}).get("ticker")
        if not sym:
            continue
        strike = o.get("strike_price") or o.get("details", {}).get("strike_price")
        typ = o.get("contract_type") or o.get("details", {}).get("contract_type")
        exp = o.get("expiration_date") or o.get("details", {}).get("expiration_date")
        bid = (o.get("last_quote") or {}).get("bid_price") or o.get("bid")
        ask = (o.get("last_quote") or {}).get("ask_price") or o.get("ask")
        last = (o.get("last_trade") or {}).get("price") or o.get("last_price")
        iv = o.get("implied_volatility") or o.get("iv")
        delta = o.get("delta")
        gamma = o.get("gamma")
        theta = o.get("theta")
        vega = o.get("vega")
        vol = (o.get("day") or {}).get("volume") or o.get("volume")
        oi = (o.get("open_interest") or (o.get("day") or {}).get("open_interest"))

        if not strike or not typ or not exp:
            continue
        exp_dt = _parse_exp_to_utc(exp)
        if exp_dt is None:
            continue
        dte = _calc_dte_inclusive(exp_dt, now)
        if dte < min_dte or dte > max_dte:
            continue
        # Moneyness (calls: U/strike, puts: strike/U)
        try:
            is_call = str(typ).strip().upper().startswith("C")
            mny = (float(underlying_price) / float(strike)) if is_call else (float(strike) / float(underlying_price))
        except Exception:
            continue
        if not (moneyness_band[0] <= mny <= moneyness_band[1]):
            continue
        # Spread sanity
        if bid is None or ask is None:
            continue
        try:
            mid = (float(bid) + float(ask)) / 2.0
            if mid <= 0:
                continue
            rel_spread = (float(ask) - float(bid)) / mid
            if rel_spread > 0.08:  # 8% max width
                continue
        except Exception:
            continue
        # Rank by tight spread then by volume
        score = (1.0 - rel_spread) + (float(vol or 0) / 1e6)
        sel.append((score, {
            "ticker": sym,
            "type": ("C" if is_call else "P"),
            "expiry": str(exp)[:10],
            "strike": float(strike),
            "bid": float(bid),
            "ask": float(ask),
            "last": float(last) if last is not None else None,
            "iv": float(iv) if iv is not None else None,
            "delta": float(delta) if delta is not None else None,
            "gamma": float(gamma) if gamma is not None else None,
            "theta": float(theta) if theta is not None else None,
            "vega": float(vega) if vega is not None else None,
            "volume": int(vol or 0),
            "open_interest": int(oi or 0),
        }))

    sel.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in sel[:top_n]]


def _fmt(x: Any) -> str:
    try:
        if isinstance(x, (int, float)):
            return f"{x:.2f}"
        if x is None:
            return "?"
        return str(x)
    except Exception:
        return "?"

# ==============================================================================
# Scoring & baselines (in-memory)
# ==============================================================================
BASELINES: Dict[str, Dict[str, float]] = {}

def _key(sym: str, expiry: str, strike: float, typ: str) -> str:
    return f"{sym}:{expiry}:{strike}:{typ}"


def update_and_zscores(symbol: str, opt: Dict[str, Any]) -> Dict[str, float]:
    """Maintain a tiny running baseline for volume and OI deltas; returns z-scores."""
    k = _key(symbol, str(opt.get("expiry")), float(opt.get("strike", 0)), str(opt.get("type")))
    b = BASELINES.setdefault(k, {"n": 0.0, "vol_mean": 0.0, "vol_m2": 0.0, "oi_mean": 0.0, "oi_m2": 0.0, "prev_oi": float(opt.get("open_interest") or 0.0)})

    vol = float(opt.get("volume") or 0.0)
    oi = float(opt.get("open_interest") or 0.0)

    # update volume baseline
    b["n"] += 1.0
    dv = vol - b["vol_mean"]
    b["vol_mean"] += dv / b["n"]
    b["vol_m2"] += dv * (vol - b["vol_mean"])  # M2

    # update OI baseline
    do = oi - b["oi_mean"]
    b["oi_mean"] += do / b["n"]
    b["oi_m2"] += do * (oi - b["oi_mean"])  # M2

    # z-scores
    vol_std = math.sqrt(max(b["vol_m2"], 0.0) / max(b["n"] - 1.0, 1.0))
    oi_std = math.sqrt(max(b["oi_m2"], 0.0) / max(b["n"] - 1.0, 1.0))
    z_vol = 0.0 if vol_std == 0 else (vol - b["vol_mean"]) / vol_std

    prev_oi = b.get("prev_oi", 0.0)
    oi_delta = oi - prev_oi
    b["prev_oi"] = oi
    z_oi_delta = 0.0 if oi_std == 0 else (oi_delta - 0.0) / oi_std

    return {"z_volume": z_vol, "z_oi_change": z_oi_delta}


def score_setup(stock: Dict[str, Any], opt: Dict[str, Any], direction: str, iv_rank: float | None = None, iv_pct: float | None = None) -> int:
    """Return 0–100 score from heuristic rubric."""
    score = 0

    # Trend + VWAP alignment
    ema9, ema21 = stock.get("ema9"), stock.get("ema21")
    px, vwap = stock.get("price"), stock.get("vwap")
    if direction == "CALL":
        if ema9 and ema21 and ema9 > ema21 and px and vwap and px > vwap:
            score += 25
    else:
        if ema9 and ema21 and ema9 < ema21 and px and vwap and px < vwap:
            score += 25

    # Breakout vs premarket / last5
    prem_hi, prem_lo = stock.get("premkt_hi"), stock.get("premkt_lo")
    last5_hi, last5_lo = stock.get("last5_high"), stock.get("last5_low")
    if direction == "CALL":
        if (px and prem_hi and px > prem_hi) or (px and last5_hi and px > last5_hi):
            score += 15
    else:
        if (px and prem_lo and px < prem_lo) or (px and last5_lo and px < last5_lo):
            score += 15

    # RSI band (if present)
    rsi5 = stock.get("rsi5")
    if rsi5 is not None:
        if direction == "CALL" and 55 <= rsi5 <= 75:
            score += 10
        if direction == "PUT" and 25 <= rsi5 <= 45:
            score += 10

    # Options anomalies
    zV = opt.get("z_volume")
    if zV is not None:
        if 2 <= zV < 3:
            score += 10
        elif zV >= 3:
            score += 20

    # IV band
    if iv_rank is not None and 15 <= iv_rank <= 85:
        score += 10

    # Spread quality
    bid, ask = opt.get("bid"), opt.get("ask")
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and (bid + ask) > 0:
        mid = (bid + ask) / 2
        spread = (ask - bid) / mid
        if spread <= 0.015:
            score += 10

    # Overextension penalty: price far from EMA20 (if provided)
    ema20, stdev20 = stock.get("ema20"), stock.get("stdev20")
    if isinstance(px, (int, float)) and isinstance(ema20, (int, float)) and isinstance(stdev20, (int, float)) and stdev20 > 0:
        z = (px - ema20) / stdev20
        if direction == "CALL" and z > 2:
            score -= 15
        if direction == "PUT" and z < -2:
            score -= 15

    return int(max(0, min(100, score)))

# ==============================================================================
# Intraday scanning job (cooldown + dedupe) and Telegram formatter
# ==============================================================================
_COOLDOWN: Dict[Tuple[str, str], float] = {}  # (symbol, direction) -> unix ts
_DEDUP: Dict[str, float] = {}                 # hash key -> unix ts


def _now_ts() -> float:
    return time.time()


def _in_session(now_local: datetime) -> bool:
    t = now_local.time()
    return dt_time(9, 35) <= t <= dt_time(15, 55)


def _cooldown_ok(symbol: str, direction: str, minutes: int = 15) -> bool:
    until = _COOLDOWN.get((symbol, direction), 0.0)
    return _now_ts() >= until


def _mark_cooldown(symbol: str, direction: str, minutes: int = 15) -> None:
    _COOLDOWN[(symbol, direction)] = _now_ts() + minutes * 60


def _dedupe_key(symbol: str, opt: Dict[str, Any], direction: str) -> str:
    return f"{symbol}:{direction}:{opt['strike']}:{opt['expiry']}"


def _dedupe_ok(key: str, minutes: int = 60) -> bool:
    ts = _DEDUP.get(key, 0.0)
    return _now_ts() >= ts


def _mark_dedupe(key: str, minutes: int = 60) -> None:
    _DEDUP[key] = _now_ts() + minutes * 60


async def intraday_options_scan_one(
    client: "httpx.AsyncClient",
    polygon_api_key: str,
    symbol: str,
    stock_ctx: Dict[str, Any],
    now_local: datetime,
) -> List[Dict[str, Any]]:
    """Return alert dicts for a single symbol (0..n)."""
    alerts: List[Dict[str, Any]] = []
    if not _in_session(now_local):
        return alerts

    if not _cooldown_ok(symbol, "CALL") and not _cooldown_ok(symbol, "PUT"):
        return alerts

    underlying = await fetch_underlying_snapshot(client, polygon_api_key, symbol)
    if not underlying:
        return alerts

    try:
        last = float(underlying.get("ticker", {}).get("lastTrade", {}).get("p"))
    except Exception:
        last = None
    if not last:
        return alerts

    chain = await fetch_options_chain_top(client, polygon_api_key, symbol)
    contracts = pick_near_atm_near_dated(chain, last, min_dte=5, max_dte=21, top_n=4)
    if not contracts:
        return alerts

    # Direction preference from stock_ctx
    direction_pref: Optional[str] = None
    ema9, ema21, vwap = stock_ctx.get("ema9"), stock_ctx.get("ema21"), stock_ctx.get("vwap")
    if all(isinstance(x, (int, float)) for x in (ema9, ema21, vwap, last)):
        if ema9 > ema21 and last > vwap:
            direction_pref = "CALL"
        elif ema9 < ema21 and last < vwap:
            direction_pref = "PUT"

    for opt in contracts:
        # Direction by contract type if pref is unclear
        direction = direction_pref or ("CALL" if opt["type"] == "C" else "PUT")

        # z-scores
        z = update_and_zscores(symbol, opt)
        opt.update(z)

        # Basic flow & spread checks
        bid, ask = opt.get("bid"), opt.get("ask")
        if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
            continue
        mid = (bid + ask) / 2.0
        if mid < 0.3:
            continue
        if (ask - bid) / max(mid, 1e-9) > 0.08:
            continue

        # Assemble minimal stock state for scoring
        stock_state = {
            "price": last,
            "vwap": stock_ctx.get("vwap"),
            "ema9": stock_ctx.get("ema9"),
            "ema21": stock_ctx.get("ema21"),
            "premkt_hi": stock_ctx.get("premkt_hi") or stock_ctx.get("premarket_high"),
            "premkt_lo": stock_ctx.get("premkt_lo") or stock_ctx.get("premarket_low"),
            "last5_high": stock_ctx.get("last5_high"),
            "last5_low": stock_ctx.get("last5_low"),
            "ema20": stock_ctx.get("ema20"),
            "stdev20": stock_ctx.get("stdev20"),
            "rsi5": stock_ctx.get("rsi5"),
        }

        score = score_setup(stock_state, opt, direction)
        if score < 60:
            continue

        # Cooldown & dedupe
        if not _cooldown_ok(symbol, direction):
            continue
        key = _dedupe_key(symbol, opt, direction)
        if not _dedupe_ok(key):
            continue

        _mark_cooldown(symbol, direction, minutes=15)
        _mark_dedupe(key, minutes=60)

        alerts.append({
            "symbol": symbol,
            "direction": direction,
            "stock": stock_state,
            "option": opt,
            "ivs": {"rank_52w": None, "percentile_90d": None},
            "score": score,
            "rationale": "Trend+VWAP align, premkt/swing context, near‑ATM contract with flow anomaly",
        })

    return alerts


def render_telegram(alert: Dict[str, Any]) -> str:
    s = alert.get("stock", {})
    o = alert.get("option", {})
    bid, ask = o.get("bid"), o.get("ask")
    spread = (ask - bid) if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) else None
    def _f(v: Any) -> str:
        try:
            if isinstance(v, (int, float)):
                return f"{v:.2f}"
            if v is None:
                return "?"
            return str(v)
        except Exception:
            return "?"
    return (
        f"📈 <b>{alert.get('direction','?')} Alert</b> — {alert.get('symbol','?')} "
        f"{_f(o.get('strike'))}{o.get('type','?')} {o.get('expiry','?')} | Score {alert.get('score','?')}/100\n"
        f"Stock {_f(s.get('price'))} (VWAP {_f(s.get('vwap'))})  EMA9/21: {_f(s.get('ema9'))}/{_f(s.get('ema21'))}\n"
        f"Flow zVol {_f(o.get('z_volume'))}  OIΔ z {_f(o.get('z_oi_change'))}  Spread {_f(spread)}\n"
        f"Vol IV {_f(o.get('iv'))}  Δ {_f(o.get('delta'))}  Γ {_f(o.get('gamma'))}\n"
        f"Triggers: trend/VWAP, near‑ATM, tight spread"
    )

# ==============================================================================
# API Routes
# ==============================================================================

@router.post("/options")
async def ingest_external_option_alert(payload: Dict[str, Any]):
    required = ["symbol", "direction", "stock", "option", "score"]
    if not all(k in payload for k in required):
        raise HTTPException(400, f"Missing one of required fields: {required}")
    preview = render_telegram(payload)
    return {"status": "ok", "message": "received", "preview": preview[:280]}

# Attach router (noop under shim)
app.include_router(router)  # type: ignore

# Health endpoints (decorators are no-ops under shim)
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "component": "alert_server", "router_attached": True, "fastapi_available": FASTAPI_AVAILABLE}

@app.get("/")
async def index() -> Dict[str, Any]:
    return {"message": "Options Alerts Server running", "routes": ["/health", "/alerts/options"], "fastapi_available": FASTAPI_AVAILABLE}

# ==============================================================================
# Offline self-tests (no network) — run this file directly to verify behavior
# ==============================================================================

def _make_fake_chain() -> List[Dict[str, Any]]:
    return [
        {  # good call, near-ATM, tight spread
            "ticker": "TSLA250816C00245000",
            "strike_price": 245,
            "contract_type": "call",
            "expiration_date": "2025-08-16",
            "bid": 3.4,
            "ask": 3.5,
            "last_price": 3.45,
            "volume": 18000,
            "open_interest": 21000,
            "delta": 0.53,
            "gamma": 0.09,
            "iv": 0.47,
        },
        {  # too wide spread -> rejected
            "ticker": "TSLA250816C00260000",
            "strike_price": 260,
            "contract_type": "call",
            "expiration_date": "2025-08-16",
            "bid": 1.00,
            "ask": 1.40,
            "last_price": 1.20,
            "volume": 1200,
            "open_interest": 8000,
        },
        {  # put also near-ATM
            "ticker": "TSLA250816P00245000",
            "strike_price": 245,
            "contract_type": "put",
            "expiration_date": "2025-08-16",
            "bid": 3.3,
            "ask": 3.45,
            "last_price": 3.38,
            "volume": 15000,
            "open_interest": 17000,
            "delta": -0.47,
            "gamma": 0.08,
            "iv": 0.49,
        },
    ]


def _test_pick_filter() -> None:
    chain = _make_fake_chain()
    picks = pick_near_atm_near_dated(chain, underlying_price=246, min_dte=5, max_dte=400, top_n=10)
    assert any(p["type"] == "C" for p in picks), "Should include at least one call"
    assert any(p["type"] == "P" for p in picks), "Should include at least one put"
    # ensure the wide-spread one was filtered out
    assert all(not (p["strike"] == 260 and p["type"] == "C") for p in picks), "Wide spread contract must be filtered"


def _test_pick_filter_with_fixed_now() -> None:
    """Deterministic DTE test: fix 'now' to ensure inclusive DTE keeps 2025-08-16."""
    chain = _make_fake_chain()
    fixed_now = datetime(2025, 8, 12, 12, 0, tzinfo=timezone.utc)
    picks = pick_near_atm_near_dated(chain, underlying_price=246, min_dte=5, max_dte=400, top_n=10, now_utc=fixed_now)
    assert any(p["type"] == "C" for p in picks), "Inclusive DTE should keep a call"
    assert any(p["type"] == "P" for p in picks), "Inclusive DTE should keep a put"


def _test_score_setup() -> None:
    stock = {"price": 246.0, "vwap": 245.0, "ema9": 245.2, "ema21": 244.9, "premkt_hi": 243.0, "last5_high": 244.0, "rsi5": 62.0}
    opt = {"bid": 3.4, "ask": 3.5, "z_volume": 3.2}
    score = score_setup(stock, opt, direction="CALL")
    assert score >= 60, f"Score should pass threshold, got {score}"


def _test_in_session() -> None:
    dt1 = datetime(2025, 8, 11, 10, 0)  # 10:00
    dt2 = datetime(2025, 8, 11, 9, 20)   # 09:20
    assert _in_session(dt1) is True
    assert _in_session(dt2) is False


def _test_fmt() -> None:
    assert _fmt(1.2345) == "1.23"
    assert _fmt(None) == "?"
    assert _fmt("x") == "x"


def _test_app_router_presence() -> None:
    # App and router should exist under both real FastAPI and shim
    assert app is not None
    assert hasattr(app, "include_router")
    assert router is not None


def run_self_tests() -> None:
    _test_pick_filter()
    _test_pick_filter_with_fixed_now()
    _test_score_setup()
    _test_in_session()
    _test_fmt()
    _test_app_router_presence()
    print("All self-tests passed.")

if __name__ == "__main__":  # pragma: no cover
    run_self_tests()
    # For quick local run with reloader (requires FastAPI/uvicorn installed):
    # import uvicorn
    # uvicorn.run("alert_server:app", host="0.0.0.0", port=8000, reload=True)
