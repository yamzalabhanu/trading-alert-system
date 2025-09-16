# volume_scanner.py
import os
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

import httpx

try:
    # Use shared HTTP client if your app has one
    from engine_runtime import get_http_client
except Exception:
    get_http_client = None  # type: ignore

try:
    from telegram_client import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, send_telegram
except Exception:
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, send_telegram = None, None, None  # type: ignore

LOG = logging.getLogger("trading_engine.uv_scan")

# Public state for /uvscan/status
_STATE: Dict[str, Optional[object]] = {
    "running": False,
    "tickers": None,
    "interval_sec": None,
    "last_run_utc": None,
    "last_alerts": 0,
    "last_error": None,
    "polygon_key_present": False,
}

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

def _env_truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _get_http() -> httpx.AsyncClient:
    if callable(get_http_client):
        cli = get_http_client()
        if cli is not None:
            return cli
    # fallback local client
    return httpx.AsyncClient(timeout=8.0)

async def _poly_json(url: str, params: dict, timeout: float = 8.0) -> Optional[dict]:
    try:
        async with _get_http() as HTTP:
            r = await HTTP.get(url, params=params, timeout=timeout)
            if r.status_code in (402, 403, 404, 429):
                return {"status": r.status_code, "body": r.text[:400]}
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, dict) else None
    except Exception as e:
        return {"error": str(e)}

async def _fetch_equity_bars(symbol: str, minutes_back: int = 90) -> Optional[List[dict]]:
    """
    Pull recent 1-min bars for the current UTC day, then slice last N minutes.
    """
    if not POLYGON_API_KEY:
        return None
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start.isoformat()}/{now.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": POLYGON_API_KEY}
    js = await _poly_json(url, params, timeout=10.0)
    if not isinstance(js, dict):
        return None
    res = js.get("results") or []
    if not isinstance(res, list):
        return None
    return res[-minutes_back:]

def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])

async def _detect_equity_vol_spike(symbol: str, lookback: int, spike_mult: float) -> Optional[Dict[str, object]]:
    bars = await _fetch_equity_bars(symbol, minutes_back=max(lookback + 1, 10))
    if not bars or len(bars) < lookback + 1:
        return None
    vols = [float(b.get("v") or 0) for b in bars[:-1]]
    last = bars[-1]
    last_vol = float(last.get("v") or 0)
    base = _median(vols[-lookback:])
    if base <= 0:
        return None
    if last_vol >= spike_mult * base:
        return {
            "type": "stock_vol_spike",
            "symbol": symbol,
            "last_vol": last_vol,
            "median": base,
            "mult": (last_vol / base),
            "t": last.get("t"),
        }
    return None

async def _fetch_option_snapshots(underlying: str, limit: int = 60) -> Optional[List[dict]]:
    """
    Pull top volume option snapshots for underlying. Some symbols may return 400 from Polygon – we swallow and move on.
    """
    if not POLYGON_API_KEY:
        return None
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
    params = {"limit": limit, "greeks": "true", "order": "desc", "sort": "volume", "apiKey": POLYGON_API_KEY}
    js = await _poly_json(url, params, timeout=10.0)
    if not isinstance(js, dict):
        return None
    res = js.get("results") or []
    return res if isinstance(res, list) else None

async def _detect_options_activity(underlying: str, min_volume: int, min_oi: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    snaps = await _fetch_option_snapshots(underlying, limit=120)
    if not snaps:
        return out
    for s in snaps:
        vol = int(s.get("volume") or 0)
        oi = int(s.get("open_interest") or 0)
        if vol >= min_volume and oi >= min_oi:
            out.append({
                "type": "opt_liq_block",
                "underlying": underlying,
                "contract": s.get("ticker"),
                "vol": vol,
                "oi": oi,
                "last": s.get("last_quote", {}).get("p"),
            })
    return out

async def _send_alert(msg: str) -> None:
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and _env_truthy(os.getenv("UV_SCAN_SEND_ALERTS", "1")):
        try:
            await send_telegram(msg)
        except Exception:
            LOG.exception("[uv-scan] telegram send failed")

async def run_scanner_once() -> Dict[str, object]:
    """
    Run one scan pass; returns summary and updates _STATE.
    Controlled by env:
      UV_SCAN_TICKERS = "WMT,NIO,ARM"
      UV_SCAN_STOCK_LOOKBACK_MIN = 20
      UV_SCAN_STOCK_SPIKE_MULT = 3.0
      UV_SCAN_MIN_OPT_VOL = 5000
      UV_SCAN_MIN_OPT_OI  = 1000
      UV_SCAN_INCLUDE_OPTIONS = "1"
    """
    tickers = [t.strip().upper() for t in (os.getenv("UV_SCAN_TICKERS", "").split(",")) if t.strip()]
    lookback = int(os.getenv("UV_SCAN_STOCK_LOOKBACK_MIN", "20"))
    spike_mult = float(os.getenv("UV_SCAN_STOCK_SPIKE_MULT", "3.0"))
    min_opt_vol = int(os.getenv("UV_SCAN_MIN_OPT_VOL", "5000"))
    min_opt_oi = int(os.getenv("UV_SCAN_MIN_OPT_OI", "1000"))
    include_options = _env_truthy(os.getenv("UV_SCAN_INCLUDE_OPTIONS", "1"))

    _STATE["last_run_utc"] = _now_utc_iso()
    _STATE["last_alerts"] = 0
    _STATE["last_error"] = None

    if not POLYGON_API_KEY:
        _STATE["polygon_key_present"] = False
        LOG.info("[uv-scan] POLYGON_API_KEY missing; scan skipped")
        return {"alerts": 0, "skipped": True}

    _STATE["polygon_key_present"] = True
    alerts = 0

    for i, sym in enumerate(tickers):
        # Stagger requests a little to avoid bursts
        await asyncio.sleep(float(os.getenv("UV_SCAN_STAGGER_SEC", "0.25")))
        try:
            spike = await _detect_equity_vol_spike(sym, lookback, spike_mult)
            if spike:
                alerts += 1
                mult = spike["mult"]
                msg = f"🔔 Unusual STOCK volume: {sym} — last/min-median ≈ {mult:.1f}×"
                LOG.info("[uv-scan] %s", msg)
                await _send_alert(msg)
        except Exception as e:
            _STATE["last_error"] = str(e)
            LOG.warning("[uv-scan] stock check error %s: %s", sym, e)

        if include_options:
            try:
                hits = await _detect_options_activity(sym, min_opt_vol, min_opt_oi)
                if hits:
                    alerts += len(hits)
                    top = hits[0]
                    msg = (f"🔔 Unusual OPTIONS activity: {sym} — "
                           f"{len(hits)} contracts ≥ vol {min_opt_vol}/oi {min_opt_oi}\n"
                           f"Top: {top.get('contract')}  Vol={top.get('vol')}  OI={top.get('oi')}")
                    LOG.info("[uv-scan] %s", msg.replace("\n", " | "))
                    await _send_alert(msg)
            except Exception as e:
                _STATE["last_error"] = str(e)
                LOG.warning("[uv-scan] opt check error %s: %s", sym, e)

    _STATE["last_alerts"] = alerts
    return {"alerts": alerts, "skipped": False}

async def run_scanner_loop() -> None:
    """
    Background loop. Controlled by:
      UV_SCAN_ENABLED = 1
      UV_SCAN_INTERVAL_SEC = 300
    It quietly idles if disabled or no tickers are configured.
    """
    task = asyncio.current_task()
    try:
        if task is not None:
            try:
                task.set_name("uv-scan")
            except Exception:
                pass

        interval = int(os.getenv("UV_SCAN_INTERVAL_SEC", "300"))
        tickers_env = os.getenv("UV_SCAN_TICKERS", "").strip()
        enabled = _env_truthy(os.getenv("UV_SCAN_ENABLED", "1"))
        _STATE["interval_sec"] = interval
        _STATE["tickers"] = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]

        if not enabled:
            LOG.info("[uv-scan] disabled by UV_SCAN_ENABLED=0")
        elif not _STATE["tickers"]:
            LOG.info("[uv-scan] no tickers set (UV_SCAN_TICKERS), idling")
        else:
            LOG.info("[uv-scan] started | interval=%ss | tickers=%s",
                     interval, ",".join(_STATE["tickers"]))

        _STATE["running"] = True
        while True:
            if enabled and _STATE["tickers"]:
                await run_scanner_once()
            await asyncio.sleep(interval)
            # Re-read toggle each loop, so you can enable/disable on the fly
            enabled = _env_truthy(os.getenv("UV_SCAN_ENABLED", "1"))
    except asyncio.CancelledError:
        LOG.info("[uv-scan] cancelled; exiting loop")
        raise
    except Exception:
        LOG.exception("[uv-scan] fatal error")
    finally:
        _STATE["running"] = False

def get_state() -> Dict[str, Optional[object]]:
    # Shallow copy for safety
    return dict(_STATE)
