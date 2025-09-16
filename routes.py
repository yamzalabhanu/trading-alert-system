# routes.py
import os
import asyncio
import logging
from typing import Optional
from urllib.parse import quote
from datetime import datetime, timezone, timedelta, date as dt_date

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Orchestrator (trading engine) module
import trading_engine as engine
# Unusual-activity scanner (NEW)
from volume_scanner import run_scanner_loop  # <-- runs in background
# Daily reporter (NEW)
from daily_reporter import build_daily_report, send_daily_report_to_telegram

# Logger
log = logging.getLogger("trading_engine.routes")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()

# Keep a handle to the background scanner task (NEW)
_scanner_task = None  # type: ignore

# ----- App lifecycle wiring -----
def bind_lifecycle(app: FastAPI):
    @app.on_event("startup")
    async def _startup():
        global _scanner_task
        await engine.startup()
        # Kick off the unusual-activity scanner (runs forever unless cancelled).
        # It will no-op if POLYGON_API_KEY or UV_SCAN_TICKERS is not set.
        _scanner_task = asyncio.create_task(run_scanner_loop())

    @app.on_event("shutdown")
    async def _shutdown():
        global _scanner_task
        # Stop scanner first so it doesn't race engine shutdown
        if _scanner_task:
            _scanner_task.cancel()
            try:
                await _scanner_task
            except Exception:
                pass
            _scanner_task = None
        await engine.shutdown()

# Optional alternative name
def mount(app: FastAPI):
    return bind_lifecycle(app)

# ----- Health / status -----
@router.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "workers": engine.get_worker_stats(),
        "llm_quota": engine.llm_quota_snapshot(),
    }

@router.get("/engine/stats")
async def engine_stats():
    return {"workers": engine.get_worker_stats()}

@router.get("/engine/quota")
async def engine_quota():
    return engine.llm_quota_snapshot()

# ----- Webhook (TradingView) -----
@router.post("/webhook")
async def webhook(
    request: Request,
    ib: bool = False,     # ?ib=1 to allow IBKR order when decision == buy
    force: bool = False,  # ?force=1 to override to buy
    qty: int = 1,         # ?qty=1 default
):
    log.info("webhook received; ib=%s force=%s qty=%s", ib, force, qty)
    text = await engine.get_alert_text_from_request(request)
    if not text:
        raise HTTPException(status_code=400, detail="Empty alert payload")

    ok = engine.enqueue_webhook_job(
        alert_text=text,
        flags={"ib_enabled": bool(ib), "qty": int(qty), "force_buy": bool(force)},
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Queue is full")

    return JSONResponse({
        "queued": True,
        "queue_stats": engine.get_worker_stats(),
        "llm_quota": engine.llm_quota_snapshot(),
    })

# Exact alias for TradingView (some users configure /webhook/tradingview)
@router.post("/webhook/tradingview")
async def webhook_tradingview(
    request: Request,
    ib: bool = False,
    force: bool = False,
    qty: int = 1,
):
    return await webhook(request=request, ib=ib, force=force, qty=qty)

# ----- Diagnostics -----
@router.get("/net/debug")
async def net_debug():
    return await engine.net_debug_info()

# Lightweight Polygon diagnostics (single, last_quote, open_close[yesterday])
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

async def _http_json(client: httpx.AsyncClient, url: str, params=None, timeout: float = 8.0):
    try:
        r = await client.get(url, params=params or {}, timeout=timeout)
        if r.status_code in (402, 403, 404, 429):
            return {"status": r.status_code, "body": r.text[:400]}
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception as e:
        return {"error": str(e)}

@router.get("/diag/polygon")
async def diag_polygon(underlying: str, contract: str):
    if not POLYGON_API_KEY:
        raise HTTPException(400, "POLYGON_API_KEY not configured")
    enc = quote(contract, safe="")
    out = {}
    async with httpx.AsyncClient(timeout=6.0) as HTTP:
        out["single"] = await _http_json(
            HTTP,
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}",
            {"apiKey": POLYGON_API_KEY},
            6.0,
        )
        out["last_quote"] = await _http_json(
            HTTP,
            f"https://api.polygon.io/v3/quotes/options/{enc}/last",
            {"apiKey": POLYGON_API_KEY},
            6.0,
        )
        yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        out["open_close"] = await _http_json(
            HTTP,
            f"https://api.polygon.io/v1/open-close/options/{enc}/{yday}",
            {"apiKey": POLYGON_API_KEY},
            6.0,
        )

    def skim(d):
        if not isinstance(d, dict):
            return d
        res = d.get("results")
        return {
            "keys": list(d.keys())[:10],
            "sample": (res[:2] if isinstance(res, list) else (res if isinstance(res, dict) else d)),
            "status_hint": d.get("status"),
        }

    return {
        "single": skim(out.get("single")),
        "last_quote": skim(out.get("last_quote")),
        "open_close": skim(out.get("open_close")),
    }

# NBBO diag: show raw status/body for a contract
@router.get("/diag/nbbo")
async def diag_nbbo(ticker: str):
    """
    Example:
      /diag/nbbo?ticker=O:AAPL250912C00245000
    Shows raw HTTP status/body from Polygon last-quote for quick debugging.
    """
    from trading_engine import _http_get_any, _encode_ticker_path, POLYGON_API_KEY as PK
    if not PK:
        raise HTTPException(400, "POLYGON_API_KEY not configured")

    enc = _encode_ticker_path(ticker)
    url = f"https://api.polygon.io/v3/quotes/options/{enc}/last"
    res = await _http_get_any(url, params={"apiKey": PK}, timeout=6.0)
    body = res.get("body")
    if isinstance(body, dict):
        body_sample = {k: body[k] for k in list(body.keys())[:6]}
    else:
        body_sample = (body or "")[:800]
    return {
        "status": res.get("status"),
        "body_sample": body_sample,
    }

# ----- Daily EOD Report (NEW) -----
@router.get("/reports/daily")
async def get_daily_report(date: Optional[str] = None):
    """
    Returns JSON + a monospace table for the given date (CDT).
    date format: YYYY-MM-DD (defaults to today in CDT)
    """
    if date:
        try:
            y, m, d = [int(x) for x in date.split("-")]
            target = dt_date(y, m, d)
        except Exception:
            raise HTTPException(400, "Invalid date; use YYYY-MM-DD")
    else:
        target = None
    rep = await build_daily_report(target)
    return rep

@router.post("/reports/daily/send")
async def send_daily_report(date: Optional[str] = None):
    """
    Sends the daily report to Telegram.
    """
    if date:
        try:
            y, m, d = [int(x) for x in date.split("-")]
            target = dt_date(y, m, d)
        except Exception:
            raise HTTPException(400, "Invalid date; use YYYY-MM-DD")
    else:
        target = None
    res = await send_daily_report_to_telegram(target)
    if not res.get("ok"):
        raise HTTPException(500, res.get("error", "send failed"))
    return res
