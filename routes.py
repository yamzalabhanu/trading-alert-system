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

# Timezone
from config import CDT_TZ

# Orchestrator (trading engine) module
import trading_engine as engine
# Unusual-activity scanner (runs in background)
from volume_scanner import run_scanner_loop, get_state as uvscan_state
# Daily reporter
from daily_reporter import build_daily_report, send_daily_report_to_telegram

# Logger
log = logging.getLogger("trading_engine.routes")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()

# Background tasks
_scanner_task: Optional[asyncio.Task] = None
_uvscan_controller_task: Optional[asyncio.Task] = None  # manages the time window

# ---------- UV-scan window gating (10:30–14:00 CDT) ----------
def _in_uvscan_window_cdt(now: Optional[datetime] = None) -> bool:
    """
    Only allow UV-scan between 10:30 and 14:00 CDT.
    Optional weekday restriction with UVSCAN_WEEKDAYS_ONLY=1 (default).
    """
    now = now or datetime.now(CDT_TZ)
    if os.getenv("UVSCAN_WEEKDAYS_ONLY", "1") == "1" and now.weekday() > 4:
        return False
    minutes = now.hour * 60 + now.minute
    return (10 * 60 + 30) <= minutes < (14 * 60)

async def _stop_uvscan_task():
    global _scanner_task
    if _scanner_task:
        _scanner_task.cancel()
        try:
            await _scanner_task
        except Exception:
            pass
        _scanner_task = None
        log.info("[routes] uv-scan task stopped")

def _start_uvscan_task():
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return
    try:
        _scanner_task = asyncio.create_task(run_scanner_loop(), name="uv-scan")
    except TypeError:
        _scanner_task = asyncio.create_task(run_scanner_loop())
    log.info("[routes] uv-scan task started")

async def _uvscan_window_controller_loop():
    """
    Enforces the 10:30–14:00 CDT window. Checks every 30s by default.
    Disable enforcement by UVSCAN_ENFORCE_WINDOW=0 (for debugging).
    """
    enforce = os.getenv("UVSCAN_ENFORCE_WINDOW", "1") == "1"
    poll = int(os.getenv("UVSCAN_WINDOW_POLL_SECONDS", "30"))
    if not enforce:
        log.info("[routes] uv-scan window enforcement disabled")
    while True:
        try:
            if enforce:
                in_window = _in_uvscan_window_cdt()
                running = bool(_scanner_task and not _scanner_task.done())
                if in_window and not running:
                    _start_uvscan_task()
                elif not in_window and running:
                    await _stop_uvscan_task()
            await asyncio.sleep(poll)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("[routes] uv-scan controller loop error: %r", e)
            await asyncio.sleep(5)

# ---------- App lifecycle ----------
def bind_lifecycle(app: FastAPI):
    @app.on_event("startup")
    async def _startup():
        global _uvscan_controller_task
        await engine.startup()
        # Start controller; it will start/stop the scan task inside the window.
        if _uvscan_controller_task is None or _uvscan_controller_task.done():
            try:
                _uvscan_controller_task = asyncio.create_task(_uvscan_window_controller_loop(), name="uv-scan-ctl")
            except TypeError:
                _uvscan_controller_task = asyncio.create_task(_uvscan_window_controller_loop())
            log.info("[routes] uv-scan controller started")

    @app.on_event("shutdown")
    async def _shutdown():
        global _uvscan_controller_task
        # Stop controller first so it doesn't restart uv-scan mid-shutdown
        if _uvscan_controller_task:
            _uvscan_controller_task.cancel()
            try:
                await _uvscan_controller_task
            except Exception:
                pass
            _uvscan_controller_task = None
            log.info("[routes] uv-scan controller stopped")
        # Then stop the scan task if running
        await _stop_uvscan_task()
        await engine.shutdown()

# Optional alternative name
def mount(app: FastAPI):
    return bind_lifecycle(app)

# ---------- Health / status ----------
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

# ---------- Webhook (TradingView) ----------
@router.post("/webhook")
async def webhook(
    request: Request,
    ib: bool = False,
    force: bool = False,
    qty: int = 1,
    bypass_window: bool = False,  # for manual testing of the alert window guard
):
    log.info("webhook received; ib=%s force=%s qty=%s", ib, force, qty)
    # (Webhook windows handled elsewhere if you already added them)
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

@router.post("/webhook/tradingview")
async def webhook_tradingview(
    request: Request,
    ib: bool = False,
    force: bool = False,
    qty: int = 1,
    bypass_window: bool = False,
):
    return await webhook(request=request, ib=ib, force=force, qty=qty, bypass_window=bypass_window)

# ---------- uv-scan control & status ----------
@router.get("/uvscan/status")
async def uvscan_status():
    running = bool(_scanner_task and not _scanner_task.done())
    ctl_running = bool(_uvscan_controller_task and not _uvscan_controller_task.done())
    in_window = _in_uvscan_window_cdt()
    return {
        "running": running,
        "controller_running": ctl_running,
        "in_window_now_cdt": in_window,
        "window_cdt": "10:30–14:00",
        "state": uvscan_state(),
    }

@router.post("/uvscan/start")
async def uvscan_start():
    # Respect the enforced window
    if not _in_uvscan_window_cdt() and os.getenv("UVSCAN_ENFORCE_WINDOW", "1") == "1":
        return {"started": False, "reason": "outside_uvscan_window", "window_cdt": "10:30–14:00"}
    _start_uvscan_task()
    return {"started": True}

@router.post("/uvscan/stop")
async def uvscan_stop():
    await _stop_uvscan_task()
    return {"stopped": True}

# ---------- Diagnostics ----------
@router.get("/net/debug")
async def net_debug():
    return await engine.net_debug_info()

# Lightweight Polygon diagnostics
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
            f"https://api.polygon.io/v3/snapshot/options/{underlying}/{enc}?",
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

# NBBO diag
@router.get("/diag/nbbo")
async def diag_nbbo(ticker: str):
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

# ---------- Daily EOD Report ----------
@router.get("/reports/daily")
async def get_daily_report(date: Optional[str] = None):
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
