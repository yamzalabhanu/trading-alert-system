# routes.py
import os
import asyncio
import logging
from contextlib import suppress
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ---- Project imports (match your repo) ----
# If any of these module names differ in your tree, adjust the import line(s) only.
from config import CDT_TZ, allowed_now_cdt
import trading_engine as engine

from volume_scanner import run_scanner_loop, get_state as uvscan_state
from daily_reporter import build_daily_report, send_daily_report_to_telegram


# ---------------- Logging ----------------
log = logging.getLogger("trading_engine.routes")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()

# ---------------- Background task handles ----------------
_scanner_task: asyncio.Task | None = None
_scanner_ctrl_task: asyncio.Task | None = None
_daily_task: asyncio.Task | None = None


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _bypass_requested(request: Request) -> bool:
    qp = request.query_params
    return _truthy(qp.get("bypass_window")) or _truthy(qp.get("bypass"))


# =========================
# UV Scanner task control
# =========================
async def _start_uvscan_task() -> None:
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return
    _scanner_task = asyncio.create_task(run_scanner_loop(), name="uvscan")
    log.info("uvscan task started")


async def _stop_uvscan_task() -> None:
    global _scanner_task
    if not _scanner_task:
        return

    task = _scanner_task
    _scanner_task = None  # clear first to avoid double-stop races

    if task.done():
        with suppress(Exception):
            task.result()
        log.info("uvscan task already done")
        return

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    log.info("uvscan task stopped")


async def _scanner_window_controller() -> None:
    """
    Starts/stops uvscan based on a CDT window.
    Defaults: 10:30–14:00 CDT (env overridable).
    """
    tz = CDT_TZ if isinstance(CDT_TZ, ZoneInfo) else ZoneInfo("America/Chicago")

    start_h = int(os.getenv("UVSCAN_START_HOUR", "10"))
    start_m = int(os.getenv("UVSCAN_START_MINUTE", "30"))
    end_h = int(os.getenv("UVSCAN_END_HOUR", "14"))
    end_m = int(os.getenv("UVSCAN_END_MINUTE", "0"))

    start_t = dt_time(start_h, start_m)
    end_t = dt_time(end_h, end_m)

    enforce = _truthy(os.getenv("ENFORCE_UVSCAN_WINDOW", "1"))
    poll_s = int(os.getenv("UVSCAN_WINDOW_POLL_SECONDS", "10"))

    log.info(
        "uvscan controller started enforce=%s window=%02d:%02d-%02d:%02d CDT",
        enforce, start_h, start_m, end_h, end_m
    )

    try:
        while True:
            now = datetime.now(tz).time()
            in_window = (start_t <= now <= end_t)
            should_run = (not enforce) or in_window

            if should_run:
                await _start_uvscan_task()
            else:
                await _stop_uvscan_task()

            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        # controller cancelled at shutdown: stop scanner
        with suppress(asyncio.CancelledError):
            await _stop_uvscan_task()
        raise


# =========================
# Daily report scheduler
# =========================
async def _daily_report_scheduler() -> None:
    """
    Sends daily report at configured time (CDT).
    Default: 14:45 CDT (env overridable).
    """
    tz = CDT_TZ if isinstance(CDT_TZ, ZoneInfo) else ZoneInfo("America/Chicago")
    hour = int(os.getenv("DAILY_REPORT_HOUR", "14"))
    minute = int(os.getenv("DAILY_REPORT_MINUTE", "45"))
    poll_s = int(os.getenv("DAILY_REPORT_POLL_SECONDS", "20"))
    enabled = _truthy(os.getenv("DAILY_REPORT_ENABLED", "1"))

    log.info("daily reporter started enabled=%s time=%02d:%02d CDT", enabled, hour, minute)

    last_sent_key: str | None = None

    try:
        while True:
            if not enabled:
                await asyncio.sleep(poll_s)
                continue

            now = datetime.now(tz)
            key = now.strftime("%Y-%m-%d")

            # send once per day when we cross HH:MM
            if now.hour == hour and now.minute == minute and last_sent_key != key:
                try:
                    report = await build_daily_report()
                    await send_daily_report_to_telegram(report)
                    last_sent_key = key
                    log.info("daily report sent for %s", key)
                except Exception:
                    log.exception("daily report failed")

            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        raise


# =========================
# Routes
# =========================
@router.get("/healthz")
async def healthz() -> dict:
    st = {}
    try:
        st = uvscan_state() or {}
    except Exception:
        st = {"ok": False, "error": "uvscan_state failed"}

    return {
        "ok": True,
        "now_cdt": datetime.now(CDT_TZ).isoformat() if CDT_TZ else datetime.now().isoformat(),
        "allowed_now_cdt": bool(allowed_now_cdt()),
        "scanner": st,
        "tasks": {
            "uvscan_running": bool(_scanner_task and not _scanner_task.done()),
            "uvscan_controller_running": bool(_scanner_ctrl_task and not _scanner_ctrl_task.done()),
            "daily_scheduler_running": bool(_daily_task and not _daily_task.done()),
        },
    }


@router.get("/uvscan/status")
async def uvscan_status() -> dict:
    return uvscan_state()


@router.post("/webhook")
@router.post("/webhook/tradingview")
async def webhook_tradingview(request: Request) -> JSONResponse:
    """
    Accepts TradingView webhook. Supports either JSON payload or raw text.
    Enforces CDT alert window via config.allowed_now_cdt() unless bypass_window=1.
    """
    bypass = _bypass_requested(request)

    if (not bypass) and (not allowed_now_cdt()):
        raise HTTPException(status_code=403, detail="Outside allowed alert window (CDT). Use ?bypass_window=1 to test.")

    # Try JSON first; fallback to raw body text
    payload = None
    body_text = ""
    try:
        payload = await request.json()
    except Exception:
        payload = None

    if payload is None:
        raw = await request.body()
        body_text = (raw or b"").decode("utf-8", errors="ignore").strip()
        payload = {"text": body_text} if body_text else {}

    # Hand off to your engine. Your engine can decide how to parse this.
    try:
        await engine.enqueue_alert(payload)
    except AttributeError:
        # If your engine uses a different entrypoint, replace this line.
        # In your repo, you may have engine.process_webhook(payload) or similar.
        await engine.process_webhook(payload)

    return JSONResponse({"ok": True, "bypass_window": bypass})


# =========================
# Lifecycle wiring
# =========================
def bind_lifecycle(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup() -> None:
        # engine startup
        with suppress(Exception):
            await engine.startup()

        # background tasks
        global _scanner_ctrl_task, _daily_task
        _scanner_ctrl_task = asyncio.create_task(_scanner_window_controller(), name="uvscan-controller")
        _daily_task = asyncio.create_task(_daily_report_scheduler(), name="daily-reporter")

        log.info("startup complete")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Cancel controllers first (they will stop scanner)
        global _scanner_ctrl_task, _daily_task
        for t in (_daily_task, _scanner_ctrl_task):
            if t and not t.done():
                t.cancel()

        # Await tasks (CancelledError is expected)
        for t in (_daily_task, _scanner_ctrl_task):
            if t:
                with suppress(asyncio.CancelledError):
                    await t

        _daily_task = None
        _scanner_ctrl_task = None

        # Ensure scanner is stopped even if controller didn’t run
        with suppress(asyncio.CancelledError):
            await _stop_uvscan_task()

        # engine shutdown
        with suppress(Exception):
            await engine.shutdown()

        log.info("shutdown complete")
