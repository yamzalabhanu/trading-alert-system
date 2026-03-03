# routes.py
import os
import asyncio
import logging
from contextlib import suppress
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import trading_engine as engine

with suppress(Exception):
    from volume_scanner import run_scanner_loop  # type: ignore
with suppress(Exception):
    run_scanner_loop = None  # type: ignore

with suppress(Exception):
    from daily_reporter import send_daily_report_to_telegram  # type: ignore
with suppress(Exception):
    send_daily_report_to_telegram = None  # type: ignore


log = logging.getLogger("trading_engine.routes")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


ENFORCE_WINDOW = _truthy(os.getenv("ENFORCE_WINDOW", "1"))
WINDOW_TZ = os.getenv("WINDOW_TZ", "America/Denver")
WINDOW_START = os.getenv("WINDOW_START", "10:30")
WINDOW_END = os.getenv("WINDOW_END", "14:00")

with suppress(Exception):
    from engine_common import window_ok_now  # type: ignore
with suppress(Exception):
    window_ok_now = None  # type: ignore


def _window_ok() -> bool:
    if callable(window_ok_now):
        try:
            return bool(window_ok_now(WINDOW_START, WINDOW_END, WINDOW_TZ))
        except Exception:
            pass
    return True


_scanner_task: Optional[asyncio.Task] = None
_reporter_task: Optional[asyncio.Task] = None


async def _start_uvscan_task() -> None:
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return
    if run_scanner_loop is None:
        log.info("uvscan not available (volume_scanner missing); skipping")
        return
    _scanner_task = asyncio.create_task(run_scanner_loop(), name="uvscan")
    log.info("uvscan task started")


async def _stop_uvscan_task() -> None:
    global _scanner_task
    if not _scanner_task:
        return
    t = _scanner_task
    _scanner_task = None
    if t.done():
        with suppress(Exception):
            t.result()
        return
    t.cancel()
    with suppress(asyncio.CancelledError):
        await t
    log.info("uvscan task stopped")


async def _start_daily_reporter() -> None:
    global _reporter_task
    if _reporter_task and not _reporter_task.done():
        return
    if send_daily_report_to_telegram is None:
        log.info("daily reporter not available; skipping")
        return

    async def _loop() -> None:
        while True:
            try:
                await send_daily_report_to_telegram()
            except Exception as e:
                log.warning("daily reporter loop error: %r", e)
            await asyncio.sleep(60)

    _reporter_task = asyncio.create_task(_loop(), name="daily_reporter")
    log.info("daily reporter started")


async def _stop_daily_reporter() -> None:
    global _reporter_task
    if not _reporter_task:
        return
    t = _reporter_task
    _reporter_task = None
    if t.done():
        with suppress(Exception):
            t.result()
        return
    t.cancel()
    with suppress(asyncio.CancelledError):
        await t
    log.info("daily reporter stopped")


def bind_lifecycle(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup() -> None:
        if hasattr(engine, "startup") and callable(getattr(engine, "startup")):
            r = engine.startup()  # type: ignore
            if asyncio.iscoroutine(r):
                await r
        await _start_uvscan_task()
        await _start_daily_reporter()
        log.info("startup complete")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await _stop_uvscan_task()
        await _stop_daily_reporter()
        if hasattr(engine, "shutdown") and callable(getattr(engine, "shutdown")):
            r = engine.shutdown()  # type: ignore
            if asyncio.iscoroutine(r):
                await r
        log.info("shutdown complete")


@router.get("/health")
async def health() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if hasattr(engine, "net_debug_info") and callable(getattr(engine, "net_debug_info")):
        with suppress(Exception):
            r = engine.net_debug_info()  # type: ignore
            info = (await r) if asyncio.iscoroutine(r) else (r or {})
    return {"ok": True, **(info or {})}


@router.post("/webhook/tradingview")
async def webhook_tradingview(request: Request, bypass_window: int = 0) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    flags: Dict[str, Any] = {
        "bypass_window": bool(bypass_window),
        "ip": request.client.host if request.client else None,
        "ua": request.headers.get("user-agent"),
        "path": str(request.url.path),
    }

    if ENFORCE_WINDOW and not flags["bypass_window"]:
        if not _window_ok():
            return JSONResponse({"ok": False, "blocked": "outside_window", "bypass_window": False}, status_code=403)

    # Preferred contract: enqueue_webhook_job(alert_text, flags) BUT it might be sync or async.
    enqueue_webhook = getattr(engine, "enqueue_webhook_job", None)
    if callable(enqueue_webhook):
        try:
            res = enqueue_webhook(payload, flags)  # type: ignore[arg-type]
            if asyncio.iscoroutine(res):
                res = await res
            # res is typically bool; we don't require it, but it's useful
            return JSONResponse({"ok": True, "enqueued": bool(res), "bypass_window": bool(bypass_window)})
        except Exception as e:
            log.exception("enqueue_webhook_job failed: %r", e)
            raise HTTPException(status_code=500, detail=f"enqueue_webhook_job failed: {e}")

    # Fallback to older job-dict enqueue contracts
    job = {"alert_text": payload, "flags": flags}

    enqueue_fn = None
    for name in ("enqueue", "enqueue_job", "enqueue_alert", "submit"):
        fn = getattr(engine, name, None)
        if callable(fn):
            enqueue_fn = fn
            break

    if enqueue_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Engine enqueue function not found (expected enqueue_webhook_job or enqueue/enqueue_job/enqueue_alert/submit)",
        )

    try:
        res = enqueue_fn(job)  # type: ignore[misc]
        if asyncio.iscoroutine(res):
            await res
    except Exception as e:
        log.exception("enqueue failed: %r", e)
        raise HTTPException(status_code=500, detail=f"enqueue failed: {e}")

    return JSONResponse({"ok": True, "bypass_window": bool(bypass_window)})
