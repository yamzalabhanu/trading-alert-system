# routes.py
import asyncio
import logging
from contextlib import suppress
from fastapi import APIRouter, FastAPI

# IMPORTANT: import your scanner loop from wherever it lives
# If it's in volume_scanner.py, keep this:
from volume_scanner import run_scanner_loop

log = logging.getLogger(__name__)

router = APIRouter()

_scanner_task: asyncio.Task | None = None


async def _start_uvscan_task() -> None:
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return  # already running

    _scanner_task = asyncio.create_task(run_scanner_loop(), name="uvscan")
    log.info("uvscan task started")


async def _stop_uvscan_task() -> None:
    global _scanner_task
    if not _scanner_task:
        return

    task = _scanner_task
    _scanner_task = None  # clear first to prevent re-entrancy / double-stop

    if task.done():
        with suppress(Exception):
            task.result()
        log.info("uvscan task already done")
        return

    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    log.info("uvscan task stopped")


def bind_lifecycle(app: FastAPI) -> None:
    @app.on_event("startup")
    async def _startup() -> None:
        # start whatever you want at boot
        await _start_uvscan_task()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # graceful stop
        try:
            await _stop_uvscan_task()
        except asyncio.CancelledError:
            log.info("Shutdown cancelled (expected)")
        except Exception:
            log.exception("Shutdown handler failed")
