# routes.py
import asyncio
import logging
from contextlib import suppress

log = logging.getLogger(__name__)

_scanner_task: asyncio.Task | None = None

async def _start_uvscan_task() -> None:
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return  # already running

    # Create task
    _scanner_task = asyncio.create_task(run_scanner_loop(), name="uvscan")
    log.info("uvscan task started")

async def _stop_uvscan_task() -> None:
    global _scanner_task
    if not _scanner_task:
        return

    task = _scanner_task
    _scanner_task = None  # clear first to prevent re-entrancy / double-stop

    if task.done():
        # consume exception if any (prevents "Task exception was never retrieved")
        with suppress(Exception):
            task.result()
        log.info("uvscan task already done")
        return

    task.cancel()
    # Cancellation is expected during shutdown
    with suppress(asyncio.CancelledError):
        await task
    log.info("uvscan task stopped")
