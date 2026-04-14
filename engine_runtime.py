# engine_runtime.py
from __future__ import annotations

import os
import asyncio
import logging
import json
from typing import Dict, Any, Optional

import httpx

# ----- logger -----
logger = logging.getLogger("trading_engine")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ----- shared resources -----
HTTP: Optional[httpx.AsyncClient] = None
WORK_Q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
WORKER_COUNT = 0


def get_http_client() -> Optional[httpx.AsyncClient]:
    return HTTP


def get_worker_stats() -> Dict[str, Any]:
    return {"queue_size": WORK_Q.qsize(), "queue_maxsize": WORK_Q.maxsize, "workers": WORKER_COUNT}


def enqueue_webhook_job(alert_text: Any, flags: Dict[str, Any]) -> bool:
    """
    Enqueue a webhook job for async processing.

    alert_text may be:
      - dict (preferred) : TradingView JSON payload
      - str             : raw JSON string
    """
    job = {"alert_text": alert_text, "flags": flags}
    try:
        WORK_Q.put_nowait(job)
        logger.info("enqueue ok; flags=%s", flags)
        return True
    except asyncio.QueueFull:
        logger.warning("enqueue failed: queue full")
        return False


def _preview_alert_text(x: Any, n: int = 200) -> str:
    try:
        if isinstance(x, str):
            return x[:n]
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", errors="replace")[:n]
        if isinstance(x, dict):
            s = json.dumps(x, separators=(",", ":"), ensure_ascii=False)
            return s[:n]
        return str(x)[:n]
    except Exception:
        return "<unprintable alert_text>"


# ----- lifecycle -----
async def startup() -> None:
    global HTTP, WORKER_COUNT
    HTTP = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(read=6.0, write=6.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    WORKER_COUNT = int(os.getenv("WORKERS", "3") or "3")
    for _ in range(WORKER_COUNT):
        asyncio.create_task(_worker())
    logger.info("startup complete; HTTP ready; workers=%d", WORKER_COUNT)


async def shutdown() -> None:
    global HTTP
    if HTTP:
        await HTTP.aclose()
        HTTP = None
        logger.info("shutdown complete; HTTP closed")


# ----- worker loop -----
async def _worker() -> None:
    logger.info("worker task started")
    from engine_logic import process_tradingview_job  # import inside task (avoids import cycles)

    while True:
        job = await WORK_Q.get()
        try:
            logger.info("processing alert job: %s", _preview_alert_text(job.get("alert_text"), 200))
            await process_tradingview_job(job)
            logger.info("job processed")
        except Exception as e:
            logger.exception("[worker] error: %r", e)
        finally:
            WORK_Q.task_done()


__all__ = [
    "startup",
    "shutdown",
    "enqueue_webhook_job",
    "get_worker_stats",
    "get_http_client",
]
