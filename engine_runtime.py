# engine_runtime.py
import os
import asyncio
import logging
from typing import Dict, Any
import httpx

from config import CDT_TZ  # used indirectly via engine_logic.market_now

# ----- logger -----
logger = logging.getLogger("trading_engine")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ----- shared resources -----
HTTP: httpx.AsyncClient | None = None
WORK_Q: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
WORKER_COUNT = 0

def get_http_client() -> httpx.AsyncClient | None:
    return HTTP

def get_worker_stats() -> Dict[str, Any]:
    return {"queue_size": WORK_Q.qsize(), "queue_maxsize": WORK_Q.maxsize, "workers": WORKER_COUNT}

def enqueue_webhook_job(alert_text: str, flags: Dict[str, Any]) -> bool:
    job = {"alert_text": alert_text, "flags": flags}
    try:
        WORK_Q.put_nowait(job)
        logger.info("enqueue ok; flags=%s", flags)
        return True
    except asyncio.QueueFull:
        logger.warning("enqueue failed: queue full")
        return False

# ----- lifecycle -----
async def startup():
    """
    Initialize HTTP client and spawn workers.
    """
    global HTTP, WORKER_COUNT
    HTTP = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(read=6.0, write=6.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    WORKER_COUNT = int(os.getenv("WORKERS", "3"))
    for _ in range(WORKER_COUNT):
        asyncio.create_task(_worker())
    logger.info("startup complete; HTTP ready; workers=%d", WORKER_COUNT)

async def shutdown():
    """
    Gracefully close HTTP client.
    """
    global HTTP
    if HTTP:
        await HTTP.aclose()
        HTTP = None
        logger.info("shutdown complete; HTTP closed")

# ----- worker loop -----
async def _worker():
    logger.info("worker task started")
    # Import inside function to avoid import-cycle at module import time
    from engine_logic import process_tradingview_job
    while True:
        job = await WORK_Q.get()
        try:
            logger.info("processing alert job: %s", (job.get("alert_text") or "")[:200])
            await process_tradingview_job(job)  # engine_logic pulls HTTP via get_http_client()
            logger.info("job processed")
        except Exception as e:
            logger.exception("[worker] error: %r", e)
        finally:
            WORK_Q.task_done()

__all__ = [
    "startup", "shutdown",
    "enqueue_webhook_job", "get_worker_stats",
    "get_http_client",
]
