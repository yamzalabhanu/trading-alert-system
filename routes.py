# routes.py
import os
from fastapi import APIRouter, HTTPException, Request, FastAPI

# Runtime surface from the new package
from trading_engine import (
    startup, shutdown, enqueue_webhook_job,
    diag_polygon_bundle, net_debug_info,
    get_worker_stats, llm_quota_snapshot,
)

# Request/alert parsing helper from utils
from trading_engine.engine_utils import get_alert_text_from_request

router = APIRouter()

# Optional: if you prefer a single function to mount lifecycle onto your FastAPI app
def mount(app: FastAPI):
    @app.on_event("startup")
    async def _on_startup():
        await startup()

    @app.on_event("shutdown")
    async def _on_shutdown():
        await shutdown()

# --- Health / status ---

@router.get("/healthz")
async def healthz():
    # very cheap liveness; add more if you like
    return {"ok": True, "workers": get_worker_stats(), "llm_quota": llm_quota_snapshot()}

@router.get("/engine/stats")
async def engine_stats():
    return {"workers": get_worker_stats()}

@router.get("/engine/quota")
async def engine_quota():
    return llm_quota_snapshot()

# --- Diagnostics (unchanged behavior, new imports) ---

@router.get("/net/debug")
async def net_debug():
    return await net_debug_info()

@router.get("/poly/diag")
async def poly_diag(sym: str, contract: str):
    """
    Example:
      /poly/diag?sym=RKLB&contract=O:RKLB250919C00050000
    """
    return await diag_polygon_bundle(sym, contract)

# --- Webhook (TradingView) ---

@router.post("/webhook")
async def webhook(
    request: Request,
    ib: bool = False,           # ?ib=1 to allow IBKR order when decision == buy
    force: bool = False,        # ?force=1 to override to buy (still logs preflight)
    qty: int = 1,               # ?qty=1 default
):
    """
    Accepts JSON {"message": "..."} or raw text body containing one of:
      CALL Signal: <TICKER> at <UL_PRICE> Strike: <STRIKE> Expiry: YYYY-MM-DD
      CALL Signal: <TICKER> at <UL_PRICE> Strike: <STRIKE>
    And same for PUT.

    Example curl:
      curl -X POST 'http://localhost:8000/webhook?ib=false&force=false&qty=1' \
        -H 'content-type: application/json' \
        -d '{"message":"CALL Signal: RKLB at 47.66 Strike: 50 Expiry: 2025-09-19"}'
    """
    alert_text = await get_alert_text_from_request(request)
    if not alert_text:
        raise HTTPException(400, "Empty alert payload")

    flags = {
        "ib_enabled": bool(ib),
        "force_buy": bool(force),
        "qty": int(qty),
    }

    ok = enqueue_webhook_job(alert_text, flags)
    if not ok:
        raise HTTPException(503, "Worker queue is full")

    return {
        "queued": True,
        "workers": get_worker_stats(),
        "llm_quota": llm_quota_snapshot(),
    }
