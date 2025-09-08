# routes.py
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from datetime import datetime

from config import COOLDOWN_SECONDS, CDT_TZ, WINDOWS_CDT, MAX_LLM_PER_DAY
from models import Alert, WebhookResponse  # kept for compatibility if referenced elsewhere
from reporting import _DECISIONS_LOG, _send_daily_report_now, _summarize_day_for_report, _chunk_lines_for_telegram
from telegram_client import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

from trading_engine import (
    startup, shutdown,
    market_now, llm_quota_snapshot,
    enqueue_webhook_job, parse_alert_text, get_worker_stats,
    diag_polygon_bundle, net_debug_info, get_http_client,
)

router = APIRouter()

# =========================
# Lifespan
# =========================
@router.on_event("startup")
async def _startup():
    await startup()

@router.on_event("shutdown")
async def _shutdown():
    await shutdown()

# =========================
# Routes
# =========================
@router.get("/health")
def health():
    return {"ok": True, "component": "alert_server", "fastapi_available": True}

@router.get("/healthz")
def healthz():
    return {"ok": True}

@router.get("/quota")
def quota():
    return {"ok": True, "quota": llm_quota_snapshot()}

@router.get("/config")
def get_config():
    cfg = {
        "COOLDOWN_SECONDS": COOLDOWN_SECONDS,
        "WINDOWS_CDT": WINDOWS_CDT,
        "MAX_LLM_PER_DAY": MAX_LLM_PER_DAY,
        "CDT_TZ": str(CDT_TZ),
    }
    return {"ok": True, "config": cfg}

@router.get("/logs/today")
def logs_today(limit: int = 50):
    limit = max(1, min(int(limit), 500))
    today_local = market_now().date()
    todays = [x for x in reversed(_DECISIONS_LOG)
              if isinstance(x.get("timestamp_local"), datetime) and x["timestamp_local"].date() == today_local]
    return {"ok": True, "count": len(todays[:limit]), "items": todays[:limit]}

@router.get("/worker/stats")
def worker_stats():
    return {"ok": True, **get_worker_stats()}

@router.post("/run/daily_report")
async def run_daily_report():
    res = await _send_daily_report_now()
    return {"ok": True, "trigger": "manual", **res}

@router.get("/net/debug")
async def net_debug():
    return await net_debug_info()

@router.get("/report/preview")
def report_preview():
    today_local = market_now().date()
    rep = _summarize_day_for_report(today_local)
    chunks = _chunk_lines_for_telegram(rep["contracts"], prefix=f"🧾 Contracts ({rep['count']}):")
    return {"ok": True, "header": rep["header"], "contract_chunks": chunks, "count": rep["count"]}

# --- Non-blocking webhook: ACK + enqueue ---
@router.post("/webhook", response_class=JSONResponse)
@router.post("/webhook/tradingview", response_class=JSONResponse)
async def webhook_tradingview(
    request: Request,
    offline: int = Query(default=0),
    ib: int = Query(default=0),
    qty: int = Query(default=1),
    force: int = Query(default=0),
    force_buy: int = Query(default=0),
    debug: int = Query(default=0),
):
    # Parse/validate payload now (fast-fail on bad format)
    from trading_engine import get_alert_text_from_request  # lazy import to avoid circulars
    payload_text = await get_alert_text_from_request(request)
    if not payload_text:
        raise HTTPException(status_code=400, detail="Empty alert payload")
    try:
        _ = parse_alert_text(payload_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid alert: {e}")

    # Determine effective IBKR enablement
    from trading_engine import IBKR_ENABLED, IBKR_DEFAULT_QTY
    effective_ib_enabled = bool(ib) if request.query_params.get("ib") is not None else IBKR_ENABLED

    flags = {
        "ib_enabled": effective_ib_enabled and (not offline),
        "force": bool(force),
        "force_buy": bool(force_buy),
        "qty": int(qty) if qty is not None else IBKR_DEFAULT_QTY,
        "debug": bool(debug),
    }
    accepted = enqueue_webhook_job(payload_text, flags)
    if not accepted:
        return JSONResponse({"status": "busy", "detail": "queue full"}, status_code=429)
    return JSONResponse({"status": "accepted"}, status_code=202)

# ================ Diagnostics ================
@router.get("/diag/polygon")
async def diag_polygon(underlying: str, contract: str):
    return await diag_polygon_bundle(underlying=underlying, contract=contract)
