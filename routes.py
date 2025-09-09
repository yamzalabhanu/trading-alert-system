# routes.py
import os
import logging
from urllib.parse import quote

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Use the orchestrator module directly
import trading_engine as engine

router = APIRouter()

# ----- Logger -----
logger = logging.getLogger("trading_engine.routes")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ----- App lifecycle wiring -----
def bind_lifecycle(app: FastAPI):
    @app.on_event("startup")
    async def _startup():
        await engine.startup()

    @app.on_event("shutdown")
    async def _shutdown():
        await engine.shutdown()

# Optional alternative if you prefer this name
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
    ib: bool = False,           # ?ib=1 to allow IBKR order when decision == buy
    force: bool = False,        # ?force=1 to override to buy (still logs preflight)
    qty: int = 1,               # ?qty=1 default
):
    """
    Accepts JSON {"message": "..."} or raw text body containing one of:
      CALL Signal: <TICKER> at <UL_PRICE> Strike: <STRIKE> Expiry: YYYY-MM-DD
      CALL Signal: <TICKER> at <UL_PRICE> Strike: <STRIKE>
    (And same for PUT.)
    """
    logger.info("webhook received; ib=%s force=%s qty=%s", ib, force, qty)
    text = await engine.get_alert_text_from_request(request)
    if not text:
        raise HTTPException(status_code=400, detail="Empty alert payload")

    ok = engine.enqueue_webhook_job(
        alert_text=text,
        flags={
            "ib_enabled": bool(ib),
            "qty": int(qty),
            "force_buy": bool(force),
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Queue is full")

    return JSONResponse({
        "queued": True,
        "queue_stats": engine.get_worker_stats(),
        "llm_quota": engine.llm_quota_snapshot(),
    })

# Alias endpoint for TradingView (same behavior as /webhook)
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
            return None
        r.raise_for_status()
        js = r.json()
        return js if isinstance(js, dict) else None
    except Exception:
        return None

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
        from datetime import datetime, timezone, timedelta
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
