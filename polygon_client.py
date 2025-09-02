# routes.py (FastAPI)
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx, os, json, asyncio

app = FastAPI()

@app.get("/healthz")
async def healthz():
    return {"ok": True}

async def process_tradingview(payload: dict):
    # 1) Validate shape early
    symbol = payload.get("symbol") or payload.get("ticker")
    if not symbol:
        return

    # 2) Call Polygon with short, explicit timeouts
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
    if not POLYGON_API_KEY:
        return

    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0)) as client:
        # Example: get underlying quote to validate connectivity
        try:
            r = await client.get(
                "https://api.polygon.io/v2/last/trade/" + symbol,
                params={"apiKey": POLYGON_API_KEY},
            )
            r.raise_for_status()
        except Exception as e:
            # log and bail; never raise into request stack
            print(f"[polygon] error: {e}")
            return

    # 3) TODO: enqueue for LLM/Telegram, etc.

@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request, bg: BackgroundTasks):
    # 1) Parse body safely
    try:
        payload = await request.json()
    except Exception:
        # Even if not JSON, never blow up (TradingView sometimes sends text)
        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            payload = {"raw": body.decode("utf-8", errors="ignore")}

    # 2) Kick background work and ACK immediately
    bg.add_task(process_tradingview, payload)
    return JSONResponse({"status": "ok"}, status_code=200)
