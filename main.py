# main.py
import os
import uvicorn
from fastapi import FastAPI
from routes import router
import asyncio
import contextlib

app = FastAPI(title="TradingView Options Alert Ingestor + Telegram")
app.include_router(router)

@app.on_event("startup")
async def on_startup():
    # Start the report scheduler
    from reporting import _daily_report_scheduler
    app.state.report_task = asyncio.create_task(_daily_report_scheduler())

@app.on_event("shutdown")
async def on_shutdown():
    # Cancel the report task
    task = getattr(app.state, "report_task", None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(Exception):
            await task

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
