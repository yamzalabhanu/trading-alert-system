# main.py
import uvicorn
from fastapi import FastAPI
from routes import router
from reporting import start_report_scheduler, stop_report_scheduler

app = FastAPI(title="TradingView Options Alert Ingestor + Telegram")
app.include_router(router)

@app.on_event("startup")
async def on_startup():
    app.state.report_task = start_report_scheduler()

@app.on_event("shutdown")
async def on_shutdown():
    await stop_report_scheduler(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))