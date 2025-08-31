# main.py - minimal version to test
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="TradingView Options Alert Ingestor + Telegram")

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
