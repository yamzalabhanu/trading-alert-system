# flow_server.py
import os
import httpx
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

app = FastAPI()

@app.get("/flow/{symbol}")
async def get_flow_score(symbol: str):
    url = f"https://api.polygon.io/v3/reference/options/contracts?ticker={symbol}&apiKey={POLYGON_API_KEY}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
        data = res.json()
        flow = compute_flow(data)
        return {"symbol": symbol, "flow": flow}

def compute_flow(data):
    if "results" not in data:
        return "neutral"
    call_flow = sum(1 for opt in data["results"] if opt["type"] == "call" and opt["volume"] > opt["open_interest"] * 1.5)
    put_flow = sum(1 for opt in data["results"] if opt["type"] == "put" and opt["volume"] > opt["open_interest"] * 1.5)
    if call_flow > put_flow:
        return "bullish"
    elif put_flow > call_flow:
        return "bearish"
    return "neutral"
