import os
import httpx
from typing import Any, Dict, List, Optional


BASE_URL = "https://api.perplexity.ai"
PPLX_API_KEY = os.getenv("PPLX_API_KEY", "")
TIMEOUT_SECS = float(os.getenv("PPLX_TIMEOUT_SECS", "15"))


class PerplexityClient:
"""Thin async client for Perplexity Search + Chat (Sonar / pplx-api)."""


def __init__(self, *, timeout: Optional[float] = None):
self.timeout = timeout or TIMEOUT_SECS
self._client: Optional[httpx.AsyncClient] = None


async def _get_client(self) -> httpx.AsyncClient:
if self._client is None:
self._client = httpx.AsyncClient(
base_url=BASE_URL,
timeout=self.timeout,
headers={
"Authorization": f"Bearer {PPLX_API_KEY}",
"Accept": "application/json",
"Content-Type": "application/json",
},
)
return self._client


async def search(self, query: str, **params) -> Dict[str, Any]:
"""Perplexity Search API — ranked, structured web results.
Example returns: {"results":[{"title","url","snippet","published_at",...}, ...]}
"""
client = await self._get_client()
resp = await client.post("/search", json={"query": query, **params})
resp.raise_for_status()
return resp.json()


async def chat(self, model: str, messages: List[Dict[str, str]], **params) -> Dict[str, Any]:
"""OpenAI‑compatible chat (Sonar / pplx‑api) with optional citations via return_citations=True."""
client = await self._get_client()
payload = {"model": model, "messages": messages} | params
resp = await client.post("/chat/completions", json=payload)
resp.raise_for_status()
return resp.json()


async def aclose(self) -> None:
if self._client is not None:
await self._client.aclose()
self._client = None
