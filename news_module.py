import os, time
item = self._news.get(sym)
if not item: return None
ts, payload = item
return payload if time.time() - ts < NEWS_TTL else None


def put_news(self, sym: str, items: List[Catalyst]):
self._news[sym] = (time.time(), items)


def get_sonar(self, sym: str) -> Optional[dict]:
item = self._sonar.get(sym)
if not item: return None
ts, payload = item
return payload if time.time() - ts < SONAR_TTL else None


def put_sonar(self, sym: str, verdict: dict):
self._sonar[sym] = (time.time(), verdict)


cache = NewsCache()


async def fetch_catalysts(pplx: PerplexityClient, sym: str) -> List[Catalyst]:
if not SEARCH_ENABLED:
return []
cached = cache.get_news(sym)
if cached is not None:
return cached
q = f"{sym} earnings OR guidance OR downgrade OR upgrade OR SEC 8-K OR ‘halt’ OR ‘merger’ today"
data = await pplx.search(q, num_results=8)
out: List[Catalyst] = []
for r in data.get("results", [])[:5]:
out.append(Catalyst(
title=r.get("title", ""), url=r.get("url", ""),
published=r.get("published_at"), snippet=r.get("snippet")
))
cache.put_news(sym, out)
return out


async def sonar_iv_verdict(pplx: PerplexityClient, sym: str) -> dict:
if not SONAR_ENABLED:
return {"verdict": None, "answer": None, "citations": []}
cached = cache.get_sonar(sym)
if cached is not None:
return cached


messages = [{
"role": "user",
"content": (
f"Given credible news today for {sym}, is 1–2 week implied volatility likely to rise? "
f"Answer strictly as JSON with fields: verdict (yes/no/unclear) and rationale (<=25 words)."
f" Cite sources."
)
}]
res = await pplx.chat(MODEL_SONAR, messages, temperature=0.2, max_tokens=200, return_citations=True)
msg = res.get("choices", [{}])[0].get("message", {})
content = msg.get("content", "")
cites = msg.get("citations", []) or []


# Soft parse for yes/no/unclear
low = content.lower()
if "yes" in low: verdict = True
elif "no" in low: verdict = False
else: verdict = None


payload = {"verdict": verdict, "answer": content, "citations": cites}
cache.put_sonar(sym, payload)
return payload
