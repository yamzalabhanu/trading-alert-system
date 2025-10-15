"""
Thin façade preserving the original import surface,
plus Perplexity-powered news enrichment & scoring helpers.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

# ===== Original façade re-exports =====
from engine_runtime import (
    startup, shutdown,
    enqueue_webhook_job, get_worker_stats,
    get_http_client,
)
from engine_logic import (
    market_now, llm_quota_snapshot,
    get_alert_text_from_request, parse_alert_text,
    preflight_ok, compose_telegram_text,
    diag_polygon_bundle, net_debug_info,
)

# ===== Optional Perplexity integration (safe to import if files exist) =====
try:
    from perplexity_client import PerplexityClient  # new module from patch
    from news_module import fetch_catalysts, sonar_iv_verdict  # new module from patch
    from scoring import apply_news_boost  # your updated scoring.py
except Exception:
    # Soft-fail if Perplexity modules aren’t present; helpers will no-op.
    PerplexityClient = None  # type: ignore
    fetch_catalysts = None   # type: ignore
    sonar_iv_verdict = None  # type: ignore
    apply_news_boost = None  # type: ignore


async def perplexity_enrich(symbol: str) -> Dict[str, Any]:
    """
    Fetches recent catalysts (Search API) and an IV-impact verdict (Sonar) for `symbol`.
    Returns a portable dict you can attach to your signal/decision context.

    Shape:
    {
      "news_catalysts": [{"title","url","published","snippet"}, ...],
      "sonar_iv_verdict": True|False|None,
      "sonar_iv_view": "<short text / JSON-ish>",
      "sonar_citations": [<url>...]
    }
    """
    if PerplexityClient is None or fetch_catalysts is None or sonar_iv_verdict is None:
        # Perplexity not installed; return empty context.
        return {
            "news_catalysts": [],
            "sonar_iv_verdict": None,
            "sonar_iv_view": None,
            "sonar_citations": [],
        }

    pplx = PerplexityClient()
    try:
        cats = await fetch_catalysts(pplx, symbol)
        sonar = await sonar_iv_verdict(pplx, symbol)
    finally:
        try:
            await pplx.aclose()
        except Exception:
            pass

    catalysts_list = []
    for c in (cats or []):
        # c may be a dataclass Catalyst or a dict depending on your import path
        if isinstance(c, dict):
            catalysts_list.append({
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "published": c.get("published"),
                "snippet": c.get("snippet"),
            })
        else:
            # dataclass with attributes
            catalysts_list.append({
                "title": getattr(c, "title", ""),
                "url": getattr(c, "url", ""),
                "published": getattr(c, "published", None),
                "snippet": getattr(c, "snippet", None),
            })

    return {
        "news_catalysts": catalysts_list,
        "sonar_iv_verdict": sonar.get("verdict") if isinstance(sonar, dict) else None,
        "sonar_iv_view": sonar.get("answer") if isinstance(sonar, dict) else None,
        "sonar_citations": sonar.get("citations", []) if isinstance(sonar, dict) else [],
    }


def boost_score_with_perplexity(base_score: float, ctx: Dict[str, Any], *, max_total_boost: float = 0.10) -> float:
    """
    Applies a light additive boost to `base_score` using Perplexity context.
    Safe no-op if scoring.apply_news_boost is unavailable.

    `ctx` should be the dict returned by `perplexity_enrich`, or at least contain:
      - ctx["sonar_iv_verdict"] -> True/False/None
      - ctx["news_catalysts"]   -> list of catalysts (may be empty)
    """
    if apply_news_boost is None:
        return base_score

    verdict = ctx.get("sonar_iv_verdict")
    catalysts = ctx.get("news_catalysts") or []
    return apply_news_boost(
        base_score,
        sonar_verdict=verdict,
        catalysts=catalysts,
        max_total_boost=max_total_boost,
    )


__all__ = [
    # lifecycle / runtime
    "startup", "shutdown", "enqueue_webhook_job", "get_worker_stats", "get_http_client",
    # logic / helpers
    "market_now", "llm_quota_snapshot", "get_alert_text_from_request", "parse_alert_text",
    "preflight_ok", "compose_telegram_text",
    # diagnostics
    "diag_polygon_bundle", "net_debug_info",
    # perplexity helpers
    "perplexity_enrich", "boost_score_with_perplexity",
]
