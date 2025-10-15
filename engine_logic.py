# Temporary shim so existing imports keep working. Safe to delete once you update imports.
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from engine_common import (
    POLYGON_API_KEY, IBKR_ENABLED, IBKR_DEFAULT_QTY, IBKR_TIF, IBKR_ORDER_MODE, IBKR_USE_MID_AS_LIMIT,
    TARGET_DELTA_CALL, TARGET_DELTA_PUT, MAX_SPREAD_PCT, MAX_QUOTE_AGE_S,
    MIN_VOL_TODAY, MIN_OI, MIN_DTE, MAX_DTE,
    SCAN_MIN_VOL_RTH, SCAN_MIN_OI_RTH, SCAN_MIN_VOL_AH, SCAN_MIN_OI_AH,
    SEND_CHAIN_SCAN_ALERTS, SEND_CHAIN_SCAN_TOPN_ALERTS, REPLACE_IF_NO_NBBO,
    market_now, llm_quota_snapshot, consume_llm,
    get_alert_text_from_request, parse_alert_text,
    round_strike_to_common_increment, _next_friday, same_week_friday, two_weeks_friday, is_same_week,
    _encode_ticker_path, _is_rth_now, _occ_meta, _ticker_matches_side,
    preflight_ok, compose_telegram_text, _ibkr_result_to_dict, _build_plus_minus_contracts,
)

from engine_processor import (
    process_tradingview_job, diag_polygon_bundle, net_debug_info
)

# ========= Perplexity integration (helpers imported from trading_engine) =========
# These exist if you've added the Perplexity patch. If not present, the helpers no-op.
try:
    from trading_engine import perplexity_enrich, boost_score_with_perplexity
except Exception:  # soft-fail to keep legacy behavior
    perplexity_enrich = None          # type: ignore
    boost_score_with_perplexity = None  # type: ignore

# Tunables (env-driven) for when to trigger Sonar and how much to boost
PPLX_BORDERLINE_MIN = float(os.getenv("PPLX_BORDERLINE_MIN", "0.45"))
PPLX_BORDERLINE_MAX = float(os.getenv("PPLX_BORDERLINE_MAX", "0.60"))
PPLX_MAX_NEWS_BOOST = float(os.getenv("PPLX_MAX_NEWS_BOOST", "0.10"))


def _is_borderline(score: Optional[float]) -> bool:
    if score is None:
        return False
    return PPLX_BORDERLINE_MIN <= float(score) <= PPLX_BORDERLINE_MAX


async def perplexity_enrich_if_needed(symbol: str, score: Optional[float]) -> Dict[str, Any]:
    """
    If `score` is borderline, fetch Perplexity news context for `symbol`.
    Returns a dict (possibly empty) with:
      - news_catalysts: list
      - sonar_iv_verdict: True|False|None
      - sonar_iv_view: str|None
      - sonar_citations: list[str]
    """
    if not _is_borderline(score) or perplexity_enrich is None:
        return {
            "news_catalysts": [],
            "sonar_iv_verdict": None,
            "sonar_iv_view": None,
            "sonar_citations": [],
        }
    return await perplexity_enrich(symbol)  # provided by trading_engine patch


def boost_score_with_perplexity_if_any(
    base_score: float,
    px_ctx: Dict[str, Any],
    *,
    max_total_boost: float = PPLX_MAX_NEWS_BOOST,
) -> float:
    """
    Apply Perplexity news boost if context is present.
    Falls back to base_score if boost helper is unavailable.
    """
    if not px_ctx or boost_score_with_perplexity is None:
        return base_score
    return boost_score_with_perplexity(base_score, px_ctx, max_total_boost=max_total_boost)


__all__ = [
    # common
    "POLYGON_API_KEY","IBKR_ENABLED","IBKR_DEFAULT_QTY","IBKR_TIF","IBKR_ORDER_MODE","IBKR_USE_MID_AS_LIMIT",
    "TARGET_DELTA_CALL","TARGET_DELTA_PUT","MAX_SPREAD_PCT","MAX_QUOTE_AGE_S",
    "MIN_VOL_TODAY","MIN_OI","MIN_DTE","MAX_DTE",
    "SCAN_MIN_VOL_RTH","SCAN_MIN_OI_RTH","SCAN_MIN_VOL_AH","SCAN_MIN_OI_AH",
    "SEND_CHAIN_SCAN_ALERTS","SEND_CHAIN_SCAN_TOPN_ALERTS","REPLACE_IF_NO_NBBO",
    "market_now","llm_quota_snapshot","consume_llm",
    "get_alert_text_from_request","parse_alert_text",
    "round_strike_to_common_increment","_next_friday","same_week_friday","two_weeks_friday","is_same_week",
    "_encode_ticker_path","_is_rth_now","_occ_meta","_ticker_matches_side",
    "preflight_ok","compose_telegram_text","_ibkr_result_to_dict","_build_plus_minus_contracts",
    # processor
    "process_tradingview_job","diag_polygon_bundle","net_debug_info",
    # perplexity knobs + helpers
    "PPLX_BORDERLINE_MIN","PPLX_BORDERLINE_MAX","PPLX_MAX_NEWS_BOOST",
    "perplexity_enrich_if_needed","boost_score_with_perplexity_if_any",
]
