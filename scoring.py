from __future__ import annotations

from typing import Dict, Any, Optional, List


# ==========
# Utilities
# ==========

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _nz_num(v: Optional[float]) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _bool01(v: Any) -> float:
    # Treat None/False as 0, True as 1
    return 1.0 if v is True else 0.0


def _delta_band_ok(side: str, delta: Optional[float]) -> Optional[bool]:
    if delta is None:
        return None
    side = (side or "").upper()
    if side == "CALL":
        return 0.30 <= delta <= 0.60
    if side == "PUT":
        return -0.60 <= delta <= -0.30
    return None


def _dte_band_ok(dte: Optional[float]) -> Optional[bool]:
    if dte is None:
        return None
    return 7.0 <= dte <= 21.0


# ===============================================
# Core score (unchanged behavior; 0..1 BUY score)
# ===============================================

def compute_decision_score(features: Dict[str, Any], llm: Dict[str, Any]) -> float:
    """
    Heuristic 0..1 score for BUY quality.
    Uses only 'features' (Polygon-derived) + a few simple rules.
    """
    oi = _nz_num(features.get("oi"))            # open interest (contracts)
    vol = _nz_num(features.get("vol"))          # day volume (contracts)
    spr = _nz_num(features.get("option_spread_pct"))
    qage = _nz_num(features.get("quote_age_sec"))
    delta = _nz_num(features.get("delta"))
    dte = _nz_num(features.get("dte"))
    iv_rank = _nz_num(features.get("iv_rank"))
    side = str(llm.get("side") or features.get("side") or "").upper()

    # Subscores (each 0..1), with weights summing ~1.0
    score = 0.0

    # Liquidity: OI and VOL (0.30 total)
    if oi is not None:
        score += _clamp01(oi / 1000.0) * 0.15
    if vol is not None:
        score += _clamp01(vol / 1000.0) * 0.15

    # Spread tightness (0.15): 0% is perfect, >=30% is bad
    if spr is not None:
        score += (1.0 - _clamp01(spr / 0.30)) * 0.15

    # Quote freshness (0.05): 0s is perfect, >=60s is stale
    if qage is not None:
        score += (1.0 - _clamp01(qage / 60.0)) * 0.05

    # Greeks delta band (0.10)
    d_ok = _delta_band_ok(side, delta)
    if d_ok is not None:
        score += _bool01(d_ok) * 0.10

    # DTE band (0.10): prefer ~1–3 weeks
    dte_ok = _dte_band_ok(dte)
    if dte_ok is not None:
        score += _bool01(dte_ok) * 0.10

    # Expected-move vs breakeven (0.10)
    score += _bool01(features.get("em_vs_be_ok")) * 0.10

    # Alignment (0.10)
    score += _bool01(features.get("mtf_align")) * 0.10

    # S/R headroom (0.10)
    score += _bool01(features.get("sr_headroom_ok")) * 0.10

    # IV rank (0.05): prefer mid regime ~[0.33, 0.66]
    if iv_rank is not None:
        iv_mid = 1.0 if 0.33 <= iv_rank <= 0.66 else 0.0
        score += iv_mid * 0.05

    return _clamp01(score)


# =======================================================
# News / Sonar boost (from Perplexity integration patch)
# =======================================================

def compute_news_catalyst_boost(
    *,
    sonar_verdict: Optional[bool],
    catalysts: Optional[List[Any]],
    max_total_boost: float = 0.10,
) -> float:
    """
    Returns an additive boost in [0..max_total_boost].
    - +0.05 if Sonar says IV is likely to rise (sonar_verdict=True).
    - +0.01 per catalyst headline found, capped at +0.05.
    Overall boost capped by max_total_boost (default 0.10).
    """
    boost = 0.0

    # Sonar IV view: True => likely IV↑
    if sonar_verdict is True:
        boost += 0.05

    # Number of credible catalysts (titles/urls from Search API)
    n = len(catalysts or [])
    if n > 0:
        boost += min(0.05, 0.01 * n)

    return min(max_total_boost, max(0.0, boost))


def apply_news_boost(
    base_score: float,
    *,
    sonar_verdict: Optional[bool],
    catalysts: Optional[List[Any]],
    max_total_boost: float = 0.10,
) -> float:
    """
    Apply the news-catalyst boost (light-touch) to the base score and clamp to [0..1].
    """
    boost = compute_news_catalyst_boost(
        sonar_verdict=sonar_verdict,
        catalysts=catalysts,
        max_total_boost=max_total_boost,
    )
    return _clamp01(base_score + boost)


# ==========================================
# Rating mapping (unchanged semantics)
# ==========================================

def map_score_to_rating(score: Optional[float], decision: Optional[str]) -> Optional[str]:
    """
    Map score to a label but only if the LLM (or system) decided to BUY.
    """
    if score is None:
        return None
    if str(decision or "").lower() != "buy":
        return None
    if score >= 0.75:
        return "Strong Buy"
    if score >= 0.55:
        return "Moderate Buy"
    if score >= 0.40:
        return "Cautious Buy"
    return "Cautious Buy"  # still a buy, but weak edge
