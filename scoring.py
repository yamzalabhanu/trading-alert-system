# scoring.py
from typing import Dict, Any, Optional

def _bool_score(v: Optional[bool], w_true: float = 1.0, w_false: float = -0.5) -> float:
       if v is True:
        return w_true
    if v is False:
        return w_false
    return 0.0  # unknown / None
    pass

def compute_decision_score(f: Dict[str, Any], llm: Dict[str, Any]) -> float:
      """
    Blend of heuristic feature checks and LLM confidence.
    Scale returns 0..100.
    """
    checklist = llm.get("checklist", {}) if isinstance(llm, dict) else {}
    pts = 0.0
    # Core confirmations
    pts += _bool_score(checklist.get("mtf_trend_alignment"), 8, -5)
    pts += _bool_score(checklist.get("sr_headroom_ok"),       6, -4)
    pts += _bool_score(checklist.get("em_vs_breakeven_ok"),   6, -6)
    pts += _bool_score(checklist.get("delta_band_ok"),        5, -3)
    pts += _bool_score(checklist.get("dte_band_ok"),          4, -6)

    # Levels & VWAP context
    pts += _bool_score(checklist.get("above_pdh"), 3, -1)
    pts += _bool_score(checklist.get("below_pdl"), 3, -1)
    pts += _bool_score(checklist.get("above_pmh"), 2, -1)
    pts += _bool_score(checklist.get("below_pml"), 2, -1)

    # Spread & liquidity mild penalty
    spread = f.get("option_spread_pct")
    if isinstance(spread, (int, float)):
        if spread <= 0.08:  # tight
            pts += 4
        elif spread <= 0.15:
            pts += 1
        else:
            pts -= 4

    # OI/Vol sanity
    oi = f.get("oi") or 0
    vol = f.get("vol") or 0
    if oi >= 500 and vol >= 250:
        pts += 4
    elif oi >= 200:
        pts += 2
    else:
        pts -= 2

    # Regime bonus if trending
    if f.get("regime_flag") == "trending":
        pts += 2

    # IV context preference: medium/low often friendlier for long premium
    iv_ctx = checklist.get("iv_context")
    if iv_ctx == "low":
        pts += 3
    elif iv_ctx == "medium":
        pts += 1
    elif iv_ctx == "high":
        pts -= 2

    # LLM confidence weight
    try:
        conf = float(llm.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    pts += conf * 20.0  # up to +20

    # Clamp and rescale to ~0..100
    pts = max(0.0, min(100.0, 50.0 + pts))  # center 50, allow +/- 50 swing
    return round(pts, 1)
    pass

def map_score_to_rating(score: float, llm_decision: str) -> Optional[str]:
       """
    Convert numeric score + LLM decision to a graded recommendation.
    Only defined for buy decisions.
    """
    if str(llm_decision).lower() != "buy":
        return None
    if score >= 80:
        return "Strong Buy"
    if score >= 60:
        return "Moderate Buy"
    if score >= 50:
        return "Cautious Buy"
    return None

# =======================
# Output formatting
# =======================
    pass