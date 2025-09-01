# feature_engine.py
from __future__ import annotations

from datetime import datetime, timezone, date
from math import sqrt
from typing import Any, Dict, Optional

# If you have a proper market clock util, import and use it; otherwise use UTC now.
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    if ask <= 0:
        return None
    return (bid + ask) / 2.0

def _spread_pct(bid: Optional[float], ask: Optional[float], mid: Optional[float]) -> Optional[float]:
    if bid is None or ask is None or mid is None or mid <= 0:
        return None
    return (ask - bid) / mid

def _quote_age_seconds(last_quote_ts: Optional[int]) -> Optional[float]:
    """
    Polygon often reports nanosecond timestamps. Accept seconds or ns.
    """
    if last_quote_ts is None:
        return None
    # Detect ns vs s
    try:
        # If it's too large, treat as ns
        if last_quote_ts > 10_000_000_000:  # > year 2286 in seconds; clearly ns
            ts_sec = last_quote_ts / 1_000_000_000.0
        else:
            ts_sec = float(last_quote_ts)
        return max(0.0, _utcnow().timestamp() - ts_sec)
    except Exception:
        return None

def _dte(expiry_iso: Optional[str]) -> Optional[float]:
    if not expiry_iso:
        return None
    try:
        exp = date.fromisoformat(expiry_iso)
        today = _utcnow().date()
        return float((exp - today).days)
    except Exception:
        return None

def _expected_move_pct(iv: Optional[float], dte_days: Optional[float]) -> Optional[float]:
    if iv is None or dte_days is None or dte_days < 0:
        return None
    # simple 1σ move (annualized IV) scaled by sqrt(time)
    return iv * sqrt(dte_days / 365.0)

async def build_features(
    client,  # httpx.AsyncClient (unused directly here but kept for future fetches)
    alert: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize Polygon option snapshot + alert context into the fields routes.py expects.
    This is intentionally defensive: any missing field becomes None, and booleans are derived simply.
    """

    # --- Parse snapshot (Polygon shape can vary; handle multiple shapes) ---
    # Common shapes:
    # snapshot.get("last_quote", {"bid":..., "ask":..., "timestamp":...})
    # snapshot.get("greeks", {"delta":..., "gamma":..., "theta":..., "vega":...})
    # snapshot.get("implied_volatility"), snapshot.get("open_interest"), snapshot.get("day", {"volume":...})
    lq = snapshot.get("last_quote") or snapshot.get("last_quote_us") or {}
    bid = _safe_float(lq.get("bid"))
    ask = _safe_float(lq.get("ask"))
    last_quote_ts = lq.get("timestamp") or lq.get("t")  # ns or sec

    greeks = snapshot.get("greeks") or {}
    delta = _safe_float(greeks.get("delta"))
    gamma = _safe_float(greeks.get("gamma"))
    theta = _safe_float(greeks.get("theta"))
    vega  = _safe_float(greeks.get("vega"))

    iv = _safe_float(snapshot.get("implied_volatility") or snapshot.get("iv"))
    oi = snapshot.get("open_interest")
    oi = int(oi) if isinstance(oi, (int, float)) and oi == int(oi) else _safe_float(oi)
    vol = snapshot.get("day", {}).get("volume")
    vol = int(vol) if isinstance(vol, (int, float)) and vol == int(vol) else _safe_float(vol)

    mid = _mid(bid, ask)
    spread_pct = _spread_pct(bid, ask, mid)
    quote_age_sec = _quote_age_seconds(last_quote_ts)

    # --- From alert context (our recommendation policy already stamped strike/expiry) ---
    expiry_iso = alert.get("expiry")
    dte = _dte(expiry_iso)
    side = (alert.get("side") or "").upper()
    ul_px = _safe_float(alert.get("underlying_price_from_alert"))

    # --- Simple IV rank placeholder (0..1) if not provided elsewhere ---
    # If you later compute IV rank historically, pass it through snapshot/features and override here.
    iv_rank = snapshot.get("iv_rank")
    iv_rank = _safe_float(iv_rank)
    if iv_rank is None and iv is not None:
        # Crude proxy: clamp IV to [0.05, 1.00] and normalize
        iv_rank = max(0.0, min(1.0, (iv - 0.05) / (1.00 - 0.05)))

    # --- RV20 placeholder (set None unless you fetch history) ---
    rv20 = _safe_float(snapshot.get("rv20"))

    # --- Expected move vs breakeven sanity (very rough heuristic) ---
    em_pct = _expected_move_pct(iv, dte)
    # Use mid as premium proxy; breakeven distance ≈ mid/ul (calls) or mid/ul (puts)
    be_pct = None
    if ul_px and mid:
        be_pct = mid / ul_px
    em_vs_be_ok = None
    if em_pct is not None and be_pct is not None:
        # If expected 1σ move covers breakeven within ~2 weeks, call it "ok"
        em_vs_be_ok = em_pct >= (be_pct * 0.8)

    # --- Liquidity sanity checks / regime placeholders ---
    # If you compute a real regime elsewhere, pass it through snapshot and override here.
    regime_flag = snapshot.get("regime") or "trending"

    # Levels/VWAP placeholders (routes.py only displays them; leave None if you don't compute them elsewhere)
    prev_day_high = _safe_float(snapshot.get("prev_day_high"))
    prev_day_low  = _safe_float(snapshot.get("prev_day_low"))
    premarket_high = _safe_float(snapshot.get("premarket_high"))
    premarket_low  = _safe_float(snapshot.get("premarket_low"))
    vwap = _safe_float(snapshot.get("vwap"))
    vwap_dist = None
    if vwap is not None and ul_px is not None and vwap > 0:
        vwap_dist = (ul_px - vwap) / vwap

    above_pdh = (ul_px is not None and prev_day_high is not None and ul_px > prev_day_high) or None
    below_pdl = (ul_px is not None and prev_day_low  is not None and ul_px < prev_day_low) or None
    above_pmh = (ul_px is not None and premarket_high is not None and ul_px > premarket_high) or None
    below_pml = (ul_px is not None and premarket_low  is not None and ul_px < premarket_low) or None

    # MTF alignment & S/R headroom placeholders (replace with your real logic if available)
    mtf_align = snapshot.get("mtf_align")
    if mtf_align is None:
        mtf_align = bool(above_pdh) if side == "CALL" else bool(below_pdl) if side == "PUT" else None

    sr_headroom_ok = snapshot.get("sr_headroom_ok")
    if sr_headroom_ok is None:
        # crude: require spread sane and OI/Vol not terrible
        sr_headroom_ok = (spread_pct is not None and spread_pct <= 0.25) and \
                         ((oi or 0) >= 100 or (vol or 0) >= 100)

    # Return the normalized feature dict
    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "option_spread_pct": spread_pct,
        "quote_age_sec": quote_age_sec,

        "oi": oi,
        "vol": vol,

        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,

        "iv": iv,
        "iv_rank": iv_rank,
        "rv20": rv20,

        "dte": dte,
        "em_vs_be_ok": em_vs_be_ok,

        "mtf_align": mtf_align,
        "sr_headroom_ok": sr_headroom_ok,
        "regime_flag": regime_flag,

        "prev_day_high": prev_day_high,
        "prev_day_low": prev_day_low,
        "premarket_high": premarket_high,
        "premarket_low": premarket_low,

        "vwap": vwap,
        "vwap_dist": vwap_dist,

        "above_pdh": above_pdh,
        "below_pdl": below_pdl,
        "above_pmh": above_pmh,
        "below_pml": below_pml,
    }
