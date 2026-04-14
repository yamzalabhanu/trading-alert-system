# trading_engine.py
from __future__ import annotations

from typing import Any, Dict
import json

# ---------------- Runtime facade (used by routes.py) ----------------
from engine_runtime import (
    startup,
    shutdown,
    enqueue_webhook_job,
    get_worker_stats,
    get_http_client,
)

# ---------------- Core constants + helpers ----------------
from engine_common import (
    POLYGON_API_KEY,
    IBKR_ENABLED,
    IBKR_DEFAULT_QTY,
    IBKR_TIF,
    IBKR_ORDER_MODE,
    IBKR_USE_MID_AS_LIMIT,
    TARGET_DELTA_CALL,
    TARGET_DELTA_PUT,
    MAX_SPREAD_PCT,
    MAX_QUOTE_AGE_S,
    MIN_VOL_TODAY,
    MIN_OI,
    MIN_DTE,
    MAX_DTE,
    SCAN_MIN_VOL_RTH,
    SCAN_MIN_OI_RTH,
    SCAN_MIN_VOL_AH,
    SCAN_MIN_OI_AH,
    SEND_CHAIN_SCAN_ALERTS,
    SEND_CHAIN_SCAN_TOPN_ALERTS,
    REPLACE_IF_NO_NBBO,
    market_now,
    llm_quota_snapshot,
    consume_llm,
    get_alert_text_from_request,
    parse_alert_text as _parse_alert_text_raw,  # <-- keep original for string parsing
    round_strike_to_common_increment,
    _next_friday,
    same_week_friday,
    two_weeks_friday,
    is_same_week,
    _encode_ticker_path,
    _is_rth_now,
    _occ_meta,
    _ticker_matches_side,
    preflight_ok,
    compose_telegram_text,
)

# Worker entrypoint + debug info
from engine_processor import process_tradingview_job, net_debug_info


# -------------------------------------------------------------------
# FIX: make parse_alert_text accept dict payloads (TV webhook JSON)
# -------------------------------------------------------------------
def parse_alert_text(payload: Any) -> Dict[str, Any]:
    """
    Accept:
      - dict: already parsed JSON -> return as-is
      - str/bytes: parse via engine_common.parse_alert_text (or json.loads fallback)
    """
    if payload is None:
        raise ValueError("empty payload")

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")

    # If engine_common's parser can handle strings, prefer it
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            raise ValueError("empty payload string")
        try:
            return _parse_alert_text_raw(s)  # engine_common behavior (JSON string, etc.)
        except Exception:
            # fallback: strict JSON object
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError("JSON payload must be an object")
            return obj

    # last resort: stringify then parse
    s2 = str(payload).strip()
    if not s2:
        raise ValueError("empty payload")
    obj2 = json.loads(s2)
    if not isinstance(obj2, dict):
        raise ValueError("JSON payload must be an object")
    return obj2


__all__ = [
    # runtime facade
    "startup",
    "shutdown",
    "enqueue_webhook_job",
    "get_worker_stats",
    "get_http_client",

    # constants/toggles
    "POLYGON_API_KEY",
    "IBKR_ENABLED",
    "IBKR_DEFAULT_QTY",
    "IBKR_TIF",
    "IBKR_ORDER_MODE",
    "IBKR_USE_MID_AS_LIMIT",
    "TARGET_DELTA_CALL",
    "TARGET_DELTA_PUT",
    "MAX_SPREAD_PCT",
    "MAX_QUOTE_AGE_S",
    "MIN_VOL_TODAY",
    "MIN_OI",
    "MIN_DTE",
    "MAX_DTE",
    "SCAN_MIN_VOL_RTH",
    "SCAN_MIN_OI_RTH",
    "SCAN_MIN_VOL_AH",
    "SCAN_MIN_OI_AH",
    "SEND_CHAIN_SCAN_ALERTS",
    "SEND_CHAIN_SCAN_TOPN_ALERTS",
    "REPLACE_IF_NO_NBBO",

    # helpers
    "market_now",
    "llm_quota_snapshot",
    "consume_llm",
    "get_alert_text_from_request",
    "parse_alert_text",
    "round_strike_to_common_increment",
    "_next_friday",
    "same_week_friday",
    "two_weeks_friday",
    "is_same_week",
    "_encode_ticker_path",
    "_is_rth_now",
    "_occ_meta",
    "_ticker_matches_side",
    "preflight_ok",
    "compose_telegram_text",

    # engine processor
    "process_tradingview_job",
    "net_debug_info",
]
