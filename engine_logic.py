# engine_logic.py
# Temporary shim so existing imports keep working. Safe to delete once you update imports.
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
]
