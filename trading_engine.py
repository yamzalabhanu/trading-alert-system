# trading_engine.py
"""
Thin façade preserving the original import surface.
"""

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

__all__ = [
    # lifecycle / runtime
    "startup", "shutdown", "enqueue_webhook_job", "get_worker_stats", "get_http_client",
    # logic / helpers
    "market_now", "llm_quota_snapshot", "get_alert_text_from_request", "parse_alert_text",
    "preflight_ok", "compose_telegram_text",
    # diagnostics
    "diag_polygon_bundle", "net_debug_info",
]
