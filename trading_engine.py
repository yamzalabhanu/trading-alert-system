"""Thin facade exposing runtime + core logic helpers."""

from engine_runtime import (
    startup, shutdown,
    enqueue_webhook_job, get_worker_stats,
    get_http_client,
)
from engine_logic import (
    market_now, llm_quota_snapshot,
    get_alert_text_from_request, parse_alert_text,
    preflight_ok, compose_telegram_text,
    net_debug_info,
)

__all__ = [
    "startup", "shutdown", "enqueue_webhook_job", "get_worker_stats", "get_http_client",
    "market_now", "llm_quota_snapshot", "get_alert_text_from_request", "parse_alert_text",
    "preflight_ok", "compose_telegram_text", "net_debug_info",
]
