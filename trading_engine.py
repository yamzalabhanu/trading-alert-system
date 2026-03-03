"""
Thin facade exposing runtime + core logic helpers.

This module is imported by routes.py as `import trading_engine as engine`.

Key compatibility shims added:
- process_webhook(payload, flags=None): enqueues a job in the shape expected by your worker
- enqueue / enqueue_job / enqueue_alert / submit: aliases to enqueue_webhook_job
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engine_runtime import (
    startup,
    shutdown,
    enqueue_webhook_job,
    get_worker_stats,
    get_http_client,
)

from engine_logic import (
    market_now,
    llm_quota_snapshot,
    get_alert_text_from_request,
    parse_alert_text,
    preflight_ok,
    compose_telegram_text,
    net_debug_info,
)


# ---------------------------------------------------------------------
# Compatibility shims for older/newer routes.py versions
# ---------------------------------------------------------------------
async def process_webhook(payload: Dict[str, Any], flags: Optional[Dict[str, Any]] = None) -> None:
    """
    Back-compat entry point used by some routes.py versions:
        await engine.process_webhook(payload)

    We normalize into the "job" envelope consumed by your worker:
        {"alert_text": <payload>, "flags": {...}}
    """
    job = {"alert_text": payload, "flags": (flags or {})}
    await enqueue_webhook_job(job)


# Common enqueue aliases (so routes.py can call any of these)
async def enqueue(job: Dict[str, Any]) -> None:
    await enqueue_webhook_job(job)


async def enqueue_job(job: Dict[str, Any]) -> None:
    await enqueue_webhook_job(job)


async def enqueue_alert(job: Dict[str, Any]) -> None:
    await enqueue_webhook_job(job)


async def submit(job: Dict[str, Any]) -> None:
    await enqueue_webhook_job(job)


__all__ = [
    # runtime
    "startup",
    "shutdown",
    "enqueue_webhook_job",
    "enqueue",
    "enqueue_job",
    "enqueue_alert",
    "submit",
    "process_webhook",
    "get_worker_stats",
    "get_http_client",
    # logic helpers
    "market_now",
    "llm_quota_snapshot",
    "get_alert_text_from_request",
    "parse_alert_text",
    "preflight_ok",
    "compose_telegram_text",
    "net_debug_info",
]
