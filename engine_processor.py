# engine_processor.py
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from engine_runtime import get_http_client
from engine_common import (
    market_now,
    consume_llm,
    parse_alert_text,
    preflight_ok,
    compose_telegram_text,
)
from llm_client import analyze_with_openai
from telegram_client import send_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from scoring import compute_decision_score, map_score_to_rating
from daily_reporter import log_alert_snapshot

try:
    from reporting import _DECISIONS_LOG
except Exception:
    _DECISIONS_LOG = []

logger = logging.getLogger("trading_engine")


async def process_tradingview_job(job: Dict[str, Any]) -> None:
    client = get_http_client()
    if client is None:
        logger.warning("[worker] HTTP client not ready")
        return

    try:
        alert = parse_alert_text(job["alert_text"])
    except Exception as e:
        logger.warning("[worker] bad alert payload: %s", e)
        return

    expiry = alert.get("expiry")
    dte = None
    if expiry:
        try:
            dte = (datetime.fromisoformat(expiry).date() - datetime.now(timezone.utc).date()).days
        except Exception:
            dte = None

    # Lightweight, provider-free feature set
    f: Dict[str, Any] = {
        "dte": dte,
        "last": alert.get("underlying_price_from_alert"),
        "mid": alert.get("underlying_price_from_alert"),
        "option_spread_pct": None,
        "quote_age_sec": 0,
        "vol": None,
        "oi": None,
        "em_vs_be_ok": True,
        "mtf_align": True,
        "sr_headroom_ok": True,
        "nbbo_provider": "disabled",
    }

    pf_ok, _ = preflight_ok(f)

    try:
        llm = await analyze_with_openai(alert, f)
        consume_llm()
    except Exception as e:
        llm = {"decision": "wait", "confidence": 0.0, "reason": f"LLM error: {e}"}

    decision_final = "buy" if llm.get("decision") == "buy" else ("skip" if llm.get("decision") == "skip" else "wait")

    try:
        score = compute_decision_score(f, llm)
        rating = map_score_to_rating(score, llm.get("decision"))
    except Exception:
        score, rating = None, None

    try:
        tg_text = compose_telegram_text(
            alert=alert,
            option_ticker=None,
            f=f,
            llm=llm,
            llm_ran=True,
            llm_reason="",
            score=score,
            rating=rating,
            diff_note="⚙️ External broker/data integrations disabled (IBKR/Polygon/Perplexity removed).",
        )
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await send_telegram(tg_text)
    except Exception as e:
        logger.exception("[worker] Telegram error: %s", e)

    try:
        log_alert_snapshot(alert, None, f)
    except Exception as e:
        logger.warning("[daily-report] log snapshot failed: %r", e)

    try:
        _DECISIONS_LOG.append({
            "timestamp_local": market_now(),
            "symbol": alert.get("symbol"),
            "side": alert.get("side"),
            "decision_final": decision_final,
            "llm": llm,
            "features": f,
            "preflight_ok": pf_ok,
            "ibkr": {"enabled": False, "attempted": False, "result": None},
        })
    except Exception as e:
        logger.warning("decision log append failed: %r", e)


async def net_debug_info() -> Dict[str, Any]:
    return {"integrations": {"ibkr": "disabled", "polygon": "disabled", "perplexity": "disabled"}}


__all__ = ["process_tradingview_job", "net_debug_info"]
