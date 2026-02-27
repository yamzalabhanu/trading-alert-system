# routes.py
import os
import asyncio
import logging
import json
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, date as dt_date
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import CDT_TZ, MARKET_TZ
import trading_engine as engine

from volume_scanner import run_scanner_loop, get_state as uvscan_state
from daily_reporter import build_daily_report, send_daily_report_to_telegram


# ---------------- Env helpers ----------------
def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


# ---------------- Logging ----------------
log = logging.getLogger("trading_engine.routes")

# ✅ Prevent double logging via root logger handlers
log.propagate = False

# ✅ Add handler only once for this logger
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)

log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()

# ---------------- Background tasks ----------------
_scanner_task: Optional[asyncio.Task] = None
_uvscan_controller_task: Optional[asyncio.Task] = None
_daily_task: Optional[asyncio.Task] = None


# ---------------- Daily report scheduler knobs ----------------
ENABLE_DAILY_REPORT_SCHED = _truthy(os.getenv("ENABLE_DAILY_REPORT_SCHED", "1"))
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "14"))       # 14:xx CDT default
DAILY_REPORT_MINUTE = int(os.getenv("DAILY_REPORT_MINUTE", "45"))   # 14:45 CDT default
SEND_EMPTY_REPORT = _truthy(os.getenv("DR_SEND_EMPTY", "1"))


# ---------------- UV-scan window gating (CDT) ----------------
def _in_uvscan_window_cdt(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(CDT_TZ)
    if os.getenv("UVSCAN_WEEKDAYS_ONLY", "1") == "1" and now.weekday() > 4:
        return False
    minutes = now.hour * 60 + now.minute
    return (10 * 60 + 30) <= minutes < (14 * 60)


async def _stop_uvscan_task():
    global _scanner_task
    if _scanner_task:
        _scanner_task.cancel()
        try:
            await _scanner_task
        except Exception:
            pass
        _scanner_task = None
    log.info("[routes] uv-scan task stopped")


def _start_uvscan_task():
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        return
    try:
        _scanner_task = asyncio.create_task(run_scanner_loop(), name="uv-scan")
    except TypeError:
        _scanner_task = asyncio.create_task(run_scanner_loop())
    log.info("[routes] uv-scan task started")


async def _uvscan_window_controller_loop():
    enforce = os.getenv("UVSCAN_ENFORCE_WINDOW", "1") == "1"
    poll = int(os.getenv("UVSCAN_WINDOW_POLL_SECONDS", "30"))
    if not enforce:
        log.info("[routes] uv-scan window enforcement disabled")
    while True:
        try:
            if enforce:
                in_window = _in_uvscan_window_cdt()
                running = bool(_scanner_task and not _scanner_task.done())
                if in_window and not running:
                    _start_uvscan_task()
                elif not in_window and running:
                    await _stop_uvscan_task()
            await asyncio.sleep(poll)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("[routes] uv-scan controller loop error: %r", e)
            await asyncio.sleep(5)


# ---------------- Daily report scheduler ----------------
def _next_report_run_from(now_cdt: datetime) -> datetime:
    tgt = now_cdt.replace(hour=DAILY_REPORT_HOUR, minute=DAILY_REPORT_MINUTE, second=0, microsecond=0)
    if now_cdt >= tgt:
        tgt = tgt + timedelta(days=1)
    while tgt.weekday() >= 5:
        tgt += timedelta(days=1)
    return tgt


async def _daily_report_dispatcher_loop():
    log.info("[reports] scheduler loop started (enabled=%s)", ENABLE_DAILY_REPORT_SCHED)
    while True:
        try:
            now_cdt = datetime.now(CDT_TZ)
            next_run = _next_report_run_from(now_cdt)
            sleep_s = max(1, int((next_run - now_cdt).total_seconds()))
            log.info("[reports] next daily report at %s CDT (in %ss)", next_run.strftime("%Y-%m-%d %H:%M:%S"), sleep_s)
            await asyncio.sleep(sleep_s)

            report_date = next_run.date()
            try:
                res = await send_daily_report_to_telegram(report_date)
                ok = bool(res.get("ok"))
                count = int(res.get("count", 0)) if isinstance(res.get("count", 0), (int, float)) else 0
                if not ok:
                    log.warning("[reports] send failed for %s: %s", report_date, res)
                elif count == 0 and not SEND_EMPTY_REPORT:
                    log.info("[reports] no rows for %s (DR_SEND_EMPTY=0) — skipped", report_date)
                else:
                    log.info("[reports] sent for %s: %s", report_date, res)
            except Exception as e:
                log.exception("[reports] exception while sending for %s: %r", report_date, e)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("[reports] scheduler loop error: %r", e)
            await asyncio.sleep(5)


# ---------------- App lifecycle ----------------
def bind_lifecycle(app: FastAPI):
    @app.on_event("startup")
    async def _startup():
        global _uvscan_controller_task, _daily_task
        await engine.startup()

        if _uvscan_controller_task is None or _uvscan_controller_task.done():
            try:
                _uvscan_controller_task = asyncio.create_task(_uvscan_window_controller_loop(), name="uv-scan-ctl")
            except TypeError:
                _uvscan_controller_task = asyncio.create_task(_uvscan_window_controller_loop())
            log.info("[routes] uv-scan controller started")

        if ENABLE_DAILY_REPORT_SCHED and (_daily_task is None or _daily_task.done()):
            try:
                _daily_task = asyncio.create_task(_daily_report_dispatcher_loop(), name="daily-report")
            except TypeError:
                _daily_task = asyncio.create_task(_daily_report_dispatcher_loop())
            log.info("[routes] daily-report scheduler started")

    @app.on_event("shutdown")
    async def _shutdown():
        global _uvscan_controller_task, _daily_task

        if _uvscan_controller_task:
            _uvscan_controller_task.cancel()
            try:
                await _uvscan_controller_task
            except Exception:
                pass
            _uvscan_controller_task = None
            log.info("[routes] uv-scan controller stopped")

        await _stop_uvscan_task()

        if _daily_task:
            _daily_task.cancel()
            try:
                await _daily_task
            except Exception:
                pass
            _daily_task = None
            log.info("[routes] daily-report scheduler stopped")

        await engine.shutdown()

    return app


def mount(app: FastAPI):
    return bind_lifecycle(app)


# ============================================================================
# WEBHOOK QUALITY GATES (NEW)
# ============================================================================

# 1) RTH-only windows (MARKET_TZ, default New York)
ENFORCE_RTH_ONLY = _truthy(os.getenv("ENFORCE_RTH_ONLY", "1"))
RTH1 = os.getenv("RTH_WINDOW_1", "09:35-11:30")
RTH2 = os.getenv("RTH_WINDOW_2", "13:30-15:45")
RTH_WEEKDAYS_ONLY = _truthy(os.getenv("RTH_WEEKDAYS_ONLY", "1"))

# 2) Event filtering (reduce spam)
ALLOW_EVENTS = {e.strip() for e in os.getenv("ALLOW_EVENTS", "entry").split(",") if e.strip()}
ALLOW_PRE_ENTRY = _truthy(os.getenv("ALLOW_PRE_ENTRY", "0"))
DROP_MANAGEMENT_EVENTS = _truthy(os.getenv("DROP_MANAGEMENT_EVENTS", "1"))

# 3) Allowlist tickers
ALLOWED_TICKERS = {t.strip().upper() for t in os.getenv("ALLOWED_TICKERS", "").split(",") if t.strip()}

# 4) Webhook cooldown (extra anti-spam)
WEBHOOK_COOLDOWN_SECONDS = int(os.getenv("WEBHOOK_COOLDOWN_SECONDS", "240"))

# 5) Top-N per day intake limiter
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))

# in-memory state (per process)
_state_day: Optional[dt_date] = None
_day_count: int = 0
_last_seen: Dict[str, datetime] = {}


def _parse_hhmm_range(s: str) -> Tuple[int, int, int, int]:
    # "HH:MM-HH:MM" -> (sh, sm, eh, em)
    a, b = s.split("-")
    sh, sm = [int(x) for x in a.split(":")]
    eh, em = [int(x) for x in b.split(":")]
    return sh, sm, eh, em


_R1 = _parse_hhmm_range(RTH1)
_R2 = _parse_hhmm_range(RTH2)


def _in_rth_window_market(now: Optional[datetime] = None) -> bool:
    if not ENFORCE_RTH_ONLY:
        return True
    now = now or datetime.now(MARKET_TZ)
    if RTH_WEEKDAYS_ONLY and now.weekday() > 4:
        return False
    m = now.hour * 60 + now.minute

    def in_win(r):
        sh, sm, eh, em = r
        return (sh * 60 + sm) <= m < (eh * 60 + em)

    return in_win(_R1) or in_win(_R2)


def _reset_day_if_needed(now_market: Optional[datetime] = None) -> None:
    global _state_day, _day_count, _last_seen
    now_market = now_market or datetime.now(MARKET_TZ)
    d = now_market.date()
    if _state_day != d:
        _state_day = d
        _day_count = 0
        _last_seen = {}


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _score_payload(p: Dict[str, Any]) -> float:
    """
    Lightweight priority score (0..100) based on Pine JSON fields:
    - higher adx, higher relVol
    - chop=false gets a boost
    - entry > pre_entry
    """
    adx = _safe_float(p.get("adx"))
    rv = _safe_float(p.get("relVol") if p.get("relVol") is not None else p.get("relvol"))
    chop = p.get("chop")
    event = str(p.get("event") or "").lower()

    score = 0.0
    if adx is not None:
        score += max(0.0, min(1.0, (adx - 18.0) / 20.0)) * 45.0  # ~18..38 -> 0..1
    if rv is not None:
        score += max(0.0, min(1.0, (rv - 1.2) / 1.5)) * 45.0     # ~1.2..2.7 -> 0..1
    if chop is False:
        score += 10.0
    if event == "entry":
        score += 5.0
    return float(score)


def _fingerprint(p: Dict[str, Any]) -> str:
    # cooldown key: ticker|side|event
    t = str(p.get("symbol") or p.get("ticker") or "").upper()
    s = str(p.get("side") or "").upper()
    e = str(p.get("event") or "").lower()
    return f"{t}|{s}|{e}"


# ---------------- Health / status ----------------
@router.get("/healthz")
async def healthz():
    _reset_day_if_needed()
    return {
        "ok": True,
        "workers": engine.get_worker_stats(),
        "llm_quota": engine.llm_quota_snapshot(),
        "uvscan_running": bool(_scanner_task and not _scanner_task.done()),
        "uvscan_controller_running": bool(_uvscan_controller_task and not _uvscan_controller_task.done()),
        "uvscan_in_window_now_cdt": _in_uvscan_window_cdt(),
        "daily_sched_running": bool(_daily_task and not _daily_task.done()),
        "next_daily_cdt": _next_report_run_from(datetime.now(CDT_TZ)).isoformat() if ENABLE_DAILY_REPORT_SCHED else None,
        "rth_only": ENFORCE_RTH_ONLY,
        "in_rth_now_market_tz": _in_rth_window_market(),
        "market_tz": str(MARKET_TZ),
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "day_count": _day_count,
        "allowed_events": sorted(list(ALLOW_EVENTS | ({"pre_entry"} if ALLOW_PRE_ENTRY else set()))),
    }


@router.get("/engine/stats")
async def engine_stats():
    return {"workers": engine.get_worker_stats()}


@router.get("/engine/quota")
async def engine_quota():
    return engine.llm_quota_snapshot()


@router.get("/alerts/window")
async def alerts_window_status():
    now_m = datetime.now(MARKET_TZ)
    return {
        "market_tz": str(MARKET_TZ),
        "now_market": now_m.isoformat(),
        "enforce_rth_only": ENFORCE_RTH_ONLY,
        "rth_windows": [RTH1, RTH2],
        "in_rth_now": _in_rth_window_market(now_m),
        "allowed_tickers_count": len(ALLOWED_TICKERS),
        "max_trades_per_day": MAX_TRADES_PER_DAY,
        "day_count": _day_count,
    }


# ---------------- Webhook (TradingView) ----------------
@router.post("/webhook")
async def webhook(
    request: Request,
    force: bool = False,
    qty: int = 1,
    bypass_window: bool = False,   # manual testing
):
    # ✅ Fix UnboundLocalError: we mutate _day_count in this function
    global _day_count, _last_seen

    _reset_day_if_needed()
    log.info("webhook received; force=%s qty=%s bypass=%s", force, qty, bypass_window)

    # RTH gating (market tz)
    if not bypass_window and not _in_rth_window_market():
        return JSONResponse(
            {
                "queued": False,
                "reason": "outside_rth_window",
                "market_tz": str(MARKET_TZ),
                "rth_windows": [RTH1, RTH2],
                "now_market": datetime.now(MARKET_TZ).isoformat(),
            },
            status_code=200,
        )

    # Get raw text (supports application/json and text/plain)
    text = await engine.get_alert_text_from_request(request)
    if not text:
        raise HTTPException(status_code=400, detail="Empty alert payload")

    # Parse into normalized dict (handles your Pine JSON as a string, plus plaintext formats)
    parsed = engine.parse_alert_text(text)

    # Pull key fields (support both schemas)
    ticker = str(parsed.get("symbol") or parsed.get("ticker") or "").upper().strip()
    side = str(parsed.get("side") or "").upper().strip()
    event = str(parsed.get("event") or "").lower().strip()

    # Enforce allowlist (if configured)
    if ALLOWED_TICKERS and ticker not in ALLOWED_TICKERS:
        return JSONResponse(
            {"queued": False, "reason": "ticker_not_allowed", "ticker": ticker},
            status_code=200,
        )

    # Filter events to reduce noise
    if not event:
        # If no event in plaintext alerts, treat as "entry" (backward compatible)
        event = "entry"
        parsed["event"] = "entry"

    if event == "pre_entry" and not ALLOW_PRE_ENTRY:
        return JSONResponse({"queued": False, "reason": "pre_entry_disabled"}, status_code=200)

    mgmt_events = {"tp1", "tp2", "tp3", "exit", "exit_fast_1m", "trail", "time", "opposite", "fast_stop"}
    if DROP_MANAGEMENT_EVENTS and event in mgmt_events:
        return JSONResponse({"queued": False, "reason": "management_event_dropped", "event": event}, status_code=200)

    allowed = set(ALLOW_EVENTS)
    if ALLOW_PRE_ENTRY:
        allowed.add("pre_entry")
    if event not in allowed:
        return JSONResponse({"queued": False, "reason": "event_not_allowed", "event": event}, status_code=200)

    # Webhook cooldown per (ticker, side, event)
    fp = _fingerprint({"symbol": ticker, "side": side, "event": event})
    now = datetime.now(MARKET_TZ)
    last = _last_seen.get(fp)
    if last and (now - last).total_seconds() < WEBHOOK_COOLDOWN_SECONDS:
        return JSONResponse(
            {"queued": False, "reason": "webhook_cooldown", "cooldown_s": WEBHOOK_COOLDOWN_SECONDS},
            status_code=200,
        )

    # Daily max trades guard (Top 5/day)
    if _day_count >= MAX_TRADES_PER_DAY:
        return JSONResponse(
            {
                "queued": False,
                "reason": "daily_limit_reached",
                "max_trades_per_day": MAX_TRADES_PER_DAY,
                "day_count": _day_count,
            },
            status_code=200,
        )

    # Priority score (optional; included in flags for engine / telegram)
    score = _score_payload(parsed)

    ok = engine.enqueue_webhook_job(
        alert_text=text,
        flags={
            "qty": int(qty),
            "force_buy": bool(force),
            "event": event,
            "ticker": ticker,
            "side": side,
            "priority_score": score,
        },
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Queue is full")

    # ✅ Mutations now safe (global declared)
    _day_count += 1
    _last_seen[fp] = now

    return JSONResponse(
        {
            "queued": True,
            "priority_score": score,
            "day_count": _day_count,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "queue_stats": engine.get_worker_stats(),
            "llm_quota": engine.llm_quota_snapshot(),
        }
    )


@router.post("/webhook/tradingview")
async def webhook_tradingview(
    request: Request,
    force: bool = False,
    qty: int = 1,
    bypass_window: bool = False,
):
    return await webhook(request=request, force=force, qty=qty, bypass_window=bypass_window)


# ---------------- uv-scan control & status ----------------
@router.get("/uvscan/status")
async def uvscan_status():
    running = bool(_scanner_task and not _scanner_task.done())
    ctl_running = bool(_uvscan_controller_task and not _uvscan_controller_task.done())
    in_window = _in_uvscan_window_cdt()
    return {
        "running": running,
        "controller_running": ctl_running,
        "in_window_now_cdt": in_window,
        "window_cdt": "10:30–14:00",
        "state": uvscan_state(),
    }


@router.post("/uvscan/start")
async def uvscan_start():
    if not _in_uvscan_window_cdt() and os.getenv("UVSCAN_ENFORCE_WINDOW", "1") == "1":
        return {"started": False, "reason": "outside_uvscan_window", "window_cdt": "10:30–14:00"}
    _start_uvscan_task()
    return {"started": True}


@router.post("/uvscan/stop")
async def uvscan_stop():
    await _stop_uvscan_task()
    return {"stopped": True}


# ---------------- Diagnostics ----------------
@router.get("/net/debug")
async def net_debug():
    return await engine.net_debug_info()


# ---------------- Daily EOD Report ----------------
@router.get("/reports/daily")
async def get_daily_report(date: Optional[str] = None):
    if date:
        try:
            y, m, d = [int(x) for x in date.split("-")]
            target = dt_date(y, m, d)
        except Exception:
            raise HTTPException(400, "Invalid date; use YYYY-MM-DD")
    else:
        target = None
    rep = await build_daily_report(target)
    return rep


@router.post("/reports/daily/send")
async def send_daily_report(date: Optional[str] = None):
    if date:
        try:
            y, m, d = [int(x) for x in date.split("-")]
            target = dt_date(y, m, d)
        except Exception:
            raise HTTPException(400, "Invalid date; use YYYY-MM-DD")
    else:
        target = None
    res = await send_daily_report_to_telegram(target)
    if not res.get("ok"):
        raise HTTPException(500, res.get("error", "send failed"))
    return res


@router.post("/reports/daily/test_send")
async def reports_test_send():
    target = datetime.now(CDT_TZ).date()
    return await send_daily_report_to_telegram(target)


@router.get("/reports/daily/next")
async def reports_next():
    return {
        "enabled": ENABLE_DAILY_REPORT_SCHED,
        "next_cdt": _next_report_run_from(datetime.now(CDT_TZ)).isoformat() if ENABLE_DAILY_REPORT_SCHED else None,
        "running": bool(_daily_task and not _daily_task.done()),
    }
