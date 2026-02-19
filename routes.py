# routes.py
import os
import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta, date as dt_date

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import CDT_TZ

import trading_engine as engine
from volume_scanner import run_scanner_loop, get_state as uvscan_state
from daily_reporter import build_daily_report, send_daily_report_to_telegram


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on")


# Daily report scheduler knobs (overridable via env)
ENABLE_DAILY_REPORT_SCHED = _truthy(os.getenv("ENABLE_DAILY_REPORT_SCHED", "1"))
DAILY_REPORT_HOUR = int(os.getenv("DAILY_REPORT_HOUR", "14"))        # 14:xx = 2:xx pm CDT
DAILY_REPORT_MINUTE = int(os.getenv("DAILY_REPORT_MINUTE", "45"))    # default 2:45 pm CDT
SEND_EMPTY_REPORT = _truthy(os.getenv("DR_SEND_EMPTY", "1"))         # send even if no rows

# TradingView alert window knobs (CDT)
ALERT_ENFORCE_WINDOW = _truthy(os.getenv("ALERT_ENFORCE_WINDOW", "1"))
ALERT_WEEKDAYS_ONLY = _truthy(os.getenv("ALERT_WEEKDAYS_ONLY", "1"))
ALERT_MORNING_START = os.getenv("ALERT_WINDOW_MORNING_START", "08:30")
ALERT_MORNING_END = os.getenv("ALERT_WINDOW_MORNING_END", "10:30")
ALERT_PM_START = os.getenv("ALERT_WINDOW_PM_START", "14:00")
ALERT_PM_END = os.getenv("ALERT_WINDOW_PM_END", "15:00")


def _hhmm_to_minutes(s: str) -> int:
    hh, mm = s.split(":")
    return int(hh) * 60 + int(mm)


_ALERT_M_START_MIN = _hhmm_to_minutes(ALERT_MORNING_START)
_ALERT_M_END_MIN = _hhmm_to_minutes(ALERT_MORNING_END)
_ALERT_P_START_MIN = _hhmm_to_minutes(ALERT_PM_START)
_ALERT_P_END_MIN = _hhmm_to_minutes(ALERT_PM_END)


def _in_alert_window_cdt(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(CDT_TZ)
    if ALERT_WEEKDAYS_ONLY and now.weekday() > 4:
        return False
    minutes = now.hour * 60 + now.minute
    in_morning = _ALERT_M_START_MIN <= minutes < _ALERT_M_END_MIN
    in_afternoon = _ALERT_P_START_MIN <= minutes < _ALERT_P_END_MIN
    return in_morning or in_afternoon


log = logging.getLogger("trading_engine.routes")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    log.addHandler(_h)
log.setLevel(os.getenv("LOG_LEVEL", "INFO"))

router = APIRouter()

_scanner_task: Optional[asyncio.Task] = None
_uvscan_controller_task: Optional[asyncio.Task] = None
_daily_task: Optional[asyncio.Task] = None


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
            log.info("[reports] next daily report at %s CDT (in %ss)",
                     next_run.strftime("%Y-%m-%d %H:%M:%S"), sleep_s)
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


def mount(app: FastAPI):
    return bind_lifecycle(app)


@router.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "workers": engine.get_worker_stats(),
        "llm_quota": engine.llm_quota_snapshot(),
        "uvscan_running": bool(_scanner_task and not _scanner_task.done()),
        "uvscan_controller_running": bool(_uvscan_controller_task and not _uvscan_controller_task.done()),
        "uvscan_in_window_now_cdt": _in_uvscan_window_cdt(),
        "daily_sched_running": bool(_daily_task and not _daily_task.done()),
        "next_daily_cdt": _next_report_run_from(datetime.now(CDT_TZ)).isoformat() if ENABLE_DAILY_REPORT_SCHED else None,
        "alerts_in_window_now_cdt": _in_alert_window_cdt(),
        "alerts_window_cdt": f"{ALERT_MORNING_START}–{ALERT_MORNING_END}, {ALERT_PM_START}–{ALERT_PM_END}",
    }


@router.get("/engine/stats")
async def engine_stats():
    return {"workers": engine.get_worker_stats()}


@router.get("/engine/quota")
async def engine_quota():
    return engine.llm_quota_snapshot()


@router.get("/alerts/window")
async def alerts_window_status():
    now = datetime.now(CDT_TZ)
    return {
        "in_window_now_cdt": _in_alert_window_cdt(now),
        "now_cdt": now.isoformat(),
        "window_cdt": f"{ALERT_MORNING_START}–{ALERT_MORNING_END}, {ALERT_PM_START}–{ALERT_PM_END}",
        "weekdays_only": ALERT_WEEKDAYS_ONLY,
        "enforced": ALERT_ENFORCE_WINDOW,
    }


@router.post("/webhook")
async def webhook(
    request: Request,
    force: bool = False,
    qty: int = 1,
    bypass_window: bool = False,  # current name
    bypass: bool = False,         # ✅ compat: older/shorter name
):
    # ✅ treat either flag as bypass
    bypass_effective = bool(bypass_window or bypass)
    log.info("webhook received; force=%s qty=%s bypass=%s", force, qty, bypass_effective)

    if ALERT_ENFORCE_WINDOW and not bypass_effective and not _in_alert_window_cdt():
        return JSONResponse(
            {
                "queued": False,
                "reason": "outside_alert_window",
                "window_cdt": f"{ALERT_MORNING_START}–{ALERT_MORNING_END}, {ALERT_PM_START}–{ALERT_PM_END}",
                "now_cdt": datetime.now(CDT_TZ).isoformat(),
                "hint": "Use ?bypass_window=1 (or ?bypass=1) for manual testing",
            },
            status_code=200,
        )

    text = await engine.get_alert_text_from_request(request)
    if not text:
        raise HTTPException(status_code=400, detail="Empty alert payload")

    ok = engine.enqueue_webhook_job(
        alert_text=text,
        flags={"qty": int(qty), "force_buy": bool(force)},
    )
    if not ok:
        raise HTTPException(status_code=503, detail="Queue is full")

    return JSONResponse(
        {
            "queued": True,
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
    bypass: bool = False,  # ✅ compat
):
    return await webhook(request=request, force=force, qty=qty, bypass_window=bypass_window, bypass=bypass)


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


@router.get("/net/debug")
async def net_debug():
    return await engine.net_debug_info()


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
