# volume_scanner.py
import os
import asyncio
import logging
from statistics import median
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from engine_runtime import get_http_client
from engine_common import market_now, POLYGON_API_KEY
from telegram_client import send_telegram

logger = logging.getLogger("volume_scanner")

# -----------------------
# Env knobs (tweak freely)
# -----------------------
SCAN_TICKERS               = [s.strip().upper() for s in os.getenv("UV_SCAN_TICKERS", "").split(",") if s.strip()]
SCAN_INTERVAL_SEC          = int(os.getenv("UV_SCAN_INTERVAL_SEC", "300"))  # 5 minutes
MAX_CONCURRENCY            = int(os.getenv("UV_SCAN_MAX_CONCURRENCY", "3"))
# Stock spike definition
STOCK_VOL_LOOKBACK_MIN     = int(os.getenv("UV_STOCK_VOL_LOOKBACK_MIN", "30"))
STOCK_VOL_MULT             = float(os.getenv("UV_STOCK_VOL_MULT", "3.0"))
STOCK_VOL_MIN_ABS          = int(os.getenv("UV_STOCK_VOL_MIN_ABS", "150000"))  # last 1m volume must exceed this
# Options scanning
OPT_SNAPSHOT_LIMIT         = int(os.getenv("UV_OPT_SNAPSHOT_LIMIT", "120"))
OPT_TOPN_CHECK             = int(os.getenv("UV_OPT_TOPN_CHECK", "15"))  # fetch yday for top N only (to limit calls)
OPT_VOL_MULT               = float(os.getenv("UV_OPT_VOL_MULT", "5.0"))
OPT_VOL_MIN_ABS            = int(os.getenv("UV_OPT_VOL_MIN_ABS", "1000"))
OI_MIN_ABS                 = int(os.getenv("UV_OI_MIN_ABS", "5000"))
OI_ABS_DELTA_MIN           = int(os.getenv("UV_OI_ABS_DELTA_MIN", "2000"))
OI_MULT                    = float(os.getenv("UV_OI_MULT", "1.25"))
# API pacing
API_SLEEP_MS_BETWEEN_CALLS = int(os.getenv("UV_API_SLEEP_MS", "250"))  # 0.25s between calls to ease 429s


def _today_utc_bounds() -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)
    return start.isoformat(), now.isoformat()


def _yday_iso() -> str:
    return (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()


async def _poly_json(url: str, params: Dict[str, Any], timeout: float = 8.0) -> Dict[str, Any]:
    client = get_http_client()
    if client is None:
        raise RuntimeError("HTTP client not ready")
    params = dict(params or {})
    if POLYGON_API_KEY:
        params.setdefault("apiKey", POLYGON_API_KEY)
    try:
        r = await client.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        await asyncio.sleep(API_SLEEP_MS_BETWEEN_CALLS / 1000.0)
        return r.json()
    except Exception as e:
        logger.warning("[uv-scan] http error %s %s -> %r", url, params, e)
        return {}


# -----------------------
# STOCK: 1m volume spike
# -----------------------
async def _check_stock_intraday(symbol: str) -> Optional[Dict[str, Any]]:
    frm, to = _today_utc_bounds()
    data = await _poly_json(
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{frm}/{to}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    results = data.get("results") or []
    if len(results) < max(10, STOCK_VOL_LOOKBACK_MIN + 1):
        return None
    vols = [float(x.get("v") or 0) for x in results]
    last_v = vols[-1]
    baseline = median(vols[-(STOCK_VOL_LOOKBACK_MIN + 1):-1])
    if baseline <= 0:
        return None
    ratio = last_v / baseline
    if ratio >= STOCK_VOL_MULT and last_v >= STOCK_VOL_MIN_ABS:
        last_c = results[-1].get("c")
        return {
            "symbol": symbol,
            "last_vol": int(last_v),
            "baseline": int(baseline),
            "ratio": round(ratio, 2),
            "last_close": float(last_c) if last_c is not None else None,
            "bars": len(results),
            "kind": "stock_vol_spike",
        }
    return None


# -----------------------
# OPTIONS: volume & OI
# -----------------------
def _fmt_occ(underlying: str, exp: str, cp: str, strike: float) -> str:
    # Build OCC like O:WMT250919C00105000 (strike x1000, 5 digits)
    dt = datetime.fromisoformat(exp).date()
    yy = str(dt.year)[2:4]
    mm = f"{dt.month:02d}"
    dd = f"{dt.day:02d}"
    right = "C" if cp.upper().startswith("C") else "P"
    strike_int = int(round(float(strike) * 1000))
    return f"O:{underlying}{yy}{mm}{dd}{right}{strike_int:08d}"

async def _yday_open_close_occ(occ: str) -> Dict[str, Any]:
    yday = _yday_iso()
    return await _poly_json(
        f"https://api.polygon.io/v1/open-close/options/{occ}/{yday}",
        {},
        timeout=6.0,
    )

async def _check_options_unusual(underlying: str) -> List[Dict[str, Any]]:
    snap = await _poly_json(
        f"https://api.polygon.io/v3/snapshot/options/{underlying}",
        {"limit": OPT_SNAPSHOT_LIMIT, "greeks": "true", "order": "desc", "sort": "volume"},
    )
    res = []
    items = snap.get("results") or []
    if not items:
        return res

    # Consider top N by current volume and compare to yesterday
    top = items[:OPT_TOPN_CHECK]
    # Build tasks cautiously to respect API pacing
    findings: List[Dict[str, Any]] = []
    for it in top:
        try:
            details = it.get("details") or {}
            day = it.get("day") or {}
            cp = details.get("contract_type") or ""
            exp = details.get("expiration_date")
            strike = details.get("strike_price")
            tkr = details.get("ticker") or ""
            # Some datasets include the OCC in 'ticker'; if not, synthesize:
            occ = tkr or (_fmt_occ(underlying, exp, cp, strike) if (exp and strike) else None)
            if not occ:
                continue

            vol_today = int(day.get("volume") or 0)
            oi_today = int(day.get("open_interest") or it.get("open_interest") or 0)

            # Skip tiny noise
            if vol_today < OPT_VOL_MIN_ABS and oi_today < OI_MIN_ABS:
                continue

            yday = await _yday_open_close_occ(occ)
            y_vol = int(yday.get("volume") or 0)
            y_oi = int(yday.get("open_interest") or 0)

            vol_ratio = (vol_today / max(1, y_vol)) if vol_today else 0.0
            oi_ratio = (oi_today / max(1, y_oi)) if oi_today else 0.0
            oi_delta = oi_today - y_oi

            interesting = False
            flags = []
            if vol_today >= OPT_VOL_MIN_ABS and vol_ratio >= OPT_VOL_MULT:
                interesting = True
                flags.append(f"Vol {vol_today:,} ({vol_ratio:.1f}× yday)")
            if oi_today >= OI_MIN_ABS and (oi_ratio >= OI_MULT or oi_delta >= OI_ABS_DELTA_MIN):
                interesting = True
                flags.append(f"OI {oi_today:,} ({oi_ratio:.2f}×; +{oi_delta:,} vs yday)")

            if interesting:
                findings.append({
                    "underlying": underlying,
                    "occ": occ,
                    "contract_type": "CALL" if cp.upper().startswith("C") else "PUT",
                    "expiry": exp,
                    "strike": strike,
                    "vol_today": vol_today,
                    "vol_yday": y_vol,
                    "vol_ratio": round(vol_ratio, 2),
                    "oi_today": oi_today,
                    "oi_yday": y_oi,
                    "oi_ratio": round(oi_ratio, 2),
                    "oi_delta": oi_delta,
                    "flags": "; ".join(flags),
                    "kind": "options_unusual",
                })
        except Exception as e:
            logger.debug("[uv-scan] parse/single error for %s: %r", underlying, e)

    return findings


# -----------------------
# Telegram formatting
# -----------------------
def _fmt_pct(x: Optional[float], nd=1) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"

def _fmt_num(x: Any) -> str:
    try:
        xi = int(x)
        return f"{xi:,}"
    except Exception:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

def _compose_msg(symbol: str, stock_hit: Optional[Dict[str, Any]], opt_hits: List[Dict[str, Any]]) -> Optional[str]:
    lines: List[str] = []
    ts = market_now().strftime("%Y-%m-%d %H:%M")
    header = f"📊 Unusual Activity Scan — {symbol} ({ts})"
    lines.append(header)

    if stock_hit:
        lines.append(
            f"Stock: 1m vol spike {stock_hit['ratio']}× "
            f"(last={_fmt_num(stock_hit['last_vol'])} vs median{STOCK_VOL_LOOKBACK_MIN}={_fmt_num(stock_hit['baseline'])})"
            + (f"; px={_fmt_num(stock_hit['last_close'])}" if stock_hit.get("last_close") is not None else "")
        )

    if opt_hits:
        lines.append("Options (top hits):")
        for h in opt_hits[:8]:
            right = "C" if h["contract_type"] == "CALL" else "P"
            exp = (h.get("expiry") or "")[:10]
            strike = h.get("strike")
            lines.append(
                f"• {symbol} {strike}{right} {exp} — {h['flags']}"
            )

    if len(lines) == 1:
        return None
    return "\n".join(lines)


# -----------------------
# Runner
# -----------------------
async def scan_once_for_symbol(symbol: str) -> None:
    stock_hit = await _check_stock_intraday(symbol)
    opt_hits = await _check_options_unusual(symbol)
    msg = _compose_msg(symbol, stock_hit, opt_hits)
    if msg:
        try:
            await send_telegram(msg)
        except Exception as e:
            logger.warning("[uv-scan] telegram error: %r", e)

async def run_scanner_loop() -> None:
    if not POLYGON_API_KEY:
        logger.warning("[uv-scan] POLYGON_API_KEY missing; scanner disabled.")
        return
    if not SCAN_TICKERS:
        logger.info("[uv-scan] UV_SCAN_TICKERS empty; scanner idle.")
        return

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async def _guarded(sym: str):
        async with sem:
            await scan_once_for_symbol(sym)

    logger.info("[uv-scan] starting scanner; tickers=%s; interval=%ss", ",".join(SCAN_TICKERS), SCAN_INTERVAL_SEC)
    try:
        while True:
            tasks = [asyncio.create_task(_guarded(sym)) for sym in SCAN_TICKERS]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(SCAN_INTERVAL_SEC)
    except asyncio.CancelledError:
        logger.info("[uv-scan] scanner loop cancelled.")
        raise
    except Exception as e:
        logger.exception("[uv-scan] fatal scanner error: %r", e)
