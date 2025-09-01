# config.py
import os
from zoneinfo import ZoneInfo

from datetime import time as dt_time


CDT_TZ = ZoneInfo("America/Chicago")

# All configuration constants from the original file
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_LLM_PER_DAY = int(os.getenv("MAX_LLM_PER_DAY", "20"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "600"))
MARKET_TZ = ZoneInfo(os.getenv("MARKET_TZ", "America/New_York"))
REPORT_HHMM = os.getenv("REPORT_HHMM", "16:15")
TRADE_STYLE = os.getenv("TRADE_STYLE", "intraday").lower()
DELTA_MIN_INTRADAY = float(os.getenv("DELTA_MIN_INTRADAY", "0.35"))
DELTA_MAX_INTRADAY = float(os.getenv("DELTA_MAX_INTRADAY", "0.55"))
DTE_MIN_INTRADAY = int(os.getenv("DTE_MIN_INTRADAY", "3"))
DTE_MAX_INTRADAY = int(os.getenv("DTE_MAX_INTRADAY", "10"))
DELTA_MIN_SWING = float(os.getenv("DELTA_MIN_SWING", "0.25"))
DELTA_MAX_SWING = float(os.getenv("DELTA_MAX_SWING", "0.45"))
DTE_MIN_SWING = int(os.getenv("DTE_MIN_SWING", "7"))
DTE_MAX_SWING = int(os.getenv("DTE_MAX_SWING", "21"))
EM_VS_BE_RATIO_MIN = float(os.getenv("EM_VS_BE_RATIO_MIN", "0.80"))
HEADROOM_MIN_R = float(os.getenv("HEADROOM_MIN_R", "1.0"))
SPY_ATR_PCT_DAYS = int(os.getenv("SPY_ATR_PCT_DAYS", "14"))
REGIME_TREND_EMA_FAST = int(os.getenv("REGIME_TREND_EMA_FAST", "9"))
REGIME_TREND_EMA_SLOW = int(os.getenv("REGIME_TREND_EMA_SLOW", "21"))
IV_HISTORY_LEN = int(os.getenv("IV_HISTORY_LEN", "120"))
MACRO_EVENT_DATES = [d.strip() for d in os.getenv("MACRO_EVENT_DATES", "").split(",") if d.strip()]
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")

# Windows in CDT
CDT_TZ = ZoneInfo("America/Chicago")
WINDOWS_CDT = [
    (dt_time(8, 30, tzinfo=CDT_TZ), dt_time(11, 30, tzinfo=CDT_TZ)),
    (dt_time(14, 0, tzinfo=CDT_TZ), dt_time(15, 0, tzinfo=CDT_TZ)),
]

def allowed_now_cdt() -> bool:
    """
    Accepts WINDOWS_CDT as either:
      - [(dt_time(..., tzinfo=CDT_TZ), dt_time(..., tzinfo=CDT_TZ)), ...]  OR
      - [(8,30,11,30), (14,0,15,0)]
    """
    now = market_now().time()  # aware time in CDT
    for win in WINDOWS_CDT:
        # Case 1: tuple/list of two time objects
        if isinstance(win, (tuple, list)) and len(win) == 2 \
           and isinstance(win[0], dt_time) and isinstance(win[1], dt_time):
            start, end = win
            if start <= now <= end:
                return True

        # Case 2: 4-tuple of ints
        elif isinstance(win, (tuple, list)) and len(win) == 4:
            sh, sm, eh, em = win
            if dt_time(sh, sm, tzinfo=CDT_TZ) <= now <= dt_time(eh, em, tzinfo=CDT_TZ):
                return True

        # Anything else: ignore gracefully
    return FalseTZ) <= now <= dt_time(eh, em, tzinfo=CDT_TZ):
                return True
    return False
