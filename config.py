# config.py
import os
from typing import List, Tuple, Union
from zoneinfo import ZoneInfo
from datetime import time as dt_time, datetime

# Timezones
CDT_TZ = ZoneInfo("America/Chicago")
MARKET_TZ = ZoneInfo(os.getenv("MARKET_TZ", "America/New_York"))

# Model / API keys
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Limits / knobs
MAX_LLM_PER_DAY = int(os.getenv("MAX_LLM_PER_DAY", "40"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "600"))
REPORT_HHMM = os.getenv("REPORT_HHMM", "16:15")

# Strategy config
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

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_THREAD_ID = os.getenv("TELEGRAM_THREAD_ID")

# Processing windows in CDT
# You may also set these as 4-int tuples like [(8,30,11,30), (14,0,15,0)]
WINDOWS_CDT: List[Union[Tuple[dt_time, dt_time], Tuple[int, int, int, int]]] = [
    (dt_time(8, 30, tzinfo=CDT_TZ), dt_time(11, 30, tzinfo=CDT_TZ)),
    (dt_time(14, 0, tzinfo=CDT_TZ), dt_time(15, 0, tzinfo=CDT_TZ)),
]

def allowed_now_cdt() -> bool:
    """
    Accepts WINDOWS_CDT as either:
      - [(dt_time(..., tzinfo=CDT_TZ), dt_time(..., tzinfo=CDT_TZ)), ...]  OR
      - [(8,30,11,30), (14,0,15,0)]  OR
      - [("08:30:00","11:30:00"), ...]
    """
    now = datetime.now(CDT_TZ).time()

    def to_time_pair(win):
        # (time, time)
        if isinstance(win, (tuple, list)) and len(win) == 2 and \
           isinstance(win[0], dt_time) and isinstance(win[1], dt_time):
            return win[0], win[1]
        # (sh, sm, eh, em)
        if isinstance(win, (tuple, list)) and len(win) == 4 and all(isinstance(x, int) for x in win):
            return dt_time(win[0], win[1], tzinfo=CDT_TZ), dt_time(win[2], win[3], tzinfo=CDT_TZ)
        # ("HH:MM[:SS]", "HH:MM[:SS]")
        if isinstance(win, (tuple, list)) and len(win) == 2 and all(isinstance(x, str) for x in win):
            try:
                s = dt_time.fromisoformat(win[0])
                e = dt_time.fromisoformat(win[1])
                if s.tzinfo is None: s = s.replace(tzinfo=CDT_TZ)
                if e.tzinfo is None: e = e.replace(tzinfo=CDT_TZ)
                return s, e
            except Exception:
                return None
        return None

    for win in WINDOWS_CDT:
        pair = to_time_pair(win)
        if not pair:
            continue
        start, end = pair
        if start <= now <= end:
            return True
    return False
