# config.py
import os
from zoneinfo import ZoneInfo
from typing import List, Tuple, Union

# Timezones
CDT_TZ = ZoneInfo("America/Chicago")
MARKET_TZ = ZoneInfo(os.getenv("MARKET_TZ", "America/New_York"))

# Model / API keys
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Limits / knobs
MAX_LLM_PER_DAY = int(os.getenv("MAX_LLM_PER_DAY", "20"))
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
