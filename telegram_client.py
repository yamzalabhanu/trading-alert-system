# telegram_client.py
import httpx
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_THREAD_ID

async def send_telegram(text: str) -> Optional[Dict[str, Any]]:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    if TELEGRAM_THREAD_ID:
        payload["message_thread_id"] = int(TELEGRAM_THREAD_ID)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        with contextlib.suppress(Exception):
            return r.json()
        return {"status_code": r.status_code, "text": r.text}