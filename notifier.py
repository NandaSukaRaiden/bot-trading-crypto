"""
notifier.py
Kirim notifikasi ke Telegram (opsional).
"""
import aiohttp
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from rich.console import Console

console = Console()


async def send_telegram_notification(message: str) -> bool:
    """Kirim pesan ke Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       f"🤖 Crypto Futures Bot\n\n{message}",
        "parse_mode": "HTML",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return True
    except Exception as e:
        console.print(f"[dim]Telegram notif gagal: {e}[/dim]")
    return False
