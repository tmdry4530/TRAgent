"""Telegram bot with command handling for trading bot control."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

import aiohttp

from src.utils.config import get_settings
from src.utils.database import get_database
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramBot:
    """Telegram bot with command handling.

    Commands:
    - /status - Current bot status
    - /stats - Today's trading statistics
    - /stop - Stop the bot gracefully
    - /help - Show available commands
    """

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self) -> None:
        """Initialize Telegram bot."""
        self.settings = get_settings()
        self.bot_token = self.settings.telegram_bot_token
        self.chat_id = self.settings.telegram_chat_id
        self.enabled = bool(self.bot_token and self.chat_id and
                           self.bot_token != "your_bot_token_here")

        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._last_update_id = 0
        self._poll_task: Optional[asyncio.Task] = None

        # Command handlers
        self._commands: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {}

        # Callbacks for bot control
        self._stop_callback: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
        self._status_callback: Optional[Callable[[], dict[str, Any]]] = None

        if self.enabled:
            logger.info("TelegramBot initialized")

    def set_stop_callback(self, callback: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Set callback for /stop command."""
        self._stop_callback = callback

    def set_status_callback(self, callback: Callable[[], dict[str, Any]]) -> None:
        """Set callback for /status command."""
        self._status_callback = callback

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False

        try:
            session = await self._ensure_session()
            url = f"{self.BASE_URL.format(token=self.bot_token)}/sendMessage"

            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }

            async with session.post(url, json=payload) as response:
                return response.status == 200

        except Exception as e:
            logger.error("Failed to send message", error=str(e))
            return False

    async def _get_updates(self) -> list[dict]:
        """Get updates from Telegram."""
        if not self.enabled:
            return []

        try:
            session = await self._ensure_session()
            url = f"{self.BASE_URL.format(token=self.bot_token)}/getUpdates"

            params = {
                "offset": self._last_update_id + 1,
                "timeout": 30,
                "allowed_updates": ["message"],
            }

            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=35)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", [])
                return []

        except asyncio.TimeoutError:
            return []
        except Exception as e:
            logger.debug("Get updates error", error=str(e))
            return []

    async def _handle_command(self, message: dict) -> None:
        """Handle incoming command message."""
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")

        # Verify it's from authorized chat
        if str(chat_id) != str(self.chat_id):
            return

        if not text.startswith("/"):
            return

        command = text.split()[0].lower().replace("@", "").split("@")[0]

        if command == "/status":
            response = await self._cmd_status()
        elif command == "/stats":
            response = await self._cmd_stats()
        elif command == "/stop":
            response = await self._cmd_stop()
        elif command == "/help":
            response = await self._cmd_help()
        elif command == "/start":
            response = "🤖 Trading Bot이 연결되었습니다.\n/help 로 명령어를 확인하세요."
        else:
            response = f"❓ 알 수 없는 명령어: {command}\n/help 로 명령어를 확인하세요."

        await self.send_message(response)

    async def _cmd_help(self) -> str:
        """Handle /help command."""
        return """📋 <b>Available Commands</b>

/status - 현재 봇 상태 확인
/stats - 오늘 거래 통계
/stop - 봇 중지
/help - 명령어 도움말"""

    async def _cmd_status(self) -> str:
        """Handle /status command."""
        if self._status_callback:
            try:
                status = self._status_callback()

                running = "🟢 Running" if status.get("running") else "🔴 Stopped"
                mode = "🧪 Testnet" if status.get("testnet") else "💵 Mainnet"

                positions = status.get("positions", {})
                active = positions.get("active", 0)
                max_pos = positions.get("max", 1)

                metrics = status.get("metrics", {})
                signals_gen = metrics.get("signals_generated", 0)
                signals_exec = metrics.get("signals_executed", 0)
                errors = metrics.get("errors", 0)

                return f"""📊 <b>Bot Status</b>

<b>Status:</b> {running}
<b>Mode:</b> {mode}
<b>Symbol:</b> {status.get('symbol', 'N/A')}
<b>Leverage:</b> {status.get('leverage', 0)}x

<b>━━━ Positions ━━━</b>
<b>Active:</b> {active}/{max_pos}

<b>━━━ Session Metrics ━━━</b>
<b>Signals Generated:</b> {signals_gen}
<b>Signals Executed:</b> {signals_exec}
<b>Errors:</b> {errors}

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"""
            except Exception as e:
                return f"❌ Status error: {str(e)}"
        else:
            return "⚠️ Status callback not configured"

    async def _cmd_stats(self) -> str:
        """Handle /stats command."""
        try:
            db = get_database()
            stats = db.get_daily_stats()
            daily_pnl = db.get_daily_pnl()

            if stats is None:
                return """📈 <b>Today's Statistics</b>

거래 기록이 없습니다.

<code>{}</code>""".format(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))

            pnl_emoji = "💰" if daily_pnl >= 0 else "💸"
            win_rate = stats.win_rate * 100

            return f"""📈 <b>Today's Statistics</b>

<b>Total Trades:</b> {stats.total_trades}
<b>Wins:</b> {stats.winning_trades} | <b>Losses:</b> {stats.losing_trades}
<b>Win Rate:</b> {win_rate:.1f}%

{pnl_emoji} <b>Daily P&L:</b> ${daily_pnl:+,.2f}
<b>Max Drawdown:</b> ${stats.max_drawdown:.2f}

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>"""

        except Exception as e:
            return f"❌ Stats error: {str(e)}"

    async def _cmd_stop(self) -> str:
        """Handle /stop command."""
        if self._stop_callback:
            await self.send_message("🛑 봇을 중지합니다...")
            try:
                asyncio.create_task(self._stop_callback())
                return "✅ 봇 중지 명령이 실행되었습니다."
            except Exception as e:
                return f"❌ Stop error: {str(e)}"
        else:
            return "⚠️ Stop callback not configured"

    async def _poll_loop(self) -> None:
        """Poll for updates in background."""
        logger.info("Telegram bot polling started")

        while self._running:
            try:
                updates = await self._get_updates()

                for update in updates:
                    self._last_update_id = update.get("update_id", self._last_update_id)

                    if "message" in update:
                        await self._handle_command(update["message"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Poll loop error", error=str(e))
                await asyncio.sleep(5)

        logger.info("Telegram bot polling stopped")

    async def _skip_old_updates(self) -> None:
        """Skip all pending updates on startup to avoid processing old commands."""
        if not self.enabled:
            return

        try:
            session = await self._ensure_session()
            url = f"{self.BASE_URL.format(token=self.bot_token)}/getUpdates"

            # Get latest updates without timeout
            params = {"offset": -1, "limit": 1}

            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("result", [])
                    if results:
                        # Set offset to skip all old messages
                        self._last_update_id = results[-1].get("update_id", 0)
                        logger.info(
                            "Skipped old Telegram updates",
                            last_update_id=self._last_update_id,
                        )

        except Exception as e:
            logger.debug("Skip old updates error", error=str(e))

    def start_polling(self) -> None:
        """Start polling for commands in background."""
        if not self.enabled:
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._start_polling_async())

    async def _start_polling_async(self) -> None:
        """Initialize and start polling."""
        # Skip old updates first to avoid processing stale commands
        await self._skip_old_updates()
        # Then start the poll loop
        await self._poll_loop()

    async def stop_polling(self) -> None:
        """Stop polling."""
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_bot: Optional[TelegramBot] = None


def get_telegram_bot() -> TelegramBot:
    """Get singleton bot instance."""
    global _bot
    if _bot is None:
        _bot = TelegramBot()
    return _bot
