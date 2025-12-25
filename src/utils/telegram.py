"""Telegram notification module for trading alerts."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Send trading notifications via Telegram bot.

    Notifications:
    - Signal generated (new trading opportunity)
    - Trade executed (position opened)
    - Trade closed (position closed with P&L)
    """

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self) -> None:
        """Initialize Telegram notifier."""
        self.settings = get_settings()
        self.bot_token = self.settings.telegram_bot_token
        self.chat_id = self.settings.telegram_chat_id
        self.enabled = bool(self.bot_token and self.chat_id and
                           self.bot_token != "your_bot_token_here")
        self._session: Optional[aiohttp.ClientSession] = None

        if self.enabled:
            logger.info("Telegram notifier enabled")
        else:
            logger.warning("Telegram notifier disabled (token/chat_id not set)")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        try:
            session = await self._ensure_session()
            url = self.BASE_URL.format(token=self.bot_token)

            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug("Telegram message sent")
                    return True
                else:
                    error = await response.text()
                    logger.warning("Telegram API error", status=response.status, error=error)
                    return False

        except Exception as e:
            logger.error("Failed to send Telegram message", error=str(e))
            return False

    async def notify_signal(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        reason: str,
    ) -> bool:
        """Notify about a new trading signal.

        Args:
            direction: LONG or SHORT
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Signal confidence (0-1)
            reason: Signal reason

        Returns:
            True if sent successfully
        """
        emoji = "🟢" if direction == "LONG" else "🔴"
        sl_pct = abs(entry_price - stop_loss) / entry_price * 100
        tp_pct = abs(take_profit - entry_price) / entry_price * 100

        text = f"""
{emoji} <b>Signal Generated</b>

<b>Direction:</b> {direction}
<b>Entry:</b> ${entry_price:,.2f}
<b>Stop Loss:</b> ${stop_loss:,.2f} (-{sl_pct:.2f}%)
<b>Take Profit:</b> ${take_profit:,.2f} (+{tp_pct:.2f}%)
<b>Confidence:</b> {confidence:.1%}

<i>{reason}</i>

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def notify_trade_opened(
        self,
        direction: str,
        entry_price: float,
        quantity: float,
        leverage: int,
        position_usd: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """Notify about a trade execution (position opened).

        Args:
            direction: LONG or SHORT
            entry_price: Actual entry price
            quantity: Position quantity
            leverage: Leverage used
            position_usd: Position size in USD
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            True if sent successfully
        """
        emoji = "✅"
        notional = position_usd * leverage

        text = f"""
{emoji} <b>Trade Executed</b>

<b>Direction:</b> {direction}
<b>Entry Price:</b> ${entry_price:,.2f}
<b>Quantity:</b> {quantity:.4f} BTC
<b>Leverage:</b> {leverage}x
<b>Margin:</b> ${position_usd:.2f}
<b>Notional:</b> ${notional:,.2f}

<b>Stop Loss:</b> ${stop_loss:,.2f}
<b>Take Profit:</b> ${take_profit:,.2f}

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def notify_trade_closed(
        self,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> bool:
        """Notify about a trade closure.

        Args:
            direction: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            pnl: Profit/loss in USD
            pnl_pct: Profit/loss percentage
            reason: Close reason

        Returns:
            True if sent successfully
        """
        if pnl >= 0:
            emoji = "💰"
            result = "WIN"
        else:
            emoji = "💸"
            result = "LOSS"

        text = f"""
{emoji} <b>Trade Closed - {result}</b>

<b>Direction:</b> {direction}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Quantity:</b> {quantity:.4f} BTC

<b>P&L:</b> ${pnl:+,.2f} ({pnl_pct:+.2f}%)

<i>Reason: {reason}</i>

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def notify_error(self, error_type: str, message: str) -> bool:
        """Notify about an error.

        Args:
            error_type: Type of error
            message: Error message

        Returns:
            True if sent successfully
        """
        text = f"""
⚠️ <b>Error Alert</b>

<b>Type:</b> {error_type}
<b>Message:</b> {message}

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def notify_bot_started(self, mode: str, symbol: str, leverage: int) -> bool:
        """Notify that bot has started.

        Args:
            mode: testnet or mainnet
            symbol: Trading symbol
            leverage: Leverage setting

        Returns:
            True if sent successfully
        """
        emoji = "🤖"
        mode_emoji = "🧪" if mode == "testnet" else "💵"

        text = f"""
{emoji} <b>Trading Bot Started</b>

{mode_emoji} <b>Mode:</b> {mode.upper()}
<b>Symbol:</b> {symbol}
<b>Leverage:</b> {leverage}x
<b>Max Position:</b> ${self.settings.max_position_usd:.0f}

<b>━━━ Strategy Settings ━━━</b>
<b>Wick Ratio:</b> {self.settings.wick_ratio_threshold:.0%}
<b>Volume Multiplier:</b> {self.settings.wick_volume_multiplier}x
<b>SL Buffer:</b> {self.settings.wick_stop_loss_buffer:.1%}
<b>Max Holding:</b> {self.settings.max_holding_minutes}분

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def notify_bot_stopped(
        self,
        signals_generated: int,
        signals_executed: int,
        total_pnl: float,
    ) -> bool:
        """Notify that bot has stopped.

        Args:
            signals_generated: Number of signals generated
            signals_executed: Number of signals executed
            total_pnl: Total P&L

        Returns:
            True if sent successfully
        """
        text = f"""
🛑 <b>Trading Bot Stopped</b>

<b>Signals Generated:</b> {signals_generated}
<b>Signals Executed:</b> {signals_executed}
<b>Total P&L:</b> ${total_pnl:+,.2f}

<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
"""
        return await self.send_message(text.strip())

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get singleton notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
