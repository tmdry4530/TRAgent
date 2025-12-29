"""High Win Rate Trading Bot.

Uses 80% WR signal generator with consecutive loss position adjustment.
Strategy: Volume Climax + Channel Bounce
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.collectors import BinanceRestCollector, BinanceWebSocketCollector
from src.executor import BinanceExecutor, PositionManager
from src.risk.daily_limit import DailyLossLimitChecker, get_daily_limit_checker
from src.risk.loss_adjuster import (
    ConsecutiveLossAdjuster,
    LossAdjusterConfig,
    RecoveryMode,
)
from src.signals.high_wr import HighWinRateSignalGenerator
from src.signals.base import Signal
from src.utils.config import get_settings
from src.utils.database import TradeDatabase, TradeRecord, get_database
from src.utils.logger import get_logger
from src.utils.telegram import TelegramNotifier, get_notifier
from src.utils.telegram_bot import TelegramBot, get_telegram_bot

logger = get_logger(__name__)


class HighWRTradingBot:
    """High Win Rate Trading Bot.

    Features:
    - 68% win rate signal generator (Volume Climax + Channel Bounce)
    - Dynamic position sizing based on consecutive losses
    - Automatic recovery after wins
    - Daily loss limit protection

    Optimized Settings:
    - Risk per trade: 30%
    - R:R Ratio: 1.5
    - Leverage: 50x

    Expected Performance:
    - Win Rate: ~68%
    - Annual Return: ~500%+
    - Max Drawdown: ~110% (high risk, use with caution)
    """

    def __init__(self) -> None:
        """Initialize the trading bot."""
        self.settings = get_settings()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Symbol (single symbol for High WR strategy)
        self.symbol = self.settings.trading_symbol

        # Components (initialized in start)
        self.ws_collector: Optional[BinanceWebSocketCollector] = None
        self.rest_collector: Optional[BinanceRestCollector] = None
        self.executor: Optional[BinanceExecutor] = None
        self.position_manager: Optional[PositionManager] = None

        # High Win Rate Signal Generator
        self.signal_generator = HighWinRateSignalGenerator(
            vol_multiplier=3.0,
            consecutive_bars=3,
            wick_pct=0.50,
            rsi_threshold=40,
            channel_vol_threshold=1.2,
            rr_ratio=1.5,
        )

        # Consecutive Loss Adjuster (Optimized for 30% risk)
        loss_config = LossAdjusterConfig(
            base_risk_per_trade=0.30,  # 30% base risk (optimized)
            loss_levels=[
                (1, 1.0),    # 1 loss: 100%
                (2, 0.50),   # 2 consecutive: 50% → 15% effective
                (3, 0.25),   # 3 consecutive: 25% → 7.5% effective
                (4, 0.10),   # 4 consecutive: 10% → 3% effective
                (5, 0.0),    # 5+ consecutive: STOP
            ],
            recovery_mode=RecoveryMode.GRADUAL,
            max_cooldown_hours=4,
            daily_loss_limit_pct=0.15,  # 15% daily limit (half of base risk)
        )
        self.loss_adjuster = ConsecutiveLossAdjuster(loss_config)

        # Telegram notifier
        self.notifier: TelegramNotifier = get_notifier()

        # Database and daily limit checker
        self.db: TradeDatabase = get_database()
        self.daily_limit_checker: DailyLossLimitChecker = get_daily_limit_checker()

        # Telegram bot for commands
        self.telegram_bot: TelegramBot = get_telegram_bot()
        self.telegram_bot.set_status_callback(self.get_status)
        self.telegram_bot.set_stop_callback(self._telegram_stop)

        # State
        self._candle_history: list[dict] = []
        self._last_candle_time: Optional[datetime] = None
        self._warmup_complete = False

        # Metrics
        self._signals_generated = 0
        self._signals_executed = 0
        self._errors_count = 0

        # Background tasks
        self._ws_task: Optional[asyncio.Task] = None

        logger.info(
            "HighWRTradingBot initialized",
            testnet=self.settings.binance_testnet,
            symbol=self.symbol,
            base_risk=f"{loss_config.base_risk_per_trade*100:.0f}%",
        )

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        logger.debug("Initializing components...")

        # Collectors
        self.ws_collector = BinanceWebSocketCollector(symbols=[self.symbol])
        self.rest_collector = BinanceRestCollector()

        # Executor and Position Manager
        self.executor = BinanceExecutor()
        self.position_manager = PositionManager(
            max_scalp_positions=1,  # Only 1 position at a time
            max_swing_positions=0,
        )

        # Register kline callback
        if self.ws_collector:
            self.ws_collector.on_kline(self._on_kline_update)

        # Load historical candles for indicator calculation
        await self._load_historical_candles()

        logger.debug("All components initialized")

    async def _load_historical_candles(self) -> None:
        """Load historical 1h candles for signal generation."""
        if not self.rest_collector:
            return

        logger.info("Loading historical 1h candles...")

        try:
            # Need 100+ candles for indicators (EMA 50, RSI 14, etc.)
            klines = await self.rest_collector.get_klines(
                symbol=self.symbol,
                interval="1h",
                limit=200,
            )

            for kline in klines[:-1]:  # Exclude last (incomplete)
                self._candle_history.append({
                    "timestamp": kline["timestamp"],
                    "open": kline["open"],
                    "high": kline["high"],
                    "low": kline["low"],
                    "close": kline["close"],
                    "volume": kline["volume"],
                })

            logger.info(
                "Historical candles loaded",
                count=len(self._candle_history),
                latest=self._candle_history[-1]["timestamp"] if self._candle_history else None,
            )

            self._warmup_complete = len(self._candle_history) >= 60

        except Exception as e:
            logger.error("Failed to load historical candles", error=str(e))

    def _on_kline_update(self, kline_data: Any) -> None:
        """Handle incoming kline data from WebSocket."""
        # Only process 1h candles
        if kline_data.interval != "1h":
            return

        # Only process closed candles
        if not kline_data.is_closed:
            return

        # Avoid duplicates
        if self._last_candle_time and kline_data.timestamp <= self._last_candle_time:
            return

        self._last_candle_time = kline_data.timestamp

        # Add to history
        self._candle_history.append({
            "timestamp": kline_data.timestamp,
            "open": kline_data.open,
            "high": kline_data.high,
            "low": kline_data.low,
            "close": kline_data.close,
            "volume": kline_data.volume,
        })

        # Keep only last 200 candles
        if len(self._candle_history) > 200:
            self._candle_history = self._candle_history[-200:]

        logger.debug(
            "New 1h candle added",
            timestamp=kline_data.timestamp,
            close=kline_data.close,
        )

    async def _check_signal(self) -> Optional[Signal]:
        """Check for trading signal from current candle data."""
        if not self._warmup_complete or len(self._candle_history) < 60:
            return None

        try:
            # Convert to DataFrame
            df = pd.DataFrame(self._candle_history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Generate signals
            signals = self.signal_generator.generate_from_dataframe(df)

            # Return the latest signal if it's from the most recent candle
            if signals:
                latest_signal = signals[-1]
                latest_candle_time = self._candle_history[-1]["timestamp"]

                # Signal must be from current or previous candle
                time_diff = abs((latest_signal.timestamp - latest_candle_time).total_seconds())
                if time_diff <= 3600:  # Within 1 hour
                    self._signals_generated += 1
                    return latest_signal

        except Exception as e:
            logger.error("Signal check error", error=str(e))

        return None

    async def _execute_signal(self, signal: Signal) -> bool:
        """Execute a trading signal with dynamic position sizing."""
        if not self.executor or not self.position_manager:
            return False

        try:
            # Check if we can trade
            can_open, reason = self.position_manager.can_open_position(signal)
            if not can_open:
                logger.info("Cannot open position", reason=reason)
                return False

            # Get adjusted risk from loss adjuster
            account = await self.executor.get_account_info()
            adjustment = self.loss_adjuster.get_adjusted_risk(account.available_balance)

            if not adjustment.can_trade:
                logger.warning("Loss adjuster blocked trading", reason=adjustment.reason)
                await self.notifier.send_message(
                    f"⚠️ <b>거래 차단 (연속 손실)</b>\n\n{adjustment.reason}"
                )
                return False

            # Check daily loss limit
            can_trade, limit_reason = self.daily_limit_checker.check_can_trade()
            if not can_trade:
                logger.warning("Daily loss limit reached", reason=limit_reason)
                await self.notifier.send_message(
                    f"⚠️ <b>일일 손실 한도 도달</b>\n\n{limit_reason}"
                )
                return False

            # Calculate position size
            risk_per_trade = adjustment.risk_per_trade
            multiplier = adjustment.size_multiplier

            # Calculate SL distance
            sl_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

            # Position sizing based on risk
            risk_amount = account.available_balance * risk_per_trade
            position_usd = risk_amount / sl_distance_pct

            # Cap at max position
            max_position = account.available_balance * 0.95  # 95% max
            position_usd = min(position_usd, max_position)

            # Get leverage
            leverage = self.settings.get_leverage(self.symbol)

            # Calculate quantity
            notional_value = position_usd * leverage
            quantity = notional_value / signal.entry_price

            # Round quantity
            if self.symbol == "SOLUSDT":
                quantity = round(quantity, 0)
            else:
                quantity = round(quantity, 3)

            if quantity < 0.001:
                logger.warning("Quantity too small", quantity=quantity)
                return False

            logger.info(
                "Executing signal",
                direction=signal.direction,
                risk=f"{risk_per_trade*100:.0f}%",
                multiplier=f"{multiplier:.0%}",
                position_usd=f"${position_usd:.2f}",
                quantity=quantity,
            )

            # Set leverage
            await self.executor.set_leverage(leverage, self.symbol)

            # Execute entry order
            entry_order = await self.executor.execute_signal(
                signal=signal,
                quantity=quantity,
                leverage=leverage,
                symbol=self.symbol,
            )

            if not entry_order or entry_order.status.value == "FAILED":
                logger.error("Entry order failed")
                return False

            # Create and activate position
            position = self.position_manager.create_position(
                signal=signal,
                quantity=entry_order.quantity,
                leverage=leverage,
                entry_order_id=entry_order.order_id,
                symbol=self.symbol,
            )

            self.position_manager.activate_position(
                position_id=position.id,
                entry_price=entry_order.avg_price or signal.entry_price,
            )

            self._signals_executed += 1

            # Send notification
            await self.notifier.send_message(
                f"🎯 <b>High WR 시그널 체결</b>\n\n"
                f"방향: {signal.direction}\n"
                f"진입가: ${signal.entry_price:,.2f}\n"
                f"손절: ${signal.stop_loss:,.2f}\n"
                f"익절: ${signal.take_profit:,.2f}\n"
                f"수량: {quantity}\n"
                f"리스크: {risk_per_trade*100:.0f}% (x{multiplier:.0%})\n"
                f"포지션: ${position_usd:,.2f}\n\n"
                f"사유: {signal.reason}"
            )

            return True

        except Exception as e:
            logger.error("Signal execution error", error=str(e))
            self._errors_count += 1
            return False

    async def _check_exit_conditions(self, current_price: float) -> None:
        """Check exit conditions for active positions."""
        if not self.position_manager or not self.executor:
            return

        for position in self.position_manager.active_positions:
            if position.symbol != self.symbol:
                continue

            try:
                should_exit = False
                exit_reason = ""

                # Check Stop Loss
                if position.is_long:
                    if current_price <= position.stop_loss:
                        should_exit = True
                        exit_reason = f"Stop Loss (${position.stop_loss:,.2f})"
                else:
                    if current_price >= position.stop_loss:
                        should_exit = True
                        exit_reason = f"Stop Loss (${position.stop_loss:,.2f})"

                # Check Take Profit
                if not should_exit:
                    if position.is_long:
                        if current_price >= position.take_profit:
                            should_exit = True
                            exit_reason = f"Take Profit (${position.take_profit:,.2f})"
                    else:
                        if current_price <= position.take_profit:
                            should_exit = True
                            exit_reason = f"Take Profit (${position.take_profit:,.2f})"

                if should_exit:
                    logger.info("Closing position", reason=exit_reason)

                    # Close position
                    exit_side = "SELL" if position.is_long else "BUY"
                    await self.executor.place_market_order(
                        side=exit_side,
                        quantity=position.quantity,
                        symbol=self.symbol,
                    )

                    # Calculate PnL
                    pnl = position.calculate_unrealized_pnl(current_price)
                    pnl_pct = position.calculate_pnl_pct(current_price) * 100

                    # Record trade result in loss adjuster
                    self.loss_adjuster.record_trade(pnl, pnl_pct)

                    # Close position in manager
                    self.position_manager.close_position(
                        position_id=position.id,
                        exit_price=current_price,
                        reason=exit_reason,
                    )

                    # Save to database
                    trade_record = TradeRecord(
                        id=None,
                        symbol=position.symbol,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=current_price,
                        quantity=position.quantity,
                        leverage=position.leverage,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(timezone.utc),
                        exit_reason=exit_reason,
                        signal_reason=position.signal_reason,
                    )
                    self.db.save_trade(trade_record)

                    # Get loss adjuster stats
                    stats = self.loss_adjuster.get_stats()

                    # Send notification
                    result_emoji = "✅" if pnl > 0 else "❌"
                    await self.notifier.send_message(
                        f"{result_emoji} <b>포지션 청산</b>\n\n"
                        f"방향: {position.direction}\n"
                        f"진입가: ${position.entry_price:,.2f}\n"
                        f"청산가: ${current_price:,.2f}\n"
                        f"수익: {pnl:+,.2f} USD ({pnl_pct:+.2f}%)\n"
                        f"사유: {exit_reason}\n\n"
                        f"<b>연속 손실 상태:</b>\n"
                        f"연속 손실: {stats['consecutive_losses']}회\n"
                        f"연속 승리: {stats['consecutive_wins']}회\n"
                        f"현재 배수: {stats['current_multiplier']:.0%}\n"
                        f"다음 리스크: {stats['effective_risk']*100:.1f}%"
                    )

            except Exception as e:
                logger.error("Exit check error", error=str(e))

    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting main trading loop")

        last_signal_check = datetime.now(timezone.utc)
        signal_check_interval = 60  # Check every minute (for 1h candles)

        while self._running and not self._shutdown_event.is_set():
            try:
                # Get current price
                if self.ws_collector:
                    ticker = await self.ws_collector.get_latest_data("ticker", symbol=self.symbol)
                    if ticker:
                        current_price = float(ticker.get("price", 0))

                        # Check exit conditions
                        if current_price > 0:
                            await self._check_exit_conditions(current_price)

                # Check for new signals periodically
                now = datetime.now(timezone.utc)
                if (now - last_signal_check).total_seconds() >= signal_check_interval:
                    last_signal_check = now

                    # Only check if no active position
                    if self.position_manager and len(self.position_manager.active_positions) == 0:
                        signal = await self._check_signal()
                        if signal:
                            await self._execute_signal(signal)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error("Main loop error", error=str(e))
                self._errors_count += 1
                await asyncio.sleep(5)

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting HighWRTradingBot...")

        try:
            await self._initialize_components()

            # Subscribe to streams
            if self.ws_collector:
                await self.ws_collector.subscribe_klines(["1h"])
                await self.ws_collector.subscribe_trades()
                self._ws_task = asyncio.create_task(self.ws_collector.connect())

            self._running = True

            # Start Telegram bot
            self.telegram_bot.start_polling()

            mode = "testnet" if self.settings.binance_testnet else "mainnet"
            leverage = self.settings.get_leverage(self.symbol)

            logger.info(
                "HighWRTradingBot started",
                mode=mode,
                symbol=self.symbol,
                leverage=leverage,
            )

            # Send startup notification
            stats = self.loss_adjuster.get_stats()
            await self.notifier.send_message(
                f"🚀 <b>High WR Trading Bot 시작</b>\n\n"
                f"모드: {mode}\n"
                f"심볼: {self.symbol}\n"
                f"레버리지: {leverage}x\n"
                f"기본 리스크: 30%\n\n"
                f"<b>전략 (최적화됨):</b>\n"
                f"- Volume Climax + Channel Bounce\n"
                f"- 예상 승률: 68%\n"
                f"- R:R 비율: 1.5\n"
                f"- 예상 연간 수익률: 500%+\n\n"
                f"<b>연속 손실 보호:</b>\n"
                f"- 2연속: 50%로 축소 (15% 리스크)\n"
                f"- 3연속: 25%로 축소 (7.5% 리스크)\n"
                f"- 5연속: 거래 중단"
            )

            await self._main_loop()

        except Exception as e:
            logger.error("Bot start error", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Stopping HighWRTradingBot...")

        self._running = False
        self._shutdown_event.set()

        # Stop Telegram bot
        await self.telegram_bot.stop_polling()

        # Cancel WebSocket task
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        # Disconnect collectors
        if self.ws_collector:
            await self.ws_collector.disconnect()

        if self.rest_collector:
            await self.rest_collector.close()

        # Close executor
        if self.executor:
            await self.executor.close()

        # Get final stats
        total_pnl = 0.0
        if self.position_manager:
            stats = self.position_manager.get_session_stats()
            total_pnl = stats["total_pnl"]

        loss_stats = self.loss_adjuster.get_stats()

        logger.info(
            "HighWRTradingBot stopped",
            signals_generated=self._signals_generated,
            signals_executed=self._signals_executed,
            total_pnl=total_pnl,
        )

        # Send shutdown notification
        await self.notifier.send_message(
            f"🛑 <b>High WR Trading Bot 종료</b>\n\n"
            f"생성된 시그널: {self._signals_generated}\n"
            f"실행된 시그널: {self._signals_executed}\n"
            f"총 수익: {total_pnl:+,.2f} USD\n"
            f"에러: {self._errors_count}\n\n"
            f"<b>연속 손실 상태:</b>\n"
            f"연속 손실: {loss_stats['consecutive_losses']}회\n"
            f"총 거래: {loss_stats['total_trades']}회"
        )

        await self.notifier.close()

    def get_status(self) -> dict[str, Any]:
        """Get current bot status."""
        loss_stats = self.loss_adjuster.get_stats()

        status = {
            "running": self._running,
            "testnet": self.settings.binance_testnet,
            "symbol": self.symbol,
            "strategy": "High Win Rate (Vol Climax + Channel Bounce)",
            "expected_win_rate": "80%",
            "metrics": {
                "signals_generated": self._signals_generated,
                "signals_executed": self._signals_executed,
                "errors": self._errors_count,
            },
            "loss_adjuster": {
                "consecutive_losses": loss_stats["consecutive_losses"],
                "consecutive_wins": loss_stats["consecutive_wins"],
                "current_multiplier": f"{loss_stats['current_multiplier']:.0%}",
                "effective_risk": f"{loss_stats['effective_risk']*100:.1f}%",
                "can_trade": loss_stats["can_trade"],
            },
            "warmup_complete": self._warmup_complete,
            "candle_history": len(self._candle_history),
        }

        if self.position_manager:
            status["positions"] = {
                "active": len(self.position_manager.active_positions),
                "session_stats": self.position_manager.get_session_stats(),
            }

        return status

    async def _telegram_stop(self) -> None:
        """Handle stop command from Telegram."""
        logger.info("Stop command received from Telegram")
        await self.stop()


async def main() -> None:
    """Main entry point."""
    bot = HighWRTradingBot()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    else:
        signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
