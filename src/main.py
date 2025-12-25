"""Main trading loop for Wick Reversal scalping bot.

Simplified architecture:
Collectors → WickReversalSignal → Risk Manager → Executor
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from src.collectors import BinanceRestCollector, BinanceWebSocketCollector
from src.executor import BinanceExecutor, PositionManager
from src.risk.manager import RiskManager
from src.signals.scalp import WickReversalSignal, CandleData
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.telegram import TelegramNotifier, get_notifier

logger = get_logger(__name__)


class TradingBot:
    """Wick Reversal Scalping Bot.

    Simplified architecture focusing on wick reversal signals only:
    - Collects 1m and 15m kline data
    - Generates signals on wick patterns with volume confirmation
    - Executes trades with dynamic exit conditions
    """

    def __init__(self) -> None:
        """Initialize the trading bot."""
        self.settings = get_settings()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized in start)
        self.ws_collector: Optional[BinanceWebSocketCollector] = None
        self.rest_collector: Optional[BinanceRestCollector] = None
        self.executor: Optional[BinanceExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None

        # Signal generator
        self.wick_signal: Optional[WickReversalSignal] = None

        # Telegram notifier
        self.notifier: TelegramNotifier = get_notifier()

        # State
        self._last_signal_check = datetime.now(timezone.utc)
        self._signal_check_interval = 1.0  # Check signals every second

        # Metrics
        self._signals_generated = 0
        self._signals_executed = 0
        self._errors_count = 0

        # Background tasks
        self._ws_task: Optional[asyncio.Task] = None

        logger.info(
            "TradingBot initialized",
            testnet=self.settings.binance_testnet,
            symbol=self.settings.trading_symbol,
        )

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        logger.info("Initializing components...")

        # Collectors
        self.ws_collector = BinanceWebSocketCollector()
        self.rest_collector = BinanceRestCollector()

        # Executor and Position Manager
        self.executor = BinanceExecutor()
        self.position_manager = PositionManager(
            max_scalp_positions=self.settings.max_concurrent_positions,
            max_swing_positions=0,  # No swing trading
        )

        # Risk Manager
        risk_config = {
            "risk": {
                "daily_loss_limit": self.settings.daily_loss_limit_pct,
                "consecutive_loss_cooldown": self.settings.max_consecutive_losses,
                "event_blackout_minutes": 0,  # No event blackout
                "llm_confidence_threshold": 0.0,  # No LLM
            }
        }
        self.risk_manager = RiskManager(risk_config)

        # Signal generator
        self.wick_signal = WickReversalSignal()

        # Register kline callback for real-time updates
        if self.ws_collector:
            self.ws_collector.on_kline(self._on_kline_update)

        # Load historical candles for immediate signal generation
        await self._load_historical_candles()

        logger.info("All components initialized")

    async def _load_historical_candles(self) -> None:
        """Load historical candles from REST API for immediate use."""
        if not self.rest_collector or not self.wick_signal:
            return

        logger.info("Loading historical candles...")

        try:
            symbol = self.settings.trading_symbol

            # Load 1m candles (need 21+ for volume average)
            klines_1m = await self.rest_collector.get_klines(
                symbol=symbol, interval="1m", limit=50
            )
            for kline in klines_1m[:-1]:  # Exclude last (potentially incomplete)
                candle = CandleData(
                    timestamp=kline["timestamp"],
                    open=kline["open"],
                    high=kline["high"],
                    low=kline["low"],
                    close=kline["close"],
                    volume=kline["volume"],
                    interval="1m",
                )
                self.wick_signal.add_candle(candle)

            # Load 15m candles (need 20+ for EMA)
            klines_15m = await self.rest_collector.get_klines(
                symbol=symbol, interval="15m", limit=50
            )
            for kline in klines_15m[:-1]:  # Exclude last (potentially incomplete)
                candle = CandleData(
                    timestamp=kline["timestamp"],
                    open=kline["open"],
                    high=kline["high"],
                    low=kline["low"],
                    close=kline["close"],
                    volume=kline["volume"],
                    interval="15m",
                )
                self.wick_signal.add_candle(candle)

            logger.info(
                "Historical candles loaded",
                candles_1m=len(klines_1m) - 1,
                candles_15m=len(klines_15m) - 1,
            )

        except Exception as e:
            logger.error("Failed to load historical candles", error=str(e))

    def _on_kline_update(self, kline_data: Any) -> None:
        """Handle incoming kline data.

        Args:
            kline_data: KlineData from WebSocket
        """
        if not self.wick_signal:
            return

        # Only process closed candles
        if not kline_data.is_closed:
            return

        candle = CandleData(
            timestamp=kline_data.timestamp,
            open=kline_data.open,
            high=kline_data.high,
            low=kline_data.low,
            close=kline_data.close,
            volume=kline_data.volume,
            interval=kline_data.interval,
        )

        self.wick_signal.add_candle(candle)

    async def _collect_market_data(self) -> dict[str, Any]:
        """Collect current market data from all sources.

        Returns:
            Market data dictionary
        """
        market_data: dict[str, Any] = {}

        try:
            if self.ws_collector:
                # Get latest price from trades
                ticker = await self.ws_collector.get_latest_data("ticker")
                if ticker:
                    price = float(ticker.get("price", 0))
                    market_data["current_price"] = price
                    market_data["price"] = price

                # Get latest klines
                kline_1m = await self.ws_collector.get_latest_data("kline_1m")
                if kline_1m:
                    market_data["kline_1m"] = kline_1m

                kline_15m = await self.ws_collector.get_latest_data("kline_15m")
                if kline_15m:
                    market_data["kline_15m"] = kline_15m

        except Exception as e:
            logger.warning("Error collecting market data", error=str(e))

        return market_data

    async def _generate_signal(self, market_data: dict[str, Any]) -> Optional[Any]:
        """Generate trading signal from market data.

        Args:
            market_data: Current market data

        Returns:
            Signal if conditions are met, None otherwise
        """
        if not self.wick_signal:
            return None

        try:
            signal = await self.wick_signal.generate(market_data)
            if signal:
                self._signals_generated += 1
                logger.info(
                    "Signal generated",
                    direction=signal.direction,
                    confidence=signal.confidence,
                    entry=signal.entry_price,
                )

                # Send Telegram notification for signal
                await self.notifier.notify_signal(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    confidence=signal.confidence,
                    reason=signal.reason,
                )

            return signal

        except Exception as e:
            logger.warning("Signal generation error", error=str(e))
            return None

    async def _execute_signal(
        self,
        signal: Any,
        market_data: dict[str, Any],
    ) -> bool:
        """Execute a trading signal.

        Args:
            signal: Trading signal
            market_data: Current market data

        Returns:
            True if execution succeeded
        """
        if not self.executor or not self.position_manager or not self.risk_manager:
            return False

        try:
            # Check if position can be opened
            can_open, reason = self.position_manager.can_open_position(signal)
            if not can_open:
                logger.info("Cannot open position", reason=reason)
                return False

            # Get account info
            account = await self.executor.get_account_info()

            # Apply risk check
            account_state = {
                "balance": account.available_balance,
                "positions": [
                    {"type": p.signal_type, "direction": p.direction}
                    for p in self.position_manager.active_positions
                ],
                "llm_confidence": signal.confidence,  # Use signal confidence
            }
            risk_result = self.risk_manager.check(signal, account_state)

            if not risk_result.approved:
                logger.info("Risk manager rejected signal", reason=risk_result.reason)
                return False

            # Get symbol-specific leverage
            leverage = self.settings.get_leverage(self.settings.trading_symbol)

            # Calculate position size (USD)
            if self.settings.max_position_usd > 0:
                # Use fixed USD amount
                position_usd = self.settings.max_position_usd
            else:
                # Use percentage of account
                position_usd = account.available_balance * risk_result.adjusted_size

            # Calculate quantity based on price and leverage
            current_price = market_data.get("current_price", signal.entry_price)
            notional_value = position_usd * leverage  # Leveraged position value
            quantity = notional_value / current_price

            # Round to appropriate precision (BTC: 3 decimals)
            quantity = round(quantity, 3)

            logger.info(
                "Position sizing",
                position_usd=position_usd,
                leverage=leverage,
                notional_value=notional_value,
                quantity=quantity,
                price=current_price,
            )

            # Execute signal with calculated quantity (SL/TP via price monitoring)
            entry_order = await self.executor.execute_signal(
                signal=signal,
                quantity=quantity,
                leverage=leverage,
            )

            if not entry_order or entry_order.status.value == "FAILED":
                logger.error("Entry order failed")
                return False

            # Create position
            position = self.position_manager.create_position(
                signal=signal,
                quantity=entry_order.quantity,
                leverage=leverage,
                entry_order_id=entry_order.order_id,
                symbol=self.settings.trading_symbol,
            )

            # Activate position (no SL/TP orders - managed by price monitoring)
            self.position_manager.activate_position(
                position_id=position.id,
                entry_price=entry_order.avg_price or signal.entry_price,
            )

            self._signals_executed += 1

            logger.info(
                "Signal executed successfully",
                position_id=position.id,
                direction=signal.direction,
                quantity=entry_order.quantity,
                leverage=leverage,
            )

            # Send Telegram notification for trade opened
            await self.notifier.notify_trade_opened(
                direction=signal.direction,
                entry_price=entry_order.avg_price or signal.entry_price,
                quantity=entry_order.quantity,
                leverage=leverage,
                position_usd=position_usd,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            return True

        except Exception as e:
            logger.error("Signal execution error", error=str(e))
            self._errors_count += 1
            return False

    async def _check_exit_conditions(self, market_data: dict[str, Any]) -> None:
        """Check and execute exit conditions for active positions.

        Exit conditions (checked in order of priority):
        1. Stop Loss - price hits SL level
        2. Take Profit - price hits TP level
        3. Max holding time exceeded
        4. Volume spike exit

        Args:
            market_data: Current market data
        """
        if not self.position_manager or not self.executor or not self.wick_signal:
            return

        current_price = market_data.get("current_price", 0)
        if current_price <= 0:
            return

        for position in self.position_manager.active_positions:
            try:
                should_exit = False
                exit_reason = ""

                # Check Stop Loss (highest priority)
                if position.is_long:
                    # LONG: exit if price <= stop_loss
                    if current_price <= position.stop_loss:
                        should_exit = True
                        exit_reason = f"Stop Loss triggered (${position.stop_loss:,.2f})"
                else:
                    # SHORT: exit if price >= stop_loss
                    if current_price >= position.stop_loss:
                        should_exit = True
                        exit_reason = f"Stop Loss triggered (${position.stop_loss:,.2f})"

                # Check Take Profit
                if not should_exit:
                    if position.is_long:
                        # LONG: exit if price >= take_profit
                        if current_price >= position.take_profit:
                            should_exit = True
                            exit_reason = f"Take Profit triggered (${position.take_profit:,.2f})"
                    else:
                        # SHORT: exit if price <= take_profit
                        if current_price <= position.take_profit:
                            should_exit = True
                            exit_reason = f"Take Profit triggered (${position.take_profit:,.2f})"

                # Check time-based exit
                if not should_exit:
                    if self.wick_signal.should_exit_on_time(self.settings.max_holding_minutes):
                        should_exit = True
                        exit_reason = f"Max holding time ({self.settings.max_holding_minutes}m) exceeded"

                # Check volume-based exit
                if not should_exit:
                    kline_1m = market_data.get("kline_1m")
                    if kline_1m and isinstance(kline_1m, dict):
                        current_volume = kline_1m.get("volume", 0)
                        if self.wick_signal.should_exit_on_volume(current_volume):
                            should_exit = True
                            exit_reason = "Volume exit trigger"

                if should_exit:
                    logger.info(
                        "Exiting position",
                        position_id=position.id,
                        reason=exit_reason,
                    )

                    # Close position at market
                    exit_side = "SELL" if position.is_long else "BUY"
                    await self.executor.place_market_order(
                        side=exit_side,
                        quantity=position.quantity,
                    )

                    # Mark position as closed
                    current_price = market_data.get("current_price", 0)

                    # Calculate P&L
                    pnl = position.calculate_unrealized_pnl(current_price)
                    pnl_pct = position.calculate_pnl_pct(current_price) * 100

                    self.position_manager.close_position(
                        position_id=position.id,
                        exit_price=current_price,
                        reason=exit_reason,
                    )

                    # Send Telegram notification for trade closed
                    await self.notifier.notify_trade_closed(
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=current_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason=exit_reason,
                    )

            except Exception as e:
                logger.error(
                    "Error checking exit conditions",
                    position_id=position.id,
                    error=str(e),
                )

    async def _process_signals(self, market_data: dict[str, Any]) -> None:
        """Process generated signals through the pipeline.

        Args:
            market_data: Current market data
        """
        # Check exit conditions first
        await self._check_exit_conditions(market_data)

        # Check if we can open new position
        if self.position_manager and len(self.position_manager.active_positions) >= self.settings.max_concurrent_positions:
            return

        # Generate and execute signal
        signal = await self._generate_signal(market_data)
        if signal:
            await self._execute_signal(signal, market_data)

    async def _update_positions(self, market_data: dict[str, Any]) -> None:
        """Update position tracking with current prices.

        Args:
            market_data: Current market data
        """
        if not self.position_manager:
            return

        current_price = market_data.get("current_price", 0)
        if current_price <= 0:
            return

        # Update trailing stops
        await self.position_manager.update_prices(current_price)

        # Log position status
        for position in self.position_manager.active_positions:
            pnl = position.calculate_unrealized_pnl(current_price)
            pnl_pct = position.calculate_pnl_pct(current_price)
            logger.debug(
                "Position status",
                position_id=position.id,
                direction=position.direction,
                unrealized_pnl=f"{pnl:.2f}",
                pnl_pct=f"{pnl_pct*100:.2f}%",
            )

    async def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting main trading loop")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Collect market data
                market_data = await self._collect_market_data()

                if not market_data:
                    await asyncio.sleep(1)
                    continue

                # Update positions
                await self._update_positions(market_data)

                # Check for new signals
                now = datetime.now(timezone.utc)
                if (now - self._last_signal_check).total_seconds() >= self._signal_check_interval:
                    await self._process_signals(market_data)
                    self._last_signal_check = now

                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error("Main loop error", error=str(e))
                self._errors_count += 1
                await asyncio.sleep(1)

    async def start(self) -> None:
        """Start the trading bot."""
        logger.info("Starting TradingBot...")

        try:
            # Initialize components
            await self._initialize_components()

            # Subscribe to streams BEFORE connecting
            if self.ws_collector:
                await self.ws_collector.subscribe_klines(["1m", "15m"])
                await self.ws_collector.subscribe_trades()

                # Start WebSocket connection in background task
                self._ws_task = asyncio.create_task(self.ws_collector.connect())

            self._running = True

            mode = "testnet" if self.settings.binance_testnet else "mainnet"
            leverage = self.settings.get_leverage(self.settings.trading_symbol)

            logger.info(
                "TradingBot started",
                mode=mode,
                symbol=self.settings.trading_symbol,
                leverage=leverage,
            )

            # Send Telegram notification for bot started
            await self.notifier.notify_bot_started(
                mode=mode,
                symbol=self.settings.trading_symbol,
                leverage=leverage,
            )

            # Run main loop
            await self._main_loop()

        except Exception as e:
            logger.error("TradingBot start error", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Stopping TradingBot...")

        self._running = False
        self._shutdown_event.set()

        # Cancel WebSocket background task
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

        # Close executor session
        if self.executor:
            await self.executor.close()

        # Log final stats
        total_pnl = 0.0
        if self.position_manager:
            stats = self.position_manager.get_session_stats()
            total_pnl = stats["total_pnl"]
            logger.info(
                "Session statistics",
                total_trades=stats["total_trades"],
                win_rate=f"{stats['win_rate']*100:.1f}%",
                total_pnl=total_pnl,
            )

        logger.info(
            "TradingBot stopped",
            signals_generated=self._signals_generated,
            signals_executed=self._signals_executed,
            errors=self._errors_count,
        )

        # Send Telegram notification for bot stopped
        await self.notifier.notify_bot_stopped(
            signals_generated=self._signals_generated,
            signals_executed=self._signals_executed,
            total_pnl=total_pnl,
        )

        # Close notifier session
        await self.notifier.close()

    def get_status(self) -> dict[str, Any]:
        """Get current bot status.

        Returns:
            Status dictionary
        """
        status = {
            "running": self._running,
            "testnet": self.settings.binance_testnet,
            "symbol": self.settings.trading_symbol,
            "leverage": self.settings.get_leverage(self.settings.trading_symbol),
            "metrics": {
                "signals_generated": self._signals_generated,
                "signals_executed": self._signals_executed,
                "errors": self._errors_count,
            },
        }

        if self.position_manager:
            status["positions"] = {
                "active": len(self.position_manager.active_positions),
                "max": self.settings.max_concurrent_positions,
                "session_stats": self.position_manager.get_session_stats(),
            }

        return status


async def main() -> None:
    """Main entry point."""
    bot = TradingBot()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.stop())

    # Handle SIGINT (Ctrl+C) and SIGTERM
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    else:
        # Windows doesn't support add_signal_handler
        signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
