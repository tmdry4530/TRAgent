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
from src.risk.daily_limit import DailyLossLimitChecker, get_daily_limit_checker
from src.risk.manager import RiskManager
from src.signals.scalp import WickReversalSignal, CandleData
from src.utils.config import get_settings
from src.utils.database import TradeDatabase, TradeRecord, get_database
from src.utils.logger import get_logger
from src.utils.telegram import TelegramNotifier, get_notifier
from src.utils.telegram_bot import TelegramBot, get_telegram_bot

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

        # Multi-symbol support
        self.symbols = self.settings.trading_symbols

        # Components (initialized in start)
        self.ws_collector: Optional[BinanceWebSocketCollector] = None
        self.rest_collector: Optional[BinanceRestCollector] = None
        self.executor: Optional[BinanceExecutor] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None

        # Signal generators per symbol
        self.wick_signals: dict[str, WickReversalSignal] = {}

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
        self._last_signal_check = datetime.now(timezone.utc)
        self._signal_check_interval = 1.0  # Check signals every second
        self._warmup_seconds = 10  # Wait before generating signals after startup
        self._start_time: Optional[datetime] = None
        self._warmup_logged = False

        # Metrics
        self._signals_generated = 0
        self._signals_executed = 0
        self._errors_count = 0

        # Background tasks
        self._ws_task: Optional[asyncio.Task] = None

        logger.info(
            "TradingBot initialized",
            testnet=self.settings.binance_testnet,
            symbols=self.symbols,
        )

    async def _initialize_components(self) -> None:
        """Initialize all trading components."""
        logger.debug("Initializing components...")

        # Collectors (multi-symbol support)
        self.ws_collector = BinanceWebSocketCollector(symbols=self.symbols)
        self.rest_collector = BinanceRestCollector()

        # Executor and Position Manager
        self.executor = BinanceExecutor()
        self.position_manager = PositionManager(
            max_scalp_positions=self.settings.max_concurrent_positions * len(self.symbols),
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

        # Signal generators per symbol
        for symbol in self.symbols:
            self.wick_signals[symbol] = WickReversalSignal()
            logger.debug("Signal generator created", symbol=symbol)

        # Register kline callback for real-time updates
        if self.ws_collector:
            self.ws_collector.on_kline(self._on_kline_update)

        # Load historical candles for immediate signal generation
        await self._load_historical_candles()

        logger.debug("All components initialized", symbols=self.symbols)

    async def _load_historical_candles(self) -> None:
        """Load historical candles from REST API for all symbols."""
        if not self.rest_collector or not self.wick_signals:
            return

        logger.debug("Loading historical candles for all symbols...")

        for symbol in self.symbols:
            try:
                wick_signal = self.wick_signals.get(symbol)
                if not wick_signal:
                    continue

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
                    wick_signal.add_candle(candle)

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
                    wick_signal.add_candle(candle)

                logger.debug(
                    "Historical candles loaded",
                    symbol=symbol,
                    candles_1m=len(klines_1m) - 1,
                    candles_15m=len(klines_15m) - 1,
                )

            except Exception as e:
                logger.error("Failed to load historical candles", symbol=symbol, error=str(e))

    def _on_kline_update(self, kline_data: Any) -> None:
        """Handle incoming kline data.

        Args:
            kline_data: KlineData from WebSocket (now includes symbol)
        """
        # Only process closed candles
        if not kline_data.is_closed:
            return

        # Get signal generator for this symbol
        symbol = kline_data.symbol
        wick_signal = self.wick_signals.get(symbol)
        if not wick_signal:
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

        wick_signal.add_candle(candle)

    async def _collect_market_data(self, symbol: str) -> dict[str, Any]:
        """Collect current market data for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Market data dictionary for the symbol
        """
        market_data: dict[str, Any] = {"symbol": symbol}

        try:
            if self.ws_collector:
                # Get latest price from trades for this symbol
                ticker = await self.ws_collector.get_latest_data("ticker", symbol=symbol)
                if ticker:
                    price = float(ticker.get("price", 0))
                    market_data["current_price"] = price
                    market_data["price"] = price

                # Get latest klines for this symbol
                kline_1m = await self.ws_collector.get_latest_data("kline_1m", symbol=symbol)
                if kline_1m:
                    market_data["kline_1m"] = kline_1m

                kline_15m = await self.ws_collector.get_latest_data("kline_15m", symbol=symbol)
                if kline_15m:
                    market_data["kline_15m"] = kline_15m

        except Exception as e:
            logger.warning("Error collecting market data", symbol=symbol, error=str(e))

        return market_data

    async def _generate_signal(self, market_data: dict[str, Any]) -> Optional[Any]:
        """Generate trading signal from market data.

        Args:
            market_data: Current market data (includes symbol)

        Returns:
            Signal if conditions are met, None otherwise
        """
        symbol = market_data.get("symbol")
        if not symbol:
            return None

        wick_signal = self.wick_signals.get(symbol)
        if not wick_signal:
            return None

        try:
            signal = await wick_signal.generate(market_data)
            if signal:
                self._signals_generated += 1
                logger.info(
                    "Signal generated",
                    symbol=symbol,
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
                    reason=f"[{symbol}] {signal.reason}",
                )

            return signal

        except Exception as e:
            logger.warning("Signal generation error", symbol=symbol, error=str(e))
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
            # Check daily loss limit first
            can_trade, limit_reason = self.daily_limit_checker.check_can_trade()
            if not can_trade:
                logger.warning("Daily loss limit check failed", reason=limit_reason)
                await self.notifier.send_message(
                    f"⚠️ <b>거래 차단</b>\n\n{limit_reason}"
                )
                return False

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

            # Get symbol from market_data
            symbol = market_data.get("symbol", self.settings.trading_symbol)

            # Get symbol-specific leverage
            leverage = self.settings.get_leverage(symbol)

            # Calculate position size (USD) with dynamic sizing
            if self.settings.max_position_usd > 0:
                # Get dynamic position size based on daily performance
                base_position = self.settings.max_position_usd
                position_usd = self.daily_limit_checker.get_dynamic_position_size(
                    base_position_usd=base_position,
                    max_loss_pct=0.02,  # Assume 2% max loss per trade
                )
            else:
                # Use percentage of account
                position_usd = account.available_balance * risk_result.adjusted_size

            # Calculate quantity based on price and leverage
            current_price = market_data.get("current_price", signal.entry_price)
            notional_value = position_usd * leverage  # Leveraged position value
            quantity = notional_value / current_price

            # Round to appropriate precision (SOL: 0 decimals, BTC: 3 decimals)
            if symbol == "SOLUSDT":
                quantity = round(quantity, 0)
            else:
                quantity = round(quantity, 3)

            logger.debug(
                "Position sizing",
                symbol=symbol,
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
                symbol=symbol,
            )

            if not entry_order or entry_order.status.value == "FAILED":
                logger.error("Entry order failed", symbol=symbol)
                return False

            # Create position
            position = self.position_manager.create_position(
                signal=signal,
                quantity=entry_order.quantity,
                leverage=leverage,
                entry_order_id=entry_order.order_id,
                symbol=symbol,
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
        1. Partial Take Profit - close 50% at halfway to TP (SCALP only)
        2. Trailing Stop activation - activate when profit threshold reached
        3. Stop Loss - price hits SL level (may be trailing)
        4. Take Profit - price hits TP level
        5. Max holding time exceeded
        6. Volume spike exit

        Args:
            market_data: Current market data (includes symbol)
        """
        if not self.position_manager or not self.executor:
            return

        current_price = market_data.get("current_price", 0)
        symbol = market_data.get("symbol")
        if current_price <= 0 or not symbol:
            return

        # Only check positions for this symbol
        for position in self.position_manager.active_positions:
            if position.symbol != symbol:
                continue

            # Get wick signal for this symbol
            wick_signal = self.wick_signals.get(symbol)

            try:
                # === PARTIAL TAKE PROFIT (SCALP only) ===
                partial_qty = position.check_partial_tp(current_price)
                if partial_qty and partial_qty > 0:
                    logger.info(
                        "Executing partial take profit",
                        position_id=position.id,
                        symbol=symbol,
                        partial_qty=partial_qty,
                        remaining_qty=position.quantity,
                    )
                    # Execute partial close
                    exit_side = "SELL" if position.is_long else "BUY"
                    await self.executor.place_market_order(
                        side=exit_side,
                        quantity=partial_qty,
                        symbol=symbol,
                    )
                    # Notify
                    pnl_pct = position.calculate_pnl_pct(current_price) * 100
                    await self.notifier.send_message(
                        f"📊 <b>부분 익절 실행</b> [{symbol}]\n\n"
                        f"방향: {position.direction}\n"
                        f"청산 수량: {partial_qty:.4f}\n"
                        f"남은 수량: {position.quantity:.4f}\n"
                        f"현재 수익률: {pnl_pct:+.2f}%"
                    )

                # === TRAILING STOP ACTIVATION ===
                # Use scalp-specific activation threshold
                if position.is_scalp:
                    activation_pct = self.position_manager.scalp_trailing_activation_pct
                else:
                    activation_pct = self.position_manager.swing_trailing_activation_pct

                if position.activate_trailing_stop(current_price, activation_pct):
                    await self.notifier.send_message(
                        f"📈 <b>트레일링 스탑 활성화</b>\n\n"
                        f"방향: {position.direction}\n"
                        f"현재가: ${current_price:,.2f}\n"
                        f"트레일링 거리: {position.trailing_stop_distance*100:.2f}%"
                    )

                # === UPDATE TRAILING STOP ===
                new_sl = position.update_trailing_stop(current_price)
                if new_sl:
                    logger.debug(
                        "Trailing stop updated",
                        position_id=position.id,
                        new_stop_loss=new_sl,
                    )

                should_exit = False
                exit_reason = ""

                # Check Stop Loss (may be trailing stop)
                if position.is_long:
                    # LONG: exit if price <= stop_loss
                    if current_price <= position.stop_loss:
                        should_exit = True
                        sl_type = "Trailing" if position.trailing_stop_activated else "Stop"
                        exit_reason = f"{sl_type} Loss triggered (${position.stop_loss:,.2f})"
                else:
                    # SHORT: exit if price >= stop_loss
                    if current_price >= position.stop_loss:
                        should_exit = True
                        sl_type = "Trailing" if position.trailing_stop_activated else "Stop"
                        exit_reason = f"{sl_type} Loss triggered (${position.stop_loss:,.2f})"

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
                if not should_exit and wick_signal:
                    if wick_signal.should_exit_on_time(self.settings.max_holding_minutes):
                        should_exit = True
                        exit_reason = f"Max holding time ({self.settings.max_holding_minutes}m) exceeded"

                # Check volume-based exit
                if not should_exit and wick_signal:
                    kline_1m = market_data.get("kline_1m")
                    if kline_1m and isinstance(kline_1m, dict):
                        current_volume = kline_1m.get("volume", 0)
                        if wick_signal.should_exit_on_volume(current_volume):
                            should_exit = True
                            exit_reason = "Volume exit trigger"

                if should_exit:
                    logger.info(
                        "Exiting position",
                        position_id=position.id,
                        symbol=symbol,
                        reason=exit_reason,
                    )

                    # Close position at market
                    exit_side = "SELL" if position.is_long else "BUY"
                    await self.executor.place_market_order(
                        side=exit_side,
                        quantity=position.quantity,
                        symbol=symbol,
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

                    # Save trade to database
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
        # Check exit conditions first (always process exits even during warmup)
        await self._check_exit_conditions(market_data)

        # Skip new signal generation during warmup period
        if self._start_time:
            elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            if elapsed < self._warmup_seconds:
                return
            elif not self._warmup_logged:
                self._warmup_logged = True
                logger.info("Warmup complete, signal generation enabled")

        # Check if we can open new position
        if self.position_manager and len(self.position_manager.active_positions) >= self.settings.max_concurrent_positions:
            return

        # Generate and execute signal
        signal = await self._generate_signal(market_data)
        if signal:
            await self._execute_signal(signal, market_data)

    async def _update_positions(self, market_data: dict[str, Any]) -> None:
        """Update position tracking with current prices for a specific symbol.

        Args:
            market_data: Current market data (includes symbol)
        """
        if not self.position_manager:
            return

        current_price = market_data.get("current_price", 0)
        symbol = market_data.get("symbol")
        if current_price <= 0 or not symbol:
            return

        # Update trailing stops for positions of this symbol
        for position in self.position_manager.active_positions:
            if position.symbol != symbol:
                continue

            # Use position's own activation threshold based on type
            if position.is_scalp:
                activation_pct = self.position_manager.scalp_trailing_activation_pct
            else:
                activation_pct = self.position_manager.swing_trailing_activation_pct

            # Activate trailing stop if threshold reached
            position.activate_trailing_stop(current_price, activation_pct)
            # Update trailing stop
            position.update_trailing_stop(current_price)

            # Log position status
            pnl = position.calculate_unrealized_pnl(current_price)
            pnl_pct = position.calculate_pnl_pct(current_price)
            logger.debug(
                "Position status",
                position_id=position.id,
                symbol=position.symbol,
                direction=position.direction,
                unrealized_pnl=f"{pnl:.2f}",
                pnl_pct=f"{pnl_pct*100:.2f}%",
            )

    async def _main_loop(self) -> None:
        """Main trading loop - processes all symbols."""
        logger.debug("Starting main trading loop", symbols=self.symbols)

        while self._running and not self._shutdown_event.is_set():
            try:
                # Process each symbol
                for symbol in self.symbols:
                    # Collect market data for this symbol
                    market_data = await self._collect_market_data(symbol)

                    if not market_data or not market_data.get("current_price"):
                        continue

                    # Update positions for this symbol
                    await self._update_positions(market_data)

                    # Check for new signals
                    now = datetime.now(timezone.utc)
                    if (now - self._last_signal_check).total_seconds() >= self._signal_check_interval:
                        await self._process_signals(market_data)

                # Update last signal check after processing all symbols
                now = datetime.now(timezone.utc)
                if (now - self._last_signal_check).total_seconds() >= self._signal_check_interval:
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
            self._start_time = datetime.now(timezone.utc)

            # Start Telegram bot command polling
            self.telegram_bot.start_polling()

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

        # Stop Telegram bot polling
        await self.telegram_bot.stop_polling()

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

    async def _telegram_stop(self) -> None:
        """Handle stop command from Telegram."""
        logger.info("Stop command received from Telegram")
        await self.stop()


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
