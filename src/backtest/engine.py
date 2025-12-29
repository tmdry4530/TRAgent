"""Backtesting engine for strategy validation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.signals.base import Signal


@dataclass
class Trade:
    """Completed trade record.

    Attributes:
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        direction: Trade direction (LONG or SHORT)
        entry_price: Actual entry price (with slippage)
        exit_price: Actual exit price (with slippage)
        size: Position size in base currency
        pnl: Profit/loss in quote currency
        pnl_pct: Profit/loss percentage
        signal_type: Signal type that triggered this trade (SCALP or SWING)
        exit_reason: Reason for exit (TP, SL, or SIGNAL)
    """

    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    signal_type: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Backtest performance metrics.

    Attributes:
        trades: List of all completed trades
        total_return: Total return percentage
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / Gross loss
        max_drawdown: Maximum drawdown percentage
        sharpe_ratio: Sharpe ratio (annualized)
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        avg_win: Average winning trade PnL
        avg_loss: Average losing trade PnL
        max_win: Largest winning trade PnL
        max_loss: Largest losing trade PnL
        avg_hold_time: Average holding time per trade
        total_commission: Total commission paid
        initial_capital: Starting capital
        final_capital: Ending capital
    """

    trades: list[Trade]
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_hold_time: float
    total_commission: float
    initial_capital: float
    final_capital: float

    def print_summary(self) -> None:
        """Print formatted backtest summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Total Trades:      {self.total_trades}")
        print(f"Winning Trades:    {self.winning_trades} ({self.win_rate:.2f}%)")
        print(f"Losing Trades:     {self.losing_trades}")
        print(f"\nTotal Return:      {self.total_return:.2f}%")
        print(f"Initial Capital:   ${self.initial_capital:,.2f}")
        print(f"Final Capital:     ${self.final_capital:,.2f}")
        print(f"\nProfit Factor:     {self.profit_factor:.2f}")
        print(f"Max Drawdown:      {self.max_drawdown:.2f}%")
        print(f"Sharpe Ratio:      {self.sharpe_ratio:.2f}")
        print(f"\nAvg Win:           ${self.avg_win:.2f}")
        print(f"Avg Loss:          ${self.avg_loss:.2f}")
        print(f"Max Win:           ${self.max_win:.2f}")
        print(f"Max Loss:          ${self.max_loss:.2f}")
        print(f"\nAvg Hold Time:     {self.avg_hold_time:.2f} hours")
        print(f"Total Commission:  ${self.total_commission:.2f}")
        print("=" * 60 + "\n")


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates trading with realistic slippage, commission, and position management.
    Supports both SCALP and SWING signals with proper conflict resolution.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0004,  # 0.04% taker fee
        slippage: float = 0.0001,  # 0.01% slippage
        risk_per_trade: float = 0.01,  # Risk 1% of capital per trade
        max_position_pct: float = 0.05,  # Max 5% of capital per position
        min_confidence: float = 0.65,  # Minimum signal confidence to trade
        fixed_position_value: float | None = None,  # Fixed position value (overrides risk calc)
        partial_tp_enabled: bool = True,  # Phase 2: Enable partial take profit
        partial_tp_ratio: float = 0.5,  # Take 50% profit at R:R 1:2
        partial_tp_rr: float = 2.0,  # R:R ratio for partial TP
        move_sl_to_be: bool = True,  # Move SL to breakeven after partial TP
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital in quote currency (USDT)
            commission: Commission rate per trade (0.0004 = 0.04%)
            slippage: Slippage rate per trade (0.0001 = 0.01%)
            risk_per_trade: Fraction of capital to risk per trade (0.01 = 1%)
            max_position_pct: Maximum position size as fraction of capital (0.05 = 5%)
            min_confidence: Minimum signal confidence to execute (0.65 = 65%)
            fixed_position_value: Fixed USD amount per trade (if set, overrides risk calc)
            partial_tp_enabled: Enable partial take profit at R:R milestone
            partial_tp_ratio: Fraction of position to close at partial TP (0.5 = 50%)
            partial_tp_rr: Risk:Reward ratio for partial TP trigger (2.0 = 1:2)
            move_sl_to_be: Move stop loss to breakeven after partial TP
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        self.fixed_position_value = fixed_position_value

        # Phase 2: Partial take profit settings
        self.partial_tp_enabled = partial_tp_enabled
        self.partial_tp_ratio = partial_tp_ratio
        self.partial_tp_rr = partial_tp_rr
        self.move_sl_to_be = move_sl_to_be

        self.capital = initial_capital
        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[Trade] = []
        self.current_position: dict | None = None

    def run(
        self,
        data: pd.DataFrame,
        signals: list[Signal],
    ) -> BacktestResult:
        """
        Run backtest on historical data with given signals.

        Args:
            data: DataFrame with OHLCV data (must have timestamp, open, high, low, close columns)
            signals: List of Signal objects to backtest

        Returns:
            BacktestResult with performance metrics

        Example:
            >>> engine = BacktestEngine(initial_capital=10000)
            >>> result = engine.run(df, signals)
            >>> result.print_summary()
        """
        logger.info(f"Starting backtest with {len(signals)} signals on {len(data)} bars")

        # Reset state
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.current_position = None

        # Sort signals by timestamp
        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        # Iterate through each bar
        for idx, row in data.iterrows():
            current_time = row["timestamp"]
            current_price = row["close"]

            # Check if current position hit SL/TP
            if self.current_position:
                self._check_exit(row)

            # Process signals at this timestamp
            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                # Only process valid signals
                if not signal.validate():
                    logger.warning(f"Invalid signal at {signal.timestamp}: {signal.reason}")
                    continue

                # Check if we can enter (no conflicting position)
                if self._can_enter(signal):
                    self._enter_position(signal, current_price, current_time)

            # Update equity curve
            equity = self.capital
            if self.current_position:
                equity += self._calculate_unrealized_pnl(current_price)
            self.equity_curve.append(equity)

        # Close any open position at the end
        if self.current_position:
            last_row = data.iloc[-1]
            self._exit_position(last_row["close"], last_row["timestamp"], "END")

        # Calculate metrics
        result = self._calculate_metrics()

        logger.info(
            f"Backtest complete: {result.total_trades} trades, "
            f"{result.total_return:.2f}% return, {result.win_rate:.2f}% win rate"
        )

        return result

    def _can_enter(self, signal: Signal) -> bool:
        """Check if we can enter a new position.

        Applies confidence filter and position checks.
        """
        # Only one position at a time - no pyramiding
        if self.current_position:
            return False

        # Confidence filter - reject low confidence signals
        if signal.confidence < self.min_confidence:
            logger.debug(
                f"Signal rejected: confidence {signal.confidence:.2f} < {self.min_confidence:.2f}"
            )
            return False

        return True

    def _enter_position(self, signal: Signal, current_price: float, current_time: datetime) -> None:
        """Enter a new position based on signal.

        Uses risk-based position sizing:
        - Calculates position size based on risk_per_trade and stop loss distance
        - Adjusts size based on signal confidence
        - Caps at max_position_pct of capital
        """
        # Calculate entry price with slippage
        entry_price = self._apply_slippage(signal.entry_price, signal.direction, "ENTRY")

        # Calculate position size using risk-based sizing
        if self.fixed_position_value is not None:
            # Use fixed position value
            position_value = min(self.fixed_position_value, self.capital * self.max_position_pct)
        else:
            # Risk-based position sizing
            position_value = self._calculate_risk_based_position(
                entry_price=entry_price,
                stop_loss=signal.stop_loss,
                confidence=signal.confidence,
            )

        # Ensure we have enough capital
        if position_value <= 0 or self.capital <= 0:
            return

        # Cap position at available capital
        position_value = min(position_value, self.capital * 0.99)  # Keep 1% buffer
        size = position_value / entry_price

        # Apply commission
        commission_cost = self._apply_commission(position_value)

        # Deduct position value AND commission from capital (margin used)
        self.capital -= (position_value + commission_cost)

        self.current_position = {
            "signal": signal,
            "entry_price": entry_price,
            "entry_time": current_time,
            "size": size,
            "position_value": position_value,  # Track original position value
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "commission": commission_cost,
        }

        logger.debug(
            f"Entered {signal.direction} position at {entry_price:.2f} "
            f"(size: {size:.4f}, value: ${position_value:.2f}, "
            f"SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}, "
            f"confidence: {signal.confidence:.2f})"
        )

    def _calculate_risk_based_position(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float,
    ) -> float:
        """Calculate position size based on risk management.

        Args:
            entry_price: Entry price after slippage
            stop_loss: Stop loss price
            confidence: Signal confidence (0-1)

        Returns:
            Position value in quote currency (USDT)
        """
        # Calculate stop loss distance as percentage
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price

        if sl_distance_pct == 0:
            return 0.0

        # Base risk amount (1% of capital by default)
        base_risk = self.capital * self.risk_per_trade

        # Adjust risk based on confidence
        # Low confidence (0.65-0.75): use 50% of risk
        # Medium confidence (0.75-0.85): use 75% of risk
        # High confidence (0.85+): use 100% of risk
        if confidence < 0.75:
            confidence_multiplier = 0.5
        elif confidence < 0.85:
            confidence_multiplier = 0.75
        else:
            confidence_multiplier = 1.0

        adjusted_risk = base_risk * confidence_multiplier

        # Position size = Risk amount / SL distance percentage
        position_value = adjusted_risk / sl_distance_pct

        # Cap at max position percentage
        max_position_value = self.capital * self.max_position_pct
        position_value = min(position_value, max_position_value)

        logger.debug(
            f"Risk-based sizing: risk=${adjusted_risk:.2f}, "
            f"sl_dist={sl_distance_pct:.4f}, pos_value=${position_value:.2f}, "
            f"conf_mult={confidence_multiplier}"
        )

        return position_value

    def _check_exit(self, row: pd.Series) -> None:
        """Check if current position should exit via SL/TP or partial TP."""
        if not self.current_position:
            return

        direction = self.current_position["direction"]
        stop_loss = self.current_position["stop_loss"]
        take_profit = self.current_position["take_profit"]
        entry_price = self.current_position["entry_price"]

        high = row["high"]
        low = row["low"]
        close = row["close"]
        timestamp = row["timestamp"]

        # Phase 2: Check partial take profit first
        if self.partial_tp_enabled and not self.current_position.get("partial_closed"):
            self._check_partial_tp(row, timestamp)

        # Check stop loss (may have moved to breakeven after partial TP)
        stop_loss = self.current_position["stop_loss"]  # Re-read in case it changed

        if direction == "LONG" and low <= stop_loss:
            self._exit_position(stop_loss, timestamp, "SL")
        elif direction == "SHORT" and high >= stop_loss:
            self._exit_position(stop_loss, timestamp, "SL")
        # Check take profit
        elif direction == "LONG" and high >= take_profit:
            self._exit_position(take_profit, timestamp, "TP")
        elif direction == "SHORT" and low <= take_profit:
            self._exit_position(take_profit, timestamp, "TP")

    def _check_partial_tp(self, row: pd.Series, timestamp: datetime) -> None:
        """Check and execute partial take profit at R:R milestone.

        Phase 2: When price reaches R:R 1:2, close 50% of position
        and move stop loss to breakeven.
        """
        if not self.current_position or self.current_position.get("partial_closed"):
            return

        pos = self.current_position
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        stop_loss = pos["stop_loss"]
        size = pos["size"]

        # Calculate risk distance
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return

        # Calculate partial TP price (R:R ratio)
        if direction == "LONG":
            partial_tp_price = entry_price + (risk * self.partial_tp_rr)
            triggered = row["high"] >= partial_tp_price
        else:
            partial_tp_price = entry_price - (risk * self.partial_tp_rr)
            triggered = row["low"] <= partial_tp_price

        if triggered:
            # Close partial position
            partial_size = size * self.partial_tp_ratio
            exit_price = self._apply_slippage(partial_tp_price, direction, "EXIT")

            # Calculate partial PnL
            if direction == "LONG":
                partial_pnl = (exit_price - entry_price) * partial_size
            else:
                partial_pnl = (entry_price - exit_price) * partial_size

            # Apply commission
            exit_value = exit_price * partial_size
            commission = self._apply_commission(exit_value)
            partial_pnl -= commission

            # Return partial position value + PnL to capital
            partial_position_value = pos.get("position_value", entry_price * size) * self.partial_tp_ratio
            self.capital += partial_position_value + partial_pnl

            # Update position
            self.current_position["size"] = size - partial_size
            self.current_position["partial_closed"] = True
            self.current_position["partial_pnl"] = partial_pnl

            # Move stop loss to breakeven if enabled
            if self.move_sl_to_be:
                self.current_position["stop_loss"] = entry_price

            logger.debug(
                f"Partial TP executed: {self.partial_tp_ratio*100:.0f}% at {exit_price:.2f}, "
                f"PnL ${partial_pnl:.2f}, SL moved to BE: {entry_price:.2f}"
            )

    def _exit_position(self, exit_price: float, exit_time: datetime, reason: str) -> None:
        """Exit current position and record trade.

        Phase 2: Includes partial TP PnL if applicable.
        """
        if not self.current_position:
            return

        pos = self.current_position
        direction = pos["direction"]
        original_size = pos["signal"].confidence  # Store for record

        # Apply slippage to exit
        exit_price = self._apply_slippage(exit_price, direction, "EXIT")

        # Calculate PnL for remaining position
        if direction == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]

        # Apply exit commission
        exit_value = exit_price * pos["size"]
        exit_commission = self._apply_commission(exit_value)

        remaining_pnl = pnl - exit_commission

        # Phase 2: Add partial TP PnL if any
        partial_pnl = pos.get("partial_pnl", 0)
        total_pnl = remaining_pnl + partial_pnl

        # Calculate total size for PnL percentage
        # If partial closed, we need original size for proper calculation
        if pos.get("partial_closed"):
            original_total_size = pos["size"] / (1 - self.partial_tp_ratio)
            original_position_value = pos.get("position_value", pos["entry_price"] * original_total_size)
            remaining_position_value = original_position_value * (1 - self.partial_tp_ratio)
        else:
            original_total_size = pos["size"]
            remaining_position_value = pos.get("position_value", pos["entry_price"] * pos["size"])

        pnl_pct = (total_pnl / pos.get("position_value", pos["entry_price"] * original_total_size)) * 100

        # Return remaining position value + PnL to capital
        self.capital += remaining_position_value + remaining_pnl

        # Build exit reason
        exit_reason = reason
        if pos.get("partial_closed"):
            exit_reason = f"PARTIAL+{reason}"

        # Record trade
        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            size=original_total_size,  # Record original size
            pnl=total_pnl,  # Total PnL including partial
            pnl_pct=pnl_pct,
            signal_type=pos["signal"].type,
            exit_reason=exit_reason,
        )
        self.trades.append(trade)

        logger.debug(
            f"Exited {direction} position at {exit_price:.2f} "
            f"({exit_reason}): PnL ${total_pnl:.2f} ({pnl_pct:.2f}%)"
            f"{f', partial: ${partial_pnl:.2f}' if partial_pnl else ''}"
        )

        self.current_position = None

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL for current position."""
        if not self.current_position:
            return 0.0

        pos = self.current_position
        if pos["direction"] == "LONG":
            return (current_price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - current_price) * pos["size"]

    def _apply_slippage(self, price: float, direction: str, side: str) -> float:
        """
        Apply slippage to price.

        Args:
            price: Original price
            direction: Position direction (LONG or SHORT)
            side: ENTRY or EXIT

        Returns:
            Price with slippage applied
        """
        # Entry: pay more for LONG, receive less for SHORT
        # Exit: receive less for LONG, pay more for SHORT
        if (direction == "LONG" and side == "ENTRY") or (direction == "SHORT" and side == "EXIT"):
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _apply_commission(self, value: float) -> float:
        """Calculate commission cost."""
        return value * self.commission

    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics from completed trades."""
        if not self.trades:
            logger.warning("No trades completed")
            return BacktestResult(
                trades=[],
                total_return=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                avg_hold_time=0.0,
                total_commission=0.0,
                initial_capital=self.initial_capital,
                final_capital=self.capital,
            )

        # Basic counts
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl <= 0)

        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        # PnL stats
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Total return
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Average holding time
        hold_times = [
            (t.exit_time - t.entry_time).total_seconds() / 3600  # hours
            for t in self.trades
        ]
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0

        # Total commission
        total_commission = sum(
            self._apply_commission(t.entry_price * t.size) +
            self._apply_commission(t.exit_price * t.size)
            for t in self.trades
        )

        return BacktestResult(
            trades=self.trades,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            avg_hold_time=avg_hold_time,
            total_commission=total_commission,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100

        return float(np.max(drawdown))

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize (assuming hourly data)
        periods_per_year = 365 * 24
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

        return float(sharpe)


def run_simple_backtest(
    data: pd.DataFrame,
    signals: list[Signal],
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """
    Convenience function to run a simple backtest.

    Args:
        data: OHLCV DataFrame
        signals: List of signals
        initial_capital: Starting capital

    Returns:
        BacktestResult with metrics
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(data, signals)
    result.print_summary()
    return result
