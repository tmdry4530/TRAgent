"""Tests for backtest engine."""

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from src.signals.base import Signal
from src.backtest.engine import BacktestEngine, Trade, BacktestResult


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = [base_time + timedelta(hours=i) for i in range(100)]

        # Generate simple price data with trend
        base_price = 50000.0
        prices = []
        for i in range(100):
            if i < 30:
                # Uptrend
                price = base_price + i * 50
            elif i < 60:
                # Downtrend
                price = base_price + 30 * 50 - (i - 30) * 60
            else:
                # Recovery
                price = base_price + 30 * 50 - 30 * 60 + (i - 60) * 40
            prices.append(price)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 100 for p in prices],
            "low": [p - 100 for p in prices],
            "close": prices,
            "volume": [1000.0] * 100,
        })
        return df

    @pytest.fixture
    def engine(self) -> BacktestEngine:
        """Create backtest engine."""
        return BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            slippage=0.0001,
        )

    def test_init(self, engine: BacktestEngine) -> None:
        """Test engine initialization."""
        assert engine.initial_capital == 10000.0
        assert engine.commission == 0.0004
        assert engine.slippage == 0.0001
        assert engine.capital == 10000.0

    def test_run_with_no_signals(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test running with no signals."""
        result = engine.run(sample_data, [])

        assert result.total_trades == 0
        assert result.total_return == 0.0
        assert result.final_capital == engine.initial_capital

    def test_run_with_winning_long(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test winning LONG trade."""
        entry_price = 50000.0
        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=entry_price,
            stop_loss=entry_price * 0.985,  # 1.5% SL
            take_profit=entry_price * 1.03,  # 3% TP
            reason="Test signal",
            timestamp=datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc),  # During uptrend
        )

        result = engine.run(sample_data, [signal])

        assert result.total_trades == 1
        assert result.winning_trades == 1 or result.trades[0].exit_reason == "END"

    def test_run_with_losing_short(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test losing SHORT trade during uptrend."""
        entry_price = 50500.0
        signal = Signal(
            type="SCALP",
            direction="SHORT",
            confidence=0.8,
            entry_price=entry_price,
            stop_loss=entry_price * 1.015,  # 1.5% SL (above for SHORT)
            take_profit=entry_price * 0.97,  # 3% TP (below for SHORT)
            reason="Test signal",
            timestamp=datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc),
        )

        result = engine.run(sample_data, [signal])

        assert result.total_trades == 1

    def test_slippage_applied(self, engine: BacktestEngine) -> None:
        """Test that slippage is applied correctly."""
        price = 50000.0

        # LONG entry: should pay more
        long_entry = engine._apply_slippage(price, "LONG", "ENTRY")
        assert long_entry > price

        # LONG exit: should receive less
        long_exit = engine._apply_slippage(price, "LONG", "EXIT")
        assert long_exit < price

        # SHORT entry: should receive more
        short_entry = engine._apply_slippage(price, "SHORT", "ENTRY")
        assert short_entry < price

        # SHORT exit: should pay more
        short_exit = engine._apply_slippage(price, "SHORT", "EXIT")
        assert short_exit > price

    def test_commission_applied(self, engine: BacktestEngine) -> None:
        """Test commission calculation."""
        value = 10000.0
        commission = engine._apply_commission(value)

        assert commission == value * 0.0004
        assert commission == 4.0

    def test_position_conflict_prevention(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test that conflicting positions are not allowed."""
        entry_price = 50000.0

        # Create two signals with opposite directions
        signal1 = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=entry_price,
            stop_loss=entry_price * 0.98,
            take_profit=entry_price * 1.10,  # Won't hit during test
            reason="Signal 1",
            timestamp=datetime(2024, 1, 1, 5, 0, 0, tzinfo=timezone.utc),
        )

        signal2 = Signal(
            type="SCALP",
            direction="SHORT",  # Opposite direction
            confidence=0.8,
            entry_price=entry_price + 1000,
            stop_loss=entry_price * 1.12,
            take_profit=entry_price * 0.95,
            reason="Signal 2",
            timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        )

        result = engine.run(sample_data, [signal1, signal2])

        # Second signal should not be entered due to conflict
        # Only 1 trade should occur
        assert result.total_trades == 1

    def test_multiple_signals(
        self, engine: BacktestEngine, sample_data: pd.DataFrame
    ) -> None:
        """Test running with multiple signals."""
        signals = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Create 3 sequential LONG signals
        for i in range(3):
            entry_price = 50000.0 + i * 100
            signal = Signal(
                type="SCALP",
                direction="LONG",
                confidence=0.8,
                entry_price=entry_price,
                stop_loss=entry_price * 0.98,
                take_profit=entry_price * 1.01,
                reason=f"Signal {i+1}",
                timestamp=base_time + timedelta(hours=i * 20),
            )
            signals.append(signal)

        result = engine.run(sample_data, signals)

        # At least 1 trade should complete
        assert result.total_trades >= 1

    def test_max_drawdown_calculation(self, engine: BacktestEngine) -> None:
        """Test max drawdown calculation."""
        # Simulate equity curve with known drawdown
        engine.equity_curve = [10000, 11000, 12000, 10800, 9600, 10500]

        mdd = engine._calculate_max_drawdown()

        # Max drawdown should be (12000 - 9600) / 12000 = 20%
        assert mdd == pytest.approx(20.0, rel=0.01)

    def test_sharpe_ratio_calculation(self, engine: BacktestEngine) -> None:
        """Test Sharpe ratio calculation."""
        # Simulate equity curve with positive returns
        equity = [10000.0]
        for _ in range(100):
            equity.append(equity[-1] * 1.001)  # 0.1% daily return
        engine.equity_curve = equity

        sharpe = engine._calculate_sharpe_ratio()

        # Should have positive Sharpe ratio
        assert sharpe > 0


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_empty_result(self) -> None:
        """Test creating empty result."""
        result = BacktestResult(
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
            initial_capital=10000.0,
            final_capital=10000.0,
        )

        assert result.total_trades == 0
        assert result.initial_capital == result.final_capital

    def test_result_with_trades(self) -> None:
        """Test result with completed trades."""
        trades = [
            Trade(
                entry_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                direction="LONG",
                entry_price=50000.0,
                exit_price=51000.0,
                size=0.1,
                pnl=100.0,
                pnl_pct=2.0,
                signal_type="SCALP",
                exit_reason="TP",
            ),
            Trade(
                entry_time=datetime(2024, 1, 1, 2, tzinfo=timezone.utc),
                exit_time=datetime(2024, 1, 1, 3, tzinfo=timezone.utc),
                direction="SHORT",
                entry_price=51000.0,
                exit_price=51500.0,
                size=0.1,
                pnl=-50.0,
                pnl_pct=-1.0,
                signal_type="SCALP",
                exit_reason="SL",
            ),
        ]

        result = BacktestResult(
            trades=trades,
            total_return=5.0,
            win_rate=50.0,
            profit_factor=2.0,
            max_drawdown=5.0,
            sharpe_ratio=1.5,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            avg_win=100.0,
            avg_loss=-50.0,
            max_win=100.0,
            max_loss=-50.0,
            avg_hold_time=1.0,
            total_commission=8.0,
            initial_capital=10000.0,
            final_capital=10500.0,
        )

        assert result.total_trades == 2
        assert result.win_rate == 50.0
        assert result.profit_factor == 2.0


class TestTrade:
    """Tests for Trade dataclass."""

    def test_create_trade(self) -> None:
        """Test creating a Trade instance."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            exit_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            direction="LONG",
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.1,
            pnl=100.0,
            pnl_pct=2.0,
            signal_type="SCALP",
            exit_reason="TP",
        )

        assert trade.direction == "LONG"
        assert trade.pnl == 100.0
        assert trade.exit_reason == "TP"
