"""Tests for position size calculator."""

import pytest

from src.risk.calculator import PositionCalculator
from src.signals.base import Signal
from datetime import datetime


@pytest.fixture
def config():
    """Trading configuration fixture."""
    return {
        "scalp": {
            "position_size": 0.30,
            "max_leverage": 40,
            "stop_loss": 0.015,
        },
        "swing": {
            "position_size": 0.50,
            "max_leverage": 10,
            "stop_loss": 0.05,
        },
        "risk": {
            "max_total_exposure": 0.80,
        },
    }


@pytest.fixture
def calculator(config):
    """PositionCalculator fixture."""
    return PositionCalculator(config)


@pytest.fixture
def scalp_signal():
    """Scalp signal fixture."""
    return Signal(
        type="SCALP",
        direction="LONG",
        confidence=0.8,
        entry_price=50000.0,
        stop_loss=49250.0,  # 1.5% stop loss
        take_profit=52500.0,
        reason="Test scalp signal",
        timestamp=datetime.now(),
    )


@pytest.fixture
def swing_signal():
    """Swing signal fixture."""
    return Signal(
        type="SWING",
        direction="LONG",
        confidence=0.75,
        entry_price=50000.0,
        stop_loss=47500.0,  # 5% stop loss
        take_profit=62500.0,
        reason="Test swing signal",
        timestamp=datetime.now(),
    )


class TestPositionCalculator:
    """Test PositionCalculator class."""

    def test_scalp_position_calculation(self, calculator, scalp_signal):
        """Test scalp position size calculation."""
        account_balance = 10000.0
        current_price = 50000.0

        result = calculator.calculate(
            signal=scalp_signal,
            account_balance=account_balance,
            current_price=current_price,
        )

        assert result is not None
        assert result.size_usd > 0
        assert result.size_btc > 0
        assert result.leverage > 0
        assert result.leverage <= calculator.scalp_max_leverage
        assert result.risk_amount > 0

        # Check that position doesn't exceed max
        max_position = account_balance * calculator.scalp_max_position_pct
        assert result.size_usd <= max_position * calculator.scalp_max_leverage

    def test_swing_position_calculation(self, calculator, swing_signal):
        """Test swing position size calculation."""
        account_balance = 10000.0
        current_price = 50000.0

        result = calculator.calculate(
            signal=swing_signal,
            account_balance=account_balance,
            current_price=current_price,
        )

        assert result is not None
        assert result.size_usd > 0
        assert result.size_btc > 0
        assert result.leverage > 0
        assert result.leverage <= calculator.swing_max_leverage
        assert result.risk_amount > 0

        # Swing should use lower leverage than scalp
        assert result.leverage <= 10

    def test_total_exposure_limit(self, calculator, scalp_signal):
        """Test total exposure limit enforcement."""
        account_balance = 10000.0
        current_price = 50000.0

        # Existing position at 70% exposure
        existing_positions = [
            {"notional": 7000.0}
        ]

        result = calculator.calculate(
            signal=scalp_signal,
            account_balance=account_balance,
            current_price=current_price,
            existing_positions=existing_positions,
        )

        # Should still allow small position (up to 10% remaining)
        assert result is not None
        assert result.size_usd > 0

    def test_total_exposure_exceeded(self, calculator, scalp_signal):
        """Test rejection when total exposure exceeded."""
        account_balance = 10000.0
        current_price = 50000.0

        # Existing position at 80% exposure (at limit)
        existing_positions = [
            {"notional": 8000.0}
        ]

        result = calculator.calculate(
            signal=scalp_signal,
            account_balance=account_balance,
            current_price=current_price,
            existing_positions=existing_positions,
        )

        # Should reject new position
        assert result is None

    def test_zero_stop_loss_uses_default(self, calculator):
        """Test that zero stop loss falls back to default."""
        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=50000.0,  # Same as entry (0% difference)
            take_profit=52500.0,
            reason="Test signal",
            timestamp=datetime.now(),
        )

        account_balance = 10000.0
        current_price = 50000.0

        result = calculator.calculate(
            signal=signal,
            account_balance=account_balance,
            current_price=current_price,
        )

        assert result is not None
        assert result.size_usd > 0

    def test_invalid_account_balance(self, calculator, scalp_signal):
        """Test handling of invalid account balance."""
        current_price = 50000.0

        result = calculator.calculate(
            signal=scalp_signal,
            account_balance=0.0,
            current_price=current_price,
        )

        # Should handle gracefully
        assert result is None or result.size_usd == 0

    def test_btc_size_calculation(self, calculator, scalp_signal):
        """Test BTC size is correctly calculated."""
        account_balance = 10000.0
        current_price = 50000.0

        result = calculator.calculate(
            signal=scalp_signal,
            account_balance=account_balance,
            current_price=current_price,
        )

        assert result is not None

        # BTC size should match USD size / price
        expected_btc = result.size_usd / current_price
        assert abs(result.size_btc - expected_btc) < 0.00001
