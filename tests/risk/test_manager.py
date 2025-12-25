"""Tests for risk manager."""

import pytest
from datetime import datetime, timedelta

from src.risk.manager import RiskManager, TradeResult
from src.signals.base import Signal


@pytest.fixture
def config():
    """Trading configuration fixture."""
    return {
        "risk": {
            "daily_loss_limit": 0.10,
            "consecutive_loss_cooldown": 3,
            "event_blackout_minutes": 30,
            "llm_confidence_threshold": 0.6,
        },
    }


@pytest.fixture
def risk_manager(config):
    """RiskManager fixture."""
    return RiskManager(config)


@pytest.fixture
def scalp_signal():
    """Scalp signal fixture."""
    return Signal(
        type="SCALP",
        direction="LONG",
        confidence=0.8,
        entry_price=50000.0,
        stop_loss=49250.0,
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
        stop_loss=47500.0,
        take_profit=62500.0,
        reason="Test swing signal",
        timestamp=datetime.now(),
    )


class TestRiskManager:
    """Test RiskManager class."""

    def test_check_passes_for_valid_signal(self, risk_manager, scalp_signal):
        """Test that valid signal passes risk check."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
            "llm_confidence": 0.8,
        }

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is True
        assert result.reason is None
        assert result.adjusted_size == 1.0

    def test_daily_loss_limit_blocks_trade(self, risk_manager, scalp_signal):
        """Test that daily loss limit blocks new trades."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
        }

        # Simulate losses reaching limit
        risk_manager.daily_pnl = -1000.0  # 10% loss

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is False
        assert "loss limit" in result.reason.lower()

    def test_cooldown_blocks_trade(self, risk_manager, scalp_signal):
        """Test that active cooldown blocks new trades."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
        }

        # Set cooldown
        risk_manager.cooldown_until = datetime.now() + timedelta(minutes=30)

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is False
        assert "cooldown" in result.reason.lower()
        assert result.cooldown_until is not None

    def test_event_blackout_blocks_trade(self, risk_manager, scalp_signal):
        """Test that event blackout blocks new trades."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
        }

        # Add upcoming event within blackout window
        upcoming_event_time = datetime.now() + timedelta(minutes=15)
        risk_manager.update_events([
            {
                "name": "FOMC",
                "time": upcoming_event_time.isoformat(),
            }
        ])

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is False
        assert "blackout" in result.reason.lower()

    def test_low_llm_confidence_reduces_size(self, risk_manager, scalp_signal):
        """Test that low LLM confidence reduces position size."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
            "llm_confidence": 0.5,  # Below 0.6 threshold
        }

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is True
        assert result.adjusted_size == 0.5
        assert len(result.warnings) > 0

    def test_consecutive_losses_reduces_size(self, risk_manager, scalp_signal):
        """Test that consecutive losses reduce position size."""
        account_state = {
            "balance": 10000.0,
            "positions": [],
        }

        # Record 2 consecutive losses
        risk_manager.consecutive_losses = 2

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is True
        assert result.adjusted_size == 0.7
        assert len(result.warnings) > 0

    def test_consecutive_losses_trigger_cooldown(self, risk_manager):
        """Test that max consecutive losses trigger cooldown."""
        trade_result = TradeResult(
            timestamp=datetime.now(),
            pnl=-100.0,
            pnl_pct=-1.5,
            signal_type="SCALP",
            direction="LONG",
        )

        # Record 3 consecutive losses
        for _ in range(3):
            risk_manager.record_trade_result(trade_result)

        assert risk_manager.consecutive_losses == 3
        assert risk_manager.cooldown_until is not None

    def test_win_resets_consecutive_losses(self, risk_manager):
        """Test that a win resets consecutive loss counter."""
        loss_result = TradeResult(
            timestamp=datetime.now(),
            pnl=-100.0,
            pnl_pct=-1.5,
            signal_type="SCALP",
            direction="LONG",
        )

        win_result = TradeResult(
            timestamp=datetime.now(),
            pnl=200.0,
            pnl_pct=3.0,
            signal_type="SCALP",
            direction="LONG",
        )

        # Record 2 losses then a win
        risk_manager.record_trade_result(loss_result)
        risk_manager.record_trade_result(loss_result)
        assert risk_manager.consecutive_losses == 2

        risk_manager.record_trade_result(win_result)
        assert risk_manager.consecutive_losses == 0

    def test_position_conflict_same_type(self, risk_manager, scalp_signal):
        """Test that existing position of same type blocks new trade."""
        account_state = {
            "balance": 10000.0,
            "positions": [
                {
                    "type": "SCALP",
                    "direction": "LONG",
                    "notional": 3000.0,
                }
            ],
        }

        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is False
        assert "existing" in result.reason.lower()

    def test_position_conflict_opposite_direction(self, risk_manager, scalp_signal):
        """Test that opposite direction position blocks new trade."""
        account_state = {
            "balance": 10000.0,
            "positions": [
                {
                    "type": "SWING",
                    "direction": "SHORT",
                    "notional": 5000.0,
                }
            ],
        }

        # Try to open LONG position
        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is False
        assert "opposite" in result.reason.lower() or "short" in result.reason.lower()

    def test_same_direction_different_type_allowed(self, risk_manager, scalp_signal):
        """Test that same direction, different type is allowed."""
        account_state = {
            "balance": 10000.0,
            "positions": [
                {
                    "type": "SWING",
                    "direction": "LONG",
                    "notional": 5000.0,
                }
            ],
        }

        # SCALP LONG should be allowed with SWING LONG
        result = risk_manager.check(scalp_signal, account_state)

        assert result.approved is True

    def test_invalid_signal_rejected(self, risk_manager):
        """Test that invalid signal is rejected."""
        # Invalid signal with stop loss above entry for LONG
        invalid_signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=51000.0,  # Invalid: above entry for LONG
            take_profit=52500.0,
            reason="Invalid signal",
            timestamp=datetime.now(),
        )

        account_state = {
            "balance": 10000.0,
            "positions": [],
        }

        result = risk_manager.check(invalid_signal, account_state)

        assert result.approved is False
        assert "invalid" in result.reason.lower()

    def test_daily_reset(self, risk_manager):
        """Test daily statistics reset."""
        risk_manager.daily_pnl = -500.0
        initial_date = risk_manager.daily_start.date()

        # Manually trigger reset
        risk_manager.reset_daily_stats()

        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.daily_start.date() >= initial_date

    def test_get_stats(self, risk_manager):
        """Test statistics retrieval."""
        stats = risk_manager.get_stats()

        assert "daily_pnl" in stats
        assert "consecutive_losses" in stats
        assert "cooldown_until" in stats
        assert "total_trades_today" in stats
        assert "event_blackout" in stats

    def test_event_update(self, risk_manager):
        """Test event list update."""
        events = [
            {
                "name": "FOMC",
                "time": (datetime.now() + timedelta(hours=2)).isoformat(),
            },
            {
                "name": "CPI",
                "time": (datetime.now() + timedelta(days=1)).isoformat(),
            },
        ]

        risk_manager.update_events(events)

        assert len(risk_manager.upcoming_events) == 2
        assert risk_manager.upcoming_events[0]["name"] == "FOMC"
