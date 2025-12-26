"""Tests for daily loss limit checker."""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from src.risk.daily_limit import DailyLossLimitChecker
from src.utils.database import TradeDatabase, TradeRecord


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_trades.db"
        db = TradeDatabase(db_path)
        yield db


@pytest.fixture
def limit_checker(temp_db):
    """Create a daily limit checker with test database."""
    with patch("src.risk.daily_limit.get_database", return_value=temp_db):
        checker = DailyLossLimitChecker(
            loss_limit_pct=0.05,  # 5% limit
            account_balance=1000.0,  # $1000 account
        )
        yield checker


class TestDailyLossLimitChecker:
    """Tests for DailyLossLimitChecker class."""

    def test_init(self, limit_checker):
        """Test initialization."""
        assert limit_checker.loss_limit_pct == 0.05
        assert limit_checker._account_balance == 1000.0

    def test_get_loss_limit_usd(self, limit_checker):
        """Test calculating loss limit in USD."""
        # 5% of $1000 = $50
        assert limit_checker.get_loss_limit_usd() == 50.0

    def test_set_account_balance(self, limit_checker):
        """Test updating account balance."""
        limit_checker.set_account_balance(2000.0)
        # 5% of $2000 = $100
        assert limit_checker.get_loss_limit_usd() == 100.0

    def test_check_can_trade_no_trades(self, limit_checker):
        """Test that trading is allowed with no trades."""
        can_trade, reason = limit_checker.check_can_trade()
        assert can_trade is True
        assert reason == ""

    def test_check_can_trade_with_profit(self, temp_db, limit_checker):
        """Test that trading is allowed when in profit."""
        # Add a winning trade
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=101000.0,
            quantity=0.001,
            leverage=50,
            pnl=1.0,  # $1 profit
            pnl_pct=1.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="TP",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        can_trade, reason = limit_checker.check_can_trade()
        assert can_trade is True

    def test_check_can_trade_under_limit(self, temp_db, limit_checker):
        """Test that trading is allowed when loss is under limit."""
        # Add a losing trade ($30 loss, limit is $50)
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=97000.0,
            quantity=0.01,
            leverage=50,
            pnl=-30.0,
            pnl_pct=-3.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        can_trade, reason = limit_checker.check_can_trade()
        assert can_trade is True

    def test_check_can_trade_at_limit(self, temp_db, limit_checker):
        """Test that trading is blocked when loss equals limit."""
        # Add a losing trade ($50 loss = exactly at limit)
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=95000.0,
            quantity=0.01,
            leverage=50,
            pnl=-50.0,
            pnl_pct=-5.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        can_trade, reason = limit_checker.check_can_trade()
        assert can_trade is False
        assert "Daily loss limit exceeded" in reason

    def test_check_can_trade_over_limit(self, temp_db, limit_checker):
        """Test that trading is blocked when loss exceeds limit."""
        # Add a losing trade ($60 loss > $50 limit)
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=94000.0,
            quantity=0.01,
            leverage=50,
            pnl=-60.0,
            pnl_pct=-6.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        can_trade, reason = limit_checker.check_can_trade()
        assert can_trade is False
        assert "Daily loss limit exceeded" in reason

    def test_is_blocked_persists(self, temp_db, limit_checker):
        """Test that blocked state persists."""
        # Add a losing trade
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=94000.0,
            quantity=0.01,
            leverage=50,
            pnl=-60.0,
            pnl_pct=-6.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        # Check once - should block
        can_trade, _ = limit_checker.check_can_trade()
        assert can_trade is False

        # Check again - should still be blocked
        assert limit_checker.is_blocked() is True

    def test_get_remaining_loss_budget_no_trades(self, limit_checker):
        """Test remaining budget with no trades."""
        # Full limit available
        assert limit_checker.get_remaining_loss_budget() == 50.0

    def test_get_remaining_loss_budget_with_loss(self, temp_db, limit_checker):
        """Test remaining budget after loss."""
        # Add a $20 loss
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=98000.0,
            quantity=0.01,
            leverage=50,
            pnl=-20.0,
            pnl_pct=-2.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        # Remaining should be $50 - $20 = $30
        assert limit_checker.get_remaining_loss_budget() == pytest.approx(30.0, abs=0.01)

    def test_get_remaining_loss_budget_with_profit(self, temp_db, limit_checker):
        """Test remaining budget in profit."""
        # Add a $10 profit
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=101000.0,
            quantity=0.01,
            leverage=50,
            pnl=10.0,
            pnl_pct=1.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="TP",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        # Full limit still available when in profit
        assert limit_checker.get_remaining_loss_budget() == 50.0

    def test_get_status(self, temp_db, limit_checker):
        """Test getting status dictionary."""
        status = limit_checker.get_status()

        assert "daily_pnl" in status
        assert "loss_limit" in status
        assert "remaining_budget" in status
        assert "is_blocked" in status
        assert status["loss_limit"] == 50.0
        assert status["is_blocked"] is False

    def test_reset(self, temp_db, limit_checker):
        """Test manual reset."""
        # Block by exceeding limit
        trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="LONG",
            entry_price=100000.0,
            exit_price=94000.0,
            quantity=0.01,
            leverage=50,
            pnl=-60.0,
            pnl_pct=-6.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            exit_time=datetime.now(timezone.utc),
            exit_reason="SL",
            signal_reason="Test",
        )
        temp_db.save_trade(trade)

        limit_checker.check_can_trade()
        assert limit_checker.is_blocked() is True

        # Reset
        limit_checker.reset()
        assert limit_checker._is_blocked is False
