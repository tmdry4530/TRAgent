"""Tests for SQLite trade database."""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile

from src.utils.database import TradeDatabase, TradeRecord, DailyStats


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_trades.db"
        db = TradeDatabase(db_path)
        yield db


@pytest.fixture
def sample_trade():
    """Create a sample trade record."""
    now = datetime.now(timezone.utc)
    return TradeRecord(
        id=None,
        symbol="BTCUSDT",
        direction="LONG",
        entry_price=100000.0,
        exit_price=100500.0,
        quantity=0.001,
        leverage=50,
        pnl=0.5,  # (100500 - 100000) * 0.001
        pnl_pct=0.5,  # 0.5%
        entry_time=now - timedelta(minutes=5),
        exit_time=now,
        exit_reason="Take Profit triggered",
        signal_reason="Wick reversal: UP trend",
    )


class TestTradeDatabase:
    """Tests for TradeDatabase class."""

    def test_init_creates_tables(self, temp_db):
        """Test that database initialization creates required tables."""
        with temp_db._get_connection() as conn:
            cursor = conn.cursor()

            # Check trades table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
            )
            assert cursor.fetchone() is not None

            # Check daily_stats table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_stats'"
            )
            assert cursor.fetchone() is not None

    def test_save_trade(self, temp_db, sample_trade):
        """Test saving a trade record."""
        trade_id = temp_db.save_trade(sample_trade)

        assert trade_id > 0

        # Verify trade was saved
        trades = temp_db.get_recent_trades(1)
        assert len(trades) == 1
        assert trades[0].symbol == "BTCUSDT"
        assert trades[0].direction == "LONG"
        assert trades[0].pnl == 0.5

    def test_get_daily_pnl(self, temp_db, sample_trade):
        """Test getting daily PnL."""
        # Initially should be 0
        assert temp_db.get_daily_pnl() == 0.0

        # Save a winning trade
        temp_db.save_trade(sample_trade)
        assert temp_db.get_daily_pnl() == 0.5

        # Save a losing trade
        losing_trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="SHORT",
            entry_price=100000.0,
            exit_price=100200.0,
            quantity=0.001,
            leverage=50,
            pnl=-0.2,
            pnl_pct=-0.2,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
            exit_time=datetime.now(timezone.utc),
            exit_reason="Stop Loss triggered",
            signal_reason="Test",
        )
        temp_db.save_trade(losing_trade)

        # Total should be 0.5 - 0.2 = 0.3
        assert temp_db.get_daily_pnl() == pytest.approx(0.3, abs=0.01)

    def test_get_daily_stats(self, temp_db, sample_trade):
        """Test getting daily statistics."""
        # No stats before trades
        assert temp_db.get_daily_stats() is None

        # Save trades
        temp_db.save_trade(sample_trade)

        losing_trade = TradeRecord(
            id=None,
            symbol="BTCUSDT",
            direction="SHORT",
            entry_price=100000.0,
            exit_price=100200.0,
            quantity=0.001,
            leverage=50,
            pnl=-0.2,
            pnl_pct=-0.2,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
            exit_time=datetime.now(timezone.utc),
            exit_reason="Stop Loss",
            signal_reason="Test",
        )
        temp_db.save_trade(losing_trade)

        stats = temp_db.get_daily_stats()
        assert stats is not None
        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.win_rate == 0.5
        assert stats.total_pnl == pytest.approx(0.3, abs=0.01)

    def test_get_trades_today(self, temp_db, sample_trade):
        """Test getting today's trades."""
        temp_db.save_trade(sample_trade)

        trades = temp_db.get_trades_today()
        assert len(trades) == 1
        assert trades[0].direction == "LONG"

    def test_get_recent_trades(self, temp_db, sample_trade):
        """Test getting recent trades with limit."""
        # Save 5 trades
        for i in range(5):
            trade = TradeRecord(
                id=None,
                symbol="BTCUSDT",
                direction="LONG",
                entry_price=100000.0 + i * 100,
                exit_price=100100.0 + i * 100,
                quantity=0.001,
                leverage=50,
                pnl=0.1,
                pnl_pct=0.1,
                entry_time=datetime.now(timezone.utc) - timedelta(minutes=5-i),
                exit_time=datetime.now(timezone.utc) - timedelta(minutes=4-i),
                exit_reason="Test",
                signal_reason="Test",
            )
            temp_db.save_trade(trade)

        # Get last 3
        trades = temp_db.get_recent_trades(3)
        assert len(trades) == 3

    def test_get_all_time_stats(self, temp_db, sample_trade):
        """Test getting all-time statistics."""
        # Empty stats
        stats = temp_db.get_all_time_stats()
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

        # Add trades
        temp_db.save_trade(sample_trade)

        stats = temp_db.get_all_time_stats()
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1
        assert stats["win_rate"] == 1.0
        assert stats["total_pnl"] == 0.5

    def test_max_drawdown_calculation(self, temp_db):
        """Test max drawdown calculation in daily stats."""
        now = datetime.now(timezone.utc)

        # Create trades that form a drawdown pattern: +1, -2, +0.5
        trades = [
            TradeRecord(
                id=None, symbol="BTCUSDT", direction="LONG",
                entry_price=100000, exit_price=101000, quantity=0.001,
                leverage=50, pnl=1.0, pnl_pct=1.0,
                entry_time=now - timedelta(minutes=15),
                exit_time=now - timedelta(minutes=10),
                exit_reason="TP", signal_reason="Test",
            ),
            TradeRecord(
                id=None, symbol="BTCUSDT", direction="LONG",
                entry_price=101000, exit_price=99000, quantity=0.001,
                leverage=50, pnl=-2.0, pnl_pct=-2.0,
                entry_time=now - timedelta(minutes=10),
                exit_time=now - timedelta(minutes=5),
                exit_reason="SL", signal_reason="Test",
            ),
            TradeRecord(
                id=None, symbol="BTCUSDT", direction="LONG",
                entry_price=99000, exit_price=99500, quantity=0.001,
                leverage=50, pnl=0.5, pnl_pct=0.5,
                entry_time=now - timedelta(minutes=5),
                exit_time=now,
                exit_reason="TP", signal_reason="Test",
            ),
        ]

        for trade in trades:
            temp_db.save_trade(trade)

        stats = temp_db.get_daily_stats()
        assert stats is not None
        # Cumulative: 1, -1, -0.5
        # Peak was 1, dropped to -1, so max drawdown = 2
        assert stats.max_drawdown == pytest.approx(2.0, abs=0.01)
