"""Unit tests for PositionManager."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.executor.position import (
    Position,
    PositionManager,
    PositionState,
)
from src.signals.base import Signal


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return Signal(
        type="SCALP",
        direction="LONG",
        confidence=0.85,
        entry_price=50000.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        reason="Test signal",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def swing_signal():
    """Create a sample swing signal."""
    return Signal(
        type="SWING",
        direction="LONG",
        confidence=0.80,
        entry_price=50000.0,
        stop_loss=47500.0,
        take_profit=57500.0,
        reason="Test swing signal",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def position_manager():
    """Create a PositionManager instance."""
    return PositionManager(
        max_scalp_positions=1,
        max_swing_positions=1,
        swing_trailing_stop_distance=0.02,
        swing_trailing_activation_pct=0.01,
    )


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            id="POS_000001",
            symbol="BTCUSDT",
            signal_type="SCALP",
            direction="LONG",
            entry_price=50000.0,
            quantity=0.1,
            stop_loss=49500.0,
            take_profit=51000.0,
            leverage=20,
        )

        assert position.id == "POS_000001"
        assert position.is_long
        assert position.is_scalp
        assert position.state == PositionState.PENDING

    def test_position_is_long(self):
        """Test is_long property."""
        long_pos = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000, quantity=0.1,
            stop_loss=49500, take_profit=51000, leverage=20,
        )
        short_pos = Position(
            id="2", symbol="BTCUSDT", signal_type="SCALP",
            direction="SHORT", entry_price=50000, quantity=0.1,
            stop_loss=50500, take_profit=49000, leverage=20,
        )

        assert long_pos.is_long is True
        assert short_pos.is_long is False

    def test_calculate_unrealized_pnl_long(self):
        """Test unrealized PnL calculation for long position."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=49500, take_profit=51000, leverage=20,
        )

        # Price up - profit
        pnl = position.calculate_unrealized_pnl(51000.0)
        assert pnl == 100.0  # (51000 - 50000) * 0.1 = 100

        # Price down - loss
        pnl = position.calculate_unrealized_pnl(49000.0)
        assert pnl == -100.0  # (49000 - 50000) * 0.1 = -100

    def test_calculate_unrealized_pnl_short(self):
        """Test unrealized PnL calculation for short position."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="SHORT", entry_price=50000.0, quantity=0.1,
            stop_loss=50500, take_profit=49000, leverage=20,
        )

        # Price down - profit
        pnl = position.calculate_unrealized_pnl(49000.0)
        assert pnl == 100.0  # (50000 - 49000) * 0.1 = 100

        # Price up - loss
        pnl = position.calculate_unrealized_pnl(51000.0)
        assert pnl == -100.0  # (50000 - 51000) * 0.1 = -100

    def test_calculate_pnl_pct(self):
        """Test PnL percentage calculation."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=49500, take_profit=51000, leverage=20,
        )

        pnl_pct = position.calculate_pnl_pct(51000.0)
        assert pnl_pct == pytest.approx(0.02)  # 2%

    def test_trailing_stop_update_long(self):
        """Test trailing stop update for long position."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SWING",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=47500.0, take_profit=57500.0, leverage=10,
            trailing_stop_enabled=True,
            trailing_stop_distance=0.02,
            highest_price=50000.0,
        )

        # Price goes up
        new_sl = position.update_trailing_stop(52000.0)

        # New SL should be 52000 * 0.98 = 50960
        assert position.highest_price == 52000.0
        assert position.stop_loss == pytest.approx(50960.0)
        assert new_sl == pytest.approx(50960.0)

    def test_trailing_stop_update_short(self):
        """Test trailing stop update for short position."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SWING",
            direction="SHORT", entry_price=50000.0, quantity=0.1,
            stop_loss=52500.0, take_profit=42500.0, leverage=10,
            trailing_stop_enabled=True,
            trailing_stop_distance=0.02,
            lowest_price=50000.0,
        )

        # Price goes down
        new_sl = position.update_trailing_stop(48000.0)

        # New SL should be 48000 * 1.02 = 48960
        assert position.lowest_price == 48000.0
        assert position.stop_loss == pytest.approx(48960.0)
        assert new_sl == pytest.approx(48960.0)

    def test_trailing_stop_no_update_when_disabled(self):
        """Test trailing stop doesn't update when disabled."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=49500.0, take_profit=51000.0, leverage=20,
            trailing_stop_enabled=False,
        )

        new_sl = position.update_trailing_stop(52000.0)
        assert new_sl is None
        assert position.stop_loss == 49500.0

    def test_position_close(self):
        """Test closing a position."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=49500, take_profit=51000, leverage=20,
            state=PositionState.ACTIVE,
        )

        position.close(51000.0, "take_profit")

        assert position.state == PositionState.CLOSED
        assert position.exit_price == 51000.0
        assert position.realized_pnl == 100.0
        assert position.close_reason == "take_profit"
        assert position.exit_time is not None

    def test_position_to_dict(self):
        """Test converting position to dictionary."""
        position = Position(
            id="1", symbol="BTCUSDT", signal_type="SCALP",
            direction="LONG", entry_price=50000.0, quantity=0.1,
            stop_loss=49500, take_profit=51000, leverage=20,
        )

        data = position.to_dict()

        assert data["id"] == "1"
        assert data["symbol"] == "BTCUSDT"
        assert data["direction"] == "LONG"
        assert data["state"] == "PENDING"


class TestPositionManager:
    """Tests for PositionManager."""

    def test_init(self, position_manager):
        """Test PositionManager initialization."""
        assert position_manager.max_scalp_positions == 1
        assert position_manager.max_swing_positions == 1
        assert position_manager.swing_trailing_stop_distance == 0.02

    def test_create_position(self, position_manager, sample_signal):
        """Test creating a position."""
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
            entry_order_id="12345",
        )

        assert position.id.startswith("POS_")
        assert position.direction == "LONG"
        assert position.quantity == 0.1
        assert position.state == PositionState.PENDING

    def test_create_swing_position_enables_trailing(
        self, position_manager, swing_signal
    ):
        """Test that swing positions have trailing stop enabled."""
        position = position_manager.create_position(
            signal=swing_signal,
            quantity=0.1,
            leverage=10,
        )

        assert position.trailing_stop_enabled is True
        assert position.trailing_stop_distance == 0.02

    def test_activate_position(self, position_manager, sample_signal):
        """Test activating a position."""
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
        )

        activated = position_manager.activate_position(
            position_id=position.id,
            entry_price=50100.0,
            sl_order_id="sl_123",
            tp_order_id="tp_123",
        )

        assert activated is not None
        assert activated.state == PositionState.ACTIVE
        assert activated.entry_price == 50100.0
        assert activated.sl_order_id == "sl_123"

    def test_can_open_position_no_existing(self, position_manager, sample_signal):
        """Test can open position when none exist."""
        can_open, reason = position_manager.can_open_position(sample_signal)
        assert can_open is True
        assert reason == ""

    def test_can_open_position_direction_conflict(
        self, position_manager, sample_signal
    ):
        """Test cannot open position with direction conflict."""
        # Create and activate a LONG position
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
        )
        position_manager.activate_position(position.id)

        # Try to open a SHORT position
        short_signal = Signal(
            type="SCALP",
            direction="SHORT",
            confidence=0.85,
            entry_price=50000.0,
            stop_loss=50500.0,
            take_profit=49000.0,
            reason="Test short",
            timestamp=datetime.now(timezone.utc),
        )

        can_open, reason = position_manager.can_open_position(short_signal)
        assert can_open is False
        assert "conflict" in reason.lower()

    def test_can_open_position_max_reached(
        self, position_manager, sample_signal
    ):
        """Test cannot open position when max reached."""
        # Create and activate a scalp position
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
        )
        position_manager.activate_position(position.id)

        # Try to open another scalp position
        can_open, reason = position_manager.can_open_position(sample_signal)
        assert can_open is False
        assert "max" in reason.lower()

    def test_close_position(self, position_manager, sample_signal):
        """Test closing a position."""
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
        )
        position_manager.activate_position(position.id)

        closed = position_manager.close_position(
            position.id,
            exit_price=51000.0,
            reason="take_profit",
        )

        assert closed is not None
        assert closed.state == PositionState.CLOSED
        assert closed.realized_pnl == 100.0
        assert len(position_manager.active_positions) == 0

    def test_close_all_positions(self, position_manager, sample_signal, swing_signal):
        """Test closing all positions."""
        # Create and activate multiple positions (same direction)
        pos1 = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos1.id)

        # For swing, use same direction
        swing_signal.direction = "LONG"
        pos2 = position_manager.create_position(swing_signal, 0.1, 10)
        position_manager.activate_position(pos2.id)

        closed = position_manager.close_all_positions(51000.0, "shutdown")

        assert len(closed) == 2
        assert len(position_manager.active_positions) == 0

    def test_active_positions_property(self, position_manager, sample_signal):
        """Test active_positions property."""
        pos = position_manager.create_position(sample_signal, 0.1, 20)

        # Before activation
        assert len(position_manager.active_positions) == 0

        # After activation
        position_manager.activate_position(pos.id)
        assert len(position_manager.active_positions) == 1

    def test_scalp_swing_positions_properties(
        self, position_manager, sample_signal, swing_signal
    ):
        """Test scalp_positions and swing_positions properties."""
        # Same direction for both
        swing_signal.direction = "LONG"

        scalp = position_manager.create_position(sample_signal, 0.1, 20)
        swing = position_manager.create_position(swing_signal, 0.1, 10)

        position_manager.activate_position(scalp.id)
        position_manager.activate_position(swing.id)

        assert len(position_manager.scalp_positions) == 1
        assert len(position_manager.swing_positions) == 1

    def test_get_position_by_order(self, position_manager, sample_signal):
        """Test getting position by order ID."""
        position = position_manager.create_position(
            signal=sample_signal,
            quantity=0.1,
            leverage=20,
            entry_order_id="entry_123",
        )
        position_manager.activate_position(
            position.id,
            sl_order_id="sl_123",
            tp_order_id="tp_123",
        )

        # Find by entry order
        found = position_manager.get_position_by_order("entry_123")
        assert found is not None
        assert found.id == position.id

        # Find by SL order
        found = position_manager.get_position_by_order("sl_123")
        assert found is not None

        # Not found
        found = position_manager.get_position_by_order("unknown")
        assert found is None

    @pytest.mark.asyncio
    async def test_update_prices_trailing_stop(
        self, position_manager, swing_signal
    ):
        """Test updating prices triggers trailing stop."""
        position = position_manager.create_position(swing_signal, 0.1, 10)
        position_manager.activate_position(position.id, entry_price=50000.0)

        # Move price up past activation threshold (1%)
        current_price = 51000.0  # 2% profit

        updates = await position_manager.update_prices(current_price)

        # Should have trailing stop update
        assert len(updates) == 1
        assert updates[0][0].id == position.id

    @pytest.mark.asyncio
    async def test_update_prices_no_update_below_threshold(
        self, position_manager, swing_signal
    ):
        """Test no trailing stop update below profit threshold."""
        position = position_manager.create_position(swing_signal, 0.1, 10)
        position_manager.activate_position(position.id, entry_price=50000.0)

        # Move price up but below 1% activation threshold
        current_price = 50400.0  # 0.8% profit

        updates = await position_manager.update_prices(current_price)

        assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_stop_loss_update_callback(
        self, position_manager, swing_signal
    ):
        """Test stop loss update callback is called."""
        callback = AsyncMock()
        position_manager.set_stop_loss_update_callback(callback)

        position = position_manager.create_position(swing_signal, 0.1, 10)
        position_manager.activate_position(position.id, entry_price=50000.0)

        # Move price up past activation threshold
        await position_manager.update_prices(52000.0)

        callback.assert_called_once()

    def test_get_total_unrealized_pnl(self, position_manager, sample_signal):
        """Test total unrealized PnL calculation."""
        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id, entry_price=50000.0)

        pnl = position_manager.get_total_unrealized_pnl(51000.0)
        assert pnl == 100.0

    def test_get_total_exposure(self, position_manager, sample_signal):
        """Test total exposure calculation."""
        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id)

        exposure = position_manager.get_total_exposure()
        assert exposure == 5000.0  # 0.1 * 50000

    def test_get_session_pnl(self, position_manager, sample_signal):
        """Test session PnL calculation."""
        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id)
        position_manager.close_position(pos.id, 51000.0, "tp")

        pnl = position_manager.get_session_pnl()
        assert pnl == 100.0

    def test_get_session_stats(self, position_manager, sample_signal):
        """Test session statistics."""
        # Create and close winning trade
        pos1 = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos1.id)
        position_manager.close_position(pos1.id, 51000.0, "tp")

        # Create and close losing trade
        pos2 = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos2.id)
        position_manager.close_position(pos2.id, 49500.0, "sl")

        stats = position_manager.get_session_stats()

        assert stats["total_trades"] == 2
        assert stats["winning_trades"] == 1
        assert stats["losing_trades"] == 1
        assert stats["win_rate"] == 0.5
        assert stats["total_pnl"] == 50.0  # 100 - 50

    def test_get_session_stats_empty(self, position_manager):
        """Test session stats when no trades."""
        stats = position_manager.get_session_stats()

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == 0.0

    def test_to_dict(self, position_manager, sample_signal):
        """Test converting manager state to dictionary."""
        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id)

        data = position_manager.to_dict()

        assert "active_positions" in data
        assert "closed_positions" in data
        assert "session_stats" in data
        assert len(data["active_positions"]) == 1

    def test_current_direction(self, position_manager, sample_signal):
        """Test current direction detection."""
        # No position
        assert position_manager.current_direction is None

        # Add long position
        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id)

        assert position_manager.current_direction == "LONG"

    def test_has_position(self, position_manager, sample_signal):
        """Test has_position property."""
        assert position_manager.has_position is False

        pos = position_manager.create_position(sample_signal, 0.1, 20)
        position_manager.activate_position(pos.id)

        assert position_manager.has_position is True
