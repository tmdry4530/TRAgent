"""Position manager for tracking and managing active positions."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from src.signals.base import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionState(Enum):
    """Position state enumeration."""

    PENDING = "PENDING"  # Entry order placed, waiting for fill
    ACTIVE = "ACTIVE"  # Position is active
    CLOSING = "CLOSING"  # Close order placed, waiting for fill
    CLOSED = "CLOSED"  # Position is closed


@dataclass
class Position:
    """Represents an active trading position."""

    id: str
    symbol: str
    signal_type: str  # SCALP or SWING
    direction: str  # LONG or SHORT
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    leverage: int
    state: PositionState = PositionState.PENDING
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    # Order tracking
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None

    # Trailing stop
    trailing_stop_enabled: bool = False
    trailing_stop_distance: float = 0.0
    trailing_stop_activated: bool = False
    highest_price: float = 0.0  # For LONG trailing
    lowest_price: float = float("inf")  # For SHORT trailing

    # Partial take profit
    partial_tp_enabled: bool = False
    partial_tp_pct: float = 0.5  # Close 50% at first TP
    partial_tp_triggered: bool = False
    original_quantity: float = 0.0  # Track original size for partial exit

    # Metadata
    signal_reason: str = ""
    close_reason: str = ""

    @property
    def is_active(self) -> bool:
        """Check if position is active."""
        return self.state == PositionState.ACTIVE

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.state == PositionState.CLOSED

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.direction == "LONG"

    @property
    def is_scalp(self) -> bool:
        """Check if this is a scalp position."""
        return self.signal_type == "SCALP"

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL.

        Args:
            current_price: Current market price

        Returns:
            Unrealized PnL in quote currency
        """
        if self.is_long:
            pnl = (current_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - current_price) * self.quantity
        return pnl

    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate PnL percentage.

        Args:
            current_price: Current market price

        Returns:
            PnL as percentage (0.01 = 1%)
        """
        if self.is_long:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        return pnl_pct

    def check_partial_tp(self, current_price: float) -> Optional[float]:
        """Check if partial take profit should trigger.

        Partial TP triggers when price reaches 50% of the way to TP.

        Args:
            current_price: Current market price

        Returns:
            Quantity to close if partial TP triggered, None otherwise
        """
        if not self.partial_tp_enabled or self.partial_tp_triggered:
            return None

        # Calculate progress toward TP (0.0 to 1.0)
        if self.is_long:
            total_distance = self.take_profit - self.entry_price
            current_progress = current_price - self.entry_price
        else:
            total_distance = self.entry_price - self.take_profit
            current_progress = self.entry_price - current_price

        if total_distance <= 0:
            return None

        progress_pct = current_progress / total_distance

        # Trigger at 50% progress toward TP
        if progress_pct >= 0.5:
            self.partial_tp_triggered = True
            close_quantity = self.quantity * self.partial_tp_pct
            # Update remaining quantity
            self.quantity = self.quantity - close_quantity

            logger.info(
                "Partial take profit triggered",
                position_id=self.id,
                progress_pct=f"{progress_pct:.1%}",
                close_quantity=close_quantity,
                remaining_quantity=self.quantity,
            )

            return close_quantity

        return None

    def activate_trailing_stop(self, current_price: float, activation_pct: float = 0.01) -> bool:
        """Check if trailing stop should be activated.

        Trailing stop activates when profit reaches activation threshold.
        When activated, also sets the initial trailing stop loss level.

        Args:
            current_price: Current market price
            activation_pct: Profit percentage to activate (default: 1%)

        Returns:
            True if just activated, False otherwise
        """
        if not self.trailing_stop_enabled or self.trailing_stop_activated:
            return False

        pnl_pct = self.calculate_pnl_pct(current_price)

        if pnl_pct >= activation_pct:
            self.trailing_stop_activated = True

            # Set highest/lowest price and initial trailing stop
            if self.is_long:
                self.highest_price = current_price
                new_sl = current_price * (1 - self.trailing_stop_distance)
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
            else:
                self.lowest_price = current_price
                new_sl = current_price * (1 + self.trailing_stop_distance)
                if new_sl < self.stop_loss:
                    self.stop_loss = new_sl

            logger.info(
                "Trailing stop activated",
                position_id=self.id,
                pnl_pct=f"{pnl_pct:.2%}",
                current_price=current_price,
                new_stop_loss=self.stop_loss,
            )
            return True

        return False

    def update_trailing_stop(self, current_price: float) -> Optional[float]:
        """Update trailing stop based on current price.

        Args:
            current_price: Current market price

        Returns:
            New stop loss price if updated, None otherwise
        """
        if not self.trailing_stop_enabled or not self.trailing_stop_activated:
            return None

        old_sl = self.stop_loss
        new_sl = None

        if self.is_long:
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
                # Calculate new stop loss
                calculated_sl = self.highest_price * (1 - self.trailing_stop_distance)
                if calculated_sl > self.stop_loss:
                    self.stop_loss = calculated_sl
                    new_sl = calculated_sl
                    logger.info(
                        "Trailing stop updated",
                        position_id=self.id,
                        new_stop_loss=new_sl,
                        highest_price=self.highest_price,
                    )
        else:
            # Update lowest price
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                # Calculate new stop loss
                calculated_sl = self.lowest_price * (1 + self.trailing_stop_distance)
                if calculated_sl < self.stop_loss:
                    self.stop_loss = calculated_sl
                    new_sl = calculated_sl
                    logger.info(
                        "Trailing stop updated",
                        position_id=self.id,
                        new_stop_loss=new_sl,
                        lowest_price=self.lowest_price,
                    )

        return new_sl if new_sl and new_sl != old_sl else None

    def close(
        self,
        exit_price: float,
        reason: str = "manual",
    ) -> None:
        """Mark position as closed.

        Args:
            exit_price: Exit price
            reason: Close reason
        """
        self.state = PositionState.CLOSED
        self.exit_price = exit_price
        self.exit_time = datetime.now(timezone.utc)
        self.close_reason = reason
        self.realized_pnl = self.calculate_unrealized_pnl(exit_price)

        logger.info(
            "Position closed",
            position_id=self.id,
            direction=self.direction,
            entry_price=self.entry_price,
            exit_price=exit_price,
            realized_pnl=self.realized_pnl,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "leverage": self.leverage,
            "state": self.state.value,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "signal_reason": self.signal_reason,
            "close_reason": self.close_reason,
        }


class PositionManager:
    """Manages active positions and position lifecycle.

    Features:
    - Track multiple positions (scalp + swing)
    - Trailing stop management for swing positions
    - PnL calculation and tracking
    - Position conflict detection (same direction only)
    """

    def __init__(
        self,
        max_scalp_positions: int = 1,
        max_swing_positions: int = 1,
        # Scalp settings
        scalp_trailing_enabled: bool = True,
        scalp_trailing_distance: float = 0.003,  # 0.3% for scalp
        scalp_trailing_activation_pct: float = 0.005,  # Activate after 0.5% profit
        scalp_partial_tp_enabled: bool = True,
        scalp_partial_tp_pct: float = 0.5,  # Close 50% at first TP
        # Swing settings
        swing_trailing_stop_distance: float = 0.02,  # 2% for swing
        swing_trailing_activation_pct: float = 0.01,  # Activate after 1% profit
    ) -> None:
        """Initialize position manager.

        Args:
            max_scalp_positions: Maximum concurrent scalp positions
            max_swing_positions: Maximum concurrent swing positions
            scalp_trailing_enabled: Enable trailing stop for scalp
            scalp_trailing_distance: Trailing distance for scalp (percentage)
            scalp_trailing_activation_pct: Profit % to activate scalp trailing
            scalp_partial_tp_enabled: Enable partial TP for scalp
            scalp_partial_tp_pct: Portion to close at partial TP
            swing_trailing_stop_distance: Trailing distance for swing (percentage)
            swing_trailing_activation_pct: Profit % to activate trailing stop
        """
        self.max_scalp_positions = max_scalp_positions
        self.max_swing_positions = max_swing_positions

        # Scalp settings
        self.scalp_trailing_enabled = scalp_trailing_enabled
        self.scalp_trailing_distance = scalp_trailing_distance
        self.scalp_trailing_activation_pct = scalp_trailing_activation_pct
        self.scalp_partial_tp_enabled = scalp_partial_tp_enabled
        self.scalp_partial_tp_pct = scalp_partial_tp_pct

        # Swing settings
        self.swing_trailing_stop_distance = swing_trailing_stop_distance
        self.swing_trailing_activation_pct = swing_trailing_activation_pct

        # Active positions
        self._positions: dict[str, Position] = {}
        self._position_counter = 0

        # Closed positions (for history)
        self._closed_positions: list[Position] = []

        # Callbacks
        self._on_stop_loss_update: Optional[Callable[[Position, float], Any]] = None

        logger.info(
            "PositionManager initialized",
            max_scalp=max_scalp_positions,
            max_swing=max_swing_positions,
            trailing_distance=swing_trailing_stop_distance,
        )

    def set_stop_loss_update_callback(
        self, callback: Callable[[Position, float], Any]
    ) -> None:
        """Set callback for when trailing stop is updated.

        Args:
            callback: Async function(position, new_stop_loss)
        """
        self._on_stop_loss_update = callback

    @property
    def active_positions(self) -> list[Position]:
        """Get all active positions."""
        return [p for p in self._positions.values() if p.is_active]

    @property
    def scalp_positions(self) -> list[Position]:
        """Get active scalp positions."""
        return [p for p in self.active_positions if p.is_scalp]

    @property
    def swing_positions(self) -> list[Position]:
        """Get active swing positions."""
        return [p for p in self.active_positions if not p.is_scalp]

    @property
    def has_position(self) -> bool:
        """Check if any active position exists."""
        return len(self.active_positions) > 0

    @property
    def current_direction(self) -> Optional[str]:
        """Get current position direction (for conflict detection)."""
        positions = self.active_positions
        if not positions:
            return None
        return positions[0].direction

    def can_open_position(self, signal: Signal) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Args:
            signal: Signal to check

        Returns:
            Tuple of (can_open, reason)
        """
        # Check direction conflict
        if self.has_position:
            current_dir = self.current_direction
            if current_dir and current_dir != signal.direction:
                return False, f"Direction conflict: current={current_dir}, signal={signal.direction}"

        # Check position limits
        if signal.type == "SCALP":
            if len(self.scalp_positions) >= self.max_scalp_positions:
                return False, f"Max scalp positions reached ({self.max_scalp_positions})"
        else:
            if len(self.swing_positions) >= self.max_swing_positions:
                return False, f"Max swing positions reached ({self.max_swing_positions})"

        return True, ""

    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"POS_{self._position_counter:06d}"

    def create_position(
        self,
        signal: Signal,
        quantity: float,
        leverage: int,
        entry_order_id: Optional[str] = None,
        symbol: str = "BTCUSDT",
    ) -> Position:
        """Create a new position from a signal.

        Args:
            signal: Trading signal
            quantity: Position quantity
            leverage: Position leverage
            entry_order_id: Entry order ID
            symbol: Trading symbol

        Returns:
            Created Position object
        """
        position = Position(
            id=self._generate_position_id(),
            symbol=symbol,
            signal_type=signal.type,
            direction=signal.direction,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=leverage,
            entry_order_id=entry_order_id,
            signal_reason=signal.reason,
            original_quantity=quantity,  # Track original for partial TP
        )

        # Configure position based on signal type
        if signal.type == "SCALP":
            # Enable trailing stop for scalp
            if self.scalp_trailing_enabled:
                position.trailing_stop_enabled = True
                position.trailing_stop_distance = self.scalp_trailing_distance
                position.highest_price = signal.entry_price
                position.lowest_price = signal.entry_price
            # Enable partial take profit for scalp
            if self.scalp_partial_tp_enabled:
                position.partial_tp_enabled = True
                position.partial_tp_pct = self.scalp_partial_tp_pct
        elif signal.type == "SWING":
            # Enable trailing stop for swing positions
            position.trailing_stop_enabled = True
            position.trailing_stop_distance = self.swing_trailing_stop_distance
            position.highest_price = signal.entry_price
            position.lowest_price = signal.entry_price

        self._positions[position.id] = position

        logger.info(
            "Position created",
            position_id=position.id,
            signal_type=signal.type,
            direction=signal.direction,
            quantity=quantity,
        )

        return position

    def activate_position(
        self,
        position_id: str,
        entry_price: Optional[float] = None,
        sl_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
    ) -> Optional[Position]:
        """Mark a position as active (entry filled).

        Args:
            position_id: Position ID
            entry_price: Actual entry price (if different from signal)
            sl_order_id: Stop loss order ID
            tp_order_id: Take profit order ID

        Returns:
            Updated Position or None if not found
        """
        position = self._positions.get(position_id)
        if not position:
            logger.warning("Position not found", position_id=position_id)
            return None

        position.state = PositionState.ACTIVE
        position.sl_order_id = sl_order_id
        position.tp_order_id = tp_order_id

        if entry_price:
            position.entry_price = entry_price
            position.highest_price = entry_price
            position.lowest_price = entry_price

        logger.info(
            "Position activated",
            position_id=position_id,
            entry_price=position.entry_price,
        )

        return position

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        return self._positions.get(position_id)

    def get_position_by_order(self, order_id: str) -> Optional[Position]:
        """Get position by any of its order IDs."""
        for position in self._positions.values():
            if order_id in [
                position.entry_order_id,
                position.sl_order_id,
                position.tp_order_id,
            ]:
                return position
        return None

    async def update_prices(self, current_price: float) -> list[tuple[Position, float]]:
        """Update positions with current price and handle trailing stops.

        Args:
            current_price: Current market price

        Returns:
            List of (position, new_stop_loss) for positions that need SL update
        """
        updates = []

        for position in self.active_positions:
            if not position.trailing_stop_enabled:
                continue

            # Use appropriate activation threshold based on position type
            if position.is_scalp:
                activation_pct = self.scalp_trailing_activation_pct
            else:
                activation_pct = self.swing_trailing_activation_pct

            # Track previous stop loss for detecting changes
            old_sl = position.stop_loss

            # Activate trailing stop if not already activated and threshold reached
            just_activated = position.activate_trailing_stop(current_price, activation_pct)

            # Update trailing stop (only works if activated)
            new_sl = position.update_trailing_stop(current_price)

            # Check if SL changed (either from activation or update)
            if just_activated and position.stop_loss != old_sl:
                new_sl = position.stop_loss

            if new_sl:
                updates.append((position, new_sl))

                # Call callback if set
                if self._on_stop_loss_update:
                    try:
                        result = self._on_stop_loss_update(position, new_sl)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(
                            "Stop loss update callback failed",
                            error=str(e),
                            position_id=position.id,
                        )

        return updates

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[Position]:
        """Close a position.

        Args:
            position_id: Position ID
            exit_price: Exit price
            reason: Close reason

        Returns:
            Closed Position or None if not found
        """
        position = self._positions.get(position_id)
        if not position:
            logger.warning("Position not found for close", position_id=position_id)
            return None

        position.close(exit_price, reason)

        # Move to closed positions
        self._closed_positions.append(position)
        del self._positions[position_id]

        return position

    def close_all_positions(
        self,
        exit_price: float,
        reason: str = "manual",
    ) -> list[Position]:
        """Close all active positions.

        Args:
            exit_price: Exit price
            reason: Close reason

        Returns:
            List of closed positions
        """
        closed = []
        for position_id in list(self._positions.keys()):
            position = self.close_position(position_id, exit_price, reason)
            if position:
                closed.append(position)

        return closed

    def get_total_unrealized_pnl(self, current_price: float) -> float:
        """Calculate total unrealized PnL across all positions.

        Args:
            current_price: Current market price

        Returns:
            Total unrealized PnL
        """
        total = 0.0
        for position in self.active_positions:
            total += position.calculate_unrealized_pnl(current_price)
        return total

    def get_total_exposure(self) -> float:
        """Get total position exposure (sum of all position sizes)."""
        return sum(p.quantity * p.entry_price for p in self.active_positions)

    def get_session_pnl(self) -> float:
        """Get total realized PnL for the session."""
        return sum(p.realized_pnl or 0 for p in self._closed_positions)

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        closed = self._closed_positions

        if not closed:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        winning = [p for p in closed if (p.realized_pnl or 0) > 0]
        losing = [p for p in closed if (p.realized_pnl or 0) < 0]
        total_pnl = sum(p.realized_pnl or 0 for p in closed)

        return {
            "total_trades": len(closed),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed) if closed else 0.0,
            "scalp_trades": len([p for p in closed if p.is_scalp]),
            "swing_trades": len([p for p in closed if not p.is_scalp]),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert manager state to dictionary."""
        return {
            "active_positions": [p.to_dict() for p in self.active_positions],
            "closed_positions": [p.to_dict() for p in self._closed_positions],
            "session_stats": self.get_session_stats(),
        }
