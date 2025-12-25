"""Risk manager for enforcing trading rules."""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.signals.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of risk check.

    Attributes:
        approved: Whether the trade is approved
        reason: Reason for rejection if not approved
        adjusted_size: Position size multiplier (0.0 to 1.0)
        cooldown_until: Cooldown expiration time if applicable
        warnings: List of warning messages
    """

    approved: bool
    reason: str | None = None
    adjusted_size: float = 1.0
    cooldown_until: datetime | None = None
    warnings: list[str] | None = None

    def __post_init__(self):
        """Initialize warnings list if not provided."""
        if self.warnings is None:
            self.warnings = []


@dataclass
class TradeResult:
    """Result of a completed trade.

    Attributes:
        timestamp: Trade completion time
        pnl: Profit/loss amount
        pnl_pct: Profit/loss percentage
        signal_type: Type of signal (SCALP or SWING)
        direction: Trade direction (LONG or SHORT)
    """

    timestamp: datetime
    pnl: float
    pnl_pct: float
    signal_type: str
    direction: str


class RiskManager:
    """Manage trading risk and enforce protection rules."""

    def __init__(self, config: dict):
        """Initialize risk manager.

        Args:
            config: Trading configuration from trading.yaml
        """
        self.config = config
        self.daily_loss_limit_pct = config["risk"]["daily_loss_limit"]
        self.max_consecutive_losses = config["risk"]["consecutive_loss_cooldown"]
        self.event_blackout_minutes = config["risk"]["event_blackout_minutes"]
        self.llm_confidence_threshold = config["risk"]["llm_confidence_threshold"]

        # State tracking
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.consecutive_losses = 0
        self.cooldown_until: datetime | None = None
        self.trade_history: deque[TradeResult] = deque(maxlen=100)
        self.upcoming_events: list[dict[str, Any]] = []

    def check(self, signal: Signal, account_state: dict) -> RiskCheckResult:
        """Perform risk check on a trading signal.

        Args:
            signal: Trading signal to check
            account_state: Current account state with keys:
                - balance: Account balance
                - positions: List of existing positions
                - llm_confidence: LLM confidence score (optional)

        Returns:
            RiskCheckResult with approval status and details
        """
        warnings = []
        adjusted_size = 1.0

        # Reset daily tracking if new day
        self._check_daily_reset()

        # 1. Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return RiskCheckResult(
                approved=False,
                reason=f"Cooldown active until {self.cooldown_until.strftime('%H:%M:%S')}",
                cooldown_until=self.cooldown_until,
                warnings=warnings,
            )

        # 2. Check daily loss limit
        account_balance = account_state.get("balance", 0)
        if account_balance <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Invalid account balance",
                warnings=warnings,
            )

        daily_loss_limit = account_balance * self.daily_loss_limit_pct
        if abs(self.daily_pnl) >= daily_loss_limit and self.daily_pnl < 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit reached: {self.daily_pnl:.2f} USD",
                warnings=warnings,
            )

        # 3. Check event blackout
        if self.is_event_blackout():
            event_names = [e.get("name", "Unknown") for e in self.upcoming_events[:3]]
            return RiskCheckResult(
                approved=False,
                reason=f"Event blackout active: {', '.join(event_names)}",
                warnings=warnings,
            )

        # 4. Check position conflicts (same direction only)
        positions = account_state.get("positions", [])
        if positions:
            conflict = self._check_position_conflict(signal, positions)
            if conflict:
                return RiskCheckResult(
                    approved=False,
                    reason=conflict,
                    warnings=warnings,
                )

        # 5. Adjust size based on LLM confidence
        llm_confidence = account_state.get("llm_confidence")
        if llm_confidence is not None:
            if llm_confidence < self.llm_confidence_threshold:
                adjusted_size = 0.5
                warnings.append(
                    f"Position reduced to 50% due to low LLM confidence: {llm_confidence:.2f}"
                )

        # 6. Adjust size based on recent losses
        if self.consecutive_losses >= 2:
            adjusted_size *= 0.7
            warnings.append(
                f"Position reduced to 70% due to {self.consecutive_losses} consecutive losses"
            )

        # 7. Validate signal
        if not signal.validate():
            return RiskCheckResult(
                approved=False,
                reason="Invalid signal parameters (stop loss / take profit)",
                warnings=warnings,
            )

        # All checks passed
        logger.info(
            f"Risk check passed for {signal.type} {signal.direction} "
            f"(confidence: {signal.confidence:.2f}, adjusted_size: {adjusted_size:.2f})"
        )

        return RiskCheckResult(
            approved=True,
            adjusted_size=adjusted_size,
            warnings=warnings,
        )

    def record_trade_result(self, trade_result: TradeResult) -> None:
        """Record a completed trade result.

        Args:
            trade_result: Completed trade information
        """
        self.trade_history.append(trade_result)
        self.daily_pnl += trade_result.pnl

        # Update consecutive losses
        if trade_result.pnl < 0:
            self.consecutive_losses += 1
            logger.warning(
                f"Loss recorded: {trade_result.pnl:.2f} USD "
                f"({self.consecutive_losses} consecutive)"
            )

            # Trigger cooldown after max consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_until = datetime.now() + timedelta(hours=1)
                logger.warning(
                    f"Cooldown triggered until {self.cooldown_until.strftime('%H:%M:%S')}"
                )
        else:
            self.consecutive_losses = 0

        logger.info(
            f"Trade recorded: {trade_result.signal_type} {trade_result.direction} "
            f"PnL: {trade_result.pnl:.2f} USD ({trade_result.pnl_pct:.2f}%)"
        )

    def is_event_blackout(self) -> bool:
        """Check if currently in event blackout period.

        Returns:
            True if within blackout period of a major event
        """
        if not self.upcoming_events:
            return False

        now = datetime.now()
        blackout_delta = timedelta(minutes=self.event_blackout_minutes)

        for event in self.upcoming_events:
            event_time = event.get("time")
            if not event_time:
                continue

            # Convert to datetime if string
            if isinstance(event_time, str):
                try:
                    event_time = datetime.fromisoformat(event_time)
                except (ValueError, TypeError):
                    continue

            # Check if within blackout window
            if event_time - blackout_delta <= now <= event_time:
                logger.info(
                    f"Event blackout active: {event.get('name', 'Unknown')} "
                    f"at {event_time.strftime('%H:%M')}"
                )
                return True

        return False

    def update_events(self, events: list[dict[str, Any]]) -> None:
        """Update upcoming economic events.

        Args:
            events: List of event dictionaries with 'time' and 'name' keys
        """
        self.upcoming_events = events
        logger.debug(f"Updated {len(events)} upcoming events")

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (called at day change)."""
        logger.info(f"Resetting daily stats. Final PnL: {self.daily_pnl:.2f} USD")
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def get_stats(self) -> dict[str, Any]:
        """Get current risk manager statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "total_trades_today": len([
                t for t in self.trade_history
                if t.timestamp.date() == datetime.now().date()
            ]),
            "event_blackout": self.is_event_blackout(),
        }

    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily stats."""
        now = datetime.now()
        if now.date() > self.daily_start.date():
            self.reset_daily_stats()

    def _check_position_conflict(self, signal: Signal, positions: list) -> str | None:
        """Check for position conflicts.

        Args:
            signal: New signal to check
            positions: Existing positions

        Returns:
            Error message if conflict exists, None otherwise
        """
        for pos in positions:
            pos_type = pos.get("type")
            pos_direction = pos.get("direction")

            # Don't allow opposite direction positions
            if pos_direction and pos_direction != signal.direction:
                return (
                    f"Cannot open {signal.direction} position: "
                    f"existing {pos_direction} {pos_type} position"
                )

            # Don't allow duplicate signal types
            if pos_type and pos_type == signal.type:
                return f"Cannot open new {signal.type}: existing {signal.type} position active"

        return None
