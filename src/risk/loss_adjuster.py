"""Consecutive Loss Position Adjuster.

Dynamically adjusts position size based on consecutive losses.
Protects capital during losing streaks and recovers after wins.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RecoveryMode(Enum):
    """Recovery mode after consecutive losses."""
    IMMEDIATE = "immediate"  # Full size after 1 win
    GRADUAL = "gradual"      # Step-by-step recovery
    CONSERVATIVE = "conservative"  # Require multiple wins


@dataclass
class LossAdjusterConfig:
    """Configuration for consecutive loss adjuster.

    Attributes:
        base_risk_per_trade: Base risk percentage (e.g., 0.15 = 15%)
        loss_levels: List of (consecutive_losses, size_multiplier) tuples
        recovery_mode: How to recover position size after wins
        recovery_wins_required: Wins required for full recovery (for CONSERVATIVE mode)
        max_cooldown_hours: Maximum cooldown hours after extreme losses
        daily_loss_limit_pct: Daily loss limit as percentage of capital
    """
    base_risk_per_trade: float = 0.15
    loss_levels: list[tuple[int, float]] = None
    recovery_mode: RecoveryMode = RecoveryMode.GRADUAL
    recovery_wins_required: int = 2
    max_cooldown_hours: int = 4
    daily_loss_limit_pct: float = 0.10

    def __post_init__(self):
        """Initialize default loss levels if not provided."""
        if self.loss_levels is None:
            # Default: step-down position sizing
            self.loss_levels = [
                (1, 1.0),    # 1 loss: 100% position (no change)
                (2, 0.50),   # 2 consecutive: 50% position
                (3, 0.25),   # 3 consecutive: 25% position
                (4, 0.10),   # 4 consecutive: 10% position
                (5, 0.0),    # 5+ consecutive: STOP trading (cooldown)
            ]


@dataclass
class AdjustmentResult:
    """Result of position adjustment calculation.

    Attributes:
        risk_per_trade: Adjusted risk per trade
        size_multiplier: Position size multiplier (0.0 to 1.0)
        can_trade: Whether trading is allowed
        reason: Explanation for the adjustment
        cooldown_until: Cooldown expiration if trading is blocked
    """
    risk_per_trade: float
    size_multiplier: float
    can_trade: bool
    reason: str
    cooldown_until: datetime | None = None


class ConsecutiveLossAdjuster:
    """Manages position sizing based on consecutive losses.

    Key Features:
    - Automatic position reduction on consecutive losses
    - Cooldown period after excessive losses
    - Gradual recovery after winning trades
    - Daily loss limit enforcement
    """

    def __init__(self, config: LossAdjusterConfig | None = None):
        """Initialize the loss adjuster.

        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or LossAdjusterConfig()

        # State tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_multiplier = 1.0
        self.cooldown_until: datetime | None = None
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        # Trade history
        self.trade_history: list[dict[str, Any]] = []

        logger.info(f"ConsecutiveLossAdjuster initialized with base risk: {self.config.base_risk_per_trade*100:.0f}%")

    def get_adjusted_risk(self, account_balance: float) -> AdjustmentResult:
        """Get adjusted risk per trade based on current state.

        Args:
            account_balance: Current account balance

        Returns:
            AdjustmentResult with adjusted risk and details
        """
        # Reset daily tracking if new day
        self._check_daily_reset()

        # Check cooldown
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return AdjustmentResult(
                risk_per_trade=0,
                size_multiplier=0,
                can_trade=False,
                reason=f"Cooldown active until {self.cooldown_until.strftime('%H:%M:%S')}",
                cooldown_until=self.cooldown_until,
            )

        # Check daily loss limit
        if account_balance > 0:
            daily_loss_limit = account_balance * self.config.daily_loss_limit_pct
            if self.daily_pnl < 0 and abs(self.daily_pnl) >= daily_loss_limit:
                # Trigger cooldown for rest of day
                tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                return AdjustmentResult(
                    risk_per_trade=0,
                    size_multiplier=0,
                    can_trade=False,
                    reason=f"Daily loss limit reached: {self.daily_pnl:.2f} USD (limit: {daily_loss_limit:.2f})",
                    cooldown_until=tomorrow,
                )

        # Get size multiplier based on consecutive losses
        multiplier = self._get_size_multiplier()

        # If multiplier is 0, trigger cooldown
        if multiplier <= 0:
            self.cooldown_until = datetime.now() + timedelta(hours=self.config.max_cooldown_hours)
            return AdjustmentResult(
                risk_per_trade=0,
                size_multiplier=0,
                can_trade=False,
                reason=f"Trading paused: {self.consecutive_losses} consecutive losses",
                cooldown_until=self.cooldown_until,
            )

        # Calculate adjusted risk
        adjusted_risk = self.config.base_risk_per_trade * multiplier

        reason = self._build_reason(multiplier)

        return AdjustmentResult(
            risk_per_trade=adjusted_risk,
            size_multiplier=multiplier,
            can_trade=True,
            reason=reason,
        )

    def record_trade(self, pnl: float, pnl_pct: float = 0) -> None:
        """Record a completed trade result.

        Args:
            pnl: Profit/loss amount in USD
            pnl_pct: Profit/loss percentage
        """
        timestamp = datetime.now()

        # Update daily PnL
        self.daily_pnl += pnl

        # Record trade
        self.trade_history.append({
            "timestamp": timestamp,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "is_win": pnl > 0,
        })

        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        if pnl > 0:
            # WIN
            self._handle_win()
        else:
            # LOSS
            self._handle_loss()

        logger.info(
            f"Trade recorded: PnL {pnl:+.2f} USD | "
            f"Consecutive: W={self.consecutive_wins} L={self.consecutive_losses} | "
            f"Multiplier: {self.current_multiplier:.0%}"
        )

    def _handle_loss(self) -> None:
        """Handle a losing trade."""
        self.consecutive_losses += 1
        self.consecutive_wins = 0

        # Update multiplier
        self.current_multiplier = self._get_size_multiplier()

        logger.warning(
            f"Loss #{self.consecutive_losses} | "
            f"Position reduced to {self.current_multiplier:.0%}"
        )

        # Check if we need to trigger cooldown
        if self.current_multiplier <= 0:
            self.cooldown_until = datetime.now() + timedelta(hours=self.config.max_cooldown_hours)
            logger.error(
                f"Trading halted after {self.consecutive_losses} consecutive losses! "
                f"Cooldown until {self.cooldown_until.strftime('%H:%M:%S')}"
            )

    def _handle_win(self) -> None:
        """Handle a winning trade."""
        self.consecutive_wins += 1

        # Recovery based on mode
        if self.config.recovery_mode == RecoveryMode.IMMEDIATE:
            # Full recovery after 1 win
            self.consecutive_losses = 0
            self.current_multiplier = 1.0

        elif self.config.recovery_mode == RecoveryMode.GRADUAL:
            # Step-up recovery: reduce consecutive loss count by 1
            if self.consecutive_losses > 0:
                self.consecutive_losses -= 1
                self.current_multiplier = self._get_size_multiplier()

        elif self.config.recovery_mode == RecoveryMode.CONSERVATIVE:
            # Require multiple wins for full recovery
            if self.consecutive_wins >= self.config.recovery_wins_required:
                self.consecutive_losses = 0
                self.current_multiplier = 1.0

        # Clear cooldown
        if self.cooldown_until:
            self.cooldown_until = None
            logger.info("Cooldown cleared after winning trade")

        logger.info(
            f"Win #{self.consecutive_wins} | "
            f"Position restored to {self.current_multiplier:.0%}"
        )

    def _get_size_multiplier(self) -> float:
        """Get position size multiplier based on consecutive losses."""
        # Sort loss levels by consecutive losses (ascending)
        sorted_levels = sorted(self.config.loss_levels, key=lambda x: x[0])

        multiplier = 1.0
        for losses, mult in sorted_levels:
            if self.consecutive_losses >= losses:
                multiplier = mult

        return multiplier

    def _build_reason(self, multiplier: float) -> str:
        """Build explanation string."""
        if multiplier >= 1.0:
            return "Full position size"
        elif self.consecutive_losses == 0:
            return f"Recovering: {self.consecutive_wins} consecutive wins"
        else:
            reduction_pct = (1 - multiplier) * 100
            return f"Position reduced {reduction_pct:.0f}% ({self.consecutive_losses} consecutive losses)"

    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily stats."""
        now = datetime.now()
        if now.date() > self.daily_start.date():
            logger.info(f"Daily reset. Previous day PnL: {self.daily_pnl:.2f} USD")
            self.daily_pnl = 0.0
            self.daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            # Clear cooldown on new day (optional)
            if self.cooldown_until and self.cooldown_until.date() < now.date():
                self.cooldown_until = None
                logger.info("Cooldown cleared on new day")

    def get_stats(self) -> dict[str, Any]:
        """Get current adjuster statistics.

        Returns:
            Dictionary with current stats
        """
        recent_trades = self.trade_history[-10:] if self.trade_history else []
        recent_wins = sum(1 for t in recent_trades if t["is_win"])
        recent_win_rate = recent_wins / len(recent_trades) * 100 if recent_trades else 0

        return {
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "current_multiplier": self.current_multiplier,
            "effective_risk": self.config.base_risk_per_trade * self.current_multiplier,
            "daily_pnl": self.daily_pnl,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "can_trade": self.cooldown_until is None or datetime.now() >= self.cooldown_until,
            "total_trades": len(self.trade_history),
            "recent_win_rate": recent_win_rate,
        }

    def reset(self) -> None:
        """Reset all state."""
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_multiplier = 1.0
        self.cooldown_until = None
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.trade_history = []
        logger.info("ConsecutiveLossAdjuster reset")


# Factory function for common configurations
def create_conservative_adjuster() -> ConsecutiveLossAdjuster:
    """Create a conservative loss adjuster (slower recovery)."""
    config = LossAdjusterConfig(
        base_risk_per_trade=0.10,
        loss_levels=[
            (1, 1.0),
            (2, 0.50),
            (3, 0.25),
            (4, 0.0),
        ],
        recovery_mode=RecoveryMode.CONSERVATIVE,
        recovery_wins_required=2,
        max_cooldown_hours=6,
    )
    return ConsecutiveLossAdjuster(config)


def create_aggressive_adjuster() -> ConsecutiveLossAdjuster:
    """Create an aggressive loss adjuster (faster recovery)."""
    config = LossAdjusterConfig(
        base_risk_per_trade=0.20,
        loss_levels=[
            (1, 1.0),
            (2, 0.70),
            (3, 0.50),
            (4, 0.25),
            (5, 0.0),
        ],
        recovery_mode=RecoveryMode.IMMEDIATE,
        max_cooldown_hours=2,
    )
    return ConsecutiveLossAdjuster(config)


def create_default_adjuster() -> ConsecutiveLossAdjuster:
    """Create default loss adjuster (balanced)."""
    config = LossAdjusterConfig(
        base_risk_per_trade=0.15,
        loss_levels=[
            (1, 1.0),    # 1 loss: 100% position
            (2, 0.50),   # 2 consecutive: 50%
            (3, 0.25),   # 3 consecutive: 25%
            (4, 0.10),   # 4 consecutive: 10%
            (5, 0.0),    # 5+ consecutive: STOP
        ],
        recovery_mode=RecoveryMode.GRADUAL,
        recovery_wins_required=2,
        max_cooldown_hours=4,
        daily_loss_limit_pct=0.10,
    )
    return ConsecutiveLossAdjuster(config)
