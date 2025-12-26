"""Daily loss limit checker.

Prevents trading when daily loss exceeds the configured limit.
"""

from datetime import date, datetime, timezone
from typing import Optional

from src.utils.config import get_settings
from src.utils.database import get_database
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DailyLossLimitChecker:
    """Check and enforce daily loss limits.

    Features:
    - Track daily PnL from database
    - Block trading when limit exceeded
    - Auto-reset at midnight UTC
    """

    def __init__(
        self,
        loss_limit_pct: Optional[float] = None,
        account_balance: Optional[float] = None,
    ) -> None:
        """Initialize daily loss limit checker.

        Args:
            loss_limit_pct: Loss limit as percentage (e.g., 0.05 for 5%)
                           Default: from settings
            account_balance: Account balance in USD for calculating limit
                           If None, uses fixed USD limit
        """
        self.settings = get_settings()
        self.db = get_database()

        self.loss_limit_pct = loss_limit_pct or self.settings.daily_loss_limit_pct
        self._account_balance = account_balance

        # Track current date for reset
        self._current_date: Optional[date] = None
        self._is_blocked = False
        self._block_reason = ""

        logger.info(
            "DailyLossLimitChecker initialized",
            loss_limit_pct=f"{self.loss_limit_pct:.1%}",
        )

    def set_account_balance(self, balance: float) -> None:
        """Update account balance for limit calculation.

        Args:
            balance: Current account balance in USD
        """
        self._account_balance = balance

    def get_loss_limit_usd(self) -> float:
        """Get current loss limit in USD.

        Returns:
            Loss limit in USD (positive number)
        """
        if self._account_balance is None or self._account_balance <= 0:
            # Fallback to fixed max_position_usd as reference
            return self.settings.max_position_usd * 2  # 2x position as limit
        return self._account_balance * self.loss_limit_pct

    def get_daily_pnl(self) -> float:
        """Get today's total PnL.

        Returns:
            Total PnL in USD (negative = loss)
        """
        return self.db.get_daily_pnl()

    def check_can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on daily loss limit.

        Returns:
            Tuple of (can_trade, reason)
        """
        today = datetime.now(timezone.utc).date()

        # Reset at midnight
        if self._current_date != today:
            self._current_date = today
            self._is_blocked = False
            self._block_reason = ""
            logger.info("Daily loss limit reset for new day", date=today.isoformat())

        # If already blocked, return early
        if self._is_blocked:
            return False, self._block_reason

        # Check daily PnL
        daily_pnl = self.get_daily_pnl()
        loss_limit = self.get_loss_limit_usd()

        # PnL is negative for losses, so check if loss exceeds limit
        if daily_pnl < 0 and abs(daily_pnl) >= loss_limit:
            self._is_blocked = True
            self._block_reason = (
                f"Daily loss limit exceeded: ${abs(daily_pnl):.2f} >= ${loss_limit:.2f} "
                f"({self.loss_limit_pct:.1%}). Trading blocked until midnight UTC."
            )
            logger.warning(
                "Daily loss limit exceeded - trading blocked",
                daily_pnl=daily_pnl,
                loss_limit=loss_limit,
            )
            return False, self._block_reason

        return True, ""

    def get_remaining_loss_budget(self) -> float:
        """Get remaining loss budget for today.

        Returns:
            Remaining budget in USD (positive number)
        """
        daily_pnl = self.get_daily_pnl()
        loss_limit = self.get_loss_limit_usd()

        if daily_pnl >= 0:
            # In profit, full limit available
            return loss_limit
        else:
            # In loss, remaining = limit - current loss
            return max(0, loss_limit - abs(daily_pnl))

    def get_dynamic_position_size(
        self,
        base_position_usd: float,
        max_loss_pct: float = 0.02,
    ) -> float:
        """Calculate dynamic position size based on daily performance.

        Position sizing logic:
        - In profit: Can increase position (up to 1.5x base)
        - In loss: Reduce position so max single trade loss fits remaining budget
        - Near limit: Minimum viable position size

        Args:
            base_position_usd: Base position size in USD
            max_loss_pct: Maximum loss per trade as decimal (default: 2%)

        Returns:
            Adjusted position size in USD
        """
        daily_pnl = self.get_daily_pnl()
        remaining_budget = self.get_remaining_loss_budget()
        loss_limit = self.get_loss_limit_usd()

        # Calculate max loss per trade at base size
        max_loss_per_trade = base_position_usd * max_loss_pct

        if daily_pnl >= 0:
            # In profit - can increase position size
            profit_bonus = min(daily_pnl / loss_limit, 0.5)  # Up to 50% bonus
            size_multiplier = 1.0 + profit_bonus
            adjusted_size = base_position_usd * size_multiplier
            logger.debug(
                "Dynamic sizing (profit mode)",
                daily_pnl=daily_pnl,
                bonus_pct=f"{profit_bonus:.1%}",
                adjusted_size=adjusted_size,
            )
        else:
            # In loss - reduce position to fit remaining budget
            if remaining_budget <= 0:
                return 0.0  # No budget left

            # Calculate how much we can afford to lose
            # If remaining budget can cover base max loss, use normal sizing
            if remaining_budget >= max_loss_per_trade:
                # Scale down based on how much of budget is used
                budget_usage = 1 - (remaining_budget / loss_limit)
                size_multiplier = max(0.5, 1.0 - budget_usage)  # Min 50% of base
                adjusted_size = base_position_usd * size_multiplier
            else:
                # Can't afford full loss, scale position to fit budget
                # Position size = remaining_budget / max_loss_pct
                adjusted_size = remaining_budget / max_loss_pct
                # Minimum viable position
                adjusted_size = max(adjusted_size, base_position_usd * 0.25)

            logger.debug(
                "Dynamic sizing (loss mode)",
                daily_pnl=daily_pnl,
                remaining_budget=remaining_budget,
                adjusted_size=adjusted_size,
            )

        # Cap at 1.5x base size
        max_size = base_position_usd * 1.5
        return min(adjusted_size, max_size)

    def get_status(self) -> dict:
        """Get current daily limit status.

        Returns:
            Status dictionary
        """
        daily_pnl = self.get_daily_pnl()
        loss_limit = self.get_loss_limit_usd()
        remaining = self.get_remaining_loss_budget()

        return {
            "daily_pnl": daily_pnl,
            "loss_limit": loss_limit,
            "loss_limit_pct": self.loss_limit_pct,
            "remaining_budget": remaining,
            "is_blocked": self._is_blocked,
            "block_reason": self._block_reason,
            "usage_pct": abs(daily_pnl) / loss_limit if loss_limit > 0 else 0,
        }

    def is_blocked(self) -> bool:
        """Check if trading is currently blocked.

        Returns:
            True if blocked
        """
        self.check_can_trade()  # Update state
        return self._is_blocked

    def reset(self) -> None:
        """Manually reset the daily limit (for testing)."""
        self._is_blocked = False
        self._block_reason = ""
        self._current_date = None
        logger.info("Daily loss limit manually reset")


# Global instance
_checker_instance: Optional[DailyLossLimitChecker] = None


def get_daily_limit_checker() -> DailyLossLimitChecker:
    """Get or create daily limit checker instance."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = DailyLossLimitChecker()
    return _checker_instance
