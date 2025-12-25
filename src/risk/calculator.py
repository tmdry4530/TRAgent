"""Position size calculator."""

from dataclasses import dataclass

from src.signals.base import Signal


@dataclass
class PositionSize:
    """Calculated position size.

    Attributes:
        size_usd: Position size in USD
        size_btc: Position size in BTC
        leverage: Applied leverage
        risk_amount: Amount at risk (based on stop loss)
    """

    size_usd: float
    size_btc: float
    leverage: int
    risk_amount: float


class PositionCalculator:
    """Calculate position sizes based on signal type and risk parameters."""

    def __init__(self, config: dict):
        """Initialize position calculator.

        Args:
            config: Trading configuration from trading.yaml
        """
        self.config = config
        self.scalp_max_position_pct = config["scalp"]["position_size"]
        self.swing_max_position_pct = config["swing"]["position_size"]
        self.max_total_exposure_pct = config["risk"].get("max_total_exposure", 0.80)
        self.scalp_max_leverage = config["scalp"]["max_leverage"]
        self.swing_max_leverage = config["swing"]["max_leverage"]
        self.scalp_stop_loss = config["scalp"]["stop_loss"]
        self.swing_stop_loss = config["swing"]["stop_loss"]

    def calculate(
        self,
        signal: Signal,
        account_balance: float,
        current_price: float,
        existing_positions: list | None = None,
    ) -> PositionSize | None:
        """Calculate position size for a signal.

        Args:
            signal: Trading signal
            account_balance: Current account balance in USD
            current_price: Current BTC price
            existing_positions: List of existing positions (optional)

        Returns:
            PositionSize object if calculation succeeds, None if position not allowed
        """
        if existing_positions is None:
            existing_positions = []

        # Validate account balance
        if account_balance <= 0:
            return None

        # Check total exposure
        total_exposure = self._calculate_total_exposure(existing_positions, account_balance)
        if total_exposure >= self.max_total_exposure_pct:
            return None

        # Get max position size based on signal type
        if signal.type == "SCALP":
            max_position_pct = self.scalp_max_position_pct
            max_leverage = self.scalp_max_leverage
            default_stop_loss = self.scalp_stop_loss
        else:  # SWING
            max_position_pct = self.swing_max_position_pct
            max_leverage = self.swing_max_leverage
            default_stop_loss = self.swing_stop_loss

        # Calculate available allocation
        remaining_exposure = self.max_total_exposure_pct - total_exposure
        effective_max_pct = min(max_position_pct, remaining_exposure)

        if effective_max_pct <= 0:
            return None

        # Calculate stop loss distance
        stop_loss_pct = abs(signal.stop_loss - signal.entry_price) / signal.entry_price
        if stop_loss_pct == 0:
            stop_loss_pct = default_stop_loss

        # Calculate position size using Kelly-like approach
        # Risk a fixed percentage of account per trade
        risk_per_trade = 0.02 if signal.type == "SCALP" else 0.03  # 2% for scalp, 3% for swing
        risk_amount = account_balance * risk_per_trade

        # Position value = risk amount / stop loss percentage
        position_value = risk_amount / stop_loss_pct

        # Cap at maximum position size
        max_position_value = account_balance * effective_max_pct
        position_value = min(position_value, max_position_value)

        # Calculate leverage needed
        collateral_needed = position_value / max_leverage
        if collateral_needed > account_balance * effective_max_pct:
            # Reduce position to fit collateral limit
            position_value = account_balance * effective_max_pct * max_leverage

        # Determine actual leverage
        actual_leverage = min(
            max_leverage,
            int(position_value / (account_balance * effective_max_pct)) or 1,
        )

        # Calculate BTC size
        size_btc = position_value / current_price

        return PositionSize(
            size_usd=position_value,
            size_btc=size_btc,
            leverage=actual_leverage,
            risk_amount=risk_amount,
        )

    def _calculate_total_exposure(
        self, existing_positions: list, account_balance: float
    ) -> float:
        """Calculate total exposure from existing positions.

        Args:
            existing_positions: List of position dictionaries
            account_balance: Current account balance

        Returns:
            Total exposure as percentage of account balance
        """
        if not existing_positions:
            return 0.0

        total_notional = sum(
            abs(pos.get("notional", 0)) for pos in existing_positions
        )
        return total_notional / account_balance if account_balance > 0 else 0.0
