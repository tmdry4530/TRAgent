"""Base classes for signal generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.collectors import MarketState


@dataclass
class Signal:
    """Trading signal with entry/exit parameters.

    Attributes:
        type: Signal type (SCALP or SWING)
        direction: Trade direction (LONG or SHORT)
        confidence: Signal confidence level (0.0 to 1.0)
        entry_price: Intended entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        reason: Human-readable explanation for the signal
        timestamp: Signal generation timestamp
    """

    type: Literal["SCALP", "SWING"]
    direction: Literal["LONG", "SHORT"]
    confidence: float  # 0.0 ~ 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime

    def validate(self) -> bool:
        """Validate signal parameters.

        Returns:
            True if signal is valid, False otherwise.
        """
        if not 0 <= self.confidence <= 1:
            return False
        if self.direction == "LONG":
            return self.stop_loss < self.entry_price < self.take_profit
        else:
            return self.take_profit < self.entry_price < self.stop_loss


class BaseSignalGenerator(ABC):
    """Base class for signal generators.

    All signal generators must inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    async def generate(self, market_state: "MarketState") -> Signal | None:
        """Generate signal if conditions are met.

        Args:
            market_state: Current market state with all relevant data.

        Returns:
            Signal object if conditions are met, None otherwise.
        """
        pass

    @abstractmethod
    def get_required_data(self) -> list[str]:
        """Return list of required data channels.

        Returns:
            List of required data channel names (e.g., ['liquidations', 'funding_rate']).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this signal generator.

        Returns:
            Human-readable signal generator name.
        """
        pass
