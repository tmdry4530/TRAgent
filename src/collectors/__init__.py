"""Data collectors for market data, macro indicators, and news."""

from abc import ABC, abstractmethod

from src.collectors.binance import (
    BinanceRestCollector,
    BinanceWebSocketCollector,
    FundingRateData,
    KlineData,
    LiquidationData,
    LongShortRatioData,
    OpenInterestData,
    OrderbookData,
    TradeData,
)
from src.collectors.macro import (
    EconomicCalendarCollector,
    EconomicEvent,
    FearGreedCollector,
    FearGreedData,
    MacroContext,
    MacroDataCollector,
    MarketIndexCollector,
    MarketIndexData,
)
from src.collectors.news import (
    CryptoPanicCollector,
    NewsCollector,
    NewsContext,
    NewsItem,
    WhaleAlert,
    WhaleAlertCollector,
)


class BaseCollector(ABC):
    """Base class for all data collectors."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def subscribe(self, channels: list[str]) -> None:
        """Subscribe to data channels."""
        pass


__all__ = [
    # Base
    "BaseCollector",
    # Binance
    "BinanceWebSocketCollector",
    "BinanceRestCollector",
    "KlineData",
    "OrderbookData",
    "TradeData",
    "LiquidationData",
    "FundingRateData",
    "OpenInterestData",
    "LongShortRatioData",
    # Macro
    "MacroDataCollector",
    "MacroContext",
    "FearGreedCollector",
    "FearGreedData",
    "MarketIndexCollector",
    "MarketIndexData",
    "EconomicCalendarCollector",
    "EconomicEvent",
    # News
    "NewsCollector",
    "NewsContext",
    "NewsItem",
    "CryptoPanicCollector",
    "WhaleAlertCollector",
    "WhaleAlert",
]
