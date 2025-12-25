"""Macro data collectors for economic indicators and market indices."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FearGreedData:
    """Fear and Greed Index data from Alternative.me."""

    value: int  # 0-100
    value_classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    timestamp: datetime
    time_until_update: int  # seconds


@dataclass
class MarketIndexData:
    """Market index data from Yahoo Finance."""

    symbol: str
    name: str
    price: float
    change: float  # absolute change
    change_percent: float  # percentage change
    timestamp: datetime


@dataclass
class EconomicEvent:
    """Economic calendar event."""

    name: str
    date: datetime
    importance: str  # low, medium, high
    country: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass
class TreasuryYieldData:
    """US Treasury yield data."""

    symbol: str
    name: str
    yield_rate: float
    change: float
    timestamp: datetime


@dataclass
class MacroContext:
    """Complete macro context for trading decisions.

    Attributes:
        timestamp: Data collection timestamp
        fear_greed_index: Fear and Greed Index value (0-100)
        fear_greed_label: Fear and Greed classification
        dxy: US Dollar Index value
        dxy_change: DXY percentage change
        sp500_change: S&P 500 percentage change
        nasdaq_change: NASDAQ percentage change
        us10y_yield: US 10-year Treasury yield
        upcoming_events: Economic events in next 24 hours
    """

    timestamp: datetime
    fear_greed_index: int
    fear_greed_label: str
    dxy: float
    dxy_change: float
    sp500_change: float
    nasdaq_change: float
    us10y_yield: float
    upcoming_events: list[EconomicEvent] = field(default_factory=list)


class FearGreedCollector:
    """Collects Fear and Greed Index from Alternative.me API.

    Polling interval: 1 hour (data updates daily)
    """

    API_URL = "https://api.alternative.me/fng/"
    POLL_INTERVAL = 3600  # 1 hour

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._callbacks: list[Callable[[FearGreedData], None]] = []
        self._last_data: Optional[FearGreedData] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def fetch(self) -> FearGreedData:
        """Fetch current Fear and Greed Index.

        Returns:
            FearGreedData with current index value

        Raises:
            ValueError: On API errors
        """
        session = await self._ensure_session()

        try:
            async with session.get(self.API_URL, params={"limit": 1}) as response:
                if response.status != 200:
                    raise ValueError(f"API error: {response.status}")

                data = await response.json()

                if "data" not in data or len(data["data"]) == 0:
                    raise ValueError("No data in response")

                item = data["data"][0]

                result = FearGreedData(
                    value=int(item["value"]),
                    value_classification=item["value_classification"],
                    timestamp=datetime.fromtimestamp(
                        int(item["timestamp"]), tz=timezone.utc
                    ),
                    time_until_update=int(item.get("time_until_update", 0)),
                )

                logger.info(
                    "Fetched Fear & Greed Index",
                    value=result.value,
                    classification=result.value_classification,
                )

                self._last_data = result
                return result

        except aiohttp.ClientError as e:
            logger.error("Network error fetching Fear & Greed", error=str(e))
            raise

    def on_update(self, callback: Callable[[FearGreedData], None]) -> None:
        """Register callback for data updates."""
        self._callbacks.append(callback)

    async def start_polling(self) -> None:
        """Start polling loop."""
        self.is_running = True
        logger.info("Starting Fear & Greed polling", interval=self.POLL_INTERVAL)

        while self.is_running:
            try:
                data = await self.fetch()

                for callback in self._callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error("Error in Fear & Greed callback", error=str(e))

            except Exception as e:
                logger.error("Error in Fear & Greed polling", error=str(e))

            await asyncio.sleep(self.POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self.is_running = False
        if self.session and not self.session.closed:
            await self.session.close()

    @property
    def last_value(self) -> Optional[int]:
        """Get last fetched Fear & Greed value."""
        return self._last_data.value if self._last_data else None


class MarketIndexCollector:
    """Collects market index data from Yahoo Finance.

    Supports: DXY, S&P 500, NASDAQ
    Polling interval: 5 minutes
    """

    POLL_INTERVAL = 300  # 5 minutes

    # Yahoo Finance symbols
    SYMBOLS = {
        "DXY": "DX-Y.NYB",  # US Dollar Index
        "SPX": "^GSPC",  # S&P 500
        "NDX": "^IXIC",  # NASDAQ Composite
        "US10Y": "^TNX",  # 10-year Treasury yield
    }

    NAMES = {
        "DXY": "US Dollar Index",
        "SPX": "S&P 500",
        "NDX": "NASDAQ Composite",
        "US10Y": "US 10Y Treasury Yield",
    }

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._callbacks: list[Callable[[dict[str, MarketIndexData]], None]] = []
        self._last_data: dict[str, MarketIndexData] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def fetch(self, symbol_key: str) -> MarketIndexData:
        """Fetch current data for a market index.

        Args:
            symbol_key: Key from SYMBOLS dict (e.g., "DXY", "SPX")

        Returns:
            MarketIndexData with current values

        Raises:
            ValueError: On API errors or unknown symbol
        """
        if symbol_key not in self.SYMBOLS:
            raise ValueError(f"Unknown symbol: {symbol_key}")

        yahoo_symbol = self.SYMBOLS[symbol_key]
        session = await self._ensure_session()

        # Yahoo Finance v8 API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {
            "interval": "1d",
            "range": "2d",
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise ValueError(f"API error: {response.status}")

                data = await response.json()

                result = data.get("chart", {}).get("result", [])
                if not result:
                    raise ValueError(f"No data for {symbol_key}")

                meta = result[0].get("meta", {})
                indicators = result[0].get("indicators", {}).get("quote", [{}])[0]

                # Get latest close price
                closes = indicators.get("close", [])
                if not closes or closes[-1] is None:
                    # Try to get previous close if today's close is not available
                    closes = [c for c in closes if c is not None]

                if not closes:
                    raise ValueError(f"No price data for {symbol_key}")

                current_price = closes[-1]
                previous_close = meta.get("previousClose", current_price)

                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close else 0

                result_data = MarketIndexData(
                    symbol=symbol_key,
                    name=self.NAMES[symbol_key],
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    timestamp=datetime.now(timezone.utc),
                )

                logger.debug(
                    "Fetched market index",
                    symbol=symbol_key,
                    price=current_price,
                    change_pct=f"{change_percent:.2f}%",
                )

                self._last_data[symbol_key] = result_data
                return result_data

        except aiohttp.ClientError as e:
            logger.error(
                "Network error fetching market index",
                symbol=symbol_key,
                error=str(e),
            )
            raise

    async def fetch_all(self) -> dict[str, MarketIndexData]:
        """Fetch all market indices.

        Returns:
            Dict mapping symbol keys to MarketIndexData
        """
        results = {}

        for symbol_key in self.SYMBOLS:
            try:
                data = await self.fetch(symbol_key)
                results[symbol_key] = data
                # Small delay between requests to avoid rate limiting
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(
                    "Failed to fetch market index",
                    symbol=symbol_key,
                    error=str(e),
                )

        logger.info("Fetched all market indices", count=len(results))
        return results

    def on_update(self, callback: Callable[[dict[str, MarketIndexData]], None]) -> None:
        """Register callback for data updates."""
        self._callbacks.append(callback)

    async def start_polling(self) -> None:
        """Start polling loop."""
        self.is_running = True
        logger.info("Starting market index polling", interval=self.POLL_INTERVAL)

        while self.is_running:
            try:
                data = await self.fetch_all()

                for callback in self._callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error("Error in market index callback", error=str(e))

            except Exception as e:
                logger.error("Error in market index polling", error=str(e))

            await asyncio.sleep(self.POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self.is_running = False
        if self.session and not self.session.closed:
            await self.session.close()

    def get_dxy(self) -> Optional[float]:
        """Get last DXY value."""
        return self._last_data.get("DXY", {}).price if "DXY" in self._last_data else None

    def get_sp500_change(self) -> Optional[float]:
        """Get last S&P 500 change percentage."""
        return (
            self._last_data.get("SPX", {}).change_percent
            if "SPX" in self._last_data
            else None
        )

    def get_nasdaq_change(self) -> Optional[float]:
        """Get last NASDAQ change percentage."""
        return (
            self._last_data.get("NDX", {}).change_percent
            if "NDX" in self._last_data
            else None
        )

    def get_us10y_yield(self) -> Optional[float]:
        """Get last US 10Y Treasury yield."""
        return (
            self._last_data.get("US10Y", {}).price
            if "US10Y" in self._last_data
            else None
        )


class EconomicCalendarCollector:
    """Collects economic calendar events.

    Uses free public APIs to get upcoming economic events.
    Polling interval: 1 hour
    """

    POLL_INTERVAL = 3600  # 1 hour

    # Major events to track (simplified list)
    MAJOR_EVENTS = [
        "FOMC",
        "Federal Reserve",
        "Fed Interest Rate",
        "CPI",
        "Inflation",
        "NFP",
        "Nonfarm Payrolls",
        "Employment",
        "GDP",
        "Retail Sales",
        "PMI",
    ]

    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._callbacks: list[Callable[[list[EconomicEvent]], None]] = []
        self._upcoming_events: list[EconomicEvent] = []

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def fetch_upcoming_events(self) -> list[EconomicEvent]:
        """Fetch upcoming economic events for the next 7 days.

        Note: This is a simplified implementation. In production,
        you would integrate with FRED API or a dedicated economic
        calendar service.

        Returns:
            List of upcoming EconomicEvent objects
        """
        # For MVP, we return a static list of known recurring events
        # In production, integrate with FRED API or investing.com scraper
        logger.info("Fetching economic calendar (mock implementation)")

        # Check for FOMC dates (simplified - real implementation would use FRED API)
        # FOMC meets roughly every 6 weeks
        events = []

        # This is a placeholder - real implementation would fetch from API
        self._upcoming_events = events

        return events

    def on_update(self, callback: Callable[[list[EconomicEvent]], None]) -> None:
        """Register callback for data updates."""
        self._callbacks.append(callback)

    async def start_polling(self) -> None:
        """Start polling loop."""
        self.is_running = True
        logger.info("Starting economic calendar polling", interval=self.POLL_INTERVAL)

        while self.is_running:
            try:
                events = await self.fetch_upcoming_events()

                for callback in self._callbacks:
                    try:
                        callback(events)
                    except Exception as e:
                        logger.error("Error in calendar callback", error=str(e))

            except Exception as e:
                logger.error("Error in calendar polling", error=str(e))

            await asyncio.sleep(self.POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self.is_running = False
        if self.session and not self.session.closed:
            await self.session.close()

    def get_upcoming_events(self, hours: int = 24) -> list[EconomicEvent]:
        """Get events within the next N hours."""
        now = datetime.now(timezone.utc)
        return [
            e
            for e in self._upcoming_events
            if (e.date - now).total_seconds() <= hours * 3600
        ]

    def has_major_event_soon(self, hours: int = 24) -> bool:
        """Check if there's a major event in the next N hours."""
        upcoming = self.get_upcoming_events(hours)
        return any(e.importance == "high" for e in upcoming)


class MacroDataCollector:
    """Unified macro data collector combining all data sources.

    Manages:
    - Fear & Greed Index (1 hour polling)
    - Market Indices (5 minute polling)
    - Economic Calendar (1 hour polling)
    """

    def __init__(self) -> None:
        self.fear_greed = FearGreedCollector()
        self.market_index = MarketIndexCollector()
        self.calendar = EconomicCalendarCollector()

        self._callbacks: list[Callable[[MacroContext], None]] = []
        self._last_context: Optional[MacroContext] = None

        # Internal state tracking
        self._fear_greed_data: Optional[FearGreedData] = None
        self._market_data: dict[str, MarketIndexData] = {}

        # Register internal callbacks
        self.fear_greed.on_update(self._on_fear_greed_update)
        self.market_index.on_update(self._on_market_update)

    def _on_fear_greed_update(self, data: FearGreedData) -> None:
        """Handle Fear & Greed update."""
        self._fear_greed_data = data
        self._update_context()

    def _on_market_update(self, data: dict[str, MarketIndexData]) -> None:
        """Handle market index update."""
        self._market_data = data
        self._update_context()

    def _update_context(self) -> None:
        """Build and emit updated MacroContext."""
        try:
            context = MacroContext(
                timestamp=datetime.now(timezone.utc),
                fear_greed_index=self._fear_greed_data.value if self._fear_greed_data else 50,
                fear_greed_label=self._fear_greed_data.value_classification
                if self._fear_greed_data
                else "Neutral",
                dxy=self._market_data.get("DXY").price
                if self._market_data.get("DXY")
                else 0.0,
                dxy_change=self._market_data.get("DXY").change_percent
                if self._market_data.get("DXY")
                else 0.0,
                sp500_change=self._market_data.get("SPX").change_percent
                if self._market_data.get("SPX")
                else 0.0,
                nasdaq_change=self._market_data.get("NDX").change_percent
                if self._market_data.get("NDX")
                else 0.0,
                us10y_yield=self._market_data.get("US10Y").price
                if self._market_data.get("US10Y")
                else 0.0,
                upcoming_events=self.calendar.get_upcoming_events(24),
            )

            self._last_context = context

            for callback in self._callbacks:
                try:
                    callback(context)
                except Exception as e:
                    logger.error("Error in macro context callback", error=str(e))

        except Exception as e:
            logger.error("Error building macro context", error=str(e))

    def on_update(self, callback: Callable[[MacroContext], None]) -> None:
        """Register callback for context updates."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start all collectors."""
        logger.info("Starting macro data collectors")

        # Fetch initial data
        await self._initial_fetch()

        # Start polling loops
        await asyncio.gather(
            self.fear_greed.start_polling(),
            self.market_index.start_polling(),
            self.calendar.start_polling(),
        )

    async def _initial_fetch(self) -> None:
        """Perform initial data fetch for all collectors."""
        try:
            # Fetch in parallel
            fear_greed_task = asyncio.create_task(self.fear_greed.fetch())
            market_task = asyncio.create_task(self.market_index.fetch_all())
            calendar_task = asyncio.create_task(self.calendar.fetch_upcoming_events())

            results = await asyncio.gather(
                fear_greed_task,
                market_task,
                calendar_task,
                return_exceptions=True,
            )

            if isinstance(results[0], FearGreedData):
                self._fear_greed_data = results[0]

            if isinstance(results[1], dict):
                self._market_data = results[1]

            self._update_context()

            logger.info("Initial macro data fetch complete")

        except Exception as e:
            logger.error("Error in initial macro fetch", error=str(e))

    async def stop(self) -> None:
        """Stop all collectors."""
        await asyncio.gather(
            self.fear_greed.stop(),
            self.market_index.stop(),
            self.calendar.stop(),
        )
        logger.info("Macro data collectors stopped")

    def get_context(self) -> Optional[MacroContext]:
        """Get current macro context."""
        return self._last_context

    async def fetch_once(self) -> MacroContext:
        """Fetch all data once without starting polling.

        Returns:
            Current MacroContext
        """
        await self._initial_fetch()
        return self._last_context
