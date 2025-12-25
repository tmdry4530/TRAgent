"""Tests for macro data collectors."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class TestFearGreedCollector:
    """Tests for FearGreedCollector."""

    @pytest.fixture
    def collector(self) -> FearGreedCollector:
        """Create collector instance."""
        return FearGreedCollector()

    @pytest.mark.asyncio
    async def test_fetch_success(self, collector: FearGreedCollector) -> None:
        """Test successful Fear & Greed fetch."""
        mock_response = {
            "data": [
                {
                    "value": "25",
                    "value_classification": "Extreme Fear",
                    "timestamp": "1734567890",
                    "time_until_update": "3600",
                }
            ]
        }

        # Create proper async context manager mock
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = mock_cm

        with patch.object(
            collector, "_ensure_session", new_callable=AsyncMock, return_value=session
        ):
            result = await collector.fetch()

            assert isinstance(result, FearGreedData)
            assert result.value == 25
            assert result.value_classification == "Extreme Fear"

    @pytest.mark.asyncio
    async def test_fetch_api_error(self, collector: FearGreedCollector) -> None:
        """Test handling of API errors."""
        mock_resp = MagicMock()
        mock_resp.status = 500

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = mock_cm

        with patch.object(
            collector, "_ensure_session", new_callable=AsyncMock, return_value=session
        ):
            with pytest.raises(ValueError, match="API error"):
                await collector.fetch()

    def test_on_update_callback(self, collector: FearGreedCollector) -> None:
        """Test callback registration."""
        callback = MagicMock()
        collector.on_update(callback)

        assert callback in collector._callbacks

    def test_last_value_none(self, collector: FearGreedCollector) -> None:
        """Test last_value when no data fetched."""
        assert collector.last_value is None


class TestMarketIndexCollector:
    """Tests for MarketIndexCollector."""

    @pytest.fixture
    def collector(self) -> MarketIndexCollector:
        """Create collector instance."""
        return MarketIndexCollector()

    def test_symbols_defined(self, collector: MarketIndexCollector) -> None:
        """Test that all expected symbols are defined."""
        assert "DXY" in collector.SYMBOLS
        assert "SPX" in collector.SYMBOLS
        assert "NDX" in collector.SYMBOLS
        assert "US10Y" in collector.SYMBOLS

    @pytest.mark.asyncio
    async def test_fetch_success(self, collector: MarketIndexCollector) -> None:
        """Test successful market index fetch."""
        mock_response = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "previousClose": 100.0,
                        },
                        "indicators": {
                            "quote": [
                                {
                                    "close": [101.0, 102.0],
                                }
                            ]
                        },
                    }
                ]
            }
        }

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = mock_cm

        with patch.object(
            collector, "_ensure_session", new_callable=AsyncMock, return_value=session
        ):
            result = await collector.fetch("DXY")

            assert isinstance(result, MarketIndexData)
            assert result.symbol == "DXY"
            assert result.price == 102.0
            assert result.change == 2.0
            assert result.change_percent == 2.0

    @pytest.mark.asyncio
    async def test_fetch_unknown_symbol(self, collector: MarketIndexCollector) -> None:
        """Test fetching unknown symbol."""
        with pytest.raises(ValueError, match="Unknown symbol"):
            await collector.fetch("UNKNOWN")

    def test_get_helpers(self, collector: MarketIndexCollector) -> None:
        """Test helper methods when no data."""
        assert collector.get_dxy() is None
        assert collector.get_sp500_change() is None
        assert collector.get_nasdaq_change() is None
        assert collector.get_us10y_yield() is None


class TestEconomicCalendarCollector:
    """Tests for EconomicCalendarCollector."""

    @pytest.fixture
    def collector(self) -> EconomicCalendarCollector:
        """Create collector instance."""
        return EconomicCalendarCollector()

    @pytest.mark.asyncio
    async def test_fetch_upcoming_events(
        self, collector: EconomicCalendarCollector
    ) -> None:
        """Test fetching upcoming events."""
        events = await collector.fetch_upcoming_events()

        assert isinstance(events, list)

    def test_get_upcoming_events_empty(
        self, collector: EconomicCalendarCollector
    ) -> None:
        """Test getting upcoming events when none exist."""
        events = collector.get_upcoming_events(24)
        assert events == []

    def test_has_major_event_soon_false(
        self, collector: EconomicCalendarCollector
    ) -> None:
        """Test major event check when none exist."""
        assert collector.has_major_event_soon(24) is False


class TestMacroDataCollector:
    """Tests for MacroDataCollector."""

    @pytest.fixture
    def collector(self) -> MacroDataCollector:
        """Create collector instance."""
        return MacroDataCollector()

    def test_init_creates_sub_collectors(
        self, collector: MacroDataCollector
    ) -> None:
        """Test that sub-collectors are created."""
        assert collector.fear_greed is not None
        assert collector.market_index is not None
        assert collector.calendar is not None

    def test_on_update_callback(self, collector: MacroDataCollector) -> None:
        """Test callback registration."""
        callback = MagicMock()
        collector.on_update(callback)

        assert callback in collector._callbacks

    def test_get_context_none(self, collector: MacroDataCollector) -> None:
        """Test get_context when no data."""
        assert collector.get_context() is None

    def test_update_context_with_data(self, collector: MacroDataCollector) -> None:
        """Test context update with mock data."""
        # Set up mock data
        collector._fear_greed_data = FearGreedData(
            value=30,
            value_classification="Fear",
            timestamp=datetime.now(timezone.utc),
            time_until_update=3600,
        )

        collector._market_data = {
            "DXY": MarketIndexData(
                symbol="DXY",
                name="US Dollar Index",
                price=104.5,
                change=0.5,
                change_percent=0.48,
                timestamp=datetime.now(timezone.utc),
            ),
            "SPX": MarketIndexData(
                symbol="SPX",
                name="S&P 500",
                price=4500.0,
                change=50.0,
                change_percent=1.12,
                timestamp=datetime.now(timezone.utc),
            ),
            "NDX": MarketIndexData(
                symbol="NDX",
                name="NASDAQ Composite",
                price=15000.0,
                change=100.0,
                change_percent=0.67,
                timestamp=datetime.now(timezone.utc),
            ),
            "US10Y": MarketIndexData(
                symbol="US10Y",
                name="US 10Y Treasury Yield",
                price=4.25,
                change=0.05,
                change_percent=1.19,
                timestamp=datetime.now(timezone.utc),
            ),
        }

        # Trigger update
        collector._update_context()

        context = collector.get_context()
        assert context is not None
        assert context.fear_greed_index == 30
        assert context.fear_greed_label == "Fear"
        assert context.dxy == 104.5
        assert context.sp500_change == 1.12
        assert context.nasdaq_change == 0.67
        assert context.us10y_yield == 4.25


class TestMacroContext:
    """Tests for MacroContext dataclass."""

    def test_create_macro_context(self) -> None:
        """Test creating MacroContext."""
        context = MacroContext(
            timestamp=datetime.now(timezone.utc),
            fear_greed_index=50,
            fear_greed_label="Neutral",
            dxy=104.0,
            dxy_change=0.1,
            sp500_change=0.5,
            nasdaq_change=-0.2,
            us10y_yield=4.0,
        )

        assert context.fear_greed_index == 50
        assert context.fear_greed_label == "Neutral"
        assert context.upcoming_events == []

    def test_create_macro_context_with_events(self) -> None:
        """Test MacroContext with events."""
        event = EconomicEvent(
            name="FOMC Meeting",
            date=datetime.now(timezone.utc),
            importance="high",
            country="US",
        )

        context = MacroContext(
            timestamp=datetime.now(timezone.utc),
            fear_greed_index=50,
            fear_greed_label="Neutral",
            dxy=104.0,
            dxy_change=0.1,
            sp500_change=0.5,
            nasdaq_change=-0.2,
            us10y_yield=4.0,
            upcoming_events=[event],
        )

        assert len(context.upcoming_events) == 1
        assert context.upcoming_events[0].name == "FOMC Meeting"
