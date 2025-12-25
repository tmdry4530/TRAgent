"""Tests for Binance WebSocket and REST collectors."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


class TestKlineData:
    """Tests for KlineData dataclass."""

    def test_create_kline_data(self) -> None:
        """Test creating a KlineData instance."""
        kline = KlineData(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.5,
            interval="1m",
            is_closed=True,
        )
        assert kline.open == 50000.0
        assert kline.high == 50100.0
        assert kline.low == 49900.0
        assert kline.close == 50050.0
        assert kline.volume == 100.5
        assert kline.interval == "1m"
        assert kline.is_closed is True


class TestLiquidationData:
    """Tests for LiquidationData dataclass."""

    def test_create_liquidation_data(self) -> None:
        """Test creating a LiquidationData instance."""
        liq = LiquidationData(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            symbol="BTCUSDT",
            side="SELL",
            price=50000.0,
            quantity=1.0,
            usd_value=50000.0,
        )
        assert liq.symbol == "BTCUSDT"
        assert liq.side == "SELL"
        assert liq.price == 50000.0
        assert liq.quantity == 1.0
        assert liq.usd_value == 50000.0


class TestBinanceWebSocketCollector:
    """Tests for BinanceWebSocketCollector."""

    @pytest.fixture
    def collector(self) -> BinanceWebSocketCollector:
        """Create a collector instance for testing."""
        with patch("src.collectors.binance.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                trading_symbol="BTCUSDT",
                binance_ws_url="wss://fstream.binance.com",
                binance_testnet=False,
            )
            return BinanceWebSocketCollector()

    def test_initial_state(self, collector: BinanceWebSocketCollector) -> None:
        """Test initial state of collector."""
        assert collector.is_connected is False
        assert collector.is_running is False
        assert collector.streams == []

    @pytest.mark.asyncio
    async def test_subscribe_klines(self, collector: BinanceWebSocketCollector) -> None:
        """Test subscribing to kline streams."""
        await collector.subscribe_klines(["1m", "15m"])
        assert "btcusdt@kline_1m" in collector.streams
        assert "btcusdt@kline_15m" in collector.streams

    @pytest.mark.asyncio
    async def test_subscribe_klines_no_duplicates(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test that subscribing to the same stream twice doesn't duplicate."""
        await collector.subscribe_klines(["1m"])
        await collector.subscribe_klines(["1m"])
        assert collector.streams.count("btcusdt@kline_1m") == 1

    @pytest.mark.asyncio
    async def test_subscribe_orderbook(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test subscribing to orderbook stream."""
        await collector.subscribe_orderbook()
        assert "btcusdt@depth20@100ms" in collector.streams

    @pytest.mark.asyncio
    async def test_subscribe_trades(self, collector: BinanceWebSocketCollector) -> None:
        """Test subscribing to trade stream."""
        await collector.subscribe_trades()
        assert "btcusdt@aggTrade" in collector.streams

    @pytest.mark.asyncio
    async def test_subscribe_liquidations(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test subscribing to liquidation stream."""
        await collector.subscribe_liquidations()
        assert "btcusdt@forceOrder" in collector.streams

    def test_register_kline_callback(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test registering a kline callback."""
        callback = MagicMock()
        collector.on_kline(callback)
        assert callback in collector._kline_callbacks

    def test_register_orderbook_callback(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test registering an orderbook callback."""
        callback = MagicMock()
        collector.on_orderbook(callback)
        assert callback in collector._orderbook_callbacks

    def test_register_trade_callback(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test registering a trade callback."""
        callback = MagicMock()
        collector.on_trade(callback)
        assert callback in collector._trade_callbacks

    def test_register_liquidation_callback(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test registering a liquidation callback."""
        callback = MagicMock()
        collector.on_liquidation(callback)
        assert callback in collector._liquidation_callbacks

    @pytest.mark.asyncio
    async def test_handle_kline_message(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test handling a kline message."""
        callback = MagicMock()
        collector.on_kline(callback)

        # Simulate kline message
        kline_data = {
            "k": {
                "t": 1735689600000,  # 2025-01-01 00:00:00
                "o": "50000.00",
                "h": "50100.00",
                "l": "49900.00",
                "c": "50050.00",
                "v": "100.50",
                "i": "1m",
                "x": True,
            }
        }

        await collector._handle_kline(kline_data)

        callback.assert_called_once()
        kline = callback.call_args[0][0]
        assert isinstance(kline, KlineData)
        assert kline.open == 50000.0
        assert kline.is_closed is True

    @pytest.mark.asyncio
    async def test_handle_liquidation_message(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test handling a liquidation message."""
        callback = MagicMock()
        collector.on_liquidation(callback)

        # Simulate liquidation message
        liq_data = {
            "o": {
                "T": 1735689600000,
                "s": "BTCUSDT",
                "S": "SELL",
                "p": "50000.00",
                "q": "1.0",
            }
        }

        await collector._handle_liquidation(liq_data)

        callback.assert_called_once()
        liq = callback.call_args[0][0]
        assert isinstance(liq, LiquidationData)
        assert liq.side == "SELL"
        assert liq.usd_value == 50000.0

    @pytest.mark.asyncio
    async def test_process_message_routes_kline(
        self, collector: BinanceWebSocketCollector
    ) -> None:
        """Test that _process_message correctly routes kline messages."""
        callback = MagicMock()
        collector.on_kline(callback)

        message = json.dumps(
            {
                "stream": "btcusdt@kline_1m",
                "data": {
                    "k": {
                        "t": 1735689600000,
                        "o": "50000.00",
                        "h": "50100.00",
                        "l": "49900.00",
                        "c": "50050.00",
                        "v": "100.50",
                        "i": "1m",
                        "x": False,
                    }
                },
            }
        )

        await collector._process_message(message)
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, collector: BinanceWebSocketCollector) -> None:
        """Test disconnecting the collector."""
        collector.ws = AsyncMock()
        collector.ws.closed = False

        await collector.disconnect()

        assert collector.is_running is False
        assert collector.is_connected is False


class TestBinanceRestCollector:
    """Tests for BinanceRestCollector."""

    @pytest.fixture
    def collector(self) -> BinanceRestCollector:
        """Create a REST collector instance for testing."""
        with patch("src.collectors.binance.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(binance_testnet=False)
            return BinanceRestCollector(base_url="https://fapi.binance.com")

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, collector: BinanceRestCollector) -> None:
        """Test getting funding rate."""
        mock_funding_response = [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": 1735689600000,
            }
        ]
        mock_premium_response = {"markPrice": "50000.00"}

        with patch.object(
            collector,
            "_request",
            side_effect=[mock_funding_response, mock_premium_response],
        ):
            result = await collector.get_funding_rate("BTCUSDT")

            assert isinstance(result, FundingRateData)
            assert result.symbol == "BTCUSDT"
            assert result.funding_rate == 0.0001
            assert result.mark_price == 50000.0

    @pytest.mark.asyncio
    async def test_get_open_interest(self, collector: BinanceRestCollector) -> None:
        """Test getting open interest."""
        mock_response = {
            "symbol": "BTCUSDT",
            "openInterest": "10000.0",
            "time": 1735689600000,
        }

        with patch.object(collector, "_request", return_value=mock_response):
            result = await collector.get_open_interest("BTCUSDT")

            assert isinstance(result, OpenInterestData)
            assert result.symbol == "BTCUSDT"
            assert result.open_interest == 10000.0

    @pytest.mark.asyncio
    async def test_get_long_short_ratio(self, collector: BinanceRestCollector) -> None:
        """Test getting long/short ratio."""
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "longAccount": "0.55",
                "shortAccount": "0.45",
                "longShortRatio": "1.222",
                "timestamp": 1735689600000,
            }
        ]

        with patch.object(collector, "_request", return_value=mock_response):
            result = await collector.get_long_short_ratio("BTCUSDT", "5m")

            assert isinstance(result, LongShortRatioData)
            assert result.symbol == "BTCUSDT"
            assert result.long_account == 0.55
            assert result.short_account == 0.45
            assert result.long_short_ratio == 1.222

    @pytest.mark.asyncio
    async def test_get_funding_rate_empty_response(
        self, collector: BinanceRestCollector
    ) -> None:
        """Test handling empty funding rate response."""
        with patch.object(collector, "_request", return_value=[]):
            with pytest.raises(ValueError, match="No funding rate data"):
                await collector.get_funding_rate("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_long_short_ratio_empty_response(
        self, collector: BinanceRestCollector
    ) -> None:
        """Test handling empty long/short ratio response."""
        with patch.object(collector, "_request", return_value=[]):
            with pytest.raises(ValueError, match="No long/short ratio data"):
                await collector.get_long_short_ratio("BTCUSDT")

    @pytest.mark.asyncio
    async def test_close_session(self, collector: BinanceRestCollector) -> None:
        """Test closing the aiohttp session."""
        collector.session = AsyncMock()
        collector.session.closed = False

        await collector.close()

        collector.session.close.assert_called_once()


class TestDataClassIntegrity:
    """Tests for data class integrity."""

    def test_orderbook_data(self) -> None:
        """Test OrderbookData structure."""
        orderbook = OrderbookData(
            timestamp=datetime.now(),
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.5), (50002.0, 0.5)],
        )
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0] == (50000.0, 1.0)

    def test_trade_data(self) -> None:
        """Test TradeData structure."""
        trade = TradeData(
            timestamp=datetime.now(),
            price=50000.0,
            quantity=1.5,
            is_buyer_maker=True,
            trade_id=12345,
        )
        assert trade.price == 50000.0
        assert trade.quantity == 1.5
        assert trade.is_buyer_maker is True
        assert trade.trade_id == 12345
