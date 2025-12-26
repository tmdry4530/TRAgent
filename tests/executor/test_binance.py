"""Unit tests for BinanceExecutor."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.executor.binance import (
    BinanceExecutor,
    OrderResult,
    OrderStatus,
    AccountInfo,
    PositionInfo,
)
from src.signals.base import Signal


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.trading_symbol = "BTCUSDT"
    settings.binance_testnet = True
    settings.binance_rest_url = "https://testnet.binancefuture.com"
    settings.binance_api_key = MagicMock()
    settings.binance_api_key.get_secret_value.return_value = "test_api_key"
    settings.binance_secret_key = MagicMock()
    settings.binance_secret_key.get_secret_value.return_value = "test_secret_key"
    settings.scalp_max_position_pct = 0.30
    settings.scalp_leverage = 20
    settings.swing_max_position_pct = 0.50
    settings.swing_leverage = 10
    return settings


@pytest.fixture
def executor(mock_settings):
    """Create BinanceExecutor with mocked settings."""
    with patch("src.executor.binance.get_settings", return_value=mock_settings):
        return BinanceExecutor()


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return Signal(
        type="SCALP",
        direction="LONG",
        confidence=0.85,
        entry_price=50000.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        reason="Test signal",
        timestamp=datetime.now(timezone.utc),
    )


class TestBinanceExecutorInit:
    """Tests for BinanceExecutor initialization."""

    def test_init_with_testnet(self, executor, mock_settings):
        """Test initialization with testnet settings."""
        assert executor.symbol == "BTCUSDT"
        assert executor.base_url == "https://testnet.binancefuture.com"
        assert executor.api_key == "test_api_key"
        assert executor.secret_key == "test_secret_key"

    def test_init_sets_rate_limit(self, executor):
        """Test rate limiting is configured."""
        assert executor.min_order_interval == 0.1


class TestSignRequest:
    """Tests for request signing."""

    def test_sign_request_adds_timestamp(self, executor):
        """Test that signing adds timestamp."""
        params = {"symbol": "BTCUSDT"}
        query_string = executor._sign_request(params)

        # Should be a string containing timestamp
        assert isinstance(query_string, str)
        assert "timestamp=" in query_string

    def test_sign_request_adds_signature(self, executor):
        """Test that signing adds signature."""
        params = {"symbol": "BTCUSDT"}
        query_string = executor._sign_request(params)

        # Should contain 64-char hex signature
        assert "signature=" in query_string
        # Extract signature value
        signature = query_string.split("signature=")[1].split("&")[0]
        assert len(signature) == 64  # SHA256 hex length


class TestAccountInfo:
    """Tests for account info retrieval."""

    @pytest.mark.asyncio
    async def test_get_account_info(self, executor):
        """Test getting account information."""
        mock_response = {
            "totalWalletBalance": "10000.0",
            "availableBalance": "8000.0",
            "totalUnrealizedProfit": "500.0",
            "totalMarginBalance": "10500.0",
            "positions": [
                {"symbol": "BTCUSDT", "positionAmt": "0.1"},
                {"symbol": "ETHUSDT", "positionAmt": "0"},
            ],
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            account = await executor.get_account_info()

            assert isinstance(account, AccountInfo)
            assert account.total_balance == 10000.0
            assert account.available_balance == 8000.0
            assert account.total_unrealized_pnl == 500.0
            assert len(account.positions) == 1  # Only non-zero positions


class TestPositionInfo:
    """Tests for position info retrieval."""

    @pytest.mark.asyncio
    async def test_get_position_long(self, executor):
        """Test getting a long position."""
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.1",
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "100.0",
                "leverage": "20",
                "marginType": "cross",
                "liquidationPrice": "45000.0",
            }
        ]

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            position = await executor.get_position()

            assert isinstance(position, PositionInfo)
            assert position.side == "LONG"
            assert position.quantity == 0.1
            assert position.entry_price == 50000.0

    @pytest.mark.asyncio
    async def test_get_position_short(self, executor):
        """Test getting a short position."""
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "-0.1",
                "entryPrice": "50000.0",
                "markPrice": "49000.0",
                "unRealizedProfit": "100.0",
                "leverage": "20",
                "marginType": "isolated",
                "liquidationPrice": "55000.0",
            }
        ]

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            position = await executor.get_position()

            assert position.side == "SHORT"
            assert position.quantity == 0.1
            assert position.margin_type == "ISOLATED"

    @pytest.mark.asyncio
    async def test_get_position_none(self, executor):
        """Test when no position exists."""
        mock_response = [
            {"symbol": "BTCUSDT", "positionAmt": "0"}
        ]

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            position = await executor.get_position()

            assert position is None


class TestLeverageAndMargin:
    """Tests for leverage and margin type settings."""

    @pytest.mark.asyncio
    async def test_set_leverage(self, executor):
        """Test setting leverage."""
        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"leverage": 20}
            result = await executor.set_leverage(20)

            assert result is True
            mock_req.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_margin_type(self, executor):
        """Test setting margin type."""
        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {}
            result = await executor.set_margin_type("CROSSED")

            assert result is True


class TestMarketOrders:
    """Tests for market order placement."""

    @pytest.mark.asyncio
    async def test_place_market_order_buy(self, executor):
        """Test placing a buy market order."""
        mock_response = {
            "orderId": "12345",
            "clientOrderId": "test_client_id",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "0.1",
            "price": "0",
            "avgPrice": "50000.0",
            "executedQty": "0.1",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await executor.place_market_order("BUY", 0.1)

            assert isinstance(result, OrderResult)
            assert result.order_id == "12345"
            assert result.side == "BUY"
            assert result.status == OrderStatus.FILLED
            assert result.is_filled

    @pytest.mark.asyncio
    async def test_place_market_order_sell(self, executor):
        """Test placing a sell market order."""
        mock_response = {
            "orderId": "12346",
            "clientOrderId": "test_client_id",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "0.1",
            "price": "0",
            "avgPrice": "50000.0",
            "executedQty": "0.1",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await executor.place_market_order("SELL", 0.1, reduce_only=True)

            assert result.side == "SELL"
            assert result.is_filled

    @pytest.mark.asyncio
    async def test_place_market_order_failure(self, executor):
        """Test market order failure handling."""
        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = ValueError("Insufficient balance")
            result = await executor.place_market_order("BUY", 0.1)

            assert result.status == OrderStatus.FAILED
            assert "Insufficient balance" in result.error_message


class TestStopOrders:
    """Tests for stop-loss and take-profit orders."""

    @pytest.mark.asyncio
    async def test_place_stop_loss(self, executor):
        """Test placing a stop-loss order."""
        mock_response = {
            "orderId": "12347",
            "clientOrderId": "test_sl",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "status": "NEW",
            "origQty": "0",
            "stopPrice": "49500.0",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await executor.place_stop_loss("SELL", 49500.0)

            assert result.order_type == "STOP_MARKET"
            assert result.price == 49500.0
            assert result.status == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_place_take_profit(self, executor):
        """Test placing a take-profit order."""
        mock_response = {
            "orderId": "12348",
            "clientOrderId": "test_tp",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "TAKE_PROFIT_MARKET",
            "status": "NEW",
            "origQty": "0",
            "stopPrice": "51000.0",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await executor.place_take_profit("SELL", 51000.0)

            assert result.order_type == "TAKE_PROFIT_MARKET"
            assert result.price == 51000.0


class TestOrderManagement:
    """Tests for order management operations."""

    @pytest.mark.asyncio
    async def test_cancel_order(self, executor):
        """Test cancelling an order."""
        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"status": "CANCELED"}
            result = await executor.cancel_order("12345")

            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, executor):
        """Test cancelling all orders."""
        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {}
            result = await executor.cancel_all_orders()

            assert result is True

    @pytest.mark.asyncio
    async def test_get_open_orders(self, executor):
        """Test getting open orders."""
        mock_response = [
            {
                "orderId": "12345",
                "clientOrderId": "test",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "status": "NEW",
                "origQty": "0.1",
                "stopPrice": "49500.0",
                "avgPrice": "0",
                "executedQty": "0",
                "time": 1700000000000,
            }
        ]

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            orders = await executor.get_open_orders()

            assert len(orders) == 1
            assert orders[0].order_id == "12345"


class TestExecuteSignal:
    """Tests for signal execution."""

    @pytest.mark.asyncio
    async def test_execute_signal_success(self, executor, sample_signal):
        """Test successful signal execution."""
        # Mock responses
        leverage_resp = {"leverage": 20}
        entry_resp = {
            "orderId": "100",
            "clientOrderId": "entry",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "0.1",
            "avgPrice": "50000.0",
            "executedQty": "0.1",
        }
        sl_resp = {
            "orderId": "101",
            "clientOrderId": "sl",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "STOP_MARKET",
            "status": "NEW",
            "origQty": "0",
            "stopPrice": "49500.0",
        }
        tp_resp = {
            "orderId": "102",
            "clientOrderId": "tp",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "TAKE_PROFIT_MARKET",
            "status": "NEW",
            "origQty": "0",
            "stopPrice": "51000.0",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            # Only entry order is placed now (SL/TP managed by price monitoring)
            mock_req.side_effect = [leverage_resp, entry_resp]

            entry = await executor.execute_signal(
                signal=sample_signal,
                position_size=1.0,
                account_balance=10000.0,
            )

            # execute_signal now returns single OrderResult (entry only)
            assert entry is not None
            assert entry.order_id == "100"

    @pytest.mark.asyncio
    async def test_execute_signal_quantity_too_small(self, executor, sample_signal):
        """Test signal execution with quantity too small."""
        # Modify signal for very small position
        sample_signal.entry_price = 100000.0  # Very high price

        with patch.object(executor, "set_leverage", new_callable=AsyncMock) as mock_lev:
            mock_lev.return_value = True

            result = await executor.execute_signal(
                signal=sample_signal,
                position_size=0.00001,  # Very small size
                account_balance=1.0,  # Very small balance
            )

            # Should return None due to quantity too small
            assert result is None


class TestClosePosition:
    """Tests for position closing."""

    @pytest.mark.asyncio
    async def test_close_position_long(self, executor):
        """Test closing a long position."""
        position_resp = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.1",
                "entryPrice": "50000.0",
                "markPrice": "51000.0",
                "unRealizedProfit": "100.0",
                "leverage": "20",
                "marginType": "cross",
                "liquidationPrice": "45000.0",
            }
        ]
        cancel_resp = {}
        close_resp = {
            "orderId": "200",
            "clientOrderId": "close",
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "0.1",
            "avgPrice": "51000.0",
            "executedQty": "0.1",
        }

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [position_resp, cancel_resp, close_resp]

            result = await executor.close_position()

            assert result is not None
            assert result.side == "SELL"
            assert result.is_filled

    @pytest.mark.asyncio
    async def test_close_position_no_position(self, executor):
        """Test closing when no position exists."""
        position_resp = [{"symbol": "BTCUSDT", "positionAmt": "0"}]

        with patch.object(executor, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = position_resp

            result = await executor.close_position()

            assert result is None


class TestOrderResultProperties:
    """Tests for OrderResult properties."""

    def test_order_result_is_filled(self):
        """Test is_filled property."""
        order = OrderResult(
            order_id="1",
            client_order_id="test",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            status=OrderStatus.FILLED,
            quantity=0.1,
            price=0,
            avg_price=50000.0,
            filled_qty=0.1,
            timestamp=datetime.now(timezone.utc),
        )
        assert order.is_filled is True

    def test_order_result_is_active(self):
        """Test is_active property."""
        order = OrderResult(
            order_id="1",
            client_order_id="test",
            symbol="BTCUSDT",
            side="BUY",
            order_type="STOP_MARKET",
            status=OrderStatus.NEW,
            quantity=0.1,
            price=49500.0,
            avg_price=0,
            filled_qty=0,
            timestamp=datetime.now(timezone.utc),
        )
        assert order.is_active is True

    def test_order_result_not_active_when_filled(self):
        """Test is_active is False when filled."""
        order = OrderResult(
            order_id="1",
            client_order_id="test",
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            status=OrderStatus.FILLED,
            quantity=0.1,
            price=0,
            avg_price=50000.0,
            filled_qty=0.1,
            timestamp=datetime.now(timezone.utc),
        )
        assert order.is_active is False
