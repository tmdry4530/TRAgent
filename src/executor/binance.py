"""Binance Futures executor for order execution."""

import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from src.signals.base import Signal
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"
    FAILED = "FAILED"


@dataclass
class OrderResult:
    """Result of an order execution."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, STOP_MARKET, etc.
    status: OrderStatus
    quantity: float
    price: float
    avg_price: float
    filled_qty: float
    timestamp: datetime
    error_message: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]


@dataclass
class AccountInfo:
    """Account information."""

    total_balance: float
    available_balance: float
    total_unrealized_pnl: float
    total_margin_balance: float
    positions: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PositionInfo:
    """Position information from Binance."""

    symbol: str
    side: str  # LONG or SHORT
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str  # CROSS or ISOLATED
    liquidation_price: float


class BinanceExecutor:
    """Executor for Binance Futures orders.

    Handles:
    - Market orders for entry
    - Stop-loss orders
    - Take-profit orders
    - Position queries
    - Order status monitoring
    """

    # Binance error codes
    ERROR_CODES = {
        -1000: "Unknown error",
        -1003: "Rate limit exceeded",
        -1021: "Timestamp out of range",
        -2010: "Insufficient balance",
        -2011: "Order cancel failed",
        -2019: "Margin insufficient",
        -4003: "Quantity too small",
    }

    def __init__(self) -> None:
        """Initialize the executor."""
        self.settings = get_settings()
        self.symbol = self.settings.trading_symbol

        # Get API credentials
        api_key = self.settings.binance_api_key
        secret_key = self.settings.binance_secret_key

        self.api_key = api_key.get_secret_value() if api_key else ""
        self.secret_key = secret_key.get_secret_value() if secret_key else ""

        # Base URL based on testnet setting
        self.base_url = self.settings.binance_rest_url

        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0

        # Time offset for server time sync (will be set on first request)
        self._time_offset: int = 0
        self._time_synced: bool = False

        # Rate limiting: 10 orders per second
        self.min_order_interval = 0.1

        logger.info(
            "BinanceExecutor initialized",
            testnet=self.settings.binance_testnet,
            symbol=self.symbol,
            base_url=self.base_url,
        )

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _sync_time(self) -> None:
        """Synchronize local time with Binance server time."""
        if self._time_synced:
            return

        session = await self._ensure_session()
        try:
            async with session.get(f"{self.base_url}/fapi/v1/time") as resp:
                data = await resp.json()
                server_time = data["serverTime"]
                local_time = int(time.time() * 1000)
                self._time_offset = server_time - local_time
                self._time_synced = True
                logger.info(
                    "Time synchronized with Binance server",
                    offset_ms=self._time_offset,
                )
        except Exception as e:
            logger.warning("Failed to sync time with server", error=str(e))
            self._time_offset = 0

    def _sign_request(self, params: dict[str, Any]) -> str:
        """Sign request with HMAC SHA256.

        Args:
            params: Request parameters

        Returns:
            Query string with timestamp and signature added
        """
        # Apply time offset for server sync
        params["timestamp"] = int(time.time() * 1000) + self._time_offset
        params["recvWindow"] = 5000  # 5 second tolerance

        # Build query string (order doesn't matter, but must be consistent)
        query_string = "&".join(f"{k}={v}" for k, v in params.items())

        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Return the full query string with signature appended
        return f"{query_string}&signature={signature}"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key."""
        return {"X-MBX-APIKEY": self.api_key}

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.min_order_interval:
            await asyncio.sleep(self.min_order_interval - time_since_last)

        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        signed: bool = True,
    ) -> dict[str, Any]:
        """Make HTTP request to Binance API.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request

        Returns:
            JSON response

        Raises:
            ValueError: On API errors
        """
        await self._rate_limit()

        # Sync time with server before signed requests
        if signed:
            await self._sync_time()

        session = await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        params = params or {}
        headers = self._get_headers()

        # For signed requests, build query string manually to ensure consistency
        if signed:
            query_string = self._sign_request(params)
            full_url = f"{url}?{query_string}"
            request_params = None
        else:
            full_url = url
            request_params = params if params else None

        try:
            if method == "GET":
                if signed:
                    async with session.get(full_url, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with session.get(full_url, params=request_params, headers=headers) as resp:
                        data = await resp.json()
            elif method == "POST":
                if signed:
                    async with session.post(full_url, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with session.post(full_url, params=request_params, headers=headers) as resp:
                        data = await resp.json()
            elif method == "DELETE":
                if signed:
                    async with session.delete(full_url, headers=headers) as resp:
                        data = await resp.json()
                else:
                    async with session.delete(full_url, params=request_params, headers=headers) as resp:
                        data = await resp.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for errors (code 200 is success for some endpoints)
            if isinstance(data, dict) and "code" in data and data["code"] != 200:
                error_code = data["code"]
                error_msg = data.get("msg", self.ERROR_CODES.get(error_code, "Unknown"))
                logger.error(
                    "Binance API error",
                    code=error_code,
                    message=error_msg,
                    endpoint=endpoint,
                )
                raise ValueError(f"Binance API error {error_code}: {error_msg}")

            return data

        except aiohttp.ClientError as e:
            logger.error("Network error", error=str(e), endpoint=endpoint)
            raise

    async def get_account_info(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo object with balance and position data
        """
        data = await self._request("GET", "/fapi/v2/account")

        positions = []
        for pos in data.get("positions", []):
            if float(pos.get("positionAmt", 0)) != 0:
                positions.append(pos)

        return AccountInfo(
            total_balance=float(data.get("totalWalletBalance", 0)),
            available_balance=float(data.get("availableBalance", 0)),
            total_unrealized_pnl=float(data.get("totalUnrealizedProfit", 0)),
            total_margin_balance=float(data.get("totalMarginBalance", 0)),
            positions=positions,
        )

    async def get_position(self, symbol: Optional[str] = None) -> Optional[PositionInfo]:
        """Get position information for a symbol.

        Args:
            symbol: Trading symbol (default: configured symbol)

        Returns:
            PositionInfo if position exists, None otherwise
        """
        symbol = symbol or self.symbol
        data = await self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})

        for pos in data:
            position_amt = float(pos.get("positionAmt", 0))
            if position_amt != 0:
                return PositionInfo(
                    symbol=pos["symbol"],
                    side="LONG" if position_amt > 0 else "SHORT",
                    quantity=abs(position_amt),
                    entry_price=float(pos.get("entryPrice", 0)),
                    mark_price=float(pos.get("markPrice", 0)),
                    unrealized_pnl=float(pos.get("unRealizedProfit", 0)),
                    leverage=int(pos.get("leverage", 1)),
                    margin_type=pos.get("marginType", "CROSS").upper(),
                    liquidation_price=float(pos.get("liquidationPrice", 0)),
                )

        return None

    async def set_leverage(self, leverage: int, symbol: Optional[str] = None) -> bool:
        """Set leverage for a symbol.

        Args:
            leverage: Leverage value (1-125)
            symbol: Trading symbol (default: configured symbol)

        Returns:
            True if successful
        """
        symbol = symbol or self.symbol
        try:
            await self._request(
                "POST",
                "/fapi/v1/leverage",
                {"symbol": symbol, "leverage": leverage},
            )
            logger.info("Leverage set", symbol=symbol, leverage=leverage)
            return True
        except Exception as e:
            logger.error("Failed to set leverage", error=str(e))
            return False

    async def set_margin_type(
        self, margin_type: str = "CROSSED", symbol: Optional[str] = None
    ) -> bool:
        """Set margin type for a symbol.

        Args:
            margin_type: CROSSED or ISOLATED
            symbol: Trading symbol (default: configured symbol)

        Returns:
            True if successful
        """
        symbol = symbol or self.symbol
        try:
            await self._request(
                "POST",
                "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": margin_type},
            )
            logger.info("Margin type set", symbol=symbol, margin_type=margin_type)
            return True
        except ValueError as e:
            # Already set to this margin type
            if "No need to change" in str(e) or "-4046" in str(e):
                return True
            logger.error("Failed to set margin type", error=str(e))
            return False

    async def place_market_order(
        self,
        side: str,
        quantity: float,
        symbol: Optional[str] = None,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a market order.

        Args:
            side: BUY or SELL
            quantity: Order quantity
            symbol: Trading symbol (default: configured symbol)
            reduce_only: Whether this is a reduce-only order

        Returns:
            OrderResult with order details
        """
        symbol = symbol or self.symbol

        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        try:
            data = await self._request("POST", "/fapi/v1/order", params)

            result = OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", symbol),
                side=data.get("side", side),
                order_type="MARKET",
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", quantity)),
                price=float(data.get("price", 0)),
                avg_price=float(data.get("avgPrice", 0)),
                filled_qty=float(data.get("executedQty", 0)),
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                "Market order placed",
                order_id=result.order_id,
                side=side,
                quantity=quantity,
                status=result.status.value,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to place market order",
                side=side,
                quantity=quantity,
                error=str(e),
            )
            return OrderResult(
                order_id="",
                client_order_id="",
                symbol=symbol,
                side=side,
                order_type="MARKET",
                status=OrderStatus.FAILED,
                quantity=quantity,
                price=0,
                avg_price=0,
                filled_qty=0,
                timestamp=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def place_stop_loss(
        self,
        side: str,
        stop_price: float,
        quantity: Optional[float] = None,
        symbol: Optional[str] = None,
        close_position: bool = True,
    ) -> OrderResult:
        """Place a stop-loss market order.

        Args:
            side: SELL for long position, BUY for short position
            stop_price: Trigger price
            quantity: Order quantity (not needed if close_position=True)
            symbol: Trading symbol (default: configured symbol)
            close_position: Whether to close entire position

        Returns:
            OrderResult with order details
        """
        symbol = symbol or self.symbol

        params = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price,
        }

        if close_position:
            params["closePosition"] = "true"
        elif quantity:
            params["quantity"] = quantity

        try:
            data = await self._request("POST", "/fapi/v1/order", params)

            result = OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", symbol),
                side=data.get("side", side),
                order_type="STOP_MARKET",
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", quantity or 0)),
                price=float(data.get("stopPrice", stop_price)),
                avg_price=0,
                filled_qty=0,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                "Stop-loss order placed",
                order_id=result.order_id,
                side=side,
                stop_price=stop_price,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to place stop-loss",
                side=side,
                stop_price=stop_price,
                error=str(e),
            )
            return OrderResult(
                order_id="",
                client_order_id="",
                symbol=symbol,
                side=side,
                order_type="STOP_MARKET",
                status=OrderStatus.FAILED,
                quantity=quantity or 0,
                price=stop_price,
                avg_price=0,
                filled_qty=0,
                timestamp=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def place_take_profit(
        self,
        side: str,
        take_profit_price: float,
        quantity: Optional[float] = None,
        symbol: Optional[str] = None,
        close_position: bool = True,
    ) -> OrderResult:
        """Place a take-profit market order.

        Args:
            side: SELL for long position, BUY for short position
            take_profit_price: Trigger price
            quantity: Order quantity (not needed if close_position=True)
            symbol: Trading symbol (default: configured symbol)
            close_position: Whether to close entire position

        Returns:
            OrderResult with order details
        """
        symbol = symbol or self.symbol

        params = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit_price,
        }

        if close_position:
            params["closePosition"] = "true"
        elif quantity:
            params["quantity"] = quantity

        try:
            data = await self._request("POST", "/fapi/v1/order", params)

            result = OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", symbol),
                side=data.get("side", side),
                order_type="TAKE_PROFIT_MARKET",
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", quantity or 0)),
                price=float(data.get("stopPrice", take_profit_price)),
                avg_price=0,
                filled_qty=0,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                "Take-profit order placed",
                order_id=result.order_id,
                side=side,
                take_profit_price=take_profit_price,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to place take-profit",
                side=side,
                take_profit_price=take_profit_price,
                error=str(e),
            )
            return OrderResult(
                order_id="",
                client_order_id="",
                symbol=symbol,
                side=side,
                order_type="TAKE_PROFIT_MARKET",
                status=OrderStatus.FAILED,
                quantity=quantity or 0,
                price=take_profit_price,
                avg_price=0,
                filled_qty=0,
                timestamp=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def cancel_order(
        self, order_id: str, symbol: Optional[str] = None
    ) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol (default: configured symbol)

        Returns:
            True if successful
        """
        symbol = symbol or self.symbol

        try:
            await self._request(
                "DELETE",
                "/fapi/v1/order",
                {"symbol": symbol, "orderId": order_id},
            )
            logger.info("Order cancelled", order_id=order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders for a symbol.

        Args:
            symbol: Trading symbol (default: configured symbol)

        Returns:
            True if successful
        """
        symbol = symbol or self.symbol

        try:
            await self._request(
                "DELETE",
                "/fapi/v1/allOpenOrders",
                {"symbol": symbol},
            )
            logger.info("All orders cancelled", symbol=symbol)
            return True
        except Exception as e:
            logger.error("Failed to cancel all orders", error=str(e))
            return False

    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> list[OrderResult]:
        """Get all open orders for a symbol.

        Args:
            symbol: Trading symbol (default: configured symbol)

        Returns:
            List of open OrderResult objects
        """
        symbol = symbol or self.symbol

        try:
            data = await self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol})

            orders = []
            for order in data:
                orders.append(
                    OrderResult(
                        order_id=str(order.get("orderId", "")),
                        client_order_id=order.get("clientOrderId", ""),
                        symbol=order.get("symbol", symbol),
                        side=order.get("side", ""),
                        order_type=order.get("type", ""),
                        status=OrderStatus(order.get("status", "NEW")),
                        quantity=float(order.get("origQty", 0)),
                        price=float(order.get("price", 0) or order.get("stopPrice", 0)),
                        avg_price=float(order.get("avgPrice", 0)),
                        filled_qty=float(order.get("executedQty", 0)),
                        timestamp=datetime.fromtimestamp(
                            order.get("time", 0) / 1000, tz=timezone.utc
                        ),
                    )
                )

            return orders

        except Exception as e:
            logger.error("Failed to get open orders", error=str(e))
            return []

    async def get_order_status(
        self, order_id: str, symbol: Optional[str] = None
    ) -> Optional[OrderResult]:
        """Get status of a specific order.

        Args:
            order_id: Order ID to query
            symbol: Trading symbol (default: configured symbol)

        Returns:
            OrderResult if found, None otherwise
        """
        symbol = symbol or self.symbol

        try:
            data = await self._request(
                "GET",
                "/fapi/v1/order",
                {"symbol": symbol, "orderId": order_id},
            )

            return OrderResult(
                order_id=str(data.get("orderId", "")),
                client_order_id=data.get("clientOrderId", ""),
                symbol=data.get("symbol", symbol),
                side=data.get("side", ""),
                order_type=data.get("type", ""),
                status=OrderStatus(data.get("status", "NEW")),
                quantity=float(data.get("origQty", 0)),
                price=float(data.get("price", 0) or data.get("stopPrice", 0)),
                avg_price=float(data.get("avgPrice", 0)),
                filled_qty=float(data.get("executedQty", 0)),
                timestamp=datetime.fromtimestamp(
                    data.get("time", 0) / 1000, tz=timezone.utc
                ),
            )

        except Exception as e:
            logger.error("Failed to get order status", order_id=order_id, error=str(e))
            return None

    async def execute_signal(
        self,
        signal: Signal,
        position_size: float = 1.0,
        account_balance: float = 0.0,
        quantity: Optional[float] = None,
        leverage: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> Optional[OrderResult]:
        """Execute a trading signal with market entry order.

        Note: SL/TP are managed by price monitoring in main loop,
        not via Binance conditional orders (due to API limitations).

        Args:
            signal: Trading signal to execute
            position_size: Position size multiplier (0.0 to 1.0), ignored if quantity provided
            account_balance: Current account balance, ignored if quantity provided
            quantity: Direct quantity to trade (overrides position_size calculation)
            leverage: Leverage to use (overrides default)
            symbol: Trading symbol (default: configured symbol)

        Returns:
            Entry order result
        """
        symbol = symbol or self.symbol
        # Determine leverage
        if leverage is None:
            if signal.type == "SCALP":
                leverage = self.settings.scalp_leverage
            else:  # SWING
                leverage = self.settings.swing_leverage

        # Set leverage
        await self.set_leverage(leverage, symbol)

        # Calculate quantity if not provided directly
        if quantity is None:
            if signal.type == "SCALP":
                max_position_pct = self.settings.scalp_max_position_pct
            else:  # SWING
                max_position_pct = self.settings.swing_max_position_pct

            position_value = account_balance * max_position_pct * position_size
            quantity = round(position_value * leverage / signal.entry_price, 3)
        else:
            quantity = round(quantity, 3)

        # Minimum quantity check
        if quantity < 0.001:
            logger.warning("Quantity too small", quantity=quantity)
            return None

        # Determine order side
        entry_side = "BUY" if signal.direction == "LONG" else "SELL"

        # Place entry order only (SL/TP managed by price monitoring)
        entry_order = await self.place_market_order(entry_side, quantity, symbol)

        if not entry_order.is_filled and entry_order.status != OrderStatus.NEW:
            logger.error("Entry order failed", status=entry_order.status.value)
            return entry_order

        logger.info(
            "Signal executed (SL/TP via price monitoring)",
            symbol=symbol,
            signal_type=signal.type,
            direction=signal.direction,
            quantity=quantity,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

        return entry_order

    async def close_position(
        self, symbol: Optional[str] = None
    ) -> Optional[OrderResult]:
        """Close the current position for a symbol.

        Args:
            symbol: Trading symbol (default: configured symbol)

        Returns:
            OrderResult if successful, None otherwise
        """
        position = await self.get_position(symbol)

        if not position:
            logger.info("No position to close")
            return None

        # Determine close side
        close_side = "SELL" if position.side == "LONG" else "BUY"

        # Cancel all open orders first
        await self.cancel_all_orders(symbol)

        # Place market order to close
        return await self.place_market_order(
            close_side,
            position.quantity,
            symbol,
            reduce_only=True,
        )

    async def close(self) -> None:
        """Close the executor and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("BinanceExecutor session closed")
