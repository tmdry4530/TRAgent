"""Binance WebSocket and REST API data collectors for real-time market data."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import aiohttp
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KlineData:
    """Candlestick data from Binance kline stream."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
    is_closed: bool


@dataclass
class OrderbookData:
    """Orderbook snapshot data."""

    timestamp: datetime
    bids: list[tuple[float, float]]  # [(price, quantity), ...]
    asks: list[tuple[float, float]]  # [(price, quantity), ...]


@dataclass
class TradeData:
    """Aggregated trade data."""

    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool  # True if seller was maker
    trade_id: int


@dataclass
class LiquidationData:
    """Liquidation order data."""

    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    price: float
    quantity: float
    usd_value: float


@dataclass
class FundingRateData:
    """Funding rate data from Binance Futures API."""

    symbol: str
    funding_rate: float
    funding_time: datetime
    mark_price: float


@dataclass
class OpenInterestData:
    """Open interest data from Binance Futures API."""

    symbol: str
    open_interest: float
    open_interest_value: float
    timestamp: datetime


@dataclass
class LongShortRatioData:
    """Long/Short account ratio data from Binance Futures API."""

    symbol: str
    long_account: float
    short_account: float
    long_short_ratio: float
    timestamp: datetime


class BinanceWebSocketCollector:
    """Collects real-time market data from Binance Futures WebSocket.

    Features:
    - Multiple stream subscriptions (klines, orderbook, trades, liquidations)
    - Automatic reconnection with exponential backoff
    - Callback-based data delivery
    - Handles connection failures gracefully
    """

    def __init__(self) -> None:
        """Initialize the WebSocket collector."""
        self.settings = get_settings()
        self.symbol = self.settings.trading_symbol.lower()
        self.ws = None
        self.is_connected = False
        self.is_running = False

        # Reconnection parameters
        self.reconnect_delay = 1
        self.max_reconnect_delay = 60
        self.reconnect_attempts = 0

        # Subscribed streams
        self.streams: list[str] = []

        # Callbacks
        self._kline_callbacks: list[Callable[[KlineData], None]] = []
        self._orderbook_callbacks: list[Callable[[OrderbookData], None]] = []
        self._trade_callbacks: list[Callable[[TradeData], None]] = []
        self._liquidation_callbacks: list[Callable[[LiquidationData], None]] = []

        # Data storage (for polling access)
        self._latest_klines: dict[str, KlineData] = {}  # keyed by interval
        self._latest_orderbook: Optional[OrderbookData] = None
        self._latest_trade: Optional[TradeData] = None
        self._recent_trades: list[TradeData] = []  # Last 100 trades
        self._recent_liquidations: list[LiquidationData] = []  # Last 50 liquidations
        self._max_trades = 100
        self._max_liquidations = 50

        # Tasks
        self._receive_task = None

    async def connect(self) -> None:
        """Establish WebSocket connection with auto-reconnect."""
        self.is_running = True

        while self.is_running:
            try:
                if not self.streams:
                    logger.warning("No streams subscribed. Waiting for subscriptions...")
                    await asyncio.sleep(1)
                    continue

                # Build WebSocket URL
                base_url = self.settings.binance_ws_url
                stream_path = "/".join(self.streams)
                ws_url = f"{base_url}/stream?streams={stream_path}"

                logger.info("Connecting to Binance WebSocket", url=ws_url)

                self.ws = await websockets.connect(ws_url)
                self.is_connected = True
                self.reconnect_delay = 1
                self.reconnect_attempts = 0

                logger.info("WebSocket connected successfully")

                # Start receiving messages
                await self._handle_messages()

            except ConnectionClosed as e:
                self.is_connected = False
                logger.warning(
                    "WebSocket connection closed",
                    code=e.code,
                    reason=e.reason,
                    attempt=self.reconnect_attempts,
                )
                await self._handle_reconnect()

            except WebSocketException as e:
                self.is_connected = False
                logger.error("WebSocket error", error=str(e), attempt=self.reconnect_attempts)
                await self._handle_reconnect()

            except Exception as e:
                self.is_connected = False
                logger.error(
                    "Unexpected error in WebSocket connection",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._handle_reconnect()

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.is_running = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.ws and not self.ws.closed:
            await self.ws.close()
            logger.info("WebSocket disconnected")

        self.is_connected = False

    async def subscribe_klines(self, intervals: list[str]) -> None:
        """Subscribe to kline (candlestick) streams.

        Args:
            intervals: List of kline intervals (e.g., ["1m", "15m"])
        """
        for interval in intervals:
            stream = f"{self.symbol}@kline_{interval}"
            if stream not in self.streams:
                self.streams.append(stream)
                logger.info("Subscribed to kline stream", interval=interval)

    async def subscribe_orderbook(self) -> None:
        """Subscribe to orderbook depth stream (20 levels, 100ms updates)."""
        stream = f"{self.symbol}@depth20@100ms"
        if stream not in self.streams:
            self.streams.append(stream)
            logger.info("Subscribed to orderbook stream")

    async def subscribe_trades(self) -> None:
        """Subscribe to aggregated trade stream."""
        stream = f"{self.symbol}@aggTrade"
        if stream not in self.streams:
            self.streams.append(stream)
            logger.info("Subscribed to trade stream")

    async def subscribe_liquidations(self) -> None:
        """Subscribe to liquidation order stream."""
        stream = f"{self.symbol}@forceOrder"
        if stream not in self.streams:
            self.streams.append(stream)
            logger.info("Subscribed to liquidation stream")

    def on_kline(self, callback: Callable[[KlineData], None]) -> None:
        """Register a callback for kline data.

        Args:
            callback: Function to call when kline data is received
        """
        self._kline_callbacks.append(callback)

    def on_orderbook(self, callback: Callable[[OrderbookData], None]) -> None:
        """Register a callback for orderbook data.

        Args:
            callback: Function to call when orderbook data is received
        """
        self._orderbook_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeData], None]) -> None:
        """Register a callback for trade data.

        Args:
            callback: Function to call when trade data is received
        """
        self._trade_callbacks.append(callback)

    def on_liquidation(self, callback: Callable[[LiquidationData], None]) -> None:
        """Register a callback for liquidation data.

        Args:
            callback: Function to call when liquidation data is received
        """
        self._liquidation_callbacks.append(callback)

    async def get_latest_data(self, data_type: str) -> Optional[Any]:
        """Get the latest cached data by type.

        Args:
            data_type: Type of data to retrieve:
                - "ticker" or "trade": Latest trade data with price
                - "orderbook": Latest orderbook snapshot
                - "trades": List of recent trades
                - "kline_1m", "kline_15m", etc.: Latest kline for interval
                - "liquidations": List of recent liquidations

        Returns:
            Cached data or None if not available
        """
        if data_type in ("ticker", "trade"):
            if self._latest_trade:
                return {
                    "price": self._latest_trade.price,
                    "quantity": self._latest_trade.quantity,
                    "timestamp": self._latest_trade.timestamp,
                }
            return None

        elif data_type == "orderbook":
            if self._latest_orderbook:
                return {
                    "bids": self._latest_orderbook.bids,
                    "asks": self._latest_orderbook.asks,
                    "timestamp": self._latest_orderbook.timestamp,
                }
            return None

        elif data_type == "trades":
            return self._recent_trades.copy() if self._recent_trades else None

        elif data_type.startswith("kline_"):
            interval = data_type.replace("kline_", "")
            kline = self._latest_klines.get(interval)
            if kline:
                return {
                    "open": kline.open,
                    "high": kline.high,
                    "low": kline.low,
                    "close": kline.close,
                    "volume": kline.volume,
                    "interval": kline.interval,
                    "is_closed": kline.is_closed,
                    "timestamp": kline.timestamp,
                }
            return None

        elif data_type == "liquidations":
            return self._recent_liquidations.copy() if self._recent_liquidations else None

        else:
            logger.warning("Unknown data type requested", data_type=data_type)
            return None

    async def _handle_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        while self.is_running and self.is_connected:
            try:
                message = await self.ws.recv()
                await self._process_message(message)
            except ConnectionClosed:
                logger.warning("Connection closed while receiving messages")
                break
            except Exception as e:
                logger.error(
                    "Error processing message",
                    error=str(e),
                    error_type=type(e).__name__,
                )

    async def _process_message(self, message: str) -> None:
        """Process a WebSocket message and route to appropriate handler.

        Args:
            message: Raw WebSocket message
        """
        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "...", "data": {...}}
            if "stream" in data:
                stream = data["stream"]
                payload = data["data"]

                if "@kline_" in stream:
                    await self._handle_kline(payload)
                elif "@depth" in stream:
                    await self._handle_orderbook(payload)
                elif "@aggTrade" in stream:
                    await self._handle_trade(payload)
                elif "@forceOrder" in stream:
                    await self._handle_liquidation(payload)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse WebSocket message", error=str(e))
        except Exception as e:
            logger.error(
                "Error processing message",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _handle_kline(self, data: dict[str, Any]) -> None:
        """Handle kline message.

        Args:
            data: Kline data payload
        """
        try:
            k = data["k"]
            kline = KlineData(
                timestamp=datetime.fromtimestamp(k["t"] / 1000),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                interval=k["i"],
                is_closed=k["x"],
            )

            # Store latest kline by interval
            self._latest_klines[kline.interval] = kline

            for callback in self._kline_callbacks:
                try:
                    callback(kline)
                except Exception as e:
                    logger.error(
                        "Error in kline callback",
                        error=str(e),
                        callback=callback.__name__,
                    )

        except Exception as e:
            logger.error("Error handling kline data", error=str(e))

    async def _handle_orderbook(self, data: dict[str, Any]) -> None:
        """Handle orderbook message.

        Args:
            data: Orderbook data payload
        """
        try:
            orderbook = OrderbookData(
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                bids=[(float(p), float(q)) for p, q in data["b"]],
                asks=[(float(p), float(q)) for p, q in data["a"]],
            )

            # Store latest orderbook
            self._latest_orderbook = orderbook

            for callback in self._orderbook_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    logger.error(
                        "Error in orderbook callback",
                        error=str(e),
                        callback=callback.__name__,
                    )

        except Exception as e:
            logger.error("Error handling orderbook data", error=str(e))

    async def _handle_trade(self, data: dict[str, Any]) -> None:
        """Handle trade message.

        Args:
            data: Trade data payload
        """
        try:
            trade = TradeData(
                timestamp=datetime.fromtimestamp(data["T"] / 1000),
                price=float(data["p"]),
                quantity=float(data["q"]),
                is_buyer_maker=data["m"],
                trade_id=int(data["a"]),
            )

            # Store latest trade and maintain recent trades list
            self._latest_trade = trade
            self._recent_trades.append(trade)
            if len(self._recent_trades) > self._max_trades:
                self._recent_trades = self._recent_trades[-self._max_trades:]

            for callback in self._trade_callbacks:
                try:
                    callback(trade)
                except Exception as e:
                    logger.error(
                        "Error in trade callback",
                        error=str(e),
                        callback=callback.__name__,
                    )

        except Exception as e:
            logger.error("Error handling trade data", error=str(e))

    async def _handle_liquidation(self, data: dict[str, Any]) -> None:
        """Handle liquidation message.

        Args:
            data: Liquidation data payload
        """
        try:
            o = data["o"]
            liquidation = LiquidationData(
                timestamp=datetime.fromtimestamp(o["T"] / 1000),
                symbol=o["s"],
                side=o["S"],
                price=float(o["p"]),
                quantity=float(o["q"]),
                usd_value=float(o["p"]) * float(o["q"]),
            )

            # Store recent liquidations
            self._recent_liquidations.append(liquidation)
            if len(self._recent_liquidations) > self._max_liquidations:
                self._recent_liquidations = self._recent_liquidations[-self._max_liquidations:]

            for callback in self._liquidation_callbacks:
                try:
                    callback(liquidation)
                except Exception as e:
                    logger.error(
                        "Error in liquidation callback",
                        error=str(e),
                        callback=callback.__name__,
                    )

        except Exception as e:
            logger.error("Error handling liquidation data", error=str(e))

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if not self.is_running:
            return

        self.reconnect_attempts += 1
        logger.info(
            "Attempting to reconnect",
            delay=self.reconnect_delay,
            attempt=self.reconnect_attempts,
        )

        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)


class BinanceRestCollector:
    """Collects market data from Binance Futures REST API.

    Features:
    - Funding rate queries
    - Open interest queries
    - Long/Short ratio queries
    - Automatic retry with exponential backoff
    - Rate limit compliance (1200 req/min)
    """

    # Rate limit: 1200 requests per minute = 50ms minimum interval
    MIN_REQUEST_INTERVAL = 0.05

    def __init__(self, base_url: Optional[str] = None) -> None:
        """Initialize the REST API collector.

        Args:
            base_url: Optional base URL override (default: based on testnet setting)
        """
        self.settings = get_settings()
        self.base_url = base_url or self.settings.binance_rest_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0.0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created.

        Returns:
            Active aiohttp session
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _request(
        self, endpoint: str, params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Make an HTTP GET request to Binance API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Optional query parameters

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On HTTP errors or invalid responses
        """
        await self._rate_limit()

        session = await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", "5"))
                    logger.warning("Rate limit exceeded", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    raise ValueError("Rate limit exceeded, retrying...")

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        "Binance API error",
                        status=response.status,
                        error=error_text,
                        endpoint=endpoint,
                    )
                    raise ValueError(f"HTTP {response.status}: {error_text}")

                data = await response.json()
                return data

        except aiohttp.ClientError as e:
            logger.error("Network error", error=str(e), endpoint=endpoint)
            raise

    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> FundingRateData:
        """Get current funding rate for a symbol.

        Args:
            symbol: Trading symbol (default: BTCUSDT)

        Returns:
            FundingRateData object with funding rate information

        Raises:
            ValueError: On API errors or invalid responses
        """
        endpoint = "/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}

        try:
            data = await self._request(endpoint, params)

            if not data or len(data) == 0:
                raise ValueError(f"No funding rate data for {symbol}")

            latest = data[0]

            # Get current mark price
            mark_price_data = await self._request("/fapi/v1/premiumIndex", {"symbol": symbol})
            mark_price = float(mark_price_data.get("markPrice", 0))

            funding_rate = FundingRateData(
                symbol=latest["symbol"],
                funding_rate=float(latest["fundingRate"]),
                funding_time=datetime.fromtimestamp(int(latest["fundingTime"]) / 1000),
                mark_price=mark_price,
            )

            logger.debug(
                "Fetched funding rate",
                symbol=symbol,
                rate=funding_rate.funding_rate,
                mark_price=mark_price,
            )

            return funding_rate

        except Exception as e:
            logger.error("Error fetching funding rate", symbol=symbol, error=str(e))
            raise

    async def get_open_interest(self, symbol: str = "BTCUSDT") -> OpenInterestData:
        """Get current open interest for a symbol.

        Args:
            symbol: Trading symbol (default: BTCUSDT)

        Returns:
            OpenInterestData object with open interest information

        Raises:
            ValueError: On API errors or invalid responses
        """
        endpoint = "/fapi/v1/openInterest"
        params = {"symbol": symbol}

        try:
            data = await self._request(endpoint, params)

            open_interest = OpenInterestData(
                symbol=data["symbol"],
                open_interest=float(data["openInterest"]),
                open_interest_value=float(data["openInterest"])
                * float(data.get("price", 0)),  # Approximate value
                timestamp=datetime.fromtimestamp(int(data["time"]) / 1000),
            )

            logger.debug(
                "Fetched open interest",
                symbol=symbol,
                oi=open_interest.open_interest,
            )

            return open_interest

        except Exception as e:
            logger.error("Error fetching open interest", symbol=symbol, error=str(e))
            raise

    async def get_long_short_ratio(
        self, symbol: str = "BTCUSDT", period: str = "5m"
    ) -> LongShortRatioData:
        """Get long/short account ratio for a symbol.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)

        Returns:
            LongShortRatioData object with ratio information

        Raises:
            ValueError: On API errors or invalid responses
        """
        endpoint = "/futures/data/globalLongShortAccountRatio"
        params = {"symbol": symbol, "period": period, "limit": 1}

        try:
            data = await self._request(endpoint, params)

            if not data or len(data) == 0:
                raise ValueError(f"No long/short ratio data for {symbol}")

            latest = data[0]

            long_account = float(latest["longAccount"])
            short_account = float(latest["shortAccount"])
            long_short_ratio = float(latest["longShortRatio"])

            ratio_data = LongShortRatioData(
                symbol=symbol,
                long_account=long_account,
                short_account=short_account,
                long_short_ratio=long_short_ratio,
                timestamp=datetime.fromtimestamp(int(latest["timestamp"]) / 1000),
            )

            logger.debug(
                "Fetched long/short ratio",
                symbol=symbol,
                ratio=long_short_ratio,
                period=period,
            )

            return ratio_data

        except Exception as e:
            logger.error(
                "Error fetching long/short ratio",
                symbol=symbol,
                period=period,
                error=str(e),
            )
            raise

    async def get_klines(
        self, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get historical kline (candlestick) data.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, etc.)
            limit: Number of klines to fetch (max 1500)

        Returns:
            List of kline dictionaries with OHLCV data
        """
        endpoint = "/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            data = await self._request(endpoint, params)

            klines = []
            for k in data:
                klines.append({
                    "timestamp": datetime.fromtimestamp(k[0] / 1000),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": datetime.fromtimestamp(k[6] / 1000),
                    "interval": interval,
                    "is_closed": True,  # Historical klines are always closed
                })

            logger.info(
                "Fetched historical klines",
                symbol=symbol,
                interval=interval,
                count=len(klines),
            )

            return klines

        except Exception as e:
            logger.error(
                "Error fetching klines",
                symbol=symbol,
                interval=interval,
                error=str(e),
            )
            raise

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("REST API session closed")
