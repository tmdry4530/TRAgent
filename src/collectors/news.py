"""News and whale alert collectors for crypto market intelligence."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Callable, Literal, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsItem:
    """News item from CryptoPanic or other sources.

    Attributes:
        id: Unique identifier
        title: News headline
        source: Source name (e.g., "cointelegraph", "coindesk")
        url: Link to full article
        published_at: Publication timestamp
        currencies: List of related crypto symbols
        kind: Type of news (news, media)
        sentiment: Detected sentiment (positive, negative, neutral)
        importance: Importance score (1-10)
    """

    id: str
    title: str
    source: str
    url: str
    published_at: datetime
    currencies: list[str]
    kind: Literal["news", "media"]
    sentiment: Literal["positive", "negative", "neutral"]
    importance: int  # 1-10


@dataclass
class WhaleAlert:
    """Whale transaction alert.

    Attributes:
        id: Transaction ID
        blockchain: Blockchain name (bitcoin, ethereum, etc.)
        symbol: Asset symbol (BTC, ETH, etc.)
        transaction_type: Type (transfer, mint, burn)
        amount: Amount in native units
        amount_usd: USD value
        from_owner: Source (exchange name or "unknown")
        to_owner: Destination (exchange name or "unknown")
        from_address: Source address (truncated)
        to_address: Destination address (truncated)
        timestamp: Transaction timestamp
        tx_hash: Transaction hash
    """

    id: str
    blockchain: str
    symbol: str
    transaction_type: str
    amount: float
    amount_usd: float
    from_owner: str
    to_owner: str
    from_address: str
    to_address: str
    timestamp: datetime
    tx_hash: str


@dataclass
class NewsContext:
    """Aggregated news context for trading decisions.

    Attributes:
        timestamp: Context generation timestamp
        recent_news: List of recent news items (last 1 hour)
        whale_alerts: List of recent whale alerts (last 30 minutes)
        bullish_count: Number of bullish news items
        bearish_count: Number of bearish news items
        exchange_inflows: Large deposits to exchanges (bearish signal)
        exchange_outflows: Large withdrawals from exchanges (bullish signal)
    """

    timestamp: datetime
    recent_news: list[NewsItem]
    whale_alerts: list[WhaleAlert]
    bullish_count: int
    bearish_count: int
    exchange_inflows: float  # USD value
    exchange_outflows: float  # USD value


class CryptoPanicCollector:
    """Collects crypto news from CryptoPanic API.

    Free tier: 5 requests per minute
    Polling interval: 1 minute
    """

    API_URL = "https://cryptopanic.com/api/v1/posts/"
    POLL_INTERVAL = 60  # 1 minute
    RATE_LIMIT = 5  # requests per minute

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize CryptoPanic collector.

        Args:
            api_key: CryptoPanic API key (optional for public access)
        """
        if api_key:
            self.api_key = api_key
        else:
            settings = get_settings()
            secret = getattr(settings, "cryptopanic_api_key", None)
            if secret and hasattr(secret, "get_secret_value"):
                value = secret.get_secret_value()
                self.api_key = value if value else None
            else:
                self.api_key = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._callbacks: list[Callable[[list[NewsItem]], None]] = []
        self._recent_news: list[NewsItem] = []
        self._seen_ids: set[str] = set()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def fetch(
        self,
        currencies: str = "BTC",
        filter_kind: str = "all",
        public: bool = True,
    ) -> list[NewsItem]:
        """Fetch recent news from CryptoPanic.

        Args:
            currencies: Comma-separated currency codes (e.g., "BTC,ETH")
            filter_kind: Filter type (all, rising, hot, bullish, bearish, important)
            public: Whether to use public API (no auth required)

        Returns:
            List of NewsItem objects

        Raises:
            ValueError: On API errors
        """
        session = await self._ensure_session()

        params = {
            "currencies": currencies,
            "filter": filter_kind,
            "public": "true" if public else "false",
        }

        if self.api_key:
            params["auth_token"] = self.api_key

        try:
            async with session.get(self.API_URL, params=params) as response:
                if response.status == 429:
                    logger.warning("CryptoPanic rate limit hit")
                    raise ValueError("Rate limit exceeded")

                if response.status != 200:
                    raise ValueError(f"API error: {response.status}")

                data = await response.json()

                if "results" not in data:
                    return []

                news_items = []
                for item in data["results"]:
                    try:
                        # Skip if already seen
                        item_id = str(item["id"])
                        if item_id in self._seen_ids:
                            continue

                        # Parse sentiment from votes
                        votes = item.get("votes", {})
                        positive = votes.get("positive", 0)
                        negative = votes.get("negative", 0)

                        if positive > negative * 2:
                            sentiment = "positive"
                        elif negative > positive * 2:
                            sentiment = "negative"
                        else:
                            sentiment = "neutral"

                        # Calculate importance based on various factors
                        importance = self._calculate_importance(item)

                        # Extract currencies
                        currencies_list = [
                            c["code"] for c in item.get("currencies", [])
                        ]

                        news_item = NewsItem(
                            id=item_id,
                            title=item["title"],
                            source=item.get("source", {}).get("title", "Unknown"),
                            url=item.get("url", ""),
                            published_at=datetime.fromisoformat(
                                item["published_at"].replace("Z", "+00:00")
                            ),
                            currencies=currencies_list,
                            kind=item.get("kind", "news"),
                            sentiment=sentiment,
                            importance=importance,
                        )

                        news_items.append(news_item)
                        self._seen_ids.add(item_id)

                    except Exception as e:
                        logger.error(
                            "Error parsing news item",
                            error=str(e),
                            item_id=item.get("id"),
                        )

                # Keep only recent news (last 1 hour)
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                self._recent_news = [
                    n for n in self._recent_news + news_items if n.published_at > cutoff
                ]

                logger.info(
                    "Fetched CryptoPanic news",
                    new_items=len(news_items),
                    total_recent=len(self._recent_news),
                )

                return news_items

        except aiohttp.ClientError as e:
            logger.error("Network error fetching CryptoPanic", error=str(e))
            raise

    def _calculate_importance(self, item: dict) -> int:
        """Calculate importance score (1-10) based on item attributes."""
        score = 5  # Base score

        votes = item.get("votes", {})
        total_votes = sum(votes.values()) if votes else 0

        # Votes boost
        if total_votes > 100:
            score += 2
        elif total_votes > 50:
            score += 1

        # Hot/Important filter
        if item.get("filter") in ["hot", "important"]:
            score += 2

        # Source reputation (simplified)
        source = item.get("source", {}).get("title", "").lower()
        reputable_sources = ["cointelegraph", "coindesk", "decrypt", "theblock"]
        if any(s in source for s in reputable_sources):
            score += 1

        return min(10, max(1, score))

    def on_update(self, callback: Callable[[list[NewsItem]], None]) -> None:
        """Register callback for news updates."""
        self._callbacks.append(callback)

    async def start_polling(self) -> None:
        """Start polling loop."""
        self.is_running = True
        logger.info("Starting CryptoPanic polling", interval=self.POLL_INTERVAL)

        while self.is_running:
            try:
                news = await self.fetch()

                if news:
                    for callback in self._callbacks:
                        try:
                            callback(news)
                        except Exception as e:
                            logger.error("Error in news callback", error=str(e))

            except Exception as e:
                logger.error("Error in CryptoPanic polling", error=str(e))

            await asyncio.sleep(self.POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self.is_running = False
        if self.session and not self.session.closed:
            await self.session.close()

    def get_recent_news(self, hours: float = 1.0) -> list[NewsItem]:
        """Get recent news within the specified time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [n for n in self._recent_news if n.published_at > cutoff]

    def get_sentiment_summary(self) -> tuple[int, int, int]:
        """Get count of (bullish, bearish, neutral) news items."""
        recent = self.get_recent_news()
        bullish = sum(1 for n in recent if n.sentiment == "positive")
        bearish = sum(1 for n in recent if n.sentiment == "negative")
        neutral = sum(1 for n in recent if n.sentiment == "neutral")
        return bullish, bearish, neutral


class WhaleAlertCollector:
    """Collects large transaction alerts from Whale Alert API.

    Note: Whale Alert requires a paid subscription (no free tier available).
    If no API key is configured, this collector returns empty results.
    Polling interval: 1 minute
    """

    API_URL = "https://api.whale-alert.io/v1/transactions"
    POLL_INTERVAL = 60  # 1 minute
    MIN_USD_VALUE = 1_000_000  # $1M minimum for alerts

    # Known exchange addresses (simplified mapping)
    EXCHANGES = {
        "binance",
        "coinbase",
        "kraken",
        "bitfinex",
        "huobi",
        "okex",
        "ftx",
        "kucoin",
        "bybit",
        "bittrex",
        "gemini",
        "bitstamp",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Whale Alert collector.

        Args:
            api_key: Whale Alert API key
        """
        if api_key:
            self.api_key = api_key
        else:
            settings = get_settings()
            secret = getattr(settings, "whale_alert_api_key", None)
            if secret and hasattr(secret, "get_secret_value"):
                value = secret.get_secret_value()
                self.api_key = value if value else None
            else:
                self.api_key = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self._callbacks: list[Callable[[list[WhaleAlert]], None]] = []
        self._recent_alerts: list[WhaleAlert] = []
        self._seen_ids: set[str] = set()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def fetch(self, min_value: int = 1_000_000) -> list[WhaleAlert]:
        """Fetch recent whale transactions.

        Args:
            min_value: Minimum USD value for transactions

        Returns:
            List of WhaleAlert objects

        Raises:
            ValueError: On API errors
        """
        if not self.api_key:
            logger.debug("No Whale Alert API key configured, skipping fetch")
            return []

        session = await self._ensure_session()

        # Fetch transactions from last 10 minutes
        start_time = int((datetime.now(timezone.utc) - timedelta(minutes=10)).timestamp())

        params = {
            "api_key": self.api_key,
            "min_value": min_value,
            "start": start_time,
            "cursor": "",
        }

        try:
            async with session.get(self.API_URL, params=params) as response:
                if response.status == 401:
                    logger.error("Invalid Whale Alert API key")
                    return []

                if response.status == 429:
                    logger.warning("Whale Alert rate limit hit")
                    raise ValueError("Rate limit exceeded")

                if response.status != 200:
                    raise ValueError(f"API error: {response.status}")

                data = await response.json()

                if data.get("result") != "success":
                    logger.warning("Whale Alert API returned error", data=data)
                    return []

                transactions = data.get("transactions", [])
                alerts = []

                for tx in transactions:
                    try:
                        tx_id = tx.get("id", tx.get("hash", ""))
                        if tx_id in self._seen_ids:
                            continue

                        # Parse from/to owner types
                        from_owner = tx.get("from", {}).get("owner_type", "unknown")
                        to_owner = tx.get("to", {}).get("owner_type", "unknown")

                        alert = WhaleAlert(
                            id=tx_id,
                            blockchain=tx.get("blockchain", "unknown"),
                            symbol=tx.get("symbol", "UNKNOWN").upper(),
                            transaction_type=tx.get("transaction_type", "transfer"),
                            amount=float(tx.get("amount", 0)),
                            amount_usd=float(tx.get("amount_usd", 0)),
                            from_owner=from_owner,
                            to_owner=to_owner,
                            from_address=tx.get("from", {}).get("address", "")[:16],
                            to_address=tx.get("to", {}).get("address", "")[:16],
                            timestamp=datetime.fromtimestamp(
                                tx.get("timestamp", 0), tz=timezone.utc
                            ),
                            tx_hash=tx.get("hash", ""),
                        )

                        alerts.append(alert)
                        self._seen_ids.add(tx_id)

                    except Exception as e:
                        logger.error(
                            "Error parsing whale alert",
                            error=str(e),
                            tx_id=tx.get("id"),
                        )

                # Keep only recent alerts (last 30 minutes)
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
                self._recent_alerts = [
                    a for a in self._recent_alerts + alerts if a.timestamp > cutoff
                ]

                logger.info(
                    "Fetched whale alerts",
                    new_alerts=len(alerts),
                    total_recent=len(self._recent_alerts),
                )

                return alerts

        except aiohttp.ClientError as e:
            logger.error("Network error fetching Whale Alert", error=str(e))
            raise

    def on_update(self, callback: Callable[[list[WhaleAlert]], None]) -> None:
        """Register callback for whale alert updates."""
        self._callbacks.append(callback)

    async def start_polling(self) -> None:
        """Start polling loop."""
        self.is_running = True
        logger.info("Starting Whale Alert polling", interval=self.POLL_INTERVAL)

        while self.is_running:
            try:
                alerts = await self.fetch()

                if alerts:
                    for callback in self._callbacks:
                        try:
                            callback(alerts)
                        except Exception as e:
                            logger.error("Error in whale alert callback", error=str(e))

            except Exception as e:
                logger.error("Error in Whale Alert polling", error=str(e))

            await asyncio.sleep(self.POLL_INTERVAL)

    async def stop(self) -> None:
        """Stop polling and close session."""
        self.is_running = False
        if self.session and not self.session.closed:
            await self.session.close()

    def get_recent_alerts(self, minutes: float = 30.0) -> list[WhaleAlert]:
        """Get recent alerts within the specified time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [a for a in self._recent_alerts if a.timestamp > cutoff]

    def get_exchange_flows(self) -> tuple[float, float]:
        """Calculate exchange inflows and outflows from recent alerts.

        Returns:
            Tuple of (inflows_usd, outflows_usd)
        """
        recent = self.get_recent_alerts()
        inflows = 0.0
        outflows = 0.0

        for alert in recent:
            if alert.to_owner.lower() in self.EXCHANGES:
                inflows += alert.amount_usd
            if alert.from_owner.lower() in self.EXCHANGES:
                outflows += alert.amount_usd

        return inflows, outflows


class NewsCollector:
    """Unified news collector combining all news sources.

    Manages:
    - CryptoPanic (1 minute polling)
    - Whale Alert (1 minute polling)
    """

    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        whale_alert_api_key: Optional[str] = None,
    ) -> None:
        self.cryptopanic = CryptoPanicCollector(api_key=cryptopanic_api_key)
        self.whale_alert = WhaleAlertCollector(api_key=whale_alert_api_key)

        self._callbacks: list[Callable[[NewsContext], None]] = []
        self._last_context: Optional[NewsContext] = None

        # Register internal callbacks
        self.cryptopanic.on_update(self._on_news_update)
        self.whale_alert.on_update(self._on_whale_update)

    def _on_news_update(self, news: list[NewsItem]) -> None:
        """Handle news update."""
        self._update_context()

    def _on_whale_update(self, alerts: list[WhaleAlert]) -> None:
        """Handle whale alert update."""
        self._update_context()

    def _update_context(self) -> None:
        """Build and emit updated NewsContext."""
        try:
            recent_news = self.cryptopanic.get_recent_news()
            whale_alerts = self.whale_alert.get_recent_alerts()
            bullish, bearish, _ = self.cryptopanic.get_sentiment_summary()
            inflows, outflows = self.whale_alert.get_exchange_flows()

            context = NewsContext(
                timestamp=datetime.now(timezone.utc),
                recent_news=recent_news,
                whale_alerts=whale_alerts,
                bullish_count=bullish,
                bearish_count=bearish,
                exchange_inflows=inflows,
                exchange_outflows=outflows,
            )

            self._last_context = context

            for callback in self._callbacks:
                try:
                    callback(context)
                except Exception as e:
                    logger.error("Error in news context callback", error=str(e))

        except Exception as e:
            logger.error("Error building news context", error=str(e))

    def on_update(self, callback: Callable[[NewsContext], None]) -> None:
        """Register callback for context updates."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start all collectors."""
        logger.info("Starting news collectors")

        # Fetch initial data
        await self._initial_fetch()

        # Start polling loops
        await asyncio.gather(
            self.cryptopanic.start_polling(),
            self.whale_alert.start_polling(),
        )

    async def _initial_fetch(self) -> None:
        """Perform initial data fetch for all collectors."""
        try:
            await asyncio.gather(
                self.cryptopanic.fetch(),
                self.whale_alert.fetch(),
                return_exceptions=True,
            )

            self._update_context()
            logger.info("Initial news data fetch complete")

        except Exception as e:
            logger.error("Error in initial news fetch", error=str(e))

    async def stop(self) -> None:
        """Stop all collectors."""
        await asyncio.gather(
            self.cryptopanic.stop(),
            self.whale_alert.stop(),
        )
        logger.info("News collectors stopped")

    def get_context(self) -> Optional[NewsContext]:
        """Get current news context."""
        return self._last_context

    async def fetch_once(self) -> NewsContext:
        """Fetch all data once without starting polling.

        Returns:
            Current NewsContext
        """
        await self._initial_fetch()
        return self._last_context
