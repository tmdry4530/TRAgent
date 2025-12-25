"""Tests for news collectors."""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.collectors.news import (
    CryptoPanicCollector,
    NewsCollector,
    NewsContext,
    NewsItem,
    WhaleAlert,
    WhaleAlertCollector,
)


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_create_news_item(self) -> None:
        """Test creating NewsItem."""
        item = NewsItem(
            id="123",
            title="Bitcoin hits new ATH",
            source="CoinDesk",
            url="https://coindesk.com/article",
            published_at=datetime.now(timezone.utc),
            currencies=["BTC"],
            kind="news",
            sentiment="positive",
            importance=8,
        )

        assert item.id == "123"
        assert item.title == "Bitcoin hits new ATH"
        assert item.sentiment == "positive"
        assert item.importance == 8


class TestWhaleAlert:
    """Tests for WhaleAlert dataclass."""

    def test_create_whale_alert(self) -> None:
        """Test creating WhaleAlert."""
        alert = WhaleAlert(
            id="tx_123",
            blockchain="bitcoin",
            symbol="BTC",
            transaction_type="transfer",
            amount=1000.0,
            amount_usd=50_000_000.0,
            from_owner="unknown",
            to_owner="binance",
            from_address="bc1q...",
            to_address="bc1q...",
            timestamp=datetime.now(timezone.utc),
            tx_hash="abc123...",
        )

        assert alert.symbol == "BTC"
        assert alert.amount_usd == 50_000_000.0
        assert alert.to_owner == "binance"


class TestCryptoPanicCollector:
    """Tests for CryptoPanicCollector."""

    @pytest.fixture
    def collector(self) -> CryptoPanicCollector:
        """Create collector instance."""
        return CryptoPanicCollector()

    @pytest.mark.asyncio
    async def test_fetch_success(self, collector: CryptoPanicCollector) -> None:
        """Test successful CryptoPanic fetch."""
        mock_response = {
            "results": [
                {
                    "id": 12345,
                    "title": "Bitcoin price analysis",
                    "source": {"title": "CoinDesk"},
                    "url": "https://example.com",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "currencies": [{"code": "BTC"}],
                    "kind": "news",
                    "votes": {"positive": 10, "negative": 2},
                }
            ]
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
            result = await collector.fetch()

            assert len(result) == 1
            assert result[0].title == "Bitcoin price analysis"
            assert result[0].sentiment == "positive"

    @pytest.mark.asyncio
    async def test_fetch_rate_limit(self, collector: CryptoPanicCollector) -> None:
        """Test handling of rate limit."""
        mock_resp = MagicMock()
        mock_resp.status = 429

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.get.return_value = mock_cm

        with patch.object(
            collector, "_ensure_session", new_callable=AsyncMock, return_value=session
        ):
            with pytest.raises(ValueError, match="Rate limit"):
                await collector.fetch()

    def test_calculate_importance(self, collector: CryptoPanicCollector) -> None:
        """Test importance calculation."""
        # Low importance item
        low_item = {
            "votes": {"positive": 5, "negative": 5},
            "filter": None,
            "source": {"title": "Unknown Blog"},
        }
        assert collector._calculate_importance(low_item) == 5

        # High importance item
        high_item = {
            "votes": {"positive": 150, "negative": 10},
            "filter": "hot",
            "source": {"title": "CoinDesk"},
        }
        assert collector._calculate_importance(high_item) >= 8

    def test_get_sentiment_summary(self, collector: CryptoPanicCollector) -> None:
        """Test sentiment summary."""
        # Add some mock news
        collector._recent_news = [
            NewsItem(
                id="1",
                title="Bull news",
                source="Test",
                url="",
                published_at=datetime.now(timezone.utc),
                currencies=["BTC"],
                kind="news",
                sentiment="positive",
                importance=5,
            ),
            NewsItem(
                id="2",
                title="Bear news",
                source="Test",
                url="",
                published_at=datetime.now(timezone.utc),
                currencies=["BTC"],
                kind="news",
                sentiment="negative",
                importance=5,
            ),
            NewsItem(
                id="3",
                title="Neutral news",
                source="Test",
                url="",
                published_at=datetime.now(timezone.utc),
                currencies=["BTC"],
                kind="news",
                sentiment="neutral",
                importance=5,
            ),
        ]

        bullish, bearish, neutral = collector.get_sentiment_summary()
        assert bullish == 1
        assert bearish == 1
        assert neutral == 1

    def test_on_update_callback(self, collector: CryptoPanicCollector) -> None:
        """Test callback registration."""
        callback = MagicMock()
        collector.on_update(callback)

        assert callback in collector._callbacks


class TestWhaleAlertCollector:
    """Tests for WhaleAlertCollector."""

    @pytest.fixture
    def collector(self) -> WhaleAlertCollector:
        """Create collector instance without API key."""
        return WhaleAlertCollector()

    @pytest.fixture
    def collector_with_key(self) -> WhaleAlertCollector:
        """Create collector instance with API key."""
        return WhaleAlertCollector(api_key="test_key")

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self, collector: WhaleAlertCollector) -> None:
        """Test fetch without API key returns empty list."""
        result = await collector.fetch()
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_success(
        self, collector_with_key: WhaleAlertCollector
    ) -> None:
        """Test successful Whale Alert fetch."""
        mock_response = {
            "result": "success",
            "transactions": [
                {
                    "id": "tx_123",
                    "blockchain": "bitcoin",
                    "symbol": "BTC",
                    "transaction_type": "transfer",
                    "amount": 1000,
                    "amount_usd": 50000000,
                    "from": {
                        "owner_type": "unknown",
                        "address": "bc1q...",
                    },
                    "to": {
                        "owner_type": "binance",
                        "address": "bc1q...",
                    },
                    "timestamp": int(datetime.now(timezone.utc).timestamp()),
                    "hash": "abc123",
                }
            ],
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
            collector_with_key, "_ensure_session", new_callable=AsyncMock, return_value=session
        ):
            result = await collector_with_key.fetch()

            assert len(result) == 1
            assert result[0].symbol == "BTC"
            assert result[0].amount_usd == 50000000

    def test_get_exchange_flows(self, collector: WhaleAlertCollector) -> None:
        """Test exchange flow calculation."""
        # Add mock alerts
        collector._recent_alerts = [
            WhaleAlert(
                id="1",
                blockchain="bitcoin",
                symbol="BTC",
                transaction_type="transfer",
                amount=100,
                amount_usd=5_000_000,
                from_owner="unknown",
                to_owner="binance",  # Inflow
                from_address="",
                to_address="",
                timestamp=datetime.now(timezone.utc),
                tx_hash="",
            ),
            WhaleAlert(
                id="2",
                blockchain="bitcoin",
                symbol="BTC",
                transaction_type="transfer",
                amount=50,
                amount_usd=2_500_000,
                from_owner="coinbase",  # Outflow
                to_owner="unknown",
                from_address="",
                to_address="",
                timestamp=datetime.now(timezone.utc),
                tx_hash="",
            ),
        ]

        inflows, outflows = collector.get_exchange_flows()
        assert inflows == 5_000_000
        assert outflows == 2_500_000


class TestNewsCollector:
    """Tests for unified NewsCollector."""

    @pytest.fixture
    def collector(self) -> NewsCollector:
        """Create collector instance."""
        return NewsCollector()

    def test_init_creates_sub_collectors(self, collector: NewsCollector) -> None:
        """Test that sub-collectors are created."""
        assert collector.cryptopanic is not None
        assert collector.whale_alert is not None

    def test_on_update_callback(self, collector: NewsCollector) -> None:
        """Test callback registration."""
        callback = MagicMock()
        collector.on_update(callback)

        assert callback in collector._callbacks

    def test_get_context_none(self, collector: NewsCollector) -> None:
        """Test get_context when no data."""
        assert collector.get_context() is None

    def test_update_context(self, collector: NewsCollector) -> None:
        """Test context update."""
        # Add some mock data to sub-collectors
        collector.cryptopanic._recent_news = [
            NewsItem(
                id="1",
                title="Test news",
                source="Test",
                url="",
                published_at=datetime.now(timezone.utc),
                currencies=["BTC"],
                kind="news",
                sentiment="positive",
                importance=5,
            ),
        ]

        collector.whale_alert._recent_alerts = [
            WhaleAlert(
                id="1",
                blockchain="bitcoin",
                symbol="BTC",
                transaction_type="transfer",
                amount=100,
                amount_usd=5_000_000,
                from_owner="unknown",
                to_owner="binance",
                from_address="",
                to_address="",
                timestamp=datetime.now(timezone.utc),
                tx_hash="",
            ),
        ]

        # Trigger update
        collector._update_context()

        context = collector.get_context()
        assert context is not None
        assert len(context.recent_news) == 1
        assert len(context.whale_alerts) == 1
        assert context.bullish_count == 1
        assert context.exchange_inflows == 5_000_000


class TestNewsContext:
    """Tests for NewsContext dataclass."""

    def test_create_news_context(self) -> None:
        """Test creating NewsContext."""
        context = NewsContext(
            timestamp=datetime.now(timezone.utc),
            recent_news=[],
            whale_alerts=[],
            bullish_count=5,
            bearish_count=2,
            exchange_inflows=10_000_000,
            exchange_outflows=5_000_000,
        )

        assert context.bullish_count == 5
        assert context.bearish_count == 2
        assert context.exchange_inflows == 10_000_000
