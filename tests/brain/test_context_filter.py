"""Tests for LLM context filter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain import LLMDecision, LLMContextFilter, MarketState
from src.collectors.macro import MacroContext
from src.collectors.news import NewsContext, NewsItem, WhaleAlert
from src.signals.base import Signal


class TestLLMDecision:
    """Tests for LLMDecision dataclass."""

    def test_create_decision(self) -> None:
        """Test creating LLMDecision."""
        decision = LLMDecision(
            execute=True,
            confidence=0.85,
            adjusted_size=0.8,
            reason="Strong bullish signal",
        )

        assert decision.execute is True
        assert decision.confidence == 0.85
        assert decision.adjusted_size == 0.8
        assert decision.reason == "Strong bullish signal"


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_create_market_state(self) -> None:
        """Test creating MarketState."""
        state = MarketState(
            timestamp=datetime.now(timezone.utc),
            price=50000.0,
            price_change_24h=2.5,
            funding_rate=0.0001,
            open_interest=10_000_000_000,
            long_short_ratio=1.2,
            volume_24h=5_000_000_000,
        )

        assert state.price == 50000.0
        assert state.funding_rate == 0.0001
        assert state.long_short_ratio == 1.2


class TestLLMContextFilter:
    """Tests for LLMContextFilter."""

    @pytest.fixture
    def signal(self) -> Signal:
        """Create test signal."""
        return Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.75,
            entry_price=50000.0,
            stop_loss=49250.0,
            take_profit=51500.0,
            reason="Volume breakout detected",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def market_state(self) -> MarketState:
        """Create test market state."""
        return MarketState(
            timestamp=datetime.now(timezone.utc),
            price=50000.0,
            price_change_24h=2.5,
            funding_rate=0.0001,
            open_interest=10_000_000_000,
            long_short_ratio=1.2,
        )

    @pytest.fixture
    def macro_context(self) -> MacroContext:
        """Create test macro context."""
        return MacroContext(
            timestamp=datetime.now(timezone.utc),
            fear_greed_index=45,
            fear_greed_label="Fear",
            dxy=104.5,
            dxy_change=-0.2,
            sp500_change=0.5,
            nasdaq_change=0.8,
            us10y_yield=4.2,
        )

    @pytest.fixture
    def news_context(self) -> NewsContext:
        """Create test news context."""
        return NewsContext(
            timestamp=datetime.now(timezone.utc),
            recent_news=[
                NewsItem(
                    id="1",
                    title="Bitcoin ETF sees record inflows",
                    source="CoinDesk",
                    url="",
                    published_at=datetime.now(timezone.utc),
                    currencies=["BTC"],
                    kind="news",
                    sentiment="positive",
                    importance=8,
                ),
            ],
            whale_alerts=[
                WhaleAlert(
                    id="1",
                    blockchain="bitcoin",
                    symbol="BTC",
                    transaction_type="transfer",
                    amount=500,
                    amount_usd=25_000_000,
                    from_owner="coinbase",
                    to_owner="unknown",
                    from_address="",
                    to_address="",
                    timestamp=datetime.now(timezone.utc),
                    tx_hash="",
                ),
            ],
            bullish_count=3,
            bearish_count=1,
            exchange_inflows=10_000_000,
            exchange_outflows=30_000_000,
        )

    def test_init_without_api_key(self) -> None:
        """Test initialization without API key."""
        with patch("src.brain.context_filter.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            filter_instance = LLMContextFilter()
            assert filter_instance.client is None

    def test_init_with_api_key(self) -> None:
        """Test initialization with API key."""
        filter_instance = LLMContextFilter(api_key="test_key")
        assert filter_instance.client is not None
        assert filter_instance.model == "claude-sonnet-4-20250514"

    def test_create_cache_key(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: MacroContext,
    ) -> None:
        """Test cache key creation."""
        filter_instance = LLMContextFilter(api_key="test_key")
        key = filter_instance._create_cache_key(signal, market_state, macro_context)

        assert isinstance(key, str)
        assert "SCALP" in key
        assert "LONG" in key

    def test_build_prompt(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: MacroContext,
        news_context: NewsContext,
    ) -> None:
        """Test prompt building."""
        filter_instance = LLMContextFilter(api_key="test_key")
        prompt = filter_instance._build_prompt(
            signal, market_state, macro_context, news_context
        )

        assert "SCALP" in prompt
        assert "LONG" in prompt
        assert "$50,000" in prompt
        assert "Fear & Greed Index: 45" in prompt
        assert "Bitcoin ETF" in prompt

    def test_parse_response_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        filter_instance = LLMContextFilter(api_key="test_key")

        response = """{
            "execute": true,
            "confidence": 0.85,
            "adjusted_size": 0.9,
            "reason": "Market conditions favorable"
        }"""

        decision = filter_instance._parse_response(response)

        assert decision.execute is True
        assert decision.confidence == 0.85
        assert decision.adjusted_size == 0.9
        assert decision.reason == "Market conditions favorable"

    def test_parse_response_with_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown."""
        filter_instance = LLMContextFilter(api_key="test_key")

        response = """Based on my analysis:

```json
{
    "execute": false,
    "confidence": 0.4,
    "adjusted_size": 0.0,
    "reason": "Too risky"
}
```

This is because..."""

        decision = filter_instance._parse_response(response)

        assert decision.execute is False
        assert decision.confidence == 0.4

    def test_parse_response_invalid(self) -> None:
        """Test parsing invalid response."""
        filter_instance = LLMContextFilter(api_key="test_key")

        with pytest.raises(ValueError, match="No JSON object found"):
            filter_instance._parse_response("This is not JSON")

    def test_parse_response_clamps_values(self) -> None:
        """Test that confidence and adjusted_size are clamped."""
        filter_instance = LLMContextFilter(api_key="test_key")

        response = """{
            "execute": true,
            "confidence": 1.5,
            "adjusted_size": -0.5,
            "reason": "Test"
        }"""

        decision = filter_instance._parse_response(response)

        assert decision.confidence == 1.0  # Clamped to max
        assert decision.adjusted_size == 0.0  # Clamped to min

    @pytest.mark.asyncio
    async def test_evaluate_without_client(
        self,
        signal: Signal,
        market_state: MarketState,
    ) -> None:
        """Test evaluate when client is not initialized."""
        with patch("src.brain.context_filter.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None
            filter_instance = LLMContextFilter()

            decision = await filter_instance.evaluate(signal, market_state)

            assert decision.execute is True
            assert "disabled" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluate_success(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: MacroContext,
        news_context: NewsContext,
    ) -> None:
        """Test successful evaluation."""
        filter_instance = LLMContextFilter(api_key="test_key")

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"execute": true, "confidence": 0.8, "adjusted_size": 0.9, "reason": "Good setup"}'
            )
        ]

        with patch.object(
            filter_instance.client,
            "messages",
            MagicMock(create=MagicMock(return_value=mock_response)),
        ):
            decision = await filter_instance.evaluate(
                signal, market_state, macro_context, news_context
            )

            assert decision.execute is True
            assert decision.confidence == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_uses_cache(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: MacroContext,
    ) -> None:
        """Test that cache is used on repeated calls."""
        filter_instance = LLMContextFilter(api_key="test_key")

        # Pre-populate cache
        cache_key = filter_instance._create_cache_key(
            signal, market_state, macro_context
        )
        cached_decision = LLMDecision(
            execute=True,
            confidence=0.9,
            adjusted_size=1.0,
            reason="Cached result",
        )
        filter_instance._cache[cache_key] = cached_decision

        decision = await filter_instance.evaluate(
            signal, market_state, macro_context
        )

        assert decision.reason == "Cached result"
        assert filter_instance._cache_hits == 1

    @pytest.mark.asyncio
    async def test_evaluate_error_returns_conservative(
        self,
        signal: Signal,
        market_state: MarketState,
    ) -> None:
        """Test that errors return conservative decision."""
        filter_instance = LLMContextFilter(api_key="test_key")

        with patch.object(
            filter_instance,
            "_call_llm",
            side_effect=Exception("API Error"),
        ):
            decision = await filter_instance.evaluate(signal, market_state)

            assert decision.execute is False
            assert decision.confidence == 0.0
            assert "failed" in decision.reason.lower()

    def test_get_stats(self) -> None:
        """Test getting filter statistics."""
        filter_instance = LLMContextFilter(api_key="test_key")

        filter_instance._total_calls = 100
        filter_instance._cache_hits = 25
        filter_instance._errors = 2

        stats = filter_instance.get_stats()

        assert stats["total_calls"] == 100
        assert stats["cache_hits"] == 25
        assert stats["cache_hit_rate"] == 0.25
        assert stats["errors"] == 2
        assert stats["model"] == "claude-sonnet-4-20250514"

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        filter_instance = LLMContextFilter(api_key="test_key")

        # Add something to cache
        filter_instance._cache["test_key"] = LLMDecision(
            execute=True, confidence=0.9, adjusted_size=1.0, reason="Test"
        )

        filter_instance.clear_cache()

        assert len(filter_instance._cache) == 0


class TestEvaluateSignalFunction:
    """Tests for evaluate_signal convenience function."""

    @pytest.mark.asyncio
    async def test_evaluate_signal_quick(self) -> None:
        """Test quick signal evaluation."""
        from src.brain.context_filter import evaluate_signal

        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.7,
            entry_price=50000.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            reason="Test",
            timestamp=datetime.now(timezone.utc),
        )

        # Without API key, should auto-approve
        with patch("src.brain.context_filter.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = None

            decision = await evaluate_signal(
                signal=signal,
                price=50000.0,
                funding_rate=0.0001,
                fear_greed=50,
            )

            assert decision.execute is True
