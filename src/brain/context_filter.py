"""LLM Context Filter for evaluating trading signals with market context.

Uses Claude API to make informed decisions about signal execution
based on current market conditions, macro context, and recent news.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from anthropic import Anthropic, APIError, RateLimitError
from cachetools import TTLCache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.brain import LLMDecision
from src.collectors.macro import MacroContext
from src.collectors.news import NewsContext, NewsItem
from src.signals.base import Signal
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketState:
    """Current market state for LLM evaluation.

    Attributes:
        timestamp: Current timestamp
        price: Current BTC price
        price_change_24h: 24h price change percentage
        funding_rate: Current funding rate
        open_interest: Current open interest in USD
        long_short_ratio: Long/Short account ratio
        volume_24h: 24h trading volume in USD
    """

    timestamp: datetime
    price: float
    price_change_24h: float
    funding_rate: float
    open_interest: float
    long_short_ratio: float
    volume_24h: float = 0.0


# Prompt template for signal evaluation
EVALUATION_PROMPT = """You are a crypto trading AI assistant. Evaluate whether to execute the following trading signal based on the current market context.

## Trading Signal
- Type: {signal_type}
- Direction: {direction}
- Entry Price: ${entry_price:,.2f}
- Stop Loss: ${stop_loss:,.2f}
- Take Profit: ${take_profit:,.2f}
- Signal Confidence: {signal_confidence:.1%}
- Reason: {signal_reason}

## Current Market State
- Current Price: ${current_price:,.2f}
- 24h Price Change: {price_change_24h:+.2f}%
- Funding Rate: {funding_rate:.4%}
- Open Interest: ${open_interest:,.0f}
- Long/Short Ratio: {long_short_ratio:.2f}

## Macro Context
- Fear & Greed Index: {fear_greed} ({fear_greed_label})
- DXY (US Dollar Index): {dxy:.2f} ({dxy_change:+.2f}%)
- S&P 500 Change: {sp500_change:+.2f}%
- NASDAQ Change: {nasdaq_change:+.2f}%
- US 10Y Yield: {us10y_yield:.2f}%
{upcoming_events}

## Recent News (Last 1 Hour)
{recent_news}

## Whale Activity (Last 30 Minutes)
{whale_activity}

## Evaluation Criteria
1. Does the news support or contradict the signal direction?
2. Is the macro environment favorable for this trade?
3. Are there any red flags or unusual market conditions?
4. Is the risk/reward ratio appropriate given current volatility?

## Response Instructions
Respond ONLY with a valid JSON object in this exact format:
{{
    "execute": true or false,
    "confidence": 0.0 to 1.0,
    "adjusted_size": 0.0 to 1.0,
    "reason": "Brief explanation (1-2 sentences)"
}}

Important:
- Set execute=false if there are significant concerns
- Reduce adjusted_size if conditions are uncertain
- Be conservative with confidence scores
- Keep reason concise but informative"""


class LLMContextFilter:
    """Evaluates trading signals using Claude LLM with full market context.

    Features:
    - Integrates market state, macro context, and news
    - Uses TTL cache to reduce API calls (5 minute cache)
    - Automatic retry with exponential backoff
    - Conservative fallback on errors
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    CACHE_TTL = 300  # 5 minutes
    CACHE_MAXSIZE = 100

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the LLM context filter.

        Args:
            api_key: Anthropic API key (uses env if not provided)
            model: Model to use (default: claude-sonnet-4-20250514)
        """
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or self.DEFAULT_MODEL

        if not self.api_key:
            logger.warning("No Anthropic API key configured - LLM filter disabled")
            self.client = None
        else:
            self.client = Anthropic(api_key=self.api_key)

        # Response cache to reduce API calls
        self._cache: TTLCache = TTLCache(
            maxsize=self.CACHE_MAXSIZE,
            ttl=self.CACHE_TTL,
        )

        # Statistics
        self._total_calls = 0
        self._cache_hits = 0
        self._errors = 0

    def _create_cache_key(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: Optional[MacroContext],
    ) -> str:
        """Create cache key from signal and context.

        Uses signal type, direction, and rounded values to group similar requests.
        """
        price_bucket = round(market_state.price / 100) * 100  # $100 buckets
        fg_bucket = (macro_context.fear_greed_index // 10) * 10 if macro_context else 50

        return f"{signal.type}:{signal.direction}:{price_bucket}:{fg_bucket}"

    def _build_prompt(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: Optional[MacroContext],
        news_context: Optional[NewsContext],
    ) -> str:
        """Build the evaluation prompt from all context."""
        # Format upcoming events
        upcoming_events = ""
        if macro_context and macro_context.upcoming_events:
            events_list = [
                f"  - {e.name} ({e.importance})"
                for e in macro_context.upcoming_events[:5]
            ]
            upcoming_events = "- Upcoming Events:\n" + "\n".join(events_list)

        # Format recent news
        recent_news = "No recent news available."
        if news_context and news_context.recent_news:
            news_items = []
            for n in news_context.recent_news[:10]:  # Limit to 10 items
                sentiment_icon = (
                    "[+]" if n.sentiment == "positive" else
                    "[-]" if n.sentiment == "negative" else "[=]"
                )
                news_items.append(f"  {sentiment_icon} {n.title} (Source: {n.source})")
            recent_news = "\n".join(news_items)

        # Format whale activity
        whale_activity = "No significant whale activity."
        if news_context and news_context.whale_alerts:
            alerts = []
            for a in news_context.whale_alerts[:5]:  # Limit to 5 items
                alerts.append(
                    f"  - {a.symbol}: ${a.amount_usd:,.0f} "
                    f"({a.from_owner} -> {a.to_owner})"
                )
            if alerts:
                whale_activity = "\n".join(alerts)
                whale_activity += (
                    f"\n  Exchange Inflows: ${news_context.exchange_inflows:,.0f}\n"
                    f"  Exchange Outflows: ${news_context.exchange_outflows:,.0f}"
                )

        # Build prompt
        prompt = EVALUATION_PROMPT.format(
            signal_type=signal.type,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_confidence=signal.confidence,
            signal_reason=signal.reason,
            current_price=market_state.price,
            price_change_24h=market_state.price_change_24h,
            funding_rate=market_state.funding_rate,
            open_interest=market_state.open_interest,
            long_short_ratio=market_state.long_short_ratio,
            fear_greed=macro_context.fear_greed_index if macro_context else 50,
            fear_greed_label=macro_context.fear_greed_label if macro_context else "Neutral",
            dxy=macro_context.dxy if macro_context else 0,
            dxy_change=macro_context.dxy_change if macro_context else 0,
            sp500_change=macro_context.sp500_change if macro_context else 0,
            nasdaq_change=macro_context.nasdaq_change if macro_context else 0,
            us10y_yield=macro_context.us10y_yield if macro_context else 0,
            upcoming_events=upcoming_events,
            recent_news=recent_news,
            whale_activity=whale_activity,
        )

        return prompt

    def _parse_response(self, response_text: str) -> LLMDecision:
        """Parse LLM response into LLMDecision.

        Args:
            response_text: Raw response from LLM

        Returns:
            LLMDecision object

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response (handles markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON object found in response")

            json_str = json_match.group()
            data = json.loads(json_str)

            # Validate required fields
            execute = bool(data.get("execute", False))
            confidence = float(data.get("confidence", 0.5))
            adjusted_size = float(data.get("adjusted_size", 1.0))
            reason = str(data.get("reason", "No reason provided"))

            # Clamp values to valid ranges
            confidence = max(0.0, min(1.0, confidence))
            adjusted_size = max(0.0, min(1.0, adjusted_size))

            return LLMDecision(
                execute=execute,
                confidence=confidence,
                adjusted_size=adjusted_size,
                reason=reason,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                "Failed to parse LLM response",
                error=str(e),
                response=response_text[:200],
            )
            raise ValueError(f"Failed to parse response: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _call_llm(self, prompt: str) -> str:
        """Call the Claude API with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            Response text from Claude

        Raises:
            APIError: On API errors (after retries)
        """
        if not self.client:
            raise ValueError("LLM client not initialized - check API key")

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.content[0].text

        except RateLimitError:
            logger.warning("Claude API rate limit hit, retrying...")
            raise
        except APIError as e:
            logger.error("Claude API error", error=str(e))
            raise

    async def evaluate(
        self,
        signal: Signal,
        market_state: MarketState,
        macro_context: Optional[MacroContext] = None,
        news_context: Optional[NewsContext] = None,
    ) -> LLMDecision:
        """Evaluate a trading signal with full market context.

        Args:
            signal: Trading signal to evaluate
            market_state: Current market state
            macro_context: Optional macro economic context
            news_context: Optional recent news context

        Returns:
            LLMDecision with execution recommendation

        Note:
            Returns conservative decision (execute=False) on any error
        """
        self._total_calls += 1

        # Check if LLM is disabled
        if not self.client:
            logger.debug("LLM disabled - auto-approving signal")
            return LLMDecision(
                execute=True,
                confidence=signal.confidence,
                adjusted_size=1.0,
                reason="LLM filter disabled - signal passed through",
            )

        # Check cache
        cache_key = self._create_cache_key(signal, market_state, macro_context)
        if cache_key in self._cache:
            self._cache_hits += 1
            logger.debug("LLM cache hit", key=cache_key)
            return self._cache[cache_key]

        try:
            # Build prompt
            prompt = self._build_prompt(
                signal, market_state, macro_context, news_context
            )

            # Call LLM
            response_text = await self._call_llm(prompt)

            # Parse response
            decision = self._parse_response(response_text)

            # Cache result
            self._cache[cache_key] = decision

            logger.info(
                "LLM evaluation complete",
                signal_type=signal.type,
                direction=signal.direction,
                execute=decision.execute,
                confidence=decision.confidence,
                reason=decision.reason,
            )

            return decision

        except Exception as e:
            self._errors += 1
            logger.error(
                "LLM evaluation failed, returning conservative decision",
                error=str(e),
                signal_type=signal.type,
            )

            # Return conservative fallback
            return LLMDecision(
                execute=False,
                confidence=0.0,
                adjusted_size=0.0,
                reason=f"LLM evaluation failed: {str(e)[:50]}",
            )

    def get_stats(self) -> dict:
        """Get filter statistics.

        Returns:
            Dict with total_calls, cache_hits, cache_hit_rate, errors
        """
        hit_rate = (
            self._cache_hits / self._total_calls
            if self._total_calls > 0
            else 0.0
        )

        return {
            "total_calls": self._total_calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": hit_rate,
            "errors": self._errors,
            "model": self.model,
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        logger.info("LLM cache cleared")


# Convenience function for quick evaluation
async def evaluate_signal(
    signal: Signal,
    price: float,
    funding_rate: float = 0.0,
    fear_greed: int = 50,
    api_key: Optional[str] = None,
) -> LLMDecision:
    """Quick signal evaluation with minimal context.

    Args:
        signal: Trading signal to evaluate
        price: Current market price
        funding_rate: Current funding rate
        fear_greed: Fear and Greed index value
        api_key: Optional API key override

    Returns:
        LLMDecision with recommendation
    """
    filter_instance = LLMContextFilter(api_key=api_key)

    market_state = MarketState(
        timestamp=datetime.now(timezone.utc),
        price=price,
        price_change_24h=0.0,
        funding_rate=funding_rate,
        open_interest=0.0,
        long_short_ratio=1.0,
    )

    macro_context = MacroContext(
        timestamp=datetime.now(timezone.utc),
        fear_greed_index=fear_greed,
        fear_greed_label="Neutral",
        dxy=0.0,
        dxy_change=0.0,
        sp500_change=0.0,
        nasdaq_change=0.0,
        us10y_yield=0.0,
    )

    return await filter_instance.evaluate(signal, market_state, macro_context)
