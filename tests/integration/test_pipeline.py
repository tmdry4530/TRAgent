"""Integration tests for Wick Reversal scalping pipeline.

Tests the complete flow:
Collector → WickReversalSignal → Risk Manager → Executor
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.risk.manager import RiskManager, TradeResult
from src.signals.base import Signal
from src.signals.scalp import (
    CandleData,
    WickReversalSignal,
    WickSignalContext,
)


# Default risk config for testing
DEFAULT_RISK_CONFIG = {
    "risk": {
        "daily_loss_limit": 0.10,
        "consecutive_loss_cooldown": 3,
        "event_blackout_minutes": 0,
        "llm_confidence_threshold": 0.0,  # No LLM
    },
}


@dataclass
class PipelineResult:
    """Result of a single pipeline run."""

    signal: Optional[Signal]
    risk_approved: bool
    final_action: str  # "EXECUTE", "RISK_REJECTED", "NO_SIGNAL"
    position_size: float


class MockMarketDataGenerator:
    """Generate mock market data for testing Wick Reversal signals."""

    def __init__(self):
        self.base_price = 50000.0

    def generate_uptrend_15m(self, num_candles: int = 25) -> list[CandleData]:
        """Generate 15m candles showing uptrend."""
        candles = []
        for i in range(num_candles):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (num_candles - i)),
                open=self.base_price + i * 50,
                high=self.base_price + i * 50 + 100,
                low=self.base_price + i * 50 - 50,
                close=self.base_price + i * 50 + 80,
                volume=100.0,
                interval="15m",
            )
            candles.append(candle)
        return candles

    def generate_downtrend_15m(self, num_candles: int = 25) -> list[CandleData]:
        """Generate 15m candles showing downtrend."""
        candles = []
        for i in range(num_candles):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (num_candles - i)),
                open=self.base_price - i * 50,
                high=self.base_price - i * 50 + 50,
                low=self.base_price - i * 50 - 100,
                close=self.base_price - i * 50 - 80,
                volume=100.0,
                interval="15m",
            )
            candles.append(candle)
        return candles

    def generate_normal_1m(self, num_candles: int = 20, start_price: float = 51200.0) -> list[CandleData]:
        """Generate 1m candles with normal volume."""
        candles = []
        for i in range(num_candles):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=num_candles + 1 - i),
                open=start_price + i * 10,
                high=start_price + i * 10 + 50,
                low=start_price + i * 10 - 50,
                close=start_price + i * 10 + 20,
                volume=100.0,
                interval="1m",
            )
            candles.append(candle)
        return candles

    def generate_bullish_hammer(self, current_price: float) -> CandleData:
        """Generate a bullish hammer candle with high volume."""
        return CandleData(
            timestamp=datetime.now(timezone.utc),
            open=current_price - 10,
            high=current_price,
            low=current_price - 200,  # Long lower wick
            close=current_price - 5,
            volume=400.0,  # 4x normal volume
            interval="1m",
        )

    def generate_bearish_shooting_star(self, current_price: float) -> CandleData:
        """Generate a bearish shooting star candle with high volume."""
        return CandleData(
            timestamp=datetime.now(timezone.utc),
            open=current_price + 10,
            high=current_price + 200,  # Long upper wick
            low=current_price,
            close=current_price + 5,
            volume=400.0,  # 4x normal volume
            interval="1m",
        )


class TestWickReversalPipeline:
    """Integration tests for Wick Reversal trading pipeline."""

    @pytest.fixture
    def market_gen(self) -> MockMarketDataGenerator:
        """Create mock market data generator."""
        return MockMarketDataGenerator()

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Create risk manager instance."""
        return RiskManager(DEFAULT_RISK_CONFIG)

    @pytest.fixture
    def signal_generator(self) -> WickReversalSignal:
        """Create signal generator with RSI disabled for testing."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            return WickReversalSignal(rsi_enabled=False, atr_enabled=False)

    async def _run_pipeline(
        self,
        signal: Optional[Signal],
        risk_manager: RiskManager,
        account_balance: float = 10000.0,
    ) -> PipelineResult:
        """Run a signal through the pipeline."""
        if signal is None:
            return PipelineResult(
                signal=None,
                risk_approved=False,
                final_action="NO_SIGNAL",
                position_size=0.0,
            )

        # Apply risk check
        account_state = {
            "balance": account_balance,
            "positions": [],
            "llm_confidence": signal.confidence,
        }
        risk_result = risk_manager.check(signal, account_state)

        if not risk_result.approved:
            return PipelineResult(
                signal=signal,
                risk_approved=False,
                final_action="RISK_REJECTED",
                position_size=0.0,
            )

        return PipelineResult(
            signal=signal,
            risk_approved=True,
            final_action="EXECUTE",
            position_size=risk_result.adjusted_size,
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_long_signal(
        self,
        market_gen: MockMarketDataGenerator,
        risk_manager: RiskManager,
        signal_generator: WickReversalSignal,
    ) -> None:
        """Test full pipeline with bullish wick reversal signal."""
        # Setup uptrend with bullish hammer
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()
        hammer = market_gen.generate_bullish_hammer(51480.0)

        for candle in candles_15m:
            signal_generator.add_candle(candle)
        for candle in candles_1m:
            signal_generator.add_candle(candle)
        signal_generator.add_candle(hammer)

        # Generate signal
        market_state = {"price": 51475.0}
        signal = await signal_generator.generate(market_state)

        # Run through pipeline
        result = await self._run_pipeline(signal, risk_manager)

        assert result.signal is not None
        assert result.signal.direction == "LONG"
        assert result.risk_approved is True
        assert result.final_action == "EXECUTE"
        assert result.position_size > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_short_signal(
        self,
        market_gen: MockMarketDataGenerator,
        risk_manager: RiskManager,
        signal_generator: WickReversalSignal,
    ) -> None:
        """Test full pipeline with bearish wick reversal signal."""
        # Setup downtrend with shooting star
        candles_15m = market_gen.generate_downtrend_15m()
        candles_1m = market_gen.generate_normal_1m(start_price=48800.0)
        shooting_star = market_gen.generate_bearish_shooting_star(48530.0)

        for candle in candles_15m:
            signal_generator.add_candle(candle)
        for candle in candles_1m:
            signal_generator.add_candle(candle)
        signal_generator.add_candle(shooting_star)

        # Generate signal
        market_state = {"price": 48535.0}
        signal = await signal_generator.generate(market_state)

        # Run through pipeline
        result = await self._run_pipeline(signal, risk_manager)

        assert result.signal is not None
        assert result.signal.direction == "SHORT"
        assert result.risk_approved is True
        assert result.final_action == "EXECUTE"

    @pytest.mark.asyncio
    async def test_pipeline_no_signal(
        self,
        market_gen: MockMarketDataGenerator,
        risk_manager: RiskManager,
        signal_generator: WickReversalSignal,
    ) -> None:
        """Test pipeline when no signal conditions are met."""
        # Setup uptrend without hammer pattern
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()

        for candle in candles_15m:
            signal_generator.add_candle(candle)
        for candle in candles_1m:
            signal_generator.add_candle(candle)

        # Add normal candle (no wick pattern)
        normal_candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=51400.0,
            high=51450.0,
            low=51350.0,
            close=51420.0,
            volume=150.0,  # Not enough volume spike
            interval="1m",
        )
        signal_generator.add_candle(normal_candle)

        # Generate signal
        market_state = {"price": 51420.0}
        signal = await signal_generator.generate(market_state)

        # Run through pipeline
        result = await self._run_pipeline(signal, risk_manager)

        assert result.final_action == "NO_SIGNAL"
        assert result.position_size == 0.0

    @pytest.mark.asyncio
    async def test_pipeline_risk_rejection_consecutive_losses(
        self,
        market_gen: MockMarketDataGenerator,
        signal_generator: WickReversalSignal,
    ) -> None:
        """Test pipeline rejection due to consecutive losses."""
        # Create risk manager with consecutive losses
        risk_manager = RiskManager(DEFAULT_RISK_CONFIG)
        for _ in range(4):
            trade_result = TradeResult(
                timestamp=datetime.now(timezone.utc),
                pnl=-100.0,
                pnl_pct=-1.0,
                signal_type="SCALP",
                direction="LONG",
            )
            risk_manager.record_trade_result(trade_result)

        # Setup valid signal scenario
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()
        hammer = market_gen.generate_bullish_hammer(51480.0)

        for candle in candles_15m:
            signal_generator.add_candle(candle)
        for candle in candles_1m:
            signal_generator.add_candle(candle)
        signal_generator.add_candle(hammer)

        # Generate signal
        market_state = {"price": 51475.0}
        signal = await signal_generator.generate(market_state)

        # Run through pipeline
        result = await self._run_pipeline(signal, risk_manager)

        assert result.signal is not None
        assert result.risk_approved is False
        assert result.final_action == "RISK_REJECTED"


class TestExitConditions:
    """Test exit condition logic."""

    @pytest.fixture
    def signal_generator(self) -> WickReversalSignal:
        """Create signal generator with RSI disabled for testing."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            return WickReversalSignal(rsi_enabled=False, atr_enabled=False)

    def test_time_based_exit(self, signal_generator: WickReversalSignal) -> None:
        """Test time-based exit triggering."""
        # Set up old signal context
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=6),
            volume_threshold=200.0,
        )

        assert signal_generator.should_exit_on_time(5) is True
        assert signal_generator.should_exit_on_time(10) is False

    def test_volume_based_exit(self, signal_generator: WickReversalSignal) -> None:
        """Test volume-based exit triggering."""
        # Entry 1 minute ago - should allow volume exit
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=60),
            volume_threshold=200.0,
        )

        assert signal_generator.should_exit_on_volume(250.0) is True
        assert signal_generator.should_exit_on_volume(150.0) is False

        # Entry just now - should not trigger volume exit (min holding time)
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc),
            volume_threshold=200.0,
        )
        assert signal_generator.should_exit_on_volume(250.0) is False


class TestMultipleScenarios:
    """Test multiple market scenarios."""

    @pytest.fixture
    def market_gen(self) -> MockMarketDataGenerator:
        return MockMarketDataGenerator()

    @pytest.fixture
    def signal_generator(self) -> WickReversalSignal:
        """Create signal generator with RSI disabled for testing."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            return WickReversalSignal(rsi_enabled=False, atr_enabled=False)

    @pytest.mark.asyncio
    async def test_trading_session_simulation(
        self,
        market_gen: MockMarketDataGenerator,
        signal_generator: WickReversalSignal,
    ) -> None:
        """Simulate a trading session with various conditions."""
        risk_manager = RiskManager(DEFAULT_RISK_CONFIG)
        results = []

        # Scenario 1: Valid long setup
        signal_generator.clear_buffers()
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()
        hammer = market_gen.generate_bullish_hammer(51480.0)

        for c in candles_15m:
            signal_generator.add_candle(c)
        for c in candles_1m:
            signal_generator.add_candle(c)
        signal_generator.add_candle(hammer)

        signal = await signal_generator.generate({"price": 51475.0})
        results.append({"scenario": "valid_long", "signal": signal})

        # Scenario 2: Mismatch (uptrend + upper wick)
        signal_generator.clear_buffers()
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()
        shooting_star = market_gen.generate_bearish_shooting_star(51480.0)

        for c in candles_15m:
            signal_generator.add_candle(c)
        for c in candles_1m:
            signal_generator.add_candle(c)
        signal_generator.add_candle(shooting_star)

        signal = await signal_generator.generate({"price": 51485.0})
        results.append({"scenario": "mismatch", "signal": signal})

        # Scenario 3: Valid short setup
        signal_generator.clear_buffers()
        candles_15m = market_gen.generate_downtrend_15m()
        candles_1m = market_gen.generate_normal_1m(start_price=48800.0)
        shooting_star = market_gen.generate_bearish_shooting_star(48530.0)

        for c in candles_15m:
            signal_generator.add_candle(c)
        for c in candles_1m:
            signal_generator.add_candle(c)
        signal_generator.add_candle(shooting_star)

        signal = await signal_generator.generate({"price": 48535.0})
        results.append({"scenario": "valid_short", "signal": signal})

        # Scenario 4: Low volume (no signal)
        signal_generator.clear_buffers()
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()

        for c in candles_15m:
            signal_generator.add_candle(c)
        for c in candles_1m:
            signal_generator.add_candle(c)

        # Add hammer with low volume
        low_volume_hammer = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=51470.0,
            high=51500.0,
            low=51300.0,
            close=51480.0,
            volume=150.0,  # Only 1.5x, not 3x
            interval="1m",
        )
        signal_generator.add_candle(low_volume_hammer)

        signal = await signal_generator.generate({"price": 51480.0})
        results.append({"scenario": "low_volume", "signal": signal})

        # Verify results
        valid_long = next(r for r in results if r["scenario"] == "valid_long")
        mismatch = next(r for r in results if r["scenario"] == "mismatch")
        valid_short = next(r for r in results if r["scenario"] == "valid_short")
        low_volume = next(r for r in results if r["scenario"] == "low_volume")

        assert valid_long["signal"] is not None
        assert valid_long["signal"].direction == "LONG"

        assert mismatch["signal"] is None  # Trend-wick mismatch

        assert valid_short["signal"] is not None
        assert valid_short["signal"].direction == "SHORT"

        assert low_volume["signal"] is None  # Volume too low

        # Print summary
        print("\n" + "=" * 60)
        print("TRADING SESSION SIMULATION RESULTS")
        print("=" * 60)
        for r in results:
            signal_dir = r["signal"].direction if r["signal"] else "NO_SIGNAL"
            print(f"  {r['scenario']:<15} -> {signal_dir}")
        print("=" * 60)

    @pytest.mark.asyncio
    async def test_signal_validation(self) -> None:
        """Test that generated signals are valid."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            signal_generator = WickReversalSignal(rsi_enabled=False, atr_enabled=False)

        market_gen = MockMarketDataGenerator()

        # Generate valid long signal
        candles_15m = market_gen.generate_uptrend_15m()
        candles_1m = market_gen.generate_normal_1m()
        hammer = market_gen.generate_bullish_hammer(51480.0)

        for c in candles_15m:
            signal_generator.add_candle(c)
        for c in candles_1m:
            signal_generator.add_candle(c)
        signal_generator.add_candle(hammer)

        signal = await signal_generator.generate({"price": 51475.0})

        assert signal is not None
        assert signal.validate() is True
        assert signal.type == "SCALP"
        assert 0 <= signal.confidence <= 1
        assert signal.stop_loss < signal.entry_price < signal.take_profit
