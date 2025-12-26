"""Tests for Wick Reversal signal generator."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.signals.base import Signal
from src.signals.scalp import (
    CandleData,
    WickReversalSignal,
    WickSignalContext,
)


class TestSignalDataclass:
    """Tests for Signal dataclass."""

    def test_create_signal(self) -> None:
        """Test creating a Signal instance."""
        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=49250.0,
            take_profit=51500.0,
            reason="Test signal",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.type == "SCALP"
        assert signal.direction == "LONG"
        assert signal.confidence == 0.8

    def test_validate_long_signal_valid(self) -> None:
        """Test validating a valid LONG signal."""
        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=51500.0,
            reason="Test",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.validate() is True

    def test_validate_short_signal_valid(self) -> None:
        """Test validating a valid SHORT signal."""
        signal = Signal(
            type="SCALP",
            direction="SHORT",
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=51000.0,
            take_profit=48500.0,
            reason="Test",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.validate() is True

    def test_validate_invalid_confidence(self) -> None:
        """Test that confidence outside [0, 1] is invalid."""
        signal = Signal(
            type="SCALP",
            direction="LONG",
            confidence=1.5,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,
            reason="Test",
            timestamp=datetime.now(timezone.utc),
        )
        assert signal.validate() is False


class TestCandleData:
    """Tests for CandleData dataclass."""

    def test_create_candle(self) -> None:
        """Test creating a CandleData instance."""
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=100.0,
            interval="1m",
        )
        assert candle.open == 50000.0
        assert candle.interval == "1m"


class TestWickReversalSignal:
    """Tests for WickReversalSignal generator."""

    @pytest.fixture
    def signal_generator(self) -> WickReversalSignal:
        """Create signal generator with mocked settings and RSI disabled."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            # Disable RSI filter for basic tests
            return WickReversalSignal(rsi_enabled=False)

    def test_name(self, signal_generator: WickReversalSignal) -> None:
        """Test signal generator name."""
        assert signal_generator.name == "WickReversal"

    def test_required_data(self, signal_generator: WickReversalSignal) -> None:
        """Test required data channels."""
        required = signal_generator.get_required_data()
        assert "kline_1m" in required
        assert "kline_15m" in required
        assert "price" in required

    def test_add_candle_1m(self, signal_generator: WickReversalSignal) -> None:
        """Test adding 1m candle to buffer."""
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=100.0,
            interval="1m",
        )
        signal_generator.add_candle(candle)
        assert len(signal_generator._candles_1m) == 1

    def test_add_candle_15m(self, signal_generator: WickReversalSignal) -> None:
        """Test adding 15m candle to buffer."""
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=100.0,
            interval="15m",
        )
        signal_generator.add_candle(candle)
        assert len(signal_generator._candles_15m) == 1

    def test_calculate_wick_ratio_lower_wick(self, signal_generator: WickReversalSignal) -> None:
        """Test wick ratio calculation for lower wick (bullish hammer)."""
        # Candle with 70% lower wick
        # Total range: 100 (50100 - 50000)
        # Body: close to open = 50080 - 50070 = 10 (top 10% of range)
        # Lower wick: body_bottom - low = 50070 - 50000 = 70 (70% of range)
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50070.0,
            high=50100.0,
            low=50000.0,
            close=50080.0,
            volume=100.0,
            interval="1m",
        )
        lower_ratio, upper_ratio, wick_type = signal_generator._calculate_wick_ratio(candle)
        assert lower_ratio == pytest.approx(0.7, rel=0.01)
        assert wick_type == "LOWER"

    def test_calculate_wick_ratio_upper_wick(self, signal_generator: WickReversalSignal) -> None:
        """Test wick ratio calculation for upper wick (shooting star)."""
        # Candle with 70% upper wick
        # Total range: 100 (50100 - 50000)
        # Body: open to close = 50030 - 50020 = 10 (bottom 10% of range)
        # Upper wick: high - body_top = 50100 - 50030 = 70 (70% of range)
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50030.0,
            high=50100.0,
            low=50000.0,
            close=50020.0,
            volume=100.0,
            interval="1m",
        )
        lower_ratio, upper_ratio, wick_type = signal_generator._calculate_wick_ratio(candle)
        assert upper_ratio == pytest.approx(0.7, rel=0.01)
        assert wick_type == "UPPER"

    def test_calculate_wick_ratio_no_significant_wick(self, signal_generator: WickReversalSignal) -> None:
        """Test wick ratio calculation with no significant wick."""
        # Balanced candle with body in the middle
        candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50040.0,
            high=50100.0,
            low=50000.0,
            close=50060.0,
            volume=100.0,
            interval="1m",
        )
        lower_ratio, upper_ratio, wick_type = signal_generator._calculate_wick_ratio(candle)
        assert wick_type == "NONE"

    @pytest.mark.asyncio
    async def test_generate_insufficient_data(self, signal_generator: WickReversalSignal) -> None:
        """Test no signal with insufficient candle data."""
        market_state = {
            "price": 50000.0,
        }
        signal = await signal_generator.generate(market_state)
        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_long_signal(self, signal_generator: WickReversalSignal) -> None:
        """Test LONG signal generation on uptrend + lower wick."""
        # Create 25 15m candles showing uptrend (price above EMA20)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (25 - i)),
                open=base_price + i * 50,
                high=base_price + i * 50 + 100,
                low=base_price + i * 50 - 50,
                close=base_price + i * 50 + 80,  # Uptrend: closes higher
                volume=100.0,
                interval="15m",
            )
            signal_generator.add_candle(candle)

        # Create 21 1m candles with normal volume
        for i in range(20):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=21 - i),
                open=51200.0 + i * 10,
                high=51250.0 + i * 10,
                low=51150.0 + i * 10,
                close=51220.0 + i * 10,
                volume=100.0,  # Normal volume
                interval="1m",
            )
            signal_generator.add_candle(candle)

        # Add high volume candle with lower wick (bullish hammer)
        # This should trigger a LONG signal
        hammer_candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=51470.0,
            high=51500.0,
            low=51300.0,  # Long lower wick
            close=51480.0,
            volume=400.0,  # 4x normal volume
            interval="1m",
        )
        signal_generator.add_candle(hammer_candle)

        market_state = {
            "price": 51480.0,
        }
        signal = await signal_generator.generate(market_state)

        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.type == "SCALP"
        assert signal.stop_loss < signal.entry_price < signal.take_profit

    @pytest.mark.asyncio
    async def test_generate_short_signal(self, signal_generator: WickReversalSignal) -> None:
        """Test SHORT signal generation on downtrend + upper wick."""
        # Create 25 15m candles showing downtrend (price below EMA20)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (25 - i)),
                open=base_price - i * 50,
                high=base_price - i * 50 + 50,
                low=base_price - i * 50 - 100,
                close=base_price - i * 50 - 80,  # Downtrend: closes lower
                volume=100.0,
                interval="15m",
            )
            signal_generator.add_candle(candle)

        # Create 20 1m candles with normal volume
        for i in range(20):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=21 - i),
                open=48800.0 - i * 10,
                high=48850.0 - i * 10,
                low=48750.0 - i * 10,
                close=48780.0 - i * 10,
                volume=100.0,
                interval="1m",
            )
            signal_generator.add_candle(candle)

        # Add high volume candle with upper wick (shooting star)
        shooting_star = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=48520.0,
            high=48700.0,  # Long upper wick
            low=48500.0,
            close=48530.0,
            volume=400.0,  # 4x normal volume
            interval="1m",
        )
        signal_generator.add_candle(shooting_star)

        market_state = {
            "price": 48530.0,
        }
        signal = await signal_generator.generate(market_state)

        assert signal is not None
        assert signal.direction == "SHORT"
        assert signal.take_profit < signal.entry_price < signal.stop_loss

    @pytest.mark.asyncio
    async def test_no_signal_on_trend_wick_mismatch(self, signal_generator: WickReversalSignal) -> None:
        """Test no signal when trend and wick direction don't match."""
        # Create uptrend
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (25 - i)),
                open=base_price + i * 50,
                high=base_price + i * 50 + 100,
                low=base_price + i * 50 - 50,
                close=base_price + i * 50 + 80,
                volume=100.0,
                interval="15m",
            )
            signal_generator.add_candle(candle)

        # Create 1m candles with normal volume
        for i in range(20):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=21 - i),
                open=51200.0 + i * 10,
                high=51250.0 + i * 10,
                low=51150.0 + i * 10,
                close=51220.0 + i * 10,
                volume=100.0,
                interval="1m",
            )
            signal_generator.add_candle(candle)

        # Add upper wick (shooting star) in uptrend - mismatch!
        shooting_star = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=51420.0,
            high=51600.0,  # Long upper wick
            low=51400.0,
            close=51430.0,
            volume=400.0,
            interval="1m",
        )
        signal_generator.add_candle(shooting_star)

        market_state = {
            "price": 51430.0,
        }
        signal = await signal_generator.generate(market_state)

        # Should not generate signal due to mismatch
        assert signal is None

    @pytest.mark.asyncio
    async def test_no_signal_low_volume(self, signal_generator: WickReversalSignal) -> None:
        """Test no signal when volume is not spiking."""
        # Create uptrend
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (25 - i)),
                open=base_price + i * 50,
                high=base_price + i * 50 + 100,
                low=base_price + i * 50 - 50,
                close=base_price + i * 50 + 80,
                volume=100.0,
                interval="15m",
            )
            signal_generator.add_candle(candle)

        # Create 1m candles with normal volume
        for i in range(20):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=21 - i),
                open=51200.0 + i * 10,
                high=51250.0 + i * 10,
                low=51150.0 + i * 10,
                close=51220.0 + i * 10,
                volume=100.0,
                interval="1m",
            )
            signal_generator.add_candle(candle)

        # Add hammer with normal volume (not 3x)
        hammer_candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=51470.0,
            high=51500.0,
            low=51300.0,
            close=51480.0,
            volume=150.0,  # Only 1.5x, not 3x
            interval="1m",
        )
        signal_generator.add_candle(hammer_candle)

        market_state = {
            "price": 51480.0,
        }
        signal = await signal_generator.generate(market_state)

        assert signal is None

    def test_should_exit_on_time(self, signal_generator: WickReversalSignal) -> None:
        """Test time-based exit condition."""
        # No context - should return False
        assert signal_generator.should_exit_on_time(5) is False

        # Set context with old entry time
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=6),
            volume_threshold=200.0,
        )
        assert signal_generator.should_exit_on_time(5) is True

        # Set context with recent entry time
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=3),
            volume_threshold=200.0,
        )
        assert signal_generator.should_exit_on_time(5) is False

    def test_should_exit_on_volume(self, signal_generator: WickReversalSignal) -> None:
        """Test volume-based exit condition."""
        from datetime import timedelta

        # No context - should return False
        assert signal_generator.should_exit_on_volume(300.0) is False

        # Set context with volume threshold (entry time 1 minute ago)
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc) - timedelta(seconds=60),
            volume_threshold=200.0,
        )

        # Volume above threshold (after min holding time)
        assert signal_generator.should_exit_on_volume(300.0) is True

        # Volume below threshold
        assert signal_generator.should_exit_on_volume(150.0) is False

        # Test min holding time - entry just now should not exit
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=50000.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc),
            volume_threshold=200.0,
        )
        assert signal_generator.should_exit_on_volume(300.0) is False  # Too early

    def test_clear_buffers(self, signal_generator: WickReversalSignal) -> None:
        """Test clearing candle buffers."""
        # Add some candles
        candle_1m = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            interval="1m",
        )
        candle_15m = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=100.0,
            interval="15m",
        )
        signal_generator.add_candle(candle_1m)
        signal_generator.add_candle(candle_15m)
        signal_generator._last_signal_context = WickSignalContext(
            wick_low=49900.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc),
            volume_threshold=200.0,
        )

        assert len(signal_generator._candles_1m) == 1
        assert len(signal_generator._candles_15m) == 1
        assert signal_generator._last_signal_context is not None

        # Clear buffers
        signal_generator.clear_buffers()

        assert len(signal_generator._candles_1m) == 0
        assert len(signal_generator._candles_15m) == 0
        assert signal_generator._last_signal_context is None


class TestWickSignalContext:
    """Tests for WickSignalContext dataclass."""

    def test_create_context(self) -> None:
        """Test creating a WickSignalContext instance."""
        context = WickSignalContext(
            wick_low=49900.0,
            wick_high=50100.0,
            entry_time=datetime.now(timezone.utc),
            volume_threshold=200.0,
        )
        assert context.wick_low == 49900.0
        assert context.wick_high == 50100.0
        assert context.volume_threshold == 200.0


class TestRSIFilter:
    """Tests for RSI filter in WickReversalSignal."""

    @pytest.fixture
    def signal_generator_rsi_enabled(self) -> WickReversalSignal:
        """Create signal generator with RSI enabled."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            return WickReversalSignal(
                rsi_enabled=True,
                rsi_oversold=40,
                rsi_overbought=60,
            )

    @pytest.fixture
    def signal_generator_rsi_disabled(self) -> WickReversalSignal:
        """Create signal generator with RSI disabled."""
        with patch("src.signals.scalp.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            return WickReversalSignal(rsi_enabled=False)

    def test_rsi_calculation(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI calculation."""
        # Create candles with consistent upward movement (should have high RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price + i * 100,
                high=base_price + i * 100 + 50,
                low=base_price + i * 100 - 20,
                close=base_price + i * 100 + 30,  # Consistently up
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        rsi = signal_generator_rsi_enabled._calculate_rsi(candles, 14)

        assert rsi is not None
        assert 60 < rsi <= 100  # Should be high RSI (overbought territory)

    def test_rsi_calculation_downtrend(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI calculation in downtrend."""
        # Create candles with consistent downward movement (should have low RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price - i * 100,
                high=base_price - i * 100 + 20,
                low=base_price - i * 100 - 50,
                close=base_price - i * 100 - 30,  # Consistently down
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        rsi = signal_generator_rsi_enabled._calculate_rsi(candles, 14)

        assert rsi is not None
        assert 0 <= rsi < 40  # Should be low RSI (oversold territory)

    def test_rsi_filter_long_passes(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI filter allows LONG when RSI < 40."""
        # Create downtrend candles (low RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price - i * 100,
                high=base_price - i * 100 + 20,
                low=base_price - i * 100 - 50,
                close=base_price - i * 100 - 30,
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        passes, rsi = signal_generator_rsi_enabled._check_rsi_filter(candles, "LONG")

        assert passes is True
        assert rsi < 40

    def test_rsi_filter_long_blocked(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI filter blocks LONG when RSI > 40."""
        # Create uptrend candles (high RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price + i * 100,
                high=base_price + i * 100 + 50,
                low=base_price + i * 100 - 20,
                close=base_price + i * 100 + 30,
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        passes, rsi = signal_generator_rsi_enabled._check_rsi_filter(candles, "LONG")

        assert passes is False
        assert rsi > 40

    def test_rsi_filter_short_passes(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI filter allows SHORT when RSI > 60."""
        # Create uptrend candles (high RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price + i * 100,
                high=base_price + i * 100 + 50,
                low=base_price + i * 100 - 20,
                close=base_price + i * 100 + 30,
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        passes, rsi = signal_generator_rsi_enabled._check_rsi_filter(candles, "SHORT")

        assert passes is True
        assert rsi > 60

    def test_rsi_filter_short_blocked(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI filter blocks SHORT when RSI < 60."""
        # Create downtrend candles (low RSI)
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=25 - i),
                open=base_price - i * 100,
                high=base_price - i * 100 + 20,
                low=base_price - i * 100 - 50,
                close=base_price - i * 100 - 30,
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        passes, rsi = signal_generator_rsi_enabled._check_rsi_filter(candles, "SHORT")

        assert passes is False
        assert rsi < 60

    def test_rsi_filter_disabled(self, signal_generator_rsi_disabled: WickReversalSignal) -> None:
        """Test RSI filter passes when disabled."""
        # No candles needed - should pass regardless
        candles: list[CandleData] = []
        passes, rsi = signal_generator_rsi_disabled._check_rsi_filter(candles, "LONG")

        assert passes is True
        assert rsi == 50.0  # Neutral when disabled

    def test_rsi_insufficient_data(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test RSI filter with insufficient data."""
        # Only 10 candles (need 15 for RSI 14)
        for i in range(10):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=10 - i),
                open=50000.0 + i * 10,
                high=50050.0 + i * 10,
                low=49950.0 + i * 10,
                close=50020.0 + i * 10,
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        candles = list(signal_generator_rsi_enabled._candles_1m)
        passes, rsi = signal_generator_rsi_enabled._check_rsi_filter(candles, "LONG")

        assert passes is False
        assert rsi == 0.0

    @pytest.mark.asyncio
    async def test_generate_long_blocked_by_rsi(self, signal_generator_rsi_enabled: WickReversalSignal) -> None:
        """Test LONG signal blocked by high RSI."""
        # Create 25 15m candles showing uptrend
        base_price = 50000.0
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=15 * (25 - i)),
                open=base_price + i * 50,
                high=base_price + i * 50 + 100,
                low=base_price + i * 50 - 50,
                close=base_price + i * 50 + 80,
                volume=100.0,
                interval="15m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        # Create 1m candles showing strong uptrend (HIGH RSI)
        for i in range(25):
            candle = CandleData(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=26 - i),
                open=51200.0 + i * 50,
                high=51300.0 + i * 50,
                low=51180.0 + i * 50,
                close=51280.0 + i * 50,  # Strong uptrend
                volume=100.0,
                interval="1m",
            )
            signal_generator_rsi_enabled.add_candle(candle)

        # Add high volume candle with lower wick (bullish hammer)
        hammer_candle = CandleData(
            timestamp=datetime.now(timezone.utc),
            open=52670.0,
            high=52700.0,
            low=52500.0,  # Long lower wick
            close=52680.0,
            volume=400.0,  # 4x normal volume
            interval="1m",
        )
        signal_generator_rsi_enabled.add_candle(hammer_candle)

        market_state = {"price": 52680.0}
        signal = await signal_generator_rsi_enabled.generate(market_state)

        # Should be blocked by RSI (RSI is high due to strong uptrend)
        assert signal is None
