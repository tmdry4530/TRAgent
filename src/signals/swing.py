"""Swing trading signal generators."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import pandas_ta as ta

from src.signals.base import BaseSignalGenerator, Signal
from src.utils.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.collectors import MarketState

logger = get_logger(__name__)


class EmaRsiSwingSignal(BaseSignalGenerator):
    """Signal generator for EMA+RSI+MACD swing trading strategy.

    Phase 2 강화:
    - EMA alignment (7 > 25 > 99 for bullish, reverse for bearish)
    - RSI in neutral zone (40-60)
    - MACD momentum confirmation

    This strategy captures medium-term trends with multiple confirmations.
    """

    # Phase 2: MACD 설정
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MACD_ENABLED = True

    def __init__(self) -> None:
        """Initialize EMA+RSI+MACD swing signal generator."""
        self.settings = get_settings()
        self.ema_periods = [7, 25, 99]
        self.rsi_period = 14
        self.rsi_entry_range = (self.settings.rsi_entry_min, self.settings.rsi_entry_max)

    @property
    def name(self) -> str:
        """Return signal generator name."""
        return "EmaRsiSwing"

    def get_required_data(self) -> list[str]:
        """Return required data channels."""
        return ["ohlcv_4h", "price"]

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA, RSI, and MACD indicators.

        Args:
            df: OHLCV dataframe with 'close' column

        Returns:
            DataFrame with added indicator columns
        """
        # Calculate EMAs
        for period in self.ema_periods:
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)

        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)

        # Phase 2: Calculate MACD
        if self.MACD_ENABLED:
            macd_result = ta.macd(
                df["close"],
                fast=self.MACD_FAST,
                slow=self.MACD_SLOW,
                signal=self.MACD_SIGNAL,
            )
            if macd_result is not None:
                df["macd"] = macd_result.iloc[:, 0]  # MACD line
                df["macd_signal"] = macd_result.iloc[:, 1]  # Signal line
                df["macd_histogram"] = macd_result.iloc[:, 2]  # Histogram

        return df

    def _check_macd_momentum(
        self, macd: float, signal: float, histogram: float, direction: str
    ) -> tuple[bool, float]:
        """Check if MACD confirms the trade direction.

        For LONG: MACD > Signal (bullish momentum)
        For SHORT: MACD < Signal (bearish momentum)

        Args:
            macd: MACD line value
            signal: Signal line value
            histogram: MACD histogram value
            direction: Trade direction (LONG or SHORT)

        Returns:
            Tuple of (passes_filter, histogram_value)
        """
        if not self.MACD_ENABLED:
            return True, 0.0

        if pd.isna(macd) or pd.isna(signal):
            return False, 0.0

        if direction == "LONG":
            # For LONG: MACD should be above signal
            passes = macd > signal
        else:
            # For SHORT: MACD should be below signal
            passes = macd < signal

        return passes, histogram if not pd.isna(histogram) else 0.0

    def _check_ema_alignment(
        self, ema_7: float, ema_25: float, ema_99: float
    ) -> str | None:
        """Check EMA alignment for trend direction.

        Args:
            ema_7: 7-period EMA value
            ema_25: 25-period EMA value
            ema_99: 99-period EMA value

        Returns:
            'LONG' for bullish alignment, 'SHORT' for bearish, None otherwise
        """
        # Bullish alignment: 7 > 25 > 99
        if ema_7 > ema_25 > ema_99:
            return "LONG"
        # Bearish alignment: 7 < 25 < 99
        elif ema_7 < ema_25 < ema_99:
            return "SHORT"
        else:
            return None

    def _check_rsi_entry_zone(self, rsi: float) -> bool:
        """Check if RSI is in entry zone (40-60).

        Args:
            rsi: Current RSI value

        Returns:
            True if RSI is in entry zone, False otherwise
        """
        min_rsi, max_rsi = self.rsi_entry_range
        return min_rsi <= rsi <= max_rsi

    def _calculate_confidence(
        self, ema_7: float, ema_25: float, ema_99: float, rsi: float
    ) -> float:
        """Calculate signal confidence based on indicator strength.

        Phase 1: 더 엄격한 신뢰도 계산
        - Base를 0.55로 낮춤 (0.7에서)
        - EMA 갭 기준을 3%로 상향 (2%에서)
        - 추세 강도에 따른 단계별 보너스

        Args:
            ema_7: 7-period EMA value
            ema_25: 25-period EMA value
            ema_99: 99-period EMA value
            rsi: Current RSI value

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Phase 1: Base confidence를 낮춰 더 선별적으로
        confidence = 0.55

        # EMA 갭 기반 보너스 (더 엄격한 기준)
        ema_7_25_gap = abs(ema_7 - ema_25) / ema_25
        ema_25_99_gap = abs(ema_25 - ema_99) / ema_99

        # 단계별 EMA 갭 보너스
        if ema_7_25_gap > 0.05:  # >5% gap: 강한 추세
            confidence += 0.15
        elif ema_7_25_gap > 0.03:  # >3% gap: 중간 추세
            confidence += 0.10
        elif ema_7_25_gap > 0.02:  # >2% gap: 약한 추세
            confidence += 0.05

        if ema_25_99_gap > 0.05:
            confidence += 0.15
        elif ema_25_99_gap > 0.03:
            confidence += 0.10
        elif ema_25_99_gap > 0.02:
            confidence += 0.05

        # RSI가 50에 가까울수록 보너스 (중립 영역)
        rsi_distance_from_50 = abs(rsi - 50)
        if rsi_distance_from_50 < 3:  # RSI 47-53: 가장 중립
            confidence += 0.05
        elif rsi_distance_from_50 < 5:  # RSI 45-55
            confidence += 0.03

        return min(0.95, confidence)

    async def generate(self, market_state: "MarketState") -> Signal | None:
        """Generate swing signal based on EMA+RSI strategy.

        Args:
            market_state: Current market state

        Returns:
            Signal if conditions are met, None otherwise
        """
        # Extract OHLCV data
        ohlcv_df = market_state.get("ohlcv_4h")
        if ohlcv_df is None or len(ohlcv_df) < max(self.ema_periods):
            logger.debug("Insufficient OHLCV data for EMA calculation")
            return None

        current_price = market_state.get("price")
        if not current_price:
            logger.warning("Missing current price data")
            return None

        # Calculate indicators
        df = self._calculate_indicators(ohlcv_df.copy())

        # Get latest values
        latest = df.iloc[-1]
        ema_7 = latest.get("ema_7")
        ema_25 = latest.get("ema_25")
        ema_99 = latest.get("ema_99")
        rsi = latest.get("rsi")

        # Check for NaN values
        if pd.isna([ema_7, ema_25, ema_99, rsi]).any():
            logger.debug("Indicators not yet calculated (NaN values)")
            return None

        # Check EMA alignment
        direction = self._check_ema_alignment(ema_7, ema_25, ema_99)
        if not direction:
            logger.debug(
                "No valid EMA alignment",
                ema_7=ema_7,
                ema_25=ema_25,
                ema_99=ema_99,
            )
            return None

        # Check RSI entry zone
        if not self._check_rsi_entry_zone(rsi):
            logger.debug(
                "RSI not in entry zone",
                rsi=rsi,
                required_range=self.rsi_entry_range,
            )
            return None

        # Phase 2: Check MACD momentum filter
        macd = latest.get("macd", 0)
        macd_signal = latest.get("macd_signal", 0)
        macd_histogram = latest.get("macd_histogram", 0)

        macd_passes, histogram_value = self._check_macd_momentum(
            macd, macd_signal, macd_histogram, direction
        )
        if not macd_passes:
            logger.debug(
                "MACD filter rejected swing signal",
                direction=direction,
                macd=macd,
                signal=macd_signal,
            )
            return None

        # Calculate confidence (with MACD bonus)
        confidence = self._calculate_confidence(ema_7, ema_25, ema_99, rsi)

        # MACD 보너스 추가 (히스토그램 강도)
        if not pd.isna(histogram_value):
            macd_bonus = min(abs(histogram_value) / 100, 0.05)
            confidence = min(0.95, confidence + macd_bonus)

        # Calculate entry/exit prices
        entry_price = current_price
        if direction == "LONG":
            stop_loss = entry_price * (1 - self.settings.swing_stop_loss_pct)
            take_profit = entry_price * (1 + self.settings.swing_take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.settings.swing_stop_loss_pct)
            take_profit = entry_price * (1 - self.settings.swing_take_profit_pct)

        reason = (
            f"EMA alignment: 7={ema_7:.2f}, 25={ema_25:.2f}, 99={ema_99:.2f} "
            f"({direction.lower()} trend), RSI={rsi:.1f}, MACD confirmed"
        )

        signal = Signal(
            type="SWING",
            direction=direction,  # type: ignore[arg-type]
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            "EMA+RSI swing signal generated",
            direction=direction,
            ema_7=ema_7,
            ema_25=ema_25,
            ema_99=ema_99,
            rsi=rsi,
            confidence=confidence,
        )

        return signal


class FearGreedFilter(BaseSignalGenerator):
    """Fear & Greed Index filter for swing signals.

    Filters swing signals based on market sentiment.
    LONG allowed: Fear & Greed < 70 (not overheated)
    SHORT allowed: Fear & Greed > 30 (not oversold)

    This acts as a sentiment filter to avoid extreme conditions.
    """

    def __init__(self) -> None:
        """Initialize Fear & Greed filter."""
        self.settings = get_settings()
        self.long_max_threshold = 70  # Don't long when greed is high
        self.short_min_threshold = 30  # Don't short when fear is high

    @property
    def name(self) -> str:
        """Return signal generator name."""
        return "FearGreedFilter"

    def get_required_data(self) -> list[str]:
        """Return required data channels."""
        return ["fear_greed_index", "price"]

    def can_long(self, fear_greed: int) -> bool:
        """Check if long signals are allowed.

        Args:
            fear_greed: Fear & Greed Index value (0-100)

        Returns:
            True if long is allowed, False otherwise
        """
        return fear_greed < self.long_max_threshold

    def can_short(self, fear_greed: int) -> bool:
        """Check if short signals are allowed.

        Args:
            fear_greed: Fear & Greed Index value (0-100)

        Returns:
            True if short is allowed, False otherwise
        """
        return fear_greed > self.short_min_threshold

    async def generate(self, market_state: "MarketState") -> Signal | None:
        """Generate filter signal based on Fear & Greed Index.

        Note: This is primarily used as a filter, not a standalone signal.
        Returns None in most cases, but can be queried for filtering.

        Args:
            market_state: Current market state

        Returns:
            None (this is a filter, not a signal generator)
        """
        # This is a filter class, not a signal generator
        # It's meant to be used by other components to filter signals
        return None
