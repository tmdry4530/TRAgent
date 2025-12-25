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
    """Signal generator for EMA+RSI swing trading strategy.

    Generates swing signals based on EMA alignment and RSI positioning.
    LONG: EMA 7 > 25 > 99 (bullish alignment) + RSI 40-60 (neutral zone)
    SHORT: EMA 7 < 25 < 99 (bearish alignment) + RSI 40-60 (neutral zone)

    This strategy captures medium-term trends with confirmation.
    """

    def __init__(self) -> None:
        """Initialize EMA+RSI swing signal generator."""
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
        """Calculate EMA and RSI indicators.

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

        return df

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

        Args:
            ema_7: 7-period EMA value
            ema_25: 25-period EMA value
            ema_99: 99-period EMA value
            rsi: Current RSI value

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.7

        # Bonus for strong EMA separation
        ema_7_25_gap = abs(ema_7 - ema_25) / ema_25
        ema_25_99_gap = abs(ema_25 - ema_99) / ema_99

        if ema_7_25_gap > 0.02:  # >2% gap
            confidence += 0.1
        if ema_25_99_gap > 0.02:
            confidence += 0.1

        # Bonus for RSI near 50 (most neutral)
        rsi_distance_from_50 = abs(rsi - 50)
        if rsi_distance_from_50 < 5:  # RSI 45-55
            confidence += 0.05

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

        # Calculate confidence
        confidence = self._calculate_confidence(ema_7, ema_25, ema_99, rsi)

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
            f"({direction.lower()} trend), RSI={rsi:.1f} in neutral zone"
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
