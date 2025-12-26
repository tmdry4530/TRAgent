"""Wick Reversal scalp trading signal generator.

Strategy: 15분봉 추세 + 1분봉 꼬리 반전 신호
- 15분봉 EMA(20)로 추세 판단
- 1분봉 고거래량 + 긴 꼬리 캔들 감지
- 추세 방향과 일치하는 반전 신호만 진입
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, Optional

import pandas as pd

from src.signals.base import BaseSignalGenerator, Signal
from src.utils.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class CandleData:
    """Single candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


@dataclass
class WickSignalContext:
    """Context for wick reversal signal evaluation."""

    wick_low: float  # 꼬리 저점 (롱 손절 기준)
    wick_high: float  # 꼬리 고점 (숏 손절 기준)
    entry_time: datetime
    volume_threshold: float  # 익절용 거래량 기준


class WickReversalSignal(BaseSignalGenerator):
    """Wick Reversal Signal Generator.

    진입 조건:
    1. 15분봉: 가격이 EMA(20) 위 → 상승 추세 → 롱만
                가격이 EMA(20) 아래 → 하락 추세 → 숏만
    2. 1분봉:
       - 거래량: 최근 20봉 평균 대비 3배 이상
       - 꼬리 비율: 캔들 전체 길이의 60% 이상이 꼬리
       - 롱: 긴 아래꼬리 (하락 후 반등)
       - 숏: 긴 위꼬리 (상승 후 하락)
    3. 추세-시그널 방향 일치
    4. RSI 필터:
       - 롱: RSI < 40 (과매도 영역)
       - 숏: RSI > 60 (과매수 영역)

    손절:
    - 롱: 꼬리 저점 - 0.1%
    - 숏: 꼬리 고점 + 0.1%

    익절:
    - 동적: 거래량 2배 이상 터지면 익절
    - 시간: 최대 5분 홀딩 후 청산
    - 안전장치: 손익비 1:2 도달 시 50% 익절
    """

    # 상수
    EMA_PERIOD = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14  # ATR 계산 기간
    RSI_OVERSOLD = 40  # 롱 진입 허용 RSI 상한
    RSI_OVERBOUGHT = 60  # 숏 진입 허용 RSI 하한
    VOLUME_MULTIPLIER = 2.0  # 진입용 거래량 배수
    WICK_RATIO_THRESHOLD = 0.6  # 꼬리 비율 임계값 (60%)
    STOP_LOSS_BUFFER = 0.001  # 0.1% 손절 버퍼 (ATR 사용 안할 때)
    ATR_SL_MULTIPLIER = 1.5  # ATR x 1.5 = Stop Loss
    ATR_TP_MULTIPLIER = 3.0  # ATR x 3.0 = Take Profit (RR 1:2)
    MAX_CANDLES = 100  # 최대 캔들 저장 수

    def __init__(
        self,
        rsi_enabled: bool = True,
        rsi_oversold: float = 40,
        rsi_overbought: float = 60,
        atr_enabled: bool = True,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
    ) -> None:
        """Initialize Wick Reversal signal generator.

        Args:
            rsi_enabled: Whether to use RSI filter
            rsi_oversold: RSI threshold for long entries (default: 40)
            rsi_overbought: RSI threshold for short entries (default: 60)
            atr_enabled: Whether to use ATR-based SL/TP
            atr_sl_multiplier: ATR multiplier for stop loss (default: 1.5)
            atr_tp_multiplier: ATR multiplier for take profit (default: 3.0)
        """
        self.settings = get_settings()

        # RSI 설정
        self.rsi_enabled = rsi_enabled
        self.RSI_OVERSOLD = rsi_oversold
        self.RSI_OVERBOUGHT = rsi_overbought

        # ATR 설정
        self.atr_enabled = atr_enabled
        self.ATR_SL_MULTIPLIER = atr_sl_multiplier
        self.ATR_TP_MULTIPLIER = atr_tp_multiplier

        # 캔들 데이터 저장소
        self._candles_1m: deque[CandleData] = deque(maxlen=self.MAX_CANDLES)
        self._candles_15m: deque[CandleData] = deque(maxlen=self.MAX_CANDLES)

        # 마지막 시그널 컨텍스트 (포지션 관리용)
        self._last_signal_context: Optional[WickSignalContext] = None

        logger.debug(
            "WickReversalSignal initialized",
            ema_period=self.EMA_PERIOD,
            volume_multiplier=self.VOLUME_MULTIPLIER,
            wick_ratio=self.WICK_RATIO_THRESHOLD,
            rsi_enabled=self.rsi_enabled,
            rsi_oversold=self.RSI_OVERSOLD,
            rsi_overbought=self.RSI_OVERBOUGHT,
            atr_enabled=self.atr_enabled,
            atr_sl_mult=self.ATR_SL_MULTIPLIER,
            atr_tp_mult=self.ATR_TP_MULTIPLIER,
        )

    @property
    def name(self) -> str:
        """Return signal generator name."""
        return "WickReversal"

    def get_required_data(self) -> list[str]:
        """Return required data channels."""
        return ["kline_1m", "kline_15m", "price"]

    def add_candle(self, candle: CandleData) -> None:
        """Add a closed candle to the buffer.

        Args:
            candle: Closed candle data
        """
        if candle.interval == "1m":
            self._candles_1m.append(candle)
        elif candle.interval == "15m":
            self._candles_15m.append(candle)

    def _calculate_ema(self, candles: list[CandleData], period: int) -> Optional[float]:
        """Calculate EMA from candle closes.

        Args:
            candles: List of candles
            period: EMA period

        Returns:
            EMA value or None if insufficient data
        """
        if len(candles) < period:
            return None

        closes = [c.close for c in candles[-period * 2 :]]  # Use more data for accuracy
        if len(closes) < period:
            return None

        df = pd.DataFrame({"close": closes})
        ema = df["close"].ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])

    def _calculate_rsi(self, candles: list[CandleData], period: int = 14) -> Optional[float]:
        """Calculate RSI from candle closes.

        Args:
            candles: List of candles
            period: RSI period (default: 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(candles) < period + 1:
            return None

        closes = [c.close for c in candles[-(period + 10) :]]  # Extra for accuracy
        if len(closes) < period + 1:
            return None

        df = pd.DataFrame({"close": closes})
        delta = df["close"].diff().dropna()

        if len(delta) < period:
            return None

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Use simple moving average for first calculation, then EWM
        first_avg_gain = gain.iloc[:period].mean()
        first_avg_loss = loss.iloc[:period].mean()

        if first_avg_loss == 0:
            return 100.0 if first_avg_gain > 0 else 50.0

        # Calculate RSI using Wilder's smoothing
        avg_gain = first_avg_gain
        avg_loss = first_avg_loss

        for i in range(period, len(gain)):
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _check_rsi_filter(
        self, candles_1m: list[CandleData], direction: str
    ) -> tuple[bool, float]:
        """Check RSI filter for entry.

        Args:
            candles_1m: List of 1m candles
            direction: LONG or SHORT

        Returns:
            Tuple of (passes_filter, rsi_value)
        """
        if not self.rsi_enabled:
            return True, 50.0  # Neutral RSI if disabled

        rsi = self._calculate_rsi(candles_1m, self.RSI_PERIOD)
        if rsi is None:
            logger.debug("Insufficient data for RSI calculation")
            return False, 0.0

        if direction == "LONG":
            # Long: RSI should be below oversold threshold
            passes = rsi < self.RSI_OVERSOLD
            if not passes:
                logger.debug(
                    "RSI filter blocked LONG",
                    rsi=rsi,
                    threshold=self.RSI_OVERSOLD,
                )
        else:
            # Short: RSI should be above overbought threshold
            passes = rsi > self.RSI_OVERBOUGHT
            if not passes:
                logger.debug(
                    "RSI filter blocked SHORT",
                    rsi=rsi,
                    threshold=self.RSI_OVERBOUGHT,
                )

        return passes, rsi

    def _calculate_atr(self, candles: list[CandleData], period: int = 14) -> Optional[float]:
        """Calculate Average True Range (ATR) from candles.

        ATR measures volatility using high, low, and close prices.
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = SMA(TR, period)

        Args:
            candles: List of candles
            period: ATR period (default: 14)

        Returns:
            ATR value or None if insufficient data
        """
        if len(candles) < period + 1:
            return None

        # Calculate True Range for each candle
        true_ranges: list[float] = []

        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        # Calculate ATR using simple moving average of last 'period' true ranges
        recent_tr = true_ranges[-period:]
        atr = sum(recent_tr) / len(recent_tr)

        return atr

    def _get_trend(self, candles_15m: list[CandleData]) -> Optional[Literal["UP", "DOWN"]]:
        """Determine trend from 15m candles using EMA(20).

        Args:
            candles_15m: List of 15m candles

        Returns:
            "UP" if price > EMA, "DOWN" if price < EMA, None if insufficient data
        """
        if len(candles_15m) < self.EMA_PERIOD:
            logger.debug("Insufficient 15m candles for EMA", count=len(candles_15m))
            return None

        ema = self._calculate_ema(candles_15m, self.EMA_PERIOD)
        if ema is None:
            return None

        current_price = candles_15m[-1].close

        if current_price > ema:
            return "UP"
        elif current_price < ema:
            return "DOWN"
        return None

    def _calculate_wick_ratio(
        self, candle: CandleData
    ) -> tuple[float, float, Literal["LOWER", "UPPER", "NONE"]]:
        """Calculate wick ratios for a candle.

        Args:
            candle: Candle to analyze

        Returns:
            (lower_wick_ratio, upper_wick_ratio, dominant_wick_type)
        """
        total_range = candle.high - candle.low
        if total_range == 0:
            return 0.0, 0.0, "NONE"

        body_top = max(candle.open, candle.close)
        body_bottom = min(candle.open, candle.close)

        lower_wick = body_bottom - candle.low
        upper_wick = candle.high - body_top

        lower_ratio = lower_wick / total_range
        upper_ratio = upper_wick / total_range

        if lower_ratio >= self.WICK_RATIO_THRESHOLD:
            return lower_ratio, upper_ratio, "LOWER"
        elif upper_ratio >= self.WICK_RATIO_THRESHOLD:
            return lower_ratio, upper_ratio, "UPPER"
        return lower_ratio, upper_ratio, "NONE"

    def _check_volume_spike(self, candles_1m: list[CandleData]) -> bool:
        """Check if current volume is 3x average.

        Args:
            candles_1m: List of 1m candles

        Returns:
            True if volume spike detected
        """
        if len(candles_1m) < 21:  # Need 20 for average + 1 current
            return False

        # Calculate average of last 20 (excluding current)
        volumes = [c.volume for c in candles_1m[-21:-1]]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        if avg_volume == 0:
            return False

        current_volume = candles_1m[-1].volume
        ratio = current_volume / avg_volume

        is_spike = ratio >= self.VOLUME_MULTIPLIER

        if is_spike:
            logger.debug(
                "Volume spike detected",
                current=current_volume,
                avg=avg_volume,
                ratio=ratio,
            )

        return is_spike

    def _get_volume_exit_threshold(self, candles_1m: list[CandleData]) -> float:
        """Get volume threshold for exit signal (2x average).

        Args:
            candles_1m: List of 1m candles

        Returns:
            Volume threshold for exit
        """
        if len(candles_1m) < 20:
            return float("inf")

        volumes = [c.volume for c in candles_1m[-20:]]
        avg_volume = sum(volumes) / len(volumes)
        return avg_volume * 2.0

    async def generate(self, market_state: dict) -> Optional[Signal]:
        """Generate wick reversal signal.

        Args:
            market_state: Current market state containing:
                - kline_1m: Latest 1m kline data
                - kline_15m: Latest 15m kline data
                - candles_1m: List of recent 1m candles (optional)
                - candles_15m: List of recent 15m candles (optional)
                - price: Current price

        Returns:
            Signal if conditions are met, None otherwise
        """
        # Get current price
        current_price = market_state.get("price")
        if not current_price:
            return None

        # Update candle buffers if provided
        if "candles_1m" in market_state:
            for candle_data in market_state["candles_1m"]:
                if isinstance(candle_data, dict):
                    candle = CandleData(
                        timestamp=candle_data.get("timestamp", datetime.now(timezone.utc)),
                        open=candle_data["open"],
                        high=candle_data["high"],
                        low=candle_data["low"],
                        close=candle_data["close"],
                        volume=candle_data["volume"],
                        interval="1m",
                    )
                    self._candles_1m.append(candle)
                elif isinstance(candle_data, CandleData):
                    self._candles_1m.append(candle_data)

        if "candles_15m" in market_state:
            for candle_data in market_state["candles_15m"]:
                if isinstance(candle_data, dict):
                    candle = CandleData(
                        timestamp=candle_data.get("timestamp", datetime.now(timezone.utc)),
                        open=candle_data["open"],
                        high=candle_data["high"],
                        low=candle_data["low"],
                        close=candle_data["close"],
                        volume=candle_data["volume"],
                        interval="15m",
                    )
                    self._candles_15m.append(candle)
                elif isinstance(candle_data, CandleData):
                    self._candles_15m.append(candle_data)

        # Handle single kline updates
        kline_1m = market_state.get("kline_1m")
        if kline_1m and isinstance(kline_1m, dict) and kline_1m.get("is_closed"):
            candle = CandleData(
                timestamp=kline_1m.get("timestamp", datetime.now(timezone.utc)),
                open=kline_1m["open"],
                high=kline_1m["high"],
                low=kline_1m["low"],
                close=kline_1m["close"],
                volume=kline_1m["volume"],
                interval="1m",
            )
            self._candles_1m.append(candle)

        kline_15m = market_state.get("kline_15m")
        if kline_15m and isinstance(kline_15m, dict) and kline_15m.get("is_closed"):
            candle = CandleData(
                timestamp=kline_15m.get("timestamp", datetime.now(timezone.utc)),
                open=kline_15m["open"],
                high=kline_15m["high"],
                low=kline_15m["low"],
                close=kline_15m["close"],
                volume=kline_15m["volume"],
                interval="15m",
            )
            self._candles_15m.append(candle)

        # Check minimum data requirements
        candles_1m = list(self._candles_1m)
        candles_15m = list(self._candles_15m)

        if len(candles_1m) < 21 or len(candles_15m) < self.EMA_PERIOD:
            logger.debug(
                "Insufficient candle data",
                candles_1m=len(candles_1m),
                candles_15m=len(candles_15m),
            )
            return None

        # 1. Check 15m trend
        trend = self._get_trend(candles_15m)
        if trend is None:
            logger.debug("Unable to determine trend")
            return None

        # 2. Check volume spike on latest 1m candle
        if not self._check_volume_spike(candles_1m):
            return None

        # 3. Check wick pattern on latest 1m candle
        latest_candle = candles_1m[-1]
        lower_ratio, upper_ratio, wick_type = self._calculate_wick_ratio(latest_candle)

        if wick_type == "NONE":
            logger.debug(
                "No significant wick detected",
                lower_ratio=lower_ratio,
                upper_ratio=upper_ratio,
            )
            return None

        # 4. Match trend with wick direction
        direction: Optional[Literal["LONG", "SHORT"]] = None

        if trend == "UP" and wick_type == "LOWER":
            # 상승 추세 + 아래꼬리 → 롱
            direction = "LONG"
        elif trend == "DOWN" and wick_type == "UPPER":
            # 하락 추세 + 위꼬리 → 숏
            direction = "SHORT"
        else:
            logger.debug(
                "Trend-wick direction mismatch",
                trend=trend,
                wick_type=wick_type,
            )
            return None

        # 5. Check RSI filter
        rsi_passes, rsi_value = self._check_rsi_filter(candles_1m, direction)
        if not rsi_passes:
            logger.debug(
                "RSI filter rejected signal",
                direction=direction,
                rsi=rsi_value,
            )
            return None

        # 6. Calculate entry/exit prices
        entry_price = current_price

        # Calculate ATR for dynamic SL/TP
        atr = self._calculate_atr(candles_1m, self.ATR_PERIOD) if self.atr_enabled else None

        if direction == "LONG":
            if atr and self.atr_enabled:
                # ATR 기반 동적 SL/TP
                stop_loss = entry_price - (atr * self.ATR_SL_MULTIPLIER)
                take_profit = entry_price + (atr * self.ATR_TP_MULTIPLIER)
            else:
                # 꼬리 기반 고정 SL/TP (fallback)
                stop_loss = latest_candle.low * (1 - self.STOP_LOSS_BUFFER)
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2)
        else:
            if atr and self.atr_enabled:
                # ATR 기반 동적 SL/TP
                stop_loss = entry_price + (atr * self.ATR_SL_MULTIPLIER)
                take_profit = entry_price - (atr * self.ATR_TP_MULTIPLIER)
            else:
                # 꼬리 기반 고정 SL/TP (fallback)
                stop_loss = latest_candle.high * (1 + self.STOP_LOSS_BUFFER)
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)

        # Calculate confidence based on wick strength and volume
        wick_strength = max(lower_ratio, upper_ratio)
        volume_ratio = latest_candle.volume / (
            sum(c.volume for c in candles_1m[-21:-1]) / 20
        )
        confidence = min(0.9, 0.5 + (wick_strength * 0.3) + (min(volume_ratio / 10, 0.1)))

        # Create signal context for position management
        self._last_signal_context = WickSignalContext(
            wick_low=latest_candle.low,
            wick_high=latest_candle.high,
            entry_time=datetime.now(timezone.utc),
            volume_threshold=self._get_volume_exit_threshold(candles_1m),
        )

        # Build reason string
        atr_info = f", ATR ${atr:.2f}" if atr else ""
        reason = (
            f"Wick reversal: {trend} trend + "
            f"{'lower' if wick_type == 'LOWER' else 'upper'} wick "
            f"({wick_strength:.1%}), volume {volume_ratio:.1f}x avg, "
            f"RSI {rsi_value:.1f}{atr_info}"
        )

        signal = Signal(
            type="SCALP",
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            "Wick reversal signal generated",
            direction=direction,
            trend=trend,
            wick_type=wick_type,
            wick_ratio=wick_strength,
            volume_ratio=volume_ratio,
            rsi=rsi_value,
            atr=atr,
            confidence=confidence,
            entry=entry_price,
            sl=stop_loss,
            tp=take_profit,
        )

        return signal

    def get_signal_context(self) -> Optional[WickSignalContext]:
        """Get the last signal context for position management.

        Returns:
            Last signal context or None
        """
        return self._last_signal_context

    def should_exit_on_volume(self, current_volume: float, min_holding_seconds: int = 30) -> bool:
        """Check if should exit based on volume spike.

        Args:
            current_volume: Current candle volume
            min_holding_seconds: Minimum seconds to hold before volume exit (default: 30)

        Returns:
            True if should exit
        """
        if self._last_signal_context is None:
            return False

        # Don't trigger volume exit too early (avoid exiting on entry candle)
        time_held = (datetime.now(timezone.utc) - self._last_signal_context.entry_time).total_seconds()
        if time_held < min_holding_seconds:
            return False

        return current_volume >= self._last_signal_context.volume_threshold

    def should_exit_on_time(self, max_minutes: int = 5) -> bool:
        """Check if should exit based on time.

        Args:
            max_minutes: Maximum holding time in minutes

        Returns:
            True if should exit
        """
        if self._last_signal_context is None:
            return False

        elapsed = datetime.now(timezone.utc) - self._last_signal_context.entry_time
        return elapsed.total_seconds() >= max_minutes * 60

    def clear_buffers(self) -> None:
        """Clear all candle buffers."""
        self._candles_1m.clear()
        self._candles_15m.clear()
        self._last_signal_context = None
