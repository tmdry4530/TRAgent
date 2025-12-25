# Trading Patterns Skill

트레이딩 시그널 및 패턴 구현 시 참조하는 스킬입니다.

## 사용 시점

- 시그널 로직 구현
- 기술적 지표 계산
- 진입/청산 조건 정의

## 단타 시그널 패턴

### 1. 청산 캐스케이드 반대매매

```python
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta

@dataclass
class LiquidationEvent:
    timestamp: datetime
    side: str  # "LONG" or "SHORT"
    quantity: float
    price: float

class LiquidationCascadeSignal:
    """대량 청산 발생 시 반대 포지션 시그널"""
    
    def __init__(
        self,
        threshold_usd: float = 50_000_000,  # $50M
        window_seconds: int = 300  # 5분
    ):
        self.threshold = threshold_usd
        self.window = timedelta(seconds=window_seconds)
        self.events: deque[LiquidationEvent] = deque()
    
    def add_event(self, event: LiquidationEvent):
        self.events.append(event)
        self._cleanup()
    
    def _cleanup(self):
        """윈도우 밖 이벤트 제거"""
        cutoff = datetime.now() - self.window
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()
    
    def check_signal(self) -> Signal | None:
        """시그널 체크"""
        long_liquidated = sum(
            e.quantity * e.price 
            for e in self.events 
            if e.side == "LONG"
        )
        short_liquidated = sum(
            e.quantity * e.price 
            for e in self.events 
            if e.side == "SHORT"
        )
        
        # 롱 청산 > 임계값 → 롱 진입 (바닥 예상)
        if long_liquidated >= self.threshold:
            return Signal(
                type="SCALP",
                direction="LONG",
                confidence=min(long_liquidated / self.threshold, 2.0) / 2,
                reason=f"롱 청산 캐스케이드: ${long_liquidated:,.0f}"
            )
        
        # 숏 청산 > 임계값 → 숏 진입 (천장 예상)
        if short_liquidated >= self.threshold:
            return Signal(
                type="SCALP",
                direction="SHORT",
                confidence=min(short_liquidated / self.threshold, 2.0) / 2,
                reason=f"숏 청산 캐스케이드: ${short_liquidated:,.0f}"
            )
        
        return None
```

### 2. 펀딩비 극단값

```python
class FundingRateSignal:
    """펀딩비 극단값 역추세 시그널"""
    
    def __init__(self, threshold: float = 0.0008):  # 0.08%
        self.threshold = threshold
    
    def check_signal(self, funding_rate: float) -> Signal | None:
        # 펀딩비 높음 = 롱 과열 → 숏 시그널
        if funding_rate >= self.threshold:
            return Signal(
                type="SCALP",
                direction="SHORT",
                confidence=min(abs(funding_rate) / self.threshold, 2.0) / 2,
                reason=f"펀딩비 과열 (롱): {funding_rate:.4%}"
            )
        
        # 펀딩비 낮음 = 숏 과열 → 롱 시그널
        if funding_rate <= -self.threshold:
            return Signal(
                type="SCALP",
                direction="LONG",
                confidence=min(abs(funding_rate) / self.threshold, 2.0) / 2,
                reason=f"펀딩비 과열 (숏): {funding_rate:.4%}"
            )
        
        return None
```

### 3. 거래량 돌파

```python
class VolumeBreakoutSignal:
    """거래량 급증 + 가격 돌파 시그널"""
    
    def __init__(
        self,
        volume_multiplier: float = 3.0,
        lookback: int = 20
    ):
        self.multiplier = volume_multiplier
        self.lookback = lookback
    
    def check_signal(self, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.lookback + 1:
            return None
        
        current = df.iloc[-1]
        avg_volume = df['volume'].iloc[-self.lookback-1:-1].mean()
        
        # 거래량 급증 체크
        if current['volume'] < avg_volume * self.multiplier:
            return None
        
        # 가격 돌파 방향
        prev_high = df['high'].iloc[-self.lookback-1:-1].max()
        prev_low = df['low'].iloc[-self.lookback-1:-1].min()
        
        if current['close'] > prev_high:
            return Signal(
                type="SCALP",
                direction="LONG",
                confidence=0.7,
                reason=f"거래량 돌파 (상승): {current['volume']/avg_volume:.1f}x"
            )
        
        if current['close'] < prev_low:
            return Signal(
                type="SCALP",
                direction="SHORT",
                confidence=0.7,
                reason=f"거래량 돌파 (하락): {current['volume']/avg_volume:.1f}x"
            )
        
        return None
```

## 스윙 시그널 패턴

### EMA 크로스 + RSI 필터

```python
import pandas_ta as ta

class EmaRsiSwingSignal:
    """EMA 정배열/역배열 + RSI 필터"""
    
    def __init__(
        self,
        ema_periods: list[int] = [7, 25, 99],
        rsi_period: int = 14,
        rsi_range: tuple[int, int] = (40, 60)
    ):
        self.ema_periods = ema_periods
        self.rsi_period = rsi_period
        self.rsi_range = rsi_range
    
    def check_signal(self, df: pd.DataFrame) -> Signal | None:
        # 지표 계산
        for period in self.ema_periods:
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        current = df.iloc[-1]
        
        # RSI 필터 (극단값 제외)
        if not (self.rsi_range[0] <= current['rsi'] <= self.rsi_range[1]):
            return None
        
        ema_7 = current['ema_7']
        ema_25 = current['ema_25']
        ema_99 = current['ema_99']
        
        # 정배열: 7 > 25 > 99 → 롱
        if ema_7 > ema_25 > ema_99:
            return Signal(
                type="SWING",
                direction="LONG",
                confidence=0.65,
                reason=f"EMA 정배열, RSI: {current['rsi']:.1f}"
            )
        
        # 역배열: 7 < 25 < 99 → 숏
        if ema_7 < ema_25 < ema_99:
            return Signal(
                type="SWING",
                direction="SHORT",
                confidence=0.65,
                reason=f"EMA 역배열, RSI: {current['rsi']:.1f}"
            )
        
        return None
```

### Fear & Greed 필터

```python
class FearGreedFilter:
    """Fear & Greed Index로 시그널 필터링"""
    
    def __init__(
        self,
        extreme_fear: int = 25,
        extreme_greed: int = 75
    ):
        self.extreme_fear = extreme_fear
        self.extreme_greed = extreme_greed
    
    def filter_signal(
        self, 
        signal: Signal, 
        fear_greed: int
    ) -> Signal | None:
        """시그널 필터링 및 신뢰도 조정"""
        
        # 극단적 공포에서 숏 → 거부
        if fear_greed <= self.extreme_fear and signal.direction == "SHORT":
            return None
        
        # 극단적 탐욕에서 롱 → 거부
        if fear_greed >= self.extreme_greed and signal.direction == "LONG":
            return None
        
        # 공포에서 롱 → 신뢰도 상승
        if fear_greed < 40 and signal.direction == "LONG":
            signal.confidence = min(signal.confidence * 1.2, 1.0)
            signal.reason += f" (Fear: {fear_greed})"
        
        # 탐욕에서 숏 → 신뢰도 상승
        if fear_greed > 60 and signal.direction == "SHORT":
            signal.confidence = min(signal.confidence * 1.2, 1.0)
            signal.reason += f" (Greed: {fear_greed})"
        
        return signal
```

## 손절/익절 계산

```python
def calculate_stop_take(
    entry_price: float,
    direction: str,
    signal_type: str
) -> tuple[float, float]:
    """손절/익절 가격 계산"""
    
    if signal_type == "SCALP":
        stop_pct = 0.015   # 1.5%
        take_pct = 0.04    # 4%
    else:  # SWING
        stop_pct = 0.05    # 5%
        take_pct = 0.20    # 20%
    
    if direction == "LONG":
        stop_loss = entry_price * (1 - stop_pct)
        take_profit = entry_price * (1 + take_pct)
    else:
        stop_loss = entry_price * (1 + stop_pct)
        take_profit = entry_price * (1 - take_pct)
    
    return stop_loss, take_profit
```
