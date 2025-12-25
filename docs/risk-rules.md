# Risk Management Rules

## 개요

리스크 관리는 트레이딩 시스템의 핵심입니다. 모든 규칙은 **강제적**이며, LLM이나 다른 컴포넌트가 우회할 수 없습니다.

## 포지션 사이징

### 단타 (SCALP)

| 항목 | 값 | 설명 |
|------|-----|------|
| 최대 포지션 | 계좌 30% | 마진 기준 |
| 레버리지 범위 | 20~40배 | 신뢰도에 따라 조정 |
| 1회 최대 손실 | 계좌 1% | 손절 시 손실 |

**레버리지 조정:**
```python
def calculate_leverage(confidence: float) -> int:
    # 신뢰도 0.5 → 20배, 신뢰도 1.0 → 40배
    base_leverage = 20
    max_additional = 20
    return int(base_leverage + (confidence * max_additional))
```

### 스윙 (SWING)

| 항목 | 값 | 설명 |
|------|-----|------|
| 최대 포지션 | 계좌 50% | 마진 기준 |
| 레버리지 범위 | 5~10배 | 신뢰도에 따라 조정 |
| 1회 최대 손실 | 계좌 3% | 손절 시 손실 |

### 총 노출 한도

```
총 노출 = 단타 포지션 + 스윙 포지션 <= 계좌 80%
```

---

## 손절/익절

### 단타

| 항목 | 값 |
|------|-----|
| 손절 | 진입가 대비 1.5% |
| 익절 | 진입가 대비 3~5% |
| 트레일링 | 없음 |

### 스윙

| 항목 | 값 |
|------|-----|
| 손절 | 진입가 대비 5% |
| 익절 | 진입가 대비 15~25% |
| 트레일링 | 고점 대비 7% 이탈 시 청산 |

**트레일링 스탑 로직:**
```python
class TrailingStop:
    def __init__(self, entry_price: float, direction: str, trail_pct: float = 0.07):
        self.entry_price = entry_price
        self.direction = direction
        self.trail_pct = trail_pct
        self.highest = entry_price if direction == "LONG" else float('inf')
        self.lowest = entry_price if direction == "SHORT" else 0
    
    def update(self, current_price: float) -> bool:
        """가격 업데이트 후 청산 여부 반환"""
        if self.direction == "LONG":
            self.highest = max(self.highest, current_price)
            trail_price = self.highest * (1 - self.trail_pct)
            return current_price <= trail_price
        else:
            self.lowest = min(self.lowest, current_price)
            trail_price = self.lowest * (1 + self.trail_pct)
            return current_price >= trail_price
```

---

## 자동 보호 규칙

### 1. 일일 손실 한도

```python
DAILY_LOSS_LIMIT = 0.10  # 10%

class DailyLossProtection:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.daily_pnl = 0.0
    
    def record_trade(self, pnl: float):
        self.daily_pnl += pnl
    
    def is_blocked(self) -> bool:
        return self.daily_pnl <= -self.initial_balance * DAILY_LOSS_LIMIT
    
    def reset_daily(self):
        """매일 UTC 00:00에 리셋"""
        self.daily_pnl = 0.0
```

**트리거:** 당일 누적 손실 >= 계좌의 10%
**액션:** 익일 UTC 00:00까지 모든 신규 진입 금지

### 2. 연속 손절 쿨다운

```python
CONSECUTIVE_LOSS_LIMIT = 3
COOLDOWN_HOURS = 1

class ConsecutiveLossProtection:
    def __init__(self):
        self.consecutive_losses = 0
        self.cooldown_until = None
    
    def record_trade(self, pnl: float):
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                self.cooldown_until = datetime.now() + timedelta(hours=COOLDOWN_HOURS)
        else:
            self.consecutive_losses = 0
    
    def is_blocked(self) -> bool:
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return True
        if datetime.now() >= self.cooldown_until:
            self.cooldown_until = None
            self.consecutive_losses = 0
        return False
```

**트리거:** 연속 3회 손절
**액션:** 1시간 쿨다운 (신규 진입 금지)

### 3. 이벤트 블랙아웃

```python
EVENT_BLACKOUT_MINUTES = 30

HIGH_IMPACT_EVENTS = [
    "FOMC",
    "CPI",
    "NFP",
    "PPI",
    "GDP",
]

class EventBlackout:
    def __init__(self, calendar: EconomicCalendar):
        self.calendar = calendar
    
    async def is_blocked(self) -> bool:
        upcoming = await self.calendar.get_upcoming(minutes=EVENT_BLACKOUT_MINUTES)
        high_impact = [e for e in upcoming if e.name in HIGH_IMPACT_EVENTS]
        return len(high_impact) > 0
    
    def get_block_reason(self) -> str | None:
        # 블랙아웃 사유 반환
        ...
```

**트리거:** 주요 경제지표 발표 30분 전
**액션:** 신규 진입 금지 (기존 포지션 유지)

### 4. LLM 확신도 임계값

```python
LLM_CONFIDENCE_THRESHOLD = 0.60
SIZE_REDUCTION_FACTOR = 0.50

class LLMConfidenceFilter:
    def filter(self, decision: LLMDecision, signal: Signal) -> tuple[bool, float]:
        """
        Returns:
            (execute, size_multiplier)
        """
        if decision.confidence < LLM_CONFIDENCE_THRESHOLD:
            # 확신도 낮으면 포지션 50% 축소
            return True, SIZE_REDUCTION_FACTOR
        return True, 1.0
```

**트리거:** LLM 확신도 < 60%
**액션:** 포지션 크기 50% 축소

---

## 포지션 계산 공식

### 켈리 기준 간소화

```python
def calculate_position_size(
    balance: float,
    risk_per_trade: float,  # 1회 손실 감수 비율 (예: 0.01 = 1%)
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    포지션 크기 계산
    
    Args:
        balance: 계좌 잔고
        risk_per_trade: 1회 거래 손실 감수 비율
        entry_price: 진입가
        stop_loss_price: 손절가
    
    Returns:
        포지션 크기 (BTC 수량)
    """
    risk_amount = balance * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price) / entry_price
    position_value = risk_amount / price_risk
    position_size = position_value / entry_price
    return position_size
```

**예시:**
```
잔고: $10,000
리스크: 1% ($100)
진입가: $50,000
손절가: $49,250 (1.5% 하락)

계산:
price_risk = |50000 - 49250| / 50000 = 0.015
position_value = 100 / 0.015 = $6,666.67
position_size = 6666.67 / 50000 = 0.133 BTC
```

---

## 리스크 체크 플로우

```python
async def check_risk(signal: Signal, account: Account) -> RiskCheck:
    warnings = []
    
    # 1. 일일 손실 한도
    if daily_loss_protection.is_blocked():
        return RiskCheck(approved=False, blocked_reason="일일 손실 한도 도달")
    
    # 2. 연속 손절 쿨다운
    if consecutive_loss_protection.is_blocked():
        remaining = (consecutive_loss_protection.cooldown_until - datetime.now()).seconds // 60
        return RiskCheck(approved=False, blocked_reason=f"쿨다운 중: {remaining}분 남음")
    
    # 3. 이벤트 블랙아웃
    if await event_blackout.is_blocked():
        reason = event_blackout.get_block_reason()
        return RiskCheck(approved=False, blocked_reason=f"이벤트 블랙아웃: {reason}")
    
    # 4. 총 노출 한도
    current_exposure = calculate_current_exposure(account)
    if current_exposure >= MAX_EXPOSURE:
        return RiskCheck(approved=False, blocked_reason="총 노출 한도 도달")
    
    # 5. 포지션 사이징
    max_size = calculate_max_position_size(signal, account, current_exposure)
    leverage = calculate_leverage(signal)
    
    return RiskCheck(
        approved=True,
        max_position_size=max_size,
        adjusted_leverage=leverage,
        warnings=warnings,
        blocked_reason=None
    )
```

---

## 모니터링 지표

| 지표 | 경고 임계값 | 위험 임계값 |
|------|-------------|-------------|
| 일일 손실 | 5% | 10% |
| 총 노출 | 60% | 80% |
| 연속 손절 | 2회 | 3회 |
| 미실현 손실 | 3% | 5% |
