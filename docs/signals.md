# Signal Definitions

## 시그널 구조

```python
@dataclass
class Signal:
    type: Literal["SCALP", "SWING"]
    direction: Literal["LONG", "SHORT"]
    confidence: float  # 0.0 ~ 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    timestamp: datetime
```

## 단타 시그널 (SCALP)

### 1. 청산 캐스케이드 (LiquidationCascade)

| 항목 | 값 |
|------|-----|
| 트리거 | 5분 내 $50M+ 청산 |
| 방향 | 역추세 (롱청산 → 롱진입) |
| 신뢰도 | 0.5 ~ 1.0 (청산량 비례) |
| 손절 | 1.5% |
| 익절 | 3~5% |

**로직:**
```
IF 5분간_롱_청산액 >= $50M:
    SIGNAL = LONG (바닥 반등 기대)
    
IF 5분간_숏_청산액 >= $50M:
    SIGNAL = SHORT (천장 하락 기대)
```

**예시:**
- 롱 청산 $80M 발생 → 롱 진입 시그널 (confidence: 0.8)
- 숏 청산 $60M 발생 → 숏 진입 시그널 (confidence: 0.6)

---

### 2. 펀딩비 극단값 (FundingRateExtreme)

| 항목 | 값 |
|------|-----|
| 트리거 | 펀딩비 ±0.08% 이상 |
| 방향 | 역추세 |
| 신뢰도 | 0.5 ~ 1.0 (펀딩비 크기 비례) |
| 손절 | 1.5% |
| 익절 | 3~5% |

**로직:**
```
IF 펀딩비 >= +0.08%:
    SIGNAL = SHORT (롱 과열 → 하락 기대)
    
IF 펀딩비 <= -0.08%:
    SIGNAL = LONG (숏 과열 → 상승 기대)
```

**펀딩비 해석:**
- 양수: 롱이 숏에게 지불 = 롱 포지션 많음
- 음수: 숏이 롱에게 지불 = 숏 포지션 많음

---

### 3. 거래량 돌파 (VolumeBreakout)

| 항목 | 값 |
|------|-----|
| 트리거 | 거래량 평균 3배 + 가격 돌파 |
| 방향 | 추세 추종 |
| 신뢰도 | 0.7 |
| 손절 | 1.5% |
| 익절 | 3~5% |

**로직:**
```
IF 현재_거래량 >= 20봉_평균 * 3:
    IF 종가 > 20봉_최고가:
        SIGNAL = LONG (상승 돌파)
    IF 종가 < 20봉_최저가:
        SIGNAL = SHORT (하락 돌파)
```

---

## 스윙 시그널 (SWING)

### 1. EMA + RSI (EmaRsiSwing)

| 항목 | 값 |
|------|-----|
| 트리거 | EMA 정배열/역배열 + RSI 40~60 |
| 방향 | 추세 추종 |
| 신뢰도 | 0.65 |
| 손절 | 5% |
| 익절 | 15~25% |
| 트레일링 | 고점 대비 7% |

**로직:**
```
롱 조건:
    EMA(7) > EMA(25) > EMA(99)  # 정배열
    AND 40 <= RSI(14) <= 60     # 과열 아님
    
숏 조건:
    EMA(7) < EMA(25) < EMA(99)  # 역배열
    AND 40 <= RSI(14) <= 60     # 과열 아님
```

**타임프레임:** 4시간봉

---

### 2. Fear & Greed 필터

시그널 자체를 생성하지 않고, 다른 시그널을 필터링/강화합니다.

| Fear & Greed | 롱 시그널 | 숏 시그널 |
|--------------|-----------|-----------|
| 0~25 (극단적 공포) | 신뢰도 +20% | 거부 |
| 26~40 | 신뢰도 +10% | 통과 |
| 41~59 | 통과 | 통과 |
| 60~74 | 통과 | 신뢰도 +10% |
| 75~100 (극단적 탐욕) | 거부 | 신뢰도 +20% |

---

## 시그널 조합

### 단타 시그널 우선순위

1. 청산 캐스케이드 (가장 급한 기회)
2. 펀딩비 극단값
3. 거래량 돌파

**동시 발생 시:** 가장 높은 신뢰도 시그널 선택

### 단타 + 스윙 조합

| 단타 방향 | 스윙 방향 | 액션 |
|-----------|-----------|------|
| LONG | LONG | 둘 다 진입 가능 |
| SHORT | SHORT | 둘 다 진입 가능 |
| LONG | SHORT | 단타만 진입 |
| SHORT | LONG | 단타만 진입 |

**원칙:** 단타와 스윙이 반대 방향이면 단타만 진입 (리스크 분산)

---

## 시그널 무효화 조건

### 공통

- LLM 확신도 < 60%
- 리스크 한도 초과
- 이벤트 블랙아웃 기간

### 단타 전용

- 5분 내 동일 방향 시그널 이미 실행됨
- 스프레드 > 0.1%

### 스윙 전용

- 기존 스윙 포지션 보유 중 (같은 방향)
- Fear & Greed 극단값으로 거부됨

---

## 시그널 로깅 형식

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "signal": {
        "type": "SCALP",
        "direction": "LONG",
        "confidence": 0.75,
        "entry_price": 42500.00,
        "stop_loss": 41862.50,
        "take_profit": 44200.00,
        "reason": "롱 청산 캐스케이드: $65,000,000"
    },
    "context": {
        "funding_rate": 0.0003,
        "fear_greed": 35,
        "recent_news_sentiment": "neutral"
    },
    "decision": {
        "execute": true,
        "llm_confidence": 0.72,
        "adjusted_size": 0.8,
        "llm_reason": "뉴스 중립, 매크로 지지"
    },
    "execution": {
        "executed": true,
        "actual_entry": 42510.00,
        "size": 0.024,
        "leverage": 25
    }
}
```
