---
description: 새로운 트레이딩 시그널 추가 - 템플릿 기반 시그널 클래스 생성
---

새로운 트레이딩 시그널을 추가합니다.

## 인자

$ARGUMENTS에서 시그널 정보를 파싱합니다:
- 시그널 이름
- 타입 (scalp/swing)

예: `/add-signal VolumeImbalance scalp`

## 실행 단계

### 1. 시그널 타입 확인

```python
signal_name = "$ARGUMENTS".split()[0] if "$ARGUMENTS" else None
signal_type = "$ARGUMENTS".split()[1] if len("$ARGUMENTS".split()) > 1 else "scalp"
```

### 2. 템플릿 생성

scalp 시그널이면 `src/signals/scalp.py`에 추가:

```python
class {SignalName}Signal:
    """
    {시그널 설명}
    
    Trigger: {트리거 조건}
    Direction: {방향 로직}
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        # TODO: 설정값 초기화
    
    def check_signal(self, market_state: MarketState) -> Signal | None:
        """
        시그널 체크
        
        Args:
            market_state: 현재 시장 상태
            
        Returns:
            Signal if triggered, None otherwise
        """
        # TODO: 시그널 로직 구현
        return None
```

swing 시그널이면 `src/signals/swing.py`에 추가.

### 3. 테스트 파일 생성

`tests/signals/test_{signal_name}.py`:

```python
import pytest
from src.signals.{type} import {SignalName}Signal

class Test{SignalName}Signal:
    
    def test_no_signal_when_condition_not_met(self):
        """조건 미충족 시 None 반환"""
        signal_gen = {SignalName}Signal()
        # TODO: 테스트 구현
        assert signal_gen.check_signal(mock_state) is None
    
    def test_long_signal(self):
        """롱 시그널 생성"""
        # TODO: 테스트 구현
        pass
    
    def test_short_signal(self):
        """숏 시그널 생성"""
        # TODO: 테스트 구현
        pass
```

### 4. docs/signals.md 업데이트

시그널 정의 문서에 새 시그널 추가.

### 5. 확인

- [ ] 시그널 클래스 생성됨
- [ ] 테스트 파일 생성됨
- [ ] 문서 업데이트됨
- [ ] 린트 통과

## 다음 단계

1. TODO 주석 부분 구현
2. 테스트 작성 및 실행
3. 백테스트로 검증
