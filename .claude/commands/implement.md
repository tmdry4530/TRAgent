---
description: TODO.md 기반 자동 구현 파이프라인 - 모듈별 구현, 테스트, 검증 자동화
---

# 구현 파이프라인

TODO.md의 미완료 항목을 체계적으로 구현합니다.

## 인자

- `/implement` → Phase 1 전체 순차 진행
- `/implement collectors` → collectors 모듈만
- `/implement signals` → signals 모듈만
- `/implement signals/scalp` → 단타 시그널만
- `/implement risk` → risk 모듈만
- `/implement backtest` → backtest 모듈만

## 실행 규칙

### 1. 사전 준비

매 모듈 구현 전:

```
1. docs/design.md 에서 해당 모듈 스펙 확인
2. docs/architecture.md 에서 데이터 흐름 확인
3. 관련 스킬 로드:
   - 바이낸스 관련 → .claude/skills/binance-api/SKILL.md
   - 시그널 관련 → .claude/skills/trading-patterns/SKILL.md
```

### 2. 서브에이전트 위임

모듈별 담당 에이전트:

| 모듈 | 서브에이전트 | 역할 |
|------|--------------|------|
| src/collectors/ | @data-collector | WebSocket, REST, 데이터 정규화 |
| src/signals/ | @signal-generator | 시그널 로직, 지표 계산 |
| src/risk/ | @risk-manager | 포지션 사이징, 보호 규칙 |
| src/backtest/ | @backtester | 백테스트 엔진, 성과 분석 |
| src/brain/ | @llm-brain | Claude API 연동 |
| 코드 리뷰 | @code-reviewer | 품질 검증 |

### 3. 구현 순서

각 모듈에서:

```
Step 1: 디렉토리/파일 생성
        └─ src/{module}/__init__.py
        └─ src/{module}/{파일}.py

Step 2: 기본 클래스 구조 작성
        └─ 데이터클래스, 인터페이스 정의

Step 3: 핵심 로직 구현
        └─ 서브에이전트 위임하여 구현

Step 4: 테스트 작성
        └─ tests/{module}/test_{파일}.py

Step 5: 테스트 실행
        └─ pytest tests/{module}/ -v

Step 6: 코드 리뷰
        └─ @code-reviewer로 검증

Step 7: TODO.md 업데이트
        └─ [ ] → [x] 변경
```

### 4. 모듈별 상세

#### collectors (1순위)

```
구현 순서:
1. src/utils/config.py - 설정 로더
2. src/utils/logger.py - 로깅 설정
3. src/collectors/binance.py - BinanceWebSocketCollector
   - connect(), disconnect()
   - subscribe_klines()
   - subscribe_orderbook()
   - subscribe_liquidations()
   - 재연결 로직 (exponential backoff)
4. src/collectors/binance.py - BinanceRestCollector
   - get_funding_rate()
   - get_open_interest()
   - get_long_short_ratio()

테스트:
- WebSocket 연결/해제
- 메시지 파싱
- 재연결 로직
```

#### signals (2순위)

```
구현 순서:
1. src/signals/base.py - Signal 데이터클래스, BaseSignalGenerator
2. src/signals/scalp.py
   - LiquidationCascadeSignal
   - FundingRateSignal
   - VolumeBreakoutSignal
3. src/signals/swing.py
   - EmaRsiSwingSignal
   - FearGreedFilter

테스트:
- 각 시그널 트리거 조건
- 경계값 테스트
- 시그널 유효성 검증
```

#### risk (3순위)

```
구현 순서:
1. src/risk/calculator.py - 포지션 사이징 계산
2. src/risk/manager.py - RiskManager
   - check()
   - record_trade_result()
   - 일일 손실 한도
   - 연속 손절 쿨다운
   - 이벤트 블랙아웃

테스트:
- 각 보호 규칙 단위 테스트
- 한도 경계값 테스트
```

#### backtest (4순위)

```
구현 순서:
1. src/backtest/download_data.py - 과거 데이터 다운로드
2. src/backtest/engine.py - BacktestEngine
   - run()
   - 수수료/슬리피지 적용
   - 성과 지표 계산
3. src/backtest/run.py - CLI 엔트리포인트

테스트:
- 알려진 결과와 비교
- 수수료 적용 검증
```

## 중단 조건

다음 상황에서 멈추고 사용자 확인 요청:

```
🛑 중단 조건:
- pytest 3회 연속 실패 → 원인 분석 후 재시도 여부 확인
- 외부 API 키 필요 → .env 설정 안내 후 대기
- 설계 모호 → 선택지 제시 후 결정 요청
- 의존성 설치 필요 → 설치 여부 확인
```

## 진행 상황 보고

각 Step 완료 시:

```
✅ [collectors] BinanceWebSocketCollector 구현 완료
   - 파일: src/collectors/binance.py
   - 테스트: tests/collectors/test_binance.py (5/5 passed)
   - 다음: BinanceRestCollector
```

## 시작

$ARGUMENTS 파싱하여 해당 모듈부터 시작.
인자 없으면 TODO.md의 첫 번째 미완료 항목부터 순차 진행.

---

지금 시작합니다. 먼저 TODO.md를 확인하고 진행 상황을 파악합니다.
