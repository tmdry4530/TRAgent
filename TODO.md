# Development TODO

## Phase 1: MVP

### 인프라 설정
- [ ] 프로젝트 초기화 (`/setup` 실행)
- [ ] .env 파일 설정
- [ ] Redis 설치 및 연결 테스트
- [ ] 바이낸스 테스트넷 API 키 발급

### 데이터 수집기
- [x] `src/collectors/binance.py` - WebSocket 연결
  - [x] Kline 스트림
  - [x] Orderbook 스트림
  - [x] Trade 스트림
  - [x] Liquidation 스트림
  - [x] 재연결 로직
- [x] `src/collectors/binance.py` - REST 폴링
  - [x] 펀딩비
  - [x] 미결제약정
  - [x] 롱숏비율
- [ ] Redis 저장 로직

### 시그널 생성기
- [x] `src/signals/scalp.py`
  - [x] LiquidationCascadeSignal
  - [x] FundingRateSignal
  - [x] VolumeBreakoutSignal
- [x] `src/signals/swing.py`
  - [x] EmaRsiSwingSignal
  - [x] FearGreedFilter

### 리스크 관리
- [x] `src/risk/manager.py`
  - [x] 포지션 사이징
  - [x] 일일 손실 한도
  - [x] 연속 손절 쿨다운
  - [x] 이벤트 블랙아웃

### 백테스팅
- [x] `src/backtest/download_data.py` - 과거 데이터 다운로드
- [x] `src/backtest/engine.py` - 백테스트 엔진
- [ ] 단타 전략 백테스트
- [ ] 스윙 전략 백테스트

### 테스트
- [x] 시그널 단위 테스트
- [x] 리스크 관리 테스트
- [x] 백테스트 검증 테스트

---

## Phase 2: LLM 통합 ✅ (완료)

### Claude API 연동
- [x] `src/brain/context_filter.py`
  - [x] 프롬프트 템플릿 (EVALUATION_PROMPT)
  - [x] API 호출 로직 (Anthropic SDK)
  - [x] 응답 파싱 (JSON 추출, 마크다운 처리)
  - [x] TTL 캐싱 (5분, 100개)

### 매크로 데이터
- [x] `src/collectors/macro.py`
  - [x] FRED API (경제 캘린더)
  - [x] Yahoo Finance (DXY, SPX, NDX, US10Y)
  - [x] Fear & Greed Index (Alternative.me)
  - [x] 경제 캘린더 (이벤트 필터링)

### 뉴스 수집
- [x] `src/collectors/news.py`
  - [x] CryptoPanic API (뉴스 + 감성분석)
  - [x] Whale Alert API (대량 이체 모니터링) - 유료 구독 필요
  - [x] 거래소 유입/유출 흐름 계산

### 통합 테스트
- [x] 전체 파이프라인 테스트 (148 tests passing)
- [x] LLM 판단 검증 (tests/brain/test_context_filter.py)

---

## Phase 3: 고도화

### 추가 기능
- [ ] Coinglass 연동
- [ ] 텔레그램 알림
- [ ] 성과 대시보드 (웹)
- [ ] 멀티 코인 지원

### 운영
- [ ] VPS 배포 스크립트
- [ ] 모니터링 설정
- [ ] 로그 분석 도구

### 최적화
- [ ] 파라미터 튜닝
- [ ] 슬리피지 최적화
- [ ] API 호출 최적화

---

## 완료된 항목

*(완료 시 위에서 여기로 이동)*

---

## 메모

- 테스트넷에서 충분히 테스트 후 메인넷 전환
- 백테스트 결과 기록 유지
- 매주 성과 리뷰
