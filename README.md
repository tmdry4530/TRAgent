# AI Trading Agent

바이낸스 BTC 선물 자동매매 에이전트

## 특징

- **하이브리드 구조**: 룰 기반 시그널 + LLM 컨텍스트 판단
- **이중 전략**: 단타(스캘핑) + 스윙 트레이딩
- **리스크 관리**: 자동 손절, 일일 한도, 이벤트 블랙아웃
- **실시간 데이터**: 바이낸스 WebSocket, 매크로 지표, 뉴스 피드

## 아키텍처

```
Data Collectors → Signal Generator → LLM Filter → Risk Manager → Executor
```

## 빠른 시작

### 1. 클론 및 설치

```bash
git clone https://github.com/yourname/trading-agent.git
cd trading-agent
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env 파일 편집하여 API 키 입력
```

### 3. 상태 확인

```bash
# Claude Code에서
/status
```

### 4. 백테스트

```bash
make backtest
```

### 5. 실행 (테스트넷)

```bash
BINANCE_TESTNET=true make dev
```

## Claude Code 사용법

### 서브에이전트

```
@data-collector   - 데이터 수집 관련
@signal-generator - 시그널 로직
@risk-manager     - 리스크 관리
@backtester       - 백테스팅
@code-reviewer    - 코드 리뷰
@llm-brain        - LLM 연동
```

### 슬래시 명령어

```
/setup     - 프로젝트 초기 설정
/status    - 시스템 상태 확인
/backtest  - 백테스트 실행
```

### 스킬

- `binance-api` - 바이낸스 API 레퍼런스
- `trading-patterns` - 트레이딩 패턴 구현 가이드

## 프로젝트 구조

```
trading-agent/
├── src/
│   ├── collectors/      # 데이터 수집
│   ├── signals/         # 시그널 생성
│   ├── brain/           # LLM 연동
│   ├── risk/            # 리스크 관리
│   ├── executor/        # 주문 실행
│   ├── backtest/        # 백테스팅
│   └── utils/           # 유틸리티
├── tests/               # 테스트
├── docs/                # 문서
├── config/              # 설정
├── data/                # 데이터
└── .claude/             # Claude Code 설정
    ├── agents/          # 서브에이전트
    ├── commands/        # 명령어
    ├── hooks/           # 훅
    └── skills/          # 스킬
```

## 전략

### 단타 (SCALP)

| 시그널 | 방향 | 조건 |
|--------|------|------|
| 청산 캐스케이드 | 역추세 | $50M+ 청산 |
| 펀딩비 극단값 | 역추세 | ±0.08% |
| 거래량 돌파 | 추세 | 평균 3배 |

### 스윙 (SWING)

| 조건 | 롱 | 숏 |
|------|----|----|
| EMA | 정배열 | 역배열 |
| RSI | 40~60 | 40~60 |
| Fear & Greed | < 70 | > 30 |

## 리스크 관리

| 규칙 | 값 |
|------|-----|
| 일일 손실 한도 | 10% |
| 연속 손절 쿨다운 | 3회 → 1시간 |
| 이벤트 블랙아웃 | 30분 전 |
| 단타 최대 포지션 | 30% |
| 스윙 최대 포지션 | 50% |

## 문서

- [Architecture](docs/architecture.md) - 상세 아키텍처
- [Signals](docs/signals.md) - 시그널 정의
- [Risk Rules](docs/risk-rules.md) - 리스크 규칙
- [API Specs](docs/api-specs.md) - API 스펙

## 라이선스

MIT

## 주의사항

⚠️ **이 프로젝트는 교육 목적입니다. 실제 자금으로 거래 시 손실이 발생할 수 있습니다.**

- 먼저 테스트넷에서 충분히 테스트하세요
- 백테스트 결과가 실제 성과를 보장하지 않습니다
- 감당할 수 있는 금액만 투자하세요
