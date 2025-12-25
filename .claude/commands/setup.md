---
description: 프로젝트 초기 설정 - 의존성 설치, 환경 설정, 디렉토리 생성
---

프로젝트 초기 설정을 진행합니다.

## 1. 프로젝트 구조 생성

다음 디렉토리 구조를 생성하세요:

```
trading-agent/
├── src/
│   ├── collectors/
│   ├── signals/
│   ├── brain/
│   ├── risk/
│   ├── executor/
│   ├── backtest/
│   └── utils/
├── tests/
├── data/
├── config/
└── logs/
```

## 2. 의존성 설치

requirements.txt를 생성하고 설치하세요:

```txt
# Core
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Async
aiohttp>=3.9.0
websockets>=12.0
asyncio>=3.4.3

# Data
pandas>=2.1.0
numpy>=1.26.0
pandas-ta>=0.3.14b0
redis>=5.0.0

# Trading
python-binance>=1.0.19
ccxt>=4.2.0

# LLM
anthropic>=0.18.0

# Utils
structlog>=24.1.0
tenacity>=8.2.0
cachetools>=5.3.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0

# Dev
ruff>=0.3.0
mypy>=1.8.0
bandit>=1.7.0
```

```bash
pip install -r requirements.txt
```

## 3. 환경 설정

.env.example 파일을 생성하고, 사용자에게 .env 파일 생성을 안내하세요:

```bash
# Binance API
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
BINANCE_TESTNET=true  # 테스트넷 사용 여부

# Anthropic
ANTHROPIC_API_KEY=your_api_key

# External APIs
FRED_API_KEY=your_api_key

# Redis
REDIS_URL=redis://localhost:6379

# Environment
ENV=development
LOG_LEVEL=DEBUG
```

## 4. 설정 파일 생성

config/trading.yaml 생성:

```yaml
scalp:
  max_leverage: 40
  position_size: 0.3
  stop_loss: 0.015
  take_profit: 0.05

swing:
  max_leverage: 10
  position_size: 0.5
  stop_loss: 0.05
  take_profit: 0.25
  trailing_stop: 0.07

risk:
  daily_loss_limit: 0.1
  consecutive_loss_cooldown: 3
  event_blackout_minutes: 30
  llm_confidence_threshold: 0.6

signals:
  liquidation_threshold: 50000000  # $50M
  funding_rate_threshold: 0.0008   # 0.08%
  volume_multiplier: 3.0
  ema_periods: [7, 25, 99]
  rsi_period: 14
```

## 5. Makefile 생성

```makefile
.PHONY: dev test lint backtest deploy

dev:
	python -m src.main

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/
	mypy src/

backtest:
	python -m src.backtest.run

backtest-scalp:
	python -m src.backtest.run --strategy scalp

backtest-swing:
	python -m src.backtest.run --strategy swing

deploy:
	./scripts/deploy.sh
```

## 6. 기본 모듈 스캐폴딩

각 모듈의 __init__.py와 기본 클래스 생성:

- `src/collectors/__init__.py`
- `src/signals/__init__.py`
- `src/brain/__init__.py`
- `src/risk/__init__.py`
- `src/executor/__init__.py`
- `src/backtest/__init__.py`
- `src/utils/__init__.py`

## 7. 확인

설정 완료 후 다음을 확인하세요:
- [ ] 모든 디렉토리 생성됨
- [ ] requirements.txt 설치 완료
- [ ] .env 파일 생성 (사용자)
- [ ] config/trading.yaml 생성됨
- [ ] 기본 모듈 파일 생성됨
