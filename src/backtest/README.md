# Backtest Module

백테스팅 엔진과 데이터 다운로더를 제공합니다.

## 구성 요소

### 1. BinanceDataDownloader

바이낸스 Futures API에서 과거 캔들 데이터를 다운로드합니다.

```python
from src.backtest import BinanceDataDownloader

async with BinanceDataDownloader() as downloader:
    # 1시간봉 데이터 다운로드
    df = await downloader.download_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # CSV로 저장
    await downloader.save_to_csv(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
        output_path="data/btc_1h_2024.csv",
    )
```

**지원 인터벌**: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`

### 2. BacktestEngine

이벤트 기반 백테스팅 엔진으로 실제 거래 조건을 시뮬레이션합니다.

```python
from src.backtest import BacktestEngine, run_simple_backtest

# 기본 사용법
engine = BacktestEngine(
    initial_capital=10000.0,
    commission=0.0004,  # 0.04% (Binance taker fee)
    slippage=0.0001,    # 0.01%
    position_size=0.95, # 자본의 95% 사용
)

result = engine.run(data=df, signals=signals)
result.print_summary()

# 또는 간편 함수 사용
result = run_simple_backtest(df, signals, initial_capital=10000)
```

## 주요 기능

### 현실적인 시뮬레이션

- **슬리피지**: 진입/청산 시 0.01% 기본 슬리피지 적용
- **수수료**: 바이낸스 테이커 수수료 0.04% 적용
- **Stop Loss/Take Profit**: 캔들의 high/low를 체크하여 정확한 청산 시뮬레이션
- **포지션 관리**: LONG/SHORT 동시 포지션 방지 (같은 방향만 허용)

### 성과 지표

```python
@dataclass
class BacktestResult:
    total_trades: int        # 총 거래 수
    winning_trades: int      # 승리한 거래 수
    losing_trades: int       # 손실 거래 수
    win_rate: float         # 승률 (%)
    profit_factor: float    # 손익비 (총이익/총손실)
    max_drawdown: float     # 최대 낙폭 (%)
    sharpe_ratio: float     # 샤프 비율 (연율화)
    total_return: float     # 총 수익률 (%)
    avg_win: float          # 평균 승리 금액
    avg_loss: float         # 평균 손실 금액
    avg_hold_time: float    # 평균 보유 시간 (시간)
    total_commission: float # 총 수수료
```

## 백테스팅 워크플로우

### 1. 데이터 다운로드

```python
async with BinanceDataDownloader() as downloader:
    df = await downloader.download_klines(
        symbol="BTCUSDT",
        interval="1h",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
```

### 2. 시그널 생성

```python
from src.signals.base import Signal
from datetime import datetime

signals = [
    Signal(
        type="SWING",
        direction="LONG",
        confidence=0.7,
        entry_price=40000,
        stop_loss=39200,  # 2% SL
        take_profit=41600,  # 4% TP
        reason="Bullish momentum",
        timestamp=datetime(2024, 1, 15, 10, 0),
    ),
    # ... more signals
]
```

### 3. 백테스트 실행

```python
engine = BacktestEngine(initial_capital=10000)
result = engine.run(data=df, signals=signals)
result.print_summary()
```

### 4. 결과 분석

```python
# 개별 거래 분석
for trade in result.trades:
    print(f"{trade.direction} @ {trade.entry_price} -> {trade.exit_price}")
    print(f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
    print(f"Exit: {trade.exit_reason}")  # TP, SL, or END

# 목표 대비 성과 평가
targets = {
    "승률": (result.win_rate, 45.0),
    "손익비": (result.profit_factor, 1.5),
    "최대 낙폭": (result.max_drawdown, 20.0),
    "샤프 비율": (result.sharpe_ratio, 1.0),
}
```

## 예제

전체 예제는 `examples/backtest_example.py` 참조:

```bash
python examples/backtest_example.py
```

## 주의사항

### 미래 데이터 누수 방지

백테스트 엔진은 각 시점에서 **과거 데이터만** 사용하도록 설계되었습니다:

- 시그널 타임스탬프가 현재 캔들 시간보다 이전인 경우에만 처리
- 진입/청산 시 현재 캔들의 close 가격 사용
- SL/TP는 캔들의 high/low 체크

### 현실적인 파라미터

```python
# 권장 설정
BacktestEngine(
    initial_capital=10000.0,
    commission=0.0004,      # Binance taker fee
    slippage=0.0001,        # 1분봉: 0.01%, 1시간봉: 0.05%
    position_size=0.95,     # 버퍼 5% 유지
)
```

### 샤프 비율 계산

- 데이터 주기에 맞춰 자동 연율화
- 1시간봉: `sqrt(365 * 24)` 적용
- 1분봉: `sqrt(365 * 24 * 60)` 적용

## 파일 구조

```
src/backtest/
├── __init__.py           # 모듈 진입점
├── download_data.py      # 바이낸스 데이터 다운로더
├── engine.py             # 백테스트 엔진
└── README.md            # 이 문서

data/                     # CSV 데이터 저장 위치
examples/
└── backtest_example.py  # 사용 예제
```

## 성능

- 1년치 1시간봉 데이터 (8760 캔들) 백테스트: ~0.1초
- 1년치 1분봉 데이터 (525600 캔들) 백테스트: ~5초

## 향후 개선 사항

- [ ] 복수 심볼 동시 백테스트
- [ ] 포트폴리오 백테스트 (여러 전략 조합)
- [ ] Walk-forward 분석
- [ ] 몬테카르로 시뮬레이션
- [ ] 시각화 (equity curve, drawdown chart)
