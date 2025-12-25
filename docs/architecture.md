# Architecture Document

## 시스템 개요

바이낸스 BTC 선물 자동매매 에이전트의 상세 아키텍처입니다.

## 컴포넌트 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Layer                                   │
├─────────────────┬─────────────────┬─────────────────┬───────────────┤
│   Binance WS    │   Binance REST  │   Macro APIs    │    News       │
│   - Klines      │   - Funding     │   - FRED        │  - CryptoPanic│
│   - Orderbook   │   - OI          │   - Yahoo       │  - Whale Alert│
│   - Trades      │   - Long/Short  │   - Fear&Greed  │               │
│   - Liquidation │                 │                 │               │
└────────┬────────┴────────┬────────┴────────┬────────┴───────┬───────┘
         │                 │                 │                │
         ▼                 ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Redis Cache                                  │
│   - market:btcusdt:price     - macro:fear_greed                     │
│   - market:btcusdt:orderbook - macro:dxy                            │
│   - market:btcusdt:funding   - news:recent                          │
│   - market:btcusdt:oi        - events:upcoming                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Signal Generator Layer                            │
├─────────────────────────────────┬───────────────────────────────────┤
│         Scalp Signals           │          Swing Signals            │
│   - LiquidationCascade          │   - EmaRsiSwing                   │
│   - FundingRateExtreme          │   - FearGreedFilter               │
│   - VolumeBreakout              │                                   │
└─────────────────────────────────┴───────────────────────────────────┘
                                 │
                                 │ Signal 발생
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM Context Filter                              │
│                                                                      │
│   Input:                        Output:                              │
│   - Signal                      - execute: bool                      │
│   - MarketState                 - confidence: float                  │
│   - MacroContext                - adjusted_size: float               │
│   - RecentNews                  - reason: str                        │
│                                                                      │
│   Claude API (claude-sonnet-4-20250514)                                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 │ 실행 결정
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Risk Manager                                   │
│                                                                      │
│   Checks:                       Actions:                             │
│   - Daily loss limit            - Position sizing                    │
│   - Consecutive losses          - Leverage adjustment                │
│   - Event blackout              - Block/Allow                        │
│   - LLM confidence threshold                                         │
└─────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 │ 승인됨
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Executor                                      │
│                                                                      │
│   - Order placement             - Position tracking                  │
│   - Stop loss management        - Take profit management             │
│   - Order status monitoring     - Trailing stop (Swing)              │
│                                                                      │
│   Binance Futures API                                                │
└─────────────────────────────────────────────────────────────────────┘
```

## 데이터 흐름

### 실시간 데이터 흐름

```python
async def main_loop():
    """메인 이벤트 루프"""
    
    # 1. 데이터 수집기 시작
    collectors = [
        BinanceWebSocketCollector(),
        MacroDataCollector(),
        NewsCollector(),
    ]
    
    for collector in collectors:
        await collector.start()
    
    # 2. 메인 루프
    while True:
        # 2.1 시장 상태 구성
        market_state = await build_market_state()
        macro_context = await build_macro_context()
        recent_news = await get_recent_news()
        
        # 2.2 시그널 체크
        signals = []
        for generator in signal_generators:
            signal = await generator.check(market_state)
            if signal:
                signals.append(signal)
        
        # 2.3 LLM 필터
        for signal in signals:
            decision = await llm_filter.evaluate(
                signal, market_state, macro_context, recent_news
            )
            
            if not decision.execute:
                continue
            
            # 2.4 리스크 체크
            risk_check = await risk_manager.check(signal, account)
            
            if not risk_check.approved:
                continue
            
            # 2.5 주문 실행
            await executor.execute(
                signal,
                size=decision.adjusted_size * risk_check.max_position_size,
                leverage=risk_check.adjusted_leverage
            )
        
        await asyncio.sleep(1)
```

### 이벤트 기반 트리거

```python
# WebSocket 메시지 핸들러
class EventHandlers:
    
    async def on_liquidation(self, event: LiquidationEvent):
        """청산 이벤트 처리"""
        liquidation_signal.add_event(event)
        
        # 시그널 즉시 체크
        signal = liquidation_signal.check_signal()
        if signal:
            await process_signal(signal)
    
    async def on_funding_rate(self, rate: float):
        """펀딩비 업데이트"""
        signal = funding_signal.check_signal(rate)
        if signal:
            await process_signal(signal)
    
    async def on_kline(self, kline: dict):
        """캔들 업데이트"""
        # DataFrame 업데이트
        update_dataframe(kline)
        
        # 캔들 완성 시에만 스윙 시그널 체크
        if kline['x']:  # 캔들 완성
            signal = swing_signal.check_signal(df)
            if signal:
                await process_signal(signal)
```

## 모듈 상세

### collectors/

| 파일 | 클래스 | 역할 |
|------|--------|------|
| binance.py | BinanceWebSocketCollector | 바이낸스 실시간 데이터 |
| binance.py | BinanceRestCollector | 바이낸스 REST 데이터 |
| macro.py | MacroDataCollector | FRED, Yahoo 데이터 |
| news.py | NewsCollector | CryptoPanic, Whale Alert |

### signals/

| 파일 | 클래스 | 역할 |
|------|--------|------|
| scalp.py | LiquidationCascadeSignal | 청산 캐스케이드 |
| scalp.py | FundingRateSignal | 펀딩비 역추세 |
| scalp.py | VolumeBreakoutSignal | 거래량 돌파 |
| swing.py | EmaRsiSwingSignal | EMA+RSI 스윙 |
| swing.py | FearGreedFilter | Fear&Greed 필터 |

### brain/

| 파일 | 클래스 | 역할 |
|------|--------|------|
| context_filter.py | LLMContextFilter | Claude API 연동 |
| prompts.py | - | 프롬프트 템플릿 |

### risk/

| 파일 | 클래스 | 역할 |
|------|--------|------|
| manager.py | RiskManager | 리스크 체크 |
| calculator.py | PositionCalculator | 포지션 사이징 |

### executor/

| 파일 | 클래스 | 역할 |
|------|--------|------|
| binance.py | BinanceExecutor | 주문 실행 |
| position.py | PositionManager | 포지션 관리 |

## 에러 처리

```python
class TradingError(Exception):
    """트레이딩 에러 베이스"""
    pass

class ConnectionError(TradingError):
    """연결 에러 - 재시도"""
    pass

class InsufficientBalanceError(TradingError):
    """잔고 부족 - 로깅 후 스킵"""
    pass

class RiskLimitError(TradingError):
    """리스크 한도 초과 - 대기"""
    pass

# 에러 핸들러
async def handle_error(error: Exception):
    if isinstance(error, ConnectionError):
        await reconnect()
    elif isinstance(error, InsufficientBalanceError):
        logger.warning(f"잔고 부족: {error}")
    elif isinstance(error, RiskLimitError):
        logger.info(f"리스크 한도: {error}")
        await wait_until_reset()
    else:
        logger.error(f"알 수 없는 에러: {error}")
        raise
```

## 로깅

```python
import structlog

logger = structlog.get_logger()

# 로그 레벨별 사용
logger.debug("상세 디버그 정보")
logger.info("정상 작동 정보", signal=signal, decision=decision)
logger.warning("주의 필요", reason="펀딩비 높음")
logger.error("에러 발생", error=str(e), traceback=traceback.format_exc())

# 거래 로그 (별도 파일)
trade_logger = structlog.get_logger("trades")
trade_logger.info(
    "trade_executed",
    signal_type=signal.type,
    direction=signal.direction,
    entry_price=entry_price,
    size=size,
    leverage=leverage
)
```
