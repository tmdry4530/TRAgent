# Binance Futures API Skill

바이낸스 선물 API 연동 시 참조하는 스킬입니다.

## 사용 시점

- 바이낸스 WebSocket 연결 구현
- REST API 호출 구현
- 주문 실행 로직 구현

## WebSocket 엔드포인트

### Futures (선물)

```
Base URL: wss://fstream.binance.com

# 스트림
/ws/btcusdt@kline_1m          # 1분봉 캔들
/ws/btcusdt@depth20@100ms     # 오더북 (20레벨, 100ms)
/ws/btcusdt@aggTrade          # 체결 데이터
/ws/btcusdt@forceOrder        # 청산 데이터
/ws/btcusdt@markPrice@1s      # 마크 가격

# 복합 스트림
/stream?streams=btcusdt@kline_1m/btcusdt@depth20@100ms
```

### 메시지 형식

```python
# Kline (캔들)
{
    "e": "kline",
    "E": 1672531200000,  # 이벤트 시간
    "s": "BTCUSDT",
    "k": {
        "t": 1672531200000,  # 캔들 시작 시간
        "T": 1672531259999,  # 캔들 종료 시간
        "s": "BTCUSDT",
        "i": "1m",          # 인터벌
        "o": "16500.00",    # 시가
        "c": "16510.00",    # 종가
        "h": "16520.00",    # 고가
        "l": "16490.00",    # 저가
        "v": "100.00",      # 거래량
        "x": false          # 캔들 완성 여부
    }
}

# 청산
{
    "e": "forceOrder",
    "E": 1672531200000,
    "o": {
        "s": "BTCUSDT",
        "S": "SELL",        # 청산 방향 (롱 청산 = SELL)
        "o": "LIMIT",
        "q": "1.000",       # 수량
        "p": "16500.00",    # 가격
        "ap": "16490.00",   # 평균 체결가
        "X": "FILLED",
        "T": 1672531200000
    }
}
```

## REST API 엔드포인트

```
Base URL: https://fapi.binance.com

# 시장 데이터 (인증 불필요)
GET /fapi/v1/ticker/price           # 현재가
GET /fapi/v1/depth                  # 오더북
GET /fapi/v1/klines                 # 캔들 데이터
GET /fapi/v1/fundingRate            # 펀딩비
GET /fapi/v1/openInterest           # 미결제약정

# 계정 데이터 (인증 필요)
GET /fapi/v2/account                # 계정 정보
GET /fapi/v2/positionRisk           # 포지션 정보
GET /fapi/v1/userTrades             # 거래 내역

# 주문 (인증 필요)
POST /fapi/v1/order                 # 주문
DELETE /fapi/v1/order               # 주문 취소
GET /fapi/v1/openOrders             # 미체결 주문
```

## 인증 방식

```python
import hmac
import hashlib
import time

def sign_request(params: dict, secret_key: str) -> dict:
    """요청 서명"""
    params['timestamp'] = int(time.time() * 1000)
    query_string = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
    signature = hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    params['signature'] = signature
    return params

# 헤더
headers = {
    'X-MBX-APIKEY': api_key
}
```

## 주문 예시

```python
# 시장가 롱 진입
params = {
    'symbol': 'BTCUSDT',
    'side': 'BUY',
    'type': 'MARKET',
    'quantity': 0.001,
}

# 지정가 숏 진입
params = {
    'symbol': 'BTCUSDT',
    'side': 'SELL',
    'type': 'LIMIT',
    'quantity': 0.001,
    'price': 50000,
    'timeInForce': 'GTC',
}

# 스탑로스
params = {
    'symbol': 'BTCUSDT',
    'side': 'SELL',  # 롱 포지션 청산
    'type': 'STOP_MARKET',
    'stopPrice': 49000,
    'closePosition': 'true',
}
```

## Rate Limit

```
# 요청 제한
- 분당 요청: 1200회
- 초당 주문: 10회
- 일일 주문: 200,000회

# 헤더로 확인
X-MBX-USED-WEIGHT-1M: 현재 사용량
```

## 에러 코드

```python
ERROR_CODES = {
    -1000: "알 수 없는 에러",
    -1003: "Rate limit 초과",
    -1021: "타임스탬프 범위 초과",
    -2010: "잔고 부족",
    -2011: "주문 취소 실패",
    -2019: "마진 부족",
    -4003: "수량이 너무 작음",
}
```

## 테스트넷

```
REST: https://testnet.binancefuture.com
WebSocket: wss://stream.binancefuture.com

# 테스트넷 API 키 발급
https://testnet.binancefuture.com/
```
