---
description: 시스템 상태 확인 - Redis, API 연결, 환경 설정 검증
---

시스템 상태를 확인합니다.

## 체크 항목

### 1. 환경 설정 확인

```bash
# .env 파일 존재 확인
if [ -f .env ]; then
    echo "✅ .env 파일 존재"
else
    echo "❌ .env 파일 없음"
fi

# 필수 환경변수 확인
python -c "
from dotenv import load_dotenv
import os

load_dotenv()

required = [
    'BINANCE_API_KEY',
    'BINANCE_SECRET_KEY', 
    'ANTHROPIC_API_KEY',
    'REDIS_URL'
]

for var in required:
    if os.getenv(var):
        print(f'✅ {var} 설정됨')
    else:
        print(f'❌ {var} 미설정')
"
```

### 2. Redis 연결 확인

```bash
python -c "
import redis
import os

try:
    r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    r.ping()
    print('✅ Redis 연결 성공')
except Exception as e:
    print(f'❌ Redis 연결 실패: {e}')
"
```

### 3. 바이낸스 API 연결 확인

```bash
python -c "
from binance.client import Client
import os

try:
    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_SECRET_KEY'),
        testnet=os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    )
    info = client.futures_account_balance()
    print('✅ 바이낸스 API 연결 성공')
    
    # USDT 잔고 표시
    for asset in info:
        if asset['asset'] == 'USDT':
            print(f'   USDT 잔고: {float(asset[\"balance\"]):.2f}')
except Exception as e:
    print(f'❌ 바이낸스 API 연결 실패: {e}')
"
```

### 4. Claude API 연결 확인

```bash
python -c "
from anthropic import Anthropic
import os

try:
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    # 간단한 테스트 호출
    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=10,
        messages=[{'role': 'user', 'content': 'Hi'}]
    )
    print('✅ Claude API 연결 성공')
except Exception as e:
    print(f'❌ Claude API 연결 실패: {e}')
"
```

### 5. 설정 파일 검증

```bash
python -c "
import yaml
from pathlib import Path

config_path = Path('config/trading.yaml')
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_keys = ['scalp', 'swing', 'risk', 'signals']
    missing = [k for k in required_keys if k not in config]
    
    if missing:
        print(f'❌ 설정 누락: {missing}')
    else:
        print('✅ 설정 파일 유효')
else:
    print('❌ config/trading.yaml 없음')
"
```

### 6. 데이터 디렉토리 확인

```bash
echo "📁 데이터 디렉토리 상태:"
du -sh data/ 2>/dev/null || echo "   data/ 디렉토리 없음"
ls data/*.csv 2>/dev/null | wc -l | xargs -I {} echo "   CSV 파일: {} 개"
```

## 요약 출력

모든 체크 완료 후 요약을 출력하세요:

```
================================
       시스템 상태 요약
================================
환경 설정:  ✅/❌
Redis:      ✅/❌
바이낸스:   ✅/❌
Claude:     ✅/❌
설정 파일:  ✅/❌
================================
```
