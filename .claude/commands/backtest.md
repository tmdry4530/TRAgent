---
description: 백테스트 실행 - 전략별 성과 분석 및 리포트 생성
---

백테스트를 실행합니다.

## 인자

$ARGUMENTS가 주어지면 해당 전략만 테스트:
- `scalp` - 단타 전략만
- `swing` - 스윙 전략만
- 없으면 전체 전략

## 실행 단계

### 1. 데이터 확인

```bash
# data/ 디렉토리에 과거 데이터가 있는지 확인
ls -la data/

# 없으면 다운로드 필요
python -m src.backtest.download_data --symbol BTCUSDT --interval 1m --days 365
```

### 2. 백테스트 실행

```bash
# 전체 전략
python -m src.backtest.run

# 또는 특정 전략
python -m src.backtest.run --strategy $ARGUMENTS
```

### 3. 결과 분석

백테스트 결과를 분석하고 다음 지표를 확인하세요:

| 지표 | 목표 | 결과 |
|------|------|------|
| 승률 | > 45% | ? |
| 손익비 | > 1.5 | ? |
| MDD | < 20% | ? |
| 샤프 비율 | > 1.0 | ? |

### 4. 리포트 생성

결과를 `data/backtest_results/` 디렉토리에 저장:

```python
# 자동 생성되는 파일들
data/backtest_results/
├── {timestamp}_summary.json      # 요약 지표
├── {timestamp}_trades.csv        # 개별 거래
└── {timestamp}_equity_curve.png  # 자산 곡선
```

### 5. 결과 해석

결과를 해석하고 다음 질문에 답하세요:

1. 목표 지표를 달성했는가?
2. MDD가 허용 범위 내인가?
3. 특정 시장 상황에서 성과가 나빴는가?
4. 파라미터 튜닝이 필요한가?

## 주의사항

- 백테스트 결과가 실제 성과를 보장하지 않음
- 슬리피지/수수료가 실제와 다를 수 있음
- 과적합(overfitting) 주의
