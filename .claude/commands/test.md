---
description: 테스트 실행 및 커버리지 리포트
---

# 테스트 실행

## 인자

- `/test` → 전체 테스트
- `/test collectors` → collectors 테스트만
- `/test signals` → signals 테스트만
- `/test --coverage` → 커버리지 포함

## 실행

```bash
# 전체 테스트
pytest tests/ -v

# 모듈별
pytest tests/$ARGUMENTS/ -v

# 커버리지
pytest tests/ -v --cov=src --cov-report=html
```

## 실패 시

테스트 실패하면:

1. 실패한 테스트 분석
2. 원인 파악
3. @code-reviewer로 코드 검토
4. 수정 후 재실행

## 결과 리포트

```
================================
       테스트 결과 요약
================================
총 테스트: N개
성공: N개 ✅
실패: N개 ❌
스킵: N개 ⏭️
커버리지: N%
================================
```