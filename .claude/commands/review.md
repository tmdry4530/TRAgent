---
description: 코드 리뷰 실행 - @code-reviewer 서브에이전트 활용
---

# 코드 리뷰

## 인자

- `/review` → 변경된 파일 전체 리뷰
- `/review src/collectors/` → 특정 디렉토리 리뷰
- `/review src/signals/scalp.py` → 특정 파일 리뷰

## 실행

@code-reviewer 서브에이전트에게 위임하여:

1. **보안 검사**
   - API 키 하드코딩 여부
   - 민감 정보 로깅 여부
   - 입력 검증

2. **트레이딩 로직 검증**
   - 손절 로직 누락 여부
   - 포지션 사이징 정확성
   - 레버리지 한도 준수

3. **코드 품질**
   - 타입 힌트
   - docstring
   - 테스트 존재 여부

4. **자동 도구 실행**
   ```bash
   ruff check $ARGUMENTS
   mypy $ARGUMENTS
   bandit -r $ARGUMENTS
   ```

## 결과 형식

```markdown
## 코드 리뷰 결과: {파일/디렉토리}

### 🔴 Critical (즉시 수정)
- [파일:라인] 이슈

### 🟡 Warning (권장)
- [파일:라인] 이슈

### 🟢 Suggestion (개선)
- [파일:라인] 제안

### ✅ Good
- 잘된 점
```

## 리뷰 후

Critical 이슈 있으면:
1. 수정 제안
2. 사용자 확인 후 자동 수정
3. 재리뷰