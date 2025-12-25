#!/usr/bin/env python3
"""
Stop Hook - Claude Code 세션 종료 시 실행

작업 요약 및 다음 단계 안내
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def get_git_status():
    """Git 상태 확인"""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return ""

def get_uncommitted_changes():
    """커밋되지 않은 변경사항 개수"""
    status = get_git_status()
    if not status:
        return 0
    return len(status.split('\n'))

def main():
    changes = get_uncommitted_changes()
    
    print("\n" + "=" * 50)
    print("       세션 종료 요약")
    print("=" * 50)
    
    # 변경사항 알림
    if changes > 0:
        print(f"\n⚠️  커밋되지 않은 변경: {changes}개 파일")
        print("   다음 명령어로 확인: git status")
    else:
        print("\n✅ 모든 변경사항 커밋됨")
    
    # 테스트 실행 권장
    print("\n📋 다음 세션 전 확인사항:")
    print("   1. pytest tests/ 실행")
    print("   2. ruff check src/ 실행")
    print("   3. git push (필요시)")
    
    # TODO 파일 확인
    todo_file = Path('TODO.md')
    if todo_file.exists():
        content = todo_file.read_text()
        unchecked = content.count('[ ]')
        if unchecked > 0:
            print(f"\n📝 남은 TODO: {unchecked}개")
    
    print("\n" + "=" * 50)
    sys.exit(0)

if __name__ == "__main__":
    main()
