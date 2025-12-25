#!/usr/bin/env python3
"""
PreToolUse Hook - Bash 명령어 실행 전 검증

위험한 명령어 차단 및 보안 검사
"""

import sys
import json
import re

# 차단할 위험 명령어 패턴
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',           # rm -rf /
    r'rm\s+-rf\s+~',           # rm -rf ~
    r'>\s*/dev/sd',            # 디스크 직접 쓰기
    r'mkfs\.',                 # 파일시스템 포맷
    r'dd\s+if=',               # dd 명령어
    r':(){ :|:& };:',          # Fork bomb
    r'chmod\s+-R\s+777\s+/',   # 루트 권한 변경
    r'curl.*\|\s*bash',        # 원격 스크립트 실행
    r'wget.*\|\s*bash',
]

# 경고할 명령어 패턴
WARNING_PATTERNS = [
    r'pip\s+install(?!.*requirements)',  # 직접 pip install
    r'npm\s+install\s+-g',                # 전역 npm 설치
    r'sudo\s+',                           # sudo 사용
]

def check_command(command: str) -> tuple[bool, str]:
    """명령어 검증
    
    Returns:
        (allowed, message)
    """
    # 위험 명령어 체크
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"🚫 위험한 명령어 차단: {pattern}"
    
    # 경고 명령어 체크
    for pattern in WARNING_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            print(f"⚠️ 주의 필요: {command}", file=sys.stderr)
    
    return True, ""

def main():
    if len(sys.argv) < 2:
        sys.exit(0)
    
    try:
        tool_input = json.loads(sys.argv[1])
        command = tool_input.get('command', '')
    except json.JSONDecodeError:
        command = sys.argv[1]
    
    allowed, message = check_command(command)
    
    if not allowed:
        print(message, file=sys.stderr)
        sys.exit(2)  # Exit code 2 = 실행 차단
    
    sys.exit(0)

if __name__ == "__main__":
    main()
