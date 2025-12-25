#!/usr/bin/env python3
"""
PostToolUse Hook - Write/Edit 후 자동 검증

코드 품질 검사 및 보안 스캔
"""

import sys
import json
import subprocess
import os
from pathlib import Path

def run_checks(filepath: str):
    """파일 타입에 따른 검사 실행"""
    path = Path(filepath)
    
    if not path.exists():
        return
    
    results = []
    
    # Python 파일 검사
    if path.suffix == '.py':
        # Ruff 린트 (빠름)
        try:
            result = subprocess.run(
                ['ruff', 'check', str(path), '--select=E,F,W'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                results.append(f"🔍 Lint 이슈:\n{result.stdout}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # 보안 검사 (src/ 파일만)
        if 'src/' in str(path):
            try:
                result = subprocess.run(
                    ['bandit', '-q', str(path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.stdout.strip():
                    results.append(f"🔒 보안 이슈:\n{result.stdout}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    
    # 민감 정보 노출 검사 (모든 파일)
    sensitive_patterns = [
        'api_key',
        'secret_key', 
        'password',
        'BINANCE_API_KEY',
        'ANTHROPIC_API_KEY',
    ]
    
    try:
        content = path.read_text()
        for pattern in sensitive_patterns:
            # 실제 값이 하드코딩된 경우만 경고 (환경변수 참조는 OK)
            if f'{pattern}=' in content.lower() and 'os.getenv' not in content and 'environ' not in content:
                if not any(x in str(path) for x in ['.env', '.example', 'test']):
                    results.append(f"⚠️ 민감 정보 노출 가능: {pattern}")
    except Exception:
        pass
    
    # 결과 출력
    if results:
        print("\n".join(results), file=sys.stderr)

def main():
    if len(sys.argv) < 2:
        sys.exit(0)
    
    try:
        tool_input = json.loads(sys.argv[1])
        filepath = tool_input.get('file_path') or tool_input.get('path', '')
    except json.JSONDecodeError:
        filepath = sys.argv[1]
    
    if filepath:
        run_checks(filepath)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
