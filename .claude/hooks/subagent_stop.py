#!/usr/bin/env python3
"""
SubagentStop Hook - 서브에이전트 작업 완료 시 실행

작업 결과 요약 및 다음 단계 안내
"""

import sys

def main():
    print("\n✅ 서브에이전트 작업 완료")
    print("   메인 에이전트로 결과가 전달됩니다.")
    sys.exit(0)

if __name__ == "__main__":
    main()
