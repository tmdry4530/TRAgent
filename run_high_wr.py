#!/usr/bin/env python3
"""Run High Win Rate Trading Bot.

Usage:
    python run_high_wr.py
    python run_high_wr.py --testnet  # Run on testnet
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main_high_wr import main

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           HIGH WIN RATE TRADING BOT (OPTIMIZED)                  ║
    ║                                                                  ║
    ║   Strategy: Volume Climax + Channel Bounce                       ║
    ║   Risk per Trade: 30%  |  R:R Ratio: 1.5                         ║
    ║   Expected Win Rate: 68%  |  Annual Return: 500%+                ║
    ║                                                                  ║
    ║   Risk Management: Consecutive Loss Protection                   ║
    ║   - 2 consecutive losses: 50% position (15% risk)                ║
    ║   - 3 consecutive losses: 25% position (7.5% risk)               ║
    ║   - 5 consecutive losses: STOP trading                           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
