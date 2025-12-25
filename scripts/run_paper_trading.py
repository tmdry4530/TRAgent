#!/usr/bin/env python3
"""Run paper trading on Binance Futures testnet.

Usage:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --log-level DEBUG
    python scripts/run_paper_trading.py --symbol ETHUSDT
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_environment(args: argparse.Namespace) -> None:
    """Setup environment variables for paper trading.

    Args:
        args: Command line arguments
    """
    # Force testnet mode
    os.environ["BINANCE_TESTNET"] = "true"

    # Set log level
    os.environ["LOG_LEVEL"] = args.log_level

    # Override symbol if specified
    if args.symbol:
        os.environ["TRADING_SYMBOL"] = args.symbol

    # Environment mode
    os.environ["ENV"] = "development"


def validate_credentials() -> bool:
    """Validate that required API credentials are set.

    Returns:
        True if credentials are valid
    """
    required_vars = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
    ]

    missing = []
    for var in required_vars:
        value = os.environ.get(var, "")
        if not value or value == "":
            missing.append(var)

    if missing:
        print("Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or environment.")
        print("\nFor testnet credentials, visit:")
        print("  https://testnet.binancefuture.com/")
        return False

    return True


async def run_paper_trading() -> None:
    """Run the paper trading bot."""
    from src.main import TradingBot
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    print("=" * 60)
    print("  BINANCE FUTURES PAPER TRADING (TESTNET)")
    print("=" * 60)
    print()
    print("Mode: TESTNET (No real funds at risk)")
    print(f"Symbol: {os.environ.get('TRADING_SYMBOL', 'BTCUSDT')}")
    print(f"Log Level: {os.environ.get('LOG_LEVEL', 'INFO')}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    bot = TradingBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        await bot.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run paper trading on Binance Futures testnet"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (default: BTCUSDT)",
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment(args)

    # Load .env file if exists
    env_file = project_root / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    # Validate credentials
    if not validate_credentials():
        sys.exit(1)

    # Run bot
    asyncio.run(run_paper_trading())


if __name__ == "__main__":
    main()
