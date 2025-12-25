#!/usr/bin/env python3
"""Run trading on Binance Futures mainnet.

WARNING: This script trades with REAL FUNDS. Use with caution.

Usage:
    python scripts/run_mainnet.py --confirm-mainnet
    python scripts/run_mainnet.py --confirm-mainnet --log-level INFO
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
    """Setup environment variables for mainnet trading.

    Args:
        args: Command line arguments
    """
    # Set mainnet mode
    os.environ["BINANCE_TESTNET"] = "false"

    # Set log level
    os.environ["LOG_LEVEL"] = args.log_level

    # Override symbol if specified
    if args.symbol:
        os.environ["TRADING_SYMBOL"] = args.symbol

    # Environment mode
    os.environ["ENV"] = "production"


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
        return False

    return True


def confirm_mainnet_trading() -> bool:
    """Get user confirmation for mainnet trading.

    Returns:
        True if user confirms
    """
    print()
    print("=" * 60)
    print("  WARNING: MAINNET TRADING MODE")
    print("=" * 60)
    print()
    print("You are about to start trading with REAL FUNDS on Binance Futures.")
    print()
    print("This bot will:")
    print("  - Open LONG and SHORT positions")
    print("  - Use leverage (up to 20x for scalp, 10x for swing)")
    print("  - Place market orders, stop-loss, and take-profit orders")
    print()
    print("Risk Parameters:")
    print(f"  - Daily loss limit: {os.environ.get('DAILY_LOSS_LIMIT_PCT', '10')}%")
    print(f"  - Max exposure: {os.environ.get('MAX_TOTAL_EXPOSURE_PCT', '80')}%")
    print()

    response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")

    return response.strip() == "I UNDERSTAND THE RISKS"


async def run_mainnet_trading() -> None:
    """Run the mainnet trading bot."""
    from src.main import TradingBot
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    print()
    print("=" * 60)
    print("  BINANCE FUTURES MAINNET TRADING")
    print("=" * 60)
    print()
    print("Mode: MAINNET (Real funds)")
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
        description="Run trading on Binance Futures mainnet (REAL FUNDS)"
    )
    parser.add_argument(
        "--confirm-mainnet",
        action="store_true",
        required=True,
        help="Required flag to confirm mainnet trading",
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
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip interactive confirmation (for automated deployments)",
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

    # Confirm mainnet trading (unless skipped)
    if not args.skip_confirmation:
        if not confirm_mainnet_trading():
            print("\nMainnet trading cancelled.")
            sys.exit(0)

    # Run bot
    asyncio.run(run_mainnet_trading())


if __name__ == "__main__":
    main()
