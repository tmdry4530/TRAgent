"""Example usage of swing signal generators.

This example demonstrates how to use EmaRsiSwingSignal and FearGreedFilter
in a trading system.
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals import EmaRsiSwingSignal, FearGreedFilter


async def example_ema_rsi_signal():
    """Example: Generate swing signal using EMA+RSI strategy."""
    print("=" * 60)
    print("Example 1: EMA+RSI Swing Signal")
    print("=" * 60 + "\n")

    # Create signal generator
    signal_gen = EmaRsiSwingSignal()

    print(f"Signal Generator: {signal_gen.name}")
    print(f"Required Data: {signal_gen.get_required_data()}")
    print()

    # Simulate market data (in production, this comes from data collectors)
    # Example: Create 150 4-hour candles
    dates = pd.date_range(start="2024-01-01", periods=150, freq="4h")
    close_prices = [50000 + i * 50 for i in range(150)]

    ohlcv_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices,
            "high": [p * 1.01 for p in close_prices],
            "low": [p * 0.99 for p in close_prices],
            "close": close_prices,
            "volume": [1000] * 150,
        }
    )

    # Market state dictionary
    market_state = {
        "ohlcv_4h": ohlcv_df,
        "price": close_prices[-1],
    }

    # Generate signal
    signal = await signal_gen.generate(market_state)

    if signal:
        print("SIGNAL GENERATED!")
        print(f"  Type: {signal.type}")
        print(f"  Direction: {signal.direction}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Entry: ${signal.entry_price:,.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${signal.take_profit:,.2f}")
        print(f"  Reason: {signal.reason}")
        print(f"  Timestamp: {signal.timestamp}")
        print(f"  Valid: {signal.validate()}")
    else:
        print("No signal - conditions not met")
        print("(This is expected with synthetic data)")

    print()


async def example_fear_greed_filter():
    """Example: Use Fear & Greed filter to validate signals."""
    print("=" * 60)
    print("Example 2: Fear & Greed Filter")
    print("=" * 60 + "\n")

    # Create filter
    filter_gen = FearGreedFilter()

    # Test various market sentiment scenarios
    scenarios = [
        (20, "Extreme Fear"),
        (30, "Fear"),
        (50, "Neutral"),
        (70, "Greed"),
        (80, "Extreme Greed"),
    ]

    print("Long Signal Filter (allowed when F&G < 70):")
    for fg_index, sentiment in scenarios:
        allowed = filter_gen.can_long(fg_index)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  F&G={fg_index:2d} ({sentiment:15s}): {status}")

    print("\nShort Signal Filter (allowed when F&G > 30):")
    for fg_index, sentiment in scenarios:
        allowed = filter_gen.can_short(fg_index)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  F&G={fg_index:2d} ({sentiment:15s}): {status}")

    print()


async def example_integrated_workflow():
    """Example: Integrated workflow with both signal and filter."""
    print("=" * 60)
    print("Example 3: Integrated Workflow")
    print("=" * 60 + "\n")

    # Initialize components
    signal_gen = EmaRsiSwingSignal()
    sentiment_filter = FearGreedFilter()

    # Simulate market data
    dates = pd.date_range(start="2024-01-01", periods=150, freq="4h")
    close_prices = [50000 + i * 50 for i in range(150)]

    ohlcv_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices,
            "high": [p * 1.01 for p in close_prices],
            "low": [p * 0.99 for p in close_prices],
            "close": close_prices,
            "volume": [1000] * 150,
        }
    )

    market_state = {
        "ohlcv_4h": ohlcv_df,
        "price": close_prices[-1],
        "fear_greed_index": 55,  # Neutral sentiment
    }

    print("Step 1: Generate signal...")
    signal = await signal_gen.generate(market_state)

    if signal:
        print(f"  Signal: {signal.direction} at ${signal.entry_price:,.2f}")
        print(f"  Confidence: {signal.confidence:.1%}")

        print("\nStep 2: Apply sentiment filter...")
        fear_greed = market_state["fear_greed_index"]

        if signal.direction == "LONG":
            filter_passed = sentiment_filter.can_long(fear_greed)
        else:
            filter_passed = sentiment_filter.can_short(fear_greed)

        print(f"  Fear & Greed: {fear_greed}")
        print(f"  Filter: {'PASS' if filter_passed else 'FAIL'}")

        if filter_passed:
            print("\nStep 3: Signal ready for LLM evaluation")
            print("  -> Next: Send to LLM filter for context analysis")
            print("  -> Then: Risk manager validates position size")
            print("  -> Finally: Executor places order")
        else:
            print("\nSignal rejected by sentiment filter")
    else:
        print("  No signal generated")

    print()


async def example_signal_validation():
    """Example: Validate signal parameters."""
    print("=" * 60)
    print("Example 4: Signal Validation")
    print("=" * 60 + "\n")

    from src.signals.base import Signal

    # Create a valid LONG signal
    valid_signal = Signal(
        type="SWING",
        direction="LONG",
        confidence=0.85,
        entry_price=60000.0,
        stop_loss=57000.0,  # -5%
        take_profit=69000.0,  # +15%
        reason="EMA aligned bullish, RSI neutral",
        timestamp=datetime.now(timezone.utc),
    )

    print("Valid LONG Signal:")
    print(f"  Entry: ${valid_signal.entry_price:,.2f}")
    print(f"  Stop Loss: ${valid_signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${valid_signal.take_profit:,.2f}")
    print(f"  Validates: {valid_signal.validate()}")
    print()

    # Create an invalid signal (SL above entry for LONG)
    invalid_signal = Signal(
        type="SWING",
        direction="LONG",
        confidence=0.85,
        entry_price=60000.0,
        stop_loss=61000.0,  # Wrong direction!
        take_profit=69000.0,
        reason="Invalid test",
        timestamp=datetime.now(timezone.utc),
    )

    print("Invalid LONG Signal (SL above entry):")
    print(f"  Entry: ${invalid_signal.entry_price:,.2f}")
    print(f"  Stop Loss: ${invalid_signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${invalid_signal.take_profit:,.2f}")
    print(f"  Validates: {invalid_signal.validate()}")
    print()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Swing Signal Examples")
    print("=" * 60 + "\n")

    await example_ema_rsi_signal()
    await example_fear_greed_filter()
    await example_integrated_workflow()
    await example_signal_validation()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
