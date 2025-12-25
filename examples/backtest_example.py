"""Example usage of backtesting engine."""

import asyncio
from datetime import datetime, timedelta

from src.backtest import BinanceDataDownloader, BacktestEngine
from src.signals.base import Signal


async def main():
    """Run a simple backtest example."""
    print("=" * 60)
    print("BACKTEST EXAMPLE")
    print("=" * 60)

    # Step 1: Download historical data
    print("\n1. Downloading historical data...")
    async with BinanceDataDownloader() as downloader:
        df = await downloader.download_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-03-31",  # 3 months for quick test
        )

    print(f"   Downloaded {len(df)} hourly candles")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Generate dummy signals (replace with real signal generator)
    print("\n2. Generating example signals...")
    signals = []

    # Create some example signals based on simple price movements
    for i in range(10, len(df), 50):  # Every 50 hours
        row = df.iloc[i]
        price = row["close"]
        timestamp = row["timestamp"]

        # Alternating LONG/SHORT signals
        if i % 100 == 10:
            # LONG signal
            signals.append(
                Signal(
                    type="SWING",
                    direction="LONG",
                    confidence=0.7,
                    entry_price=price,
                    stop_loss=price * 0.98,  # 2% stop loss
                    take_profit=price * 1.04,  # 4% take profit
                    reason="Example LONG signal",
                    timestamp=timestamp,
                )
            )
        else:
            # SHORT signal
            signals.append(
                Signal(
                    type="SWING",
                    direction="SHORT",
                    confidence=0.6,
                    entry_price=price,
                    stop_loss=price * 1.02,  # 2% stop loss
                    take_profit=price * 0.96,  # 4% take profit
                    reason="Example SHORT signal",
                    timestamp=timestamp,
                )
            )

    print(f"   Generated {len(signals)} signals")

    # Step 3: Run backtest
    print("\n3. Running backtest...")
    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,  # 0.04%
        slippage=0.0001,  # 0.01%
        position_size=0.95,  # Use 95% of capital
    )

    result = engine.run(data=df, signals=signals)

    # Step 4: Display results
    print("\n4. Results:")
    result.print_summary()

    # Additional analysis
    if result.trades:
        print("\nFirst 5 trades:")
        for i, trade in enumerate(result.trades[:5], 1):
            print(f"\n  Trade {i}:")
            print(f"    Type:      {trade.signal_type} {trade.direction}")
            print(f"    Entry:     {trade.entry_time} @ ${trade.entry_price:.2f}")
            print(f"    Exit:      {trade.exit_time} @ ${trade.exit_price:.2f}")
            print(f"    Exit Reason: {trade.exit_reason}")
            print(f"    PnL:       ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")

    # Performance assessment
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 60)

    targets = {
        "Win Rate": (result.win_rate, 45.0, "%"),
        "Profit Factor": (result.profit_factor, 1.5, ""),
        "Max Drawdown": (result.max_drawdown, 20.0, "%"),
        "Sharpe Ratio": (result.sharpe_ratio, 1.0, ""),
    }

    for metric, (value, target, unit) in targets.items():
        status = "PASS" if (
            (metric == "Max Drawdown" and value < target) or
            (metric != "Max Drawdown" and value > target)
        ) else "FAIL"

        print(f"{metric:20s} {value:8.2f}{unit:1s} (target: {target:.1f}{unit}) [{status}]")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
