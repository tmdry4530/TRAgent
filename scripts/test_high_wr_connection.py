"""Test High WR Bot connection and signal generation."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collectors import BinanceRestCollector
from src.executor import BinanceExecutor
from src.signals.high_wr import HighWinRateSignalGenerator
from src.risk.loss_adjuster import ConsecutiveLossAdjuster, LossAdjusterConfig, RecoveryMode
from src.utils.config import get_settings
import pandas as pd


async def test_connection():
    """Test Binance API connection."""
    print("\n" + "=" * 60)
    print("  HIGH WR BOT - CONNECTION TEST")
    print("=" * 60)

    settings = get_settings()
    print(f"\nTestnet: {settings.binance_testnet}")
    print(f"Symbol: {settings.trading_symbol}")

    # Track test results
    rest_ok = False
    executor_ok = False
    signal_ok = False
    adjuster_ok = False
    account = None
    signals = []

    # Test REST API connection
    print("\n[1] Testing REST API connection...")
    rest = BinanceRestCollector()
    klines = []

    try:
        klines = await rest.get_klines(
            symbol=settings.trading_symbol,
            interval="1h",
            limit=100
        )
        print(f"    [OK] Loaded {len(klines)} 1h candles")
        print(f"    Latest: {klines[-1]['timestamp']} @ ${klines[-1]['close']:,.2f}")
        rest_ok = True
    except Exception as e:
        print(f"    [FAIL] Error: {e}")

    # Test Executor
    print("\n[2] Testing Executor connection...")
    executor = BinanceExecutor()

    try:
        account = await executor.get_account_info()
        print(f"    [OK] Connected to Binance Futures")
        print(f"    Balance: ${account.total_balance:,.2f}")
        print(f"    Available: ${account.available_balance:,.2f}")

        # Check position
        position = await executor.get_position(settings.trading_symbol)
        if position:
            print(f"    Active position: {position.side} {position.quantity}")
        else:
            print(f"    No active position")
        executor_ok = True
    except Exception as e:
        print(f"    [FAIL] Error: {e}")

    # Test Signal Generator
    print("\n[3] Testing Signal Generator...")
    signal_gen = HighWinRateSignalGenerator(
        vol_multiplier=3.0,
        consecutive_bars=3,
        wick_pct=0.50,
        rsi_threshold=40,
        channel_vol_threshold=1.2,
        rr_ratio=1.5,
    )

    try:
        if klines:
            # Convert klines to DataFrame
            df = pd.DataFrame(klines)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Generate signals
            signals = signal_gen.generate_from_dataframe(df)
            print(f"    [OK] Signal generator working")
            print(f"    Generated {len(signals)} signals from historical data")

            if signals:
                latest = signals[-1]
                print(f"    Latest signal: {latest.direction} @ ${latest.entry_price:,.2f}")
                print(f"    SL: ${latest.stop_loss:,.2f}, TP: ${latest.take_profit:,.2f}")
            signal_ok = True
        else:
            print(f"    [SKIP] No klines data available")
    except Exception as e:
        print(f"    [FAIL] Error: {e}")

    # Test Loss Adjuster
    print("\n[4] Testing Loss Adjuster...")
    config = LossAdjusterConfig(
        base_risk_per_trade=0.15,
        loss_levels=[
            (1, 1.0),
            (2, 0.50),
            (3, 0.25),
            (4, 0.10),
            (5, 0.0),
        ],
        recovery_mode=RecoveryMode.GRADUAL,
    )
    adjuster = ConsecutiveLossAdjuster(config)

    try:
        # Test adjustment with available balance or default
        test_balance = account.available_balance if account else 10000.0
        result = adjuster.get_adjusted_risk(test_balance)
        print(f"    [OK] Loss adjuster working")
        print(f"    Can trade: {result.can_trade}")
        print(f"    Risk per trade: {result.risk_per_trade*100:.0f}%")
        print(f"    Size multiplier: {result.size_multiplier:.0%}")

        # Simulate some trades
        print("\n    Simulating consecutive losses...")
        for i in range(3):
            adjuster.record_trade(-100, -10)  # Simulate loss
            result = adjuster.get_adjusted_risk(test_balance)
            print(f"    After {i+1} loss(es): multiplier = {result.size_multiplier:.0%}")

        # Simulate win
        print("\n    Simulating win for recovery...")
        adjuster.record_trade(150, 15)  # Simulate win
        result = adjuster.get_adjusted_risk(test_balance)
        print(f"    After win: multiplier = {result.size_multiplier:.0%}")
        adjuster_ok = True

    except Exception as e:
        print(f"    [FAIL] Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    rest_status = "[OK]" if rest_ok else "[FAIL]"
    executor_status = "[OK]" if executor_ok else "[FAIL]"
    signal_status = "[OK]" if signal_ok else "[FAIL]"
    adjuster_status = "[OK]" if adjuster_ok else "[FAIL]"
    balance_str = f"${account.available_balance:,.2f}" if account else "N/A"

    print(f"""
    REST API:      {rest_status} {'Connected' if rest_ok else 'Failed'}
    Executor:      {executor_status} {'Connected' if executor_ok else 'Failed - Check API keys'}
    Balance:       {balance_str}
    Signal Gen:    {signal_status} {'Working' if signal_ok else 'Failed'} ({len(signals)} signals)
    Loss Adjuster: {adjuster_status} {'Working' if adjuster_ok else 'Failed'}
    """)

    if rest_ok and executor_ok and signal_ok and adjuster_ok:
        print("    Ready to run: python run_high_wr.py")
    else:
        print("    [!] Fix the issues above before running live trading")

    # Cleanup
    await rest.close()
    await executor.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
