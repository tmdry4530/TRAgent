"""Example usage of Risk module."""

import yaml
from datetime import datetime
from pathlib import Path

from src.risk import (
    PositionCalculator,
    RiskManager,
    TradeResult,
)
from src.signals.base import Signal


def load_config():
    """Load trading configuration."""
    config_path = Path(__file__).parent.parent / "config" / "trading.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Demonstrate risk management functionality."""
    # Load configuration
    config = load_config()

    # Initialize components
    calculator = PositionCalculator(config)
    risk_manager = RiskManager(config)

    print("=" * 60)
    print("Risk Management Example")
    print("=" * 60)

    # Example 1: Calculate position size for scalp signal
    print("\n1. Calculate Scalp Position Size")
    print("-" * 60)

    scalp_signal = Signal(
        type="SCALP",
        direction="LONG",
        confidence=0.85,
        entry_price=50000.0,
        stop_loss=49250.0,  # 1.5% stop loss
        take_profit=52500.0,  # 5% take profit
        reason="Volume breakout + funding rate extreme",
        timestamp=datetime.now(),
    )

    account_balance = 10000.0
    current_price = 50000.0

    position = calculator.calculate(
        signal=scalp_signal,
        account_balance=account_balance,
        current_price=current_price,
    )

    if position:
        print(f"Signal: {scalp_signal.type} {scalp_signal.direction}")
        print(f"Entry: ${scalp_signal.entry_price:,.2f}")
        print(f"Stop Loss: ${scalp_signal.stop_loss:,.2f} ({((scalp_signal.stop_loss - scalp_signal.entry_price) / scalp_signal.entry_price * 100):.2f}%)")
        print(f"Take Profit: ${scalp_signal.take_profit:,.2f} ({((scalp_signal.take_profit - scalp_signal.entry_price) / scalp_signal.entry_price * 100):.2f}%)")
        print(f"\nPosition Size:")
        print(f"  USD: ${position.size_usd:,.2f}")
        print(f"  BTC: {position.size_btc:.4f}")
        print(f"  Leverage: {position.leverage}x")
        print(f"  Risk Amount: ${position.risk_amount:.2f} ({position.risk_amount / account_balance * 100:.1f}%)")
    else:
        print("Position calculation failed")

    # Example 2: Risk check for signal
    print("\n\n2. Risk Check")
    print("-" * 60)

    account_state = {
        "balance": account_balance,
        "positions": [],
        "llm_confidence": 0.75,
    }

    risk_check = risk_manager.check(scalp_signal, account_state)

    print(f"Approved: {risk_check.approved}")
    print(f"Adjusted Size: {risk_check.adjusted_size * 100:.0f}%")
    if risk_check.warnings:
        print(f"Warnings: {', '.join(risk_check.warnings)}")
    if risk_check.reason:
        print(f"Rejection Reason: {risk_check.reason}")

    # Example 3: Position conflict detection
    print("\n\n3. Position Conflict Detection")
    print("-" * 60)

    # Try to open SHORT when already LONG
    short_signal = Signal(
        type="SCALP",
        direction="SHORT",
        confidence=0.8,
        entry_price=50000.0,
        stop_loss=50750.0,
        take_profit=47500.0,
        reason="Opposite direction test",
        timestamp=datetime.now(),
    )

    account_state_with_position = {
        "balance": account_balance,
        "positions": [
            {
                "type": "SWING",
                "direction": "LONG",
                "notional": 5000.0,
            }
        ],
    }

    risk_check = risk_manager.check(short_signal, account_state_with_position)

    print(f"Signal: {short_signal.type} {short_signal.direction}")
    print(f"Existing Position: SWING LONG")
    print(f"Approved: {risk_check.approved}")
    print(f"Reason: {risk_check.reason}")

    # Example 4: Simulate trading and loss tracking
    print("\n\n4. Trade Tracking & Cooldown")
    print("-" * 60)

    print("Simulating 3 consecutive losses...")

    for i in range(3):
        loss = TradeResult(
            timestamp=datetime.now(),
            pnl=-150.0,
            pnl_pct=-1.5,
            signal_type="SCALP",
            direction="LONG",
        )
        risk_manager.record_trade_result(loss)
        print(f"  Loss {i + 1}: PnL ${loss.pnl:.2f} ({loss.pnl_pct:.1f}%)")

    print(f"\nConsecutive Losses: {risk_manager.consecutive_losses}")
    print(f"Cooldown Until: {risk_manager.cooldown_until.strftime('%H:%M:%S') if risk_manager.cooldown_until else 'None'}")

    # Try to trade during cooldown
    risk_check = risk_manager.check(scalp_signal, account_state)
    print(f"\nNew trade during cooldown:")
    print(f"  Approved: {risk_check.approved}")
    print(f"  Reason: {risk_check.reason}")

    # Example 5: Event blackout
    print("\n\n5. Event Blackout")
    print("-" * 60)

    from datetime import timedelta

    # Add upcoming FOMC event in 15 minutes
    upcoming_event = datetime.now() + timedelta(minutes=15)
    risk_manager.update_events([
        {
            "name": "FOMC Rate Decision",
            "time": upcoming_event.isoformat(),
        }
    ])

    print(f"Upcoming Event: FOMC Rate Decision at {upcoming_event.strftime('%H:%M:%S')}")
    print(f"Event Blackout Active: {risk_manager.is_event_blackout()}")

    # Reset cooldown for this example
    risk_manager.cooldown_until = None

    risk_check = risk_manager.check(scalp_signal, account_state)
    print(f"\nNew trade during blackout:")
    print(f"  Approved: {risk_check.approved}")
    print(f"  Reason: {risk_check.reason}")

    # Example 6: Statistics
    print("\n\n6. Risk Manager Statistics")
    print("-" * 60)

    stats = risk_manager.get_stats()
    print(f"Daily PnL: ${stats['daily_pnl']:.2f}")
    print(f"Consecutive Losses: {stats['consecutive_losses']}")
    print(f"Trades Today: {stats['total_trades_today']}")
    print(f"Cooldown Until: {stats['cooldown_until'] or 'None'}")
    print(f"Event Blackout: {stats['event_blackout']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
