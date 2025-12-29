"""Test new High Win Rate Signal Generator with Consecutive Loss Adjuster."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from collections import defaultdict

from src.signals.high_wr import HighWinRateSignalGenerator
from src.risk.loss_adjuster import (
    ConsecutiveLossAdjuster,
    LossAdjusterConfig,
    RecoveryMode,
    create_default_adjuster,
)


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str
    risk_used: float
    multiplier: float


class BacktestWithLossAdjuster:
    """Backtest with consecutive loss position adjustment."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        loss_adjuster: ConsecutiveLossAdjuster = None,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage
        self.loss_adjuster = loss_adjuster or create_default_adjuster()

    def run(self, df: pd.DataFrame, signals: list) -> dict:
        """Run backtest with dynamic position sizing."""
        self.capital = self.initial_capital
        self.trades: list[Trade] = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row["timestamp"]

            # Check exit
            if self.current_position:
                self._check_exit(row)

            # Check entry
            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if not self.current_position:
                    self._enter(signal, row["close"], current_time)

            # Track equity
            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(row["close"])
            self.equity_curve.append(equity)

        # Close any open position
        if self.current_position:
            self._exit(df.iloc[-1]["close"], df.iloc[-1]["timestamp"], "END")

        return self._calculate_metrics()

    def _enter(self, signal, price, time):
        """Enter a position with dynamic risk."""
        # Get adjusted risk from loss adjuster
        adjustment = self.loss_adjuster.get_adjusted_risk(self.capital)

        if not adjustment.can_trade:
            return

        risk_per_trade = adjustment.risk_per_trade
        multiplier = adjustment.size_multiplier

        entry_price = price * (1 + self.slippage if signal.direction == "LONG" else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price

        if sl_distance == 0 or sl_distance > 0.05:
            return

        risk_amount = self.capital * risk_per_trade
        position_value = risk_amount / sl_distance
        margin = position_value / self.leverage
        max_margin = self.capital * 0.95

        if margin > max_margin:
            margin = max_margin
            position_value = margin * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission

        if margin + entry_comm >= self.capital:
            return

        self.capital -= (margin + entry_comm)

        self.current_position = {
            "entry_price": entry_price,
            "entry_time": time,
            "size": size,
            "margin": margin,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_used": risk_per_trade,
            "multiplier": multiplier,
        }

    def _check_exit(self, row):
        """Check exit conditions."""
        pos = self.current_position
        if not pos:
            return

        if pos["direction"] == "LONG":
            if row["low"] <= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            elif row["high"] >= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")
        else:
            if row["high"] >= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            elif row["low"] <= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")

    def _exit(self, price, time, reason):
        """Exit position and record trade."""
        pos = self.current_position
        if not pos:
            return

        exit_price = price * (1 - self.slippage if pos["direction"] == "LONG" else 1 + self.slippage)

        if pos["direction"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]

        pnl -= exit_price * pos["size"] * self.commission
        self.capital += pos["margin"] + pnl

        # Record trade result in loss adjuster
        pnl_pct = pnl / pos["margin"] * 100
        self.loss_adjuster.record_trade(pnl, pnl_pct)

        self.trades.append(Trade(
            entry_time=pos["entry_time"],
            exit_time=time,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason,
            risk_used=pos["risk_used"],
            multiplier=pos["multiplier"],
        ))

        self.current_position = None

    def _unrealized_pnl(self, price):
        """Calculate unrealized PnL."""
        pos = self.current_position
        if not pos:
            return 0
        if pos["direction"] == "LONG":
            return (price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - price) * pos["size"]

    def _calculate_metrics(self) -> dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return None

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))

        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_return": total_return,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "final_capital": self.capital,
            "trade_list": self.trades,
        }


def main():
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "BTCUSDT_1h_365d.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} BTC 1h candles")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    # Create signal generator
    signal_gen = HighWinRateSignalGenerator(
        vol_multiplier=3.0,
        consecutive_bars=3,
        wick_pct=0.50,
        rsi_threshold=40,
        channel_vol_threshold=1.2,
        rr_ratio=1.5,
    )

    # Generate signals
    signals = signal_gen.generate_from_dataframe(df)
    print(f"\nGenerated {len(signals)} signals")

    print("\n" + "=" * 100)
    print("  TEST 1: WITHOUT Loss Adjuster (Fixed 15% Risk)")
    print("=" * 100)

    # Backtest without loss adjuster (fixed risk)
    class FixedRiskAdjuster:
        def get_adjusted_risk(self, balance):
            class Result:
                risk_per_trade = 0.15
                size_multiplier = 1.0
                can_trade = True
            return Result()
        def record_trade(self, pnl, pnl_pct):
            pass

    bt_fixed = BacktestWithLossAdjuster(loss_adjuster=FixedRiskAdjuster())
    result_fixed = bt_fixed.run(df, signals)

    if result_fixed:
        print(f"\nTrades: {result_fixed['trades']}")
        print(f"Win Rate: {result_fixed['win_rate']:.1f}%")
        print(f"Return: {result_fixed['total_return']:+.1f}%")
        print(f"Max DD: {result_fixed['max_drawdown']:.1f}%")
        print(f"Profit Factor: {result_fixed['profit_factor']:.2f}")

    print("\n" + "=" * 100)
    print("  TEST 2: WITH Loss Adjuster (Dynamic Risk)")
    print("=" * 100)

    # Create loss adjuster
    config = LossAdjusterConfig(
        base_risk_per_trade=0.15,
        loss_levels=[
            (1, 1.0),    # 1 loss: 100%
            (2, 0.50),   # 2 consecutive: 50%
            (3, 0.25),   # 3 consecutive: 25%
            (4, 0.10),   # 4 consecutive: 10%
            (5, 0.0),    # 5+ consecutive: STOP
        ],
        recovery_mode=RecoveryMode.GRADUAL,
        max_cooldown_hours=4,
        daily_loss_limit_pct=0.10,
    )
    loss_adjuster = ConsecutiveLossAdjuster(config)

    bt_dynamic = BacktestWithLossAdjuster(loss_adjuster=loss_adjuster)
    result_dynamic = bt_dynamic.run(df, signals)

    if result_dynamic:
        print(f"\nTrades: {result_dynamic['trades']}")
        print(f"Win Rate: {result_dynamic['win_rate']:.1f}%")
        print(f"Return: {result_dynamic['total_return']:+.1f}%")
        print(f"Max DD: {result_dynamic['max_drawdown']:.1f}%")
        print(f"Profit Factor: {result_dynamic['profit_factor']:.2f}")

    # Compare
    print("\n" + "=" * 100)
    print("  COMPARISON")
    print("=" * 100)

    if result_fixed and result_dynamic:
        print(f"\n{'Metric':<20} {'Fixed Risk':>15} {'Dynamic Risk':>15} {'Diff':>15}")
        print("-" * 65)
        print(f"{'Trades':<20} {result_fixed['trades']:>15} {result_dynamic['trades']:>15} {result_dynamic['trades'] - result_fixed['trades']:>+15}")
        print(f"{'Win Rate':<20} {result_fixed['win_rate']:>14.1f}% {result_dynamic['win_rate']:>14.1f}% {result_dynamic['win_rate'] - result_fixed['win_rate']:>+14.1f}%")
        print(f"{'Return':<20} {result_fixed['total_return']:>+14.1f}% {result_dynamic['total_return']:>+14.1f}% {result_dynamic['total_return'] - result_fixed['total_return']:>+14.1f}%")
        print(f"{'Max DD':<20} {result_fixed['max_drawdown']:>14.1f}% {result_dynamic['max_drawdown']:>14.1f}% {result_dynamic['max_drawdown'] - result_fixed['max_drawdown']:>+14.1f}%")
        print(f"{'Profit Factor':<20} {result_fixed['profit_factor']:>15.2f} {result_dynamic['profit_factor']:>15.2f} {result_dynamic['profit_factor'] - result_fixed['profit_factor']:>+15.2f}")

    # Show trade details
    print("\n" + "=" * 100)
    print("  TRADE DETAILS (With Loss Adjuster)")
    print("=" * 100)

    if result_dynamic and result_dynamic.get("trade_list"):
        print(f"\n{'#':>3} {'Date':<12} {'Dir':<6} {'Result':<6} {'PnL':>12} {'Risk':>8} {'Mult':>8}")
        print("-" * 70)

        for i, t in enumerate(result_dynamic["trade_list"], 1):
            result_str = "WIN" if t.pnl > 0 else "LOSS"
            print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} {result_str:<6} "
                  f"{t.pnl:>+11.1f} {t.risk_used*100:>7.0f}% {t.multiplier:>7.0%}")

    # Loss adjuster stats
    print("\n" + "=" * 100)
    print("  LOSS ADJUSTER FINAL STATE")
    print("=" * 100)

    stats = loss_adjuster.get_stats()
    print(f"\nConsecutive Losses: {stats['consecutive_losses']}")
    print(f"Consecutive Wins: {stats['consecutive_wins']}")
    print(f"Current Multiplier: {stats['current_multiplier']:.0%}")
    print(f"Effective Risk: {stats['effective_risk']*100:.1f}%")
    print(f"Can Trade: {stats['can_trade']}")


if __name__ == "__main__":
    main()
