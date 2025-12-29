"""Aggressive backtest with trailing stop and relaxed conditions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from src.signals.high_wr import HighWinRateSignalGenerator


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    highest_pnl_pct: float = 0  # Max unrealized profit


class AggressiveBacktest:
    """Backtest with trailing stop to ride trends."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.30,  # 30% risk
        commission: float = 0.0004,
        slippage: float = 0.0002,
        trailing_activation: float = 0.02,  # Activate trailing after 2% profit
        trailing_distance: float = 0.015,   # Trail by 1.5%
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance

    def run(self, df: pd.DataFrame, signals: list) -> dict:
        """Run backtest with trailing stops."""
        self.capital = self.initial_capital
        self.trades: list[Trade] = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row["timestamp"]

            # Check exit with trailing stop
            if self.current_position:
                self._check_exit_trailing(row)

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
        """Enter position."""
        entry_price = price * (1 + self.slippage if signal.direction == "LONG" else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price

        if sl_distance == 0 or sl_distance > 0.05:
            return

        risk_amount = self.capital * self.risk_per_trade
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
            "initial_sl": signal.stop_loss,
            "trailing_active": False,
            "highest_price": entry_price if signal.direction == "LONG" else entry_price,
            "lowest_price": entry_price if signal.direction == "SHORT" else entry_price,
            "highest_pnl_pct": 0,
        }

    def _check_exit_trailing(self, row):
        """Check exit with trailing stop logic."""
        pos = self.current_position
        if not pos:
            return

        current_price = row["close"]

        if pos["direction"] == "LONG":
            # Update highest price
            if row["high"] > pos["highest_price"]:
                pos["highest_price"] = row["high"]

            # Calculate current profit %
            profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

            # Track highest unrealized profit
            high_profit_pct = (pos["highest_price"] - pos["entry_price"]) / pos["entry_price"]
            if high_profit_pct > pos["highest_pnl_pct"]:
                pos["highest_pnl_pct"] = high_profit_pct

            # Activate trailing stop after threshold
            if profit_pct >= self.trailing_activation and not pos["trailing_active"]:
                pos["trailing_active"] = True
                pos["stop_loss"] = pos["highest_price"] * (1 - self.trailing_distance)

            # Update trailing stop
            if pos["trailing_active"]:
                new_sl = pos["highest_price"] * (1 - self.trailing_distance)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            # Check stop loss hit
            if row["low"] <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                reason = "TRAIL" if pos["trailing_active"] else "SL"
                self._exit(exit_price, row["timestamp"], reason)

        else:  # SHORT
            # Update lowest price
            if row["low"] < pos["lowest_price"]:
                pos["lowest_price"] = row["low"]

            # Calculate current profit %
            profit_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

            # Track highest unrealized profit
            high_profit_pct = (pos["entry_price"] - pos["lowest_price"]) / pos["entry_price"]
            if high_profit_pct > pos["highest_pnl_pct"]:
                pos["highest_pnl_pct"] = high_profit_pct

            # Activate trailing stop
            if profit_pct >= self.trailing_activation and not pos["trailing_active"]:
                pos["trailing_active"] = True
                pos["stop_loss"] = pos["lowest_price"] * (1 + self.trailing_distance)

            # Update trailing stop
            if pos["trailing_active"]:
                new_sl = pos["lowest_price"] * (1 + self.trailing_distance)
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            # Check stop loss hit
            if row["high"] >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                reason = "TRAIL" if pos["trailing_active"] else "SL"
                self._exit(exit_price, row["timestamp"], reason)

    def _exit(self, price, time, reason):
        """Exit position."""
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

        pnl_pct = pnl / pos["margin"] * 100

        self.trades.append(Trade(
            entry_time=pos["entry_time"],
            exit_time=time,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            highest_pnl_pct=pos["highest_pnl_pct"] * 100,
        ))

        self.current_position = None

    def _unrealized_pnl(self, price):
        pos = self.current_position
        if not pos:
            return 0
        if pos["direction"] == "LONG":
            return (price - pos["entry_price"]) * pos["size"]
        else:
            return (pos["entry_price"] - price) * pos["size"]

    def _calculate_metrics(self) -> dict:
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

        # Average winner/loser
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_return": total_return,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "final_capital": self.capital,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
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

    # Test configurations
    configs = [
        {
            "name": "ORIGINAL (Conservative)",
            "signal_params": {
                "vol_multiplier": 3.0,
                "consecutive_bars": 3,
                "wick_pct": 0.50,
                "rsi_threshold": 40,
                "channel_vol_threshold": 1.2,
                "rr_ratio": 1.5,
            },
            "backtest_params": {
                "risk_per_trade": 0.15,
                "trailing_activation": 999,  # Never activate (use fixed TP)
                "trailing_distance": 0.015,
            }
        },
        {
            "name": "RELAXED + TRAILING",
            "signal_params": {
                "vol_multiplier": 2.0,      # 3.0 → 2.0
                "consecutive_bars": 2,       # 3 → 2
                "wick_pct": 0.40,            # 0.50 → 0.40
                "rsi_threshold": 50,         # 40 → 50
                "channel_vol_threshold": 0.8, # 1.2 → 0.8
                "rr_ratio": 3.0,             # Not used with trailing
            },
            "backtest_params": {
                "risk_per_trade": 0.30,
                "trailing_activation": 0.02,  # 2% profit activates trailing
                "trailing_distance": 0.015,   # 1.5% trail
            }
        },
        {
            "name": "AGGRESSIVE + TIGHT TRAIL",
            "signal_params": {
                "vol_multiplier": 1.5,
                "consecutive_bars": 2,
                "wick_pct": 0.35,
                "rsi_threshold": 55,
                "channel_vol_threshold": 0.6,
                "rr_ratio": 3.0,
            },
            "backtest_params": {
                "risk_per_trade": 0.40,
                "trailing_activation": 0.015,  # 1.5% activates
                "trailing_distance": 0.01,     # 1% tight trail
            }
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"  {config['name']}")
        print(f"{'='*80}")

        # Create signal generator
        signal_gen = HighWinRateSignalGenerator(**config["signal_params"])
        signals = signal_gen.generate_from_dataframe(df)
        print(f"Signals generated: {len(signals)}")

        # Run backtest
        bt = AggressiveBacktest(**config["backtest_params"])
        result = bt.run(df, signals)

        if result:
            results.append({"name": config["name"], **result})
            print(f"\nTrades: {result['trades']}")
            print(f"Win Rate: {result['win_rate']:.1f}%")
            print(f"Return: {result['total_return']:+.1f}%")
            print(f"Max DD: {result['max_drawdown']:.1f}%")
            print(f"Profit Factor: {result['profit_factor']:.2f}")
            print(f"Avg Win: {result['avg_win_pct']:+.1f}%")
            print(f"Avg Loss: {result['avg_loss_pct']:.1f}%")

    # Comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON")
    print(f"{'='*100}")

    print(f"\n{'Config':<30} {'Trades':>8} {'WR':>8} {'Return':>12} {'MDD':>10} {'PF':>8} {'AvgWin':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<30} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['total_return']:>+11.1f}% "
              f"{r['max_drawdown']:>9.1f}% {r['profit_factor']:>8.2f} {r['avg_win_pct']:>+9.1f}%")

    # Show best result trade details
    if len(results) > 1:
        best = max(results, key=lambda x: x['total_return'])
        print(f"\n{'='*100}")
        print(f"  BEST CONFIG: {best['name']} - TRADE DETAILS")
        print(f"{'='*100}")

        print(f"\n{'#':>3} {'Date':<12} {'Dir':<6} {'Exit':<6} {'PnL%':>10} {'MaxPnL%':>10} {'PnL$':>12}")
        print("-" * 80)

        for i, t in enumerate(best['trade_list'][:30], 1):  # Show first 30
            print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} "
                  f"{t.exit_reason:<6} {t.pnl_pct:>+9.1f}% {t.highest_pnl_pct:>+9.1f}% {t.pnl:>+11.1f}")

        if len(best['trade_list']) > 30:
            print(f"... and {len(best['trade_list']) - 30} more trades")

    # Monthly breakdown for best
    if len(results) > 1:
        best = max(results, key=lambda x: x['total_return'])
        print(f"\n{'='*100}")
        print(f"  MONTHLY PERFORMANCE: {best['name']}")
        print(f"{'='*100}")

        from collections import defaultdict
        monthly = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})

        for t in best['trade_list']:
            month = t.entry_time.strftime('%Y-%m')
            monthly[month]['pnl'] += t.pnl
            monthly[month]['trades'] += 1
            if t.pnl > 0:
                monthly[month]['wins'] += 1

        capital = 10000
        print(f"\n{'Month':<10} {'Trades':>8} {'WR':>8} {'PnL$':>12} {'Return':>10} {'Capital':>12}")
        print("-" * 70)

        for month in sorted(monthly.keys()):
            data = monthly[month]
            start_cap = capital
            capital += data['pnl']
            wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
            ret = data['pnl'] / start_cap * 100
            print(f"{month:<10} {data['trades']:>8} {wr:>7.0f}% {data['pnl']:>+11.0f} {ret:>+9.1f}% {capital:>11,.0f}")


if __name__ == "__main__":
    main()
