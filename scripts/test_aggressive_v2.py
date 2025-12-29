"""Aggressive backtest v2: Partial TP + Trailing for remainder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

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
    max_price_seen: float = 0


class HybridBacktest:
    """
    Hybrid strategy:
    1. First TP hit -> close 50%, move SL to breakeven
    2. Trail remainder for bigger trends
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.30,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        first_tp_ratio: float = 1.5,    # First TP at 1.5x SL
        partial_close_pct: float = 0.5,  # Close 50% at first TP
        trail_distance: float = 0.02,    # Trail 2% for remainder
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.first_tp_ratio = first_tp_ratio
        self.partial_close_pct = partial_close_pct
        self.trail_distance = trail_distance

    def run(self, df: pd.DataFrame, signals: list) -> dict:
        self.capital = self.initial_capital
        self.trades: list[Trade] = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row["timestamp"]

            if self.current_position:
                self._check_exit_hybrid(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if not self.current_position:
                    self._enter(signal, row["close"], current_time)

            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(row["close"])
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]["close"], df.iloc[-1]["timestamp"], "END", 1.0)

        return self._calculate_metrics()

    def _enter(self, signal, price, time):
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

        # Calculate first TP
        tp_distance = sl_distance * self.first_tp_ratio
        if signal.direction == "LONG":
            first_tp = entry_price * (1 + tp_distance)
        else:
            first_tp = entry_price * (1 - tp_distance)

        self.current_position = {
            "entry_price": entry_price,
            "entry_time": time,
            "size": size,
            "original_size": size,
            "margin": margin,
            "original_margin": margin,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "initial_sl": signal.stop_loss,
            "first_tp": first_tp,
            "first_tp_hit": False,
            "trailing_active": False,
            "highest_price": entry_price,
            "lowest_price": entry_price,
        }

    def _check_exit_hybrid(self, row):
        pos = self.current_position
        if not pos:
            return

        if pos["direction"] == "LONG":
            # Update tracking
            if row["high"] > pos["highest_price"]:
                pos["highest_price"] = row["high"]

            # Check first TP
            if not pos["first_tp_hit"] and row["high"] >= pos["first_tp"]:
                # Partial close at first TP
                self._partial_close(pos["first_tp"], row["timestamp"], "TP1")
                pos["first_tp_hit"] = True
                # Move SL to breakeven + small profit
                pos["stop_loss"] = pos["entry_price"] * 1.005  # 0.5% above entry
                pos["trailing_active"] = True

            # Update trailing stop
            if pos["trailing_active"]:
                new_sl = pos["highest_price"] * (1 - self.trail_distance)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            # Check SL
            if row["low"] <= pos["stop_loss"]:
                reason = "TRAIL" if pos["first_tp_hit"] else "SL"
                self._exit(pos["stop_loss"], row["timestamp"], reason, 1.0)

        else:  # SHORT
            if row["low"] < pos["lowest_price"]:
                pos["lowest_price"] = row["low"]

            # Check first TP
            if not pos["first_tp_hit"] and row["low"] <= pos["first_tp"]:
                self._partial_close(pos["first_tp"], row["timestamp"], "TP1")
                pos["first_tp_hit"] = True
                pos["stop_loss"] = pos["entry_price"] * 0.995  # 0.5% below entry
                pos["trailing_active"] = True

            # Update trailing stop
            if pos["trailing_active"]:
                new_sl = pos["lowest_price"] * (1 + self.trail_distance)
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            # Check SL
            if row["high"] >= pos["stop_loss"]:
                reason = "TRAIL" if pos["first_tp_hit"] else "SL"
                self._exit(pos["stop_loss"], row["timestamp"], reason, 1.0)

    def _partial_close(self, price, time, reason):
        """Close partial position at first TP."""
        pos = self.current_position
        if not pos:
            return

        close_size = pos["size"] * self.partial_close_pct
        close_margin = pos["margin"] * self.partial_close_pct

        exit_price = price * (1 - self.slippage if pos["direction"] == "LONG" else 1 + self.slippage)

        if pos["direction"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * close_size
        else:
            pnl = (pos["entry_price"] - exit_price) * close_size

        pnl -= exit_price * close_size * self.commission
        self.capital += close_margin + pnl

        pnl_pct = pnl / close_margin * 100

        self.trades.append(Trade(
            entry_time=pos["entry_time"],
            exit_time=time,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            max_price_seen=pos["highest_price"] if pos["direction"] == "LONG" else pos["lowest_price"],
        ))

        # Update position for remainder
        pos["size"] -= close_size
        pos["margin"] -= close_margin

    def _exit(self, price, time, reason, fraction=1.0):
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

        pnl_pct = pnl / pos["margin"] * 100 if pos["margin"] > 0 else 0

        self.trades.append(Trade(
            entry_time=pos["entry_time"],
            exit_time=time,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            max_price_seen=pos["highest_price"] if pos["direction"] == "LONG" else pos["lowest_price"],
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
    data_path = Path(__file__).parent.parent / "data" / "BTCUSDT_1h_365d.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} BTC 1h candles")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    configs = [
        {
            "name": "ORIGINAL (Fixed TP 1.5x)",
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
                "first_tp_ratio": 1.5,
                "partial_close_pct": 1.0,  # 100% close at TP (no trailing)
                "trail_distance": 0.02,
            }
        },
        {
            "name": "HYBRID (50% TP + 50% Trail)",
            "signal_params": {
                "vol_multiplier": 2.5,
                "consecutive_bars": 2,
                "wick_pct": 0.45,
                "rsi_threshold": 45,
                "channel_vol_threshold": 1.0,
                "rr_ratio": 1.5,
            },
            "backtest_params": {
                "risk_per_trade": 0.25,
                "first_tp_ratio": 1.5,
                "partial_close_pct": 0.5,  # 50% at TP, 50% trail
                "trail_distance": 0.02,
            }
        },
        {
            "name": "AGGRESSIVE (30% TP + 70% Trail)",
            "signal_params": {
                "vol_multiplier": 2.0,
                "consecutive_bars": 2,
                "wick_pct": 0.40,
                "rsi_threshold": 50,
                "channel_vol_threshold": 0.8,
                "rr_ratio": 2.0,
            },
            "backtest_params": {
                "risk_per_trade": 0.30,
                "first_tp_ratio": 2.0,
                "partial_close_pct": 0.3,  # 30% at TP, 70% trail
                "trail_distance": 0.025,
            }
        },
        {
            "name": "MAX TREND (20% TP + 80% Trail, Tight)",
            "signal_params": {
                "vol_multiplier": 2.0,
                "consecutive_bars": 2,
                "wick_pct": 0.40,
                "rsi_threshold": 50,
                "channel_vol_threshold": 0.8,
                "rr_ratio": 2.0,
            },
            "backtest_params": {
                "risk_per_trade": 0.35,
                "first_tp_ratio": 1.5,
                "partial_close_pct": 0.2,  # 20% at TP, 80% trail
                "trail_distance": 0.015,   # Tighter trail
            }
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"  {config['name']}")
        print(f"{'='*80}")

        signal_gen = HighWinRateSignalGenerator(**config["signal_params"])
        signals = signal_gen.generate_from_dataframe(df)
        print(f"Signals: {len(signals)}")

        bt = HybridBacktest(**config["backtest_params"])
        result = bt.run(df, signals)

        if result:
            results.append({"name": config["name"], **result})
            print(f"Trades: {result['trades']} | WR: {result['win_rate']:.1f}%")
            print(f"Return: {result['total_return']:+.1f}% | MDD: {result['max_drawdown']:.1f}%")
            print(f"PF: {result['profit_factor']:.2f} | AvgWin: {result['avg_win_pct']:+.1f}% | AvgLoss: {result['avg_loss_pct']:.1f}%")

    # Comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON")
    print(f"{'='*100}")
    print(f"\n{'Config':<35} {'Trades':>7} {'WR':>7} {'Return':>10} {'MDD':>8} {'PF':>7} {'AvgW':>8} {'AvgL':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<35} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_return']:>+9.1f}% "
              f"{r['max_drawdown']:>7.1f}% {r['profit_factor']:>7.2f} {r['avg_win_pct']:>+7.1f}% {r['avg_loss_pct']:>7.1f}%")

    # Best result details
    best = max(results, key=lambda x: x['total_return'])
    print(f"\n{'='*100}")
    print(f"  BEST: {best['name']}")
    print(f"{'='*100}")

    print(f"\n{'#':>3} {'Date':<12} {'Dir':<6} {'Exit':<6} {'PnL%':>10} {'PnL$':>12}")
    print("-" * 60)

    for i, t in enumerate(best['trade_list'][:40], 1):
        print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} "
              f"{t.exit_reason:<6} {t.pnl_pct:>+9.1f}% {t.pnl:>+11.1f}")

    if len(best['trade_list']) > 40:
        print(f"... +{len(best['trade_list']) - 40} more trades")

    # Monthly for best
    print(f"\n{'='*100}")
    print(f"  MONTHLY: {best['name']}")
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
    print(f"\n{'Month':<10} {'Trades':>6} {'WR':>6} {'PnL$':>12} {'Return':>10} {'Capital':>12}")
    print("-" * 65)

    for month in sorted(monthly.keys()):
        data = monthly[month]
        start_cap = capital
        capital += data['pnl']
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        ret = data['pnl'] / start_cap * 100
        print(f"{month:<10} {data['trades']:>6} {wr:>5.0f}% {data['pnl']:>+11.0f} {ret:>+9.1f}% {capital:>11,.0f}")

    print(f"\n{'TOTAL':<10} {len(best['trade_list']):>6} {best['win_rate']:>5.1f}% {best['total_return']*100:>+11.0f} "
          f"{best['total_return']:>+9.1f}% {best['final_capital']:>11,.0f}")


if __name__ == "__main__":
    main()
