"""Optimized backtest: High WR signals + optimized risk + fixed TP."""

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


class OptimizedBacktest:
    """Backtest with fixed TP (proven to work)."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.25,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        rr_ratio: float = 1.5,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.rr_ratio = rr_ratio

    def run(self, df: pd.DataFrame, signals: list) -> dict:
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row["timestamp"]

            if self.current_position:
                self._check_exit(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if not self.current_position:
                    self._enter(signal, row)

            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(row["close"])
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]["close"], df.iloc[-1]["timestamp"], "END")

        return self._calculate_metrics()

    def _enter(self, signal, row):
        entry_price = row['close'] * (1 + self.slippage if signal.direction == "LONG" else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price

        if sl_distance == 0 or sl_distance > 0.05:
            return

        # Calculate TP based on R:R ratio
        tp_distance = sl_distance * self.rr_ratio
        if signal.direction == "LONG":
            take_profit = entry_price * (1 + tp_distance)
        else:
            take_profit = entry_price * (1 - tp_distance)

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
            "entry_time": row['timestamp'],
            "size": size,
            "margin": margin,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "take_profit": take_profit,
        }

    def _check_exit(self, row):
        pos = self.current_position
        if not pos:
            return

        if pos["direction"] == "LONG":
            # Check SL first (more conservative)
            if row["low"] <= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            # Then TP
            elif row["high"] >= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")
        else:
            if row["high"] >= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            elif row["low"] <= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")

    def _exit(self, price, time, reason):
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

    def _calculate_metrics(self):
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

    # Signal configurations to test
    signal_configs = [
        {
            "name": "STRICT",
            "params": {
                "vol_multiplier": 3.0,
                "consecutive_bars": 3,
                "wick_pct": 0.50,
                "rsi_threshold": 40,
                "channel_vol_threshold": 1.2,
                "rr_ratio": 1.5,
            }
        },
        {
            "name": "RELAXED",
            "params": {
                "vol_multiplier": 2.0,
                "consecutive_bars": 2,
                "wick_pct": 0.40,
                "rsi_threshold": 50,
                "channel_vol_threshold": 0.8,
                "rr_ratio": 1.5,
            }
        },
    ]

    # Backtest configurations
    backtest_configs = [
        {"name": "15% Risk, R:R 1.5", "risk": 0.15, "rr": 1.5},
        {"name": "25% Risk, R:R 1.5", "risk": 0.25, "rr": 1.5},
        {"name": "30% Risk, R:R 1.5", "risk": 0.30, "rr": 1.5},
        {"name": "25% Risk, R:R 2.0", "risk": 0.25, "rr": 2.0},
        {"name": "30% Risk, R:R 2.0", "risk": 0.30, "rr": 2.0},
        {"name": "35% Risk, R:R 2.0", "risk": 0.35, "rr": 2.0},
        {"name": "30% Risk, R:R 2.5", "risk": 0.30, "rr": 2.5},
        {"name": "35% Risk, R:R 2.5", "risk": 0.35, "rr": 2.5},
    ]

    all_results = []

    for sig_cfg in signal_configs:
        print(f"\n{'='*100}")
        print(f"  SIGNAL CONFIG: {sig_cfg['name']}")
        print(f"{'='*100}")

        signal_gen = HighWinRateSignalGenerator(**sig_cfg["params"])
        signals = signal_gen.generate_from_dataframe(df)
        print(f"Signals generated: {len(signals)}")

        for bt_cfg in backtest_configs:
            bt = OptimizedBacktest(
                risk_per_trade=bt_cfg["risk"],
                rr_ratio=bt_cfg["rr"],
            )
            result = bt.run(df, signals)

            if result:
                result_name = f"{sig_cfg['name']} + {bt_cfg['name']}"
                all_results.append({"name": result_name, "sig": sig_cfg['name'], **result})

    # Sort by return
    all_results.sort(key=lambda x: x['total_return'], reverse=True)

    # Show top 10
    print(f"\n{'='*120}")
    print(f"  TOP 10 CONFIGURATIONS BY RETURN")
    print(f"{'='*120}")
    print(f"\n{'#':>2} {'Config':<40} {'Trades':>7} {'WR':>7} {'Return':>12} {'MDD':>10} {'PF':>7}")
    print("-" * 100)

    for i, r in enumerate(all_results[:10], 1):
        print(f"{i:>2} {r['name']:<40} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['total_return']:>+11.1f}% {r['max_drawdown']:>9.1f}% {r['profit_factor']:>7.2f}")

    # Show best detail
    best = all_results[0]
    print(f"\n{'='*100}")
    print(f"  BEST: {best['name']}")
    print(f"{'='*100}")
    print(f"\nTrades: {best['trades']} | Wins: {best['wins']} | Losses: {best['losses']}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Total Return: {best['total_return']:+.1f}%")
    print(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"Profit Factor: {best['profit_factor']:.2f}")
    print(f"Avg Win: {best['avg_win_pct']:+.1f}% | Avg Loss: {best['avg_loss_pct']:.1f}%")
    print(f"Final Capital: ${best['final_capital']:,.0f}")

    # Trade details
    print(f"\n{'#':>3} {'Date':<12} {'Dir':<6} {'Exit':<4} {'PnL%':>10} {'PnL$':>12}")
    print("-" * 60)

    for i, t in enumerate(best['trade_list'], 1):
        print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} "
              f"{t.exit_reason:<4} {t.pnl_pct:>+9.1f}% {t.pnl:>+11.1f}")

    # Monthly
    print(f"\n{'='*100}")
    print(f"  MONTHLY BREAKDOWN")
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
    print(f"\n{'Month':<10} {'Trades':>6} {'WR':>6} {'PnL$':>12} {'Return':>10} {'Capital':>14}")
    print("-" * 70)

    for month in sorted(monthly.keys()):
        data = monthly[month]
        start_cap = capital
        capital += data['pnl']
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        ret = data['pnl'] / start_cap * 100
        print(f"{month:<10} {data['trades']:>6} {wr:>5.0f}% {data['pnl']:>+11.0f} {ret:>+9.1f}% {capital:>13,.0f}")

    print("-" * 70)
    print(f"{'TOTAL':<10} {len(best['trade_list']):>6} {best['win_rate']:>5.1f}% "
          f"{capital - 10000:>+11.0f} {best['total_return']:>+9.1f}% {capital:>13,.0f}")


if __name__ == "__main__":
    main()
