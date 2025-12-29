"""Test trend-following signals with trailing stop."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List


@dataclass
class Signal:
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    stop_loss: float
    signal_type: str
    strength: float = 1.0


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
    signal_type: str


class TrendSignalGenerator:
    """Generate trend-following signals: breakouts, momentum, EMA crossovers."""

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        atr_period: int = 14,
        atr_sl_mult: float = 2.0,
        breakout_period: int = 20,
        volume_confirm: float = 1.2,
        rsi_period: int = 14,
        rsi_momentum_threshold: float = 55,
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.breakout_period = breakout_period
        self.volume_confirm = volume_confirm
        self.rsi_period = rsi_period
        self.rsi_momentum_threshold = rsi_momentum_threshold

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        df = df.copy()

        # Calculate indicators
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()

        # ATR for stop loss
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(self.atr_period).mean()

        # Breakout levels
        df['high_break'] = df['high'].rolling(self.breakout_period).max().shift(1)
        df['low_break'] = df['low'].rolling(self.breakout_period).min().shift(1)

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Trend direction (EMA)
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df['trend_change'] = df['trend'] != df['trend'].shift(1)

        signals = []

        for i in range(max(self.breakout_period, self.ema_slow, 50), len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            # Skip if ATR is invalid
            if pd.isna(row['atr']) or row['atr'] == 0:
                continue

            atr = row['atr']

            # === SIGNAL 1: Breakout with volume ===
            # Long: price breaks above recent high with volume
            if row['close'] > prev['high_break'] and row['vol_ratio'] > self.volume_confirm:
                if row['trend'] == 1:  # Confirm with trend
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="LONG",
                        entry_price=row['close'],
                        stop_loss=row['close'] - atr * self.atr_sl_mult,
                        signal_type="BREAKOUT",
                        strength=row['vol_ratio'],
                    ))

            # Short: price breaks below recent low with volume
            if row['close'] < prev['low_break'] and row['vol_ratio'] > self.volume_confirm:
                if row['trend'] == -1:
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="SHORT",
                        entry_price=row['close'],
                        stop_loss=row['close'] + atr * self.atr_sl_mult,
                        signal_type="BREAKOUT",
                        strength=row['vol_ratio'],
                    ))

            # === SIGNAL 2: EMA Crossover ===
            if row['trend_change']:
                if row['trend'] == 1 and row['rsi'] > self.rsi_momentum_threshold:
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="LONG",
                        entry_price=row['close'],
                        stop_loss=row['close'] - atr * self.atr_sl_mult,
                        signal_type="EMA_CROSS",
                        strength=1.0,
                    ))
                elif row['trend'] == -1 and row['rsi'] < (100 - self.rsi_momentum_threshold):
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="SHORT",
                        entry_price=row['close'],
                        stop_loss=row['close'] + atr * self.atr_sl_mult,
                        signal_type="EMA_CROSS",
                        strength=1.0,
                    ))

            # === SIGNAL 3: Momentum continuation ===
            # Strong RSI + price above EMA + volume
            if (row['rsi'] > 60 and row['close'] > row['ema_fast'] > row['ema_slow']
                and row['vol_ratio'] > 1.0):
                # Check for pullback entry (price touched EMA fast recently)
                recent_lows = df['low'].iloc[i-5:i]
                if recent_lows.min() <= df['ema_fast'].iloc[i-5:i].mean():
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="LONG",
                        entry_price=row['close'],
                        stop_loss=row['close'] - atr * self.atr_sl_mult,
                        signal_type="MOMENTUM",
                        strength=row['rsi'] / 100,
                    ))

            if (row['rsi'] < 40 and row['close'] < row['ema_fast'] < row['ema_slow']
                and row['vol_ratio'] > 1.0):
                recent_highs = df['high'].iloc[i-5:i]
                if recent_highs.max() >= df['ema_fast'].iloc[i-5:i].mean():
                    signals.append(Signal(
                        timestamp=row['timestamp'],
                        direction="SHORT",
                        entry_price=row['close'],
                        stop_loss=row['close'] + atr * self.atr_sl_mult,
                        signal_type="MOMENTUM",
                        strength=(100 - row['rsi']) / 100,
                    ))

        return signals


class TrendBacktest:
    """Backtest with pure trailing stop for trend following."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.25,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        trail_activation: float = 0.01,  # Activate after 1% profit
        trail_distance: float = 0.02,    # 2% trailing stop
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.trail_activation = trail_activation
        self.trail_distance = trail_distance

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

        if sl_distance == 0 or sl_distance > 0.10:
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
            "entry_time": row['timestamp'],
            "size": size,
            "margin": margin,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "trailing_active": False,
            "best_price": entry_price,
            "signal_type": signal.signal_type,
        }

    def _check_exit(self, row):
        pos = self.current_position
        if not pos:
            return

        if pos["direction"] == "LONG":
            if row["high"] > pos["best_price"]:
                pos["best_price"] = row["high"]

            profit_pct = (row["close"] - pos["entry_price"]) / pos["entry_price"]

            if profit_pct >= self.trail_activation and not pos["trailing_active"]:
                pos["trailing_active"] = True
                pos["stop_loss"] = pos["best_price"] * (1 - self.trail_distance)

            if pos["trailing_active"]:
                new_sl = pos["best_price"] * (1 - self.trail_distance)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            if row["low"] <= pos["stop_loss"]:
                reason = "TRAIL" if pos["trailing_active"] else "SL"
                self._exit(pos["stop_loss"], row["timestamp"], reason)

        else:
            if row["low"] < pos["best_price"]:
                pos["best_price"] = row["low"]

            profit_pct = (pos["entry_price"] - row["close"]) / pos["entry_price"]

            if profit_pct >= self.trail_activation and not pos["trailing_active"]:
                pos["trailing_active"] = True
                pos["stop_loss"] = pos["best_price"] * (1 + self.trail_distance)

            if pos["trailing_active"]:
                new_sl = pos["best_price"] * (1 + self.trail_distance)
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl

            if row["high"] >= pos["stop_loss"]:
                reason = "TRAIL" if pos["trailing_active"] else "SL"
                self._exit(pos["stop_loss"], row["timestamp"], reason)

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
            signal_type=pos["signal_type"],
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

        # By signal type
        by_type = {}
        for t in self.trades:
            if t.signal_type not in by_type:
                by_type[t.signal_type] = {"trades": 0, "wins": 0, "pnl": 0}
            by_type[t.signal_type]["trades"] += 1
            if t.pnl > 0:
                by_type[t.signal_type]["wins"] += 1
            by_type[t.signal_type]["pnl"] += t.pnl

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
            "by_type": by_type,
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
            "name": "CONSERVATIVE TREND",
            "signal_params": {
                "ema_fast": 20,
                "ema_slow": 50,
                "atr_sl_mult": 2.0,
                "breakout_period": 20,
                "volume_confirm": 1.5,
            },
            "backtest_params": {
                "risk_per_trade": 0.20,
                "trail_activation": 0.015,
                "trail_distance": 0.025,
            }
        },
        {
            "name": "BALANCED TREND",
            "signal_params": {
                "ema_fast": 12,
                "ema_slow": 26,
                "atr_sl_mult": 1.5,
                "breakout_period": 14,
                "volume_confirm": 1.2,
            },
            "backtest_params": {
                "risk_per_trade": 0.25,
                "trail_activation": 0.01,
                "trail_distance": 0.02,
            }
        },
        {
            "name": "AGGRESSIVE TREND",
            "signal_params": {
                "ema_fast": 9,
                "ema_slow": 21,
                "atr_sl_mult": 1.5,
                "breakout_period": 10,
                "volume_confirm": 1.0,
            },
            "backtest_params": {
                "risk_per_trade": 0.30,
                "trail_activation": 0.008,
                "trail_distance": 0.015,
            }
        },
        {
            "name": "TIGHT TRAIL",
            "signal_params": {
                "ema_fast": 12,
                "ema_slow": 26,
                "atr_sl_mult": 1.2,
                "breakout_period": 14,
                "volume_confirm": 1.2,
            },
            "backtest_params": {
                "risk_per_trade": 0.30,
                "trail_activation": 0.005,
                "trail_distance": 0.01,
            }
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"  {config['name']}")
        print(f"{'='*80}")

        sig_gen = TrendSignalGenerator(**config["signal_params"])
        signals = sig_gen.generate(df)
        print(f"Signals: {len(signals)}")

        if signals:
            by_type = {}
            for s in signals:
                by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1
            print(f"  By type: {by_type}")

        bt = TrendBacktest(**config["backtest_params"])
        result = bt.run(df, signals)

        if result:
            results.append({"name": config["name"], **result})
            print(f"\nTrades: {result['trades']} | WR: {result['win_rate']:.1f}%")
            print(f"Return: {result['total_return']:+.1f}% | MDD: {result['max_drawdown']:.1f}%")
            print(f"PF: {result['profit_factor']:.2f} | AvgWin: {result['avg_win_pct']:+.1f}% | AvgLoss: {result['avg_loss_pct']:.1f}%")

            print(f"\nBy Signal Type:")
            for stype, data in result['by_type'].items():
                wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
                print(f"  {stype}: {data['trades']} trades, {wr:.0f}% WR, ${data['pnl']:+,.0f}")

    # Comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON")
    print(f"{'='*100}")
    print(f"\n{'Config':<25} {'Trades':>7} {'WR':>7} {'Return':>12} {'MDD':>10} {'PF':>7} {'AvgW':>9} {'AvgL':>9}")
    print("-" * 100)

    for r in results:
        print(f"{r['name']:<25} {r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_return']:>+11.1f}% "
              f"{r['max_drawdown']:>9.1f}% {r['profit_factor']:>7.2f} {r['avg_win_pct']:>+8.1f}% {r['avg_loss_pct']:>8.1f}%")

    # Best result
    if results:
        best = max(results, key=lambda x: x['total_return'])
        print(f"\n{'='*100}")
        print(f"  BEST: {best['name']} ({best['total_return']:+.1f}%)")
        print(f"{'='*100}")

        # Monthly
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


if __name__ == "__main__":
    main()
