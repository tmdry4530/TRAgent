"""
Final Configuration Test - 80% WR + 100%+ Return with Daily Limit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from collections import defaultdict
from src.signals.base import Signal


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str


class Backtest:
    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0002, max_daily_trades=999):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.max_daily_trades = max_daily_trades

    def run(self, df, signals):
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]
        self.daily_trades = defaultdict(int)

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row['timestamp']
            current_date = current_time.date()

            if self.current_position:
                self._check_exit(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                # Check daily limit
                if self.daily_trades[current_date] >= self.max_daily_trades:
                    continue

                if not self.current_position:
                    if self._enter(signal, row['close'], current_time):
                        self.daily_trades[current_date] += 1

            equity = self.capital + (self._unrealized_pnl(row['close']) if self.current_position else 0)
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal, price, time):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price
        if sl_distance == 0 or sl_distance > 0.05:
            return False

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
            return False
        self.capital -= (margin + entry_comm)

        self.current_position = {
            'entry_price': entry_price, 'entry_time': time, 'size': size,
            'margin': margin, 'direction': signal.direction,
            'stop_loss': signal.stop_loss, 'take_profit': signal.take_profit,
        }
        return True

    def _check_exit(self, row):
        pos = self.current_position
        if not pos:
            return
        if pos['direction'] == 'LONG':
            if row['low'] <= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif row['high'] >= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')
        else:
            if row['high'] >= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif row['low'] <= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')

    def _exit(self, price, time, reason):
        pos = self.current_position
        if not pos:
            return
        exit_price = price * (1 - self.slippage if pos['direction'] == 'LONG' else 1 + self.slippage)
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']
        pnl -= exit_price * pos['size'] * self.commission
        self.capital += pos['margin'] + pnl

        self.trades.append(Trade(
            pos['entry_time'], time, pos['direction'],
            pos['entry_price'], exit_price, pnl, reason
        ))
        self.current_position = None

    def _unrealized_pnl(self, price):
        pos = self.current_position
        if not pos:
            return 0
        return (price - pos['entry_price']) * pos['size'] if pos['direction'] == 'LONG' else (pos['entry_price'] - price) * pos['size']

    def _metrics(self):
        if not self.trades:
            return None
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))

        # Unique trading days
        unique_days = len(set(t.entry_time.date() for t in self.trades))
        trades_per_day = len(self.trades) / unique_days if unique_days > 0 else 0

        return {
            'trades': len(self.trades), 'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_return': total_return, 'profit_factor': pf,
            'max_drawdown': max_dd, 'final': self.capital,
            'trade_list': self.trades,
            'trades_per_day': trades_per_day,
            'trading_days': unique_days
        }


def add_indicators(df):
    df = df.copy()

    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()

    for span in [10, 20, 50, 100, 200]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    df['dc_upper'] = df['high'].rolling(20).max()
    df['dc_lower'] = df['low'].rolling(20).min()
    df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2

    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    return df


def generate_signals(df, vol_mult=3.0, consec=3, wick_pct=0.45, rsi_thresh=40, vol_thresh=1.2, rr=1.5):
    """Generate high win rate signals."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['volume_ratio']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Volume Climax check
        consec_down = sum(1 for j in range(1, min(consec + 1, i)) if df.iloc[i-j]['close'] < df.iloc[i-j]['open'])
        consec_up = sum(1 for j in range(1, min(consec + 1, i)) if df.iloc[i-j]['close'] > df.iloc[i-j]['open'])

        # LONG: Volume Climax
        if (vol_ratio >= vol_mult and consec_down >= consec and lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and rsi < rsi_thresh):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.90,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol Climax", timestamp=row['timestamp'],
            ))

        # SHORT: Volume Climax
        if (vol_ratio >= vol_mult and consec_up >= consec and upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and rsi > (100 - rsi_thresh)):
            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr
            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.90,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol Climax", timestamp=row['timestamp'],
            ))

        # Channel Bounce
        uptrend = price > row['ema50'] if not pd.isna(row['ema50']) else True
        downtrend = price < row['ema50'] if not pd.isna(row['ema50']) else True

        # LONG: Channel Bounce
        if (uptrend and
            row['low'] <= row['dc_lower'] * 1.005 and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= vol_thresh and
            rsi < 50):

            sl = row['dc_lower'] - atr * 0.1
            tp = row['dc_mid']

            if tp > price:
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=0.85,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel Bounce", timestamp=row['timestamp'],
                ))

        # SHORT: Channel Bounce
        if (downtrend and
            row['high'] >= row['dc_upper'] * 0.995 and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            vol_ratio >= vol_thresh and
            rsi > 50):

            sl = row['dc_upper'] + atr * 0.1
            tp = row['dc_mid']

            if tp < price:
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=0.85,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel Bounce", timestamp=row['timestamp'],
                ))

    return signals


def main():
    data_path = Path(__file__).parent.parent / "data" / "BTCUSDT_1h_365d.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(df)} BTC 1h candles")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    df = add_indicators(df)

    # Best configurations found
    configs = [
        {"name": "Config A: 80% WR", "vol_mult": 3.0, "consec": 3, "wick_pct": 0.50, "rsi_thresh": 40, "vol_thresh": 1.2, "rr": 1.5},
        {"name": "Config B: 79% WR", "vol_mult": 3.0, "consec": 3, "wick_pct": 0.45, "rsi_thresh": 40, "vol_thresh": 1.2, "rr": 1.5},
        {"name": "Config C: 77% WR", "vol_mult": 3.0, "consec": 2, "wick_pct": 0.45, "rsi_thresh": 40, "vol_thresh": 1.2, "rr": 1.5},
        {"name": "Config D: 80% WR Strict", "vol_mult": 3.0, "consec": 2, "wick_pct": 0.50, "rsi_thresh": 40, "vol_thresh": 1.5, "rr": 1.5},
    ]

    print("\n" + "=" * 120)
    print("  FINAL CONFIGURATIONS TEST - 80% WR + 100%+ RETURN")
    print("=" * 120)

    for config in configs:
        name = config.pop("name")
        signals = generate_signals(df, **config)

        print(f"\n{'-' * 100}")
        print(f"  {name}")
        print(f"  Settings: Vol {config['vol_mult']}x, Consec {config['consec']}, Wick {config['wick_pct']*100:.0f}%, RSI {config['rsi_thresh']}, VolTh {config['vol_thresh']}x, R:R {config['rr']}")
        print(f"  Total signals: {len(signals)}")
        print(f"{'-' * 100}")

        print(f"\n{'Daily Limit':>12} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6} {'Days':>6} {'T/Day':>6}")
        print("-" * 85)

        for daily_limit in [1, 2, 3, 999]:
            for risk in [0.10, 0.15, 0.20, 0.25, 0.30]:
                bt = Backtest(risk_per_trade=risk, max_daily_trades=daily_limit)
                result = bt.run(df, signals)

                if result:
                    dl_str = "∞" if daily_limit == 999 else str(daily_limit)
                    mark = " ***" if result['win_rate'] >= 75 and result['total_return'] >= 100 else ""
                    print(f"{dl_str:>12} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% "
                          f"{result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}% {result['profit_factor']:>6.2f} "
                          f"{result['trading_days']:>6} {result['trades_per_day']:>5.2f}{mark}")

        # Reset config name
        config["name"] = name

    # Summary of best results
    print("\n" + "=" * 120)
    print("  RECOMMENDED CONFIGURATIONS")
    print("=" * 120)

    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  CONSERVATIVE (안정적)                                                       │
    │  - Config A (80% WR) + Risk 15% + Daily Limit 3                             │
    │  - Expected: ~20 trades, 80% WR, ~100% return                               │
    │  - MDD: ~98% (High risk!)                                                   │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │  AGGRESSIVE (공격적)                                                          │
    │  - Config B (79% WR) + Risk 25% + No Limit                                  │
    │  - Expected: ~24 trades, 79% WR, ~170% return                               │
    │  - MDD: ~111% (Very high risk!)                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

    NOTE: MDD > 100% means account can be wiped out.
    Consider reducing risk per trade or using stricter position sizing.
    """)

    # Show individual trades for best config
    print("\n" + "=" * 120)
    print("  TRADE DETAILS - Config A (80% WR) @ 15% Risk")
    print("=" * 120)

    config = {"vol_mult": 3.0, "consec": 3, "wick_pct": 0.50, "rsi_thresh": 40, "vol_thresh": 1.2, "rr": 1.5}
    signals = generate_signals(df, **config)

    bt = Backtest(risk_per_trade=0.15)
    result = bt.run(df, signals)

    if result and 'trade_list' in result:
        print(f"\nTotal: {result['trades']} trades, {result['win_rate']:.1f}% WR, {result['total_return']:+.1f}% return\n")

        print(f"{'#':>3} {'Date':<12} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'Result':<6} {'PnL':>12}")
        print("-" * 75)

        for i, t in enumerate(result['trade_list'], 1):
            result_str = "WIN" if t.pnl > 0 else "LOSS"
            print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} {t.entry_price:>10.1f} {t.exit_price:>10.1f} {result_str:<6} {t.pnl:>+11.1f}")


if __name__ == "__main__":
    main()
