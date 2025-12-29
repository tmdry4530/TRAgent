"""
Test relaxed parameters to increase trade count while maintaining high WR.
Goal: 70%+ WR, 100%+ Return, More trades
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
                 commission=0.0004, slippage=0.0002):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage

    def run(self, df, signals):
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row['timestamp']
            if self.current_position:
                self._check_exit(row)
            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1
                if not self.current_position:
                    self._enter(signal, row['close'], current_time)

            equity = self.capital + (self._unrealized_pnl(row['close']) if self.current_position else 0)
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal, price, time):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)
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
            'entry_price': entry_price, 'entry_time': time, 'size': size,
            'margin': margin, 'direction': signal.direction,
            'stop_loss': signal.stop_loss, 'take_profit': signal.take_profit,
        }

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
        return {
            'trades': len(self.trades), 'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_return': total_return, 'profit_factor': pf,
            'max_drawdown': max_dd, 'final': self.capital, 'trade_list': self.trades,
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

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

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


def generate_signals_with_params(df, vol_mult, consec, wick_pct, rsi_threshold, vol_thresh, rr):
    """Generate Volume Climax + Channel Bounce signals with custom params."""
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

        # Volume Climax
        consec_down = sum(1 for j in range(1, min(consec + 1, i)) if df.iloc[i-j]['close'] < df.iloc[i-j]['open'])
        consec_up = sum(1 for j in range(1, min(consec + 1, i)) if df.iloc[i-j]['close'] > df.iloc[i-j]['open'])

        # LONG: Volume Climax
        if (vol_ratio >= vol_mult and consec_down >= consec and lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and rsi < rsi_threshold):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.90,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol Climax", timestamp=row['timestamp'],
            ))

        # SHORT: Volume Climax
        if (vol_ratio >= vol_mult and consec_up >= consec and upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and rsi > (100 - rsi_threshold)):
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

    # Parameter grid search
    print("\n" + "=" * 120)
    print("  PARAMETER OPTIMIZATION: Finding 70%+ WR with most trades")
    print("=" * 120)

    results = []

    # Grid search
    for vol_mult in [2.0, 2.5, 3.0]:
        for consec in [2, 3]:
            for wick_pct in [0.35, 0.40, 0.45, 0.50]:
                for rsi_thresh in [40, 45]:
                    for vol_thresh in [1.2, 1.5]:
                        for rr in [1.5, 2.0]:
                            signals = generate_signals_with_params(
                                df, vol_mult, consec, wick_pct, rsi_thresh, vol_thresh, rr
                            )

                            if len(signals) < 3:
                                continue

                            bt = Backtest(risk_per_trade=0.15)
                            result = bt.run(df, signals)

                            if result and result['win_rate'] >= 70 and result['total_return'] > 0:
                                results.append({
                                    'vol_mult': vol_mult,
                                    'consec': consec,
                                    'wick_pct': wick_pct,
                                    'rsi_thresh': rsi_thresh,
                                    'vol_thresh': vol_thresh,
                                    'rr': rr,
                                    **result
                                })

    # Sort by trades (more trades better)
    results = sorted(results, key=lambda x: x['trades'], reverse=True)

    print(f"\nFound {len(results)} configs with WR >= 70% and positive return")

    if results:
        print(f"\n{'Vol':>5} {'Con':>4} {'Wick':>5} {'RSI':>4} {'VolTh':>5} {'R:R':>4} {'Trades':>7} {'Win%':>7} {'Return':>9} {'MDD':>7}")
        print("-" * 75)

        for r in results[:20]:  # Top 20
            print(f"{r['vol_mult']:>5.1f} {r['consec']:>4} {r['wick_pct']:>5.2f} {r['rsi_thresh']:>4} {r['vol_thresh']:>5.1f} {r['rr']:>4.1f} "
                  f"{r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_return']:>+8.1f}% {r['max_drawdown']:>6.1f}%")

        # Best by return among high trade counts
        best = results[0]  # Most trades
        print("\n" + "=" * 120)
        print("  BEST CONFIG (Most trades with WR >= 70%)")
        print("=" * 120)
        print(f"Volume Multiplier: {best['vol_mult']}")
        print(f"Consecutive bars: {best['consec']}")
        print(f"Wick %: {best['wick_pct']}")
        print(f"RSI Threshold: {best['rsi_thresh']}")
        print(f"Volume Threshold: {best['vol_thresh']}")
        print(f"R:R Ratio: {best['rr']}")
        print(f"\nTrades: {best['trades']}")
        print(f"Win Rate: {best['win_rate']:.1f}%")
        print(f"Return: {best['total_return']:+.1f}%")
        print(f"Max DD: {best['max_drawdown']:.1f}%")

        # Test best config with different risk levels
        print("\n" + "-" * 80)
        print("  Best config at different risk levels:")
        print("-" * 80)

        signals = generate_signals_with_params(
            df, best['vol_mult'], best['consec'], best['wick_pct'],
            best['rsi_thresh'], best['vol_thresh'], best['rr']
        )

        print(f"\n{'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8}")
        print("-" * 50)

        for risk in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            bt = Backtest(risk_per_trade=risk)
            result = bt.run(df, signals)
            if result:
                print(f"{risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}%")

    else:
        print("\nNo configurations found with WR >= 70%")

        # Try looser criteria
        print("\n" + "=" * 120)
        print("  RELAXED SEARCH: WR >= 60%")
        print("=" * 120)

        results = []
        for vol_mult in [2.0, 2.5]:
            for consec in [2]:
                for wick_pct in [0.30, 0.35, 0.40]:
                    for rsi_thresh in [45, 50]:
                        for vol_thresh in [1.0, 1.2]:
                            for rr in [1.5, 2.0]:
                                signals = generate_signals_with_params(
                                    df, vol_mult, consec, wick_pct, rsi_thresh, vol_thresh, rr
                                )

                                if len(signals) < 5:
                                    continue

                                bt = Backtest(risk_per_trade=0.15)
                                result = bt.run(df, signals)

                                if result and result['win_rate'] >= 60 and result['total_return'] > 0:
                                    results.append({
                                        'vol_mult': vol_mult,
                                        'consec': consec,
                                        'wick_pct': wick_pct,
                                        'rsi_thresh': rsi_thresh,
                                        'vol_thresh': vol_thresh,
                                        'rr': rr,
                                        **result
                                    })

        results = sorted(results, key=lambda x: (x['win_rate'], x['trades']), reverse=True)

        if results:
            print(f"\nFound {len(results)} configs with WR >= 60%")
            print(f"\n{'Vol':>5} {'Con':>4} {'Wick':>5} {'RSI':>4} {'VolTh':>5} {'R:R':>4} {'Trades':>7} {'Win%':>7} {'Return':>9} {'MDD':>7}")
            print("-" * 75)

            for r in results[:15]:
                print(f"{r['vol_mult']:>5.1f} {r['consec']:>4} {r['wick_pct']:>5.2f} {r['rsi_thresh']:>4} {r['vol_thresh']:>5.1f} {r['rr']:>4.1f} "
                      f"{r['trades']:>7} {r['win_rate']:>6.1f}% {r['total_return']:>+8.1f}% {r['max_drawdown']:>6.1f}%")


if __name__ == "__main__":
    main()
