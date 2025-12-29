"""
Original High Win Rate Strategy Test with Daily Limit
Uses exact same logic from test_combined_high_wr.py
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


def volume_climax_signals(df, vol_mult=3.0, consec=3, wick_pct=0.50, rr=1.5):
    """100% WR Volume Climax signals - ORIGINAL LOGIC."""
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

        consec_down = all(df.iloc[i-j]['close'] < df.iloc[i-j]['open'] for j in range(1, consec + 1))
        consec_up = all(df.iloc[i-j]['close'] > df.iloc[i-j]['open'] for j in range(1, consec + 1))

        # LONG: After consecutive down + volume climax + lower wick + bullish close + RSI oversold
        if (vol_ratio >= vol_mult and consec_down and lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and rsi < 35):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.90,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol Climax", timestamp=row['timestamp'],
            ))

        # SHORT: After consecutive up + volume climax + upper wick + bearish close + RSI overbought
        if (vol_ratio >= vol_mult and consec_up and upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and rsi > 65):
            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr
            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.90,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol Climax", timestamp=row['timestamp'],
            ))

    return signals


def channel_bounce_signals(df, wick_pct=0.50, rr=1.5):
    """100% WR Channel Bounce signals - ORIGINAL LOGIC."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['dc_lower']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        uptrend = price > row['ema50'] if not pd.isna(row['ema50']) else True
        downtrend = price < row['ema50'] if not pd.isna(row['ema50']) else True

        # LONG: Touch channel lower in uptrend with wick
        if (uptrend and
            row['low'] <= row['dc_lower'] * 1.002 and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= 1.5 and
            rsi < 45):

            sl = row['dc_lower'] - atr * 0.1
            tp = row['dc_mid']

            if tp > price:
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=0.85,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel Bounce", timestamp=row['timestamp'],
                ))

        # SHORT: Touch channel upper in downtrend with wick
        if (downtrend and
            row['high'] >= row['dc_upper'] * 0.998 and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            vol_ratio >= 1.5 and
            rsi > 55):

            sl = row['dc_upper'] + atr * 0.1
            tp = row['dc_mid']

            if tp < price:
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=0.85,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel Bounce", timestamp=row['timestamp'],
                ))

    return signals


def swing_signals(df, rr=3.0):
    """4H EMA Swing Strategy (for comparison)."""
    signals = []

    # Resample to 4H
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()

    df_4h['ema20'] = df_4h['close'].ewm(span=20).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50).mean()
    df_4h['atr'] = (df_4h['high'] - df_4h['low']).rolling(14).mean()
    df_4h['body'] = abs(df_4h['close'] - df_4h['open'])
    df_4h['range'] = df_4h['high'] - df_4h['low']
    df_4h['lower_wick'] = df_4h[['open', 'close']].min(axis=1) - df_4h['low']
    df_4h['upper_wick'] = df_4h['high'] - df_4h[['open', 'close']].max(axis=1)
    df_4h['lower_wick_pct'] = df_4h['lower_wick'] / df_4h['range'].replace(0, np.nan)
    df_4h['upper_wick_pct'] = df_4h['upper_wick'] / df_4h['range'].replace(0, np.nan)

    for i in range(50, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i-1]
        price = row['close']

        if pd.isna(row['ema20']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # LONG: Uptrend + pullback to EMA + wick rejection
        if (uptrend and
            row['low'] <= row['ema20'] * 1.01 and
            lower_wick_pct >= 0.4 and
            row['close'] > row['open']):

            sl = row['low'] - atr * 0.2
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Swing EMA", timestamp=row['timestamp'],
            ))

        # SHORT: Downtrend + pullback to EMA + wick rejection
        if (downtrend and
            row['high'] >= row['ema20'] * 0.99 and
            upper_wick_pct >= 0.4 and
            row['close'] < row['open']):

            sl = row['high'] + atr * 0.2
            tp = price - (sl - price) * rr
            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Swing EMA", timestamp=row['timestamp'],
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

    # Generate signals
    vol_signals = volume_climax_signals(df)
    channel_signals = channel_bounce_signals(df)
    swing_sigs = swing_signals(df)

    print(f"\nVolume Climax signals: {len(vol_signals)}")
    print(f"Channel Bounce signals: {len(channel_signals)}")
    print(f"Swing signals: {len(swing_sigs)}")

    # Test individual strategies without daily limit
    print("\n" + "=" * 100)
    print("  INDIVIDUAL STRATEGIES (No Daily Limit)")
    print("=" * 100)

    risk_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"\n{'Strategy':<25} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6}")
    print("-" * 85)

    for risk in risk_levels:
        bt = Backtest(risk_per_trade=risk)
        result = bt.run(df, vol_signals)
        if result:
            print(f"{'Volume Climax':<25} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}% {result['profit_factor']:>6.2f}")

    print()
    for risk in risk_levels:
        bt = Backtest(risk_per_trade=risk)
        result = bt.run(df, channel_signals)
        if result:
            print(f"{'Channel Bounce':<25} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}% {result['profit_factor']:>6.2f}")

    print()
    for risk in risk_levels:
        bt = Backtest(risk_per_trade=risk)
        result = bt.run(df, swing_sigs)
        if result:
            print(f"{'Swing 4H EMA':<25} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}% {result['profit_factor']:>6.2f}")

    # Combined signals
    all_signals = vol_signals + channel_signals
    print("\n" + "=" * 100)
    print(f"  COMBINED (Vol Climax + Channel Bounce) = {len(all_signals)} signals")
    print("=" * 100)

    print(f"\n{'Daily Limit':>12} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6}")
    print("-" * 70)

    best_results = []

    for daily_limit in [1, 2, 3, 5, 999]:
        for risk in [0.10, 0.15, 0.20, 0.25, 0.30]:
            bt = Backtest(risk_per_trade=risk, max_daily_trades=daily_limit)
            result = bt.run(df, all_signals)
            if result:
                dl_str = "∞" if daily_limit == 999 else str(daily_limit)
                mark = " ***" if result['win_rate'] >= 70 and result['total_return'] > 0 else ""
                print(f"{dl_str:>12} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['total_return']:>+9.1f}% {result['max_drawdown']:>7.1f}% {result['profit_factor']:>6.2f}{mark}")

                if result['win_rate'] >= 70 and result['total_return'] > 0:
                    best_results.append({
                        'daily_limit': daily_limit,
                        'risk': risk,
                        **result
                    })

    # Best high WR config
    if best_results:
        best = max(best_results, key=lambda x: x['total_return'])
        print("\n" + "=" * 100)
        print("  BEST HIGH WIN RATE CONFIG (WR >= 70%)")
        print("=" * 100)
        print(f"Daily Limit: {best['daily_limit']}")
        print(f"Risk: {best['risk']*100:.0f}%")
        print(f"Trades: {best['trades']}")
        print(f"Win Rate: {best['win_rate']:.1f}%")
        print(f"Return: {best['total_return']:+.1f}%")
        print(f"Max DD: {best['max_drawdown']:.1f}%")

    # Compare with Swing
    print("\n" + "=" * 100)
    print("  COMPARISON: HIGH WR vs SWING")
    print("=" * 100)

    for risk in [0.10, 0.15]:
        # High WR
        bt = Backtest(risk_per_trade=risk)
        combined_result = bt.run(df, all_signals)

        # Swing
        bt = Backtest(risk_per_trade=risk)
        swing_result = bt.run(df, swing_sigs)

        if combined_result and swing_result:
            print(f"\nRisk {risk*100:.0f}%:")
            print(f"  High WR: {combined_result['trades']} trades, {combined_result['win_rate']:.1f}% WR, {combined_result['total_return']:+.1f}% return")
            print(f"  Swing:   {swing_result['trades']} trades, {swing_result['win_rate']:.1f}% WR, {swing_result['total_return']:+.1f}% return")


if __name__ == "__main__":
    main()
