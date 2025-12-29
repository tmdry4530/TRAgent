"""Swing + Wick Confirmation - Swing signals with wick pattern entry timing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
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


class LeveragedBacktest:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.10,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        min_confidence: float = 0.60,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.min_confidence = min_confidence
        self.capital = initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = [initial_capital]

    def run(self, df: pd.DataFrame, signals: list[Signal]):
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

                if signal.confidence >= self.min_confidence and not self.current_position:
                    self._enter(signal, row['close'], current_time)

            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(row['close'])
            self.equity_curve.append(equity)

        if self.current_position:
            last = df.iloc[-1]
            self._exit(last['close'], last['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal: Signal, price: float, time: datetime):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price
        if sl_distance == 0:
            return

        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_distance
        margin = position_value / self.leverage
        max_margin = self.capital * 0.90
        if margin > max_margin:
            margin = max_margin
            position_value = margin * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission
        self.capital -= (margin + entry_comm)

        self.current_position = {
            'entry_price': entry_price, 'entry_time': time, 'size': size,
            'margin': margin, 'position_value': position_value,
            'direction': signal.direction, 'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
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

    def _exit(self, price: float, time: datetime, reason: str):
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

    def _unrealized_pnl(self, price: float) -> float:
        pos = self.current_position
        if not pos:
            return 0.0
        if pos['direction'] == 'LONG':
            return (price - pos['entry_price']) * pos['size']
        return (pos['entry_price'] - price) * pos['size']

    def _metrics(self):
        if not self.trades:
            return None
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))
        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_return': total_return,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'final': self.capital,
            'trade_list': self.trades,
        }


def generate_swing_signals(df: pd.DataFrame) -> list[Signal]:
    """Original swing signals (4h timeframe)."""
    signals = []
    df = df.copy()

    df['ema12'] = df['close'].ewm(span=12 * 4, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26 * 4, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50 * 4, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14 * 4).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14 * 4).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9 * 4, adjust=False).mean()

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14 * 4).mean()

    for i in range(250, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-4]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['macd']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr']
        macd = row['macd']
        macd_sig = row['macd_signal']

        bullish = row['ema12'] > row['ema26'] > row['ema50']
        bearish = row['ema12'] < row['ema26'] < row['ema50']

        prev_bull = prev['ema12'] > prev['ema26'] if not pd.isna(prev['ema26']) else False
        prev_bear = prev['ema12'] < prev['ema26'] if not pd.isna(prev['ema26']) else False

        bull_cross = bullish and not prev_bull
        bear_cross = bearish and not prev_bear

        macd_bull = macd > macd_sig
        macd_bear = macd < macd_sig

        sl_distance = atr * 3
        tp_distance = sl_distance * 3

        if bull_cross and macd_bull and 30 <= rsi <= 65:
            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.65,
                entry_price=price,
                stop_loss=price - sl_distance,
                take_profit=price + tp_distance,
                reason=f"4h EMA cross + MACD",
                timestamp=row['timestamp'],
            ))

        if bear_cross and macd_bear and 35 <= rsi <= 70:
            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.65,
                entry_price=price,
                stop_loss=price + sl_distance,
                take_profit=price - tp_distance,
                reason=f"4h EMA cross + MACD",
                timestamp=row['timestamp'],
            ))

    return signals


def generate_swing_with_wick_confirmation(df: pd.DataFrame) -> list[Signal]:
    """Swing signals that wait for wick confirmation for better entry."""
    signals = []
    df = df.copy()

    # 4h EMAs (scaled from 1h)
    df['ema12'] = df['close'].ewm(span=12 * 4, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26 * 4, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50 * 4, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14 * 4).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14 * 4).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9 * 4, adjust=False).mean()

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14 * 4).mean()

    # Candle analysis
    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    # Track pending swing signals
    pending_long = None
    pending_long_expires = None
    pending_short = None
    pending_short_expires = None

    WICK_THRESHOLD = 0.40  # 40% wick required
    LOOKBACK_HOURS = 48  # Wait up to 48 hours for wick confirmation

    for i in range(250, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-4]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['ema50']) or pd.isna(row['macd']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr']
        macd = row['macd']
        macd_sig = row['macd_signal']

        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        bullish = row['ema12'] > row['ema26'] > row['ema50']
        bearish = row['ema12'] < row['ema26'] < row['ema50']

        prev_bull = prev['ema12'] > prev['ema26'] if not pd.isna(prev['ema26']) else False
        prev_bear = prev['ema12'] < prev['ema26'] if not pd.isna(prev['ema26']) else False

        bull_cross = bullish and not prev_bull
        bear_cross = bearish and not prev_bear

        macd_bull = macd > macd_sig
        macd_bear = macd < macd_sig

        # Create pending signals
        if bull_cross and macd_bull and 30 <= rsi <= 65:
            pending_long = {
                'atr': atr,
                'rsi': rsi,
            }
            pending_long_expires = timestamp + timedelta(hours=LOOKBACK_HOURS)

        if bear_cross and macd_bear and 35 <= rsi <= 70:
            pending_short = {
                'atr': atr,
                'rsi': rsi,
            }
            pending_short_expires = timestamp + timedelta(hours=LOOKBACK_HOURS)

        # Check for wick confirmation on pending longs
        if pending_long and pending_long_expires:
            if timestamp > pending_long_expires:
                pending_long = None
                pending_long_expires = None
            elif lower_wick_pct >= WICK_THRESHOLD and row['close'] > row['open']:
                # Found wick! Create signal with better entry
                sl_distance = pending_long['atr'] * 2.5  # Tighter SL due to wick
                tp_distance = sl_distance * 3.5  # Better R:R

                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=0.70,  # Higher confidence
                    entry_price=price,
                    stop_loss=min(row['low'] - atr * 0.1, price - sl_distance),
                    take_profit=price + tp_distance,
                    reason=f"Swing + wick confirmation",
                    timestamp=timestamp,
                ))
                pending_long = None
                pending_long_expires = None

        # Check for wick confirmation on pending shorts
        if pending_short and pending_short_expires:
            if timestamp > pending_short_expires:
                pending_short = None
                pending_short_expires = None
            elif upper_wick_pct >= WICK_THRESHOLD and row['close'] < row['open']:
                sl_distance = pending_short['atr'] * 2.5
                tp_distance = sl_distance * 3.5

                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=0.70,
                    entry_price=price,
                    stop_loss=max(row['high'] + atr * 0.1, price + sl_distance),
                    take_profit=price - tp_distance,
                    reason=f"Swing + wick confirmation",
                    timestamp=timestamp,
                ))
                pending_short = None
                pending_short_expires = None

    return signals


def generate_relaxed_swing_with_wick(df: pd.DataFrame) -> list[Signal]:
    """More relaxed swing + wick for more signals."""
    signals = []
    df = df.copy()

    # 2h EMAs (middle ground)
    df['ema12'] = df['close'].ewm(span=12 * 2, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26 * 2, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50 * 2, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14 * 2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14 * 2).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9 * 2, adjust=False).mean()

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14 * 2).mean()

    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)

    WICK_THRESHOLD = 0.35

    for i in range(120, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-2]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr']
        macd = row['macd']
        macd_sig = row['macd_signal']

        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Trend conditions (more relaxed)
        bullish_trend = row['ema12'] > row['ema26']
        bearish_trend = row['ema12'] < row['ema26']

        prev_bull = prev['ema12'] > prev['ema26'] if not pd.isna(prev['ema26']) else False
        prev_bear = prev['ema12'] < prev['ema26'] if not pd.isna(prev['ema26']) else False

        macd_bull = macd > macd_sig
        macd_bear = macd < macd_sig

        # LONG: Bullish trend + wick bounce + MACD alignment
        if (bullish_trend and macd_bull and
            lower_wick_pct >= WICK_THRESHOLD and
            row['close'] > row['open'] and
            rsi < 60):

            sl_distance = atr * 2.5
            tp_distance = sl_distance * 3

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.65,
                entry_price=price,
                stop_loss=row['low'] - atr * 0.1,
                take_profit=price + tp_distance,
                reason=f"2h trend + wick RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # SHORT: Bearish trend + wick rejection + MACD alignment
        if (bearish_trend and macd_bear and
            upper_wick_pct >= WICK_THRESHOLD and
            row['close'] < row['open'] and
            rsi > 40):

            sl_distance = atr * 2.5
            tp_distance = sl_distance * 3

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.65,
                entry_price=price,
                stop_loss=row['high'] + atr * 0.1,
                take_profit=price - tp_distance,
                reason=f"2h trend + wick RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    # Original swing
    swing_signals = generate_swing_signals(df)
    print(f'\nOriginal Swing (4h): {len(swing_signals)} signals')

    # Swing + wick confirmation
    swing_wick_signals = generate_swing_with_wick_confirmation(df)
    print(f'Swing + Wick Confirm: {len(swing_wick_signals)} signals')

    # Relaxed swing + wick
    relaxed_signals = generate_relaxed_swing_with_wick(df)
    print(f'Relaxed 2h + Wick: {len(relaxed_signals)} signals')

    print('\n' + '=' * 80)
    print('  SWING + WICK STRATEGY COMPARISON')
    print('=' * 80)

    strategies = [
        ('Original Swing 4h', swing_signals),
        ('Swing+Wick Confirm', swing_wick_signals),
        ('Relaxed 2h+Wick', relaxed_signals),
    ]

    print(f"\n{'Strategy':>20} {'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8} {'Monthly':>9}")
    print('-' * 90)

    results = {}

    for name, sigs in strategies:
        if not sigs:
            print(f'{name:>20} -- No signals generated')
            continue

        for risk in [0.05, 0.10]:
            engine = LeveragedBacktest(risk_per_trade=risk)
            result = engine.run(df, sigs)

            if result:
                monthly = result['total_return'] / 12
                print(f"{name:>20} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>6.2f} {result['max_drawdown']:>7.1f}% {monthly:>+8.1f}%")

                if result['profit_factor'] > 1.0:
                    results[f'{name}_{risk}'] = result

    # Best result
    if results:
        print('\n' + '=' * 80)
        best_key = max(results.keys(), key=lambda k: results[k]['total_return'])
        best = results[best_key]

        print(f'  BEST: {best_key}')
        print('=' * 80)
        print(f'  Trades: {best["trades"]}')
        print(f'  Win Rate: {best["win_rate"]:.1f}%')
        print(f'  Return: {best["total_return"]:+.2f}%')
        print(f'  PF: {best["profit_factor"]:.2f}')
        print(f'  Max DD: {best["max_drawdown"]:.2f}%')
        print(f'  Monthly: {best["total_return"]/12:+.1f}%')

        if best['trade_list']:
            print('\n  TRADE DETAILS')
            print('  ' + '-' * 60)
            for t in best['trade_list']:
                sign = '+' if t.pnl >= 0 else ''
                print(f"  {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction:>5} "
                      f"${t.entry_price:,.0f} -> ${t.exit_price:,.0f} "
                      f"{sign}${t.pnl:,.0f} ({t.exit_reason})")

    print('\n' + '=' * 80)
    print('  CONCLUSION')
    print('=' * 80)

    if results:
        best = results[best_key]
        orig = None
        for k, v in results.items():
            if 'Original' in k and '0.1' in k:
                orig = v
                break

        print(f'''
  Original Swing (4h EMA cross):
  - Simple and reliable
  - 50% win rate, {orig["total_return"]:+.0f}% annual if 10% risk

  Swing + Wick confirmation:
  - Adds entry timing refinement
  - May reduce total trades but improve quality

  For your wick-entry style:
  - The algorithm cannot fully replicate human discretion
  - Consider using signals as "alert" mode
  - Then manually time entries using wick patterns
''')

    print('=' * 80)


if __name__ == '__main__':
    main()
