"""Volume Climax Strategy Optimization - Find optimal parameters for 80% WR."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from itertools import product
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
                 commission=0.0004, slippage=0.0002, min_confidence=0.60):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.min_confidence = min_confidence

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
                if signal.confidence >= self.min_confidence and not self.current_position:
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

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
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

    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    return df


def volume_climax_strategy(df, vol_mult=3.0, consec_bars=3, wick_pct=0.50, rsi_long=35, rsi_short=65, rr_ratio=1.5):
    """Parameterized volume climax strategy."""
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

        # Check consecutive moves
        consec_down = all(df.iloc[i-j]['close'] < df.iloc[i-j]['open'] for j in range(1, consec_bars + 1))
        consec_up = all(df.iloc[i-j]['close'] > df.iloc[i-j]['open'] for j in range(1, consec_bars + 1))

        # LONG
        if (vol_ratio >= vol_mult and
            consec_down and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            rsi < rsi_long):

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Vol climax",
                timestamp=row['timestamp'],
            ))

        # SHORT
        if (vol_ratio >= vol_mult and
            consec_up and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            rsi > rsi_short):

            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Vol climax",
                timestamp=row['timestamp'],
            ))

    return signals


def smart_money_reversal(df, vol_mult=2.5, wick_pct=0.55, rsi_extreme=30, rr_ratio=1.2):
    """Smart money reversal - extreme conditions with confirmation."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['volume_ratio']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # LONG: Price spike down with immediate recovery
        big_down_move = (prev['close'] - prev['open']) / prev['open'] < -0.01  # >1% down candle
        immediate_recovery = row['close'] > prev['close']

        if (big_down_move and
            immediate_recovery and
            vol_ratio >= vol_mult and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            rsi < 40):

            sl = min(row['low'], prev['low']) - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Smart money reversal",
                timestamp=row['timestamp'],
            ))

        # SHORT
        big_up_move = (prev['close'] - prev['open']) / prev['open'] > 0.01
        immediate_drop = row['close'] < prev['close']

        if (big_up_move and
            immediate_drop and
            vol_ratio >= vol_mult and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            rsi > 60):

            sl = max(row['high'], prev['high']) + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Smart money reversal",
                timestamp=row['timestamp'],
            ))

    return signals


def exhaustion_candle(df, vol_mult=2.0, range_mult=2.0, wick_pct=0.50, rr_ratio=1.0):
    """Exhaustion candle - large range candle with reversal wick."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['volume_ratio']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio']
        full_range = row['full_range']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Large range candle
        big_range = full_range >= atr * range_mult

        # LONG: Big range + long lower wick + bullish close
        if (big_range and
            vol_ratio >= vol_mult and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            rsi < 45):

            sl = row['low'] - atr * 0.05
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Exhaustion candle",
                timestamp=row['timestamp'],
            ))

        # SHORT
        if (big_range and
            vol_ratio >= vol_mult and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            rsi > 55):

            sl = row['high'] + atr * 0.05
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Exhaustion candle",
                timestamp=row['timestamp'],
            ))

    return signals


def bb_squeeze_breakout(df, vol_mult=2.0, squeeze_periods=10, rr_ratio=1.5):
    """Bollinger Band squeeze followed by breakout with volume."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['bb_std']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1

        # Check for prior squeeze (narrow BB)
        recent_bb_widths = [(df.iloc[i-j]['bb_upper'] - df.iloc[i-j]['bb_lower'])
                           for j in range(1, squeeze_periods + 1)
                           if not pd.isna(df.iloc[i-j]['bb_upper'])]

        if len(recent_bb_widths) < squeeze_periods:
            continue

        avg_width = np.mean(recent_bb_widths)
        current_width = row['bb_upper'] - row['bb_lower']

        # Expanding from squeeze
        was_squeezed = all(w < avg_width * 0.8 for w in recent_bb_widths[-5:])

        # LONG: Breakout above BB upper with volume
        if (was_squeezed and
            row['close'] > row['bb_upper'] and
            vol_ratio >= vol_mult and
            row['close'] > row['open']):

            sl = row['bb_mid'] - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"BB squeeze breakout",
                timestamp=row['timestamp'],
            ))

        # SHORT: Breakdown below BB lower with volume
        if (was_squeezed and
            row['close'] < row['bb_lower'] and
            vol_ratio >= vol_mult and
            row['close'] < row['open']):

            sl = row['bb_mid'] + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"BB squeeze breakdown",
                timestamp=row['timestamp'],
            ))

    return signals


def trend_continuation_wick(df, wick_pct=0.45, rsi_min=40, rsi_max=60, rr_ratio=1.5):
    """Wick in direction of trend - continuation signal."""
    signals = []

    for i in range(200, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['ema200']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1

        # Strong trend conditions
        strong_uptrend = (price > row['ema20'] > row['ema50'] > row['ema100'] > row['ema200'])
        strong_downtrend = (price < row['ema20'] < row['ema50'] < row['ema100'] < row['ema200'])

        # LONG: Strong uptrend + pullback wick + not overbought
        if (strong_uptrend and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            rsi_min <= rsi <= rsi_max and
            vol_ratio >= 1.5 and
            row['low'] <= row['ema20'] * 1.01):  # Touch EMA20

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Trend continuation wick",
                timestamp=row['timestamp'],
            ))

        # SHORT: Strong downtrend + rejection wick + not oversold
        if (strong_downtrend and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            rsi_min <= rsi <= rsi_max and
            vol_ratio >= 1.5 and
            row['high'] >= row['ema20'] * 0.99):

            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Trend continuation wick",
                timestamp=row['timestamp'],
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    df = add_indicators(df)

    print('\n' + '=' * 90)
    print('  VOLUME CLIMAX PARAMETER OPTIMIZATION')
    print('=' * 90)

    # Test different parameters
    vol_mults = [2.0, 2.5, 3.0]
    consec_bars_opts = [2, 3]
    wick_pcts = [0.40, 0.50, 0.60]
    rr_ratios = [1.0, 1.5, 2.0]

    print(f"\n{'Vol':>5} {'Bars':>5} {'Wick':>5} {'R:R':>5} {'Sigs':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7}")
    print('-' * 75)

    best_results = []

    for vol, bars, wick, rr in product(vol_mults, consec_bars_opts, wick_pcts, rr_ratios):
        signals = volume_climax_strategy(df, vol_mult=vol, consec_bars=bars, wick_pct=wick, rr_ratio=rr)

        if not signals:
            continue

        engine = Backtest(risk_per_trade=0.10)
        result = engine.run(df, signals)

        if result and result['trades'] >= 3:
            print(f"{vol:>5.1f} {bars:>5} {wick:>5.2f} {rr:>5.1f} {len(signals):>6} "
                  f"{result['trades']:>7} {result['win_rate']:>6.1f}% "
                  f"{result['total_return']:>+9.1f}% {result['profit_factor']:>7.2f}")

            if result['win_rate'] >= 60:
                best_results.append({
                    'params': f"Vol{vol}_B{bars}_W{wick}_RR{rr}",
                    'signals': len(signals),
                    'result': result,
                })

    print('\n' + '=' * 90)
    print('  OTHER STRATEGIES')
    print('=' * 90)

    other_strategies = [
        ("Smart Money Reversal", smart_money_reversal(df)),
        ("Exhaustion Candle", exhaustion_candle(df)),
        ("BB Squeeze Breakout", bb_squeeze_breakout(df)),
        ("Trend Continuation", trend_continuation_wick(df)),
    ]

    print(f"\n{'Strategy':<25} {'Sigs':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7}")
    print('-' * 70)

    for name, signals in other_strategies:
        if not signals:
            print(f"{name:<25} {'0':>6}")
            continue

        engine = Backtest(risk_per_trade=0.10)
        result = engine.run(df, signals)

        if result:
            print(f"{name:<25} {len(signals):>6} {result['trades']:>7} "
                  f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                  f"{result['profit_factor']:>7.2f}")

            if result['win_rate'] >= 60:
                best_results.append({
                    'params': name,
                    'signals': len(signals),
                    'result': result,
                })

    # Summary of high win rate strategies
    print('\n' + '=' * 90)
    print('  HIGH WIN RATE STRATEGIES (>=60%)')
    print('=' * 90)

    if best_results:
        best_results.sort(key=lambda x: x['result']['total_return'], reverse=True)

        print(f"\n{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Return':>10} {'Monthly':>9}")
        print('-' * 70)

        for r in best_results:
            res = r['result']
            monthly = res['total_return'] / 12
            print(f"{r['params']:<30} {res['trades']:>7} {res['win_rate']:>6.1f}% "
                  f"{res['total_return']:>+9.1f}% {monthly:>+8.1f}%")

        # Best overall
        best = best_results[0]
        print('\n' + '=' * 90)
        print(f"  BEST: {best['params']}")
        print('=' * 90)
        print(f"  Signals: {best['signals']}")
        print(f"  Trades: {best['result']['trades']}")
        print(f"  Win Rate: {best['result']['win_rate']:.1f}%")
        print(f"  Return: {best['result']['total_return']:+.2f}%")
        print(f"  Monthly: {best['result']['total_return']/12:+.1f}%")

        if best['result']['trade_list']:
            print('\n  TRADES:')
            for t in best['result']['trade_list']:
                sign = '+' if t.pnl >= 0 else ''
                status = 'WIN' if t.pnl > 0 else 'LOSS'
                print(f"    {t.entry_time.strftime('%Y-%m-%d')} {t.direction:>5} "
                      f"${t.entry_price:,.0f} -> ${t.exit_price:,.0f} {sign}${t.pnl:,.0f} [{status}]")
    else:
        print("\n  No strategies found with 60%+ win rate")

    # Try combining strategies
    print('\n' + '=' * 90)
    print('  COMBINED SELECTIVE SIGNALS')
    print('=' * 90)

    # Combine best performing signals
    combined = []
    combined.extend(volume_climax_strategy(df, vol_mult=2.5, consec_bars=3, wick_pct=0.50, rr_ratio=1.5))
    combined.extend(smart_money_reversal(df))
    combined.extend(exhaustion_candle(df, vol_mult=2.0, range_mult=2.0, rr_ratio=1.0))

    combined.sort(key=lambda s: s.timestamp)

    print(f"\nCombined signals: {len(combined)}")

    for risk in [0.05, 0.10, 0.15, 0.20]:
        engine = Backtest(risk_per_trade=risk)
        result = engine.run(df, combined)

        if result:
            print(f"  Risk {risk*100:>4.0f}%: {result['trades']} trades, "
                  f"{result['win_rate']:.1f}% WR, {result['total_return']:+.1f}% return, "
                  f"PF {result['profit_factor']:.2f}")

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
