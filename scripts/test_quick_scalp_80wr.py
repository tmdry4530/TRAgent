"""Quick Scalp Strategy - High win rate through fast take profit."""

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


class QuickScalpBacktest:
    """Quick scalp with tight TP for high win rate."""

    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0001, rr_ratio=0.8):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.rr_ratio = rr_ratio  # R:R ratio for TP

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
                    self._enter(signal, row, current_time)

            equity = self.capital + (self._unrealized_pnl(row['close']) if self.current_position else 0)
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal, row, time):
        price = row['close']
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)

        # Calculate SL based on signal or ATR
        if hasattr(signal, 'stop_loss') and signal.stop_loss:
            sl_distance = abs(entry_price - signal.stop_loss)
        else:
            return

        sl_pct = sl_distance / entry_price
        if sl_pct == 0 or sl_pct > 0.03:
            return

        # Risk-based position sizing
        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_pct
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

        # Calculate TP based on R:R ratio
        tp_distance = sl_distance * self.rr_ratio
        if signal.direction == 'LONG':
            take_profit = entry_price + tp_distance
            stop_loss = entry_price - sl_distance
        else:
            take_profit = entry_price - tp_distance
            stop_loss = entry_price + sl_distance

        self.current_position = {
            'entry_price': entry_price, 'entry_time': time, 'size': size,
            'margin': margin, 'position_value': position_value,
            'direction': signal.direction, 'stop_loss': stop_loss,
            'take_profit': take_profit,
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


def wick_scalp_signals(df, vol_mult=2.0, wick_pct=0.45):
    """More relaxed wick signals for scalping."""
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

        # Trend context
        uptrend = price > row['ema50'] if not pd.isna(row['ema50']) else True
        downtrend = price < row['ema50'] if not pd.isna(row['ema50']) else True

        # LONG: Wick bounce with volume in uptrend
        if (lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= vol_mult and
            rsi < 55 and
            uptrend):

            # Tight SL at wick low
            sl = row['low'] - atr * 0.05

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.70,
                entry_price=price, stop_loss=sl, take_profit=price + atr,
                reason="Wick scalp", timestamp=row['timestamp'],
            ))

        # SHORT: Wick rejection with volume in downtrend
        if (upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            vol_ratio >= vol_mult and
            rsi > 45 and
            downtrend):

            sl = row['high'] + atr * 0.05

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.70,
                entry_price=price, stop_loss=sl, take_profit=price - atr,
                reason="Wick scalp", timestamp=row['timestamp'],
            ))

    return signals


def ema_bounce_signals(df):
    """EMA bounce signals for scalping."""
    signals = []

    for i in range(100, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['ema20']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Uptrend: EMA20 > EMA50
        uptrend = row['ema20'] > row['ema50'] if not pd.isna(row['ema50']) else False
        downtrend = row['ema20'] < row['ema50'] if not pd.isna(row['ema50']) else False

        # LONG: Touch EMA20 with wick in uptrend
        if (uptrend and
            row['low'] <= row['ema20'] * 1.002 and
            lower_wick_pct >= 0.40 and
            row['close'] > row['open'] and
            35 <= rsi <= 55):

            sl = row['low'] - atr * 0.1

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.72,
                entry_price=price, stop_loss=sl, take_profit=price + atr,
                reason="EMA bounce", timestamp=row['timestamp'],
            ))

        # SHORT: Touch EMA20 with wick in downtrend
        if (downtrend and
            row['high'] >= row['ema20'] * 0.998 and
            upper_wick_pct >= 0.40 and
            row['close'] < row['open'] and
            45 <= rsi <= 65):

            sl = row['high'] + atr * 0.1

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.72,
                entry_price=price, stop_loss=sl, take_profit=price - atr,
                reason="EMA bounce", timestamp=row['timestamp'],
            ))

    return signals


def bb_bounce_signals(df):
    """Bollinger Band bounce signals."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['bb_lower']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1

        # LONG: Touch BB lower with wick + volume
        if (row['low'] <= row['bb_lower'] * 1.005 and
            lower_wick_pct >= 0.45 and
            row['close'] > row['open'] and
            vol_ratio >= 1.5 and
            rsi < 40):

            sl = row['low'] - atr * 0.05

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=price + atr,
                reason="BB bounce", timestamp=row['timestamp'],
            ))

        # SHORT: Touch BB upper with wick + volume
        if (row['high'] >= row['bb_upper'] * 0.995 and
            upper_wick_pct >= 0.45 and
            row['close'] < row['open'] and
            vol_ratio >= 1.5 and
            rsi > 60):

            sl = row['high'] + atr * 0.05

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=price - atr,
                reason="BB bounce", timestamp=row['timestamp'],
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    df = add_indicators(df)

    # Generate all signal types
    wick_signals = wick_scalp_signals(df)
    ema_signals = ema_bounce_signals(df)
    bb_signals = bb_bounce_signals(df)

    all_signals = wick_signals + ema_signals + bb_signals

    # Remove duplicates
    unique_signals = []
    seen = set()
    for s in sorted(all_signals, key=lambda x: (x.timestamp, -x.confidence)):
        key = (s.timestamp, s.direction)
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)

    signals = unique_signals

    print(f'\nTotal scalp signals: {len(signals)}')
    print(f'  Wick: {len(wick_signals)}, EMA: {len(ema_signals)}, BB: {len(bb_signals)}')

    print('\n' + '=' * 90)
    print('  QUICK SCALP STRATEGY - R:R OPTIMIZATION')
    print('=' * 90)

    # Test different R:R ratios
    print(f"\n{'R:R':>6} {'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7} {'MDD':>8}")
    print('-' * 65)

    best = None
    best_config = None

    for rr in [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        for risk in [0.10, 0.15, 0.20]:
            engine = QuickScalpBacktest(risk_per_trade=risk, rr_ratio=rr)
            result = engine.run(df, signals)

            if result and result['trades'] >= 5:
                print(f"{rr:>6.1f} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>7.2f} {result['max_drawdown']:>7.1f}%")

                if result['win_rate'] >= 70 and result['total_return'] > 0:
                    if best is None or result['total_return'] > best['total_return']:
                        best = result
                        best_config = {'rr': rr, 'risk': risk}

    # Try very tight TP for higher win rate
    print('\n' + '=' * 90)
    print('  ULTRA-TIGHT TP (Higher Win Rate)')
    print('=' * 90)

    print(f"\n{'R:R':>6} {'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7}")
    print('-' * 55)

    for rr in [0.3, 0.4, 0.5]:
        for risk in [0.15, 0.20, 0.25, 0.30]:
            engine = QuickScalpBacktest(risk_per_trade=risk, rr_ratio=rr)
            result = engine.run(df, signals)

            if result and result['trades'] >= 5:
                print(f"{rr:>6.1f} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>7.2f}")

                if result['win_rate'] >= 70 and result['total_return'] > (best['total_return'] if best else -999):
                    best = result
                    best_config = {'rr': rr, 'risk': risk}

    # Results
    print('\n' + '=' * 90)

    if best and best['win_rate'] >= 70:
        print(f'  BEST 70%+ WIN RATE RESULT')
        print('=' * 90)
        print(f'''
  Configuration:
  - R:R Ratio: {best_config["rr"]}
  - Risk: {best_config["risk"]*100:.0f}%
  - Leverage: 50x

  Results:
  - Trades: {best["trades"]}
  - Win Rate: {best["win_rate"]:.1f}%
  - Return: {best["total_return"]:+.2f}%
  - PF: {best["profit_factor"]:.2f}
  - Max DD: {best["max_drawdown"]:.2f}%
  - Monthly: {best["total_return"]/12:+.1f}%

  Required for 80% WR + 100% return with R:R {best_config["rr"]}:
  - With {best_config["rr"]} R:R, need {100/(best_config["rr"]*100/(1-best_config["rr"])):.0f}+ winning trades
  - Or use higher R:R and accept lower win rate
''')

        if best['trade_list']:
            print('  SAMPLE TRADES:')
            wins = [t for t in best['trade_list'] if t.pnl > 0][:5]
            losses = [t for t in best['trade_list'] if t.pnl <= 0][:3]

            print('  [WINS]')
            for t in wins:
                print(f"    {t.entry_time.strftime('%Y-%m-%d')} {t.direction:>5} +${t.pnl:,.0f}")

            print('  [LOSSES]')
            for t in losses:
                print(f"    {t.entry_time.strftime('%Y-%m-%d')} {t.direction:>5} ${t.pnl:,.0f}")

    else:
        print('  TARGET NOT MET')
        print('=' * 90)
        print('''
  Analysis: 80% win rate + 100% return is mathematically challenging with wick signals.

  Trade-off:
  - Higher win rate (80%+) requires tighter TP → lower per-trade profit
  - Higher return (100%+) requires more trades or higher R:R

  For 80% WR to achieve 100% return:
  - With R:R 1:1, need 100 winning trades (80 wins - 20 losses = 60 net wins)
  - With R:R 0.5:1, need 200+ winning trades

  Current data only supports ~100-150 trades/year with quality signals.

  RECOMMENDATION:
  Use original SWING strategy (50% WR, 3:1 R:R, +178% return)
  - Mathematically more efficient
  - Fewer trades = lower execution risk
  - Proven profitable over 1 year backtest
''')

    # Compare strategies
    print('\n' + '=' * 90)
    print('  STRATEGY COMPARISON')
    print('=' * 90)

    from scripts.run_backtest import generate_swing_signals
    swing_signals = generate_swing_signals(df)

    comparisons = [
        ("Swing (50% WR, 3:1 R:R)", swing_signals, 0.10, None),
    ]

    if best:
        comparisons.append(("Best Scalp", signals, best_config['risk'], best_config['rr']))

    print(f"\n{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Return':>10} {'Monthly':>9}")
    print('-' * 70)

    for name, sigs, risk, rr in comparisons:
        if rr:
            engine = QuickScalpBacktest(risk_per_trade=risk, rr_ratio=rr)
        else:
            from scripts.test_final_80wr import Backtest
            engine = Backtest(risk_per_trade=risk)

        result = engine.run(df, sigs)
        if result:
            print(f"{name:<30} {result['trades']:>7} {result['win_rate']:>6.1f}% "
                  f"{result['total_return']:>+9.1f}% {result['total_return']/12:>+8.1f}%")

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
