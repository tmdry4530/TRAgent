"""Final BTC 50x Strategy Test - All strategies compared."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from collections import defaultdict

from scripts.run_backtest import generate_scalp_signals, generate_swing_signals
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
    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0002, min_confidence=0.60):
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


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles (1 year)')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    # Generate all signals
    scalp_signals = generate_scalp_signals(df)
    swing_signals = generate_swing_signals(df)

    print(f'\nSignals: {len(scalp_signals)} scalp, {len(swing_signals)} swing')

    print('\n' + '=' * 75)
    print('  FINAL STRATEGY COMPARISON - BTC 50x LEVERAGE')
    print('=' * 75)

    results = {}

    # Test Scalp
    print('\n[SCALP Strategy]')
    print(f"{'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8} {'Monthly':>9}")
    print('-' * 60)

    for risk in [0.02, 0.03, 0.05]:
        engine = LeveragedBacktest(risk_per_trade=risk, min_confidence=0.60)
        result = engine.run(df, scalp_signals)
        if result:
            monthly = result['total_return'] / 12
            print(f"{risk*100:>5.0f}% {result['trades']:>7} {result['win_rate']:>6.1f}% "
                  f"{result['total_return']:>+9.1f}% {result['profit_factor']:>6.2f} "
                  f"{result['max_drawdown']:>7.1f}% {monthly:>+8.1f}%")
            if result['profit_factor'] > 1.0:
                results[f'scalp_{risk}'] = result

    # Test Swing
    print('\n[SWING Strategy]')
    print(f"{'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8} {'Monthly':>9}")
    print('-' * 60)

    for risk in [0.05, 0.10, 0.15]:
        engine = LeveragedBacktest(risk_per_trade=risk, min_confidence=0.60)
        result = engine.run(df, swing_signals)
        if result:
            monthly = result['total_return'] / 12
            print(f"{risk*100:>5.0f}% {result['trades']:>7} {result['win_rate']:>6.1f}% "
                  f"{result['total_return']:>+9.1f}% {result['profit_factor']:>6.2f} "
                  f"{result['max_drawdown']:>7.1f}% {monthly:>+8.1f}%")
            if result['profit_factor'] > 1.0:
                results[f'swing_{risk}'] = result

    # Best strategy
    print('\n' + '=' * 75)

    if results:
        best_key = max(results.keys(), key=lambda k: results[k]['total_return'])
        best = results[best_key]

        print(f'  BEST STRATEGY: {best_key.upper()}')
        print('=' * 75)
        print(f'  Initial Capital:  $10,000')
        print(f'  Final Capital:    ${best["final"]:,.2f}')
        print(f'  Total Return:     {best["total_return"]:+.2f}%')
        print(f'  Total Trades:     {best["trades"]}')
        print(f'  Win Rate:         {best["win_rate"]:.1f}%')
        print(f'  Profit Factor:    {best["profit_factor"]:.2f}')
        print(f'  Max Drawdown:     {best["max_drawdown"]:.2f}%')
        print()
        print(f'  Monthly Avg:      {best["trades"]/12:.1f} trades')
        print(f'  Monthly Return:   {best["total_return"]/12:+.1f}%')

        # Monthly breakdown
        if best['trade_list']:
            print('\n  MONTHLY PERFORMANCE')
            print('  ' + '-' * 50)

            monthly_pnl = defaultdict(float)
            monthly_trades = defaultdict(int)
            monthly_wins = defaultdict(int)

            for t in best['trade_list']:
                m = t.exit_time.strftime('%Y-%m')
                monthly_pnl[m] += t.pnl
                monthly_trades[m] += 1
                if t.pnl > 0:
                    monthly_wins[m] += 1

            cap = 10000
            total_months = 0
            positive_months = 0

            for month in sorted(monthly_pnl.keys()):
                pnl = monthly_pnl[month]
                trades = monthly_trades[month]
                wins = monthly_wins[month]
                ret = (pnl / cap) * 100
                cap += pnl
                sign = '+' if ret >= 0 else ''
                bar = '#' * int(abs(ret) / 5) if ret != 0 else ''

                total_months += 1
                if ret > 0:
                    positive_months += 1

                print(f"  {month}: {trades}거래 {wins}승 {sign}{ret:>7.2f}%  {bar}")

            print()
            print(f'  Positive Months: {positive_months}/{total_months}')

    else:
        print('  No profitable strategy found!')

    # Final recommendation
    print('\n' + '=' * 75)
    print('  RECOMMENDATION')
    print('=' * 75)

    if 'swing_0.1' in results:
        r = results['swing_0.1']
        print(f'''
  Strategy: SWING (4h EMA Cross + MACD)
  Leverage: 50x
  Risk:     10% per trade

  Expected Performance:
  - Win Rate:      ~50%
  - Annual Return: ~{r["total_return"]:.0f}%
  - Max Drawdown:  ~{r["max_drawdown"]:.0f}%
  - Monthly Trades: ~1

  NOTE: Returns vary significantly month-to-month.
        50% chance of no trade in any given month.
        Recommend running for 3-6 months minimum.
''')

    print('=' * 75)


if __name__ == '__main__':
    main()
