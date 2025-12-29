"""Daily Trade Limit Strategy - Max 3 trades per day."""

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


class DailyLimitBacktest:
    """Backtest with daily trade limit."""

    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0002, max_daily_trades=3):
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

        # Daily stats
        trading_days = len(set(t.entry_time.date() for t in self.trades))

        return {
            'trades': len(self.trades), 'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_return': total_return, 'profit_factor': pf,
            'max_drawdown': max_dd, 'final': self.capital, 'trade_list': self.trades,
            'trading_days': trading_days,
            'trades_per_day': len(self.trades) / trading_days if trading_days > 0 else 0,
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


def generate_scalp_signals(df, wick_pct=0.40, vol_mult=1.5, rr=1.5):
    """More frequent scalp signals for daily trading."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['bb_lower']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Trend
        uptrend = price > row['ema50'] if not pd.isna(row['ema50']) else True
        downtrend = price < row['ema50'] if not pd.isna(row['ema50']) else True

        # BB touch
        bb_lower_touch = row['low'] <= row['bb_lower'] * 1.005
        bb_upper_touch = row['high'] >= row['bb_upper'] * 0.995

        # Channel touch
        dc_lower_touch = row['low'] <= row['dc_lower'] * 1.005
        dc_upper_touch = row['high'] >= row['dc_upper'] * 0.995

        # === LONG Signals ===
        # Type 1: BB lower + wick in uptrend
        if (uptrend and bb_lower_touch and lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and vol_ratio >= vol_mult and rsi < 45):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="BB bounce", timestamp=row['timestamp'],
            ))

        # Type 2: Channel lower + wick in uptrend
        elif (uptrend and dc_lower_touch and lower_wick_pct >= wick_pct and
              row['close'] > row['open'] and vol_ratio >= vol_mult and rsi < 45):
            sl = row['dc_lower'] - atr * 0.1
            tp = row['dc_mid']
            if tp > price:
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=0.72,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel bounce", timestamp=row['timestamp'],
                ))

        # Type 3: Volume climax
        consec_down = all(df.iloc[i-j]['close'] < df.iloc[i-j]['open'] for j in range(1, 3))
        if (vol_ratio >= 2.5 and consec_down and lower_wick_pct >= 0.50 and
            row['close'] > row['open'] and rsi < 40):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.85,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol climax", timestamp=row['timestamp'],
            ))

        # === SHORT Signals ===
        # Type 1: BB upper + wick in downtrend
        if (downtrend and bb_upper_touch and upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and vol_ratio >= vol_mult and rsi > 55):
            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr
            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="BB rejection", timestamp=row['timestamp'],
            ))

        # Type 2: Channel upper + wick in downtrend
        elif (downtrend and dc_upper_touch and upper_wick_pct >= wick_pct and
              row['close'] < row['open'] and vol_ratio >= vol_mult and rsi > 55):
            sl = row['dc_upper'] + atr * 0.1
            tp = row['dc_mid']
            if tp < price:
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=0.72,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason="Channel rejection", timestamp=row['timestamp'],
                ))

        # Type 3: Volume climax
        consec_up = all(df.iloc[i-j]['close'] > df.iloc[i-j]['open'] for j in range(1, 3))
        if (vol_ratio >= 2.5 and consec_up and upper_wick_pct >= 0.50 and
            row['close'] < row['open'] and rsi > 60):
            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr
            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.85,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason="Vol climax", timestamp=row['timestamp'],
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    df = add_indicators(df)

    # Generate signals
    signals = generate_scalp_signals(df)

    # Remove duplicates
    unique = []
    seen = set()
    for s in sorted(signals, key=lambda x: (x.timestamp, -x.confidence)):
        key = (s.timestamp, s.direction)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    signals = unique

    print(f'\nTotal signals: {len(signals)}')

    longs = sum(1 for s in signals if s.direction == 'LONG')
    shorts = sum(1 for s in signals if s.direction == 'SHORT')
    print(f'  LONG: {longs}, SHORT: {shorts}')

    print('\n' + '=' * 95)
    print('  DAILY LIMIT STRATEGY TEST (Max 3 trades/day)')
    print('=' * 95)

    print(f"\n{'Daily':>6} {'Risk':>6} {'Trades':>7} {'Days':>6} {'T/Day':>6} {'Win%':>7} {'Return':>10} {'PF':>7} {'MDD':>8}")
    print('-' * 85)

    best = None
    best_cfg = None

    for max_daily in [1, 2, 3]:
        for risk in [0.05, 0.10, 0.15, 0.20]:
            engine = DailyLimitBacktest(risk_per_trade=risk, max_daily_trades=max_daily)
            result = engine.run(df, signals)

            if result and result['trades'] >= 5:
                print(f"{max_daily:>6} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['trading_days']:>6} {result['trades_per_day']:>5.1f} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>7.2f} {result['max_drawdown']:>7.1f}%")

                if result['win_rate'] >= 60 and result['total_return'] > 0:
                    if best is None or result['total_return'] > best['total_return']:
                        best = result
                        best_cfg = {'max_daily': max_daily, 'risk': risk}

    # Test with higher risk for best config
    if best_cfg:
        print('\n' + '-' * 85)
        print(f"Best config (Daily={best_cfg['max_daily']}) - Testing higher risk:")
        print('-' * 85)

        for risk in [0.25, 0.30, 0.35, 0.40]:
            engine = DailyLimitBacktest(risk_per_trade=risk, max_daily_trades=best_cfg['max_daily'])
            result = engine.run(df, signals)

            if result:
                print(f"{best_cfg['max_daily']:>6} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['trading_days']:>6} {result['trades_per_day']:>5.1f} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>7.2f} {result['max_drawdown']:>7.1f}%")

                if result['total_return'] > best['total_return']:
                    best = result
                    best_cfg['risk'] = risk

    print('\n' + '=' * 95)

    if best:
        print(f"  BEST RESULT: Daily Limit {best_cfg['max_daily']}, Risk {best_cfg['risk']*100:.0f}%")
        print('=' * 95)
        print(f'''
  Trades: {best["trades"]}
  Trading Days: {best["trading_days"]}
  Trades/Day: {best["trades_per_day"]:.1f}
  Win Rate: {best["win_rate"]:.1f}%
  Total Return: {best["total_return"]:+.2f}%
  Profit Factor: {best["profit_factor"]:.2f}
  Max Drawdown: {best["max_drawdown"]:.2f}%
  Final Capital: ${best["final"]:,.2f}

  Monthly: {best["total_return"]/12:+.1f}%
  Weekly: {best["total_return"]/52:+.1f}%
''')

        # Trade breakdown
        if best['trade_list']:
            wins = [t for t in best['trade_list'] if t.pnl > 0]
            losses = [t for t in best['trade_list'] if t.pnl <= 0]

            print(f'  Wins: {len(wins)}, Losses: {len(losses)}')

            # Monthly breakdown
            print('\n  MONTHLY BREAKDOWN:')
            print('  ' + '-' * 55)
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
            positive_months = 0
            total_months = 0

            for month in sorted(monthly_pnl.keys()):
                pnl = monthly_pnl[month]
                trades = monthly_trades[month]
                wins = monthly_wins[month]
                ret = (pnl / cap) * 100
                cap += pnl
                sign = '+' if ret >= 0 else ''

                total_months += 1
                if ret > 0:
                    positive_months += 1

                wr = wins / trades * 100 if trades > 0 else 0
                print(f"    {month}: {trades:>2}거래 {wins}승({wr:.0f}%) {sign}{ret:>6.1f}%")

            print(f'\n  Positive Months: {positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)')

            # Sample trades
            print('\n  RECENT TRADES:')
            print('  ' + '-' * 70)
            for t in best['trade_list'][-10:]:
                sign = '+' if t.pnl >= 0 else ''
                status = 'WIN ' if t.pnl > 0 else 'LOSS'
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction:>5} "
                      f"${t.entry_price:,.0f} {sign}${t.pnl:,.0f} [{status}]")

    # Compare with other strategies
    print('\n' + '=' * 95)
    print('  COMPARISON')
    print('=' * 95)

    from scripts.run_backtest import generate_swing_signals
    swing_signals = generate_swing_signals(df)

    print(f"\n{'Strategy':<40} {'Trades':>7} {'Win%':>7} {'Return':>10} {'Monthly':>9}")
    print('-' * 80)

    # Swing
    from scripts.test_combined_high_wr import Backtest
    for risk in [0.10, 0.15]:
        engine = Backtest(risk_per_trade=risk)
        result = engine.run(df, swing_signals)
        if result:
            print(f"{'Swing (4h EMA) @ ' + str(int(risk*100)) + '%':<40} {result['trades']:>7} "
                  f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                  f"{result['total_return']/12:>+8.1f}%")

    # Daily limit
    if best:
        name = f"Daily Limit ({best_cfg['max_daily']}/day) @ {int(best_cfg['risk']*100)}%"
        print(f"{name:<40} {best['trades']:>7} "
              f"{best['win_rate']:>6.1f}% {best['total_return']:>+9.1f}% "
              f"{best['total_return']/12:>+8.1f}%")

    print('\n' + '=' * 95)


if __name__ == '__main__':
    main()
