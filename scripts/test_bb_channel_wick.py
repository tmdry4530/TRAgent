"""BB + Channel + Wick Strategy - User's actual trading style."""

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

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # EMAs
    for span in [10, 20, 50, 100, 200]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # Bollinger Bands (20, 2)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # Bollinger Bands (20, 2.5) - wider for extreme
    df['bb_upper_wide'] = df['bb_mid'] + 2.5 * df['bb_std']
    df['bb_lower_wide'] = df['bb_mid'] - 2.5 * df['bb_std']

    # Donchian Channel (20)
    df['dc_upper'] = df['high'].rolling(20).max()
    df['dc_lower'] = df['low'].rolling(20).min()
    df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2

    # Keltner Channel (20, 1.5 ATR)
    df['kc_mid'] = df['close'].ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['kc_mid'] + 1.5 * df['atr']
    df['kc_lower'] = df['kc_mid'] - 1.5 * df['atr']

    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Candle analysis
    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    return df


def bb_channel_wick_signals(df, bb_touch=True, channel_touch=True, wick_pct=0.50,
                            vol_mult=1.5, rr_ratio=1.5):
    """BB + Channel + Wick combined signals."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['bb_lower']) or pd.isna(row['dc_lower']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # === LONG CONDITIONS ===
        # 1. BB lower touch
        bb_lower_touch = row['low'] <= row['bb_lower'] * 1.002

        # 2. Channel lower touch (Donchian or Keltner)
        dc_lower_touch = row['low'] <= row['dc_lower'] * 1.005
        kc_lower_touch = row['low'] <= row['kc_lower'] * 1.002

        # 3. Wick confirmation
        has_lower_wick = lower_wick_pct >= wick_pct and row['close'] > row['open']

        # 4. Volume confirmation
        has_volume = vol_ratio >= vol_mult

        # Combined LONG signal
        long_conditions = []
        if bb_touch:
            long_conditions.append(bb_lower_touch)
        if channel_touch:
            long_conditions.append(dc_lower_touch or kc_lower_touch)

        if (any(long_conditions) and has_lower_wick and has_volume and rsi < 40):
            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            # Confidence based on confluence
            conf = 0.65
            if bb_lower_touch:
                conf += 0.05
            if dc_lower_touch:
                conf += 0.05
            if kc_lower_touch:
                conf += 0.05
            if vol_ratio >= 2.0:
                conf += 0.05
            if rsi < 30:
                conf += 0.05

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=min(0.90, conf),
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"BB+Channel bounce RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # === SHORT CONDITIONS ===
        bb_upper_touch = row['high'] >= row['bb_upper'] * 0.998
        dc_upper_touch = row['high'] >= row['dc_upper'] * 0.995
        kc_upper_touch = row['high'] >= row['kc_upper'] * 0.998

        has_upper_wick = upper_wick_pct >= wick_pct and row['close'] < row['open']

        short_conditions = []
        if bb_touch:
            short_conditions.append(bb_upper_touch)
        if channel_touch:
            short_conditions.append(dc_upper_touch or kc_upper_touch)

        if (any(short_conditions) and has_upper_wick and has_volume and rsi > 60):
            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            conf = 0.65
            if bb_upper_touch:
                conf += 0.05
            if dc_upper_touch:
                conf += 0.05
            if kc_upper_touch:
                conf += 0.05
            if vol_ratio >= 2.0:
                conf += 0.05
            if rsi > 70:
                conf += 0.05

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=min(0.90, conf),
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"BB+Channel rejection RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def bb_squeeze_reversal(df, wick_pct=0.50, rr_ratio=1.5):
    """BB squeeze followed by reversal at band touch."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['bb_width']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Check for squeeze (BB inside Keltner)
        bb_inside_kc = (row['bb_upper'] < row['kc_upper'] and
                        row['bb_lower'] > row['kc_lower'])

        # Recent squeeze
        recent_squeeze = any(
            df.iloc[i-j]['bb_upper'] < df.iloc[i-j]['kc_upper'] and
            df.iloc[i-j]['bb_lower'] > df.iloc[i-j]['kc_lower']
            for j in range(1, 6)
            if not pd.isna(df.iloc[i-j]['bb_upper'])
        )

        # LONG: After squeeze, touch BB lower with wick
        if (recent_squeeze and
            row['low'] <= row['bb_lower'] * 1.002 and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= 1.5 and
            rsi < 45):

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Squeeze + BB bounce",
                timestamp=row['timestamp'],
            ))

        # SHORT: After squeeze, touch BB upper with wick
        if (recent_squeeze and
            row['high'] >= row['bb_upper'] * 0.998 and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            vol_ratio >= 1.5 and
            rsi > 55):

            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Squeeze + BB rejection",
                timestamp=row['timestamp'],
            ))

    return signals


def extreme_bb_reversal(df, wick_pct=0.55, rr_ratio=1.2):
    """Extreme BB touch (2.5 std) with strong reversal."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['bb_lower_wide']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # LONG: Touch wide BB lower (2.5 std)
        if (row['low'] <= row['bb_lower_wide'] and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= 2.0 and
            rsi < 30):

            sl = row['low'] - atr * 0.05
            tp = price + (price - sl) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Extreme BB bounce RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # SHORT: Touch wide BB upper
        if (row['high'] >= row['bb_upper_wide'] and
            upper_wick_pct >= wick_pct and
            row['close'] < row['open'] and
            vol_ratio >= 2.0 and
            rsi > 70):

            sl = row['high'] + atr * 0.05
            tp = price - (sl - price) * rr_ratio

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Extreme BB rejection RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def channel_bounce_wick(df, wick_pct=0.50, rr_ratio=1.5):
    """Donchian channel bounce with wick confirmation."""
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

        # Trend context
        uptrend = price > row['ema50'] if not pd.isna(row['ema50']) else True
        downtrend = price < row['ema50'] if not pd.isna(row['ema50']) else True

        # LONG: Touch channel lower in uptrend
        if (uptrend and
            row['low'] <= row['dc_lower'] * 1.002 and
            lower_wick_pct >= wick_pct and
            row['close'] > row['open'] and
            vol_ratio >= 1.5 and
            rsi < 45):

            sl = row['dc_lower'] - atr * 0.1
            tp = row['dc_mid']  # Target mid-channel

            if tp > price:  # Only if TP is above entry
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=0.72,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason=f"Channel bounce",
                    timestamp=row['timestamp'],
                ))

        # SHORT: Touch channel upper in downtrend
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
                    type='SCALP', direction='SHORT', confidence=0.72,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason=f"Channel rejection",
                    timestamp=row['timestamp'],
                ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    df = add_indicators(df)

    print('\n' + '=' * 90)
    print('  BB + CHANNEL + WICK STRATEGY TEST')
    print('=' * 90)

    # Test different strategies
    strategies = [
        ("BB + Channel + Wick", bb_channel_wick_signals(df)),
        ("BB Squeeze Reversal", bb_squeeze_reversal(df)),
        ("Extreme BB (2.5std)", extreme_bb_reversal(df)),
        ("Channel Bounce", channel_bounce_wick(df)),
    ]

    print(f"\n{'Strategy':<25} {'Sigs':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7}")
    print('-' * 70)

    best_results = []

    for name, signals in strategies:
        if not signals:
            print(f"{name:<25} {'0':>6}")
            continue

        # Remove duplicates
        unique = []
        seen = set()
        for s in sorted(signals, key=lambda x: (x.timestamp, -x.confidence)):
            key = (s.timestamp, s.direction)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        signals = unique

        engine = Backtest(risk_per_trade=0.10)
        result = engine.run(df, signals)

        if result:
            print(f"{name:<25} {len(signals):>6} {result['trades']:>7} "
                  f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                  f"{result['profit_factor']:>7.2f}")

            best_results.append({
                'name': name,
                'signals': signals,
                'result': result,
            })

    # Test combined signals with different R:R
    print('\n' + '=' * 90)
    print('  COMBINED SIGNALS - R:R OPTIMIZATION')
    print('=' * 90)

    # Combine all signals
    all_signals = []
    for name, sigs in strategies:
        all_signals.extend(sigs)

    # Remove duplicates
    unique = []
    seen = set()
    for s in sorted(all_signals, key=lambda x: (x.timestamp, -x.confidence)):
        key = (s.timestamp, s.direction)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    all_signals = unique

    print(f"\nTotal combined signals: {len(all_signals)}")

    print(f"\n{'R:R':>5} {'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7} {'MDD':>8}")
    print('-' * 60)

    best = None
    best_cfg = None

    for rr in [0.8, 1.0, 1.2, 1.5, 2.0]:
        # Adjust TP based on R:R
        adjusted_signals = []
        for s in all_signals:
            sl_dist = abs(s.entry_price - s.stop_loss)
            if s.direction == 'LONG':
                new_tp = s.entry_price + sl_dist * rr
            else:
                new_tp = s.entry_price - sl_dist * rr

            adjusted_signals.append(Signal(
                type=s.type, direction=s.direction, confidence=s.confidence,
                entry_price=s.entry_price, stop_loss=s.stop_loss, take_profit=new_tp,
                reason=s.reason, timestamp=s.timestamp,
            ))

        for risk in [0.10, 0.15, 0.20]:
            engine = Backtest(risk_per_trade=risk)
            result = engine.run(df, adjusted_signals)

            if result and result['trades'] >= 5:
                print(f"{rr:>5.1f} {risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>7.2f} {result['max_drawdown']:>7.1f}%")

                if result['win_rate'] >= 60 and result['total_return'] > 0:
                    if best is None or result['total_return'] > best['total_return']:
                        best = result
                        best_cfg = {'rr': rr, 'risk': risk}

    # Best result details
    print('\n' + '=' * 90)

    if best and best['win_rate'] >= 60:
        print(f'  BEST RESULT: R:R {best_cfg["rr"]}, Risk {best_cfg["risk"]*100:.0f}%')
        print('=' * 90)
        print(f'''
  Trades: {best["trades"]}
  Win Rate: {best["win_rate"]:.1f}%
  Return: {best["total_return"]:+.2f}%
  PF: {best["profit_factor"]:.2f}
  Max DD: {best["max_drawdown"]:.2f}%
  Monthly: {best["total_return"]/12:+.1f}%
''')

        if best['trade_list']:
            wins = sum(1 for t in best['trade_list'] if t.pnl > 0)
            losses = sum(1 for t in best['trade_list'] if t.pnl <= 0)
            print(f'  Wins: {wins}, Losses: {losses}')

            print('\n  TRADES:')
            for t in best['trade_list']:
                sign = '+' if t.pnl >= 0 else ''
                status = 'WIN' if t.pnl > 0 else 'LOSS'
                print(f"    {t.entry_time.strftime('%Y-%m-%d %H:%M')} {t.direction:>5} "
                      f"${t.entry_price:,.0f} {sign}${t.pnl:,.0f} [{status}]")

    else:
        print('  HIGH WIN RATE RESULT NOT FOUND')
        print('=' * 90)

        if best_results:
            best_r = max(best_results, key=lambda x: x['result']['total_return'])
            print(f'''
  Best Available: {best_r["name"]}
  - Win Rate: {best_r["result"]["win_rate"]:.1f}%
  - Return: {best_r["result"]["total_return"]:+.2f}%
''')

    # Compare with swing
    print('\n' + '=' * 90)
    print('  COMPARISON')
    print('=' * 90)

    from scripts.run_backtest import generate_swing_signals
    swing_signals = generate_swing_signals(df)
    swing_engine = Backtest(risk_per_trade=0.10)
    swing_result = swing_engine.run(df, swing_signals)

    print(f"\n{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Return':>10}")
    print('-' * 60)

    if swing_result:
        print(f"{'Swing (4h EMA)':30} {swing_result['trades']:>7} "
              f"{swing_result['win_rate']:>6.1f}% {swing_result['total_return']:>+9.1f}%")

    if best:
        print(f"{'BB+Channel+Wick':30} {best['trades']:>7} "
              f"{best['win_rate']:>6.1f}% {best['total_return']:>+9.1f}%")

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
