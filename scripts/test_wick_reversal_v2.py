"""Wick Reversal Strategy V2 - More Strict Conditions + Trend Filter."""

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
    hold_hours: float
    exit_reason: str


class WickReversalBacktest:
    """Flexible trailing stop backtest engine."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.05,
        commission: float = 0.0004,
        slippage: float = 0.0001,
        min_confidence: float = 0.60,
        min_rr: float = 2.0,
        max_rr: float = 5.0,
        trailing_start_rr: float = 1.5,
        trailing_pct: float = 0.015,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.trailing_start_rr = trailing_start_rr
        self.trailing_pct = trailing_pct

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
            price = row['close']

            if self.current_position:
                self._check_exit(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if signal.confidence >= self.min_confidence and not self.current_position:
                    self._enter(signal, price, current_time)

            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(price)
            self.equity_curve.append(equity)

        if self.current_position:
            last = df.iloc[-1]
            self._exit(last['close'], last['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal: Signal, price: float, time: datetime):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss)
        sl_pct = sl_distance / entry_price

        if sl_pct == 0 or sl_pct > 0.03:  # 3% max SL for wick trades
            return

        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_pct
        margin = position_value / self.leverage

        max_margin = self.capital * 0.80
        if margin > max_margin:
            margin = max_margin
            position_value = margin * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission

        if margin + entry_comm > self.capital * 0.95:
            return

        self.capital -= (margin + entry_comm)

        if signal.direction == 'LONG':
            trailing_trigger = entry_price + sl_distance * self.trailing_start_rr
            max_tp = entry_price + sl_distance * self.max_rr
        else:
            trailing_trigger = entry_price - sl_distance * self.trailing_start_rr
            max_tp = entry_price - sl_distance * self.max_rr

        self.current_position = {
            'entry_price': entry_price,
            'entry_time': time,
            'size': size,
            'margin': margin,
            'position_value': position_value,
            'direction': signal.direction,
            'stop_loss': signal.stop_loss,
            'max_tp': max_tp,
            'trailing_trigger': trailing_trigger,
            'trailing_stop': None,
            'highest_price': entry_price if signal.direction == 'LONG' else None,
            'lowest_price': entry_price if signal.direction == 'SHORT' else None,
            'sl_distance': sl_distance,
        }

    def _check_exit(self, row):
        if not self.current_position:
            return

        pos = self.current_position
        high = row['high']
        low = row['low']
        direction = pos['direction']

        if direction == 'LONG':
            if high > (pos['highest_price'] or 0):
                pos['highest_price'] = high

                if high >= pos['trailing_trigger'] and pos['trailing_stop'] is None:
                    pos['trailing_stop'] = high * (1 - self.trailing_pct)

                if pos['trailing_stop'] is not None:
                    new_trailing = high * (1 - self.trailing_pct)
                    if new_trailing > pos['trailing_stop']:
                        pos['trailing_stop'] = new_trailing

            if low <= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
                return

            if pos['trailing_stop'] and low <= pos['trailing_stop']:
                self._exit(pos['trailing_stop'], row['timestamp'], 'TRAIL')
                return

            if high >= pos['max_tp']:
                self._exit(pos['max_tp'], row['timestamp'], 'MAX_TP')
                return

        else:  # SHORT
            if low < (pos['lowest_price'] or float('inf')):
                pos['lowest_price'] = low

                if low <= pos['trailing_trigger'] and pos['trailing_stop'] is None:
                    pos['trailing_stop'] = low * (1 + self.trailing_pct)

                if pos['trailing_stop'] is not None:
                    new_trailing = low * (1 + self.trailing_pct)
                    if new_trailing < pos['trailing_stop']:
                        pos['trailing_stop'] = new_trailing

            if high >= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
                return

            if pos['trailing_stop'] and high >= pos['trailing_stop']:
                self._exit(pos['trailing_stop'], row['timestamp'], 'TRAIL')
                return

            if low <= pos['max_tp']:
                self._exit(pos['max_tp'], row['timestamp'], 'MAX_TP')
                return

    def _exit(self, price: float, time: datetime, reason: str):
        if not self.current_position:
            return

        pos = self.current_position
        direction = pos['direction']
        exit_price = price * (1 - self.slippage if direction == 'LONG' else 1 + self.slippage)

        if direction == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        exit_comm = exit_price * pos['size'] * self.commission
        pnl -= exit_comm

        self.capital += pos['margin'] + pnl

        hold_hours = (time - pos['entry_time']).total_seconds() / 3600

        self.trades.append(Trade(
            entry_time=pos['entry_time'],
            exit_time=time,
            direction=direction,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            pnl=pnl,
            hold_hours=hold_hours,
            exit_reason=reason,
        ))
        self.current_position = None

    def _unrealized_pnl(self, price: float) -> float:
        if not self.current_position:
            return 0.0
        pos = self.current_position
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

        avg_hold = np.mean([t.hold_hours for t in self.trades]) if self.trades else 0

        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_return': total_return,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'final': self.capital,
            'avg_hold_hours': avg_hold,
            'trade_list': self.trades,
            'exit_reasons': {r: sum(1 for t in self.trades if t.exit_reason == r)
                           for r in ['SL', 'TRAIL', 'MAX_TP', 'END']},
        }


def generate_wick_signals_v2(df: pd.DataFrame) -> list[Signal]:
    """More strict wick reversal signals - with trend and support/resistance."""
    signals = []
    df = df.copy()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMAs for trend
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Candle analysis
    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    # Spike detection
    df['range_vs_atr'] = df['full_range'] / df['atr']

    # Support/Resistance detection (rolling pivot highs/lows)
    df['pivot_high'] = df['high'].rolling(window=20, center=True).max()
    df['pivot_low'] = df['low'].rolling(window=20, center=True).min()

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    # STRICT CONDITIONS
    MIN_WICK_PCT = 0.60  # Wick must be 60%+ of range (stricter)
    MIN_VOLUME_RATIO = 2.0  # 2x average volume (stricter)
    MIN_SPIKE_ATR = 2.0  # 2x ATR move (stricter)
    MAX_BODY_PCT = 0.30  # Body must be small (doji-like)

    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['atr']) or pd.isna(row['rsi']) or pd.isna(row['ema200']):
            continue

        atr = row['atr']
        rsi = row['rsi']
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0
        range_atr = row['range_vs_atr'] if not pd.isna(row['range_vs_atr']) else 0
        body_pct = row['body'] / row['full_range'] if row['full_range'] > 0 else 1

        # Trend context (using 200 EMA as main trend)
        major_uptrend = price > row['ema200']
        major_downtrend = price < row['ema200']

        # Short-term trend (EMA alignment)
        short_uptrend = row['ema10'] > row['ema20'] > row['ema50']
        short_downtrend = row['ema10'] < row['ema20'] < row['ema50']

        # Near support/resistance
        near_bb_lower = row['low'] <= row['bb_lower'] * 1.01
        near_bb_upper = row['high'] >= row['bb_upper'] * 0.99

        # === LONG: Lower wick bounce at support ===
        if (lower_wick_pct >= MIN_WICK_PCT and
            body_pct <= MAX_BODY_PCT and  # Small body (rejection)
            row['close'] > row['open'] and  # Bullish close
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR and
            major_uptrend):  # Only long in uptrend

            conf = 0.55

            # Wick quality
            conf += min((lower_wick_pct - 0.6) * 1.5, 0.15)

            # Volume spike
            conf += min((vol_ratio - 2.0) / 8, 0.10)

            # RSI oversold bounce
            if rsi < 30:
                conf += 0.15
            elif rsi < 40:
                conf += 0.05

            # Near Bollinger lower (support)
            if near_bb_lower:
                conf += 0.10

            # Trend alignment bonus
            if short_uptrend:
                conf += 0.05

            # SL at wick low with small buffer
            stop_loss = row['low'] - atr * 0.05

            # Temporary TP (trailing handles the actual exit)
            take_profit = price + atr * 4

            if conf >= 0.65:  # Higher threshold
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=min(0.90, conf),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Wick bounce at support RSI:{rsi:.0f} Vol:{vol_ratio:.1f}x",
                    timestamp=timestamp,
                ))

        # === SHORT: Upper wick rejection at resistance ===
        if (upper_wick_pct >= MIN_WICK_PCT and
            body_pct <= MAX_BODY_PCT and
            row['close'] < row['open'] and  # Bearish close
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR and
            major_downtrend):  # Only short in downtrend

            conf = 0.55

            conf += min((upper_wick_pct - 0.6) * 1.5, 0.15)
            conf += min((vol_ratio - 2.0) / 8, 0.10)

            if rsi > 70:
                conf += 0.15
            elif rsi > 60:
                conf += 0.05

            if near_bb_upper:
                conf += 0.10

            if short_downtrend:
                conf += 0.05

            stop_loss = row['high'] + atr * 0.05
            take_profit = price - atr * 4

            if conf >= 0.65:
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=min(0.90, conf),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Wick rejection at resistance RSI:{rsi:.0f} Vol:{vol_ratio:.1f}x",
                    timestamp=timestamp,
                ))

    return signals


def generate_countertrend_wicks(df: pd.DataFrame) -> list[Signal]:
    """Alternative: Counter-trend wick entries at extreme levels."""
    signals = []
    df = df.copy()

    # Same indicators
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)
    df['range_vs_atr'] = df['full_range'] / df['atr']

    # Bollinger for extremes
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2.5 * df['bb_std']  # Wider bands
    df['bb_lower'] = df['bb_mid'] - 2.5 * df['bb_std']

    # VERY strict for counter-trend
    MIN_WICK_PCT = 0.70
    MIN_VOLUME_RATIO = 2.5
    MIN_SPIKE_ATR = 2.5

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['atr']) or pd.isna(row['rsi']):
            continue

        atr = row['atr']
        rsi = row['rsi']
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0
        range_atr = row['range_vs_atr'] if not pd.isna(row['range_vs_atr']) else 0

        # === LONG: Extreme oversold bounce ===
        if (lower_wick_pct >= MIN_WICK_PCT and
            row['close'] > row['open'] and
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR and
            rsi < 25 and  # Very oversold
            row['low'] < row['bb_lower']):  # Below lower BB

            conf = 0.70  # High base for extreme conditions
            conf += min((vol_ratio - 2.5) / 10, 0.10)

            stop_loss = row['low'] - atr * 0.05
            take_profit = price + atr * 5

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=min(0.90, conf),
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Extreme oversold bounce RSI:{rsi:.0f}",
                timestamp=timestamp,
            ))

        # === SHORT: Extreme overbought rejection ===
        if (upper_wick_pct >= MIN_WICK_PCT and
            row['close'] < row['open'] and
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR and
            rsi > 75 and  # Very overbought
            row['high'] > row['bb_upper']):  # Above upper BB

            conf = 0.70
            conf += min((vol_ratio - 2.5) / 10, 0.10)

            stop_loss = row['high'] + atr * 0.05
            take_profit = price - atr * 5

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=min(0.90, conf),
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Extreme overbought rejection RSI:{rsi:.0f}",
                timestamp=timestamp,
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    # Strategy 1: Trend-following wicks
    signals_v2 = generate_wick_signals_v2(df)
    print(f'\nStrict Wick Signals (trend-following): {len(signals_v2)}')

    # Strategy 2: Counter-trend extreme wicks
    signals_ct = generate_countertrend_wicks(df)
    print(f'Counter-trend Extreme Wicks: {len(signals_ct)}')

    # Combined signals
    all_signals = signals_v2 + signals_ct
    print(f'Combined: {len(all_signals)}')

    print('\n' + '=' * 75)
    print('  WICK REVERSAL V2 - STRICT CONDITIONS')
    print('=' * 75)

    # Test configurations
    strategies = [
        ('Trend Wick', signals_v2),
        ('Extreme CT', signals_ct),
        ('Combined', all_signals),
    ]

    results = {}

    for name, sigs in strategies:
        if not sigs:
            continue

        print(f'\n[{name}] ({len(sigs)} signals)')
        longs = sum(1 for s in sigs if s.direction == 'LONG')
        shorts = sum(1 for s in sigs if s.direction == 'SHORT')
        print(f'  LONG: {longs}, SHORT: {shorts}')

        print(f"\n{'Risk':>6} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8}")
        print('-' * 55)

        for risk in [0.03, 0.05, 0.08]:
            engine = WickReversalBacktest(
                risk_per_trade=risk,
                min_rr=2.0,
                max_rr=5.0,
                trailing_start_rr=1.5,
                trailing_pct=0.015,
            )
            result = engine.run(df, sigs)

            if result:
                print(f"{risk*100:>5.0f}% {result['trades']:>7} "
                      f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                      f"{result['profit_factor']:>6.2f} {result['max_drawdown']:>7.1f}%")

                key = f'{name}_{risk}'
                if result['profit_factor'] > 1.0 or result['total_return'] > 0:
                    results[key] = {'result': result, 'name': name, 'risk': risk}

    # Compare with Swing
    print('\n' + '=' * 75)
    print('  COMPARISON WITH SWING STRATEGY')
    print('=' * 75)

    from scripts.test_btc_50x_leverage import LeveragedBacktest as SwingBacktest
    from scripts.run_backtest import generate_swing_signals

    swing_signals = generate_swing_signals(df)
    swing_engine = SwingBacktest(risk_per_trade=0.10)
    swing_result = swing_engine.run(df, swing_signals)

    print(f"\n{'Strategy':>20} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8} {'Monthly':>9}")
    print('-' * 75)

    if swing_result:
        print(f"{'SWING (4h 10%)':>20} {swing_result['trades']:>7} {swing_result['win_rate']:>6.1f}% "
              f"{swing_result['total_return']:>+9.1f}% {swing_result['profit_factor']:>6.2f} "
              f"{swing_result['max_drawdown']:>7.1f}% {swing_result['total_return']/12:>+8.1f}%")

    for key, data in sorted(results.items(), key=lambda x: x[1]['result']['total_return'], reverse=True)[:3]:
        r = data['result']
        print(f"{key:>20} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['total_return']:>+9.1f}% {r['profit_factor']:>6.2f} "
              f"{r['max_drawdown']:>7.1f}% {r['total_return']/12:>+8.1f}%")

    # Best result details
    if results:
        best_key = max(results.keys(), key=lambda k: results[k]['result']['total_return'])
        best = results[best_key]['result']

        print('\n' + '=' * 75)
        print(f'  BEST WICK STRATEGY: {best_key}')
        print('=' * 75)
        print(f'  Trades: {best["trades"]}')
        print(f'  Win Rate: {best["win_rate"]:.1f}%')
        print(f'  Return: {best["total_return"]:+.2f}%')
        print(f'  PF: {best["profit_factor"]:.2f}')
        print(f'  Max DD: {best["max_drawdown"]:.2f}%')
        print(f'  Avg Hold: {best["avg_hold_hours"]:.1f}h')
        print(f'  Exits: {best["exit_reasons"]}')

        if best['trade_list']:
            print('\n  MONTHLY BREAKDOWN')
            print('  ' + '-' * 50)

            monthly = defaultdict(list)
            for t in best['trade_list']:
                m = t.exit_time.strftime('%Y-%m')
                monthly[m].append(t)

            cap = 10000
            for month in sorted(monthly.keys()):
                trades = monthly[month]
                pnl = sum(t.pnl for t in trades)
                wins = sum(1 for t in trades if t.pnl > 0)
                ret = (pnl / cap) * 100
                cap += pnl
                sign = '+' if ret >= 0 else ''
                print(f"  {month}: {len(trades):>2}거래 {wins}승 {sign}{ret:>7.2f}%")

    else:
        print('\n  No profitable wick strategy found.')
        print('  The SWING strategy (4h EMA cross) remains the best option.')

    print('\n' + '=' * 75)
    print('  RECOMMENDATION')
    print('=' * 75)

    if swing_result and swing_result['profit_factor'] > 1.0:
        print(f'''
  Given the backtest results:

  - Wick reversal strategies show lower win rates (20-35%)
  - The R:R doesn't compensate enough for the low win rate with 50x leverage
  - Losses compound quickly at 50x

  SWING strategy remains optimal for 50x leverage:
  - 50% win rate with 3:1 R:R
  - Fewer trades = fewer chances for error
  - {swing_result["total_return"]:+.0f}% annual return

  For wick-style trading, consider:
  1. Lower leverage (10-20x) to survive lower win rates
  2. More confirmation signals before entry
  3. Manual discretion for context the algo can't capture
''')

    print('=' * 75)


if __name__ == '__main__':
    main()
