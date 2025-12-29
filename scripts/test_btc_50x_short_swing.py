"""BTC 50x Short Swing - 1h timeframe swing for more trades."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from src.signals.base import Signal


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    margin_used: float
    pnl: float
    pnl_pct: float
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
        margin_required = position_value / self.leverage
        max_margin = self.capital * 0.90
        if margin_required > max_margin:
            margin_required = max_margin
            position_value = margin_required * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission
        self.capital -= (margin_required + entry_comm)

        self.current_position = {
            'signal': signal,
            'entry_price': entry_price,
            'entry_time': time,
            'size': size,
            'margin': margin_required,
            'position_value': position_value,
            'direction': signal.direction,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
        }

    def _check_exit(self, row):
        if not self.current_position:
            return
        pos = self.current_position
        high, low = row['high'], row['low']

        if pos['direction'] == 'LONG':
            if low <= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif high >= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')
        else:
            if high >= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif low <= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')

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
        pnl_pct = (pnl / pos['margin']) * 100
        self.capital += pos['margin'] + pnl

        self.trades.append(Trade(
            entry_time=pos['entry_time'], exit_time=time, direction=direction,
            entry_price=pos['entry_price'], exit_price=exit_price, size=pos['size'],
            margin_used=pos['margin'], pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason,
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
        win_rate = len(wins) / len(self.trades) * 100
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        max_dd = float(np.max(dd))
        return {
            'trades': len(self.trades), 'wins': len(wins), 'losses': len(losses),
            'win_rate': win_rate, 'total_return': total_return,
            'profit_factor': pf, 'max_drawdown': max_dd,
            'initial': self.initial_capital, 'final': self.capital,
            'trade_list': self.trades,
        }


def generate_short_swing_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate swing signals on 1h timeframe (not 4h) for more frequency."""
    signals = []
    df = df.copy()

    # EMAs on 1h
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    MIN_CONFIDENCE = 0.60

    for i in range(60, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['macd']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr']
        macd = row['macd']
        macd_sig = row['macd_signal']
        macd_hist = row['macd_hist']

        # EMA alignment
        bullish = row['ema12'] > row['ema26'] > row['ema50']
        bearish = row['ema12'] < row['ema26'] < row['ema50']

        # EMA crossover
        prev_bull = prev['ema12'] > prev['ema26'] if not pd.isna(prev['ema26']) else False
        prev_bear = prev['ema12'] < prev['ema26'] if not pd.isna(prev['ema26']) else False

        bull_cross = bullish and not prev_bull
        bear_cross = bearish and not prev_bear

        # MACD confirmation
        macd_bull = macd > macd_sig
        macd_bear = macd < macd_sig

        # RSI zone
        rsi_ok_long = 35 <= rsi <= 60
        rsi_ok_short = 40 <= rsi <= 65

        # SL/TP (same R:R as original swing but based on ATR)
        sl_distance = atr * 3  # 3x ATR for swing
        tp_distance = sl_distance * 3  # R:R 1:3

        # Confidence based on EMA gap
        def calc_conf(ema12, ema26, ema50):
            conf = 0.55
            gap1 = abs(ema12 - ema26) / ema26
            gap2 = abs(ema26 - ema50) / ema50
            if gap1 > 0.02:
                conf += 0.10
            if gap2 > 0.02:
                conf += 0.10
            return min(0.85, conf)

        # LONG
        if bull_cross and macd_bull and rsi_ok_long:
            conf = calc_conf(row['ema12'], row['ema26'], row['ema50'])
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=conf,
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason=f"1h EMA cross + MACD + RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

        # SHORT
        if bear_cross and macd_bear and rsi_ok_short:
            conf = calc_conf(row['ema12'], row['ema26'], row['ema50'])
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=conf,
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason=f"1h EMA cross + MACD + RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')

    # 1h Short Swing signals
    short_swing = generate_short_swing_signals(df)
    print(f'1h Short Swing signals: {len(short_swing)}')

    # 4h Original Swing signals
    from scripts.run_backtest import generate_swing_signals
    orig_swing = generate_swing_signals(df)
    print(f'4h Original Swing signals: {len(orig_swing)}')

    print('\n' + '=' * 70)
    print('  TIMEFRAME COMPARISON (50x Leverage, 10% Risk)')
    print('=' * 70)

    # Test 1h swing
    print('\n--- 1H SHORT SWING ---')
    if short_swing:
        print(f"\n{'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10}")
        print('-' * 60)

        for risk in [0.05, 0.10, 0.15]:
            engine = LeveragedBacktest(
                initial_capital=10000.0,
                leverage=50,
                risk_per_trade=risk,
            )
            result = engine.run(df, short_swing)
            if result:
                print(f"{risk*100:>7.0f}% {result['trades']:>8} "
                      f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                      f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}%")

    # Test 4h swing
    print('\n--- 4H ORIGINAL SWING ---')
    if orig_swing:
        print(f"\n{'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10}")
        print('-' * 60)

        for risk in [0.05, 0.10, 0.15]:
            engine = LeveragedBacktest(
                initial_capital=10000.0,
                leverage=50,
                risk_per_trade=risk,
            )
            result = engine.run(df, orig_swing)
            if result:
                print(f"{risk*100:>7.0f}% {result['trades']:>8} "
                      f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                      f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}%")

    # Combined
    print('\n--- COMBINED (1H + 4H) ---')
    combined = short_swing + orig_swing
    combined.sort(key=lambda s: s.timestamp)
    print(f'Combined signals: {len(combined)}')

    if combined:
        print(f"\n{'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10}")
        print('-' * 60)

        best = None
        best_risk = None

        for risk in [0.05, 0.08, 0.10]:
            engine = LeveragedBacktest(
                initial_capital=10000.0,
                leverage=50,
                risk_per_trade=risk,
            )
            result = engine.run(df, combined)
            if result:
                print(f"{risk*100:>7.0f}% {result['trades']:>8} "
                      f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                      f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}%")

                if result['profit_factor'] > 1.0:
                    if best is None or result['total_return'] > best['total_return']:
                        best = result
                        best_risk = risk

        if best:
            print('\n' + '=' * 70)
            print(f'  BEST COMBINED RESULT ({best_risk*100:.0f}% risk)')
            print('=' * 70)
            print(f'  Trades:        {best["trades"]}')
            print(f'  Win Rate:      {best["win_rate"]:.1f}%')
            print(f'  Total Return:  {best["total_return"]:+.2f}%')
            print(f'  Profit Factor: {best["profit_factor"]:.2f}')
            print(f'  Max Drawdown:  {best["max_drawdown"]:.2f}%')
            print()
            print(f'  Monthly avg:   {best["trades"]/12:.1f} trades')
            print(f'  Monthly return:{best["total_return"]/12:+.1f}%')

            # Monthly
            if best['trade_list']:
                print('\n  MONTHLY BREAKDOWN')
                print('  ' + '-' * 45)
                from collections import defaultdict
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

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
