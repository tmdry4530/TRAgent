"""BTC 50x Scalp Strategy V2 - Strict filters for higher win rate."""

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


class LeveragedScalpBacktest:
    """Scalp backtest with 50x leverage."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.02,
        commission: float = 0.0004,
        slippage: float = 0.0001,
        min_confidence: float = 0.65,
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
        if sl_distance == 0 or sl_distance > 0.03:
            return

        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_distance
        margin_required = position_value / self.leverage

        max_margin = self.capital * 0.30
        if margin_required > max_margin:
            margin_required = max_margin
            position_value = margin_required * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission

        if margin_required + entry_comm > self.capital * 0.95:
            return

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

        exit_value = exit_price * pos['size']
        exit_comm = exit_value * self.commission
        pnl -= exit_comm

        pnl_pct = (pnl / pos['margin']) * 100

        self.capital += pos['margin'] + pnl

        trade = Trade(
            entry_time=pos['entry_time'],
            exit_time=time,
            direction=direction,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            size=pos['size'],
            margin_used=pos['margin'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.trades.append(trade)
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

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        max_dd = float(np.max(dd))

        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'initial': self.initial_capital,
            'final': self.capital,
            'avg_pnl': np.mean([t.pnl for t in self.trades]),
            'avg_win': np.mean([t.pnl for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl for t in losses]) if losses else 0,
            'trade_list': self.trades,
        }


def generate_strict_scalp_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate scalp signals with STRICT filters - quality over quantity."""
    signals = []
    df = df.copy()

    # Indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Price momentum
    df['momentum'] = df['close'].pct_change(5)

    MIN_CONFIDENCE = 0.70  # Higher threshold

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['atr']) or pd.isna(row['rsi']) or pd.isna(row['ema50']):
            continue

        rsi = row['rsi']
        macd = row['macd']
        macd_sig = row['macd_signal']
        macd_hist = row['macd_hist']
        bb_pct = row['bb_pct'] if not pd.isna(row['bb_pct']) else 0.5
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        atr = row['atr']
        momentum = row['momentum'] if not pd.isna(row['momentum']) else 0

        # Strong trend alignment
        strong_uptrend = row['ema9'] > row['ema21'] > row['ema50']
        strong_downtrend = row['ema9'] < row['ema21'] < row['ema50']

        # MACD momentum
        macd_bullish = macd > macd_sig and macd_hist > prev['macd_hist']
        macd_bearish = macd < macd_sig and macd_hist < prev['macd_hist']

        # Consecutive candles
        bull_candles = row['close'] > row['open'] and prev['close'] > prev['open']
        bear_candles = row['close'] < row['open'] and prev['close'] < prev['open']

        # === STRICT SIGNAL: Multiple confirmation required ===

        # LONG: Strong uptrend + RSI not overbought + MACD bullish + Volume + Momentum
        long_conditions = [
            strong_uptrend,
            25 < rsi < 55,  # Not overbought, not too oversold
            macd_bullish,
            vol_ratio >= 1.5,
            momentum > 0,
            bull_candles,
        ]

        # SHORT: Strong downtrend + RSI not oversold + MACD bearish + Volume + Momentum
        short_conditions = [
            strong_downtrend,
            45 < rsi < 75,  # Not oversold, not too overbought
            macd_bearish,
            vol_ratio >= 1.5,
            momentum < 0,
            bear_candles,
        ]

        long_score = sum(long_conditions)
        short_score = sum(short_conditions)

        # Tighter SL/TP for scalp (R:R 1:1.2 for higher win rate)
        sl_distance = atr * 0.8
        tp_distance = atr * 1.0

        # Need 5/6 conditions for signal
        if long_score >= 5:
            conf = 0.70 + (long_score - 5) * 0.08
            if vol_ratio >= 2.0:
                conf += 0.05

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=min(0.90, conf),
                entry_price=price,
                stop_loss=price - sl_distance,
                take_profit=price + tp_distance,
                reason=f"Trend+MACD+Vol ({long_score}/6)",
                timestamp=timestamp,
            ))

        if short_score >= 5:
            conf = 0.70 + (short_score - 5) * 0.08
            if vol_ratio >= 2.0:
                conf += 0.05

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=min(0.90, conf),
                entry_price=price,
                stop_loss=price + sl_distance,
                take_profit=price - tp_distance,
                reason=f"Trend+MACD+Vol ({short_score}/6)",
                timestamp=timestamp,
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')

    signals = generate_strict_scalp_signals(df)
    print(f'Generated {len(signals)} STRICT scalp signals')

    if not signals:
        print("No signals!")
        return

    print('\n' + '=' * 70)
    print('  BTC 50x STRICT SCALP STRATEGY')
    print('=' * 70)

    print(f"\n{'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10}")
    print('-' * 60)

    best = None
    best_risk = None

    for risk in [0.02, 0.03, 0.05, 0.08, 0.10]:
        engine = LeveragedScalpBacktest(
            initial_capital=10000.0,
            leverage=50,
            risk_per_trade=risk,
            min_confidence=0.70,
        )
        result = engine.run(df, signals)

        if result:
            print(f"{risk*100:>7.0f}% {result['trades']:>8} "
                  f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                  f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}%")

            if result['profit_factor'] > 1.0:
                if best is None or result['total_return'] > best['total_return']:
                    best = result
                    best_risk = risk

    print('-' * 60)

    if best:
        print(f'\n  BEST CONFIG: {best_risk*100:.0f}% risk')
        print(f'  Trades: {best["trades"]} | Win: {best["win_rate"]:.1f}%')
        print(f'  Return: {best["total_return"]:+.2f}% | MDD: {best["max_drawdown"]:.2f}%')
        print(f'  PF: {best["profit_factor"]:.2f}')
        print(f'  Monthly avg: {best["total_return"]/12:+.1f}% | {best["trades"]/12:.1f} trades')

        # Monthly breakdown
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

    # Compare with Swing
    print('\n' + '=' * 70)
    print('  SCALP vs SWING')
    print('=' * 70)

    from scripts.test_btc_50x_leverage import LeveragedBacktest
    from scripts.run_backtest import generate_swing_signals

    swing_signals = generate_swing_signals(df)
    swing_engine = LeveragedBacktest(
        initial_capital=10000.0,
        leverage=50,
        risk_per_trade=0.10,
    )
    swing_result = swing_engine.run(df, swing_signals)

    print(f"\n{'Strategy':>12} {'Trades':>8} {'Win%':>8} {'Return':>12} {'Monthly':>10}")
    print('-' * 55)

    if best:
        print(f"{'SCALP':>12} {best['trades']:>8} {best['win_rate']:>7.1f}% "
              f"{best['total_return']:>+11.2f}% {best['total_return']/12:>+9.1f}%")

    if swing_result:
        print(f"{'SWING':>12} {swing_result['trades']:>8} {swing_result['win_rate']:>7.1f}% "
              f"{swing_result['total_return']:>+11.2f}% {swing_result['total_return']/12:>+9.1f}%")

    # Combined strategy
    if best and swing_result:
        combined_return = best['total_return'] + swing_result['total_return']
        combined_trades = best['trades'] + swing_result['trades']
        print(f"{'COMBINED':>12} {combined_trades:>8} {'---':>8} "
              f"{combined_return:>+11.2f}% {combined_return/12:>+9.1f}%")

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
