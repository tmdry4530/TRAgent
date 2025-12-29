"""BTC 50x Scalp Strategy - High frequency trading."""

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
        risk_per_trade: float = 0.02,  # 2% risk per scalp
        commission: float = 0.0004,
        slippage: float = 0.0001,  # Tighter slippage for scalp
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
        if sl_distance == 0 or sl_distance > 0.05:  # Skip if SL > 5%
            return

        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_distance
        margin_required = position_value / self.leverage

        max_margin = self.capital * 0.50  # Max 50% margin for scalp
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


def generate_scalp_signals_1h(df: pd.DataFrame) -> list[Signal]:
    """Generate scalp signals from 1h data - optimized for frequency."""
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

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # EMA for trend
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ATR for dynamic SL/TP
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    MIN_CONFIDENCE = 0.60

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['atr']) or pd.isna(row['rsi']):
            continue

        rsi = row['rsi']
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_sig = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0
        bb_pct = row['bb_pct'] if not pd.isna(row['bb_pct']) else 0.5
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        atr = row['atr']

        # Trend filter
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']

        # Scalp SL/TP (tighter than swing)
        sl_distance = atr * 1.0  # 1x ATR
        tp_distance = atr * 1.5  # 1.5x ATR (R:R 1:1.5)

        # === SIGNAL 1: RSI Oversold/Overbought + Volume ===
        if vol_ratio >= 1.5:
            if rsi < 30 and row['close'] > row['open'] and uptrend:
                conf = 0.60 + min((30 - rsi) / 50, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason=f"RSI {rsi:.0f} oversold",
                    timestamp=timestamp,
                ))

            if rsi > 70 and row['close'] < row['open'] and downtrend:
                conf = 0.60 + min((rsi - 70) / 50, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason=f"RSI {rsi:.0f} overbought",
                    timestamp=timestamp,
                ))

        # === SIGNAL 2: Bollinger Band Touch ===
        if vol_ratio >= 1.2:
            if bb_pct < 0.05 and row['close'] > row['open']:
                conf = 0.62 + min((0.05 - bb_pct) * 5, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=min(0.80, conf),
                    entry_price=price,
                    stop_loss=row['bb_lower'] - atr * 0.3,
                    take_profit=row['bb_mid'],
                    reason="BB lower touch",
                    timestamp=timestamp,
                ))

            if bb_pct > 0.95 and row['close'] < row['open']:
                conf = 0.62 + min((bb_pct - 0.95) * 5, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=min(0.80, conf),
                    entry_price=price,
                    stop_loss=row['bb_upper'] + atr * 0.3,
                    take_profit=row['bb_mid'],
                    reason="BB upper touch",
                    timestamp=timestamp,
                ))

        # === SIGNAL 3: MACD Cross with Volume ===
        if vol_ratio >= 1.3:
            macd_cross_up = macd > macd_sig and prev['macd'] <= prev['macd_signal']
            macd_cross_down = macd < macd_sig and prev['macd'] >= prev['macd_signal']

            if macd_cross_up and rsi < 60 and uptrend:
                conf = 0.65 + min(vol_ratio / 10, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason="MACD cross up",
                    timestamp=timestamp,
                ))

            if macd_cross_down and rsi > 40 and downtrend:
                conf = 0.65 + min(vol_ratio / 10, 0.15)
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason="MACD cross down",
                    timestamp=timestamp,
                ))

        # === SIGNAL 4: EMA Pullback ===
        ema_dist = (price - row['ema20']) / row['ema20']

        if uptrend and -0.01 < ema_dist < 0.005 and row['close'] > row['open'] and rsi < 55:
            conf = 0.63 + min(vol_ratio / 10, 0.12)
            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=min(0.80, conf),
                entry_price=price,
                stop_loss=row['ema20'] - atr * 0.5,
                take_profit=price + tp_distance,
                reason="EMA20 pullback",
                timestamp=timestamp,
            ))

        if downtrend and -0.005 < ema_dist < 0.01 and row['close'] < row['open'] and rsi > 45:
            conf = 0.63 + min(vol_ratio / 10, 0.12)
            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=min(0.80, conf),
                entry_price=price,
                stop_loss=row['ema20'] + atr * 0.5,
                take_profit=price - tp_distance,
                reason="EMA20 pullback",
                timestamp=timestamp,
            ))

    return signals


def main():
    # Load 1h data
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Generate scalp signals
    signals = generate_scalp_signals_1h(df)
    print(f'Generated {len(signals)} scalp signals')

    if not signals:
        print("No signals generated!")
        return

    print('\n' + '=' * 70)
    print('  BTC 50x SCALP STRATEGY')
    print('=' * 70)

    print(f"\n{'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10} {'월평균':>10}")
    print('-' * 76)

    test_configs = [
        0.01,  # 1% risk
        0.02,  # 2% risk
        0.03,  # 3% risk
        0.05,  # 5% risk
    ]

    best_result = None
    best_risk = None

    for risk in test_configs:
        engine = LeveragedScalpBacktest(
            initial_capital=10000.0,
            leverage=50,
            risk_per_trade=risk,
            min_confidence=0.60,
        )
        result = engine.run(df, signals)

        if result:
            monthly_avg = result['total_return'] / 12
            print(f"{risk*100:>7.0f}% {result['trades']:>8} "
                  f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                  f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}% "
                  f"{monthly_avg:>+9.1f}%")

            if result['profit_factor'] > 1.0:
                if best_result is None or result['total_return'] > best_result['total_return']:
                    best_result = result
                    best_risk = risk

    print('-' * 76)

    if best_result:
        print(f'\n  BEST SCALP CONFIGURATION')
        print(f'  Risk per trade: {best_risk*100:.0f}%')
        print()
        print(f'  Initial Capital:   ${best_result["initial"]:,.2f}')
        print(f'  Final Capital:     ${best_result["final"]:,.2f}')
        print(f'  Total Return:      {best_result["total_return"]:+.2f}%')
        print()
        print(f'  Total Trades:      {best_result["trades"]}')
        print(f'  Win Rate:          {best_result["win_rate"]:.1f}%')
        print(f'  Profit Factor:     {best_result["profit_factor"]:.2f}')
        print(f'  Max Drawdown:      {best_result["max_drawdown"]:.2f}%')
        print()
        print(f'  월평균 거래:       {best_result["trades"] / 12:.1f}회')
        print(f'  월평균 수익률:     {best_result["total_return"] / 12:+.1f}%')
        print()
        print(f'  Avg PnL per Trade: ${best_result["avg_pnl"]:+,.2f}')
        print(f'  Avg Win:           ${best_result["avg_win"]:+,.2f}')
        print(f'  Avg Loss:          ${best_result["avg_loss"]:,.2f}')

        # Monthly breakdown
        if best_result['trade_list']:
            print('\n  MONTHLY BREAKDOWN')
            print('  ' + '-' * 50)
            from collections import defaultdict
            monthly_trades = defaultdict(list)
            for t in best_result['trade_list']:
                m = t.exit_time.strftime('%Y-%m')
                monthly_trades[m].append(t)

            cap = 10000
            for month in sorted(monthly_trades.keys()):
                trades = monthly_trades[month]
                pnl = sum(t.pnl for t in trades)
                wins = sum(1 for t in trades if t.pnl > 0)
                ret = (pnl / cap) * 100
                cap += pnl
                bar = '#' * int(abs(ret) / 3) if ret != 0 else ''
                sign = '+' if ret >= 0 else ''
                print(f"  {month}: {len(trades):>3}거래 {wins}승 {sign}{ret:>7.2f}%  {bar}")

    # Compare with Swing
    print('\n' + '=' * 70)
    print('  SCALP vs SWING 비교')
    print('=' * 70)

    from scripts.run_backtest import generate_swing_signals

    swing_signals = generate_swing_signals(df)

    # Import the leveraged backtest from previous script
    sys.path.insert(0, str(Path(__file__).parent))
    from test_btc_50x_leverage import LeveragedBacktest

    swing_engine = LeveragedBacktest(
        initial_capital=10000.0,
        leverage=50,
        risk_per_trade=0.10,
        min_confidence=0.60,
    )
    swing_result = swing_engine.run(df, swing_signals)

    print(f"\n{'전략':>12} {'거래수':>8} {'승률':>8} {'수익률':>12} {'MDD':>10} {'월평균':>10}")
    print('-' * 66)

    if best_result:
        print(f"{'SCALP':>12} {best_result['trades']:>8} "
              f"{best_result['win_rate']:>7.1f}% {best_result['total_return']:>+11.2f}% "
              f"{best_result['max_drawdown']:>9.2f}% {best_result['total_return']/12:>+9.1f}%")

    if swing_result:
        print(f"{'SWING':>12} {swing_result['trades']:>8} "
              f"{swing_result['win_rate']:>7.1f}% {swing_result['total_return']:>+11.2f}% "
              f"{swing_result['max_drawdown']:>9.2f}% {swing_result['total_return']/12:>+9.1f}%")

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
