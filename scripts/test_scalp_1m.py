"""Test scalp strategy with 1-minute data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.signals.base import Signal


def generate_scalp_signals_1m(df):
    """Generate scalp signals from 1-minute data."""
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

    # Wick patterns
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['full_range'] = df['high'] - df['low']
    df['wick_down'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)
    df['wick_up'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)

    MIN_CONFIDENCE = 0.60

    for i in range(100, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        price = row['close']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_sig = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0
        bb_pct = row['bb_pct'] if not pd.isna(row['bb_pct']) else 0.5
        wick_down = row['wick_down'] if not pd.isna(row['wick_down']) else 0
        wick_up = row['wick_up'] if not pd.isna(row['wick_up']) else 0

        if pd.isna(row['volume_ratio']):
            continue

        def check_macd(direction):
            if direction == 'LONG':
                return macd > macd_sig
            return macd < macd_sig

        # 1. Volume spike + RSI
        if row['volume_ratio'] >= 3.0:
            if row['close'] > row['open'] and rsi < 40:
                direction = 'LONG'
            elif row['close'] < row['open'] and rsi > 60:
                direction = 'SHORT'
            else:
                direction = None

            if direction and check_macd(direction):
                confidence = 0.55 + min(row['volume_ratio'] / 30, 0.2) + min(abs(50-rsi)/100, 0.1)
                sl = price * (0.995 if direction == 'LONG' else 1.005)
                tp = price * (1.01 if direction == 'LONG' else 0.99)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP', direction=direction, confidence=min(0.85, confidence),
                        entry_price=price, stop_loss=sl, take_profit=tp,
                        reason=f"Vol {row['volume_ratio']:.1f}x",
                        timestamp=timestamp,
                    ))

        # 2. BB reversal
        if row['volume_ratio'] >= 2.0:
            if bb_pct < 0.1 and row['close'] > row['open'] and rsi < 45 and check_macd('LONG'):
                confidence = 0.6 + min((0.1 - bb_pct) * 3, 0.15)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP', direction='LONG', confidence=min(0.85, confidence),
                        entry_price=price, stop_loss=row['bb_lower'] * 0.999,
                        take_profit=row['bb_mid'],
                        reason='BB lower',
                        timestamp=timestamp,
                    ))

            if bb_pct > 0.9 and row['close'] < row['open'] and rsi > 55 and check_macd('SHORT'):
                confidence = 0.6 + min((bb_pct - 0.9) * 3, 0.15)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP', direction='SHORT', confidence=min(0.85, confidence),
                        entry_price=price, stop_loss=row['bb_upper'] * 1.001,
                        take_profit=row['bb_mid'],
                        reason='BB upper',
                        timestamp=timestamp,
                    ))

        # 3. Wick reversal
        if row['volume_ratio'] >= 1.5:
            if wick_down > 0.65 and row['close'] > row['open'] and rsi < 45 and check_macd('LONG'):
                confidence = 0.55 + min(wick_down * 0.3, 0.2)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP', direction='LONG', confidence=min(0.85, confidence),
                        entry_price=price, stop_loss=row['low'] * 0.999,
                        take_profit=price * 1.01,
                        reason=f"Wick {wick_down*100:.0f}%",
                        timestamp=timestamp,
                    ))

            if wick_up > 0.65 and row['close'] < row['open'] and rsi > 55 and check_macd('SHORT'):
                confidence = 0.55 + min(wick_up * 0.3, 0.2)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP', direction='SHORT', confidence=min(0.85, confidence),
                        entry_price=price, stop_loss=row['high'] * 1.001,
                        take_profit=price * 0.99,
                        reason=f"Wick {wick_up*100:.0f}%",
                        timestamp=timestamp,
                    ))

    return signals


def main():
    # Load 1m data (30 days)
    df = pd.read_csv('data/BTCUSDT_1m_30d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} 1m candles (30 days)')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Generate signals
    print('\nGenerating scalp signals from 1m data...')
    signals = generate_scalp_signals_1m(df)
    print(f'Generated {len(signals)} scalp signals')

    # Run backtest
    if len(signals) > 0:
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            slippage=0.0001,
            risk_per_trade=0.03,
            max_position_pct=0.15,
            min_confidence=0.60,
            partial_tp_enabled=True,
            partial_tp_ratio=0.5,
            partial_tp_rr=2.0,
            move_sl_to_be=True,
        )
        result = engine.run(df, signals)

        print('\n' + '=' * 60)
        print('  SCALP Phase 3 (30일 1분봉 데이터)')
        print('=' * 60)
        print(f'  Total Trades:     {result.total_trades}')
        print(f'  Win Rate:         {result.win_rate:.1f}%')
        print(f'  Total Return:     {result.total_return:+.2f}%')
        print(f'  Profit Factor:    {result.profit_factor:.2f}')
        print(f'  Max Drawdown:     {result.max_drawdown:.2f}%')
        print(f'  Sharpe Ratio:     {result.sharpe_ratio:.2f}')
        print(f'  Avg Win:          ${result.avg_win:.2f}')
        print(f'  Avg Loss:         ${result.avg_loss:.2f}')

        # 월간 수익률 추정 (30일 데이터)
        monthly_return = result.total_return
        yearly_estimate = monthly_return * 12
        print(f'\n  월간 수익률:      {monthly_return:+.2f}%')
        print(f'  연간 추정:        {yearly_estimate:+.2f}%')
        print('=' * 60)
    else:
        print('No signals generated')


if __name__ == '__main__':
    main()
