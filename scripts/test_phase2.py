"""Phase 2 backtest test script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine
from src.signals.base import Signal


def generate_improved_scalp_signals(df):
    """
    Phase 2 개선: 다양한 시그널 패턴 추가
    - Volume Spike
    - Wick Reversal Pattern (윅 반전)
    - RSI Extreme Reversal
    """
    signals = []
    df = df.copy()

    # Calculate indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['price_change'] = df['close'].pct_change()

    # Candle body and wick calculations
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['full_range'] = df['high'] - df['low']
    df['wick_ratio_up'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['wick_ratio_down'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    # EMA for trend
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

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

    MIN_CONFIDENCE = 0.65

    for i in range(50, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        price = row['close']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_sig = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0

        if pd.isna(row['volume_ratio']) or pd.isna(row['wick_ratio_up']):
            continue

        def check_macd(direction):
            if direction == 'LONG':
                return macd > macd_sig
            else:
                return macd < macd_sig

        # Signal 1: Wick Reversal (하락 후 긴 하위꼬리)
        if (row['wick_ratio_down'] > 0.65 and
            row['volume_ratio'] >= 2.0 and
            rsi < 40 and
            row['close'] > row['open']):  # Bullish candle

            if check_macd('LONG'):
                confidence = 0.55 + min(row['wick_ratio_down'] * 0.2, 0.15) + min((40-rsi)/100, 0.1)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP',
                        direction='LONG',
                        confidence=min(0.9, confidence),
                        entry_price=price,
                        stop_loss=row['low'] * 0.999,  # Below wick
                        take_profit=price * 1.02,
                        reason=f"Wick reversal: {row['wick_ratio_down']*100:.0f}% lower wick",
                        timestamp=timestamp,
                    ))

        # Signal 2: Wick Reversal (상승 후 긴 상위꼬리)
        if (row['wick_ratio_up'] > 0.65 and
            row['volume_ratio'] >= 2.0 and
            rsi > 60 and
            row['close'] < row['open']):  # Bearish candle

            if check_macd('SHORT'):
                confidence = 0.55 + min(row['wick_ratio_up'] * 0.2, 0.15) + min((rsi-60)/100, 0.1)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP',
                        direction='SHORT',
                        confidence=min(0.9, confidence),
                        entry_price=price,
                        stop_loss=row['high'] * 1.001,  # Above wick
                        take_profit=price * 0.98,
                        reason=f"Wick reversal: {row['wick_ratio_up']*100:.0f}% upper wick",
                        timestamp=timestamp,
                    ))

        # Signal 3: RSI Extreme + Volume
        if row['volume_ratio'] >= 3.0:
            if rsi < 30 and row['close'] > row['open'] and check_macd('LONG'):
                confidence = 0.6 + min((30-rsi)/50, 0.15) + min((row['volume_ratio']-3)/10, 0.1)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP',
                        direction='LONG',
                        confidence=min(0.9, confidence),
                        entry_price=price,
                        stop_loss=price * 0.985,
                        take_profit=price * 1.025,
                        reason=f"RSI extreme {rsi:.0f} + Vol {row['volume_ratio']:.1f}x",
                        timestamp=timestamp,
                    ))

            elif rsi > 70 and row['close'] < row['open'] and check_macd('SHORT'):
                confidence = 0.6 + min((rsi-70)/50, 0.15) + min((row['volume_ratio']-3)/10, 0.1)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type='SCALP',
                        direction='SHORT',
                        confidence=min(0.9, confidence),
                        entry_price=price,
                        stop_loss=price * 1.015,
                        take_profit=price * 0.975,
                        reason=f"RSI extreme {rsi:.0f} + Vol {row['volume_ratio']:.1f}x",
                        timestamp=timestamp,
                    ))

    return signals


def main():
    # Load 1h data for longer period testing
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} 1h candles (1 year)')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Generate signals
    print('\nGenerating improved scalp signals...')
    signals = generate_improved_scalp_signals(df)
    print(f'Generated {len(signals)} signals')

    # Categorize signals
    wick_signals = [s for s in signals if 'Wick' in s.reason]
    rsi_signals = [s for s in signals if 'RSI' in s.reason]
    print(f'  - Wick reversal: {len(wick_signals)}')
    print(f'  - RSI extreme: {len(rsi_signals)}')

    # Run backtest
    if signals:
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            slippage=0.0001,
            risk_per_trade=0.01,
            max_position_pct=0.05,
            min_confidence=0.65,
            partial_tp_enabled=True,
            partial_tp_ratio=0.5,
            partial_tp_rr=2.0,
            move_sl_to_be=True,
        )
        result = engine.run(df, signals)

        print('\n' + '=' * 60)
        print('  개선된 SCALP Phase 2 (1년 1시간봉 데이터)')
        print('=' * 60)
        print(f'  Total Trades:     {result.total_trades}')
        print(f'  Win Rate:         {result.win_rate:.1f}%')
        print(f'  Total Return:     {result.total_return:+.2f}%')
        print(f'  Profit Factor:    {result.profit_factor:.2f}')
        print(f'  Max Drawdown:     {result.max_drawdown:.2f}%')
        print(f'  Sharpe Ratio:     {result.sharpe_ratio:.2f}')
        print(f'  Avg Win:          ${result.avg_win:.2f}')
        print(f'  Avg Loss:         ${result.avg_loss:.2f}')
        print('=' * 60)
    else:
        print('No signals generated')


if __name__ == '__main__':
    main()
