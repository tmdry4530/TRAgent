"""BTC-focused backtest with 50x leverage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.signals.base import Signal


def generate_btc_swing_signals(df):
    """Generate optimized swing signals for BTC with 50x leverage.

    50x 레버리지용 최적화:
    - 더 타이트한 손절 (2.5%)
    - 더 높은 익절 (7.5%) = R:R 1:3
    - 엄격한 필터로 높은 승률 추구
    """
    signals = []

    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # EMAs
    df_4h['ema7'] = df_4h['close'].ewm(span=7, adjust=False).mean()
    df_4h['ema25'] = df_4h['close'].ewm(span=25, adjust=False).mean()
    df_4h['ema99'] = df_4h['close'].ewm(span=99, adjust=False).mean()

    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df_4h['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_4h['close'].ewm(span=26, adjust=False).mean()
    df_4h['macd'] = ema_fast - ema_slow
    df_4h['macd_signal'] = df_4h['macd'].ewm(span=9, adjust=False).mean()
    df_4h['macd_hist'] = df_4h['macd'] - df_4h['macd_signal']

    # ATR for dynamic SL
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(window=14).mean()

    MIN_CONFIDENCE = 0.65

    for i in range(100, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i-1]
        price = row['close']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr'] if not pd.isna(row['atr']) else price * 0.02

        if pd.isna(row['ema99']) or pd.isna(row['macd']):
            continue

        # EMA 정렬
        bullish = row['ema7'] > row['ema25'] > row['ema99']
        bearish = row['ema7'] < row['ema25'] < row['ema99']

        # EMA 크로스오버 감지
        prev_bull = prev['ema7'] > prev['ema25'] if not pd.isna(prev['ema25']) else False
        prev_bear = prev['ema7'] < prev['ema25'] if not pd.isna(prev['ema25']) else False

        bull_cross = bullish and not prev_bull
        bear_cross = bearish and not prev_bear

        # MACD 조건
        macd_bull = row['macd'] > row['macd_signal'] and row['macd_hist'] > 0
        macd_bear = row['macd'] < row['macd_signal'] and row['macd_hist'] < 0

        # RSI 조건 (더 엄격)
        rsi_ok_long = 40 <= rsi <= 55
        rsi_ok_short = 45 <= rsi <= 60

        # EMA 갭 기반 신뢰도
        def calc_confidence(ema7, ema25, ema99):
            conf = 0.60
            gap1 = abs(ema7 - ema25) / ema25
            gap2 = abs(ema25 - ema99) / ema99
            if gap1 > 0.03:
                conf += 0.10
            if gap2 > 0.03:
                conf += 0.10
            return min(0.90, conf)

        # 50x용 타이트 SL/TP (ATR 기반)
        sl_distance = max(atr * 1.5, price * 0.02)  # 최소 2%
        tp_distance = sl_distance * 3  # R:R 1:3

        # LONG 시그널
        if bull_cross and macd_bull and rsi_ok_long:
            conf = calc_confidence(row['ema7'], row['ema25'], row['ema99'])
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=conf,
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason=f"EMA cross + MACD + RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

        # SHORT 시그널
        if bear_cross and macd_bear and rsi_ok_short:
            conf = calc_confidence(row['ema7'], row['ema25'], row['ema99'])
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=conf,
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason=f"EMA cross + MACD + RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

    return signals


def run_backtest_50x(df, signals, risk_pct, max_pos_pct):
    """Run backtest with 50x leverage settings."""
    if not signals:
        return None

    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,  # 0.04% taker fee
        slippage=0.0002,    # 0.02% slippage (conservative for 50x)
        risk_per_trade=risk_pct,
        max_position_pct=max_pos_pct,
        min_confidence=0.65,
        partial_tp_enabled=True,
        partial_tp_ratio=0.5,
        partial_tp_rr=2.0,
        move_sl_to_be=True,
    )
    return engine.run(df, signals)


def main():
    # Load BTC data
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Generate signals
    print('\nGenerating BTC swing signals (50x optimized)...')
    signals = generate_btc_swing_signals(df)
    print(f'Generated {len(signals)} signals')

    if not signals:
        print('No signals generated!')
        return

    # Test different risk levels
    print('\n' + '=' * 70)
    print('  BTC 50x LEVERAGE - RISK LEVEL COMPARISON')
    print('=' * 70)
    print(f"{'Risk%':>8} {'MaxPos%':>8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}")
    print('-' * 70)

    test_configs = [
        (0.05, 0.30),  # 5% risk, 30% max pos
        (0.08, 0.40),  # 8% risk, 40% max pos
        (0.10, 0.50),  # 10% risk, 50% max pos
        (0.12, 0.60),  # 12% risk, 60% max pos
        (0.15, 0.70),  # 15% risk, 70% max pos
        (0.20, 0.80),  # 20% risk, 80% max pos
    ]

    best_result = None
    best_config = None

    for risk, max_pos in test_configs:
        result = run_backtest_50x(df, signals, risk, max_pos)
        if result:
            print(f"{risk*100:>7.0f}% {max_pos*100:>7.0f}% {result.total_trades:>8} "
                  f"{result.win_rate:>7.1f}% {result.total_return:>+9.2f}% "
                  f"{result.profit_factor:>8.2f} {result.max_drawdown:>7.2f}%")

            # Track best Sharpe
            if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                best_result = result
                best_config = (risk, max_pos)

    print('-' * 70)

    if best_result:
        print(f'\n  BEST CONFIG: Risk {best_config[0]*100:.0f}%, MaxPos {best_config[1]*100:.0f}%')
        print()
        print('=' * 70)
        print('  DETAILED RESULTS (Best Config)')
        print('=' * 70)
        print(f'  Total Trades:       {best_result.total_trades}')
        print(f'  Winning Trades:     {best_result.winning_trades}')
        print(f'  Losing Trades:      {best_result.losing_trades}')
        print(f'  Win Rate:           {best_result.win_rate:.1f}%')
        print()
        print(f'  Total Return:       {best_result.total_return:+.2f}%')
        print(f'  Initial Capital:    ${best_result.initial_capital:,.2f}')
        print(f'  Final Capital:      ${best_result.final_capital:,.2f}')
        print(f'  Net Profit:         ${best_result.final_capital - best_result.initial_capital:+,.2f}')
        print()
        print(f'  Profit Factor:      {best_result.profit_factor:.2f}')
        print(f'  Max Drawdown:       {best_result.max_drawdown:.2f}%')
        print(f'  Sharpe Ratio:       {best_result.sharpe_ratio:.2f}')
        print()
        print(f'  Avg Win:            ${best_result.avg_win:+.2f}')
        print(f'  Avg Loss:           ${best_result.avg_loss:.2f}')
        print(f'  Avg Hold Time:      {best_result.avg_hold_time:.1f} hours')
        print('=' * 70)

        # Monthly breakdown
        if best_result.trades:
            print('\n  MONTHLY PERFORMANCE')
            print('-' * 40)
            from collections import defaultdict
            monthly = defaultdict(float)
            for trade in best_result.trades:
                month = trade.exit_time.strftime('%Y-%m')
                monthly[month] += trade.pnl

            running = 10000
            for month in sorted(monthly.keys()):
                pnl = monthly[month]
                ret = (pnl / running) * 100
                running += pnl
                bar = '#' * int(abs(ret) / 2) if ret != 0 else ''
                sign = '+' if ret >= 0 else ''
                print(f"  {month}: {sign}{ret:6.2f}%  {bar}")


if __name__ == '__main__':
    main()
