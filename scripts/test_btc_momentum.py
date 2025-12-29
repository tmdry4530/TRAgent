"""BTC 50x Momentum Strategy - Focus on R:R optimization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.signals.base import Signal


def generate_momentum_signals(df, rr_ratio=2.0):
    """Generate momentum-based signals with optimized R:R.

    Momentum Strategy:
    - Trade breakouts of recent highs/lows
    - Use ATR for dynamic SL
    - Adjustable R:R ratio for optimization
    - Volume confirmation
    """
    signals = []

    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # Donchian Channel (breakout levels)
    df_4h['high_20'] = df_4h['high'].rolling(window=20).max()
    df_4h['low_20'] = df_4h['low'].rolling(window=20).min()

    # EMAs for trend filter
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()

    # ATR
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(window=14).mean()

    # Volume
    df_4h['volume_ma'] = df_4h['volume'].rolling(window=20).mean()
    df_4h['volume_ratio'] = df_4h['volume'] / df_4h['volume_ma']

    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi'] = 100 - (100 / (1 + rs))

    MIN_CONFIDENCE = 0.60

    for i in range(25, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i-1]
        price = row['close']

        if pd.isna(row['high_20']) or pd.isna(row['atr']) or pd.isna(row['ema50']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0

        # Trend filter
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']

        # Dynamic SL/TP based on ATR and R:R
        sl_distance = atr * 1.5
        tp_distance = sl_distance * rr_ratio

        # Breakout conditions
        breakout_up = (price > prev['high_20']) and (prev['close'] <= prev['high_20'])
        breakout_down = (price < prev['low_20']) and (prev['close'] >= prev['low_20'])

        # Volume confirmation
        volume_ok = vol_ratio >= 1.3

        # LONG: Breakout above 20-period high in uptrend
        if breakout_up and uptrend and volume_ok and rsi < 70:
            conf = 0.60 + min((vol_ratio - 1.3) * 0.1, 0.2)
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason=f"Breakout up, Vol {vol_ratio:.1f}x",
                    timestamp=row['timestamp'],
                ))

        # SHORT: Breakout below 20-period low in downtrend
        if breakout_down and downtrend and volume_ok and rsi > 30:
            conf = 0.60 + min((vol_ratio - 1.3) * 0.1, 0.2)
            if conf >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=min(0.85, conf),
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason=f"Breakout down, Vol {vol_ratio:.1f}x",
                    timestamp=row['timestamp'],
                ))

    return signals


def generate_mean_reversion_signals(df, rr_ratio=1.5):
    """Generate mean reversion signals at extremes.

    Mean Reversion:
    - Trade when price extends far from EMA
    - Enter on reversal candle patterns
    - Tight R:R for quick profits
    """
    signals = []

    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # EMA
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    df_4h['bb_mid'] = df_4h['close'].rolling(window=20).mean()
    df_4h['bb_std'] = df_4h['close'].rolling(window=20).std()
    df_4h['bb_upper'] = df_4h['bb_mid'] + (df_4h['bb_std'] * 2.5)  # Wider bands
    df_4h['bb_lower'] = df_4h['bb_mid'] - (df_4h['bb_std'] * 2.5)

    # ATR
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(window=14).mean()

    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi'] = 100 - (100 / (1 + rs))

    # Distance from EMA
    df_4h['ema_dist'] = (df_4h['close'] - df_4h['ema20']) / df_4h['ema20']

    MIN_CONFIDENCE = 0.65

    for i in range(25, len(df_4h)):
        row = df_4h.iloc[i]
        price = row['close']

        if pd.isna(row['bb_lower']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        ema_dist = row['ema_dist'] if not pd.isna(row['ema_dist']) else 0

        # Reversal candle patterns
        bullish_reversal = (row['close'] > row['open'] and
                           row['close'] - row['open'] > (row['high'] - row['low']) * 0.6)
        bearish_reversal = (row['close'] < row['open'] and
                           row['open'] - row['close'] > (row['high'] - row['low']) * 0.6)

        # SL/TP
        sl_distance = atr * 1.2
        tp_distance = sl_distance * rr_ratio

        # LONG: Price below lower BB + bullish reversal + RSI oversold
        if price < row['bb_lower'] and bullish_reversal and rsi < 30:
            conf = 0.70 + min((30 - rsi) / 50, 0.15)
            signals.append(Signal(
                type='SWING', direction='LONG', confidence=min(0.85, conf),
                entry_price=price,
                stop_loss=price - sl_distance,
                take_profit=price + tp_distance,
                reason=f"BB lower + RSI {rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # SHORT: Price above upper BB + bearish reversal + RSI overbought
        if price > row['bb_upper'] and bearish_reversal and rsi > 70:
            conf = 0.70 + min((rsi - 70) / 50, 0.15)
            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=min(0.85, conf),
                entry_price=price,
                stop_loss=price + sl_distance,
                take_profit=price - tp_distance,
                reason=f"BB upper + RSI {rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def run_test(df, signals, name, risk_levels):
    """Run backtest across multiple risk levels."""
    if not signals:
        print(f"  No signals generated for {name}")
        return None

    print(f"\n{'Risk%':>8} {'MaxPos%':>8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}")
    print('-' * 70)

    best = None
    best_cfg = None

    for risk, max_pos in risk_levels:
        engine = BacktestEngine(
            initial_capital=10000.0,
            commission=0.0004,
            slippage=0.0002,
            risk_per_trade=risk,
            max_position_pct=max_pos,
            min_confidence=0.60,
            partial_tp_enabled=True,
            partial_tp_ratio=0.5,
            partial_tp_rr=1.5,  # Partial TP at 1.5x risk
            move_sl_to_be=True,
        )
        result = engine.run(df, signals)

        print(f"{risk*100:>7.0f}% {max_pos*100:>7.0f}% {result.total_trades:>8} "
              f"{result.win_rate:>7.1f}% {result.total_return:>+9.2f}% "
              f"{result.profit_factor:>8.2f} {result.max_drawdown:>7.2f}%")

        if best is None or (result.profit_factor > 1.0 and result.sharpe_ratio > best.sharpe_ratio):
            best = result
            best_cfg = (risk, max_pos)

    return best, best_cfg


def main():
    # Load data
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    risk_levels = [
        (0.05, 0.30),
        (0.08, 0.40),
        (0.10, 0.50),
        (0.12, 0.60),
    ]

    # Test different R:R ratios for momentum strategy
    print('\n' + '=' * 70)
    print('  MOMENTUM BREAKOUT - R:R OPTIMIZATION')
    print('=' * 70)

    best_overall = None
    best_rr = None

    for rr in [1.5, 2.0, 2.5, 3.0]:
        print(f'\n--- R:R Ratio = 1:{rr} ---')
        signals = generate_momentum_signals(df, rr_ratio=rr)
        print(f'Generated {len(signals)} signals')

        result, cfg = run_test(df, signals, f"Momentum R:R 1:{rr}", risk_levels)
        if result and (best_overall is None or result.profit_factor > best_overall.profit_factor):
            best_overall = result
            best_rr = rr

    if best_overall:
        print(f'\n  Best Momentum R:R: 1:{best_rr}')
        print(f'  Win Rate: {best_overall.win_rate:.1f}%')
        print(f'  Return: {best_overall.total_return:+.2f}%')
        print(f'  PF: {best_overall.profit_factor:.2f}')

    # Test mean reversion
    print('\n' + '=' * 70)
    print('  MEAN REVERSION AT EXTREMES')
    print('=' * 70)

    for rr in [1.2, 1.5, 2.0]:
        print(f'\n--- R:R Ratio = 1:{rr} ---')
        signals = generate_mean_reversion_signals(df, rr_ratio=rr)
        print(f'Generated {len(signals)} signals')
        run_test(df, signals, f"Mean Reversion R:R 1:{rr}", risk_levels)

    # Use Phase 3 swing signals (the ones that worked)
    print('\n' + '=' * 70)
    print('  PHASE 3 SWING STRATEGY (BASELINE)')
    print('=' * 70)

    # Import and use the Phase 3 signals
    from scripts.run_backtest import generate_swing_signals
    phase3_signals = generate_swing_signals(df)
    print(f'Generated {len(phase3_signals)} Phase 3 swing signals')

    run_test(df, phase3_signals, "Phase 3 Swing", risk_levels)

    # Combine: Phase 3 + Mean Reversion
    print('\n' + '=' * 70)
    print('  COMBINED: PHASE 3 + MEAN REVERSION')
    print('=' * 70)

    mr_signals = generate_mean_reversion_signals(df, rr_ratio=1.5)
    combined = phase3_signals + mr_signals
    combined.sort(key=lambda s: s.timestamp)
    print(f'Combined {len(combined)} signals')

    best_combined, best_cfg = run_test(df, combined, "Combined", risk_levels)

    if best_combined and best_cfg:
        print('\n' + '-' * 70)
        print(f'\n  BEST COMBINED RESULT')
        print(f'  Config: Risk {best_cfg[0]*100:.0f}%, MaxPos {best_cfg[1]*100:.0f}%')
        print(f'  Trades: {best_combined.total_trades}')
        print(f'  Win Rate: {best_combined.win_rate:.1f}%')
        print(f'  Return: {best_combined.total_return:+.2f}%')
        print(f'  Profit Factor: {best_combined.profit_factor:.2f}')
        print(f'  Max Drawdown: {best_combined.max_drawdown:.2f}%')
        print(f'  Sharpe: {best_combined.sharpe_ratio:.2f}')
        print(f'  Final Capital: ${best_combined.final_capital:,.2f}')

        # Monthly
        if best_combined.trades:
            print('\n  MONTHLY BREAKDOWN')
            print('  ' + '-' * 40)
            from collections import defaultdict
            monthly = defaultdict(float)
            for t in best_combined.trades:
                m = t.exit_time.strftime('%Y-%m')
                monthly[m] += t.pnl

            cap = 10000
            for m in sorted(monthly.keys()):
                pnl = monthly[m]
                ret = (pnl / cap) * 100
                cap += pnl
                sign = '+' if ret >= 0 else ''
                print(f"  {m}: {sign}{ret:6.2f}%")

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
