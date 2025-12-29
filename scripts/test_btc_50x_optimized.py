"""BTC 50x optimized strategy for 70%+ win rate."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.backtest.engine import BacktestEngine
from src.signals.base import Signal


def generate_high_winrate_signals(df):
    """Generate high win-rate signals optimized for BTC 50x.

    High Win Rate Strategy (70%+ target):
    - Multiple confirmation (EMA + RSI + MACD + Volume)
    - Tight take profit (R:R 1:1.5) for faster wins
    - Only trade with strong momentum
    - ATR-based dynamic SL for volatility adaptation
    """
    signals = []

    # Resample to 4h for swing analysis
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # EMAs for trend
    df_4h['ema7'] = df_4h['close'].ewm(span=7, adjust=False).mean()
    df_4h['ema21'] = df_4h['close'].ewm(span=21, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()

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

    # ATR for volatility
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(window=14).mean()
    df_4h['atr_pct'] = df_4h['atr'] / df_4h['close']

    # Volume analysis
    df_4h['volume_ma'] = df_4h['volume'].rolling(window=20).mean()
    df_4h['volume_ratio'] = df_4h['volume'] / df_4h['volume_ma']

    # Momentum strength
    df_4h['momentum'] = df_4h['close'].pct_change(5)  # 5-bar momentum

    # Trend strength (ADX-like)
    df_4h['trend_strength'] = abs(df_4h['ema7'] - df_4h['ema50']) / df_4h['ema50']

    MIN_CONFIDENCE = 0.70  # Higher threshold for high win rate

    for i in range(100, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i-1]
        prev2 = df_4h.iloc[i-2]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['macd']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr'] if not pd.isna(row['atr']) else price * 0.02
        volume_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        momentum = row['momentum'] if not pd.isna(row['momentum']) else 0
        trend_str = row['trend_strength'] if not pd.isna(row['trend_strength']) else 0

        # === STRICT CONFIRMATION RULES ===

        # 1. Trend alignment (all EMAs aligned)
        bullish_trend = row['ema7'] > row['ema21'] > row['ema50']
        bearish_trend = row['ema7'] < row['ema21'] < row['ema50']

        # 2. Momentum confirmation
        macd_bullish = row['macd'] > row['macd_signal'] and row['macd_hist'] > prev['macd_hist']
        macd_bearish = row['macd'] < row['macd_signal'] and row['macd_hist'] < prev['macd_hist']

        # 3. RSI in favorable zone (not extreme)
        rsi_long_ok = 35 <= rsi <= 55  # Avoid overbought
        rsi_short_ok = 45 <= rsi <= 65  # Avoid oversold

        # 4. Volume confirmation
        volume_ok = volume_ratio >= 1.2  # Above average volume

        # 5. Trend strength
        strong_trend = trend_str >= 0.02  # At least 2% gap between EMAs

        # 6. Price above/below key EMA
        price_above_ema = price > row['ema21']
        price_below_ema = price < row['ema21']

        # 7. Consecutive bullish/bearish bars (momentum)
        consec_bullish = (row['close'] > row['open'] and
                         prev['close'] > prev['open'])
        consec_bearish = (row['close'] < row['open'] and
                         prev['close'] < prev['open'])

        # === SIGNAL GENERATION ===

        # LONG: All bullish conditions met
        long_conditions = [
            bullish_trend,
            macd_bullish,
            rsi_long_ok,
            volume_ok,
            price_above_ema,
            momentum > 0,
        ]

        # SHORT: All bearish conditions met
        short_conditions = [
            bearish_trend,
            macd_bearish,
            rsi_short_ok,
            volume_ok,
            price_below_ema,
            momentum < 0,
        ]

        # Count confirmations
        long_score = sum(long_conditions)
        short_score = sum(short_conditions)

        # Dynamic SL/TP based on ATR (tighter for high win rate)
        sl_distance = atr * 1.5  # 1.5x ATR stop loss
        tp_distance = sl_distance * 1.5  # R:R 1:1.5 for higher win rate

        # LONG signal - need at least 5/6 conditions
        if long_score >= 5:
            conf_base = 0.60 + (long_score - 5) * 0.10

            # Bonus for strong trend
            if strong_trend:
                conf_base += 0.05
            if consec_bullish:
                conf_base += 0.05
            if volume_ratio >= 1.5:
                conf_base += 0.05

            confidence = min(0.95, conf_base)

            if confidence >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=confidence,
                    entry_price=price,
                    stop_loss=price - sl_distance,
                    take_profit=price + tp_distance,
                    reason=f"Strong trend ({long_score}/6) RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

        # SHORT signal - need at least 5/6 conditions
        if short_score >= 5:
            conf_base = 0.60 + (short_score - 5) * 0.10

            # Bonus for strong trend
            if strong_trend:
                conf_base += 0.05
            if consec_bearish:
                conf_base += 0.05
            if volume_ratio >= 1.5:
                conf_base += 0.05

            confidence = min(0.95, conf_base)

            if confidence >= MIN_CONFIDENCE:
                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=confidence,
                    entry_price=price,
                    stop_loss=price + sl_distance,
                    take_profit=price - tp_distance,
                    reason=f"Strong trend ({short_score}/6) RSI {rsi:.0f}",
                    timestamp=row['timestamp'],
                ))

    return signals


def generate_pullback_signals(df):
    """Generate pullback-to-trend signals (high win rate).

    Pullback Strategy:
    - Wait for established trend
    - Enter on pullback to EMA support/resistance
    - Higher win rate due to trend continuation
    """
    signals = []

    # Resample to 4h
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    # EMAs
    df_4h['ema20'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50'] = df_4h['close'].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    df_4h['tr'] = np.maximum(
        df_4h['high'] - df_4h['low'],
        np.maximum(
            abs(df_4h['high'] - df_4h['close'].shift(1)),
            abs(df_4h['low'] - df_4h['close'].shift(1))
        )
    )
    df_4h['atr'] = df_4h['tr'].rolling(window=14).mean()

    # Distance from EMA20
    df_4h['dist_ema20'] = (df_4h['close'] - df_4h['ema20']) / df_4h['ema20']

    MIN_CONFIDENCE = 0.70

    for i in range(100, len(df_4h)):
        row = df_4h.iloc[i]
        prev = df_4h.iloc[i-1]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['atr']):
            continue

        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        atr = row['atr'] if not pd.isna(row['atr']) else price * 0.02
        dist = row['dist_ema20'] if not pd.isna(row['dist_ema20']) else 0

        # Trend direction
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']

        # Pullback detection
        pullback_to_support = uptrend and abs(dist) < 0.01  # Within 1% of EMA20
        pullback_to_resistance = downtrend and abs(dist) < 0.01

        # Bounce confirmation (price closing in trend direction)
        bullish_bounce = row['close'] > row['open'] and row['close'] > prev['close']
        bearish_bounce = row['close'] < row['open'] and row['close'] < prev['close']

        # RSI not extreme
        rsi_ok_long = 40 <= rsi <= 60
        rsi_ok_short = 40 <= rsi <= 60

        # Tight R:R for pullback (1:1.2)
        sl_distance = atr * 1.2
        tp_distance = sl_distance * 1.2

        # LONG: Pullback to support in uptrend
        if pullback_to_support and bullish_bounce and rsi_ok_long:
            confidence = 0.72
            signals.append(Signal(
                type='SWING', direction='LONG', confidence=confidence,
                entry_price=price,
                stop_loss=price - sl_distance,
                take_profit=price + tp_distance,
                reason=f"Pullback bounce RSI {rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # SHORT: Pullback to resistance in downtrend
        if pullback_to_resistance and bearish_bounce and rsi_ok_short:
            confidence = 0.72
            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=confidence,
                entry_price=price,
                stop_loss=price + sl_distance,
                take_profit=price - tp_distance,
                reason=f"Pullback bounce RSI {rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def run_backtest(df, signals, risk_pct, max_pos_pct, partial_tp_rr=1.0):
    """Run backtest with given parameters."""
    if not signals:
        return None

    engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,
        slippage=0.0002,
        risk_per_trade=risk_pct,
        max_position_pct=max_pos_pct,
        min_confidence=0.70,
        partial_tp_enabled=True,
        partial_tp_ratio=0.5,
        partial_tp_rr=partial_tp_rr,
        move_sl_to_be=True,
    )
    return engine.run(df, signals)


def main():
    # Load BTC data
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Strategy 1: High Win Rate Signals
    print('\n' + '=' * 70)
    print('  STRATEGY 1: HIGH WIN RATE (Multi-Confirmation)')
    print('=' * 70)

    signals1 = generate_high_winrate_signals(df)
    print(f'Generated {len(signals1)} high-winrate signals')

    if signals1:
        # Test different risk levels
        print(f"\n{'Risk%':>8} {'MaxPos%':>8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}")
        print('-' * 70)

        best_result = None
        best_config = None

        for risk, max_pos in [(0.08, 0.40), (0.10, 0.50), (0.12, 0.60), (0.15, 0.70)]:
            result = run_backtest(df, signals1, risk, max_pos, partial_tp_rr=1.0)
            if result:
                print(f"{risk*100:>7.0f}% {max_pos*100:>7.0f}% {result.total_trades:>8} "
                      f"{result.win_rate:>7.1f}% {result.total_return:>+9.2f}% "
                      f"{result.profit_factor:>8.2f} {result.max_drawdown:>7.2f}%")

                # Best by Sharpe
                if best_result is None or result.sharpe_ratio > best_result.sharpe_ratio:
                    best_result = result
                    best_config = (risk, max_pos)

        if best_result:
            print(f'\n  Best Config: Risk {best_config[0]*100:.0f}%, MaxPos {best_config[1]*100:.0f}%')
            print(f'  Win Rate: {best_result.win_rate:.1f}%')
            print(f'  Return: {best_result.total_return:+.2f}%')

    # Strategy 2: Pullback Signals
    print('\n' + '=' * 70)
    print('  STRATEGY 2: PULLBACK TO TREND')
    print('=' * 70)

    signals2 = generate_pullback_signals(df)
    print(f'Generated {len(signals2)} pullback signals')

    if signals2:
        print(f"\n{'Risk%':>8} {'MaxPos%':>8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}")
        print('-' * 70)

        for risk, max_pos in [(0.08, 0.40), (0.10, 0.50), (0.12, 0.60)]:
            result = run_backtest(df, signals2, risk, max_pos, partial_tp_rr=0.8)
            if result:
                print(f"{risk*100:>7.0f}% {max_pos*100:>7.0f}% {result.total_trades:>8} "
                      f"{result.win_rate:>7.1f}% {result.total_return:>+9.2f}% "
                      f"{result.profit_factor:>8.2f} {result.max_drawdown:>7.2f}%")

    # Strategy 3: Combined (both strategies)
    print('\n' + '=' * 70)
    print('  STRATEGY 3: COMBINED (High WR + Pullback)')
    print('=' * 70)

    combined_signals = signals1 + signals2
    # Sort by timestamp
    combined_signals.sort(key=lambda s: s.timestamp)
    print(f'Combined {len(combined_signals)} signals')

    if combined_signals:
        print(f"\n{'Risk%':>8} {'MaxPos%':>8} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}")
        print('-' * 70)

        best_combined = None
        best_combined_config = None

        for risk, max_pos in [(0.08, 0.40), (0.10, 0.50), (0.12, 0.60), (0.15, 0.70)]:
            result = run_backtest(df, combined_signals, risk, max_pos, partial_tp_rr=1.0)
            if result:
                print(f"{risk*100:>7.0f}% {max_pos*100:>7.0f}% {result.total_trades:>8} "
                      f"{result.win_rate:>7.1f}% {result.total_return:>+9.2f}% "
                      f"{result.profit_factor:>8.2f} {result.max_drawdown:>7.2f}%")

                if best_combined is None or result.sharpe_ratio > best_combined.sharpe_ratio:
                    best_combined = result
                    best_combined_config = (risk, max_pos)

        print('-' * 70)

        if best_combined:
            print(f'\n  BEST COMBINED CONFIG')
            print(f'  Risk: {best_combined_config[0]*100:.0f}%, MaxPos: {best_combined_config[1]*100:.0f}%')
            print()
            print(f'  Total Trades:     {best_combined.total_trades}')
            print(f'  Win Rate:         {best_combined.win_rate:.1f}%')
            print(f'  Total Return:     {best_combined.total_return:+.2f}%')
            print(f'  Profit Factor:    {best_combined.profit_factor:.2f}')
            print(f'  Max Drawdown:     {best_combined.max_drawdown:.2f}%')
            print(f'  Sharpe Ratio:     {best_combined.sharpe_ratio:.2f}')
            print(f'  Final Capital:    ${best_combined.final_capital:,.2f}')
            print()

            # Monthly breakdown
            if best_combined.trades:
                print('  MONTHLY PERFORMANCE')
                print('  ' + '-' * 40)
                from collections import defaultdict
                monthly = defaultdict(float)
                for trade in best_combined.trades:
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

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
