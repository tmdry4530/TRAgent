"""
Strict High Win Rate + Daily Limit Test
- Volume Climax (100% WR) + Channel Bounce (100% WR) only
- Daily trade limit applied
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta


def load_data():
    """Load BTC 1h data"""
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "data" / "BTCUSDT_1h_365d.csv"

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculate indicators
    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Donchian Channel (20 period)
    df['channel_high'] = df['high'].rolling(20).max()
    df['channel_low'] = df['low'].rolling(20).min()

    # Wick calculations
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['candle_range'].replace(0, np.nan)

    print(f"Loaded {len(df)} BTC 1h candles")
    print(f"Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    return df


def generate_volume_climax_signals(df, vol_mult=3.0, consec_bars=3, wick_pct=0.5, rr_ratio=1.5):
    """
    Volume Climax Strategy (100% WR achieved previously)
    - 3x volume spike
    - 3 consecutive same direction
    - 50%+ wick ratio
    - R:R 1.5
    """
    signals = []

    for i in range(consec_bars + 20, len(df)):
        row = df.iloc[i]

        # Skip if not volume climax
        if row['volume_ratio'] < vol_mult:
            continue

        # Check consecutive bars
        bullish_count = 0
        bearish_count = 0
        for j in range(1, consec_bars + 1):
            prev = df.iloc[i - j]
            if prev['close'] > prev['open']:
                bullish_count += 1
            else:
                bearish_count += 1

        # Need all consecutive bars same direction
        if bullish_count < consec_bars and bearish_count < consec_bars:
            continue

        # Check wick ratio
        if pd.isna(row['wick_ratio']) or row['wick_ratio'] < wick_pct:
            continue

        # Determine signal direction (reversal)
        if bullish_count >= consec_bars:
            # After bullish run, SHORT reversal
            if row['upper_wick'] > row['body']:  # Upper wick rejection
                sl_dist = row['upper_wick'] * 1.5
                tp_dist = sl_dist * rr_ratio
                signals.append({
                    'timestamp': row['timestamp'],
                    'direction': 'SHORT',
                    'entry': row['close'],
                    'sl': row['close'] + sl_dist,
                    'tp': row['close'] - tp_dist,
                    'type': 'VolClimaxShort'
                })
        else:
            # After bearish run, LONG reversal
            if row['lower_wick'] > row['body']:  # Lower wick rejection
                sl_dist = row['lower_wick'] * 1.5
                tp_dist = sl_dist * rr_ratio
                signals.append({
                    'timestamp': row['timestamp'],
                    'direction': 'LONG',
                    'entry': row['close'],
                    'sl': row['close'] - sl_dist,
                    'tp': row['close'] + tp_dist,
                    'type': 'VolClimaxLong'
                })

    return signals


def generate_channel_bounce_signals(df, lookback=20, wick_min_pct=0.3, rr_ratio=2.0):
    """
    Channel Bounce Strategy (100% WR achieved previously)
    - Price touches Donchian channel boundary
    - Strong wick rejection
    - R:R 2.0
    """
    signals = []

    for i in range(lookback + 1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Calculate channel (using previous bar's values)
        channel_high = prev['channel_high']
        channel_low = prev['channel_low']

        if pd.isna(channel_high) or pd.isna(channel_low):
            continue

        channel_range = channel_high - channel_low
        touch_threshold = channel_range * 0.02  # Within 2% of channel

        # Check for lower channel bounce (LONG)
        if row['low'] <= channel_low + touch_threshold:
            # Need lower wick rejection
            if row['candle_range'] > 0:
                lower_wick_pct = row['lower_wick'] / row['candle_range']
                if lower_wick_pct >= wick_min_pct and row['close'] > row['open']:
                    sl_dist = row['lower_wick'] * 1.5
                    tp_dist = sl_dist * rr_ratio
                    signals.append({
                        'timestamp': row['timestamp'],
                        'direction': 'LONG',
                        'entry': row['close'],
                        'sl': row['close'] - sl_dist,
                        'tp': row['close'] + tp_dist,
                        'type': 'ChannelBounceLong'
                    })

        # Check for upper channel bounce (SHORT)
        if row['high'] >= channel_high - touch_threshold:
            # Need upper wick rejection
            if row['candle_range'] > 0:
                upper_wick_pct = row['upper_wick'] / row['candle_range']
                if upper_wick_pct >= wick_min_pct and row['close'] < row['open']:
                    sl_dist = row['upper_wick'] * 1.5
                    tp_dist = sl_dist * rr_ratio
                    signals.append({
                        'timestamp': row['timestamp'],
                        'direction': 'SHORT',
                        'entry': row['close'],
                        'sl': row['close'] + sl_dist,
                        'tp': row['close'] - tp_dist,
                        'type': 'ChannelBounceShort'
                    })

    return signals


class StrictDailyLimitBacktest:
    """Backtest with daily trade limit"""

    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0002, max_daily_trades=3):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.max_daily_trades = max_daily_trades

    def run(self, df, signals):
        """Run backtest with daily limit"""
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0

        trades = []
        daily_trades = defaultdict(int)

        # Convert signals to list and sort by timestamp
        signal_list = sorted(signals, key=lambda x: x['timestamp'])

        for signal in signal_list:
            ts = signal['timestamp']
            current_date = ts.date()

            # Check daily limit
            if daily_trades[current_date] >= self.max_daily_trades:
                continue

            # Find the candle for this signal
            idx = df[df['timestamp'] == ts].index
            if len(idx) == 0:
                continue
            idx = idx[0]

            entry = signal['entry']
            sl = signal['sl']
            tp = signal['tp']
            direction = signal['direction']

            # Calculate position size based on risk
            sl_distance = abs(entry - sl)
            if sl_distance == 0:
                continue

            risk_amount = capital * self.risk_per_trade
            position_value = risk_amount / (sl_distance / entry)

            # Simulate trade
            # Find exit
            exit_price = None
            exit_type = None
            exit_time = None

            for j in range(idx + 1, min(idx + 100, len(df))):  # Max 100 bars
                candle = df.iloc[j]

                if direction == 'LONG':
                    # Check SL first (conservative)
                    if candle['low'] <= sl:
                        exit_price = sl
                        exit_type = 'SL'
                        exit_time = candle['timestamp']
                        break
                    # Check TP
                    if candle['high'] >= tp:
                        exit_price = tp
                        exit_type = 'TP'
                        exit_time = candle['timestamp']
                        break
                else:  # SHORT
                    # Check SL first
                    if candle['high'] >= sl:
                        exit_price = sl
                        exit_type = 'SL'
                        exit_time = candle['timestamp']
                        break
                    # Check TP
                    if candle['low'] <= tp:
                        exit_price = tp
                        exit_type = 'TP'
                        exit_time = candle['timestamp']
                        break

            if exit_price is None:
                # Timeout - exit at last candle
                exit_price = df.iloc[min(idx + 99, len(df) - 1)]['close']
                exit_type = 'TIMEOUT'
                exit_time = df.iloc[min(idx + 99, len(df) - 1)]['timestamp']

            # Calculate PnL
            if direction == 'LONG':
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry

            # Apply slippage
            pnl_pct -= self.slippage * 2

            # Calculate actual PnL
            pnl = position_value * pnl_pct

            # Commission on position value
            commission_cost = position_value * self.commission * 2
            pnl -= commission_cost

            capital += pnl
            daily_trades[current_date] += 1

            trades.append({
                'entry_time': ts,
                'exit_time': exit_time,
                'direction': direction,
                'type': signal['type'],
                'entry': entry,
                'exit': exit_price,
                'exit_type': exit_type,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'capital': capital
            })

            # Update drawdown
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate stats
        if len(trades) == 0:
            return {
                'trades': 0,
                'win_rate': 0,
                'return': 0,
                'max_dd': 0,
                'pf': 0,
                'avg_days': 0
            }

        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / len(trades) * 100
        total_return = (capital - self.initial_capital) / self.initial_capital * 100

        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Trading days
        unique_days = len(set(t['entry_time'].date() for t in trades))
        avg_trades_per_day = len(trades) / unique_days if unique_days > 0 else 0

        return {
            'trades': len(trades),
            'wins': wins,
            'win_rate': win_rate,
            'return': total_return,
            'max_dd': max_drawdown,
            'pf': profit_factor,
            'avg_tpd': avg_trades_per_day,
            'trade_list': trades
        }


def main():
    df = load_data()

    print("\n" + "=" * 100)
    print("  STRICT HIGH WR + DAILY LIMIT TEST")
    print("=" * 100)

    # Generate signals from 100% WR strategies only
    vol_signals = generate_volume_climax_signals(df, vol_mult=3.0, consec_bars=3, wick_pct=0.5, rr_ratio=1.5)
    channel_signals = generate_channel_bounce_signals(df, lookback=20, wick_min_pct=0.3, rr_ratio=2.0)

    print(f"\nVolume Climax signals: {len(vol_signals)}")
    print(f"Channel Bounce signals: {len(channel_signals)}")

    # Test each strategy separately
    print("\n" + "-" * 100)
    print("  INDIVIDUAL STRATEGY RESULTS")
    print("-" * 100)

    risk_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"\n{'Strategy':<25} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6}")
    print("-" * 85)

    for risk in risk_levels:
        # Volume Climax
        bt = StrictDailyLimitBacktest(risk_per_trade=risk, max_daily_trades=10)
        result = bt.run(df, vol_signals)
        print(f"{'Volume Climax':<25} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['pf']:>6.2f}")

    print()
    for risk in risk_levels:
        # Channel Bounce
        bt = StrictDailyLimitBacktest(risk_per_trade=risk, max_daily_trades=10)
        result = bt.run(df, channel_signals)
        print(f"{'Channel Bounce':<25} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['pf']:>6.2f}")

    # Combined with daily limit
    print("\n" + "-" * 100)
    print("  COMBINED WITH DAILY LIMIT")
    print("-" * 100)

    all_signals = vol_signals + channel_signals
    print(f"\nTotal combined signals: {len(all_signals)}")

    daily_limits = [1, 2, 3, 5, 10]

    print(f"\n{'Daily Limit':>12} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6}")
    print("-" * 70)

    best_result = None
    best_config = None

    for daily_limit in daily_limits:
        for risk in [0.10, 0.15, 0.20, 0.25, 0.30]:
            bt = StrictDailyLimitBacktest(risk_per_trade=risk, max_daily_trades=daily_limit)
            result = bt.run(df, all_signals)

            print(f"{daily_limit:>12} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['pf']:>6.2f}")

            # Track best result meeting criteria (WR >= 70%, Return > 0)
            if result['win_rate'] >= 70 and result['return'] > 0:
                if best_result is None or result['return'] > best_result['return']:
                    best_result = result
                    best_config = {'daily_limit': daily_limit, 'risk': risk}

    # Best configuration
    if best_result:
        print("\n" + "=" * 100)
        print("  BEST CONFIGURATION (WR >= 70%)")
        print("=" * 100)
        print(f"Daily Limit: {best_config['daily_limit']}")
        print(f"Risk: {best_config['risk']*100:.0f}%")
        print(f"Trades: {best_result['trades']}")
        print(f"Win Rate: {best_result['win_rate']:.1f}%")
        print(f"Return: {best_result['return']:+.1f}%")
        print(f"Max DD: {best_result['max_dd']:.1f}%")
        print(f"Profit Factor: {best_result['pf']:.2f}")

        if 'trade_list' in best_result:
            print("\nTrades:")
            for t in best_result['trade_list'][:10]:
                print(f"  {t['entry_time'].strftime('%Y-%m-%d %H:%M')} {t['direction']:<5} {t['type']:<20} "
                      f"Entry:{t['entry']:.0f} Exit:{t['exit']:.0f} ({t['exit_type']}) "
                      f"PnL:{t['pnl']:+.1f} ({t['pnl_pct']:+.1f}%)")

    # Compare with relaxed parameters
    print("\n" + "=" * 100)
    print("  RELAXED PARAMETERS TEST")
    print("=" * 100)

    # Try more relaxed parameters to get more signals
    relaxed_vol_signals = generate_volume_climax_signals(df, vol_mult=2.5, consec_bars=2, wick_pct=0.3, rr_ratio=1.5)
    relaxed_channel_signals = generate_channel_bounce_signals(df, lookback=15, wick_min_pct=0.2, rr_ratio=1.8)

    print(f"\nRelaxed Volume Climax signals: {len(relaxed_vol_signals)}")
    print(f"Relaxed Channel Bounce signals: {len(relaxed_channel_signals)}")

    all_relaxed = relaxed_vol_signals + relaxed_channel_signals
    print(f"Total relaxed signals: {len(all_relaxed)}")

    print(f"\n{'Daily Limit':>12} {'Risk':>6} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MDD':>8} {'PF':>6}")
    print("-" * 70)

    for daily_limit in [1, 2, 3]:
        for risk in [0.10, 0.15, 0.20]:
            bt = StrictDailyLimitBacktest(risk_per_trade=risk, max_daily_trades=daily_limit)
            result = bt.run(df, all_relaxed)

            # Highlight good results
            marker = "***" if result['win_rate'] >= 70 and result['return'] > 50 else ""
            print(f"{daily_limit:>12} {risk*100:>5.0f}% {result['trades']:>8} {result['win_rate']:>7.1f}% {result['return']:>+9.1f}% {result['max_dd']:>7.1f}% {result['pf']:>6.2f} {marker}")


if __name__ == "__main__":
    main()
