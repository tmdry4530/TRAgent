"""80% Win Rate Strategy Search - Ultra Selective Signals."""

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
    exit_reason: str


class Backtest:
    def __init__(self, initial_capital=10000.0, leverage=50, risk_per_trade=0.10,
                 commission=0.0004, slippage=0.0002, min_confidence=0.60):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.min_confidence = min_confidence

    def run(self, df, signals):
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

            equity = self.capital + (self._unrealized_pnl(row['close']) if self.current_position else 0)
            self.equity_curve.append(equity)

        if self.current_position:
            self._exit(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal, price, time):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price
        if sl_distance == 0 or sl_distance > 0.05:
            return

        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_distance
        margin = position_value / self.leverage
        max_margin = self.capital * 0.90
        if margin > max_margin:
            margin = max_margin
            position_value = margin * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission
        self.capital -= (margin + entry_comm)

        self.current_position = {
            'entry_price': entry_price, 'entry_time': time, 'size': size,
            'margin': margin, 'position_value': position_value,
            'direction': signal.direction, 'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
        }

    def _check_exit(self, row):
        pos = self.current_position
        if not pos:
            return
        if pos['direction'] == 'LONG':
            if row['low'] <= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif row['high'] >= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')
        else:
            if row['high'] >= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
            elif row['low'] <= pos['take_profit']:
                self._exit(pos['take_profit'], row['timestamp'], 'TP')

    def _exit(self, price, time, reason):
        pos = self.current_position
        if not pos:
            return
        exit_price = price * (1 - self.slippage if pos['direction'] == 'LONG' else 1 + self.slippage)
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']
        pnl -= exit_price * pos['size'] * self.commission
        self.capital += pos['margin'] + pnl

        self.trades.append(Trade(
            pos['entry_time'], time, pos['direction'],
            pos['entry_price'], exit_price, pnl, reason
        ))
        self.current_position = None

    def _unrealized_pnl(self, price):
        pos = self.current_position
        if not pos:
            return 0
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
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))
        return {
            'trades': len(self.trades), 'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'total_return': total_return, 'profit_factor': pf,
            'max_drawdown': max_dd, 'final': self.capital, 'trade_list': self.trades,
        }


def add_indicators(df):
    """Add all indicators to dataframe."""
    df = df.copy()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # EMAs
    for span in [10, 20, 50, 100, 200]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Candle analysis
    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)
    df['body_pct'] = df['body'] / df['full_range'].replace(0, np.nan)

    # Key levels (round numbers)
    df['near_round'] = (df['close'] % 1000) < 200

    # Previous highs/lows
    df['prev_high_20'] = df['high'].rolling(20).max().shift(1)
    df['prev_low_20'] = df['low'].rolling(20).min().shift(1)
    df['prev_high_50'] = df['high'].rolling(50).max().shift(1)
    df['prev_low_50'] = df['low'].rolling(50).min().shift(1)

    # RSI divergence detection
    df['rsi_prev'] = df['rsi'].shift(5)
    df['price_prev'] = df['close'].shift(5)

    # Consecutive candles
    df['bullish'] = df['close'] > df['open']
    df['bearish'] = df['close'] < df['open']
    df['consec_bull'] = df['bullish'].rolling(3).sum()
    df['consec_bear'] = df['bearish'].rolling(3).sum()

    return df


def strategy_1_extreme_oversold_bounce(df):
    """Strategy 1: Extreme oversold with wick bounce at support."""
    signals = []

    for i in range(200, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1

        # LONG: Extreme conditions
        if (rsi < 25 and  # Very oversold
            lower_wick_pct >= 0.60 and  # Long lower wick
            row['close'] > row['open'] and  # Bullish close
            vol_ratio >= 2.0 and  # High volume
            row['low'] <= row['bb_lower'] and  # Touch BB lower
            price > row['ema200']):  # Above major trend

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * 1.5  # R:R 1:1.5 for higher win rate

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Extreme oversold bounce RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_2_trend_pullback(df):
    """Strategy 2: Pullback in strong trend to EMA with wick."""
    signals = []

    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if pd.isna(row['ema50']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Strong uptrend
        strong_uptrend = (row['ema20'] > row['ema50'] > row['ema100'] and
                         price > row['ema20'] and
                         row['macd'] > row['macd_signal'] > 0)

        # Strong downtrend
        strong_downtrend = (row['ema20'] < row['ema50'] < row['ema100'] and
                          price < row['ema20'] and
                          row['macd'] < row['macd_signal'] < 0)

        # LONG: Pullback to EMA20 in uptrend with wick
        if (strong_uptrend and
            row['low'] <= row['ema20'] * 1.005 and  # Touch EMA20
            lower_wick_pct >= 0.50 and  # Wick rejection
            row['close'] > row['open'] and  # Bullish close
            35 <= rsi <= 55):  # Not overbought

            sl = min(row['low'], row['ema50']) - atr * 0.1
            tp = price + (price - sl) * 1.5

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Trend pullback to EMA20",
                timestamp=row['timestamp'],
            ))

        # SHORT: Pullback to EMA20 in downtrend with wick
        if (strong_downtrend and
            row['high'] >= row['ema20'] * 0.995 and
            upper_wick_pct >= 0.50 and
            row['close'] < row['open'] and
            45 <= rsi <= 65):

            sl = max(row['high'], row['ema50']) + atr * 0.1
            tp = price - (sl - price) * 1.5

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Trend pullback to EMA20",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_3_double_bottom_wick(df):
    """Strategy 3: Double bottom/top pattern with wick confirmation."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Look for double bottom (low similar to recent low)
        recent_lows = df.iloc[i-30:i]['low']
        min_low = recent_lows.min()
        min_low_idx = recent_lows.idxmin()

        # Double bottom condition
        double_bottom = (
            abs(row['low'] - min_low) / min_low < 0.01 and  # Within 1%
            i - df.index.get_loc(min_low_idx) >= 10 and  # At least 10 bars apart
            lower_wick_pct >= 0.50 and
            row['close'] > row['open'] and
            rsi < 40
        )

        if double_bottom:
            sl = min(row['low'], min_low) - atr * 0.2
            tp = price + (price - sl) * 2

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Double bottom with wick",
                timestamp=row['timestamp'],
            ))

        # Look for double top
        recent_highs = df.iloc[i-30:i]['high']
        max_high = recent_highs.max()
        max_high_idx = recent_highs.idxmax()

        double_top = (
            abs(row['high'] - max_high) / max_high < 0.01 and
            i - df.index.get_loc(max_high_idx) >= 10 and
            upper_wick_pct >= 0.50 and
            row['close'] < row['open'] and
            rsi > 60
        )

        if double_top:
            sl = max(row['high'], max_high) + atr * 0.2
            tp = price - (sl - price) * 2

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.75,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Double top with wick",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_4_rsi_divergence_wick(df):
    """Strategy 4: RSI divergence with wick confirmation."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['rsi']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Look for bullish divergence (lower low in price, higher low in RSI)
        lookback = 20
        price_low_idx = df.iloc[i-lookback:i]['low'].idxmin()
        price_low = df.loc[price_low_idx, 'low']
        rsi_at_low = df.loc[price_low_idx, 'rsi']

        bullish_div = (
            row['low'] < price_low and  # New low in price
            rsi > rsi_at_low and  # Higher RSI (divergence)
            rsi < 40 and  # Still oversold area
            lower_wick_pct >= 0.50 and
            row['close'] > row['open']
        )

        if bullish_div:
            sl = row['low'] - atr * 0.2
            tp = price + (price - sl) * 2

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Bullish RSI divergence + wick",
                timestamp=row['timestamp'],
            ))

        # Bearish divergence
        price_high_idx = df.iloc[i-lookback:i]['high'].idxmax()
        price_high = df.loc[price_high_idx, 'high']
        rsi_at_high = df.loc[price_high_idx, 'rsi']

        bearish_div = (
            row['high'] > price_high and
            rsi < rsi_at_high and
            rsi > 60 and
            upper_wick_pct >= 0.50 and
            row['close'] < row['open']
        )

        if bearish_div:
            sl = row['high'] + atr * 0.2
            tp = price - (sl - price) * 2

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Bearish RSI divergence + wick",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_5_breakout_retest(df):
    """Strategy 5: Breakout and retest of key levels."""
    signals = []

    # Track broken levels
    broken_resistances = []
    broken_supports = []

    for i in range(100, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Detect breakout of 50-bar high
        prev_high = df.iloc[i-51:i-1]['high'].max()
        prev_low = df.iloc[i-51:i-1]['low'].min()

        # New breakout above resistance
        if prev['close'] <= prev_high and row['close'] > prev_high:
            broken_resistances.append({'level': prev_high, 'bar': i, 'expires': i + 20})

        # New breakout below support
        if prev['close'] >= prev_low and row['close'] < prev_low:
            broken_supports.append({'level': prev_low, 'bar': i, 'expires': i + 20})

        # Clean expired levels
        broken_resistances = [b for b in broken_resistances if b['expires'] > i]
        broken_supports = [b for b in broken_supports if b['expires'] > i]

        # Check for retest of broken resistance (now support)
        for br in broken_resistances:
            if (row['low'] <= br['level'] * 1.005 and  # Retest the level
                row['low'] >= br['level'] * 0.99 and
                lower_wick_pct >= 0.40 and
                row['close'] > row['open'] and
                i - br['bar'] >= 3):  # At least 3 bars after breakout

                sl = br['level'] - atr * 0.5
                tp = price + (price - sl) * 2

                signals.append(Signal(
                    type='SWING', direction='LONG', confidence=0.75,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason=f"Breakout retest (support)",
                    timestamp=row['timestamp'],
                ))
                broken_resistances.remove(br)
                break

        # Check for retest of broken support (now resistance)
        for bs in broken_supports:
            if (row['high'] >= bs['level'] * 0.995 and
                row['high'] <= bs['level'] * 1.01 and
                upper_wick_pct >= 0.40 and
                row['close'] < row['open'] and
                i - bs['bar'] >= 3):

                sl = bs['level'] + atr * 0.5
                tp = price - (sl - price) * 2

                signals.append(Signal(
                    type='SWING', direction='SHORT', confidence=0.75,
                    entry_price=price, stop_loss=sl, take_profit=tp,
                    reason=f"Breakout retest (resistance)",
                    timestamp=row['timestamp'],
                ))
                broken_supports.remove(bs)
                break

    return signals


def strategy_6_volume_climax_reversal(df):
    """Strategy 6: Volume climax with reversal candle."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['atr']) or pd.isna(row['volume_ratio']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        vol_ratio = row['volume_ratio']
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # Check for consecutive down moves before
        consec_down = all(df.iloc[i-j]['close'] < df.iloc[i-j]['open'] for j in range(1, 4))
        consec_up = all(df.iloc[i-j]['close'] > df.iloc[i-j]['open'] for j in range(1, 4))

        # Volume climax reversal LONG
        if (vol_ratio >= 3.0 and  # Very high volume
            consec_down and  # After down move
            lower_wick_pct >= 0.50 and
            row['close'] > row['open'] and
            rsi < 35):

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * 1.5

            signals.append(Signal(
                type='SCALP', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Volume climax reversal Vol:{vol_ratio:.1f}x",
                timestamp=row['timestamp'],
            ))

        # Volume climax reversal SHORT
        if (vol_ratio >= 3.0 and
            consec_up and
            upper_wick_pct >= 0.50 and
            row['close'] < row['open'] and
            rsi > 65):

            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * 1.5

            signals.append(Signal(
                type='SCALP', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Volume climax reversal Vol:{vol_ratio:.1f}x",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_7_multi_timeframe_confluence(df):
    """Strategy 7: Multiple timeframe alignment with wick."""
    signals = []

    # Simulate higher timeframes
    df['ema20_4h'] = df['close'].ewm(span=20*4, adjust=False).mean()
    df['ema50_4h'] = df['close'].ewm(span=50*4, adjust=False).mean()

    delta_4h = df['close'].diff(4)
    gain_4h = delta_4h.where(delta_4h > 0, 0).rolling(14*4).mean()
    loss_4h = (-delta_4h.where(delta_4h < 0, 0)).rolling(14*4).mean()
    df['rsi_4h'] = 100 - (100 / (1 + gain_4h / loss_4h))

    for i in range(200, len(df)):
        row = df.iloc[i]
        price = row['close']

        if pd.isna(row['ema50_4h']) or pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi_1h = row['rsi'] if not pd.isna(row['rsi']) else 50
        rsi_4h = row['rsi_4h'] if not pd.isna(row['rsi_4h']) else 50
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0

        # 4H trend up, 1H pullback with wick
        trend_4h_up = row['ema20_4h'] > row['ema50_4h'] and price > row['ema20_4h']
        trend_4h_down = row['ema20_4h'] < row['ema50_4h'] and price < row['ema20_4h']

        # LONG: 4H uptrend, 1H oversold with wick
        if (trend_4h_up and
            rsi_1h < 35 and
            rsi_4h > 45 and  # 4H not oversold
            lower_wick_pct >= 0.50 and
            row['close'] > row['open']):

            sl = row['low'] - atr * 0.1
            tp = price + (price - sl) * 2

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"MTF confluence: 4H up + 1H oversold",
                timestamp=row['timestamp'],
            ))

        # SHORT: 4H downtrend, 1H overbought with wick
        if (trend_4h_down and
            rsi_1h > 65 and
            rsi_4h < 55 and
            upper_wick_pct >= 0.50 and
            row['close'] < row['open']):

            sl = row['high'] + atr * 0.1
            tp = price - (sl - price) * 2

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"MTF confluence: 4H down + 1H overbought",
                timestamp=row['timestamp'],
            ))

    return signals


def strategy_8_hammer_engulfing(df):
    """Strategy 8: Hammer/Shooting star with engulfing confirmation."""
    signals = []

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']

        if pd.isna(row['atr']):
            continue

        atr = row['atr']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50

        # Hammer detection (on previous candle)
        prev_lower_wick = prev[['open', 'close']].min() - prev['low']
        prev_body = abs(prev['close'] - prev['open'])
        prev_full_range = prev['high'] - prev['low']

        is_hammer = (
            prev_full_range > 0 and
            prev_lower_wick / prev_full_range >= 0.60 and
            prev_body / prev_full_range <= 0.30
        )

        # Bullish engulfing on current candle
        bullish_engulf = (
            row['close'] > row['open'] and
            row['close'] > prev['open'] and
            row['open'] < prev['close']
        )

        # Hammer + Bullish engulfing
        if is_hammer and bullish_engulf and rsi < 40:
            sl = min(prev['low'], row['low']) - atr * 0.1
            tp = price + (price - sl) * 2

            signals.append(Signal(
                type='SWING', direction='LONG', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Hammer + Engulfing RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

        # Shooting star detection
        prev_upper_wick = prev['high'] - prev[['open', 'close']].max()

        is_shooting_star = (
            prev_full_range > 0 and
            prev_upper_wick / prev_full_range >= 0.60 and
            prev_body / prev_full_range <= 0.30
        )

        # Bearish engulfing
        bearish_engulf = (
            row['close'] < row['open'] and
            row['close'] < prev['open'] and
            row['open'] > prev['close']
        )

        if is_shooting_star and bearish_engulf and rsi > 60:
            sl = max(prev['high'], row['high']) + atr * 0.1
            tp = price - (sl - price) * 2

            signals.append(Signal(
                type='SWING', direction='SHORT', confidence=0.80,
                entry_price=price, stop_loss=sl, take_profit=tp,
                reason=f"Shooting star + Engulfing RSI:{rsi:.0f}",
                timestamp=row['timestamp'],
            ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    df = add_indicators(df)

    strategies = [
        ("1. Extreme Oversold Bounce", strategy_1_extreme_oversold_bounce),
        ("2. Trend Pullback", strategy_2_trend_pullback),
        ("3. Double Bottom/Top", strategy_3_double_bottom_wick),
        ("4. RSI Divergence", strategy_4_rsi_divergence_wick),
        ("5. Breakout Retest", strategy_5_breakout_retest),
        ("6. Volume Climax", strategy_6_volume_climax_reversal),
        ("7. MTF Confluence", strategy_7_multi_timeframe_confluence),
        ("8. Hammer/Engulfing", strategy_8_hammer_engulfing),
    ]

    print('\n' + '=' * 85)
    print('  80% WIN RATE STRATEGY SEARCH')
    print('=' * 85)
    print(f"\n{'Strategy':<30} {'Signals':>8} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>7}")
    print('-' * 85)

    results = []

    for name, func in strategies:
        signals = func(df)

        if not signals:
            print(f"{name:<30} {'0':>8} {'-':>7} {'-':>7} {'-':>10} {'-':>7}")
            continue

        engine = Backtest(risk_per_trade=0.10)
        result = engine.run(df, signals)

        if result:
            print(f"{name:<30} {len(signals):>8} {result['trades']:>7} "
                  f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                  f"{result['profit_factor']:>7.2f}")

            results.append({
                'name': name,
                'signals': len(signals),
                'result': result,
            })

    # Find strategies with 70%+ win rate
    print('\n' + '=' * 85)
    print('  HIGH WIN RATE STRATEGIES (>=50%)')
    print('=' * 85)

    high_wr = [r for r in results if r['result']['win_rate'] >= 50]
    high_wr.sort(key=lambda x: x['result']['win_rate'], reverse=True)

    if high_wr:
        print(f"\n{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Return':>10} {'Monthly':>9}")
        print('-' * 70)

        for r in high_wr:
            res = r['result']
            monthly = res['total_return'] / 12
            print(f"{r['name']:<30} {res['trades']:>7} {res['win_rate']:>6.1f}% "
                  f"{res['total_return']:>+9.1f}% {monthly:>+8.1f}%")
    else:
        print("\n  No strategies found with 50%+ win rate individually.")

    # Try combinations
    print('\n' + '=' * 85)
    print('  COMBINED STRATEGIES')
    print('=' * 85)

    # Combine high-quality signals
    all_signals = []
    for name, func in strategies:
        all_signals.extend(func(df))

    # Sort by timestamp
    all_signals.sort(key=lambda s: s.timestamp)

    print(f"\nTotal combined signals: {len(all_signals)}")

    for risk in [0.05, 0.10, 0.15]:
        engine = Backtest(risk_per_trade=risk)
        result = engine.run(df, all_signals)

        if result:
            print(f"Combined @ {risk*100:.0f}% risk: {result['trades']} trades, "
                  f"{result['win_rate']:.1f}% WR, {result['total_return']:+.1f}% return")

    # Best strategy details
    if results:
        best = max(results, key=lambda r: r['result']['total_return'] if r['result']['win_rate'] >= 40 else -999)

        print('\n' + '=' * 85)
        print(f'  BEST STRATEGY: {best["name"]}')
        print('=' * 85)

        res = best['result']
        print(f'''
  Signals:     {best["signals"]}
  Trades:      {res["trades"]}
  Win Rate:    {res["win_rate"]:.1f}%
  Return:      {res["total_return"]:+.2f}%
  PF:          {res["profit_factor"]:.2f}
  Max DD:      {res["max_drawdown"]:.2f}%
  Monthly:     {res["total_return"]/12:+.1f}%
''')

        if res['trade_list']:
            print('  TRADE DETAILS')
            print('  ' + '-' * 70)
            for t in res['trade_list'][:20]:  # First 20
                sign = '+' if t.pnl >= 0 else ''
                print(f"  {t.entry_time.strftime('%Y-%m-%d')} {t.direction:>5} "
                      f"${t.entry_price:,.0f} -> ${t.exit_price:,.0f} "
                      f"{sign}${t.pnl:,.0f} ({t.exit_reason})")

    print('\n' + '=' * 85)


if __name__ == '__main__':
    main()
