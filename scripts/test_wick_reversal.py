"""Wick Reversal Strategy - 꼬리 반전 진입, 유동적 청산."""

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
    hold_hours: float
    exit_reason: str


class WickReversalBacktest:
    """Wick reversal with flexible exit (scalp to swing)."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.05,
        commission: float = 0.0004,
        slippage: float = 0.0001,
        min_confidence: float = 0.60,
        # 유동적 청산 설정
        min_rr: float = 1.5,      # 최소 R:R (단타 수준)
        max_rr: float = 5.0,      # 최대 R:R (스윙 수준)
        trailing_start_rr: float = 2.0,  # 트레일링 시작 R:R
        trailing_pct: float = 0.02,      # 트레일링 스탑 %
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.min_confidence = min_confidence
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.trailing_start_rr = trailing_start_rr
        self.trailing_pct = trailing_pct

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
            price = row['close']
            high = row['high']
            low = row['low']

            if self.current_position:
                self._check_exit(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if signal.confidence >= self.min_confidence and not self.current_position:
                    self._enter(signal, price, current_time)

            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(price)
            self.equity_curve.append(equity)

        if self.current_position:
            last = df.iloc[-1]
            self._exit(last['close'], last['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal: Signal, price: float, time: datetime):
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)

        # 손절은 꼬리 끝 (signal.stop_loss)
        sl_distance = abs(entry_price - signal.stop_loss)
        sl_pct = sl_distance / entry_price

        if sl_pct == 0 or sl_pct > 0.05:  # 5% 이상 손절은 스킵
            return

        # 리스크 기반 포지션 사이징
        risk_amount = self.capital * self.risk_per_trade
        position_value = risk_amount / sl_pct
        margin = position_value / self.leverage

        max_margin = self.capital * 0.80
        if margin > max_margin:
            margin = max_margin
            position_value = margin * self.leverage

        size = position_value / entry_price
        entry_comm = position_value * self.commission

        if margin + entry_comm > self.capital * 0.95:
            return

        self.capital -= (margin + entry_comm)

        # 유동적 TP 계산
        min_tp_distance = sl_distance * self.min_rr
        max_tp_distance = sl_distance * self.max_rr

        if signal.direction == 'LONG':
            min_tp = entry_price + min_tp_distance
            max_tp = entry_price + max_tp_distance
            trailing_trigger = entry_price + sl_distance * self.trailing_start_rr
        else:
            min_tp = entry_price - min_tp_distance
            max_tp = entry_price - max_tp_distance
            trailing_trigger = entry_price - sl_distance * self.trailing_start_rr

        self.current_position = {
            'entry_price': entry_price,
            'entry_time': time,
            'size': size,
            'margin': margin,
            'position_value': position_value,
            'direction': signal.direction,
            'stop_loss': signal.stop_loss,
            'min_tp': min_tp,
            'max_tp': max_tp,
            'trailing_trigger': trailing_trigger,
            'trailing_stop': None,
            'highest_price': entry_price if signal.direction == 'LONG' else None,
            'lowest_price': entry_price if signal.direction == 'SHORT' else None,
            'sl_distance': sl_distance,
        }

    def _check_exit(self, row):
        if not self.current_position:
            return

        pos = self.current_position
        high = row['high']
        low = row['low']
        close = row['close']
        direction = pos['direction']

        # Update highest/lowest for trailing
        if direction == 'LONG':
            if high > (pos['highest_price'] or 0):
                pos['highest_price'] = high

                # 트레일링 활성화 체크
                if high >= pos['trailing_trigger'] and pos['trailing_stop'] is None:
                    pos['trailing_stop'] = high * (1 - self.trailing_pct)

                # 트레일링 스탑 업데이트
                if pos['trailing_stop'] is not None:
                    new_trailing = high * (1 - self.trailing_pct)
                    if new_trailing > pos['trailing_stop']:
                        pos['trailing_stop'] = new_trailing

        else:  # SHORT
            if low < (pos['lowest_price'] or float('inf')):
                pos['lowest_price'] = low

                if low <= pos['trailing_trigger'] and pos['trailing_stop'] is None:
                    pos['trailing_stop'] = low * (1 + self.trailing_pct)

                if pos['trailing_stop'] is not None:
                    new_trailing = low * (1 + self.trailing_pct)
                    if new_trailing < pos['trailing_stop']:
                        pos['trailing_stop'] = new_trailing

        # Exit checks
        if direction == 'LONG':
            # 1. 손절
            if low <= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
                return

            # 2. 트레일링 스탑
            if pos['trailing_stop'] and low <= pos['trailing_stop']:
                self._exit(pos['trailing_stop'], row['timestamp'], 'TRAIL')
                return

            # 3. 최대 TP
            if high >= pos['max_tp']:
                self._exit(pos['max_tp'], row['timestamp'], 'MAX_TP')
                return

        else:  # SHORT
            # 1. 손절
            if high >= pos['stop_loss']:
                self._exit(pos['stop_loss'], row['timestamp'], 'SL')
                return

            # 2. 트레일링 스탑
            if pos['trailing_stop'] and high >= pos['trailing_stop']:
                self._exit(pos['trailing_stop'], row['timestamp'], 'TRAIL')
                return

            # 3. 최대 TP
            if low <= pos['max_tp']:
                self._exit(pos['max_tp'], row['timestamp'], 'MAX_TP')
                return

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

        self.capital += pos['margin'] + pnl

        hold_hours = (time - pos['entry_time']).total_seconds() / 3600

        self.trades.append(Trade(
            entry_time=pos['entry_time'],
            exit_time=time,
            direction=direction,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            pnl=pnl,
            hold_hours=hold_hours,
            exit_reason=reason,
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

        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))

        # 홀딩 시간 분석
        avg_hold = np.mean([t.hold_hours for t in self.trades])
        scalp_trades = sum(1 for t in self.trades if t.hold_hours < 4)
        day_trades = sum(1 for t in self.trades if 4 <= t.hold_hours < 24)
        swing_trades = sum(1 for t in self.trades if t.hold_hours >= 24)

        return {
            'trades': len(self.trades),
            'wins': len(wins),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_return': total_return,
            'profit_factor': pf,
            'max_drawdown': max_dd,
            'final': self.capital,
            'avg_hold_hours': avg_hold,
            'scalp_trades': scalp_trades,
            'day_trades': day_trades,
            'swing_trades': swing_trades,
            'trade_list': self.trades,
            'exit_reasons': {r: sum(1 for t in self.trades if t.exit_reason == r)
                           for r in ['SL', 'TRAIL', 'MAX_TP', 'END']},
        }


def generate_wick_reversal_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate wick reversal signals - 꼬리 감지 후 반전 진입."""
    signals = []
    df = df.copy()

    # 기본 지표
    df['atr'] = df['tr'].rolling(window=14).mean() if 'tr' in df.columns else None
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=14).mean()

    # 거래량
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA for trend context
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 캔들 분석
    df['body'] = abs(df['close'] - df['open'])
    df['full_range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # 꼬리 비율
    df['upper_wick_pct'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['lower_wick_pct'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    # 스파이크 감지 (ATR 대비 큰 움직임)
    df['range_vs_atr'] = df['full_range'] / df['atr']

    MIN_WICK_PCT = 0.50  # 꼬리가 전체의 50% 이상
    MIN_VOLUME_RATIO = 1.5  # 평균 거래량의 1.5배 이상
    MIN_SPIKE_ATR = 1.5  # ATR의 1.5배 이상 움직임

    for i in range(30, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        timestamp = row['timestamp']

        if pd.isna(row['atr']) or pd.isna(row['rsi']):
            continue

        atr = row['atr']
        rsi = row['rsi']
        vol_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1.0
        lower_wick_pct = row['lower_wick_pct'] if not pd.isna(row['lower_wick_pct']) else 0
        upper_wick_pct = row['upper_wick_pct'] if not pd.isna(row['upper_wick_pct']) else 0
        range_atr = row['range_vs_atr'] if not pd.isna(row['range_vs_atr']) else 0

        # 트렌드 컨텍스트
        uptrend = row['ema20'] > row['ema50']
        downtrend = row['ema20'] < row['ema50']

        # === 아래꼬리 반전 (LONG) ===
        # 조건: 긴 아래꼬리 + 양봉 마감 + 거래량 + 스파이크
        if (lower_wick_pct >= MIN_WICK_PCT and
            row['close'] > row['open'] and  # 양봉
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR):

            # 신뢰도 계산
            conf = 0.55
            conf += min(lower_wick_pct - 0.5, 0.2) * 0.5  # 꼬리 길수록 +
            conf += min((vol_ratio - 1.5) / 5, 0.15)  # 거래량 많을수록 +
            if rsi < 35:  # 과매도
                conf += 0.10
            if uptrend:  # 추세 방향과 일치
                conf += 0.05

            # 손절: 꼬리 끝 (저점) - 약간의 버퍼
            stop_loss = row['low'] - atr * 0.1

            # TP는 백테스트 엔진에서 유동적으로 관리
            take_profit = price + atr * 3  # 임시 (실제로는 트레일링)

            if conf >= 0.60:
                signals.append(Signal(
                    type='SCALP', direction='LONG', confidence=min(0.90, conf),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Lower wick {lower_wick_pct*100:.0f}% Vol {vol_ratio:.1f}x",
                    timestamp=timestamp,
                ))

        # === 윗꼬리 반전 (SHORT) ===
        # 조건: 긴 윗꼬리 + 음봉 마감 + 거래량 + 스파이크
        if (upper_wick_pct >= MIN_WICK_PCT and
            row['close'] < row['open'] and  # 음봉
            vol_ratio >= MIN_VOLUME_RATIO and
            range_atr >= MIN_SPIKE_ATR):

            conf = 0.55
            conf += min(upper_wick_pct - 0.5, 0.2) * 0.5
            conf += min((vol_ratio - 1.5) / 5, 0.15)
            if rsi > 65:  # 과매수
                conf += 0.10
            if downtrend:
                conf += 0.05

            stop_loss = row['high'] + atr * 0.1
            take_profit = price - atr * 3

            if conf >= 0.60:
                signals.append(Signal(
                    type='SCALP', direction='SHORT', confidence=min(0.90, conf),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Upper wick {upper_wick_pct*100:.0f}% Vol {vol_ratio:.1f}x",
                    timestamp=timestamp,
                ))

    return signals


def main():
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min().date()} to {df["timestamp"].max().date()}')

    # Generate wick reversal signals
    signals = generate_wick_reversal_signals(df)
    print(f'\nWick Reversal Signals: {len(signals)}')

    if not signals:
        print("No signals generated!")
        return

    # Signal breakdown
    longs = sum(1 for s in signals if s.direction == 'LONG')
    shorts = sum(1 for s in signals if s.direction == 'SHORT')
    print(f'  LONG: {longs}, SHORT: {shorts}')

    print('\n' + '=' * 75)
    print('  WICK REVERSAL STRATEGY - 50x LEVERAGE')
    print('=' * 75)

    # Test different configurations
    configs = [
        {'risk': 0.03, 'min_rr': 1.5, 'max_rr': 4.0, 'trailing_rr': 2.0},
        {'risk': 0.05, 'min_rr': 1.5, 'max_rr': 5.0, 'trailing_rr': 2.0},
        {'risk': 0.05, 'min_rr': 2.0, 'max_rr': 6.0, 'trailing_rr': 2.5},
        {'risk': 0.08, 'min_rr': 1.5, 'max_rr': 5.0, 'trailing_rr': 2.0},
    ]

    print(f"\n{'Risk':>6} {'R:R':>10} {'Trades':>7} {'Win%':>7} {'Return':>10} {'PF':>6} {'MDD':>8}")
    print('-' * 65)

    best = None
    best_config = None

    for cfg in configs:
        engine = WickReversalBacktest(
            initial_capital=10000.0,
            leverage=50,
            risk_per_trade=cfg['risk'],
            min_rr=cfg['min_rr'],
            max_rr=cfg['max_rr'],
            trailing_start_rr=cfg['trailing_rr'],
        )
        result = engine.run(df, signals)

        if result:
            rr_str = f"{cfg['min_rr']}-{cfg['max_rr']}"
            print(f"{cfg['risk']*100:>5.0f}% {rr_str:>10} {result['trades']:>7} "
                  f"{result['win_rate']:>6.1f}% {result['total_return']:>+9.1f}% "
                  f"{result['profit_factor']:>6.2f} {result['max_drawdown']:>7.1f}%")

            if result['profit_factor'] > 1.0:
                if best is None or result['total_return'] > best['total_return']:
                    best = result
                    best_config = cfg

    print('-' * 65)

    if best:
        print(f'\n  BEST CONFIGURATION')
        print(f'  Risk: {best_config["risk"]*100:.0f}%, R:R: {best_config["min_rr"]}-{best_config["max_rr"]}')
        print()
        print(f'  Total Trades:     {best["trades"]}')
        print(f'  Win Rate:         {best["win_rate"]:.1f}%')
        print(f'  Total Return:     {best["total_return"]:+.2f}%')
        print(f'  Profit Factor:    {best["profit_factor"]:.2f}')
        print(f'  Max Drawdown:     {best["max_drawdown"]:.2f}%')
        print(f'  Final Capital:    ${best["final"]:,.2f}')
        print()
        print(f'  Avg Hold Time:    {best["avg_hold_hours"]:.1f} hours')
        print(f'  Trade Types:      Scalp(<4h): {best["scalp_trades"]}, '
              f'Day(4-24h): {best["day_trades"]}, Swing(>24h): {best["swing_trades"]}')
        print(f'  Exit Reasons:     {best["exit_reasons"]}')
        print()
        print(f'  Monthly Trades:   {best["trades"]/12:.1f}')
        print(f'  Monthly Return:   {best["total_return"]/12:+.1f}%')

        # Monthly breakdown
        if best['trade_list']:
            print('\n  MONTHLY PERFORMANCE')
            print('  ' + '-' * 55)

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
                avg_hold = np.mean([t.hold_hours for t in trades])
                sign = '+' if ret >= 0 else ''
                print(f"  {month}: {len(trades):>2}거래 {wins}승 {sign}{ret:>7.2f}% (평균 {avg_hold:.0f}시간)")

    else:
        print("\n  No profitable configuration found!")

    # Compare with original swing
    print('\n' + '=' * 75)
    print('  COMPARISON: WICK vs SWING')
    print('=' * 75)

    from scripts.test_btc_50x_leverage import LeveragedBacktest as SwingBacktest
    from scripts.run_backtest import generate_swing_signals

    swing_signals = generate_swing_signals(df)
    swing_engine = SwingBacktest(risk_per_trade=0.10)
    swing_result = swing_engine.run(df, swing_signals)

    print(f"\n{'Strategy':>15} {'Trades':>7} {'Win%':>7} {'Return':>10} {'Monthly':>9}")
    print('-' * 55)

    if best:
        print(f"{'WICK REVERSAL':>15} {best['trades']:>7} {best['win_rate']:>6.1f}% "
              f"{best['total_return']:>+9.1f}% {best['total_return']/12:>+8.1f}%")

    if swing_result:
        print(f"{'SWING (4h)':>15} {swing_result['trades']:>7} {swing_result['win_rate']:>6.1f}% "
              f"{swing_result['total_return']:>+9.1f}% {swing_result['total_return']/12:>+8.1f}%")

    print('\n' + '=' * 75)


if __name__ == '__main__':
    main()
