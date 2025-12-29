"""BTC 50x Leverage Backtest - Proper leverage simulation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from scripts.run_backtest import generate_swing_signals
from src.signals.base import Signal


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float  # Position size in BTC
    margin_used: float  # Margin used (position / leverage)
    pnl: float
    pnl_pct: float  # PnL as % of margin
    exit_reason: str


class LeveragedBacktest:
    """Backtest engine with proper leverage simulation."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.10,  # Risk 10% of capital per trade
        commission: float = 0.0004,  # 0.04% taker
        slippage: float = 0.0002,  # 0.02%
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
        """Run backtest."""
        self.capital = self.initial_capital
        self.trades = []
        self.current_position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row['timestamp']

            # Check exit conditions
            if self.current_position:
                self._check_exit(row)

            # Process new signals
            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1

                if signal.confidence >= self.min_confidence and not self.current_position:
                    self._enter(signal, row['close'], current_time)

            # Update equity
            equity = self.capital
            if self.current_position:
                equity += self._unrealized_pnl(row['close'])
            self.equity_curve.append(equity)

        # Close at end
        if self.current_position:
            last = df.iloc[-1]
            self._exit(last['close'], last['timestamp'], 'END')

        return self._metrics()

    def _enter(self, signal: Signal, price: float, time: datetime):
        """Enter position with leverage."""
        entry_price = price * (1 + self.slippage if signal.direction == 'LONG' else 1 - self.slippage)

        # Calculate position size based on risk
        sl_distance = abs(entry_price - signal.stop_loss) / entry_price
        if sl_distance == 0:
            return

        # Risk amount
        risk_amount = self.capital * self.risk_per_trade

        # Position value = Risk / SL Distance
        # This is the notional value of the position
        position_value = risk_amount / sl_distance

        # With leverage, margin required = position_value / leverage
        margin_required = position_value / self.leverage

        # Cap margin at 90% of capital (safety buffer)
        max_margin = self.capital * 0.90
        if margin_required > max_margin:
            margin_required = max_margin
            position_value = margin_required * self.leverage

        # Position size in BTC
        size = position_value / entry_price

        # Commission on entry
        entry_comm = position_value * self.commission

        # Deduct margin and commission
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
        """Check SL/TP conditions."""
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
        """Exit position."""
        if not self.current_position:
            return

        pos = self.current_position
        direction = pos['direction']
        exit_price = price * (1 - self.slippage if direction == 'LONG' else 1 + self.slippage)

        # Calculate PnL on full position
        if direction == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']

        # Exit commission
        exit_value = exit_price * pos['size']
        exit_comm = exit_value * self.commission
        pnl -= exit_comm

        # PnL as % of margin used
        pnl_pct = (pnl / pos['margin']) * 100

        # Return margin + PnL to capital
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
        """Calculate metrics."""
        if not self.trades:
            return None

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown
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


def main():
    # Load data
    df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    print(f'Loaded {len(df)} BTC 1h candles')
    print(f'Period: {df["timestamp"].min()} to {df["timestamp"].max()}')

    # Generate Phase 3 swing signals
    signals = generate_swing_signals(df)
    print(f'Generated {len(signals)} swing signals')

    print('\n' + '=' * 70)
    print('  BTC 50x LEVERAGE BACKTEST (PROPER SIMULATION)')
    print('=' * 70)

    # Test different risk levels with 50x leverage
    print(f"\n{'Leverage':>10} {'Risk%':>8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'PF':>8} {'MDD':>10}")
    print('-' * 70)

    test_configs = [
        (50, 0.05),  # 50x, 5% risk
        (50, 0.10),  # 50x, 10% risk
        (50, 0.15),  # 50x, 15% risk
        (50, 0.20),  # 50x, 20% risk
        (50, 0.25),  # 50x, 25% risk
        (50, 0.30),  # 50x, 30% risk (aggressive)
    ]

    best_result = None
    best_config = None

    for lev, risk in test_configs:
        engine = LeveragedBacktest(
            initial_capital=10000.0,
            leverage=lev,
            risk_per_trade=risk,
            min_confidence=0.60,
        )
        result = engine.run(df, signals)

        if result:
            print(f"{lev:>10}x {risk*100:>7.0f}% {result['trades']:>8} "
                  f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                  f"{result['profit_factor']:>8.2f} {result['max_drawdown']:>9.2f}%")

            if best_result is None or (result['profit_factor'] > 1.0 and
                                       result['total_return'] > best_result['total_return']):
                best_result = result
                best_config = (lev, risk)

    print('-' * 70)

    if best_result:
        print(f'\n  BEST CONFIGURATION')
        print(f'  Leverage: {best_config[0]}x, Risk: {best_config[1]*100:.0f}%')
        print()
        print(f'  Initial Capital:   ${best_result["initial"]:,.2f}')
        print(f'  Final Capital:     ${best_result["final"]:,.2f}')
        print(f'  Total Return:      {best_result["total_return"]:+.2f}%')
        print(f'  Net Profit:        ${best_result["final"] - best_result["initial"]:+,.2f}')
        print()
        print(f'  Total Trades:      {best_result["trades"]}')
        print(f'  Win Rate:          {best_result["win_rate"]:.1f}%')
        print(f'  Profit Factor:     {best_result["profit_factor"]:.2f}')
        print(f'  Max Drawdown:      {best_result["max_drawdown"]:.2f}%')
        print()
        print(f'  Avg PnL per Trade: ${best_result["avg_pnl"]:+,.2f}')
        print(f'  Avg Win:           ${best_result["avg_win"]:+,.2f}')
        print(f'  Avg Loss:          ${best_result["avg_loss"]:,.2f}')

        # Monthly breakdown
        if best_result['trade_list']:
            print('\n  MONTHLY BREAKDOWN')
            print('  ' + '-' * 40)
            from collections import defaultdict
            monthly = defaultdict(float)
            for t in best_result['trade_list']:
                m = t.exit_time.strftime('%Y-%m')
                monthly[m] += t.pnl

            cap = 10000
            for month in sorted(monthly.keys()):
                pnl = monthly[month]
                ret = (pnl / cap) * 100
                cap += pnl
                bar = '#' * int(abs(ret) / 5) if ret != 0 else ''
                sign = '+' if ret >= 0 else ''
                print(f"  {month}: {sign}{ret:7.2f}%  {bar}")

        # Trade-by-trade detail
        print('\n  TRADE DETAILS')
        print('  ' + '-' * 60)
        print(f"  {'Exit':>12} {'Dir':>6} {'Entry':>10} {'Exit':>10} {'PnL':>12} {'Reason':>8}")
        for t in best_result['trade_list']:
            print(f"  {t.exit_time.strftime('%Y-%m-%d'):>12} {t.direction:>6} "
                  f"{t.entry_price:>10,.2f} {t.exit_price:>10,.2f} "
                  f"{t.pnl:>+12,.2f} {t.exit_reason:>8}")

    # Compare leverage levels
    print('\n' + '=' * 70)
    print('  LEVERAGE COMPARISON (10% Risk Fixed)')
    print('=' * 70)
    print(f"\n{'Leverage':>10} {'Trades':>8} {'Win%':>8} {'Return':>12} {'MDD':>10} {'Risk/Reward':>12}")
    print('-' * 70)

    for lev in [10, 20, 30, 40, 50, 75, 100]:
        engine = LeveragedBacktest(
            initial_capital=10000.0,
            leverage=lev,
            risk_per_trade=0.10,
            min_confidence=0.60,
        )
        result = engine.run(df, signals)

        if result:
            rr = result['total_return'] / result['max_drawdown'] if result['max_drawdown'] > 0 else 0
            print(f"{lev:>10}x {result['trades']:>8} "
                  f"{result['win_rate']:>7.1f}% {result['total_return']:>+11.2f}% "
                  f"{result['max_drawdown']:>9.2f}% {rr:>11.2f}")

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
