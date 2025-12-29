"""Multi-asset backtest for BTC + SOL."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.backtest.engine import BacktestEngine
from scripts.run_backtest import generate_swing_signals, generate_scalp_signals


def run_backtest(df, signals, initial_capital=5000.0):
    """Run backtest with given signals."""
    if not signals:
        return None
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.0004,
        slippage=0.0001,
        risk_per_trade=0.10,
        max_position_pct=0.40,
        min_confidence=0.60,
        partial_tp_enabled=True,
        partial_tp_ratio=0.5,
        partial_tp_rr=2.0,
        move_sl_to_be=True,
    )
    return engine.run(df, signals)


def main():
    # Load both datasets
    btc_df = pd.read_csv('data/BTCUSDT_1h_365d.csv', parse_dates=['timestamp'])
    sol_df = pd.read_csv('data/SOLUSDT_1h_365d.csv', parse_dates=['timestamp'])

    print('=' * 60)
    print('  MULTI-ASSET BACKTEST (BTC + SOL) - Phase 3')
    print('=' * 60)
    print()

    # Generate signals for each
    print('Generating signals...')
    btc_swing = generate_swing_signals(btc_df)
    sol_swing = generate_swing_signals(sol_df)
    btc_scalp = generate_scalp_signals(btc_df)
    sol_scalp = generate_scalp_signals(sol_df)

    print(f'  BTC Swing: {len(btc_swing)} signals')
    print(f'  SOL Swing: {len(sol_swing)} signals')
    print(f'  BTC Scalp: {len(btc_scalp)} signals')
    print(f'  SOL Scalp: {len(sol_scalp)} signals')

    # Run backtests (각 자산에 $5,000 배정)
    print('\nRunning backtests...')
    btc_swing_result = run_backtest(btc_df, btc_swing, 5000.0)
    sol_swing_result = run_backtest(sol_df, sol_swing, 5000.0)

    print()
    print('=' * 60)
    print('  SWING STRATEGY RESULTS')
    print('=' * 60)
    header = f"{'Asset':<12} {'Trades':>8} {'Win%':>8} {'Return':>10} {'PF':>8} {'MDD':>8}"
    print(header)
    print('-' * 60)

    if btc_swing_result:
        print(f"{'BTC':<12} {btc_swing_result.total_trades:>8} "
              f"{btc_swing_result.win_rate:>7.1f}% "
              f"{btc_swing_result.total_return:>+9.2f}% "
              f"{btc_swing_result.profit_factor:>8.2f} "
              f"{btc_swing_result.max_drawdown:>7.2f}%")

    if sol_swing_result:
        print(f"{'SOL':<12} {sol_swing_result.total_trades:>8} "
              f"{sol_swing_result.win_rate:>7.1f}% "
              f"{sol_swing_result.total_return:>+9.2f}% "
              f"{sol_swing_result.profit_factor:>8.2f} "
              f"{sol_swing_result.max_drawdown:>7.2f}%")

    # Combined results
    btc_trades = btc_swing_result.total_trades if btc_swing_result else 0
    sol_trades = sol_swing_result.total_trades if sol_swing_result else 0
    total_trades = btc_trades + sol_trades

    btc_profit = (btc_swing_result.final_capital - 5000) if btc_swing_result else 0
    sol_profit = (sol_swing_result.final_capital - 5000) if sol_swing_result else 0
    total_profit = btc_profit + sol_profit
    total_return = (total_profit / 10000) * 100

    btc_wins = btc_swing_result.winning_trades if btc_swing_result else 0
    sol_wins = sol_swing_result.winning_trades if sol_swing_result else 0
    total_wins = btc_wins + sol_wins
    combined_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    print('-' * 60)
    print(f"{'COMBINED':<12} {total_trades:>8} "
          f"{combined_win_rate:>7.1f}% "
          f"{total_return:>+9.2f}%")

    print()
    print('=' * 60)
    print('  COMBINED PORTFOLIO SUMMARY')
    print('=' * 60)
    print(f'  Initial Capital:    $10,000.00')
    print(f'  Final Capital:      ${10000 + total_profit:,.2f}')
    print(f'  Total Profit:       ${total_profit:+,.2f}')
    print(f'  Total Return:       {total_return:+.2f}%')
    print(f'  Total Trades:       {total_trades}')
    print(f'  Combined Win Rate:  {combined_win_rate:.1f}%')
    print()

    # Annualized projection
    monthly_return = total_return / 12
    print(f'  Monthly Avg Return: {monthly_return:+.2f}%')
    print(f'  Projected Annual:   {total_return:+.2f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()
