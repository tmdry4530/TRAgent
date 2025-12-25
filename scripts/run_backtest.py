"""Full backtest script for scalp and swing strategies."""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Suppress debug logs before importing other modules
logging.basicConfig(level=logging.WARNING)
for logger_name in ['src.backtest.engine', 'src.backtest.download_data', 'src.signals', 'src.utils']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.download_data import BinanceDataDownloader
from src.backtest.engine import BacktestEngine, BacktestResult
from src.signals.base import Signal
from src.utils.logger import get_logger

# Set script logger to INFO to show progress
logger = get_logger(__name__)
logging.getLogger(__name__).setLevel(logging.INFO)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


async def download_data(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 365,
) -> pd.DataFrame:
    """Download historical data from Binance."""

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    cache_file = DATA_DIR / f"{symbol}_{interval}_{days}d.csv"

    # Check cache
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        logger.info(f"Loaded {len(df)} candles from cache")
        return df

    logger.info(f"Downloading {symbol} {interval} data for {days} days...")

    async with BinanceDataDownloader() as downloader:
        df = await downloader.download_klines(
            symbol=symbol,
            interval=interval,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

    # Save to cache
    df.to_csv(cache_file, index=False)
    logger.info(f"Saved {len(df)} candles to {cache_file}")

    return df


def generate_scalp_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate scalp signals from historical data.

    Simulates signal generation based on historical conditions:
    1. LiquidationCascade: Large volume spikes with price reversal
    2. FundingRate: Simulated from price volatility patterns
    3. VolumeBreakout: Volume > 3x average with price breakout
    """
    signals = []

    # Calculate indicators
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=60).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['price_change'] = df['close'].pct_change()
    df['price_ma'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['price_change'].rolling(window=60).std()

    # Skip first 100 rows for indicator warmup
    for i in range(100, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # Skip if indicators are NaN
        if pd.isna(row['volume_ratio']) or pd.isna(row['volatility']):
            continue

        # 1. Volume Breakout Signal
        if row['volume_ratio'] > 3.0:
            # Determine direction based on price movement
            if row['close'] > row['open']:
                direction = "LONG"
                stop_loss = price * 0.985
                take_profit = price * 1.03
            else:
                direction = "SHORT"
                stop_loss = price * 1.015
                take_profit = price * 0.97

            confidence = min(0.8, row['volume_ratio'] / 6.0)

            signal = Signal(
                type="SCALP",
                direction=direction,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Volume breakout: {row['volume_ratio']:.1f}x average",
                timestamp=timestamp,
            )
            signals.append(signal)

        # 2. Simulated Liquidation Cascade (large price move + high volume)
        if abs(row['price_change']) > 0.01 and row['volume_ratio'] > 2.0:
            # Counter-trend entry after cascade
            if row['price_change'] < -0.01:
                direction = "LONG"
                stop_loss = price * 0.985
                take_profit = price * 1.03
            else:
                direction = "SHORT"
                stop_loss = price * 1.015
                take_profit = price * 0.97

            confidence = min(0.85, abs(row['price_change']) * 50)

            signal = Signal(
                type="SCALP",
                direction=direction,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"Liquidation cascade: {row['price_change']*100:.2f}% move",
                timestamp=timestamp,
            )
            signals.append(signal)

        # 3. Simulated Funding Rate Extreme (extended trend)
        # Check for extreme volatility as proxy
        if row['volatility'] > 0.005:
            # Look for trend reversal
            recent_prices = df.iloc[i-10:i]['close']
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]

            if trend > 0.02:  # Extended uptrend -> SHORT
                direction = "SHORT"
                stop_loss = price * 1.015
                take_profit = price * 0.97

                signal = Signal(
                    type="SCALP",
                    direction=direction,
                    confidence=min(0.75, abs(trend) * 20),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Funding rate extreme: {trend*100:.2f}% extended trend",
                    timestamp=timestamp,
                )
                signals.append(signal)
            elif trend < -0.02:  # Extended downtrend -> LONG
                direction = "LONG"
                stop_loss = price * 0.985
                take_profit = price * 1.03

                signal = Signal(
                    type="SCALP",
                    direction=direction,
                    confidence=min(0.75, abs(trend) * 20),
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"Funding rate extreme: {trend*100:.2f}% extended trend",
                    timestamp=timestamp,
                )
                signals.append(signal)

    logger.info(f"Generated {len(signals)} scalp signals")
    return signals


def generate_swing_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate swing signals from historical data.

    Uses EMA crossover and RSI for swing trading signals.
    """
    signals = []

    # Resample to 4h for swing signals
    df_4h = df.set_index('timestamp').resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()

    # Calculate EMAs
    df_4h['ema7'] = df_4h['close'].ewm(span=7, adjust=False).mean()
    df_4h['ema25'] = df_4h['close'].ewm(span=25, adjust=False).mean()
    df_4h['ema99'] = df_4h['close'].ewm(span=99, adjust=False).mean()

    # Calculate RSI
    delta = df_4h['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_4h['rsi'] = 100 - (100 / (1 + rs))

    # Skip first 100 rows for indicator warmup
    for i in range(100, len(df_4h)):
        row = df_4h.iloc[i]
        prev_row = df_4h.iloc[i-1]
        timestamp = row['timestamp']
        price = row['close']

        # Skip if indicators are NaN
        if pd.isna(row['ema99']) or pd.isna(row['rsi']):
            continue

        # Check EMA alignment
        bullish_alignment = row['ema7'] > row['ema25'] > row['ema99']
        bearish_alignment = row['ema7'] < row['ema25'] < row['ema99']

        # Check RSI in neutral zone (40-60)
        rsi_neutral = 40 <= row['rsi'] <= 60

        # Detect crossover
        prev_bullish = prev_row['ema7'] > prev_row['ema25'] if not pd.isna(prev_row['ema25']) else False
        prev_bearish = prev_row['ema7'] < prev_row['ema25'] if not pd.isna(prev_row['ema25']) else False

        bullish_cross = bullish_alignment and not prev_bullish
        bearish_cross = bearish_alignment and not prev_bearish

        if bullish_cross and rsi_neutral:
            signal = Signal(
                type="SWING",
                direction="LONG",
                confidence=0.75,
                entry_price=price,
                stop_loss=price * 0.95,  # 5% SL
                take_profit=price * 1.15,  # 15% TP
                reason=f"EMA bullish alignment + RSI {row['rsi']:.1f}",
                timestamp=timestamp,
            )
            signals.append(signal)

        elif bearish_cross and rsi_neutral:
            signal = Signal(
                type="SWING",
                direction="SHORT",
                confidence=0.75,
                entry_price=price,
                stop_loss=price * 1.05,  # 5% SL
                take_profit=price * 0.85,  # 15% TP
                reason=f"EMA bearish alignment + RSI {row['rsi']:.1f}",
                timestamp=timestamp,
            )
            signals.append(signal)

    logger.info(f"Generated {len(signals)} swing signals")
    return signals


def calculate_monthly_returns(trades: list, initial_capital: float) -> dict:
    """Calculate monthly returns from trades."""
    monthly_pnl = defaultdict(float)

    for trade in trades:
        month_key = trade.exit_time.strftime("%Y-%m")
        monthly_pnl[month_key] += trade.pnl

    # Convert to returns percentage
    running_capital = initial_capital
    monthly_returns = {}

    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        return_pct = (pnl / running_capital) * 100
        monthly_returns[month] = return_pct
        running_capital += pnl

    return monthly_returns


def calculate_max_consecutive_losses(trades: list) -> int:
    """Calculate maximum consecutive losing trades."""
    max_losses = 0
    current_losses = 0

    for trade in trades:
        if trade.pnl < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0

    return max_losses


def print_detailed_report(
    result: BacktestResult,
    strategy_name: str,
    monthly_returns: dict,
    max_consecutive_losses: int,
) -> None:
    """Print detailed backtest report."""
    print("\n" + "=" * 70)
    print(f"  {strategy_name} BACKTEST REPORT")
    print("=" * 70)

    # Basic metrics
    print("\n[PERFORMANCE METRICS]")
    print("-" * 40)
    print(f"  Total Trades:        {result.total_trades:,}")
    print(f"  Winning Trades:      {result.winning_trades:,} ({result.win_rate:.1f}%)")
    print(f"  Losing Trades:       {result.losing_trades:,}")
    print(f"  ")
    print(f"  Total Return:        {result.total_return:+.2f}%")
    print(f"  Initial Capital:     ${result.initial_capital:,.2f}")
    print(f"  Final Capital:       ${result.final_capital:,.2f}")
    print(f"  Net Profit:          ${result.final_capital - result.initial_capital:+,.2f}")

    # Risk metrics
    print("\n[RISK METRICS]")
    print("-" * 40)
    print(f"  Profit Factor:       {result.profit_factor:.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown:.2f}%")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:.2f}")
    print(f"  Max Consec. Losses:  {max_consecutive_losses}")

    # Trade metrics
    print("\n[TRADE METRICS]")
    print("-" * 40)
    print(f"  Avg Win:             ${result.avg_win:+,.2f}")
    print(f"  Avg Loss:            ${result.avg_loss:+,.2f}")
    print(f"  Max Win:             ${result.max_win:+,.2f}")
    print(f"  Max Loss:            ${result.max_loss:+,.2f}")
    print(f"  Avg Hold Time:       {result.avg_hold_time:.1f} hours")
    print(f"  Total Commission:    ${result.total_commission:,.2f}")

    # Monthly returns
    print("\n[MONTHLY RETURNS]")
    print("-" * 40)
    for month, ret in sorted(monthly_returns.items()):
        bar = "#" * int(abs(ret) / 2) if ret != 0 else ""
        sign = "+" if ret >= 0 else ""
        status = "[+]" if ret >= 0 else "[-]"
        print(f"  {month}: {status} {sign}{ret:6.2f}%  {bar}")

    # Summary statistics
    if monthly_returns:
        returns = list(monthly_returns.values())
        print(f"\n  Avg Monthly:         {np.mean(returns):+.2f}%")
        print(f"  Best Month:          {max(returns):+.2f}%")
        print(f"  Worst Month:         {min(returns):+.2f}%")
        print(f"  Positive Months:     {sum(1 for r in returns if r > 0)}/{len(returns)}")

    print("\n" + "=" * 70)


async def main():
    """Run full backtest."""
    print("\n" + "=" * 60)
    print("           BINANCE BTC FUTURES BACKTEST")
    print("=" * 60 + "\n")

    # Step 1: Download data
    print("[Step 1] Downloading 1 year of BTCUSDT 1h data...")
    df = await download_data(symbol="BTCUSDT", interval="1h", days=365)
    print(f"   [OK] Downloaded {len(df):,} candles")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Generate signals
    print("\n[Step 2] Generating signals...")
    scalp_signals = generate_scalp_signals(df)
    swing_signals = generate_swing_signals(df)
    print(f"   [OK] Scalp signals: {len(scalp_signals):,}")
    print(f"   [OK] Swing signals: {len(swing_signals):,}")

    # Step 3: Run scalp backtest
    print("\n[Step 3] Running SCALP strategy backtest...")
    scalp_engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,
        slippage=0.0001,
        fixed_position_value=1000.0,  # Fixed $1000 per trade
    )
    scalp_result = scalp_engine.run(df, scalp_signals)

    scalp_monthly = calculate_monthly_returns(scalp_result.trades, 10000.0)
    scalp_max_losses = calculate_max_consecutive_losses(scalp_result.trades)

    print_detailed_report(
        scalp_result,
        "SCALP STRATEGY",
        scalp_monthly,
        scalp_max_losses,
    )

    # Step 4: Run swing backtest
    print("\n[Step 4] Running SWING strategy backtest...")
    swing_engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,
        slippage=0.0001,
        fixed_position_value=2000.0,  # Fixed $2000 per trade (larger for swing)
    )
    swing_result = swing_engine.run(df, swing_signals)

    swing_monthly = calculate_monthly_returns(swing_result.trades, 10000.0)
    swing_max_losses = calculate_max_consecutive_losses(swing_result.trades)

    print_detailed_report(
        swing_result,
        "SWING STRATEGY",
        swing_monthly,
        swing_max_losses,
    )

    # Combined summary
    print("\n" + "=" * 70)
    print("                    COMBINED SUMMARY")
    print("=" * 70)
    print("\n  Strategy Comparison:")
    print(f"  {'Metric':<25} {'Scalp':>15} {'Swing':>15}")
    print("  " + "-" * 55)
    print(f"  {'Total Trades':<25} {scalp_result.total_trades:>15,} {swing_result.total_trades:>15,}")
    print(f"  {'Win Rate':<25} {scalp_result.win_rate:>14.1f}% {swing_result.win_rate:>14.1f}%")
    print(f"  {'Total Return':<25} {scalp_result.total_return:>+14.1f}% {swing_result.total_return:>+14.1f}%")
    print(f"  {'Profit Factor':<25} {scalp_result.profit_factor:>15.2f} {swing_result.profit_factor:>15.2f}")
    print(f"  {'Max Drawdown':<25} {scalp_result.max_drawdown:>14.1f}% {swing_result.max_drawdown:>14.1f}%")
    print(f"  {'Sharpe Ratio':<25} {scalp_result.sharpe_ratio:>15.2f} {swing_result.sharpe_ratio:>15.2f}")
    print(f"  {'Max Consec. Losses':<25} {scalp_max_losses:>15} {swing_max_losses:>15}")

    # Final recommendation
    print("\n" + "=" * 70)
    print("                    RECOMMENDATIONS")
    print("=" * 70)

    if scalp_result.profit_factor > 1.5 and scalp_result.max_drawdown < 20:
        print("  [PASS] Scalp strategy meets target criteria (PF > 1.5, MDD < 20%)")
    else:
        print("  [WARN] Scalp strategy needs optimization")

    if swing_result.profit_factor > 1.5 and swing_result.max_drawdown < 20:
        print("  [PASS] Swing strategy meets target criteria (PF > 1.5, MDD < 20%)")
    else:
        print("  [WARN] Swing strategy needs optimization")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
