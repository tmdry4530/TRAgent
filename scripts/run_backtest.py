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

    Phase 3 강화 버전:
    - 거래량 기준: 3x (완화)
    - RSI 필터: 38/62 (완화)
    - MACD OR 조건 (완화)
    - 볼린저 밴드 시그널 추가
    - 신뢰도 기준 60%로 하향
    """
    signals = []

    # Calculate indicators
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['price_change'] = df['close'].pct_change()
    df['price_ma'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['price_change'].rolling(window=20).std()

    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD calculation
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Phase 3: 볼린저 밴드 계산
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Phase 3: 캔들 패턴 계산
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['full_range'] = df['high'] - df['low']
    df['wick_ratio_up'] = df['upper_wick'] / df['full_range'].replace(0, np.nan)
    df['wick_ratio_down'] = df['lower_wick'] / df['full_range'].replace(0, np.nan)

    # Phase 3 완화된 기준
    VOLUME_THRESHOLD = 3.0  # 5.0 → 3.0
    RSI_OVERSOLD = 38  # 35 → 38
    RSI_OVERBOUGHT = 62  # 65 → 62
    MIN_CONFIDENCE = 0.60  # 0.65 → 0.60

    # Skip first 100 rows for indicator warmup
    for i in range(100, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        price = row['close']
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_signal = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0
        macd_histogram = row['macd_histogram'] if not pd.isna(row['macd_histogram']) else 0
        bb_pct = row['bb_pct'] if not pd.isna(row['bb_pct']) else 0.5
        wick_up = row['wick_ratio_up'] if not pd.isna(row['wick_ratio_up']) else 0
        wick_down = row['wick_ratio_down'] if not pd.isna(row['wick_ratio_down']) else 0

        # Skip if indicators are NaN
        if pd.isna(row['volume_ratio']) or pd.isna(row['volatility']):
            continue

        # Phase 3: MACD 완화된 체크 (OR 조건)
        def check_macd_relaxed(direction):
            if direction == "LONG":
                # MACD > Signal OR 히스토그램 상승 중 OR 히스토그램 > 0
                return macd > macd_signal or macd_histogram > 0
            else:
                return macd < macd_signal or macd_histogram < 0

        # 1. Volume Breakout Signal (Phase 3: 완화된 조건)
        if row['volume_ratio'] >= VOLUME_THRESHOLD:
            if row['close'] > row['open'] and rsi < RSI_OVERSOLD:
                direction = "LONG"
                stop_loss = price * 0.985
                take_profit = price * 1.03
            elif row['close'] < row['open'] and rsi > RSI_OVERBOUGHT:
                direction = "SHORT"
                stop_loss = price * 1.015
                take_profit = price * 0.97
            else:
                direction = None

            if direction and check_macd_relaxed(direction):
                base_conf = 0.55
                volume_bonus = min(0.15, (row['volume_ratio'] - 3) / 20)
                rsi_bonus = 0.1 if (rsi < 32 or rsi > 68) else 0.05
                macd_bonus = min(abs(macd_histogram) * 40, 0.1)
                confidence = min(0.9, base_conf + volume_bonus + rsi_bonus + macd_bonus)

                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type="SCALP", direction=direction, confidence=confidence,
                        entry_price=price, stop_loss=stop_loss, take_profit=take_profit,
                        reason=f"Volume {row['volume_ratio']:.1f}x + RSI {rsi:.0f}",
                        timestamp=timestamp,
                    ))

        # 2. Phase 3: 볼린저 밴드 반전 시그널
        if row['volume_ratio'] >= 2.0:
            # 하단 밴드 터치 + 양봉 = LONG
            if bb_pct < 0.05 and row['close'] > row['open'] and rsi < 45:
                if check_macd_relaxed("LONG"):
                    confidence = 0.6 + min((0.05 - bb_pct) * 4, 0.15) + min((45 - rsi) / 100, 0.1)
                    if confidence >= MIN_CONFIDENCE:
                        signals.append(Signal(
                            type="SCALP", direction="LONG", confidence=min(0.85, confidence),
                            entry_price=price, stop_loss=row['bb_lower'] * 0.998,
                            take_profit=row['bb_mid'],
                            reason=f"BB lower touch + RSI {rsi:.0f}",
                            timestamp=timestamp,
                        ))

            # 상단 밴드 터치 + 음봉 = SHORT
            if bb_pct > 0.95 and row['close'] < row['open'] and rsi > 55:
                if check_macd_relaxed("SHORT"):
                    confidence = 0.6 + min((bb_pct - 0.95) * 4, 0.15) + min((rsi - 55) / 100, 0.1)
                    if confidence >= MIN_CONFIDENCE:
                        signals.append(Signal(
                            type="SCALP", direction="SHORT", confidence=min(0.85, confidence),
                            entry_price=price, stop_loss=row['bb_upper'] * 1.002,
                            take_profit=row['bb_mid'],
                            reason=f"BB upper touch + RSI {rsi:.0f}",
                            timestamp=timestamp,
                        ))

        # 3. Phase 3: 윅 반전 패턴 (완화된 조건)
        if row['volume_ratio'] >= 1.5:
            # 긴 하위꼬리 + 양봉 = LONG
            if wick_down > 0.6 and row['close'] > row['open'] and rsi < 45:
                if check_macd_relaxed("LONG"):
                    confidence = 0.55 + min(wick_down * 0.25, 0.2) + min((45 - rsi) / 100, 0.1)
                    if confidence >= MIN_CONFIDENCE:
                        signals.append(Signal(
                            type="SCALP", direction="LONG", confidence=min(0.85, confidence),
                            entry_price=price, stop_loss=row['low'] * 0.998,
                            take_profit=price * 1.025,
                            reason=f"Wick reversal {wick_down*100:.0f}% + RSI {rsi:.0f}",
                            timestamp=timestamp,
                        ))

            # 긴 상위꼬리 + 음봉 = SHORT
            if wick_up > 0.6 and row['close'] < row['open'] and rsi > 55:
                if check_macd_relaxed("SHORT"):
                    confidence = 0.55 + min(wick_up * 0.25, 0.2) + min((rsi - 55) / 100, 0.1)
                    if confidence >= MIN_CONFIDENCE:
                        signals.append(Signal(
                            type="SCALP", direction="SHORT", confidence=min(0.85, confidence),
                            entry_price=price, stop_loss=row['high'] * 1.002,
                            take_profit=price * 0.975,
                            reason=f"Wick reversal {wick_up*100:.0f}% + RSI {rsi:.0f}",
                            timestamp=timestamp,
                        ))

        # 4. Liquidation Cascade (완화된 조건)
        if abs(row['price_change']) > 0.012 and row['volume_ratio'] >= 3.0:
            if row['price_change'] < -0.012 and rsi < RSI_OVERSOLD + 5:
                direction = "LONG"
                stop_loss = price * 0.985
                take_profit = price * 1.03
            elif row['price_change'] > 0.012 and rsi > RSI_OVERBOUGHT - 5:
                direction = "SHORT"
                stop_loss = price * 1.015
                take_profit = price * 0.97
            else:
                direction = None

            if direction and check_macd_relaxed(direction):
                confidence = min(0.85, 0.6 + abs(row['price_change']) * 10)
                if confidence >= MIN_CONFIDENCE:
                    signals.append(Signal(
                        type="SCALP", direction=direction, confidence=confidence,
                        entry_price=price, stop_loss=stop_loss, take_profit=take_profit,
                        reason=f"Cascade {row['price_change']*100:.1f}%",
                        timestamp=timestamp,
                    ))

    logger.info(f"Generated {len(signals)} scalp signals (Phase 3 + BB)")
    return signals


def generate_swing_signals(df: pd.DataFrame) -> list[Signal]:
    """Generate swing signals from historical data.

    Phase 3 강화 버전:
    - EMA 갭 기반 신뢰도 계산
    - RSI 범위 확장 (35-65)
    - MACD OR 조건 (완화)
    - 신뢰도 기준 60%로 하향
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

    # Phase 2: MACD calculation
    ema_fast = df_4h['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df_4h['close'].ewm(span=26, adjust=False).mean()
    df_4h['macd'] = ema_fast - ema_slow
    df_4h['macd_signal'] = df_4h['macd'].ewm(span=9, adjust=False).mean()
    df_4h['macd_histogram'] = df_4h['macd'] - df_4h['macd_signal']

    MIN_CONFIDENCE = 0.60  # Phase 3: 최소 신뢰도 하향

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

        # Phase 3: Check RSI in expanded zone (35-65)
        rsi_neutral = 35 <= row['rsi'] <= 65

        # Detect crossover
        prev_bullish = prev_row['ema7'] > prev_row['ema25'] if not pd.isna(prev_row['ema25']) else False
        prev_bearish = prev_row['ema7'] < prev_row['ema25'] if not pd.isna(prev_row['ema25']) else False

        bullish_cross = bullish_alignment and not prev_bullish
        bearish_cross = bearish_alignment and not prev_bearish

        # Phase 1: 강화된 신뢰도 계산
        def calculate_swing_confidence(ema7, ema25, ema99, rsi):
            """EMA 갭 기반 신뢰도 계산"""
            confidence = 0.55  # Phase 1: 낮춘 base

            # EMA 갭 보너스
            ema_7_25_gap = abs(ema7 - ema25) / ema25
            ema_25_99_gap = abs(ema25 - ema99) / ema99

            if ema_7_25_gap > 0.05:
                confidence += 0.15
            elif ema_7_25_gap > 0.03:
                confidence += 0.10
            elif ema_7_25_gap > 0.02:
                confidence += 0.05

            if ema_25_99_gap > 0.05:
                confidence += 0.15
            elif ema_25_99_gap > 0.03:
                confidence += 0.10
            elif ema_25_99_gap > 0.02:
                confidence += 0.05

            # RSI 중립 보너스
            rsi_dist = abs(rsi - 50)
            if rsi_dist < 3:
                confidence += 0.05
            elif rsi_dist < 5:
                confidence += 0.03

            return min(0.95, confidence)

        # Phase 2: MACD 값 추출
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_signal_val = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0
        macd_histogram = row['macd_histogram'] if not pd.isna(row['macd_histogram']) else 0

        # Phase 2: MACD 모멘텀 확인 함수
        def check_swing_macd(direction):
            if direction == "LONG":
                return macd > macd_signal_val  # MACD가 시그널 위에 있어야 함
            else:
                return macd < macd_signal_val  # MACD가 시그널 아래 있어야 함

        if bullish_cross and rsi_neutral:
            # Phase 2: MACD 필터
            if not check_swing_macd("LONG"):
                continue

            confidence = calculate_swing_confidence(
                row['ema7'], row['ema25'], row['ema99'], row['rsi']
            )

            # Phase 2: MACD 보너스
            macd_bonus = min(abs(macd_histogram) / 100, 0.05)
            confidence = min(0.95, confidence + macd_bonus)

            if confidence >= MIN_CONFIDENCE:
                signal = Signal(
                    type="SWING",
                    direction="LONG",
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=price * 0.95,  # 5% SL
                    take_profit=price * 1.15,  # 15% TP
                    reason=f"EMA bullish + RSI {row['rsi']:.1f} + MACD confirmed",
                    timestamp=timestamp,
                )
                signals.append(signal)

        elif bearish_cross and rsi_neutral:
            # Phase 2: MACD 필터
            if not check_swing_macd("SHORT"):
                continue

            confidence = calculate_swing_confidence(
                row['ema7'], row['ema25'], row['ema99'], row['rsi']
            )

            # Phase 2: MACD 보너스
            macd_bonus = min(abs(macd_histogram) / 100, 0.05)
            confidence = min(0.95, confidence + macd_bonus)

            if confidence >= MIN_CONFIDENCE:
                signal = Signal(
                    type="SWING",
                    direction="SHORT",
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=price * 1.05,  # 5% SL
                    take_profit=price * 0.85,  # 15% TP
                    reason=f"EMA bearish + RSI {row['rsi']:.1f} + MACD confirmed",
                    timestamp=timestamp,
                )
                signals.append(signal)

    logger.info(f"Generated {len(signals)} swing signals (Phase 3)")
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

    # Step 3: Run scalp backtest (Phase 3: 공격적 포지션 사이징)
    print("\n[Step 3] Running SCALP strategy backtest (Phase 3)...")
    scalp_engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,
        slippage=0.0001,
        risk_per_trade=0.03,  # Phase 3: 거래당 3% 리스크
        max_position_pct=0.15,  # Phase 3: 최대 15% 포지션
        min_confidence=0.60,  # Phase 3: 최소 신뢰도 60%
        # 부분익절 설정
        partial_tp_enabled=True,
        partial_tp_ratio=0.5,
        partial_tp_rr=2.0,
        move_sl_to_be=True,
    )
    scalp_result = scalp_engine.run(df, scalp_signals)

    scalp_monthly = calculate_monthly_returns(scalp_result.trades, 10000.0)
    scalp_max_losses = calculate_max_consecutive_losses(scalp_result.trades)

    print_detailed_report(
        scalp_result,
        "SCALP STRATEGY (Phase 3: BB + Aggressive Sizing)",
        scalp_monthly,
        scalp_max_losses,
    )

    # Step 4: Run swing backtest (Phase 3: 공격적 포지션 사이징)
    print("\n[Step 4] Running SWING strategy backtest (Phase 3)...")
    swing_engine = BacktestEngine(
        initial_capital=10000.0,
        commission=0.0004,
        slippage=0.0001,
        risk_per_trade=0.05,  # Phase 3: 스윙은 거래당 5% 리스크
        max_position_pct=0.20,  # Phase 3: 최대 20% 포지션
        min_confidence=0.60,  # Phase 3: 최소 신뢰도 60%
        partial_tp_enabled=True,
        partial_tp_ratio=0.5,
        partial_tp_rr=2.0,
        move_sl_to_be=True,
    )
    swing_result = swing_engine.run(df, swing_signals)

    swing_monthly = calculate_monthly_returns(swing_result.trades, 10000.0)
    swing_max_losses = calculate_max_consecutive_losses(swing_result.trades)

    print_detailed_report(
        swing_result,
        "SWING STRATEGY (Phase 3: Aggressive Sizing)",
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
