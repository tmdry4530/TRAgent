"""Test volume-based signal strategy (no wick requirement)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List


@dataclass
class Signal:
    timestamp: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str


class VolumeSignalGenerator:
    """Volume spike based signal generator."""

    def __init__(
        self,
        vol_multiplier: float = 2.5,      # 2.5x volume spike
        consecutive_bars: int = 2,         # 2 consecutive same direction
        rsi_threshold: float = 45,         # RSI threshold
        rr_ratio: float = 1.5,
        use_wick: bool = False,            # Disable wick requirement
        wick_pct: float = 0.0,             # No wick requirement
    ):
        self.vol_multiplier = vol_multiplier
        self.consecutive_bars = consecutive_bars
        self.rsi_threshold = rsi_threshold
        self.rr_ratio = rr_ratio
        self.use_wick = use_wick
        self.wick_pct = wick_pct

    def generate(self, df: pd.DataFrame) -> List[Signal]:
        df = df.copy()

        # ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(14).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # Volume
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

        # Wick (optional)
        df["full_range"] = df["high"] - df["low"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["upper_wick_pct"] = df["upper_wick"] / df["full_range"].replace(0, np.nan)
        df["lower_wick_pct"] = df["lower_wick"] / df["full_range"].replace(0, np.nan)

        signals = []

        for i in range(50, len(df)):
            row = df.iloc[i]

            if pd.isna(row["volume_ratio"]) or pd.isna(row["rsi"]) or pd.isna(row["atr"]):
                continue

            vol_ratio = row["volume_ratio"]
            rsi = row["rsi"]
            atr = row["atr"]
            price = row["close"]

            # Count consecutive bars
            consec_down = 0
            consec_up = 0
            for j in range(1, self.consecutive_bars + 1):
                if i - j < 0:
                    break
                prev = df.iloc[i - j]
                if prev["close"] < prev["open"]:
                    consec_down += 1
                else:
                    consec_up += 1

            # Volume spike check
            if vol_ratio < self.vol_multiplier:
                continue

            # LONG: Volume spike after downtrend
            if (consec_down >= self.consecutive_bars and
                row["close"] > row["open"] and
                rsi < self.rsi_threshold):

                # Optional wick check
                if self.use_wick and row["lower_wick_pct"] < self.wick_pct:
                    continue

                sl = row["low"] - atr * 0.1
                tp = price + (price - sl) * self.rr_ratio

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    direction="LONG",
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Vol {vol_ratio:.1f}x, RSI {rsi:.0f}",
                ))

            # SHORT: Volume spike after uptrend
            elif (consec_up >= self.consecutive_bars and
                  row["close"] < row["open"] and
                  rsi > (100 - self.rsi_threshold)):

                if self.use_wick and row["upper_wick_pct"] < self.wick_pct:
                    continue

                sl = row["high"] + atr * 0.1
                tp = price - (sl - price) * self.rr_ratio

                signals.append(Signal(
                    timestamp=row["timestamp"],
                    direction="SHORT",
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    reason=f"Vol {vol_ratio:.1f}x, RSI {rsi:.0f}",
                ))

        return signals


class SimpleBacktest:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: int = 50,
        risk_per_trade: float = 0.30,
        commission: float = 0.0004,
        slippage: float = 0.0002,
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage

    def run(self, df: pd.DataFrame, signals: list) -> dict:
        self.capital = self.initial_capital
        self.trades = []
        self.position = None
        self.equity_curve = [self.initial_capital]

        signals = sorted(signals, key=lambda s: s.timestamp)
        signal_idx = 0

        for idx, row in df.iterrows():
            current_time = row["timestamp"]

            if self.position:
                self._check_exit(row)

            while signal_idx < len(signals) and signals[signal_idx].timestamp <= current_time:
                signal = signals[signal_idx]
                signal_idx += 1
                if not self.position:
                    self._enter(signal, row)

            equity = self.capital
            if self.position:
                equity += self._unrealized_pnl(row["close"])
            self.equity_curve.append(equity)

        if self.position:
            self._exit(df.iloc[-1]["close"], df.iloc[-1]["timestamp"], "END")

        return self._calc_metrics()

    def _enter(self, signal, row):
        entry_price = row['close'] * (1 + self.slippage if signal.direction == "LONG" else 1 - self.slippage)
        sl_dist = abs(entry_price - signal.stop_loss) / entry_price

        if sl_dist == 0 or sl_dist > 0.05:
            return

        risk_amt = self.capital * self.risk_per_trade
        pos_val = risk_amt / sl_dist
        margin = pos_val / self.leverage

        if margin > self.capital * 0.95:
            margin = self.capital * 0.95
            pos_val = margin * self.leverage

        size = pos_val / entry_price
        comm = pos_val * self.commission

        if margin + comm >= self.capital:
            return

        self.capital -= (margin + comm)
        self.position = {
            "entry_price": entry_price,
            "entry_time": row['timestamp'],
            "size": size,
            "margin": margin,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        }

    def _check_exit(self, row):
        pos = self.position
        if not pos:
            return

        if pos["direction"] == "LONG":
            if row["low"] <= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            elif row["high"] >= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")
        else:
            if row["high"] >= pos["stop_loss"]:
                self._exit(pos["stop_loss"], row["timestamp"], "SL")
            elif row["low"] <= pos["take_profit"]:
                self._exit(pos["take_profit"], row["timestamp"], "TP")

    def _exit(self, price, time, reason):
        pos = self.position
        if not pos:
            return

        exit_price = price * (1 - self.slippage if pos["direction"] == "LONG" else 1 + self.slippage)

        if pos["direction"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]

        pnl -= exit_price * pos["size"] * self.commission
        self.capital += pos["margin"] + pnl
        pnl_pct = pnl / pos["margin"] * 100

        self.trades.append(Trade(
            entry_time=pos["entry_time"],
            exit_time=time,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        ))
        self.position = None

    def _unrealized_pnl(self, price):
        pos = self.position
        if not pos:
            return 0
        if pos["direction"] == "LONG":
            return (price - pos["entry_price"]) * pos["size"]
        return (pos["entry_price"] - price) * pos["size"]

    def _calc_metrics(self):
        if not self.trades:
            return None

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_ret = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        max_dd = float(np.max((peak - eq) / peak * 100))

        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_return": total_ret,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "final_capital": self.capital,
            "trade_list": self.trades,
        }


def main():
    data_path = Path(__file__).parent.parent / "data" / "BTCUSDT_1h_365d.csv"
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Data: {len(df)} candles ({df['timestamp'].min().date()} ~ {df['timestamp'].max().date()})")

    configs = [
        # Original with wick
        {"name": "ORIGINAL (wick 50%)", "vol": 3.0, "consec": 3, "rsi": 40, "wick": True, "wick_pct": 0.50},

        # Volume only - no wick
        {"name": "VOL 3.0x (no wick)", "vol": 3.0, "consec": 3, "rsi": 40, "wick": False, "wick_pct": 0.0},
        {"name": "VOL 2.5x (no wick)", "vol": 2.5, "consec": 2, "rsi": 45, "wick": False, "wick_pct": 0.0},
        {"name": "VOL 2.0x (no wick)", "vol": 2.0, "consec": 2, "rsi": 50, "wick": False, "wick_pct": 0.0},

        # Reduced wick requirement
        {"name": "VOL 2.5x + wick 30%", "vol": 2.5, "consec": 2, "rsi": 45, "wick": True, "wick_pct": 0.30},
        {"name": "VOL 2.5x + wick 20%", "vol": 2.5, "consec": 2, "rsi": 45, "wick": True, "wick_pct": 0.20},
    ]

    results = []

    for cfg in configs:
        sig_gen = VolumeSignalGenerator(
            vol_multiplier=cfg["vol"],
            consecutive_bars=cfg["consec"],
            rsi_threshold=cfg["rsi"],
            use_wick=cfg["wick"],
            wick_pct=cfg["wick_pct"],
        )
        signals = sig_gen.generate(df)

        bt = SimpleBacktest(risk_per_trade=0.30)
        result = bt.run(df, signals)

        if result:
            results.append({"name": cfg["name"], **result})
            print(f"\n{cfg['name']}: {len(signals)} signals, {result['trades']} trades, "
                  f"{result['win_rate']:.1f}% WR, {result['total_return']:+.1f}% return")

    # Comparison
    print(f"\n{'='*100}")
    print(f"  COMPARISON (30% Risk, 50x Leverage)")
    print(f"{'='*100}")
    print(f"\n{'Config':<25} {'Sigs':>6} {'Trades':>7} {'WR':>7} {'Return':>12} {'MDD':>10} {'PF':>7}")
    print("-" * 85)

    results.sort(key=lambda x: x['total_return'], reverse=True)

    for r in results:
        print(f"{r['name']:<25} {'-':>6} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['total_return']:>+11.1f}% {r['max_drawdown']:>9.1f}% {r['profit_factor']:>7.2f}")

    # Best detail
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"  BEST: {best['name']}")
        print(f"{'='*100}")

        print(f"\n{'#':>3} {'Date':<12} {'Dir':<6} {'Exit':<4} {'PnL%':>10} {'PnL$':>12}")
        print("-" * 55)

        for i, t in enumerate(best['trade_list'][:25], 1):
            print(f"{i:>3} {t.entry_time.strftime('%Y-%m-%d'):<12} {t.direction:<6} "
                  f"{t.exit_reason:<4} {t.pnl_pct:>+9.1f}% {t.pnl:>+11.1f}")

        if len(best['trade_list']) > 25:
            print(f"... +{len(best['trade_list']) - 25} more trades")


if __name__ == "__main__":
    main()
