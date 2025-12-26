"""SQLite database for trade history and daily statistics."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/trades.db")


@dataclass
class TradeRecord:
    """Trade record for database storage."""

    id: Optional[int]
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    signal_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "exit_reason": self.exit_reason,
            "signal_reason": self.signal_reason,
        }


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    win_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
        }


class TradeDatabase:
    """SQLite database for trade history.

    Features:
    - Store all trade records
    - Calculate daily statistics
    - Track daily PnL for loss limit
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize database.

        Args:
            db_path: Path to database file. Defaults to data/trades.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info("TradeDatabase initialized", path=str(self.db_path))

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    exit_reason TEXT,
                    signal_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Daily stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_exit_time
                ON trades(exit_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol)
            """)

    def save_trade(self, trade: TradeRecord) -> int:
        """Save a trade record.

        Args:
            trade: Trade record to save

        Returns:
            Trade ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades (
                    symbol, direction, entry_price, exit_price, quantity,
                    leverage, pnl, pnl_pct, entry_time, exit_time,
                    exit_reason, signal_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.symbol,
                    trade.direction,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.leverage,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    trade.exit_reason,
                    trade.signal_reason,
                ),
            )
            trade_id = cursor.lastrowid or 0

            logger.info(
                "Trade saved",
                trade_id=trade_id,
                symbol=trade.symbol,
                direction=trade.direction,
                pnl=trade.pnl,
            )

            # Update daily stats
            self._update_daily_stats(conn, trade.exit_time.date())

            return trade_id

    def _update_daily_stats(self, conn: sqlite3.Connection, trade_date: date) -> None:
        """Update daily statistics for a given date.

        Args:
            conn: Database connection
            trade_date: Date to update
        """
        cursor = conn.cursor()

        # Get all trades for the date
        date_str = trade_date.isoformat()
        cursor.execute(
            """
            SELECT pnl FROM trades
            WHERE date(exit_time) = ?
            ORDER BY exit_time
            """,
            (date_str,),
        )

        rows = cursor.fetchall()
        if not rows:
            return

        pnls = [row["pnl"] for row in rows]
        total_trades = len(pnls)
        winning_trades = sum(1 for p in pnls if p > 0)
        losing_trades = sum(1 for p in pnls if p < 0)
        total_pnl = sum(pnls)

        # Calculate max drawdown (cumulative)
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Upsert daily stats
        cursor.execute(
            """
            INSERT OR REPLACE INTO daily_stats (
                date, total_trades, winning_trades, losing_trades,
                total_pnl, max_drawdown, win_rate, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                total_trades,
                winning_trades,
                losing_trades,
                total_pnl,
                max_drawdown,
                win_rate,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def get_daily_pnl(self, target_date: Optional[date] = None) -> float:
        """Get total PnL for a specific date.

        Args:
            target_date: Date to query (default: today)

        Returns:
            Total PnL for the date
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) as total_pnl
                FROM trades
                WHERE date(exit_time) = ?
                """,
                (target_date.isoformat(),),
            )
            row = cursor.fetchone()
            return float(row["total_pnl"]) if row else 0.0

    def get_daily_stats(self, target_date: Optional[date] = None) -> Optional[DailyStats]:
        """Get daily statistics.

        Args:
            target_date: Date to query (default: today)

        Returns:
            DailyStats or None if no trades
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM daily_stats WHERE date = ?
                """,
                (target_date.isoformat(),),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return DailyStats(
                date=date.fromisoformat(row["date"]),
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                losing_trades=row["losing_trades"],
                total_pnl=row["total_pnl"],
                max_drawdown=row["max_drawdown"],
                win_rate=row["win_rate"],
            )

    def get_trades_today(self) -> list[TradeRecord]:
        """Get all trades from today.

        Returns:
            List of trade records
        """
        today = datetime.now(timezone.utc).date()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                WHERE date(exit_time) = ?
                ORDER BY exit_time DESC
                """,
                (today.isoformat(),),
            )

            trades = []
            for row in cursor.fetchall():
                trades.append(
                    TradeRecord(
                        id=row["id"],
                        symbol=row["symbol"],
                        direction=row["direction"],
                        entry_price=row["entry_price"],
                        exit_price=row["exit_price"],
                        quantity=row["quantity"],
                        leverage=row["leverage"],
                        pnl=row["pnl"],
                        pnl_pct=row["pnl_pct"],
                        entry_time=datetime.fromisoformat(row["entry_time"]),
                        exit_time=datetime.fromisoformat(row["exit_time"]),
                        exit_reason=row["exit_reason"],
                        signal_reason=row["signal_reason"],
                    )
                )

            return trades

    def get_recent_trades(self, limit: int = 10) -> list[TradeRecord]:
        """Get recent trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                ORDER BY exit_time DESC
                LIMIT ?
                """,
                (limit,),
            )

            trades = []
            for row in cursor.fetchall():
                trades.append(
                    TradeRecord(
                        id=row["id"],
                        symbol=row["symbol"],
                        direction=row["direction"],
                        entry_price=row["entry_price"],
                        exit_price=row["exit_price"],
                        quantity=row["quantity"],
                        leverage=row["leverage"],
                        pnl=row["pnl"],
                        pnl_pct=row["pnl_pct"],
                        entry_time=datetime.fromisoformat(row["entry_time"]),
                        exit_time=datetime.fromisoformat(row["exit_time"]),
                        exit_reason=row["exit_reason"],
                        signal_reason=row["signal_reason"],
                    )
                )

            return trades

    def get_all_time_stats(self) -> dict[str, Any]:
        """Get all-time trading statistics.

        Returns:
            Statistics dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                """
            )

            row = cursor.fetchone()

            if not row or row["total_trades"] == 0:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                }

            total = row["total_trades"]
            winning = row["winning_trades"] or 0

            return {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": row["losing_trades"] or 0,
                "win_rate": winning / total if total > 0 else 0.0,
                "total_pnl": row["total_pnl"] or 0.0,
                "avg_pnl": row["avg_pnl"] or 0.0,
                "best_trade": row["best_trade"] or 0.0,
                "worst_trade": row["worst_trade"] or 0.0,
            }


# Global instance
_db_instance: Optional[TradeDatabase] = None


def get_database() -> TradeDatabase:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = TradeDatabase()
    return _db_instance
