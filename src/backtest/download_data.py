"""Binance historical data downloader for backtesting."""

import asyncio
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BinanceDataDownloader:
    """Download historical kline data from Binance Futures API."""

    BASE_URL = "https://fapi.binance.com"
    KLINES_ENDPOINT = "/fapi/v1/klines"
    MAX_LIMIT = 1500  # Binance limit per request

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def download_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1m",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        Download historical kline data from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with OHLCV data and timestamp

        Example:
            >>> async with BinanceDataDownloader() as downloader:
            ...     df = await downloader.download_klines(
            ...         symbol="BTCUSDT",
            ...         interval="1h",
            ...         start_date="2024-01-01",
            ...         end_date="2024-12-31",
            ...     )
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            should_close = True
        else:
            should_close = False

        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

            logger.info(
                f"Downloading {symbol} {interval} klines from {start_date} to {end_date}"
            )

            all_klines = []
            current_ts = start_ts

            while current_ts < end_ts:
                klines = await self._fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_ts,
                    end_time=end_ts,
                    limit=self.MAX_LIMIT,
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Update timestamp to last kline's close time + 1ms
                current_ts = klines[-1][6] + 1

                logger.debug(
                    f"Downloaded {len(klines)} klines, total: {len(all_klines)}"
                )

                # Rate limiting - Binance allows 2400 requests per minute
                # Use conservative delay to avoid 429 errors
                await asyncio.sleep(0.5)

            df = self._process_klines(all_klines)

            logger.info(
                f"Downloaded {len(df)} klines from {df['timestamp'].min()} to {df['timestamp'].max()}"
            )

            return df

        finally:
            if should_close and self.session:
                await self.session.close()
                self.session = None

    async def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int,
        max_retries: int = 5,
    ) -> list[list]:
        """Fetch single batch of klines from Binance API with retry logic."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }

        url = f"{self.BASE_URL}{self.KLINES_ENDPOINT}"

        if not self.session:
            raise RuntimeError("Session not initialized")

        for attempt in range(max_retries):
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    # Rate limited - wait and retry
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s...
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Binance API error: {response.status} - {error_text}"
                    )

        raise RuntimeError("Max retries exceeded for Binance API")

    def _process_klines(self, klines: list[list]) -> pd.DataFrame:
        """
        Process raw kline data into DataFrame.

        Binance kline format:
        [
            [
                1499040000000,      # Open time
                "0.01634000",       # Open
                "0.80000000",       # High
                "0.01575800",       # Low
                "0.01577100",       # Close
                "148976.11427815",  # Volume
                1499644799999,      # Close time
                "2434.19055334",    # Quote asset volume
                308,                # Number of trades
                "1756.87402397",    # Taker buy base asset volume
                "28.46694368",      # Taker buy quote asset volume
                "17928899.62484339" # Ignore
            ]
        ]
        """
        df = pd.DataFrame(klines, columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ])

        # Convert types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Use open_time as primary timestamp
        df["timestamp"] = df["open_time"]

        # Convert price and volume columns to float
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        # Select and reorder columns
        df = df[[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
        ]]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)

        return df

    async def save_to_csv(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        output_path: str | None = None,
    ) -> str:
        """
        Download and save kline data to CSV file.

        Args:
            symbol: Trading pair
            interval: Kline interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_path: Output CSV path (auto-generated if None)

        Returns:
            Path to saved CSV file
        """
        df = await self.download_klines(symbol, interval, start_date, end_date)

        if output_path is None:
            output_path = f"data/{symbol}_{interval}_{start_date}_{end_date}.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} klines to {output_path}")

        return output_path


async def main():
    """Example usage."""
    async with BinanceDataDownloader() as downloader:
        # Download 1-hour data for backtesting
        df = await downloader.download_klines(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        print(f"Downloaded {len(df)} klines")
        print(df.head())
        print(df.tail())

        # Save to CSV
        await downloader.save_to_csv(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/btc_1h_2024.csv",
        )


if __name__ == "__main__":
    asyncio.run(main())
