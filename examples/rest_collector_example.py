"""Example usage of BinanceRestCollector for fetching market data.

This script demonstrates how to use the REST API collector to fetch:
- Funding rates
- Open interest
- Long/Short account ratios
"""

import asyncio

from src.collectors.binance import BinanceRestCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def main() -> None:
    """Main example function."""
    collector = BinanceRestCollector()

    try:
        # Fetch funding rate
        logger.info("Fetching funding rate...")
        funding_rate = await collector.get_funding_rate("BTCUSDT")
        logger.info(
            "Funding Rate",
            symbol=funding_rate.symbol,
            rate=funding_rate.funding_rate,
            mark_price=funding_rate.mark_price,
            funding_time=funding_rate.funding_time,
        )

        # Fetch open interest
        logger.info("Fetching open interest...")
        open_interest = await collector.get_open_interest("BTCUSDT")
        logger.info(
            "Open Interest",
            symbol=open_interest.symbol,
            oi=open_interest.open_interest,
            oi_value=open_interest.open_interest_value,
        )

        # Fetch long/short ratio
        logger.info("Fetching long/short ratio...")
        ls_ratio = await collector.get_long_short_ratio("BTCUSDT", period="5m")
        logger.info(
            "Long/Short Ratio",
            symbol=ls_ratio.symbol,
            long_pct=ls_ratio.long_account,
            short_pct=ls_ratio.short_account,
            ratio=ls_ratio.long_short_ratio,
        )

    except Exception as e:
        logger.error("Error in example", error=str(e))
    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
