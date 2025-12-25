"""Example script to test BinanceWebSocketCollector.

This script demonstrates how to use the BinanceWebSocketCollector
to subscribe to real-time market data from Binance Futures.
"""

import asyncio

from src.collectors.binance import (
    BinanceWebSocketCollector,
    KlineData,
    LiquidationData,
    OrderbookData,
    TradeData,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def handle_kline(kline: KlineData) -> None:
    """Handle kline data callback.

    Args:
        kline: Kline data from WebSocket
    """
    if kline.is_closed:
        logger.info(
            "Kline closed",
            interval=kline.interval,
            timestamp=kline.timestamp.isoformat(),
            close=kline.close,
            volume=kline.volume,
        )


def handle_orderbook(orderbook: OrderbookData) -> None:
    """Handle orderbook data callback.

    Args:
        orderbook: Orderbook data from WebSocket
    """
    if orderbook.bids and orderbook.asks:
        best_bid = orderbook.bids[0][0]
        best_ask = orderbook.asks[0][0]
        spread = best_ask - best_bid
        logger.info(
            "Orderbook update",
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            timestamp=orderbook.timestamp.isoformat(),
        )


def handle_trade(trade: TradeData) -> None:
    """Handle trade data callback.

    Args:
        trade: Trade data from WebSocket
    """
    logger.info(
        "Trade",
        price=trade.price,
        quantity=trade.quantity,
        is_buyer_maker=trade.is_buyer_maker,
        timestamp=trade.timestamp.isoformat(),
    )


def handle_liquidation(liquidation: LiquidationData) -> None:
    """Handle liquidation data callback.

    Args:
        liquidation: Liquidation data from WebSocket
    """
    logger.warning(
        "Liquidation",
        side=liquidation.side,
        price=liquidation.price,
        quantity=liquidation.quantity,
        usd_value=liquidation.usd_value,
        timestamp=liquidation.timestamp.isoformat(),
    )


async def main() -> None:
    """Main function to run the WebSocket collector example."""
    collector = BinanceWebSocketCollector()

    # Register callbacks
    collector.on_kline(handle_kline)
    collector.on_orderbook(handle_orderbook)
    collector.on_trade(handle_trade)
    collector.on_liquidation(handle_liquidation)

    # Subscribe to streams
    await collector.subscribe_klines(["1m", "15m"])
    await collector.subscribe_orderbook()
    await collector.subscribe_trades()
    await collector.subscribe_liquidations()

    logger.info("Starting Binance WebSocket collector...")

    # Connect and run
    try:
        await collector.connect()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
