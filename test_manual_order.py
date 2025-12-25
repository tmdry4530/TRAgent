"""Manual order test script."""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.executor.binance import BinanceExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_api():
    """Test API connection and order placement."""
    executor = BinanceExecutor()

    try:
        # Test 1: Get account info
        print("\n=== Test 1: Get Account Info ===")
        account = await executor.get_account_info()
        print(f"Total Balance: {account.total_balance} USDT")
        print(f"Available Balance: {account.available_balance} USDT")
        print(f"Unrealized PNL: {account.total_unrealized_pnl} USDT")

        # Test 2: Get current position
        print("\n=== Test 2: Get Position ===")
        position = await executor.get_position()
        if position:
            print(f"Position: {position.side} {position.quantity} @ {position.entry_price}")
            print(f"Unrealized PNL: {position.unrealized_pnl}")
        else:
            print("No open position")

        # Test 3: Set leverage (should work on testnet)
        print("\n=== Test 3: Set Leverage ===")
        success = await executor.set_leverage(10)
        print(f"Set leverage to 10x: {'Success' if success else 'Failed'}")

        # Test 4: Place a small market order (must be > $100 notional)
        # With BTC at ~$97,000, we need at least 0.002 BTC ($194)
        print("\n=== Test 4: Place Market Order ===")
        order = await executor.place_market_order(
            side="BUY",
            quantity=0.002,  # ~$194 at $97k
        )
        print(f"Order Status: {order.status.value}")
        if order.error_message:
            print(f"Error: {order.error_message}")
        else:
            print(f"Order ID: {order.order_id}")
            print(f"Filled Qty: {order.filled_qty}")
            print(f"Avg Price: {order.avg_price}")

        # Test 5: Check position after order
        if order.status.value in ["NEW", "FILLED"]:
            print("\n=== Test 5: Check Position After Order ===")
            position = await executor.get_position()
            if position:
                print(f"Position: {position.side} {position.quantity} @ {position.entry_price}")

                # Test 6: Close position
                print("\n=== Test 6: Close Position ===")
                close_order = await executor.close_position()
                if close_order:
                    print(f"Close Order Status: {close_order.status.value}")
                else:
                    print("No position to close")

        print("\n=== All Tests Completed ===")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await executor.close()


if __name__ == "__main__":
    asyncio.run(test_api())
