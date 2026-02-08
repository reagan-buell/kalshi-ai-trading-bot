import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.clients.coinbase_ws_client import CoinbaseWSClient
from src.utils.logging_setup import setup_logging

async def test_coinbase_feed():
    setup_logging(log_level="INFO")
    client = CoinbaseWSClient(product_id="BTC-USD")
    
    def on_price(price):
        print(f"🔥 BTC Price Update: ${price:,.2f}")

    client.register_callback(on_price)
    
    print("Connecting to Coinbase... (will run for 10 seconds)")
    task = asyncio.create_task(client.start())
    
    await asyncio.sleep(10)
    
    print(f"Final cached price: ${client.get_price():,.2f}")
    await client.stop()
    task.cancel()

if __name__ == "__main__":
    asyncio.run(test_coinbase_feed())
