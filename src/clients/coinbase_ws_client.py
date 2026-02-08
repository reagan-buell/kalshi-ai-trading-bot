import asyncio
import json
import time
import aiohttp
from typing import Optional, Callable, Dict, Any
from src.utils.logging_setup import get_trading_logger

class CoinbaseWSClient:
    """
    High-speed Coinbase WebSocket client for low-latency price updates.
    Reliable for US-based users.
    """
    
    def __init__(self, product_id: str = "BTC-USD"):
        self.product_id = product_id
        self.url = "wss://ws-feed.exchange.coinbase.com"
        self.logger = get_trading_logger("coinbase_ws")
        self.current_price: float = 0.0
        self.last_update_ts: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._is_running = False
        self._callbacks = []

    def register_callback(self, callback: Callable[[float], Any]):
        """Register a function to be called on every price update."""
        self._callbacks.append(callback)

    async def start(self):
        """Start the WebSocket connection and message processing loop."""
        self._is_running = True
        self._session = aiohttp.ClientSession()
        
        while self._is_running:
            try:
                self.logger.info(f"Connecting to Coinbase WS: {self.url}")
                async with self._session.ws_connect(self.url) as ws:
                    self._ws = ws
                    
                    # Subscribe to ticker channel
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": [self.product_id],
                        "channels": ["ticker"]
                    }
                    await ws.send_str(json.dumps(subscribe_msg))
                    
                    async for msg in ws:
                        if not self._is_running:
                            break
                            
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            
                            if data.get('type') == 'ticker':
                                price = float(data.get('price', 0))
                                if price > 0:
                                    self.current_price = price
                                    self.last_update_ts = time.time()
                                    
                                    # Trigger callbacks
                                    for cb in self._callbacks:
                                        if asyncio.iscoroutinefunction(cb):
                                            await cb(price)
                                        else:
                                            cb(price)
                                            
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception as e:
                self.logger.error(f"Coinbase WS error: {e}")
                if self._is_running:
                    await asyncio.sleep(5)  # Backoff before reconnect

    async def stop(self):
        """Stop the client and close connections."""
        self._is_running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self.logger.info("Coinbase WS client stopped")

    def get_price(self) -> float:
        """Get the latest cached price."""
        return self.current_price
