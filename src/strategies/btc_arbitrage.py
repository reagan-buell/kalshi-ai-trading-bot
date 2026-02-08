import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.clients.kalshi_client import KalshiClient
from src.clients.coinbase_ws_client import CoinbaseWSClient
from src.utils.database import DatabaseManager, Market, Position
from src.utils.logging_setup import get_trading_logger
from src.jobs.execute import execute_position

class BTCArbitrageStrategy:
    """
    High-frequency arbitrage strategy for Bitcoin 15-min markets.
    Compares real-time Binance prices with lagging Kalshi contract prices.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        coinbase_client: CoinbaseWSClient,
        min_profit_threshold: float = 0.05,  # 5% minimum net edge
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.coinbase_client = coinbase_client
        self.logger = get_trading_logger("btc_arbitrage")
        self.min_profit_threshold = min_profit_threshold
        self.active_ticker_patterns = ["BTC-"] # Tickers starting with BTC-
        
    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """
        Main loop to scan for price discrepancies.
        """
        opportunities = []
        btc_price = self.coinbase_client.get_price()
        
        if btc_price <= 0:
            self.logger.warning("No Coinbase price available yet")
            return []

        # 1. Fetch relevant BTC markets from Kalshi
        # Focus on those expiring soon (15-min markets)
        markets = await self.db_manager.get_eligible_markets(
            volume_min=10, 
            max_days_to_expiry=1
        )
        
        btc_markets = [m for m in markets if m.market_id.startswith("BTC-")]
        
        for market in btc_markets:
            # Parse strike price from title or ticker if possible
            # Kalshi BTC titles usually: "Bitcoin will be above $68,500.00 at 11:15 AM ET?"
            strike_price = self._extract_strike_price(market)
            if not strike_price:
                continue
                
            # 2. Compare current Binance price to Strike
            # Determine "Fair Value"
            fair_value = self._calculate_fair_value(btc_price, strike_price, market)
            
            # 3. Check Kalshi prices (Yes/No)
            # Kalshi Yes price is the cost to buy a 'Yes' contract
            yes_ask = market.yes_price # This is already normalized to 0-1
            no_ask = market.no_price
            
            # Arb Case A: Kalshi Yes is too cheap
            if fair_value > yes_ask + self.min_profit_threshold:
                opportunities.append({
                    "market": market,
                    "side": "YES",
                    "edge": fair_value - yes_ask,
                    "binance_price": btc_price,
                    "strike_price": strike_price,
                    "kalshi_price": yes_ask
                })
                
            # Arb Case B: Kalshi No is too cheap (Fair Yes is low, but No is priced higher)
            # Fair No price = 1 - fair_value
            elif (1 - fair_value) > no_ask + self.min_profit_threshold:
                opportunities.append({
                    "market": market,
                    "side": "NO",
                    "edge": (1 - fair_value) - no_ask,
                    "binance_price": btc_price,
                    "strike_price": strike_price,
                    "kalshi_price": no_ask
                })
                
        return opportunities

    def _extract_strike_price(self, market: Market) -> Optional[float]:
        """Extract the numeric strike price from market metadata."""
        import re
        # Example title: "Bitcoin will be above $68,500.00 at 11:15 AM ET?"
        match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', market.title)
        if match:
            return float(match.group(1).replace(',', ''))
        return None

    def _calculate_fair_value(self, btc_price: float, strike_price: float, market: Market) -> float:
        """
        Simplified 'Fair Value' calculation. 
        In high-frequency arbitrage near expiry, it's often close to 1 if price > strike + buffer.
        """
        # Distance to strike
        diff = btc_price - strike_price
        
        # Time to expiration in seconds
        time_left = market.expiration_ts - datetime.now().timestamp()
        
        if time_left <= 0:
            return 1.0 if diff > 0 else 0.0
            
        # Simplified probability based on distance and standard deviation
        # For 15-min markets, volatility is key
        # Using a very basic heuristic for now:
        if diff > 100: return 0.95
        if diff < -100: return 0.05
        
        # Linear approximation for demo, can be improved with Black-Scholes/CDF
        prob = 0.5 + (diff / 200) # 200$ range maps to 0 to 1
        return max(0.01, min(0.99, prob))

    async def execute_arbitrage(self, opportunity: Dict, capital: float):
        """Execute the arbitrage trade."""
        market = opportunity['market']
        side = opportunity['side']
        price = opportunity['kalshi_price']
        
        quantity = int(capital / price) if price > 0 else 0
        if quantity < 1: return

        position = Position(
            market_id=market.market_id,
            side=side,
            entry_price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            rationale=f"BTC ARB: Binance ${opportunity['binance_price']} vs Strike ${opportunity['strike_price']}. Edge: {opportunity['edge']:.1%}",
            confidence=0.9,
            live=True, # Arbs are usually live-only targets
            strategy="btc_arbitrage"
        )
        
        self.logger.info(f"⚡ EXECUTING BTC ARB: {side} {quantity} contracts on {market.market_id}")
        # Add to DB
        await self.db_manager.add_position(position)
        
        # Execute
        await execute_position(position, live_mode=True, db_manager=self.db_manager, kalshi_client=self.kalshi_client)
