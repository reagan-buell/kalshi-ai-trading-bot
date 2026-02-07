import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from src.strategies.portfolio_optimization import create_market_opportunities_from_markets
from src.utils.database import Market
from src.utils.market_categorization import MarketType
from src.config.settings import settings

class TestPortfolioOptimizationWeights(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mock clients
        self.xai_client = MagicMock()
        self.xai_client.get_completion = AsyncMock(return_value='{"probability": 0.7, "confidence": 0.8}')
        
        self.kalshi_client = MagicMock()
        self.kalshi_client.get_market = AsyncMock(return_value={
            'market': {
                'yes_price': 60,
                'no_price': 40,
                'status': 'active'
            }
        })

    async def test_weight_adjustments(self):
        # 1. Standard Market
        standard_market = Market(
            market_id="STD-1",
            title="Will it rain in NY?",
            yes_price=0.6,
            no_price=0.4,
            volume=50000,
            expiration_ts=1700000000,
            category="Weather",
            status="active",
            last_updated=None
        )
        
        # 2. Player Prop Market
        prop_market = Market(
            market_id="PROP-1",
            title="Patrick Mahomes over 300 passing yards",
            yes_price=0.6,
            no_price=0.4,
            volume=50000,
            expiration_ts=1700000000,
            category="Sports",
            status="active",
            last_updated=None
        )
        
        # 3. Combo Market
        combo_market = Market(
            market_id="COMBO-1",
            title="Combo: Chiefs and Eagles win",
            yes_price=0.6,
            no_price=0.4,
            volume=50000,
            expiration_ts=1700000000,
            category="Sports",
            status="active",
            last_updated=None
        )
        
        markets = [standard_market, prop_market, combo_market]
        
        # Run opportunity creation
        opportunities = await create_market_opportunities_from_markets(
            markets, self.xai_client, self.kalshi_client, None, 10000
        )
        
        # Map by ID for easy checking
        opp_map = {o.market_id: o for o in opportunities}
        
        # Check Standard Market (No change to confidence)
        self.assertAlmostEqual(opp_map["STD-1"].confidence, 0.8)
        
        # Check Player Prop (Confidence boost: 0.8 * 1.25 = 1.0)
        self.assertAlmostEqual(opp_map["PROP-1"].confidence, 1.0)
        
        # Check Combo Market (Confidence penalty: 0.8 * 0.5 = 0.4)
        self.assertAlmostEqual(opp_map["COMBO-1"].confidence, 0.4)

if __name__ == "__main__":
    unittest.main()
