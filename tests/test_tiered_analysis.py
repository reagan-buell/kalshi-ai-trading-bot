import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from src.jobs.decide import make_decision_for_market
from src.utils.database import Market
from src.clients.xai_client import TradingDecision

class TestTieredAnalysis(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db_manager = MagicMock()
        self.db_manager.get_daily_ai_cost = AsyncMock(return_value=0.0)
        self.db_manager.was_recently_analyzed = AsyncMock(return_value=False)
        self.db_manager.get_market_analysis_count_today = AsyncMock(return_value=0)
        self.db_manager.record_market_analysis = AsyncMock()
        
        self.xai_client = MagicMock()
        self.kalshi_client = MagicMock()
        self.kalshi_client.get_balance = AsyncMock(return_value={"balance": 10000})
        
        self.market = Market(
            market_id="TEST-1",
            title="Test Market",
            yes_price=0.5,
            no_price=0.5,
            volume=1000,
            expiration_ts=int(asyncio.get_event_loop().time() + 3600),
            category="Test",
            status="active",
            last_updated=None
        )

    async def test_tiered_gate_skip(self):
        """Verify that if Tier 1 says SKIP, we don't proceed to Tier 2."""
        self.xai_client.get_fast_analysis = AsyncMock(return_value=TradingDecision(
            action="SKIP", side="NONE", confidence=0.2, reasoning="Low potential"
        ))
        
        decision = await make_decision_for_market(
            self.market, self.db_manager, self.xai_client, self.kalshi_client
        )
        
        self.assertIsNone(decision)
        self.xai_client.get_trading_decision.assert_not_called()
        # Verify it was recorded as SKIP_TIER1
        self.db_manager.record_market_analysis.assert_called_with(
            "TEST-1", "SKIP_TIER1", 0.0, 0.005, "fast_filter_rejected"
        )

    async def test_tiered_gate_proceed(self):
        """Verify that if Tier 1 says BUY, we proceed to Tier 2."""
        self.xai_client.get_fast_analysis = AsyncMock(return_value=TradingDecision(
            action="BUY", side="YES", confidence=0.7, reasoning="High potential"
        ))
        self.xai_client.get_trading_decision = AsyncMock(return_value=TradingDecision(
            action="BUY", side="YES", confidence=0.8, limit_price=60, reasoning="Deep reasoning"
        ))
        self.kalshi_client.get_market = AsyncMock(return_value={"market": {"rules": "Test rules"}})
        
        # Mock search to avoid real calls
        self.xai_client.search = AsyncMock(return_value="Diverse news content")
        
        decision = await make_decision_for_market(
            self.market, self.db_manager, self.xai_client, self.kalshi_client
        )
        
        self.assertIsNotNone(decision)
        self.xai_client.get_trading_decision.assert_called()
        self.assertEqual(decision.market_id, "TEST-1")

if __name__ == "__main__":
    unittest.main()
