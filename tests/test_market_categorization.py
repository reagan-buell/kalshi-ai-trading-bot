import unittest
from src.utils.market_categorization import get_market_type, MarketType

class TestMarketCategorization(unittest.TestCase):
    def test_player_props(self):
        self.assertEqual(get_market_type("Will Patrick Mahomes have over 300 passing yards?"), MarketType.PLAYER_PROP)
        self.assertEqual(get_market_type("How many touchdowns will Justin Jefferson score?"), MarketType.PLAYER_PROP)
        self.assertEqual(get_market_type("NBA Player Performance: LeBron James points scored"), MarketType.PLAYER_PROP)
        self.assertEqual(get_market_type("NFL Player Prop: Travis Kelce receiving yards"), MarketType.PLAYER_PROP)
        
    def test_combo_bets(self):
        self.assertEqual(get_market_type("Combo: Will the Chiefs and Eagles both win?"), MarketType.COMBO)
        self.assertEqual(get_market_type("Will it rain AND will the price of gold go up?"), MarketType.COMBO)
        self.assertEqual(get_market_type("Multi-event: S&P 500 up + Bitcoin down"), MarketType.COMBO)
        self.assertEqual(get_market_type("Will player X score 20 points and player Y get 10 rebounds?"), MarketType.COMBO)

    def test_standard_markets(self):
        self.assertEqual(get_market_type("Will the federal reserve raise rates in March?"), MarketType.STANDARD)
        self.assertEqual(get_market_type("Will the S&P 500 close above 5000?"), MarketType.STANDARD)
        self.assertEqual(get_market_type("Will it snow in Chicago on Christmas?"), MarketType.STANDARD)

    def test_category_fallback(self):
        self.assertEqual(get_market_type("Unknown Title", "NFL Prop"), MarketType.PLAYER_PROP)
        self.assertEqual(get_market_type("Unknown Title", "Daily Combo"), MarketType.COMBO)

if __name__ == "__main__":
    unittest.main()
