"""
Market Categorization Utility

Identifies market types (PLAYER_PROP, COMBO, STANDARD) based on titles and categories
to allow for weighted trading strategies.
"""

import re
from enum import Enum
from typing import Optional

class MarketType(Enum):
    PLAYER_PROP = "PLAYER_PROP"
    COMBO = "COMBO"
    STANDARD = "STANDARD"

# Patterns for identifying player props
PROP_PATTERNS = [
    r"passing yards",
    r"rushing yards",
    r"receiving yards",
    r"touchdowns",
    r"total points",
    r"points scored",
    r"rebounds",
    r"assists",
    r"player prop",
    r"player performance",
    r"to score",
    r"over/under"
]

# Patterns for identifying combo bets
COMBO_PATTERNS = [
    r"combo:",
    r"and",
    r"\+",
    r"multi-event",
    r"parlay",
    r"combination"
]

def get_market_type(title: str, category: Optional[str] = None) -> MarketType:
    """
    Identify the type of market based on its title and category.
    """
    if not title:
        return MarketType.STANDARD
        
    title_lower = title.lower()
    
    # Check for combo bets first as they might contain prop-like terms
    for pattern in COMBO_PATTERNS:
        if re.search(pattern, title_lower):
            # Special case: if it's a "player prop combo", it still counts as a combo
            return MarketType.COMBO
            
    # Check for player props
    for pattern in PROP_PATTERNS:
        if re.search(pattern, title_lower):
            return MarketType.PLAYER_PROP
            
    # Check category if provided
    if category:
        cat_lower = category.lower()
        if "prop" in cat_lower:
            return MarketType.PLAYER_PROP
        if "combo" in cat_lower:
            return MarketType.COMBO
            
    return MarketType.STANDARD
