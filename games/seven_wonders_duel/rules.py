"""Small, authoritative rule primitives for the 7 Wonders Duel base game.

This is intentionally not a complete engine yet.  It establishes the arithmetic
and constants that later state transitions can depend on without duplicating
rules throughout the codebase.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class Resource(str, Enum):
    """Resources used to pay construction costs."""

    WOOD = "wood"
    CLAY = "clay"
    STONE = "stone"
    GLASS = "glass"
    PAPYRUS = "papyrus"


RAW_MATERIALS = frozenset({Resource.WOOD, Resource.CLAY, Resource.STONE})
MANUFACTURED_GOODS = frozenset({Resource.GLASS, Resource.PAPYRUS})

NUM_PLAYERS = 2
NUM_AGES = 3
STARTING_COINS = 7
AGE_CARDS_IN_TABLEAU = 20
CARDS_REMOVED_PER_AGE = 3
GUILDS_ADDED_TO_AGE_III = 3
WONDERS_PER_PLAYER = 4
MAX_BUILT_WONDERS = 7
PROGRESS_TOKENS_AVAILABLE = 5
SCIENCE_SYMBOLS_FOR_VICTORY = 6
COINS_PER_VICTORY_POINT = 3


def normal_trade_unit_cost(opponent_brown_grey_production: int) -> int:
    """Return the bank price for one missing resource.

    Only matching symbols on the opponent's brown and grey buildings count.
    Production from yellow buildings and Wonders is deliberately excluded by
    the caller.  Commercial discounts that fix a price at one coin should also
    be applied by the caller instead of using this function.
    """

    if opponent_brown_grey_production < 0:
        raise ValueError("production cannot be negative")
    return 2 + opponent_brown_grey_production


def trade_cost(
    missing: Mapping[Resource, int],
    opponent_brown_grey: Mapping[Resource, int],
    fixed_one_coin: frozenset[Resource] = frozenset(),
) -> int:
    """Return the total cost of buying all missing construction resources."""

    total = 0
    for resource, quantity in missing.items():
        if quantity < 0:
            raise ValueError("missing resource quantities cannot be negative")
        unit_cost = (
            1
            if resource in fixed_one_coin
            else normal_trade_unit_cost(opponent_brown_grey.get(resource, 0))
        )
        total += quantity * unit_cost
    return total


def discard_income(yellow_buildings: int) -> int:
    """Coins received for discarding an accessible Age card."""

    if yellow_buildings < 0:
        raise ValueError("yellow building count cannot be negative")
    return 2 + yellow_buildings


def treasury_victory_points(coins: int) -> int:
    """Civilian-victory points from coins at the end of Age III."""

    if coins < 0:
        raise ValueError("coins cannot be negative")
    return coins // COINS_PER_VICTORY_POINT

