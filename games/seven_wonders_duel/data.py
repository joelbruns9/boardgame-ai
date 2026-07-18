"""Typed base-game component data transcribed from the official rulebook.

Source pages are the printed page numbers in the English 2015 rulebook linked
from README.md.  Card faces and chain relationships are on pages 18-19;
Progress, Guild, and Wonder effects are on pages 14-17; tableau layouts are on
page 20.

The rulebook does not provide a table of Wonder construction costs. Eight costs
are visible in its setup photograph; the remaining four were checked against a
single component photograph linked in TRANSCRIPTION_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .rules import Resource


class CardColor(str, Enum):
    BROWN = "brown"
    GREY = "grey"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    PURPLE = "purple"


class BackType(str, Enum):
    """Printed card backs. Backs are public information even when faces are not."""

    AGE_I = "age_i"
    AGE_II = "age_ii"
    AGE_III = "age_iii"
    GUILD = "guild"


class ScienceSymbol(str, Enum):
    ARMILLARY_SPHERE = "armillary_sphere"
    WHEEL = "wheel"
    SUNDIAL = "sundial"
    MORTAR_AND_PESTLE = "mortar_and_pestle"
    SET_SQUARE = "set_square"
    QUILL_AND_INK = "quill_and_ink"
    LAW = "law"


class EffectKind(str, Enum):
    IMMEDIATE_COINS = "immediate_coins"
    OPPONENT_LOSES_COINS = "opponent_loses_coins"
    PLAY_AGAIN = "play_again"
    COINS_PER_OWN_COLOR = "coins_per_own_color"
    COINS_PER_OWN_WONDER = "coins_per_own_wonder"
    COINS_PER_MOST_COLOR = "coins_per_most_color"
    COINS_PER_MOST_BROWN_GREY = "coins_per_most_brown_grey"
    VP_PER_MOST_COLOR = "vp_per_most_color"
    VP_PER_MOST_WONDER = "vp_per_most_wonder"
    VP_PER_RICHEST_COIN_SET = "vp_per_richest_coin_set"
    VP_PER_MOST_BROWN_GREY = "vp_per_most_brown_grey"
    DESTROY_OPPONENT_BROWN = "destroy_opponent_brown"
    DESTROY_OPPONENT_GREY = "destroy_opponent_grey"
    BUILD_FROM_DISCARD_FREE = "build_from_discard_free"
    CHOOSE_UNUSED_PROGRESS = "choose_unused_progress"
    FUTURE_WONDER_RESOURCE_DISCOUNT = "future_wonder_resource_discount"
    RECEIVE_OPPONENT_TRADE_SPEND = "receive_opponent_trade_spend"
    FUTURE_BLUE_RESOURCE_DISCOUNT = "future_blue_resource_discount"
    VP_PER_PROGRESS = "vp_per_progress"
    FUTURE_RED_EXTRA_SHIELD = "future_red_extra_shield"
    FUTURE_WONDER_PLAY_AGAIN = "future_wonder_play_again"
    COINS_PER_CHAIN_BUILD = "coins_per_chain_build"


@dataclass(frozen=True, slots=True)
class Cost:
    coins: int = 0
    wood: int = 0
    clay: int = 0
    stone: int = 0
    glass: int = 0
    papyrus: int = 0

    def resource_count(self, resource: Resource) -> int:
        return getattr(self, resource.value)

    @property
    def total_resources(self) -> int:
        return self.wood + self.clay + self.stone + self.glass + self.papyrus


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    amount: int = 1
    color: CardColor | None = None


@dataclass(frozen=True, slots=True)
class CardData:
    name: str
    age: int
    color: CardColor
    cost: Cost = Cost()
    victory_points: int = 0
    shields: int = 0
    fixed_production: tuple[Resource, ...] = ()
    choice_production: tuple[Resource, ...] = ()
    trade_discount: frozenset[Resource] = frozenset()
    science: ScienceSymbol | None = None
    chain_from: str | None = None
    chain_to: str | None = None
    effects: tuple[Effect, ...] = ()
    source_aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProgressData:
    name: str
    victory_points: int = 0
    science: ScienceSymbol | None = None
    effects: tuple[Effect, ...] = ()


@dataclass(frozen=True, slots=True)
class WonderData:
    name: str
    cost: Cost | None
    victory_points: int = 0
    shields: int = 0
    choice_production: tuple[Resource, ...] = ()
    effects: tuple[Effect, ...] = ()
    cost_source: str | None = None


@dataclass(frozen=True, slots=True)
class TableauSlot:
    """One slot using integer half-column coordinates from rulebook page 20."""

    row: int
    x: int
    face_up: bool


def _e(kind: EffectKind, amount: int = 1, color: CardColor | None = None) -> Effect:
    return Effect(kind, amount, color)


def _card(name: str, age: int, color: CardColor, **kwargs) -> CardData:
    return CardData(name=name, age=age, color=color, **kwargs)


# Chain identifiers describe the printed icon rather than a source card name.
HORSESHOE = "horseshoe"
SWORD = "sword"
FORT = "fort"
TARGET = "target"
HELMET = "helmet"
BOOK = "book"
GEAR = "gear"
LYRE = "lyre"
LAMP = "lamp"
MASK = "mask"
MOON = "moon"
WATER_DROP = "water_drop"
COLUMN = "column"
SUN = "sun"
BUILDING = "building"
JUG = "jug"
BARREL = "barrel"


AGE_I_CARDS = (
    _card("Lumber Yard", 1, CardColor.BROWN, fixed_production=(Resource.WOOD,)),
    _card("Logging Camp", 1, CardColor.BROWN, cost=Cost(coins=1), fixed_production=(Resource.WOOD,)),
    _card("Clay Pool", 1, CardColor.BROWN, fixed_production=(Resource.CLAY,)),
    _card("Clay Pit", 1, CardColor.BROWN, cost=Cost(coins=1), fixed_production=(Resource.CLAY,)),
    _card("Quarry", 1, CardColor.BROWN, fixed_production=(Resource.STONE,)),
    _card("Stone Pit", 1, CardColor.BROWN, cost=Cost(coins=1), fixed_production=(Resource.STONE,)),
    _card("Glassworks", 1, CardColor.GREY, cost=Cost(coins=1), fixed_production=(Resource.GLASS,)),
    _card("Press", 1, CardColor.GREY, cost=Cost(coins=1), fixed_production=(Resource.PAPYRUS,)),
    _card("Guard Tower", 1, CardColor.RED, shields=1),
    _card("Workshop", 1, CardColor.GREEN, cost=Cost(papyrus=1), victory_points=1, science=ScienceSymbol.SET_SQUARE),
    _card("Apothecary", 1, CardColor.GREEN, cost=Cost(glass=1), victory_points=1, science=ScienceSymbol.WHEEL),
    _card("Stone Reserve", 1, CardColor.YELLOW, cost=Cost(coins=3), trade_discount=frozenset({Resource.STONE})),
    _card("Clay Reserve", 1, CardColor.YELLOW, cost=Cost(coins=3), trade_discount=frozenset({Resource.CLAY})),
    _card("Wood Reserve", 1, CardColor.YELLOW, cost=Cost(coins=3), trade_discount=frozenset({Resource.WOOD})),
    _card("Stable", 1, CardColor.RED, cost=Cost(wood=1), shields=1, chain_to=HORSESHOE),
    _card("Garrison", 1, CardColor.RED, cost=Cost(clay=1), shields=1, chain_to=SWORD),
    _card("Palisade", 1, CardColor.RED, cost=Cost(coins=2), shields=1, chain_to=FORT),
    _card("Scriptorium", 1, CardColor.GREEN, cost=Cost(coins=2), science=ScienceSymbol.QUILL_AND_INK, chain_to=BOOK),
    _card("Pharmacist", 1, CardColor.GREEN, cost=Cost(coins=2), science=ScienceSymbol.MORTAR_AND_PESTLE, chain_to=GEAR),
    _card("Theater", 1, CardColor.BLUE, victory_points=3, chain_to=MASK),
    _card("Altar", 1, CardColor.BLUE, victory_points=3, chain_to=MOON),
    _card("Baths", 1, CardColor.BLUE, cost=Cost(stone=1), victory_points=3, chain_to=WATER_DROP),
    _card("Tavern", 1, CardColor.YELLOW, chain_to=JUG, effects=(_e(EffectKind.IMMEDIATE_COINS, 4),)),
)


AGE_II_CARDS = (
    _card("Sawmill", 2, CardColor.BROWN, cost=Cost(coins=2), fixed_production=(Resource.WOOD, Resource.WOOD)),
    _card("Brickyard", 2, CardColor.BROWN, cost=Cost(coins=2), fixed_production=(Resource.CLAY, Resource.CLAY)),
    _card("Shelf Quarry", 2, CardColor.BROWN, cost=Cost(coins=2), fixed_production=(Resource.STONE, Resource.STONE)),
    _card("Glass-Blower", 2, CardColor.GREY, fixed_production=(Resource.GLASS,), source_aliases=("Glassblower",)),
    _card("Drying Room", 2, CardColor.GREY, fixed_production=(Resource.PAPYRUS,)),
    _card("Walls", 2, CardColor.RED, cost=Cost(stone=2), shields=2),
    _card("Forum", 2, CardColor.YELLOW, cost=Cost(coins=3, clay=1), choice_production=(Resource.GLASS, Resource.PAPYRUS)),
    _card("Caravansery", 2, CardColor.YELLOW, cost=Cost(coins=2, glass=1, papyrus=1), choice_production=(Resource.WOOD, Resource.CLAY, Resource.STONE)),
    _card("Customs House", 2, CardColor.YELLOW, cost=Cost(coins=4), trade_discount=frozenset({Resource.GLASS, Resource.PAPYRUS})),
    _card("Courthouse", 2, CardColor.BLUE, cost=Cost(wood=2, glass=1), victory_points=5),
    _card("Horse Breeders", 2, CardColor.RED, cost=Cost(wood=1, clay=1), shields=1, chain_from=HORSESHOE),
    _card("Barracks", 2, CardColor.RED, cost=Cost(coins=3), shields=1, chain_from=SWORD),
    _card("Archery Range", 2, CardColor.RED, cost=Cost(wood=1, stone=1, papyrus=1), shields=2, chain_to=TARGET),
    _card("Parade Ground", 2, CardColor.RED, cost=Cost(clay=2, glass=1), shields=2, chain_to=HELMET),
    _card("Library", 2, CardColor.GREEN, cost=Cost(wood=1, stone=1, glass=1), victory_points=2, science=ScienceSymbol.QUILL_AND_INK, chain_from=BOOK),
    _card("Dispensary", 2, CardColor.GREEN, cost=Cost(clay=2, stone=1), victory_points=2, science=ScienceSymbol.MORTAR_AND_PESTLE, chain_from=GEAR),
    _card("School", 2, CardColor.GREEN, cost=Cost(wood=1, papyrus=2), victory_points=1, science=ScienceSymbol.WHEEL, chain_to=LYRE),
    _card("Laboratory", 2, CardColor.GREEN, cost=Cost(wood=1, glass=2), victory_points=1, science=ScienceSymbol.SET_SQUARE, chain_to=LAMP),
    _card("Statue", 2, CardColor.BLUE, cost=Cost(clay=2), victory_points=4, chain_from=MASK, chain_to=COLUMN),
    _card("Temple", 2, CardColor.BLUE, cost=Cost(wood=1, papyrus=1), victory_points=4, chain_from=MOON, chain_to=SUN),
    _card("Aqueduct", 2, CardColor.BLUE, cost=Cost(stone=3), victory_points=5, chain_from=WATER_DROP),
    _card("Rostrum", 2, CardColor.BLUE, cost=Cost(wood=1, stone=1), victory_points=4, chain_to=BUILDING),
    _card("Brewery", 2, CardColor.YELLOW, chain_to=BARREL, effects=(_e(EffectKind.IMMEDIATE_COINS, 6),)),
)


AGE_III_CARDS = (
    _card("Arsenal", 3, CardColor.RED, cost=Cost(wood=2, clay=3), shields=3),
    _card("Pretorium", 3, CardColor.RED, cost=Cost(coins=8), shields=3, source_aliases=("Praetorium",)),
    _card("Academy", 3, CardColor.GREEN, cost=Cost(wood=1, stone=1, glass=2), victory_points=3, science=ScienceSymbol.SUNDIAL),
    _card("Study", 3, CardColor.GREEN, cost=Cost(wood=2, glass=1, papyrus=1), victory_points=3, science=ScienceSymbol.SUNDIAL),
    _card("Chamber of Commerce", 3, CardColor.YELLOW, cost=Cost(papyrus=2), victory_points=3, effects=(_e(EffectKind.COINS_PER_OWN_COLOR, 3, CardColor.GREY),)),
    _card("Port", 3, CardColor.YELLOW, cost=Cost(wood=1, glass=1, papyrus=1), victory_points=3, effects=(_e(EffectKind.COINS_PER_OWN_COLOR, 2, CardColor.BROWN),)),
    _card("Armory", 3, CardColor.YELLOW, cost=Cost(stone=2, glass=1), victory_points=3, effects=(_e(EffectKind.COINS_PER_OWN_COLOR, 1, CardColor.RED),)),
    _card("Palace", 3, CardColor.BLUE, cost=Cost(wood=1, clay=1, stone=1, glass=2), victory_points=7),
    _card("Town Hall", 3, CardColor.BLUE, cost=Cost(wood=2, stone=3), victory_points=7),
    _card("Obelisk", 3, CardColor.BLUE, cost=Cost(stone=2, glass=1), victory_points=5),
    _card("Fortifications", 3, CardColor.RED, cost=Cost(clay=1, stone=2, papyrus=1), shields=2, chain_from=FORT),
    _card("Siege Workshop", 3, CardColor.RED, cost=Cost(wood=3, glass=1), shields=2, chain_from=TARGET),
    _card("Circus", 3, CardColor.RED, cost=Cost(clay=2, stone=2), shields=2, chain_from=HELMET),
    _card("University", 3, CardColor.GREEN, cost=Cost(clay=1, glass=1, papyrus=1), victory_points=2, science=ScienceSymbol.ARMILLARY_SPHERE, chain_from=LYRE),
    _card("Observatory", 3, CardColor.GREEN, cost=Cost(stone=1, papyrus=2), victory_points=2, science=ScienceSymbol.ARMILLARY_SPHERE, chain_from=LAMP),
    _card("Gardens", 3, CardColor.BLUE, cost=Cost(wood=2, clay=2), victory_points=6, chain_from=COLUMN),
    _card("Pantheon", 3, CardColor.BLUE, cost=Cost(wood=1, clay=1, papyrus=2), victory_points=6, chain_from=SUN),
    _card("Senate", 3, CardColor.BLUE, cost=Cost(clay=2, stone=1, papyrus=1), victory_points=5, chain_from=BUILDING),
    _card("Lighthouse", 3, CardColor.YELLOW, cost=Cost(clay=2, glass=1), victory_points=3, chain_from=JUG, effects=(_e(EffectKind.COINS_PER_OWN_COLOR, 1, CardColor.YELLOW),)),
    _card("Arena", 3, CardColor.YELLOW, cost=Cost(wood=1, clay=1, stone=1), victory_points=3, chain_from=BARREL, effects=(_e(EffectKind.COINS_PER_OWN_WONDER, 2),)),
)


GUILD_CARDS = (
    _card("Merchants Guild", 3, CardColor.PURPLE, cost=Cost(wood=1, clay=1, glass=1, papyrus=1), effects=(_e(EffectKind.COINS_PER_MOST_COLOR, 1, CardColor.YELLOW), _e(EffectKind.VP_PER_MOST_COLOR, 1, CardColor.YELLOW)), source_aliases=("Traders Guild",)),
    _card("Shipowners Guild", 3, CardColor.PURPLE, cost=Cost(clay=1, stone=1, glass=1, papyrus=1), effects=(_e(EffectKind.COINS_PER_MOST_BROWN_GREY), _e(EffectKind.VP_PER_MOST_BROWN_GREY))),
    _card("Builders Guild", 3, CardColor.PURPLE, cost=Cost(wood=1, clay=1, stone=2, glass=1), effects=(_e(EffectKind.VP_PER_MOST_WONDER, 2),)),
    _card("Magistrates Guild", 3, CardColor.PURPLE, cost=Cost(wood=2, clay=1, papyrus=1), effects=(_e(EffectKind.COINS_PER_MOST_COLOR, 1, CardColor.BLUE), _e(EffectKind.VP_PER_MOST_COLOR, 1, CardColor.BLUE)),),
    _card("Scientists Guild", 3, CardColor.PURPLE, cost=Cost(wood=2, clay=2), effects=(_e(EffectKind.COINS_PER_MOST_COLOR, 1, CardColor.GREEN), _e(EffectKind.VP_PER_MOST_COLOR, 1, CardColor.GREEN)),),
    _card("Moneylenders Guild", 3, CardColor.PURPLE, cost=Cost(wood=2, stone=2), effects=(_e(EffectKind.VP_PER_RICHEST_COIN_SET),)),
    _card("Tacticians Guild", 3, CardColor.PURPLE, cost=Cost(clay=1, stone=2, papyrus=1), effects=(_e(EffectKind.COINS_PER_MOST_COLOR, 1, CardColor.RED), _e(EffectKind.VP_PER_MOST_COLOR, 1, CardColor.RED)),),
)


ALL_AGE_CARDS = AGE_I_CARDS + AGE_II_CARDS + AGE_III_CARDS
ALL_BUILDING_CARDS = ALL_AGE_CARDS + GUILD_CARDS
CARDS_BY_NAME = {card.name: card for card in ALL_BUILDING_CARDS}

# Canonical integer ids (CODEC_SPEC.md §1): position in the data tuples above.
CARD_IDS = {card.name: index for index, card in enumerate(ALL_BUILDING_CARDS)}

_BACK_BY_AGE = {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}


def back_type_of(card_name: str) -> BackType:
    card = CARDS_BY_NAME[card_name]
    if card.color is CardColor.PURPLE:
        return BackType.GUILD
    return _BACK_BY_AGE[card.age]


PROGRESS_TOKENS = (
    ProgressData("Agriculture", victory_points=4, effects=(_e(EffectKind.IMMEDIATE_COINS, 6),)),
    ProgressData("Architecture", effects=(_e(EffectKind.FUTURE_WONDER_RESOURCE_DISCOUNT, 2),)),
    ProgressData("Economy", effects=(_e(EffectKind.RECEIVE_OPPONENT_TRADE_SPEND),)),
    ProgressData("Law", science=ScienceSymbol.LAW),
    ProgressData("Masonry", effects=(_e(EffectKind.FUTURE_BLUE_RESOURCE_DISCOUNT, 2),)),
    ProgressData("Mathematics", effects=(_e(EffectKind.VP_PER_PROGRESS, 3),)),
    ProgressData("Philosophy", victory_points=7),
    ProgressData("Strategy", effects=(_e(EffectKind.FUTURE_RED_EXTRA_SHIELD),)),
    ProgressData("Theology", effects=(_e(EffectKind.FUTURE_WONDER_PLAY_AGAIN),)),
    ProgressData("Urbanism", effects=(_e(EffectKind.IMMEDIATE_COINS, 6), _e(EffectKind.COINS_PER_CHAIN_BUILD, 4))),
)
PROGRESS_BY_NAME = {token.name: token for token in PROGRESS_TOKENS}
PROGRESS_IDS = {token.name: index for index, token in enumerate(PROGRESS_TOKENS)}


WONDERS = (
    WonderData("The Appian Way", Cost(clay=2, stone=2, papyrus=1), victory_points=3, effects=(_e(EffectKind.IMMEDIATE_COINS, 3), _e(EffectKind.OPPONENT_LOSES_COINS, 3), _e(EffectKind.PLAY_AGAIN)), cost_source="rulebook page 6 setup photograph"),
    WonderData("Circus Maximus", Cost(wood=1, stone=2, glass=1), victory_points=3, shields=1, effects=(_e(EffectKind.DESTROY_OPPONENT_GREY),), cost_source="rulebook page 6 setup photograph"),
    WonderData("The Colossus", Cost(clay=3, glass=1), victory_points=3, shields=2, cost_source="rulebook page 11 construction example"),
    WonderData("The Great Library", Cost(wood=3, glass=1, papyrus=1), victory_points=4, effects=(_e(EffectKind.CHOOSE_UNUSED_PROGRESS, 3),), cost_source="component photograph; see TRANSCRIPTION_NOTES.md"),
    WonderData("The Great Lighthouse", Cost(wood=1, stone=1, papyrus=2), victory_points=4, choice_production=(Resource.WOOD, Resource.CLAY, Resource.STONE), cost_source="rulebook page 6 setup photograph"),
    WonderData("The Hanging Gardens", Cost(wood=2, glass=1, papyrus=1), victory_points=3, effects=(_e(EffectKind.IMMEDIATE_COINS, 6), _e(EffectKind.PLAY_AGAIN)), cost_source="component photograph; see TRANSCRIPTION_NOTES.md"),
    WonderData("The Mausoleum", Cost(clay=2, glass=2, papyrus=1), victory_points=2, effects=(_e(EffectKind.BUILD_FROM_DISCARD_FREE),), cost_source="component photograph; see TRANSCRIPTION_NOTES.md"),
    WonderData("Piraeus", Cost(wood=2, clay=1, stone=1), victory_points=2, choice_production=(Resource.GLASS, Resource.PAPYRUS), effects=(_e(EffectKind.PLAY_AGAIN),), cost_source="rulebook page 6 setup photograph"),
    WonderData("The Pyramids", Cost(stone=3, papyrus=1), victory_points=9, cost_source="rulebook page 4 component diagram"),
    WonderData("The Sphinx", Cost(clay=1, stone=1, glass=2), victory_points=6, effects=(_e(EffectKind.PLAY_AGAIN),), cost_source="component photograph; see TRANSCRIPTION_NOTES.md"),
    WonderData("The Statue of Zeus", Cost(wood=1, clay=1, stone=1, papyrus=2), victory_points=3, shields=1, effects=(_e(EffectKind.DESTROY_OPPONENT_BROWN),), cost_source="rulebook page 6 setup photograph"),
    WonderData("The Temple of Artemis", Cost(wood=1, stone=1, glass=1, papyrus=1), effects=(_e(EffectKind.IMMEDIATE_COINS, 12), _e(EffectKind.PLAY_AGAIN)), cost_source="rulebook page 6 setup photograph"),
)
WONDERS_BY_NAME = {wonder.name: wonder for wonder in WONDERS}
WONDER_IDS = {wonder.name: index for index, wonder in enumerate(WONDERS)}


def _rows(spec: tuple[tuple[tuple[int, ...], bool], ...]) -> tuple[TableauSlot, ...]:
    return tuple(
        TableauSlot(row=row, x=x, face_up=face_up)
        for row, (positions, face_up) in enumerate(spec)
        for x in positions
    )


TABLEAU_LAYOUTS = {
    1: _rows((((5, 7), True), ((4, 6, 8), False), ((3, 5, 7, 9), True), ((2, 4, 6, 8, 10), False), ((1, 3, 5, 7, 9, 11), True))),
    2: _rows((((1, 3, 5, 7, 9, 11), True), ((2, 4, 6, 8, 10), False), ((3, 5, 7, 9), True), ((4, 6, 8), False), ((5, 7), True))),
    3: _rows((((5, 7), True), ((4, 6, 8), False), ((3, 5, 7, 9), True), ((4, 8), False), ((3, 5, 7, 9), True), ((4, 6, 8), False), ((5, 7), True))),
}


def covering_slots(layout: tuple[TableauSlot, ...], slot: TableauSlot) -> tuple[TableauSlot, ...]:
    """Return cards in the next printed row that overlap and cover ``slot``."""

    return tuple(
        candidate
        for candidate in layout
        if candidate.row == slot.row + 1 and abs(candidate.x - slot.x) == 1
    )
