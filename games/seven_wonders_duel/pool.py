"""UnseenPool: the single source of truth for hidden-card structure.

CODEC_SPEC.md §4.1/§4.3. Three consumers share this module so their views can
never diverge: encoder unseen-pool features, closed-loop chance enumeration,
and open-loop determinization (``resample_hidden``).

Pools are computed from a ``PlayerObservation`` only — never from hidden
``GameState`` fields. Hidden information is symmetric, so the viewer does not
matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from .data import (
    ALL_BUILDING_CARDS,
    CARD_IDS,
    GUILD_CARDS,
    PROGRESS_IDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    WONDER_IDS,
    WONDERS,
    BackType,
    back_type_of,
)
from .game import GameState, Phase, PlayerObservation, TableauState

BACK_UNIVERSES: dict[BackType, frozenset[str]] = {
    back: frozenset(
        card.name for card in ALL_BUILDING_CARDS if back_type_of(card.name) is back
    )
    for back in BackType
}
_ALL_WONDER_NAMES = frozenset(wonder.name for wonder in WONDERS)
_ALL_PROGRESS_NAMES = frozenset(token.name for token in PROGRESS_TOKENS)


@dataclass(frozen=True, slots=True)
class UnseenPool:
    """Per-back-type unseen card sets plus the wonder and progress analogues.

    ``cards[back]``: card names not visible anywhere (face-up tableau, either
    city, discard, burials). Future-age pools are full universes minus nothing —
    they exist from the start (encoder §5.6 uses them for cross-age features).

    ``wonders``: wonders not yet seen in an offer or a city. Only meaningful
    during draft round 0 (the WONDER_GROUP_REVEAL pool); afterwards it is the
    4 never-revealed box wonders, which no longer matter.

    ``offboard_progress``: the 5 tokens not on the board and not owned — the
    GREAT_LIBRARY_DRAW pool. Deducible by both players (token set is public).
    """

    cards: dict[BackType, frozenset[str]]
    wonders: frozenset[str]
    offboard_progress: frozenset[str]


def visible_card_names(observation: PlayerObservation) -> frozenset[str]:
    visible: set[str] = set()
    for card in observation.tableau:
        if card.card_name is not None:
            visible.add(card.card_name)
    for city in observation.cities:
        visible.update(city.buildings)
    visible.update(observation.discard_pile)
    visible.update(observation.buried_cards)
    return frozenset(visible)


def unseen_pool(observation: PlayerObservation) -> UnseenPool:
    visible = visible_card_names(observation)
    cards = {back: universe - visible for back, universe in BACK_UNIVERSES.items()}

    seen_wonders = set(observation.wonder_offer)
    for city in observation.cities:
        seen_wonders.update(city.wonders)
    wonders = _ALL_WONDER_NAMES - seen_wonders

    owned_progress: set[str] = set(observation.available_progress_tokens)
    for city in observation.cities:
        owned_progress.update(city.progress_tokens)
    offboard = _ALL_PROGRESS_NAMES - owned_progress

    return UnseenPool(cards=cards, wonders=wonders, offboard_progress=offboard)


def enumerate_card_reveal(
    pool: UnseenPool, back: BackType
) -> tuple[tuple[str, float], ...]:
    """CARD_REVEAL outcomes for a slot of the given back, canonical id order."""

    names = sorted(pool.cards[back], key=CARD_IDS.__getitem__)
    if not names:
        raise ValueError(f"empty unseen pool for back {back.value}")
    probability = 1.0 / len(names)
    return tuple((name, probability) for name in names)


def enumerate_great_library(
    pool: UnseenPool,
) -> tuple[tuple[tuple[str, ...], float], ...]:
    """GREAT_LIBRARY_DRAW outcomes: 3-subsets of the off-board tokens in
    lexicographic canonical order (CODEC_SPEC.md §4.2) — 10 when 5 remain."""

    names = sorted(pool.offboard_progress, key=PROGRESS_IDS.__getitem__)
    subsets = tuple(combinations(names, 3))
    if not subsets:
        raise ValueError("no Great Library outcomes available")
    probability = 1.0 / len(subsets)
    return tuple((subset, probability) for subset in subsets)


def enumerate_wonder_flip(pool: UnseenPool) -> tuple[tuple[tuple[str, ...], float], ...]:
    """WONDER_GROUP_REVEAL outcomes: 4-subsets of the unseen wonders in
    lexicographic canonical order — C(8,4)=70 at the standard flip point."""

    names = sorted(pool.wonders, key=WONDER_IDS.__getitem__)
    subsets = tuple(combinations(names, 4))
    if not subsets:
        raise ValueError("no wonder-flip outcomes available")
    probability = 1.0 / len(subsets)
    return tuple((subset, probability) for subset in subsets)


def resample_hidden(state: GameState, rng: random.Random) -> None:
    """Re-randomize every hidden assignment in place, preserving the visible
    projection exactly (the determinizer of CODEC_SPEC.md §4.3, formulated on a
    state clone rather than observation reconstruction — search always starts
    from a clone, and only hidden entities are moved).

    Redistributes: unrevealed current-tableau slots + removed cards (within
    back type), guild selection, future-age decks, and the second wonder group
    during draft round 0. Decouples ``state.rng`` from the original stream.
    """

    if state.phase is Phase.WONDER_DRAFT:
        if state.wonder_round == 0:
            hidden_wonders = list(state.wonder_groups[1]) + list(state.unused_wonders)
            rng.shuffle(hidden_wonders)
            state.wonder_groups = (state.wonder_groups[0], tuple(hidden_wonders[:4]))
            state.unused_wonders = tuple(hidden_wonders[4:])
        # The Age I structure is dealt at setup but nothing is observable until
        # the draft ends, so the whole age is hidden and fully re-dealt.
        _resample_future_age(state, 1, rng)
        state.tableau = TableauState.from_deck(1, state.age_decks[1])
    else:
        _resample_current_tableau(state, rng)

    for age in (2, 3):
        if age > state.age or (age == state.age and state.phase is Phase.WONDER_DRAFT):
            _resample_future_age(state, age, rng)

    state.rng = random.Random(rng.getrandbits(64))


def _resample_current_tableau(state: GameState, rng: random.Random) -> None:
    hidden = [
        card
        for card in state.tableau.cards.values()
        if card.present and not card.revealed
    ]
    if state.age in (1, 2):
        _shuffle_hidden_class(state, hidden, state.age, rng)
        return
    age_hidden = [c for c in hidden if back_type_of(c.card_name) is BackType.AGE_III]
    guild_hidden = [c for c in hidden if back_type_of(c.card_name) is BackType.GUILD]
    _shuffle_hidden_class(state, age_hidden, 3, rng)

    guild_pool = [card.card_name for card in guild_hidden] + list(state.unused_guilds)
    rng.shuffle(guild_pool)
    for card, name in zip(guild_hidden, guild_pool, strict=False):
        card.card_name = name
    state.unused_guilds = tuple(guild_pool[len(guild_hidden):])
    state.selected_guilds = tuple(
        card.name for card in GUILD_CARDS if card.name not in state.unused_guilds
    )


def _shuffle_hidden_class(
    state: GameState, hidden_cards: list, age: int, rng: random.Random
) -> None:
    pool = [card.card_name for card in hidden_cards] + list(
        state.removed_age_cards[age]
    )
    rng.shuffle(pool)
    for card, name in zip(hidden_cards, pool, strict=False):
        card.card_name = name
    state.removed_age_cards[age] = tuple(pool[len(hidden_cards):])


def _resample_future_age(state: GameState, age: int, rng: random.Random) -> None:
    names = sorted(BACK_UNIVERSES[_AGE_BACKS[age]], key=CARD_IDS.__getitem__)
    rng.shuffle(names)
    state.removed_age_cards[age] = tuple(names[:3])
    deck = names[3:]
    if age == 3:
        guilds = sorted(BACK_UNIVERSES[BackType.GUILD], key=CARD_IDS.__getitem__)
        rng.shuffle(guilds)
        state.selected_guilds = tuple(guilds[:3])
        state.unused_guilds = tuple(guilds[3:])
        deck = deck + list(state.selected_guilds)
        rng.shuffle(deck)
    state.age_decks[age] = tuple(deck)
    layout_size = len(TABLEAU_LAYOUTS[age])
    if len(state.age_decks[age]) != layout_size:
        raise AssertionError(f"Age {age} deck must hold {layout_size} cards")


_AGE_BACKS = {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}
