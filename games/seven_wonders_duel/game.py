"""Seeded setup and observation-safe state for the base game.

This module deliberately stops before resolving the three uses of an Age card.
It owns setup randomness, Wonder drafting, tableau accessibility/reveals, and the
boundary between complete simulator state and information visible to a player.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import random

from .data import (
    AGE_I_CARDS,
    AGE_II_CARDS,
    AGE_III_CARDS,
    GUILD_CARDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    WONDERS,
    BackType,
    ScienceSymbol,
    TableauSlot,
    back_type_of,
    covering_slots,
)
from .rules import STARTING_COINS


SlotId = tuple[int, int]


class ChanceKind(str, Enum):
    """First-class chance events (CODEC_SPEC.md §4.2)."""

    CARD_REVEAL = "card_reveal"
    GREAT_LIBRARY_DRAW = "great_library_draw"
    WONDER_GROUP_REVEAL = "wonder_group_reveal"
    AGE_DEAL = "age_deal"


class HiddenInformationError(RuntimeError):
    """Raised when resolving chance would read hidden state behind the search barrier."""


@dataclass(frozen=True, slots=True)
class ResolvedChance:
    """One chance event that fired during an action, with the outcome used.

    ``context``: CARD_REVEAL → (slot_id, BackType); AGE_DEAL → (age,); else ().
    ``outcome``: CARD_REVEAL → card name; GREAT_LIBRARY_DRAW → 3 token names in
    canonical id order; WONDER_GROUP_REVEAL → 4 wonder names in canonical id
    order; AGE_DEAL → the full dealt tuple in layout order.
    """

    kind: ChanceKind
    context: tuple
    outcome: str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepResult:
    events: tuple[ResolvedChance, ...] = ()


class Phase(str, Enum):
    WONDER_DRAFT = "wonder_draft"
    PLAY_AGE = "play_age"
    CHOOSE_NEXT_START_PLAYER = "choose_next_start_player"
    COMPLETE = "complete"


class VictoryType(str, Enum):
    MILITARY = "military"
    SCIENTIFIC = "scientific"
    CIVILIAN = "civilian"
    SHARED_CIVILIAN = "shared_civilian"


class PendingChoiceKind(str, Enum):
    DESTROY_OPPONENT_BROWN = "destroy_opponent_brown"
    DESTROY_OPPONENT_GREY = "destroy_opponent_grey"
    BUILD_FROM_DISCARD_FREE = "build_from_discard_free"
    CHOOSE_UNUSED_PROGRESS = "choose_unused_progress"
    CHOOSE_AVAILABLE_PROGRESS = "choose_available_progress"


@dataclass(frozen=True, slots=True)
class PendingChoice:
    kind: PendingChoiceKind
    player: int
    options: tuple[str, ...]
    consume_all_options: bool = False


@dataclass(slots=True)
class CityState:
    coins: int = STARTING_COINS
    wonders: list[str] = field(default_factory=list)
    built_wonders: list[str] = field(default_factory=list)
    buildings: list[str] = field(default_factory=list)
    progress_tokens: list[str] = field(default_factory=list)
    claimed_science_pairs: set[ScienceSymbol] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PublicCity:
    coins: int
    wonders: tuple[str, ...]
    built_wonders: tuple[str, ...]
    buildings: tuple[str, ...]
    progress_tokens: tuple[str, ...]
    claimed_science_pairs: frozenset[ScienceSymbol]


@dataclass(slots=True)
class TableauCard:
    slot: TableauSlot
    card_name: str
    revealed: bool
    present: bool = True

    @property
    def slot_id(self) -> SlotId:
        return (self.slot.row, self.slot.x)


@dataclass(frozen=True, slots=True)
class PublicTableauCard:
    slot_id: SlotId
    present: bool
    revealed: bool
    accessible: bool
    card_name: str | None
    back: BackType | None


@dataclass(slots=True)
class TableauState:
    age: int
    cards: dict[SlotId, TableauCard]

    @classmethod
    def from_deck(cls, age: int, card_names: tuple[str, ...]) -> "TableauState":
        layout = TABLEAU_LAYOUTS[age]
        if len(card_names) != len(layout):
            raise ValueError(f"Age {age} tableau requires {len(layout)} cards")
        cards = {
            (slot.row, slot.x): TableauCard(
                slot=slot,
                card_name=name,
                revealed=slot.face_up,
            )
            for slot, name in zip(layout, card_names, strict=True)
        }
        return cls(age=age, cards=cards)

    def is_accessible(self, slot_id: SlotId) -> bool:
        card = self.cards.get(slot_id)
        if card is None or not card.present:
            return False
        layout = TABLEAU_LAYOUTS[self.age]
        return not any(
            self.cards[(coverer.row, coverer.x)].present
            for coverer in covering_slots(layout, card.slot)
        )

    def accessible_slot_ids(self) -> tuple[SlotId, ...]:
        return tuple(slot_id for slot_id in self.cards if self.is_accessible(slot_id))

    def take_accessible(self, slot_id: SlotId) -> tuple[str, tuple[SlotId, ...]]:
        """Remove one accessible card; report newly accessible face-down slots.

        Newly accessible cards are NOT revealed here: revelation is a chance
        event owned by ``engine._process_reveals``, which resolves the returned
        slots sequentially in the canonical (row, x) ascending order of
        CODEC_SPEC.md §4.2 and calls :meth:`reveal` on each. Keeping siblings
        unrevealed until their own event fires is what lets a supplied outcome
        swap with a card locked in a simultaneously exposed slot.
        """

        if not self.is_accessible(slot_id):
            raise ValueError(f"tableau slot is not accessible: {slot_id}")
        card = self.cards[slot_id]
        if not card.revealed:
            raise AssertionError("an accessible card must have been revealed")
        card.present = False
        newly_accessible = tuple(
            sorted(
                candidate.slot_id
                for candidate in self.cards.values()
                if candidate.present
                and not candidate.revealed
                and self.is_accessible(candidate.slot_id)
            )
        )
        return card.card_name, newly_accessible

    def reveal(self, slot_id: SlotId) -> None:
        card = self.cards[slot_id]
        if not card.present:
            raise ValueError(f"cannot reveal an absent card: {slot_id}")
        card.revealed = True

    def public_cards(self) -> tuple[PublicTableauCard, ...]:
        return tuple(
            PublicTableauCard(
                slot_id=card.slot_id,
                present=card.present,
                revealed=card.revealed if card.present else False,
                accessible=self.is_accessible(card.slot_id),
                card_name=card.card_name if card.present and card.revealed else None,
                back=back_type_of(card.card_name) if card.present else None,
            )
            for card in self.cards.values()
        )


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    viewer: int
    phase: Phase
    active_player: int
    age: int
    cities: tuple[PublicCity, PublicCity]
    available_progress_tokens: tuple[str, ...]
    wonder_offer: tuple[str, ...]
    tableau: tuple[PublicTableauCard, ...]
    discard_pile: tuple[str, ...]
    buried_cards: tuple[str, ...]
    wonder_burials: tuple[tuple[str, str], ...]
    retired_wonders: frozenset[str]
    pending_choice: PendingChoice | None
    pending_extra_turn: bool
    pending_shields: int
    conflict_position: int
    military_tokens_remaining: tuple[tuple[int, int], ...]
    winner: int | None
    victory_type: VictoryType | None
    final_scores: tuple[int, int] | None


@dataclass(slots=True)
class GameState:
    """Complete simulator state, including information hidden from both players."""

    seed: int
    first_player: int
    phase: Phase
    active_player: int
    age: int
    cities: tuple[CityState, CityState]
    available_progress_tokens: tuple[str, ...]
    unused_progress_tokens: tuple[str, ...]
    wonder_groups: tuple[tuple[str, ...], tuple[str, ...]]
    unused_wonders: tuple[str, ...]
    wonder_offer: list[str]
    wonder_round: int
    wonder_pick_index: int
    age_decks: dict[int, tuple[str, ...]]
    removed_age_cards: dict[int, tuple[str, ...]]
    selected_guilds: tuple[str, ...]
    unused_guilds: tuple[str, ...]
    tableau: TableauState
    discard_pile: list[str]
    buried_cards: list[str]
    retired_wonders: set[str]
    pending_choice: PendingChoice | None
    pending_extra_turn: bool
    pending_shields: int
    conflict_position: int
    military_tokens_remaining: dict[int, int]
    winner: int | None
    victory_type: VictoryType | None
    final_scores: tuple[int, int] | None
    rng: random.Random = field(repr=False)
    wonder_burials: dict[str, str] = field(default_factory=dict)
    search_barrier: bool = False
    """When True (set on clones handed to search), resolving any chance event
    without an explicitly supplied outcome raises HiddenInformationError instead
    of silently reading the locked deal / RNG stream (CODEC_SPEC.md §4.3)."""

    @classmethod
    def new(cls, seed: int, first_player: int = 0) -> "GameState":
        if first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")
        rng = random.Random(seed)

        progress = [token.name for token in PROGRESS_TOKENS]
        rng.shuffle(progress)

        wonders = [wonder.name for wonder in WONDERS]
        rng.shuffle(wonders)
        wonder_groups = (tuple(wonders[:4]), tuple(wonders[4:8]))

        age_decks: dict[int, tuple[str, ...]] = {}
        removed_age_cards: dict[int, tuple[str, ...]] = {}
        for age, definitions in ((1, AGE_I_CARDS), (2, AGE_II_CARDS)):
            names = [card.name for card in definitions]
            rng.shuffle(names)
            removed_age_cards[age] = tuple(names[:3])
            age_decks[age] = tuple(names[3:])

        age_three = [card.name for card in AGE_III_CARDS]
        rng.shuffle(age_three)
        removed_age_cards[3] = tuple(age_three[:3])
        age_three = age_three[3:]

        guilds = [card.name for card in GUILD_CARDS]
        rng.shuffle(guilds)
        selected_guilds = tuple(guilds[:3])
        age_three.extend(selected_guilds)
        rng.shuffle(age_three)
        age_decks[3] = tuple(age_three)

        return cls(
            seed=seed,
            first_player=first_player,
            phase=Phase.WONDER_DRAFT,
            active_player=first_player,
            age=1,
            cities=(CityState(), CityState()),
            available_progress_tokens=tuple(progress[:5]),
            unused_progress_tokens=tuple(progress[5:]),
            wonder_groups=wonder_groups,
            unused_wonders=tuple(wonders[8:]),
            wonder_offer=list(wonder_groups[0]),
            wonder_round=0,
            wonder_pick_index=0,
            age_decks=age_decks,
            removed_age_cards=removed_age_cards,
            selected_guilds=selected_guilds,
            unused_guilds=tuple(guilds[3:]),
            tableau=TableauState.from_deck(1, age_decks[1]),
            discard_pile=[],
            buried_cards=[],
            retired_wonders=set(),
            pending_choice=None,
            pending_extra_turn=False,
            pending_shields=0,
            conflict_position=0,
            military_tokens_remaining={-7: 5, -4: 2, 4: 2, 7: 5},
            winner=None,
            victory_type=None,
            final_scores=None,
            rng=rng,
        )

    def _draft_order(self, round_index: int) -> tuple[int, int, int, int]:
        first = self.first_player if round_index == 0 else 1 - self.first_player
        second = 1 - first
        return (first, second, second, first)

    def legal_wonder_choices(self) -> tuple[str, ...]:
        if self.phase is not Phase.WONDER_DRAFT:
            return ()
        return tuple(self.wonder_offer)

    def pick_wonder(self, wonder_name: str) -> bool:
        """Apply one draft pick. Returns True when the pick flipped the second
        Wonder group face-up (a WONDER_GROUP_REVEAL chance event — the caller in
        engine.apply_action records/overrides it)."""

        if self.phase is not Phase.WONDER_DRAFT:
            raise ValueError("Wonder selection is complete")
        if wonder_name not in self.wonder_offer:
            raise ValueError(f"Wonder is not in the current offer: {wonder_name}")

        expected_player = self._draft_order(self.wonder_round)[self.wonder_pick_index]
        if self.active_player != expected_player:
            raise AssertionError("active player does not match Wonder draft order")
        self.cities[self.active_player].wonders.append(wonder_name)
        self.wonder_offer.remove(wonder_name)
        self.wonder_pick_index += 1

        if self.wonder_pick_index < 4:
            self.active_player = self._draft_order(self.wonder_round)[self.wonder_pick_index]
            return False

        if self.wonder_round == 0:
            self.wonder_round = 1
            self.wonder_pick_index = 0
            self.wonder_offer = list(self.wonder_groups[1])
            self.active_player = self._draft_order(1)[0]
            return True

        self.wonder_offer = []
        self.phase = Phase.PLAY_AGE
        self.active_player = self.first_player
        return False

    def observation(self, viewer: int) -> PlayerObservation:
        if viewer not in (0, 1):
            raise ValueError("viewer must be 0 or 1")
        cities = tuple(
            PublicCity(
                coins=city.coins,
                wonders=tuple(city.wonders),
                built_wonders=tuple(city.built_wonders),
                buildings=tuple(city.buildings),
                progress_tokens=tuple(city.progress_tokens),
                claimed_science_pairs=frozenset(city.claimed_science_pairs),
            )
            for city in self.cities
        )
        tableau = self.tableau.public_cards() if self.phase is not Phase.WONDER_DRAFT else ()
        return PlayerObservation(
            viewer=viewer,
            phase=self.phase,
            active_player=self.active_player,
            age=self.age,
            cities=cities,  # type: ignore[arg-type]
            available_progress_tokens=self.available_progress_tokens,
            wonder_offer=tuple(self.wonder_offer),
            tableau=tableau,
            discard_pile=tuple(self.discard_pile),
            buried_cards=tuple(self.buried_cards),
            wonder_burials=tuple(sorted(self.wonder_burials.items())),
            retired_wonders=frozenset(self.retired_wonders),
            pending_choice=self.pending_choice,
            pending_extra_turn=self.pending_extra_turn,
            pending_shields=self.pending_shields,
            conflict_position=self.conflict_position,
            military_tokens_remaining=tuple(sorted(self.military_tokens_remaining.items())),
            winner=self.winner,
            victory_type=self.victory_type,
            final_scores=self.final_scores,
        )

    def setup_fingerprint(self) -> tuple:
        """Stable complete-state summary for reproducibility tests and manifests."""

        return (
            self.first_player,
            self.available_progress_tokens,
            self.unused_progress_tokens,
            self.wonder_groups,
            self.unused_wonders,
            tuple((age, self.age_decks[age]) for age in (1, 2, 3)),
            tuple((age, self.removed_age_cards[age]) for age in (1, 2, 3)),
            self.selected_guilds,
            self.unused_guilds,
        )

    def clone(self) -> "GameState":
        """Return an independent copy, including the exact future RNG stream."""

        return copy.deepcopy(self)


def new_game(seed: int, first_player: int = 0) -> GameState:
    return GameState.new(seed=seed, first_player=first_player)
