"""Observation encoder: typed entity-token sequence (CODEC_SPEC.md §5).

Pure function of ``(PlayerObservation, UnseenPool)`` — never of hidden state.
Always actor-relative (§2): "my" is the player to act, and the encoding of a
mirrored state is bit-identical.

Economy features are computed by calling the engine's own pricing helpers on a
stub ``GameState`` populated from public observation fields only — zero logic
duplication, so encoder prices can never drift from engine prices. The private
imports below are deliberate for that reason.

The feature *semantics* are fixed by the spec; the offsets here are pinned by
``ENCODER_SIGNATURE`` (sha256 over the full schema), which golden tests assert
and the eventual Rust loader re-checks (§5.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import random

from .data import (
    ALL_BUILDING_CARDS,
    CARD_IDS,
    CARDS_BY_NAME,
    PROGRESS_BY_NAME,
    PROGRESS_IDS,
    PROGRESS_TOKENS,
    TABLEAU_LAYOUTS,
    WONDER_IDS,
    WONDERS,
    WONDERS_BY_NAME,
    BackType,
    CardColor,
    EffectKind,
    ScienceSymbol,
)
from .engine import (
    _choice_producers,
    _fixed_production,
    _opponent_trade_production,
    _trade_discounts,
    minimum_payment,
    score_player,
)
from .game import (
    CityState,
    GameState,
    PendingChoiceKind,
    Phase,
    PlayerObservation,
    TableauState,
)
from .pool import BACK_UNIVERSES, UnseenPool, unseen_pool
from .rules import Resource, discard_income

ENCODER_VERSION = "7wd-encoder-1"

_RESOURCES = tuple(Resource)
_SYMBOLS = tuple(ScienceSymbol)
_COLORS = tuple(CardColor)
_BACKS = tuple(BackType)
_BACK_ID = {back: index for index, back in enumerate(_BACKS)}
_CARD_NAMES = tuple(card.name for card in ALL_BUILDING_CARDS)
_WONDER_NAMES = tuple(wonder.name for wonder in WONDERS)
_PROGRESS_NAMES = tuple(token.name for token in PROGRESS_TOKENS)

_DECISIONS = (
    "wonder_draft",
    "main_turn",
    "destroy_brown",
    "destroy_grey",
    "build_from_discard",
    "great_library_pick",
    "board_progress_pick",
    "next_age_starter",
    "complete",
)
_PENDING_DECISION = {
    PendingChoiceKind.DESTROY_OPPONENT_BROWN: "destroy_brown",
    PendingChoiceKind.DESTROY_OPPONENT_GREY: "destroy_grey",
    PendingChoiceKind.BUILD_FROM_DISCARD_FREE: "build_from_discard",
    PendingChoiceKind.CHOOSE_UNUSED_PROGRESS: "great_library_pick",
    PendingChoiceKind.CHOOSE_AVAILABLE_PROGRESS: "board_progress_pick",
}


class TokenType(str, Enum):
    GLOBAL = "global"
    DRAFT_OFFER = "draft_offer"
    TABLEAU = "tableau"
    CITY_CARD = "city_card"
    WONDER = "wonder"
    PROGRESS = "progress"
    DISCARD = "discard"
    POOL = "pool"
    POOL_WONDER = "pool_wonder"


@dataclass(frozen=True, slots=True)
class Token:
    """One entity token: per-type entity id + per-type feature vector.

    ``aux_id`` is a second embedding slot (-1 when unused); today only WONDER
    uses it, for the buried card id.
    """

    type: TokenType
    entity_id: int
    aux_id: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Encoding:
    actor: int
    tokens: tuple[Token, ...]


# --- feature-name schema (fixes semantics; hashed into ENCODER_SIGNATURE) ---


def _per_player_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}coins",
        f"{prefix}coins_s",
        f"{prefix}sci_distinct",
        f"{prefix}sci_to_win",
        *(f"{prefix}sym_{s.value}" for s in _SYMBOLS),
        *(f"{prefix}color_{c.value}" for c in _COLORS),
        f"{prefix}unbuilt_wonders",
        f"{prefix}unbuilt_extra_turn_wonders",
        *(f"{prefix}prod_{r.value}" for r in _RESOURCES),
        *(f"{prefix}choice_prod_{r.value}" for r in _RESOURCES),
        *(f"{prefix}trade_price_{r.value}" for r in _RESOURCES),
        f"{prefix}discard_income",
        f"{prefix}score_military",
        f"{prefix}score_buildings",
        f"{prefix}score_wonders",
        f"{prefix}score_progress",
        f"{prefix}score_treasury",
        f"{prefix}score_total",
        f"{prefix}score_blue",
        f"{prefix}mil_shields_obtainable",
        f"{prefix}mil_win_feasible",
        f"{prefix}sci_missing_obtainable",
        f"{prefix}sci_win_feasible",
    )


GLOBAL_FEATURES = (
    *(f"decision_{d}" for d in _DECISIONS),
    "age_1",
    "age_2",
    "age_3",
    "cards_remaining",
    "cards_remaining_s",
    "face_down_remaining",
    "face_down_remaining_s",
    "military",
    "military_s",
    "dist_my_mil_win",
    "dist_my_mil_win_s",
    "dist_opp_mil_win",
    "dist_opp_mil_win_s",
    "my_next_token_dist",
    "my_next_token_dist_s",
    "my_next_token_penalty",
    "opp_next_token_dist",
    "opp_next_token_dist_s",
    "opp_next_token_penalty",
    "pending_shields",
    "pending_extra_turn",
    *_per_player_names("my_"),
    *_per_player_names("opp_"),
)

_TABLEAU_PER_PLAYER = (
    "affordable",
    "cost",
    "cost_s",
    "chain_free",
    "completes_pair",
    "gives_sixth_symbol",
    "eff_shields",
    "would_cross_token",
    "would_win_military",
)
TABLEAU_FEATURES = (
    "row",
    "row_s",
    "x",
    "x_s",
    "face_up_layer",
    "accessible",
    "coverers",
    "covers_hidden",
    *(f"my_{name}" for name in _TABLEAU_PER_PLAYER),
    *(f"opp_{name}" for name in _TABLEAU_PER_PLAYER),
)

DRAFT_OFFER_FEATURES = ("second_round",)
CITY_CARD_FEATURES = ("mine",)
WONDER_FEATURES = (
    "mine",
    "built",
    "retired",
    "affordable",
    "cost",
    "cost_s",
    "grants_extra_turn",
    "shields",
)
PROGRESS_FEATURES = ("on_board", "mine", "theirs", "library_candidate")
DISCARD_FEATURES = ("revive_candidate",)
POOL_FEATURES = (
    "count",
    "count_s",
    "my_mean_cost",
    "my_min_cost",
    "opp_mean_cost",
    "opp_min_cost",
    *(f"member_{name}" for name in _CARD_NAMES),
)
POOL_WONDER_FEATURES = (
    "count",
    "count_s",
    *(f"member_{name}" for name in _WONDER_NAMES),
)

_SCHEMA = {
    "version": ENCODER_VERSION,
    "entity_spaces": {
        TokenType.GLOBAL.value: 1,
        TokenType.DRAFT_OFFER.value: 12,
        TokenType.TABLEAU.value: 77,  # 73 cards + 4 back types (face-down)
        TokenType.CITY_CARD.value: 73,
        TokenType.WONDER.value: 12,
        TokenType.PROGRESS.value: 10,
        TokenType.DISCARD.value: 73,
        TokenType.POOL.value: 4,
        TokenType.POOL_WONDER.value: 1,
    },
    "features": {
        TokenType.GLOBAL.value: GLOBAL_FEATURES,
        TokenType.DRAFT_OFFER.value: DRAFT_OFFER_FEATURES,
        TokenType.TABLEAU.value: TABLEAU_FEATURES,
        TokenType.CITY_CARD.value: CITY_CARD_FEATURES,
        TokenType.WONDER.value: WONDER_FEATURES,
        TokenType.PROGRESS.value: PROGRESS_FEATURES,
        TokenType.DISCARD.value: DISCARD_FEATURES,
        TokenType.POOL.value: POOL_FEATURES,
        TokenType.POOL_WONDER.value: POOL_WONDER_FEATURES,
    },
}
ENCODER_SIGNATURE = hashlib.sha256(
    json.dumps(_SCHEMA, sort_keys=True).encode()
).hexdigest()

_FEATURE_COUNTS = {
    TokenType[key.upper()]: len(names) for key, names in _SCHEMA["features"].items()
}


def _token(
    token_type: TokenType, entity_id: int, features: list[float], aux_id: int = -1
) -> Token:
    if len(features) != _FEATURE_COUNTS[token_type]:
        raise AssertionError(
            f"{token_type.value}: {len(features)} features, "
            f"schema says {_FEATURE_COUNTS[token_type]}"
        )
    return Token(
        type=token_type,
        entity_id=entity_id,
        aux_id=aux_id,
        features=tuple(float(v) for v in features),
    )


# --- public-fields stub for engine pricing helpers --------------------------


def _stub_state(observation: PlayerObservation) -> GameState:
    cities = tuple(
        CityState(
            coins=city.coins,
            wonders=list(city.wonders),
            built_wonders=list(city.built_wonders),
            buildings=list(city.buildings),
            progress_tokens=list(city.progress_tokens),
            claimed_science_pairs=set(city.claimed_science_pairs),
        )
        for city in observation.cities
    )
    return GameState(
        seed=0,
        first_player=0,
        phase=observation.phase,
        active_player=observation.active_player,
        age=max(observation.age, 1),
        cities=cities,  # type: ignore[arg-type]
        available_progress_tokens=observation.available_progress_tokens,
        unused_progress_tokens=(),
        wonder_groups=((), ()),
        unused_wonders=(),
        wonder_offer=list(observation.wonder_offer),
        wonder_round=0,
        wonder_pick_index=0,
        age_decks={},
        removed_age_cards={},
        selected_guilds=(),
        unused_guilds=(),
        tableau=TableauState(age=max(observation.age, 1), cards={}),
        discard_pile=list(observation.discard_pile),
        buried_cards=list(observation.buried_cards),
        retired_wonders=set(observation.retired_wonders),
        pending_choice=observation.pending_choice,
        pending_extra_turn=observation.pending_extra_turn,
        pending_shields=observation.pending_shields,
        conflict_position=observation.conflict_position,
        military_tokens_remaining=dict(observation.military_tokens_remaining),
        winner=observation.winner,
        victory_type=observation.victory_type,
        final_scores=observation.final_scores,
        rng=random.Random(0),
    )


# --- shared derived quantities ----------------------------------------------


class _Derived:
    """Per-encode cache of quantities several token types share."""

    def __init__(self, observation: PlayerObservation, pool: UnseenPool, actor: int):
        self.obs = observation
        self.pool = pool
        self.actor = actor
        self.stub = _stub_state(observation)
        self.symbols = tuple(self._symbols(seat) for seat in (0, 1))
        self.relevant_backs = self._relevant_backs()
        self.obtainable_cards = self._obtainable_cards()
        self.min_costs = {
            seat: {
                name: minimum_payment(
                    self.stub, seat, CARDS_BY_NAME[name].cost, card=CARDS_BY_NAME[name]
                ).total_coins
                for back in self.pool.cards
                for name in self.pool.cards[back]
            }
            for seat in (0, 1)
        }

    def _symbols(self, seat: int) -> frozenset[ScienceSymbol]:
        found: set[ScienceSymbol] = set()
        city = self.obs.cities[seat]
        for name in city.buildings:
            if CARDS_BY_NAME[name].science is not None:
                found.add(CARDS_BY_NAME[name].science)
        for name in city.progress_tokens:
            if PROGRESS_BY_NAME[name].science is not None:
                found.add(PROGRESS_BY_NAME[name].science)
        return frozenset(found)

    def _relevant_backs(self) -> frozenset[BackType]:
        """Backs whose pool cards can still enter play (loose: includes the
        current age's removed cards by design — spec §5.2)."""

        age_backs = {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}
        backs = {age_backs[a] for a in range(max(self.obs.age, 1), 4)}
        backs.add(BackType.GUILD)
        return frozenset(backs)

    def _obtainable_cards(self) -> tuple[str, ...]:
        names = [
            card.card_name
            for card in self.obs.tableau
            if card.card_name is not None and card.present
        ]
        for back in self.relevant_backs:
            names.extend(self.pool.cards[back])
        return tuple(names)

    # military ---------------------------------------------------------------

    def rel_position(self, seat: int) -> int:
        return (
            self.obs.conflict_position
            if seat == 0
            else -self.obs.conflict_position
        )

    def next_token(self, seat: int) -> tuple[int, int]:
        """(distance, penalty) to the next unclaimed token ahead of ``seat``'s
        advance; (18, 0) when none remain in that direction."""

        position = self.rel_position(seat)
        best: tuple[int, int] | None = None
        for absolute, penalty in self.obs.military_tokens_remaining:
            relative = absolute if seat == 0 else -absolute
            if relative > position:
                distance = relative - position
                if best is None or distance < best[0]:
                    best = (distance, penalty)
        return best if best is not None else (18, 0)

    def progress_obtainable(self, seat: int, token_name: str) -> bool:
        if token_name in self.obs.available_progress_tokens:
            return True
        if token_name in self.obs.cities[seat].progress_tokens:
            return True
        if token_name in self.obs.cities[1 - seat].progress_tokens:
            return False
        # Off-board: reachable only through an unbuilt Great Library.
        city = self.obs.cities[seat]
        return (
            "The Great Library" in city.wonders
            and "The Great Library" not in city.built_wonders
            and "The Great Library" not in self.obs.retired_wonders
        )

    def effective_shields(self, seat: int, card_name: str) -> int:
        card = CARDS_BY_NAME[card_name]
        shields = card.shields
        if (
            card.color is CardColor.RED
            and "Strategy" in self.obs.cities[seat].progress_tokens
        ):
            shields += 1
        return shields

    def military_bound(self, seat: int) -> int:
        total = sum(CARDS_BY_NAME[name].shields for name in self.obtainable_cards)
        city = self.obs.cities[seat]
        for name in city.wonders:
            if name not in city.built_wonders and name not in self.obs.retired_wonders:
                total += WONDERS_BY_NAME[name].shields
        if self.progress_obtainable(seat, "Strategy"):
            total += sum(
                1
                for name in self.obtainable_cards
                if CARDS_BY_NAME[name].color is CardColor.RED
            )
        return total

    def science_missing_obtainable(self, seat: int) -> int:
        have = self.symbols[seat]
        obtainable: set[ScienceSymbol] = set()
        for name in self.obtainable_cards:
            if CARDS_BY_NAME[name].science is not None:
                obtainable.add(CARDS_BY_NAME[name].science)
        if self.progress_obtainable(seat, "Law"):
            obtainable.add(ScienceSymbol.LAW)
        return len(obtainable - have)


# --- per-player global block ------------------------------------------------


def _unbuilt_wonder_stats(derived: _Derived, seat: int) -> tuple[int, int]:
    city = derived.obs.cities[seat]
    theology = "Theology" in city.progress_tokens
    unbuilt = 0
    extra_turn = 0
    for name in city.wonders:
        if name in city.built_wonders or name in derived.obs.retired_wonders:
            continue
        unbuilt += 1
        wonder = WONDERS_BY_NAME[name]
        if theology or any(e.kind is EffectKind.PLAY_AGAIN for e in wonder.effects):
            extra_turn += 1
    return unbuilt, extra_turn


def _per_player_values(derived: _Derived, seat: int) -> list[float]:
    obs = derived.obs
    stub = derived.stub
    city = obs.cities[seat]
    have = derived.symbols[seat]
    color_counts = {color: 0 for color in _COLORS}
    for name in city.buildings:
        color_counts[CARDS_BY_NAME[name].color] += 1
    unbuilt, extra_turn = _unbuilt_wonder_stats(derived, seat)
    fixed = _fixed_production(stub, seat)
    choices = _choice_producers(stub, seat)
    discounts = _trade_discounts(stub, seat)
    opponent_production = _opponent_trade_production(stub, seat)
    score = score_player(stub, seat)
    mil_bound = derived.military_bound(seat)
    dist_win = 9 - derived.rel_position(seat)
    sci_missing = derived.science_missing_obtainable(seat)
    return [
        city.coins,
        city.coins / 10,
        len(have),
        max(0, 6 - len(have)),
        *(1.0 if s in have else 0.0 for s in _SYMBOLS),
        *(color_counts[c] for c in _COLORS),
        unbuilt,
        extra_turn,
        *(fixed[r] for r in _RESOURCES),
        *(sum(1 for group in choices if r in group) for r in _RESOURCES),
        *(
            1 if r in discounts else 2 + opponent_production[r]
            for r in _RESOURCES
        ),
        discard_income(color_counts[CardColor.YELLOW]),
        score.military,
        score.buildings,
        score.wonders,
        score.progress,
        score.treasury,
        score.total,
        score.blue_buildings,
        mil_bound,
        1.0 if mil_bound >= dist_win else 0.0,
        sci_missing,
        1.0 if len(have) + sci_missing >= 6 else 0.0,
    ]


def _global_token(derived: _Derived) -> Token:
    obs = derived.obs
    if obs.pending_choice is not None:
        decision = _PENDING_DECISION[obs.pending_choice.kind]
    elif obs.phase is Phase.WONDER_DRAFT:
        decision = "wonder_draft"
    elif obs.phase is Phase.CHOOSE_NEXT_START_PLAYER:
        decision = "next_age_starter"
    elif obs.phase is Phase.COMPLETE:
        decision = "complete"
    else:
        decision = "main_turn"
    present = [card for card in obs.tableau if card.present]
    face_down = sum(1 for card in present if not card.revealed)
    military = derived.rel_position(derived.actor)
    my_token = derived.next_token(derived.actor)
    opp_token = derived.next_token(1 - derived.actor)
    values = [
        *(1.0 if d == decision else 0.0 for d in _DECISIONS),
        *(1.0 if obs.age == a else 0.0 for a in (1, 2, 3)),
        len(present),
        len(present) / 20,
        face_down,
        face_down / 10,
        military,
        military / 9,
        9 - military,
        (9 - military) / 18,
        9 + military,
        (9 + military) / 18,
        my_token[0],
        my_token[0] / 18,
        my_token[1],
        opp_token[0],
        opp_token[0] / 18,
        opp_token[1],
        obs.pending_shields,
        1.0 if obs.pending_extra_turn else 0.0,
    ]
    values.extend(_per_player_values(derived, derived.actor))
    values.extend(_per_player_values(derived, 1 - derived.actor))
    return _token(TokenType.GLOBAL, 0, values)


# --- tableau tokens ---------------------------------------------------------


def _tableau_card_per_player(derived: _Derived, seat: int, card_name: str) -> list[float]:
    card = CARDS_BY_NAME[card_name]
    payment = minimum_payment(derived.stub, seat, card.cost, card=card)
    cost = payment.total_coins
    affordable = derived.obs.cities[seat].coins >= cost
    have = derived.symbols[seat]
    claimed = derived.obs.cities[seat].claimed_science_pairs
    completes_pair = (
        card.science is not None and card.science in have and card.science not in claimed
    )
    gives_sixth = (
        card.science is not None
        and card.science not in have
        and len(have) + 1 >= 6
    )
    shields = derived.effective_shields(seat, card_name)
    next_dist, _ = derived.next_token(seat)
    dist_win = 9 - derived.rel_position(seat)
    return [
        1.0 if affordable else 0.0,
        cost,
        cost / 10,
        1.0 if payment.used_chain else 0.0,
        1.0 if completes_pair else 0.0,
        1.0 if gives_sixth else 0.0,
        shields,
        1.0 if shields >= next_dist else 0.0,
        1.0 if shields >= dist_win else 0.0,
    ]


def _tableau_tokens(derived: _Derived) -> list[Token]:
    obs = derived.obs
    if not obs.tableau:
        return []
    layout = {
        (slot.row, slot.x): slot for slot in TABLEAU_LAYOUTS[max(obs.age, 1)]
    }
    present = {card.slot_id: card for card in obs.tableau if card.present}
    tokens = []
    for slot_id in sorted(present):
        public = present[slot_id]
        slot = layout[slot_id]
        coverers = sum(
            1
            for other in present
            if other[0] == slot_id[0] + 1 and abs(other[1] - slot_id[1]) == 1
        )
        covers_hidden = sum(
            1
            for other, other_card in present.items()
            if other[0] == slot_id[0] - 1
            and abs(other[1] - slot_id[1]) == 1
            and not other_card.revealed
        )
        values = [
            slot_id[0],
            slot_id[0] / 6,
            slot_id[1],
            slot_id[1] / 11,
            1.0 if slot.face_up else 0.0,
            1.0 if public.accessible else 0.0,
            coverers,
            covers_hidden,
        ]
        if public.card_name is not None:
            entity = CARD_IDS[public.card_name]
            values.extend(_tableau_card_per_player(derived, derived.actor, public.card_name))
            values.extend(
                _tableau_card_per_player(derived, 1 - derived.actor, public.card_name)
            )
        else:
            entity = 73 + _BACK_ID[public.back]
            values.extend([0.0] * (2 * len(_TABLEAU_PER_PLAYER)))
        tokens.append(_token(TokenType.TABLEAU, entity, values))
    return tokens


# --- remaining token types --------------------------------------------------


def _draft_offer_tokens(derived: _Derived) -> list[Token]:
    obs = derived.obs
    if obs.phase is not Phase.WONDER_DRAFT:
        return []
    picked = sum(len(city.wonders) for city in obs.cities)
    second_round = 1.0 if picked >= 4 else 0.0
    return [
        _token(TokenType.DRAFT_OFFER, WONDER_IDS[name], [second_round])
        for name in sorted(obs.wonder_offer, key=WONDER_IDS.__getitem__)
    ]


def _city_card_tokens(derived: _Derived) -> list[Token]:
    tokens = []
    for mine, seat in ((1.0, derived.actor), (0.0, 1 - derived.actor)):
        for name in derived.obs.cities[seat].buildings:
            tokens.append(_token(TokenType.CITY_CARD, CARD_IDS[name], [mine]))
    return tokens


def _wonder_tokens(derived: _Derived) -> list[Token]:
    obs = derived.obs
    burials = dict(obs.wonder_burials)
    tokens = []
    for mine, seat in ((1.0, derived.actor), (0.0, 1 - derived.actor)):
        city = obs.cities[seat]
        theology = "Theology" in city.progress_tokens
        for name in city.wonders:
            wonder = WONDERS_BY_NAME[name]
            built = name in city.built_wonders
            retired = name in obs.retired_wonders
            if built or retired:
                affordable, cost = 0.0, 0.0
            else:
                payment = minimum_payment(derived.stub, seat, wonder.cost, is_wonder=True)
                cost = payment.total_coins
                affordable = 1.0 if city.coins >= cost else 0.0
            grants_extra = theology or any(
                e.kind is EffectKind.PLAY_AGAIN for e in wonder.effects
            )
            buried = burials.get(name)
            tokens.append(
                _token(
                    TokenType.WONDER,
                    WONDER_IDS[name],
                    [
                        mine,
                        1.0 if built else 0.0,
                        1.0 if retired else 0.0,
                        affordable,
                        cost,
                        cost / 10,
                        1.0 if grants_extra else 0.0,
                        wonder.shields,
                    ],
                    aux_id=CARD_IDS[buried] if buried is not None else -1,
                )
            )
    return tokens


def _progress_tokens(derived: _Derived) -> list[Token]:
    obs = derived.obs
    candidates = (
        set(obs.pending_choice.options)
        if obs.pending_choice is not None
        and obs.pending_choice.kind is PendingChoiceKind.CHOOSE_UNUSED_PROGRESS
        else set()
    )
    tokens = []
    for name in obs.available_progress_tokens:
        tokens.append(
            _token(TokenType.PROGRESS, PROGRESS_IDS[name], [1.0, 0.0, 0.0, 0.0])
        )
    for mine, seat in ((1.0, derived.actor), (0.0, 1 - derived.actor)):
        for name in obs.cities[seat].progress_tokens:
            tokens.append(
                _token(
                    TokenType.PROGRESS,
                    PROGRESS_IDS[name],
                    [0.0, mine, 1.0 - mine, 0.0],
                )
            )
    for name in sorted(candidates, key=PROGRESS_IDS.__getitem__):
        tokens.append(
            _token(TokenType.PROGRESS, PROGRESS_IDS[name], [0.0, 0.0, 0.0, 1.0])
        )
    return tokens


def _discard_tokens(derived: _Derived) -> list[Token]:
    obs = derived.obs
    mausoleum_pending = (
        obs.pending_choice is not None
        and obs.pending_choice.kind is PendingChoiceKind.BUILD_FROM_DISCARD_FREE
    )
    return [
        _token(
            TokenType.DISCARD,
            CARD_IDS[name],
            [1.0 if mausoleum_pending else 0.0],
        )
        for name in obs.discard_pile
    ]


def _pool_tokens(derived: _Derived) -> list[Token]:
    tokens = []
    for back in _BACKS:
        members = derived.pool.cards[back]
        if not members:
            continue
        my_costs = [derived.min_costs[derived.actor][name] for name in members]
        opp_costs = [derived.min_costs[1 - derived.actor][name] for name in members]
        membership = [1.0 if name in members else 0.0 for name in _CARD_NAMES]
        tokens.append(
            _token(
                TokenType.POOL,
                _BACK_ID[back],
                [
                    len(members),
                    len(members) / 23,
                    sum(my_costs) / len(my_costs),
                    min(my_costs),
                    sum(opp_costs) / len(opp_costs),
                    min(opp_costs),
                    *membership,
                ],
            )
        )
    return tokens


def _pool_wonder_token(derived: _Derived) -> list[Token]:
    obs = derived.obs
    if obs.phase is not Phase.WONDER_DRAFT:
        return []
    picked = sum(len(city.wonders) for city in obs.cities)
    if picked >= 4:
        return []  # second group is face-up; the unseen four never matter
    members = derived.pool.wonders
    membership = [1.0 if name in members else 0.0 for name in _WONDER_NAMES]
    return [
        _token(
            TokenType.POOL_WONDER,
            0,
            [len(members), len(members) / 12, *membership],
        )
    ]


# --- entry point ------------------------------------------------------------


def encode(observation: PlayerObservation) -> Encoding:
    """Encode one observation from the actor's perspective (spec §2/§5)."""

    actor = (
        observation.pending_choice.player
        if observation.pending_choice is not None
        else observation.active_player
    )
    pool = unseen_pool(observation)
    derived = _Derived(observation, pool, actor)
    tokens = [
        _global_token(derived),
        *_draft_offer_tokens(derived),
        *_tableau_tokens(derived),
        *_city_card_tokens(derived),
        *_wonder_tokens(derived),
        *_progress_tokens(derived),
        *_discard_tokens(derived),
        *_pool_tokens(derived),
        *_pool_wonder_token(derived),
    ]
    return Encoding(actor=actor, tokens=tuple(tokens))
