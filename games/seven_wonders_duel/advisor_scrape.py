"""Reconstruct a searchable 7WD state from a public observation.

The BGA-scrape path of the advisor.  A browser extension can observe only
public information -- no seed, no hidden deck.  Closed-mode search never reads
hidden identities directly (it predicts chance from the observation and steps
barred clones with explicit outcomes), so a *determinization* -- any full state
whose public projection matches the observation -- is all search needs.

Strategy (leaning on the existing determinizer):
  1. Build a skeleton full state, overwriting every public field from the
     observation.
  2. Assign the *current age's* unseen multiset (from ``unseen_pool``) to its
     face-down tableau slots + removed pile, keeping counts and back types
     right so the visible projection is exact.
  3. Call ``resample_hidden`` -- it re-deals all *future* ages from scratch and
     reshuffles the current age within the multiset from step 2, producing a
     valid, uniformly-random determinization.

Supported: ``PLAY_AGE`` (the position a human actually asks about) and, for
state-building only, ``COMPLETE`` (terminal -> no search).  ``WONDER_DRAFT`` and
the between-age ``CHOOSE_NEXT_START_PLAYER`` transition are rejected: their
hidden structure is not reconstructable from a single public observation, and
the seed+prefix wire covers local analysis of those.
"""

from __future__ import annotations

import random

from .data import (
    CARD_IDS,
    PROGRESS_IDS,
    WONDER_IDS,
    BackType,
    ScienceSymbol,
    TABLEAU_LAYOUTS,
    back_type_of,
)
from .engine import back_type_of_age
from .game import (
    CityState,
    GameState,
    Phase,
    PendingChoice,
    PendingChoiceKind,
    PlayerObservation,
    PublicCity,
    PublicTableauCard,
    TableauCard,
    TableauState,
    VictoryType,
    new_game,
)
from .pool import BACK_UNIVERSES, unseen_pool

_SUPPORTED = (Phase.PLAY_AGE, Phase.COMPLETE)


def determinize_observation(obs: PlayerObservation, rng: random.Random) -> GameState:
    """A full state whose ``observation(0)`` equals ``obs`` (public-exact) with
    hidden information filled by a valid random determinization."""

    if obs.phase not in _SUPPORTED:
        raise ValueError(
            f"scrape codec supports PLAY_AGE/COMPLETE, not {obs.phase.name}; "
            "use the seed+prefix wire for that position"
        )

    state = new_game(0, 0)

    # 1. public fields ------------------------------------------------------
    state.phase = obs.phase
    state.active_player = obs.active_player
    state.age = obs.age
    state.cities = tuple(  # type: ignore[assignment]
        CityState(
            coins=pc.coins,
            wonders=list(pc.wonders),
            built_wonders=list(pc.built_wonders),
            buildings=list(pc.buildings),
            progress_tokens=list(pc.progress_tokens),
            claimed_science_pairs=set(pc.claimed_science_pairs),
        )
        for pc in obs.cities
    )
    state.available_progress_tokens = tuple(obs.available_progress_tokens)
    state.discard_pile = list(obs.discard_pile)
    state.buried_cards = list(obs.buried_cards)
    state.wonder_burials = dict(obs.wonder_burials)
    state.retired_wonders = set(obs.retired_wonders)
    state.pending_choice = obs.pending_choice
    state.pending_extra_turn = obs.pending_extra_turn
    state.pending_shields = obs.pending_shields
    state.conflict_position = obs.conflict_position
    state.military_tokens_remaining = dict(obs.military_tokens_remaining)
    state.winner = obs.winner
    state.victory_type = obs.victory_type
    state.final_scores = obs.final_scores
    state.wonder_offer = []
    state.wonder_round = 1
    state.wonder_pick_index = 4

    pool = unseen_pool(obs)
    state.unused_progress_tokens = tuple(
        sorted(pool.offboard_progress, key=PROGRESS_IDS.__getitem__)
    )
    state.unused_wonders = tuple(sorted(pool.wonders, key=WONDER_IDS.__getitem__))

    # 2. current-age tableau + hidden multiset ------------------------------
    layout = TABLEAU_LAYOUTS[obs.age]
    slot_by_id = {(slot.row, slot.x): slot for slot in layout}
    cards: dict = {}
    facedown_by_back: dict[BackType, list] = {}
    _filler = next(iter(BACK_UNIVERSES[back_type_of_age(obs.age)]))
    for pc in obs.tableau:
        slot = slot_by_id[pc.slot_id]
        if pc.present and pc.revealed:
            name = pc.card_name
        else:
            name = _filler  # face-down / absent placeholder (fixed below)
        cards[pc.slot_id] = TableauCard(
            slot=slot, card_name=name, revealed=pc.revealed, present=pc.present
        )
        if pc.present and not pc.revealed:
            facedown_by_back.setdefault(pc.back, []).append(pc.slot_id)

    if obs.age in (1, 2):
        back = back_type_of_age(obs.age)
        unseen = list(pool.cards[back])
        fds = facedown_by_back.get(back, [])
        if len(unseen) != len(fds) + 3:
            raise ValueError(
                f"age {obs.age} unseen={len(unseen)} != facedown {len(fds)} + 3 removed"
            )
        rng.shuffle(unseen)
        for slot_id, name in zip(fds, unseen):
            cards[slot_id].card_name = name
        state.removed_age_cards[obs.age] = tuple(unseen[len(fds):])
    else:  # age 3: AGE_III + GUILD backs, guilds split select/unused
        age3 = list(pool.cards[BackType.AGE_III])
        guild = list(pool.cards[BackType.GUILD])
        fds3 = facedown_by_back.get(BackType.AGE_III, [])
        fdsg = facedown_by_back.get(BackType.GUILD, [])
        if len(age3) != len(fds3) + 3:
            raise ValueError(f"age3 AGE_III {len(age3)} != facedown {len(fds3)} + 3")
        if len(guild) != len(fdsg) + 4:
            raise ValueError(f"age3 GUILD {len(guild)} != facedown {len(fdsg)} + 4 unused")
        rng.shuffle(age3)
        rng.shuffle(guild)
        for slot_id, name in zip(fds3, age3):
            cards[slot_id].card_name = name
        state.removed_age_cards[3] = tuple(age3[len(fds3):])
        for slot_id, name in zip(fdsg, guild):
            cards[slot_id].card_name = name
        state.unused_guilds = tuple(guild[len(fdsg):])
        all_guilds = set(BACK_UNIVERSES[BackType.GUILD])
        state.selected_guilds = tuple(
            sorted(all_guilds - set(state.unused_guilds), key=CARD_IDS.__getitem__)
        )

    state.tableau = TableauState(age=obs.age, cards=cards)
    state.age_decks[obs.age] = tuple(c.card_name for c in cards.values() if c.present)

    # 3. determinize hidden: futures re-dealt, current reshuffled -----------
    from .pool import resample_hidden

    resample_hidden(state, rng)
    return state


# --------------------------------------------------------------------------
# JSON wire for a public observation (the schema a scraper emits).
# --------------------------------------------------------------------------


def observation_to_wire(obs: PlayerObservation) -> dict:
    """Serialize a PlayerObservation to the scrape wire dict."""

    def city(pc: PublicCity) -> dict:
        return {
            "coins": pc.coins,
            "wonders": list(pc.wonders),
            "built_wonders": list(pc.built_wonders),
            "buildings": list(pc.buildings),
            "progress_tokens": list(pc.progress_tokens),
            "science_pairs": sorted(s.value for s in pc.claimed_science_pairs),
        }

    def card(pc: PublicTableauCard) -> dict:
        return {
            "slot_id": [pc.slot_id[0], pc.slot_id[1]],
            "present": pc.present,
            "revealed": pc.revealed,
            "accessible": pc.accessible,
            "card_name": pc.card_name,
            "back": None if pc.back is None else pc.back.value,
        }

    pc_choice = obs.pending_choice
    return {
        "phase": obs.phase.value,
        "active_player": obs.active_player,
        "age": obs.age,
        "cities": [city(c) for c in obs.cities],
        "available_progress_tokens": list(obs.available_progress_tokens),
        "wonder_offer": list(obs.wonder_offer),
        "tableau": [card(c) for c in obs.tableau],
        "discard_pile": list(obs.discard_pile),
        "buried_cards": list(obs.buried_cards),
        "wonder_burials": [list(pair) for pair in obs.wonder_burials],
        "retired_wonders": sorted(obs.retired_wonders),
        "pending_choice": None
        if pc_choice is None
        else {
            "kind": pc_choice.kind.value,
            "player": pc_choice.player,
            "options": list(pc_choice.options),
            "consume_all_options": pc_choice.consume_all_options,
        },
        "pending_extra_turn": obs.pending_extra_turn,
        "pending_shields": obs.pending_shields,
        "conflict_position": obs.conflict_position,
        "military_tokens_remaining": [list(t) for t in obs.military_tokens_remaining],
        "winner": obs.winner,
        "victory_type": None if obs.victory_type is None else obs.victory_type.value,
        "final_scores": None if obs.final_scores is None else list(obs.final_scores),
    }


def observation_from_wire(data: dict) -> PlayerObservation:
    """Parse the scrape wire dict back into a PlayerObservation."""

    def city(d: dict) -> PublicCity:
        return PublicCity(
            coins=int(d["coins"]),
            wonders=tuple(d["wonders"]),
            built_wonders=tuple(d["built_wonders"]),
            buildings=tuple(d["buildings"]),
            progress_tokens=tuple(d["progress_tokens"]),
            claimed_science_pairs=frozenset(ScienceSymbol(s) for s in d["science_pairs"]),
        )

    def card(d: dict) -> PublicTableauCard:
        return PublicTableauCard(
            slot_id=(int(d["slot_id"][0]), int(d["slot_id"][1])),
            present=bool(d["present"]),
            revealed=bool(d["revealed"]),
            accessible=bool(d["accessible"]),
            card_name=d["card_name"],
            back=None if d["back"] is None else BackType(d["back"]),
        )

    pc = data.get("pending_choice")
    return PlayerObservation(
        viewer=0,
        phase=Phase(data["phase"]),
        active_player=int(data["active_player"]),
        age=int(data["age"]),
        cities=(city(data["cities"][0]), city(data["cities"][1])),
        available_progress_tokens=tuple(data["available_progress_tokens"]),
        wonder_offer=tuple(data.get("wonder_offer", ())),
        tableau=tuple(card(c) for c in data["tableau"]),
        discard_pile=tuple(data["discard_pile"]),
        buried_cards=tuple(data["buried_cards"]),
        wonder_burials=tuple((p[0], p[1]) for p in data["wonder_burials"]),
        retired_wonders=frozenset(data["retired_wonders"]),
        pending_choice=None
        if pc is None
        else PendingChoice(
            kind=PendingChoiceKind(pc["kind"]),
            player=int(pc["player"]),
            options=tuple(pc["options"]),
            consume_all_options=bool(pc.get("consume_all_options", False)),
        ),
        pending_extra_turn=bool(data["pending_extra_turn"]),
        pending_shields=int(data["pending_shields"]),
        conflict_position=int(data["conflict_position"]),
        military_tokens_remaining=tuple(
            (int(t[0]), int(t[1])) for t in data["military_tokens_remaining"]
        ),
        winner=data["winner"],
        victory_type=None if data["victory_type"] is None else VictoryType(data["victory_type"]),
        final_scores=None if data["final_scores"] is None else tuple(data["final_scores"]),
    )
