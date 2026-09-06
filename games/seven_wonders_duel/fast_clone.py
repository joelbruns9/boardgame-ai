"""A structure-aware copy of ``GameState``, for proof search.

``GameState.clone`` is ``copy.deepcopy``. On the sudden-death certifier that is
77% of total runtime -- 17.2 M ``deepcopy`` calls for 12,763 clones -- because
deepcopy walks every string, frozen dataclass and nested tuple as though it
might be mutated. Almost none of it can be.

This module copies exactly what the engine mutates and shares the rest.

**Why this is not ``GameState.clone``.** That method is on the hot path of
search, self-play, and the replay/equivalence gates. A sharing bug there would
surface as a heisenbug in training data, months later, with no clean signal. So
the fast path is opt-in, lives here, and is pinned by an equivalence test
(``test_fast_clone.py``) that compares it against ``deepcopy`` field by field
and checks independence by mutation.

**What is shared, and why that is safe.**

* ``str``, ``int``, ``bool``, ``None``, ``Enum`` -- immutable.
* ``tuple`` of the above (``available_progress_tokens``, ``wonder_groups``,
  ``age_decks`` values, ...) -- immutable, and the engine replaces these
  wholesale rather than mutating them.
* ``TableauSlot``, ``PendingChoice`` -- ``frozen=True`` dataclasses.

**What is copied.** ``CityState`` (mutable, holds lists and a set),
``TableauCard`` (mutable ``revealed`` / ``present``), and every ``list``,
``dict`` and ``set`` the engine writes through.

**The RNG.** ``random.Random`` is rebuilt from ``getstate()``, whose payload is
an immutable tuple of ints -- no recursive copy, and the future stream is
identical, which is the contract ``clone``'s docstring promises.

**The guard that matters.** ``_FIELDS`` pins the dataclass layout. Add a field to
``GameState`` and this module raises on import-time first use rather than
silently failing to copy it. A missed mutable field is precisely the bug class
this design is exposed to, so it is made loud instead of possible.
"""

from __future__ import annotations

import random

from .game import CityState, GameState, TableauCard, TableauState

#: Pinned field layout. A mismatch means someone changed GameState and this
#: module has not been reviewed against the change -- which could mean a mutable
#: field is being shared between a state and its copy.
_FIELDS = (
    "seed",
    "first_player",
    "phase",
    "active_player",
    "age",
    "cities",
    "available_progress_tokens",
    "unused_progress_tokens",
    "wonder_groups",
    "unused_wonders",
    "wonder_offer",
    "wonder_round",
    "wonder_pick_index",
    "age_decks",
    "removed_age_cards",
    "selected_guilds",
    "unused_guilds",
    "tableau",
    "discard_pile",
    "buried_cards",
    "retired_wonders",
    "pending_choice",
    "pending_extra_turn",
    "pending_shields",
    "conflict_position",
    "military_tokens_remaining",
    "winner",
    "victory_type",
    "final_scores",
    "rng",
    "wonder_burials",
    "search_barrier",
)


def check_layout() -> None:
    """Raise if ``GameState``'s fields no longer match what this module copies."""

    import dataclasses

    live = tuple(f.name for f in dataclasses.fields(GameState))
    if live != _FIELDS:
        added = set(live) - set(_FIELDS)
        removed = set(_FIELDS) - set(live)
        raise RuntimeError(
            "GameState layout changed since fast_clone was written; review "
            "whether the new fields are mutable before copying them. "
            f"added={sorted(added)} removed={sorted(removed)} "
            "(order matters too)"
        )


_checked = False


def _copy_city(city: CityState) -> CityState:
    return CityState(
        coins=city.coins,
        wonders=list(city.wonders),
        built_wonders=list(city.built_wonders),
        buildings=list(city.buildings),
        progress_tokens=list(city.progress_tokens),
        claimed_science_pairs=set(city.claimed_science_pairs),
    )


def _copy_tableau(tableau: TableauState) -> TableauState:
    # `slot` is a frozen dataclass, so it is shared rather than rebuilt.
    return TableauState(
        age=tableau.age,
        cards={
            slot_id: TableauCard(
                slot=card.slot,
                card_name=card.card_name,
                revealed=card.revealed,
                present=card.present,
            )
            for slot_id, card in tableau.cards.items()
        },
    )


def _copy_rng(rng: random.Random) -> random.Random:
    out = random.Random()
    out.setstate(rng.getstate())
    return out


def fast_clone(state: GameState) -> GameState:
    """An independent copy of ``state``, equivalent to ``state.clone()``."""

    global _checked
    if not _checked:
        check_layout()
        _checked = True

    return GameState(
        seed=state.seed,
        first_player=state.first_player,
        phase=state.phase,
        active_player=state.active_player,
        age=state.age,
        cities=(_copy_city(state.cities[0]), _copy_city(state.cities[1])),
        available_progress_tokens=state.available_progress_tokens,
        unused_progress_tokens=state.unused_progress_tokens,
        wonder_groups=state.wonder_groups,
        unused_wonders=state.unused_wonders,
        wonder_offer=list(state.wonder_offer),
        wonder_round=state.wonder_round,
        wonder_pick_index=state.wonder_pick_index,
        age_decks=dict(state.age_decks),
        removed_age_cards=dict(state.removed_age_cards),
        selected_guilds=state.selected_guilds,
        unused_guilds=state.unused_guilds,
        tableau=_copy_tableau(state.tableau),
        discard_pile=list(state.discard_pile),
        buried_cards=list(state.buried_cards),
        retired_wonders=set(state.retired_wonders),
        pending_choice=state.pending_choice,
        pending_extra_turn=state.pending_extra_turn,
        pending_shields=state.pending_shields,
        conflict_position=state.conflict_position,
        military_tokens_remaining=dict(state.military_tokens_remaining),
        winner=state.winner,
        victory_type=state.victory_type,
        final_scores=state.final_scores,
        rng=_copy_rng(state.rng),
        wonder_burials=dict(state.wonder_burials),
        search_barrier=state.search_barrier,
    )
