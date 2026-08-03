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

Supported: ``WONDER_DRAFT`` and ``PLAY_AGE`` (the positions a human actually asks
about) and, for state-building only, ``COMPLETE`` (terminal -> no search).  The
between-age ``CHOOSE_NEXT_START_PLAYER`` transition is still rejected; the
seed+prefix wire covers local analysis of it.

The draft was long assumed unreconstructable. It is not: BGA shows only the
current group, so the hidden part is a uniform 4-of-8 partition into
(group 2 | never-dealt box) -- 70 equally likely splits -- which is the same kind
of object the age-deck determinizer already samples. It needs its own branch
because no age is dealt yet, so there is no tableau at all. See
``_determinize_draft``.
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

_SUPPORTED = (Phase.WONDER_DRAFT, Phase.PLAY_AGE, Phase.COMPLETE)


def determinize_observation(
    obs: PlayerObservation,
    rng: random.Random,
    *,
    unknown_burial_ages: tuple[int, ...] = (),
) -> GameState:
    """A full state whose ``observation(0)`` equals ``obs`` (public-exact) with
    hidden information filled by a valid random determinization.

    ``unknown_burial_ages``: ages of cards buried under constructed wonders whose
    *identity* we could not recover (BGA's ``gamedatas`` exposes only the age).
    Constructing a wonder buries an age card permanently, so each such burial is
    one more card of that age that is out of play and can never be revealed --
    exactly the role ``removed_age_cards`` already plays. Passing the ages here
    keeps the pool arithmetic honest: without it the buried cards stay in the
    unseen pool and can be dealt into a face-down slot they can never occupy.

    Burials whose identity *is* known arrive in ``obs.wonder_burials`` and are
    excluded from the unseen pool directly, which is strictly sharper.
    """

    if obs.phase not in _SUPPORTED:
        raise ValueError(
            f"scrape codec supports WONDER_DRAFT/PLAY_AGE/COMPLETE, not "
            f"{obs.phase.name}; use the seed+prefix wire for that position"
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
    # A finished draft is the right default for PLAY_AGE/COMPLETE: round 1
    # (0-indexed: the second group), all four taken. The draft branch below
    # overwrites all three.
    state.wonder_offer = []
    state.wonder_round = 1
    state.wonder_pick_index = 4

    pool = unseen_pool(obs)
    state.unused_progress_tokens = tuple(
        sorted(pool.offboard_progress, key=PROGRESS_IDS.__getitem__)
    )
    state.unused_wonders = tuple(sorted(pool.wonders, key=WONDER_IDS.__getitem__))

    if obs.phase is Phase.WONDER_DRAFT:
        _determinize_draft(state, obs, pool, rng)
        return state

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

    # Cards buried under wonders are out of play. Known identities leave the
    # unseen pool outright; unknown ones only add to the count that must land in
    # the "removed" remainder. BGA reports a buried guild's age as 4.
    buried_known = {card for _wonder, card in obs.wonder_burials}
    unknown_by_age: dict[int, int] = {}
    for age in unknown_burial_ages:
        unknown_by_age[int(age)] = unknown_by_age.get(int(age), 0) + 1

    def _unseen(back: BackType) -> list[str]:
        return [name for name in pool.cards[back] if name not in buried_known]

    if obs.age in (1, 2):
        back = back_type_of_age(obs.age)
        unseen = _unseen(back)
        fds = facedown_by_back.get(back, [])
        out_of_play = 3 + unknown_by_age.get(obs.age, 0)
        if len(unseen) != len(fds) + out_of_play:
            raise ValueError(
                f"age {obs.age} unseen={len(unseen)} != facedown {len(fds)} + "
                f"{out_of_play} out of play (3 removed + "
                f"{unknown_by_age.get(obs.age, 0)} unidentified wonder burials)"
            )
        rng.shuffle(unseen)
        for slot_id, name in zip(fds, unseen):
            cards[slot_id].card_name = name
        state.removed_age_cards[obs.age] = tuple(unseen[len(fds):])
    else:  # age 3: AGE_III + GUILD backs, guilds split select/unused
        age3 = _unseen(BackType.AGE_III)
        guild = _unseen(BackType.GUILD)
        fds3 = facedown_by_back.get(BackType.AGE_III, [])
        fdsg = facedown_by_back.get(BackType.GUILD, [])
        out3 = 3 + unknown_by_age.get(3, 0)
        outg = 4 + unknown_by_age.get(4, 0)
        if len(age3) != len(fds3) + out3:
            raise ValueError(
                f"age3 AGE_III {len(age3)} != facedown {len(fds3)} + {out3} "
                f"out of play (3 removed + {unknown_by_age.get(3, 0)} burials)"
            )
        if len(guild) != len(fdsg) + outg:
            raise ValueError(
                f"age3 GUILD {len(guild)} != facedown {len(fdsg)} + {outg} "
                f"out of play (4 unused + {unknown_by_age.get(4, 0)} burials)"
            )
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

    # Resolve any CARD_REVEAL the capture caught mid-flight. Taking a card
    # uncovers its neighbours; the engine reveals them via a CARD_REVEAL chance
    # event fired as part of that same action (TableauState.take_accessible,
    # game.py:176), and `_card_at` treats an accessible-but-face-down slot as a
    # hard error. A scrape taken mid-move -- during a pending choice, with the
    # triggering card already off the structure -- can land in exactly that gap,
    # because BGA defers its flip until the whole move resolves. The identity is
    # unknowable from the snapshot, so it is sampled like any other face-down
    # slot above; revealing it here is what the engine would have done.
    for slot_id, card in state.tableau.cards.items():
        if card.present and not card.revealed and state.tableau.is_accessible(slot_id):
            card.revealed = True
    state.age_decks[obs.age] = tuple(c.card_name for c in cards.values() if c.present)

    # 3. determinize hidden: futures re-dealt, current reshuffled -----------
    from .pool import resample_hidden

    resample_hidden(state, rng)
    return state


def _determinize_draft(state, obs, pool, rng: random.Random) -> None:
    """Fill a WONDER_DRAFT position.

    Nothing here touches the tableau: at draft time no age has been dealt, so the
    PLAY_AGE block's assumption of a live structure with face-down slots does not
    hold. `resample_hidden` deals all three ages from scratch for a draft state.

    The hidden information is a partition, not a deck. In round 0 BGA shows only
    the current group (`Wonders::getSituation` returns just `selection{round}`),
    so the unknown is which 4 of the 8 unseen wonders form group 2 and which 4
    are the never-dealt box -- C(8,4) = 70 equally likely partitions. In round 1
    group 2 is visible and only the 4 box wonders remain hidden, which can no
    longer affect play.
    """

    picked = sum(len(city.wonders) + len(city.built_wonders) for city in obs.cities)
    state.wonder_round = picked // 4  # 0-indexed: 0 is the first group
    state.wonder_pick_index = picked % 4
    state.wonder_offer = list(obs.wonder_offer)

    # `pick_wonder` asserts active_player == _draft_order(round)[pick_index], and
    # new_game(0, 0) leaves first_player = 0, so a draft state MUST set it or
    # that assertion fires. _draft_order is (f, 1-f, 1-f, f) for round 0 and
    # (1-f, f, f, 1-f) for round 1 -- BGA's "A-B-B-A, then B-A-A-B"
    # (WonderSelectedTrait.php:10) -- so the seat that picks at index 0 or 3 of
    # round 0 is the first player, and round 1 mirrors it.
    at_ends = state.wonder_pick_index in (0, 3)
    if state.wonder_round == 0:
        state.first_player = obs.active_player if at_ends else 1 - obs.active_player
    else:
        state.first_player = 1 - obs.active_player if at_ends else obs.active_player

    # wonder_groups must be reconstructed because pick_wonder reads
    # wonder_groups[1] when it flips the second group face-up.
    #
    # This needs picks in CHRONOLOGICAL order, and the observation only gives
    # them per player. Concatenating the two lists is wrong the moment round 1
    # starts -- it interleaves rounds and `taken[:4]` stops being group 0. The
    # global order is recoverable though: the draft sequence is fixed once
    # first_player is known, so replay it and pop from each player's list, which
    # `pick_wonder` appends to in pick order.
    def _order(round_index: int) -> tuple[int, int, int, int]:
        first = state.first_player if round_index == 0 else 1 - state.first_player
        return (first, 1 - first, 1 - first, first)

    queues = [
        list(city.wonders) + list(city.built_wonders) for city in obs.cities
    ]
    sequence: list[str] = []
    for seat in (_order(0) + _order(1))[:picked]:
        sequence.append(queues[seat].pop(0))

    taken = sequence
    if state.wonder_round == 0:
        # Everything picked so far came out of group 0, and the rest of group 0
        # is still on offer. Group 1 is a uniform 4-subset of the unseen 8.
        group0 = list(taken) + list(state.wonder_offer)
        hidden = sorted(pool.wonders, key=WONDER_IDS.__getitem__)
        rng.shuffle(hidden)
        group1 = hidden[:4]
        state.unused_wonders = tuple(hidden[4:])
    else:
        # Group 0 is exactly the first four picks; group 1 is the rest plus what
        # is still on offer. Only the 4 box wonders stay hidden.
        group0 = list(taken[:4])
        group1 = list(taken[4:]) + list(state.wonder_offer)
    state.wonder_groups = (tuple(group0), tuple(group1))

    # No age is dealt yet, so there is no tableau and no current-age multiset.
    state.tableau = TableauState(age=obs.age, cards={})

    from .pool import resample_hidden

    resample_hidden(state, rng)


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
