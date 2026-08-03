"""Wonder-draft support in the scrape codec and the BGA mapper (item A).

The draft is 8 picks of which 6 are real decisions -- BGA auto-takes the last
wonder of each round (`SelectWonderTrait.php:31-41`) -- and it shapes the whole
game, so the advisor being silent through it was the last real gap.

What makes the draft different from PLAY_AGE is the *shape* of the hidden
information. A play position hides a deck; a draft position hides a **partition**.
BGA shows only the current group (`Wonders::getSituation` returns just
`selection{round}`), so in round 0 the unknown is which 4 of the 8 unseen wonders
form group 2 and which 4 are the never-dealt box: C(8,4) = 70 equally likely
splits. In round 1 group 2 is visible and only the box remains hidden, which can
no longer affect play.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import pytest

from .advisor_scrape import (
    determinize_observation,
    observation_from_wire,
    observation_to_wire,
)
from .codec import decode_action, legal_action_indices
from .data import WONDERS
from .engine import apply_action, legal_actions
from .game import Phase, new_game

_TESTDATA = Path(__file__).parent / "testdata"
_ALL_WONDERS = {w.name for w in WONDERS}


def _draft_positions(seeds=range(20), picks=9):
    """Every draft position reachable by playing the draft out."""
    for seed in seeds:
        game = new_game(seed, seed % 2)
        rng = random.Random(seed)
        for step in range(picks):
            if game.phase is not Phase.WONDER_DRAFT:
                break
            yield seed, step, game
            actions = legal_actions(game)
            if not actions:
                break
            apply_action(game, rng.choice(actions))


def test_draft_positions_round_trip_public_exact():
    seen = 0
    for seed, step, game in _draft_positions():
        obs = game.observation(0)
        rebuilt = observation_from_wire(observation_to_wire(obs))
        state = determinize_observation(rebuilt, random.Random(seed * 31 + step))
        assert state.observation(0) == obs, f"seed {seed} step {step}"
        seen += 1
    assert seen > 100, seen


def test_draft_determinization_is_legal_and_partitions_all_twelve_wonders():
    """Two invariants that together pin the reconstruction down.

    Legality is the sharper one: `pick_wonder` asserts
    `active_player == _draft_order(round)[pick_index]`, and `new_game(0, 0)`
    leaves `first_player = 0`, so a draft state that does not derive
    `first_player` correctly fires that assertion.
    """
    for seed, step, game in _draft_positions():
        obs = game.observation(0)
        state = determinize_observation(obs, random.Random(seed * 7 + step))

        offered = {decode_action(state, i).wonder_name for i in legal_action_indices(state)}
        assert offered == set(obs.wonder_offer), f"seed {seed} step {step}"

        counts = collections.Counter(state.wonder_groups[0])
        counts += collections.Counter(state.wonder_groups[1])
        counts += collections.Counter(state.unused_wonders)
        assert set(counts) == _ALL_WONDERS
        assert all(n == 1 for n in counts.values()), "a wonder appears twice"
        assert (
            len(state.wonder_groups[0]),
            len(state.wonder_groups[1]),
            len(state.unused_wonders),
        ) == (4, 4, 4)

        # The engine must accept the move the human is about to make.
        state.clone().pick_wonder(next(iter(obs.wonder_offer)))


def test_group_two_is_sampled_across_the_unseen_wonders():
    """Round 0 hides a 4-of-8 partition. Over many seeds every unseen wonder
    should turn up in group 2 -- a determinizer that fixed the split would still
    pass every other test here."""
    game = new_game(3, 0)
    obs = game.observation(0)
    seen: collections.Counter = collections.Counter()
    for seed in range(300):
        state = determinize_observation(obs, random.Random(seed))
        assert state.wonder_round == 0 and state.wonder_pick_index == 0
        seen.update(state.wonder_groups[1])
    unseen_count = 12 - len(obs.wonder_offer)
    assert len(seen) == unseen_count, f"only {len(seen)} of {unseen_count} ever sampled"
    assert min(seen.values()) > 20, seen


def test_round_one_reveals_group_two():
    """Once the second group is on offer, nothing about its *membership* is
    hidden, so both groups must be fully determined by what is visible.

    Compared as sets, not sequences: the engine holds each group in deal order
    while a reconstruction can only see pick order plus the current offer, and
    deal order is not recoverable from a public observation. It also does not
    matter -- the observation does not expose it (see the round-trip test) and
    the codec orders actions by wonder id, so nothing downstream can tell.
    """
    checked = 0
    for seed, _step, game in _draft_positions(seeds=range(12)):
        if game.wonder_round != 1:
            continue
        obs = game.observation(0)
        state = determinize_observation(obs, random.Random(seed))
        for i in (0, 1):
            assert set(state.wonder_groups[i]) == set(game.wonder_groups[i]), (
                f"seed {seed} group {i}"
            )
        assert set(state.unused_wonders) == set(game.unused_wonders), f"seed {seed}"
        checked += 1
    assert checked > 10, checked


# --- the real BGA capture ---------------------------------------------------


def _live_draft():
    raw = json.loads(
        (_TESTDATA / "bga_892846644_draft.json").read_text(encoding="utf-8")
    )
    return {"bga": raw["bga"], "args": raw["args"], "dom": raw["dom"]}


def test_live_bga_draft_capture_maps_and_determinizes():
    from .bga_extract import wire_from_bga_payload

    wire = wire_from_bga_payload(_live_draft())
    obs_wire = wire["observation"]
    assert obs_wire["phase"] == Phase.WONDER_DRAFT.value
    assert obs_wire["tableau"] == [], "no age is dealt during the draft"
    assert len(obs_wire["wonder_offer"]) == 4
    assert all(c["wonders"] == [] for c in obs_wire["cities"]), "first pick of the game"

    obs = observation_from_wire(obs_wire)
    state = determinize_observation(obs, random.Random(0))
    assert state.observation(0) == obs
    assert state.wonder_round == 0 and state.wonder_pick_index == 0
    assert {decode_action(state, i).wonder_name for i in legal_action_indices(state)} == set(
        obs_wire["wonder_offer"]
    )


def test_draft_resample_seed_actually_varies_the_determinization():
    """Draft advice must be aggregated over determinizations -- a round-0 query
    has 70 partitions times however many age deals -- so the seed has to move
    the hidden state."""
    from .bga_extract import wire_from_bga_payload

    payload = _live_draft()
    groups = set()
    for seed in range(8):
        wire = wire_from_bga_payload(payload, resample_seed=seed)
        obs = observation_from_wire(wire["observation"])
        state = determinize_observation(obs, random.Random(seed))
        groups.add(tuple(sorted(state.wonder_groups[1])))
    assert len(groups) > 1, "every seed produced the same hidden group 2"


def test_between_age_start_player_choice_is_still_rejected():
    """Supporting the draft must not quietly widen the mapper's scope."""
    from .bga_extract import UnsupportedBgaState, wire_from_bga_payload

    payload = _live_draft()
    payload["bga"]["gamestate"] = dict(payload["bga"]["gamestate"], name="selectStartPlayer")
    with pytest.raises(UnsupportedBgaState):
        wire_from_bga_payload(payload)
