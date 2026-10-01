"""Encoder v3's blocking tests -- ``ENCODER_V3_SPEC.md`` §10.4 and §10.5.

The rest of §10 gates the helpers these features are built on and lives with
them: §10.1/§10.3 in ``test_plan_feasibility.py``, §10.2/§10.2a in
``test_plan_threat.py``, §10.6 in ``rust_encode_equiv.py`` (step 6).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from games.welcome_to import encoder as enc
from games.welcome_to.game import GameConfig, GameState, Phase


def _states(seeds=range(6), plies=140, players=2, advanced=True):
    """Mid-game states, sampled at every phase rather than only at boundaries."""
    for seed in seeds:
        rng = random.Random(seed)
        state = GameState.new(
            seed=seed, config=GameConfig(players=players, advanced=advanced)
        )
        for _ in range(plies):
            if state.is_terminal:
                break
            yield state
            state = state.step(rng.choice(state.legal_actions()))


# ──────────────────────────────────────────────────────────────────────────
# §10.4 Temp-split exactness -- blocking
# ──────────────────────────────────────────────────────────────────────────
def test_the_temp_split_partitions_the_shipped_writable_plane():
    """§10.4: ``plane 8 | plane 12`` is v2's plane 8, and the two are disjoint.

    This is what proves §5.2's split loses nothing.  v2 unioned ``numbers_for``
    into one writable mask, so a box reachable only by SPENDING A TEMP was
    marked identically to a box that is free -- a lie in the input, not an
    omission.  Splitting it gains a bit and must lose none.

    ⚠ Driven through :func:`encoder._sheet_planes` rather than
    ``encode_state``.  It is the same production code that writes both planes,
    and it is the only way to reach the spec's >=5k states: the §6.4 threat
    block is ~96% of a full encode, and nothing in this assertion depends on it.
    """
    checked = 0
    samples = list(_states(seeds=range(10), plies=250, players=4))
    samples += list(_states(seeds=range(10), plies=250, players=2))
    for state in samples:
        viewer = state.actor
        base, every = enc._offered(state, viewer)
        view = enc._DeckView(state, viewer)
        for seat in enc.seat_order(state, viewer):
            sheet = state.sheet_for(viewer, seat)
            out = np.zeros(
                (enc.SHEET_PLANES, enc.NUM_STREETS, enc.MAX_STREET_LEN),
                dtype=np.float32,
            )
            enc._sheet_planes(state, viewer, seat, sheet, base, every, view, out)

            union_truth = np.zeros_like(out[enc.P_WRITABLE])
            for n in every:
                for x, y in sheet.available_locations(n):
                    union_truth[x, y] = 1.0

            no_temp = out[enc.P_WRITABLE]
            temp_only = out[enc.P_WRITABLE_TEMP]
            assert not np.any(
                (no_temp > 0) & (temp_only > 0)
            ), "the two halves overlap"
            assert np.array_equal(
                np.maximum(no_temp, temp_only), union_truth
            ), "the union is not the shipped writable mask"
            checked += 1
    assert checked >= 5000, f"only {checked} seat-states covered"


def test_a_temp_only_box_is_marked_temp_only():
    """The split must actually fire, not merely be consistent with itself."""
    seen_temp_only = False
    for state in _states(seeds=range(4)):
        planes = enc.encode_state(state, state.actor)[0]
        if planes[:, enc.P_WRITABLE_TEMP].any():
            seen_temp_only = True
            break
    assert seen_temp_only, "no temp-only box in any sampled state"


# ──────────────────────────────────────────────────────────────────────────
# §10.5 Symmetry and leak -- carried, still blocking
#
# ⚠ TWO tests, not one.  At a turn boundary the live sheet and the public
# snapshot are EQUAL, so a helper reaching for `state.sheets[p]` passes a
# boundary symmetry test unnoticed; the leak is only visible mid-turn.
# ──────────────────────────────────────────────────────────────────────────
def _seat_block(state, viewer, seat):
    planes, scalars, _, _ = enc.encode_state(state, viewer)
    k = enc.seat_order(state, viewer).index(seat)
    return planes[k].copy(), scalars[k].copy()


def test_a_seat_encodes_identically_from_either_viewer_at_a_boundary():
    """§10.5 part 1.  The per-seat block is a function of the SEAT, not the viewer.

    Sampled at ``CHOOSE_CARDS`` with nothing written this turn, where the live
    sheet and the public snapshot agree, so any asymmetry is a real one.
    """
    checked = 0
    for state in _states(seeds=range(5)):
        if state.phase is not Phase.CHOOSE_CARDS or state.ctx.last_house is not None:
            continue
        if state.actor != 0:
            continue
        for seat in range(state.config.players):
            a_planes, a_scalars = _seat_block(state, 0, seat)
            b_planes, b_scalars = _seat_block(state, 1, seat)
            # `is_viewer` is viewer-relative BY DESIGN and is the one exception.
            iv = enc.block_slice("is_viewer")
            a_scalars[iv] = b_scalars[iv] = 0.0
            assert np.array_equal(a_planes, b_planes), f"planes differ for seat {seat}"
            assert np.allclose(
                a_scalars, b_scalars, atol=0, rtol=0
            ), f"scalars differ for seat {seat}"
            checked += 1
    assert checked > 200, f"only {checked} comparisons"


def test_an_opponents_mid_turn_write_does_not_reach_the_viewer():
    """§10.5 part 2 -- the LEAK test, and the one a boundary test cannot see.

    Welcome To is concurrent and BGA reveals nothing until a turn resolves, so
    what seat 1 writes during its own turn must not move a single float of what
    seat 0 encodes.  A helper reaching for ``state.sheets[seat]`` instead of
    ``sheet_for(viewer, seat)`` passes the symmetry test above and fails here.

    The write is applied **directly to the live sheet**, leaving
    ``public_sheets`` alone.  Driving it with real actions makes the test
    skip-prone -- the turn resolves and the snapshot legitimately updates --
    and that is exactly the case where the test proves nothing.
    """
    rng = random.Random(4)
    state = GameState.new(seed=4, config=GameConfig(players=2, advanced=True))
    for _ in range(30):
        if state.is_terminal:
            break
        state = state.step(rng.choice(state.legal_actions()))
    assert not state.is_terminal

    before = enc.encode_state(state, 0)

    live = state.sheets[1]
    target = next(
        ((x, y) for x in range(enc.NUM_STREETS)
         for y in range(len(live.numbers[x]))
         if live.numbers[x][y] is None and live.available_locations(7)
         and (x, y) in live.available_locations(7)),
        None,
    )
    assert target is not None, "no legal write for seat 1; the test proves nothing"
    live.write(7, target, turn=state.turn)
    live.temps += 1
    live.parks[target[0]] = min(live.parks[target[0]] + 1, 3)

    after = enc.encode_state(state, 0)
    for name, a, b in zip(("planes", "sheets", "viewer", "global"), before, after):
        assert np.array_equal(a, b), f"{name} moved while seat 1 was mid-turn"


def test_encoding_does_not_mutate_the_state():
    """Both §10.5 tests are required to be mutation-checked.

    §9.2a top-fences a COPY to evaluate its kills, and §8 builds a hypothetical
    roundabout sheet; either would corrupt the caller's state if it reached for
    the real one.
    """
    from games.welcome_to.snapshot import to_snapshot

    for state in _states(seeds=range(3), plies=60):
        before = to_snapshot(state)
        enc.encode_state(state, state.actor)
        assert to_snapshot(state) == before, "encode_state mutated the state"
