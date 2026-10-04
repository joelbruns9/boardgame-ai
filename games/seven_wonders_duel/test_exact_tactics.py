"""G4: the Rust proven-node predicate and its use in search.

`tactics.rs` is a port of `tactics.py`, so the Python module is the oracle: the
two must agree on every position, and every Rust positive must name a terminal
the right player actually wins. `tactics.forced_win` is checked to contain
`phase_e.guaranteed_win_now` (the narrower trap-suite predicate) and to find
extra-turn and civilian wins beyond it.
"""

from __future__ import annotations

from collections import Counter

import pytest

swr = pytest.importorskip("seven_wonders_rust")

from . import phase_e as pe
from . import tactics as tc
from .buffer import replay
from .game import Phase
from .rust_bridge import rust_game_from_state
from .search import state_actor


@pytest.fixture(scope="module")
def positions():
    """Every decision of science- and military-rush bot games: the curriculum
    bots drive both win conditions to the edge, so positives are common."""

    records = pe.fresh_bot_records(24, seed=4242)
    # Game 12 of seed 99 holds a win inside a PENDING choice (found by scanning
    # 200 games: 3 such positions in 11,566, all agreeing). The 24 above have
    # none, and the pending path is the one this predicate exists for.
    records.append(pe.fresh_bot_records(13, seed=99)[12])
    states = []
    for record in records:
        replay(record, on_state=lambda game, _move: states.append(game.clone()))
    return states


def test_rust_agrees_with_the_python_reference(positions):
    counts = Counter()
    mismatches = []
    for state in positions:
        if state.phase is Phase.COMPLETE:
            continue
        expected = tc.forced_win(state)
        narrow = pe.guaranteed_win_now(state)
        assert expected or not narrow  # the reference contains the trap predicate
        proven = rust_game_from_state(state).guaranteed_win_now()
        counts["positions"] += 1
        counts["wins"] += expected
        counts["beyond_narrow"] += expected and not narrow
        counts["pending_wins"] += expected and state.pending_choice is not None
        if (proven is not None) != expected:
            mismatches.append((state.age, state_actor(state), expected))
            continue
        if proven is not None:
            value, outlook = proven
            # Actor frame: the MOVER wins, whoever that is.
            assert value == (1.0 if state_actor(state) == 0 else -1.0)
            mover = state_actor(state)
            won = outlook[0:3] if mover == 0 else outlook[3:6]
            assert sum(won) == 1.0
    assert not mismatches, mismatches[:10]
    # The gate must not pass vacuously.
    assert counts["wins"] >= 10, counts
    assert counts["pending_wins"] >= 1, counts
    assert counts["beyond_narrow"] >= 1, counts


def test_military_positive_and_its_negative(positions):
    """Push the track to the mover's doorstep: a shield build then wins."""

    checked = 0
    for state in positions:
        if state.phase is not Phase.PLAY_AGE or state.pending_choice is not None:
            continue
        mover = state_actor(state)
        state = state.clone()
        state.conflict_position = 8 if mover == 0 else -8
        expected = tc.forced_win(state)
        assert (rust_game_from_state(state).guaranteed_win_now() is not None) == expected
        checked += expected
        if checked >= 5:
            break
    assert checked >= 5


def test_the_switch_is_off_by_default_and_round_trips():
    assert swr.exact_tactics() is False
    try:
        swr.set_exact_tactics(True)
        assert swr.exact_tactics() is True
    finally:
        swr.set_exact_tactics(False)


def _predecessor_of_a_proven_position(positions):
    """`(state, action)` whose action deterministically reaches a position the
    NEXT mover has a guaranteed win in -- the node search should mark proven."""

    from .search import chance_signature
    from .codec import decode_action, legal_action_indices
    from .engine import apply_action

    for state in positions:
        if state.phase is Phase.COMPLETE:
            continue
        for index in legal_action_indices(state):
            action = decode_action(state, index)
            if chance_signature(state, action):
                continue
            child = state.clone()
            apply_action(child, action)
            if child.phase is not Phase.COMPLETE and pe.guaranteed_win_now(child):
                return state, index, child
    raise AssertionError("no deterministic predecessor of a proven position")


def test_search_backs_up_the_exact_value_through_a_proven_child(positions):
    state, action, child = _predecessor_of_a_proven_position(positions)
    exact = 1.0 if state_actor(child) == 0 else -1.0

    def edge_mean(enabled: bool):
        swr.set_exact_tactics(enabled)
        try:
            handle = swr.RustPuctSearch.open_mock(rust_game_from_state(state), 400, 3)
            handle.advance(400)
            edges = {a: (v, s) for a, v, s, _p in handle.snapshot()[4]}
        finally:
            swr.set_exact_tactics(False)
        visits, value_sum = edges[action]
        assert visits > 0
        return value_sum / visits

    # On: every visit to that edge lands on the proven child and returns its
    # exact value -- no network estimate is ever averaged in.
    assert edge_mean(True) == pytest.approx(exact, abs=1e-12)
    # Off: the same edge is a network-valued subtree, not the exact result.
    assert edge_mean(False) != pytest.approx(exact, abs=1e-6)


# --- layer 1b: proven losses -------------------------------------------------


@pytest.fixture(scope="module")
def loss_positions(positions):
    """Played positions, plus each one with the track pushed to the OPPONENT's
    doorstep: proven losses are rare on played lines and common there."""

    states = list(positions[::2])
    for state in positions[::3]:
        if state.phase is not Phase.PLAY_AGE:
            continue
        pushed = state.clone()
        pushed.conflict_position = -7 if state_actor(pushed) == 0 else 7
        states.append(pushed)
    return states


def test_rust_loss_agrees_with_the_python_reference(loss_positions):
    counts = Counter()
    mismatches = []
    for state in loss_positions:
        if state.phase is Phase.COMPLETE:
            continue
        expected = tc.forced_loss(state)
        proven = rust_game_from_state(state).guaranteed_loss_now()
        counts["positions"] += 1
        counts["losses"] += expected
        if (proven is not None) != expected:
            mismatches.append((state.age, state_actor(state), expected))
            continue
        if proven is not None:
            value, outlook = proven
            # Actor frame: the MOVER loses.
            assert value == (-1.0 if state_actor(state) == 0 else 1.0)
            assert not tc.forced_win(state)
    assert not mismatches, mismatches[:10]
    assert counts["losses"] >= 10, counts


# --- layer 2: proofs propagate up the tree ----------------------------------


def _deterministic_children(state):
    from .codec import decode_action, legal_action_indices
    from .engine import apply_action
    from .search import chance_signature

    for index in legal_action_indices(state):
        action = decode_action(state, index)
        if chance_signature(state, action):
            continue
        child = state.clone()
        apply_action(child, action)
        if child.phase is not Phase.COMPLETE:
            yield index, child


def _two_level_win(positions):
    """`(root, action, mid)`: `action` leads deterministically to `mid`, whose
    mover has a move into a forced LOSS for the opponent -- a win two moves deep
    that no single-node predicate proves, only propagation."""

    for root in positions:
        if root.phase is not Phase.PLAY_AGE or root.age < 2:
            continue
        for index, mid in _deterministic_children(root):
            if tc.forced_win(mid) or tc.forced_loss(mid):
                continue
            for _j, leaf in _deterministic_children(mid):
                if state_actor(leaf) != state_actor(mid) and tc.forced_loss(leaf):
                    return root, index, mid
    raise AssertionError("no two-level win in the fixture")


def test_a_proof_two_levels_down_reaches_the_root_edge(positions):
    root, action, mid = _two_level_win(positions)
    exact = 1.0 if state_actor(mid) == 0 else -1.0
    swr.set_exact_tactics(True)
    try:
        handle = swr.RustPuctSearch.open_mock(rust_game_from_state(root), 3000, 5)
        handle.advance(3000)
        edges = {a: (v, s) for a, v, s, _p in handle.snapshot()[4]}
    finally:
        swr.set_exact_tactics(False)
    visits, value_sum = edges[action]
    assert visits > 0
    # The edge into `mid` reads the exact value: its statistics were reset
    # when `mid` was solved, and every later visit returned the proof.
    assert value_sum / visits == pytest.approx(exact, abs=1e-12)
