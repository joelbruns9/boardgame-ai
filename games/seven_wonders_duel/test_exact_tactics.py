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


# --- G4b: Mausoleum retrievals expanded over every option --------------------


def _mausoleum_root(records):
    """A position whose mover can build the Mausoleum deterministically and
    would then choose among two or more discarded cards."""

    from .codec import decode_action, legal_action_indices
    from .engine import ActionUse, apply_action
    from .search import chance_signature

    for record in records:
        states = []
        replay(record, on_state=lambda game, _move: states.append(game.clone()))
        for state in states:
            if state.phase is not Phase.PLAY_AGE or state.pending_choice is not None:
                continue
            for index in legal_action_indices(state):
                action = decode_action(state, index)
                if (
                    action.use is not ActionUse.CONSTRUCT_WONDER
                    or action.wonder_name != "The Mausoleum"
                    or chance_signature(state, action)
                ):
                    continue
                child = state.clone()
                apply_action(child, action)
                pending = child.pending_choice
                if pending is not None and len(legal_action_indices(child)) >= 2:
                    return state, index, len(legal_action_indices(child))
    return None


def test_a_mausoleum_retrieval_is_expanded_over_every_option():
    found = _mausoleum_root(pe.fresh_bot_records(120, seed=2468))
    if found is None:
        pytest.skip("no deterministic Mausoleum build with 2+ retrievals in the fixture")
    root, action, options = found

    def run(enabled: bool):
        swr.set_exact_tactics(enabled)
        try:
            handle = swr.RustPuctSearch.open_mock(rust_game_from_state(root), 600, 9)
            handle.advance(600)
            return handle.tactics_metrics(), handle.follow_ups()
        finally:
            swr.set_exact_tactics(False)

    on, follow_on = run(True)
    off, _ = run(False)
    assert off["option_expansions"] == 0 and off["option_rows"] == 0
    assert on["option_expansions"] >= 1
    # The options were valued in the expanding request itself. (No per-node
    # bound from the ROOT's option count: deeper Mausoleum nodes see bigger
    # discard piles, and Library expansions count here too.)
    assert on["option_rows"] >= 1
    ranked = {root_action: follow for root_action, follow, _c in follow_on}
    if action in ranked:
        # Seeded options all carry a visit, so the ranking can name them.
        assert len(ranked[action]) >= min(options, 3)


# --- G8.0: Great Library token afterstates are shared across offers ----------


def _library_root(records):
    """A position whose mover can build the Great Library without a reveal."""

    from .codec import decode_action, legal_action_indices
    from .engine import ActionUse
    from .search import chance_signature
    from .game import ChanceKind

    for record in records:
        states = []
        replay(record, on_state=lambda game, _move: states.append(game.clone()))
        for state in states:
            if state.phase is not Phase.PLAY_AGE or state.pending_choice is not None:
                continue
            for index in legal_action_indices(state):
                action = decode_action(state, index)
                if (
                    action.use is ActionUse.CONSTRUCT_WONDER
                    and action.wonder_name == "The Great Library"
                ):
                    specs = chance_signature(state, action)
                    if all(spec.kind is ChanceKind.GREAT_LIBRARY_DRAW for spec in specs):
                        return state, index
    return None


def test_the_unused_token_pool_is_invisible_to_the_network(positions):
    """The safety argument for sharing: two positions that differ ONLY in which
    tokens went back to the box must encode identically, for either seat."""

    import numpy as np

    from .dataset import vectorize
    from .encoder import encode

    checked = 0
    for state in positions:
        if state.phase is Phase.COMPLETE or not state.unused_progress_tokens:
            continue
        stripped = state.clone()
        stripped.unused_progress_tokens = ()
        for seat in (0, 1):
            a = vectorize(encode(state.observation(seat)))
            b = vectorize(encode(stripped.observation(seat)))
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
        checked += 1
        if checked >= 50:
            break
    assert checked >= 10


def test_a_library_build_is_one_offer_node_over_the_whole_pool():
    """G4b Library: the 3-of-5 draw is not sampled. A reveal-free build has ONE
    child, an offer node holding all five tokens, expanded over every token and
    valued by the exact best-of-offer expectation (0.6 / 0.3 / 0.1)."""

    found = _library_root(pe.fresh_bot_records(120, seed=1357))
    if found is None:
        pytest.skip("no reveal-free Great Library build in the fixture")
    root, _action = found

    def run(enabled: bool):
        swr.set_exact_tactics(enabled)
        try:
            handle = swr.RustPuctSearch.open_mock(rust_game_from_state(root), 800, 4)
            handle.advance(800)
            return handle.tactics_metrics()
        finally:
            swr.set_exact_tactics(False)

    on, off = run(True), run(False)
    assert off["library_offer_nodes"] == 0
    # Reveal-free: the root's draw collapses to ONE offer node where the
    # sampled representation grows up to ten offer children. (Deeper lines
    # where the Library is built later add their own, hence >=.)
    assert on["library_offer_nodes"] >= 1
    assert on["option_expansions"] >= 1


def _reveal_positions(records):
    """`(state, previous_state)` for every position with a revealing action that
    leaves the opponent a forced win in some world (`losing_mass` > 0)."""

    found = []
    for record in records:
        previous = {}

        def visit(state, move, previous=previous):
            if state.phase is Phase.PLAY_AGE and state.pending_choice is None:
                masses = rust_game_from_state(state).losing_mass()
                if any(m is not None and m[1] and m[0] > 0.0 for m in masses):
                    found.append((state.clone(), previous.get("s")))
            previous["s"] = state.clone()

        replay(record, on_state=visit)
    return found


def test_reveal_strata_agree_with_the_losing_mass_reference():
    """G4 layer 2b: an interior reveal's proven worlds include exactly the
    worlds `losing_mass` (the G0 reference) finds, and their summed value is
    what those worlds plus any proven wins are worth."""

    positions = _reveal_positions(pe.fresh_bot_records(60, seed=8080))
    assert positions, "fixture has no reveal with forced-loss exposure"
    swr.set_exact_tactics(True)
    checked = 0
    for state, _previous in positions:
        game = rust_game_from_state(state)
        sign = 1.0 if state_actor(state) == 0 else -1.0
        for position, entry in enumerate(game.losing_mass()):
            if entry is None or not entry[1]:
                continue
            losing = entry[0]
            strata = game.reveal_strata(position)
            if strata is None:
                continue  # an over-cap (two-card) reveal stays sampled
            mass, value_p0, open_worlds = strata
            assert mass >= losing - 1e-9
            assert 0.0 < mass <= 1.0 + 1e-9
            # Every `losing_mass` world is in, at -1 for the mover. The other
            # proven worlds are worth +-1: the mover's wins, and its losses the
            # reference does not look for (an extra turn whose every follow-up
            # loses) -- a superset, never fewer.
            mover_value = sign * value_p0
            assert -mass - 1e-9 <= mover_value <= mass - 2.0 * losing + 1e-9
            assert (open_worlds == 0) == (mass >= 1.0 - 1e-9)
            checked += 1
    assert checked, "no reveal was stratified"


def test_interior_reveals_are_stratified_only_when_switched_on():
    positions = [p for _s, p in _reveal_positions(pe.fresh_bot_records(60, seed=8080)) if p]
    assert positions

    def run(strata: bool) -> int:
        swr.set_exact_tactics(True)
        swr.set_exact_reveal_strata(strata)
        try:
            total = 0
            for root in positions[:12]:
                handle = swr.RustPuctSearch.open_mock(rust_game_from_state(root), 400, 11)
                handle.advance(400)
                total += handle.tactics_metrics()["strata_edges"]
            return total
        finally:
            swr.set_exact_reveal_strata(True)

    assert run(True) > 0
    assert run(False) == 0
