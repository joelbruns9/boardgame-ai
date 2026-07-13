"""Step 2 gate: validate the Rust `SearchEngine` (Python-drivable make/unmake)
against the functional `RustGameState.step` oracle across the full PUBLIC game
state, plus `official_outcome` vs the Python `determine_winner` cascade and the
`make_with_row` chance-child mechanics (happy path AND input validation).

`_fp` compares phase, all scalars, the deck/row/claim vectors, discards, scores,
and BOTH boards' terrain+crowns — the full public state. (The board bbox /
`occupied` are internal and reflected in `scores()`; the byte-level make==step
guarantee is the Step-1 Rust differential test `make_unmake_tests`.)
"""
from __future__ import annotations

import itertools
import random

import pytest

import kingdomino_rust as kr

from games.kingdomino.game import Phase, determine_winner
from games.kingdomino.bots import RandomBot
from games.kingdomino.bot_match import play_bot_game
from games.kingdomino.endgame_solver import _rust_state_from_python

GAME_OVER = int(Phase.GAME_OVER)  # 3
PLACE = int(Phase.PLACE_AND_SELECT)  # 1


def _fp(s: "kr.RustGameState"):
    """Full public-state fingerprint of a RustGameState via its getters."""
    return (
        s.phase,
        s.actor_index,
        s.initial_pick_count,
        s.start_player,
        tuple(s.deck()),
        tuple(s.current_row()),
        tuple(s.pending_claims()),
        tuple(s.next_claims()),
        s.discards(),
        s.scores(),
        tuple(s.board_terrain(0)),
        tuple(s.board_crowns(0)),
        tuple(s.board_terrain(1)),
        tuple(s.board_crowns(1)),
    )


def test_make_matches_step_and_unwinds():
    """Driving `SearchEngine.make` action-for-action with the functional
    `RustGameState.step` yields identical full public state at every ply;
    unwinding the whole stack via `unmake` returns to the exact start."""
    for seed in range(40):
        rng = random.Random(seed ^ 0x515)
        base = kr.batched_new_game(seed, True, True)
        start = _fp(base)
        func = base  # functional path (reassigned each step; base is immutable)
        eng = kr.SearchEngine(base)
        plies = 0
        for _ in range(400):
            if func.phase == GAME_OVER:
                break
            legal = func.legal_actions()
            p, pk = rng.choice(legal)
            stepped = func.step(p, pk)
            eng.make(p, pk)
            assert _fp(eng.snapshot()) == _fp(stepped), (
                f"seed {seed} ply {plies}: make diverged from step"
            )
            func = stepped
            plies += 1
        assert eng.phase == GAME_OVER, f"seed {seed}: did not reach GAME_OVER"
        assert eng.depth() == plies
        for _ in range(plies):
            eng.unmake()
        assert eng.depth() == 0
        assert _fp(eng.snapshot()) == start, f"seed {seed}: unwind != start"


def test_official_outcome_matches_determine_winner():
    """`SearchEngine.official_outcome` agrees with the authoritative Python
    `determine_winner` cascade at real terminals (both wins observed)."""
    seen_p0 = seen_p1 = 0
    for seed in range(60):
        _, state = play_bot_game(seed=seed, bot0=RandomBot(), bot1=RandomBot())
        assert state.phase == Phase.GAME_OVER
        rs = _rust_state_from_python(state)
        assert rs is not None, f"seed {seed}: could not build RustGameState"
        eng = kr.SearchEngine(rs)
        got = eng.official_outcome()
        w = determine_winner(state)
        expect = 0 if w is None else (1 if w == 0 else -1)
        assert got == expect, f"seed {seed}: outcome {got} != expected {expect}"
        seen_p0 += expect == 1
        seen_p1 += expect == -1
    assert seen_p0 and seen_p1, f"lopsided sample: p0={seen_p0} p1={seen_p1}"


def _first_boundary(last_actor: bool, seed_base: int = 0, max_seed: int = 160):
    """Find a PLACE_AND_SELECT state with deck>=8. `last_actor=True` returns a
    round boundary (next move DEALS); False returns a mid-round state (does not).
    Returns (RustGameState, first_legal_action, sorted_pre_deck)."""
    for seed in range(seed_base, seed_base + max_seed):
        rng = random.Random(seed * 5 + 9)
        s = kr.batched_new_game(seed, True, True)
        for _ in range(400):
            if s.phase == GAME_OVER:
                break
            n_pending = len(s.pending_claims())
            is_last = s.actor_index == n_pending - 1
            if s.phase == PLACE and len(s.deck()) >= 8 and is_last == last_actor:
                return s, s.legal_actions()[0], sorted(s.deck())
            p, pk = rng.choice(s.legal_actions())
            s = s.step(p, pk)
    raise AssertionError(f"no {'boundary' if last_actor else 'mid-round'} state found")


def test_make_with_row_installs_chance_child_and_unwinds():
    """At a round boundary, `make_with_row` installs a chosen enumerated draw as
    the new row (deck = pre-deal bag minus row, both sorted); `unmake` restores."""
    rs, (p, pk), pre_deck = _first_boundary(last_actor=True)
    eng = kr.SearchEngine(rs)
    before = _fp(eng.snapshot())
    row = list(itertools.combinations(pre_deck, 4))[0]
    eng.make_with_row(p, pk, list(row))
    assert tuple(eng.current_row()) == tuple(sorted(row))
    resid = pre_deck.copy()
    for r in row:
        resid.remove(r)
    assert tuple(eng.deck()) == tuple(sorted(resid))
    eng.unmake()
    assert _fp(eng.snapshot()) == before, "unmake did not restore the pre-boundary state"


def test_make_with_row_rejects_bad_rows_without_mutating():
    """Malformed rows (wrong length, empty, duplicate, alien tile) and a row for
    a NON-dealing action are all rejected with the engine and undo depth
    unchanged — the foundational-infra guarantee."""
    rs, (p, pk), pre_deck = _first_boundary(last_actor=True)
    eng = kr.SearchEngine(rs)
    before, d0 = _fp(eng.snapshot()), eng.depth()
    good = list(itertools.combinations(pre_deck, 4))[0]
    bad_rows = [
        [],                                      # empty
        list(good[:3]),                          # too short
        list(good) + [good[0]],                  # too long
        [good[0], good[0], good[1], good[2]],    # duplicate tile
        [good[0], good[1], good[2], 9999],       # alien tile not in bag
    ]
    for bad in bad_rows:
        with pytest.raises(Exception):
            eng.make_with_row(p, pk, bad)
        assert eng.depth() == d0, f"undo depth changed after rejecting {bad}"
        assert _fp(eng.snapshot()) == before, f"state changed after rejecting {bad}"

    # A well-formed row supplied for an action that does NOT deal is also rejected
    # (never silently keeps the hidden-order actual deal).
    rs2, (p2, pk2), pre_deck2 = _first_boundary(last_actor=False, seed_base=1)
    eng2 = kr.SearchEngine(rs2)
    before2, d2 = _fp(eng2.snapshot()), eng2.depth()
    valid = list(list(itertools.combinations(pre_deck2, 4))[0])
    with pytest.raises(Exception):
        eng2.make_with_row(p2, pk2, valid)
    assert eng2.depth() == d2, "undo depth changed after non-dealing rejection"
    assert _fp(eng2.snapshot()) == before2, "state changed after non-dealing rejection"
