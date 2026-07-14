"""Python/Rust parity gates for the frozen Step-3 NNUE reference encoder.

The Rust path is intentionally recomputing/stateless.  It becomes the reference
that the later incremental accumulator must match, so parity covers both player
frames, terminals, forced discards, and D4-transformed states before any delta
logic is introduced.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

import kingdomino_rust as kr

from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.game import GameConfig, GameState, Phase
from games.kingdomino.nnue import d4
from games.kingdomino.nnue.sparse_encoder import (
    CORE_SIZE,
    core_schema_hash,
    encode_core,
)
from games.kingdomino.nnue.summary_encoder import (
    SUMMARY_SIZE,
    encode_summary,
    summary_schema_hash,
)


def _rust(state: GameState):
    rs = _rust_state_from_python(state)
    assert rs is not None, "Python GameState did not convert to RustGameState"
    return rs


def _states(seeds=range(12)):
    for seed in seeds:
        state = GameState.new(seed=seed)
        rng = random.Random(seed * 17 + 3)
        while state.phase != Phase.GAME_OVER:
            yield state
            state = state.step(rng.choice(state.legal_actions()))
        yield state


def _assert_parity(state: GameState, perspective: int):
    got_sparse, got_summary = _rust(state).nnue_features(perspective)
    want_sparse = encode_core(state, perspective)
    want_summary = encode_summary(state, perspective)
    assert got_sparse.dtype == np.int32
    assert got_summary.dtype == np.float32
    assert np.array_equal(got_sparse, want_sparse), (
        f"sparse mismatch phase={state.phase} perspective={perspective}"
    )
    assert np.array_equal(got_summary, want_summary), (
        f"summary mismatch phase={state.phase} perspective={perspective}; "
        f"maxdiff={float(np.max(np.abs(got_summary - want_summary))):.9g}"
    )


def test_rust_schema_contract_matches_python():
    core_size, summary_size, core_hash, summary_hash = kr.nnue_schema_info()
    assert core_size == CORE_SIZE == 5710
    assert summary_size == SUMMARY_SIZE == 171
    assert core_hash == core_schema_hash()
    assert summary_hash == summary_schema_hash()


def test_rust_features_match_python_every_ply_both_perspectives():
    phases = set()
    saw_terminal = saw_discard = False
    for state in _states():
        phases.add(state.phase)
        saw_terminal |= state.phase == Phase.GAME_OVER
        saw_discard |= sum(state.discards) > 0
        _assert_parity(state, 0)
        _assert_parity(state, 1)
    assert phases == {
        Phase.INITIAL_SELECTION,
        Phase.PLACE_AND_SELECT,
        Phase.FINAL_PLACEMENT,
        Phase.GAME_OVER,
    }
    assert saw_terminal
    assert saw_discard, "random fixtures never exercised the discard feature"


def test_rust_features_obey_d4_maps():
    samples = []
    for state in _states(range(4)):
        if any(len(board.occupied_cells()) > 5 for board in state.boards):
            samples.append(state)
        if len(samples) == 8:
            break
    assert len(samples) == 8

    for state in samples:
        for perspective in (0, 1):
            base_sparse, base_summary = _rust(state).nnue_features(perspective)
            for k, flip in d4.D4_ELEMENTS:
                transformed = d4.transform_state(state, k, flip)
                got_sparse, got_summary = _rust(transformed).nnue_features(perspective)
                assert set(got_sparse.tolist()) == d4.apply_sparse(
                    base_sparse.tolist(), k, flip
                )
                assert np.array_equal(
                    got_summary,
                    d4.summary_perm(k, flip)(base_summary),
                )


def test_python_to_rust_converter_preserves_discards():
    state = next(s for s in _states(range(80)) if sum(s.discards) > 0)
    assert _rust(state).discards() == tuple(state.discards)
    _assert_parity(state, 0)
    _assert_parity(state, 1)


def test_rust_features_match_all_rules_variants():
    for harmony in (False, True):
        for middle in (False, True):
            config = GameConfig(harmony=harmony, middle_kingdom=middle)
            state = GameState.new(seed=31 + 2 * harmony + middle, config=config)
            rng = random.Random(900 + 2 * harmony + middle)
            while state.phase != Phase.GAME_OVER:
                _assert_parity(state, 0)
                _assert_parity(state, 1)
                state = state.step(rng.choice(state.legal_actions()))
            _assert_parity(state, 0)
            _assert_parity(state, 1)


def test_rust_encoder_rejects_bad_perspective():
    state = _rust(GameState.new(seed=0))
    for bad in (2, 255):
        with pytest.raises(ValueError, match="perspective"):
            state.nnue_features(bad)


def test_rust_encoder_rejects_bad_domino_id():
    state = kr.RustGameState(0, [999], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="domino id"):
        state.nnue_features(0)
