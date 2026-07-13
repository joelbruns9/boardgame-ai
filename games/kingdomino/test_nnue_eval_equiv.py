"""Step 2b gate: the Rust `NnueEvaluator` (dense forward over the ported encoder)
must match the PyTorch net it was exported from, on REAL encoded positions.

Covers the reviewer's 2b checklist: PyTorch-vs-Rust forward equivalence on real
positions; actor->P0 frame equivalence for BOTH actors; explicit 2*sigmoid-1
expected-outcome conversion; a separate check of the normalized margin output;
finite outputs; and load-error handling. Isolates the FORWARD pass by feeding both
sides the SAME Rust-encoded features (the encoder is bit-exact per its own gate).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

import kingdomino_rust as kr

from games.kingdomino.game import GameState, Phase
from games.kingdomino.endgame_solver import _rust_state_from_python
from games.kingdomino.nnue.net import TwoHeadNNUE

CKPT = "games/kingdomino/nnue/checkpoints/dense_v1.pt"
KNNUE = "games/kingdomino/nnue/checkpoints/dense_v1.knnue"

pytestmark = pytest.mark.skipif(
    not (Path(CKPT).exists() and Path(KNNUE).exists()),
    reason="train + export the dense net first (nnue.train, nnue.export)",
)


def _load_net():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = TwoHeadNNUE(**ck["config"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, float(ck["margin_scale"])


def _collect_states(want_per_actor=6):
    """PLACE_AND_SELECT positions with their current actor, ensuring BOTH actors are
    represented (the frame flip differs by actor)."""
    by_actor = {0: [], 1: []}
    seed = 0
    while (len(by_actor[0]) < want_per_actor or len(by_actor[1]) < want_per_actor) and seed < 600:
        st = GameState.new(seed=seed)
        rng = random.Random(seed * 5 + 1)
        for _ in range(400):
            if st.phase == Phase.GAME_OVER:
                break
            if st.phase == Phase.PLACE_AND_SELECT and len(st.legal_actions()) >= 2:
                rs = _rust_state_from_python(st)
                if rs is not None:
                    actor = kr.SearchEngine(rs).current_actor()
                    if len(by_actor[actor]) < want_per_actor:
                        by_actor[actor].append((st, rs, actor))
            st = st.step(rng.choice(st.legal_actions()))
        seed += 1
    return by_actor[0] + by_actor[1], by_actor


def test_rust_forward_matches_pytorch_both_actors():
    net, margin_scale = _load_net()
    ev = kr.NnueEvaluator(KNNUE)
    states, by_actor = _collect_states()
    assert by_actor[0] and by_actor[1], "need positions with both actors to move"

    for pystate, rs, actor in states:
        mb, ob, flat = rs.encode(actor)  # same bit-exact encoder the Rust eval uses
        x = np.concatenate([mb.reshape(-1), ob.reshape(-1), flat]).astype(np.float32)
        with torch.no_grad():
            logit, margin_norm = net(torch.from_numpy(x)[None])
        expected_py = float(torch.sigmoid(logit))
        margin_pts_py = float(margin_norm) * margin_scale
        actor_val = 2.0 * expected_py - 1.0
        p0_py = actor_val if actor == 0 else -actor_val

        p0_r, expected_r, margin_pts_r = ev.evaluate(rs)

        assert np.isfinite([p0_r, expected_r, margin_pts_r]).all(), "non-finite Rust output"
        assert abs(expected_r - expected_py) < 1e-3, (
            f"expected_score rust {expected_r} vs py {expected_py} (actor {actor})")
        assert abs(p0_r - p0_py) < 2e-3, f"p0 rust {p0_r} vs py {p0_py} (actor {actor})"
        assert abs(margin_pts_r - margin_pts_py) < 2e-2, (
            f"margin rust {margin_pts_r} vs py {margin_pts_py} (actor {actor})")


def test_frame_flip_internal_consistency():
    """p0_value = (+1 if actor==P0 else -1) * (2*expected_score - 1), for both actors."""
    ev = kr.NnueEvaluator(KNNUE)
    _, by_actor = _collect_states()
    for actor in (0, 1):
        for _pystate, rs, a in by_actor[actor]:
            p0_r, expected_r, _ = ev.evaluate(rs)
            sign = 1.0 if a == 0 else -1.0
            # Rust computes 2*expected-1 in f32 then widens to f64, so the recomputed
            # f64 value agrees only to f32 precision (~1e-7), not bit-for-bit.
            assert abs(p0_r - sign * (2.0 * expected_r - 1.0)) < 1e-6
            assert -1.0 <= p0_r <= 1.0 and 0.0 <= expected_r <= 1.0


def test_search_with_nnue_eval_runs():
    """RustSearch(eval='nnue') loads the weights and returns a finite root value."""
    states, _ = _collect_states(want_per_actor=1)
    rs = states[0][1]
    search = kr.RustSearch(depth=2, eval="nnue", nnue_path=KNNUE)
    v = search.value(rs, 2)
    assert np.isfinite(v) and -1.5 <= v <= 1.5
    assert search.nodes > 0
    a = search.choose_action(rs)
    assert a is not None


def test_load_errors():
    with pytest.raises(Exception):
        kr.NnueEvaluator("does_not_exist.knnue")
    with pytest.raises(ValueError):
        kr.RustSearch(depth=2, eval="nnue")  # missing nnue_path
    with pytest.raises(ValueError):
        kr.RustSearch(depth=2, eval="pick_aware", nnue_path=KNNUE)  # path but not nnue
