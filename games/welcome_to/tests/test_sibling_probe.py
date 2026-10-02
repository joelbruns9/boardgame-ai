"""Sibling probe: candidate selection, paired futures, and decision metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from games.welcome_to import macro_codec as mc
from games.welcome_to import mcts, network as nw, rust_search, self_play
from games.welcome_to import sibling_probe as sp

wr = pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(3)
    net = nw.WelcomeToNet(_SMALL).eval()
    cfg = mcts.SearchConfig(simulations=2)
    packed = rust_search.PackedNetEvaluator(net, torch.device("cpu"), cfg)
    games, _ = self_play.generate(
        net,
        config=self_play.SelfPlayConfig(games=3, inflight=3, max_batch=3, seed=22_000),
        search_config=self_play.default_search_config(simulations=2),
        device="cpu",
    )
    roots = sp.select_roots(games, packed, max_candidates=5, seed=1)
    assert roots, "no eligible roots"
    sp.rollout_roots(roots, packed, cfg, futures=3, seed=9)
    return net, cfg, packed, games, roots


def test_candidates_are_the_played_move_then_same_slot_same_delta_boxes(setup):
    _net, _cfg, _packed, games, roots = setup
    assert {r.game_seed for r in roots} <= {g.seed for g in games}
    for root in roots:
        state = wr.RustGameState.from_snapshot(root.snapshot)
        assert state.actor == 0 and state.turn == root.turn
        assert root.candidates[0] == root.played
        assert len(set(root.candidates)) == len(root.candidates) >= 2
        legal = set(state.legal_macros())
        for candidate in root.candidates:
            assert candidate in legal
            assert sp._slot_delta(candidate) == sp._slot_delta(root.played)
        assert len(root.candidates) <= 5
    buckets = {}
    for root in roots:
        bucket = next(i for i, (lo, hi) in enumerate(sp.TURN_BUCKETS) if lo <= root.turn <= hi)
        key = (root.game_seed, bucket)
        assert key not in buckets, "at most one root per game and turn bucket"
        buckets[key] = root


def test_outcomes_are_complete_and_reproducible(setup):
    _net, cfg, packed, _games, roots = setup
    for root in roots:
        n = len(root.candidates)
        assert root.scores.shape == (n, 3, root.players)
        assert root.ranks.shape == (n, 3, 4)
        np.testing.assert_allclose(root.ranks.sum(-1), 1.0, rtol=1e-6)
        assert np.all(np.abs(root.blend) <= 1.0)
        assert len(root.afterstates) == n
    again = [sp.Root(**{**r.__dict__, "scores": None, "ranks": None, "seat_ranks": None, "blend": None, "afterstates": []}) for r in roots]
    sp.rollout_roots(again, packed, cfg, futures=3, seed=9, chunk_roots=1)
    for first, second in zip(roots, again):
        np.testing.assert_array_equal(first.scores, second.scores)


def test_saved_chunks_resume_identically_and_refuse_other_roots(setup, tmp_path):
    _net, cfg, packed, _games, roots = setup
    blank = lambda: [sp.Root(**{**r.__dict__, "scores": None, "ranks": None, "seat_ranks": None, "blend": None, "afterstates": []}) for r in roots]
    first = blank()
    sp.rollout_roots(first, packed, cfg, futures=3, seed=9, chunk_roots=2, checkpoint_dir=tmp_path)
    assert len(list(tmp_path.glob("chunk_*.pt"))) == (len(roots) + 1) // 2
    resumed = blank()
    calls = []
    original = sp._rollout_chunk
    sp._rollout_chunk = lambda *a, **k: calls.append(1)
    try:
        sp.rollout_roots(resumed, packed, cfg, futures=3, seed=9, chunk_roots=2, checkpoint_dir=tmp_path)
    finally:
        sp._rollout_chunk = original
    assert not calls, "a saved chunk was rolled out again"
    for a, b in zip(first, resumed):
        np.testing.assert_array_equal(a.scores, b.scores)
        np.testing.assert_array_equal(a.blend, b.blend)
    other = blank()
    other[0].candidates = list(reversed(other[0].candidates))
    with pytest.raises(ValueError, match="different roots"):
        sp.rollout_roots(other, packed, cfg, futures=3, seed=9, chunk_roots=2, checkpoint_dir=tmp_path)


def test_candidates_in_one_future_share_the_reshuffled_deck(setup):
    """Paired randomness: within a future every candidate sees the same deck."""
    _net, _cfg, _packed, _games, roots = setup
    root = roots[0]
    base = wr.RustGameState.from_snapshot(root.snapshot)
    seed = 12345
    decks = []
    for candidate in root.candidates:
        drawn, _ = base.redeterminize(seed)
        decks.append(drawn.step_macro(candidate).snapshot()["deck"])
    assert all(deck == decks[0] for deck in decks)


def test_decision_metrics_on_synthetic_truth():
    truth = np.array([[0.1] * 4 + [0.1] * 4, [0.5] * 4 + [0.5] * 4, [0.3] * 4 + [0.3] * 4])
    scores = np.zeros((3, 8, 2), dtype=np.float32)
    scores[:, :, 0] = np.array([10.0, 30.0, 20.0])[:, None]
    root = {"blend": truth, "scores": scores}
    perfect = sp.score_decisions([np.array([0.0, 2.0, 1.0])], [root], fit=4)
    assert perfect["regret"]["n"] == 1
    worst = sp.score_decisions([np.array([2.0, 0.0, 1.0])], [root], fit=4)
    assert worst["pair_sign_accuracy"]["mean"] == 0.0
    best = sp.score_decisions([np.array([0.0, 2.0, 1.0]), np.array([0.0, 2.0, 1.0])], [root, root], fit=4)
    assert best["regret"]["mean"] == 0.0
    assert best["pair_sign_accuracy"]["mean"] == 1.0
    assert best["regret_played"]["mean"] == pytest.approx(0.4)
    assert best["score_regret"]["mean"] == 0.0
    assert best["score_regret_played"]["mean"] == pytest.approx(20.0)
    assert best["regret_fit_oracle"]["mean"] == 0.0
