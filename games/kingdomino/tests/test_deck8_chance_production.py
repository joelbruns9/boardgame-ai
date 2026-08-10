"""Production BatchedMCTS integration tests for exhaustive deck=8 chance nodes."""

from __future__ import annotations

import numpy as np
import pytest

kingdomino_rust = pytest.importorskip("kingdomino_rust")


def test_batched_open_loop_commits_complete_deck8_panels_without_target_changes():
    """Drive real games over step/update with a deterministic mock evaluator.

    This exercises the Python-visible production boundary without depending on
    Torch. Policy logits and values are all zero; the test concerns panel
    admission, accounting, game recycling, and replay tuple compatibility.
    """

    mcts = kingdomino_rust.BatchedMCTS(
        2,
        2,
        20260810,
        100,
        leaf_batch=2,
        open_loop=True,
        dirichlet_eps=0.0,
        exact_endgame_max_secs=0.0,
        deck8_chance_enumeration=True,
    )
    finished = []
    max_batch = 0
    ticks = 0
    while not mcts.done():
        my, opp, flat, legal_indices = mcts.step()
        batch = int(my.shape[0])
        assert opp.shape[0] == batch
        assert flat.shape[0] == batch
        assert len(legal_indices) == batch
        max_batch = max(max_batch, batch)
        values = np.zeros(batch, dtype=np.float32)
        gathered = [np.zeros(len(indices), dtype=np.float32) for indices in legal_indices]
        finished.extend(mcts.update(values, gathered))
        ticks += 1
        assert ticks < 20_000

    panels = int(mcts.deck8_chance_panel_count)
    bootstrap_rows = int(mcts.deck8_chance_bootstrap_rows)
    assert len(finished) == 2
    assert panels > 0
    assert bootstrap_rows == 70 * panels
    assert max_batch >= 70

    # The learner-facing tuple remains unchanged and every recorded root policy
    # is a normalized distribution over real MCTS visits.
    for _seed, examples, _scores in finished:
        assert len(examples) == 52
        for example in examples:
            assert len(example) == 12
            policy = np.asarray(example[4], dtype=np.float64)
            assert policy.size > 0
            assert np.isclose(policy.sum(), 1.0, atol=1e-6)


def test_self_play_rejects_deck8_enumeration_on_closed_loop_engine():
    from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched

    cfg = SelfPlayConfig(
        engine="batched",
        deck8_chance_enumeration=True,
    )
    with pytest.raises(ValueError, match="requires engine='batched_open_loop'"):
        play_selfplay_games_batched(None, cfg, n_games=1, game_seed_start=1)
