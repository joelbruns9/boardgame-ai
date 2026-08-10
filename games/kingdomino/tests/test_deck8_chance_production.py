"""Production BatchedMCTS integration tests for exhaustive deck=8 chance nodes."""

from __future__ import annotations

import numpy as np
import pytest

kingdomino_rust = pytest.importorskip("kingdomino_rust")


def _state_sensitive_values(my, opp, flat):
    """Cheap deterministic stand-in for a network value head.

    The coefficients deliberately touch every encoded feature.  Equal encoded
    states therefore receive bit-identical values while distinct reveal rows
    normally receive different values.
    """
    arrays = [np.asarray(x, dtype=np.float64).reshape(len(flat), -1)
              for x in (my, opp, flat)]
    signal = np.zeros(len(flat), dtype=np.float64)
    for scale, array in zip((0.7, -0.4, 1.1), arrays):
        weights = np.linspace(-1.0, 1.0, array.shape[1], dtype=np.float64)
        signal += scale * (array @ weights) / max(1, array.shape[1])
    return np.tanh(signal).astype(np.float32)


def _finish_with_zero_evaluator(mcts):
    finished = []
    while not mcts.done():
        my, _opp, _flat, legal_indices = mcts.step()
        values = np.zeros(len(my), dtype=np.float32)
        gathered = [np.zeros(len(indices), dtype=np.float32)
                    for indices in legal_indices]
        finished.extend(mcts.update(values, gathered))
    return finished


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


def test_production_panel_mean_uses_all_nonconstant_rows_and_only_one_panel_per_move():
    """Exercise real step/update panel commitment with nonconstant leaf values.

    This catches row-order/value-frame mistakes that an all-zero evaluator
    cannot detect.  It also follows the same live tree until the move finishes
    and verifies the per-move admission cap directly.
    """
    mcts = kingdomino_rust.BatchedMCTS(
        1,
        1,
        20260811,
        400,
        leaf_batch=6,
        open_loop=True,
        dirichlet_eps=0.0,
        exact_endgame_max_secs=0.0,
        deck8_chance_enumeration=True,
    )
    saw_panel = False
    saw_panel_retire = False
    ticks = 0
    while not mcts.done() and not saw_panel_retire:
        my, opp, flat, legal_indices = mcts.step()
        values = _state_sensitive_values(my, opp, flat)
        actors = np.asarray(mcts.row_actors(), dtype=np.int64)
        values_p0 = np.where(actors == 0, values, -values).astype(np.float64)
        gathered = [np.zeros(len(indices), dtype=np.float32)
                    for indices in legal_indices]

        panel_rows = int(mcts.deck8_chance_bootstrap_rows)
        expected_panel_mean = None
        if len(values) >= 70 and panel_rows == 0:
            candidate = values_p0[-70:]
            if np.ptp(candidate) > 1e-6:
                expected_panel_mean = float(candidate.mean())

        mcts.update(values, gathered)
        panels = mcts.debug_ol_panel_values(0)
        assert len(panels) <= 1
        if panels:
            if not saw_panel:
                assert expected_panel_mean is not None
                _node_id, actual_mean, visits, support = panels[0]
                assert support == 70
                assert visits >= 0
                assert actual_mean == pytest.approx(expected_panel_mean, abs=1e-9)
                saw_panel = True
        elif saw_panel:
            # finalize_move replaced the tree, which is also where the
            # one-panel admission guard is reset for the next real move.
            saw_panel_retire = True

        ticks += 1
        assert ticks < 10_000

    assert saw_panel
    assert saw_panel_retire


def test_self_play_rejects_deck8_enumeration_on_closed_loop_engine():
    from games.kingdomino.self_play import SelfPlayConfig, play_selfplay_games_batched

    cfg = SelfPlayConfig(
        engine="batched",
        deck8_chance_enumeration=True,
    )
    with pytest.raises(ValueError, match="requires engine='batched_open_loop'"):
        play_selfplay_games_batched(None, cfg, n_games=1, game_seed_start=1)


def test_hof_rejects_closed_loop_and_forces_chance_mode_off(monkeypatch):
    import games.kingdomino.self_play as self_play

    closed_cfg = self_play.SelfPlayConfig(
        engine="batched",
        deck8_chance_enumeration=True,
    )
    with pytest.raises(ValueError, match="requires engine='batched_open_loop'"):
        self_play.play_hof_games_batched(
            None, None, closed_cfg, n_games=1, game_seed_start=1
        )

    constructor_args = []

    class FakeBatchedMCTS:
        def __init__(self, *args, **kwargs):
            constructor_args.append(kwargs)

    monkeypatch.setattr(kingdomino_rust, "BatchedMCTS", FakeBatchedMCTS)
    monkeypatch.setattr(self_play, "make_rust_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(self_play, "_run_hof_orientation", lambda *args, **kwargs: [])
    open_cfg = self_play.SelfPlayConfig(
        engine="batched_open_loop",
        deck8_chance_enumeration=True,
    )
    self_play.play_hof_games_batched(
        None, None, open_cfg, n_games=1, game_seed_start=1
    )
    assert constructor_args
    assert all(args["deck8_chance_enumeration"] is False for args in constructor_args)


def test_evaluation_seat_filter_partitions_the_both_seats_panel_count():
    def run(selector):
        mcts = kingdomino_rust.BatchedMCTS(
            1,
            1,
            20260812,
            100,
            leaf_batch=2,
            open_loop=True,
            dirichlet_eps=0.0,
            exact_endgame_max_secs=0.0,
            deck8_chance_enumeration=True,
            deck8_chance_enumeration_seat=selector,
        )
        assert len(_finish_with_zero_evaluator(mcts)) == 1
        return int(mcts.deck8_chance_panel_count)

    both = run(-1)
    seat0 = run(0)
    seat1 = run(1)
    assert seat0 > 0
    assert seat1 > 0
    assert both == seat0 + seat1
