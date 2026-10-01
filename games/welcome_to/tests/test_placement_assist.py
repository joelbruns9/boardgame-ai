"""Placement assist as a phased-out generation scaffold."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import placement_assist as pa
from games.welcome_to import s2_run, self_play
from games.welcome_to.game import GameState

wr = pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(
    sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16
)
THROUGH = 8


def test_assignment_is_deterministic_honours_the_fraction_and_skips_restarts():
    config = self_play.SelfPlayConfig(games=2_000, seed=12_000)
    first = self_play.assisted_games(config, 0.3, 16)
    assert first == self_play.assisted_games(config, 0.3, 16)
    assert 0.26 < len(first) / config.games < 0.34
    assert set(first.values()) == {16}
    excluded = set(list(first)[:50])
    assert not excluded & set(self_play.assisted_games(config, 0.3, 16, exclude=excluded))
    assert self_play.assisted_games(config, 0.0, 16) == {}


def test_the_driver_phases_assistance_out_by_the_end_iteration():
    args = argparse.Namespace(assist_fraction=0.5, assist_end_iteration=6)
    shares = [s2_run.assist_fraction(i, args) for i in range(1, 9)]
    assert shares == pytest.approx([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0])
    off = argparse.Namespace(assist_fraction=0.0, assist_end_iteration=6)
    assert s2_run.assist_fraction(1, off) == 0.0


@pytest.fixture(scope="module")
def assisted_run(tmp_path_factory):
    torch.manual_seed(31)
    net = nw.WelcomeToNet(_SMALL).eval()
    prefix = tmp_path_factory.mktemp("assist") / "trajectories.jsonl"
    config = self_play.SelfPlayConfig(
        games=8, inflight=8, max_batch=8, seed=13_300, opening_temperature_turns=2
    )
    plan = self_play.assisted_games(config, 0.5, THROUGH)
    assert 0 < len(plan) < config.games
    writer = wr.RustSampleShardWriter(prefix, shard_games=4, queue_games=4)
    try:
        trajectories, metrics = self_play.generate(
            net,
            config=config,
            search_config=self_play.default_search_config(simulations=3),
            device="cpu",
            on_captured=lambda _t, captured: writer.add(captured),
            assisted=plan,
        )
    finally:
        writer.close()
    return trajectories, metrics, plan, prefix, config


def _learner_writes(trajectory, through):
    """Learner writes through ``through``; each must be its own assistance."""
    state = GameState.new(seed=trajectory.engine_seed, config=trajectory.config, rng_kind=trajectory.rng)
    count = 0
    for action in trajectory.actions:
        if pa.is_write(action) and state.actor == 0 and state.turn <= through:
            assert pa.assisted_choice(state, action) == action
            count += 1
        mc.apply_macro(state, action)
    return count


def test_assisted_games_are_recorded_and_only_the_learner_is_assisted(assisted_run):
    trajectories, metrics, plan, _prefix, config = assisted_run
    assert {t.seed: t.assist_through for t in trajectories if t.assist_through} == plan
    writes = sum(_learner_writes(t, THROUGH) for t in trajectories if t.assist_through)
    assert metrics["assisted_decisions"] == writes > 0
    assert metrics["assisted_games"] == len(plan)
    assert metrics["natural_games"] == config.games - len(plan)
    for trajectory in trajectories:
        if trajectory.assist_through:
            assert trajectory.searches, "assisted games still carry search targets"


def test_assisted_rust_rows_are_exactly_the_python_oracle(assisted_run):
    trajectories, _metrics, _plan, prefix, _config = assisted_run
    cached = {t.seed: t for t in self_play.read_trajectories(prefix)}
    checked = 0
    for trajectory in trajectories:
        if not trajectory.assist_through:
            continue
        oracle = list(self_play.replay(self_play.SelfPlayTrajectory.from_json(trajectory.to_json())))
        rows = list(self_play.replay(cached[trajectory.seed]))
        assert len(rows) == len(oracle) == len(trajectory.searches)
        for actual, expected in zip(rows, oracle):
            assert np.array_equal(actual.policy, expected.policy)
            assert np.array_equal(actual.sheet_planes, expected.sheet_planes)
            for name in actual.targets:
                assert np.array_equal(
                    np.asarray(actual.targets[name], dtype=np.float32),
                    np.asarray(expected.targets[name], dtype=np.float32),
                ), name
            checked += 1
    assert checked > 0


def test_resume_refuses_a_corpus_that_disagrees_with_the_assist_plan(assisted_run):
    trajectories, _metrics, plan, _prefix, config = assisted_run
    assert self_play.validate_resume(trajectories, config, {}, plan)
    with pytest.raises(ValueError, match="assist plan"):
        self_play.validate_resume(trajectories, config, {}, {})


def test_a_curriculum_game_is_never_assisted():
    config = self_play.SelfPlayConfig(games=4, seed=1)
    with pytest.raises(ValueError, match="never placement-assisted"):
        self_play.SelfPlayTrajectory(
            seed=1, players=2, actions=(1, 2, 3), searches=(), scores=(0, 0),
            opponents=("a", "b"),
            restart=self_play.curriculum.Restart(source_seed=1, at=1, reshuffle_seed=1, distance=1, slot=0),
            assist_through=16,
        )
