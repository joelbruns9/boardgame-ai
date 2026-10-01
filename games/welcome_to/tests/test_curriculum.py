"""Near-completion plan curriculum: rewind points, restart plans, and the
restarted games' replay on both engines."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from games.welcome_to import curriculum
from games.welcome_to import network as nw
from games.welcome_to import self_play

wr = pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(
    sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16
)


def _net(seed: int = 23) -> nw.WelcomeToNet:
    torch.manual_seed(seed)
    return nw.WelcomeToNet(_SMALL).eval()


def _random_source(seed: int, players: int) -> self_play.SelfPlayTrajectory:
    """An ordinary finished game played by random macros, as a source."""
    rng = random.Random(seed)
    state = wr.RustGameState(seed, players=players, advanced=True, expert=False, solo_rules=False)
    actions = []
    while not state.is_terminal:
        action = rng.choice(state.legal_macros())
        actions.append(action)
        state.apply_macro(action)
    return self_play.SelfPlayTrajectory(
        seed=seed,
        players=players,
        actions=tuple(actions),
        searches=(),
        scores=tuple(state.scores()),
        opponents=("learner",) + ("random",) * (players - 1),
    )


@pytest.fixture(scope="module")
def sources():
    games = [_random_source(70_000 + 10 * i + players, players) for i in range(60) for players in (2, 3)]
    return games


@pytest.fixture(scope="module")
def pool(sources):
    found = curriculum.candidates(sources)
    assert len(found) >= 10, "random sources gave too few learner completions"
    return found


def test_every_rewind_point_is_a_learner_turn_start_k_turns_before_its_plan(sources, pool):
    by_seed = {t.seed: t for t in sources}
    for candidate in pool:
        source = by_seed[candidate.source_seed]
        state = wr.RustGameState(
            source.seed, players=source.players, advanced=True, expert=False, solo_rules=False
        )
        for action in candidate.prefix:
            state.apply_macro(action)
        restart_turn = state.turn
        assert state.actor == 0
        assert candidate.prefix == source.actions[: candidate.at]
        # the previous decision belonged to an earlier turn: a turn start
        before = wr.RustGameState(
            source.seed, players=source.players, advanced=True, expert=False, solo_rules=False
        )
        for action in candidate.prefix[:-1]:
            before.apply_macro(action)
        assert before.turn < restart_turn
        for action in source.actions[candidate.at :]:
            state.apply_macro(action)
        completed = [t for seat, t in state.plan_turns_for(0, candidate.slot) if seat == 0]
        assert completed == [restart_turn + candidate.distance]
    assert {c.distance for c in pool} == set(curriculum.DEFAULT_DISTANCES)


def test_restart_plans_are_deterministic_keep_seat_counts_and_honour_the_fraction(pool):
    jobs = list(zip(range(500, 600), self_play.seat_counts(100)))
    first = curriculum.plan_restarts(jobs, pool, fraction=0.2, seed=500)
    second = curriculum.plan_restarts(list(reversed(jobs)), list(reversed(pool)), fraction=0.2, seed=500)
    assert first == second
    assert curriculum.plan_digest(first) == curriculum.plan_digest(second)
    players = dict(jobs)
    pool_players = {c.players for c in pool}
    eligible = sum(1 for _, n in jobs if n in pool_players)
    assert 0 < len(first) <= 20 and len(first) <= eligible
    by_source = {(c.source_seed, c.at): c for c in pool}
    for job_seed, (restart, prefix) in first.items():
        source = by_source[(restart.source_seed, restart.at)]
        assert source.players == players[job_seed]
        assert prefix == source.prefix
    assert curriculum.plan_restarts(jobs, pool, fraction=0.0, seed=500) == {}
    assert curriculum.plan_restarts(jobs, [], fraction=0.5, seed=500) == {}


def test_a_restart_round_trips_through_json_and_guards_its_prefix():
    source = _random_source(71_000, 2)
    restart = curriculum.Restart(
        source_seed=source.seed, at=10, reshuffle_seed=99, distance=2, slot=1
    )
    trajectory = self_play.SelfPlayTrajectory(
        seed=5, players=2, actions=source.actions, searches=(),
        scores=source.scores, opponents=source.opponents, restart=restart,
    )
    assert self_play.SelfPlayTrajectory.from_json(trajectory.to_json()) == trajectory
    assert trajectory.engine_seed == source.seed
    with pytest.raises(ValueError, match="prefix"):
        self_play.SelfPlayTrajectory(
            seed=5, players=2, actions=source.actions,
            searches=(self_play.SearchTarget(decision=3, actions=(1,), visits=(1,)),),
            scores=source.scores, opponents=source.opponents, restart=restart,
        )


@pytest.fixture(scope="module")
def restarted(tmp_path_factory, pool):
    prefix = tmp_path_factory.mktemp("curriculum") / "trajectories.jsonl"
    config = self_play.SelfPlayConfig(
        games=8, inflight=8, max_batch=8, seed=9_100, opening_temperature_turns=2
    )
    jobs = list(zip(range(config.seed, config.seed + config.games), self_play.seat_counts(config.games)))
    plan = curriculum.plan_restarts(jobs, pool, fraction=0.5, seed=config.seed)
    assert plan, "no restart could be planned"
    writer = wr.RustSampleShardWriter(prefix, shard_games=4, queue_games=4)
    try:
        trajectories, metrics = self_play.generate(
            _net(),
            config=config,
            search_config=self_play.default_search_config(simulations=3),
            device="cpu",
            on_captured=lambda _t, captured: writer.add(captured),
            restarts=plan,
        )
    finally:
        writer.close()
    return trajectories, metrics, plan, prefix, config


def test_restarted_games_start_from_their_source_prefix(restarted):
    trajectories, metrics, plan, _prefix, config = restarted
    restarts = [t for t in trajectories if t.restart is not None]
    assert len(restarts) == len(plan)
    for trajectory in restarts:
        point, prefix = plan[trajectory.seed]
        assert trajectory.restart == point
        assert trajectory.actions[: point.at] == prefix
        assert all(target.decision >= point.at for target in trajectory.searches)
        assert trajectory.searches, "the learner searched nothing after the restart"
    assert metrics["curriculum_games"] == len(plan)
    assert metrics["natural_games"] == config.games - len(plan)
    assert 0.0 <= metrics["curriculum_source_plan_rate"] <= 1.0


def test_restarted_rust_rows_are_exactly_the_python_oracle(restarted):
    """The Python replay rebuilds a restart from source_seed and redeterminize;
    the Rust capture did the same. Encodings AND targets must agree exactly."""
    trajectories, _metrics, _plan, prefix, _config = restarted
    cached = {t.seed: t for t in self_play.read_trajectories(prefix)}
    checked = 0
    for trajectory in trajectories:
        if trajectory.restart is None:
            continue
        oracle = list(self_play.replay(self_play.SelfPlayTrajectory.from_json(trajectory.to_json())))
        rows = list(self_play.replay(cached[trajectory.seed]))
        assert len(rows) == len(oracle) == len(trajectory.searches)
        for actual, expected in zip(rows, oracle):
            for name in ("sheet_planes", "sheet_scalars", "viewer_plane", "global_scalars", "legal", "policy"):
                assert np.array_equal(getattr(actual, name), getattr(expected, name)), name
            assert actual.targets.keys() == expected.targets.keys()
            for name in actual.targets:
                assert np.array_equal(
                    np.asarray(actual.targets[name], dtype=np.float32),
                    np.asarray(expected.targets[name], dtype=np.float32),
                ), name
            checked += 1
    assert checked > 0


def test_resume_refuses_a_corpus_that_disagrees_with_the_restart_plan(restarted):
    trajectories, _metrics, plan, _prefix, config = restarted
    assert self_play.validate_resume(trajectories, config, plan)
    with pytest.raises(ValueError, match="restart plan"):
        self_play.validate_resume(trajectories, config, {})


def test_rust_capture_refuses_a_wrong_reshuffle(restarted):
    trajectories, *_ = restarted
    game = next(t for t in trajectories if t.restart is not None)
    state = wr.RustGameState(
        game.engine_seed, players=game.players, advanced=True, expert=False, solo_rules=False
    )
    for decision, action in enumerate(game.actions):
        if decision == game.restart.at:
            state, _ = state.redeterminize(game.restart.reshuffle_seed)
        state.apply_macro(action)
    capture = lambda: wr.RustTrainingCapture(game.seed)
    capture().finish(state, "{}", list(game.actions), game.rust_restart)
    source, at, reshuffle = game.rust_restart
    with pytest.raises(Exception):
        capture().finish(state, "{}", list(game.actions), (source, at, reshuffle + 1))
    with pytest.raises(Exception, match="outside"):
        capture().finish(state, "{}", list(game.actions), (source, 0, reshuffle))
