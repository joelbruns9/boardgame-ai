"""Forced City Plan deals (plan-deal curriculum): legal, paired, and identical across engines."""

from __future__ import annotations

import random

import pytest

from games.welcome_to import macro_codec as mc
from games.welcome_to import snapshot
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.plans import available_plan_ids

wr = pytest.importorskip("welcome_to_rust")
CONFIG = GameConfig(players=3, advanced=True, solo_rules=False)


def _forced(rng):
    return tuple(rng.choice(available_plan_ids(stack, True)) for stack in (1, 2, 3))


def test_a_forced_deal_keeps_every_other_random_draw():
    rng = random.Random(1)
    for seed in range(20):
        natural = GameState.new(seed=seed, config=CONFIG)
        plans = _forced(rng)
        forced = GameState.new(seed=seed, config=CONFIG, plan_ids=plans)
        assert forced.plan_ids == plans
        assert forced.deck == natural.deck
        assert forced.stack_new == natural.stack_new


def test_python_and_rust_agree_on_forced_deals_through_a_whole_game():
    rng = random.Random(2)
    for seed in range(12):
        plans = _forced(rng)
        py = GameState.new(seed=seed, config=CONFIG, plan_ids=plans)
        rs = wr.RustGameState(seed, players=3, advanced=True, expert=False, solo_rules=False, plan_ids=list(plans))
        assert tuple(rs.plan_ids) == plans
        while not py.is_terminal:
            assert snapshot.to_snapshot(py) == rs.snapshot()
            macro = rng.choice(mc.legal_macros(py))
            mc.apply_macro(py, macro)
            rs.apply_macro(macro)
        assert rs.is_terminal and list(rs.scores(None)) == py.scores()


def test_an_off_stack_deal_is_refused_by_both_engines():
    wrong = (available_plan_ids(3, True)[0], available_plan_ids(2, True)[0], available_plan_ids(1, True)[0])
    with pytest.raises(ValueError, match="legal plan per stack"):
        GameState.new(seed=1, config=CONFIG, plan_ids=wrong)
    with pytest.raises(Exception, match="legal plan per stack"):
        wr.RustGameState(1, players=3, advanced=True, expert=False, solo_rules=False, plan_ids=list(wrong))


def test_forced_deals_flow_through_generation_capture_and_restarts(tmp_path):
    """A forced-deal game and a curriculum restart of it replay identically in
    Python and in the Rust capture, and are excluded from strength metrics."""
    import numpy as np
    import torch

    from games.welcome_to import curriculum, network as nw, self_play

    torch.manual_seed(5)
    net = nw.WelcomeToNet(nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16)).eval()
    config = self_play.SelfPlayConfig(games=4, inflight=4, max_batch=4, seed=77_000)
    seeds = list(range(config.seed, config.seed + config.games))
    rng = random.Random(4)
    forced = {seeds[0]: _forced(rng), seeds[2]: _forced(rng)}
    prefix = tmp_path / "trajectories.jsonl"
    writer = wr.RustSampleShardWriter(prefix, shard_games=4, queue_games=4)
    try:
        games, metrics = self_play.generate(
            net, config=config, search_config=self_play.default_search_config(simulations=2),
            device="cpu", on_captured=lambda _t, c: writer.add(c), forced_plans=forced,
        )
    finally:
        writer.close()
    by_seed = {g.seed: g for g in games}
    for seed, plans in forced.items():
        assert by_seed[seed].plan_ids == plans
        assert tuple(by_seed[seed].new_python_state().plan_ids) == plans
    assert metrics["natural_games"] == config.games - len(forced)
    cached = {t.seed: t for t in self_play.read_trajectories(prefix)}
    for game in games:
        oracle = list(self_play.replay(self_play.SelfPlayTrajectory.from_json(game.to_json())))
        rows = list(self_play.replay(cached[game.seed]))
        assert len(rows) == len(oracle)
        for a, b in zip(rows, oracle):
            assert np.array_equal(a.sheet_planes, b.sheet_planes)
            for name in a.targets:
                assert np.array_equal(np.asarray(a.targets[name], dtype=np.float32), np.asarray(b.targets[name], dtype=np.float32)), name
    # a restart of a forced-deal source inherits the deal
    source = by_seed[seeds[0]]
    restart = curriculum.Restart(source_seed=source.seed, at=6, reshuffle_seed=3, distance=1, slot=0, plan_ids=source.plan_ids)
    child = self_play.SelfPlayTrajectory(
        seed=1, players=source.players, actions=source.actions, searches=(), scores=source.scores,
        opponents=source.opponents, restart=restart,
    )
    assert child.engine_plan_ids == source.plan_ids
    assert self_play.SelfPlayTrajectory.from_json(child.to_json()) == child
    with pytest.raises(ValueError, match="forced-deal plan"):
        self_play.validate_resume(games, config, {}, {}, {})
