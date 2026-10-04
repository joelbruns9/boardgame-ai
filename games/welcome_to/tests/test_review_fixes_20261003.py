"""Regressions for TOY_RUN_PACKAGE_REVIEW.md (2026-10-03) findings F1-F7."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from games.welcome_to import curriculum
from games.welcome_to import network as nw
from games.welcome_to import paired_targets as pt
from games.welcome_to import plans as pl
from games.welcome_to import pool_rescue as pr
from games.welcome_to import s2_train, self_play, sibling_probe
from games.welcome_to.game import GameConfig, GameState

wr = pytest.importorskip("welcome_to_rust")
_SMALL = nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16, plan_hidden=16, plan_out=8)


def _root(players, turn, seed):
    return sibling_probe.Root(game_seed=seed, players=players, turn=turn, played=1, candidates=[1, 2], snapshot={})


def test_f1_root_cap_keeps_every_player_count_in_proportion():
    """Seat counts come in seed blocks; truncating a seed-sorted list had cut
    the four-player block entirely."""
    candidates = (
        [_root(2, 5 + i % 15, 1_000 + i) for i in range(600)]
        + [_root(3, 5 + i % 15, 2_000 + i) for i in range(300)]
        + [_root(4, 5 + i % 15, 3_000 + i) for i in range(100)]
    )
    chosen = pt._stratified_roots(candidates, 300, seed=4)
    counts = {n: sum(r.players == n for r in chosen) for n in (2, 3, 4)}
    assert len(chosen) == 300
    assert counts == {2: 180, 3: 90, 4: 30}
    assert chosen == sorted(chosen, key=lambda r: (r.game_seed, r.turn))
    assert pt._stratified_roots(candidates[:10], 300, seed=4) == sorted(candidates[:10], key=lambda r: (r.game_seed, r.turn))


def test_f3_complete_street_rule_ignores_streets_that_cannot_get_a_roundabout():
    state = GameState.new(seed=11, config=GameConfig(players=2, advanced=True, solo_rules=False), plan_ids=(0, 24, 14))
    sheet = state.sheets[0]
    sheet.build_roundabout((0, 0), 1)
    sheet.build_roundabout((1, 0), 1)
    assert not sheet.can_build_roundabout()
    viable = pl.pool_plan_streets(pl.PLANS[24], sheet)
    pools, parks, live = pr.needed_streets(state, 0)
    assert live == [1]
    assert 2 not in viable and 2 not in pools and 2 not in parks
    assert pools == set(viable)
    assert {x for x, _y in pl.pool_target_boxes(pl.PLANS[24], sheet)} <= set(viable)


@pytest.fixture(scope="module")
def iteration(tmp_path_factory):
    root = tmp_path_factory.mktemp("review_pairs")
    directory = root / "iter_0001"
    directory.mkdir()
    torch.manual_seed(3)
    net = nw.WelcomeToNet(_SMALL).eval()
    config = self_play.SelfPlayConfig(games=10, inflight=10, max_batch=10, seed=88_000)
    forced = {88_000: (18, 23, 14), 88_009: (19, 25, 15)}
    writer = wr.RustSampleShardWriter(directory / "trajectories.jsonl", shard_games=5, queue_games=5)
    try:
        games, _ = self_play.generate(
            net, config=config, search_config=self_play.default_search_config(simulations=2),
            device="cpu", on_captured=lambda _t, c: writer.add(c), forced_plans=forced,
        )
    finally:
        writer.close()
    checkpoint = root / "learner.pt"
    s2_train.save_checkpoint(
        checkpoint, net, torch.optim.AdamW(net.parameters()), s2_train.S2TrainConfig(),
        {"optimizer_steps_completed": 0, "training_runs_completed": 0}, source="test",
    )
    path = pt.build(directory, checkpoint, roots=12, futures=2, seed=1, device="cpu", simulations=2, plan_aware_share=1.0)
    return directory, checkpoint, path, forced


def test_f4_forced_deal_games_never_become_paired_roots(iteration):
    _directory, _checkpoint, path, forced = iteration
    payload = torch.load(path, weights_only=False)
    assert payload["roots"], "no roots were built"
    assert not {r["game_seed"] for r in payload["roots"]} & set(forced)
    assert payload["realized_roots"] == len(payload["roots"]) <= payload["requested_roots"]
    assert sum(payload["roots_by_players"].values()) == payload["realized_roots"]


def test_f5_an_existing_pairs_file_with_another_recipe_is_refused(iteration):
    directory, checkpoint, path, _forced = iteration
    assert pt.build(directory, checkpoint, roots=12, futures=2, seed=1, device="cpu", simulations=2, plan_aware_share=1.0) == path
    with pytest.raises(ValueError, match="different recipe"):
        pt.build(directory, checkpoint, roots=12, futures=3, seed=1, device="cpu", simulations=2, plan_aware_share=1.0)


def test_q1_steered_roots_carry_an_execution_audit(iteration):
    _directory, _checkpoint, path, _forced = iteration
    payload = torch.load(path, weights_only=False)
    if payload["steered_roots"] == 0:
        assert payload["execution_audit"] is None
        pytest.skip("no live pool plan among this tiny corpus's roots")
    audit = payload["execution_audit"]
    assert audit["roots"] == payload["steered_roots"]
    assert 0.0 <= audit["preferred_candidate_changed"] <= 1.0
    assert payload["steer_calls"] >= payload["steer_overrides"]


def test_f6_the_fallback_moves_a_whole_family_and_needs_two(iteration):
    _directory, _checkpoint, _path, _forced = iteration
    games = self_play.read_trajectories(_directory / "trajectories.jsonl")
    source = games[0]
    child = replace(
        source, seed=999_001, searches=(),
        restart=curriculum.Restart(source_seed=source.seed, at=1, reshuffle_seed=1, distance=1, slot=0),
    )
    with pytest.raises(ValueError, match="two game families"):
        s2_train.split_trajectories([source, child], 0.1, seed=0)
    other = games[1]
    for salt in map(str, range(20)):
        train, val = s2_train.split_trajectories([source, child, other], 0.01, seed=0, salt=salt)
        families_train = {s2_train.split_family(g) for g in train}
        families_val = {s2_train.split_family(g) for g in val}
        assert train and val and not families_train & families_val


def test_f2_helper_games_are_identified():
    base = dict(seed=1, players=2, actions=(1, 2), searches=(), scores=(0, 0), opponents=("a", "b"))
    assert not s2_train.is_helper_game(self_play.SelfPlayTrajectory(**base))
    assert s2_train.is_helper_game(self_play.SelfPlayTrajectory(**base, plan_ids=(0, 6, 12)))
    assert s2_train.is_helper_game(self_play.SelfPlayTrajectory(**base, assist_through=16))
