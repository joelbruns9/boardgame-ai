"""Paired placement targets: building, splitting, the loss, and training with them."""

from __future__ import annotations

import pytest
import torch

from games.welcome_to import network as nw
from games.welcome_to import paired_targets as pt
from games.welcome_to import s2_train, self_play

wr = pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(sheet_hidden=16, sheet_out=8, trunk_hidden=24, trunk_blocks=1, head_hidden=16)


@pytest.fixture(scope="module")
def iteration(tmp_path_factory):
    """A tiny iteration directory: shards from a small net, and its checkpoint."""
    root = tmp_path_factory.mktemp("pairs_run")
    directory = root / "iter_0001"
    directory.mkdir()
    torch.manual_seed(11)
    net = nw.WelcomeToNet(_SMALL).eval()
    writer = wr.RustSampleShardWriter(directory / "trajectories.jsonl", shard_games=4, queue_games=4)
    try:
        games, _ = self_play.generate(
            net,
            config=self_play.SelfPlayConfig(games=8, inflight=8, max_batch=8, seed=31_000),
            search_config=self_play.default_search_config(simulations=2),
            device="cpu",
            on_captured=lambda _t, captured: writer.add(captured),
        )
    finally:
        writer.close()
    checkpoint = root / "learner.pt"
    config = s2_train.S2TrainConfig()
    s2_train.save_checkpoint(
        checkpoint, net, torch.optim.AdamW(net.parameters()), config,
        {"optimizer_steps_completed": 0, "training_runs_completed": 0}, source="test",
    )
    path = pt.build(directory, checkpoint, roots=6, alternatives=2, futures=3, seed=5, device="cpu", simulations=2)
    return root, directory, checkpoint, path, games


def test_build_writes_paired_roots_from_ordinary_games_and_resumes(iteration):
    root, directory, checkpoint, path, games = iteration
    payload = torch.load(path, weights_only=False)
    roots = payload["roots"]
    assert 0 < len(roots) <= 6
    seeds = {g.seed for g in games}
    for r in roots:
        assert r["game_seed"] in seeds
        assert 2 <= len(r["candidates"]) <= 3
        assert r["scores"].shape[:2] == (len(r["candidates"]), 3)
        assert r["ranks"].shape == (len(r["candidates"]), 3, 4)
        assert len(r["afterstates"]) == len(r["candidates"])
    stamp = path.stat().st_mtime_ns
    assert pt.build(directory, checkpoint, roots=6, alternatives=2, futures=3, seed=5, device="cpu", simulations=2) == path
    assert path.stat().st_mtime_ns == stamp, "an existing pairs.pt is reused, not rebuilt"
    window = pt.load_window(root, 1, 4)
    assert [(r["game_seed"], r["turn"], r["candidates"]) for r in window] == [
        (r["game_seed"], r["turn"], r["candidates"]) for r in roots
    ]
    assert pt.load_window(root, 5, 2) == []


def test_pairs_split_by_the_same_family_rule_as_the_replay(iteration):
    """A paired root lands on the side its source game's family lands on."""
    _root, _directory, _checkpoint, path, games = iteration
    roots = torch.load(path, weights_only=False)["roots"]
    by_seed = {g.seed: g for g in games}
    for salt in ("a", "b", "c"):
        train, val = pt.split(roots, 0.5, salt)
        assert len(train) + len(val) == len(roots)
        _games_train, games_val = s2_train.split_trajectories(games, 0.5, seed=0, salt=salt)
        held = {s2_train.split_family(g) for g in games_val}
        for r in val:
            assert s2_train.split_family(by_seed[r["game_seed"]]) in held or not held
        for r in train:
            assert s2_train.split_family(by_seed[r["game_seed"]]) not in held


def test_the_paired_loss_trains_the_score_heads_toward_the_labels(iteration):
    _root, _directory, _checkpoint, path, _games = iteration
    roots = torch.load(path, weights_only=False)["roots"]
    torch.manual_seed(2)
    net = nw.WelcomeToNet(_SMALL)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    first, parts = pt.paired_loss(net, roots, "cpu", pair_weight=100.0, rank_weight=1.0)
    assert set(parts) == {"paired_score_abs", "paired_rank", "paired_diff"}
    for _ in range(60):
        loss, _ = pt.paired_loss(net, roots, "cpu", pair_weight=100.0, rank_weight=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.5 * float(first.detach())


def test_training_with_pairs_reports_before_and_after_on_held_out_roots(iteration):
    _root, _directory, _checkpoint, path, games = iteration
    roots = torch.load(path, weights_only=False)["roots"]
    torch.manual_seed(4)
    config = s2_train.S2TrainConfig(train_steps=3, batch_size=8, pairs_weight=1.0, val_fraction=0.4, log_every=1)
    net, _opt, metrics = s2_train.fit(
        games, net=nw.WelcomeToNet(_SMALL), config=config, device="cpu", pairs=roots, log=False
    )
    report = metrics["pairs"]
    if report is None:  # every root's family landed in validation
        assert not pt.split(roots, 0.4, config.val_split_salt)[0]
        return
    assert report["train_roots"] + report["val_roots"] <= len(roots)
    assert any("loss_paired" in row for row in metrics["history"])
    off = s2_train.fit(games, net=nw.WelcomeToNet(_SMALL), config=s2_train.S2TrainConfig(train_steps=1, batch_size=8), device="cpu", pairs=roots, log=False)[2]
    assert off["pairs"] is None, "pairs_weight 0 leaves training unchanged"
