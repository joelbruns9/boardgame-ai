"""S2 visit-target evaluation, optimiser, and checkpoint gates."""

from __future__ import annotations

import math
import random

import pytest
import torch

from games.welcome_to import network as nw
from games.welcome_to import s2_train
from games.welcome_to import self_play
from games.welcome_to import train as s0_train

pytest.importorskip("welcome_to_rust")


_SMALL = nw.NetConfig(
    sheet_hidden=16,
    sheet_out=8,
    trunk_hidden=24,
    trunk_blocks=1,
    head_hidden=16,
)


def _net(seed: int = 51) -> nw.WelcomeToNet:
    torch.manual_seed(seed)
    return nw.WelcomeToNet(_SMALL).eval()


@pytest.fixture(scope="module")
def trajectories():
    games, _ = self_play.generate(
        _net(),
        config=self_play.SelfPlayConfig(
            games=4,
            inflight=4,
            max_batch=4,
            seed=14_000,
        ),
        search_config=self_play.default_search_config(simulations=2),
        device="cpu",
    )
    return games


def test_s2_split_is_by_complete_game_and_reproducible(trajectories):
    train, val = s2_train.split_trajectories(trajectories, 0.25, seed=7)
    again_train, again_val = s2_train.split_trajectories(
        trajectories, 0.25, seed=7
    )
    assert train == again_train
    assert val == again_val
    assert {game.seed for game in train}.isdisjoint(game.seed for game in val)
    assert len(train) == 3
    assert len(val) == 1
    with pytest.raises(ValueError, match="at least two"):
        s2_train.split_trajectories(trajectories[:1], 0.1, seed=0)
    assert s2_train._json_safe({"missing": float("nan")}) == {"missing": None}


def test_s2_evaluation_uses_the_visit_distribution(trajectories):
    net = _net(52)
    metrics = s2_train.evaluate(net, trajectories, "cpu", batch_size=8)
    assert metrics["eval_samples"] == sum(len(game.searches) for game in trajectories)
    assert metrics["policy_cross_entropy"] >= metrics["policy_target_entropy"] - 1e-6
    assert metrics["policy_kl"] >= 0.0
    assert metrics["rank_cross_entropy"] >= metrics["rank_target_entropy"] - 1e-6
    assert 0.0 <= metrics["policy_visit_best"] <= 1.0
    assert 0.0 <= metrics["sampled_action_top1"] <= 1.0
    assert 0.0 <= metrics["rank_best"] <= 1.0
    assert "policy_top1" not in metrics
    for slot in range(3):
        assert f"support_turns_to_plan_{slot}" in metrics

    # Recompute the soft-target cross entropy directly. The sampled action is
    # intentionally absent from this expression.
    total = 0.0
    rows = 0
    with torch.no_grad():
        for raw in self_play.iter_batches(
            trajectories, 8, random.Random(0), shuffle_buffer=16
        ):
            batch = nw.to_tensors(raw)
            out = net(
                batch["sheet_planes"],
                batch["sheet_scalars"],
                batch["viewer_plane"],
                batch["global_scalars"],
            )
            logits = out["policy_logits"].masked_fill(batch["legal"] <= 0, -1e9)
            total += float(
                -(batch["policy"] * torch.log_softmax(logits, -1)).sum()
            )
            rows += int(batch["policy"].shape[0])
    assert metrics["policy_cross_entropy"] == pytest.approx(total / rows, rel=1e-6)


def test_s2_random_start_trains_and_checkpoint_resumes(trajectories, tmp_path):
    config = s2_train.S2TrainConfig(
        val_fraction=0.25,
        epochs=1,
        batch_size=16,
        lr=1e-3,
        shuffle_buffer=32,
        seed=71,
    )
    net, optimizer, metrics = s2_train.fit(
        trajectories,
        net_config=_SMALL,
        config=config,
        device="cpu",
        log=False,
    )
    assert metrics["initialization"] == "random"
    assert metrics["optimizer_steps"] > 0
    assert metrics["training_samples"] > 0
    assert math.isfinite(metrics["policy_cross_entropy"])
    assert math.isfinite(metrics["history"][0]["loss_total"])
    assert math.isfinite(metrics["history"][0]["max_gradient_norm"])

    path = s2_train.save_checkpoint(
        tmp_path / "candidate.pt",
        net,
        optimizer,
        config,
        metrics,
        source="random",
    )
    loaded, payload = s2_train.load_training_checkpoint(path)
    generic = s0_train.load(path)
    for name, value in net.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
        assert torch.equal(value, generic.state_dict()[name])
    assert payload["epochs_completed"] == 1
    assert payload["optimizer_state"]["state"]

    resumed, _, resumed_metrics = s2_train.fit(
        trajectories,
        net=loaded,
        config=config,
        device="cpu",
        optimizer_state=payload["optimizer_state"],
        epochs_completed=payload["epochs_completed"],
        log=False,
    )
    assert resumed is loaded
    assert resumed_metrics["initialization"] == "checkpoint"
    assert resumed_metrics["epochs_completed"] == 2.0
