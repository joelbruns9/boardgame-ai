"""S2 visit-target evaluation, optimiser, and checkpoint gates."""

from __future__ import annotations

import math
import random
from dataclasses import asdict

import pytest
import torch

from games.welcome_to import encoder as enc
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
    reversed_train, reversed_val = s2_train.split_trajectories(
        list(reversed(trajectories)), 0.25, seed=7
    )
    assert {game.seed for game in reversed_train} == {game.seed for game in train}
    assert {game.seed for game in reversed_val} == {game.seed for game in val}
    held, digest = s2_train.stable_family_is_validation(trajectories[0].seed, 0.25, "x")
    assert (held, digest) == s2_train.stable_family_is_validation(
        trajectories[0].seed, 0.25, "x"
    )
    assert digest != s2_train.stable_family_is_validation(
        trajectories[0].seed + 1, 0.25, "x"
    )[1]
    with pytest.raises(ValueError, match="at least two"):
        s2_train.split_trajectories(trajectories[:1], 0.1, seed=0)
    assert s2_train._json_safe({"missing": float("nan")}) == {"missing": None}


def test_a_curriculum_restart_shares_its_source_game_s_split(trajectories):
    """Review 2026-10-02 §2.3: a restart and its source must sit on one side."""
    from dataclasses import replace

    from games.welcome_to import curriculum

    sources = list(trajectories)
    restarts = [
        replace(
            source,
            seed=900_000 + index,
            searches=(),
            restart=curriculum.Restart(
                source_seed=source.seed, at=1, reshuffle_seed=index, distance=1, slot=0
            ),
        )
        for index, source in enumerate(sources)
    ]
    for salt in ("a", "b", "c", "d", "e"):
        train, val = s2_train.split_trajectories(sources + restarts, 0.4, seed=0, salt=salt)
        held = {s2_train.split_family(game) for game in val}
        trained = {s2_train.split_family(game) for game in train}
        assert held.isdisjoint(trained)
        assert all(s2_train.split_family(r) == r.restart.source_seed for r in restarts)


def test_diagnostic_evaluation_set_is_bounded_and_order_independent(trajectories):
    iterations = [1, 1, 2, 2]
    selected = s2_train._bounded_evaluation_set(
        trajectories,
        2,
        iterations=iterations,
        salt="holdout",
        domain="validation",
    )
    reversed_selected = s2_train._bounded_evaluation_set(
        list(reversed(trajectories)),
        2,
        iterations=list(reversed(iterations)),
        salt="holdout",
        domain="validation",
    )
    assert len(selected) == 2
    assert {game.seed for game in selected} == {
        game.seed for game in reversed_selected
    }


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
    for name in (
        "will_complete_plan_0",
        "plan_0_first",
        "end_trigger_all_plans",
    ):
        assert f"support_{name}" in metrics
        assert f"target_mean_{name}" in metrics
        assert f"target_std_{name}" in metrics
        assert f"bce_{name}" in metrics
        assert f"brier_{name}" in metrics
        assert f"accuracy_{name}" in metrics
        assert f"positive_rate_{name}" in metrics
        assert f"r2_{name}" not in metrics
        p = metrics[f"positive_rate_{name}"]
        if 0.0 < p < 1.0:
            # skill is measured against the constant predictor at that rate
            assert metrics[f"brier_skill_{name}"] == pytest.approx(
                1.0 - metrics[f"brier_{name}"] / (p * (1.0 - p))
            )
            assert f"bce_skill_{name}" in metrics

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
        train_steps=2,
        batch_size=16,
        lr=1e-3,
        log_every=1,
        seed=71,
    )
    net, optimizer, metrics = s2_train.fit(
        trajectories,
        net_config=_SMALL,
        config=config,
        device="cpu",
        trajectory_iterations=[4] * len(trajectories),
        log=False,
    )
    assert metrics["initialization"] == "random"
    assert metrics["optimizer_steps"] > 0
    assert metrics["training_samples"] > 0
    assert math.isfinite(metrics["policy_cross_entropy"])
    assert math.isfinite(metrics["history"][0]["loss_total"])
    assert math.isfinite(metrics["history"][0]["max_gradient_norm"])
    assert metrics["pretrain_newest_iteration"] == 4
    assert metrics["pretrain_newest_metrics"]["eval_samples"] > 0

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
    assert payload["optimizer_steps_completed"] == 2
    assert payload["optimizer_state"]["state"]
    assert payload["optimizer_parameter_names"] == [
        name for name, _parameter in loaded.named_parameters()
    ]

    rejected_optimizer = torch.optim.AdamW(loaded.parameters())
    swapped_names = list(payload["optimizer_parameter_names"])
    swapped_names[:2] = reversed(swapped_names[:2])
    with pytest.raises(ValueError, match="names/order"):
        s2_train._load_optimizer_state_checked(
            rejected_optimizer,
            payload["optimizer_state"],
            loaded,
            swapped_names,
        )

    resumed, _, resumed_metrics = s2_train.fit(
        trajectories,
        net=loaded,
        config=config,
        device="cpu",
        optimizer_state=payload["optimizer_state"],
        optimizer_parameter_names=payload["optimizer_parameter_names"],
        optimizer_steps_completed=payload["optimizer_steps_completed"],
        training_runs_completed=payload["training_runs_completed"],
        log=False,
    )
    assert resumed is loaded
    assert resumed_metrics["initialization"] == "checkpoint"
    assert resumed_metrics["optimizer_steps_completed"] == 4.0
    assert resumed_metrics["training_runs_completed"] == 2.0


def test_an_unstamped_version_one_checkpoint_is_refused(tmp_path):
    """Every version-1 checkpoint predates the ABI stamp: refused, never expanded.

    The head-row expansion that used to rescue these was deleted after review
    (2026-09-25, §0.4) -- a matching ABI must mean current shapes.
    """
    net = _net(61)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format": s2_train.CHECKPOINT_FORMAT,
            "version": 1,
            "state_dict": net.state_dict(),
            "net_config": asdict(net.config),
        },
        path,
    )
    for loader in (s2_train.load_training_checkpoint, s0_train.load):
        with pytest.raises(ValueError):
            loader(path)


def test_a_stamped_checkpoint_with_wrong_head_rows_is_refused(tmp_path):
    net = _net(62)
    state = {name: value.clone() for name, value in net.state_dict().items()}
    final = max(
        index for index, module in enumerate(net.per_seat_head)
        if isinstance(module, torch.nn.Linear)
    )
    state[f"per_seat_head.{final}.bias"] = state[f"per_seat_head.{final}.bias"][:-3]
    path = tmp_path / "short.pt"
    torch.save(
        {
            "encoder_abi": enc.ENCODER_ABI_VERSION,
            "state_dict": state,
            "net_config": asdict(net.config),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="incompatible tensor shapes"):
        s0_train.load(path)


