"""S2 visit-target evaluation, optimiser, and checkpoint gates."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict

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
    reversed_train, reversed_val = s2_train.split_trajectories(
        list(reversed(trajectories)), 0.25, seed=7
    )
    assert {game.seed for game in reversed_train} == {game.seed for game in train}
    assert {game.seed for game in reversed_val} == {game.seed for game in val}
    held, digest = s2_train.stable_is_validation(12, trajectories[0].seed, 0.25, "x")
    assert (held, digest) == s2_train.stable_is_validation(
        12, trajectories[0].seed, 0.25, "x"
    )
    assert digest != s2_train.stable_is_validation(
        13, trajectories[0].seed, 0.25, "x"
    )[1]
    with pytest.raises(ValueError, match="at least two"):
        s2_train.split_trajectories(trajectories[:1], 0.1, seed=0)
    assert s2_train._json_safe({"missing": float("nan")}) == {"missing": None}


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
        s2_train._load_optimizer_state_compatible(
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


def test_version_one_checkpoint_expands_appended_plan_heads_deterministically(tmp_path):
    net = _net(61)
    legacy = {name: value.clone() for name, value in net.state_dict().items()}
    output_keys = [
        name
        for name, value in legacy.items()
        if name.startswith("per_seat_head.")
        and value.ndim >= 1
        and value.shape[0] == len(nw.PER_SEAT_HEAD_TARGETS)
    ]
    assert len(output_keys) == 2
    old_rows = len(nw.LEGACY_PER_SEAT_HEAD_TARGETS)
    for name in output_keys:
        legacy[name] = legacy[name][:old_rows].clone()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format": s2_train.CHECKPOINT_FORMAT,
            "version": s2_train.LEGACY_CHECKPOINT_VERSION,
            "state_dict": legacy,
            "net_config": asdict(net.config),
        },
        path,
    )
    loaded, _ = s2_train.load_training_checkpoint(path)
    generic = s0_train.load(path)
    for name in output_keys:
        assert torch.equal(loaded.state_dict()[name][:old_rows], legacy[name])
        assert torch.count_nonzero(loaded.state_dict()[name][old_rows:]) == 0
        assert torch.equal(loaded.state_dict()[name], generic.state_dict()[name])


def test_version_one_adam_state_expands_only_final_head_and_can_step(trajectories):
    net = _net(62)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    raw = next(
        self_play.iter_batches(trajectories, 8, random.Random(9), shuffle_buffer=16)
    )
    batch = nw.to_tensors(raw)
    loss, _ = nw.losses(net(
        batch["sheet_planes"],
        batch["sheet_scalars"],
        batch["viewer_plane"],
        batch["global_scalars"],
    ), batch)
    loss.backward()
    optimizer.step()

    names = [name for name, _parameter in net.named_parameters()]
    final_index = max(
        index
        for index, module in enumerate(net.per_seat_head)
        if isinstance(module, torch.nn.Linear)
    )
    widened = {
        f"per_seat_head.{final_index}.weight",
        f"per_seat_head.{final_index}.bias",
    }
    old_rows = len(nw.LEGACY_PER_SEAT_HEAD_TARGETS)

    legacy_model = {name: value.clone() for name, value in net.state_dict().items()}
    for name in widened:
        legacy_model[name] = legacy_model[name][:old_rows].clone()
    legacy_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_ids = [
        item for group in legacy_optimizer["param_groups"] for item in group["params"]
    ]
    expected_moments = {}
    for saved_id, name in zip(saved_ids, names):
        if name not in widened:
            continue
        state = legacy_optimizer["state"][saved_id]
        for moment in ("exp_avg", "exp_avg_sq"):
            state[moment] = state[moment][:old_rows].clone()
        expected_moments[name] = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }

    resumed = _net(63)
    nw.load_state_dict_compatible(resumed, legacy_model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    s2_train._load_optimizer_state_compatible(
        resumed_optimizer,
        legacy_optimizer,
        resumed,
        names,
    )

    resumed_parameters = dict(resumed.named_parameters())
    for name in widened:
        assert torch.equal(
            resumed.state_dict()[name][:old_rows], legacy_model[name]
        )
        assert torch.count_nonzero(resumed.state_dict()[name][old_rows:]) == 0
        state = resumed_optimizer.state[resumed_parameters[name]]
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(
                state[moment][:old_rows], expected_moments[name][moment]
            )
            assert torch.count_nonzero(state[moment][old_rows:]) == 0
        assert torch.equal(state["step"], expected_moments[name]["step"])

    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss, _ = nw.losses(resumed(
        batch["sheet_planes"],
        batch["sheet_scalars"],
        batch["viewer_plane"],
        batch["global_scalars"],
    ), batch)
    resumed_loss.backward()
    resumed_optimizer.step()
    assert torch.isfinite(resumed_loss)
    assert all(
        torch.isfinite(parameter).all() for parameter in resumed.parameters()
    )
