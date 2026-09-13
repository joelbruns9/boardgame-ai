"""The fitted W5 weight: the fit, the ceiling, the checkpoint, and the served views.

What these protect, in order of how quietly each could go wrong:

* the fit finds the alpha the evidence implies, and never leaves ``[0, max]``;
* ``alpha = gate_max * tanh(gate)`` survives a checkpoint round trip, and a file
  without the key keeps the legacy ceiling it was trained under;
* the served policy is literally ``flat + alpha * W5``, and the ``action`` view is
  W5 alone -- the arena's pure action-token player;
* the flat logits the fit recovers do not depend on the alpha the model carried.
"""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from .action_alpha import (
    HeldOutLogits,
    collect_held_out_logits,
    cross_entropy,
    fit_alpha,
    format_alpha_fit,
    refit_alpha,
    step_toward,
)
from .buffer import GameRecorder
from .codec import legal_action_indices
from .dataset import collate, examples_from_record
from .game import Phase
from .net import SWDNet
from .train import (
    ACTION_GATE_MAX_LEGACY,
    action_gate_max_from_config,
    make_checkpoint,
    model_from_config,
)


def _synthetic(alpha: float, rows: int = 3000, width: int = 7, seed: int = 11):
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    residual = torch.randn(rows, width, generator=generator, dtype=torch.float64)
    counts = torch.randint(2, width + 1, (rows,), generator=generator)
    mask = torch.arange(width).unsqueeze(0) < counts.unsqueeze(1)
    combined = (flat + alpha * residual).masked_fill(~mask, float("-inf"))
    target = torch.softmax(combined, dim=1)
    return HeldOutLogits(flat=flat, residual=residual, target=target, mask=mask)


@pytest.mark.parametrize("truth", [0.0, 0.35, 1.0, 1.7])
def test_the_fit_recovers_the_alpha_that_generated_the_targets(truth):
    logits = _synthetic(truth)
    assert fit_alpha(logits, 2.0) == pytest.approx(truth, abs=1e-6)


def test_the_fit_is_clamped_to_the_range_at_both_ends():
    assert fit_alpha(_synthetic(3.5), 2.0) == 2.0
    assert fit_alpha(_synthetic(-0.8), 2.0) == 0.0


def test_the_fitted_alpha_is_the_held_out_minimum():
    logits = _synthetic(0.6)
    best = fit_alpha(logits, 2.0)
    for other in (0.0, 0.3, best - 0.05, best + 0.05, 1.2, 2.0):
        assert cross_entropy(logits, best) <= cross_entropy(logits, other) + 1e-12


def test_the_step_cap_limits_movement_and_respects_the_range():
    assert step_toward(0.0, 1.5, 0.1, 2.0) == pytest.approx(0.1)
    assert step_toward(0.5, 0.45, 0.1, 2.0) == pytest.approx(0.45)
    assert step_toward(1.0, 0.0, 0.1, 2.0) == pytest.approx(0.9)
    assert step_toward(1.95, 2.0, 0.1, 2.0) == pytest.approx(2.0)
    assert step_toward(0.05, 0.0, 0.1, 2.0) == 0.0
    with pytest.raises(ValueError):
        step_toward(0.0, 1.0, 0.0, 2.0)


def test_set_alpha_reaches_above_one_only_under_a_raised_ceiling():
    raised = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0).action_scorer
    assert raised.set_alpha(1.5) == pytest.approx(1.5, abs=1e-6)
    assert raised.set_alpha(2.0) == pytest.approx(2.0, abs=1e-5)
    assert raised.alpha_value() <= 2.0
    legacy = SWDNet(32, 1, 2, action_residual=True).action_scorer
    assert legacy.gate_max == ACTION_GATE_MAX_LEGACY
    assert legacy.set_alpha(1.5) < 1.0


def _playing_batch():
    from .test_action_residual import _input_batch, _playing_game

    batch, _vectorized, _legal = _input_batch(_playing_game())
    return batch


def test_the_served_policy_is_flat_plus_alpha_times_w5():
    torch.manual_seed(5)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0).eval()
    batch = _playing_batch()
    with torch.no_grad():
        flat = model(batch)["policy"].clone()
        model.action_scorer.set_alpha(1.25)
        out = model(batch)
    alpha = model.action_scorer.alpha_value()
    assert torch.allclose(out["policy"], flat + alpha * out["action_policy"], atol=1e-5)


def test_the_action_view_serves_w5_alone_and_needs_a_scorer():
    torch.manual_seed(6)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0).eval()
    model.action_scorer.set_alpha(0.7)
    batch = _playing_batch()
    model.policy_source = "action"
    with torch.no_grad():
        out = model(batch)
    assert torch.equal(out["policy"], out["action_policy"])

    plain = SWDNet(32, 1, 2).eval()
    plain.policy_source = "action"
    with pytest.raises(ValueError, match="needs the W5 action scorer"):
        with torch.no_grad():
            plain(batch)


def test_the_ceiling_and_the_weight_survive_a_checkpoint_round_trip():
    torch.manual_seed(7)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    model.action_scorer.set_alpha(1.4)
    checkpoint = make_checkpoint(model, {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2})
    assert checkpoint["config"]["action_gate_max"] == 2.0
    rebuilt = model_from_config(checkpoint["config"])
    rebuilt.load_state_dict(checkpoint["model_state"])
    assert rebuilt.action_scorer.gate_max == 2.0
    assert rebuilt.action_scorer.alpha_value() == pytest.approx(1.4, abs=1e-6)


def test_a_file_without_the_key_keeps_the_legacy_ceiling():
    assert action_gate_max_from_config({"action_residual": True}) == {
        "action_gate_max": ACTION_GATE_MAX_LEGACY
    }
    assert action_gate_max_from_config({}) == {}
    rebuilt = model_from_config({"d_model": 32, "layers": 1, "heads": 2, "action_residual": True})
    assert rebuilt.action_scorer.gate_max == ACTION_GATE_MAX_LEGACY


def test_a_config_contradicting_the_model_ceiling_is_refused():
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    with pytest.raises(ValueError, match="action_gate_max"):
        make_checkpoint(model, {"d_model": 32, "layers": 1, "action_gate_max": 1.0})


def _labeled_examples(games: int = 3):
    examples = []
    for seed in range(games):
        recorder = GameRecorder(90 + seed, agents={"p0": "test", "p1": "test"})
        rng = random.Random(9000 + seed)
        while recorder.game.phase is not Phase.COMPLETE:
            legal = legal_action_indices(recorder.game)
            choice = rng.choice(legal)
            recorder.play(choice, policy_target={choice: 1.0})
        examples.extend(examples_from_record(recorder.finish()))
    return examples


def test_recovered_flat_logits_do_not_depend_on_the_carried_alpha():
    torch.manual_seed(8)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    examples = _labeled_examples(1)
    model.action_scorer.set_alpha(0.0)
    at_zero = collect_held_out_logits(model, examples, "cpu", batch_size=16)
    model.action_scorer.set_alpha(1.3)
    at_carried = collect_held_out_logits(model, examples, "cpu", batch_size=16)
    assert at_zero.positions == at_carried.positions > 0
    real = at_zero.mask
    assert torch.allclose(at_zero.flat[real], at_carried.flat[real], atol=1e-4)
    assert torch.equal(at_zero.residual, at_carried.residual)
    assert torch.allclose(at_zero.target.sum(dim=1), torch.ones(at_zero.positions, dtype=torch.float64))


def test_refit_applies_a_capped_step_and_reports_the_evidence():
    torch.manual_seed(9)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    examples = _labeled_examples(2)
    fit = refit_alpha(
        model, examples, "cpu", alpha_max=2.0, step=0.1, batch_size=32, min_positions=1
    )
    assert fit.skipped is None
    assert fit.previous == 0.0
    assert 0.0 <= fit.fitted <= 2.0
    assert fit.applied == pytest.approx(min(fit.fitted, 0.1), abs=1e-6)
    assert model.action_scorer.alpha_value() == pytest.approx(fit.applied, abs=1e-6)
    assert fit.loss_at_fitted <= fit.loss_flat_only + 1e-9
    assert math.isfinite(fit.loss_w5_only)
    assert "fitted" in format_alpha_fit(fit.as_dict())


def test_too_little_evidence_keeps_the_previous_weight():
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    model.action_scorer.set_alpha(0.4)
    fit = refit_alpha(model, _labeled_examples(1), "cpu", alpha_max=2.0, step=0.1, min_positions=10_000)
    assert fit.skipped
    assert fit.applied == pytest.approx(0.4, abs=1e-6)
    assert model.action_scorer.alpha_value() == pytest.approx(0.4, abs=1e-6)
    assert "kept" in format_alpha_fit(fit.as_dict())


def test_refit_refuses_a_model_built_under_a_different_ceiling():
    legacy = SWDNet(32, 1, 2, action_residual=True)
    with pytest.raises(ValueError, match="action_gate_max"):
        refit_alpha(legacy, _labeled_examples(1), "cpu", alpha_max=2.0, step=0.1)


def test_phase_d_builds_the_raised_ceiling_only_when_fitting():
    from .phase_d import PhaseDConfig, PhaseDLoop

    loop = PhaseDLoop.__new__(PhaseDLoop)
    loop.config = PhaseDConfig(action_residual=True, action_policy_weight=0.1)
    assert loop._new_model().action_scorer.gate_max == ACTION_GATE_MAX_LEGACY
    loop.config = PhaseDConfig(
        action_residual=True, action_policy_weight=0.1, fit_action_alpha=True
    )
    model = loop._new_model()
    assert model.action_scorer.gate_max == 2.0
    assert not model.action_scorer.gate.requires_grad


def test_phase_d_refuses_contradictory_alpha_settings():
    from .phase_d import PhaseDConfig

    with pytest.raises(ValueError, match="--fit-action-alpha requires --action-residual"):
        PhaseDConfig(fit_action_alpha=True).validate()
    with pytest.raises(ValueError, match="choose one"):
        PhaseDConfig(
            action_residual=True,
            action_policy_weight=0.1,
            fit_action_alpha=True,
            train_action_gate=True,
        ).validate()
    with pytest.raises(ValueError, match="action_alpha_step"):
        PhaseDConfig(
            action_residual=True,
            action_policy_weight=0.1,
            fit_action_alpha=True,
            action_alpha_step=0.0,
        ).validate()


def test_the_heartbeat_reports_the_fitted_weight():
    from games.az_loop.run_controller import RunController

    line = RunController.heartbeat_line(
        {"iteration": 3, "stats": {"training": {"policy_mix_alpha": 0.35}}}
    )
    assert "alpha=0.350" in line
    quiet = RunController.heartbeat_line({"iteration": 3, "stats": {"training": {}}})
    assert "alpha=" not in quiet


def test_the_arena_can_play_w5_alone_and_labels_the_side(tmp_path):
    from .arena import load_side

    torch.manual_seed(10)
    model = SWDNet(32, 1, 2, action_residual=True, action_gate_max=2.0)
    model.action_scorer.set_alpha(0.9)
    path = tmp_path / "w5.pt"
    torch.save(make_checkpoint(model, {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2}), path)
    side = load_side("a", path, device="cpu", precision="fp32", batch_cap=8, policy_source="action")
    assert side.model.policy_source == "action"
    assert side.architecture["policy_source"] == "action"
    assert side.architecture["action_alpha"] == pytest.approx(0.9, abs=1e-6)
    assert side.name.endswith("[action]")

    plain = tmp_path / "plain.pt"
    torch.save(make_checkpoint(SWDNet(32, 1, 2), {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2}), plain)
    with pytest.raises(ValueError, match="no W5 action scorer"):
        load_side("b", plain, device="cpu", precision="fp32", batch_cap=8, policy_source="action")
