"""W6.2b: the precision arena must actually differ only in precision."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .phase_d import PhaseDConfig, PhaseDLoop
from .precision_arena import run
from .train import make_checkpoint


def _tiny_checkpoint(tmp_path: Path) -> Path:
    from .train import build_model

    model = build_model("transformer", 32, 1, 4)
    checkpoint = make_checkpoint(
        model,
        {"model": "transformer", "d_model": 32, "layers": 1, "heads": 4, "iteration": 0},
    )
    path = tmp_path / "arm.pt"
    torch.save(checkpoint, path)
    return path


def test_the_two_arms_are_the_same_weights_at_different_precisions(tmp_path, monkeypatch):
    seen = {}
    from . import phase_d

    real = PhaseDLoop._rust_model_gate_rolling

    def spy(self, candidate_spec, opponent_spec, seed_offset, games, precisions=None):
        seen["precisions"] = precisions
        seen["same_weights"] = all(
            torch.equal(candidate_spec.model_state[key], opponent_spec.model_state[key])
            for key in candidate_spec.model_state
        )
        seen["games"] = games
        raise SystemExit("stop before playing")

    monkeypatch.setattr(PhaseDLoop, "_rust_model_gate_rolling", spy)
    with pytest.raises(SystemExit):
        run(
            _tiny_checkpoint(tmp_path),
            games=4,
            device="cpu",
            sims=1,
            slots=2,
            global_batch_cap=8,
            work_dir=tmp_path / "work",
        )
    assert seen["precisions"] == ("bf16", "fp32")
    assert seen["same_weights"], "an arena over different weights measures nothing"
    assert seen["games"] == 4


def test_per_side_precision_defaults_to_the_run_precision(tmp_path):
    # Without an explicit pair, a gate must keep using one precision for both
    # sides -- the arena is the exception, not the new normal.
    config = PhaseDConfig(run_dir=str(tmp_path / "run"), precision="bf16", device="cpu")
    loop = PhaseDLoop(config)
    assert loop.config.precision == "bf16"


def test_the_null_is_inside_a_wide_interval_and_outside_a_lopsided_one():
    from games.az_loop import wilson_interval

    # A dead-even arena: 0.500 must sit inside.
    lower, upper = wilson_interval(10.0, 20)
    assert lower <= 0.50 <= upper
    # A one-sided sweep at the same size: it must not.
    lower, upper = wilson_interval(20.0, 20)
    assert not lower <= 0.50 <= upper


def test_the_arena_actually_runs_end_to_end_on_cpu(tmp_path):
    """Executes the body, not a mock of it.

    The mocked test above verifies the *call*; it cannot catch a mistake inside
    the runner. This one plays real games through the per-side-precision path.
    """

    report = run(
        _tiny_checkpoint(tmp_path),
        games=2,
        device="cpu",
        sims=1,
        slots=2,
        global_batch_cap=8,
        work_dir=tmp_path / "work",
    )
    assert report["games"] == 2
    assert report["pairs"] == 1
    assert report["arms"] == {"candidate": "bf16", "opponent": "fp32"}
    assert 0.0 <= report["bf16_score_rate"] <= 1.0
    assert report["wilson"]["lower"] <= report["wilson"]["upper"]
    assert len(report["pair_scores"]) == 1


class _FixedDtypeModel(torch.nn.Module):
    """A net whose outputs are in a chosen dtype, whatever the input is.

    Stands in for a real model under CUDA autocast: `Evaluator.autocast` is a
    null context on CPU, so on a CPU box both arms of the arena compute in
    float32 and the merge below never sees two dtypes. That is exactly why
    `test_the_arena_actually_runs_end_to_end_on_cpu` executes this path without
    exercising it.
    """

    def __init__(self, dtype, width=3):
        super().__init__()
        self.dtype_out = dtype
        self.width = width

    def forward(self, batch):
        rows = len(next(iter(batch.values())))
        return {
            "policy": torch.full((rows, self.width), 0.5, dtype=self.dtype_out),
            "value": torch.full((rows, 1), 0.25, dtype=self.dtype_out),
        }


def test_routing_merges_two_precisions_without_a_dtype_clash():
    """The bug W6.2b's arena hit on its first real run.

    The merged buffer was allocated from whichever net was evaluated first, so
    with one bf16 arm and one fp32 arm `index_copy_` raised "self and source
    expected to have the same dtype". Mixed precision is the arena's entire
    treatment, so this failed 100% of the time on CUDA and 0% on CPU.
    """

    from .rust_bridge import rust_searcher_routed_flat_batch_adapter

    class _Stub:
        def __init__(self, dtype, precision):
            self.model = _FixedDtypeModel(dtype)
            self.device = "cpu"
            self.max_batch = 8
            self.precision = precision

        def autocast(self):
            import contextlib

            return contextlib.nullcontext()

    adapter = rust_searcher_routed_flat_batch_adapter(
        [_Stub(torch.bfloat16, "bf16"), _Stub(torch.float32, "fp32")]
    )
    model = adapter.evaluator.model
    batch = {
        "net_ids": torch.tensor([0, 1, 0, 1]),
        "features": torch.zeros((4, 2)),
    }

    combined = model(batch)

    # Widened to the common dtype, and every row is filled by its own net.
    assert combined["value"].dtype is torch.float32
    assert combined["policy"].shape == (4, 3)
    assert torch.allclose(combined["value"], torch.full((4, 1), 0.25))


def test_single_precision_routing_keeps_the_nets_own_dtype():
    """The fix must not touch the path W1's routing equivalence was verified on."""

    from .rust_bridge import rust_searcher_routed_flat_batch_adapter

    class _Stub:
        def __init__(self):
            self.model = _FixedDtypeModel(torch.float32)
            self.device = "cpu"
            self.max_batch = 8
            self.precision = "fp32"

        def autocast(self):
            import contextlib

            return contextlib.nullcontext()

    adapter = rust_searcher_routed_flat_batch_adapter([_Stub(), _Stub()])
    assert adapter.evaluator.model.autocasts is None
    combined = adapter.evaluator.model(
        {"net_ids": torch.tensor([0, 1]), "features": torch.zeros((2, 2))}
    )
    assert combined["value"].dtype is torch.float32


def test_a_normal_gate_still_uses_one_precision_for_both_sides(tmp_path):
    """Regression: per-side precision must not leak into ordinary gates."""

    from .phase_d import PhaseDConfig, PhaseDLoop

    seen = []
    # Relative import: the suite runs as `seven_wonders_duel.*`, so
    # `import games.seven_wonders_duel.phase_d` would bind a *second* copy of
    # the module and patch something the code under test never looks at.
    from . import phase_d

    real_evaluator = phase_d.Evaluator

    class _Spy(real_evaluator):  # type: ignore[misc,valid-type]
        def __init__(self, model, device, cap, precision="fp32", **kwargs):
            seen.append(precision)
            super().__init__(model, device, cap, precision=precision, **kwargs)

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        device="cpu",
        d_model=32,
        layers=1,
        heads=4,
        precision="fp32",
        gate_backend="rust",
        gate_sims=1,
        gate_max_games=2,
        gate_slots=2,
        rust_slots=2,
        rust_global_batch_cap=8,
        seed_games=0,
        promotion_every=0,
    )
    loop = PhaseDLoop(config)
    spec = loop._model_agent_spec(_tiny_checkpoint(tmp_path), "candidate")
    phase_d.Evaluator = _Spy
    try:
        loop._rust_model_gate_rolling(spec, spec, 1_000, 2)
    finally:
        phase_d.Evaluator = real_evaluator
    assert seen == ["fp32", "fp32"]
