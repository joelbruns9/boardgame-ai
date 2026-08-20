"""The full-sims ramp, and the evaluation leaf batch.

Both exist because a value was being decided somewhere other than where the
operator sets it: evaluation hardcoded leaf_batch=1 regardless of config, and
simulations were fixed for a whole run because there was no way to say
otherwise.
"""

from __future__ import annotations

import pytest

from .phase_d import PhaseDConfig


def _config(**overrides) -> PhaseDConfig:
    base = dict(run_dir="x")
    base.update(overrides)
    config = PhaseDConfig(**base)
    config.validate()
    return config


# ── The ramp ─────────────────────────────────────────────────────────────────


def test_no_schedule_leaves_the_configured_budget_alone():
    config = _config(full_sims_min=1600, full_sims_max=1600)
    assert config.full_sims_knots() == ()
    assert config.full_sims_at(0) is None
    assert config.full_sims_at(10**9) is None


def test_the_ramp_is_piecewise_constant_not_interpolated():
    """Interpolation would make every iteration incomparable with every other,
    rather than only those across a step."""

    config = _config(full_sims_schedule="0:400,10000:900,25000:1600")
    assert config.full_sims_at(0) == 400
    assert config.full_sims_at(9_999) == 400
    assert config.full_sims_at(10_000) == 900
    assert config.full_sims_at(24_999) == 900, "no interpolation between knots"
    assert config.full_sims_at(25_000) == 1600
    assert config.full_sims_at(10**9) == 1600, "the last knot holds"


def test_knots_may_be_given_out_of_order():
    config = _config(full_sims_schedule="25000:1600,0:400,10000:900")
    assert config.full_sims_at(10_000) == 900


def test_a_schedule_without_a_value_at_zero_is_refused():
    """Otherwise the budget before the first knot is undefined, and the run
    would silently take whichever value the lookup happened to seed with."""

    with pytest.raises(ValueError, match="at 0 games"):
        _config(full_sims_schedule="10000:900")


@pytest.mark.parametrize(
    "text", ["0:400,nonsense", "0:400,10000", "0:0", "-1:400", "0:400,0:900"]
)
def test_malformed_schedules_are_refused_at_construction(text):
    """Not at the first iteration, twenty minutes into a rented box."""

    with pytest.raises(ValueError):
        _config(full_sims_schedule=text)


def test_the_ramp_is_part_of_the_schedule_identity():
    """A ramp changed across a resume moves every later iteration's meaning,
    exactly as the curriculum's would."""

    identity = _config(full_sims_schedule="0:400,10000:1600").schedule_identity()
    assert identity["full_sims_schedule"] == "0:400,10000:1600"


# ── The evaluation leaf batch ────────────────────────────────────────────────


def test_evaluation_follows_the_training_leaf_batch_by_default():
    """Gates hardcoded 1, so evaluation played a different search from training
    no matter what was configured -- and the advisor is certified by the gate."""

    assert _config().evaluation_leaf_batch() == 1
    assert _config(leaf_batch=6, virtual_loss_root=True).evaluation_leaf_batch() == 6


def test_evaluation_can_be_overridden_independently():
    config = _config(leaf_batch=6, eval_leaf_batch=4, virtual_loss_root=True)
    assert config.evaluation_leaf_batch() == 4


def test_a_batched_evaluation_under_puct_needs_the_opt_in():
    with pytest.raises(ValueError, match="--virtual-loss-root"):
        _config(eval_search_mode="puct", eval_leaf_batch=6)
    # Gumbel evaluation has no PUCT root to protect.
    assert _config(eval_search_mode="gumbel", eval_leaf_batch=6).evaluation_leaf_batch() == 6


def test_the_ramp_steps_on_the_real_games_clock(tmp_path):
    """End to end against the ledger, not the config object. The ledger counts
    records in the per-iteration buffer files, and `generation_clock` reads
    games BEFORE the iteration -- a schedule keyed on games *through* it would
    depend on the games it is deciding how to generate."""

    from .phase_d import PhaseDLoop

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"), schedule_basis="games",
        games_per_iteration=1000, full_sims_min=1600, full_sims_max=1600,
        full_sims_schedule="0:400,2000:900,5000:1600",
        d_model=32, layers=1, heads=4,
    )
    config.validate()
    loop = PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(8):
        loop.buffer_dir.joinpath(f"iter_{iteration:04d}.jsonl").write_text(
            "\n".join('{"x":1}' for _ in range(1000)), encoding="utf-8"
        )
    loop.games_ledger.refresh()

    resolved = {i: loop.resolved_schedules(i).full_sims for i in range(8)}
    assert resolved[0] == 400
    assert resolved[1] == 400, "still before the 2000-game knot"
    assert resolved[2] == 900
    assert resolved[4] == 900, "holds until the next knot"
    assert resolved[5] == 1600
    assert resolved[7] == 1600


def test_an_iterations_basis_run_ignores_the_games_ramp(tmp_path):
    """The ramp is defined on the games clock. Under the iterations basis there
    is no such clock, and silently applying it at iteration counts would be the
    W1.2 rescaling bug in a new place."""

    from .phase_d import PhaseDLoop

    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"), schedule_basis="iterations",
        full_sims_schedule="0:400,2000:900",
        d_model=32, layers=1, heads=4,
    )
    config.validate()
    loop = PhaseDLoop(config)
    assert loop.resolved_schedules(0).full_sims is None


# ── Wiring, not just API ─────────────────────────────────────────────────────
# Both of the following were added after a mutation SURVIVED: reverting the eval
# call site to `leaf_batch=1`, and dropping the ramp from the generation call,
# left every test above green. They checked that the config could compute the
# right value, never that the call sites used it -- which is the entire defect.


def _phase_d_source() -> str:
    from pathlib import Path

    return (Path(__file__).with_name("phase_d.py")).read_text(encoding="utf-8")


def test_both_evaluation_call_sites_use_the_configured_leaf_batch():
    """The gate and the arena. A config value nothing reads is not a setting."""

    source = _phase_d_source()
    uses = source.count("leaf_batch=self.config.evaluation_leaf_batch()")
    assert uses == 2, (
        f"expected the gate and arena to use evaluation_leaf_batch(), found "
        f"{uses}. Evaluation hardcoding leaf_batch=1 is the bug this replaced."
    )


def test_generation_takes_the_scheduled_sims_when_there_is_one():
    """`full_sims_min` and `full_sims_max` BOTH have to follow the ramp; taking
    the schedule for one and the config for the other would sample a range
    between a scheduled floor and a fixed ceiling."""

    source = _phase_d_source()
    assert source.count("schedules.full_sims\n                if schedules.full_sims is not None") == 2, (
        "the generation call must take the scheduled value for both bounds"
    )


# ── Per-path wave mode ───────────────────────────────────────────────────────
# The two batching mechanisms are mutually exclusive, and the global flag
# silently disabled the one that matters on the path carrying 1600 simulations.


def test_the_cheap_wave_flags_must_be_set_together():
    """Conflict-free waves without round-robin cut every wave to width 1, and
    round-robin without them only changes the search. Either alone is a config
    that costs something and buys nothing."""

    with pytest.raises(ValueError, match="must be set together"):
        _config(cheap_conflict_free_waves=True, cheap_round_robin_candidates=False)
    with pytest.raises(ValueError, match="must be set together"):
        _config(cheap_conflict_free_waves=False, cheap_round_robin_candidates=True)


def test_cheap_leaf_batching_accepts_the_per_path_waves():
    """The configuration the run actually wants: cheap batches without
    collisions, full batches under virtual loss."""

    config = _config(
        selfplay_search_mode="puct", cheap_search_mode="gumbel",
        leaf_batch=6, virtual_loss_root=True, cheap_leaf_batch=16,
        cheap_conflict_free_waves=True, cheap_round_robin_candidates=True,
        conflict_free_waves=False, round_robin_candidates=False,
    )
    assert config.cheap_conflict_free_waves and not config.conflict_free_waves


def test_the_launcher_uses_the_cheap_wave_flags_not_the_global_ones():
    """Measured 2026-08-20: wave 1.88 per-path against 1.56 global. The global
    flag collapses PUCT-root width to ~1.19, so a run using it batches only on
    the cheap path while believing it batches on both."""

    from pathlib import Path

    launcher = (
        Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in launcher.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--cheap-conflict-free-waves" in code
    assert "--cheap-round-robin-candidates" in code
    # The global forms must not be passed: they would override the cheap ones
    # for full moves and undo the whole point.
    assert "LEAF_BATCH_FLAGS+=(--conflict-free-waves" not in code
