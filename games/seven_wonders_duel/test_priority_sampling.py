"""G3: the sampling mixture, its signals, and the trainer's use of it."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from . import priority_sampling as ps
from .phase_e import fresh_bot_records
from .dataset import examples_from_record


def _signals(n=1000, proof_every=50, seed=0):
    rng = np.random.default_rng(seed)
    surprise = rng.exponential(1.0, n)
    surprise[::7] = np.nan  # value-only rows carry no policy label
    correction = rng.exponential(0.3, n)
    correction[::5] = np.nan
    proof = np.zeros(n, dtype=bool)
    proof[::proof_every] = True
    return ps.Signals(surprise, correction, proof)


def test_the_mixture_is_a_distribution_with_a_uniform_floor_and_a_cap():
    signals = _signals()
    for share, cap in ((0.3, 2.0), (0.0, 2.0), (1.0, 2.0), (0.3, 1.0), (0.5, 4.0)):
        prio = ps.priorities(signals, cap)
        p = ps.mixture(prio, uniform_share=share, cap=cap)
        n = len(p)
        assert p.sum() == pytest.approx(1.0)
        # Every row keeps at least its uniform share...
        assert p.min() >= share / n - 1e-15
        # ...and none is drawn more than share + (1 - share) * cap times uniform.
        assert p.max() * n <= share + (1 - share) * cap + 1e-9
    uniform = ps.mixture(ps.priorities(signals), uniform_share=1.0)
    assert np.allclose(uniform, 1 / len(uniform))


def test_the_bound_holds_for_sparse_and_proof_heavy_signals():
    """The review's counterexample: one surprising row among 1,000 drew
    700x uniform under clip-then-renormalise."""

    n = 1000
    surprise = np.zeros(n)
    surprise[0] = 1.0
    sparse = ps.Signals(surprise, np.full(n, np.nan), np.zeros(n, dtype=bool))
    for signals in (
        sparse,
        ps.Signals(surprise, np.full(n, np.nan), np.arange(n) % 100 == 0),
        ps.Signals(surprise, np.full(n, np.nan), np.ones(n, dtype=bool)),
    ):
        p = ps.mixture(ps.priorities(signals), uniform_share=0.3, cap=2.0)
        assert p.sum() == pytest.approx(1.0)
        assert p.max() * n <= 1.7 + 1e-9
        assert p.min() * n >= 0.3 - 1e-9


def test_proof_rows_are_drawn_at_the_cap():
    signals = _signals()
    prio = ps.priorities(signals, cap=2.0)
    p = ps.mixture(prio, uniform_share=0.3, cap=2.0)
    report = ps.sampling_report(p, signals)
    # Pinned at the cap: well above uniform, and the most any row can get.
    assert report["proof_relative"] > 1.3
    assert report["proof_relative"] == pytest.approx(report["max_relative"])
    assert report["effective_share"] < 1.0


def test_signals_are_normalised_so_neither_dominates_by_scale():
    base = _signals()
    scaled = ps.Signals(base.surprise, base.value_correction * 1000.0, base.proof)
    assert np.allclose(ps.priorities(base), ps.priorities(scaled))


def test_bad_settings_are_refused():
    prio = ps.priorities(_signals())
    with pytest.raises(ValueError):
        ps.mixture(prio, uniform_share=1.5)
    with pytest.raises(ValueError):
        ps.mixture(prio, cap=0.5)


@pytest.fixture(scope="module")
def examples():
    rows = []
    for record in fresh_bot_records(4, seed=31):
        rows.extend(examples_from_record(record, record_fast_moves=True))
    # Give a handful of rows a search value and a proof, as self-play would.
    rows = [
        replace(e, root_value=0.2 if i % 3 == 0 else None,
                solver_value=1.0 if i % 40 == 0 else None,
                solver_exact=i % 40 == 0)
        for i, e in enumerate(rows)
    ]
    return rows


def test_row_signals_run_on_a_model_and_leave_its_mode_alone(examples):
    from .train import build_model

    model = build_model("transformer", 32, 1)
    model.train()
    signals = ps.row_signals(model, examples, "cpu", batch_size=64)
    assert model.training
    n = len(examples)
    assert signals.surprise.shape == (n,) and signals.value_correction.shape == (n,)
    with_policy = np.array([e.has_policy for e in examples])
    assert np.isnan(signals.surprise[~with_policy]).all()
    assert (signals.surprise[with_policy] >= 0).all()
    with_root = np.array([e.root_value is not None for e in examples])
    assert np.isnan(signals.value_correction[~with_root]).all()
    assert (signals.proof == np.array([ps.is_proof_row(e) for e in examples])).all()
    assert signals.proof.sum() >= sum(1 for i in range(n) if i % 40 == 0)


def test_the_trainer_draws_by_the_weights(examples):
    from .train import build_model, train_steps

    n = len(examples)
    weights = np.full(n, 1e-9)
    favoured = [i for i in range(n) if examples[i].solver_value is not None]
    weights[favoured] = 1.0
    weights /= weights.sum()
    history, _ = train_steps(
        build_model("transformer", 32, 1), examples, None,
        device="cpu", steps=3, batch_size=32, validate_every=1000,
        sample_weights=weights, log=lambda *_: None,
    )
    # Every draw lands on a favoured (proof) row.
    final = history[-1]
    assert final["sampled_proof_rows"] == final["sampled_rows"] == 96
    with pytest.raises(ValueError):
        train_steps(build_model("transformer", 32, 1), examples, None, device="cpu",
                    steps=1, batch_size=8, sample_weights=weights[:-1], log=lambda *_: None)


def test_phase_d_refuses_bad_priority_settings():
    from .phase_d import PhaseDConfig

    with pytest.raises(ValueError):
        PhaseDConfig(priority_uniform_share=1.2).validate()
    with pytest.raises(ValueError):
        PhaseDConfig(priority_cap=0.5).validate()
