"""G2b: exact one-move tactics folded into training targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

swr = pytest.importorskip("seven_wonders_rust")

from . import phase_e as pe
from .dataset import (
    TACTIC_BLOCK,
    TACTIC_LOSS,
    TACTIC_NONE,
    TACTIC_WIN,
    apply_tactic_labels,
    derive_records_rust,
    examples_from_record,
)


@pytest.fixture(scope="module")
def records():
    # Rush-bot games reach forced wins, losses and must-block positions often.
    return pe.fresh_bot_records(12, seed=2024)


def _example(records, has_policy=True):
    example = examples_from_record(records[0])[5]
    n = len(example.legal)
    policy = np.full(n, 1.0 / n, dtype=np.float32)
    return replace(example, has_policy=has_policy, policy_target=policy,
                   solver_value=None, solver_exact=False)


def test_a_forced_win_sets_the_value_and_moves_the_target_to_winning_moves(records):
    example = _example(records)
    n = len(example.legal)
    labels = [0] * n
    labels[1] = 1
    labels[2] = -1
    out = apply_tactic_labels(example, labels)
    assert out.tactic == TACTIC_WIN
    assert out.solver_value == 1.0 and out.solver_exact
    assert out.policy_target[1] == pytest.approx(1.0)
    assert out.policy_target.sum() == pytest.approx(1.0)


def test_every_move_losing_sets_an_exact_loss_and_keeps_the_target(records):
    example = _example(records)
    out = apply_tactic_labels(example, [-1] * len(example.legal))
    assert out.tactic == TACTIC_LOSS
    assert out.solver_value == -1.0 and out.solver_exact
    assert np.array_equal(out.policy_target, example.policy_target)


def test_must_block_strips_losing_moves_and_renormalises(records):
    example = _example(records)
    n = len(example.legal)
    assert n >= 3
    labels = [-1] + [0] * (n - 1)
    out = apply_tactic_labels(example, labels)
    assert out.tactic == TACTIC_BLOCK
    assert out.solver_value is None  # nothing about the value is proven
    assert out.policy_target[0] == 0.0
    assert out.policy_target.sum() == pytest.approx(1.0)
    # All of search's mass on the losing move: uniform over the rest.
    only_losing = replace(example, policy_target=np.eye(n, dtype=np.float32)[0])
    out = apply_tactic_labels(only_losing, labels)
    assert out.policy_target[0] == 0.0
    assert np.allclose(out.policy_target[1:], 1.0 / (n - 1))


def test_nothing_proven_changes_nothing_and_exact_solver_values_win(records):
    example = _example(records)
    assert apply_tactic_labels(example, [0] * len(example.legal)) is example
    solved = replace(example, solver_value=-1.0, solver_exact=True)
    labels = [0] * len(example.legal)
    labels[0] = 1
    assert apply_tactic_labels(solved, labels).solver_value == -1.0
    value_only = replace(example, has_policy=False)
    out = apply_tactic_labels(value_only, labels)
    assert np.array_equal(out.policy_target, example.policy_target)
    with pytest.raises(ValueError):
        apply_tactic_labels(example, [0] * (len(example.legal) + 1))


def test_both_backends_label_identically(records):
    counts = Counter()
    rust = derive_records_rust(records, tactic_labels=True, batch_games=4)
    for record, (rust_rows, _stats) in zip(records, rust):
        python_rows = examples_from_record(record, tactic_labels=True)
        assert len(python_rows) == len(rust_rows)
        for a, b in zip(python_rows, rust_rows):
            assert a.tactic == b.tactic
            assert a.solver_value == b.solver_value and a.solver_exact == b.solver_exact
            assert np.allclose(a.policy_target, b.policy_target)
            counts[a.tactic] += 1
    # The gate must not pass vacuously.
    assert counts[TACTIC_WIN] and counts[TACTIC_BLOCK], counts
    plain = derive_records_rust(records[:2], batch_games=2)
    assert all(e.tactic == TACTIC_NONE for rows, _ in plain for e in rows)


def test_labelled_rows_count_as_proof_rows_for_g3(records):
    from .priority_sampling import is_proof_row

    rows = examples_from_record(records[0], tactic_labels=True)
    for row in rows:
        if row.tactic in (TACTIC_WIN, TACTIC_LOSS):
            assert is_proof_row(row)
