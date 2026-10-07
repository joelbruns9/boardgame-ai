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
    # A row this route does not train on is filtered all the same: projection
    # to another route can switch its policy on later (review of 8014a6c, #4).
    value_only = replace(example, has_policy=False)
    out = apply_tactic_labels(value_only, labels)
    assert out.has_policy is False
    assert out.policy_target[0] == pytest.approx(1.0)
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


def _as_searched(record, route=None):
    """The bot game with every move recorded as a full search (uniform target),
    so replies and policy rows exist; `route` relabels whose model it trains."""

    legal = {e.move_index: e.legal for e in examples_from_record(record)}
    moves = [
        replace(m, sims=800, policy_excluded=False,
                policy_target={int(a): 1.0 / len(legal[m.i]) for a in legal[m.i]},
                **({"target_route": route} if route else {}))
        for m in record.moves
    ]
    return replace(record, moves=moves)


def test_reply_targets_follow_the_corrected_move_targets(records):
    """Review of 8014a6c, #2: a reply label is the FOLLOWING row's final move
    target, so it never keeps mass G2b removed from that row."""

    from .dataset import sync_reply_targets  # noqa: F401  (the contract under test)

    searched = [_as_searched(r) for r in records]
    rust = derive_records_rust(searched, tactic_labels=True, batch_games=4)
    synced = 0
    for record, (rust_rows, _stats) in zip(searched, rust):
        python_rows = examples_from_record(record, tactic_labels=True)
        for rows in (python_rows, rust_rows):
            by_move = {e.move_index: e for e in rows}
            for row in rows:
                if row.reply_target is None:
                    continue
                following = by_move[row.move_index + 1]
                assert np.array_equal(row.reply_legal, following.legal)
                assert np.allclose(row.reply_target, following.policy_target)
                if following.tactic in (TACTIC_WIN, TACTIC_BLOCK):
                    synced += 1
        for a, b in zip(python_rows, rust_rows):
            assert (a.reply_target is None) == (b.reply_target is None)
            if a.reply_target is not None:
                assert np.allclose(a.reply_target, b.reply_target)
    assert synced, "no corrected reply exercised"
    # Without corrections the replies are byte-identical to the raw pairing.
    plain = examples_from_record(searched[0])
    for row in plain:
        if row.reply_target is not None:
            assert row.reply_target.sum() == pytest.approx(1.0)


def test_a_cached_general_row_projects_to_the_directly_derived_specialist_row(records):
    """Review of 8014a6c, #4: Phase D caches the general derivation and
    projects it for a specialist; the projected policy must be the one direct
    derivation for that specialist produces."""

    from .dataset import project_examples

    checked = 0
    for record in records:
        science = _as_searched(record, route="science")
        cached = examples_from_record(science, tactic_labels=True)
        direct = examples_from_record(science, tactic_labels=True, derived_for="science")
        projected = project_examples(cached, science, "science")
        for p, d in zip(projected, direct):
            assert p.has_policy == d.has_policy
            assert np.allclose(p.policy_target, d.policy_target)
            if d.tactic == TACTIC_BLOCK and d.has_policy:
                checked += 1
    assert checked, "no specialist must_block row exercised"


def test_the_phase_out_census_counts_the_changes_g2b_makes(records):
    """G2b phase-out measure: searched bot games put uniform mass on proven
    losing moves, so the census must count real target changes; a game whose
    targets already avoid them must count none."""

    from .g2b_census import census

    searched = [_as_searched(record) for record in records]
    result = census(searched, retain=0)
    assert result["labelled"] > 0
    assert 0 < result["policy_changed"] <= result["labelled"]
    assert result["value_changed"] > 0
    assert set(result["by_class"]) <= {"win", "all_lose", "block"}

    # Clean those same targets the way G2b would; the census then sees nothing
    # left to change in the policy.
    cleaned = []
    for record in searched:
        rows = {e.move_index: e for e in examples_from_record(record, tactic_labels=True)}
        moves = []
        for move in record.moves:
            row = rows.get(move.i)
            if row is None or not row.has_policy:
                moves.append(move)
                continue
            target = {int(a): float(p) for a, p in zip(row.legal, row.policy_target) if p > 0}
            moves.append(replace(move, policy_target=target))
        cleaned.append(replace(record, moves=moves))
    assert census(cleaned, retain=0)["policy_changed"] == 0


def test_the_census_counts_specialist_owned_policy_corrections(records):
    """Review of ebc70c0, #2: a specialist-owned row has no policy in the
    general projection, yet G2b rewrites the target that specialist trains on."""

    from .g2b_census import census

    owned = [_as_searched(record, route="specialist:1") for record in records]
    result = census(owned, retain=0)
    assert result["routes"]["general"]["policy_rows"] == 0
    assert result["routes"]["specialist:1"]["policy_rows"] > 0
    assert result["routes"]["specialist:1"]["policy_changed"] > 0
    assert result["policy_changed_any_route"] == result["routes"]["specialist:1"]["policy_changed"]


def test_the_census_counts_an_exactness_change_at_an_unchanged_value(records, monkeypatch):
    """Review of ebc70c0, #3: +1 expectimax -> +1 exact switches the row from
    utility to proof supervision, so it is a value change."""

    from . import g2b_census

    record = _as_searched(records[0])
    base = [
        replace(row, tactic=TACTIC_NONE, solver_value=None, solver_exact=False,
                certain_win=False)
        for row in examples_from_record(record)
    ]
    expectimax = list(base)
    expectimax[3] = replace(base[3], solver_value=1.0, solver_exact=False)
    proven = list(base)
    proven[3] = replace(base[3], solver_value=1.0, solver_exact=True, tactic=TACTIC_WIN)
    monkeypatch.setattr(
        g2b_census, "_derive",
        lambda records, *, tactic_labels, retain: [proven if tactic_labels else expectimax],
    )
    result = g2b_census.census([record], retain=0)
    assert result["value_changed"] == 1
    assert result["value_exactness_only"] == 1
    assert result["value_changed_effective"] == 1

    # Under the certain-win rule the exact win is the target either way.
    proven[3] = replace(proven[3], certain_win=True)
    expectimax[3] = replace(expectimax[3], certain_win=True)
    result = g2b_census.census([record], retain=0)
    assert result["value_changed"] == 1
    assert result["value_changed_effective"] == 0
