"""Numerical correctness of the regret measurement, on hand-checkable cases.

A measurement script fails differently from a feature: a bug here does not raise,
it returns a plausible wrong number that gets believed. Three smoke-tested
positions demonstrated execution, not correctness -- these fix the arithmetic,
the direction, the alignment and the aggregation against cases small enough to
verify by hand.

The cases deliberately include the one that motivated the review: a prediction
whose DISTRIBUTION improves while its top-1 choice is unchanged. A metric that
cannot separate those two cannot support a claim about evaluation.
"""

from __future__ import annotations

import json

import pytest

from .w3_corpus_regret import reference_positions


def _artifact(tmp_path, name, values, *, decision_row=0, checkpoint="ref.pt"):
    """One reference artifact: action index -> weighted win percentage."""

    payload = {
        "position": {
            "log": "runs/seven_wonders_duel/bga_game_log/table_1.jsonl",
            "decision_row": decision_row,
            "resample_seed": 0,
        },
        "checkpoint": {"path": checkpoint},
        "reference_values": {
            "actions": [
                {"index": index, "label": f"action_{index}",
                 "win_pct_weighted": value}
                for index, value in values.items()
            ]
        },
    }
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _regret(values: dict[int, float], chosen: int):
    """The measurement under test, isolated from search and checkpoints."""

    best = max(values.values())
    got = values.get(chosen)
    return None if got is None else round(best - got, 3)


# -- the arithmetic ---------------------------------------------------------


def test_choosing_the_reference_best_is_zero_regret():
    values = {1: 80.0, 2: 55.0, 3: 20.0}
    assert _regret(values, 1) == 0.0


def test_regret_is_the_gap_to_the_best_not_to_the_next():
    """A common off-by-one: regret against the runner-up understates every
    mistake except the second-best one."""

    values = {1: 80.0, 2: 55.0, 3: 20.0}
    assert _regret(values, 3) == 60.0          # 80 - 20, not 55 - 20
    assert _regret(values, 2) == 25.0


def test_regret_is_never_negative_and_lower_is_better():
    """Direction. A sign flip here would rank the worst arm first."""

    values = {1: 80.0, 2: 55.0, 3: 20.0}
    worst = _regret(values, 3)
    better = _regret(values, 2)
    best = _regret(values, 1)
    assert best == 0.0
    assert 0.0 <= best < better < worst


def test_an_action_the_reference_never_scored_is_unscored_not_zero():
    """Dropping these silently would bias the mean toward arms that stay inside
    the reference's action set -- and scoring them 0 would reward picking an
    action the reference never considered."""

    values = {1: 80.0, 2: 55.0}
    assert _regret(values, 99) is None


def test_a_single_action_position_has_zero_regret_for_that_action():
    """A forced position cannot discriminate between arms; it must not produce
    a spurious nonzero."""

    assert _regret({7: 42.0}, 7) == 0.0


# -- alignment: the index is the contract -----------------------------------


def test_regret_is_keyed_by_action_index_not_by_rank(tmp_path):
    """The reference stores actions best-first. Looking a choice up by POSITION
    in that list rather than by codec index would silently score the wrong
    action -- and would still return a plausible number."""

    _artifact(tmp_path, "a.json", {49: 87.1, 506: 84.3, 122: 30.0})
    positions = reference_positions(tmp_path)
    assert len(positions) == 1
    values = positions[0]["values"]
    assert values[49] == 87.1 and values[506] == 84.3
    # rank 0 is index 49, not index 0
    assert 0 not in values
    assert _regret(values, 506) == pytest.approx(2.8)


def test_labels_travel_with_the_index(tmp_path):
    _artifact(tmp_path, "a.json", {49: 87.1, 506: 84.3})
    position = reference_positions(tmp_path)[0]
    assert position["labels"][49] == "action_49"


# -- what the reference loader includes and excludes -------------------------


def test_summary_and_recheck_artifacts_are_not_read_as_positions(tmp_path):
    """`summary_shard*.json` and `_recheck` files sit in the same directory.
    Counting a summary as a position would inflate n and average in a file that
    has no action values."""

    _artifact(tmp_path, "real.json", {1: 50.0})
    _artifact(tmp_path, "real_recheck.json", {1: 50.0})
    (tmp_path / "summary_shard0.json").write_text(
        json.dumps({"positions": []}), encoding="utf-8"
    )
    (tmp_path / "triage_report.json").write_text(
        json.dumps({"cells": []}), encoding="utf-8"
    )
    names = {p["artifact"] for p in reference_positions(tmp_path)}
    assert names == {"real.json"}


def test_a_mixed_reference_directory_is_visible(tmp_path):
    """Regret is only comparable across positions if one model produced every
    reference value. A directory mixing checkpoints must be detectable."""

    _artifact(tmp_path, "a.json", {1: 50.0}, checkpoint="model_a.pt")
    _artifact(tmp_path, "b.json", {1: 50.0}, decision_row=1, checkpoint="model_b.pt")
    checkpoints = {p["checkpoint"] for p in reference_positions(tmp_path)}
    assert checkpoints == {"model_a.pt", "model_b.pt"}


# -- aggregation ------------------------------------------------------------


def _summarise(rows):
    """Mirror of the harness's aggregation, over hand-made rows."""

    summary = {}
    for row in rows:
        entry = summary.setdefault(
            row["arm"], {"n": 0, "scored": 0, "regret": 0.0, "agree": 0, "unscored": 0}
        )
        entry["n"] += 1
        if row["regret"] is None:
            entry["unscored"] += 1
            continue
        entry["scored"] += 1
        entry["regret"] += row["regret"]
        entry["agree"] += int(bool(row["agrees_with_reference_best"]))
    for entry in summary.values():
        entry["mean_regret"] = (
            round(entry["regret"] / entry["scored"], 3) if entry["scored"] else None
        )
        entry["agreement"] = (
            round(entry["agree"] / entry["scored"], 3) if entry["scored"] else None
        )
    return summary


def test_unscored_rows_do_not_dilute_the_mean():
    rows = [
        {"arm": "a", "regret": 10.0, "agrees_with_reference_best": False},
        {"arm": "a", "regret": 0.0, "agrees_with_reference_best": True},
        {"arm": "a", "regret": None, "agrees_with_reference_best": None},
    ]
    summary = _summarise(rows)["a"]
    assert summary["n"] == 3 and summary["scored"] == 2 and summary["unscored"] == 1
    assert summary["mean_regret"] == 5.0        # not 10/3
    assert summary["agreement"] == 0.5


def test_an_arm_with_nothing_scored_reports_none_not_zero():
    """Zero mean regret is the BEST possible score. An arm that scored nothing
    must not be reported as perfect."""

    rows = [{"arm": "a", "regret": None, "agrees_with_reference_best": None}]
    summary = _summarise(rows)["a"]
    assert summary["mean_regret"] is None
    assert summary["agreement"] is None


def test_a_strictly_worse_arm_ranks_worse():
    """End to end over the aggregation: an arm that picks a worse action at
    every position must not come out ahead."""

    values = {1: 80.0, 2: 55.0, 3: 20.0}
    good = [{"arm": "good", "regret": _regret(values, 1),
             "agrees_with_reference_best": True} for _ in range(5)]
    bad = [{"arm": "bad", "regret": _regret(values, 3),
            "agrees_with_reference_best": False} for _ in range(5)]
    summary = _summarise(good + bad)
    assert summary["good"]["mean_regret"] < summary["bad"]["mean_regret"]
    assert summary["good"]["agreement"] > summary["bad"]["agreement"]


def test_identical_arms_score_identically():
    """The null the experiment must be able to report: two arms that behave the
    same must not be separated by the measurement itself."""

    values = {1: 80.0, 2: 55.0}
    rows = []
    for arm in ("a", "b"):
        rows += [{"arm": arm, "regret": _regret(values, 2),
                  "agrees_with_reference_best": False} for _ in range(4)]
    summary = _summarise(rows)
    assert summary["a"]["mean_regret"] == summary["b"]["mean_regret"]
    assert summary["a"]["agreement"] == summary["b"]["agreement"]


# -- the case that motivated the review -------------------------------------


def test_regret_cannot_see_a_distribution_that_improves_under_an_unchanged_top1():
    """Documented limitation, pinned so it is not mistaken for a bug later.

    Two arms pick the SAME action; one holds a much better distribution over the
    rest. Regret and top-1 agreement are identical for both, by construction --
    neither metric can support a claim about evaluation quality. Establishing
    that needs value error against an independent reference, or gameplay.
    """

    values = {1: 80.0, 2: 55.0, 3: 20.0}
    sharp = _regret(values, 2)
    blunt = _regret(values, 2)
    assert sharp == blunt == 25.0
