"""The Workstream 9 reference-case harness.

The harness produces the numbers a Workstream 9 gate diffs, so what has to be
tested is that it reads the tree honestly, not that any particular number comes
out. Tests run at smoke budgets against a tiny random net: the shape, the
frames, and the position guard are what is asserted; the measurements are not.

The one substantive assertion about the position itself is that decision row 17
of table 908370787 still resolves to the reviewed decision. If a re-scrape, a
re-indexed log, or an extractor change moves it, every later comparison would be
measuring a different decision under the same name.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from . import w9_reference_case as w9

LOG = w9.REPO_ROOT / "runs/seven_wonders_duel/bga_game_log/table_908370787.jsonl"

pytestmark = pytest.mark.skipif(
    not LOG.exists(), reason="BGA game log for the reference table is not present"
)


@pytest.fixture(scope="module")
def position():
    return w9.load_position(LOG, w9.DEFAULT_DECISION_ROW, 0)


@pytest.fixture(scope="module")
def evaluator():
    from .inference import Evaluator
    from .train import build_model

    return Evaluator(build_model("transformer", 32, 1), "cpu")


def _args(**overrides):
    argv = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return w9.parse_args(argv)


# -- position ---------------------------------------------------------------


def test_reference_row_is_still_the_reviewed_decision(position):
    """Age II, the human's move 31: five legal actions, the played
    ``Discard for coins: Caravansery`` among them."""

    assert position.meta["age"] == 2
    assert position.meta["phase"] == "PLAY_AGE"
    assert tuple(entry["label"] for entry in position.legal) == w9.EXPECTED_LEGAL


def test_position_guard_refuses_a_different_decision():
    """The guard is the whole defence against silently measuring another
    position as if it were the reference case."""

    with pytest.raises(SystemExit, match="does not match the reference case"):
        w9.load_position(LOG, w9.DEFAULT_DECISION_ROW + 1, 0)


def test_position_guard_can_be_waived():
    other = w9.load_position(LOG, w9.DEFAULT_DECISION_ROW + 1, 0, strict=False)
    assert other.meta["decision_row"] == w9.DEFAULT_DECISION_ROW + 1


def test_position_is_reproducible_for_a_fixed_resample_seed():
    """The determinizer must be a pure function of its seed -- otherwise the
    artifact is not comparable across runs. See ``advisor_scrape._unseen``."""

    first = w9.load_position(LOG, w9.DEFAULT_DECISION_ROW, 7)
    second = w9.load_position(LOG, w9.DEFAULT_DECISION_ROW, 7)
    assert first.observation_sha256 == second.observation_sha256
    assert [c.card_name for c in first.game.tableau.cards.values()] == [
        c.card_name for c in second.game.tableau.cards.values()
    ]


def test_resolve_action_by_substring_and_rejects_ambiguity(position):
    assert (
        w9.resolve_action(position, "coins: Caravansery")["label"]
        == w9.DEFAULT_WALK_ACTION
    )
    assert (
        w9.resolve_action(position, w9.DEFAULT_WALK_ACTION)["label"]
        == w9.DEFAULT_WALK_ACTION
    )
    with pytest.raises(SystemExit, match="ambiguous"):
        w9.resolve_action(position, "Caravansery")
    with pytest.raises(SystemExit, match="no action"):
        w9.resolve_action(position, "Build: Pyramids")


def test_win_pct_matches_the_panel_frame():
    """``extension_7wd/content.js`` renders ``(q + 1) / 2 * 100``."""

    assert w9.win_pct(0.0) == 50.0
    assert w9.win_pct(1.0) == 100.0
    assert w9.win_pct(-1.0) == 0.0


# -- tree reading -----------------------------------------------------------


def test_walked_edge_carries_a_chance_partition(position, evaluator):
    """The finding in one assertion: taking the coverer fires a CARD_REVEAL, so
    the edge holds many chance children and one reply must be rediscovered in
    each of them."""

    args = _args(smoke=True)
    mcts = w9.build_mcts(evaluator, args)
    root = mcts.make_root(position.game)
    walked = w9.resolve_action(position, w9.DEFAULT_WALK_ACTION)
    edge = next(e for e in root.edges if e.action_index == walked["index"])
    for _ in range(60):
        mcts.descend(root)

    breakdown = w9.world_breakdown(edge, w9.DEFAULT_TRACKED)
    assert breakdown["rollup"]["world_count"] > 1
    assert len(breakdown["worlds"]) == breakdown["rollup"]["world_count"]
    # Every world names the card it revealed, and force-expansion makes the
    # support exhaustive, so the probabilities are a proper distribution.
    assert all(world["revealed"] for world in breakdown["worlds"])
    assert sum(world["probability"] for world in breakdown["worlds"]) == pytest.approx(1.0)


def test_world_breakdown_sums_wonder_burial_variants(position, evaluator):
    """Constructing a Wonder is one action per burial target, so the tracked
    Wonder is a GROUP of edges. Reporting only one variant would understate the
    partition; reporting only the group total would hide it."""

    args = _args(smoke=True)
    mcts = w9.build_mcts(evaluator, args)
    root = mcts.make_root(position.game)
    walked = w9.resolve_action(position, w9.DEFAULT_WALK_ACTION)
    edge = next(e for e in root.edges if e.action_index == walked["index"])
    for _ in range(120):
        mcts.descend(root)

    breakdown = w9.world_breakdown(edge, w9.DEFAULT_TRACKED)
    for world in breakdown["worlds"]:
        tracked = world["tracked"]
        assert tracked["examined"] == (tracked["group_visits"] > 0)
        if tracked["best_variant"] is not None:
            assert tracked["group_visits"] >= tracked["best_variant"]["visits"]
            assert w9.DEFAULT_TRACKED in tracked["best_variant"]["label"]
        else:
            assert tracked["group_visits"] == 0


def test_root_ranking_is_visit_ordered_and_actor_framed(position, evaluator):
    args = _args(smoke=True)
    mcts = w9.build_mcts(evaluator, args)
    root = mcts.make_root(position.game)
    for _ in range(80):
        mcts.descend(root)

    ranking = w9.root_ranking(root, position.sign, position.game)
    assert [row["rank"] for row in ranking] == list(range(1, len(ranking) + 1))
    assert [row["visits"] for row in ranking] == sorted(
        (row["visits"] for row in ranking), reverse=True
    )
    assert {row["index"] for row in ranking} == {e["index"] for e in position.legal}
    for row in ranking:
        assert -1.0 <= row["q"] <= 1.0
        assert row["win_pct"] == pytest.approx(w9.win_pct(row["q"]), abs=0.01)


# -- end to end -------------------------------------------------------------


def test_smoke_run_writes_a_complete_artifact(tmp_path, monkeypatch, evaluator):
    """The artifact is the deliverable, so its shape is what the gate depends
    on. Budgets here are meaningless by construction (``--smoke``)."""

    out = tmp_path / "smoke.json"
    monkeypatch.setattr(
        w9, "load_evaluator_for",
        lambda args: (evaluator, w9.REPO_ROOT / "extension_7wd/candidate_0085.pt"),
    )
    assert w9.main(["--smoke", "--quiet", "--out", str(out)]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["smoke"] is True
    assert report["harness_version"] == 1
    assert report["checkpoint"]["sha256"]
    assert report["position"]["age"] == 2
    assert report["search"]["force_expand_root_chance"] is True

    # Every default stage produced its section.
    assert [rung["sims"] for rung in report["ladder"]] == [40, 80]
    assert report["tree_walk"]["rollup"]["world_count"] > 1
    assert report["single_world_probes"]["worlds_probed"] == 2

    summary = report["summary"]
    assert set(summary["sims_to_promote_refutation"]) == {
        "in_any_world", "in_half_of_worlds", "in_all_worlds", "max_sims_measured"
    }
    assert summary["sims_to_promote_refutation"]["max_sims_measured"] == 80
    assert summary["partition_penalty"]["buckets_per_idea"] >= 1


def test_stage_selection_omits_unrequested_sections(tmp_path, monkeypatch, evaluator):
    out = tmp_path / "walk_only.json"
    monkeypatch.setattr(
        w9, "load_evaluator_for",
        lambda args: (evaluator, w9.REPO_ROOT / "extension_7wd/candidate_0085.pt"),
    )
    assert w9.main(
        ["--smoke", "--quiet", "--stages", "walk", "--out", str(out)]
    ) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert "tree_walk" in report
    assert "ladder" not in report
    assert "single_world_probes" not in report
    assert "reference_values" not in report


def test_summary_out_is_the_small_committable_half(tmp_path, monkeypatch, evaluator):
    """The full report holds every per-world reply and belongs under the
    gitignored ``runs/``; the summary is what a later search change is diffed
    against, so it must stand alone -- provenance included."""

    full, summary_path = tmp_path / "full.json", tmp_path / "summary.json"
    monkeypatch.setattr(
        w9, "load_evaluator_for",
        lambda args: (evaluator, w9.REPO_ROOT / "extension_7wd/candidate_0085.pt"),
    )
    assert w9.main(
        ["--smoke", "--quiet", "--out", str(full), "--summary-out", str(summary_path)]
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["checkpoint"]["sha256"]
    assert summary["search"]["seed"] == 0
    assert summary["position"]["decision_row"] == w9.DEFAULT_DECISION_ROW
    assert "sims_to_promote_refutation" in summary["summary"]
    assert summary_path.stat().st_size < full.stat().st_size

    # The bulky per-node sections stay in the full report. Checked as dict KEYS,
    # not as substrings: "worlds" occurs inside rollup names like
    # "in_half_of_worlds", which are exactly what the summary is for.
    def keys(node):
        if isinstance(node, dict):
            return set(node) | {k for v in node.values() for k in keys(v)}
        if isinstance(node, list):
            return {k for v in node for k in keys(v)}
        return set()

    bulky = {"worlds", "probes", "top_replies", "ranking", "legal"}
    assert not keys(summary) & bulky
    assert keys(json.loads(full.read_text(encoding="utf-8"))) & bulky


def test_unknown_stage_is_refused():
    with pytest.raises(SystemExit, match="unknown stage"):
        w9.parse_args(["--stages", "walk,teleport"])
