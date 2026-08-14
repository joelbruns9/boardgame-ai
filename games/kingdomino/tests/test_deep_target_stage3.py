import pytest

from games.kingdomino.deep_target_stage3 import (
    cross_seed_uplifts,
    game_clustered_interval,
    summarize_stage3,
)


def _row(position_id, table_id, regrets, s2_picks=(1, 1), deep_picks=(1, 1), best=(1, 1)):
    return {
        "position_id": position_id,
        "table_id": table_id,
        "split": "development",
        "cohort_reasons": ["synthetic"],
        "matched_pick_regret_by_repeat": list(regrets),
        "matched_best_pick_by_repeat": list(best),
        "ordinary_30000_searches": [
            {"selected_pick_domino_id": pick} for pick in deep_picks
        ],
        "stage2_4800_searches": [
            {"selected_pick_domino_id": pick} for pick in s2_picks
        ],
        "matched_10000_pick_groups_by_repeat": [
            [
                {"pick_domino_id": 1, "root_value_actor": 0.10},
                {"pick_domino_id": 2, "root_value_actor": 0.12},
            ],
            [
                {"pick_domino_id": 1, "root_value_actor": 0.11},
                {"pick_domino_id": 2, "root_value_actor": 0.09},
            ],
        ],
        "paired_comparisons": [
            {
                "pick_changed_4800_to_30000": before != after,
            }
            for before, after in zip(s2_picks, deep_picks)
        ],
    }


def test_stage3_summary_uses_paired_regret_and_source_game_clusters():
    rows = [
        _row("a", "g1", (0.0, 0.02)),
        _row("b", "g1", (0.04, 0.06), deep_picks=(1, 2), best=(1, 2)),
        _row("c", "g2", (0.0, 0.0)),
    ]
    summary = summarize_stage3(rows)

    assert summary["positions"] == 3
    assert summary["paired_repeats"] == 6
    assert summary["ordinary_pick_changes_4800_to_30000"] == 1
    assert summary["matched_pick_regret_gt_003"] == 2
    assert summary["game_clustered_regret"]["source_games"] == 2
    assert summary["game_clustered_regret"]["game_weighted_mean"] == pytest.approx(0.015)


def test_clustered_interval_is_deterministic():
    rows = [_row("a", "g1", (0.01, 0.03)), _row("b", "g2", (0.0, 0.0))]
    first = game_clustered_interval(rows, samples=100, seed=7)
    second = game_clustered_interval(rows, samples=100, seed=7)
    assert first == second


def test_cross_seed_uplift_scores_choice_on_other_repeat():
    row = _row("a", "g1", (0.0, 0.0), s2_picks=(1, 1), best=(2, 1))
    assert cross_seed_uplifts(row, chooser="matched") == pytest.approx([-0.02, 0.0])
