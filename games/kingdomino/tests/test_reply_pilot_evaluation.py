from games.kingdomino.reply_pilot_evaluation import arm_metrics
from games.kingdomino.secondary_pick_seed_test import ROOT_SEEDS


def test_arm_metrics_separates_common_rank1_offset_from_secondary_excess():
    references = {
        0: {1: 0.7, 2: 0.4},
        1: {3: 0.6, 4: 0.2},
    }
    ladder = {}
    for sims in (3200, 10000):
        for seed in ROOT_SEEDS:
            ladder[(0, sims, seed)] = {1: 0.72, 2: 0.52}
            ladder[(1, sims, seed)] = {3: 0.62, 4: 0.32}

    metrics = arm_metrics(ladder, references)
    row = metrics["by_sims"]["3200"]
    assert abs(row["rank1"]["fragility"]["median"] - 0.02) < 1e-12
    assert abs(row["secondary"]["fragility"]["median"] - 0.12) < 1e-12
    assert abs(row["secondary_minus_rank1_median_fragility"] - 0.10) < 1e-12
    assert row["rank1"]["missing_cells"] == 0
    assert row["secondary"]["missing_cells"] == 0
