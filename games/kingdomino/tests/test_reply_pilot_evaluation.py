from games.kingdomino.reply_pilot_evaluation import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, RANK1_ABS_FRAGILITY_TOLERANCE,
    _bootstrap_positions, _cluster_by_position, _excess_statistic,
    _paired_excess_statistic, _paired_median_statistic, arm_metrics,
)
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


def _two_position_fixture():
    references = {
        0: {1: 0.7, 2: 0.4},
        1: {3: 0.6, 4: 0.2},
    }
    ladder = {}
    for sims in (3200, 10000):
        for seed in ROOT_SEEDS:
            ladder[(0, sims, seed)] = {1: 0.72, 2: 0.52}
            ladder[(1, sims, seed)] = {3: 0.62, 4: 0.32}
    return references, ladder


def test_cluster_by_position_labels_ranks_and_computes_fragility():
    references, ladder = _two_position_fixture()
    clusters = _cluster_by_position(ladder, references, 3200)

    assert set(clusters) == {0, 1}
    # Two picks per position across five root seeds.
    assert len(clusters[0]) == 2 * len(ROOT_SEEDS)
    rank1 = [frag for is_rank1, frag in clusters[0] if is_rank1]
    secondary = [frag for is_rank1, frag in clusters[0] if not is_rank1]
    assert len(rank1) == len(ROOT_SEEDS)
    assert all(abs(value - 0.02) < 1e-12 for value in rank1)
    assert all(abs(value - 0.12) < 1e-12 for value in secondary)


def test_cluster_by_position_skips_missing_cells():
    references, ladder = _two_position_fixture()
    for seed in ROOT_SEEDS:
        ladder[(0, 3200, seed)] = {1: 0.72, 2: None}
    clusters = _cluster_by_position(ladder, references, 3200)
    assert len(clusters[0]) == len(ROOT_SEEDS)
    assert all(is_rank1 for is_rank1, _ in clusters[0])


def test_excess_statistic_matches_arm_metrics_point_estimate():
    references, ladder = _two_position_fixture()
    clusters = _cluster_by_position(ladder, references, 3200)
    statistic = _excess_statistic(clusters)
    assert abs(statistic([0, 1]) - 0.10) < 1e-12


def test_paired_excess_statistic_is_the_difference_of_arm_excesses():
    references, control_ladder = _two_position_fixture()
    # Treatment halves the secondary gap; rank-1 is untouched.
    treatment_ladder = dict(control_ladder)
    for sims in (3200, 10000):
        for seed in ROOT_SEEDS:
            treatment_ladder[(0, sims, seed)] = {1: 0.72, 2: 0.46}
            treatment_ladder[(1, sims, seed)] = {3: 0.62, 4: 0.26}

    control_clusters = _cluster_by_position(control_ladder, references, 3200)
    treatment_clusters = _cluster_by_position(treatment_ladder, references, 3200)
    paired = _paired_excess_statistic(control_clusters, treatment_clusters)

    # control excess 0.10, treatment excess 0.04 -> paired delta -0.06.
    assert abs(_excess_statistic(control_clusters)([0, 1]) - 0.10) < 1e-12
    assert abs(_excess_statistic(treatment_clusters)([0, 1]) - 0.04) < 1e-12
    assert abs(paired([0, 1]) - (-0.06)) < 1e-12


def test_paired_interval_is_tighter_than_the_marginals():
    """Why the paired difference is the primary endpoint.

    Both arms share positions, so a common per-position offset cancels in the
    paired statistic but inflates each marginal interval.
    """
    control = {index: [(True, 0.0), (False, 0.10 + index)] for index in range(20)}
    treatment = {index: [(True, 0.0), (False, 0.05 + index)] for index in range(20)}
    positions = sorted(control)

    marginal = _bootstrap_positions(
        _excess_statistic(control), positions, resamples=1000, seed=BOOTSTRAP_SEED)
    paired = _bootstrap_positions(
        _paired_excess_statistic(control, treatment), positions,
        resamples=1000, seed=BOOTSTRAP_SEED)

    marginal_width = marginal["ci_high"] - marginal["ci_low"]
    paired_width = paired["ci_high"] - paired["ci_low"]
    assert paired_width < marginal_width
    assert abs(paired["point"] - (-0.05)) < 1e-9


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    paired = {index: [0.01 * index] * 4 for index in range(20)}
    statistic = _paired_median_statistic(paired)
    positions = sorted(paired)

    first = _bootstrap_positions(
        statistic, positions, resamples=500, seed=BOOTSTRAP_SEED)
    second = _bootstrap_positions(
        statistic, positions, resamples=500, seed=BOOTSTRAP_SEED)

    assert first == second, "same seed must reproduce the interval exactly"
    assert first["resamples"] == 500
    assert first["ci_low"] <= first["point"] <= first["ci_high"]


def test_bootstrap_degenerates_safely_on_one_position():
    paired = {0: [0.5, 0.5]}
    result = _bootstrap_positions(
        _paired_median_statistic(paired), [0],
        resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    assert result["point"] == 0.5
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["resamples"] == 0


def test_bootstrap_draws_whole_positions_with_replacement():
    """The resampling unit must be the position, not the individual cell.

    Picks and seeds from one root are correlated; drawing them independently
    would understate the interval.  A spy statistic records exactly what each
    resample is composed of.
    """
    seen = []

    def spy(positions):
        seen.append(list(positions))
        return 0.0

    _bootstrap_positions(spy, list(range(10)), resamples=50, seed=BOOTSTRAP_SEED)

    resamples = seen[1:]  # seen[0] is the point estimate on the real set
    assert len(resamples) == 50
    assert all(len(sample) == 10 for sample in resamples), "fixed-size draws"
    assert all(set(sample) <= set(range(10)) for sample in resamples)
    # With replacement: at least one draw must repeat a position.
    assert any(len(set(sample)) < 10 for sample in resamples)


def test_bootstrap_interval_widens_with_between_position_spread():
    """Positions that genuinely disagree must produce a non-degenerate CI."""
    identical = {index: [0.05] * 4 for index in range(20)}
    spread = {index: [0.05 * (index - 10)] * 4 for index in range(20)}

    tight = _bootstrap_positions(
        _paired_median_statistic(identical), sorted(identical),
        resamples=1000, seed=BOOTSTRAP_SEED)
    wide = _bootstrap_positions(
        _paired_median_statistic(spread), sorted(spread),
        resamples=1000, seed=BOOTSTRAP_SEED)

    assert tight["ci_high"] - tight["ci_low"] == 0.0
    assert wide["ci_high"] - wide["ci_low"] > 0.05


def test_rank1_abs_fragility_guard_fails_on_material_degradation():
    """The guard endpoint: |rank-1 fragility| increasing beyond tolerance."""
    degraded = {index: [0.20, 0.22, 0.19] for index in range(20)}
    result = _bootstrap_positions(
        _paired_median_statistic(degraded), sorted(degraded),
        resamples=1000, seed=BOOTSTRAP_SEED)
    assert result["ci_high"] > RANK1_ABS_FRAGILITY_TOLERANCE
    assert not (result["ci_high"] <= RANK1_ABS_FRAGILITY_TOLERANCE)


def test_rank1_abs_fragility_guard_passes_when_rank1_is_unharmed():
    """Rank-1 Q may move a lot as long as |fragility| does not grow.

    This is the case a search-allocation arm must be allowed to pass: the
    old signed +/-0.02 guard would reject it, the |fragility| guard accepts it.
    """
    # Signed shift is large and negative; accuracy is unchanged or improved.
    unharmed = {index: [-0.001, 0.0, -0.002] for index in range(20)}
    result = _bootstrap_positions(
        _paired_median_statistic(unharmed), sorted(unharmed),
        resamples=1000, seed=BOOTSTRAP_SEED)
    assert result["ci_high"] <= RANK1_ABS_FRAGILITY_TOLERANCE
