"""Gates for the Step 4 chance-cap quality tool.

The tool decides whether capping ships, so its arithmetic is worth pinning: a
metric that silently reported zero error would look like a green light.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from games.seven_wonders_duel.chance_cap_quality import (
    compare_search,
    kl_divergence,
    measure_edge,
    percentile,
    pure_double_reveal_edges,
    summarise_level_a,
)
from games.seven_wonders_duel.codec import decode_action, legal_action_indices
from games.seven_wonders_duel.engine import Action, ActionUse, apply_action
from games.seven_wonders_duel.game import Phase, new_game
from games.seven_wonders_duel.inference import Evaluation
from games.seven_wonders_duel.search import enumerate_chains


class _SpreadEvaluator:
    """Deterministic, position-dependent values so a capped support genuinely
    differs from the full one (a constant oracle would hide every error)."""

    def evaluate(self, encodings, legal_lists):
        out = []
        for encoding, legal in zip(encodings, legal_lists):
            key = hash(tuple(sorted(token.entity_id for token in encoding.tokens)))
            win = (key % 1000) / 1000.0
            out.append(
                Evaluation(
                    policy=np.full(len(legal), 1.0 / len(legal), dtype=np.float32),
                    wdl=np.asarray([win, 0.0, 1.0 - win], dtype=np.float32),
                    joint7=np.full(7, 1.0 / 7.0, dtype=np.float32),
                    margin=0.0,
                    military=0.0,
                    science=np.zeros(2, dtype=np.float32),
                )
            )
        return out


class _ConstantEvaluator(_SpreadEvaluator):
    def evaluate(self, encodings, legal_lists):
        out = super().evaluate(encodings, legal_lists)
        for evaluation in out:
            evaluation.wdl[:] = np.asarray([0.6, 0.1, 0.3], dtype=np.float32)
        return out


def _double_uncover_state():
    game = new_game(30, first_player=0)
    rng = random.Random(30 * 13 + 5)
    while game.phase is not Phase.PLAY_AGE:
        apply_action(game, decode_action(game, rng.choice(legal_action_indices(game))))
    apply_action(game, Action((4, 1), ActionUse.DISCARD_FOR_COINS))
    apply_action(game, Action((4, 5), ActionUse.DISCARD_FOR_COINS))
    return game


def test_edge_measurement_reports_a_real_error_against_the_exact_expectation():
    game = _double_uncover_state()
    edges = pure_double_reveal_edges(game)
    assert edges
    index, specs = edges[0]
    row = measure_edge(game, index, specs, _SpreadEvaluator(), [1, 2, 3], 17)

    assert row["outcomes"] == len(enumerate_chains(game, specs))
    assert row["same_back"] and row["pool"] == 11
    assert set(row["caps"]) == {1, 2, 3}
    for offsets, cap in row["caps"].items():
        assert cap["retained"] == row["pool"] * offsets
        assert cap["worst_gap"] >= 0.0  # a subset can only lose worst cases
        assert cap["terminal_retained"] <= row["terminal_children"]
    # More offsets is more information: error must not grow on average, and the
    # retained fraction must.
    retained = [row["caps"][x]["retained"] for x in (1, 2, 3)]
    assert retained == sorted(retained)
    assert any(abs(row["caps"][x]["q_error"]) > 0 for x in (1, 2, 3))


def test_a_constant_oracle_makes_the_capped_expectation_exact():
    """The estimator is a re-weighted mean, so with every child valued the same
    it must return the exact expectation -- any residue would be a weighting
    bug, not approximation error."""

    game = _double_uncover_state()
    index, specs = pure_double_reveal_edges(game)[0]
    row = measure_edge(game, index, specs, _ConstantEvaluator(), [1, 2], 3)
    for cap in row["caps"].values():
        assert cap["q_error"] == pytest.approx(0.0, abs=1e-12)
        assert cap["worst_gap"] == pytest.approx(0.0, abs=1e-12)


def test_level_a_summary_aggregates_by_pool_and_back():
    rows = [
        {
            "outcomes": 110,
            "same_back": True,
            "pool": 11,
            "terminal_children": 1,
            "caps": {2: {"retained": 22, "q_error": 0.02, "terminal_retained": 1, "worst_gap": 0.0}},
        },
        {
            "outcomes": 40,
            "same_back": False,
            "pool": None,
            "terminal_children": 2,
            "caps": {2: {"retained": 16, "q_error": -0.04, "terminal_retained": 0, "worst_gap": 0.5}},
        },
    ]
    summary = summarise_level_a(rows, [2])[2]
    assert summary["edges"] == 2
    assert summary["q_mae"] == pytest.approx(0.03)
    assert summary["q_max"] == pytest.approx(0.04)
    assert summary["terminal_children_dropped"] == 2
    assert summary["worst_case_covered"] == pytest.approx(0.5)
    assert summary["q_mae_by_back"] == {"same": pytest.approx(0.02), "mixed": pytest.approx(0.04)}


def test_decision_comparison_scores_agreement_and_the_cost_of_disagreeing():
    baseline = {
        "policy": [0.6, 0.3, 0.1],
        "visits": [5, 3, 0],
        "topk": [0, 1],
        "completed_q": [0.5, 0.2, -0.9],
    }
    identical = compare_search(baseline, baseline)
    assert identical["action_disagreement"] == 0.0
    assert identical["policy_kl"] == pytest.approx(0.0)
    assert identical["survivor_jaccard"] == 1.0
    assert identical["topk_identical"] == 1.0

    other = dict(baseline, policy=[0.3, 0.6, 0.1], visits=[5, 3, 1], topk=[1, 0])
    differing = compare_search(baseline, other)
    assert differing["action_disagreement"] == 1.0
    # Regret is read off the BASELINE's own completed Q: 0.5 - 0.2.
    assert differing["action_regret"] == pytest.approx(0.3)
    assert differing["policy_kl"] > 0.0
    assert differing["survivor_jaccard"] == pytest.approx(2 / 3)
    assert differing["topk_identical"] == 0.0


def test_kl_and_percentile_helpers():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))
    # A zero in the reference contributes nothing rather than diverging.
    assert kl_divergence([0.0, 1.0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert percentile([1, 2, 3, 4], 0.0) == 1
    assert percentile([1, 2, 3, 4], 1.0) == 4
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert math.isnan(percentile([], 0.5))
