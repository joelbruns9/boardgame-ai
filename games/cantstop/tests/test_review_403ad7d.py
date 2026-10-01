"""Regression tests for the findings of the 403ad7d training/search review
(reviews/cantstop-training-search-403ad7d.md in the main checkout).

Run: python -m pytest games/cantstop/tests/test_review_403ad7d.py -q
"""

import numpy as np
import pytest

from games.cantstop import train
from games.cantstop.engine import RuleSet
from games.cantstop.lookahead import refine
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool import game_seeds, run_pool
from games.cantstop.rust_pool_equiv import MOCKS
from games.cantstop.self_play import (PLAIN, Search, check_targets,
                                      stack_training)
from games.cantstop.solver import TurnSolver
from games.cantstop.tests.test_lookahead import check_turns, turn_starts
from games.cantstop.value_accuracy import cluster_bootstrap, uniform_indices

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

K3_EXACT = Search(exact_root=True, lookahead_k=3)


# ---- P1: offset lookahead must keep values on the probability simplex ----

def _valid(v, n):
    v = np.asarray(v, dtype=np.float64)
    return v.min() >= 0.0 and abs(v.sum() - 1.0) < 1e-9 and len(v) == n


def test_offset_leaves_stay_probabilities_on_the_reviewers_position():
    """The reviewer's counterexample: before the fix the k=3 root value was
    [0.3098, -0.0055, 0.6957] and 2,579 of 6,029 stop vectors had a
    negative entry."""
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(3), 8)[2]
    ps = TurnSolver(state, ev)
    refine(ps, state, ev, 3, True)
    assert _valid(ps.value(state), 3)
    for key, v in ps.stop_values.items():
        assert _valid(v, 3), key
    assert _valid(ps.bust_value, 3)


def test_offset_lookahead_games_produce_valid_targets_in_rust():
    """End to end with three players on the pool, lambda = 0 (targets are
    exactly the recorded turn values)."""
    starts = turn_starts(RuleSet.make(3), 8)[:4]
    res = run_pool([s.rules for s in starts],
                   game_seeds(PortableRng(0), len(starts)),
                   [MOCKS["hashed"]], starts=starts, search=K3_EXACT)
    for r in res:
        for v in r.turn_values:
            assert v is None or _valid(v, 3)
    stack_training(res, 0.0)          # would raise on an invalid target


def test_rust_still_matches_python_with_the_projection():
    state = turn_starts(RuleSet.make(3), 8)[2]
    check_turns(state, 0, "hashed", [K3_EXACT] * 3)


@pytest.mark.parametrize("bad, match", [
    ([[0.6, 0.5, 0, 0]], "sums to"),
    ([[1.1, -0.1, 0, 0]], "negative"),
    ([[np.nan, 1.0, 0, 0]], "non-finite"),
])
def test_training_boundary_refuses_invalid_targets(bad, match):
    with pytest.raises(ValueError, match=match):
        check_targets(np.asarray(bad))


# ---- finding 2: arena must play the self-play search ----

def test_arena_drops_exact_root_only_without_lookahead():
    assert train.arena_search_for(Search(exact_root=True)) == PLAIN
    k = Search(exact_root=True, lookahead_k=4)
    assert train.arena_search_for(k) == k


def test_exact_root_changes_lookahead_play():
    """Why: with k > 0, rooting before vs after the roll picks different
    leaves to refine, so the turn itself can differ (the reviewer's
    example: 11 decisions vs 2)."""
    from games.cantstop.tests.test_lookahead import rust_turns
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(2), 1)[0]
    after, _ = rust_turns(state, 3, ev, [Search(lookahead_k=3)] * 2, 1)
    before, _ = rust_turns(state, 3, ev, [K3_EXACT] * 2, 1)
    assert after[7] != before[7]      # turn lengths differ


# ---- finding 3: the accuracy check samples uniformly ----

def test_board_sampling_is_uniform_not_lowest_indices():
    """Before the fix: max index 3,897 of 10,000, mean percentile 19%."""
    idx = uniform_indices(10_000, 60, PortableRng(0))
    assert len(set(idx)) == 60
    assert 0.35 < np.mean(idx) / 10_000 < 0.65
    assert max(idx) > 8_000


def test_cluster_bootstrap_widens_error_bars_for_correlated_boards():
    gen = np.random.default_rng(1)
    games = np.repeat(np.arange(10), 6)                   # 6 boards per game
    per_game = gen.normal(0, 1, 10)[games]                 # shared within game
    values = per_game + gen.normal(0, 0.1, 60)
    naive = values.std(ddof=1) / np.sqrt(60)
    assert cluster_bootstrap(values, list(games)) > 1.5 * naive
