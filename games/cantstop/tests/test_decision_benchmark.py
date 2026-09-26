"""The forced stop-vs-roll benchmark, the confirmation harness and the
experiment-identity record. Machinery only, on mocks and tiny sizes; the
benchmark itself is run by hand.

Run: python -m pytest games/cantstop/tests/test_decision_benchmark.py -q
"""

import numpy as np
import pytest

from games.cantstop import confirm
from games.cantstop.decision_benchmark import (
    Decision, _branches, collect, continue_game, grade, immediate_threat,
    rollout, stratify, summarize)
from games.cantstop.engine import GameState, Phase, RuleSet, can_stop, stop
from games.cantstop.experiment import identity
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool import run_pool
from games.cantstop.rust_pool_equiv import MOCKS
from games.cantstop.rust_solver import RustTurnSolver
from games.cantstop.snapshot import from_snapshot, snapshot
from games.cantstop.solver import TurnSolver

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

EV = MOCKS["hashed"]


@pytest.fixture(scope="module")
def decisions():
    return collect(EV, RuleSet.make(2), 3, seed=4)


# ---- collection ----

def test_decisions_are_real_stop_or_roll_choices(decisions):
    assert len(decisions) > 10
    for d in decisions:
        s = from_snapshot(d.snap)
        assert s.phase == Phase.AWAIT_DECISION and can_stop(s)
        after = s.clone()
        stop(after)
        assert not after.game_over                  # winning stops excluded
        assert d.chose_stop == (d.stop_value >= d.roll_value)
        assert 0.0 <= d.threat <= 1.0


def test_stop_roll_values_match_the_python_solver(decisions):
    """The binding's two sides of a decision equal Python's stop and roll
    values for the same configuration and table."""
    d = decisions[0]
    s = from_snapshot(d.snap)
    ps = TurnSolver(s, EV)          # both solved from the decision itself
    rs = RustTurnSolver(s, EV)
    stop_v, roll_v = rs.stop_roll(s)
    key = tuple(sorted(s.runners.items()))
    assert stop_v.tolist() == ps.stop_values[key].tolist()
    assert roll_v.tolist() == ps.roll_values[key].tolist()


def test_immediate_threat_is_the_exact_win_this_turn_probability(decisions):
    from games.cantstop.decision_benchmark import _zero_evaluator
    for d in decisions[:5]:
        after = from_snapshot(d.snap)
        stop(after)
        ps = TurnSolver(after, _zero_evaluator)
        assert immediate_threat(after) == pytest.approx(
            ps.value(after)[after.active_player], abs=1e-12)


# ---- the roll branch starts mid-turn on the pool ----

def test_pool_continues_a_mid_turn_state_exactly_as_python(decisions):
    """The roll branch starts the pool from AWAIT_DECISION: it must roll on
    and finish that turn, then the game, exactly as the Python reference."""
    for d in decisions[:3]:
        _, state = _branches(d)
        for seed in (1, 2):
            py = continue_game(state, EV, PortableRng(seed))
            rs = run_pool([state.rules], [seed], [EV], starts=[state])[0]
            assert rs.winner == py.winner


# ---- stratification and grading ----

def test_stratify_samples_within_strata_and_records_occurrence(decisions):
    sample, occ = stratify(decisions, 5, seed=1)
    assert occ["random"] == 1.0
    assert all(d.strata for d in sample)
    counts = {k: sum(k in d.strata for d in sample) for k in occ}
    assert counts["random"] == min(5, len(decisions))
    for k, share in occ.items():
        assert counts[k] <= 5
        assert counts[k] == min(5, round(share * len(decisions)))


def test_grade_selects_with_a_and_grades_with_b():
    d = Decision(snap=None, game=0, mover=0, stop_value=0.6, roll_value=0.5,
                 chose_stop=True, threat=0.0, contested=False)
    n = 200
    stop_w = np.r_[np.zeros(n // 2), np.zeros(n // 2)]       # stop always loses
    roll_w = np.r_[np.ones(n // 2), np.full(n // 2, 1.0)]    # roll always wins
    g = grade(d, stop_w, roll_w, delta=0.01)
    assert g["wrong"] and g["loss_b"] == pytest.approx(1.0)
    d.chose_stop = False
    g = grade(d, stop_w, roll_w, delta=0.01)
    assert not g["wrong"] and g["loss_b"] == 0.0


def test_rollout_pairs_branches_and_scores_the_mover(decisions):
    sample, _ = stratify(decisions, 2, seed=3)
    wins = rollout(sample[:2], EV, reps=4, seed=9)
    assert len(wins) == 2
    for s, r in wins:
        assert s.shape == r.shape == (4,)
        assert set(np.unique(np.r_[s, r])) <= {0.0, 1.0}
    rows = [grade(d, s, r, 0.01) for d, (s, r) in zip(sample[:2], wins)]
    summary = summarize(sample[:2], rows)
    tagged = {k for d in sample[:2] for k in d.strata}
    assert set(summary) == tagged
    assert all(v["n"] >= 1 for v in summary.values())


# ---- confirmation harness ----

def test_seat_cycles_and_fresh_seeds():
    assert confirm.seat_cycle_games(200, 3) == 201
    assert confirm.seat_cycle_games(10_000, 2) == 10_000
    assert confirm.seat_cycle_games(10_001, 4) == 10_004
    s1 = confirm.confirmation_seed("a.pt", "b.pt", 0)
    assert s1 != confirm.confirmation_seed("a.pt", "c.pt", 0)
    assert s1 != confirm.confirmation_seed("a.pt", "b.pt", 1)


@pytest.mark.parametrize("ci, want", [
    ((0.52, 0.55), "better"), ((0.45, 0.48), "worse"),
    ((0.492, 0.508), "equivalent"), ((0.48, 0.515), "inconclusive"),
])
def test_confirmation_verdicts(ci, want):
    assert confirm.decide(ci, 0.5, 0.01) == want


def test_identity_records_commit_and_checkpoint_hash(tmp_path):
    ck = tmp_path / "x.pt"
    ck.write_bytes(b"weights")
    import hashlib
    meta = identity(nets={"a": ck}, search={"k": 0})
    assert meta["commit"] and len(meta["commit"]) == 40
    assert meta["nets"]["a"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert meta["search"] == {"k": 0}
