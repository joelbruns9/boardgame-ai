"""Self-play personas (stop bias) and the per-iteration learning-rate schedule.

Run: python -m pytest games/cantstop/tests/test_personas.py -q
"""

import pytest

from games.cantstop import train
from games.cantstop.engine import RuleSet
from games.cantstop.lookahead import leaf_reach
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool_equiv import MOCKS, compare_results
from games.cantstop.self_play import AGGRESSIVE, CONSERVATIVE, PLAIN, Search
from games.cantstop.solver import TurnSolver
from games.cantstop.tests.test_lookahead import check_turns, turn_starts

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")


# ---- what a stop bias does ----

@pytest.mark.parametrize("rules", [RuleSet.make(2), RuleSet.make(3, blocking=True)],
                         ids=str)
def test_bias_moves_bust_risk_the_right_way(rules):
    """A conservative player busts less often in a turn, an aggressive one
    more often, than best play from the same position."""
    ev = MOCKS["hashed"]
    for state in turn_starts(rules, 2):
        bust = {}
        for name, bias in (("cons", 0.03), ("best", 0.0), ("aggr", -0.03)):
            ps = TurnSolver(state, ev, stop_bias=bias)
            bust[name], _ = leaf_reach(ps, state)
        assert bust["cons"] <= bust["best"] <= bust["aggr"], bust
        assert bust["cons"] < bust["aggr"], bust


def test_extreme_biases_always_or_never_stop():
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(2), 1)[0]
    always = TurnSolver(state, ev, stop_bias=1.0)
    never = TurnSolver(state, ev, stop_bias=-1.0)
    for key in always.keys:
        if always.stoppable[key] and not always.winning[key]:
            assert always.decision_values[key] is always.stop_values[key]
            assert never.decision_values[key] is not never.stop_values[key]


def test_zero_bias_is_best_play_exactly():
    ev = MOCKS["mover_wins"]          # every stop decision on a rounding error
    state = turn_starts(RuleSet.make(2), 1)[0]
    a, b = TurnSolver(state, ev), TurnSolver(state, ev, stop_bias=0.0)
    for key in a.keys:
        assert a.decision_values[key].tolist() == b.decision_values[key].tolist()


# ---- Rust matches Python ----

@pytest.mark.parametrize("search", [CONSERVATIVE, AGGRESSIVE,
                                    Search(stop_bias=0.03, exact_root=True),
                                    Search(stop_bias=-0.03, lookahead_k=2)],
                         ids=["cons", "aggr", "cons-exact", "aggr-k2"])
def test_persona_turns_match_python(search):
    for rules in (RuleSet.make(2), RuleSet.make(4, blocking=True)):
        state = turn_starts(rules, 1)[0]
        check_turns(state, 5, "hashed", [search] * rules.num_players)


def test_mixed_persona_seats_match_python():
    state = turn_starts(RuleSet.make(2), 1)[0]
    check_turns(state, 9, "hashed", [CONSERVATIVE, AGGRESSIVE], turns=2)


# ---- the persona schedule ----

def test_persona_seating_counts_and_seats():
    schedule = [RuleSet.make(3)] * 20
    seats = train.persona_seating(schedule, PortableRng(1), PLAIN,
                                  conservative=0.2, aggressive=0.1)
    kinds = []
    for i, row in enumerate(seats):
        odd = [(j, s) for j, s in enumerate(row) if s != PLAIN]
        assert len(odd) <= 1
        if odd:
            j, s = odd[0]
            assert j == i % 3                       # persona seat rotates
            kinds.append("c" if s.stop_bias > 0 else "a")
    assert kinds.count("c") == 4 and kinds.count("a") == 2


def test_no_personas_draws_no_random_numbers():
    rng = PortableRng(5)
    state = rng.state
    train.persona_seating([RuleSet.make(2)] * 10, rng, PLAIN)
    assert rng.state == state


def test_persona_generation_matches_between_backends():
    rules = [RuleSet.make(2)]
    kw = dict(conservative=0.5, aggressive=0.25)
    py = train.generate(rules, 4, MOCKS["hashed"], PortableRng(2),
                        backend="python", **kw)
    rs = train.generate(rules, 4, MOCKS["hashed"], PortableRng(2),
                        backend="rust", **kw)
    for i, (a, b) in enumerate(zip(py, rs)):
        compare_results(a, b, f"game {i}")


# ---- learning-rate schedule ----

def test_lr_schedule_steps_down_by_iteration():
    sched = train.parse_lr_schedule(["1:1e-3", "60:3e-4", "150:1e-4"])
    assert train.lr_at(1, 5e-5, sched) == 1e-3
    assert train.lr_at(59, 5e-5, sched) == 1e-3
    assert train.lr_at(60, 5e-5, sched) == 3e-4
    assert train.lr_at(400, 5e-5, sched) == 1e-4
    late = train.parse_lr_schedule(["10:1e-4"])
    assert train.lr_at(3, 7e-4, late) == 7e-4     # before the first entry
    assert train.lr_at(3, 7e-4, None) == 7e-4
    with pytest.raises(ValueError, match="ITER:LR"):
        train.parse_lr_schedule(["60"])


# ---- personas play biased, but record best-play targets ----

@pytest.mark.parametrize("exact", [False, True], ids=["sampled", "exact"])
def test_persona_turn_records_the_best_play_value(exact):
    """Same position, same opening roll: the persona's recorded value is
    exactly best play's, even though the persona may then play the turn
    differently. Otherwise ~15% of rows would teach the net that the side
    to move plays worse than it does."""
    from dataclasses import replace
    from games.cantstop.tests.test_lookahead import rust_turns
    ev = MOCKS["hashed"]
    plain = Search(exact_root=exact)
    played_differently = 0
    for rules in (RuleSet.make(2), RuleSet.make(3, blocking=True)):
        for state in turn_starts(rules, 3):
            n = rules.num_players
            ref, _ = rust_turns(state, 3, ev, [plain] * n, 1)
            for bias in (0.03, -0.03, 0.3, -0.3):
                persona = replace(plain, stop_bias=bias)
                got, _ = rust_turns(state, 3, ev, [persona] * n, 1)
                assert got[8] == ref[8]                 # turn_values
                played_differently += got[7] != ref[7]  # turn_lengths
    assert played_differently > 0        # the bias really changed play
