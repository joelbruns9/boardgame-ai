"""M2 gate: the Rust turn solver's whole table equals the Python solver's.

Bit-equality on float64, not closeness: the Rust side reproduces Python's
summation order, so any difference is a bug. Evaluators are deterministic
mocks (``rust_solver.EVALUATORS``); ``flat`` makes everything tie, which is
the only way to see the tie-breaks.

The full-size gate is ``python -m games.cantstop.rust_solver``. Skipped, not
failed, when the extension is not built.

Run: python -m pytest games/cantstop/tests/test_rust_solver_equiv.py -q
"""

import numpy as np
import pytest

from games.cantstop.engine import ALL_RULESETS, GameState, Phase, RuleSet
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_solver import (
    EVALUATORS, RustTurnSolver, compare_solvers, sample_positions,
)
from games.cantstop.snapshot import snapshot
from games.cantstop.solver import ProgressHeuristic, TurnSolver

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

BASE2 = RuleSet.make(2)
BLOCK2 = RuleSet.make(2, blocking=True)
BLOCK4 = RuleSet.make(4, blocking=True)
EXT3 = RuleSet.make(3, extended=True)

# Keep the fast suite fast: Python solves of the largest turn-start tables
# take ~10 s each and add nothing the smaller ones do not already cover.
MAX_POSITIONS = 6000


def state_with(rules=BASE2, active=0, runners=None, progress=None,
               claimed=None, phase=Phase.AWAIT_DECISION, dice=None):
    s = GameState(rules)
    s.active_player = active
    for p, cols in (progress or {}).items():
        s.progress[p].update(cols)
    for col, p in (claimed or {}).items():
        s.claimed_by[col] = p
    s.runners = dict(runners or {})
    s.phase = phase
    s.dice = dice
    return s


def _small(state):
    return rust.TurnSolver(snapshot(state)).num_positions <= MAX_POSITIONS


@pytest.mark.parametrize("rules", ALL_RULESETS, ids=str)
def test_sampled_positions_match_under_every_evaluator(rules):
    positions = [s for s in sample_positions(rules, 7, 12) if _small(s)][:3]
    assert positions, "sampler produced nothing small enough"
    for i, state in enumerate(positions):
        for name in EVALUATORS:
            compare_solvers(state, name, f"{rules} #{i}",
                            rng=PortableRng(i), max_keys=8)


CONSTRUCTED = {
    # Root is blocked: stopping is illegal here and in part of the table.
    "blocked root": state_with(
        BLOCK2, runners={7: 4, 8: 2}, progress={1: {7: 4, 6: 3}}),
    "blocked by the third seat": state_with(
        BLOCK4, active=1, runners={5: 2, 9: 6}, progress={3: {9: 6, 9: 7}}),
    # One column from winning, one step from the top: winning leaves.
    "one step from winning": state_with(
        claimed={2: 0, 12: 0}, runners={3: 4}, progress={1: {4: 6}}),
    "extended, two columns to go": state_with(
        EXT3, claimed={2: 0, 12: 0}, runners={3: 4, 11: 4}),
    # Claimed columns and saved markers at the top shrink the menus.
    "crowded board": state_with(
        BASE2, active=1, claimed={7: 0, 6: 1}, runners={8: 9},
        progress={0: {5: 8, 9: 7}, 1: {5: 3}}),
    # Rooted at the rolled options, the self-play entry point.
    "just rolled": state_with(
        runners={7: 3}, phase=Phase.AWAIT_MOVE, dice=(1, 3, 4, 6)),
    "just rolled, capped double": state_with(
        runners={2: 2, 5: 1, 9: 1}, phase=Phase.AWAIT_MOVE,
        dice=(1, 1, 1, 1)),
    # Turn start late in a game.
    "late turn start": state_with(
        BLOCK2, claimed={7: 1, 8: 0}, progress={0: {6: 9, 5: 7},
                                                 1: {6: 10, 9: 8}},
        phase=Phase.AWAIT_ROLL),
}


@pytest.mark.parametrize("name", list(CONSTRUCTED))
@pytest.mark.parametrize("evaluator", list(EVALUATORS))
def test_constructed_positions_match(name, evaluator):
    compare_solvers(CONSTRUCTED[name].clone(), evaluator, name,
                    rng=PortableRng(1), max_keys=10)


def test_constructed_positions_reach_their_corners():
    c = CONSTRUCTED
    blocked = TurnSolver(c["blocked root"], ProgressHeuristic())
    assert not blocked.stoppable[tuple(sorted(c["blocked root"].runners.items()))]
    winning = TurnSolver(c["one step from winning"], ProgressHeuristic())
    assert any(winning.winning.values())


def test_mover_wins_puts_every_choice_on_a_rounding_error():
    """The gate only exercises the stop tie-break and the summation order
    if ``mover_wins`` really makes stop and roll agree to the last bits
    while still producing both answers. (``flat`` does not: winning leaves
    make rolling strictly better, so it never stops.)"""
    state = CONSTRUCTED["crowded board"]
    ps = TurnSolver(state, EVALUATORS["mover_wins"])
    a = ps.active
    keys = [k for k in ps.keys if ps.stoppable[k] and not ps.winning[k]]
    gaps = [abs(ps.stop_values[k][a] - ps.roll_values[k][a]) for k in keys]
    stops = sum(ps.decision_values[k] is ps.stop_values[k] for k in keys)
    assert max(gaps) < 1e-14
    assert 0.1 * len(keys) < stops < 0.9 * len(keys)


def test_drop_in_for_python_solver_in_a_game():
    """Play a whole game where each turn is solved by BOTH solvers and the
    Rust answer is the one played: every decision must agree."""
    rng = PortableRng(11)
    state = GameState(RuleSet.make(2))
    from games.cantstop.engine import (apply_move, can_stop, random_dice,
                                       roll, stop)
    evaluate = ProgressHeuristic()
    turns = 0
    while not state.game_over and turns < 60:
        if not roll(state, random_dice(rng)):
            turns += 1
            continue
        ps, rs = TurnSolver(state, evaluate), RustTurnSolver(state, evaluate)
        while True:
            move = rs.choose_move(state)
            assert move == ps.choose_move(state)
            apply_move(state, move)
            assert rs.should_stop(state) == ps.should_stop(state)
            if can_stop(state) and rs.should_stop(state):
                stop(state)
                break
            if not roll(state, random_dice(rng)):
                break
        turns += 1
    assert turns > 5


def test_query_before_values_is_refused():
    s = rust.TurnSolver(snapshot(CONSTRUCTED["blocked root"]))
    with pytest.raises(RuntimeError, match="set_leaf_values"):
        s.should_stop([(7, 4), (8, 2)])


def test_wrong_number_of_values_is_refused():
    s = rust.TurnSolver(snapshot(CONSTRUCTED["blocked root"]))
    with pytest.raises(ValueError, match="expected"):
        s.set_leaf_values([[0.5, 0.5]])
    with pytest.raises(ValueError, match="expected"):
        s.set_leaf_values([[1.0]] * s.num_leaves)


def test_unreachable_configuration_is_a_key_error():
    rs = RustTurnSolver(CONSTRUCTED["blocked root"], ProgressHeuristic())
    other = CONSTRUCTED["blocked root"].clone()
    other.runners = {2: 1}
    with pytest.raises(KeyError):
        rs.should_stop(other)
    with pytest.raises(KeyError):
        TurnSolver(CONSTRUCTED["blocked root"],
                   ProgressHeuristic()).should_stop(other)


def test_game_over_is_refused():
    s = state_with(claimed={2: 0, 3: 0, 4: 0}, phase=Phase.GAME_OVER)
    s.winner = 0
    with pytest.raises(ValueError, match="game is over"):
        RustTurnSolver(s, ProgressHeuristic())
    with pytest.raises(ValueError, match="game is over"):
        rust.TurnSolver(snapshot(s))


def test_evaluator_shape_is_checked():
    with pytest.raises(ValueError, match="shape"):
        RustTurnSolver(CONSTRUCTED["blocked root"],
                       lambda boards: np.zeros((len(boards), 3)))
