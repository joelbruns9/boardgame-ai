"""Correctness tests for the exact turn solver.

Run: python -m pytest games/cantstop/tests/test_solver.py -q
"""

import random
from collections import Counter
from itertools import product

import numpy as np
import pytest

from games.cantstop.engine import (
    ALL_RULESETS, COLUMN_HEIGHTS, GameState, Phase, RuleSet, apply_move,
    bust, can_stop, dice_pairings, legal_moves, random_dice, roll, stop,
    stop_blocked,
)
from games.cantstop.solver import (
    ROLL_CLASSES, ProgressHeuristic, TurnSolver, _build_menu,
    column_signature, runners_key,
)

HEUR = ProgressHeuristic()


# ---- reference: plain expectimax over engine transitions ----

ORDERED_ROLL_WEIGHTS = Counter(tuple(sorted(d))
                               for d in product(range(1, 7), repeat=4))


class BruteForce:
    """Expectimax over all 1296 ordered rolls (grouped only by sorted dice),
    driving the engine's own roll/apply_move/stop/bust. Shares nothing with
    the solver except the evaluator."""

    def __init__(self, state, evaluate):
        self.base = state
        self.evaluate = evaluate
        self.a = state.active_player
        self.memo = {}

    def leaf(self, s):
        if s.game_over:
            v = np.zeros(s.rules.num_players)
            v[s.winner] = 1.0
            return v
        return np.asarray(self.evaluate([s]))[0]

    def decision(self, s):
        key = ("d", runners_key(s.runners))
        if key in self.memo:
            return self.memo[key]
        best = self.roll(s)
        if can_stop(s):
            t = s.clone()
            stop(t)
            sv = self.leaf(t)
            if sv[self.a] >= best[self.a]:
                best = sv
        self.memo[key] = best
        return best

    def roll(self, s):
        key = ("r", runners_key(s.runners))
        if key in self.memo:
            return self.memo[key]
        total = 0.0
        for dice, w in ORDERED_ROLL_WEIGHTS.items():
            t = s.clone()
            t.phase = Phase.AWAIT_DECISION if s.runners else Phase.AWAIT_ROLL
            moves = roll(t, dice)
            if not moves:
                total = total + w / 1296 * self.leaf(t)
                continue
            best = None
            for m in moves:
                u = t.clone()
                apply_move(u, m)
                v = self.decision(u)
                if best is None or v[self.a] > best[self.a]:
                    best = v
            total = total + w / 1296 * best
        self.memo[key] = total
        return total


# ---- position sampling ----

def random_positions(rules, rng, n, max_positions=None, games=200):
    """In-turn positions from random play: turn starts (AWAIT_ROLL, no
    runners) and post-move decisions (AWAIT_DECISION)."""
    out = []
    for _ in range(games):
        s = GameState(rules)
        while not s.game_over and len(out) < n:
            if s.phase == Phase.AWAIT_ROLL:
                if rng.random() < 0.3:
                    out.append(s.clone())
                roll(s, random_dice(rng))
            elif s.phase == Phase.AWAIT_MOVE:
                apply_move(s, rng.choice(legal_moves(s, s.dice)))
                if rng.random() < 0.3:
                    out.append(s.clone())
            elif can_stop(s) and rng.random() < 0.3:
                stop(s)
            else:
                roll(s, random_dice(rng))
        if len(out) >= n:
            break
    if max_positions is not None:
        out = [p for p in out if _size(p) <= max_positions]
    return out


def _size(state):
    """Cheap upper bound on reachable configurations."""
    total = 1
    free = [c for c in COLUMN_HEIGHTS if state.claimed_by[c] is None]
    room = sorted((COLUMN_HEIGHTS[c] - state.position(c) + 1 for c in free),
                  reverse=True)
    for r in room[:3]:
        total *= r
    return total * max(1, len(free)) ** max(0, 3 - len(state.runners))


def late_positions(rules, seed, want=6, max_positions=400):
    rng = random.Random(seed)
    got = []
    for _ in range(50):
        got += random_positions(rules, rng, 400, max_positions)
        if len(got) >= want:
            break
    rng.shuffle(got)
    return got[:want]


# ---- tests ----

def test_roll_classes_cover_all_rolls():
    assert abs(sum(p for _, p in ROLL_CLASSES) - 1.0) < 1e-12
    by_class = Counter()
    for dice in product(range(1, 7), repeat=4):
        by_class[tuple(sorted(dice_pairings(dice)))] += 1
    for dice, p in ROLL_CLASSES:
        assert by_class[tuple(sorted(dice_pairings(dice)))] == round(p * 1296)


def _direct_menu(s):
    """Menu of s's runners as {sorted move tuple: prob}, plus bust prob,
    straight from the engine with no caching."""
    menu = Counter()
    bust_p = 0.0
    for dice, p in ROLL_CLASSES:
        ms = legal_moves(s, dice)
        if ms:
            menu[tuple(ms)] += p
        else:
            bust_p += p
    return round(bust_p, 12), {k: round(v, 12) for k, v in menu.items()}


def test_signature_determines_menu():
    """Every configuration sharing a column signature must offer exactly the
    same moves with the same probabilities -- otherwise the menu cache hands
    one configuration another's menu."""
    rng = random.Random(1)
    by_sig = {}
    samples = 0
    for rules in ALL_RULESETS:
        states = random_positions(rules, rng, 150)
        states += synthetic_precap_positions(rules, rng, want=15, lo=1, hi=10**9)
        for s in states:
            me = s.active_player
            # Also every runner configuration one move away, which reaches
            # runners at the top of their columns.
            configs = [s]
            for dice, _ in ROLL_CLASSES[::7]:
                for m in legal_moves(s, dice):
                    t = s.clone()
                    for col in m:
                        t.runners[col] = t.position(col) + 1
                    configs.append(t)
            for t in configs:
                sig = column_signature(t.claimed_by, t.progress[me], t.runners)
                menu = _direct_menu(t)
                samples += 1
                if sig in by_sig:
                    assert by_sig[sig] == menu, (sig, t)
                else:
                    by_sig[sig] = menu
    assert samples > 5000 and len(by_sig) > 500
    assert any(5 in sig for sig in by_sig), "runner-at-top never sampled"
    bust_p, moves, menus = _build_menu(s)
    assert _direct_menu(s) == (round(bust_p, 12), {
        tuple(sorted(moves[i] for i in idx)): round(p, 12) for idx, p in menus})


@pytest.mark.parametrize("rules", ALL_RULESETS, ids=repr)
def test_solver_matches_brute_force(rules):
    positions = late_positions(rules, seed=rules.num_players * 7
                               + rules.columns_to_win + rules.blocking)
    assert positions, "no small positions sampled"
    for s in positions:
        sol = TurnSolver(s, HEUR)
        root = s.clone()
        bf = BruteForce(root, HEUR)
        if s.runners:
            want = bf.decision(root)
            got = sol.decision_values[runners_key(s.runners)]
        else:
            want = bf.roll(root)
            got = sol.roll_values[()]
        np.testing.assert_allclose(got, want, atol=1e-9)


def _rollout(sol, state, rng):
    """Play the solver's policy to the end of the turn; return the leaf
    value vector (terminal one-hot or evaluator)."""
    s = state.clone()
    me = s.active_player
    while s.active_player == me and not s.game_over:
        if s.phase == Phase.AWAIT_MOVE:
            apply_move(s, sol.choose_move(s))
        elif s.phase == Phase.AWAIT_DECISION and sol.should_stop(s):
            stop(s)
        else:
            roll(s, random_dice(rng))
    if s.game_over:
        v = np.zeros(s.rules.num_players)
        v[s.winner] = 1.0
        return v
    return np.asarray(HEUR([s]))[0]


@pytest.mark.parametrize("rules", [RuleSet.make(2), RuleSet.make(3, True, True),
                                   RuleSet.make(4, blocking=True)], ids=repr)
def test_policy_rollouts_match_solver_value(rules):
    rng = random.Random(3)
    positions = late_positions(rules, seed=11, want=3, max_positions=3000)
    for s in positions:
        sol = TurnSolver(s, HEUR)
        a = s.active_player
        vals = np.array([_rollout(sol, s, rng)[a] for _ in range(3000)])
        se = vals.std() / np.sqrt(len(vals)) + 1e-9
        target = sol.value(s)[a]
        assert abs(vals.mean() - target) < 4 * se + 1e-6, \
            (vals.mean(), target, se)


def test_winning_stop_is_taken_and_exact():
    rules = RuleSet.make(2)
    s = GameState(rules)
    s.claimed_by[2] = 0
    s.claimed_by[12] = 0
    s.runners = {3: 5}
    s.phase = Phase.AWAIT_DECISION
    sol = TurnSolver(s, HEUR)
    assert sol.should_stop(s)
    np.testing.assert_array_equal(sol.value(s), [1.0, 0.0])


def test_blocked_position_never_stops():
    rules = RuleSet.make(3, extended=True, blocking=True)
    rng = random.Random(5)
    checked = 0
    for s in random_positions(rules, rng, 3000):
        if s.phase != Phase.AWAIT_DECISION or not stop_blocked(s):
            continue
        if _size(s) > 3000:
            continue
        sol = TurnSolver(s, HEUR)
        assert not sol.should_stop(s)
        checked += 1
        if checked >= 5:
            break
    assert checked > 0


def test_decisions_are_legal_and_consistent():
    rules = RuleSet.make(3, extended=True, blocking=True)
    rng = random.Random(9)
    for s in late_positions(rules, seed=2, want=4, max_positions=3000):
        sol = TurnSolver(s, HEUR)
        t = s.clone()
        me = t.active_player
        while t.active_player == me and not t.game_over:
            if t.phase == Phase.AWAIT_MOVE:
                m = sol.choose_move(t)
                assert m in legal_moves(t, t.dice)
                apply_move(t, m)
            elif t.phase == Phase.AWAIT_DECISION and sol.should_stop(t):
                assert can_stop(t)
                stop(t)
            else:
                roll(t, random_dice(rng))


def test_stop_preferred_when_evaluator_punishes_rolling():
    """Evaluator: every non-bust board is great for the mover, bust is
    terrible. The solver must stop immediately wherever it can."""
    def evaluate(states):
        out = []
        for st in states:
            mover = (st.active_player - 1) % st.rules.num_players
            saved = sum(st.progress[mover].values())
            v = np.full(st.rules.num_players, 0.0)
            v[mover] = 0.9 if saved > base_saved else 0.1
            v[(mover + 1) % st.rules.num_players] = 1 - v[mover]
            out.append(v)
        return np.array(out)

    s = GameState(RuleSet.make(2))
    s.runners = {7: 1}
    s.phase = Phase.AWAIT_DECISION
    base_saved = 0
    sol = TurnSolver(s, evaluate)
    assert sol.should_stop(s)


def synthetic_precap_positions(rules, rng, want=5, lo=20, hi=600):
    """Late-game boards with few open columns and 0-2 runners, so the
    column-opening part of the turn is small enough for brute force."""
    out = []
    n = rules.num_players
    for _ in range(5000):
        s = GameState(rules)
        s.active_player = rng.randrange(n)
        counts = [0] * n
        for c in COLUMN_HEIGHTS:
            if rng.random() < 0.6:
                p = rng.randrange(n)
                if counts[p] < rules.columns_to_win - 1:
                    s.claimed_by[c] = p
                    counts[p] += 1
                    continue
            used = set()
            for p in range(n):
                if rng.random() < 0.7:
                    h = rng.randrange(max(1, COLUMN_HEIGHTS[c] - 3),
                                      COLUMN_HEIGHTS[c])
                    if rules.blocking and h in used:
                        continue
                    used.add(h)
                    s.progress[p][c] = h
        me = s.active_player
        open_cols = [c for c in COLUMN_HEIGHTS if s.claimed_by[c] is None
                     and s.progress[me][c] < COLUMN_HEIGHTS[c]]
        k = rng.randrange(3)
        if k > len(open_cols):
            continue
        for c in rng.sample(open_cols, k):
            s.runners[c] = rng.randrange(s.progress[me][c] + 1,
                                         COLUMN_HEIGHTS[c] + 1)
        s.phase = Phase.AWAIT_DECISION if s.runners else Phase.AWAIT_ROLL
        sol = TurnSolver(s, HEUR)
        if lo <= sol.num_positions <= hi:
            out.append(s)
            if len(out) >= want:
                break
    return out


@pytest.mark.parametrize("rules", ALL_RULESETS, ids=repr)
def test_solver_matches_brute_force_precap(rules):
    rng = random.Random(rules.num_players * 31 + rules.columns_to_win * 3
                        + rules.blocking)
    positions = synthetic_precap_positions(rules, rng)
    assert len(positions) >= 3
    assert any(len(s.runners) < 2 for s in positions)
    for s in positions:
        sol = TurnSolver(s, HEUR)
        bf = BruteForce(s.clone(), HEUR)
        if s.runners:
            want, got = bf.decision(s.clone()), sol.value(s)
        else:
            want, got = bf.roll(s.clone()), sol.value(s)
        np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("rules", [RuleSet.make(2), RuleSet.make(3, True, True),
                                   RuleSet.make(4, blocking=True)], ids=repr)
def test_rooting_at_the_roll_matches_full_turn_solve(rules):
    """A solve started after the dice are rolled must give the same value
    and the same decisions as the full turn-start solve, on a smaller table."""
    rng = random.Random(4)
    checked = 0
    for s in synthetic_precap_positions(rules, rng, want=6, lo=200, hi=5000):
        full = TurnSolver(s, HEUR)
        for dice, _ in ROLL_CLASSES[::5]:
            t = s.clone()
            if not roll(t, dice):
                continue
            rooted = TurnSolver(t, HEUR)
            assert rooted.num_positions <= full.num_positions
            np.testing.assert_allclose(rooted.value(t), full.value(t), atol=1e-12)
            assert rooted.choose_move(t) == full.choose_move(t)
            apply_move(t, rooted.choose_move(t))
            assert rooted.should_stop(t) == full.should_stop(t)
            np.testing.assert_allclose(rooted.value(t), full.value(t), atol=1e-12)
            checked += 1
    assert checked > 10
