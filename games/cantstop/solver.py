"""Exact turn solver: best play for the rest of the active player's turn.

Within a turn nobody else moves, so the only uncertainty is the dice, whose
probabilities are exact. The solver enumerates every runner configuration
the turn can reach, asks an evaluator how good each end-of-turn board is,
and works backwards:

    decision(k) = max( stop(k)  [if stopping is legal],  roll(k) )
    roll(k)     = sum over roll outcomes of P(outcome) *
                  ( bust value                      if no legal move
                    max over moves of decision(child) otherwise )

Values are per-seat win-probability vectors. "max" compares the active
seat's entry. A stop that wins the game is exact (one-hot) and is never
expanded further. Ties prefer stopping, then the first move in sorted order.

Every roll that does not bust moves a runner up, and columns have tops, so
the reachable configurations form a finite DAG ordered by total runner
height; values are filled from the highest configurations down.

One solve covers the whole remaining turn: every decision the turn can
reach is answered from the same table.

Evaluator contract: ``evaluate(states) -> ndarray (len(states), players)``.
Each state is a board at the start of some player's turn (AWAIT_ROLL, no
runners); rows are win probabilities in absolute seat order.
"""

from collections import defaultdict
from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from .engine import (COLUMN_HEIGHTS, COLUMNS, GameState, Phase, bust,
                     dice_pairings, legal_moves, stop, stop_blocked)


def _multiset_weight(dice):
    counts = defaultdict(int)
    for d in dice:
        counts[d] += 1
    w = factorial(4)
    for c in counts.values():
        w //= factorial(c)
    return w


def _roll_classes():
    """Group the 1296 ordered rolls by their set of sum pairings. Legal moves
    depend on the dice only through that set. Returns [(dice, probability)]
    with one representative roll per class."""
    classes = {}
    for dice in combinations_with_replacement(range(1, 7), 4):
        key = tuple(sorted(dice_pairings(dice)))
        rep, w = classes.get(key, (dice, 0))
        classes[key] = (rep, w + _multiset_weight(dice))
    return [(rep, w / 1296) for rep, w in classes.values()]


ROLL_CLASSES = _roll_classes()


def runners_key(runners):
    return tuple(sorted(runners.items()))


# ---- roll menus, cached by column signature ----
#
# Which moves a roll allows depends on each column only through its status:
#   0 closed, no runner (claimed, or saved marker at the top)
#   5 runner at the top: closed, but still occupies a runner slot
#   1 runner, one space left      2 runner, two or more left
#   3 free,   one space left      4 free,   two or more left
# (free slots = 3 - runner columns, codes 1/2/5). So the menu of a roll --
# the set of moves it offers -- is a function of that signature, shared by
# every configuration with the same signature, in every game.

_MENU_CACHE = {}


def column_signature(claimed_by, saved, runners):
    sig = []
    for c in COLUMNS:
        pos = runners.get(c)
        in_runners = pos is not None
        if not in_runners:
            pos = saved[c]
        room = COLUMN_HEIGHTS[c] - pos
        if claimed_by[c] is not None:
            sig.append(0)
        elif room <= 0:
            sig.append(5 if in_runners else 0)
        elif in_runners:
            sig.append(1 if room == 1 else 2)
        else:
            sig.append(3 if room == 1 else 4)
    return tuple(sig)


def _build_menu(state):
    """(bust prob, distinct moves, [(move indices, prob)]) for state's
    runners, using the engine's move rules as the single source of truth."""
    grouped = defaultdict(float)
    bust_p = 0.0
    index = {}
    for dice, p in ROLL_CLASSES:
        moves = legal_moves(state, dice)
        if not moves:
            bust_p += p
            continue
        idx = tuple(sorted(index.setdefault(m, len(index)) for m in moves))
        grouped[idx] += p
    moves = [None] * len(index)
    for m, i in index.items():
        moves[i] = m
    return bust_p, moves, list(grouped.items())


class TurnSolver:
    """Solve the active player's turn from ``state``'s current runners.

    ``state`` may be in any in-turn phase. The saved board (progress, claims,
    active player) is fixed for the whole turn.
    """

    def __init__(self, state, evaluate):
        if state.game_over:
            raise ValueError("game is over")
        self.base = state.clone()
        self.base.dice = None
        self.active = state.active_player
        self.num_players = state.rules.num_players
        self.root = runners_key(state.runners)
        self.evaluator_calls = 0
        self._enumerate()
        self._evaluate_leaves(evaluate)
        self._backup()

    # ---- pass 1: reachable configurations and their roll menus ----

    def _enumerate(self):
        scratch = self.base.clone()
        saved = self.base.progress[self.active]
        claimed_by = self.base.claimed_by
        cols_to_win = self.base.rules.columns_to_win
        already = len(self.base.claimed_columns(self.active))

        self.menus = {}          # key -> [(child keys, prob)]
        self.bust_prob = {}      # key -> prob
        self.stoppable = {}      # key -> bool
        self.winning = {}        # key -> bool (stoppable and stopping wins)
        stack = [self.root]
        seen = {self.root}
        while stack:
            key = stack.pop()
            runners = dict(key)
            scratch.runners = runners
            can = bool(runners) and not stop_blocked(scratch)
            wins = can and already + sum(
                pos >= COLUMN_HEIGHTS[c] for c, pos in key) >= cols_to_win
            self.stoppable[key] = can
            self.winning[key] = wins
            if wins:
                self.menus[key] = []
                self.bust_prob[key] = 0.0
                continue
            sig = column_signature(claimed_by, saved, runners)
            menu = _MENU_CACHE.get(sig)
            if menu is None:
                menu = _MENU_CACHE[sig] = _build_menu(scratch)
            bust_p, moves, move_menus = menu
            children = []
            for move in moves:
                child = dict(runners)
                for col in move:
                    child[col] = child.get(col, saved[col]) + 1
                ck = runners_key(child)
                children.append(ck)
                if ck not in seen:
                    seen.add(ck)
                    stack.append(ck)
            self.menus[key] = [(tuple(children[i] for i in idx), p)
                               for idx, p in move_menus]
            self.bust_prob[key] = bust_p
        self.keys = sorted(seen, key=lambda k: -sum(pos for _, pos in k))

    # ---- pass 2: evaluator on end-of-turn boards ----

    def _board_after(self, key):
        s = self.base.clone()
        s.runners = dict(key)
        s.phase = Phase.AWAIT_DECISION
        if key:
            stop(s)
        else:
            bust(s)
        return s

    def _evaluate_leaves(self, evaluate):
        leaf_keys = [k for k in self.keys
                     if self.stoppable[k] and not self.winning[k]]
        boards = [self._board_after(()) ] + [self._board_after(k) for k in leaf_keys]
        values = np.asarray(evaluate(boards), dtype=np.float64)
        if values.shape != (len(boards), self.num_players):
            raise ValueError(f"evaluator returned shape {values.shape}")
        self.evaluator_calls = len(boards)
        self.bust_value = values[0]
        self.stop_values = dict(zip(leaf_keys, values[1:]))
        win = np.zeros(self.num_players)
        win[self.active] = 1.0
        for k in self.keys:
            if self.winning[k]:
                self.stop_values[k] = win

    # ---- pass 3: backward induction ----

    def _backup(self):
        a = self.active
        self.decision_values = {}
        self.roll_values = {}
        for key in self.keys:  # highest configurations first
            if self.winning[key]:
                self.decision_values[key] = self.stop_values[key]
                continue
            roll_v = self.bust_prob[key] * self.bust_value
            for children, p in self.menus[key]:
                best = max(children,
                           key=lambda c: self.decision_values[c][a])
                roll_v = roll_v + p * self.decision_values[best]
            self.roll_values[key] = roll_v
            if (self.stoppable[key]
                    and self.stop_values[key][a] >= roll_v[a]):
                self.decision_values[key] = self.stop_values[key]
            else:
                self.decision_values[key] = roll_v

    # ---- queries ----

    @property
    def num_positions(self):
        return len(self.keys)

    def _key(self, state):
        key = runners_key(state.runners)
        if key not in self.decision_values:
            raise KeyError(f"runners {key} not reachable from this solve")
        return key

    def value(self, state):
        """Per-seat win probabilities under best play from ``state``."""
        key = self._key(state)
        if state.phase == Phase.AWAIT_MOVE:
            return self.decision_values[
                self.best_child(key, state.dice)]
        if state.phase == Phase.AWAIT_ROLL:
            return self.roll_values[key]
        return self.decision_values[key]

    def best_child(self, key, dice):
        scratch = self.base.clone()
        scratch.runners = dict(key)
        saved = self.base.progress[self.active]
        best, best_v = None, -1.0
        for move in legal_moves(scratch, dice):
            child = dict(key)
            for col in move:
                child[col] = child.get(col, saved[col]) + 1
            ck = runners_key(child)
            v = self.decision_values[ck][self.active]
            if v > best_v:
                best, best_v = ck, v
        return best

    def choose_move(self, state):
        """Best legal move for ``state`` (phase AWAIT_MOVE)."""
        key = self._key(state)
        target = self.best_child(key, state.dice)
        saved = self.base.progress[self.active]
        for move in legal_moves(state, state.dice):
            child = dict(key)
            for col in move:
                child[col] = child.get(col, saved[col]) + 1
            if runners_key(child) == target:
                return move
        raise AssertionError("unreachable")

    def should_stop(self, state):
        """True if stopping is legal and at least as good as rolling."""
        key = self._key(state)
        return (state.phase == Phase.AWAIT_DECISION
                and self.stoppable[key]
                and self.decision_values[key] is self.stop_values[key])


# ---- placeholder evaluator ----

class ProgressHeuristic:
    """Stand-in for the value net: softmax over a per-seat progress score.

    score = claimed columns / columns to win
            + share of the remaining columns' height already climbed.
    Only for developing and validating the solver; not a strength target.
    """

    def __init__(self, temperature=0.15):
        self.temperature = temperature

    def __call__(self, states):
        out = np.empty((len(states), states[0].rules.num_players))
        for i, s in enumerate(states):
            need = s.rules.columns_to_win
            scores = []
            for p in range(s.rules.num_players):
                claimed = sum(1 for c in COLUMNS if s.claimed_by[c] == p)
                climb = sorted((s.progress[p][c] / COLUMN_HEIGHTS[c]
                                for c in COLUMNS if s.claimed_by[c] is None),
                               reverse=True)
                scores.append((claimed + sum(climb[:max(need - claimed, 0)]))
                              / need)
            z = np.asarray(scores) / self.temperature
            z = np.exp(z - z.max())
            out[i] = z / z.sum()
        return out


def solve_turn(state, evaluate):
    return TurnSolver(state, evaluate)
