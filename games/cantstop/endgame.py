"""Exact end-game values: who wins from a late position, with no net at all.

Late in a game few columns are open and little progress remains, so every
board the rest of the game can reach fits in memory and can be solved
exactly -- the method of Glenn & Aloi for solved Can't Stop variants,
restricted to the end game. It is the only judge of the net's late-game win
probabilities and decisions that does not lean on the net itself.

Structure. A "level" is a saved board (everyone's progress and claims) with
no runners; it has one turn-start board per seat to move. Within a turn the
existing Rust turn solver is exact given the values of the boards the turn
can end on:

  * a stop ends on a board with strictly more saved progress -- a higher
    level, solved first (recursion with memo);
  * a winning stop is terminal (the solver scores it as a certain win);
  * a bust ends on the SAME level with the next seat to move.

So a level's n boards depend on each other only through busts. They are
solved together by Gauss-Seidel: re-back-up each seat's turn table with the
current value of its bust board (``rebackup`` -- no re-enumeration) until no
value moves by more than ``tol``. It converges because every turn stops
with positive probability; the rate is the product of the bust chances
around the cycle.

``budget`` caps the number of levels: a position whose remaining game is too
big raises ``TooLarge`` rather than running for hours.
"""

import numpy as np

from .engine import COLUMNS, GameState, Phase, RuleSet
from .snapshot import from_snapshot, snapshot


class TooLarge(RuntimeError):
    """The remaining game has more levels than the budget allows."""


def level_key(snap):
    """A saved board, from a snapshot: rules, progress, claims."""
    rules, _active, progress, claimed = snap[:4]
    return (tuple(rules), tuple(tuple(row) for row in progress), tuple(claimed))


def level_state(key, active):
    rules, progress, claimed = key
    state = GameState(RuleSet(*rules))
    state.active_player = active
    state.progress = [dict(zip(COLUMNS, row)) for row in progress]
    state.claimed_by = {c: (None if v == -1 else v) for c, v in zip(COLUMNS, claimed)}
    state.phase = Phase.AWAIT_ROLL
    return state


class _Placeholder:
    """Evaluator used only to construct a turn table; values come later.
    ``evaluate_features`` puts the solver on its fast path, so no Python
    board objects are built for the leaves."""

    def __call__(self, boards):
        n = boards[0].rules.num_players
        return np.full((len(boards), n), 1.0 / n)

    def evaluate_features(self, features, reference):
        n = reference.rules.num_players
        return np.full((len(features), n), 1.0 / n)


class ExactEndgame:
    def __init__(self, budget=20_000, tol=1e-12, max_sweeps=10_000):
        self.budget, self.tol, self.max_sweeps = budget, tol, max_sweeps
        self.values = {}        # level key -> (n, n) array: row = seat to move
        self._pending = {}      # level key -> successor level keys (not yet solved)
        self.sweeps = 0

    # ---- solving ----

    def board_value(self, state):
        """Exact win probabilities (absolute seats) of a turn-start board."""
        if state.game_over:
            out = np.zeros(state.rules.num_players)
            out[state.winner] = 1.0
            return out
        if state.runners or state.phase != Phase.AWAIT_ROLL:
            raise ValueError("board_value takes a turn-start board")
        key = level_key(snapshot(state))
        self._solve(key)
        return self.values[key][state.active_player].copy()

    def _solve(self, root):
        """Solve ``root`` and every level above it, deepest first, without
        Python recursion (a chain can be ~60 levels deep, and wide)."""
        stack = [root]
        while stack:
            key = stack[-1]
            if key in self.values:
                stack.pop()
                continue
            missing = [k for k in self._successors(key) if k not in self.values]
            if missing:
                stack.extend(missing)
                continue
            stack.pop()
            self._solve_level(key)

    def _seat_tables(self, key):
        """Each seat's turn table at this level, with its stop leaves as
        (level key, seat to move). Leaf 0 is the bust board: this level,
        next seat to move."""
        from .rust_solver import RustTurnSolver
        out = []
        for active in range(key[0][0]):
            solver = RustTurnSolver(level_state(key, active), _Placeholder())
            out.append((solver, [(level_key(s), s[1]) for s in solver._s.leaf_snapshots()[1:]]))
        return out

    def _successors(self, key):
        """Levels a stop can reach from ``key``. Only these small sets are
        kept while waiting; turn tables are rebuilt when the level is solved
        (cheap), so memory holds no tables for the levels on the stack."""
        succ = self._pending.get(key)
        if succ is None:
            if len(self.values) + len(self._pending) >= self.budget:
                raise TooLarge(f"more than {self.budget} levels")
            succ = {k for _, stops in self._seat_tables(key) for k, _ in stops} - {key}
            self._pending[key] = succ
        return succ

    def _solve_level(self, key):
        n = key[0][0]
        seats = self._seat_tables(key)
        fixed = [np.array([self.values[k][a] for k, a in stops]).reshape(-1, n)
                 for _, stops in seats]
        x = np.full((n, n), 1.0 / n)               # current values, row = seat to move
        for sweep in range(self.max_sweeps):
            moved = 0.0
            for active, (solver, _stops) in enumerate(seats):
                leaves = np.vstack([x[(active + 1) % n][None, :], fixed[active]])
                solver._s.rebackup(leaves.tolist())
                new = np.asarray(solver.value(level_state(key, active)))
                moved = max(moved, float(np.abs(new - x[active]).max()))
                x[active] = new
            self.sweeps += 1
            if moved < self.tol:
                break
        else:
            raise RuntimeError(f"level did not converge in {self.max_sweeps} sweeps")
        self.values[key] = x
        self._pending.pop(key, None)

    # ---- queries ----

    def turn_solver(self, state):
        """A turn table for ``state``'s turn with EXACT end-of-turn values:
        ``value`` / ``stop_roll`` / ``choose_move`` answer as the exact game."""
        start = state.clone()
        start.runners, start.dice, start.phase = {}, None, Phase.AWAIT_ROLL
        key = level_key(snapshot(start))
        self._solve(key)
        n = start.rules.num_players
        from .rust_solver import RustTurnSolver
        solver = RustTurnSolver(start, _Placeholder())
        leaves = solver._s.leaf_snapshots()
        values = [self.values[key][(start.active_player + 1) % n]]
        for s in leaves[1:]:
            k = level_key(s)
            self._solve(k)
            values.append(self.values[k][s[1]])
        solver._s.rebackup(np.asarray(values).tolist())
        return solver

    @property
    def levels(self):
        return len(self.values)
