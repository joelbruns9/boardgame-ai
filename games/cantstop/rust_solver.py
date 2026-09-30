"""The Rust turn solver behind ``solver.TurnSolver``'s interface, and the M2
equivalence gate that holds it to the Python solver.

``RustTurnSolver(state, evaluate)`` is a drop-in for ``TurnSolver``: same
constructor, same ``value`` / ``choose_move`` / ``should_stop`` /
``num_positions`` / ``evaluator_calls``. Underneath, the Rust side enumerates
and hands back the end-of-turn boards; they are evaluated here, in Python,
and the values go back for the backward induction. At M2 the boards cross
as snapshots and are rebuilt as Python ``GameState`` objects so any existing
evaluator works unchanged -- that crossing is the known cost M3 removes by
encoding in Rust.

**The gate** (``compare_solvers``) solves the same position with both and
compares the *whole* table, not the chosen move: the reachable
configurations, which are stoppable and which win, every roll menu with its
probabilities, every stop / roll / decision value, and the stop-or-roll
choice -- all with ``==`` on float64, because the Rust side reproduces
Python's summation order. It then compares the query API on a sample of
configurations under every dice multiset.

Evaluators are deterministic **mocks**, never the net: that isolates
enumeration and backup from torch's floating-point noise.

    heuristic  ProgressHeuristic, the Phase 1 stand-in
    hashed     a pseudo-random value per board (from a CRC of its snapshot),
               so ties are rare and many different choices are exercised
    flat       every board worth the same, so sibling children tie and the
               first-child-in-key-order tie-break decides
    mover_wins every leaf -- the bust board too -- is a certain win for the
               seat that just moved. Stop and roll then differ only by float
               rounding (measured: |stop - roll| <= 1.3e-15 everywhere, with
               both answers common), so every stop-or-roll choice is decided
               by the last bit of the summation and the prefer-stopping
               tie-break. The most order-sensitive case there is.

Run the gate over sampled positions from all 10 rule sets:

    python -m games.cantstop.rust_solver --positions-per-ruleset 20
"""

import argparse
import time
import zlib

import numpy as np

from games.cantstop.engine import (
    ALL_RULESETS, GameState, Phase, apply_move, can_stop, legal_moves,
    random_dice, roll, stop,
)
from games.cantstop.encoder import FEATURE_SIZE
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_equiv import ALL_DICE, Divergence
from games.cantstop.snapshot import from_snapshot, snapshot
from games.cantstop.solver import ProgressHeuristic, TurnSolver, runners_key


def _rust():
    import cantstop_rust
    return cantstop_rust


class RustTurnSolver:
    """``solver.TurnSolver``'s interface on the Rust solver."""

    def __init__(self, state, evaluate):
        if state.game_over:
            raise ValueError("game is over")
        self.active = state.active_player
        self.num_players = state.rules.num_players
        self._s = _rust().TurnSolver(snapshot(state))
        if hasattr(evaluate, "evaluate_features"):
            # Fast path: Rust encodes the leaves; only one float32 buffer
            # crosses. ``reference`` carries the two things the output
            # rotation reads -- rule set and seat to move.
            features = leaf_features(self._s)
            reference = GameState(state.rules)
            reference.active_player = self._s.leaf_active_player
            values = evaluate.evaluate_features(features, reference)
        else:
            boards = [from_snapshot(s) for s in self._s.leaf_snapshots()]
            values = evaluate(boards)
        values = np.asarray(values, dtype=np.float64)
        rows = self._s.num_leaves
        if values.shape != (rows, self.num_players):
            raise ValueError(f"evaluator returned shape {values.shape}")
        self._s.set_leaf_values_bytes(values.astype("<f8").tobytes())
        self.evaluator_calls = rows
        self._bust_value = values[0].copy()

    @property
    def bust_value(self):
        """Absolute-seat value of losing this turn, from the original leaves."""
        return self._bust_value.copy()

    @property
    def num_positions(self):
        return self._s.num_positions

    def value(self, state):
        return np.asarray(self._s.value(
            sorted(state.runners.items()), int(state.phase),
            None if state.dice is None else list(state.dice)))

    def choose_move(self, state):
        return tuple(self._s.choose_move(sorted(state.runners.items()),
                                         list(state.dice)))

    def should_stop(self, state):
        # Python looks the configuration up (KeyError if unreachable) before
        # it checks the phase; keep that order.
        answer = self._s.should_stop(sorted(state.runners.items()))
        return state.phase == Phase.AWAIT_DECISION and answer

    def stop_roll(self, state):
        """(stop value, roll-on value) at ``state``'s runners, absolute
        seats; either is None where it does not exist."""
        stop, roll = self._s.stop_roll(sorted(state.runners.items()))
        as_arr = lambda v: None if v is None else np.asarray(v)
        return as_arr(stop), as_arr(roll)


def leaf_features(rust_solver):
    """A Rust solver's leaves as an (N, FEATURE_SIZE) float32 array.

    Wrapped in a ``bytearray`` so the array is writable: torch warns on (and
    has undefined behaviour for) tensors over read-only numpy memory."""
    raw = bytearray(rust_solver.leaf_features())
    return np.frombuffer(raw, dtype="<f4").reshape(
        rust_solver.num_leaves, FEATURE_SIZE)


# ---- mock evaluators ----

def hashed_evaluator(states):
    out = np.empty((len(states), states[0].rules.num_players))
    for i, s in enumerate(states):
        rng = PortableRng(zlib.crc32(repr(snapshot(s)).encode()))
        row = np.array([rng.next_float() + 1e-3 for _ in range(len(out[i]))])
        out[i] = row / row.sum()
    return out


def flat_evaluator(states):
    n = states[0].rules.num_players
    return np.full((len(states), n), 1.0 / n)


def mover_wins_evaluator(states):
    n = states[0].rules.num_players
    out = np.zeros((len(states), n))
    for i, s in enumerate(states):
        # Leaf boards are post-stop / post-bust: the mover is the seat before.
        out[i, (s.active_player - 1) % n] = 1.0
    return out


EVALUATORS = {
    "heuristic": ProgressHeuristic(),
    "hashed": hashed_evaluator,
    "flat": flat_evaluator,
    "mover_wins": mover_wins_evaluator,
}


# ---- the gate ----

def _k(key):
    """A Rust key (list of tuples) as Python's runners_key tuple."""
    return tuple(tuple(e) for e in key)


def python_table(ps):
    table = {}
    for key in ps.keys:
        stops = (ps.stoppable[key]
                 and ps.decision_values[key] is ps.stop_values.get(key))
        table[key] = (
            ps.stoppable[key],
            ps.winning[key],
            ps.bust_prob[key],
            [(list(children), p) for children, p in ps.menus[key]],
            (ps.stop_values[key].tolist() if key in ps.stop_values
             else None),
            (ps.roll_values[key].tolist() if key in ps.roll_values
             else None),
            ps.decision_values[key].tolist(),
            stops,
        )
    return table


def rust_table(rs):
    table = {}
    for (key, stoppable, winning, bust_p, menu, sv, rv, dv, stops) in \
            rs._s.table():
        table[_k(key)] = (
            stoppable, winning, bust_p,
            [([_k(c) for c in kids], p) for kids, p in menu],
            sv, rv, dv, stops,
        )
    return table


FIELDS = ("stoppable", "winning", "bust_prob", "menu", "stop_value",
          "roll_value", "decision_value", "stops")


def compare_tables(ps, rs, where):
    pt, rt = python_table(ps), rust_table(rs)
    if set(pt) != set(rt):
        missing = sorted(set(pt) - set(rt))[:5]
        extra = sorted(set(rt) - set(pt))[:5]
        raise Divergence(f"reachable configurations differ at {where}: "
                         f"{len(pt)} python vs {len(rt)} rust; "
                         f"python-only {missing}, rust-only {extra}")
    for key in pt:
        for name, a, b in zip(FIELDS, pt[key], rt[key]):
            if a != b:
                raise Divergence(f"{name} differs for runners {key} at "
                                 f"{where}: python={a!r} rust={b!r}")
    if ps.evaluator_calls != rs.evaluator_calls:
        raise Divergence(f"leaf count differs at {where}: "
                         f"{ps.evaluator_calls} vs {rs.evaluator_calls}")


def compare_queries(ps, rs, state, where, *, max_keys=25, rng=None):
    """The query API on the root and on a sample of reachable
    configurations, under every dice multiset."""
    def same(what, a, b):
        if isinstance(a, np.ndarray):
            a, b = a.tolist(), np.asarray(b).tolist()
        if a != b:
            raise Divergence(f"{what} differs at {where}: "
                             f"python={a!r} rust={b!r}")

    same("root value", ps.value(state), rs.value(state))
    if state.phase == Phase.AWAIT_MOVE:
        same("root move", ps.choose_move(state), rs.choose_move(state))
    if state.phase == Phase.AWAIT_DECISION:
        same("root stop", ps.should_stop(state), rs.should_stop(state))

    keys = list(ps.keys)
    if rng is not None and len(keys) > max_keys:
        keys = [keys[rng.randrange(len(keys))] for _ in range(max_keys)]
    for key in keys[:max_keys]:
        if ps.winning[key]:
            continue
        s = state.clone()
        s.runners = dict(key)
        s.phase = Phase.AWAIT_DECISION
        same(f"should_stop{key}", ps.should_stop(s), rs.should_stop(s))
        same(f"value{key}", ps.value(s), rs.value(s))
        for dice in ALL_DICE:
            if not legal_moves(s, dice):
                continue
            m = s.clone()
            m.phase = Phase.AWAIT_MOVE
            m.dice = dice
            same(f"choose_move{key}{dice}", ps.choose_move(m),
                 rs.choose_move(m))
            same(f"move value{key}{dice}", ps.value(m), rs.value(m))


def compare_solvers(state, evaluator_name, where="", *, rng=None,
                    max_keys=25):
    evaluate = EVALUATORS[evaluator_name]
    ps = TurnSolver(state, evaluate)
    rs = RustTurnSolver(state, evaluate)
    where = f"{where} evaluator={evaluator_name}\n{state!r}"
    compare_tables(ps, rs, where)
    compare_queries(ps, rs, state, where, rng=rng, max_keys=max_keys)
    return ps.num_positions


# ---- positions to solve ----

def sample_positions(rules, seed, count, *, max_python_positions=None):
    """In-turn positions from random games, spread over the game and over
    all three in-turn phases: turn start (AWAIT_ROLL), just rolled
    (AWAIT_MOVE, the phase self-play solves from), mid-turn
    (AWAIT_DECISION)."""
    # Mix the rule set into the seed: the same seed would otherwise give
    # every rule set the same opening positions.
    rng = PortableRng(seed * 1_000_003 + hash(
        (rules.num_players, rules.columns_to_win, rules.blocking)) % 997)
    out = []
    state = GameState(rules)
    take = 0.04          # ~one position per 25 steps: spread over the game
    while len(out) < count:
        if state.game_over:
            state = GameState(rules)
        if state.phase == Phase.AWAIT_ROLL and rng.next_float() < take:
            out.append(state.clone())
        moves = roll(state, random_dice(rng))
        if not moves:
            continue
        if rng.next_float() < take:
            out.append(state.clone())
        apply_move(state, moves[rng.randrange(len(moves))])
        if state.game_over:
            continue
        if rng.next_float() < take:
            out.append(state.clone())
        if can_stop(state) and rng.next_float() < 0.4:
            stop(state)
    return out[:count]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--positions-per-ruleset", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    started = time.perf_counter()
    solves = configs = 0
    for rules in ALL_RULESETS:
        t = time.perf_counter()
        n = 0
        for i, state in enumerate(sample_positions(
                rules, args.seed, args.positions_per_ruleset)):
            for name in EVALUATORS:
                n += compare_solvers(state, name, f"{rules} #{i}",
                                     rng=PortableRng(i))
                solves += 1
        configs += n
        print(f"{rules}: {n} configurations, "
              f"{time.perf_counter() - t:.0f} s", flush=True)
    print(f"\nGREEN: {solves} solves, {configs} configurations compared, "
          f"0 divergences, {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
