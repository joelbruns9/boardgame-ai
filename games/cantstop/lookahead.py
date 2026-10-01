"""Selective 2-turn lookahead -- the Python reference for
``cantstop_rust/src/lookahead.rs`` and ``TurnSolver::leaf_reach``.

The turn solver scores every end-of-turn board with the net. Lookahead
re-scores the boards that matter most by solving the NEXT player's whole
turn from each, from before their opening roll: the exact expectation over
their dice and their best play, with the net at the end of *that* turn. One
turn of search across the turn boundary, where it counts:

1. solve the turn as usual;
2. ``choose``: the bust board plus the top ``k - 1`` stop leaves by
   ``leaf_reach`` -- how likely best play is to reach each leaf's
   stop-or-roll decision (ties by key; reachable leaves only);
3. solve the next turn from each chosen board; its root roll value is the
   refined value;
4. ``refine``: chosen leaves take the refined value; with ``offset``, every
   other leaf shifts by the reach-weighted mean refinement, so an option is
   neither favoured nor penalised just for having been looked at, and is
   then projected back onto the probability simplex (review finding P1: a
   shift keeps the sum but not the signs); back up again.

The offset is an UNVALIDATED heuristic (review, 2026-09-25): lookahead is
parked for training until it shows strength per unit time in balanced-seat
matches.

Cost: k extra turn-start solves per turn, each with its own leaves for the
net. Every sum runs in a fixed order so Rust matches bit for bit.
"""

from .engine import Phase
from .solver import TurnSolver, runners_key

BUST = None      # the bust board's leaf id


def _height(key):
    return sum(pos for _, pos in key)


def leaf_reach(ps, state):
    """(bust reach, {stop leaf key: reach}) under ``ps``'s current policy,
    starting from ``state`` (the position ``ps`` was built from)."""
    a = ps.active
    if state.phase == Phase.AWAIT_MOVE:
        start = ps.best_child(runners_key(state.runners), state.dice)
    else:
        start = runners_key(state.runners)
    reach = {start: 1.0}
    bust = 0.0
    stops = {}
    for key in sorted(ps.keys, key=lambda k: (_height(k), k)):
        r = reach.get(key, 0.0)
        if r == 0.0 or ps.winning[key]:
            continue
        if ps.stoppable[key]:
            stops[key] = r
            if ps.decision_values[key] is ps.stop_values[key]:
                continue                      # the policy stops here
        bust += r * ps.bust_prob[key]
        for children, p in ps.menus[key]:
            best = max(children, key=lambda c: ps.decision_values[c][a])
            reach[best] = reach.get(best, 0.0) + r * p
    return bust, stops


def choose(ps, state, k):
    """Leaves to refine, in refinement order, and their reach weights."""
    if k <= 0:
        return [], []
    bust, stops = leaf_reach(ps, state)
    ranked = sorted((key for key, r in stops.items() if r > 0.0),
                    key=lambda key: (-stops[key], key))
    leaves = [BUST] + ranked[:k - 1]
    weights = [bust] + [stops[key] for key in ranked[:k - 1]]
    return leaves, weights


def _project(row):
    """Clip negatives to zero and renormalise, in seat order -- as
    ``project_to_simplex`` in Rust. Review finding P1: the common shift
    kept each row's sum but not its entries >= 0, and negative entries
    became negative training targets."""
    row = [x if x >= 0.0 else 0.0 for x in row]
    total = 0.0
    for x in row:
        total += x
    return [x / total for x in row]


def refine(ps, state, evaluate, k, offset=True):
    """Apply the lookahead to a solved ``ps`` in place. Returns the refined
    leaves (for tests)."""
    leaves, weights = choose(ps, state, k)
    if not leaves:
        return leaves
    n = ps.num_players
    v1 = {BUST: ps.bust_value}
    for key in ps.keys:
        if ps.stoppable[key] and not ps.winning[key]:
            v1[key] = ps.stop_values[key]
    refined = []
    for leaf in leaves:
        board = ps._board_after(() if leaf is BUST else leaf)
        refined.append(TurnSolver(board, evaluate).value(board))
    new = dict(v1)
    if offset:
        d = [0.0] * n
        wsum = 0.0
        for leaf, w, r in zip(leaves, weights, refined):
            for s in range(n):
                d[s] += w * (r[s] - v1[leaf][s])
            wsum += w
        if wsum > 0.0:
            d = [x / wsum for x in d]
            chosen = set(leaves)
            for leaf, v in v1.items():
                if leaf not in chosen:
                    new[leaf] = _project([v[s] + d[s] for s in range(n)])
    for leaf, r in zip(leaves, refined):
        new[leaf] = r
    import numpy as np
    ps.bust_value = np.asarray(new.pop(BUST), dtype=np.float64)
    for key, v in new.items():
        ps.stop_values[key] = np.asarray(v, dtype=np.float64)
    ps._backup()
    return leaves
