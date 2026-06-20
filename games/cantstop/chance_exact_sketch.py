#!/usr/bin/env python
"""
chance_exact_sketch.py — gated exact (probability-weighted) chance nodes,
                         hardened against conservatism.

DESIGN (from the fan-out probe):
  Fan-out k is ~109 on open boards (bust ≈ 0, trivial decision) and collapses
  to ≤~23 at the runner cap, where bust risk is real and variable — the
  push/stop crunch. So exact weighting is GATED: on at the cap, sampled
  elsewhere.

WHAT CHANGES vs the live engine:
  - ChanceNode.Q returns the exact probability-weighted expectation on gated
    nodes instead of the sampled mean W/N.
  - Descent and backprop are UNCHANGED. Rolling dice already samples outcomes
    at true probability, so which child is deepened is already correct; only
    the *aggregation* changes. DecisionNode.select_child reads child.Q and is
    untouched.

ANTI-CONSERVATISM (the point of this revision):
  1. NEUTRAL default (CHANCE_FALLBACK = 0.5) for not-yet-evaluated outcome
     mass — never the sampled mean W/N. At a continue node bust dominates the
     early samples, so W/N is pessimistic and would make continue look worse
     than it is (over-stopping). 0.5 also blocks the opposite trap: an
     unsearched bust child (a ChanceNode, N=0) reads Q=0 in the OPPONENT's
     perspective, which flips to 1.0 in ours ("busting wins"); holding it at
     0.5 until it's actually searched prevents that false optimism too.
  2. EAGER expansion (EAGER_EXPANSION): at a gated node, create & (caller-)
     evaluate all ≤~23 outcome children up front, so the exact expectation —
     including the rare position-flipping outcomes — is right from the first
     visit, with no transient under-valuation of continue. The gate is what
     makes those ≤23 batched evals affordable.

A/B: EXACT_CHANCE_ENABLED=False → today's pure-sampled engine. EAGER_EXPANSION
toggles eager vs lazy independently. Validate with run_oracle_comparison().

INTEGRATION: replace ChanceNode in mcts.py with the class below (adds 3 slots),
add the module-level helpers, and in the chance branch of _traverse/_step_sim,
when node._exact and EAGER_EXPANSION and node has no children yet, call
node.build_outcome_children(DecisionNode, ChanceNode), batch-evaluate the
returned DecisionNodes (self.evaluate_batch), and seed each:
    child.N, child.W = 1, value ; self.expand_decision_node(child, priors)
Then proceed (descent reads the now-exact Q). Lazy mode needs no call-site
change at all.
"""

import math
import argparse
from itertools import combinations_with_replacement
from collections import Counter, defaultdict

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move, stop_turn, bust_turn, MAX_RUNNERS,
)


# ============================================================
# A/B switches + gate + defaults
# ============================================================

EXACT_CHANCE_ENABLED = True
EAGER_EXPANSION = True
CHANCE_FALLBACK = 0.5   # neutral; see anti-conservatism note (1) above.


def should_use_exact(state):
    """
    Cheap structural gate — O(1), no enumeration needed to decide.

    Primary: the runner cap. k collapses to ≤~23 and bust risk becomes real
    exactly when all MAX_RUNNERS runners are committed, so we pay enumeration
    only on the ~18% of nodes that need it.

    More precise alternative (costs one enumeration to decide): gate on
    chance_outcome_weights(state).get((), 0.0) >= TAU, which also catches the
    elevated-bust 2-runner tail (k up to ~70, still cheap). Start with the
    cap; widen only if the oracle A/B shows the tail matters.
    """
    if not EXACT_CHANCE_ENABLED:
        return False
    return len(state.runners) == MAX_RUNNERS


# ============================================================
# Exact dice enumeration (faithful to _sample_chance_outcome)
# ============================================================

_TOTAL_ROLLS = 6 ** 4  # 1296


def _build_dice_table():
    """126 distinct 4-dice multisets with multinomial multiplicities
    (sum == 1296). get_possible_moves is order-invariant, so multiset
    enumeration is exact and ~10x cheaper than all 1296 ordered rolls."""
    table = []
    for combo in combinations_with_replacement(range(1, 7), 4):
        mult = math.factorial(4)
        for c in Counter(combo).values():
            mult //= math.factorial(c)
        table.append((list(combo), mult))
    return table


_DICE_TABLE = _build_dice_table()


def chance_outcome_weights(state):
    """
    Exact probability of each canonical outcome class for a PRE-ROLL state.
    Canonicalization MATCHES _sample_chance_outcome:
        normal -> tuple(sorted(get_valid_moves(state)));  bust -> ().
    Returns {canonical_outcome: probability}, summing to 1.0. Compute once
    per node and cache. Restores state.dice to [] (pre-roll) on exit.
    """
    mass = defaultdict(int)
    for dice, mult in _DICE_TABLE:
        state.dice = dice
        valid = get_valid_moves(state)
        mass[tuple(sorted(valid)) if valid else ()] += mult
    state.dice = []
    return {o: m / _TOTAL_ROLLS for o, m in mass.items()}


# ============================================================
# ChanceNode — drop-in replacement for the one in mcts.py
# ============================================================

class ChanceNode:
    """
    Nature rolls dice (pre-roll state; dice cleared). Unchanged fields:
    state, parent, parent_action, prior, children_by_outcome, N, W,
    flip_from_parent. New: _weights, _exact, _weights_ready.
    """

    __slots__ = [
        'state', 'parent', 'parent_action', 'prior',
        'children_by_outcome', 'N', 'W', 'flip_from_parent',
        '_weights', '_exact', '_weights_ready',
    ]

    def __init__(self, state, parent=None, parent_action=None,
                 prior=0.0, flip_from_parent=False):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        self.flip_from_parent = flip_from_parent
        self.children_by_outcome = {}
        self.N = 0
        self.W = 0.0
        self._exact = should_use_exact(state)   # O(1) gate decision
        self._weights = None
        self._weights_ready = False

    def _ensure_weights(self):
        if not self._weights_ready:
            self._weights = chance_outcome_weights(self.state)
            self._weights_ready = True

    def build_outcome_children(self, make_decision, make_chance):
        """
        EAGER expansion. Create a child for every outcome class so the exact
        expectation is correct from the first visit (no fallback for the
        normal outcomes). Mirrors _sample_chance_outcome's normal/bust split.

        make_decision / make_chance: the real DecisionNode / ChanceNode
        constructors, injected (avoids importing the heavy mcts module here,
        and lets the unit test pass stubs).

        Returns the list of newly created DecisionNode children that still
        need a network value. CALLER must batch-evaluate them and seed each:
            child.N, child.W = 1, value
            expand_decision_node(child, priors)
        The bust child is a nested ChanceNode (no NN value) — it gets a value
        when the search descends into it, and uses CHANCE_FALLBACK until then.
        """
        self._ensure_weights()
        to_eval = []
        reps = {}            # canonical -> one representative rolled state
        bust_state = None

        for dice, _mult in _DICE_TABLE:
            s = self.state.clone()
            s.dice = list(dice)
            valid = get_valid_moves(s)
            if valid:
                canonical = tuple(sorted(valid))
                reps.setdefault(canonical, s)
            elif bust_state is None:
                bs = self.state.clone()
                bs.dice = list(dice)
                bust_turn(bs)
                bs.dice = []
                bust_state = bs

        for canonical, rep in reps.items():
            if canonical in self.children_by_outcome:
                continue
            child = make_decision(state=rep, parent=self, parent_action=None,
                                  prior=0.0, flip_from_parent=False)
            self.children_by_outcome[canonical] = child
            to_eval.append(child)

        if bust_state is not None and () not in self.children_by_outcome:
            flip = (bust_state.active_player != self.state.active_player)
            self.children_by_outcome[()] = make_chance(
                state=bust_state, parent=self, parent_action=None,
                prior=0.0, flip_from_parent=flip,
            )

        return to_eval

    @property
    def Q(self):
        if not self._exact:
            # Identical to the current engine: pure sampled mean.
            return self.W / self.N if self.N > 0 else 0.0

        self._ensure_weights()
        weighted = 0.0
        covered = 0.0    # probability mass of outcomes with a real estimate
        for outcome, p in self._weights.items():
            child = self.children_by_outcome.get(outcome)
            if child is not None and child.N > 0:
                cq = child.Q
                # Same perspective convention as DecisionNode.select_child:
                # a player change across the edge flips win-prob to 1 - q.
                if child.flip_from_parent:
                    cq = 1.0 - cq
                weighted += p * cq
                covered += p

        # Not-yet-evaluated mass uses the NEUTRAL default — never W/N. As
        # children fill in (immediately, under eager expansion), covered -> 1
        # and Q -> the pure exact expectation.
        return weighted + (1.0 - covered) * CHANCE_FALLBACK


# ============================================================
# Oracle comparison harness (anti-conservatism validation)
# ============================================================

def collect_cap_positions(num_games, rng, max_positions, turn_cap=400):
    """
    Random-policy rollouts; collect post-roll DECISION states at the runner
    cap (dice rolled, valid moves present, all runners committed) — the
    states where stop-vs-continue is a live, hard choice. Engine-only.
    """
    out = []
    for _ in range(num_games):
        if len(out) >= max_positions:
            break
        st = GameState(num_players=2)
        turns = 0
        while not st.game_over and turns < turn_cap and len(out) < max_positions:
            moves_this_turn = 0
            while True:
                st.roll_dice()
                valid = get_valid_moves(st)
                if not valid:
                    bust_turn(st)
                    break
                if len(st.runners) == MAX_RUNNERS and len(out) < max_positions:
                    out.append(st.clone())          # crunch decision state
                apply_move(st, rng.choice(valid))
                moves_this_turn += 1
                if rng.random() < min(0.85, 0.15 + 0.18 * moves_this_turn):
                    stop_turn(st)
                    break
            turns += 1
    return out


def summarize_oracle(results):
    """
    results: list of (exact_continue: bool, oracle_continue: bool).
    Reports continue rates, the conservatism gap (oracle - exact continue
    rate; positive => exact over-stops), and the disagreement breakdown.
    """
    n = len(results)
    if n == 0:
        return {}
    exact_cont = sum(1 for e, _ in results if e)
    oracle_cont = sum(1 for _, o in results if o)
    agree = sum(1 for e, o in results if e == o)
    conservative_err = sum(1 for e, o in results if (not e) and o)   # exact stops, oracle continues
    aggressive_err = sum(1 for e, o in results if e and (not o))     # exact continues, oracle stops
    return {
        'n': n,
        'exact_continue_rate': exact_cont / n,
        'oracle_continue_rate': oracle_cont / n,
        'conservatism_gap': (oracle_cont - exact_cont) / n,
        'agreement': agree / n,
        'conservative_errors': conservative_err / n,
        'aggressive_errors': aggressive_err / n,
    }


def run_oracle_comparison(model_path, device='cpu', num_positions=200,
                          low_sims=1200, oracle_sims=5000, seed=0):
    """
    RUN ON YOUR INTEGRATED TREE (needs the model and the wired mcts.py).

    For each cap decision state, compare:
      candidate: EXACT_CHANCE_ENABLED=True at low_sims
      oracle:    EXACT_CHANCE_ENABLED=False (pure sampled) at oracle_sims
    The premise is that exact at low budget reproduces the converged sampled
    oracle. A positive conservatism_gap means exact is over-stopping relative
    to the oracle — a fallback/calibration bug to chase (try eager first).

    Assumes mcts exposes MCTS, action_to_move_decision, and the
    EXACT_CHANCE_ENABLED flag, and self_play exposes load_model. Adjust the
    imports to your tree if names differ.
    """
    import random
    from games.cantstop import mcts as M
    from games.cantstop.mcts import MCTS, action_to_move_decision
    from games.cantstop.self_play import load_model

    model = load_model(model_path, device)
    rng = random.Random(seed)
    positions = collect_cap_positions(2000, rng, num_positions)

    def decide(state, exact, sims):
        M.EXACT_CHANCE_ENABLED = exact
        action = MCTS(model, device).get_action(state, sims, temperature=0.0)
        _, decision = action_to_move_decision(int(action))
        return decision == "continue"

    results = []
    for st in positions:
        exact_cont = decide(st.clone(), True, low_sims)
        oracle_cont = decide(st.clone(), False, oracle_sims)
        results.append((exact_cont, oracle_cont))

    M.EXACT_CHANCE_ENABLED = True
    summary = summarize_oracle(results)
    print(f"\n  Oracle comparison ({summary['n']} cap positions, "
          f"exact@{low_sims} vs sampled@{oracle_sims}):")
    print(f"    continue rate  exact {summary['exact_continue_rate']:.1%} | "
          f"oracle {summary['oracle_continue_rate']:.1%}")
    print(f"    conservatism gap (oracle - exact): "
          f"{summary['conservatism_gap']:+.1%}  "
          f"(positive => exact over-stops)")
    print(f"    agreement {summary['agreement']:.1%} | "
          f"conservative errors {summary['conservative_errors']:.1%} | "
          f"aggressive errors {summary['aggressive_errors']:.1%}")
    return summary


# ============================================================
# Self-tests (run against the real engine; no model needed)
# ============================================================

def _selftest():
    import random

    # --- 1. weights valid; capped state has real bust; gate fires ---
    st = GameState(num_players=2)
    st.active_player = 0
    st.runners = {6: 2, 8: 1, 7: 3}        # 3 runners == cap
    w = chance_outcome_weights(st)
    k = sum(1 for o in w if o != ())
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert should_use_exact(st)
    print(f"  capped state: k={k}, bust={w.get((), 0.0):.3f}, sum=1  OK")

    # --- 2. open state not gated ---
    st2 = GameState(num_players=2)
    st2.runners = {}
    assert not should_use_exact(st2)
    print("  open state: gate off (sampled)  OK")

    # --- 3. neutral fallback: empty Q == 0.5 regardless of W/N ---
    cn = ChanceNode(st)
    cn._ensure_weights()
    cn.N, cn.W = 4, 3.0                     # sampled mean 0.75 (pessimism trap if used)
    assert abs(cn.Q - CHANCE_FALLBACK) < 1e-9, f"Q {cn.Q} != neutral {CHANCE_FALLBACK}"
    print(f"  neutral fallback: empty Q == {CHANCE_FALLBACK} (ignores W/N)  OK")

    # --- 4. Q converges to the exact probability-weighted expectation ---
    class MockChild:
        __slots__ = ['N', 'W', 'flip_from_parent']
        def __init__(self, q, flip=False):
            self.N, self.W, self.flip_from_parent = 1, q, flip
        @property
        def Q(self):
            return self.W / self.N if self.N > 0 else 0.0

    cn2 = ChanceNode(st)
    cn2._ensure_weights()
    outcomes = list(cn2._weights.keys())
    val_for = {o: (0.10 if o == () else 0.70) for o in outcomes}
    for o in outcomes:
        if o == ():                          # bust child flips perspective
            cn2.children_by_outcome[o] = MockChild(1.0 - val_for[o], flip=True)
        else:
            cn2.children_by_outcome[o] = MockChild(val_for[o], flip=False)
    expected = sum(cn2._weights[o] * val_for[o] for o in outcomes)
    assert abs(cn2.Q - expected) < 1e-9, f"exact Q {cn2.Q} != {expected}"
    print(f"  exact convergence: full Q == weighted mean ({cn2.Q:.4f})  OK")

    # --- 5. eager expansion builds all outcome children faithfully ---
    class _DStub:
        __slots__ = ['state', 'parent', 'parent_action', 'prior',
                     'flip_from_parent', 'N', 'W']
        def __init__(self, state, parent, parent_action, prior, flip_from_parent):
            self.state, self.parent = state, parent
            self.parent_action, self.prior = parent_action, prior
            self.flip_from_parent, self.N, self.W = flip_from_parent, 0, 0.0

    class _CStub(_DStub):
        pass

    cn3 = ChanceNode(st)
    to_eval = cn3.build_outcome_children(_DStub, _CStub)
    keys = set(cn3.children_by_outcome)
    assert keys == set(w), "eager children must cover every outcome class"
    assert len(to_eval) == k, f"to_eval {len(to_eval)} should equal k={k}"
    assert all(c.flip_from_parent is False for c in to_eval), "normals don't flip"
    if () in keys:
        bust_child = cn3.children_by_outcome[()]
        assert isinstance(bust_child, _CStub), "bust child must be a ChanceNode"
        assert bust_child.flip_from_parent is True, "bust flips perspective"
    print(f"  eager expansion: built {len(keys)} children "
          f"({k} decisions + bust), perspectives correct  OK")

    # --- 6. oracle summary aggregation (pure logic) ---
    mock = [(True, True)] * 60 + [(False, True)] * 30 + [(False, False)] * 8 \
        + [(True, False)] * 2
    s = summarize_oracle(mock)
    assert s['n'] == 100
    assert abs(s['conservatism_gap'] - 0.28) < 1e-9   # oracle 90% - exact 62%
    assert abs(s['conservative_errors'] - 0.30) < 1e-9
    print(f"  oracle summary: gap {s['conservatism_gap']:+.0%}, "
          f"conservative errs {s['conservative_errors']:.0%}  OK")

    # --- 7. cap-position collector returns valid crunch states ---
    pos = collect_cap_positions(200, random.Random(0), 50)
    assert pos, "expected some cap positions"
    assert all(len(p.runners) == MAX_RUNNERS for p in pos)
    assert all(get_valid_moves(p) for p in pos), "cap states must have moves"
    print(f"  cap collector: {len(pos)} states, all at cap with live moves  OK")

    print("All sketch self-tests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gated exact-chance sketch")
    ap.add_argument("--oracle", action="store_true",
                    help="run exact-vs-sampled-oracle comparison (needs --model "
                         "and the integrated mcts.py)")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--low-sims", type=int, default=1200, dest="low_sims")
    ap.add_argument("--oracle-sims", type=int, default=5000, dest="oracle_sims")
    args = ap.parse_args()

    if args.oracle:
        if not args.model:
            ap.error("--oracle requires --model")
        run_oracle_comparison(args.model, args.device, args.positions,
                              args.low_sims, args.oracle_sims)
    else:
        _selftest()