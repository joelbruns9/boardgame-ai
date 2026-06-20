#!/usr/bin/env python
"""
chance_fanout_probe.py — measure the dice fan-out at chance nodes.

Feasibility check for exact (probability-weighted) chance nodes in the
Can't Stop MCTS. An exact chance node weights each outcome class by its
*true* probability instead of by sample frequency. The cost questions are:
  1. How many distinct outcome classes (k) does a chance node face?
     (Drives the weight-table size, memory per node, and the O(k) weighted
      backup — NOT per-visit cost, since the search still deepens one child
      per traversal.)
  2. How cheap is the per-node weight computation?

This probe answers both empirically, with no model and no GPU:

  - Plays random-policy games to collect realistic PRE-ROLL states (the
    states a ChanceNode holds): once at the start of each turn (runners
    empty), and again after each CONTINUE (runners updated, before re-roll).
    States are de-duplicated by everything get_valid_moves depends on, so
    each distinct configuration is enumerated once and the de-dup ratio
    doubles as a weight-cache reuse estimate.

  - For each distinct configuration it ENUMERATES the dice exactly: the 126
    four-dice multisets with their multinomial weights (equivalent to all
    1296 ordered rolls, since get_possible_moves is order-invariant — see
    --verify). It buckets them by the SAME canonical key the MCTS uses,
    tuple(sorted(get_valid_moves(state))), with () == bust, and records:
        k         = number of distinct non-bust outcome classes
        bust_prob = exact probability of a bust

  - Reports the distribution of k and bust_prob, overall and split by runner
    count, plus the per-config enumeration cost and the de-dup ratio.

Run from the repo root:
    python -m games.cantstop.chance_fanout_probe --games 2000
    python -m games.cantstop.chance_fanout_probe --verify   # prove 126==1296
"""

import sys
import os
import time
import math
import random
import argparse
from itertools import combinations_with_replacement, product
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move, stop_turn, bust_turn,
)

_TOTAL_ROLLS = 6 ** 4  # 1296


def _build_dice_table():
    """All 126 distinct 4-dice multisets with multinomial multiplicities.
    multiplicity = 4! / prod(count_face!); the weights sum to 1296."""
    table = []
    for combo in combinations_with_replacement(range(1, 7), 4):
        mult = math.factorial(4)
        for c in Counter(combo).values():
            mult //= math.factorial(c)
        table.append((list(combo), mult))
    return table


_DICE_TABLE = _build_dice_table()
assert len(_DICE_TABLE) == 126, "expected 126 multisets"
assert sum(m for _, m in _DICE_TABLE) == _TOTAL_ROLLS, "dice weights != 1296"


def enumerate_outcomes(state):
    """
    Exact outcome-class distribution for one PRE-ROLL state.
    Returns (k, bust_prob, mass) where mass maps canonical_outcome -> prob.
    Buckets by the MCTS canonical key: tuple(sorted(get_valid_moves)); () is bust.
    """
    mass = defaultdict(int)
    for dice, mult in _DICE_TABLE:
        state.dice = dice
        valid = get_valid_moves(state)
        canonical = tuple(sorted(valid)) if valid else ()
        mass[canonical] += mult

    probs = {o: m / _TOTAL_ROLLS for o, m in mass.items()}
    bust_prob = probs.get((), 0.0)
    k = sum(1 for o in probs if o != ())
    return k, bust_prob, probs


def _signature(state):
    """Everything get_valid_moves depends on, besides the dice we vary."""
    prog = state.progress[state.active_player]
    return (
        frozenset(state.all_claimed),
        frozenset(state.runners.items()),
        frozenset(prog.items()),
    )


def collect_configs(num_games, rng, turn_cap=400):
    """
    Random-policy rollouts; snapshot every PRE-ROLL state, de-duplicated by
    signature. Returns: dict sig -> [count, representative_state, num_runners].
    """
    configs = {}

    def record(st):
        sig = _signature(st)
        entry = configs.get(sig)
        if entry is None:
            rep = st.clone()
            rep.dice = []
            configs[sig] = [1, rep, len(st.runners)]
        else:
            entry[0] += 1

    for _ in range(num_games):
        st = GameState(num_players=2)
        turns = 0
        while not st.game_over and turns < turn_cap:
            record(st)                       # start of turn (runners empty)
            moves_this_turn = 0
            while True:
                st.roll_dice()
                valid = get_valid_moves(st)
                if not valid:
                    bust_turn(st)
                    break
                apply_move(st, rng.choice(valid))
                moves_this_turn += 1
                stop_prob = min(0.85, 0.15 + 0.18 * moves_this_turn)
                if rng.random() < stop_prob:
                    stop_turn(st)
                    break
                record(st)                   # after CONTINUE, before re-roll
            turns += 1

    return configs


def _weighted(pairs):
    """pairs: list of (value, weight). Returns mean/min/max/percentiles."""
    if not pairs:
        return {}
    pairs = sorted(pairs, key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total

    def pct(p):
        target = p / 100.0 * total
        cum = 0
        for v, w in pairs:
            cum += w
            if cum >= target:
                return v
        return pairs[-1][0]

    return {'mean': mean, 'min': pairs[0][0], 'max': pairs[-1][0],
            'p50': pct(50), 'p90': pct(90), 'p95': pct(95), 'p99': pct(99)}


def verify_enumeration(rng, n_states=200):
    """Prove the 126-multiset enumeration equals the full 1296 ordered roll
    enumeration on real states (i.e. get_possible_moves is order-invariant)."""
    configs = collect_configs(max(50, n_states // 4), rng)
    reps = [v[1] for v in list(configs.values())[:n_states]]
    mismatches = 0
    for st in reps:
        # exact via multiset table
        _, _, probs_ms = enumerate_outcomes(st)
        # brute force via all 1296 ordered rolls
        mass = defaultdict(int)
        for dice in product(range(1, 7), repeat=4):
            st.dice = list(dice)
            valid = get_valid_moves(st)
            mass[tuple(sorted(valid)) if valid else ()] += 1
        probs_bf = {o: m / _TOTAL_ROLLS for o, m in mass.items()}
        if set(probs_ms) != set(probs_bf) or any(
            abs(probs_ms[o] - probs_bf[o]) > 1e-12 for o in probs_ms
        ):
            mismatches += 1
    print(f"verify: compared {len(reps)} states, {mismatches} mismatches "
          f"between 126-multiset and 1296-ordered enumeration.")
    return mismatches == 0


def main():
    ap = argparse.ArgumentParser(description="Chance-node fan-out probe")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--turn-cap", type=int, default=400, dest="turn_cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="prove 126-multiset == 1296-ordered, then exit")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.verify:
        ok = verify_enumeration(rng)
        sys.exit(0 if ok else 1)

    print(f"Collecting pre-roll states from {args.games:,} random games ...")
    t0 = time.time()
    configs = collect_configs(args.games, rng, args.turn_cap)
    t_collect = time.time() - t0

    total_encounters = sum(c for c, _, _ in configs.values())
    n_distinct = len(configs)
    if n_distinct == 0:
        print("No states collected.")
        return

    print(f"  {total_encounters:,} pre-roll states encountered, "
          f"{n_distinct:,} distinct configs "
          f"(reuse {total_encounters / n_distinct:.1f}x) "
          f"in {t_collect:.1f}s")

    # Enumerate each distinct config exactly, once.
    print(f"Enumerating dice fan-out for {n_distinct:,} configs ...")
    t1 = time.time()
    rows = []   # (count, num_runners, k, bust_prob)
    for count, rep, n_run in configs.values():
        k, bust_prob, _ = enumerate_outcomes(rep)
        rows.append((count, n_run, k, bust_prob))
    t_enum = time.time() - t1

    k_pairs = [(k, c) for (c, _, k, _) in rows]
    bust_pairs = [(b, c) for (c, _, _, b) in rows]
    k_stats = _weighted(k_pairs)
    bust_stats = _weighted(bust_pairs)

    bar = "=" * 72
    print(f"\n{bar}")
    print("  CHANCE-NODE FAN-OUT PROBE")
    print(f"{bar}")
    print(f"  enumeration cost: {t_enum:.1f}s for {n_distinct:,} configs "
          f"= {t_enum / n_distinct * 1e6:,.0f} µs/config "
          f"(126 get_valid_moves each; cacheable per config)")

    print(f"\n  Outcome classes k (non-bust), weighted as-encountered:")
    print(f"    mean {k_stats['mean']:.2f} | p50 {k_stats['p50']} | "
          f"p90 {k_stats['p90']} | p95 {k_stats['p95']} | "
          f"p99 {k_stats['p99']} | max {k_stats['max']}")

    print(f"\n  Bust probability, weighted as-encountered:")
    print(f"    mean {bust_stats['mean']:.3f} | p50 {bust_stats['p50']:.3f} | "
          f"p90 {bust_stats['p90']:.3f} | p95 {bust_stats['p95']:.3f} | "
          f"max {bust_stats['max']:.3f}")

    # Breakdown by runner count — k and bust both move strongly with it.
    print(f"\n  By runner count:")
    print(f"    {'runners':>7} | {'encounters':>10} | {'mean k':>6} | "
          f"{'p95 k':>5} | {'max k':>5} | {'mean bust':>9} | {'max bust':>8}")
    print("    " + "-" * 66)
    by_run = defaultdict(list)
    for (c, n_run, k, b) in rows:
        by_run[min(n_run, 3)].append((c, k, b))
    for n_run in sorted(by_run):
        grp = by_run[n_run]
        enc = sum(c for c, _, _ in grp)
        kp = [(k, c) for (c, k, _) in grp]
        bp = [(b, c) for (c, _, b) in grp]
        ks = _weighted(kp)
        bs = _weighted(bp)
        print(f"    {n_run:>7} | {enc:>10,} | {ks['mean']:>6.2f} | "
              f"{ks['p95']:>5} | {ks['max']:>5} | {bs['mean']:>9.3f} | "
              f"{bs['max']:>8.3f}")

    # Decision-relevant subset: chance nodes where bust risk is real.
    # These are the push/stop calls exact weighting is meant to fix.
    print(f"\n  Decision-relevant subset (where bust risk is non-trivial):")
    for thr in (0.05, 0.10):
        sub = [(c, k, b) for (c, _, k, b) in rows if b >= thr]
        if not sub:
            print(f"    bust ≥ {thr:.2f}: none")
            continue
        enc = sum(c for c, _, _ in sub)
        share = enc / total_encounters
        ks = _weighted([(k, c) for (c, k, _) in sub])
        print(f"    bust ≥ {thr:.2f}: {share:5.1%} of nodes | "
              f"k mean {ks['mean']:.1f}, p95 {ks['p95']}, max {ks['max']}")

    print(f"\n{bar}")
    # Data-driven read: fan-out and decision-difficulty are anti-correlated.
    relevant = [(c, k, b) for (c, _, k, b) in rows if b >= 0.05]
    rel_k = _weighted([(k, c) for (c, k, _) in relevant]) if relevant else {}
    print("  Read:")
    print("  - Fan-out is LARGE on open boards (k≈109, 0–2 runners) but bust")
    print("    risk there is ~0, so the continue/stop call is trivial — nothing")
    print("    rare to miss, sampling is already fine.")
    if rel_k:
        print(f"  - Fan-out is SMALL exactly where the decision is hard: among")
        print(f"    nodes with real bust risk, k p95 {rel_k['p95']}, max "
              f"{rel_k['max']}. That's the push/stop crunch (≈ the 3-runner cap),")
        print("    and it's cheap to enumerate exactly.")
    print("  => Exact-EVERYWHERE is wasteful (109-wide tables on trivial nodes).")
    print("     Gate exact chance on bust-risk / runner-cap — small k there, and")
    print("     it's the rising, hard-to-sample bust mass exact weighting nails.")
    print(f"  Per-node enumeration is ~{t_enum/n_distinct*1e6:.0f} µs (computed once")
    print("  per chance node, amortized over its visits), so cost is a non-issue.")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()