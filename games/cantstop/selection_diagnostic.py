#!/usr/bin/env python
"""
selection_diagnostic.py — diagnose observation 2.

On lopsided positions (win prob >95% or <5%) the search spreads visits
~uniformly and the advisor's move looks indecisive — worst on the losing
(<5%) side. This pins down WHY by measuring, per lopsided position, three
spreads across the root's legal actions, plus an alignment check:

  Q spread       small  => the value head can't DISTINGUISH the actions
                           (a value-resolution / objective problem)
  visit entropy  high   => the search did NOT concentrate (visits ~uniform)
  prior entropy  high   => the policy-head priors are flat (undertrained
                           policy head, e.g. an early iteration)
  argmax-visit vs argmax-Q : does the move the advisor would PICK (most
                           visits) match the best-value move? Frequent
                           mismatch on lopsided positions is the bug biting.

Verdict mapping:
  compressed Q                              -> value head can't resolve the
                                               actions; needs a margin/score-
                                               aware target. More sims won't help.
  Q separated, visits uniform, priors flat  -> undertrained policy head; more
                                               training sharpens priors and
                                               concentrates visits.
  Q separated, visits uniform, priors sharp -> PUCT not exploiting the signal;
                                               tune c_puct / FPU.

Runs against your CURRENT mcts.py (no exact-chance integration needed) — it
only needs the model. Without --model it runs offline self-tests.

  python -m games.cantstop.selection_diagnostic --model models/cantstop/best_model.pt
"""

import math
import argparse
import random
from collections import Counter

from games.cantstop.engine import (
    GameState, get_valid_moves, apply_move, stop_turn, bust_turn,
)


# ============================================================
# Pure metrics (unit-tested offline)
# ============================================================

def norm_entropy(weights):
    """Entropy of `weights` normalized to [0,1]: 1.0 = uniform, 0.0 = one-hot.
    Used for both visit counts and priors over the legal actions."""
    n = len(weights)
    if n <= 1:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return 1.0  # no information => treat as maximally spread
    h = 0.0
    for w in weights:
        if w > 0:
            p = w / total
            h -= p * math.log(p)
    return h / math.log(n)


def classify_position(q_spread, visit_ent, prior_ent,
                      q_flat=0.03, visit_unif=0.90, prior_flat=0.85):
    """Map the three spreads to a likely cause."""
    if q_spread < q_flat:
        return "compressed_Q"           # value head can't tell actions apart
    if visit_ent > visit_unif:
        return "flat_priors" if prior_ent > prior_flat else "puct_diffuse"
    return "concentrating"              # search is working on this position


# ============================================================
# Position collection (engine-only; unit-tested offline)
# ============================================================

def collect_decision_positions(num_games, rng, pool_cap=6000, turn_cap=400):
    """Random-policy rollouts; snapshot post-roll DECISION states (dice set,
    valid moves present) throughout games — the lopsided ones cluster near
    game end and get filtered out later."""
    out = []
    for _ in range(num_games):
        if len(out) >= pool_cap:
            break
        st = GameState(num_players=2)
        turns = 0
        while not st.game_over and turns < turn_cap and len(out) < pool_cap:
            while True:
                st.roll_dice()
                valid = get_valid_moves(st)
                if not valid:
                    bust_turn(st)
                    break
                out.append(st.clone())          # decision state
                if len(out) >= pool_cap:
                    break
                apply_move(st, rng.choice(valid))
                if rng.random() < 0.4:
                    stop_turn(st)
                    break
            turns += 1
    return out


# ============================================================
# Live search access (mirrors MCTS.search; needs the model)
# ============================================================

def search_with_root(engine, mcts_mod, state, sims):
    """Mirror of MCTS.search's setup, but return the root DecisionNode for
    inspection. No Dirichlet noise (we want the model's raw priors). Reuses
    the engine's own evaluate / expand / simulation methods, so it stays
    faithful to the production search."""
    s = state.clone()
    if not s.dice:
        s.roll_dice()
    root = mcts_mod.DecisionNode(state=s, parent=None, parent_action=None,
                                 prior=0.0, flip_from_parent=False)
    if not root.valid_moves:
        return None
    value, priors = engine.evaluate(root.state, root.valid_moves, root.mask)
    engine.expand_decision_node(root, priors)
    root.N, root.W = 1, value
    remaining = sims - 1
    if remaining > 0:
        if engine.target_inflight <= 1:
            engine._run_sync_simulations(root, remaining)
        else:
            engine._run_async_simulations(root, remaining)
    return root


def analyze_root(root):
    """Per-action stats at the root, plus the spreads and the visit/Q
    alignment. Q is converted to the root player's perspective (1 - q on a
    perspective-flipping edge), matching DecisionNode.select_child."""
    acts = []  # (action, q_root, N, prior)
    for a, child in root.children.items():
        q = child.Q
        if getattr(child, "flip_from_parent", False):
            q = 1.0 - q
        acts.append((a, q, child.N, child.prior))

    visits = [n for (_, _, n, _) in acts]
    priors = [p for (_, _, _, p) in acts]
    visited = [(a, q, n, p) for (a, q, n, p) in acts if n >= 1]
    q_vals = [q for (_, q, _, _) in visited] or [0.0]

    argmax_visit = max(acts, key=lambda t: t[2])[0]
    argmax_q = max(visited, key=lambda t: t[1])[0] if visited else None
    total_v = sum(visits)
    return {
        "n_legal": len(acts),
        "q_spread": max(q_vals) - min(q_vals),
        "best_q": max(q_vals),
        "visit_ent": norm_entropy(visits),
        "prior_ent": norm_entropy(priors),
        "aligned": (argmax_visit == argmax_q),
        "top_visit_share": (max(visits) / total_v) if total_v else 0.0,
        "class": classify_position(
            max(q_vals) - min(q_vals),
            norm_entropy(visits), norm_entropy(priors)),
    }


def root_value_only(engine, mcts_mod, state):
    """Just the NN root value, for cheaply filtering lopsided positions."""
    s = state.clone()
    if not s.dice:
        s.roll_dice()
    node = mcts_mod.DecisionNode(state=s, parent=None, parent_action=None,
                                 prior=0.0, flip_from_parent=False)
    if not node.valid_moves:
        return None
    value, _ = engine.evaluate(node.state, node.valid_moves, node.mask)
    return value


def _summarize_bucket(name, rows):
    if not rows:
        print(f"\n  {name}: none found")
        return None
    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n
    misaligned = sum(1 for r in rows if not r["aligned"]) / n
    classes = Counter(r["class"] for r in rows)
    print(f"\n  {name} ({n} positions):")
    print(f"    mean Q spread       {mean('q_spread'):.3f}   "
          f"(small => value head can't separate actions)")
    print(f"    mean visit entropy  {mean('visit_ent'):.3f}   "
          f"(→1 = visits uniform, →0 = concentrated)")
    print(f"    mean prior entropy  {mean('prior_ent'):.3f}   "
          f"(→1 = flat priors / undertrained policy)")
    print(f"    mean top-visit share {mean('top_visit_share'):.3f}")
    print(f"    argmax-visit != argmax-Q on {misaligned:.0%} of positions "
          f"(advisor picking a non-best-value move)")
    print(f"    classes: " + ", ".join(f"{k}={v}" for k, v in classes.most_common()))
    return classes.most_common(1)[0][0]


def run_diagnostic(model_path, device="cpu", num_games=1500, sims=1600,
                   low=0.05, high=0.95, max_per_bucket=25, seed=0):
    from games.cantstop import mcts as mcts_mod
    from games.cantstop.mcts import MCTS
    from games.cantstop.evaluate import load_model

    model = load_model(model_path, device)
    engine = MCTS(model, device, target_inflight=1, warmup_sims=16)
    rng = random.Random(seed)

    print(f"Collecting positions ({num_games} random games) ...")
    pool = collect_decision_positions(num_games, rng)
    print(f"  pool of {len(pool):,} decision states; filtering for "
          f"value <{low} or >{high} ...")

    losing, winning = [], []
    for s in pool:
        if len(losing) >= max_per_bucket and len(winning) >= max_per_bucket:
            break
        v = root_value_only(engine, mcts_mod, s)
        if v is None:
            continue
        if v < low and len(losing) < max_per_bucket:
            losing.append(s)
        elif v > high and len(winning) < max_per_bucket:
            winning.append(s)
    print(f"  found {len(losing)} losing (<{low}) and "
          f"{len(winning)} winning (>{high}) positions; "
          f"searching each at {sims} sims ...")

    def analyze_all(states):
        rows = []
        for st in states:
            root = search_with_root(engine, mcts_mod, st, sims)
            if root is not None and root.children:
                rows.append(analyze_root(root))
        return rows

    losing_rows = analyze_all(losing)
    winning_rows = analyze_all(winning)

    print("\n" + "=" * 70)
    print("  SELECTION DIAGNOSTIC (observation 2)")
    print("=" * 70)
    dom_lose = _summarize_bucket(f"LOSING side (value < {low})", losing_rows)
    dom_win = _summarize_bucket(f"WINNING side (value > {high})", winning_rows)

    print("\n  Read (losing side is the one you care about):")
    verdict = {
        "compressed_Q": "value head can't resolve the actions -> a margin/"
                        "score-aware target is the fix; more sims won't help.",
        "flat_priors": "undertrained policy head -> more training iterations "
                       "sharpen priors and concentrate visits (your run helps).",
        "puct_diffuse": "priors are usable but PUCT isn't exploiting them -> "
                        "tune c_puct / FPU to concentrate on the best action.",
        "concentrating": "search is concentrating fine here — the indecision "
                         "may be cosmetic (near-equal actions).",
    }
    if dom_lose:
        print(f"    dominant cause on the losing side: {dom_lose}")
        print(f"    => {verdict[dom_lose]}")
    if dom_lose == "compressed_Q":
        print("    (To confirm it's resolution and not a truly-decided position,")
        print("     re-search those positions at a high sim count; if a real")
        print("     comeback line exists, deeper search opens the Q spread.)")
    print("=" * 70 + "\n")


# ============================================================
# Offline self-tests (no model needed)
# ============================================================

def _selftest():
    # entropy extremes
    assert abs(norm_entropy([1, 1, 1, 1]) - 1.0) < 1e-9
    assert norm_entropy([10, 0, 0, 0]) < 1e-9
    assert 0.4 < norm_entropy([5, 3, 1, 1]) < 0.95
    print("  norm_entropy: uniform=1, one-hot=0, mixed in between  OK")

    # classification of the three causes
    assert classify_position(0.01, 0.99, 0.99) == "compressed_Q"
    assert classify_position(0.20, 0.97, 0.95) == "flat_priors"
    assert classify_position(0.20, 0.97, 0.40) == "puct_diffuse"
    assert classify_position(0.20, 0.40, 0.40) == "concentrating"
    print("  classify_position: all four branches  OK")

    # position collector returns valid decision states
    pos = collect_decision_positions(200, random.Random(0))
    assert pos and all(get_valid_moves(p) for p in pos)
    print(f"  collector: {len(pos)} valid decision states  OK")

    # analyze_root on a mock root: best Q under-visited => misaligned
    class _Mock:
        def __init__(self, q, n, prior, flip=False):
            self.Q, self.N, self.prior, self.flip_from_parent = q, n, prior, flip
    class _Root:
        pass
    r = _Root()
    # action 0: best Q (0.12) but few visits; action 1: low Q, most visits
    r.children = {0: _Mock(0.12, 5, 0.25), 1: _Mock(0.03, 40, 0.25),
                  2: _Mock(0.04, 38, 0.25), 3: _Mock(0.03, 37, 0.25)}
    a = analyze_root(r)
    assert a["q_spread"] >= 0.08 and not a["aligned"], a
    assert a["visit_ent"] > 0.8          # visits diffuse (not concentrated)
    print(f"  analyze_root: detects best-Q action under-visited "
          f"(q_spread={a['q_spread']:.2f}, aligned={a['aligned']}, "
          f"visit_ent={a['visit_ent']:.2f})  OK")

    print("All selection-diagnostic self-tests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Diagnose obs 2 (visit spread on "
                                             "lopsided positions)")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--games", type=int, default=1500)
    ap.add_argument("--sims", type=int, default=1600)
    ap.add_argument("--low", type=float, default=0.05)
    ap.add_argument("--high", type=float, default=0.95)
    ap.add_argument("--per-bucket", type=int, default=25, dest="per_bucket")
    args = ap.parse_args()

    if args.model:
        run_diagnostic(args.model, args.device, args.games, args.sims,
                       args.low, args.high, args.per_bucket)
    else:
        _selftest()