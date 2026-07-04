"""Exact-endgame solver diagnostic harness.

Goal: characterize WHY the exact solver's tail is expensive (the ~20% that time
out at a 5-6s budget), and provide a persistent corpus so we can A/B solver
changes (ordering, transposition, aspiration) by node-count / solve-time RATIO.

Two modes:

  generate  Play games (random or score-greedy) with the CURRENT engine, snapshot
            every no-chance endgame root (deck in {0,4}), measure each with the
            production solver, and append it to a JSONL corpus. Every position is
            saved in the SAME field set as the training run's fallback sidecar
            (from_parts-compatible) plus its solver metrics, so harness-generated
            and real-run-tail positions are interchangeable and re-analyzable.

  analyze   Load a JSONL corpus (harness- OR sidecar-produced), optionally
            re-measure at a chosen budget/ordering, and print the tail report:
            solve-time distribution, timeout rate at several budgets, node-count
            distribution, and how hardness tracks position features (legal
            placements vs. legal picks — the placement-axis vs. draft-axis split,
            and board fullness). Also an optional value-equivalence guard (two
            orderings must return bit-identical solved values).

NOTE ON REALISM: random/greedy self-play positions are NOT the competent-play
tail. Use this to build tooling, validate value propagation, and measure RELATIVE
speedups (ratios transfer). Confirm the ABSOLUTE tail on faithful positions from
the run's `--exact_fallback_positions` sidecar (same format → `analyze` reads it).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

import kingdomino_rust as kr
from games.kingdomino.game import GameState, Phase
from games.kingdomino.dominoes import Terrain
from games.kingdomino.endgame_solver import _rust_state_from_python

EMPTY = int(Terrain.EMPTY)
# Bound the UNPRUNED tree-size count so a huge position can't hang the harness.
# NB: count_endgame_nodes_no_chance is the full minimax tree (no alpha-beta); it
# is the position's intrinsic complexity, NOT the solver's pruned work. The gap
# between this and `elapsed` (pruned solve) shows how effective pruning already is.
NODE_CAP = 50_000_000


# ── state <-> dict (from_parts field set; sidecar-compatible) ───────────────
def serialize(rs, st) -> dict:
    """Fields required to reconstruct via RustGameState.from_parts. Board arrays
    come from the RUST getters (layout-safe); castle/bonuses/discards from the
    Python state (no rust state-level getter)."""
    return {
        "deck": list(rs.deck()),
        "current_row": list(rs.current_row()),
        "pending_claims": [list(t) for t in rs.pending_claims()],
        "next_claims": [list(t) for t in rs.next_claims()],
        "phase": int(rs.phase),
        "actor_index": int(rs.actor_index),
        "initial_pick_count": int(rs.initial_pick_count),
        "start_player": int(rs.start_player),
        "board0_terrain": list(rs.board_terrain(0)),
        "board0_crowns": list(rs.board_crowns(0)),
        "board1_terrain": list(rs.board_terrain(1)),
        "board1_crowns": list(rs.board_crowns(1)),
        "castle_x": int(st.boards[0].castle_pos[0]),
        "castle_y": int(st.boards[0].castle_pos[1]),
        "harmony": bool(st.config.harmony),
        "middle_kingdom": bool(st.config.middle_kingdom),
        "discards": list(getattr(st, "discards", [0, 0])),
    }


def deserialize(d: dict):
    """dict -> RustGameState via from_parts. Tolerates the sidecar omitting
    `discards` (defaults to (0,0))."""
    dis = d.get("discards", [0, 0])
    return kr.RustGameState.from_parts(
        [int(x) for x in d["deck"]],
        [int(x) for x in d["current_row"]],
        [(int(a), int(b)) for a, b in d["pending_claims"]],
        [(int(a), int(b)) for a, b in d["next_claims"]],
        int(d["phase"]), int(d["actor_index"]),
        int(d["initial_pick_count"]), int(d["start_player"]),
        [int(x) for x in d["board0_terrain"]], [int(x) for x in d["board0_crowns"]],
        [int(x) for x in d["board1_terrain"]], [int(x) for x in d["board1_crowns"]],
        bool(d["harmony"]), bool(d["middle_kingdom"]),
        int(d["castle_x"]), int(d["castle_y"]),
        (int(dis[0]), int(dis[1])),
    )


# ── measurement ─────────────────────────────────────────────────────────────
PICK_AXIS_SIZE = 5  # joint = placement_idx * PICK_AXIS_SIZE + pick_idx


def features(rs) -> dict:
    """Cheap root position features. n_placements vs n_picks is the first-order
    placement-axis vs draft-axis branching split, decoded from joint indices."""
    legal = rs.legal_action_indices()  # Vec<u16> joint indices
    placements = {int(j) // PICK_AXIS_SIZE for j in legal}
    picks = {int(j) % PICK_AXIS_SIZE for j in legal}
    occ0 = sum(1 for v in rs.board_terrain(0) if int(v) != EMPTY)
    occ1 = sum(1 for v in rs.board_terrain(1) if int(v) != EMPTY)
    return {
        "deck_len": len(rs.deck()),
        "n_legal": len(legal),
        "n_placements": len(placements),
        "n_picks": len(picks),
        "occ0": occ0, "occ1": occ1,
    }


def measure(rs, budget: float, ordering: str) -> dict:
    value, solved, elapsed = rs.measure_endgame_tree(
        budget, 160.0, 2.0, 0.5, True, ordering)
    nodes = kr.count_endgame_nodes_no_chance(rs, NODE_CAP)
    # elapsed = PRUNED solve time (alpha-beta); nodes = UNPRUNED tree size.
    return {"value": value, "solved": bool(solved), "elapsed": elapsed,
            "nodes": int(nodes), "nodes_capped": int(nodes) >= NODE_CAP}


# ── generation ──────────────────────────────────────────────────────────────
def _greedy_action(st):
    """Score-greedy placement (fills boards efficiently, closer to competent
    geometry than random); selection chosen at random among the greedy-best."""
    best, best_score = [], None
    me = st.current_actor
    for a in st.legal_actions():
        nb = st.step(a)
        s = nb.boards[me].score(st.config.harmony, st.config.middle_kingdom).total
        if best_score is None or s > best_score:
            best_score, best = s, [a]
        elif s == best_score:
            best.append(a)
    return best


def generate(n_target: int, mode: str, budget: float, ordering: str,
             out_path: str, seed: int):
    rng = random.Random(seed)
    saved = 0
    game = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        while saved < n_target:
            st = GameState.new(seed=seed + game)
            game += 1
            r = random.Random(seed * 131 + game)
            seen_decklens = set()
            while st.phase != Phase.GAME_OVER:
                is_endgame = (
                    (st.phase == Phase.PLACE_AND_SELECT and len(st.deck) in (0, 4))
                    or (st.phase == Phase.FINAL_PLACEMENT and len(st.deck) == 0))
                # one snapshot per (game, deck_len) to avoid near-duplicate roots
                key = (st.phase, len(st.deck))
                if is_endgame and key not in seen_decklens and len(st.legal_actions()) >= 2:
                    seen_decklens.add(key)
                    rs = _rust_state_from_python(st)
                    rec = serialize(rs, st)
                    rec.update(features(rs))
                    rec.update(measure(rs, budget, ordering))
                    rec["source"] = f"{mode}_g{game}"
                    fh.write(json.dumps(rec) + "\n")
                    saved += 1
                    if saved % 25 == 0:
                        print(f"  saved {saved}/{n_target} "
                              f"(last: deck={rec['deck_len']} nodes={rec['nodes']} "
                              f"solved={rec['solved']} {rec['elapsed']:.2f}s)")
                    if saved >= n_target:
                        break
                if mode == "greedy":
                    cands = _greedy_action(st)
                    st = st.step(r.choice(cands))
                else:
                    st = st.step(r.choice(st.legal_actions()))
    print(f"wrote {saved} positions -> {out_path}")


# ── analysis ────────────────────────────────────────────────────────────────
def _pct(xs, p):
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs[0] if xs else 0)


def analyze(records, budgets):
    n = len(records)
    solved = [r for r in records if r["solved"]]
    elapsed = [r["elapsed"] for r in solved]
    nodes = [r["nodes"] for r in records]
    print("=" * 72)
    print(f"ENDGAME SOLVER TAIL REPORT  ({n} positions)")
    print("=" * 72)
    print(f"solved within measure budget: {len(solved)}/{n} "
          f"({100*len(solved)/n:.0f}%)")
    if elapsed:
        print(f"solve time (solved): median={_pct(elapsed,50):.3f}s "
              f"p90={_pct(elapsed,90):.3f}s p99={_pct(elapsed,99):.3f}s "
              f"max={max(elapsed):.3f}s")
    print(f"unpruned tree size: median={int(_pct(nodes,50)):,} p90={int(_pct(nodes,90)):,} "
          f"max={max(nodes):,}" + (" (some capped)" if any(r['nodes_capped'] for r in records) else ""))
    print("  (unpruned tree vs pruned solve time above = how much pruning already helps)")
    print("\ntimeout rate at hypothetical budgets (unsolved-at-measure count as timeout):")
    for b in budgets:
        to = sum(1 for r in records if (not r["solved"]) or r["elapsed"] > b)
        print(f"  {b:>5.1f}s : {100*to/n:5.1f}%  ({to}/{n})")

    # placement-axis vs draft-axis: which correlates with hardness?
    print("\nhardness signature — hardest 25% (by nodes) vs easiest 25%:")
    ranked = sorted(records, key=lambda r: r["nodes"])
    q = max(1, n // 4)
    easy, hard = ranked[:q], ranked[-q:]
    def avg(rs, k):
        return sum(r[k] for r in rs) / len(rs)
    for k in ("deck_len", "n_legal", "n_placements", "n_picks", "occ0", "occ1"):
        e, h = avg(easy, k), avg(hard, k)
        ratio = f"{h/e:5.2f}x" if e > 1e-6 else "   n/a"
        print(f"  {k:<13} easy={e:7.1f}   hard={h:7.1f}   ratio={ratio}")
    print("\n(n_placements >> n_picks driving hardness => placement-axis bound;\n"
          " n_picks tracking hardness => draft/selection-axis bound.)")


def value_equivalence_guard(records, budget, orderings):
    """Re-solve each position under two orderings; solved values must be identical."""
    a, b = orderings
    mism = 0
    checked = 0
    for r in records:
        rs = deserialize(r)
        va, sa, _ = rs.measure_endgame_tree(budget, 160.0, 2.0, 0.5, True, a)
        vb, sb, _ = rs.measure_endgame_tree(budget, 160.0, 2.0, 0.5, True, b)
        if sa and sb:
            checked += 1
            if va != vb:
                mism += 1
    print(f"\nvalue-equivalence guard [{a} vs {b}]: {mism} mismatches / {checked} "
          f"both-solved  -> {'PASS' if mism == 0 else 'FAIL'}")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="play games, snapshot+measure endgame roots")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--mode", choices=["random", "greedy"], default="greedy")
    g.add_argument("--budget", type=float, default=10.0, help="solver wall-clock per position")
    g.add_argument("--ordering", default="lookahead2_clustered")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=900000)

    a = sub.add_parser("analyze", help="report the tail of a saved corpus")
    a.add_argument("--in", dest="inp", required=True)
    a.add_argument("--budgets", default="1,2,5,10")
    a.add_argument("--value_guard", default="", help="two orderings 'a,b' to cross-check")
    a.add_argument("--guard_budget", type=float, default=10.0)

    args = p.parse_args()
    if args.cmd == "generate":
        generate(args.n, args.mode, args.budget, args.ordering, args.out, args.seed)
    elif args.cmd == "analyze":
        records = load(args.inp)
        analyze(records, [float(x) for x in args.budgets.split(",")])
        if args.value_guard:
            value_equivalence_guard(records, args.guard_budget,
                                    args.value_guard.split(","))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
