"""Search-frame trajectory + reveal-luck decomposition for a logged BGA game.

Part 1 — search-frame trajectory: every RELIABLE viewer-turn decision is
re-evaluated with the full advisor search (not the raw win head). The edge
(blended search value, actor frame) is the number the user actually plays
by; the approx win%% translation is linear inside |edge| <= 0.35 and labeled
compressed beyond it.

Part 2 — reveal luck: at chosen post-reveal rows, the actual revealed row is
compared against counterfactual rows sampled from the known bag (the scraper
logs hidden-deck identities by elimination). Reported per reveal:
  - luck percentile: where the ACTUAL row's evaluation sits in the
    counterfactual distribution (low = unlucky draw);
  - consistency check: mean counterfactual value vs the pre-reveal eval.
Root-inference values give the full distribution cheaply; the actual row and
a subsample are also search-evaluated to confirm the ranking isn't a
raw-head artifact.
"""
import argparse
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from games.kingdomino.web_app import (  # noqa: E402
    RecommendRequest,
    _load_nn_evaluator,
    _root_trajectory,
    recommend,
    state_from_debug_json,
)
from runs.kingdomino.bga_postmortem import segment_games  # noqa: E402


def edge_str(edge: float) -> str:
    if abs(edge) <= 0.35:
        return f"{edge:+.3f} (~{0.5 + edge:.0%})"
    return f"{edge:+.3f} ({'>~85%' if edge > 0 else '<~15%'}, compressed)"


def search_value(rec_state: dict, sims: int, device: str) -> float:
    req = RecommendRequest(engine="nn", state=rec_state, top_k=1,
                           nn_sims=sims, num_simulations=sims,
                           device=device, seed=0)
    out = recommend(req)
    v = out.get("value")
    if v is None:
        raise RuntimeError("search returned no root value")
    return float(v)  # actor frame (= viewer on reliable rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True)
    p.add_argument("--game", type=int, default=-1)
    p.add_argument("--sims", type=int, default=1600)
    p.add_argument("--luck_rows", default="",
                   help="comma-separated post-reveal row indices to luck-test")
    p.add_argument("--luck_samples", type=int, default=40)
    p.add_argument("--luck_search_subsample", type=int, default=10)
    p.add_argument("--luck_search_sims", type=int, default=800)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.log, encoding="utf-8") if l.strip()]
    g = segment_games(rows)[args.game]
    decisions = g["decisions"]
    viewer = next((d.get("viewer_id") for d in decisions if d.get("viewer_id")), None)

    class _NNReq:
        checkpoint_path = None
        device = args.device
        channels = blocks = bilinear_dim = None
        nn_sims = 50
    _, net, ckpt = _load_nn_evaluator(_NNReq())
    print(f"net: {ckpt}")

    # ── Part 1: search-frame trajectory over reliable viewer rows ──
    print(f"\nSearch-frame trajectory ({args.sims} sims/row):")
    print(f"{'#':>3} {'deck':>4} {'search edge':<26} {'raw head':>9}")
    reliable = []
    for i, rec in enumerate(decisions):
        if str(rec.get("active_player")) != str(viewer):
            continue
        if rec["state"].get("board_reconstruction_warning"):
            continue
        try:
            state = state_from_debug_json(rec["state"])
        except Exception:
            continue
        ri = _root_trajectory(net, state, args.device)
        try:
            v = search_value(rec["state"], args.sims, args.device)
        except Exception as e:
            print(f"{i:>3}  search failed: {e}")
            continue
        reliable.append((i, rec, v))
        print(f"{i:>3} {str(rec['state'].get('deck_count')):>4} "
              f"{edge_str(v):<26} {ri['win_prob']:>8.1%}")

    # ── Part 2: reveal luck at the requested rows ──
    luck_rows = [int(x) for x in args.luck_rows.split(",") if x.strip()]
    rng = random.Random(0)
    for lr in luck_rows:
        rec = decisions[lr]
        s = rec["state"]
        row_actual = sorted(s.get("current_row") or [])
        bag_after = sorted((s.get("debug") or {}).get("deck") or [])
        if not row_actual or not bag_after:
            print(f"\nrow {lr}: missing row/deck identities; cannot luck-test")
            continue
        pool = sorted(row_actual + bag_after)  # pre-reveal bag
        combos = list(itertools.combinations(pool, len(row_actual)))
        if len(combos) > args.luck_samples:
            keep = set()
            keep.add(tuple(row_actual))
            while len(keep) < args.luck_samples:
                keep.add(tuple(sorted(rng.sample(pool, len(row_actual)))))
            combos = sorted(keep)
        print(f"\n=== reveal luck at row {lr} (deck {s.get('deck_count')}): "
              f"actual row {row_actual} from bag of {len(pool)} "
              f"({len(combos)} counterfactuals) ===")

        vals = {}
        for combo in combos:
            variant = copy.deepcopy(s)
            variant["current_row"] = list(combo)
            variant["debug"] = dict(variant.get("debug") or {})
            variant["debug"]["deck"] = [t for t in pool if t not in combo]
            variant["deck_count"] = len(variant["debug"]["deck"])
            try:
                st = state_from_debug_json(variant)
                ri = _root_trajectory(net, st, args.device)
                # actor == viewer at these rows; win_prob is actor-frame
                vals[combo] = ri["win_prob"]
            except Exception:
                continue
        actual_key = tuple(row_actual)
        if actual_key not in vals:
            print("  actual row failed to evaluate; skipping")
            continue
        actual_v = vals[actual_key]
        dist = sorted(vals.values())
        pct = sum(1 for v in dist if v <= actual_v) / len(dist)
        import statistics
        print(f"  raw-head frame: actual reveal win%={actual_v:.1%}, "
              f"counterfactual mean={statistics.mean(dist):.1%} "
              f"(min {dist[0]:.1%}, max {dist[-1]:.1%})")
        print(f"  luck percentile: {pct:.0%} "
              f"({'unlucky' if pct < 0.25 else 'lucky' if pct > 0.75 else 'ordinary'})")

        # Search-frame confirmation on a subsample spanning the distribution.
        by_v = sorted(vals.items(), key=lambda kv: kv[1])
        idxs = [round(j * (len(by_v) - 1) / max(1, args.luck_search_subsample - 1))
                for j in range(args.luck_search_subsample)]
        sub = {by_v[j][0] for j in idxs} | {actual_key}
        searched = {}
        for combo in sub:
            variant = copy.deepcopy(s)
            variant["current_row"] = list(combo)
            variant["debug"] = dict(variant.get("debug") or {})
            variant["debug"]["deck"] = [t for t in pool if t not in combo]
            variant["deck_count"] = len(variant["debug"]["deck"])
            try:
                searched[combo] = search_value(variant, args.luck_search_sims,
                                               args.device)
            except Exception:
                continue
        if actual_key in searched:
            sv = sorted(searched.values())
            s_act = searched[actual_key]
            s_pct = sum(1 for v in sv if v <= s_act) / len(sv)
            print(f"  search frame ({args.luck_search_sims} sims, "
                  f"{len(searched)} rows): actual edge {edge_str(s_act)}, "
                  f"percentile {s_pct:.0%}, "
                  f"range [{sv[0]:+.3f}, {sv[-1]:+.3f}]")

    print("\nLUCK ANALYSIS DONE")


if __name__ == "__main__":
    main()
