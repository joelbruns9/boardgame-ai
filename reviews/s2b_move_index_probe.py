"""Measure which move indices S2b reanalysis actually re-searches.

Runs the production selection (`reanalysis_candidates` + `cap_reanalysis`)
over a replay window and compares the move-index distribution of the
selected positions with that of all candidates.

    python reviews/s2b_move_index_probe.py RUN_DIR/buffers --first 0 --last 8 \
        --budget 12144 [--ignore-policy-excluded]

`--budget` is the cap in positions (run08 log: "N general examples" when the
cap binds). `--ignore-policy-excluded` approximates an every-move-full run on
a buffer that searched most moves cheaply (run07).
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.seven_wonders_duel.buffer import read_records  # noqa: E402
from games.seven_wonders_duel.specialist import (  # noqa: E402
    DEFAULT_REANALYSIS_GAP,
    cap_reanalysis,
    reanalysis_candidates,
)

BUCKETS = [(0, 8), (8, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 200)]


def bucket(i: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= i < hi:
            return f"{lo:>3}-{hi - 1:<3}"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("buffers", type=Path)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--gap", type=float, default=DEFAULT_REANALYSIS_GAP)
    p.add_argument("--ignore-policy-excluded", action="store_true")
    p.add_argument("--seeded", action="store_true", help="the fixed (rng) selection")
    a = p.parse_args()

    records = []
    newest = []
    for it in range(a.first, a.last + 1):
        recs = read_records(a.buffers / f"iter_{it:04d}.jsonl")
        if a.ignore_policy_excluded:
            recs = [
                dataclasses.replace(
                    r,
                    moves=tuple(
                        dataclasses.replace(m, policy_excluded=False)
                        if getattr(m, "search_lambda", 0.0)
                        else m
                        for m in r.moves
                    ),
                )
                for r in recs
            ]
        records.extend(recs)
        newest.extend([it == a.last] * len(recs))

    selected = [reanalysis_candidates(r, min_gap=a.gap) for r in records]
    import random
    kept = cap_reanalysis(
        selected, a.budget, 1.0,
        rng=random.Random(f"s2b-cap:0:{a.last}") if a.seeded else None,
    )

    cand_games = sum(1 for s in selected if s)
    n_cand = sum(len(s) for s in selected)
    n_kept = sum(len(k) for k in kept)
    print(f"window iters {a.first}..{a.last}: {len(records)} games, "
          f"{cand_games} with candidates, {n_cand} candidates, {n_kept} kept "
          f"(budget {a.budget}); {n_cand / max(cand_games, 1):.1f} cand/game, "
          f"{n_kept / max(cand_games, 1):.1f} kept/game")
    kept_newest = sum(len(k) for k, nw in zip(kept, newest) if nw)
    cand_newest = sum(len(s) for s, nw in zip(selected, newest) if nw)
    print(f"newest iteration: {cand_newest} candidates, {kept_newest} kept "
          f"({kept_newest / max(cand_newest, 1):.0%} coverage)")

    lengths = Counter(len(r.moves) for r, s in zip(records, selected) if s)
    mean_len = sum(k * v for k, v in lengths.items()) / max(sum(lengths.values()), 1)
    print(f"mean game length (games with candidates): {mean_len:.1f} plies\n")

    c_all = Counter(bucket(i) for s in selected for i in s)
    c_kept = Counter(bucket(i) for k in kept for i in k)
    print(f"{'moves':>8} {'cands':>8} {'cand%':>6} {'kept':>8} {'kept%':>6} {'coverage':>9}")
    for lo, hi in BUCKETS:
        b = f"{lo:>3}-{hi - 1:<3}"
        c, k = c_all.get(b, 0), c_kept.get(b, 0)
        print(f"{b:>8} {c:>8} {c / max(n_cand, 1):>6.1%} {k:>8} "
              f"{k / max(n_kept, 1):>6.1%} {k / max(c, 1):>9.1%}")
    kept_idx = sorted(i for k in kept for i in k)
    cand_idx = sorted(i for s in selected for i in s)
    if kept_idx:
        print(f"\nmedian move index: candidates {cand_idx[len(cand_idx) // 2]}, "
              f"kept {kept_idx[len(kept_idx) // 2]}; max kept {kept_idx[-1]}")


if __name__ == "__main__":
    main()
