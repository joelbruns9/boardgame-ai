"""G2b phase-out census: how often does tactic relabelling change a target?

`MODEL_GROWTH_PLAN.md` (G2b, owner 2026-10-05): G2b is a bridge. With exact
tactics in self-play (G4 + the proven-loss guard) the search should produce
clean targets natively, so on the first new run's buffers measure how often G2b
still changes one -- near zero means remove it, otherwise keep it.

Derives each buffer twice through the production Rust path, tactic labels off
and on, pairs the rows by `(game, move)` and counts:

* `labelled` -- rows where `classify_actions` found anything (a forced win, a
  proven-losing move, or every move losing). Not the phase-out number: a
  labelled row whose search already put no mass on the losing moves is
  unchanged.
* `policy_changed` -- policy rows whose move target moved by total variation
  >= `--tv` (default 0.01): search put real mass where G2b removes it.
* `value_changed` -- rows whose exact value (`solver_value`) G2b set or changed.

Laptop, seconds per buffer::

    python -m games.seven_wonders_duel.g2b_census \\
        runs/seven_wonders_duel/run08/buffers/iter_0003.jsonl [more buffers...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .buffer import read_records
from .dataset import TACTIC_BLOCK, TACTIC_LOSS, TACTIC_NONE, TACTIC_WIN, derive_records_rust

_CLASS = {TACTIC_WIN: "win", TACTIC_LOSS: "all_lose", TACTIC_BLOCK: "block"}


def _rows(records, *, tactic_labels: bool, retain: int) -> dict:
    out = {}
    for rows, _stats in derive_records_rust(
        records, retain_proofs_per_game=retain, tactic_labels=tactic_labels
    ):
        for row in rows:
            if row.move_index is None:
                continue
            out[(row.iteration, row.game_key, row.move_index)] = row
    return out


def census(records, *, retain: int = 4, tv: float = 0.01) -> dict:
    """The change counts for one set of records (see the module docstring)."""

    plain = _rows(records, tactic_labels=False, retain=retain)
    labelled = _rows(records, tactic_labels=True, retain=retain)
    if plain.keys() != labelled.keys():
        raise AssertionError("tactic labels changed WHICH rows exist, not just their targets")
    counts = {
        "games": len(records),
        "rows": len(labelled),
        "policy_rows": 0,
        "labelled": 0,
        "policy_changed": 0,
        "value_changed": 0,
        "by_class": {},
    }
    moved = []
    for key, after in labelled.items():
        before = plain[key]
        if after.has_policy:
            counts["policy_rows"] += 1
        tactic = int(getattr(after, "tactic", TACTIC_NONE))
        if tactic == TACTIC_NONE:
            continue
        counts["labelled"] += 1
        bucket = counts["by_class"].setdefault(
            _CLASS.get(tactic, str(tactic)),
            {"labelled": 0, "policy_changed": 0, "value_changed": 0},
        )
        bucket["labelled"] += 1
        if (
            after.has_policy
            and after.policy_target is not None
            and before.policy_target is not None
        ):
            distance = 0.5 * float(
                np.abs(
                    after.policy_target.astype(np.float64)
                    - before.policy_target.astype(np.float64)
                ).sum()
            )
            if distance >= tv:
                counts["policy_changed"] += 1
                bucket["policy_changed"] += 1
                moved.append(distance)
        if after.solver_value != before.solver_value:
            counts["value_changed"] += 1
            bucket["value_changed"] += 1
    rows = max(counts["rows"], 1)
    policy_rows = max(counts["policy_rows"], 1)
    counts["labelled_share"] = counts["labelled"] / rows
    counts["policy_changed_share_of_policy_rows"] = counts["policy_changed"] / policy_rows
    counts["value_changed_share"] = counts["value_changed"] / rows
    counts["policy_tv_mean_when_changed"] = float(np.mean(moved)) if moved else 0.0
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("buffers", nargs="+", type=Path)
    parser.add_argument("--retain-proofs-per-game", type=int, default=4)
    parser.add_argument("--tv", type=float, default=0.01,
                        help="smallest policy-target change counted (total variation)")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {}
    for path in args.buffers:
        records = read_records(path)
        if args.max_games:
            records = records[: args.max_games]
        result = census(records, retain=args.retain_proofs_per_game, tv=args.tv)
        report[str(path)] = result
        print(
            f"{path.name}: {result['games']} games, {result['rows']} rows | "
            f"labelled {result['labelled_share']:.2%} | policy target changed "
            f"{result['policy_changed_share_of_policy_rows']:.2%} of policy rows "
            f"(mean TV {result['policy_tv_mean_when_changed']:.3f}) | value changed "
            f"{result['value_changed_share']:.2%}",
            flush=True,
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
