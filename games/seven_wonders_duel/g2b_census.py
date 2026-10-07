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
  >= `--tv` (default 0.01): search put real mass where G2b removes it. Counted
  PER TRAINED ROUTE (`general`, `specialist:N`): a specialist-owned row has no
  policy in the general's projection but G2b still rewrites the target its
  own model trains on (review of ebc70c0, #2).
* `value_changed` -- rows whose exact-value supervision G2b changed: the
  scalar (`solver_value`) OR its exactness (`solver_exact`). An unchanged +-1
  that turns exact switches the row from expectimax utility to categorical
  proof supervision (review of ebc70c0, #3). `value_changed_effective` drops
  the exactness-only rows already overridden by the certain-win rule.

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
from .dataset import (
    TACTIC_BLOCK,
    TACTIC_LOSS,
    TACTIC_NONE,
    TACTIC_WIN,
    derive_records_rust,
    project_examples,
)
from .specialist import GENERAL_ROUTE

_CLASS = {TACTIC_WIN: "win", TACTIC_LOSS: "all_lose", TACTIC_BLOCK: "block"}


def _derive(records, *, tactic_labels: bool, retain: int) -> list:
    """Per-record example lists, general derivation (the cached one)."""

    return [
        rows
        for rows, _stats in derive_records_rust(
            records, retain_proofs_per_game=retain, tactic_labels=tactic_labels
        )
    ]


def _routes(records) -> list[str]:
    routes = {GENERAL_ROUTE}
    for record in records:
        for move in record.moves:
            route = getattr(move, "target_route", None)
            if route:
                routes.add(route)
    return sorted(routes)


def _keyed(rows) -> dict:
    return {
        (row.iteration, row.game_key, row.move_index): row
        for row in rows
        if row.move_index is not None
    }


def _tv(a, b) -> float:
    return 0.5 * float(
        np.abs(a.astype(np.float64) - b.astype(np.float64)).sum()
    )


def census(records, *, retain: int = 4, tv: float = 0.01) -> dict:
    """The change counts for one set of records (see the module docstring)."""

    plain = _derive(records, tactic_labels=False, retain=retain)
    labelled = _derive(records, tactic_labels=True, retain=retain)
    counts = {
        "games": len(records),
        "rows": 0,
        "labelled": 0,
        "value_changed": 0,
        "value_exactness_only": 0,
        "value_changed_effective": 0,
        "by_class": {},
        "routes": {
            route: {"policy_rows": 0, "policy_changed": 0, "tv_sum": 0.0}
            for route in _routes(records)
        },
    }
    for record, before_rows, after_rows in zip(records, plain, labelled):
        before_by, after_by = _keyed(before_rows), _keyed(after_rows)
        if before_by.keys() != after_by.keys():
            raise AssertionError(
                "tactic labels changed WHICH rows exist, not just their targets"
            )
        counts["rows"] += len(after_by)
        changed_keys = set()
        for key, after in after_by.items():
            before = before_by[key]
            tactic = int(getattr(after, "tactic", TACTIC_NONE))
            if tactic == TACTIC_NONE:
                continue
            changed_keys.add(key)
            counts["labelled"] += 1
            bucket = counts["by_class"].setdefault(
                _CLASS.get(tactic, str(tactic)),
                {"labelled": 0, "value_changed": 0},
            )
            bucket["labelled"] += 1
            scalar = after.solver_value != before.solver_value
            exactness = bool(after.solver_exact) != bool(before.solver_exact)
            if scalar or exactness:
                counts["value_changed"] += 1
                bucket["value_changed"] += 1
                if not scalar:
                    counts["value_exactness_only"] += 1
                # A certain-win row's value target is the exact win whatever
                # the solver fields say, so exactness alone changes nothing.
                if scalar or not getattr(after, "certain_win", False):
                    counts["value_changed_effective"] += 1
        for route, tally in counts["routes"].items():
            before_view = _keyed(project_examples(before_rows, record, route))
            after_view = _keyed(project_examples(after_rows, record, route))
            for key, after in after_view.items():
                if not after.has_policy or after.policy_target is None:
                    continue
                tally["policy_rows"] += 1
                if key not in changed_keys:
                    continue
                before = before_view[key]
                if before.policy_target is None:
                    continue
                distance = _tv(after.policy_target, before.policy_target)
                if distance >= tv:
                    tally["policy_changed"] += 1
                    tally["tv_sum"] += distance
    rows = max(counts["rows"], 1)
    counts["labelled_share"] = counts["labelled"] / rows
    counts["value_changed_share"] = counts["value_changed"] / rows
    counts["value_changed_effective_share"] = counts["value_changed_effective"] / rows
    for tally in counts["routes"].values():
        tally["policy_changed_share"] = tally["policy_changed"] / max(tally["policy_rows"], 1)
        tally["policy_tv_mean_when_changed"] = (
            tally.pop("tv_sum") / tally["policy_changed"] if tally["policy_changed"] else 0.0
        )
    general = counts["routes"][GENERAL_ROUTE]
    # Kept for continuity with the run07 baseline (general route only).
    counts["policy_rows"] = general["policy_rows"]
    counts["policy_changed"] = general["policy_changed"]
    counts["policy_changed_share_of_policy_rows"] = general["policy_changed_share"]
    counts["policy_tv_mean_when_changed"] = general["policy_tv_mean_when_changed"]
    counts["policy_changed_any_route"] = sum(
        tally["policy_changed"] for tally in counts["routes"].values()
    )
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
            f"{result['value_changed_share']:.2%} "
            f"(effective {result['value_changed_effective_share']:.2%})",
            flush=True,
        )
        for route, tally in result["routes"].items():
            if route == GENERAL_ROUTE:
                continue
            print(
                f"    {route}: policy target changed in {tally['policy_changed']} of "
                f"{tally['policy_rows']} policy rows ({tally['policy_changed_share']:.2%})",
                flush=True,
            )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
