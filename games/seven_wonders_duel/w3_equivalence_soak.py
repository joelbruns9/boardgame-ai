"""Cross-language encoding equivalence at scale, and control-channel sanity.

The W3 build rests on one claim: Python and Rust encode the same position
identically, control channels included. That was checked on 205 rows. This runs
it over thousands of games and, unlike the unit gate, reports WHERE a difference
lands -- a mismatch confined to the control columns is a different bug from one
in the pre-existing features, and the fix differs accordingly.

It also answers what a pure equivalence check cannot: masked positions compare
equal for the wrong reason. Two encoders that both wrongly mask everything agree
perfectly. So the run reports the valid/masked split, the distribution of
reachability, and asserts the features are actually populated.

    python -m games.seven_wonders_duel.w3_equivalence_soak --games 4000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--out", default="runs/seven_wonders_duel/w3_equivalence_soak.json")
    args = parser.parse_args(argv)

    from .buffer import GameRecorder
    from .codec import legal_action_indices
    from .control_table import default_table, ensure_rust_table
    from .dataset import derive_records_rust, examples_from_record
    from .encoder import CONTROL_FEATURES, GLOBAL_FEATURES, TABLEAU_FEATURES, TokenType
    from .game import Phase

    table = default_table()
    print(f"table {table.manifest['content_digest'][:12]} "
          f"rule {table.manifest['rule_identity'][:12]}", flush=True)
    print("rust install:", ensure_rust_table(), flush=True)

    valid_index = GLOBAL_FEATURES.index("control_valid")
    control_start = TABLEAU_FEATURES.index(CONTROL_FEATURES[0])
    control_stop = control_start + len(CONTROL_FEATURES)

    rng = random.Random(args.seed)
    stats = Counter()
    reach_hist = Counter()
    mismatch_columns = Counter()
    examples_of_mismatch = []
    started = time.perf_counter()

    for base in range(0, args.games, args.batch):
        records = []
        for offset in range(min(args.batch, args.games - base)):
            recorder = GameRecorder(args.seed + base + offset,
                                    agents={"p0": "soak", "p1": "soak"})
            local = random.Random(args.seed * 7 + base + offset)
            while recorder.game.phase is not Phase.COMPLETE:
                choice = local.choice(legal_action_indices(recorder.game))
                recorder.play(choice, policy_target={choice: 1.0})
            records.append(recorder.finish())

        rust = derive_records_rust(records)
        for record, (rust_rows, _stats) in zip(records, rust):
            python_rows = examples_from_record(record)
            if len(python_rows) != len(rust_rows):
                stats["row_count_mismatch"] += 1
                continue
            for py, rs in zip(python_rows, rust_rows):
                stats["rows"] += 1
                if not np.array_equal(py.type_ids, rs.type_ids):
                    stats["token_id_mismatch"] += 1
                    continue
                a = np.asarray(py.features, dtype=np.float64)
                b = np.asarray(rs.features, dtype=np.float64)
                if a.shape != b.shape:
                    stats["shape_mismatch"] += 1
                    continue
                bad = np.argwhere(np.abs(a - b) > 1e-9)
                if bad.size:
                    stats["feature_mismatch"] += 1
                    for _row, column in bad[:20]:
                        mismatch_columns[int(column)] += 1
                    if len(examples_of_mismatch) < 5:
                        examples_of_mismatch.append(
                            {"table": record.table_id if hasattr(record, "table_id")
                             else None, "columns": sorted({int(c) for _r, c in bad})}
                        )

                # Populated, not merely equal: two encoders that both mask
                # everything agree perfectly and are both useless.
                types = np.asarray(py.type_ids)
                global_row = int(np.argmax(types == 0))
                is_valid = a[global_row][valid_index] == 1.0
                stats["valid" if is_valid else "masked"] += 1
                tableau = np.where(types == 2)[0]
                if is_valid and tableau.size:
                    window = a[tableau, control_start:control_stop]
                    reach_hist[int(window[:, 0].sum())] += 1
                    if not window.any():
                        stats["valid_but_all_zero"] += 1
                elif tableau.size and a[tableau, control_start:control_stop].any():
                    stats["masked_but_nonzero"] += 1

        done = min(base + args.batch, args.games)
        print(f"  {done}/{args.games} games, {stats['rows']} rows, "
              f"{stats['feature_mismatch']} mismatches "
              f"({(time.perf_counter() - started) / 60:.1f} min)", flush=True)

    elapsed = time.perf_counter() - started
    report = {
        "harness": "w3_equivalence_soak",
        "games": args.games,
        "seed": args.seed,
        "table_digest": table.manifest["content_digest"],
        "rule_identity": table.manifest["rule_identity"],
        "minutes": round(elapsed / 60, 2),
        "counts": dict(stats),
        "valid_fraction": (round(stats["valid"] / max(stats["rows"], 1), 4)),
        "mismatch_columns": dict(mismatch_columns),
        "mismatch_examples": examples_of_mismatch,
        "reachable_slots_histogram": dict(sorted(reach_hist.items())),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in
                      ("games", "minutes", "counts", "valid_fraction")}, indent=2))
    print(f"wrote {out}")
    failed = any(stats[k] for k in (
        "row_count_mismatch", "token_id_mismatch", "shape_mismatch",
        "feature_mismatch", "masked_but_nonzero", "valid_but_all_zero",
    ))
    if failed:
        print("EQUIVALENCE FAILED", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
