"""Validate schema-v2 RSS stability over the tail of a cloud training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(run_dir: Path):
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return manifest.get("iterations", [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tail", type=int, default=30)
    parser.add_argument("--max-drift", type=float, default=0.05)
    parser.add_argument("--minimum-iterations", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = _rows(args.run_dir)
    if len(rows) < args.minimum_iterations:
        raise SystemExit(
            f"need {args.minimum_iterations} completed iterations, found {len(rows)}"
        )
    tail = rows[-args.tail :]
    rss = [
        int(row["stats"]["resources"]["peak_rss_bytes"])
        for row in tail
        if row.get("stats", {}).get("resources", {}).get("peak_rss_bytes")
    ]
    if len(rss) != len(tail):
        raise SystemExit("tail contains rows without schema-v2 peak RSS")
    n = len(rss)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(rss) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, rss)) / denominator
    fitted_change = slope * (n - 1)
    relative_drift = abs(fitted_change) / mean_y
    result = {
        "iterations": [int(tail[0]["iteration"]), int(tail[-1]["iteration"])],
        "samples": n,
        "mean_rss_bytes": mean_y,
        "fitted_change_bytes": fitted_change,
        "relative_drift": relative_drift,
        "limit": args.max_drift,
        "passed": relative_drift <= args.max_drift,
    }
    output = args.output or args.run_dir / "memory_stability.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
