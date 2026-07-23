"""Select the fastest eligible F4 Rust cloud sweep row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(root: Path, output: Path) -> dict:
    candidates = []
    for path in root.glob("**/summary.json"):
        if path == output:
            continue
        if path.parent.name in {"smoke", "confirmation", "diagnostic"}:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("mode") != "rust" or not row.get("eligible"):
            continue
        candidates.append(
            {
                "summary": str(path.resolve()),
                "games_per_second": row["rust_games_per_second_mean"],
                "manifest": row["manifest"],
            }
        )
    if not candidates:
        raise SystemExit("no eligible Rust cloud sweep summaries found")
    locked = candidates[0]["manifest"]
    for candidate in candidates[1:]:
        manifest = candidate["manifest"]
        for field in (
            "contract_schema_version",
            "contract_sha256",
            "quality_lock_sha256",
            "checkpoint_sha256",
        ):
            if manifest.get(field) != locked.get(field):
                raise SystemExit(f"cloud sweep mixes incompatible {field}")
    candidates.sort(key=lambda row: row["games_per_second"], reverse=True)
    result = {"winner": candidates[0], "ranked": candidates}
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.root, args.output), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
