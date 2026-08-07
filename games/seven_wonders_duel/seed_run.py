#!/usr/bin/env python3
"""Start a new run from a previous run's weights and Hall of Fame.

`phase_d` has no "initialise from checkpoint" flag: it reads weights from
`<run-dir>/checkpoints/` and treats an empty buffer directory as a new run. So
seeding is a matter of putting the right files in the right places before the
first launch -- which is what this does, with the two corrections that are easy
to miss when doing it by hand.

WHY THE ITERATION HAS TO BE REWRITTEN. A checkpoint carries its promotion
iteration in `config['iteration']`, and `PhaseD.current_best_iteration()` reads
it back from the file rather than from run state. Copy cloud3's iteration-120
checkpoint into a fresh run and it will report 120 while the run is at 0, and
the self-anchor's lag arithmetic compares those two numbers. This rewrites it to
0 and leaves `model_state` and `encoder_signature` untouched.

WHICH HALL-OF-FAME ENTRIES TO CARRY. Not all of them. cloud3 kept
`hof_iter_0000` -- a near-bootstrap network -- in its pool for over a hundred
iterations, so 15% of every iteration's games were played against an opponent
the learner beat about 95% of the time. Those games spend the scarce per-game
outcome-label budget on foregone conclusions and pin their value targets near
+1. Seed only checkpoints strong enough to punish forgetting; `--hof-iterations`
takes the list explicitly so the weak ones cannot be included by accident.

`HallOfFame.sample(mode="recency")` weights entries by their POSITION in the
index, not by iteration number, so entries are written in ascending iteration
order and the newest keeps the most weight.

    python -m games.seven_wonders_duel.seed_run \\
        --from-run runs/seven_wonders_duel/cloud3 \\
        --to-run runs/seven_wonders_duel/cloud4 \\
        --checkpoint iter_0120_promoted_XXXX.pt \\
        --hof-iterations 65 80 120
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_weights(source: Path, run_dir: Path, *, dry_run: bool = False) -> None:
    """Install `source` as both current_best.pt and latest.pt, at iteration 0.

    Both names are written because the soft-gate controller continues the
    rolling learner from `latest.pt` while the gate compares against
    `current_best.pt`; seeding only one would have the run open by gating the
    seed against a randomly initialised network.
    """

    import torch

    checkpoints = run_dir / "checkpoints"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{source} has no config dict; refusing to guess")
    was = config.get("iteration")
    config["iteration"] = 0
    # cloud3's per-step training history rides along in config and is of no use
    # to a new run; dropping it keeps the seeded file readable.
    history = config.pop("history", None)

    print(f"  weights   : {source.name}")
    print(f"    iteration {was} -> 0" + ("  (dropped history)" if history else ""))
    print(f"    d_model={config.get('d_model')} layers={config.get('layers')} "
          f"heads={config.get('heads')} precision={config.get('precision')}")
    if dry_run:
        return
    checkpoints.mkdir(parents=True, exist_ok=True)
    for name in ("current_best.pt", "latest.pt"):
        torch.save(payload, checkpoints / name)
        print(f"    wrote {checkpoints / name}")


def seed_hof(
    source_run: Path,
    run_dir: Path,
    iterations: list[int],
    *,
    offset: int = 1000,
    dry_run: bool = False,
) -> None:
    """Copy selected HOF entries and rewrite the index for the new location.

    The index stores absolute resolved paths, so copying the checkpoints without
    rewriting it leaves every entry pointing at the old run -- which works right
    up until that directory is deleted.

    Seeded entries are renumbered by `offset` so their iterations cannot collide
    with the new run's own. Checkpoint FILES never collide -- the name carries a
    sha256 prefix, and `add()` dedupes on the full digest -- but `HOFEntry.name`
    and the `league_opponent_iteration` stat are the bare integer, so a seeded
    entry at 65 and a promotion at 65 would report identically and any later
    grouping by opponent iteration would merge two different opponents. The
    field is reporting-only (phase_d `:1373` and `:2198`), so renumbering is
    free, and 1000 is far enough above any real iteration to read as deliberate.
    """

    source_index = source_run / "hof" / "hof_index.jsonl"
    if not source_index.exists():
        raise FileNotFoundError(source_index)
    entries = [
        json.loads(line) for line in source_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = sorted(
        (e for e in entries if e["iteration"] in iterations),
        key=lambda e: e["iteration"],
    )
    missing = set(iterations) - {e["iteration"] for e in wanted}
    if missing:
        raise ValueError(
            f"no HOF entry for iteration(s) {sorted(missing)}; "
            f"available: {sorted({e['iteration'] for e in entries})}"
        )

    target_dir = run_dir / "hof"
    print(f"  hall of fame: {len(wanted)} of {len(entries)} entries "
          f"(skipping {sorted({e['iteration'] for e in entries} - set(iterations))})")
    if dry_run:
        for entry in wanted:
            print(f"    would copy iteration {entry['iteration']} -> "
                  f"{entry['iteration'] + offset}: {Path(entry['path']).name}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for entry in wanted:
        origin = Path(entry["path"])
        if not origin.is_file():
            # The index is authoritative about names, but a moved run breaks the
            # absolute path; fall back to the same basename beside the index.
            origin = source_run / "hof" / origin.name
        if not origin.is_file():
            raise FileNotFoundError(f"HOF checkpoint missing: {entry['path']}")
        # Rebuild the name from the RENUMBERED iteration, matching
        # HallOfFame.add()'s format, so the filename and the index agree. Keeping
        # cloud3's name would leave a file called iter_0065 indexed at 1065.
        safe_tag = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in entry["tag"]
        )
        destination = target_dir / (
            f"iter_{entry['iteration'] + offset:04d}_{safe_tag}_"
            f"{entry['sha256'][:12]}{origin.suffix}"
        )
        shutil.copy2(origin, destination)
        checksum = _sha256(destination)
        if checksum != entry["sha256"]:
            raise RuntimeError(f"checksum changed copying {origin}")
        lines.append(
            json.dumps(
                {
                    **entry,
                    "path": str(destination.resolve()),
                    "source": entry["path"],
                    "iteration": entry["iteration"] + offset,
                    "metadata": {
                        **entry.get("metadata", {}),
                        "seeded_from": str(source_run),
                        "seeded_at_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    },
                },
                sort_keys=True,
            )
        )
        print(f"    iteration {entry['iteration']:>4} -> "
              f"{entry['iteration'] + offset} ({destination.name})")
    (target_dir / "hof_index.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"    wrote {target_dir / 'hof_index.jsonl'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--to-run", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="filename inside <from-run>/checkpoints to seed the weights from",
    )
    parser.add_argument(
        "--hof-iterations",
        type=int,
        nargs="*",
        default=[],
        help="iterations whose HOF entries to carry over. Omit for none. Choose "
        "only checkpoints strong enough to be worth playing against.",
    )
    parser.add_argument(
        "--hof-iteration-offset",
        type=int,
        default=1000,
        help="added to seeded HOF iteration numbers so they cannot be confused "
        "with the new run's own. Files never collide (the name carries a sha "
        "prefix), but log lines and the league_opponent_iteration stat are the "
        "bare integer.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.from_run / "checkpoints" / args.checkpoint
    if not source.is_file():
        raise FileNotFoundError(source)
    buffers = args.to_run / "buffers"
    if buffers.exists() and any(buffers.glob("iter_*.jsonl")):
        raise SystemExit(
            f"{buffers} already holds iteration files -- {args.to_run} is an "
            "established run and phase_d would resume it, not start fresh. "
            "Seeding now would overwrite its checkpoints. Refusing."
        )

    print(f"seeding {args.to_run} from {args.from_run}"
          + ("  [DRY RUN]" if args.dry_run else ""))
    seed_weights(source, args.to_run, dry_run=args.dry_run)
    if args.hof_iterations:
        seed_hof(
            args.from_run,
            args.to_run,
            args.hof_iterations,
            offset=args.hof_iteration_offset,
            dry_run=args.dry_run,
        )
    else:
        print("  hall of fame: none requested")
    print("\ndone. phase_d will now skip bootstrap (current_best.pt exists) and "
          "start at iteration 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
