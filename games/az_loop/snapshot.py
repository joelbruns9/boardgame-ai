"""W6.7: a consistent, minimally-sized copy of a live run.

A run directory is written by several hands at different moments -- the pending
journal and ``_recovery/`` during an iteration, a buffer file when generation
ends, the manifest when the row commits.  Copying it while an iteration is in
flight can therefore capture a buffer whose row does not exist, or rolling
checkpoints that disagree with the manifest's hashes, and the resume that
follows fails on a hash mismatch rather than at the point of the mistake.

This waits for an iteration boundary, copies the minimal resumable set, and
writes the manifest **last** so a snapshot is never newer than the rows that
describe it.  Per-iteration buffers are immutable once written, so repeat
snapshots only copy what is new.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time


MANIFEST = "run_manifest.json"
PENDING = "pending_iteration.json"


def iteration_in_flight(run_dir: Path) -> bool:
    """True when an iteration has begun mutating the rolling checkpoints."""

    return (run_dir / "checkpoints" / PENDING).exists() or (
        run_dir / PENDING
    ).exists()


def wait_for_boundary(run_dir: Path, timeout: float, poll: float = 5.0) -> bool:
    """Block until no iteration is in flight. False on timeout."""

    deadline = time.monotonic() + timeout
    while iteration_in_flight(run_dir):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True


def resumable_set(run_dir: Path) -> list[Path]:
    """Every path a resume needs, manifest last.

    Deliberately excludes ``_recovery/`` and the pending marker: they describe an
    interrupted iteration on the *source* box, and copying them would make the
    destination reconcile a rollback that already happened here.
    """

    paths: list[Path] = []
    checkpoints = run_dir / "checkpoints"
    for name in ("latest.pt", "current_best.pt"):
        candidate = checkpoints / name
        if candidate.is_file():
            paths.append(candidate)
    for extra in sorted(checkpoints.glob("optimizer_*.pt")):
        paths.append(extra)
    buffers = run_dir / "buffers"
    if buffers.is_dir():
        paths.extend(sorted(buffers.glob("*.jsonl")))
    for name in ("training_log.jsonl", "heartbeat.log"):
        candidate = run_dir / name
        if candidate.is_file():
            paths.append(candidate)
    elo = run_dir / "elo"
    if elo.is_dir():
        paths.extend(sorted(elo.glob("*.jsonl")))
    manifest = run_dir / MANIFEST
    if manifest.is_file():
        paths.append(manifest)  # last: never newer than the rows it describes
    return paths


def snapshot(
    run_dir: Path, destination: Path, *, timeout: float = 1800.0
) -> dict[str, object]:
    if not (run_dir / MANIFEST).is_file():
        raise FileNotFoundError(f"no {MANIFEST} under {run_dir}")
    if not wait_for_boundary(run_dir, timeout):
        raise TimeoutError(
            f"an iteration was still in flight after {timeout:.0f}s; a snapshot "
            "taken now could not be resumed. Wait for the boundary or stop the run."
        )
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    written_bytes = 0
    for path in resumable_set(run_dir):
        target = destination / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            target.exists()
            and target.stat().st_size == path.stat().st_size
            and target.stat().st_mtime >= path.stat().st_mtime
        ):
            # Per-iteration buffers are immutable, so an incremental pull is the
            # normal case rather than an optimisation.
            skipped += 1
            continue
        shutil.copy2(path, target)
        copied += 1
        written_bytes += path.stat().st_size
    rows = json.loads((run_dir / MANIFEST).read_text(encoding="utf-8")).get(
        "iterations", []
    )
    return {
        "run_dir": str(run_dir),
        "destination": str(destination),
        "iterations": len(rows),
        "last_iteration": rows[-1]["iteration"] if rows else None,
        "files_copied": copied,
        "files_already_current": skipped,
        "bytes_copied": written_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for an iteration boundary before giving up",
    )
    args = parser.parse_args(argv)
    report = snapshot(args.run_dir, args.destination, timeout=args.timeout)
    print(json.dumps(report, indent=2))
    print(
        f"\nsnapshot through iteration {report['last_iteration']}: "
        f"{report['files_copied']} files "
        f"({report['bytes_copied'] / 1e6:.1f} MB), "
        f"{report['files_already_current']} already current"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
