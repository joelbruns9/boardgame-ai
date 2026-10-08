"""Are CUDA graphs replaying? Read back from a run's ``training_log.jsonl``.

run07 passed ``--cuda-graphs`` and ran EAGER the whole run: captures failed
under concurrent shard threads and the wrapper disabled itself without a trace.
The adapter now records ``graph_*`` counts in ``rust_boundary``; the launcher
reads them twice -- after the stage-9 smoke, and from the real run's first
iterations -- and both readers live here, so they cannot drift apart again.

Review of d8aa2e3, finding 1: both readers looked at
``generation_performance.rust_boundary``. Under the soft-gate controller (the
default, and every launch) the loop's stats sit one level down, at
``generation_performance.performance.rust_boundary``, so both always saw
nothing -- the smoke passed with a "cannot confirm" warning and the watcher
read zero replays and, with ``GRAPH_GUARD=stop``, would have killed a healthy
run. Missing counters are now UNAVAILABLE evidence, never a failure verdict.

    python -m games.seven_wonders_duel.graph_check smoke LOG
    python -m games.seven_wonders_duel.graph_check watch LOG PID GUARD OFFSET
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

COUNTERS = ("graph_replays", "graph_eager_calls", "graph_captures", "graph_capture_failures")
MAX_EAGER_SHARE = 0.05

OK, FAILED, UNAVAILABLE = "OK", "NOT REPLAYING", "UNAVAILABLE"


def boundary_counters(row: dict[str, Any]) -> dict[str, int] | None:
    """The row's graph counters, or None when the row carries none.

    Soft-gate controller rows nest the loop's generation stats under
    ``performance``; strict-gate rows store them directly. Both are read.
    """

    reported = row.get("generation_performance") or {}
    for source in (reported.get("performance") or {}, reported):
        boundary = source.get("rust_boundary") if isinstance(source, dict) else None
        if isinstance(boundary, dict) and any(key in boundary for key in COUNTERS):
            return {key: int(boundary.get(key, 0) or 0) for key in COUNTERS}
    return None


def verdict(counters: dict[str, int] | None) -> str:
    if counters is None:
        return UNAVAILABLE
    replays = counters["graph_replays"]
    eager = counters["graph_eager_calls"]
    share = eager / max(replays + eager, 1)
    if counters["graph_capture_failures"] or replays == 0 or share > MAX_EAGER_SHARE:
        return FAILED
    return OK


def describe(counters: dict[str, int] | None) -> str:
    if counters is None:
        return "no graph counters in this row"
    replays, eager = counters["graph_replays"], counters["graph_eager_calls"]
    share = eager / max(replays + eager, 1)
    return (
        f"replays={replays} eager={eager} ({share:.1%}) "
        f"captures={counters['graph_captures']} "
        f"capture_failures={counters['graph_capture_failures']}"
    )


class LogTail:
    """Complete, newline-terminated rows appended after a byte offset.

    The run appends large rows to the file this reads, with no lock, so the
    final line can be half-written (review finding 4). Only bytes up to the
    last newline are consumed; the rest is retried on the next read. A
    COMPLETE line that does not parse is reported and skipped -- it is not
    evidence either way.
    """

    def __init__(self, path: Path, offset: int = 0):
        self.path = Path(path)
        self.offset = int(offset)
        self.malformed = 0

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self.offset += end + 1
        rows = []
        for line in data[: end + 1].splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.malformed += 1
                print(f"WARNING: skipping a malformed training-log row: {exc}", flush=True)
        return rows


def last_iteration(path: Path, offset: int) -> int:
    """The highest iteration committed in the first ``offset`` bytes, or -1."""

    path = Path(path)
    if offset <= 0 or not path.is_file():
        return -1
    with path.open("rb") as handle:
        data = handle.read(offset)
    best = -1
    for line in data.splitlines():
        try:
            best = max(best, int(json.loads(line)["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue
    return best


def smoke(path: Path) -> int:
    """0 when the smoke replayed graphs; 1 when it did not or cannot be told."""

    rows = LogTail(path).read()
    seen = [(row.get("iteration"), boundary_counters(row)) for row in rows]
    for iteration, counters in seen:
        print(f"smoke iteration {iteration}: {verdict(counters)} {describe(counters)}")
    verdicts = [verdict(counters) for _, counters in seen]
    if not rows:
        print("FAILED: the smoke wrote no training-log rows")
        return 1
    if FAILED in verdicts:
        return 1
    if OK not in verdicts:
        # The smoke runs the Rust generator with --cuda-graphs, so the
        # counters must be there. Their absence is the run07 failure mode
        # (no evidence) and is not let through.
        print("FAILED: no graph counters in the smoke log; cannot confirm replay")
        return 1
    return 0


def watch(
    path: Path,
    pid: int,
    guard: str,
    offset: int,
    *,
    rows_wanted: int = 3,
    poll_seconds: float = 60.0,
    deadline_seconds: float = 24 * 3600,
    alive: Callable[[int], bool] | None = None,
    terminate: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Check the first ``rows_wanted`` iterations the launched process commits.

    ``offset`` is the log's size before launch (review finding 3): rows before
    it belong to an earlier process and say nothing about this one. Rows after
    it whose iteration is not past the old maximum are also historical -- the
    startup backfill (`_sync_training_log`) can append those.
    """

    def _alive(target: int) -> bool:
        try:
            os.kill(target, 0)
            return True
        except OSError:
            return False

    alive = alive or _alive
    terminate = terminate or (lambda target: os.kill(target, signal.SIGTERM))
    tail = LogTail(path, offset)
    floor = last_iteration(path, offset)
    deadline = clock() + deadline_seconds
    checked = confirmed = 0
    if floor >= 0:
        print(f"resume: ignoring iterations <= {floor} (before this launch)", flush=True)
    while checked < rows_wanted and clock() < deadline and alive(pid):
        try:
            rows = tail.read()
        except OSError as exc:
            print(f"WARNING: could not read {path}: {exc}; retrying", flush=True)
            rows = []
        for row in rows:
            try:
                iteration = int(row.get("iteration"))
            except (TypeError, ValueError):
                continue
            if iteration <= floor or checked >= rows_wanted:
                continue
            counters = boundary_counters(row)
            result = verdict(counters)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} iteration {iteration}: "
                  f"{result} {describe(counters)}", flush=True)
            checked += 1
            if result == OK:
                confirmed += 1
            elif result == FAILED:
                print("WARNING: CUDA graphs are not replaying on the real run; generation "
                      "runs eager (slower, same training).", flush=True)
                if guard == "stop":
                    print(f"GRAPH_GUARD=stop: sending SIGTERM to pid {pid}", flush=True)
                    terminate(pid)
                return 1
        if checked < rows_wanted:
            sleep(poll_seconds)
    print(f"graph check finished: {checked} iteration(s) checked, {confirmed} confirmed "
          f"replaying, {checked - confirmed} without counters", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("smoke")
    one.add_argument("log", type=Path)
    run = sub.add_parser("watch")
    run.add_argument("log", type=Path)
    run.add_argument("pid", type=int)
    run.add_argument("guard", choices=("warn", "stop"))
    run.add_argument("offset", type=int)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return smoke(args.log)
    return watch(args.log, args.pid, args.guard, args.offset)


if __name__ == "__main__":
    sys.exit(main())
