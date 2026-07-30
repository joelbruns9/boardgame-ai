"""W6.2: run the Rust/Python equivalence suite so it cannot pass vacuously.

The suite is the only thing standing between a cloud run and an engine that
disagrees with the one every measurement in the plan was taken on.  As specified
before this module existed it would have passed on a fresh box without executing
its two most important tests: ``pytest`` was absent from ``requirements.txt``,
the corpus tests skipped when ``runs/`` was missing, and ``runs/`` is gitignored.

This runner asserts what actually ran:

* zero skips -- a skipped equivalence test is a failed one for this purpose;
* every test in the committed manifest actually *ran*. Collection counts are not
  enough -- ``-k`` deselection happens after the collection hook, so a
  filtered-out test still looks collected. Deleting, renaming or filtering out a
  test is a smoke failure, not a quieter suite;
* the corpus really was exercised.

``-p no:randomly`` keeps ordering fixed and ``-W error`` promotes warnings, so a
deprecation that would become a silent behaviour change on the box fails here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pytest


SUITE = Path(__file__).with_name("test_rust_engine_equiv.py")
MANIFEST = Path(__file__).with_name("cloud_equivalence_manifest.json")


class _Outcomes:
    """Collect per-test outcomes; the terminal summary is not machine-readable."""

    def __init__(self) -> None:
        self.collected: list[str] = []
        self.passed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: dict[str, str] = {}

    def pytest_collection_modifyitems(self, items) -> None:
        self.collected = [item.nodeid for item in items]

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call" and not (
            report.when == "setup" and report.outcome == "skipped"
        ):
            if report.outcome != "failed":
                return
        if report.outcome == "passed":
            self.passed.add(report.nodeid)
        elif report.outcome == "skipped":
            reason = ""
            if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
                reason = report.longrepr[2]
            self.skipped[report.nodeid] = reason
        else:
            self.failed.add(report.nodeid)


def _node_names(nodeids) -> set[str]:
    return {nodeid.split("::", 1)[-1] for nodeid in nodeids}


def run(write_manifest: bool = False, extra: list[str] | None = None) -> int:
    outcomes = _Outcomes()
    arguments = [
        str(SUITE),
        "-p",
        "no:randomly",
        "-W",
        "error",
        "-q",
        *(extra or []),
    ]
    status = pytest.main(arguments, plugins=[outcomes])

    # What *ran*, not what was collected: `-k` deselection happens after the
    # collection hook, so a filtered-out test still looks collected.
    executed = _node_names(
        outcomes.passed | outcomes.failed | set(outcomes.skipped)
    )
    if write_manifest:
        MANIFEST.write_text(
            json.dumps({"tests": sorted(executed)}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST} with {len(executed)} tests")
        return 0

    problems: list[str] = []
    if status != 0:
        problems.append(f"pytest exited {status}")
    if outcomes.failed:
        problems.extend(f"failed: {name}" for name in sorted(outcomes.failed))
    for nodeid, reason in sorted(outcomes.skipped.items()):
        problems.append(f"skipped (a skipped equivalence test is a failure): {nodeid} -- {reason}")

    expected = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["tests"])
    missing = expected - executed
    if missing:
        problems.extend(
            f"in the manifest but did not run: {name}" for name in sorted(missing)
        )
    added = executed - expected
    if added:
        # Not a failure: new coverage is welcome, but it should be recorded.
        print(
            "note: not in the manifest (re-run with --write-manifest to record): "
            + ", ".join(sorted(added))
        )

    print(
        f"\nequivalence smoke: {len(outcomes.passed)} passed, "
        f"{len(outcomes.skipped)} skipped, {len(outcomes.failed)} failed, "
        f"{len(executed)} ran, {len(expected)} expected"
    )
    if problems:
        print("EQUIVALENCE SMOKE FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("equivalence smoke passed: engine parity verified, nothing skipped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="record the currently collected tests as the expected set",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="extra arguments forwarded to pytest",
    )
    args = parser.parse_args(argv)
    return run(write_manifest=args.write_manifest, extra=args.pytest_args)


if __name__ == "__main__":
    sys.exit(main())
