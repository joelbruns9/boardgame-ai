"""Read back what ``/api/game_log`` wrote.

The log is only worth keeping if every line reloads into a real position, so
loading is a first-class function rather than a snippet each consumer rewrites.
Two consumers are expected and they want the same bytes:

* **self-play restarts** -- real human-play positions are a diversity source no
  amount of self-play generates on its own;
* **engine validation** -- the same positions replayed through the engine, with
  BGA's own published arithmetic (fetched separately from its archive log) as
  the oracle.

Lines are skipped rather than raised on: a log is appended to live during a
game, so a truncated final line is normal, and one bad row must not cost the
rest of the corpus. :func:`load_positions` reports what it skipped.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


class UnloggableState(ValueError):
    """The supplied wire does not rebuild into a position."""


class GameLogWriter:
    """Append-only per-table log with content dedup.

    Kept out of the HTTP layer so the behaviour that matters -- validate,
    dedupe, append -- is testable without a web client, and so any caller
    (a replay importer, a batch backfill) can write the same format.
    """

    __slots__ = ("_dir", "_last")

    def __init__(self, log_dir: str | Path):
        self._dir = Path(log_dir)
        self._last: dict[str, str] = {}

    @staticmethod
    def safe_table_id(table_id: Any) -> str:
        # Becomes a filename, so it may not carry separators or traversal.
        cleaned = "".join(
            c for c in str(table_id or "unknown") if c.isalnum() or c in "-_"
        )
        return cleaned or "unknown"

    def path_for(self, table_id: Any) -> Path:
        return self._dir / f"table_{self.safe_table_id(table_id)}.jsonl"

    def append(
        self,
        adapter,
        *,
        table_id: Any = None,
        kind: str = "decision",
        state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and append one row. Returns the endpoint's response body."""

        if state is not None:
            try:
                adapter.state_from_wire(state)
            except Exception as exc:
                # A row that cannot be rebuilt is worse than a missing one: the
                # failure surfaces much later, in whatever consumes the log.
                raise UnloggableState(str(exc)) from exc

        table = self.safe_table_id(table_id)
        record = {"kind": kind, "table_id": table, "state": state, "extra": extra or {}}
        # Dedup on the payload that carries the content. For a position that is
        # ``state``, and ``extra`` is deliberately excluded: it holds things that
        # vary between re-posts of the same board (timings, what the advisor said
        # so far), which would defeat the dedup entirely. A stateless row -- a
        # notification-packet batch, say -- has its content in ``extra`` instead,
        # and keying such rows on ``state`` alone made every one of them a
        # duplicate of the last, so only the first was ever written.
        identity = {"kind": kind, "state": state}
        if state is None:
            identity["extra"] = record["extra"]
        key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if self._last.get(table) == key:
            # A streaming client re-posts the same position many times per turn.
            return {"ok": True, "appended": False, "reason": "duplicate"}

        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(table)
        record["logged_at"] = time.time()
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._last[table] = key
        return {"ok": True, "appended": True, "path": str(path)}


@dataclass(frozen=True, slots=True)
class LoggedPosition:
    """One logged row, with the state already rebuilt by the adapter."""

    table_id: str
    kind: str
    state: Any
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadReport:
    """What a load found, so a caller never silently trains on half a corpus."""

    files: int = 0
    lines: int = 0
    loaded: int = 0
    skipped_unparseable: int = 0
    skipped_no_state: int = 0
    skipped_unloadable: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.loaded} positions from {self.files} file(s) / {self.lines} lines; "
            f"skipped {self.skipped_unparseable} unparseable, "
            f"{self.skipped_no_state} stateless, "
            f"{self.skipped_unloadable} unloadable"
        )


def log_dir_for(game_id: str, root: str | Path = "runs") -> Path:
    return Path(root) / game_id / "bga_game_log"


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed rows, skipping any line that is not valid JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_positions(
    adapter,
    paths: str | Path | list[str | Path],
    *,
    kinds: tuple[str, ...] | None = ("decision",),
) -> tuple[list[LoggedPosition], LoadReport]:
    """Load every logged position under ``paths`` through ``adapter``.

    ``paths`` may be a directory, a single file, or a list of either.
    ``kinds=None`` keeps every row; the default keeps decision points, which is
    what a restart corpus wants (a terminal row has no move to make).
    """

    if not isinstance(paths, list):
        paths = [paths]
    files: list[Path] = []
    for entry in paths:
        entry = Path(entry)
        if entry.is_dir():
            files.extend(sorted(entry.glob("*.jsonl")))
        elif entry.is_file():
            files.append(entry)

    report = LoadReport(files=len(files))
    out: list[LoggedPosition] = []
    for path in files:
        for row in iter_rows(path):
            report.lines += 1
            if kinds is not None and row.get("kind") not in kinds:
                continue
            wire = row.get("state")
            if wire is None:
                report.skipped_no_state += 1
                continue
            try:
                state = adapter.state_from_wire(wire)
            except Exception as exc:  # a stale codec, an expansion, a bad row
                report.skipped_unloadable += 1
                if len(report.errors) < 10:
                    report.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            out.append(
                LoggedPosition(
                    table_id=str(row.get("table_id") or "unknown"),
                    kind=str(row.get("kind") or "decision"),
                    state=state,
                    extra=dict(row.get("extra") or {}),
                )
            )
            report.loaded += 1
    return out, report
