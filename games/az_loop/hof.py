"""Checkpoint-format-agnostic Hall-of-Fame storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class HOFEntry:
    path: str
    sha256: str
    source: str
    iteration: int
    tag: str
    created_at_utc: str
    metadata: dict[str, Any]


class HallOfFame:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.index_path = self.directory / "hof_index.jsonl"

    def entries(self) -> list[HOFEntry]:
        if not self.index_path.exists():
            return []
        entries = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(HOFEntry(**json.loads(line)))
        return entries

    def add(
        self,
        checkpoint: str | Path,
        *,
        iteration: int,
        tag: str = "promoted",
        metadata: dict[str, Any] | None = None,
    ) -> HOFEntry:
        source = Path(checkpoint)
        if not source.is_file():
            raise FileNotFoundError(source)
        checksum = _sha256(source)
        for entry in self.entries():
            if entry.sha256 == checksum:
                return entry
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
        target = self.directory / (
            f"iter_{iteration:04d}_{safe_tag}_{checksum[:12]}{source.suffix}"
        )
        shutil.copy2(source, target)
        entry = HOFEntry(
            path=str(target.resolve()),
            sha256=_sha256(target),
            source=str(source.resolve()),
            iteration=iteration,
            tag=tag,
            created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metadata=dict(metadata or {}),
        )
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def sample(self, rng: random.Random, mode: str = "recency") -> HOFEntry | None:
        entries = self.entries()
        if not entries:
            return None
        if mode == "latest":
            return entries[-1]
        if mode == "uniform":
            return rng.choice(entries)
        if mode != "recency":
            raise ValueError(f"unknown HOF sampling mode: {mode}")
        weights = list(range(1, len(entries) + 1))
        return rng.choices(entries, weights=weights, k=1)[0]
