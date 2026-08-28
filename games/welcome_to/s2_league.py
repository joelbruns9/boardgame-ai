"""Durable, bounded-memory opponent league for Welcome To S2 generation.

Every outgoing promoted best remains in an append-only manifest. Generation
loads only the current best, a few recent archives, and a deterministic sample
of older hall-of-fame entries, so diversity grows without putting every model
on the GPU at once.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


LEAGUE_FORMAT = "welcome_to_s2_league"
LEAGUE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    current_best_weight: float = 0.60
    recent_weight: float = 0.30
    hall_of_fame_weight: float = 0.10
    recent_count: int = 3
    hall_of_fame_count: int = 2
    history_ramp_promotions: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        weights = (
            self.current_best_weight,
            self.recent_weight,
            self.hall_of_fame_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("league weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("league needs positive sampling mass")
        if self.current_best_weight <= 0.0:
            raise ValueError("current best must retain positive league weight")
        if (
            self.recent_count < 0
            or self.hall_of_fame_count < 0
            or self.history_ramp_promotions < 0
        ):
            raise ValueError("league checkpoint counts must be non-negative")


@dataclass(frozen=True, slots=True)
class LeagueEntry:
    path: str
    sha256: str
    archived_at_iteration: int
    tag: str


@dataclass(frozen=True, slots=True)
class OpponentSpec:
    name: str
    path: str
    sha256: str
    weight: float
    kind: str
    archived_at_iteration: Optional[int]


@dataclass(frozen=True, slots=True)
class LeagueSelection:
    iteration: int
    opponents: tuple[OpponentSpec, ...]

    def metrics(self) -> dict[str, Any]:
        total = sum(item.weight for item in self.opponents)
        return {
            "iteration": self.iteration,
            "opponents": [
                {
                    **asdict(item),
                    "normalized_weight": item.weight / total,
                }
                for item in self.opponents
            ],
        }


class S2League:
    """Atomic manifest of immutable historical best checkpoints."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    def entries(self) -> list[LeagueEntry]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != LEAGUE_FORMAT or int(
            payload.get("version", -1)
        ) != LEAGUE_VERSION:
            raise ValueError(f"unsupported league manifest {self.manifest_path}")
        entries = [LeagueEntry(**item) for item in payload.get("entries", [])]
        if len({entry.sha256 for entry in entries}) != len(entries):
            raise ValueError("league manifest repeats a checkpoint hash")
        return entries

    def _write(self, entries: list[LeagueEntry]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=self.manifest_path.name + ".",
                suffix=".tmp",
                dir=self.manifest_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {
                        "format": LEAGUE_FORMAT,
                        "version": LEAGUE_VERSION,
                        "entries": [asdict(entry) for entry in entries],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def register(
        self,
        checkpoint: str | Path,
        *,
        archived_at_iteration: int,
        tag: str = "promoted_best",
    ) -> LeagueEntry:
        """Idempotently append an already-archived immutable checkpoint."""
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if archived_at_iteration < 0:
            raise ValueError("archived iteration must be non-negative")
        checksum = sha256_file(path)
        entries = self.entries()
        for entry in entries:
            if entry.sha256 == checksum:
                if sha256_file(entry.path) != checksum:
                    raise ValueError(f"league checkpoint changed on disk: {entry.path}")
                return entry
        entry = LeagueEntry(
            path=str(path),
            sha256=checksum,
            archived_at_iteration=archived_at_iteration,
            tag=tag,
        )
        self._write([*entries, entry])
        return entry

    def select(
        self,
        current_best: str | Path,
        *,
        iteration: int,
        config: Optional[LeagueConfig] = None,
    ) -> LeagueSelection:
        """Select a resume-stable bounded opponent pool for one iteration."""
        if iteration < 0:
            raise ValueError("league iteration must be non-negative")
        config = config or LeagueConfig()
        current_path = Path(current_best).resolve()
        if not current_path.is_file():
            raise FileNotFoundError(current_path)
        current_sha = sha256_file(current_path)
        entries = []
        for entry in self.entries():
            path = Path(entry.path)
            if not path.is_file() or sha256_file(path) != entry.sha256:
                raise ValueError(f"league checkpoint is missing or changed: {path}")
            if entry.sha256 != current_sha:
                entries.append(entry)

        recent = entries[-config.recent_count :] if config.recent_count else []
        recent_hashes = {entry.sha256 for entry in recent}
        older = [entry for entry in entries if entry.sha256 not in recent_hashes]
        rng = random.Random(config.seed ^ (iteration * 100_003) ^ 0x5332_4C4541475545)
        hall_count = min(config.hall_of_fame_count, len(older))
        hall = rng.sample(older, hall_count) if hall_count else []

        ramp = (
            1.0
            if config.history_ramp_promotions == 0
            else min(1.0, len(entries) / config.history_ramp_promotions)
        )
        recent_mass = config.recent_weight * ramp if recent else 0.0
        hall_mass = config.hall_of_fame_weight * ramp if hall else 0.0
        # Until enough promotions exist, and whenever a category has no member,
        # its unused mass stays on current best instead of causing a sudden
        # renormalized jump toward the first weak archive.
        current_mass = (
            config.current_best_weight
            + config.recent_weight
            + config.hall_of_fame_weight
            - recent_mass
            - hall_mass
        )

        opponents = [
            OpponentSpec(
                name=f"current_best_{current_sha[:10]}",
                path=str(current_path),
                sha256=current_sha,
                weight=current_mass,
                kind="current_best",
                archived_at_iteration=None,
            )
        ]
        # Linear recency weights inside the recent category: the newest archive
        # gets the largest share without evicting the preceding promoted bests.
        if recent and recent_mass > 0.0:
            denominator = sum(range(1, len(recent) + 1))
            for rank, entry in enumerate(recent, start=1):
                opponents.append(
                    OpponentSpec(
                        name=f"recent_{entry.archived_at_iteration:04d}_{entry.sha256[:10]}",
                        path=entry.path,
                        sha256=entry.sha256,
                        weight=recent_mass * rank / denominator,
                        kind="recent",
                        archived_at_iteration=entry.archived_at_iteration,
                    )
                )
        if hall and hall_mass > 0.0:
            for entry in hall:
                opponents.append(
                    OpponentSpec(
                        name=f"hof_{entry.archived_at_iteration:04d}_{entry.sha256[:10]}",
                        path=entry.path,
                        sha256=entry.sha256,
                        weight=hall_mass / len(hall),
                        kind="hall_of_fame",
                        archived_at_iteration=entry.archived_at_iteration,
                    )
                )
        return LeagueSelection(iteration=iteration, opponents=tuple(opponents))
