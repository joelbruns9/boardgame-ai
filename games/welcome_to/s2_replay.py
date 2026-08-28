"""Durable multi-iteration replay selection for Welcome To S2.

The immutable `.wts` generation shards are the buffer.  This module supplies
the missing 7WD-style window/ledger layer: discover completed iterations,
select a games-sized growing window, validate its storage ABI, and record the
exact shard set used for a candidate.  Tensor rows remain on disk and are
sampled directly by Rust during training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from games.az_loop import GrowingReplayWindow, WindowSelection
from games.welcome_to import self_play


REPLAY_FORMAT = "welcome_to_s2_replay"
REPLAY_VERSION = 1
LEDGER_FORMAT = "welcome_to_s2_replay_ledger"
LEDGER_VERSION = 1
LEDGER_NAME = "replay_ledger.json"
_ITERATION = re.compile(r"iter_(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iteration_number(path: Path) -> Optional[int]:
    match = _ITERATION.fullmatch(path.name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True, slots=True)
class ReplayIteration:
    iteration: int
    prefix: Path
    games: int
    positions: int
    generation_manifest: Path
    generation_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayLedgerEntry:
    """A completed iteration's immutable contribution to the games clock."""

    iteration: int
    games: int
    positions: int
    generation_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayCorpus:
    trajectories: tuple[self_play.SelfPlayTrajectory, ...]
    trajectory_iterations: tuple[int, ...]
    iterations: tuple[ReplayIteration, ...]
    selection: WindowSelection
    games_before_selection: int

    @property
    def positions(self) -> int:
        return sum(item.positions for item in self.iterations)

    @property
    def newest_positions(self) -> int:
        newest = self.selection.newest_iteration
        return sum(
            item.positions for item in self.iterations if item.iteration == newest
        )

    def metrics(self) -> dict[str, Any]:
        newest = self.selection.newest_iteration
        ages = [newest - item.iteration for item in self.iterations] if newest is not None else []
        return {
            "format": REPLAY_FORMAT,
            "version": REPLAY_VERSION,
            "scheduled_window_games": self.selection.target_games,
            "realized_window_games": self.selection.realised_games,
            "window_positions": self.positions,
            "newest_positions": self.newest_positions,
            "window_iterations": len(self.iterations),
            "oldest_iteration": self.selection.oldest_iteration,
            "newest_iteration": newest,
            "games_before_selection": self.games_before_selection,
            "iteration_ages": ages,
            "iterations": [
                {
                    "iteration": item.iteration,
                    "prefix": str(item.prefix.resolve()),
                    "games": item.games,
                    "positions": item.positions,
                    "generation_manifest": str(item.generation_manifest.resolve()),
                    "generation_manifest_sha256": item.generation_manifest_sha256,
                }
                for item in self.iterations
            ],
        }


def _iteration_summary(directory: Path, iteration: int) -> ReplayIteration:
    prefix = directory / "trajectories.jsonl"
    shards = self_play.training_shard_paths(prefix)
    if not shards:
        raise FileNotFoundError(f"replay iteration {directory} has no .wts shards")
    manifest_path = prefix.with_suffix(prefix.suffix + ".manifest.json")
    metrics_path = prefix.with_suffix(prefix.suffix + ".metrics.json")
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"replay iteration {directory} needs generation manifest and metrics"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    games = int(metrics.get("total_games", metrics.get("games", 0)))
    positions = int(metrics.get("searched_roots", 0))
    if games <= 0 or positions <= 0:
        raise ValueError(f"replay iteration {directory} has invalid game/position counts")
    return ReplayIteration(
        iteration=iteration,
        prefix=prefix,
        games=games,
        positions=positions,
        generation_manifest=manifest_path,
        generation_manifest_sha256=_sha256(manifest_path),
    )


def _ledger_path(root: Path) -> Path:
    return root / LEDGER_NAME


def _load_ledger(root: Path) -> tuple[ReplayLedgerEntry, ...]:
    path = _ledger_path(root)
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != LEDGER_FORMAT
        or int(payload.get("version", -1)) != LEDGER_VERSION
    ):
        raise ValueError(f"unsupported replay ledger {path}")
    entries = tuple(ReplayLedgerEntry(**item) for item in payload.get("entries", []))
    numbers = [item.iteration for item in entries]
    if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
        raise ValueError("replay ledger iterations must be unique and ordered")
    if any(item.games <= 0 or item.positions <= 0 for item in entries):
        raise ValueError("replay ledger counts must be positive")
    return entries


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _write_ledger(root: Path, entries: Sequence[ReplayLedgerEntry]) -> Path:
    return _atomic_write_json(
        _ledger_path(root),
        {
            "format": LEDGER_FORMAT,
            "version": LEDGER_VERSION,
            "entries": [
                {
                    "iteration": item.iteration,
                    "games": item.games,
                    "positions": item.positions,
                    "generation_manifest_sha256": item.generation_manifest_sha256,
                }
                for item in entries
            ],
        },
    )


def discover_iterations(
    replay_root: str | Path,
    *,
    through_iteration: Optional[int] = None,
    _known_complete: Sequence[int] = (),
) -> list[ReplayIteration]:
    """Discover complete iterations, ignoring never-completed crash residue.

    An incomplete directory that has never entered the durable ledger is an
    interrupted generation attempt and is skipped. If the ledger says the
    iteration completed, the same missing files are corruption and fail loudly.
    """
    root = Path(replay_root)
    if not root.is_dir():
        raise FileNotFoundError(f"replay root does not exist: {root}")
    found: list[ReplayIteration] = []
    known_complete = frozenset(int(item) for item in _known_complete)
    for directory in root.iterdir():
        iteration = _iteration_number(directory) if directory.is_dir() else None
        if iteration is None or (
            through_iteration is not None and iteration > through_iteration
        ):
            continue
        try:
            found.append(_iteration_summary(directory, iteration))
        except FileNotFoundError:
            if iteration in known_complete:
                raise ValueError(
                    f"completed replay iteration {iteration} is now incomplete"
                ) from None
            continue
    if not found:
        raise FileNotFoundError(f"no completed replay iterations under {root}")
    found.sort(key=lambda item: item.iteration)
    numbers = [item.iteration for item in found]
    if len(numbers) != len(set(numbers)):
        raise ValueError("replay root contains duplicate iteration numbers")
    return found


def _validate_iteration_content(
    item: ReplayIteration,
) -> tuple[list[self_play.SelfPlayTrajectory], ReplayLedgerEntry]:
    games = self_play.read_trajectories(item.prefix)
    actual_games = len(games)
    actual_positions = sum(len(game.searches) for game in games)
    if actual_games != item.games:
        raise ValueError(
            f"replay iteration {item.iteration} metrics say {item.games} games "
            f"but shards contain {actual_games}"
        )
    if actual_positions != item.positions:
        raise ValueError(
            f"replay iteration {item.iteration} metrics say {item.positions} "
            f"positions but shards contain {actual_positions}"
        )
    return games, ReplayLedgerEntry(
        iteration=item.iteration,
        games=actual_games,
        positions=actual_positions,
        generation_manifest_sha256=item.generation_manifest_sha256,
    )


def _synchronize_ledger(
    root: Path,
    available: Sequence[ReplayIteration],
    ledger: Sequence[ReplayLedgerEntry],
) -> tuple[tuple[ReplayLedgerEntry, ...], dict[int, list[self_play.SelfPlayTrajectory]]]:
    """Validate every visible iteration before it can influence the clock."""
    by_iteration = {item.iteration: item for item in ledger}
    decoded: dict[int, list[self_play.SelfPlayTrajectory]] = {}
    changed = False
    for item in available:
        recorded = by_iteration.get(item.iteration)
        if recorded is not None:
            current = ReplayLedgerEntry(
                iteration=item.iteration,
                games=item.games,
                positions=item.positions,
                generation_manifest_sha256=item.generation_manifest_sha256,
            )
            if current != recorded:
                raise ValueError(
                    f"replay iteration {item.iteration} no longer matches its ledger entry"
                )
            continue
        games, entry = _validate_iteration_content(item)
        decoded[item.iteration] = games
        by_iteration[item.iteration] = entry
        changed = True
    entries = tuple(by_iteration[index] for index in sorted(by_iteration))
    if changed:
        _write_ledger(root, entries)
    return entries, decoded


def _validate_generation_manifests(iterations: Sequence[ReplayIteration]) -> None:
    identity = None
    for item in iterations:
        manifest = json.loads(item.generation_manifest.read_text(encoding="utf-8"))
        current = (
            manifest.get("format"),
            manifest.get("version"),
            manifest.get("trajectory_version"),
            manifest.get("table_signature"),
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(
                "replay window mixes incompatible generation/table ABIs: "
                f"{identity!r} and {current!r}"
            )


def assemble_replay(
    replay_root: str | Path,
    *,
    through_iteration: Optional[int] = None,
    coefficient: float = 16.0,
    exponent: float = 0.6,
    cap_games: int = 20_000,
    floor_games: int = 500,
) -> ReplayCorpus:
    """Assemble a whole-iteration window on a durable cumulative-games clock."""
    root = Path(replay_root)
    ledger = _load_ledger(root)
    available = discover_iterations(
        root,
        through_iteration=through_iteration,
        _known_complete=[item.iteration for item in ledger],
    )
    ledger, decoded = _synchronize_ledger(root, available, ledger)
    by_iteration = {item.iteration: item for item in available}
    current = max(by_iteration)
    total_games = sum(item.games for item in ledger if item.iteration <= current)
    window = GrowingReplayWindow(
        coefficient=coefficient,
        exponent=exponent,
        cap_games=cap_games,
        floor_games=floor_games,
    )
    selection = window.select(
        total_games,
        current,
        lambda iteration: by_iteration[iteration].games,
        tuple(by_iteration),
    )
    selected = tuple(by_iteration[index] for index in selection.iterations)
    _validate_generation_manifests(selected)
    trajectories: list[self_play.SelfPlayTrajectory] = []
    trajectory_iterations: list[int] = []
    seen_seeds: set[int] = set()
    for item in selected:
        current_games = decoded.get(item.iteration)
        if current_games is None:
            current_games, recorded = _validate_iteration_content(item)
            expected = next(
                entry for entry in ledger if entry.iteration == item.iteration
            )
            if recorded != expected:
                raise ValueError(
                    f"replay iteration {item.iteration} content changed after ledgering"
                )
        actual_positions = sum(len(game.searches) for game in current_games)
        if actual_positions != item.positions:
            raise ValueError(
                f"replay iteration {item.iteration} metrics say {item.positions} "
                f"positions but shards contain {actual_positions}"
            )
        duplicates = seen_seeds.intersection(game.seed for game in current_games)
        if duplicates:
            sample = sorted(duplicates)[:8]
            raise ValueError(f"replay window repeats game seeds {sample}")
        seen_seeds.update(game.seed for game in current_games)
        trajectories.extend(current_games)
        trajectory_iterations.extend([item.iteration] * len(current_games))
    return ReplayCorpus(
        trajectories=tuple(trajectories),
        trajectory_iterations=tuple(trajectory_iterations),
        iterations=selected,
        selection=selection,
        games_before_selection=total_games,
    )


def explicit_replay(prefixes: Sequence[str | Path]) -> ReplayCorpus:
    """Build an unwindowed corpus from explicitly named trajectory prefixes."""
    if not prefixes:
        raise ValueError("at least one trajectory prefix is required")
    trajectories: list[self_play.SelfPlayTrajectory] = []
    trajectory_iterations: list[int] = []
    iterations: list[ReplayIteration] = []
    seen: set[int] = set()
    for ordinal, raw_prefix in enumerate(prefixes):
        prefix = Path(raw_prefix)
        games = self_play.read_trajectories(prefix)
        duplicates = seen.intersection(game.seed for game in games)
        if duplicates:
            raise ValueError(f"explicit replay repeats game seeds {sorted(duplicates)[:8]}")
        seen.update(game.seed for game in games)
        trajectories.extend(games)
        iteration = _iteration_number(prefix.parent)
        if iteration is None:
            iteration = ordinal
        trajectory_iterations.extend([iteration] * len(games))
        manifest = prefix.with_suffix(prefix.suffix + ".manifest.json")
        iterations.append(
            ReplayIteration(
                iteration=iteration,
                prefix=prefix,
                games=len(games),
                positions=sum(len(game.searches) for game in games),
                generation_manifest=manifest,
                generation_manifest_sha256=_sha256(manifest) if manifest.is_file() else "",
            )
        )
    iterations.sort(key=lambda item: item.iteration)
    realised = sum(item.games for item in iterations)
    selection = WindowSelection(
        target_games=realised,
        realised_games=realised,
        iterations=tuple(item.iteration for item in iterations),
    )
    return ReplayCorpus(
        trajectories=tuple(trajectories),
        trajectory_iterations=tuple(trajectory_iterations),
        iterations=tuple(iterations),
        selection=selection,
        games_before_selection=realised,
    )


def write_replay_manifest(path: str | Path, corpus: ReplayCorpus) -> Path:
    """Atomically record the exact durable replay selection for a candidate."""
    destination = Path(path)
    payload: Mapping[str, Any] = corpus.metrics()
    return _atomic_write_json(destination, payload)
