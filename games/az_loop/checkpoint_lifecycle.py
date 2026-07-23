"""Atomic rolling-checkpoint lifecycle for game-agnostic AZ training loops.

The controller owns three checkpoint *roles* -- ``latest``, ``current_best``,
and immutable per-iteration ``candidate`` snapshots -- but treats every
checkpoint as an opaque file.  It never imports torch, understands weights, or
reads a payload; a game adapter writes candidate files and the controller only
moves bytes between rolling roles.

All writes to the rolling ``latest.pt`` / ``current_best.pt`` files go through a
temporary file *in the destination directory* followed by :func:`os.replace`.
``os.replace`` is atomic only within a single volume, so the temporary is always
created beside its destination rather than in a system temp dir on another
drive.  A crash therefore leaves either the old file or the new file, never a
truncated blend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

LATEST = "latest"
CANDIDATE = "candidate"
CURRENT_BEST = "current_best"

UNTRAINED = "untrained"
TRAINED = "trained"


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Immutable description of a checkpoint file at a moment in time."""

    path: Path
    sha256: str
    role: str
    iteration: int
    training_state: str

    def as_row(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "role": self.role,
            "iteration": self.iteration,
            "training_state": self.training_state,
        }


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA-256 without loading it whole into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_for(
    path: str | Path,
    *,
    role: str,
    iteration: int,
    training_state: str,
) -> CheckpointArtifact:
    """Hash an existing checkpoint file and describe it in a given role."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {resolved}")
    if training_state not in (UNTRAINED, TRAINED):
        raise ValueError(f"invalid training_state: {training_state!r}")
    return CheckpointArtifact(
        path=resolved,
        sha256=sha256_file(resolved),
        role=role,
        iteration=iteration,
        training_state=training_state,
    )


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    """Copy ``source`` onto ``destination`` atomically within its volume.

    The temporary lands in the destination directory so the final
    :func:`os.replace` is a same-volume rename and therefore atomic even across
    drives on Windows.
    """

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    _atomic_write(destination, data)


def atomic_write_bytes(destination: str | Path, data: bytes) -> None:
    """Write ``data`` to ``destination`` atomically within its volume."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, data)


def _atomic_write(destination: Path, data: bytes) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def install(
    source: str | Path,
    destination: str | Path,
    *,
    role: str,
    iteration: int,
    training_state: str,
) -> CheckpointArtifact:
    """Atomically install ``source`` into a rolling ``destination`` role.

    Returns the artifact describing the destination after installation.  The
    hash is computed from the destination copy so the returned digest matches
    exactly what a resume will later re-verify on disk.
    """

    atomic_copy(source, destination)
    return artifact_for(
        destination, role=role, iteration=iteration, training_state=training_state
    )
