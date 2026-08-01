"""Generic, adapter-supplied run provenance for AZ training loops."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class RunManifest:
    def __init__(self, run_dir: str | Path, repo_root: str | Path):
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root)
        self.path = self.run_dir / "run_manifest.json"

    def code_identity(self) -> dict[str, Any]:
        """Commit plus a fingerprint of any uncommitted work (W6.5).

        The commit alone is not the code that ran: a dirty tree at the same SHA
        is different code.  The diff digest makes that difference comparable on
        resume without storing the patch twice.
        """

        status = _git(self.repo_root, "status", "--porcelain")
        diff = _git(self.repo_root, "diff", "--binary", "HEAD")
        return {
            "commit": _git(self.repo_root, "rev-parse", "HEAD"),
            "branch": _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status and status != "unknown"),
            "status_porcelain": status.splitlines() if status != "unknown" else [],
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "_diff": diff,
        }

    def initialize(
        self,
        *,
        config: Any,
        adapter_contract: dict[str, Any],
        model_contract: dict[str, Any],
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        identity = self.code_identity()
        (self.run_dir / "dirty_diff.patch").write_text(
            identity.pop("_diff"), encoding="utf-8"
        )
        payload = {
            "manifest_version": 1,
            "run_id": self.run_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": sys.argv,
            "git": identity,
            "config": (
                dataclasses.asdict(config)
                if dataclasses.is_dataclass(config)
                else dict(config)
            ),
            "adapter_contract": adapter_contract,
            "model_contract": model_contract,
            "hardware": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "checkpoints": [],
            "iterations": [],
            "iteration_log": {"path": "training_log.jsonl", "count": 0},
        }
        _atomic_json(self.path, payload)
        return payload

    def note_iteration(self, iteration: int) -> None:
        """Record that an iteration completed, in O(1) manifest bytes.

        This used to append the whole row, which made the manifest a verbatim
        duplicate of ``training_log.jsonl`` -- and because every append rewrites
        the file, the cost was quadratic in iterations.  Measured on run 03 at
        iteration 149: a 188 MB manifest, of which 57.3 MB was an exact copy of
        the log, costing 0.8 s to parse and 2.9 s to re-serialise *per
        iteration*, and briefly allocating ~320 MB of peak RSS on a heap that
        was already 6.5 GB.  That transient is the most plausible cause of the
        MemoryError that killed the run this one continued from.

        The rows live in the training log, which is append-only and therefore
        flat in cost.  ``iterations`` stays in the payload but is never appended
        to again: old runs keep the rows they already have, and
        ``_sync_training_log`` copies them into the log on the next start.
        """

        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        log = manifest.setdefault(
            "iteration_log", {"path": "training_log.jsonl", "count": 0}
        )
        log["count"] = int(log.get("count", 0)) + 1
        log["last_iteration"] = iteration
        log.setdefault("first_iteration", iteration)
        _atomic_json(self.path, manifest)

    def record_schedule_change(self, entry: dict[str, Any]) -> None:
        """Append a schedule change a resume was explicitly allowed to make.

        The schedule guard's objection to a mid-run change is that the run has
        no way to record it happened.  This is that way: an append-only list of
        ``{at_games, changes, recorded_at_utc}``, so a reader of the finished run
        can attribute every iteration to the regime it actually trained under.
        """

        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        manifest.setdefault("schedule_changes", []).append(
            {
                **entry,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
        )
        _atomic_json(self.path, manifest)

    def add_checkpoint(self, path: str | Path, iteration: int, promoted: bool) -> None:
        checkpoint = Path(path)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        manifest["checkpoints"].append(
            {
                "iteration": iteration,
                "path": str(checkpoint.resolve()),
                "sha256": digest,
                "promoted": promoted,
            }
        )
        _atomic_json(self.path, manifest)
