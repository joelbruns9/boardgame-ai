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
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class RunManifest:
    def __init__(self, run_dir: str | Path, repo_root: str | Path):
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root)
        self.path = self.run_dir / "run_manifest.json"

    def initialize(
        self,
        *,
        config: Any,
        adapter_contract: dict[str, Any],
        model_contract: dict[str, Any],
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        status = _git(self.repo_root, "status", "--porcelain")
        diff = _git(self.repo_root, "diff", "--binary", "HEAD")
        (self.run_dir / "dirty_diff.patch").write_text(diff, encoding="utf-8")
        payload = {
            "manifest_version": 1,
            "run_id": self.run_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "command": sys.argv,
            "git": {
                "commit": _git(self.repo_root, "rev-parse", "HEAD"),
                "branch": _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(status and status != "unknown"),
                "status_porcelain": status.splitlines() if status != "unknown" else [],
            },
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
        }
        _atomic_json(self.path, payload)
        return payload

    def append_iteration(self, row: dict[str, Any]) -> None:
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        manifest["iterations"].append(row)
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
