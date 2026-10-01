"""Experiment identity: enough to say exactly what produced a result.

Plan review (2026-09-26): result files must record their inputs -- the
code commit (and whether the tree was dirty), each checkpoint's path AND
content hash, the search settings, seeds and the command line. A score
table that does not say which nets and searches it scored is not evidence.

    meta = identity(nets={"a": path_a}, search=..., seed=...)
"""

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def identity(nets=None, **fields):
    """A JSON-ready record of what is running. ``nets`` maps a role to a
    checkpoint path; each is recorded with its sha256."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nets": {role: {"path": str(p), "sha256": file_sha256(p)}
                 for role, p in (nets or {}).items()},
        **{k: _plain(v) for k, v in fields.items()},
    }


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
