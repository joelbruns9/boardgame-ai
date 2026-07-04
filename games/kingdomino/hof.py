"""Hall-of-fame checkpoint helpers for Kingdomino training.

Milestone 7 uses the HOF pool as an opponent source, not as a source of policy
labels.  Entries are copied from promoted/current-best checkpoints and tracked
in a JSONL index so training runs can sample one stable opponent per iteration.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from games.kingdomino.network import KingdominoNet
from games.kingdomino.promotion import (
    DEFAULT_BEST_DIR,
    DEFAULT_CURRENT_BEST,
    sha256_file,
)
from games.kingdomino.round_robin_eval import checkpoint_config, checkpoint_state_dict


DEFAULT_HOF_DIR = DEFAULT_BEST_DIR / "hof"


@dataclass
class HOFEntry:
    path: str
    sha256: str
    timestamp: str
    source: str
    source_sha256: str
    tag: str
    iteration: int | None = None
    channels: int | None = None
    blocks: int | None = None
    bilinear_dim: int | None = None
    score_scale: float | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def read_hof_index(hof_dir: str | Path = DEFAULT_HOF_DIR) -> list[HOFEntry]:
    index = Path(hof_dir) / "hof_index.jsonl"
    if not index.exists():
        return []
    entries: list[HOFEntry] = []
    with index.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            fields = HOFEntry.__dataclass_fields__
            entries.append(HOFEntry(**{k: v for k, v in data.items() if k in fields}))
    return entries


def load_hof_net(path: str | Path, device: str = "cpu") -> KingdominoNet:
    """Load a HOF checkpoint using its own saved architecture config."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    net = KingdominoNet(
        channels=int(cfg.get("channels", 96)),
        blocks=int(cfg.get("blocks", 8)),
        bilinear_dim=int(cfg.get("bilinear_dim", 64)),
        score_scale=float(cfg.get("score_scale", 160.0)),
    )
    net.load_state_dict(checkpoint_state_dict(ckpt))
    net.to(device)
    net.eval()
    return net


def add_hof_entry(
    source_checkpoint: str | Path = DEFAULT_CURRENT_BEST,
    *,
    hof_dir: str | Path = DEFAULT_HOF_DIR,
    tag: str = "current_best",
    iteration: int | None = None,
    metadata: dict[str, Any] | None = None,
    allow_duplicate: bool = False,
) -> HOFEntry:
    """Copy a promoted checkpoint into the HOF pool and append index metadata."""
    source = Path(source_checkpoint)
    if not source.exists():
        raise FileNotFoundError(f"HOF source checkpoint does not exist: {source}")

    hof_dir = Path(hof_dir)
    hof_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if not allow_duplicate:
        for entry in read_hof_index(hof_dir):
            if entry.sha256 == source_hash:
                return entry
    safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    iter_part = f"iter_{int(iteration):04d}" if iteration is not None else "manual"
    target = hof_dir / f"hof_{iter_part}_{safe_tag}_{source_hash[:12]}_{stamp}.pt"
    shutil.copy2(source, target)

    ckpt = torch.load(source, map_location="cpu")
    cfg = checkpoint_config(ckpt)
    entry = HOFEntry(
        path=str(target),
        sha256=sha256_file(target),
        timestamp=now_iso(),
        source=str(source),
        source_sha256=source_hash,
        tag=str(tag),
        iteration=iteration,
        channels=int(cfg["channels"]) if "channels" in cfg else None,
        blocks=int(cfg["blocks"]) if "blocks" in cfg else None,
        bilinear_dim=int(cfg["bilinear_dim"]) if "bilinear_dim" in cfg else None,
        score_scale=float(cfg["score_scale"]) if "score_scale" in cfg else None,
    )
    payload = asdict(entry)
    if metadata:
        payload["metadata"] = metadata
    append_jsonl(hof_dir / "hof_index.jsonl", payload)
    return entry


def sample_hof_entry(
    entries: list[HOFEntry],
    *,
    rng: random.Random,
    weights: str = "recency",
) -> HOFEntry | None:
    if not entries:
        return None
    mode = (weights or "recency").lower()
    if mode == "uniform":
        return rng.choice(entries)
    if mode == "latest":
        return entries[-1]
    if mode == "mixed":
        if rng.random() < 0.7:
            return sample_hof_entry(entries, rng=rng, weights="recency")
        return rng.choice(entries)
    if mode != "recency":
        raise ValueError(f"unknown HOF sample weighting mode: {weights!r}")

    # Recent entries are more relevant, but older styles remain reachable.
    n = len(entries)
    weights_arr = [float(i + 1) for i in range(n)]
    total = sum(weights_arr)
    pick = rng.random() * total
    acc = 0.0
    for entry, weight in zip(entries, weights_arr):
        acc += weight
        if pick <= acc:
            return entry
    return entries[-1]
