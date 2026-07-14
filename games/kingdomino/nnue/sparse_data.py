"""Replay-derived packed/CSR training data for the sparse Kingdomino NNUE.

The JSONL trajectories remain the source of truth.  This module replays them
through ``RustGameState``, derives the frozen 5,710-index core and 171-value
summary before every action, and stores a disposable packed artifact.  Every
artifact is stamped with both semantic schema hashes so stale encoded data fails
loudly instead of recreating the run10 encoder lock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import datagen
from .d4 import D4_ELEMENTS, sparse_perm, summary_perm
from .sparse_encoder import CORE_SIZE, core_schema_hash
from .summary_encoder import (
    MAX_BOARD_CELLS,
    SCORE_SCALE,
    SUMMARY_SIZE,
    TOTAL_CATALOG_CROWNS,
    summary_schema_hash,
)

ARTIFACT_VERSION = 1
MARGIN_SCALE = 40.0

# Continuous auxiliary target order and fixed, corpus-independent scales.
AUX_SCORE_NAMES = (
    "my_territory",
    "my_largest_territory",
    "my_total_crowns",
    "opp_territory",
    "opp_largest_territory",
    "opp_total_crowns",
)
AUX_SCORE_SCALES = np.asarray(
    [SCORE_SCALE, MAX_BOARD_CELLS, TOTAL_CATALOG_CROWNS] * 2,
    dtype=np.float32,
)
AUX_BONUS_NAMES = (
    "my_harmony",
    "my_middle_kingdom",
    "opp_harmony",
    "opp_middle_kingdom",
)
TARGET_SCHEMA = {
    "outcome": "actor expected match score: win=1, official draw=.5, loss=0",
    "margin": f"actor final total-score margin / {MARGIN_SCALE:g}",
    "aux_scores": dict(zip(AUX_SCORE_NAMES, AUX_SCORE_SCALES.tolist())),
    "aux_bonus_logits": list(AUX_BONUS_NAMES),
    "player_order": "my(actor), opponent",
}


def _metadata(records: list[dict], position_count: int, rules: tuple[bool, bool]) -> dict:
    source_bytes = "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    ).encode()
    return {
        "artifact": "kingdomino_sparse_nnue_csr",
        "artifact_version": ARTIFACT_VERSION,
        "core_size": CORE_SIZE,
        "summary_size": SUMMARY_SIZE,
        "core_schema_hash": core_schema_hash(),
        "summary_schema_hash": summary_schema_hash(),
        "datagen_engine_version": datagen.ENGINE_VERSION,
        "datagen_format_version": datagen.FORMAT_VERSION,
        "catalog_hash": datagen.catalog_hash(),
        "rules": {"harmony": rules[0], "middle_kingdom": rules[1]},
        "game_count": len(records),
        "position_count": int(position_count),
        "source_records_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_git_commits": sorted({str(r.get("git_commit", "")) for r in records}),
        "source_git_dirty": any(bool(r.get("git_dirty", False)) for r in records),
        "source_seeds": [min(int(r["seed"]) for r in records),
                         max(int(r["seed"]) for r in records)],
        "target_schema": TARGET_SCHEMA,
        "d4": "base orientation stored; one of 8 frozen permutations applied at batch time",
    }


@dataclass
class PackedSparseData:
    """CSR active indices plus dense summaries and actor-relative targets."""

    indices: np.ndarray
    offsets: np.ndarray
    summaries: np.ndarray
    outcome: np.ndarray
    margin: np.ndarray
    aux_scores: np.ndarray
    aux_bonus: np.ndarray
    actors: np.ndarray
    game_index: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.validate()

    def __len__(self) -> int:
        return int(self.summaries.shape[0])

    def validate(self) -> None:
        n = len(self.summaries)
        if self.indices.dtype != np.int32 or self.offsets.dtype != np.int64:
            raise ValueError("CSR indices/offsets must be int32/int64")
        if self.offsets.shape != (n + 1,) or self.offsets[0] != 0:
            raise ValueError("offsets must have length N+1 and begin at zero")
        if np.any(self.offsets[1:] < self.offsets[:-1]) or self.offsets[-1] != len(self.indices):
            raise ValueError("invalid CSR offsets")
        if len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= CORE_SIZE):
            raise ValueError("active feature index outside the frozen core")
        expected = {
            "summaries": (n, SUMMARY_SIZE),
            "outcome": (n,),
            "margin": (n,),
            "aux_scores": (n, len(AUX_SCORE_NAMES)),
            "aux_bonus": (n, len(AUX_BONUS_NAMES)),
            "actors": (n,),
            "game_index": (n,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape {getattr(self, name).shape} != {shape}")
        for name in ("summaries", "outcome", "margin", "aux_scores", "aux_bonus"):
            if getattr(self, name).dtype != np.float32 or not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite float32")
        if not np.isin(self.actors, (0, 1)).all():
            raise ValueError("actors must contain only player 0/1")
        _validate_metadata(self.metadata, n)

    def batch(self, rows, *, d4_choices=None, device=None) -> dict:
        """Create one torch batch, optionally applying per-row D4 augmentation."""
        import torch

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("rows must be one-dimensional")
        if d4_choices is None:
            choices = np.zeros(len(rows), dtype=np.int64)
        elif np.isscalar(d4_choices):
            choices = np.full(len(rows), int(d4_choices), dtype=np.int64)
        else:
            choices = np.asarray(d4_choices, dtype=np.int64)
        if choices.shape != rows.shape or np.any((choices < 0) | (choices >= len(D4_ELEMENTS))):
            raise ValueError("d4_choices must select one of the 8 D4 elements per row")

        chunks: list[np.ndarray] = []
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        summaries = np.empty((len(rows), SUMMARY_SIZE), dtype=np.float32)
        for j, (row, choice) in enumerate(zip(rows, choices)):
            start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
            idx = self.indices[start:stop]
            k, flip = D4_ELEMENTS[int(choice)]
            if choice:
                idx = _sparse_permutations()[int(choice)][idx]
                summaries[j] = self.summaries[row][_summary_sources()[int(choice)]]
            else:
                summaries[j] = self.summaries[row]
            chunks.append(np.asarray(idx, dtype=np.int64))
            offsets[j + 1] = offsets[j] + len(idx)
        flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

        def tensor(a, dtype=None):
            t = torch.from_numpy(np.ascontiguousarray(a))
            if dtype is not None:
                t = t.to(dtype=dtype)
            return t.to(device) if device is not None else t

        return {
            "indices": tensor(flat, torch.long),
            "offsets": tensor(offsets, torch.long),
            "summary": tensor(summaries),
            "outcome": tensor(self.outcome[rows]),
            "margin": tensor(self.margin[rows]),
            "aux_scores": tensor(self.aux_scores[rows]),
            "aux_bonus": tensor(self.aux_bonus[rows]),
        }


_SPARSE_PERMS: Optional[list[np.ndarray]] = None
_SUMMARY_SOURCES: Optional[list[np.ndarray]] = None


def _sparse_permutations() -> list[np.ndarray]:
    global _SPARSE_PERMS
    if _SPARSE_PERMS is None:
        _SPARSE_PERMS = [sparse_perm(k, f).astype(np.int32) for k, f in D4_ELEMENTS]
    return _SPARSE_PERMS


def _summary_sources() -> list[np.ndarray]:
    """For each D4 element, source index for every transformed destination."""
    global _SUMMARY_SOURCES
    if _SUMMARY_SOURCES is None:
        probe = np.arange(SUMMARY_SIZE, dtype=np.float32)
        _SUMMARY_SOURCES = [
            summary_perm(k, f)(probe).astype(np.int64) for k, f in D4_ELEMENTS
        ]
    return _SUMMARY_SOURCES


def _aux_from_breakdowns(breakdowns, actor: int):
    # Rust tuple: total, territory, largest, crowns, harmony_bonus, middle_bonus.
    ordered = (breakdowns[actor], breakdowns[1 - actor])
    score = []
    bonus = []
    for b in ordered:
        score.extend((b[1], b[2], b[3]))
        bonus.extend((float(b[4] > 0), float(b[5] > 0)))
    return np.asarray(score, np.float32) / AUX_SCORE_SCALES, np.asarray(bonus, np.float32)


def derive_records(records: Iterable[dict]) -> PackedSparseData:
    """Replay whole records into a disposable packed feature artifact."""
    import kingdomino_rust as kr

    records = list(records)
    if not records:
        raise ValueError("at least one replay record is required")
    rules_seen = {(bool(r["harmony"]), bool(r["middle_kingdom"])) for r in records}
    if len(rules_seen) != 1:
        raise ValueError(f"records mix rules configurations: {rules_seen}")
    rust_schema = tuple(kr.nnue_schema_info())
    expected_schema = (CORE_SIZE, SUMMARY_SIZE, core_schema_hash(), summary_schema_hash())
    if rust_schema != expected_schema:
        raise RuntimeError(f"Rust/Python NNUE schema mismatch: {rust_schema} != {expected_schema}")

    chunks: list[np.ndarray] = []
    offsets = [0]
    summaries: list[np.ndarray] = []
    outcomes: list[float] = []
    margins: list[float] = []
    aux_scores: list[np.ndarray] = []
    aux_bonus: list[np.ndarray] = []
    actors: list[int] = []
    game_indices: list[int] = []

    for gi, rec in enumerate(records):
        rs = kr.RustGameState(
            rec["start_player"], list(rec["deck"]), list(rec["current_row"]),
            rec["harmony"], rec["middle_kingdom"],
        )
        game_actors: list[int] = []
        for sa in rec["actions"]:
            actor = int(rs.current_actor())
            idx, summary = rs.nnue_features(actor)
            idx = np.asarray(idx, dtype=np.int32)
            summary = np.asarray(summary, dtype=np.float32)
            chunks.append(idx)
            offsets.append(offsets[-1] + len(idx))
            summaries.append(summary)
            game_actors.append(actor)
            placement, pick = datagen._deser_action(sa)
            rs = rs.step(placement, pick)

        if rs.phase != datagen.GAME_OVER:
            raise ValueError(f"seed {rec['seed']}: trajectory did not reach game over")
        if len(game_actors) != int(rec["n_positions"]):
            raise ValueError(f"seed {rec['seed']}: n_positions disagrees with actions")
        scores = tuple(int(x) for x in rs.scores())
        outcome_p0 = int(kr.SearchEngine(rs).official_outcome())
        if list(scores) != rec["final_scores"] or outcome_p0 != int(rec["outcome_p0"]):
            raise ValueError(f"seed {rec['seed']}: replay label mismatch")
        breakdowns = rs.score_breakdowns()
        for actor in game_actors:
            actor_outcome = outcome_p0 if actor == 0 else -outcome_p0
            outcomes.append((actor_outcome + 1.0) / 2.0)
            margins.append((scores[actor] - scores[1 - actor]) / MARGIN_SCALE)
            sc, bn = _aux_from_breakdowns(breakdowns, actor)
            aux_scores.append(sc)
            aux_bonus.append(bn)
            actors.append(actor)
            game_indices.append(gi)

    n = len(summaries)
    return PackedSparseData(
        indices=np.concatenate(chunks).astype(np.int32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
        summaries=np.asarray(summaries, dtype=np.float32),
        outcome=np.asarray(outcomes, dtype=np.float32),
        margin=np.asarray(margins, dtype=np.float32),
        aux_scores=np.asarray(aux_scores, dtype=np.float32),
        aux_bonus=np.asarray(aux_bonus, dtype=np.float32),
        actors=np.asarray(actors, dtype=np.uint8),
        game_index=np.asarray(game_indices, dtype=np.int32),
        metadata=_metadata(records, n, next(iter(rules_seen))),
    )


def _validate_metadata(meta: dict, n: Optional[int] = None) -> None:
    expected = {
        "artifact_version": ARTIFACT_VERSION,
        "core_size": CORE_SIZE,
        "summary_size": SUMMARY_SIZE,
        "core_schema_hash": core_schema_hash(),
        "summary_schema_hash": summary_schema_hash(),
        "datagen_engine_version": datagen.ENGINE_VERSION,
        "datagen_format_version": datagen.FORMAT_VERSION,
        "catalog_hash": datagen.catalog_hash(),
        "target_schema": TARGET_SCHEMA,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"stale packed artifact: {key}={meta.get(key)!r} != {value!r}")
    if n is not None and int(meta.get("position_count", -1)) != n:
        raise ValueError("packed artifact position_count does not match arrays")
    source_hash = meta.get("source_records_sha256", "")
    if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("packed artifact is missing a valid source-record hash")


def save_packed(data: PackedSparseData, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=data.indices,
        offsets=data.offsets,
        summaries=data.summaries,
        outcome=data.outcome,
        margin=data.margin,
        aux_scores=data.aux_scores,
        aux_bonus=data.aux_bonus,
        actors=data.actors,
        game_index=data.game_index,
        metadata=np.asarray(json.dumps(data.metadata, sort_keys=True)),
    )
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def load_packed(path) -> PackedSparseData:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        return PackedSparseData(
            indices=z["indices"].copy(),
            offsets=z["offsets"].copy(),
            summaries=z["summaries"].copy(),
            outcome=z["outcome"].copy(),
            margin=z["margin"].copy(),
            aux_scores=z["aux_scores"].copy(),
            aux_bonus=z["aux_bonus"].copy(),
            actors=z["actors"].copy(),
            game_index=z["game_index"].copy(),
            metadata=meta,
        )


def concatenate_packed(parts: Iterable[PackedSparseData]) -> PackedSparseData:
    """Join independently derived shards without weakening schema validation.

    Replay JSONL remains the source of truth.  This only combines disposable
    packed caches, preserving each component's source hash in the aggregate
    metadata and making game indices unique across shards.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("at least one packed shard is required")
    rules = {
        (bool(p.metadata["rules"]["harmony"]),
         bool(p.metadata["rules"]["middle_kingdom"]))
        for p in parts
    }
    if len(rules) != 1:
        raise ValueError(f"packed shards mix rules configurations: {rules}")

    offsets = [0]
    indices: list[np.ndarray] = []
    game_indices: list[np.ndarray] = []
    game_base = 0
    for part in parts:
        indices.append(part.indices)
        offsets.extend((part.offsets[1:] + offsets[-1]).tolist())
        game_indices.append(part.game_index.astype(np.int64) + game_base)
        game_base += int(part.metadata["game_count"])

    component_hashes = [p.metadata["source_records_sha256"] for p in parts]
    aggregate_hash = hashlib.sha256("\n".join(component_hashes).encode()).hexdigest()
    metadata = {
        "artifact": "kingdomino_sparse_nnue_csr",
        "artifact_version": ARTIFACT_VERSION,
        "core_size": CORE_SIZE,
        "summary_size": SUMMARY_SIZE,
        "core_schema_hash": core_schema_hash(),
        "summary_schema_hash": summary_schema_hash(),
        "datagen_engine_version": datagen.ENGINE_VERSION,
        "datagen_format_version": datagen.FORMAT_VERSION,
        "catalog_hash": datagen.catalog_hash(),
        "rules": {"harmony": next(iter(rules))[0],
                  "middle_kingdom": next(iter(rules))[1]},
        "game_count": game_base,
        "position_count": sum(len(p) for p in parts),
        "source_records_sha256": aggregate_hash,
        "source_components": [p.metadata for p in parts],
        "source_git_commits": sorted({
            commit for p in parts for commit in p.metadata.get("source_git_commits", [])
        }),
        "source_git_dirty": any(p.metadata.get("source_git_dirty", False) for p in parts),
        "target_schema": TARGET_SCHEMA,
        "d4": "base orientation stored; one of 8 frozen permutations applied at batch time",
    }
    return PackedSparseData(
        indices=np.concatenate(indices).astype(np.int32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
        summaries=np.concatenate([p.summaries for p in parts]),
        outcome=np.concatenate([p.outcome for p in parts]),
        margin=np.concatenate([p.margin for p in parts]),
        aux_scores=np.concatenate([p.aux_scores for p in parts]),
        aux_bonus=np.concatenate([p.aux_bonus for p in parts]),
        actors=np.concatenate([p.actors for p in parts]),
        game_index=np.concatenate(game_indices).astype(np.int32),
        metadata=metadata,
    )


def derive_split(source_dir, split: str, out_path=None, *, max_games: int = 0):
    """Strict-load and derive one whole-game split; optionally cache it."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    source = Path(source_dir) / f"{split}.jsonl"
    records = datagen.load_records(str(source), strict=True)
    if max_games:
        records = records[:max_games]
    data = derive_records(records)
    if out_path is not None:
        save_packed(data, out_path)
    return data
