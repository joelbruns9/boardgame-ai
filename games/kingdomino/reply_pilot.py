"""Opponent-reply label generation for the secondary-pick training pilot.

The expensive forced tree remains owned by :mod:`denial_search`; this module
adds the production artifact boundary around its backed ply-1 labels:

* fresh, opponent-at-ply-1 training roots;
* deterministic modulo sharding and crash-safe per-root resume;
* compact, self-contained encoded states plus audit/reconstruction state JSON;
* locked quality-filter provenance; and
* strict canonical shard merging and validation.

No command in this module updates ``current_best`` or starts training.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from games.kingdomino.action_codec import NO_PICK_IDX, PICK_AXIS_SIZE
from games.kingdomino.denial_search import (
    ACTOR_FRAME,
    AZBatchEvaluator,
    DenialSearch,
    SearchConfig,
    generate_az_midgame_positions,
    load_checkpoint_network,
    public_state_key,
)
from games.kingdomino.denial_signal_sweep import (
    file_sha256,
    load_frozen_positions,
    write_frozen_positions,
)
from games.kingdomino.encoder import FLAT_SIZE, NUM_BOARD_CHANNELS, CANVAS_SIZE, encode_state
from games.kingdomino.game import GameState, Phase
from games.kingdomino.promotion import DEFAULT_CURRENT_BEST, sha256_file


SCHEMA_VERSION = 1
DEFAULT_DIR = Path("runs/kingdomino/reply_pilot")
DEFAULT_ROOTS = DEFAULT_DIR / "training_roots.jsonl"
DEFAULT_SHARDS = DEFAULT_DIR / "shards"
DEFAULT_MERGED = DEFAULT_DIR / "reply_labels.jsonl"
DEFAULT_FROZEN_EVAL = Path("runs/kingdomino/denial_search/signal_positions.jsonl")


def reply_root_eligible(state: GameState) -> bool:
    """True only when a round-start action is followed by the opponent."""
    if state.phase != Phase.PLACE_AND_SELECT or int(state.actor_index) != 0:
        return False
    if len(state.pending_claims) < 2:
        return False
    return int(state.pending_claims[0].player) != int(state.pending_claims[1].player)


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_dump(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _array_blob(array: np.ndarray) -> dict[str, Any]:
    compact = np.ascontiguousarray(array, dtype=np.float16)
    return {
        "dtype": "float16",
        "shape": list(compact.shape),
        "data_base64": base64.b64encode(compact.tobytes()).decode("ascii"),
    }


def decode_array_blob(blob: dict[str, Any]) -> np.ndarray:
    if blob.get("dtype") != "float16":
        raise ValueError(f"unsupported encoded dtype: {blob.get('dtype')!r}")
    shape = tuple(int(value) for value in blob["shape"])
    raw = base64.b64decode(blob["data_base64"], validate=True)
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float16).itemsize
    if len(raw) != expected:
        raise ValueError(f"encoded byte length {len(raw)} != expected {expected}")
    return np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()


def encode_reply_state(state: GameState) -> dict[str, Any]:
    actor = int(state.current_actor)
    my_board, opp_board, flat = encode_state(state, actor)
    return {
        "actor": actor,
        "my_board": _array_blob(my_board),
        "opp_board": _array_blob(opp_board),
        "flat": _array_blob(flat),
    }


def _encoded_state_hash(encoded: dict[str, Any], legal_indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for name in ("my_board", "opp_board", "flat"):
        digest.update(base64.b64decode(encoded[name]["data_base64"], validate=True))
    digest.update(np.asarray(legal_indices, dtype=np.int32).tobytes())
    return digest.hexdigest()


def _quality_decision(
    quality: dict[str, Any],
    *,
    calibration: bool,
    min_top_two_margin: Optional[float],
    max_mc_standard_error: Optional[float],
    max_target_entropy: Optional[float],
    reject_ties: bool,
    max_searched_seed_sd: Optional[float] = None,
    min_top_pick_agreement: Optional[float] = None,
) -> tuple[bool, list[str]]:
    if calibration:
        return False, ["calibration_only"]
    thresholds = (min_top_two_margin, max_mc_standard_error, max_target_entropy)
    if any(value is None for value in thresholds):
        raise ValueError("production generation requires all three locked quality thresholds")
    cross_required = (max_searched_seed_sd is not None
                      or min_top_pick_agreement is not None)
    if cross_required and (max_searched_seed_sd is None
                           or min_top_pick_agreement is None):
        raise ValueError("cross-seed thresholds must be supplied together")
    reasons = []
    if float(quality["top_two_margin"]) < float(min_top_two_margin):
        reasons.append("top_two_margin")
    if float(quality["max_mc_standard_error"]) > float(max_mc_standard_error):
        reasons.append("mc_standard_error")
    if float(quality["target_entropy"]) > float(max_target_entropy):
        reasons.append("target_entropy")
    if reject_ties and bool(quality["exact_or_near_tie"]):
        reasons.append("exact_or_near_tie")
    if cross_required and quality.get("max_searched_seed_sd") is None:
        reasons.append("cross_seed_not_computed")
    elif cross_required and float(quality["max_searched_seed_sd"]) > float(max_searched_seed_sd):
        reasons.append("searched_seed_sd")
    if cross_required and quality.get("top_pick_agreement") is None:
        if "cross_seed_not_computed" not in reasons:
            reasons.append("cross_seed_not_computed")
    elif cross_required and float(quality["top_pick_agreement"]) < float(min_top_pick_agreement):
        reasons.append("top_pick_agreement")
    return not reasons, reasons


def serialize_reply_example(
    label: dict[str, Any],
    *,
    position_index: int,
    root_state_key: str,
    source: dict[str, Any],
    calibration: bool,
    min_top_two_margin: Optional[float],
    max_mc_standard_error: Optional[float],
    max_target_entropy: Optional[float],
    reject_ties: bool,
    max_searched_seed_sd: Optional[float] = None,
    min_top_pick_agreement: Optional[float] = None,
) -> dict[str, Any]:
    state = label.get("_state")
    if not isinstance(state, GameState):
        raise TypeError("reply label is missing its in-memory GameState")
    accepted, reasons = _quality_decision(
        label["quality"], calibration=calibration,
        min_top_two_margin=min_top_two_margin,
        max_mc_standard_error=max_mc_standard_error,
        max_target_entropy=max_target_entropy,
        reject_ties=reject_ties,
        max_searched_seed_sd=max_searched_seed_sd,
        min_top_pick_agreement=min_top_pick_agreement,
    )
    from games.kingdomino.web_app import state_to_debug_json

    row = copy.deepcopy({key: value for key, value in label.items() if key != "_state"})
    row.update({
        "schema_version": SCHEMA_VERSION,
        "position_index": int(position_index),
        "root_state_key": str(root_state_key),
        "source": dict(source),
        "encoded_state": encode_reply_state(state),
        "state": state_to_debug_json(state),
        "discards": [int(value) for value in state.discards],
        "quality_accept": bool(accepted),
        "quality_rejection_reasons": reasons,
    })
    parent = "none" if row["parent_pick_domino_id"] is None else row["parent_pick_domino_id"]
    row["example_id"] = (
        f"p{position_index}:root-{parent}:chance-{row['chance_child_index']}:"
        f"{row['state_key']}"
    )
    return row


def serialize_rust_reply_example(
    label: dict[str, Any],
    *,
    rust_evaluator,
    position_index: int,
    root_state_key: str,
    source: dict[str, Any],
    calibration: bool,
    min_top_two_margin: Optional[float],
    max_mc_standard_error: Optional[float],
    max_target_entropy: Optional[float],
    reject_ties: bool,
    max_searched_seed_sd: Optional[float] = None,
    min_top_pick_agreement: Optional[float] = None,
) -> dict[str, Any]:
    """Serialize a Rust-tree reply without reconstructing a Python GameState."""
    state = label.get("_rust_state")
    if state is None:
        raise TypeError("Rust reply label is missing its state handle")
    actor = int(label["actor"])
    my_board, opp_board, flat = state.encode(actor)
    encoded = {
        "actor": actor,
        "my_board": _array_blob(np.asarray(my_board)),
        "opp_board": _array_blob(np.asarray(opp_board)),
        "flat": _array_blob(np.asarray(flat)),
    }
    legal_indices = [int(value) for value in state.legal_action_indices()]
    current_row = [int(value) for value in state.current_row()]
    _values, gathered = rust_evaluator(
        np.asarray(my_board, dtype=np.float32)[None, ...],
        np.asarray(opp_board, dtype=np.float32)[None, ...],
        np.asarray(flat, dtype=np.float32)[None, ...],
        [np.asarray(legal_indices, dtype=np.int64)],
    )
    logits = np.asarray(gathered[0], dtype=np.float64)
    logits -= float(logits.max())
    probabilities = np.exp(logits)
    probabilities /= float(probabilities.sum())

    actions_by_pick: dict[Optional[int], list[tuple[int, float]]] = {}
    legal_rows = []
    for action_idx, probability in zip(legal_indices, probabilities):
        slot = int(action_idx) % PICK_AXIS_SIZE
        pick = None if slot == NO_PICK_IDX else current_row[slot]
        legal_rows.append({"action_idx": action_idx, "pick_domino_id": pick})
        actions_by_pick.setdefault(pick, []).append((action_idx, float(probability)))

    pick_rows = []
    for source_row in label["per_pick"]:
        pick = source_row["pick_domino_id"]
        actions = actions_by_pick.get(pick, [])
        total = sum(value for _idx, value in actions)
        if not actions or total <= 0.0:
            raise ValueError(f"Rust reply pick group {pick!r} has no baseline probability")
        conditional = [
            {"action_idx": idx, "conditional_probability": value / total}
            for idx, value in actions
        ]
        entropy = -sum(
            item["conditional_probability"]
            * math.log(max(item["conditional_probability"], 1e-300))
            for item in conditional
        )
        row = dict(source_row)
        row.update({
            "baseline_pick_probability": float(total),
            "baseline_conditional_placements": conditional,
            "baseline_within_group_entropy": float(entropy),
        })
        pick_rows.append(row)

    actor_values = [float(row["searched_value_actor"]) for row in pick_rows]
    errors = [float(row["mc_standard_error"]) for row in pick_rows]
    sorted_values = sorted(actor_values, reverse=True)
    top_margin = (
        float(sorted_values[0] - sorted_values[1])
        if len(sorted_values) > 1 else 2.0)
    quality = {
        "top_two_margin": top_margin,
        "max_mc_standard_error": max(errors, default=0.0),
        "target_entropy": float(-sum(
            probability * math.log(max(probability, 1e-300))
            for probability in label["denial_policy_target"] if probability > 0.0)),
        "exact_or_near_tie": top_margin <= 1e-6,
        "max_searched_seed_sd": label.get("cross_seed", {}).get(
            "max_searched_seed_sd"),
        "top_pick_agreement": label.get("cross_seed", {}).get(
            "top_pick_agreement"),
    }
    accepted, reasons = _quality_decision(
        quality, calibration=calibration,
        min_top_two_margin=min_top_two_margin,
        max_mc_standard_error=max_mc_standard_error,
        max_target_entropy=max_target_entropy,
        reject_ties=reject_ties,
        max_searched_seed_sd=max_searched_seed_sd,
        min_top_pick_agreement=min_top_pick_agreement,
    )
    state_hash = _encoded_state_hash(encoded, legal_indices)
    parent = "none" if label["parent_pick_domino_id"] is None else label["parent_pick_domino_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "state_backend": "rust-encoded-v1",
        "state_key": state_hash,
        "encoded_state_sha256": state_hash,
        "position_index": int(position_index),
        "root_state_key": str(root_state_key),
        "source": dict(source),
        "actor": actor,
        "root_actor": 1 - actor,
        "parent_pick_domino_id": label["parent_pick_domino_id"],
        "parent_representative": {
            "action_idx": int(label["parent_representative_action_idx"])},
        "parent_raw_prior": float(label["parent_raw_prior"]),
        "parent_searched_rank": int(label["parent_searched_rank"]),
        "parent_fragility": label["parent_fragility"],
        "chance_child_index": 0,
        "chance_weight": 1.0,
        "legal_actions": legal_rows,
        "legal_pick_ids": [row["pick_domino_id"] for row in pick_rows],
        "per_pick": pick_rows,
        "denial_policy_target": [float(value) for value in label["denial_policy_target"]],
        "quality": quality,
        "encoded_state": encoded,
        "quality_accept": bool(accepted),
        "quality_rejection_reasons": reasons,
        "cross_seed": label.get("cross_seed"),
        "example_id": f"p{position_index}:root-{parent}:chance-0:{state_hash}",
    }


def validate_reply_example(row: dict[str, Any]) -> None:
    if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("reply example schema mismatch")
    target = np.asarray(row["denial_policy_target"], dtype=np.float64)
    if target.ndim != 1 or not target.size or not np.isfinite(target).all():
        raise ValueError("invalid reply target")
    if (target < 0.0).any() or not math.isclose(float(target.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("reply target must be non-negative and normalized")
    picks = row["legal_pick_ids"]
    if len(picks) != len(target) or len(row["per_pick"]) != len(target):
        raise ValueError("pick rows and reply target are not aligned")
    if int(row["actor"]) == int(row["root_actor"]):
        raise ValueError("reply example is not an opponent-at-ply-1 state")
    cross = row.get("cross_seed")
    if cross is not None:
        if len(cross.get("seeds", [])) < 2:
            raise ValueError("cross-seed confirmation has fewer than two seeds")
        agreement = float(cross["top_pick_agreement"])
        seed_sd = [float(value) for value in cross["searched_seed_sd_by_pick"].values()]
        if not 0.0 <= agreement <= 1.0 or not seed_sd or any(
            not math.isfinite(value) or value < 0.0 for value in seed_sd
        ):
            raise ValueError("invalid cross-seed quality metrics")
        if not math.isclose(max(seed_sd), float(cross["max_searched_seed_sd"]),
                            abs_tol=1e-12):
            raise ValueError("cross-seed max SD does not match per-pick SDs")
        if (not math.isclose(float(row["quality"]["max_searched_seed_sd"]),
                             float(cross["max_searched_seed_sd"]), abs_tol=1e-12)
                or not math.isclose(float(row["quality"]["top_pick_agreement"]),
                                    agreement, abs_tol=1e-12)):
            raise ValueError("quality row and cross-seed metrics differ")

    legal_indices = [int(item["action_idx"]) for item in row["legal_actions"]]
    if legal_indices != sorted(set(legal_indices)):
        raise ValueError("legal action indices must be unique and sorted")
    legal_set = set(legal_indices)
    for pick_row in row["per_pick"]:
        conditional = pick_row["baseline_conditional_placements"]
        if not conditional:
            raise ValueError("pick group has no legal placement actions")
        indices = {int(item["action_idx"]) for item in conditional}
        if not indices.issubset(legal_set):
            raise ValueError("conditional placement action is not legal")
        total = sum(float(item["conditional_probability"]) for item in conditional)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("conditional placement probabilities do not sum to one")

    expected_shapes = {
        "my_board": (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE),
        "opp_board": (NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE),
        "flat": (FLAT_SIZE,),
    }
    for name, expected in expected_shapes.items():
        observed = decode_array_blob(row["encoded_state"][name])
        if observed.shape != expected or not np.isfinite(observed).all():
            raise ValueError(f"invalid encoded {name}: {observed.shape}")

    if "state" in row:
        from games.kingdomino.web_app import state_from_debug_json
        reconstructed = state_from_debug_json(row["state"])
        reconstructed.discards = [int(value) for value in row.get("discards", [0, 0])]
        if public_state_key(reconstructed) != row["state_key"]:
            raise ValueError("serialized reply state key mismatch")
        encoded = encode_reply_state(reconstructed)
        for name in expected_shapes:
            if not np.array_equal(
                decode_array_blob(encoded[name]), decode_array_blob(row["encoded_state"][name])
            ):
                raise ValueError(f"serialized state does not reproduce encoded {name}")
    elif row.get("state_backend") == "rust-encoded-v1":
        observed_hash = _encoded_state_hash(row["encoded_state"], legal_indices)
        if observed_hash != row.get("encoded_state_sha256") or observed_hash != row["state_key"]:
            raise ValueError("Rust encoded-state hash mismatch")
    else:
        raise ValueError("reply artifact has no reproducible state representation")


def _rust_primary_quality(label: dict[str, Any]) -> dict[str, Any]:
    actor_values = sorted(
        (float(row["searched_value_actor"]) for row in label["per_pick"]),
        reverse=True)
    margin = actor_values[0] - actor_values[1] if len(actor_values) > 1 else 2.0
    target = [float(value) for value in label["denial_policy_target"]]
    return {
        "top_two_margin": float(margin),
        "max_mc_standard_error": max(
            (float(row["mc_standard_error"]) for row in label["per_pick"]),
            default=0.0),
        "target_entropy": float(-sum(
            value * math.log(max(value, 1e-300)) for value in target if value > 0.0)),
        "exact_or_near_tie": margin <= 1e-6,
    }


def _primary_passes_locked_filter(label: dict[str, Any], args: argparse.Namespace) -> bool:
    quality = _rust_primary_quality(label)
    return (
        float(quality["top_two_margin"]) >= float(args.min_top_two_margin)
        and float(quality["max_mc_standard_error"]) <= float(args.max_mc_standard_error)
        and float(quality["target_entropy"]) <= float(args.max_target_entropy)
        and not (args.reject_ties and bool(quality["exact_or_near_tie"])))


def _confirm_rust_reply_label(
    search: DenialSearch, label: dict[str, Any], *, base_seed: int,
    seed_count: int, seed_stride: int, rayon_threads: int,
) -> dict[str, Any]:
    if seed_count < 2:
        raise ValueError("cross-seed confirmation requires at least two seeds")
    state = label.get("_rust_state")
    if state is None:
        raise TypeError("cross-seed confirmation requires a Rust reply state")
    by_seed = {}
    expected = {row["pick_domino_id"]: row for row in label["per_pick"]}
    for offset in range(seed_count):
        seed = int(base_seed) + offset * int(seed_stride)
        rows = search.search_rust_reply_state(
            state, seed=seed, rayon_threads=rayon_threads)
        indexed = {row["pick_domino_id"]: row for row in rows}
        if set(indexed) != set(expected):
            raise ValueError("reply pick IDs changed across confirmation seeds")
        if offset == 0:
            for pick in expected:
                if (abs(float(indexed[pick]["searched_value_actor"])
                        - float(expected[pick]["searched_value_actor"])) > 1e-6
                        or int(indexed[pick]["selected_placement_action_idx"])
                        != int(expected[pick]["selected_placement_action_idx"])):
                    raise ValueError(
                        "fixed-state confirmation failed to reproduce parent tree: "
                        f"pick={pick!r}, expected={expected[pick]!r}, "
                        f"observed={indexed[pick]!r}")
        by_seed[seed] = indexed

    def pick_order(pick):
        return -1 if pick is None else int(pick)

    top_by_seed = {
        seed: max(indexed, key=lambda pick: (
            float(indexed[pick]["searched_value_actor"]), -pick_order(pick)))
        for seed, indexed in by_seed.items()
    }
    seed_sd = {
        pick: float(np.std([
            float(by_seed[seed][pick]["searched_value_actor"]) for seed in by_seed
        ], ddof=0))
        for pick in expected
    }
    agreement = max(
        sum(candidate == value for value in top_by_seed.values())
        for candidate in set(top_by_seed.values())) / len(top_by_seed)
    return {
        "seeds": list(by_seed),
        "top_pick_by_seed": {str(seed): pick for seed, pick in top_by_seed.items()},
        "top_pick_agreement": float(agreement),
        "searched_seed_sd_by_pick": {
            "none" if pick is None else str(pick): value for pick, value in seed_sd.items()},
        "max_searched_seed_sd": max(seed_sd.values(), default=0.0),
    }


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("denial_search.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest_contract(args: argparse.Namespace, *, positions_sha: str,
                       checkpoint_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "kingdomino_opponent_reply_shard",
        "positions_sha256": positions_sha,
        "checkpoint_sha256": checkpoint_sha,
        "implementation_sha256": _implementation_sha256(),
        "actor_frame": ACTOR_FRAME,
        "engine": str(args.engine),
        "rayon_threads": int(args.rayon_threads),
        "search": {
            "pick_plies": int(args.pick_plies),
            "chance_k": int(args.chance_k),
            "seed": int(args.seed),
            "placement_top_k": int(args.placement_top_k),
            "root_search_sims": int(args.search_sims),
            "policy_temperature": float(args.policy_temperature),
            "tie_tolerance": float(args.tie_tolerance),
            "uncertainty_z": float(args.uncertainty_z),
        },
        "quality_filter": {
            "calibration": bool(args.calibration),
            "min_top_two_margin": args.min_top_two_margin,
            "max_mc_standard_error": args.max_mc_standard_error,
            "max_target_entropy": args.max_target_entropy,
            "reject_ties": bool(args.reject_ties),
            "max_searched_seed_sd": args.max_searched_seed_sd,
            "min_top_pick_agreement": args.min_top_pick_agreement,
            "confirmation_seeds": int(args.confirmation_seeds),
            "confirmation_seed_stride": int(args.confirmation_seed_stride),
        },
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }


def freeze_training_roots(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.positions_path)
    if output.exists() and not args.force:
        records = load_frozen_positions(output)
        if not all(reply_root_eligible(state) for state, _source in records):
            raise ValueError("existing training-root artifact contains an ineligible state")
        return {"path": str(output), "positions": len(records), "sha256": file_sha256(output)}

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)),
    )
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=SearchConfig(root_search_sims=args.trajectory_sims, seed=args.seed),
    )
    records = generate_az_midgame_positions(
        search, count=args.positions, seed=args.seed,
        min_deck=args.min_deck, max_deck=args.max_deck,
        position_filter=reply_root_eligible,
    )

    blocked: dict[str, str] = {}
    for blocked_path in (args.frozen_eval_path, args.reserved_test_path):
        if blocked_path and Path(blocked_path).exists():
            for state, _source in load_frozen_positions(blocked_path):
                blocked[public_state_key(state)] = str(blocked_path)
    collisions = [(public_state_key(state), blocked[public_state_key(state)])
                  for state, _source in records if public_state_key(state) in blocked]
    if collisions:
        raise ValueError(f"training-root leakage into held-out data: {collisions[:3]}")

    result = write_frozen_positions(records, output)
    result["eligible_opponent_reply_roots"] = len(records)
    result["checkpoint_sha256"] = sha256_file(args.checkpoint)
    _atomic_json(output.with_suffix(".manifest.json"), result)
    return result


def generate_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if not args.calibration and any(value is None for value in (
        args.min_top_two_margin, args.max_mc_standard_error, args.max_target_entropy,
        args.max_searched_seed_sd, args.min_top_pick_agreement,
    )):
        raise ValueError("production generation requires locked quality thresholds")
    if args.engine != "rust" and args.confirmation_seeds > 1:
        raise ValueError("cross-seed reply confirmation currently requires --engine rust")

    records = load_frozen_positions(args.positions_path)
    if not all(reply_root_eligible(state) for state, _source in records):
        raise ValueError("all training roots must hand ply 1 to the opponent")
    positions_sha = file_sha256(args.positions_path)
    checkpoint_sha = sha256_file(args.checkpoint)
    contract = _manifest_contract(
        args, positions_sha=positions_sha, checkpoint_sha=checkpoint_sha)
    shard_dir = Path(args.shards_dir)
    stem = f"shard-{args.shard_index:04d}-of-{args.num_shards:04d}"
    output = shard_dir / f"{stem}.jsonl"
    manifest_path = shard_dir / f"{stem}.manifest.json"
    target_indices = [index for index in range(len(records))
                      if index % args.num_shards == args.shard_index]

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in contract.items():
            if existing_manifest.get(key) != value:
                raise ValueError(f"existing shard manifest differs at {key}: {manifest_path}")
    rows = _read_jsonl(output)
    completed = {int(row["position_index"]) for row in rows}
    if len(completed) != len(rows) or not completed.issubset(target_indices):
        raise ValueError("existing shard has duplicate or out-of-range position rows")

    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)),
    )
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=SearchConfig(
            pick_plies=args.pick_plies, chance_k=args.chance_k, seed=args.seed,
            placement_top_k=args.placement_top_k, root_search_sims=args.search_sims,
            policy_temperature=args.policy_temperature,
            tie_tolerance=args.tie_tolerance, uncertainty_z=args.uncertainty_z,
        ),
    )

    started = time.perf_counter()
    for index in target_indices:
        if index in completed:
            continue
        state, source = records[index]
        root_result = search._root_search(
            state, cache_namespace=f"reply-shard-{args.shard_index}")
        if args.engine == "rust":
            rust_result = search.search_position_rust(
                state, root_result=root_result, rayon_threads=args.rayon_threads)
            labels = rust_result["reply_labels"]
            for label in labels:
                if (args.calibration or _primary_passes_locked_filter(label, args)):
                    label["cross_seed"] = _confirm_rust_reply_label(
                        search, label, base_seed=args.seed,
                        seed_count=args.confirmation_seeds,
                        seed_stride=args.confirmation_seed_stride,
                        rayon_threads=args.rayon_threads)
            examples = [serialize_rust_reply_example(
                label, rust_evaluator=search._rust_evaluator,
                position_index=index, root_state_key=public_state_key(state),
                source=source, calibration=args.calibration,
                min_top_two_margin=args.min_top_two_margin,
                max_mc_standard_error=args.max_mc_standard_error,
                max_target_entropy=args.max_target_entropy,
                reject_ties=args.reject_ties,
                max_searched_seed_sd=args.max_searched_seed_sd,
                min_top_pick_agreement=args.min_top_pick_agreement,
            ) for label in labels]
        else:
            root_label = search.search_position(state, root_result=root_result)
            labels = search.extract_reply_labels(state, root_label=root_label)
            examples = [serialize_reply_example(
                label, position_index=index, root_state_key=public_state_key(state),
                source=source, calibration=args.calibration,
                min_top_two_margin=args.min_top_two_margin,
                max_mc_standard_error=args.max_mc_standard_error,
                max_target_entropy=args.max_target_entropy,
                reject_ties=args.reject_ties,
            ) for label in labels]
        for example in examples:
            validate_reply_example(example)
        _append_jsonl(output, {
            "position_index": index,
            "root_state_key": public_state_key(state),
            "examples": examples,
        })
        completed.add(index)
        _atomic_json(manifest_path, {
            **contract,
            "target_indices": target_indices,
            "completed_indices": sorted(completed),
            "complete": completed == set(target_indices),
        })
        print(f"reply shard {args.shard_index}: {len(completed)}/{len(target_indices)} roots",
              flush=True)

    all_rows = _read_jsonl(output)
    examples = [example for row in all_rows for example in row["examples"]]
    manifest = {
        **contract,
        "target_indices": target_indices,
        "completed_indices": sorted(completed),
        "complete": completed == set(target_indices),
        "root_rows": len(all_rows),
        "reply_examples": len(examples),
        "accepted_examples": sum(bool(row["quality_accept"]) for row in examples),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _merge_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    ignored = {
        "shard_index", "target_indices", "completed_indices", "complete",
        "root_rows", "reply_examples", "accepted_examples",
        "elapsed_seconds_this_invocation", "output", "output_sha256",
    }
    return {key: value for key, value in manifest.items() if key not in ignored}


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    shard_dir = Path(args.shards_dir)
    manifests = sorted(shard_dir.glob("shard-*-of-*.manifest.json"))
    if not manifests:
        raise ValueError(f"no shard manifests found in {shard_dir}")
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    if not all(bool(manifest.get("complete")) for manifest in loaded):
        raise ValueError("cannot merge incomplete shards")
    first_contract = _merge_contract(loaded[0])
    if any(_merge_contract(manifest) != first_contract for manifest in loaded[1:]):
        raise ValueError("shard provenance/configuration mismatch")
    num_shards = int(loaded[0]["num_shards"])
    indices = sorted(int(manifest["shard_index"]) for manifest in loaded)
    if indices != list(range(num_shards)) or len(loaded) != num_shards:
        raise ValueError(f"expected exactly shard indices 0..{num_shards - 1}, got {indices}")

    root_rows = []
    for path, manifest in zip(manifests, loaded):
        data_path = Path(manifest.get("output", path.with_suffix("").with_suffix(".jsonl")))
        if not data_path.is_absolute():
            data_path = Path(data_path)
        if file_sha256(data_path) != manifest["output_sha256"]:
            raise ValueError(f"shard data hash mismatch: {data_path}")
        root_rows.extend(_read_jsonl(data_path))
    root_rows.sort(key=lambda row: int(row["position_index"]))
    root_indices = [int(row["position_index"]) for row in root_rows]
    if root_indices != list(range(len(root_rows))):
        raise ValueError("merged root positions are missing or duplicated")

    examples = [example for row in root_rows for example in row["examples"]]
    examples.sort(key=lambda row: (
        int(row["position_index"]),
        -1 if row["parent_pick_domino_id"] is None else int(row["parent_pick_domino_id"]),
        int(row["chance_child_index"]), str(row["state_key"]),
    ))
    ids = [row["example_id"] for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate merged reply example IDs")
    for row in examples:
        validate_reply_example(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    selected = [row for row in examples if row["quality_accept"]] if args.accepted_only else examples
    temporary.write_text("".join(_json_dump(row) + "\n" for row in selected), encoding="utf-8")
    temporary.replace(output)
    manifest = {
        **first_contract,
        "kind": "kingdomino_opponent_reply_merged",
        "source_shards": num_shards,
        "root_rows": len(root_rows),
        "reply_examples_before_filter": len(examples),
        "reply_examples": len(selected),
        "accepted_only": bool(args.accepted_only),
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    _atomic_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def validate_artifact(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.output)
    rows = _read_jsonl(path)
    for row in rows:
        validate_reply_example(row)
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("artifact contains duplicate example IDs")
    return {
        "path": str(path), "sha256": file_sha256(path), "examples": len(rows),
        "accepted": sum(bool(row["quality_accept"]) for row in rows),
    }


def summarize_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Summarize calibration labels and emit a reproducible filter proposal.

    The proposal splits the allowed rejection rate evenly over the five
    continuous/discrete guards, then measures the actual joint retention after also
    rejecting exact/near ties.  Production still requires the resulting
    numeric thresholds to be supplied explicitly.
    """
    rows = _read_jsonl(Path(args.input))
    if not rows:
        raise ValueError("cannot summarize an empty calibration artifact")
    for row in rows:
        validate_reply_example(row)
        if row.get("quality_rejection_reasons") != ["calibration_only"]:
            raise ValueError("summarize requires calibration rows")
    if not 0.0 < args.target_retention < 1.0:
        raise ValueError("target_retention must be in (0,1)")

    names = ("top_two_margin", "max_mc_standard_error", "target_entropy",
             "max_searched_seed_sd", "top_pick_agreement")
    if any(row["quality"].get(name) is None for row in rows for name in names):
        raise ValueError("calibration rows are missing cross-seed quality metrics")
    values = {
        name: np.asarray([float(row["quality"][name]) for row in rows], dtype=np.float64)
        for name in names
    }
    percentiles = (0, 10, 25, 50, 75, 90, 100)
    distributions = {
        name: {f"p{q}": float(np.percentile(array, q)) for q in percentiles}
        for name, array in values.items()
    }
    tail = (1.0 - float(args.target_retention)) / len(names)
    proposal = {
        "min_top_two_margin": float(np.quantile(values["top_two_margin"], tail)),
        "max_mc_standard_error": float(np.quantile(
            values["max_mc_standard_error"], 1.0 - tail)),
        "max_target_entropy": float(np.quantile(values["target_entropy"], 1.0 - tail)),
        "max_searched_seed_sd": float(np.quantile(
            values["max_searched_seed_sd"], 1.0 - tail)),
        "min_top_pick_agreement": float(np.quantile(
            values["top_pick_agreement"], tail)),
        "reject_ties": True,
    }
    is_top_tie = lambda row: (
        float(row["quality"]["top_two_margin"]) <= float(args.tie_tolerance))
    accepted = [
        row for row in rows
        if (not is_top_tie(row)
            and float(row["quality"]["top_two_margin"]) >= proposal["min_top_two_margin"]
            and float(row["quality"]["max_mc_standard_error"])
                <= proposal["max_mc_standard_error"]
            and float(row["quality"]["target_entropy"])
                <= proposal["max_target_entropy"]
            and float(row["quality"]["max_searched_seed_sd"])
                <= proposal["max_searched_seed_sd"]
            and float(row["quality"]["top_pick_agreement"])
                >= proposal["min_top_pick_agreement"])
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "kingdomino_opponent_reply_calibration_summary",
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "examples": len(rows),
        "exact_or_near_ties": sum(is_top_tie(row) for row in rows),
        "distributions": distributions,
        "target_retention": float(args.target_retention),
        "proposed_filter": proposal,
        "joint_accepted_examples": len(accepted),
        "joint_retention": len(accepted) / len(rows),
    }
    output = Path(args.output)
    if output == DEFAULT_MERGED:
        output = DEFAULT_DIR / "calibration_summary.json"
    _atomic_json(output, result)
    return result


def split_artifact(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(Path(args.input))
    if not rows:
        raise ValueError("cannot split an empty reply artifact")
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        validate_reply_example(row)
        groups.setdefault(int(row["position_index"]), []).append(row)
    if len(groups) < 2:
        raise ValueError("reply train/validation split requires at least two root positions")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    ranked = sorted(groups, key=lambda index: hashlib.sha256(
        f"{args.split_seed}:{index}:{groups[index][0]['root_state_key']}".encode("ascii")
    ).digest())
    validation_count = min(len(ranked) - 1, max(1, round(
        len(ranked) * args.validation_fraction)))
    validation_roots = set(ranked[:validation_count])
    train = [row for index in sorted(groups) if index not in validation_roots
             for row in groups[index]]
    validation = [row for index in sorted(groups) if index in validation_roots
                  for row in groups[index]]
    train_keys = {row["root_state_key"] for row in train}
    validation_keys = {row["root_state_key"] for row in validation}
    if train_keys & validation_keys:
        raise AssertionError("root state leaked across reply train/validation split")

    def write(rows_to_write: list[dict[str, Any]], output: Path, role: str) -> dict[str, Any]:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text("".join(_json_dump(row) + "\n" for row in rows_to_write),
                             encoding="utf-8")
        temporary.replace(output)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "kingdomino_opponent_reply_split",
            "role": role,
            "source": str(args.input),
            "source_sha256": file_sha256(args.input),
            "split_seed": int(args.split_seed),
            "validation_fraction": float(args.validation_fraction),
            "root_positions": len({row["position_index"] for row in rows_to_write}),
            "examples": len(rows_to_write),
            "output": str(output),
            "output_sha256": file_sha256(output),
        }
        source_manifest = Path(args.input).with_suffix(".manifest.json")
        if source_manifest.exists():
            source_data = json.loads(source_manifest.read_text(encoding="utf-8"))
            manifest["checkpoint_sha256"] = source_data.get("checkpoint_sha256")
            manifest["positions_sha256"] = source_data.get("positions_sha256")
        _atomic_json(output.with_suffix(".manifest.json"), manifest)
        return manifest

    return {
        "train": write(train, Path(args.train_output), "train"),
        "validation": write(validation, Path(args.validation_output), "validation"),
        "root_state_overlap": 0,
    }


def _compare_tree_outputs(python_label: dict[str, Any], python_replies: list[dict[str, Any]],
                          rust_label: dict[str, Any]) -> dict[str, Any]:
    def by_pick(rows):
        return {row["pick_domino_id"]: row for row in rows}

    value_deltas = []
    stderr_deltas = []
    action_mismatches = 0
    left = by_pick(python_label["per_pick"])
    right = by_pick(rust_label["per_pick"])
    if left.keys() != right.keys():
        raise ValueError("Python/Rust root pick sets differ")
    for pick in left:
        value_deltas.append(abs(float(left[pick]["searched_value_player0"])
                                - float(right[pick]["searched_value_player0"])))
        stderr_deltas.append(abs(float(left[pick]["mc_standard_error"])
                                 - float(right[pick]["mc_standard_error"])))
        action_mismatches += (
            int(left[pick]["representative"]["action_idx"])
            != int(right[pick]["representative_action_idx"])
        )
    left_replies = {row["parent_pick_domino_id"]: row for row in python_replies}
    right_replies = {row["parent_pick_domino_id"]: row for row in rust_label["reply_labels"]}
    if left_replies.keys() != right_replies.keys():
        raise ValueError("Python/Rust reply parent-pick sets differ")
    target_deltas = []
    for parent in left_replies:
        left_rows = by_pick(left_replies[parent]["per_pick"])
        right_rows = by_pick(right_replies[parent]["per_pick"])
        if left_rows.keys() != right_rows.keys():
            raise ValueError("Python/Rust reply pick sets differ")
        for pick in left_rows:
            value_deltas.append(abs(float(left_rows[pick]["searched_value_player0"])
                                    - float(right_rows[pick]["searched_value_player0"])))
            stderr_deltas.append(abs(float(left_rows[pick]["mc_standard_error"])
                                     - float(right_rows[pick]["mc_standard_error"])))
            action_mismatches += (
                int(left_rows[pick]["selected_placement_action_idx"])
                != int(right_rows[pick]["selected_placement_action_idx"])
            )
        target_deltas.extend(abs(float(a) - float(b)) for a, b in zip(
            left_replies[parent]["denial_policy_target"],
            right_replies[parent]["denial_policy_target"],
        ))
    return {
        "max_abs_value_delta": max(value_deltas, default=0.0),
        "max_abs_stderr_delta": max(stderr_deltas, default=0.0),
        "max_abs_target_delta": max(target_deltas, default=0.0),
        "selected_action_mismatches": int(action_mismatches),
    }


def benchmark_engines(args: argparse.Namespace) -> dict[str, Any]:
    records = [record for record in load_frozen_positions(args.positions_path)
               if reply_root_eligible(record[0])][:args.benchmark_limit]
    if not records:
        raise ValueError("benchmark input has no eligible opponent-at-ply-1 roots")
    net, checkpoint_cfg = load_checkpoint_network(args.checkpoint, args.device)
    evaluator = AZBatchEvaluator(
        net, device=args.device, batch_size=args.leaf_batch_size,
        margin_gain=float(checkpoint_cfg.get("margin_gain", 2.0)),
        alpha=float(checkpoint_cfg.get("alpha", 0.5)),
    )
    search = DenialSearch(
        evaluator, checkpoint_path=args.checkpoint,
        config=SearchConfig(
            pick_plies=args.pick_plies, chance_k=args.chance_k, seed=args.seed,
            placement_top_k=args.placement_top_k, root_search_sims=args.search_sims,
            policy_temperature=args.policy_temperature,
            tie_tolerance=args.tie_tolerance, uncertainty_z=args.uncertainty_z,
        ),
    )
    threads = [int(value) for value in args.benchmark_threads.split(",") if value.strip()]
    if not threads or any(value < 1 for value in threads):
        raise ValueError("benchmark threads must be positive comma-separated integers")
    rows = []
    for index, (state, _source) in enumerate(records):
        root_result = search._root_search(state, cache_namespace="reply-benchmark")
        search._node_tt.clear()
        started = time.perf_counter()
        python_label = search.search_position(state, root_result=root_result)
        python_replies = search.extract_reply_labels(state, root_label=python_label)
        python_seconds = time.perf_counter() - started
        per_thread = {}
        normalized = None
        for count in threads:
            started = time.perf_counter()
            rust_label = search.search_position_rust(
                state, root_result=root_result, rayon_threads=count)
            elapsed = time.perf_counter() - started
            comparison = _compare_tree_outputs(python_label, python_replies, rust_label)
            if (comparison["max_abs_value_delta"] > args.equivalence_tolerance
                    or comparison["max_abs_stderr_delta"] > args.equivalence_tolerance
                    or comparison["max_abs_target_delta"] > args.equivalence_tolerance
                    or comparison["selected_action_mismatches"]):
                raise ValueError(f"Python/Rust equivalence failed: {comparison}")
            deterministic = {
                "per_pick": rust_label["per_pick"],
                "reply_labels": [
                    {key: value for key, value in reply.items() if key != "_rust_state"}
                    for reply in rust_label["reply_labels"]
                ],
                "structure": {key: value for key, value in rust_label["structure"].items()
                              if key != "rayon_threads"},
            }
            if normalized is None:
                normalized = deterministic
            elif deterministic != normalized:
                raise ValueError("Rayon thread count changed deterministic Rust output")
            per_thread[str(count)] = {
                "elapsed_seconds": elapsed,
                "speedup_vs_python": python_seconds / max(elapsed, 1e-12),
                "comparison": comparison,
                "structure": rust_label["structure"],
            }
        rows.append({
            "position": index,
            "state_key": public_state_key(state),
            "python_elapsed_seconds": python_seconds,
            "rust": per_thread,
        })
        print(f"benchmark {index + 1}/{len(records)}", flush=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "positions_sha256": file_sha256(args.positions_path),
        "positions": len(records),
        "pick_plies": args.pick_plies,
        "chance_k": args.chance_k,
        "threads": threads,
        "rows": rows,
        "mean_python_seconds": float(np.mean([row["python_elapsed_seconds"] for row in rows])),
        "mean_rust_seconds": {
            str(count): float(np.mean([
                row["rust"][str(count)]["elapsed_seconds"] for row in rows]))
            for count in threads
        },
    }
    result["speedup_vs_python"] = {
        str(count): result["mean_python_seconds"] / max(result["mean_rust_seconds"][str(count)], 1e-12)
        for count in threads
    }
    output = Path(args.output)
    if output == DEFAULT_MERGED:
        output = DEFAULT_DIR / "engine_benchmark.json"
    _atomic_json(output, result)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "generate", "merge", "split", "validate",
                                           "summarize", "benchmark"),
                        required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CURRENT_BEST))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positions-path", default=str(DEFAULT_ROOTS))
    parser.add_argument("--positions", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--trajectory-sims", type=int, default=3200)
    parser.add_argument("--min-deck", type=int, default=8)
    parser.add_argument("--max-deck", type=int, default=28)
    parser.add_argument("--frozen-eval-path", default=str(DEFAULT_FROZEN_EVAL))
    parser.add_argument("--reserved-test-path", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shards-dir", default=str(DEFAULT_SHARDS))
    parser.add_argument("--engine", choices=("python", "rust"), default="rust")
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pick-plies", type=int, default=8)
    parser.add_argument("--chance-k", type=int, default=16)
    parser.add_argument("--placement-top-k", type=int, default=2)
    parser.add_argument("--search-sims", type=int, default=3200)
    parser.add_argument("--leaf-batch-size", type=int, default=512)
    parser.add_argument("--policy-temperature", type=float, default=0.10)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--uncertainty-z", type=float, default=1.0)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--min-top-two-margin", type=float)
    parser.add_argument("--max-mc-standard-error", type=float)
    parser.add_argument("--max-target-entropy", type=float)
    parser.add_argument("--max-searched-seed-sd", type=float)
    parser.add_argument("--min-top-pick-agreement", type=float)
    parser.add_argument("--confirmation-seeds", type=int, default=3)
    parser.add_argument("--confirmation-seed-stride", type=int, default=1_000_003)
    parser.add_argument("--reject-ties", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_MERGED))
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--input", default=str(DEFAULT_MERGED))
    parser.add_argument("--train-output", default=str(DEFAULT_DIR / "reply_train.jsonl"))
    parser.add_argument("--validation-output", default=str(DEFAULT_DIR / "reply_validation.jsonl"))
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20_260_719)
    parser.add_argument("--target-retention", type=float, default=0.75)
    parser.add_argument("--benchmark-limit", type=int, default=10)
    parser.add_argument("--benchmark-threads", default="1,2,4,8")
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.mode == "freeze":
        result = freeze_training_roots(args)
    elif args.mode == "generate":
        result = generate_shard(args)
    elif args.mode == "merge":
        result = merge_shards(args)
    elif args.mode == "split":
        result = split_artifact(args)
    elif args.mode == "validate":
        result = validate_artifact(args)
    elif args.mode == "summarize":
        result = summarize_calibration(args)
    else:
        result = benchmark_engines(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
