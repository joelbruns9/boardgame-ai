"""Run the frozen, resumable eight-position deck-8 A1c oracle suite.

Validation is deliberately front-loaded: every oracle cell, frozen-input hash,
boundary identity, seed, and execution order is checked before the network is
loaded.  Position artifacts are the resume unit and are only written after all
repeats for that position complete successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any

from games.kingdomino.action_codec import encode_action
from games.kingdomino.deck8_oracle import (
    EXPECTED_ROWS,
    _stable_digest,
    apply_prefix_actions,
    conditioned_tails,
    validate_boundary_state,
)
from games.kingdomino.deck8_oracle_corpus import _load_oracle
from games.kingdomino.denial_search import (
    chance_public_state_key_v1,
    public_state_key,
)
from games.kingdomino.denial_signal_sweep import file_sha256, load_frozen_positions


CONFIG_SCHEMA = "kd-deck8-oracle-a1c-suite-config-v1"
RESOLVED_SCHEMA = "kd-deck8-oracle-a1c-suite-resolved-v1"
POSITION_SCHEMA = "kd-deck8-oracle-a1c-suite-position-v1"
SUMMARY_SCHEMA = "kd-deck8-oracle-a1c-suite-summary-v1"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "deck8_oracle_a1c_suite_v1.json"
DEFAULT_OUTPUT = Path(
    "runs/kingdomino/chance_correct_a1/deck8_oracle_a1c_suite_v1"
)
EXPECTED_ARM_NAMES = [
    "x0",
    "x1_hajek_balanced",
    "a1c_x4_balanced",
    "a1c_x4_iid",
    "a1c_x8_balanced",
    "a1c_x8_iid",
]
CONTROL_NAMES = ["x0", "x1_hajek_balanced"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed or unreadable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name != "configs":
        raise ValueError(f"suite config must live in games/kingdomino/configs: {resolved}")
    return resolved.parents[3]


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def derive_seed(base_seed: int, position_ordinal: int, repeat: int) -> int:
    return int(base_seed) + 100 * int(position_ordinal) + int(repeat)


def global_arm_order(
    arm_names: list[str], position_ordinal: int, repeat: int, repeat_count: int
) -> list[str]:
    if not arm_names:
        return []
    rotation = (position_ordinal * repeat_count + repeat) % len(arm_names)
    return arm_names[rotation:] + arm_names[:rotation]


def validate_config(config: dict[str, Any]) -> None:
    """Validate immutable experiment choices before touching any input files."""
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("suite configuration schema mismatch")
    oracles = config.get("oracles")
    if not isinstance(oracles, list) or len(oracles) != 8:
        raise ValueError("suite must contain exactly eight ordered oracles")
    positions = [int(row["position_index"]) for row in oracles]
    oracle_ids = [str(row["expected_oracle_id"]) for row in oracles]
    if len(set(positions)) != len(positions):
        raise ValueError("duplicate position IDs in suite configuration")
    if len(set(oracle_ids)) != len(oracle_ids):
        raise ValueError("duplicate oracle IDs in suite configuration")
    if positions != sorted(positions):
        raise ValueError("oracles must be ordered by position index")
    for row in oracles:
        if int(row["legal_actions"]) <= 0:
            raise ValueError("oracle legal-action count must be positive")
        for field in ("expected_oracle_id", "expected_sha256"):
            value = str(row[field])
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"invalid lowercase SHA-256 in {field}")

    search = config.get("search", {})
    if int(search.get("nn_eval_budget", 0)) != 4801:
        raise ValueError("sealed suite NN budget must be 4801")
    if int(search.get("simulation_ceiling", 0)) != 9600:
        raise ValueError("sealed suite simulation ceiling must be 9600")
    if int(search.get("repeat_count", 0)) != 8:
        raise ValueError("sealed suite repeat count must be 8")
    arms = search.get("arms")
    if not isinstance(arms, list):
        raise ValueError("search arms must be a list")
    names = [str(row["name"]) for row in arms]
    if names != EXPECTED_ARM_NAMES:
        raise ValueError(f"suite arm order/configuration mismatch: {names}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    for arm in arms:
        if int(arm["leaf_batch"]) != 8:
            raise ValueError("every suite arm must use leaf_batch=8")
        if str(arm["name"]).startswith("a1c_") and (
            int(arm["enum_max_rows"]) != 1 or str(arm["panel_mode"]) != "a1c"
        ):
            raise ValueError("A1c arms must force sampled deck-8 panels")
    if int(config.get("bootstrap", {}).get("resamples", 0)) < 20000:
        raise ValueError("clustered bootstrap requires at least 20,000 resamples")


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _position_identity(
    *,
    position_index: int,
    oracle_id: str,
    prefix_actions: list[int],
    root_state_key: str,
    boundary_state_key: str,
) -> str:
    return _digest(
        {
            "position_index": position_index,
            "oracle_id": oracle_id,
            "prefix_actions": prefix_actions,
            "root_state_key": root_state_key,
            "boundary_state_key": boundary_state_key,
        }
    )


def validate_suite_inputs(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate all CPU-side input and oracle evidence and return provenance."""
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_config(config)
    root = _repo_root(config_path)
    positions_path = _resolve(root, str(config["positions_path"]))
    checkpoint_path = _resolve(root, str(config["checkpoint_path"]))
    if not positions_path.is_file():
        raise FileNotFoundError(f"frozen positions missing: {positions_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint_path}")
    positions_hash = file_sha256(positions_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    positions = load_frozen_positions(positions_path)
    resolved_oracles = []

    for ordinal, spec in enumerate(config["oracles"]):
        summary_path = _resolve(root, str(spec["summary_path"]))
        summary_hash = file_sha256(summary_path)
        if summary_hash != str(spec["expected_sha256"]):
            raise ValueError(f"oracle summary hash mismatch: {summary_path}")
        summary = _read_json(summary_path)
        position_index = int(spec["position_index"])
        oracle_id = str(spec["expected_oracle_id"])
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete oracle summary: {summary_path}")
        if summary.get("oracle_id") != oracle_id:
            raise ValueError(f"oracle ID mismatch: {summary_path}")
        if int(summary.get("position_index", -1)) != position_index:
            raise ValueError(f"oracle position mismatch: {summary_path}")
        legal_actions = int(spec["legal_actions"])
        if int(summary.get("legal_actions", -1)) != legal_actions or int(
            summary.get("complete_actions", -1)
        ) != legal_actions:
            raise ValueError(f"oracle legal/complete action mismatch: {summary_path}")
        actions = summary.get("actions", [])
        if len(actions) != legal_actions or any(
            row.get("complete") is not True
            or int(row.get("rows_solved", -1)) != EXPECTED_ROWS
            or int(row.get("rows_expected", -1)) != EXPECTED_ROWS
            for row in actions
        ):
            raise ValueError(f"oracle summary lacks 70 solved rows per action: {summary_path}")
        if position_index >= len(positions):
            raise ValueError(f"oracle position outside frozen source: {position_index}")

        oracle_manifest_path = summary_path.with_name("manifest.json")
        oracle_manifest = _read_json(oracle_manifest_path)
        identity = oracle_manifest.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"oracle manifest identity missing: {oracle_manifest_path}")
        digest_identity = dict(identity)
        stored_id = digest_identity.pop("oracle_id", None)
        if stored_id != oracle_id or _stable_digest(digest_identity) != oracle_id:
            raise ValueError(f"oracle identity digest mismatch: {oracle_manifest_path}")
        prefix = [int(value) for value in summary["prefix_actions"]]
        if (
            int(identity.get("position_index", -1)) != position_index
            or [int(value) for value in identity.get("prefix_actions", [])] != prefix
            or identity.get("positions_sha256") != positions_hash
        ):
            raise ValueError(f"oracle manifest frozen-source mismatch: {oracle_manifest_path}")

        root_state, _source = positions[position_index]
        boundary = apply_prefix_actions(root_state, prefix)
        validate_boundary_state(boundary)
        root_key = public_state_key(root_state)
        boundary_key = public_state_key(boundary)
        boundary_chance_key = chance_public_state_key_v1(boundary).hex()
        if (
            identity.get("root_state_key") != root_key
            or identity.get("boundary_state_key") != boundary_key
            or identity.get("boundary_chance_key_hex") != boundary_chance_key
        ):
            raise ValueError(f"oracle boundary reconstruction mismatch: {summary_path}")
        action_indices = sorted(encode_action(action, boundary) for action in boundary.legal_actions())
        summary_indices = sorted(int(row["action_idx"]) for row in actions)
        if action_indices != summary_indices or len(action_indices) != legal_actions:
            raise ValueError(f"oracle action support mismatch: {summary_path}")
        if int(summary["boundary_actor"]) != int(boundary.current_actor):
            raise ValueError(f"oracle actor-frame mismatch: {summary_path}")

        # This validates every cell's identity, exact mean, actor-frame sign,
        # common 70-row support, and finite solved value.
        loaded = _load_oracle(summary_path)
        support = set(next(iter(loaded["cells"].values())))
        expected_support = set(itertools.combinations(sorted(int(x) for x in boundary.deck), 4))
        if support != expected_support:
            raise ValueError(f"oracle cell support does not match reconstructed deck: {summary_path}")
        expected_cells = {}
        for action in boundary.legal_actions():
            action_idx = int(encode_action(action, boundary))
            futures = conditioned_tails(boundary, action)
            if {tuple(child.current_row) for child, _ in futures} != expected_support:
                raise ValueError(
                    f"oracle conditioned-tail support mismatch: {summary_path}"
                )
            for child, probability in futures:
                row = tuple(int(value) for value in child.current_row)
                expected_cells[(action_idx, row)] = {
                    "tail_state_key": public_state_key(child),
                    "tail_chance_key_hex": chance_public_state_key_v1(child).hex(),
                    "probability": float(probability),
                }
        seen_cells = set()
        for cell_path in (summary_path.parent / "cells").glob("*.json"):
            cell = _read_json(cell_path)
            key = (
                int(cell.get("action_idx", -1)),
                tuple(int(value) for value in cell.get("row", [])),
            )
            expected = expected_cells.get(key)
            if expected is None or key in seen_cells:
                raise ValueError(f"oracle cell identity mismatch: {cell_path}")
            seen_cells.add(key)
            if (
                cell.get("oracle_id") != oracle_id
                or cell.get("solved") is not True
                or cell.get("tail_state_key") != expected["tail_state_key"]
                or cell.get("tail_chance_key_hex")
                != expected["tail_chance_key_hex"]
                or not math.isclose(
                    float(cell.get("probability", math.nan)),
                    expected["probability"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"oracle cell solve/state mismatch: {cell_path}")
        if seen_cells != set(expected_cells):
            raise ValueError(f"oracle cell set is incomplete: {summary_path}")

        resolved_oracles.append(
            {
                "ordinal": ordinal,
                "position_index": position_index,
                "legal_actions": legal_actions,
                "summary_path": str(summary_path),
                "summary_sha256": summary_hash,
                "oracle_manifest_path": str(oracle_manifest_path),
                "oracle_manifest_sha256": file_sha256(oracle_manifest_path),
                "oracle_id": oracle_id,
                "prefix_actions": prefix,
                "root_state_key": root_key,
                "boundary_state_key": boundary_key,
                "position_sha256": _position_identity(
                    position_index=position_index,
                    oracle_id=oracle_id,
                    prefix_actions=prefix,
                    root_state_key=root_key,
                    boundary_state_key=boundary_key,
                ),
            }
        )

    source_paths = {
        "suite_runner": Path(__file__).resolve(),
        "single_position_comparator": Path(__file__).with_name(
            "deck8_oracle_compare.py"
        ).resolve(),
        "rust_search_source": Path(__file__).with_name("kingdomino_rust")
        / "src"
        / "lib.rs",
    }
    search = config["search"]
    repeat_count = int(search["repeat_count"])
    arm_names = [str(arm["name"]) for arm in search["arms"]]
    cases = []
    for ordinal, oracle in enumerate(resolved_oracles):
        for repeat in range(repeat_count):
            cases.append(
                {
                    "position_ordinal": ordinal,
                    "position_index": oracle["position_index"],
                    "repeat": repeat,
                    "seed": derive_seed(
                        int(config["seeding"]["base_seed"]), ordinal, repeat
                    ),
                    "arm_execution_order": global_arm_order(
                        arm_names, ordinal, repeat, repeat_count
                    ),
                }
            )
    return {
        "schema_version": RESOLVED_SCHEMA,
        "suite_config_path": str(config_path),
        "suite_config_sha256": file_sha256(config_path),
        "suite_config": config,
        "git_commit": _git_commit(root),
        "repo_root": str(root),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        },
        "positions": {"path": str(positions_path), "sha256": positions_hash},
        "sources": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in source_paths.items()
        },
        "oracles": resolved_oracles,
        "cases": cases,
        "selection_limitation": config["selection_reason"],
    }


def _write_or_verify_resolved(output_dir: Path, resolved: dict[str, Any]) -> None:
    path = output_dir / "manifest.resolved.json"
    if path.exists():
        existing = _read_json(path)
        if existing != resolved:
            raise ValueError(f"incompatible resolved provenance in output directory: {path}")
    else:
        _atomic_json(path, resolved)


def _position_provenance(
    resolved: dict[str, Any], oracle: dict[str, Any]
) -> dict[str, Any]:
    config = resolved["suite_config"]
    repeat_count = int(config["search"]["repeat_count"])
    arm_names = [row["name"] for row in config["search"]["arms"]]
    ordinal = int(oracle["ordinal"])
    return {
        "suite_config_sha256": resolved["suite_config_sha256"],
        "checkpoint_path": resolved["checkpoint"]["path"],
        "checkpoint_sha256": resolved["checkpoint"]["sha256"],
        "positions_path": resolved["positions"]["path"],
        "positions_sha256": resolved["positions"]["sha256"],
        "oracle_path": oracle["summary_path"],
        "oracle_sha256": oracle["summary_sha256"],
        "oracle_id": oracle["oracle_id"],
        "position_sha256": oracle["position_sha256"],
        "search_configuration": config["search"],
        "seeds": [
            derive_seed(int(config["seeding"]["base_seed"]), ordinal, repeat)
            for repeat in range(repeat_count)
        ],
        "arm_execution_orders": [
            global_arm_order(arm_names, ordinal, repeat, repeat_count)
            for repeat in range(repeat_count)
        ],
    }


def validate_position_artifact(
    artifact_path: Path,
    expected_provenance: dict[str, Any],
    *,
    position_index: int,
) -> dict[str, Any]:
    artifact = _read_json(artifact_path)
    if artifact.get("schema_version") != POSITION_SCHEMA:
        raise ValueError(f"position artifact schema mismatch: {artifact_path}")
    if artifact.get("provenance") != expected_provenance:
        raise ValueError(f"position artifact provenance mismatch: {artifact_path}")
    if int(artifact.get("position_index", -1)) != position_index:
        raise ValueError(f"position artifact identity mismatch: {artifact_path}")
    records = artifact.get("records")
    seeds = expected_provenance["seeds"]
    orders = expected_provenance["arm_execution_orders"]
    if not isinstance(records, list) or len(records) != len(seeds):
        raise ValueError(f"position artifact repeat count mismatch: {artifact_path}")
    budget = int(expected_provenance["search_configuration"]["nn_eval_budget"])
    expected_arms = [row["name"] for row in expected_provenance["search_configuration"]["arms"]]
    for repeat, record in enumerate(records):
        if int(record.get("seed", -1)) != seeds[repeat] or record.get(
            "arm_execution_order"
        ) != orders[repeat]:
            raise ValueError(f"position artifact seed/order mismatch: {artifact_path}")
        arms = record.get("arms", {})
        if set(arms) != set(expected_arms):
            raise ValueError(f"position artifact arm set mismatch: {artifact_path}")
        for name, row in arms.items():
            if (
                int(row.get("nn_evaluations", -1)) != budget
                or row.get("nn_eval_budget_hit") is not True
                or row.get("simulation_limit_hit") is not False
                or row.get("python_rust_nn_accounting_match") is not True
            ):
                raise ValueError(f"invalid NN budget/accounting for {name}: {artifact_path}")
            if name in CONTROL_NAMES and row.get("a1c_nodes") != []:
                raise ValueError(f"control unexpectedly contains A1c nodes: {artifact_path}")
    return artifact


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    offset = (len(ordered) - 1) * probability
    low = math.floor(offset)
    high = math.ceil(offset)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (offset - low)


def clustered_bootstrap_interval(
    position_values: list[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    """Bootstrap whole position clusters; callers pre-average repeats."""
    if not position_values:
        raise ValueError("clustered bootstrap requires position-level values")
    rng = random.Random(int(seed))
    count = len(position_values)
    draws = []
    for _ in range(int(resamples)):
        draws.append(
            statistics.fmean(position_values[rng.randrange(count)] for _ in range(count))
        )
    return {
        "unit": "position cluster retaining all repeats before aggregation",
        "positions": count,
        "resamples": int(resamples),
        "seed": int(seed),
        "lower_95": _percentile(draws, 0.025),
        "upper_95": _percentile(draws, 0.975),
    }


def _position_arm_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    q_correct = sum(int(row["q_pairwise_correct"]) for row in rows)
    q_pairs = sum(int(row["q_pairwise_pairs"]) for row in rows)
    selections = [int(row["selected_action_idx"]) for row in rows]
    modal_count = max(selections.count(value) for value in set(selections))
    return {
        "mean_regret": statistics.fmean(float(row["exact_regret_actor"]) for row in rows),
        "p90_regret": _percentile(
            [float(row["exact_regret_actor"]) for row in rows], 0.9
        ),
        "exact_best_rate": statistics.fmean(float(row["top1_exact"]) for row in rows),
        "pairwise_accuracy": 0.0 if q_pairs == 0 else q_correct / q_pairs,
        "mean_exact_rank": statistics.fmean(
            float(row["selected_exact_rank"]) for row in rows
        ),
        "selection_agreement": modal_count / len(selections),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def classify_result(
    arm_summaries: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, dict[str, Any]]],
    violations: list[str],
) -> dict[str, Any]:
    """Apply the preregistered calibration classification without arm tuning."""
    if violations:
        return {"classification": "invalid", "reasons": violations, "arm_checks": {}}
    checks: dict[str, Any] = {}
    qualifying = []
    for candidate in EXPECTED_ARM_NAMES[2:]:
        candidate_metrics = arm_summaries[candidate]
        x0 = arm_summaries["x0"]
        x1 = arm_summaries["x1_hajek_balanced"]
        versus_x0 = paired[candidate]["x0"]
        conditions = {
            "lower_mean_regret_than_both_controls": candidate_metrics[
                "mean_exact_regret_actor"
            ]
            < x0["mean_exact_regret_actor"]
            and candidate_metrics["mean_exact_regret_actor"]
            < x1["mean_exact_regret_actor"],
            "exact_best_rate_no_lower_than_both_controls": candidate_metrics[
                "exact_best_selection_rate"
            ]
            >= x0["exact_best_selection_rate"]
            and candidate_metrics["exact_best_selection_rate"]
            >= x1["exact_best_selection_rate"],
            # Decision: interpret the preregistered diagnostic comparison against
            # incumbent x0; the primary conditions already require beating both.
            "improves_p90_or_pairwise_vs_x0": candidate_metrics[
                "p90_exact_regret_actor"
            ]
            < x0["p90_exact_regret_actor"]
            or candidate_metrics["exact_pairwise_ordering_accuracy"]
            > x0["exact_pairwise_ordering_accuracy"],
            "nonregressing_on_majority_of_positions": versus_x0[
                "position_win_tie_loss"
            ]["wins"]
            + versus_x0["position_win_tie_loss"]["ties"]
            > len(versus_x0["position_deltas"]) / 2,
            "gains_not_concentrated_in_one_or_two_positions": versus_x0[
                "position_win_tie_loss"
            ]["wins"]
            >= 3,
            "gain_not_confined_to_never_initialized_searches": not versus_x0[
                "all_regret_gain_positions_never_initialized"
            ],
        }
        checks[candidate] = conditions
        if all(conditions.values()):
            qualifying.append(candidate)
    if qualifying:
        return {
            "classification": "target-positive",
            "qualifying_arms": qualifying,
            "arm_checks": checks,
            "note": "Calibration signal only; this is not a promotion or whole-game strength claim.",
        }
    any_lower_regret = any(
        arm_summaries[name]["mean_exact_regret_actor"]
        < arm_summaries["x0"]["mean_exact_regret_actor"]
        for name in EXPECTED_ARM_NAMES[2:]
    )
    any_better_best = any(
        arm_summaries[name]["exact_best_selection_rate"]
        > arm_summaries["x0"]["exact_best_selection_rate"]
        for name in EXPECTED_ARM_NAMES[2:]
    )
    if not any_lower_regret and not any_better_best:
        label = "target-negative"
        reason = "No A1c arm lowers mean regret or improves exact-best rate versus x0."
    else:
        label = "inconclusive/weak-signal"
        reason = "Some point estimates improve, but no arm satisfies all preregistered conditions."
    return {"classification": label, "reasons": [reason], "arm_checks": checks}


def aggregate_suite(
    position_artifacts: list[dict[str, Any]], resolved: dict[str, Any]
) -> dict[str, Any]:
    config = resolved["suite_config"]
    arm_names = [row["name"] for row in config["search"]["arms"]]
    by_position: dict[int, dict[str, Any]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    all_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}
    position_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in arm_names}

    for artifact in position_artifacts:
        position_index = int(artifact["position_index"])
        exact = artifact["exact"]
        per_arm = {}
        for name in arm_names:
            rows = [record["arms"][name] for record in artifact["records"]]
            all_rows[name].extend(rows)
            metrics = _position_arm_metrics(rows)
            position_metrics[name].append(metrics)
            per_arm[name] = metrics
        by_position[position_index] = {
            "legal_actions": int(artifact["legal_actions"]),
            "top_gap_actor": exact["top_gap_actor"],
            "arms": per_arm,
        }

    bootstrap = config["bootstrap"]
    for arm_index, name in enumerate(arm_names):
        rows = all_rows[name]
        pos = position_metrics[name]
        node_rows = [node for row in rows for node in row["a1c_nodes"]]
        regrets = [entry["mean_regret"] for entry in pos]
        arm_violations = []
        if any(not row["nn_eval_budget_hit"] for row in rows):
            arm_violations.append("unused NN budget")
        if any(row["simulation_limit_hit"] for row in rows):
            arm_violations.append("simulation ceiling reached")
        if any(not row["python_rust_nn_accounting_match"] for row in rows):
            arm_violations.append("Python/Rust NN accounting mismatch")
        violations.extend(f"{name}: {value}" for value in arm_violations)
        arm_summaries[name] = {
            "searches": len(rows),
            "independent_position_clusters": len(pos),
            "exact_best_selection_rate": statistics.fmean(
                entry["exact_best_rate"] for entry in pos
            ),
            "mean_exact_regret_actor": statistics.fmean(regrets),
            "mean_exact_regret_clustered_95_interval": clustered_bootstrap_interval(
                regrets,
                seed=int(bootstrap["seed"]) + 1000 + arm_index,
                resamples=int(bootstrap["resamples"]),
            ),
            "median_exact_regret_actor": statistics.median(regrets),
            "p90_exact_regret_actor": _percentile(regrets, 0.9),
            "exact_pairwise_ordering_accuracy": statistics.fmean(
                entry["pairwise_accuracy"] for entry in pos
            ),
            "exact_best_rate_clustered_95_interval": clustered_bootstrap_interval(
                [entry["exact_best_rate"] for entry in pos],
                seed=int(bootstrap["seed"]) + 2000 + arm_index,
                resamples=int(bootstrap["resamples"]),
            ),
            "mean_selected_exact_rank": statistics.fmean(
                entry["mean_exact_rank"] for entry in pos
            ),
            "mean_selection_agreement": statistics.fmean(
                entry["selection_agreement"] for entry in pos
            ),
            "mean_nn_evaluations": _mean(rows, "nn_evaluations"),
            "max_nn_evaluations": max(int(row["nn_evaluations"]) for row in rows),
            "mean_simulations_completed": _mean(rows, "simulations_completed"),
            "mean_wall_seconds": _mean(rows, "elapsed_seconds"),
            "mean_evaluator_calls": _mean(rows, "nn_evaluator_calls"),
            "mean_evaluator_batch_size": _mean(rows, "nn_mean_batch_size"),
            "max_evaluator_batch_size": max(int(row["nn_max_batch_size"]) for row in rows),
            "mean_initialization_nn_rows": _mean(rows, "initialization_nn_evaluations"),
            "mean_initialization_evaluator_calls": _mean(
                rows, "initialization_evaluator_calls"
            ),
            "mean_initialization_nn_fraction": _mean(
                rows, "initialization_nn_fraction"
            ),
            "mean_initialized_cycles": _mean(rows, "a1c_initialized_cycles"),
            "mean_reached_chance_nodes": _mean(rows, "a1c_reached_chance_nodes"),
            "mean_initialized_chance_nodes": _mean(
                rows, "a1c_initialized_chance_nodes"
            ),
            "mean_never_initialized_chance_nodes": _mean(
                rows, "a1c_uninitialized_chance_nodes"
            ),
            "mean_pre_admission_visits_per_node": (
                0.0
                if not node_rows
                else statistics.fmean(float(node["preinit_visits"]) for node in node_rows)
            ),
            "mean_pre_admission_visit_fraction_per_node": (
                0.0
                if not node_rows
                else statistics.fmean(
                    float(node["preinit_visit_fraction"]) for node in node_rows
                )
            ),
            "late_and_never_initialized_nodes": node_rows,
            "violations": arm_violations,
        }

    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate_index, candidate in enumerate(arm_names[2:]):
        paired[candidate] = {}
        for control_index, control in enumerate(CONTROL_NAMES):
            position_deltas = []
            for artifact in position_artifacts:
                candidate_rows = [record["arms"][candidate] for record in artifact["records"]]
                control_rows = [record["arms"][control] for record in artifact["records"]]
                regret_deltas = [
                    float(left["exact_regret_actor"]) - float(right["exact_regret_actor"])
                    for left, right in zip(candidate_rows, control_rows)
                ]
                top1_deltas = [
                    float(left["top1_exact"]) - float(right["top1_exact"])
                    for left, right in zip(candidate_rows, control_rows)
                ]
                pairwise_deltas = [
                    float(left["q_pairwise_accuracy"])
                    - float(right["q_pairwise_accuracy"])
                    for left, right in zip(candidate_rows, control_rows)
                    if left["q_pairwise_accuracy"] is not None
                    and right["q_pairwise_accuracy"] is not None
                ]
                mean_delta = statistics.fmean(regret_deltas)
                initialized = any(
                    int(row["a1c_initialized_chance_nodes"]) > 0 for row in candidate_rows
                )
                position_deltas.append(
                    {
                        "position_index": int(artifact["position_index"]),
                        "mean_regret_delta": mean_delta,
                        "p90_regret_delta": _percentile(
                            [float(row["exact_regret_actor"]) for row in candidate_rows], 0.9
                        )
                        - _percentile(
                            [float(row["exact_regret_actor"]) for row in control_rows], 0.9
                        ),
                        "exact_best_rate_delta": statistics.fmean(top1_deltas),
                        "pairwise_accuracy_delta": (
                            None if not pairwise_deltas else statistics.fmean(pairwise_deltas)
                        ),
                        "candidate_initialized_any_node": initialized,
                    }
                )
            mean_regret_deltas = [row["mean_regret_delta"] for row in position_deltas]
            wins = sum(value < -1e-12 for value in mean_regret_deltas)
            losses = sum(value > 1e-12 for value in mean_regret_deltas)
            ties = len(mean_regret_deltas) - wins - losses
            gain_positions = [row for row in position_deltas if row["mean_regret_delta"] < -1e-12]
            bootstrap_seed = int(bootstrap["seed"]) + 100 * candidate_index + control_index
            exact_best_deltas = [
                row["exact_best_rate_delta"] for row in position_deltas
            ]
            available_pairwise = [
                row["pairwise_accuracy_delta"]
                for row in position_deltas
                if row["pairwise_accuracy_delta"] is not None
            ]
            paired[candidate][control] = {
                "delta_sign": "candidate minus control; negative regret/rank is better, positive exact-best/pairwise is better",
                "position_deltas": position_deltas,
                "mean_regret_delta": statistics.fmean(mean_regret_deltas),
                "mean_p90_regret_delta": statistics.fmean(
                    row["p90_regret_delta"] for row in position_deltas
                ),
                "mean_exact_best_rate_delta": statistics.fmean(
                    exact_best_deltas
                ),
                "mean_pairwise_accuracy_delta": (
                    None
                    if not available_pairwise
                    else statistics.fmean(available_pairwise)
                ),
                "position_win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
                "all_regret_gain_positions_never_initialized": bool(gain_positions)
                and all(not row["candidate_initialized_any_node"] for row in gain_positions),
                "mean_regret_delta_clustered_95_interval": clustered_bootstrap_interval(
                    mean_regret_deltas,
                    seed=bootstrap_seed,
                    resamples=int(bootstrap["resamples"]),
                ),
                "exact_best_rate_delta_clustered_95_interval": clustered_bootstrap_interval(
                    exact_best_deltas,
                    seed=bootstrap_seed + 10000,
                    resamples=int(bootstrap["resamples"]),
                ),
            }

    classification = classify_result(arm_summaries, paired, violations)
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "provenance": {
            "suite_config_sha256": resolved["suite_config_sha256"],
            "checkpoint": resolved["checkpoint"],
            "positions": resolved["positions"],
            "git_commit": resolved["git_commit"],
            "source_hashes": resolved["sources"],
            "position_artifacts": [
                {
                    "position_index": int(artifact["position_index"]),
                    "path": artifact["artifact_path"],
                    "sha256": file_sha256(artifact["artifact_path"]),
                }
                for artifact in position_artifacts
            ],
        },
        "aggregation": "Repeats are averaged within each position before the eight position clusters are aggregated.",
        "calibration_limitation": resolved["selection_limitation"],
        "arms": arm_summaries,
        "paired_deltas": paired,
        "by_position": {str(key): value for key, value in sorted(by_position.items())},
        "validity_violations": violations,
        "classification": classification,
    }


def _verify_frozen_hashes(resolved: dict[str, Any]) -> None:
    for name in ("checkpoint", "positions"):
        item = resolved[name]
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen {name} hash drifted before resumed position")


def run(args: argparse.Namespace) -> dict[str, Any]:
    resolved = validate_suite_inputs(args.config)
    output_dir = Path(args.output_dir).resolve()
    _write_or_verify_resolved(output_dir, resolved)
    if bool(args.validate_only):
        return {
            "status": "validated",
            "positions": len(resolved["oracles"]),
            "cases": len(resolved["cases"]),
            "output_dir": str(output_dir),
        }

    # GPU-dependent imports and initialization occur only after all validation.
    from games.kingdomino.deck8_oracle_compare import compare_boundary
    from games.kingdomino.denial_search import load_checkpoint_network
    from games.kingdomino.self_play import make_rust_evaluator

    config = resolved["suite_config"]
    search = config["search"]
    positions = load_frozen_positions(resolved["positions"]["path"])
    net, checkpoint_cfg = load_checkpoint_network(
        resolved["checkpoint"]["path"], search["device"]
    )
    margin_gain = float(checkpoint_cfg.get("margin_gain", 2.0))
    alpha = float(checkpoint_cfg.get("alpha", 0.5))
    evaluator = make_rust_evaluator(
        net,
        device=search["device"],
        margin_gain=margin_gain,
        alpha=alpha,
    )
    specs = {
        row["name"]: {key: value for key, value in row.items() if key != "name"}
        for row in search["arms"]
    }
    warmed = False
    artifacts = []
    for oracle in resolved["oracles"]:
        _verify_frozen_hashes(resolved)
        position_path = output_dir / "positions" / f"position_{oracle['position_index']:02d}.json"
        provenance = _position_provenance(resolved, oracle)
        if position_path.exists():
            artifact = validate_position_artifact(
                position_path,
                provenance,
                position_index=int(oracle["position_index"]),
            )
            artifact["artifact_path"] = str(position_path)
            artifacts.append(artifact)
            continue
        summary = _read_json(Path(oracle["summary_path"]))
        exact_values = {
            int(row["action_idx"]): float(row["expected_value_actor"])
            for row in summary["actions"]
        }
        root_state, _source = positions[int(oracle["position_index"])]
        boundary = apply_prefix_actions(root_state, oracle["prefix_actions"])
        comparison = compare_boundary(
            boundary,
            evaluator,
            exact_values,
            specs,
            simulation_ceiling=int(search["simulation_ceiling"]),
            nn_eval_budget=int(search["nn_eval_budget"]),
            seed=provenance["seeds"][0],
            seed_count=int(search["repeat_count"]),
            fpu=float(search["fpu"]),
            cpuct=float(search["cpuct"]),
            margin_gain=margin_gain,
            alpha=alpha,
            a1c_init_visits=int(search["chance_init_visits"]),
            a1c_widening_c=float(search["chance_widening_c"]),
            a1c_init_max_fraction=float(search["chance_init_max_fraction"]),
            rotation_offset=int(oracle["ordinal"]) * int(search["repeat_count"]),
            warm=not warmed,
        )
        warmed = True
        exact_order = sorted(exact_values, key=lambda idx: (-exact_values[idx], idx))
        artifact = {
            "schema_version": POSITION_SCHEMA,
            "position_index": int(oracle["position_index"]),
            "legal_actions": int(oracle["legal_actions"]),
            "provenance": provenance,
            "exact": {
                **comparison["exact"],
                "action_values_actor": {
                    str(idx): exact_values[idx] for idx in exact_order
                },
            },
            "records": comparison["records"],
            "single_position_aggregate": comparison["aggregate"],
        }
        _atomic_json(position_path, artifact)
        artifact = validate_position_artifact(
            position_path,
            provenance,
            position_index=int(oracle["position_index"]),
        )
        artifact["artifact_path"] = str(position_path)
        artifacts.append(artifact)

    if len(artifacts) != len(resolved["oracles"]):
        raise RuntimeError("suite aggregation attempted before all positions completed")
    summary = aggregate_suite(artifacts, resolved)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        existing = _read_json(summary_path)
        if existing != summary:
            raise ValueError(f"existing suite summary differs; refusing overwrite: {summary_path}")
    else:
        _atomic_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result if result.get("status") == "validated" else {
        "status": result["status"],
        "classification": result["classification"],
        "arms": result["arms"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
