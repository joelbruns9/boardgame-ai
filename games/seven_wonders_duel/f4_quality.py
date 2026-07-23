"""F4.2 preregistered WU/leaf-batch position-quality sweep.

Consumes a Phase-E position/truth corpus and ``f4_contract_v2.json``. It always runs
the exact ``leaf_batch=1`` arena path as the paired sequential baseline, records
raw JSONL rows, and emits the position-gate summary needed before playing-strength
calibration. It deliberately cannot write ``f4_quality_lock.json``: the separate
seat-swapped non-inferiority gate must also be green first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

from .codec import legal_action_indices
from .encoder import Encoding, Token, TokenType
from .phase_e import reconstruct
from .rust_bridge import rust_flat_batch_adapter, rust_game_from_prefix


ROOT = Path(__file__).resolve().parent
LEGACY_CONTRACT_PATH = ROOT / "f4_contract.json"
CONTRACT_PATH = ROOT / "f4_contract_v2.json"
REQUIRED_PHASE_STRATA = (
    "wonder_draft",
    "age_1",
    "age_2",
    "age_3",
    "between_ages",
    "pending_choice",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    at = q * (len(ordered) - 1)
    lo = int(math.floor(at))
    hi = int(math.ceil(at))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - at) + ordered[hi] * (at - lo)


def _js_divergence(a: list[float], b: list[float]) -> float:
    middle = [(x + y) * 0.5 for x, y in zip(a, b)]

    def kl(p, q):
        return sum(x * math.log(x / y) for x, y in zip(p, q) if x > 0.0)

    return 0.5 * (kl(a, middle) + kl(b, middle))


def _cluster_upper_bound(
    rows: list[dict], *, field: str, confidence: float, resamples: int, seed: int
) -> float | None:
    by_position: dict[str, list[float]] = {}
    for row in rows:
        by_position.setdefault(row["position_id"], []).append(float(row[field]))
    clusters = [mean(values) for values in by_position.values()]
    if not clusters:
        return None
    rng = random.Random(seed)
    samples = [
        mean(clusters[rng.randrange(len(clusters))] for _ in clusters)
        for _ in range(resamples)
    ]
    return _percentile(samples, confidence)


def _net_adapter(evaluator):
    token_types = list(TokenType)

    def adapter(tokens, actor, legal):
        encoded = Encoding(
            actor=actor,
            tokens=tuple(
                Token(token_types[type_id], entity_id, aux_id, tuple(features))
                for type_id, entity_id, aux_id, features in tokens
            ),
        )
        result = evaluator.evaluate([encoded], [list(legal)])[0]
        return float(result.wdl[0] - result.wdl[2]), [float(p) for p in result.policy]

    return adapter


def _parse_search(result, legal: list[int]) -> dict:
    if isinstance(result, dict):
        return {
            "action": int(result["action"]),
            "action_value": float(result["action_value"]),
            "root_value": float(result["root_value"]),
            "visits": list(result["visits"]),
            "policy": list(result["policy"]),
            "topk": list(result["topk"]),
            "sims": int(result["sims"]),
            "metrics": {
                **{key: int(value) for key, value in result["metrics"].items()},
                "forced_rows": int(result.get("nn_work", {}).get("forced_rows", 0)),
                "forced_cache_hits": int(
                    result.get("nn_work", {}).get("forced_cache_hits", 0)
                ),
            },
            "completed_q": {
                action: float(q) for action, q in zip(legal, result["completed_q"])
            },
        }
    metrics = result[7]
    return {
        "action": int(result[0]),
        "action_value": float(result[1]),
        "root_value": float(result[2]),
        "visits": list(result[3]),
        "policy": list(result[4]),
        "topk": list(result[5]),
        "sims": int(result[6]),
        "metrics": {
            "scheduled": int(metrics[0]),
            "requested": int(metrics[1]),
            "unique": int(metrics[2]),
            "terminal": int(metrics[3]),
            "collisions": int(metrics[4]),
            "waves": int(metrics[5]),
            "max_wave_paths": int(metrics[6]),
            "max_wave_unique": int(metrics[7]),
            "forced_rows": 0,
            "forced_cache_hits": 0,
        },
        "completed_q": {action: float(q) for action, q in zip(legal, result[8])},
    }


def summarize(rows: list[dict], positions: list[dict], truths: dict, contract: dict) -> dict:
    quality = contract["fast_search_quality_gate"]
    bounds = quality["eligibility"]
    corpus_cfg = quality["corpus"]
    confidence = quality["confidence_method"]
    consequential_ids = {
        position["id"]
        for position in positions
        if truths.get(position["id"], {}).get("trap_gap", 0.0)
        >= corpus_cfg["consequential_gap_minimum"]
    }
    sequential_traps: dict[str, list[bool]] = {}
    for row in rows:
        sequential_traps.setdefault(row["position_id"], []).append(row["sequential_trap"])
    clean_ids = {
        position_id
        for position_id, picks in sequential_traps.items()
        if position_id in consequential_ids and not any(picks)
    }
    phase_counts = {stratum: 0 for stratum in REQUIRED_PHASE_STRATA}
    for position in positions:
        stratum = position.get("stratum")
        if stratum is None and position.get("age") in (1, 2, 3):
            stratum = f"age_{position['age']}"
        if stratum in phase_counts:
            phase_counts[stratum] += 1
    phase_minimum = corpus_cfg["minimum_positions_per_game_phase"]
    phase_minimum_met = all(count >= phase_minimum for count in phase_counts.values())
    output = {
        "corpus": {
            "positions": len(positions),
            "consequential_positions": len(consequential_ids),
            "baseline_clean_consequential_positions": len(clean_ids),
            "phase_strata": phase_counts,
            "minimum_positions_per_game_phase": phase_minimum,
            "phase_minimums_met": phase_minimum_met,
            "minimums_met": (
                len(positions) >= corpus_cfg["minimum_total_positions"]
                and len(consequential_ids) >= corpus_cfg["minimum_consequential_positions"]
                and len(clean_ids)
                >= corpus_cfg["minimum_baseline_clean_consequential_positions"]
                and phase_minimum_met
            ),
        },
        "leaf_batches": {},
    }
    first_batch = min({row["leaf_batch"] for row in rows})
    sequential_by_position: dict[str, list[dict]] = {}
    for row in rows:
        if row["leaf_batch"] == first_batch:
            sequential_by_position.setdefault(row["position_id"], []).append(row)
    variance_pairs = []
    for position_rows in sequential_by_position.values():
        ordered = sorted(position_rows, key=lambda row: row["search_seed"])
        if len(ordered) < 2:
            continue
        for index, left in enumerate(ordered):
            right = ordered[(index + 1) % len(ordered)]
            variance_pairs.append(
                {
                    "action_agreement": float(
                        left["sequential_action"] == right["sequential_action"]
                    ),
                    "policy_js": _js_divergence(
                        left["sequential_policy"], right["sequential_policy"]
                    ),
                    "root_value_abs_error": abs(
                        left["sequential_root_value"] - right["sequential_root_value"]
                    ),
                }
            )
    natural_pairs_required = quality["natural_variance_control"][
        "sequential_independent_seed_pairs_per_position"
    ]
    output["natural_variance_control"] = {
        "pairs": len(variance_pairs),
        "minimum_pairs_per_position": natural_pairs_required,
        "minimum_met": bool(sequential_by_position)
        and all(
            len(position_rows) >= natural_pairs_required
            for position_rows in sequential_by_position.values()
        ),
        "action_agreement": mean([pair["action_agreement"] for pair in variance_pairs])
        if variance_pairs
        else None,
        "mean_policy_js": mean([pair["policy_js"] for pair in variance_pairs])
        if variance_pairs
        else None,
        "mean_root_value_abs_error": mean(
            [pair["root_value_abs_error"] for pair in variance_pairs]
        )
        if variance_pairs
        else None,
    }
    for leaf_batch in sorted({row["leaf_batch"] for row in rows}):
        selected = [row for row in rows if row["leaf_batch"] == leaf_batch]
        consequential = [row for row in selected if row["position_id"] in consequential_ids]
        new_blunder_ids = sorted(
            {
                row["position_id"]
                for row in selected
                if row["position_id"] in clean_ids and row["fast_trap"]
            }
        )
        trap_upper = _cluster_upper_bound(
            consequential,
            field="trap_delta",
            confidence=confidence["one_sided_confidence"],
            resamples=confidence["minimum_resamples"],
            seed=confidence["fixed_seed"] + leaf_batch,
        )
        large = [row for row in selected if row["sequential_gap"] >= bounds["large_gap_minimum"]]
        medium = [
            row
            for row in selected
            if bounds["medium_gap_minimum"]
            <= row["sequential_gap"]
            < bounds["large_gap_minimum"]
        ]
        regrets = [row["sequential_tree_regret"] for row in selected]
        divergences = [row["policy_js"] for row in selected]
        value_errors = [row["root_value_abs_error"] for row in selected]
        requested = sum(row["metrics"]["requested"] for row in selected)
        collisions = sum(row["metrics"]["collisions"] for row in selected)
        paired_counts: dict[str, int] = {}
        for row in selected:
            paired_counts[row["position_id"]] = paired_counts.get(row["position_id"], 0) + 1
        paired_seed_minimum_met = bool(paired_counts) and all(
            count >= corpus_cfg["minimum_paired_search_seeds_per_position"]
            for count in paired_counts.values()
        )
        large_agreement = (
            mean([row["action_agreement"] for row in large]) if large else None
        )
        medium_agreement = (
            mean([row["action_agreement"] for row in medium]) if medium else None
        )
        metrics = {
            "rows": len(selected),
            "new_blunder_fixtures": new_blunder_ids,
            "trap_delta_one_sided_95_upper": trap_upper,
            "mean_regret": mean(regrets),
            "p95_regret": _percentile(regrets, 0.95),
            "large_gap_agreement": large_agreement,
            "medium_gap_agreement": medium_agreement,
            "mean_policy_js": mean(divergences),
            "p95_policy_js": _percentile(divergences, 0.95),
            "mean_root_value_abs_error": mean(value_errors),
            "p95_root_value_abs_error": _percentile(value_errors, 0.95),
            "collision_fraction": collisions / requested if requested else 0.0,
            "minimum_paired_search_seeds_met": paired_seed_minimum_met,
        }
        checks = {
            "paired_search_sample_size": paired_seed_minimum_met
            and output["natural_variance_control"]["minimum_met"],
            "zero_new_clean_blunders": not new_blunder_ids,
            "trap_rate_non_inferior": trap_upper is not None
            and trap_upper <= bounds["aggregate_consequential_trap_rate_delta_one_sided_95_upper"],
            "mean_regret": metrics["mean_regret"]
            <= bounds["aggregate_mean_sequential_tree_regret_max"],
            "p95_regret": metrics["p95_regret"]
            <= bounds["aggregate_p95_sequential_tree_regret_max"],
            "large_gap_agreement": large_agreement is not None
            and large_agreement >= bounds["large_gap_chosen_action_agreement_min"],
            "medium_gap_agreement": medium_agreement is not None
            and medium_agreement >= bounds["medium_gap_chosen_action_agreement_min"],
            "mean_policy_js": metrics["mean_policy_js"]
            <= bounds["mean_root_policy_jensen_shannon_divergence_max"],
            "p95_policy_js": metrics["p95_policy_js"]
            <= bounds["p95_root_policy_jensen_shannon_divergence_max"],
            "mean_root_value": metrics["mean_root_value_abs_error"]
            <= bounds["mean_absolute_root_value_error_max"],
            "p95_root_value": metrics["p95_root_value_abs_error"]
            <= bounds["p95_absolute_root_value_error_max"],
        }
        output["leaf_batches"][str(leaf_batch)] = {
            "metrics": metrics,
            "checks": checks,
            "position_gate_eligible": output["corpus"]["minimums_met"] and all(checks.values()),
        }
    eligible = [
        int(batch)
        for batch, result in output["leaf_batches"].items()
        if result["position_gate_eligible"]
    ]
    output["largest_position_eligible_leaf_batch"] = max(eligible) if eligible else None
    output["quality_lock_written"] = False
    output["remaining_gate"] = "seat-swapped playing-strength non-inferiority"
    return output


def run(args) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    quality_contract = contract["fast_search_quality_gate"]
    required = {
        "sims": quality_contract["required_sims"],
        "top_k": quality_contract["required_top_k"],
        "force": quality_contract["required_force_expand_root_chance"],
    }
    actual = {"sims": args.sims, "top_k": args.top_k, "force": args.force}
    mismatches = [
        f"{field}={actual[field]!r} (required {value!r})"
        for field, value in required.items()
        if actual[field] != value
    ]
    if mismatches:
        raise ValueError(
            "quality run does not match f4-contract-2: " + ", ".join(mismatches)
        )
    positions_path = args.corpus / "positions.jsonl"
    truths_path = args.corpus / "ground_truth.jsonl"
    positions = _read_jsonl(positions_path)
    if args.limit:
        positions = positions[: args.limit]
    truths = {row["id"]: row for row in _read_jsonl(truths_path) if not row.get("skipped")}

    from .phase_e import load_evaluator

    evaluator = load_evaluator(str(args.checkpoint), args.device)
    evaluator.max_batch = args.global_batch_cap
    adapter = rust_flat_batch_adapter(evaluator)
    leaf_batches = args.leaf_batches or contract["fast_search_quality_gate"][
        "candidate_leaf_batches"
    ]
    results_path = args.output / "position_rows.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema": "f4-quality-run-2",
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": _sha256(CONTRACT_PATH),
        "positions_sha256": _sha256(positions_path),
        "ground_truth_sha256": _sha256(truths_path),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "sims": args.sims,
        "top_k": args.top_k,
        "seeds": args.seeds,
        "seed": args.seed,
        "force": args.force,
        "age_deal_samples": args.age_deal_samples,
        "leaf_batches": leaf_batches,
        "global_batch_cap": args.global_batch_cap,
        "position_batch": args.position_batch,
        "limit": args.limit,
    }
    run_config_path = args.output / "run_config.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError("existing quality output uses a different run configuration")
    else:
        run_config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rows = _read_jsonl(results_path) if results_path.exists() else []
    done = {
        (row["position_id"], int(row["search_seed"]), int(row["leaf_batch"]))
        for row in rows
    }
    prepared = []
    for position_index, position in enumerate(positions):
        py_state = reconstruct(position)
        reconstructed, rust_state = rust_game_from_prefix(
            position["game_seed"], position["first_player"], position["prefix"]
        )
        if legal_action_indices(reconstructed) != legal_action_indices(py_state):
            raise RuntimeError(f"Rust reconstruction diverged for {position['id']}")
        prepared.append(
            {
                "index": position_index,
                "position": position,
                "rust": rust_state,
                "legal": legal_action_indices(py_state),
                "traps": set(position.get("traps", ())),
                "consequential": truths.get(position["id"], {}).get("trap_gap", 0.0)
                >= contract["fast_search_quality_gate"]["corpus"][
                    "consequential_gap_minimum"
                ],
            }
        )
    import seven_wonders_rust as swr

    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        for seed_index in range(args.seeds):
            for start in range(0, len(prepared), args.position_batch):
                batch = [
                    item
                    for item in prepared[start : start + args.position_batch]
                    if any(
                        (
                            item["position"]["id"],
                            args.seed + item["index"] * 100_003 + seed_index,
                            leaf_batch,
                        )
                        not in done
                        for leaf_batch in leaf_batches
                    )
                ]
                if not batch:
                    continue
                games = [item["rust"] for item in batch]
                search_seeds = [
                    args.seed + item["index"] * 100_003 + seed_index for item in batch
                ]
                sequential_rows = swr.search_many_flat_net(
                    adapter,
                    games,
                    search_seeds,
                    args.global_batch_cap,
                    1,
                    args.sims,
                    args.top_k,
                    force=args.force,
                    age_deal_samples=args.age_deal_samples,
                )
                fast_by_batch = {
                    leaf_batch: swr.search_many_flat_net(
                        adapter,
                        games,
                        search_seeds,
                        args.global_batch_cap,
                        leaf_batch,
                        args.sims,
                        args.top_k,
                        force=args.force,
                        age_deal_samples=args.age_deal_samples,
                    )
                    for leaf_batch in leaf_batches
                }
                for row_index, item in enumerate(batch):
                    position = item["position"]
                    legal = item["legal"]
                    traps = item["traps"]
                    search_seed = search_seeds[row_index]
                    sequential = _parse_search(sequential_rows[row_index], legal)
                    q_values = sorted(sequential["completed_q"].values(), reverse=True)
                    # Completed Q is actor-relative in [-1, 1], so 2.0 is the
                    # maximum possible gap and cleanly represents a sole legal move.
                    gap = q_values[0] - q_values[1] if len(q_values) > 1 else 2.0
                    best_q = q_values[0]
                    for leaf_batch in leaf_batches:
                        key = (position["id"], search_seed, leaf_batch)
                        if key in done:
                            continue
                        fast = _parse_search(fast_by_batch[leaf_batch][row_index], legal)
                        row = {
                            "position_id": position["id"],
                            "age": position.get("age"),
                            "consequential": item["consequential"],
                            "search_seed": search_seed,
                            "leaf_batch": leaf_batch,
                            "sequential_action": sequential["action"],
                            "sequential_policy": sequential["policy"],
                            "sequential_root_value": sequential["root_value"],
                            "fast_action": fast["action"],
                            "action_agreement": float(fast["action"] == sequential["action"]),
                            "sequential_gap": gap,
                            "sequential_tree_regret": best_q
                            - sequential["completed_q"][fast["action"]],
                            "policy_js": _js_divergence(sequential["policy"], fast["policy"]),
                            "root_value_abs_error": abs(
                                sequential["root_value"] - fast["root_value"]
                            ),
                            "sequential_trap": sequential["action"] in traps,
                            "fast_trap": fast["action"] in traps,
                            "trap_delta": float(fast["action"] in traps)
                            - float(sequential["action"] in traps),
                            "metrics": fast["metrics"],
                        }
                        rows.append(row)
                        done.add(key)
                        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                print(
                    f"quality: seed {seed_index + 1}/{args.seeds}, positions {min(start + len(batch), len(prepared))}/{len(prepared)}",
                    flush=True,
                )

    summary = summarize(rows, positions, truths, contract)
    summary["manifest"] = {
        "contract_schema_version": contract["schema_version"],
        "contract_sha256": _sha256(CONTRACT_PATH),
        "positions_sha256": _sha256(positions_path),
        "ground_truth_sha256": _sha256(truths_path),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "sims": args.sims,
        "top_k": args.top_k,
        "seeds": args.seeds,
        "force_expand_root_chance": args.force,
        "age_deal_sample_count": args.age_deal_samples,
        "leaf_batches": leaf_batches,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--leaf-batches", type=lambda value: [int(x) for x in value.split(",")])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--age-deal-samples", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--global-batch-cap", type=int, default=256)
    parser.add_argument("--position-batch", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
