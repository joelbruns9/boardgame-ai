"""Reproducible F4.6 laptop comparison and Rust production benchmark.

Laptop mode runs the Python and flat-buffer Rust generators at the frozen
search settings, emits per-repetition JSONL, and computes a fixed-seed bootstrap
lower bound for the speedup. Rust mode is the cloud-sweep primitive and never
enters the Python self-play loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .buffer import GameRecorder
from .codec import decode_action
from .engine import apply_action
from .f4_quality import CONTRACT_PATH
from .game import Phase
from .inference import Evaluator
from .loop_inference import CoalescingEvaluator
from .phase_d import CurriculumMCTS, temperature_for_move
from .phase_e import load_evaluator
from .portable_rng import PortableRng
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play
from .search import SearchConfig


SCHEMA = "f4-throughput-row-2"
GAME_RNG_XOR = 0xC6BC279692B5CC83


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rows_csv(rows: list[dict], path: Path) -> None:
    """Mirror append-safe JSONL rows into a convenient tabular artifact."""

    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _sample_policy(policy: dict[int, float], temperature: float, rng: PortableRng) -> int:
    actions = sorted(policy)
    weights = [max(policy[action], 1e-12) ** (1.0 / temperature) for action in actions]
    target = rng.next_float() * sum(weights)
    cumulative = 0.0
    for action, weight in zip(actions, weights):
        cumulative += weight
        if target < cumulative:
            return action
    return actions[-1]


def _python_game(index: int, seed: int, evaluator, search: dict, force: bool) -> dict:
    first_player = index % 2
    recorder = GameRecorder(
        seed,
        first_player=first_player,
        agents={"p0": "network", "p1": "network", "kind": "self_play"},
        iteration=-1,
    )
    rng = PortableRng(seed ^ GAME_RNG_XOR)
    simulations = 0
    while recorder.game.phase is not Phase.COMPLETE:
        full = rng.next_float() < search["full_search_fraction"]
        low = search["full_sims_min"] if full else search["cheap_sims_min"]
        high = search["full_sims_max"] if full else search["cheap_sims_max"]
        sims = low + rng.randrange(high - low + 1)
        search_seed = rng.getrandbits(63)
        mcts = CurriculumMCTS(
            evaluator,
            SearchConfig(
                sims=sims,
                top_k=search["top_k"],
                mode="closed",
                seed=search_seed,
                force_expand_root_chance=force,
            ),
            search["draft_prior"],
        )
        result = mcts.search(recorder.game)
        action = _sample_policy(
            result.policy_target,
            temperature_for_move(len(recorder._moves)),
            rng,
        )
        recorder.play(
            action,
            visits=result.visits,
            policy_target=result.policy_target,
            root_value=result.root_value,
            sims=result.sims,
            mode=result.mode,
            gumbel_topk=result.gumbel_topk,
            policy_excluded=not full,
        )
        simulations += result.sims
    record = recorder.finish()
    return {
        "moves": len(record.moves),
        "simulations": simulations,
        "policy_eligible_targets": sum(not move.policy_excluded for move in record.moves),
    }


def _run_python(
    *, seeds: list[int], evaluator: Evaluator, search: dict, workers: int, batch_cap: int
) -> dict:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    with CoalescingEvaluator(evaluator, max_batch=batch_cap, max_wait_ms=2.0) as service:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    lambda item: _python_game(
                        item[0], item[1], service, search,
                        force=search["force_expand_root_chance"]
                    ),
                    enumerate(seeds),
                )
            )
        batches = service.batches
        positions = service.positions
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    games = len(results)
    moves = sum(row["moves"] for row in results)
    simulations = sum(row["simulations"] for row in results)
    policy_targets = sum(row["policy_eligible_targets"] for row in results)
    return {
        "implementation": "python",
        "seconds": wall,
        "games": games,
        "moves": moves,
        "simulations": simulations,
        "games_per_second": games / wall,
        "moves_per_second": moves / wall,
        "scheduled_simulations_per_second": simulations / wall,
        "policy_eligible_targets_per_second": policy_targets / wall,
        "requested_nn_leaves_per_second": positions / wall,
        "unique_nn_leaves_per_second": positions / wall,
        "global_batch_leaves_mean": positions / batches if batches else 0.0,
        "cpu_utilization": cpu / wall / max(1, os.cpu_count() or 1),
    }


def _metric_delta(adapter, before: dict, row_start: int, token_start: int) -> dict:
    total = adapter.total_metrics
    return {
        key: float(total[key]) - float(before.get(key, 0.0))
        for key in total
    } | {
        "batch_rows": adapter.batch_rows[row_start:],
        "batch_tokens": adapter.batch_tokens[token_start:],
        "batch_padded_tokens": adapter.batch_padded_tokens[token_start:],
    }


def _run_rust(
    *,
    seeds: list[int],
    evaluator: Evaluator,
    search: dict,
    lock: dict,
    slots: int,
    batch_cap: int,
    max_inflight: int,
    scheduler_workers: int,
    diagnostic_sync: bool,
    pinned_memory: bool,
) -> dict:
    import seven_wonders_rust as swr

    adapter = rust_flat_batch_adapter(
        evaluator,
        diagnostic_sync=diagnostic_sync,
        pinned_memory=pinned_memory,
    )
    before = dict(adapter.total_metrics)
    row_start = len(adapter.batch_rows)
    token_start = len(adapter.batch_tokens)
    if evaluator.device != "cpu":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    raw_metrics = []
    games_done = 0
    moves = simulations = requested = unique = terminal = collisions = 0
    policy_targets = forced_rows = forced_cache_hits = 0
    forced_by_kind = {
        "card_reveal": 0,
        "great_library": 0,
        "wonder_group": 0,
        "age_deal": 0,
    }
    for start in range(0, len(seeds), slots):
        chunk = seeds[start : start + slots]
        first_players = [(start + index) % 2 for index in range(len(chunk))]
        games = rust_games_for_self_play(chunk, first_players)
        records, metrics = swr.self_play_many_flat_net(
            adapter=adapter,
            games=games,
            game_seeds=chunk,
            global_batch_cap=batch_cap,
            leaf_batch=lock["leaf_batch"],
            cheap_sims_min=search["cheap_sims_min"],
            cheap_sims_max=search["cheap_sims_max"],
            full_sims_min=search["full_sims_min"],
            full_sims_max=search["full_sims_max"],
            full_search_fraction=search["full_search_fraction"],
            top_k=search["top_k"],
            draft_prior=search["draft_prior"],
            iteration=-1,
            force=lock["force_expand_root_chance"],
            age_deal_samples=lock["age_deal_sampler"]["sample_count"],
            max_inflight_batches=max_inflight,
            scheduler_workers=scheduler_workers,
        )
        games_done += len(records)
        moves += int(metrics["moves"])
        simulations += int(metrics["simulations"])
        requested += int(metrics["requested_nn_leaves"])
        unique += int(metrics["unique_nn_leaves"])
        terminal += int(metrics["terminal_leaves"])
        collisions += int(metrics["collisions"])
        forced_rows += int(metrics["forced_rows"])
        forced_cache_hits += int(metrics["forced_cache_hits"])
        for kind in forced_by_kind:
            forced_by_kind[kind] += int(metrics[f"forced_{kind}_rows"])
        policy_targets += sum(
            not move["policy_excluded"] for record in records for move in record["moves"]
        )
        raw_metrics.append(metrics)
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    boundary = _metric_delta(adapter, before, row_start, token_start)
    batch_rows = boundary.pop("batch_rows")
    batch_tokens = boundary.pop("batch_tokens")
    batch_padded = boundary.pop("batch_padded_tokens")
    total_tokens = sum(batch_tokens)
    total_padded = sum(batch_padded)
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(evaluator.device))
        if evaluator.device != "cpu"
        else 0
    )
    try:
        import psutil

        memory = psutil.Process().memory_info()
        peak_host = int(getattr(memory, "peak_wset", memory.rss))
    except (ImportError, OSError):
        peak_host = 0
    queue_seconds = sum(float(row["queue_wait_ns"]) for row in raw_metrics) / 1e9
    pack_seconds = sum(float(row["encode_pack_ns"]) for row in raw_metrics) / 1e9
    py_call_seconds = sum(float(row["py_call_ns"]) for row in raw_metrics) / 1e9
    extract_seconds = sum(float(row["extract_ns"]) for row in raw_metrics) / 1e9
    tree_seconds = sum(float(row["rust_tree_ns"]) for row in raw_metrics) / 1e9
    chance_seconds = sum(float(row["rust_chance_ns"]) for row in raw_metrics) / 1e9
    record_seconds = sum(float(row["rust_record_ns"]) for row in raw_metrics) / 1e9
    scatter_seconds = sum(float(row["scatter_ns"]) for row in raw_metrics) / 1e9
    ready_cycles = sum(float(row["scheduler_ready_slot_cycles"]) for row in raw_metrics)
    waiting_cycles = sum(float(row["scheduler_waiting_slot_cycles"]) for row in raw_metrics)
    idle_cycles = sum(float(row["scheduler_idle_slot_cycles"]) for row in raw_metrics)
    forced_per_search = [
        int(value)
        for row in raw_metrics
        for value in row["forced_rows_per_search"]
    ]
    slot_cycles = ready_cycles + waiting_cycles + idle_cycles
    return {
        "implementation": "rust",
        "seconds": wall,
        "games": games_done,
        "moves": moves,
        "simulations": simulations,
        "games_per_second": games_done / wall,
        "moves_per_second": moves / wall,
        "scheduled_simulations_per_second": simulations / wall,
        "policy_eligible_targets_per_second": policy_targets / wall,
        "forced_outcome_rows_per_second": forced_rows / wall,
        "forced_card_reveal_rows_per_second": forced_by_kind["card_reveal"] / wall,
        "forced_great_library_rows_per_second": forced_by_kind["great_library"] / wall,
        "forced_wonder_group_rows_per_second": forced_by_kind["wonder_group"] / wall,
        "forced_age_deal_rows_per_second": forced_by_kind["age_deal"] / wall,
        "forced_outcome_rows_per_search_mean": statistics.mean(forced_per_search)
        if forced_per_search
        else 0.0,
        "forced_outcome_rows_per_search_p50": _percentile(forced_per_search, 0.50),
        "forced_outcome_rows_per_search_p95": _percentile(forced_per_search, 0.95),
        "forced_outcome_rows_per_search_max": max(forced_per_search, default=0),
        "requested_nn_leaves_per_second": requested / wall,
        "unique_nn_leaves_per_second": unique / wall,
        "total_nn_rows_per_second": sum(batch_rows) / wall,
        "total_nn_tokens_per_second": total_tokens / wall,
        "forced_cache_hits": forced_cache_hits,
        "terminal_leaves": terminal,
        "collisions": collisions,
        "dedupe_ratio": 1.0 - unique / requested if requested else 0.0,
        "global_batch_leaves_mean": statistics.mean(batch_rows) if batch_rows else 0.0,
        "global_batch_leaves_p50": _percentile(batch_rows, 0.50),
        "global_batch_leaves_p95": _percentile(batch_rows, 0.95),
        "global_batch_tokens_mean": statistics.mean(batch_tokens) if batch_tokens else 0.0,
        "global_batch_tokens_p50": _percentile(batch_tokens, 0.50),
        "global_batch_tokens_p95": _percentile(batch_tokens, 0.95),
        "padding_ratio": 1.0 - total_tokens / total_padded if total_padded else 0.0,
        "scheduler_ready_fraction": ready_cycles / slot_cycles if slot_cycles else 0.0,
        "scheduler_waiting_fraction": waiting_cycles / slot_cycles if slot_cycles else 0.0,
        "scheduler_idle_fraction": idle_cycles / slot_cycles if slot_cycles else 0.0,
        "queue_wait_seconds": queue_seconds,
        "pyo3_call_seconds": py_call_seconds,
        "rust_tree_seconds": tree_seconds,
        "rust_chance_seconds": chance_seconds,
        "rust_encode_pack_seconds": pack_seconds,
        "rust_record_seconds": record_seconds,
        "pyo3_tensor_seconds": boundary["tensor_seconds"],
        "h2d_seconds": boundary["h2d_seconds"],
        "gpu_forward_seconds": boundary["forward_seconds"],
        "gather_d2h_seconds": boundary["gather_seconds"] + boundary["d2h_seconds"],
        "scatter_seconds": scatter_seconds + extract_seconds,
        "cpu_utilization": cpu / wall / max(1, os.cpu_count() or 1),
        "gpu_busy_fraction": min(1.0, boundary["forward_seconds"] / wall),
        "peak_host_memory_bytes": peak_host,
        "peak_gpu_memory_bytes": peak_gpu,
        "oom_count": 0,
        "global_batches": len(batch_rows),
        "leaf_batch": int(lock["leaf_batch"]),
        "force_expand_root_chance": bool(lock["force_expand_root_chance"]),
        "sims": int(search["full_sims_min"]),
        "slots": slots,
        "scheduler_workers": scheduler_workers,
    }


def speedup_summary(python_rows: list[dict], rust_rows: list[dict], seed: int) -> dict:
    ratios = [
        rust["games_per_second"] / python["games_per_second"]
        for python, rust in zip(python_rows, rust_rows)
    ]
    rng = random.Random(seed)
    samples = []
    for _ in range(10_000):
        samples.append(sum(rng.choice(ratios) for _ in ratios) / len(ratios))
    return {
        "repetitions": len(ratios),
        "ratios": ratios,
        "mean_speedup": statistics.mean(ratios),
        "speedup_one_sided_95_lower": _percentile(samples, 0.05),
    }


def native_inference_assessment(rust_rows: list[dict]) -> dict:
    """Apply the preregistered conditional F4.7 decision to measured rows.

    Native inference is considered material only when Python tensor construction
    plus the non-component PyO3 envelope consumes at least 15% of end-to-end
    wall time and removing it has an Amdahl upper bound of at least 10%.
    """

    shares = []
    for row in rust_rows:
        component = (
            row["pyo3_tensor_seconds"]
            + row["h2d_seconds"]
            + row["gpu_forward_seconds"]
            + row["gather_d2h_seconds"]
        )
        envelope = max(0.0, row["pyo3_call_seconds"] - component)
        removable = row["pyo3_tensor_seconds"] + envelope
        shares.append(min(1.0, removable / row["seconds"]))
    mean_share = statistics.mean(shares) if shares else 0.0
    upper_speedup = 1.0 / max(1e-9, 1.0 - mean_share)
    required = mean_share >= 0.15 and upper_speedup >= 1.10
    return {
        "decision_rule": {
            "minimum_removable_wall_fraction": 0.15,
            "minimum_amdahl_speedup_upper_bound": 1.10,
        },
        "removable_wall_fraction_mean": mean_share,
        "amdahl_speedup_upper_bound": upper_speedup,
        "f4_7_required": required,
        "decision": "implement_native_inference" if required else "skip_native_inference",
    }


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _lock_sha256(args, lock: dict) -> str:
    if args.quality_lock is not None:
        return _sha256(args.quality_lock)
    return hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()


def _manifest(args, contract: dict, lock: dict) -> dict:
    return {
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "dirty_worktree": bool(_command_output(["git", "status", "--porcelain"])),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config_sha256": hashlib.sha256(
            json.dumps({"contract": contract, "lock": lock}, sort_keys=True).encode()
        ).hexdigest(),
        "contract_sha256": _sha256(CONTRACT_PATH),
        "contract_schema_version": contract["schema_version"],
        "quality_lock_sha256": _lock_sha256(args, lock),
        "exploratory_leaf1": bool(args.exploratory_leaf1),
        "device": args.device,
        "inference_precision": lock["inference_precision"],
        "python_version": sys.version,
        "rustc_version": _command_output(["rustc", "--version"]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_model": platform.processor() or platform.machine(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.limit,power.draw,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cpu_thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "warmup_games": args.warmup_games,
        "measured_games": args.games,
        "repetitions": args.repetitions,
        "slots": args.slots,
        "global_batch_cap": args.global_batch_cap,
        "max_inflight_batches": args.max_inflight_batches,
        "scheduler_workers": args.scheduler_workers,
        "pinned_memory": args.pinned_memory,
        "torch_compile": args.torch_compile,
        "diagnostic_sync": args.diagnostic_sync,
        "isolated_forward_rows_per_second": args.isolated_forward_rows_per_second,
        "exploratory_seed_start": args.exploratory_seed_start,
        "exact_command": subprocess.list2cmdline(sys.argv),
    }


def run(args) -> dict:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    frozen = contract["laptop_comparative_benchmark"]
    if args.exploratory_leaf1:
        lock = {
            "schema_version": "f4-exploratory-leaf1-1",
            "contract_schema_version": contract["schema_version"],
            "contract_sha256": _sha256(CONTRACT_PATH),
            "leaf_batch": 1,
            "force_expand_root_chance": True,
            "inference_precision": "float32",
            "production_search": frozen["search"],
            "age_deal_sampler": {
                "method": "paired_common_outcome_sampling",
                "sample_count": args.exploratory_age_deal_samples,
                "diagnostic_reference_count": 32,
            },
            "checkpoint_sha256": _sha256(args.checkpoint),
            "gate_results": {"exploratory": True},
        }
    else:
        if args.exploratory_seed_start is not None:
            raise ValueError("exploratory seed override requires --exploratory-leaf1")
        lock = json.loads(args.quality_lock.read_text(encoding="utf-8"))
        required_lock_fields = contract["quality_lock_format"]["required_fields"]
        missing = [field for field in required_lock_fields if field not in lock]
        if missing:
            raise ValueError(f"quality lock is missing required fields: {missing}")
        if lock.get("schema_version") != contract["quality_lock_format"]["schema_version"]:
            raise ValueError("unsupported F4 quality-lock schema")
        if lock.get("contract_schema_version") != contract["schema_version"]:
            raise ValueError("quality lock mixes an incompatible F4 contract version")
        gates = lock.get("gate_results", {})
        position_gate = gates.get("position_summary", {})
        strength_gate = gates.get("playing_strength_summary", {})
        frontier_gate = gates.get("concurrency_frontier_summary") or {}
        if lock["leaf_batch"] == 1:
            if not frontier_gate.get("leaf1_reaches_gpu_knee"):
                raise ValueError("leaf_batch=1 lock lacks an eligible concurrency frontier")
        else:
            if position_gate.get("largest_position_eligible_leaf_batch") != lock["leaf_batch"]:
                raise ValueError("quality lock position result does not approve leaf_batch")
            if not strength_gate.get("eligible"):
                raise ValueError("quality lock playing-strength result is not eligible")
        if lock["contract_sha256"] != _sha256(CONTRACT_PATH):
            raise ValueError("quality lock does not match f4_contract_v2.json")
        if lock["checkpoint_sha256"] != _sha256(args.checkpoint):
            raise ValueError("benchmark checkpoint does not match quality lock")
    minimums_met = (
        args.warmup_games >= frozen["minimum_warmup_games"]
        and args.games >= frozen["minimum_measured_games_per_repetition"]
        and args.repetitions >= frozen["minimum_repetitions"]
    )
    if args.mode == "laptop" and not minimums_met and not args.allow_underfilled:
        raise ValueError("laptop run is below the preregistered sample minimums")
    search = dict(lock["production_search"])
    if search != frozen["search"]:
        raise ValueError("quality lock production search differs from the registered schedule")
    if not search.get("force_expand_root_chance") or lock["force_expand_root_chance"] is not True:
        raise ValueError("production throughput requires forced root chance")
    evaluator = load_evaluator(str(args.checkpoint), args.device)
    evaluator.max_batch = args.global_batch_cap
    if args.torch_compile != "none":
        evaluator.model = torch.compile(evaluator.model, mode=args.torch_compile)
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rows.jsonl"
    run_config = {
        "schema": "f4-throughput-run-2",
        "contract_schema_version": contract["schema_version"],
        "mode": args.mode,
        "contract_sha256": _sha256(CONTRACT_PATH),
        "quality_lock_sha256": _lock_sha256(args, lock),
        "exploratory_leaf1": bool(args.exploratory_leaf1),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "device": args.device,
        "warmup_games": args.warmup_games,
        "games": args.games,
        "repetitions": args.repetitions,
        "slots": args.slots,
        "global_batch_cap": args.global_batch_cap,
        "max_inflight_batches": args.max_inflight_batches,
        "scheduler_workers": args.scheduler_workers,
        "python_workers": args.python_workers,
        "pinned_memory": args.pinned_memory,
        "torch_compile": args.torch_compile,
        "diagnostic_sync": args.diagnostic_sync,
        "isolated_forward_rows_per_second": args.isolated_forward_rows_per_second,
        "exploratory_seed_start": args.exploratory_seed_start,
    }
    config_path = args.output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("existing benchmark output uses a different run configuration")
    else:
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rows = (
        [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
        if rows_path.exists()
        else []
    )
    done = {(row["implementation"], int(row["repetition"])) for row in rows}
    seed_start = (
        args.exploratory_seed_start
        if args.exploratory_seed_start is not None
        else frozen["game_seed_start"]
    )
    warmup_seeds = [seed_start - args.warmup_games + i for i in range(args.warmup_games)]
    if args.warmup_games:
        _run_rust(
            seeds=warmup_seeds,
            evaluator=evaluator,
            search=search,
            lock=lock,
            slots=args.slots,
            batch_cap=args.global_batch_cap,
            max_inflight=args.max_inflight_batches,
            scheduler_workers=args.scheduler_workers,
            diagnostic_sync=False,
            pinned_memory=args.pinned_memory,
        )
        if args.mode == "laptop":
            _run_python(
                seeds=warmup_seeds,
                evaluator=evaluator,
                search=search,
                workers=args.python_workers,
                batch_cap=args.global_batch_cap,
            )
    with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
        for repetition in range(args.repetitions):
            seeds = [
                seed_start + repetition * args.games + index
                for index in range(args.games)
            ]
            if args.mode == "laptop" and ("python", repetition) not in done:
                python_row = _run_python(
                    seeds=seeds,
                    evaluator=evaluator,
                    search=search,
                    workers=args.python_workers,
                    batch_cap=args.global_batch_cap,
                )
                python_row |= {"schema": SCHEMA, "repetition": repetition}
                rows.append(python_row)
                done.add(("python", repetition))
                handle.write(json.dumps(python_row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            if ("rust", repetition) not in done:
                rust_row = _run_rust(
                    seeds=seeds,
                    evaluator=evaluator,
                    search=search,
                    lock=lock,
                    slots=args.slots,
                    batch_cap=args.global_batch_cap,
                    max_inflight=args.max_inflight_batches,
                    scheduler_workers=args.scheduler_workers,
                    diagnostic_sync=args.diagnostic_sync,
                    pinned_memory=args.pinned_memory,
                )
                rust_row["isolated_forward_rows_ratio"] = (
                    rust_row["total_nn_rows_per_second"]
                    / args.isolated_forward_rows_per_second
                    if args.isolated_forward_rows_per_second > 0
                    else 0.0
                )
                missing_metrics = [
                    metric
                    for metric in contract["required_metrics"]
                    if metric not in rust_row
                ]
                if missing_metrics:
                    raise RuntimeError(
                        f"Rust benchmark omitted required metrics: {missing_metrics}"
                    )
                rust_row |= {"schema": SCHEMA, "repetition": repetition}
                rows.append(rust_row)
                done.add(("rust", repetition))
                handle.write(json.dumps(rust_row, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                print(
                    f"bench: repetition {repetition + 1}/{args.repetitions} "
                    f"rust={rust_row['games_per_second']:.3f} games/s",
                    flush=True,
                )
    rust_rows = sorted(
        (row for row in rows if row["implementation"] == "rust"),
        key=lambda row: row["repetition"],
    )
    summary = {
        "mode": args.mode,
        "manifest": _manifest(args, contract, lock),
        "sample_minimums_met": minimums_met,
        "rust_games_per_second_mean": statistics.mean(
            row["games_per_second"] for row in rust_rows
        ),
        "conditional_f4_7": native_inference_assessment(rust_rows),
        "rust_metrics_mean": {
            key: statistics.mean(float(row[key]) for row in rust_rows)
            for key in contract["required_metrics"]
            if rust_rows and all(isinstance(row.get(key), (int, float)) for row in rust_rows)
        },
    }
    if args.mode == "laptop":
        python_rows = sorted(
            (row for row in rows if row["implementation"] == "python"),
            key=lambda row: row["repetition"],
        )
        speed = speedup_summary(python_rows, rust_rows, frozen["game_seed_start"])
        summary["speedup"] = speed
        summary["eligible"] = (
            minimums_met
            and speed["speedup_one_sided_95_lower"]
            >= frozen["minimum_speedup_lower_confidence_bound"]
        )
    else:
        summary["eligible"] = True
    _write_rows_csv(rows, args.output / "rows.csv")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("laptop", "rust"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    lock_group = parser.add_mutually_exclusive_group(required=True)
    lock_group.add_argument("--quality-lock", type=Path)
    lock_group.add_argument("--exploratory-leaf1", action="store_true")
    parser.add_argument(
        "--exploratory-age-deal-samples", type=int, choices=(4, 8, 16, 32), default=32
    )
    parser.add_argument("--exploratory-seed-start", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-games", type=int, default=16)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--global-batch-cap", type=int, default=256)
    parser.add_argument("--max-inflight-batches", type=int, default=2)
    parser.add_argument("--scheduler-workers", type=int, default=1)
    parser.add_argument("--python-workers", type=int, default=4)
    parser.add_argument("--diagnostic-sync", action="store_true")
    parser.add_argument("--isolated-forward-rows-per-second", type=float, default=0.0)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument(
        "--torch-compile",
        choices=("none", "reduce-overhead", "max-autotune"),
        default="none",
    )
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--record-failures", action="store_true")
    args = parser.parse_args()
    if args.record_failures:
        try:
            result = run(args)
        except Exception as exc:  # operational sweep row, preserved for ranking
            args.output.mkdir(parents=True, exist_ok=True)
            message = f"{type(exc).__name__}: {exc}"
            result = {
                "mode": args.mode,
                "eligible": False,
                "error": message,
                "oom_count": int("out of memory" in message.lower()),
                "exact_command": subprocess.list2cmdline(sys.argv),
            }
            (args.output / "summary.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
