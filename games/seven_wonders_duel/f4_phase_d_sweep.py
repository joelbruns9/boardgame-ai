"""Joint sweep of the free scheduler axes on the real Phase D generation path.

The axes here are the ones that do not change what the search computes, so they
can be chosen on wall clock alone: `rust_slots`, `rust_global_batch_cap` and
`rust_max_inflight_batches`. They are swept *jointly* because they interact --
the cap binds as slots rise, so sweeping slots alone at a fixed cap finds a
ceiling that belongs to the cap and misattributes it to slots.

Two things this measures that the F4 benchmark cannot:

* **the real job mix.** Phase D spends ~15% of its games on curriculum bots,
  split across `(bot type, seat)` groups that go to the Rust scheduler as
  separate calls. A small group cannot fill a large slot pool, so the benefit of
  more slots is diluted by exactly the fraction of games that are bot games.
  Per-group timings are recorded so that dilution is visible rather than baked
  into one number.
* **whether these axes are really free.** They do not change the search, but
  they do change batch composition, and a different batch shape can change
  floating-point reductions on CUDA. Trajectory fingerprints are compared across
  every point; divergence is reported, not asserted away, because on the real
  net it is information about the axis rather than a bug.

Order is reversed on alternate repetitions so that thermal drift on a laptop GPU
averages out instead of loading onto whichever point runs last.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from . import phase_d as pd
from .train import heads_from_config


def timed_scheduler_calls():
    """Wrap the Rust entry point to record (games, seconds) per group call."""

    import seven_wonders_rust as swr

    real = swr.self_play_many_flat_net
    calls: list[dict] = []

    def wrapper(*args, **kwargs):
        games = len(kwargs.get("game_seeds", ()))
        # Per-game routing puts bot and neural games in one call, so a call is
        # no longer either/or: count the bot games inside it. The scalar form is
        # still used by the seed buffer and the arena, where a call is uniform.
        per_game = list(kwargs.get("bots_p0") or ()) + list(kwargs.get("bots_p1") or ())
        if per_game:
            bot_games = sum(
                1
                for left, right in zip(kwargs["bots_p0"], kwargs["bots_p1"])
                if left is not None or right is not None
            )
            bot = "per-game"
        else:
            bot = kwargs.get("bot_p0") or kwargs.get("bot_p1")
            bot_games = games if bot else 0
        started = time.monotonic()
        result = real(*args, **kwargs)
        # `(records, metrics)`. The batch widths are the number this sweep
        # exists to move: a run whose mean batch is 42 against a 2,048 cap is
        # paying per-call overhead, not compute, and no wall-clock total says
        # so on its own.
        rows: list[int] = []
        try:
            rows = [int(value) for value in (result[1] or {}).get("batch_rows", ())]
        except Exception:  # pragma: no cover - metrics shape is the contract
            pass
        calls.append(
            {
                "games": games,
                "bot_games": bot_games,
                "seconds": time.monotonic() - started,
                "bot": bot,
                "batch_rows": rows,
            }
        )
        return result

    swr.self_play_many_flat_net = wrapper
    return calls, (lambda: setattr(swr, "self_play_many_flat_net", real))


def geometry_from_checkpoint(path) -> dict[str, int]:
    """The model width the checkpoint was trained at.

    Read, never assumed or passed as a flag. `_load_model_checkpoint` refuses a
    width mismatch -- W0 lost a run to a checkpoint whose width was inferred --
    so a sweep left on `PhaseDConfig`'s 128x4 defaults cannot load an L
    checkpoint at all, which is how this stage died on its first cloud box.
    `w5_gate_slots_sweep` and `w5_gate_bench` already do exactly this.
    """

    stored = torch.load(path, map_location="cpu", weights_only=False).get("config", {})
    return {
        "d_model": int(stored.get("d_model", 384)),
        "layers": int(stored.get("layers", 8)),
        "heads": heads_from_config(stored),
    }


def field_default(name: str):
    """A PhaseDConfig default, read properly.

    `PhaseDConfig` is `@dataclass(slots=True)`, so `PhaseDConfig.rust_slots` is
    a slot DESCRIPTOR, not 16. The baseline comparison below used class
    attributes and therefore never matched any grid row -- it silently fell back
    to the first point for every sweep ever run.
    """

    for field in dataclasses.fields(pd.PhaseDConfig):
        if field.name == name:
            return field.default
    raise KeyError(name)


def config_from_manifest(
    manifest_path, *, output, device, games, precision, geometry
) -> "pd.PhaseDConfig":
    """The RUN's configuration, with only what the sweep controls overridden.

    Without this the sweep measured PhaseDConfig's defaults for everything it
    did not name -- `selfplay_search_mode="gumbel"` against a PUCT run,
    `full_sims_max=128` against 1600, `cheap_sims_max=24` against 100. Roughly
    50 simulations a move instead of the run's measured 522, under a different
    search algorithm. Simulations per move set the leaf arrival rate, which is
    what the slot and worker axes act on, so the optimum found that way belongs
    to a machine nobody is running.

    Same defect as the worker axis, one layer up: measuring a configuration
    other than the one being configured. Fields the manifest does not have are
    left at their defaults rather than guessed.
    """

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    stored = payload.get("config")
    if not isinstance(stored, dict) or not stored:
        raise SystemExit(
            f"{manifest_path} has no 'config' block; it is not a run manifest"
        )
    fields = {field.name for field in dataclasses.fields(pd.PhaseDConfig)}
    unknown = sorted(set(stored) - fields)
    kwargs = {key: value for key, value in stored.items() if key in fields}
    # The sweep owns these; the run's values would defeat the point.
    kwargs.update(
        run_dir=str(output / "run"),
        device=device,
        games_per_iteration=games,
        seed_games=0,
        iterations=1,
        precision=precision,
        **geometry,
    )
    config = pd.PhaseDConfig(**kwargs)
    print(
        f"config from {manifest_path}: search={config.selfplay_search_mode} "
        f"cheap_sims={config.cheap_sims_min}-{config.cheap_sims_max} "
        f"full_sims={config.full_sims_min}-{config.full_sims_max} "
        f"full_fraction={config.full_search_fraction} top_k={config.top_k}"
        + (f" ({len(unknown)} manifest fields ignored)" if unknown else ""),
        flush=True,
    )
    return config


def apply_config_overrides(config, overrides: list[str]) -> "pd.PhaseDConfig":
    """Set PhaseDConfig fields from `name=value` strings, typed by the field.

    The sweep reads its search settings from a run's manifest, which is correct
    once a run exists -- and useless before one does. Choosing geometry for a
    configuration that has never run is exactly the case here: leaf batching was
    decided by an A/B, and sweeping without it would optimise for
    `leaf_batch=1`, a setting nobody intends to use. Same defect as the sweep
    measuring Gumbel at 24/128 sims against a PUCT run at 100/1600.

    Names are checked against the dataclass, so a typo fails here rather than
    being silently ignored -- an override that does nothing is worse than none,
    because the sweep would report settings it did not measure.
    """

    if not overrides:
        return config
    types = {field.name: field.type for field in dataclasses.fields(pd.PhaseDConfig)}
    applied = {}
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--config-override {override!r} is not name=value")
        name, _, raw = override.partition("=")
        name, raw = name.strip(), raw.strip()
        if name not in types:
            raise SystemExit(
                f"--config-override {name!r} is not a PhaseDConfig field"
            )
        declared = str(types[name])
        if "bool" in declared:
            if raw.lower() not in ("true", "false", "1", "0"):
                raise SystemExit(f"{name} is a flag; use true or false")
            value = raw.lower() in ("true", "1")
        elif "int" in declared:
            value = int(raw)
        elif "float" in declared:
            value = float(raw)
        else:
            value = raw
        setattr(config, name, value)
        applied[name] = value
    # Re-validate: overrides can produce a combination the parser would refuse,
    # and finding that at the first grid point wastes the setup before it.
    config.validate()
    print(
        "config overrides: "
        + ", ".join(f"{name}={value}" for name, value in sorted(applied.items())),
        flush=True,
    )
    return config


def steady_state_schedules(loop) -> "pd.ResolvedSchedules":
    """Schedules for the part of the run this geometry has to serve.

    Both curriculum bots and the draft prior anneal out over the first ~10k
    games, so 190 of a 200-iteration run sees neither. Sweeping with a bot mix
    would measure a different cost curve on purpose: bot games run ~0.52
    games/s at every slot count while neural games scale, so they dilute exactly
    the axis being swept.
    """

    return pd.ResolvedSchedules(curriculum_mix_fraction=0.0, draft_prior=0.0)


def run_point(
    loop, model, iteration, jobs, destination, slots, cap, inflight,
    workers=1, solver_threads=0,
):
    loop.config.rust_slots = slots
    loop.config.rust_global_batch_cap = cap
    loop.config.rust_max_inflight_batches = inflight
    loop.config.rust_scheduler_workers = workers
    # Per SHARD, so the load this point puts on the CPU is threads x workers.
    # Applied per point rather than once at startup: the total moves with the
    # worker axis, and a sweep that solved on one thread count while measuring
    # another would attribute the solver's core contention to the axis.
    pd.configure_solver_threads(solver_threads, workers)

    calls, restore = timed_scheduler_calls()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.monotonic()
        records = loop._generate_iteration_rust(
            model, iteration, destination, jobs, steady_state_schedules(loop)
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.monotonic() - started
    finally:
        restore()

    # Uniform calls split cleanly; a per-game call carries both kinds at once,
    # in which case there is one pool and the split is no longer a timing.
    neural = [call for call in calls if call["bot"] is None]
    bots = [call for call in calls if call["bot"] not in (None, "per-game")]
    mixed = [call for call in calls if call["bot"] == "per-game"]
    fingerprint = tuple(
        (
            record.winner,
            record.trajectory_digest,
            record.final_digest,
            tuple(move.action for move in record.moves),
        )
        for record in records
    )
    batch_rows = [row for call in calls for row in call.get("batch_rows", ())]
    stats = {
        "slots": slots,
        "global_batch_cap": cap,
        "max_inflight_batches": inflight,
        "scheduler_workers": workers,
        "solver_threads_per_shard": solver_threads,
        "solver_threads_total": solver_threads * workers,
        "mean_batch_size": statistics.fmean(batch_rows) if batch_rows else 0.0,
        "max_batch_size": max(batch_rows) if batch_rows else 0,
        "batches": len(batch_rows),
        "wall_seconds": wall,
        "games": len(records),
        "games_per_second": len(records) / wall if wall else 0.0,
        "games_per_hour": 3600 * len(records) / wall if wall else 0.0,
        "neural_games": sum(call["games"] for call in neural),
        "neural_seconds": sum(call["seconds"] for call in neural),
        "bot_games": sum(call["games"] for call in bots),
        "bot_seconds": sum(call["seconds"] for call in bots),
        "bot_groups": len(bots),
        "mixed_calls": len(mixed),
        "mixed_games": sum(call["games"] for call in mixed),
        "mixed_bot_games": sum(call["bot_games"] for call in mixed),
        "mixed_seconds": sum(call["seconds"] for call in mixed),
        "scheduler_calls": len(calls),
        "calls": calls,
    }
    for prefix in ("neural", "bot"):
        games = stats[f"{prefix}_games"]
        seconds = stats[f"{prefix}_seconds"]
        stats[f"{prefix}_games_per_second"] = games / seconds if seconds else 0.0
    return stats, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--warmup-games", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="must match the run being configured: bf16 is 1.69x on L, so a "
        "geometry chosen at fp32 is chosen against the wrong cost curve",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a PhaseDConfig field after the manifest is read; repeatable. "
        "Needed to sweep a configuration no run has used yet, e.g. "
        "--config-override leaf_batch=6 --config-override virtual_loss_root=true",
    )
    parser.add_argument(
        "--config-from-manifest",
        default="",
        help="run_manifest.json of the run being configured. Its config block "
        "supplies every field the sweep does not override, so the measurement "
        "runs the same search the run runs. Without it the search settings are "
        "dataclass defaults.",
    )
    parser.add_argument("--slots", default="16,32,48")
    parser.add_argument("--caps", default="256,512")
    parser.add_argument("--inflight", default="1,2")
    parser.add_argument(
        "--workers",
        default="1",
        help="scheduler shard counts to sweep. Swept rather than fixed because "
        "`rust_slots` is a GLOBAL budget shared across shards: an optimum found "
        "at one shard is not the optimum at four, and this harness used to "
        "measure at PhaseDConfig's default of 1 while configuring a run at 4.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        default=0,
        help="solver threads PER SHARD, as the run passes them. The default of "
        "0 measures with the solver off, which is the wrong cost curve for a "
        "run that solves: those threads compete with generation for cores.",
    )
    parser.add_argument(
        "--solver-threads-total",
        type=int,
        default=0,
        help="total solver threads, divided across shards at each point. Use "
        "this whenever --workers has more than one value: --solver-threads is "
        "PER SHARD, so holding it fixed across the worker axis silently varies "
        "the solver load with the axis being measured, and fewer shards would "
        "lose partly because they were under-solving. Overrides "
        "--solver-threads.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    numbers = lambda text: [int(part) for part in text.split(",") if part.strip()]
    grid = list(
        itertools.product(
            numbers(args.slots),
            numbers(args.caps),
            numbers(args.inflight),
            numbers(args.workers),
        )
    )
    # A shard with no slot cannot make progress, and `SlotBudget::new` refuses
    # the combination rather than widening it. Drop those points here so the
    # sweep reports a grid it actually measured instead of dying partway.
    dropped = [point for point in grid if point[0] < point[3]]
    grid = [point for point in grid if point[0] >= point[3]]
    if dropped:
        print(
            f"skipping {len(dropped)} point(s) with fewer slots than shards: "
            f"{sorted({(slots, workers) for slots, _, _, workers in dropped})}",
            flush=True,
        )
    if not grid:
        raise SystemExit("every grid point has fewer slots than shards")

    def solver_threads_for(workers: int) -> int:
        """Per-shard threads at this point, holding the TOTAL fixed when asked."""

        if args.solver_threads_total <= 0:
            return args.solver_threads
        return max(1, args.solver_threads_total // max(1, workers))

    if args.solver_threads_total > 0 and len(numbers(args.workers)) > 1:
        print(
            "solver: holding "
            f"{args.solver_threads_total} threads TOTAL across the worker axis "
            + ", ".join(
                f"{workers}x{solver_threads_for(workers)}"
                for workers in numbers(args.workers)
            ),
            flush=True,
        )

    geometry = geometry_from_checkpoint(args.checkpoint)
    run_config = None
    if args.config_from_manifest:
        config = run_config = config_from_manifest(
            args.config_from_manifest,
            output=output,
            device=args.device,
            games=args.games,
            precision=args.precision,
            geometry=geometry,
        )
    else:
        print(
            "WARNING: no --config-from-manifest, so every search setting takes "
            f"its DEFAULT: {field_default('selfplay_search_mode')} search, "
            f"{field_default('cheap_sims_max')}/{field_default('full_sims_max')} "
            "sims. If the run being configured differs, this measures a cost "
            "curve that is not its own.",
            flush=True,
        )
        config = pd.PhaseDConfig(
            run_dir=str(output / "run"),
            device=args.device,
            games_per_iteration=args.games,
            seed_games=0,
            iterations=1,
            precision=args.precision,
            **geometry,
        )
    config = apply_config_overrides(config, args.config_override)

    # SNAPSHOT before anything runs. `run_point` assigns loop.config.rust_slots
    # and friends at every grid point, and `run_config` is the same object, so
    # reading the run's geometry after the grid reports the LAST POINT MEASURED
    # instead -- a baseline that silently renames itself.
    run_baseline = (
        (
            run_config.rust_slots,
            run_config.rust_global_batch_cap,
            run_config.rust_max_inflight_batches,
            run_config.rust_scheduler_workers,
        )
        if run_config is not None
        else None
    )
    loop = pd.PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    model = loop.load_model(args.checkpoint)

    def jobs_for(count: int, iteration: int):
        return [
            pd.GameJob(index=index, seed=config.seed + iteration * 1_000_000 + index)
            for index in range(count)
        ]

    if args.warmup_games:
        print(f"warmup: {args.warmup_games} games", flush=True)
        run_point(
            loop, model, args.iteration, jobs_for(args.warmup_games, 999),
            output / "warmup.jsonl", *grid[0],
            solver_threads=solver_threads_for(grid[0][3]),
        )

    jobs = jobs_for(args.games, args.iteration)
    if args.games <= max(numbers(args.slots)):
        print(
            f"WARNING: {args.games} games <= {max(numbers(args.slots))} slots; the "
            "pool cannot refill at the top of the grid, so that point measures "
            "activation rather than throughput",
            flush=True,
        )

    results: list[dict] = []
    fingerprints: set = set()
    for repetition in range(args.repetitions):
        order = grid if repetition % 2 == 0 else list(reversed(grid))
        for position, (slots, cap, inflight, workers) in enumerate(order):
            stats, fingerprint = run_point(
                loop, model, args.iteration, jobs,
                output
                / f"r{repetition}_{position:02d}_s{slots}_c{cap}_i{inflight}_w{workers}.jsonl",
                slots, cap, inflight, workers, solver_threads_for(workers),
            )
            stats["repetition"] = repetition
            results.append(stats)
            fingerprints.add(fingerprint)
            if stats["mixed_calls"]:
                detail = (
                    f"| 1 pool: {stats['mixed_games']} games "
                    f"({stats['mixed_bot_games']} bot) in "
                    f"{stats['scheduler_calls']} call(s)"
                )
            else:
                detail = (
                    f"| neural {stats['neural_games_per_second']:.3f} g/s, "
                    f"bots {stats['bot_games_per_second']:.3f} g/s over "
                    f"{stats['bot_groups']} groups"
                )
            print(
                f"slots={slots:<4} cap={cap:<4} inflight={inflight} workers={workers:<2} "
                f"batch={stats['mean_batch_size']:6.1f}  "
                f"{stats['wall_seconds']:7.1f}s  {stats['games_per_hour']:7.0f} games/h  "
                + detail,
                flush=True,
            )

    summary = []
    for point in grid:
        slots, cap, inflight, workers = point
        matching = [
            row for row in results
            if (
                row["slots"],
                row["global_batch_cap"],
                row["max_inflight_batches"],
                row["scheduler_workers"],
            ) == point
        ]
        summary.append(
            {
                "slots": slots,
                "global_batch_cap": cap,
                "max_inflight_batches": inflight,
                "scheduler_workers": workers,
                "solver_threads_per_shard": solver_threads_for(workers),
                "solver_threads_total": solver_threads_for(workers) * workers,
                "median_seconds": statistics.median(row["wall_seconds"] for row in matching),
                "median_games_per_hour": statistics.median(
                    row["games_per_hour"] for row in matching
                ),
                "median_batch_size": statistics.median(
                    row["mean_batch_size"] for row in matching
                ),
                "runs": len(matching),
            }
        )
    summary.sort(key=lambda row: row["median_seconds"])
    # Compare against Phase D's current defaults when they are in the grid --
    # the number that matters is "what would changing the config buy" -- and
    # fall back to the first grid point when they are not.
    key = lambda row: (
        row["slots"],
        row["global_batch_cap"],
        row["max_inflight_batches"],
        row["scheduler_workers"],
    )
    # Baseline: what the run is CURRENTLY set to, when the manifest says and the
    # grid contains it. "What would changing this buy me" is the question a
    # sweep is run to answer, and it is only answered against the status quo --
    # comparing against dataclass defaults answers a question nobody asked.
    baselines = []
    if run_baseline is not None:
        baselines.append(run_baseline)
    baselines.append(
        tuple(
            field_default(name)
            for name in (
                "rust_slots",
                "rust_global_batch_cap",
                "rust_max_inflight_batches",
                "rust_scheduler_workers",
            )
        )
    )
    baselines.append(grid[0])
    base = None
    for candidate in baselines:
        base = next((row for row in summary if key(row) == tuple(candidate)), None)
        if base is not None:
            break
    if base is None:
        base = summary[0]
    baseline_label = "/".join(str(part) for part in key(base))
    for row in summary:
        row["speedup_vs_baseline"] = base["median_seconds"] / row["median_seconds"]
    payload_baseline = baseline_label

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "games": args.games,
            "repetitions": args.repetitions,
            "grid": [list(point) for point in grid],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "opponent_fraction": config.opponent_fraction,
        },
        "runs": results,
        "summary": summary,
        "baseline": payload_baseline,
        "distinct_trajectory_sets": len(fingerprints),
    }
    (output / "phase_d_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nbest first:")
    for row in summary:
        print(
            f"  slots={row['slots']:<4} cap={row['global_batch_cap']:<5} "
            f"inflight={row['max_inflight_batches']} "
            f"workers={row['scheduler_workers']:<2} "
            f"batch={row['median_batch_size']:5.0f}  "
            f"{row['median_games_per_hour']:7.0f} games/h  "
            f"({row['speedup_vs_baseline']:.2f}x vs {baseline_label})"
        )
    print(
        f"\ndistinct trajectory sets across all points: {len(fingerprints)} "
        "(1 = these axes changed nothing the search saw)"
    )


if __name__ == "__main__":
    main()
