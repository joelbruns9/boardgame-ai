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
import pathlib
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
        wave = 0.0
        # Worker-side counters, summed across calls in `run_point`. Kept here
        # because this wrapper is the only place the metrics dict is seen.
        coalescing = {
            key: 0
            for key in (
                "worker_requests",
                "boundary_forwards",
                "boundary_forward_rows",
                "coalesce_wait_ns",
                "coalesce_carried",
                # The SOLVER's share of slot occupancy, time-weighted. A parked
                # slot yields no evaluation group until its solve returns, so
                # this is the concurrency the solver takes away from generation
                # -- and it is the quantity `--sims-divisor` assumes it holds
                # roughly fixed. Recorded so that assumption is checkable
                # against the live run's profile rather than asserted.
                "parked_slot_ns",
                "live_slot_ns",
            )
        }
        try:
            metrics = result[1] or {}
            for key in coalescing:
                coalescing[key] = int(metrics.get(key, 0) or 0)
            rows = [int(value) for value in metrics.get("batch_rows", ())]
            # Leaves in flight for ONE game. Batch width is leaves summed across
            # games, so batch alone cannot distinguish "leaf batching is not
            # engaging" from "it engages and buys nothing" -- and those lead to
            # opposite decisions.
            wave = float(metrics.get("mean_wave_width") or 0.0)
        except Exception:  # pragma: no cover - metrics shape is the contract
            pass
        calls.append(
            {
                "games": games,
                "bot_games": bot_games,
                "seconds": time.monotonic() - started,
                "bot": bot,
                "batch_rows": rows,
                "mean_wave_width": wave,
                **coalescing,
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


SIMS_FIELDS = ("cheap_sims_min", "cheap_sims_max", "full_sims_min", "full_sims_max")


def apply_sims_divisor(config, divisor: int) -> "pd.PhaseDConfig":
    """Divide the SEARCH BUDGET by `divisor`, keeping the search's shape.

    `--config-from-manifest` is what makes this necessary. Before it, the sweep
    measured `PhaseDConfig`'s laptop defaults -- 24 cheap and 128 full sims of
    Gumbel -- while configuring a run at 100 and 1600 of PUCT. Fixing that
    multiplied the box hours by roughly the same factor it fixed the fidelity
    by, against a grid that is already dozens of points times two repetitions.

    So: the honest cheap sweep is not a DIFFERENT search, it is the SAME search
    run shallower. What survives the divisor:

    * the search ALGORITHM (`selfplay_search_mode`, `cheap_search_mode`) -- the
      thing whose absence made the old default sweep meaningless;
    * the cheap/full MIX (`full_search_fraction`, `full_search_every_games`), so
      the leaf arrival rate keeps the run's bimodal shape rather than averaging
      into one;
    * `top_k` and the chance fan-out, so a full search still splits its visits
      across the same number of worlds;
    * the SOLVER-to-generation ratio, because the caller divides the node budget
      by the same factor -- see the caveat below.

    What does NOT survive, and must not be read off a divided sweep:

    * absolute throughput. `games_per_hour` at divisor 4 is roughly 4x the run's
      rate and is not a prediction of anything. The ratios BETWEEN points are
      what this sweep is for; the absolute number belongs to the undivided run.
    * the per-move leaf arrival RATE, which is exactly what the slot and worker
      axes act on. A large divisor measures a lighter machine, and a geometry
      chosen there is chosen for it. 4 keeps a 1600-sim full search at 400,
      which is still deeper than anything the old default sweep ever ran.

    The solver caveat, stated rather than buried: dividing the node budget is
    the closest cheap approximation, not an identity. The budget is an
    ADMISSION THRESHOLD, so halving it both shortens each solve and refuses the
    expensive positions -- which carry most of the nodes. Leaving it alone
    instead would over-weight the solver by the full divisor, since games finish
    that much faster while each solve costs the same. Neither is exact, the
    divided budget is the nearer of the two, and `parked_slot_fraction` is
    recorded at every point so the residual is a number the operator can compare
    against the live run's profile instead of a hope.
    """

    if divisor <= 1:
        return config
    before = {name: getattr(config, name) for name in SIMS_FIELDS}
    for name in SIMS_FIELDS:
        setattr(config, name, max(1, round(before[name] / divisor)))
    # Re-validate: `validate` enforces `1 <= min <= max` on both pairs, and
    # rounding can collapse a narrow band -- finding that at the first grid
    # point would waste the model load and the warmup before it.
    config.validate()
    print(
        f"sims divisor {divisor}x: "
        + ", ".join(
            f"{name}={before[name]}->{getattr(config, name)}" for name in SIMS_FIELDS
        )
        + " (ratios between points are the result; absolute games/hour is NOT "
        "the run's rate)",
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


def _find_manifest_value(payload, key):
    """First occurrence of `key` anywhere in a nested manifest."""

    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_manifest_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_manifest_value(value, key)
            if found is not None:
                return found
    return None


def run_point(
    loop, model, iteration, jobs, destination, slots, cap, inflight,
    workers=1, solver_threads=0, solver_max_nodes=0, solver_max_secs=0.0,
    inference_wait_ms=0.0, solver_attempt_nodes=0,
):
    loop.config.rust_slots = slots
    loop.config.rust_global_batch_cap = cap
    loop.config.rust_max_inflight_batches = inflight
    loop.config.rust_scheduler_workers = workers
    # The coalescing wait. It only has anything to buy when there is more than
    # one shard to merge ACROSS -- at workers=1 the ratio is 1.00 by
    # construction -- so this axis is read together with the worker axis, never
    # alone.
    loop.config.rust_inference_wait_ms = inference_wait_ms
    # Per SHARD, so the load this point puts on the CPU is threads x workers.
    # Applied per point rather than once at startup: the total moves with the
    # worker axis, and a sweep that solved on one thread count while measuring
    # another would attribute the solver's core contention to the axis.
    pd.configure_solver_threads(solver_threads, workers)
    # THE NODE BUDGET, which is what makes the solver run at all.
    #
    # This function used to size the thread pool and never install the budget.
    # `endgame_solver()` then reported `max_nodes = 0`, `solver_wants` returned
    # false at every position, and EVERY POINT WAS MEASURED WITH SOLVING
    # DISABLED -- while the harness printed the thread count it had dutifully
    # configured. `THROUGHPUT_LEVERS.md` §3.1 records this exact incident as
    # having already cost a day of rented box; it was documented and not fixed.
    #
    # `solve_endgames` is the second half: it is a per-CALL grant, so a budget
    # without it still solves nothing.
    solving = solver_threads > 0 and solver_max_nodes > 0
    # `_generate_iteration_rust` derives the per-call grant from this field
    # (`solve_endgames=self.config.endgame_solver_max_nodes > 0`), so setting it
    # is what the RUN does -- one source of truth rather than a second switch
    # the sweep could set differently.
    loop.config.endgame_solver_max_nodes = solver_max_nodes if solving else 0
    if solving:
        # The ATTEMPT BAR as well as the timeout. They are separate settings on
        # the run, and passing only the timeout would leave the bar defaulting
        # to it -- which on a run that narrows the bar admits a far larger set of
        # positions than the run does, so the solver load being measured is not
        # the run's. Same defect as measuring a solver that never ran, inverted.
        pd.configure_endgame_solver(
            solver_max_nodes,
            solver_max_secs or float(solver_max_nodes) / 1_200_000.0 * 5.0,
            0,
            True,
            solver_attempt_nodes,
        )
    else:
        pd.configure_endgame_solver(0, 1.0, 0, False)

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
    # Python-side boundary counters, published by `_generate_iteration_rust`.
    boundary_model_forwards = int(
        (getattr(loop, "last_generation_stats", {}) or {})
        .get("rust_boundary", {})
        .get("model_forwards", 0)
    )
    scheduler = {
        key: sum(call.get(key, 0) for call in calls)
        for key in (
            "worker_requests",
            "boundary_forwards",
            "boundary_forward_rows",
            "coalesce_wait_ns",
            "coalesce_carried",
            "parked_slot_ns",
            "live_slot_ns",
        )
    }
    waves = [call["mean_wave_width"] for call in calls if call.get("mean_wave_width")]
    stats = {
        "slots": slots,
        "global_batch_cap": cap,
        "max_inflight_batches": inflight,
        "scheduler_workers": workers,
        "inference_wait_ms": inference_wait_ms,
        # ENGAGEMENT, the same way `solves_attempted` is liveness rather than a
        # thread count. `requests_per_forward` is exactly 1.00 when nothing
        # merged, so a point that reports a wait and a ratio of 1.00 measured a
        # coalescer that did not run -- and the wall clock alone would call that
        # "the wait didn't help".
        "worker_requests": int(scheduler.get("worker_requests", 0)),
        "boundary_forwards": int(scheduler.get("boundary_forwards", 0)),
        "requests_per_forward": (
            int(scheduler.get("worker_requests", 0))
            / int(scheduler.get("boundary_forwards", 1) or 1)
        ),
        "mean_forward_rows": (
            int(scheduler.get("boundary_forward_rows", 0))
            / int(scheduler.get("boundary_forwards", 1) or 1)
        ),
        "coalesce_wait_seconds": int(scheduler.get("coalesce_wait_ns", 0)) / 1e9,
        "coalesce_carried": int(scheduler.get("coalesce_carried", 0)),
        # Slot-time parked on a solve, as a share of slot-time live. This is
        # what "the solver costs concurrency" means as a number: at 0.30 the
        # solver is holding three slots in ten, and the slot axis is really
        # being swept at 70% of its label.
        "parked_slot_ns": int(scheduler.get("parked_slot_ns", 0)),
        "live_slot_ns": int(scheduler.get("live_slot_ns", 0)),
        "parked_slot_fraction": (
            int(scheduler.get("parked_slot_ns", 0))
            / int(scheduler.get("live_slot_ns", 1) or 1)
        ),
        # GPU-side forwards, from the Python adapter. A routed model runs one
        # per network present, so this is the number that says whether a merge
        # actually reduced GPU work or only the boundary hop.
        "model_forwards": boundary_model_forwards,
        "model_forwards_per_forward": (
            boundary_model_forwards / int(scheduler.get("boundary_forwards", 1) or 1)
        ),
        "solver_threads_per_shard": solver_threads,
        "solver_threads_total": solver_threads * workers,
        # LIVENESS, not configuration. `THROUGHPUT_LEVERS.md` §3.1: "assert
        # every subsystem is live, not merely configured -- print a count of
        # work each one actually did". A thread count says what was asked for;
        # this says what happened.
        "solves_attempted": sum(
            1 for record in records for move in record.moves if move.solver_attempted
        ),
        # Attempts carrying a PREDICTION, which is what says the cost model was
        # actually installed rather than the card cap having admitted something.
        # An attempt count alone cannot tell those apart, and the card-cap path
        # is exactly how this harness measured zero solves while looking busy.
        "solves_with_prediction": sum(
            1
            for record in records
            for move in record.moves
            if move.solver_attempted and move.solver_predicted_nodes is not None
        ),
        "solves_answered": sum(
            1
            for record in records
            for move in record.moves
            if move.solver_value is not None
        ),
        "mean_batch_size": statistics.fmean(batch_rows) if batch_rows else 0.0,
        "mean_wave_width": statistics.fmean(waves) if waves else 0.0,
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


def main(argv: list[str] | None = None) -> dict:
    """Run the grid and return the payload it wrote.

    Takes `argv` and returns the payload so a DRIVER can stage this harness --
    `f4_staged_sweep` runs it twice, reads the first stage's winner and pins it
    for the second. Both would be possible over subprocesses and a JSON file;
    doing it in process keeps one parser as the single place an axis is
    validated, and keeps a driver from having to re-quote every flag.
    """

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
        "--sims-divisor",
        type=int,
        default=1,
        help="divide the run's simulation budget (and, with it, the solver's "
        "node budget and deadline) by this factor at every point. The default "
        "of 1 measures the run's own search. Use it to buy grid points with "
        "box hours once --config-from-manifest has made each point cost what "
        "the run costs: it keeps the search ALGORITHM, the cheap/full mix, "
        "top_k and the solver-to-generation ratio, and gives up absolute "
        "throughput -- games/hour from a divided sweep is not the run's rate, "
        "only the ratios between points are. See `apply_sims_divisor`.",
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
        "--inference-wait-ms",
        default="0",
        help="coalescing-wait axis, comma separated (e.g. \"0,1,2\"). 0 still "
        "merges everything already queued; a positive value blocks that long "
        "for more. It buys nothing at --workers 1, where there is only one "
        "submitter, so sweep it against the worker axis and read "
        "requests_per_forward beside the wall clock.",
    )
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
        "--solver-max-nodes",
        type=int,
        default=0,
        help="endgame-solver node budget to install at every point. Without it "
        "the budget stays 0, `solver_wants` refuses every position, and the "
        "whole solver axis measures a solver that never ran -- the defect "
        "THROUGHPUT_LEVERS.md 3.1 records. Taken from the manifest when "
        "--config-from-manifest is given and this is left at 0.",
    )
    parser.add_argument(
        "--solver-max-secs",
        type=float,
        default=0.0,
        help="deadline per solve; derived from the node budget when 0.",
    )
    parser.add_argument(
        "--solver-attempt-nodes",
        type=int,
        default=0,
        help="the budget the cost model is compared against when deciding to "
        "attempt a solve. 0 uses the timeout, which is what one shared number "
        "always meant. Taken from the manifest when --config-from-manifest is "
        "given and this is left at 0: a run that narrowed its bar attempts far "
        "fewer positions than its timeout implies, and sweeping at the timeout "
        "would measure a solver load no run carries.",
    )
    parser.add_argument(
        "--solver-threads-total",
        default="0",
        help="TOTAL solver threads, divided across shards at each point. Takes "
        "a comma-separated list, and is then a swept AXIS like the others. "
        "Use it whenever --workers has more than one value: --solver-threads "
        "is PER SHARD, so holding it fixed across the worker axis silently "
        "varies the solver load with the axis being measured, and fewer shards "
        "would lose partly because they were under-solving. Overrides "
        "--solver-threads.\n\n"
        "Swept rather than fixed because the generation/solver split is a "
        "CONTENDED one: the solver runs synchronously inside a shard, so a "
        "thread given to it is a thread taken from leaf production, and the "
        "best split is a property of the box's core count rather than of the "
        "algorithm. A single value measures one split and reports it as the "
        "answer. `0` in the list measures with the solver off, which is the "
        "honest baseline for what solving costs.",
    )
    args = parser.parse_args(argv)
    if args.sims_divisor < 1:
        raise SystemExit("--sims-divisor must be 1 or more")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    numbers = lambda text: [int(part) for part in text.split(",") if part.strip()]
    # The solver total is the fourth free axis. It is a TOTAL rather than a
    # per-shard count so that a point's solver load does not move with the
    # worker axis beside it -- see `--solver-threads-total`.
    solver_totals = numbers(args.solver_threads_total) or [0]
    waits = [float(part) for part in args.inference_wait_ms.split(",") if part.strip()]
    waits = waits or [0.0]
    grid = list(
        itertools.product(
            numbers(args.slots),
            numbers(args.caps),
            numbers(args.inflight),
            numbers(args.workers),
            solver_totals,
            waits,
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
            f"{sorted({(slots, workers) for slots, _, _, workers, _, _ in dropped})}",
            flush=True,
        )
    # A positive wait at one shard is a point that cannot differ from wait=0
    # except by adding latency: there is no second submitter to merge with. It
    # would enter the grid, cost a full measurement, and land in the ranking as
    # noise around a duplicate.
    single_shard_waits = [point for point in grid if point[3] == 1 and point[5] > 0.0]
    grid = [point for point in grid if not (point[3] == 1 and point[5] > 0.0)]
    if single_shard_waits:
        print(
            f"skipping {len(single_shard_waits)} single-shard point(s) with a "
            "positive coalescing wait: one submitter has nothing to merge with",
            flush=True,
        )
    if not grid:
        # Naming both filters, because they are both reachable and the message
        # used to name only the first. A staged sweep that pins one shard and a
        # positive wait empties the grid here, and "fewer slots than shards" is
        # then a wrong answer to a real question.
        raise SystemExit(
            "the grid is empty after dropping points with fewer slots than "
            "shards, and single-shard points with a positive coalescing wait. "
            "Raise --slots above --workers, or drop the positive waits when "
            "sweeping one shard."
        )

    # A point cannot hold more games live than it is GIVEN. The scheduler
    # activates a queued game whenever one finishes, so at `--games 200` a
    # 512-slot point never has more than 200 games live: it measures 200 slots
    # wearing a 512 label, and the slot axis is flat above the game count for
    # reasons that have nothing to do with the hardware.
    #
    # Above that floor there is still a ramp and a drain at every point, and
    # they are a larger fraction of a short run. Three games per slot is the
    # threshold `THROUGHPUT_LEVERS.md` uses to call a measurement steady state.
    max_slots = max(point[0] for point in grid)
    if args.games < max_slots:
        raise SystemExit(
            f"--games {args.games} is below the largest slot count in the grid "
            f"({max_slots}), so that point can never fill its slots and the "
            "slot axis would be measured at an occupancy no run has. Raise "
            f"--games to at least {3 * max_slots}, or drop the slot values "
            "above the game count."
        )
    if args.games < 3 * max_slots:
        print(
            f"WARNING: --games {args.games} against {max_slots} slots is "
            f"{args.games / max_slots:.1f} games per slot. Ramp and drain are a "
            "large share of a run that short, and they favour SMALL slot "
            f"counts. {3 * max_slots} or more makes the comparison steady "
            "state.",
            flush=True,
        )

    def solver_threads_for(workers: int, total: int) -> int:
        """Per-shard threads at this point, holding the TOTAL fixed.

        A total of 0 means the solver is off at this point, and that must stay
        0 rather than being widened to 1: "off" is a grid point with its own
        cost curve, and rounding it up would delete the baseline the other
        points are measured against.
        """

        if total <= 0:
            return args.solver_threads if solver_totals == [0] else 0
        return max(1, total // max(1, workers))

    if len(solver_totals) > 1:
        print(
            "solver split is an AXIS: totals "
            + ",".join(str(total) for total in solver_totals)
            + " over workers "
            + ",".join(str(workers) for workers in numbers(args.workers)),
            flush=True,
        )
    elif solver_totals != [0] and len(numbers(args.workers)) > 1:
        print(
            "solver: holding "
            f"{solver_totals[0]} threads TOTAL across the worker axis "
            + ", ".join(
                f"{workers}x{solver_threads_for(workers, solver_totals[0])}"
                for workers in numbers(args.workers)
            ),
            flush=True,
        )

    # The budget the RUN uses, unless overridden. A sweep that installs a
    # different budget from the run is measuring a different workload: the
    # budget is the solver's ADMISSION THRESHOLD, not a ceiling
    # (`THROUGHPUT_LEVERS.md` class D).
    solver_max_nodes = args.solver_max_nodes
    solver_attempt_nodes = args.solver_attempt_nodes
    if args.config_from_manifest:
        import json as _json

        manifest = _json.loads(
            pathlib.Path(args.config_from_manifest).read_text(encoding="utf-8")
        )
        # EACH fallback resolves on its OWN absence. Nesting the bar inside the
        # timeout's `if` meant a caller that passed an explicit timeout -- which
        # the launcher does -- never reached the bar, so the split silently
        # vanished for the sweep and admission widened to the timeout.
        if solver_max_nodes <= 0:
            found = _find_manifest_value(manifest, "endgame_solver_max_nodes")
            if found:
                solver_max_nodes = int(found)
                print(f"solver budget from manifest: {solver_max_nodes:,} nodes", flush=True)
        # The ATTEMPT BAR, which is a different number from the timeout on any
        # run that split them. Left at 0 it falls back to the timeout, which is
        # what one shared number always meant -- but on a run that narrowed the
        # bar, defaulting to the timeout admits a much larger set of positions
        # and measures a solver load the run does not carry.
        if solver_attempt_nodes <= 0:
            bar = _find_manifest_value(manifest, "endgame_solver_attempt_nodes")
            if bar:
                solver_attempt_nodes = int(bar)
            print(
                f"solver attempt bar from manifest: {solver_attempt_nodes:,} "
                f"nodes ({solver_max_nodes / max(1, solver_attempt_nodes):.1f}x "
                "below the timeout)",
                flush=True,
            )
    # The SAME factor as the sims, so the solver keeps its share of slot
    # occupancy instead of growing into the space cheaper generation vacates.
    # `run_point` derives the deadline from the node budget when none is given,
    # so that follows automatically; an explicit one has to be divided here or
    # the sweep would turn node declines into deadline declines, which is the
    # one kind of decline the launcher's stage 6b exists to avoid.
    if args.sims_divisor > 1 and solver_max_nodes > 0:
        divided = max(1, round(solver_max_nodes / args.sims_divisor))
        print(
            f"solver budget divided {args.sims_divisor}x: "
            f"{solver_max_nodes:,} -> {divided:,} nodes",
            flush=True,
        )
        solver_max_nodes = divided
        if args.solver_max_secs > 0:
            args.solver_max_secs = args.solver_max_secs / args.sims_divisor
        # The bar divides too, so the two keep their RATIO. Dividing the timeout
        # alone would leave the same positions admitted against a quarter of the
        # budget, turning proofs into declines and measuring a solver that fails
        # far more often than the run's does.
        if solver_attempt_nodes > 0:
            solver_attempt_nodes = max(
                1, round(solver_attempt_nodes / args.sims_divisor)
            )
    if solver_totals != [0] and solver_max_nodes <= 0:
        raise SystemExit(
            "a solver split was requested but no node budget is available, so "
            "every point would run with solving DISABLED and the axis would "
            "measure nothing. Pass --solver-max-nodes, or --config-from-manifest "
            "for a run that records one."
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
    # THE COST MODEL, installed in this process.
    #
    # `config_from_manifest` copies `endgame_cost_model` into the config, and
    # that does nothing: the trigger reads a RUST GLOBAL, and the `--emit-config`
    # subprocess that produced the manifest cannot reach this process's. With no
    # model installed `solver_wants` falls back to `cards_left <= max_cards`, and
    # `run_point` passes `max_cards = 0` -- a test no Age III mid-play position
    # can pass. Every point then attempted ZERO solves while reporting the thread
    # count it had configured, which is the defect THROUGHPUT_LEVERS 3.1 records,
    # in the one harness written to avoid it.
    if run_config is not None and getattr(run_config, "endgame_cost_model", None):
        stored = run_config.endgame_cost_model
        if isinstance(stored, dict) and "coefficients" in stored:
            import seven_wonders_rust as _swr

            names = list(_swr.endgame_cost_model_features())
            _swr.set_endgame_cost_model(
                names,
                float(stored["intercept"]),
                [float(stored["coefficients"][n]) for n in names],
                float(stored["margin_decades"]),
            )
            print(
                f"cost model installed from the manifest: margin "
                f"{stored['margin_decades']} decades over {len(names)} features",
                flush=True,
            )
        else:
            print(
                "WARNING: the manifest's cost model has no coefficients, so the "
                "trigger falls back to the card cap -- which run_point sets to "
                "0, refusing every position. The solver axis will measure "
                "nothing.",
                flush=True,
            )

    config = apply_config_overrides(config, args.config_override)
    # LAST, so an explicit --config-override names the budget the operator
    # meant and the divisor is applied to that rather than to whatever the
    # manifest happened to carry.
    config = apply_sims_divisor(config, args.sims_divisor)

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
            # `solver_threads` is PER SHARD; the grid axis is the TOTAL.
            run_config.solver_threads * run_config.rust_scheduler_workers,
            run_config.rust_inference_wait_ms,
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
            output / "warmup.jsonl", *grid[0][:4],
            solver_threads=solver_threads_for(grid[0][3], grid[0][4]),
            solver_max_nodes=solver_max_nodes,
            solver_max_secs=args.solver_max_secs,
            inference_wait_ms=grid[0][5],
            solver_attempt_nodes=solver_attempt_nodes,
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
        for position, point in enumerate(order):
            slots, cap, inflight, workers, solver_total, wait_ms = point
            stats, fingerprint = run_point(
                loop, model, args.iteration, jobs,
                output
                / (
                    f"r{repetition}_{position:02d}_s{slots}_c{cap}_i{inflight}"
                    f"_w{workers}_t{solver_total}_q{wait_ms:g}.jsonl"
                ),
                slots, cap, inflight, workers,
                solver_threads_for(workers, solver_total),
                solver_max_nodes,
                args.solver_max_secs,
                wait_ms,
                solver_attempt_nodes,
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
                f"solver={stats['solver_threads_total']:<3} "
                f"wait={wait_ms:<4g} "
                f"fwd={stats['mean_forward_rows']:6.1f}x{stats['requests_per_forward']:.2f} "
                f"batch={stats['mean_batch_size']:6.1f} "
                f"wave={stats['mean_wave_width']:4.2f} "
                f"park={stats['parked_slot_fraction']:4.0%} "
                f"{stats['wall_seconds']:7.1f}s  {stats['games_per_hour']:7.0f} games/h  "
                + detail,
                flush=True,
            )

    summary = []
    for point in grid:
        slots, cap, inflight, workers, solver_total, wait_ms = point
        per_shard = solver_threads_for(workers, solver_total)
        matching = [
            row for row in results
            if (
                row["slots"],
                row["global_batch_cap"],
                row["max_inflight_batches"],
                row["scheduler_workers"],
                row["solver_threads_per_shard"],
                row["inference_wait_ms"],
            ) == (slots, cap, inflight, workers, per_shard, wait_ms)
        ]
        summary.append(
            {
                "slots": slots,
                "global_batch_cap": cap,
                "max_inflight_batches": inflight,
                "scheduler_workers": workers,
                "inference_wait_ms": wait_ms,
                "median_requests_per_forward": statistics.median(
                    row["requests_per_forward"] for row in matching
                ),
                "median_forward_rows": statistics.median(
                    row["mean_forward_rows"] for row in matching
                ),
                "coalesce_wait_seconds": sum(
                    row["coalesce_wait_seconds"] for row in matching
                ),
                # See `parked_slot_fraction` in `run_point`. Carried into the
                # summary because `--sims-divisor` is only honest while this
                # stays near the run's own figure, and the summary is the only
                # thing `sweep_launch_env` and a human ever read.
                "median_parked_slot_fraction": statistics.median(
                    row.get("parked_slot_fraction", 0.0) for row in matching
                ),
                "median_model_forwards_per_forward": statistics.median(
                    row["model_forwards_per_forward"] for row in matching
                ),
                "solver_threads_per_shard": per_shard,
                "solver_threads_total": per_shard * workers,
                # LIVENESS. A thread count is what was configured; these are
                # what happened. `THROUGHPUT_LEVERS.md` 3.1 requires a sweep to
                # refuse a result when a subsystem it claims to measure did no
                # work -- and this harness previously configured solver threads
                # while never installing the node budget, so every point ran
                # with solving disabled.
                "solves_attempted": sum(
                    row.get("solves_attempted", 0) for row in matching
                ),
                "solves_answered": sum(
                    row.get("solves_answered", 0) for row in matching
                ),
                "solves_with_prediction": sum(
                    row.get("solves_with_prediction", 0) for row in matching
                ),
                "median_seconds": statistics.median(row["wall_seconds"] for row in matching),
                "median_games_per_hour": statistics.median(
                    row["games_per_hour"] for row in matching
                ),
                "median_batch_size": statistics.median(
                    row["mean_batch_size"] for row in matching
                ),
                "median_wave_width": statistics.median(
                    row["mean_wave_width"] for row in matching
                ),
                "runs": len(matching),
            }
        )
    summary.sort(key=lambda row: row["median_seconds"])
    # Compare against Phase D's current defaults when they are in the grid --
    # the number that matters is "what would changing the config buy" -- and
    # fall back to the first grid point when they are not.
    # EVERY swept axis, or the lookup below is ambiguous.
    #
    # `summary` is sorted fastest-first, so a key that omits an axis matches
    # several rows and `next()` silently returns the FASTEST of them. The
    # baseline then IS the winner, every speedup is measured against the best
    # point, and the axis the key forgot reports 1.00x -- the sweep concludes
    # that changing the thing it just varied bought nothing.
    #
    # This bit the wait axis the moment it was added, and the solver axis was
    # already exposed to it.
    key = lambda row: (
        row["slots"],
        row["global_batch_cap"],
        row["max_inflight_batches"],
        row["scheduler_workers"],
        row.get("solver_threads_total", 0),
        row.get("inference_wait_ms", 0.0),
    )
    # Baseline: what the run is CURRENTLY set to, when the manifest says and the
    # grid contains it. "What would changing this buy me" is the question a
    # sweep is run to answer, and it is only answered against the status quo --
    # comparing against dataclass defaults answers a question nobody asked.
    def grid_key(point):
        slots, cap, inflight, workers, solver_total, wait_ms = point
        return (slots, cap, inflight, workers, solver_total, wait_ms)

    baselines = []
    if run_baseline is not None:
        # The manifest describes the four geometry axes. The run's solver split
        # and wait complete the key; a manifest that predates either reads 0,
        # which is what such a run was doing.
        baselines.append(tuple(run_baseline))
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
        # The configured solver split and wait complete the key. Both default to
        # "off"/0, which is what a config that never set them is running.
        # Both default to "off", which is what a config that set neither runs.
        + (
            int(field_default("solver_threads"))
            * int(field_default("rust_scheduler_workers")),
            float(field_default("rust_inference_wait_ms")),
        )
    )
    baselines.append(grid_key(grid[0]))
    base = None
    for candidate in baselines:
        base = next((row for row in summary if key(row) == tuple(candidate)), None)
        if base is not None:
            break
    if base is None:
        base = summary[0]
    # Names every axis, because a label that omits one describes several points.
    baseline_label = (
        f"slots={base['slots']}/cap={base['global_batch_cap']}"
        f"/inflight={base['max_inflight_batches']}"
        f"/workers={base['scheduler_workers']}"
        f"/solver={base.get('solver_threads_total', 0)}"
        f"/wait={base.get('inference_wait_ms', 0.0):g}"
    )
    for row in summary:
        row["speedup_vs_baseline"] = base["median_seconds"] / row["median_seconds"]
    payload_baseline = baseline_label

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "games": args.games,
            "repetitions": args.repetitions,
            # Provenance, read by `sweep_launch_env` so `measured_env.sh` can
            # say the geometry was chosen at a reduced search budget. A number
            # in a file nobody reads is not provenance.
            "sims_divisor": args.sims_divisor,
            "cheap_sims": [config.cheap_sims_min, config.cheap_sims_max],
            "full_sims": [config.full_sims_min, config.full_sims_max],
            "solver_max_nodes": solver_max_nodes,
            "solver_attempt_nodes": solver_attempt_nodes,
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
            f"wait={row['inference_wait_ms']:<4g} "
            f"fwd={row['median_forward_rows']:5.0f}"
            f"x{row['median_requests_per_forward']:.2f} "
            f"batch={row['median_batch_size']:5.0f} "
            f"wave={row['median_wave_width']:4.2f} "
            f"park={row['median_parked_slot_fraction']:4.0%} "
            f"{row['median_games_per_hour']:7.0f} games/h  "
            f"({row['speedup_vs_baseline']:.2f}x vs {baseline_label})"
        )
    print(
        f"\ndistinct trajectory sets across all points: {len(fingerprints)} "
        "(1 = these axes changed nothing the search saw)"
    )
    return payload


if __name__ == "__main__":
    main()
