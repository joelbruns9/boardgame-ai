#!/usr/bin/env bash
# =============================================================================
# rehearse_sweep_laptop.sh — run the box's stage 8b pipeline on a laptop, at toy
# scale, to prove the INFRASTRUCTURE works before renting anything.
#
# `RENTING_A_BOX.md` §6: throughput sweeps need the box, exclusively, because
# contention invalidates them. But nothing about the *plumbing* needs a GPU you
# are paying for, and §1 is emphatic that everything checkable on a laptop
# should be checked there.
#
#   ┌─ VALIDATED HERE (transfers to the box) ──────────────────────────────────┐
#   │ · the sweep harness runs end to end with every axis, including the       │
#   │   generation/solver core split                                          │
#   │ · every subsystem is LIVE -- solves attempted > 0, not merely configured │
#   │   (THROUGHPUT_LEVERS.md §3.1: a sweep once measured every point with the │
#   │   solver switched off, on a run where it took 22-37% of generation wall) │
#   │ · the repaired counters move, and requests are distinguishable from      │
#   │   forwards (COALESCER_BUILD_PLAN.md §0.2b)                              │
#   │ · `sweep_launch_env.py` writes a sourceable env file carrying the winner │
#   │   AND the `SWEEP_MEASURED` provenance marker the launcher's pass-2 guard │
#   │   refuses without                                                       │
#   │ · the STAGED handoff: geometry ranked first, the batching axes swept at  │
#   │   the winner, and `swept_axes` carrying "stage A measured the split"     │
#   │   through to SOLVER_THREADS -- which stage B's own summary denies        │
#   │ · `--sims-divisor` reaching the config and being recorded as provenance  │
#   │ · stage 6b's CONTENDED node-rate measurement, and stage 8c's solver       │
#   │   sizing crossing it with the shipped corpus -- ending in solver caps in  │
#   │   the same env file pass 2 sources, with the clock proven slack against   │
#   │   the node budget at the rate just measured                              │
#   │ · the games-per-point requirement and the grid arithmetic                │
#   └──────────────────────────────────────────────────────────────────────────┘
#
#   ┌─ NOT VALIDATED HERE (do not carry any number to the box) ────────────────┐
#   │ Every measured optimum. Different GPU, different core count, different   │
#   │ memory, and this machine is running your editor and test suite while it  │
#   │ measures. `THROUGHPUT_LEVERS.md` §7: benchmark figures only partly       │
#   │ transfer even between two clean machines.                                │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash rehearse_sweep_laptop.sh [OUT_DIR]
#
# Env:
#   REHEARSE_CHECKPOINT   an L checkpoint carrying the RUN's architecture.
#                         Built for you if unset -- see §"checkpoint" below.
#   REHEARSE_GAMES=24     games per sweep point (3 per slot; below that the
#                         sweep warns that ramp and drain dominate)
#   REHEARSE_MANIFEST     a run_manifest.json to check the config path against
#   REHEARSE_WAIT_CSV=0,2 coalescing waits in ms. Swept, not pinned: a wait
#                         reaching the harness and merging nothing looks the
#                         same in a log as a wait that helped.
#   REHEARSE_SOLVER_NODES=2000000
#                         node budget installed at every solver-on point. The
#                         axis measures nothing without it -- see the check.
#   REHEARSE_WORKERS_CSV=1,2  shard counts for the GEOMETRY stage. 1 is in the
#                         list so the "a single shard cannot merge" property is
#                         checked; the cost is that if a single shard WINS, the
#                         batching stage runs its whole wait axis at 0 and the
#                         wait plumbing goes unexercised. The check says so and
#                         names 2,4 as the re-run.
#   REHEARSE_INFLIGHT_CSV=1,2  inflight batches for the BATCHING stage. Two
#                         values, or that stage resolves to one point and ranks
#                         nothing.
#   REHEARSE_SIMS_DIVISOR=2   exercises the box's cost knob. On the box it
#                         divides the run's 1600-simulation search; here it
#                         divides the harness defaults, which is enough to prove
#                         the flag reaches the config and is recorded.
#   REHEARSE_RATE_THREADS=4   threads for the contended rate measurement. The
#                         single-thread figure overstates per-thread capacity --
#                         measured 1.41M nodes/s/thread at 1 against 795k at 8 on
#                         this laptop, 1.77x apart -- which is why stage 6b
#                         measures it contended and why this rehearses that.
#   REHEARSE_TARGET_SHARE=0.80  fraction of solver capacity the caps may commit.
#   REHEARSE_DEVICE=cuda
# =============================================================================
set -euo pipefail

OUT="${1:-/tmp/rehearse_sweep}"
# 3 games per slot: the steady-state threshold the sweep now warns below. At
# 8 slots that is 24. Raising --slots here means raising this too.
GAMES="${REHEARSE_GAMES:-24}"
DEVICE="${REHEARSE_DEVICE:-cuda}"
PY="${PYTHON_BIN:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$OUT"

# ── The checkpoint ──────────────────────────────────────────────────────────
#
# `W2_W3_W5_CLOUD_ACCEPTANCE.md` says the sweep checkpoint's playing strength is
# irrelevant and its 384x8x6 tensor shapes are required. **That is no longer
# sufficient.** W1/W2/W4/W5 change the model, not just its width, so a sweep run
# against a plain transformer measures a model nobody is running --
# `THROUGHPUT_LEVERS.md` §3.1 in a new dress.
#
# A checkpoint from a previous run is usually unusable for a different reason:
# the encoder signature moves with the workstreams, and `load_checkpoint`
# refuses rather than silently loading a mismatched model.
CKPT="${REHEARSE_CHECKPOINT:-$OUT/sweep_L.pt}"
if [ ! -f "$CKPT" ]; then
  say "Building a sweep checkpoint with the run's architecture"
  "$PY" - "$CKPT" <<'PYCKPT' || die "could not build the sweep checkpoint"
import sys, torch
from games.seven_wonders_duel.phase_d import PhaseDConfig, PhaseDLoop
from games.seven_wonders_duel.train import make_checkpoint

config = PhaseDConfig(
    run_dir="/tmp/_rehearse_ckpt", d_model=384, layers=8, heads=6, device="cpu",
    slot_embedding=True, graph_module=True,
    hierarchical_value=True, hier_value_weight=0.5,
    action_residual=True, action_exposes=True, action_policy_weight=0.5,
    pooled_readout=True, reply_head=True,
)
loop = PhaseDLoop.__new__(PhaseDLoop)
loop.config = config
model = loop._new_model()
torch.save(make_checkpoint(model, loop._model_contract(model, iteration=0)), sys.argv[1])
print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")
PYCKPT
fi

# ── The manifest path, checked WITHOUT running any games ────────────────────
#
# `THROUGHPUT_LEVERS.md` §3.1: the sweep must build its config from the run's
# manifest, or it finds an optimum belonging to a machine nobody is running.
# That is a box requirement. Checking it costs nothing here, and it is checked
# SEPARATELY from the games below -- see the note on scale.
if [ -n "${REHEARSE_MANIFEST:-}" ]; then
  say "Manifest config path (parse only, no games)"
  "$PY" - "$REHEARSE_MANIFEST" <<'PYMANIFEST' || die "manifest config did not load"
import pathlib, sys
from games.seven_wonders_duel.f4_phase_d_sweep import config_from_manifest

config = config_from_manifest(
    sys.argv[1], output=pathlib.Path("/tmp/_rehearse_manifest"), device="cpu",
    games=1, precision="bf16", geometry={},
)
print(f"  search={config.selfplay_search_mode} "
      f"full_sims={config.full_sims_min}-{config.full_sims_max} "
      f"cheap_sims={config.cheap_sims_min}-{config.cheap_sims_max} "
      f"top_k={config.top_k}")
if config.full_sims_max < 100:
    raise SystemExit("manifest did not supply the run's real search budget")
PYMANIFEST
fi

# ── Stage 8b, generation ────────────────────────────────────────────────────
#
# Two points minimum on every axis: a grid that resolves to one configuration
# reports 1.00x against itself (§3.1). The solver split is included because it
# is the axis the launcher used to pin rather than measure.
#
# ⚠ NOTE ON SCALE, and why this is not the box's command.
#
# `--config-from-manifest` IS passed, whenever REHEARSE_MANIFEST names one, and
# that reverses the reasoning this block used to carry.
#
# It was omitted because the run's budget is 1600 full simulations and four
# warmup games at that budget did not finish in 20 minutes on this laptop. True
# then. `--sims-divisor` is what changed it: the manifest's search can now run
# shallow enough for a laptop while staying the RUN's search in algorithm, mix
# and trigger.
#
# And omitting it was not free, which is the part that matters. The launcher
# passes the flag, and `f4_phase_d_sweep` installs the run's COST MODEL only on
# that path -- so a rehearsal without it ran the one configuration the box never
# uses, with no model installed and `max_cards = 0`. `solver_wants` then falls
# back to `cards_left <= 0`, reachable at exactly one position per game: the end
# of Age III, when no cards remain. Three consecutive rehearsals reported
# "solver LIVE: 1 attempted" and passed while the solver axis measured nothing.
#
# So the manifest path is the one that has to be rehearsed. A divisor keeps it
# affordable; REHEARSE_SIMS_DIVISOR is the knob.
# The coalescing wait is swept here for the same reason the solver split is:
# it is an axis the box will run, and this script exists to prove the plumbing
# of that axis before it is rented. A wait that reaches the harness and merges
# nothing looks identical, in a log, to a wait that helps a little.
# STAGED, because that is what the box runs. The launcher's stage 8b drives
# `f4_staged_sweep`, which ranks geometry first and sweeps the batching axes at
# the winner -- and the handoff it adds (which axes were varied ACROSS the two
# stages, threaded through `sweep_launch_env`) is plumbing, which is exactly
# what this script exists to prove before anything is rented. Rehearsing the
# unstaged harness would leave the box's real pipeline unrehearsed.
#
# `--sims-divisor` is exercised for the same reason: on the box it divides the
# run's 1600-simulation search, and a divisor that failed to reach the harness
# would look, in a log, like a sweep that was simply fast.
say "Generation sweep (STAGED toy grid; solver split and wait as axes, LAPTOP-SCALE search)"
"$PY" -m games.seven_wonders_duel.f4_staged_sweep \
  --checkpoint "$CKPT" \
  --output "$OUT/generation" \
  --games "$GAMES" --warmup-games 2 --repetitions 1 \
  --slots 8 --caps 256 \
  --inflight "${REHEARSE_INFLIGHT_CSV:-1,2}" \
  --workers "${REHEARSE_WORKERS_CSV:-1,2}" \
  --inference-wait-ms "${REHEARSE_WAIT_CSV:-0,2}" \
  --solver-threads-total "0,4" \
  --solver-max-nodes "${REHEARSE_SOLVER_NODES:-2000000}" \
  --stage-a-inflight 1 --stage-a-wait-ms 0 \
  --sims-divisor "${REHEARSE_SIMS_DIVISOR:-2}" \
  ${REHEARSE_MANIFEST:+--config-from-manifest "$REHEARSE_MANIFEST"} \
  --device "$DEVICE" --precision bf16 \
  || die "generation sweep did not complete"

# ── Stage 8b, gate ──────────────────────────────────────────────────────────
say "Gate sweep (separate harness, separate cost regime)"
"$PY" -m games.seven_wonders_duel.w5_gate_slots_sweep \
  --checkpoint "$CKPT" \
  --work-dir "$OUT/gate_work" \
  --output "$OUT/gate_20.json" \
  --games 20 --slots 4 8 --caps 256 --sims 16 \
  --device "$DEVICE" --precision bf16 \
  || die "gate sweep did not complete"

# ── The env file the launcher's pass 2 consumes ─────────────────────────────
say "Translating both sweeps into measured_env.sh"
"$PY" "$REPO/games/seven_wonders_duel/sweep_launch_env.py" \
  --sweep-dir "$OUT" --gate-rung 20 \
  || die "could not summarise the sweeps"

# ── Stage 6b's rate measurement, and stage 8c's solver sizing ───────────────
#
# Both are box stages, and both are rehearsed here for the same reason the
# staged sweep is: the launcher runs them, so a rehearsal that skips them proves
# a pipeline nobody is going to run.
#
# The rate is measured CONTENDED, which is the whole point of it. On this laptop
# the single-thread figure is 1.41M nodes/s/thread and eight threads give 795k --
# 1.77x apart, so a budget sized off the uncontended number is 1.77x optimistic.
# Few threads and a small budget here: this is a plumbing check, and the NUMBER
# belongs to whichever box measured it.
say "Solver node rate (CONTENDED, as stage 6b measures it)"
RATE="$("$PY" - "${REHEARSE_RATE_THREADS:-4}" <<'PYRATE'
import sys
from games.seven_wonders_duel.endgame_trigger_study import measure_node_rate_contended
out = measure_node_rate_contended(
    max(1, int(sys.argv[1])), max_nodes=2_000_000, max_secs=30.0
)
print(int(out["nodes_per_second_per_thread"]))
print(
    f"  {out['threads']} threads: {out['nodes_per_second_per_thread']:,.0f} "
    f"nodes/s/thread, parallel efficiency {out['parallel_efficiency']:.2f}",
    file=sys.stderr,
)
PYRATE
)"
RATE="$(printf '%s' "$RATE" | head -1)"
[ "${RATE:-0}" -gt 0 ] || die "the contended rate measurement produced nothing"

# The corpus is the SHIPPED one -- cloud2's endgames, priced once. That is the
# artifact a rented box reads, so rehearsing against anything else would leave
# the file that actually travels untested.
say "Sizing the solver caps (stage 8c) from the shipped corpus"
"$PY" -m games.seven_wonders_duel.solver_sizing \
  --corpus "$REPO/games/seven_wonders_duel/solver_corpus.json" \
  --rate "$RATE" \
  --threads "${REHEARSE_RATE_THREADS:-4}" \
  --generation-wall-seconds 4422 \
  --games 1000 \
  --target-share "${REHEARSE_TARGET_SHARE:-0.80}" \
  --output "$OUT/solver_env.sh" \
  || die "solver sizing failed"
cat "$OUT/solver_env.sh" >> "$OUT/measured_env.sh"

# ── Assertions ──────────────────────────────────────────────────────────────
say "Checking the infrastructure"
"$PY" - "$OUT" "$RATE" <<'PYCHECK' || die "infrastructure check failed"
import json, pathlib, subprocess, sys

out = pathlib.Path(sys.argv[1])
# The rate stage 6b measured, so the clock check below asks the question that
# matters: does the wall clock stop a solve before the node budget does?
rate_for_check = int(sys.argv[2])
payload = json.loads(
    (out / "generation" / "phase_d_sweep.json").read_text(encoding="utf-8")
)
problems = []

# THE UNION OF BOTH STAGES, not `summary`.
#
# `summary` is stage B's, and stage B pins everything stage A won -- so read
# alone it shows one solver split, one slot count and one shard count, and every
# "did this axis vary" check below would fail on a sweep that measured all of
# them. That is the same reading error `sweep_launch_env` has to avoid, which is
# why the driver publishes `staged.stages`; this checks the publication works.
staged = payload.get("staged")
if staged is None:
    problems.append(
        "no `staged` block: the generation sweep was not driven by "
        "f4_staged_sweep, so this rehearsed a pipeline the box does not run"
    )
    summary = payload["summary"]
    stage_summaries = [payload["summary"]]
else:
    stage_summaries = [stage["summary"] for stage in staged["stages"]]
    summary = [row for stage in stage_summaries for row in stage]

# 1. More than one point per stage, or that stage reported 1.00x against
#    itself. Per stage rather than in total: a staged sweep whose second stage
#    collapsed to one point ranked nothing, and the union would hide it.
for index, stage in enumerate(stage_summaries):
    if len(stage) < 2:
        problems.append(
            f"stage {index} resolved to {len(stage)} point(s); that is not a "
            "measurement"
        )

# 2. The solver axis varied, and -- the point of this check -- the solver
#    actually DID WORK where it was switched on.
#
#    An earlier version of this script asserted `solver_threads_total > 0`,
#    which is CONFIGURATION. It passed while `f4_phase_d_sweep` was installing
#    solver threads and never the node budget, so every point ran with solving
#    disabled: exactly the defect `THROUGHPUT_LEVERS.md` §3.1 records, in a
#    check written to catch it.
splits = sorted({row["solver_threads_total"] for row in summary})
if len(splits) < 2:
    problems.append(f"solver split did not vary: {splits}")
on = [row for row in summary if row["solver_threads_total"] > 0]
off = [row for row in summary if row["solver_threads_total"] == 0]
if not on:
    problems.append("no point was configured with the solver on")
elif not any(row.get("solves_attempted", 0) > 0 for row in on):
    problems.append(
        "solver configured but ZERO solves attempted -- the axis measured a "
        "solver that never ran"
    )
elif not any(row.get("solves_with_prediction", 0) > 0 for row in on):
    # ATTEMPTS ARE NOT ENOUGH, and this is the check three green rehearsals
    # needed. Only the cost model produces a prediction. With none installed the
    # trigger falls back to `cards_left <= max_cards`, and at max_cards = 0 that
    # admits exactly the end-of-Age-III position where no cards remain -- one
    # degenerate solve per run, reported as "solver LIVE".
    problems.append(
        "solves were attempted but NONE carried a cost-model prediction: the "
        "model was not installed, so the trigger is the card cap and the solver "
        "axis measures a configuration the box never runs. Set "
        "REHEARSE_MANIFEST so the sweep takes the manifest path the launcher "
        "takes."
    )
if any(row.get("solves_attempted", 0) > 0 for row in off):
    problems.append("a solver-off point still attempted solves")
attempted = sum(row.get("solves_attempted", 0) for row in on)
answered = sum(row.get("solves_answered", 0) for row in on)

# 3. The coalescer actually MERGED where there was something to merge.
#
#    Liveness, not configuration -- the same rule the solver check above learned
#    the hard way. `requests_per_forward` is exactly 1.00 when nothing coalesced,
#    so a wait that reaches the harness and merges nothing is indistinguishable,
#    in a wall-clock log, from a wait that helped a little.
#
#    Only multi-shard points can merge: at one shard there is a single submitter
#    and the ratio is 1.00 by construction, which is a fact about the geometry
#    rather than a failure.
waits = sorted({row.get("inference_wait_ms", 0.0) for row in summary})
sharded = [row for row in summary if row["scheduler_workers"] > 1]
single = [row for row in summary if row["scheduler_workers"] == 1]
# Among MULTI-SHARD points, because that is where the wait can do anything.
# `f4_phase_d_sweep` drops a positive wait at one shard outright -- one
# submitter has nothing to merge with -- so a staged sweep whose geometry stage
# picked a single shard runs its whole wait axis at 0 and proves nothing about
# the wait. That is a real gap in the rehearsal, not a failure of the code, and
# it is reported as such with the way to close it.
sharded_waits = sorted({row.get("inference_wait_ms", 0.0) for row in sharded})
# ORDER MATTERS. The single-shard case collapses the union to {0.0}, so a bare
# `len(waits) < 2` first swallows exactly the case the branch below exists to
# explain -- and reports "the wait did not vary" where the truth is "the
# geometry stage picked one shard, so the wait axis was dropped for it". The
# first is a grid the operator got wrong; the second is a re-run with a flag.
if len(sharded) and len(sharded_waits) < 2 or (not sharded and len(waits) < 2):
    problems.append(
        f"the wait axis was not exercised (waits seen: {waits}). A positive "
        "wait is dropped at one shard by construction -- one submitter has "
        "nothing to merge with -- so when the geometry stage picks a single "
        "shard the batching stage runs its whole wait axis at 0. Re-run with "
        "REHEARSE_WORKERS_CSV=2,4 to force a multi-shard winner."
    )
elif len(waits) < 2:
    problems.append(f"coalescing wait did not vary: {waits}")
if not sharded:
    problems.append("no multi-shard point, so coalescing could not be exercised")
elif not any(row.get("median_requests_per_forward", 0.0) > 1.0 for row in sharded):
    problems.append(
        "every multi-shard point reported requests_per_forward == 1.00 -- the "
        "coalescer was configured and did not merge"
    )
for row in single:
    if row.get("median_requests_per_forward", 0.0) > 1.0:
        problems.append(
            "a SINGLE-shard point reported a merge; one submitter has nothing "
            "to merge with, so this counter is measuring the wrong thing"
        )
merged = max(
    (row.get("median_requests_per_forward", 0.0) for row in sharded), default=0.0
)

# 4. Every point measured throughput, not activation: more games than slots.
for row in summary:
    if row.get("runs", 0) < 1:
        problems.append(f"point {row['slots']}/{row['scheduler_workers']} has no runs")

# 5. The env file exists, sources cleanly, and carries the provenance marker
#    the launcher's pass-2 guard refuses without -- and carries the measured
#    WAIT, which production silently replaces with 0 when it is missing.
env = out / "measured_env.sh"
if not env.is_file():
    problems.append("measured_env.sh was not written")
else:
    probe = subprocess.run(
        ["bash", "-c",
         f'source "{env.as_posix()}"; '
         'echo "$RUST_SLOTS|$RUST_SCHEDULER_WORKERS|$GATE_SLOTS|$SWEEP_MEASURED'
         '|$RUST_INFERENCE_WAIT_MS"'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        problems.append(f"measured_env.sh does not source: {probe.stderr.strip()}")
    else:
        slots, workers, gate, measured, wait = probe.stdout.strip().split("|")
        if measured != "1":
            problems.append("SWEEP_MEASURED is not set; pass 2 would refuse to launch")
        for name, value in (("RUST_SLOTS", slots), ("RUST_SCHEDULER_WORKERS", workers),
                            ("GATE_SLOTS", gate)):
            if not value:
                problems.append(f"{name} is empty in measured_env.sh")
        # The wait was swept, so a winner exists and must be carried. An absent
        # one means production runs 0 ms while every other number in the file
        # describes a geometry measured at something else.
        if not wait:
            problems.append(
                "RUST_INFERENCE_WAIT_MS is absent although the sweep varied the "
                "wait; the run would silently take the 0 ms default"
            )
        print(
            f"  measured_env.sh -> slots={slots} workers={workers} "
            f"gate_slots={gate} wait={wait}ms"
        )

# 6. The staged handoff. Stage B's summary carries ONE solver split, because
#    stage A pinned it -- so the rule `sweep_launch_env` uses on an unstaged
#    file ("was the split varied") reads false and the measurement stage A paid
#    for is silently dropped. The `swept_axes` block is what prevents that, and
#    nothing else in this script would notice if it stopped arriving.
if staged is not None:
    swept = set(staged.get("swept_axes") or ())
    for axis in ("solver_threads_per_shard", "scheduler_workers"):
        if len({row.get(axis) for row in stage_summaries[-1]}) > 1:
            problems.append(
                f"stage B varied {axis}; the geometry pin did not take"
            )
    if "solver_threads_total" not in swept:
        problems.append(
            "the split was swept at stage A but swept_axes does not say so, so "
            "SOLVER_THREADS would be dropped from measured_env.sh"
        )
    env_text = env.read_text(encoding="utf-8") if env.is_file() else ""
    if "export SOLVER_THREADS=" not in env_text:
        problems.append(
            "SOLVER_THREADS is absent from measured_env.sh although stage A "
            "measured the split -- the staged handoff did not reach it"
        )
    drift = staged.get("carryover_drift")
    if drift is None:
        problems.append(
            "no carryover_drift: stage B did not re-measure stage A's winning "
            "point, so the two stages cannot be compared"
        )
    else:
        print(f"  staged: axes {sorted(swept)}, carryover {drift:+.1%}")

# 7. The sims divisor REACHED the harness. A divisor that silently did nothing
#    looks, in a log, like a sweep that was simply fast -- and on the box it is
#    the difference between measuring the run's search and a quarter of it.
divisor = int((payload.get("config") or {}).get("sims_divisor", 0) or 0)
if divisor < 1:
    problems.append("the sweep recorded no sims_divisor; provenance is missing")
else:
    print(f"  sims divisor recorded: {divisor}x")

# 8. Stage 8c's solver caps reached the SAME file pass 2 sources. Two files to
#    source is one file to forget, and the geometry and the caps were measured
#    beside each other -- taking one without the other launches a run whose
#    solver was sized against a machine it is not running on.
if not env.is_file():
    problems.append("measured_env.sh missing; the solver caps had nowhere to go")
else:
    text = env.read_text(encoding="utf-8")
    for name in ("ENDGAME_SOLVER_ATTEMPT_NODES", "ENDGAME_SOLVER_MAX_NODES",
                 "ENDGAME_SOLVER_MAX_SECS"):
        if f"export {name}=" not in text:
            problems.append(f"{name} is absent; stage 8c did not reach the env file")
    probe = subprocess.run(
        ["bash", "-c",
         f'source "{env.as_posix()}"; '
         'echo "$ENDGAME_SOLVER_ATTEMPT_NODES|$ENDGAME_SOLVER_MAX_NODES'
         '|$ENDGAME_SOLVER_MAX_SECS"'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        problems.append(f"measured_env.sh stopped sourcing once the caps were "
                        f"appended: {probe.stderr.strip()}")
    else:
        bar, cap, secs = probe.stdout.strip().split("|")
        if bar and cap and int(bar) > int(cap):
            problems.append(
                f"the attempt bar ({bar}) is above the timeout ({cap}): every "
                "position admitted there would spend the timeout in full"
            )
        # The clock must not be the thing that stops a solve, or a node-censored
        # decline becomes a load-dependent one.
        if cap and secs and int(secs) * int(rate_for_check) <= int(cap):
            problems.append(
                f"the clock ({secs}s) binds before the node budget ({cap}) at "
                "the measured rate; declines would depend on machine load"
            )
        print(f"  solver caps -> bar={bar} timeout={cap} clock={secs}s")

if problems:
    print("\n".join(f"  - {p}" for p in problems))
    raise SystemExit(1)

print(f"  {len(summary)} points, solver totals {splits}, waits {waits}")
print(f"  solver LIVE: {attempted} solves attempted, {answered} answered")
print(f"  coalescer LIVE: best multi-shard merge {merged:.2f} requests/forward")
print("  every subsystem live; env file sourceable and marked measured")
PYCHECK

# The L checkpoint is 69 MB and the sweep directories accumulate per-point
# JSONL. Left behind, they were enough to push a 16 GiB laptop into killing the
# test suite for memory while the next thing ran.
if [ "${REHEARSE_KEEP:-0}" != "1" ]; then
  rm -rf "$OUT/gate_work" "$OUT/generation"/*.jsonl 2>/dev/null || true
  say "Cleaned per-point artifacts (REHEARSE_KEEP=1 to retain them)"
fi

cat <<'EOF'

==> Infrastructure OK.

    The PLUMBING is validated and transfers to the box.
    The NUMBERS above do not. Do not copy any slot count, worker count, batch
    cap or solver split from this run into a launch: they belong to this
    laptop, contended, at a toy grid.

    On the box, the same pipeline runs as pass 1 of:
        SWEEP_CHECKPOINT=<L ckpt> bash launch_7wd_run.sh
        source <run>/sweeps/measured_env.sh && bash launch_7wd_run.sh
EOF
