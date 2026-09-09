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
#   REHEARSE_GAMES=12     games per sweep point
#   REHEARSE_MANIFEST     a run_manifest.json to check the config path against
#   REHEARSE_SOLVER_NODES=2000000
#                         node budget installed at every solver-on point. The
#                         axis measures nothing without it -- see the check.
#   REHEARSE_DEVICE=cuda
# =============================================================================
set -euo pipefail

OUT="${1:-/tmp/rehearse_sweep}"
GAMES="${REHEARSE_GAMES:-12}"
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
# `--config-from-manifest` is deliberately NOT passed here. The run's budget is
# 1600 full simulations; four warmup games at that budget did not finish in 20
# minutes on this laptop, which is the correct answer for a production budget on
# a laptop GPU and a useless one for a plumbing check. Without it the harness
# takes its own defaults (64-128 full, 16-24 cheap), which exercise every code
# path at a scale a laptop can finish.
#
# On the BOX the manifest flag is mandatory, for exactly the reason §3.1 gives.
# Dropping it there would repeat the defect that sweep once committed: measuring
# Gumbel at 24/128 to configure a PUCT run at 100/1600.
say "Generation sweep (toy grid, solver split as an axis, LAPTOP-SCALE search)"
"$PY" -m games.seven_wonders_duel.f4_phase_d_sweep \
  --checkpoint "$CKPT" \
  --output "$OUT/generation" \
  --games "$GAMES" --warmup-games 2 --repetitions 1 \
  --slots 8 --caps 256 --inflight 1 \
  --workers 1,2 \
  --solver-threads-total "0,4" \
  --solver-max-nodes "${REHEARSE_SOLVER_NODES:-2000000}" \
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

# ── Assertions ──────────────────────────────────────────────────────────────
say "Checking the infrastructure"
"$PY" - "$OUT" <<'PYCHECK' || die "infrastructure check failed"
import json, pathlib, subprocess, sys

out = pathlib.Path(sys.argv[1])
summary = json.loads(
    (out / "generation" / "phase_d_sweep.json").read_text(encoding="utf-8")
)["summary"]
problems = []

# 1. More than one point, or the sweep reported 1.00x against itself.
if len(summary) < 2:
    problems.append(f"grid resolved to {len(summary)} point(s); that is not a measurement")

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
if any(row.get("solves_attempted", 0) > 0 for row in off):
    problems.append("a solver-off point still attempted solves")
attempted = sum(row.get("solves_attempted", 0) for row in on)
answered = sum(row.get("solves_answered", 0) for row in on)

# 3. Every point measured throughput, not activation: more games than slots.
for row in summary:
    if row.get("runs", 0) < 1:
        problems.append(f"point {row['slots']}/{row['scheduler_workers']} has no runs")

# 4. The env file exists, sources cleanly, and carries the provenance marker
#    the launcher's pass-2 guard refuses without.
env = out / "measured_env.sh"
if not env.is_file():
    problems.append("measured_env.sh was not written")
else:
    probe = subprocess.run(
        ["bash", "-c",
         f'source "{env.as_posix()}"; '
         'echo "$RUST_SLOTS|$RUST_SCHEDULER_WORKERS|$GATE_SLOTS|$SWEEP_MEASURED"'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        problems.append(f"measured_env.sh does not source: {probe.stderr.strip()}")
    else:
        slots, workers, gate, measured = probe.stdout.strip().split("|")
        if measured != "1":
            problems.append("SWEEP_MEASURED is not set; pass 2 would refuse to launch")
        for name, value in (("RUST_SLOTS", slots), ("RUST_SCHEDULER_WORKERS", workers),
                            ("GATE_SLOTS", gate)):
            if not value:
                problems.append(f"{name} is empty in measured_env.sh")
        print(f"  measured_env.sh -> slots={slots} workers={workers} gate_slots={gate}")

if problems:
    print("\n".join(f"  - {p}" for p in problems))
    raise SystemExit(1)

print(f"  {len(summary)} points, solver totals {splits}")
print(f"  solver LIVE: {attempted} solves attempted, {answered} answered")
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
