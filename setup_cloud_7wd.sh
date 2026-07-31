#!/usr/bin/env bash
# =============================================================================
# setup_cloud_7wd.sh — first-login setup + launch for 7 Wonders Duel training.
#
# Brings a fresh Linux/CUDA box to "training launched", in one command:
#   1. Rust toolchain (rustup >= 1.85)
#   2. clone (or update) the repo at ~/boardgame-ai
#   3. Python deps — cu128 torch FIRST, then requirements.txt
#   4. build the seven_wonders_rust crate with maturin
#   5. HARD-FAIL GPU verification gate
#   6. W6.4 preflight: host memory at the run's MAXIMUM scheduled window, and a
#      hard VRAM floor for the shipped model
#   7. W6.2 engine-equivalence smoke — the Rust/Python parity suite, which must
#      not skip. Every number in CLOUD_TRAINING_PLAN.md assumes the Rust engine
#      and Python agree; this is the only thing that checks it on this box.
#   8. W6.2b precision arena (optional): bf16 vs fp32 from one checkpoint
#   9. Phase D plumbing smoke on CUDA
#  10. launch training detached with nohup so it survives SSH disconnect
#
# The Rust stages are the reason this file was rewritten: it previously ran pure
# Python, while every measurement in the plan assumes --generation-backend rust
# --gate-backend rust.
#
# Idempotent: re-running updates the repo and RESUMES the run. Phase D refuses a
# resume whose commit, precision, or schedules differ from the ones the run
# started on (W6.5), so an update that lands mid-run stops rather than silently
# splitting the run across two engines.
#
# Usage (fresh box):
#   curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud_7wd.sh -o setup_cloud_7wd.sh
#   bash setup_cloud_7wd.sh
# or from an existing clone:
#   bash ~/boardgame-ai/setup_cloud_7wd.sh
#
# Knobs (env vars):
#   ITERATIONS=60 GAMES_PER_ITERATION=500 SEED_GAMES=5000 WORKERS=8
#   D_MODEL=384 LAYERS=8 HEADS=6 PRECISION=bf16 LEARNING_RATE=5e-5
#   HOF_FRACTION=0.15 GATE_LADDER="100 200 400 800"
#   LAUNCH_FLAGS_JSON=<f4_cloud_finalize output>  measured --rust-* flags (W6.3)
#   PRECISION_ARENA_CHECKPOINT=<path>             runs W6.2b before launching
#   SELF_ANCHOR_GAMES=200 SELF_ANCHOR_LAG_GAMES=20000   W7a stagnation anchor
#   INTERVENTION_LADDER=0                                W7b response (off)
#   MEMORY_BUDGET_GB / VRAM_BUDGET_GB / MEMORY_HEADROOM_GB
#   RUN_DIR_REL=runs/seven_wonders_duel/cloud
#   LAUNCH=1        set 0 to stop after verification
#   SKIP_SMOKE=0    set 1 to skip the Phase D plumbing smoke
#   SKIP_EQUIV=0    set 1 to skip the equivalence suite (do not do this)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/boardgame-ai}"

# The common library lives beside this script in a checkout, and beside the
# repo once cloned; a bare curl of this file alone fetches it.
if [ -f "$SCRIPT_DIR/setup_cloud_common.sh" ]; then
  # shellcheck source=setup_cloud_common.sh
  source "$SCRIPT_DIR/setup_cloud_common.sh"
elif [ -f "$REPO_DIR/setup_cloud_common.sh" ]; then
  # shellcheck source=setup_cloud_common.sh
  source "$REPO_DIR/setup_cloud_common.sh"
else
  curl -fsSL "https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud_common.sh" \
    -o /tmp/setup_cloud_common.sh || {
      echo "[FATAL] could not fetch setup_cloud_common.sh" >&2; exit 1; }
  # shellcheck source=/dev/null
  source /tmp/setup_cloud_common.sh
fi

RUN_DIR_REL="${RUN_DIR_REL:-runs/seven_wonders_duel/cloud}"
ITERATIONS="${ITERATIONS:-60}"
GAMES_PER_ITERATION="${GAMES_PER_ITERATION:-500}"
SEED_GAMES="${SEED_GAMES:-5000}"
WORKERS="${WORKERS:-8}"
PROCESS_WORKERS="${PROCESS_WORKERS:-$(nproc)}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
PRECISION="${PRECISION:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
HOF_FRACTION="${HOF_FRACTION:-0.15}"
HOF_START_GAMES="${HOF_START_GAMES:-10000}"
GATE_LADDER="${GATE_LADDER:-100 200 400 800}"
GATE_LADDER_FLOOR_GAMES="${GATE_LADDER_FLOOR_GAMES:-10000}"
ANCHOR_GAMES="${ANCHOR_GAMES:-200}"
SELF_ANCHOR_GAMES="${SELF_ANCHOR_GAMES:-200}"
SELF_ANCHOR_LAG_GAMES="${SELF_ANCHOR_LAG_GAMES:-20000}"
SELF_ANCHOR_EVERY_GAMES="${SELF_ANCHOR_EVERY_GAMES:-10000}"
INTERVENTION_LADDER="${INTERVENTION_LADDER:-0}"
INTERVENTION_WINDOW_GAMES="${INTERVENTION_WINDOW_GAMES:-20000}"
REPLAY_WINDOW_CAP_GAMES="${REPLAY_WINDOW_CAP_GAMES:-20000}"
EXAMPLE_CACHE_GB="${EXAMPLE_CACHE_GB:-0}"
MEMORY_BUDGET_GB="${MEMORY_BUDGET_GB:-0}"
VRAM_BUDGET_GB="${VRAM_BUDGET_GB:-0}"
MEMORY_HEADROOM_GB="${MEMORY_HEADROOM_GB:-2}"
LAUNCH="${LAUNCH:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_EQUIV="${SKIP_EQUIV:-0}"
CRATE_DIR_REL="games/seven_wonders_duel/seven_wonders_rust"

common::require_python

stage 1 "Rust toolchain (rustup)"
common::rust_toolchain
stage_done 1

stage 2 "Clone repo into $REPO_DIR"
common::clone_repo "games/seven_wonders_duel/phase_d.py" "$CRATE_DIR_REL"
stage_done 2

stage 3 "Python dependencies (cu128 torch first)"
common::python_deps
stage_done 3

stage 4 "Build seven_wonders_rust"
common::build_crate "$CRATE_DIR_REL" seven_wonders_rust
stage_done 4

stage 5 "GPU verification gate"
common::gpu_gate
stage_done 5

# ── STAGE 6: W6.4 preflight — size the run at its cap, not its first iteration
stage 6 "Launch preflight (host memory at the window cap, VRAM floor)"
"$PY" -m games.seven_wonders_duel.cloud_preflight \
  --d-model "$D_MODEL" --layers "$LAYERS" --heads "$HEADS" \
  --device cuda \
  --replay-window-cap-games "$REPLAY_WINDOW_CAP_GAMES" \
  --example-cache-gb "$EXAMPLE_CACHE_GB" \
  --memory-budget-gb "$MEMORY_BUDGET_GB" \
  --memory-headroom-gb "$MEMORY_HEADROOM_GB" \
  --output "$REPO_DIR/$RUN_DIR_REL/preflight.json" \
  || die "Preflight refused this box. Destroy the instance and rent a bigger one."
stage_done 6

# ── STAGE 7: W6.2 engine equivalence — must run, must not skip ───────────────
stage 7 "Rust/Python engine equivalence suite"
if [ "$SKIP_EQUIV" = "1" ]; then
  warn "SKIP_EQUIV=1 — launching without verifying engine parity on this box."
else
  "$PY" -m games.seven_wonders_duel.cloud_equivalence_smoke \
    || die "Engine equivalence failed or was skipped — do not train on this box."
fi
stage_done 7

# ── STAGE 8: W6.2b precision arena (optional; needs an L checkpoint) ─────────
stage 8 "Precision arena (bf16 vs fp32)"
if [ -n "${PRECISION_ARENA_CHECKPOINT:-}" ]; then
  "$PY" -m games.seven_wonders_duel.precision_arena \
    --checkpoint "$PRECISION_ARENA_CHECKPOINT" \
    --work-dir "$REPO_DIR/$RUN_DIR_REL/precision_arena" \
    --output "$REPO_DIR/$RUN_DIR_REL/precision_arena.json" \
    --games "${PRECISION_ARENA_GAMES:-400}" \
    || die "bf16 differs from fp32 beyond its interval — relaunch with PRECISION=fp32."
else
  warn "PRECISION_ARENA_CHECKPOINT unset; skipping W6.2b. The shipped precision "
  warn "has then never played a scored game — set it to a W0 L checkpoint."
fi
stage_done 8

# ── STAGE 9: Phase D plumbing smoke on CUDA ──────────────────────────────────
stage 9 "Phase D plumbing smoke on CUDA"
if [ "$SKIP_SMOKE" = "1" ]; then
  warn "SKIP_SMOKE=1; skipping the CUDA plumbing smoke."
else
  SMOKE_DIR="runs/seven_wonders_duel/phase_d_smoke_$(date +%Y%m%dT%H%M%S)"
  "$PY" -m games.seven_wonders_duel.phase_d \
    --run-dir "$SMOKE_DIR" --device cuda --plumbing-smoke --process-workers 2 \
    || die "CUDA plumbing smoke failed — do not launch training."
  ok "Smoke completed: $SMOKE_DIR"
fi
stage_done 9

# ── STAGE 10: Launch training detached ───────────────────────────────────────
stage 10 "Launch training"
RUN_DIR="$REPO_DIR/$RUN_DIR_REL"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/launch_$(date +%Y%m%dT%H%M%S).log"

# W6.3: the throughput sweep and Phase D spell the same four settings
# differently. Translate rather than re-type.
TUNED_FLAGS=()
if [ -n "${LAUNCH_FLAGS_JSON:-}" ]; then
  read -r -a TUNED_FLAGS <<< "$(
    "$PY" -m games.seven_wonders_duel.f4_launch_flags "$LAUNCH_FLAGS_JSON"
  )" || die "Could not translate $LAUNCH_FLAGS_JSON into Phase D flags."
  ok "Measured launch flags: ${TUNED_FLAGS[*]}"
else
  warn "LAUNCH_FLAGS_JSON unset; launching on Phase D defaults rather than this "
  warn "box's measured sweep."
fi

read -r -a LADDER_RUNGS <<< "$GATE_LADDER"

# W7b ships present-but-disabled: detection reports either way, and enabling
# the response is a deliberate choice made at launch, not mid-run.
LADDER_FLAG=()
if [ "$INTERVENTION_LADDER" = "1" ]; then
  LADDER_FLAG=(--intervention-ladder)
  warn "W7b intervention ladder ENABLED: stagnation will change the schedules."
fi

TRAIN_CMD=(
  "$PY" -m games.seven_wonders_duel.phase_d
  --run-dir "$RUN_DIR_REL"
  --device cuda
  --iterations "$ITERATIONS"
  --games-per-iteration "$GAMES_PER_ITERATION"
  --seed-games "$SEED_GAMES"
  --workers "$WORKERS"
  --process-workers "$PROCESS_WORKERS"
  --d-model "$D_MODEL" --layers "$LAYERS" --heads "$HEADS"
  --precision "$PRECISION"
  --learning-rate "$LEARNING_RATE"
  --schedule-basis games
  --generation-backend rust --gate-backend rust
  --hof-opponent-fraction "$HOF_FRACTION" --hof-start-games "$HOF_START_GAMES"
  --selfplay-generator-mode soft_gate --revert-reset-after 2
  --promotion-min-lcb 0.50 --revert-max-ucb 0.48
  --gate-ladder-games "${LADDER_RUNGS[@]}"
  --gate-ladder-step-up-after 2
  --gate-ladder-floor-games "$GATE_LADDER_FLOOR_GAMES"
  --anchor-games "$ANCHOR_GAMES"
  --self-anchor-games "$SELF_ANCHOR_GAMES"
  --self-anchor-lag-games "$SELF_ANCHOR_LAG_GAMES"
  --self-anchor-every-games "$SELF_ANCHOR_EVERY_GAMES"
  --intervention-window-games "$INTERVENTION_WINDOW_GAMES"
  --replay-window-cap-games "$REPLAY_WINDOW_CAP_GAMES"
  --memory-budget-gb "$MEMORY_BUDGET_GB"
  --vram-budget-gb "$VRAM_BUDGET_GB"
  --memory-headroom-gb "$MEMORY_HEADROOM_GB"
  "${LADDER_FLAG[@]}"
  "${TUNED_FLAGS[@]}"
)

if [ "$LAUNCH" != "1" ]; then
  warn "LAUNCH=$LAUNCH; verified but not launching. Launch manually with:"
  warn "  cd $REPO_DIR && nohup ${TRAIN_CMD[*]} >> $LOG_FILE 2>&1 &"
  stage_done 10
  ok "Setup complete."
  exit 0
fi

cd "$REPO_DIR"
common::launch_detached "$LOG_FILE" "${TRAIN_CMD[@]}"
stage_done 10

cat <<EOF

Monitor:
  tail -f "$LOG_FILE"
  tail -f "$RUN_DIR/heartbeat.log"      # one line per iteration (W6.6)
  python -m tools.az_report "$RUN_DIR"  # full report, any time

Snapshot for download (waits for an iteration boundary, W6.7):
  python -m games.az_loop.snapshot "$RUN_DIR" ~/snapshot
  # then, from the laptop:
  scp -r -P <ssh-port> root@<instance-ip>:~/snapshot runs/seven_wonders_duel/

Resume after interruption (also the way to apply a code update — Phase D will
refuse a resume on a different commit unless --allow-resume-code-drift):
  bash $REPO_DIR/setup_cloud_7wd.sh

EOF
ok "Setup complete."
