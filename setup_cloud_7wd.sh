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
#   ITERATIONS=200 GAMES_PER_ITERATION=1000 SEED_GAMES=5000 WORKERS=8
#                   (200k games; --iterations always means "N more" on a resume)
#   D_MODEL=384 LAYERS=8 HEADS=6 PRECISION=bf16 LEARNING_RATE=5e-5
#   TRAIN_STEPS=<0.19 x games/iteration>  TRAIN_WARMUP_STEPS=<steps/3>
#                   derived, not defaulted — see the note beside them below
#   TRAIN_BATCH_SIZE=512
#   HOF_FRACTION=0.15 GATE_LADDER="200 600 1000 1500"
#   PROMOTION_EVERY=5 BOOTSTRAP_POLICY=auto_first_trained
#   PROBATION_RESET_AFTER=4 REVERT_RESET_AFTER=3
#   LAUNCH_FLAGS_JSON=<f4_cloud_finalize output>  measured --rust-* flags (W6.3)
#   PRECISION_ARENA_CHECKPOINT=<path>             runs W6.2b before launching
#   SELF_ANCHOR_GAMES=400 SELF_ANCHOR_LAG_GAMES=20000   W7a stagnation anchor
#   DISK_BUDGET_GB=0 DISK_HEADROOM_GB=5   0 = measure this box's free space
#   INTERVENTION_LADDER=0                                W7b response (off)
#   MEMORY_BUDGET_GB / VRAM_BUDGET_GB / MEMORY_HEADROOM_GB
#   RUN_DIR_REL=runs/seven_wonders_duel/cloud
#   LAUNCH=1        set 0 to stop after verification
#   SKIP_SMOKE=0    set 1 to skip the Phase D plumbing smoke
#   SKIP_EQUIV=0    set 1 to skip the equivalence suite (do not do this)
#   SWEEP_CHECKPOINT=<path>  runs the generation + gate scheduler sweeps
#                   (two-pass: stage 8b writes sweeps/measured_env.sh;
#                    source it and re-run to launch on those numbers)
#   RUST_SLOTS / RUST_GLOBAL_BATCH_CAP / RUST_MAX_INFLIGHT_BATCHES
#                   generation-side scheduler settings, normally set by
#                   sourcing measured_env.sh rather than by hand
#   SWEEP_SLOTS_CSV / SWEEP_CAPS_CSV / SWEEP_INFLIGHT_CSV  generation grid
#   SWEEP_SLOTS / SWEEP_CAPS  gate grid (space separated; different harness)
#   SWEEP_GENERATION_GAMES=200 SWEEP_REPETITIONS=1
#   SKIP_SWEEPS=0   set 1 to launch on defaults rather than this box
#   GATE_SWEEP_RUNGS  gate sizes to sweep (default: ladder's middle rung)
#   GATE_SLOTS / GATE_GLOBAL_BATCH_CAP  gate-side scheduler settings; the
#                   gate wants a wider pool and cap than generation, and
#                   sharing one value costs whichever path is misfitted
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
# Sized for a 200k-game run: 200 x 1,000. Games per iteration is deliberately
# absent from the schedule identity (W1.2), so it is free to change on a resume
# and larger iterations are pure savings -- half the checkpoint pairs, gate
# cycles, replay-derivation passes and log rows for the same games.
ITERATIONS="${ITERATIONS:-200}"
GAMES_PER_ITERATION="${GAMES_PER_ITERATION:-1000}"
SEED_GAMES="${SEED_GAMES:-5000}"
WORKERS="${WORKERS:-8}"
PROCESS_WORKERS="${PROCESS_WORKERS:-$(nproc)}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
PRECISION="${PRECISION:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
# Train steps are COUPLED to games per iteration and must be derived, never
# defaulted: `--train-steps` defaults to 300 in the parser, which at 1,000 games
# an iteration is ~8x sample reuse and at 500 is ~16x, against the ~5x this loop
# is tuned for (run 03 used 76 steps at 400 games). Leaving it unset means the
# paid run trains at whatever the parser happens to say -- the same failure as
# the lifecycle flags the run-03 remediation had to add.
#
# 0.19 x games: ~19.4 recorded positions per game (measured with
# --record-fast-moves off), 5 passes each, at batch 512. Warmup is a third of
# the budget because the parser's default 100 can exceed the whole of it.
TRAIN_STEPS="${TRAIN_STEPS:-$(( (GAMES_PER_ITERATION * 19 + 99) / 100 ))}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-$(( TRAIN_STEPS / 3 ))}"
HOF_FRACTION="${HOF_FRACTION:-0.15}"
HOF_START_GAMES="${HOF_START_GAMES:-10000}"
GATE_LADDER="${GATE_LADDER:-200 600 1000 1500}"
GATE_LADDER_FLOOR_GAMES="${GATE_LADDER_FLOOR_GAMES:-10000}"
PROMOTION_EVERY="${PROMOTION_EVERY:-5}"
BOOTSTRAP_POLICY="${BOOTSTRAP_POLICY:-auto_first_trained}"
PROBATION_RESET_AFTER="${PROBATION_RESET_AFTER:-4}"
REVERT_RESET_AFTER="${REVERT_RESET_AFTER:-3}"
ANCHOR_GAMES="${ANCHOR_GAMES:-200}"
# 400, not 200: the self-anchor is the run's stopping rule, and 100 pairs
# resolve a lagged advantage of 0.60+ easily but clear LCB > 0.50 only ~13% of
# the time at 0.55 -- blind exactly where "am I still improving" gets decided.
SELF_ANCHOR_GAMES="${SELF_ANCHOR_GAMES:-400}"
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
SKIP_SWEEPS="${SKIP_SWEEPS:-0}"
GATE_SLOTS="${GATE_SLOTS:-}"
RUST_SLOTS="${RUST_SLOTS:-}"
RUST_GLOBAL_BATCH_CAP="${RUST_GLOBAL_BATCH_CAP:-}"
RUST_MAX_INFLIGHT_BATCHES="${RUST_MAX_INFLIGHT_BATCHES:-}"
GATE_GLOBAL_BATCH_CAP="${GATE_GLOBAL_BATCH_CAP:-}"
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
stage 6 "Launch preflight (host memory at the window cap, VRAM floor, disk)"
"$PY" -m games.seven_wonders_duel.cloud_preflight \
  --d-model "$D_MODEL" --layers "$LAYERS" --heads "$HEADS" \
  --device cuda \
  --iterations "$ITERATIONS" \
  --games-per-iteration "$GAMES_PER_ITERATION" \
  --seed-games "$SEED_GAMES" \
  --promotion-every "$PROMOTION_EVERY" \
  --run-dir "$REPO_DIR/$RUN_DIR_REL" \
  --disk-budget-gb "${DISK_BUDGET_GB:-0}" \
  --disk-headroom-gb "${DISK_HEADROOM_GB:-5}" \
  --replay-window-cap-games "$REPLAY_WINDOW_CAP_GAMES" \
  --example-cache-gb "$EXAMPLE_CACHE_GB" \
  --memory-budget-gb "$MEMORY_BUDGET_GB" \
  --memory-headroom-gb "$MEMORY_HEADROOM_GB" \
  --output "$REPO_DIR/$RUN_DIR_REL/preflight.json" \
  && _preflight_status=0 || _preflight_status=$?
# A refusal (exit 1) and a crash are different events and must not share a
# message: "rent a bigger box" is terrible advice for a bug in the check.
if [ "$_preflight_status" -eq 1 ]; then
  die "Preflight REFUSED this box — see the failures above. Fix the flagged budget or destroy the instance and rent a bigger one."
elif [ "$_preflight_status" -ne 0 ]; then
  die "Preflight CRASHED (exit $_preflight_status). That is a bug in the check, not a verdict on this box; nothing here says the hardware is wrong."
fi
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

# ── STAGE 8b: Scheduler sweeps (generation and gate, separately) ─────────────
#
# Generation and the gate need *different* settings, and the axes interact, so
# both are swept jointly on the box that will run them. Measured on the laptop
# 3070 (d128 L4, 64 sims, 100-game gates, games/s):
#
#     slots \ cap     256      512     1024
#     48 (shipped)   0.605    0.571    0.581
#     144            0.752    0.816    0.840
#
# The cap's *sign* flips with slot count: at 48 slots widening it costs 4%, at
# 144 slots it gains 12%. Generation is pinned near 48 slots and is ~85% of an
# iteration, so `--gate-global-batch-cap` exists to keep a gate-sized cap away
# from it. Sweeping either axis alone concludes the shipped setting is optimal.
#
# One gate sweep is enough: the optimum is stable in gate size. 144 slots /
# 1024 cap won at 100, 200 and 600 games on the laptop 3070, and the gain over
# 48/256 barely moved (1.39x at 100 games, 1.37x at 600). GATE_SWEEP_RUNGS
# defaults to the ladder's middle rung; pass more than one value if a box looks
# unlike the others, since nothing guarantees that stability on new hardware.

stage 8b "Scheduler sweeps (generation, then gate)"
if [ "$SKIP_SWEEPS" = "1" ]; then
  warn "SKIP_SWEEPS=1 — launching on defaults rather than this box's measurement."
elif [ -z "${SWEEP_CHECKPOINT:-}" ]; then
  warn "SWEEP_CHECKPOINT unset; skipping both sweeps. Phase D will run on"
  warn "defaults measured on a different GPU. Set it to a W0 L checkpoint."
else
  SWEEP_DIR="$REPO_DIR/$RUN_DIR_REL/sweeps"
  mkdir -p "$SWEEP_DIR"

  # f4_phase_d_sweep takes COMMA-separated axes and an --output DIRECTORY (it
  # writes phase_d_sweep.json inside). w5_gate_slots_sweep takes space-separated
  # axes and an --output FILE. They are different harnesses; test_setup_cloud
  # arg-parses both invocations so this cannot drift again.
  "$PY" -m games.seven_wonders_duel.f4_phase_d_sweep \
    --checkpoint "$SWEEP_CHECKPOINT" \
    --output "$SWEEP_DIR/generation" \
    --games "${SWEEP_GENERATION_GAMES:-200}" \
    --repetitions "${SWEEP_REPETITIONS:-1}" \
    --slots "${SWEEP_SLOTS_CSV:-48,96,144}" \
    --caps "${SWEEP_CAPS_CSV:-256,1024}" \
    --inflight "${SWEEP_INFLIGHT_CSV:-1}" \
    --device cuda \
    || die "Generation sweep failed - do not launch on unmeasured settings."
  ok "Generation sweep: $SWEEP_DIR/generation/phase_d_sweep.json"

  # Sweep the ladder's *lowest* rung: the gate optimum measured stable across
  # 100/200/600-game gates on the laptop 3070, so the cheap rung answers the
  # same question at a fraction of the games. Override with GATE_SWEEP_RUNGS.
  read -r -a _RUNGS <<< "$GATE_LADDER"
  read -r -a _SWEEP_RUNGS <<< "${GATE_SWEEP_RUNGS:-${_RUNGS[0]}}"
  for RUNG in "${_SWEEP_RUNGS[@]}"; do
    "$PY" -m games.seven_wonders_duel.w5_gate_slots_sweep \
      --checkpoint "$SWEEP_CHECKPOINT" \
      --work-dir "$SWEEP_DIR/gate_$RUNG" \
      --output "$SWEEP_DIR/gate_$RUNG.json" \
      --games "$RUNG" \
      --slots ${SWEEP_SLOTS:-48 96 144} \
      --caps ${SWEEP_CAPS:-256 1024} \
      --sims "${GATE_SIMS:-64}" \
      --precision "$PRECISION" \
      || die "Gate sweep at rung $RUNG failed."
    ok "Gate sweep (rung $RUNG): $SWEEP_DIR/gate_$RUNG.json"
  done

  # Turn both results into an env file pass 2 can source. The generation sweep
  # writes {summary: [...]} sorted fastest-first; the gate sweep writes {best:
  # {...}}. Neither is in the production-manifest shape f4_launch_flags reads,
  # so the translation lives here rather than pretending LAUNCH_FLAGS_JSON can
  # consume a sweep.
  "$PY" "$REPO_DIR/games/seven_wonders_duel/sweep_launch_env.py" \
    --sweep-dir "$SWEEP_DIR" --gate-rung "${_SWEEP_RUNGS[0]}" \
    || die "Could not summarise the sweeps."

  warn "Sweeps measure but do not apply. To launch on this box's numbers:"
  warn "  source $SWEEP_DIR/measured_env.sh && bash \$0"
fi
stage_done 8b

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
[ -n "$RUST_SLOTS" ] && TUNED_FLAGS+=(--rust-slots "$RUST_SLOTS")
[ -n "$RUST_GLOBAL_BATCH_CAP" ] &&
  TUNED_FLAGS+=(--rust-global-batch-cap "$RUST_GLOBAL_BATCH_CAP")
[ -n "$RUST_MAX_INFLIGHT_BATCHES" ] &&
  TUNED_FLAGS+=(--rust-max-inflight-batches "$RUST_MAX_INFLIGHT_BATCHES")
if [ ${#TUNED_FLAGS[@]} -gt 0 ]; then
  ok "Measured generation flags: ${TUNED_FLAGS[*]}"
elif [ -n "${LAUNCH_FLAGS_JSON:-}" ]; then
  read -r -a TUNED_FLAGS <<< "$(
    "$PY" -m games.seven_wonders_duel.f4_launch_flags "$LAUNCH_FLAGS_JSON"
  )" || die "Could not translate $LAUNCH_FLAGS_JSON into Phase D flags."
  ok "Measured launch flags: ${TUNED_FLAGS[*]}"
else
  warn "LAUNCH_FLAGS_JSON unset; launching on Phase D defaults rather than this "
  warn "box's measured sweep."
fi

read -r -a LADDER_RUNGS <<< "$GATE_LADDER"

# Gate-side scheduler settings, from stage 8b. Deliberately separate from the
# generation flags in TUNED_FLAGS: the two paths run at different slot counts,
# and the batch cap helps at one and hurts at the other.
GATE_TUNED_FLAGS=()
[ -n "$GATE_SLOTS" ] && GATE_TUNED_FLAGS+=(--gate-slots "$GATE_SLOTS")
[ -n "$GATE_GLOBAL_BATCH_CAP" ] &&
  GATE_TUNED_FLAGS+=(--gate-global-batch-cap "$GATE_GLOBAL_BATCH_CAP")
if [ ${#GATE_TUNED_FLAGS[@]} -eq 0 ]; then
  warn "GATE_SLOTS/GATE_GLOBAL_BATCH_CAP unset; the gate will run on generation's"
  warn "scheduler settings, which measured ~1.2x slower on the laptop 3070."
else
  ok "Gate scheduler flags: ${GATE_TUNED_FLAGS[*]}"
fi

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
  --train-steps "$TRAIN_STEPS"
  --train-warmup-steps "$TRAIN_WARMUP_STEPS"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --schedule-basis games
  --generation-backend rust --gate-backend rust
  --hof-opponent-fraction "$HOF_FRACTION" --hof-start-games "$HOF_START_GAMES"
  --selfplay-generator-mode soft_gate
  --bootstrap-policy "$BOOTSTRAP_POLICY"
  --promotion-every "$PROMOTION_EVERY"
  --revert-reset-after "$REVERT_RESET_AFTER"
  --probation-reset-after "$PROBATION_RESET_AFTER"
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
  "${GATE_TUNED_FLAGS[@]}"
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
