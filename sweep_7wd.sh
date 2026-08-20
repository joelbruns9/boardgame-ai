#!/usr/bin/env bash
# =============================================================================
# sweep_7wd.sh — measure this box's scheduler geometry. Never launches training.
#
# setup_cloud_7wd.sh cannot be used for this: its stage 10 launches the run, and
# stage 8b's sweep is opt-in, pinned to inflight=1, and measures at one shard
# while configuring a run at four. This script does the measurement alone.
#
# What it does:
#   1. works out of a SEPARATE checkout, so the running run's repo keeps the
#      commit its manifest recorded and stays resumable
#   2. PROFILES THE LIVE RUN from what it has already written to
#      training_log.jsonl -- read-only, requires no stop, and answers whether
#      the solver blocks generation and whether the batches are wide enough.
#      PROFILE_ONLY=1 stops here, which is the zero-risk diagnosis.
#   3. refuses to go further while Phase D is alive (a sweep sharing the GPU
#      with training measures neither configuration)
#   4. copies a checkpoint out of the run rather than reading it in place
#   5. sweeps slots x cap x inflight x scheduler-workers, with the solver
#      configured as the run configures it
#   6. writes measured_env.sh and prints the flags that changed
#
# What it deliberately does NOT do:
#   * launch or resume training -- that stays a decision you make
#   * build the Rust crate. `maturin develop` installs into the shared site-
#     packages, so building from the sweep checkout would replace the extension
#     the training run uses. If the crate source differs between the two
#     checkouts, this refuses rather than building.
#   * write anything into the run directory
#
# Usage (from the run's checkout, when it already has this file):
#   bash ~/boardgame-ai/sweep_7wd.sh
# or without touching the run's checkout at all, which is the point:
#   curl -fsSL https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/sweep_7wd.sh -o sweep_7wd.sh
#   SWEEP_REF=main bash sweep_7wd.sh
#
# There is no self-update trap here, unlike the setup scripts: this never pulls
# into the directory it is running from. It fetches into $SWEEP_REPO, which is
# never the copy bash is executing.
#
#   GAMES=300 bash sweep_7wd.sh             # longer, tighter measurement
#   SWEEP_SLOTS=256,512,1024 bash sweep_7wd.sh
#
# Knobs:
#   RUN_DIR=~/boardgame-ai/runs/seven_wonders_duel/cloud   run being tuned
#   RUN_REPO=~/boardgame-ai        the checkout the run was launched from
#   SWEEP_REPO=~/sweep-checkout    where this script works (created if absent)
#   SWEEP_REF=<branch|sha>         what to check out there (default: the run's
#                                  own commit, i.e. sweep the code that is
#                                  running; set it to pick up sweep fixes)
#   CHECKPOINT=<path>              default: the run's current_best.pt
#   GAMES=200 REPETITIONS=2 WARMUP_GAMES=8
#   SWEEP_SLOTS / SWEEP_CAPS / SWEEP_INFLIGHT / SWEEP_WORKERS  (comma separated)
#   SOLVER_THREADS=3               per shard, as the run passes it
#   PRECISION=bf16                 must match the run
#   GATE_RUNG=200                  gate sweep rung; empty to skip the gate sweep
#   OUTPUT=~/sweep_7wd             results directory
#   FORCE=0                        1 to sweep anyway while training is alive
#   PROFILE_ONLY=0                 1 to read the live run and stop. Safe at any
#                                  time; nothing is stopped or measured on GPU.
#   PROFILE_ITERATIONS=10          how many iterations of history to show
#   SOLVER_NODE_RATE=<nodes/s/thread>  enables the solver capacity estimate
#                                  (stage 6b of the launcher measures this)
#   SOLVER_THREADS_TOTAL=<n>       --solver-threads x scheduler workers
# =============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/joelbruns9/boardgame-ai.git}"
RUN_REPO="${RUN_REPO:-$HOME/boardgame-ai}"
RUN_DIR="${RUN_DIR:-$RUN_REPO/runs/seven_wonders_duel/cloud}"
SWEEP_REPO="${SWEEP_REPO:-$HOME/sweep-checkout}"
OUTPUT="${OUTPUT:-$HOME/sweep_7wd}"
FORCE="${FORCE:-0}"

GAMES="${GAMES:-200}"
REPETITIONS="${REPETITIONS:-2}"
WARMUP_GAMES="${WARMUP_GAMES:-8}"
PRECISION="${PRECISION:-bf16}"
SOLVER_THREADS="${SOLVER_THREADS:-3}"
GATE_RUNG="${GATE_RUNG:-200}"

# Grids. The shipped values (256 slots, 2048 cap, inflight 1, 4 workers) are
# INSIDE every axis on purpose: a sweep whose grid excludes the current setting
# cannot tell you whether changing it is an improvement.
#
# Workers sweeps DOWN as well as up. Shards divide the concurrent games between
# them, and a batch is formed from one shard's ready games -- so at a fixed slot
# count, more shards means SMALLER batches. With 256 slots the observed mean
# batch was 42 at 4 shards; one shard would submit the same games as one batch.
# The trade is that each shard is a single OS thread doing tree work, so fewer
# shards can starve the GPU from the other direction. That is the measurement.
SWEEP_SLOTS="${SWEEP_SLOTS:-256,512,768}"
SWEEP_CAPS="${SWEEP_CAPS:-2048}"
SWEEP_INFLIGHT="${SWEEP_INFLIGHT:-1,2}"
SWEEP_WORKERS="${SWEEP_WORKERS:-1,2,4,8}"

if [ -f "$RUN_REPO/setup_cloud_common.sh" ]; then
  # shellcheck source=setup_cloud_common.sh
  source "$RUN_REPO/setup_cloud_common.sh"
else
  log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
  ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }
  warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
  die()  { printf '\033[1;31m[FATAL]\033[0m %s\n' "$*" >&2; exit 1; }
  stage()      { printf '\n\033[1;36m=== STAGE %s: %s ===\033[0m\n' "$1" "$2"; }
  stage_done() { printf '\033[1;32m=== STAGE %s COMPLETE ===\033[0m\n' "$1"; }
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

mkdir -p "$OUTPUT"
SWEEP_LOG="$OUTPUT/sweep_$(date +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$SWEEP_LOG") 2>&1
log "logging to $SWEEP_LOG"

# Which copy of THIS FILE is running.
#
# Bumped by hand on every change to this script. Without it, "the fix did not
# work" and "you ran a cached copy of the previous version" are the same
# observation -- raw.githubusercontent is CDN-cached, so a curl seconds after a
# push can legitimately return the old file. That ambiguity cost a full
# debugging round trip; the version line ends it.
SWEEP_SCRIPT_VERSION=4
log "sweep_7wd.sh version $SWEEP_SCRIPT_VERSION (checksum $(cksum < "${BASH_SOURCE[0]}" | cut -d' ' -f1))"

# ── STAGE 1: a checkout that is not the run's ────────────────────────────────
# `_refuse_changed_code` compares the whole repo's commit and dirty diff against
# what the manifest recorded, so a `git pull` in the run's own checkout ends the
# ability to resume it. Work somewhere else.
stage 1 "Sweep checkout at $SWEEP_REPO"
[ -d "$RUN_REPO/.git" ] || die "$RUN_REPO is not a git checkout"
RUN_COMMIT="$(git -C "$RUN_REPO" rev-parse HEAD)"
SWEEP_REF="${SWEEP_REF:-$RUN_COMMIT}"
log "run checkout is at ${RUN_COMMIT:0:12}; sweeping ${SWEEP_REF:0:12}"

# Cloned from GITHUB, not from $RUN_REPO. Cloning locally would mean the only
# way to get newer sweep code onto this box is `git pull` in the run's own
# checkout -- which is exactly the thing that ends the run's resumability. The
# sweep checkout fetches its own code and leaves $RUN_REPO alone.
if [ ! -d "$SWEEP_REPO/.git" ]; then
  git clone "$REPO_URL" "$SWEEP_REPO" \
    || git clone "$RUN_REPO" "$SWEEP_REPO" \
    || die "could not clone $REPO_URL or $RUN_REPO"
fi
git -C "$SWEEP_REPO" fetch --all --quiet --tags \
  || warn "fetch failed; using what is on disk"

# Check out the REMOTE ref, not the local branch of the same name.
#
# `git fetch` advances origin/main; it does NOT advance the local `main` that a
# clone left behind. So `git checkout main` in an existing clone lands on the
# commit that clone was made at, however long ago -- fetching first changes
# nothing about that. This script fetched and then checked out the local branch,
# so the second run of it re-measured the FIRST run's code and reported the
# stale commit as though it were current. Fetch and checkout are not a pull.
#
# Detached HEAD is deliberate: this checkout is disposable and exists to be at
# one exact commit, and detaching makes "which code ran" unambiguous in the log.
if git -C "$SWEEP_REPO" rev-parse --verify --quiet "origin/$SWEEP_REF^{commit}" >/dev/null; then
  git -C "$SWEEP_REPO" checkout --quiet --detach "origin/$SWEEP_REF" \
    || die "could not check out 'origin/$SWEEP_REF' in $SWEEP_REPO"
else
  # A raw SHA or a tag: not under origin/, so use it directly.
  git -C "$SWEEP_REPO" checkout --quiet --detach "$SWEEP_REF" \
    || die "could not check out '$SWEEP_REF' in $SWEEP_REPO (fetched from $REPO_URL)"
fi
SWEEP_COMMIT="$(git -C "$SWEEP_REPO" rev-parse HEAD)"
ok "sweep checkout at ${SWEEP_COMMIT:0:12}"

if [ "$SWEEP_COMMIT" != "$RUN_COMMIT" ]; then
  warn "The sweep runs different Python than the training run. That is fine for"
  warn "measuring geometry, but the run's checkout was NOT touched -- it stays"
  warn "at ${RUN_COMMIT:0:12} and stays resumable."
fi
stage_done 1

# ── STAGE 2: read what the live run has already recorded ─────────────────────
# Read-only, and deliberately BEFORE the "training must not be running" check.
# The Rust scheduler has been writing a full per-shard time breakdown into
# training_log.jsonl all along -- nothing surfaced it. It answers whether the
# solver blocks generation and whether batches are wide enough, on the real
# configuration, without stopping anything. Sweep only what this cannot settle.
stage 2 "Profile of the live run"
cd "$SWEEP_REPO"
"$PY" -m games.seven_wonders_duel.generation_profile "$RUN_DIR" \
  --last "${PROFILE_ITERATIONS:-10}" \
  ${SOLVER_NODE_RATE:+--solver-node-rate "$SOLVER_NODE_RATE"} \
  ${SOLVER_THREADS_TOTAL:+--solver-threads-total "$SOLVER_THREADS_TOTAL"} \
  || warn "could not profile $RUN_DIR (continuing)"
stage_done 2

if [ "${PROFILE_ONLY:-0}" = "1" ]; then
  ok "PROFILE_ONLY=1: nothing measured on the GPU, nothing stopped, nothing
  in the run directory touched."
  exit 0
fi

# ── STAGE 3: the GPU must be ours alone ──────────────────────────────────────
# Two processes sharing one GPU measure each other, and the numbers look
# plausible either way. This is the check that makes the sweep mean anything.
stage 3 "Training must not be running"
LIVE="$(pgrep -af "phase_d" || true)"
if [ -n "$LIVE" ]; then
  warn "Phase D looks alive:"
  printf '%s\n' "$LIVE"
  if [ "$FORCE" != "1" ]; then
    die "Stop training before sweeping (kill the pid above), or set FORCE=1 to
  accept measurements taken against a busy GPU. A sweep run beside training
  measures contention, not geometry -- and the winning point it picks will be
  the one that tolerated the interference best.
  To diagnose without stopping anything, re-run with PROFILE_ONLY=1."
  fi
  warn "FORCE=1: sweeping anyway. Every number below is contended."
fi
stage_done 3

# ── STAGE 4: the extension must match the code being swept ───────────────────
# The crate is installed into site-packages by `maturin develop`. Rebuilding it
# here would swap the extension out from under the training run, so this refuses
# instead. A Python-only difference is safe; a crate difference is not.
stage 4 "Rust extension matches the sweep checkout"
RUN_CRATE="$(git -C "$RUN_REPO" rev-parse "$RUN_COMMIT:games/seven_wonders_duel/seven_wonders_rust")"
SWEEP_CRATE="$(git -C "$SWEEP_REPO" rev-parse "$SWEEP_COMMIT:games/seven_wonders_duel/seven_wonders_rust")"
if [ "$RUN_CRATE" != "$SWEEP_CRATE" ]; then
  die "The crate source differs between the two checkouts. The installed
  extension was built from the run's copy, and rebuilding it here would replace
  the .so the training run loads. Pick a SWEEP_REF whose crate matches, or
  rebuild deliberately on a box that is not training."
fi
"$PY" - <<'PYEXT' || die "seven_wonders_rust does not import"
import seven_wonders_rust as swr
print(f"extension: {swr.__file__}")
PYEXT
ok "crate source identical in both checkouts; extension importable"
stage_done 4

# ── STAGE 5: a checkpoint, copied out ────────────────────────────────────────
# Copied rather than read in place: the run rewrites current_best.pt on every
# promotion, and a sweep that reloads it mid-grid would compare points measured
# on two different networks.
stage 5 "Checkpoint"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/current_best.pt}"
[ -f "$CHECKPOINT" ] || die "no checkpoint at $CHECKPOINT (set CHECKPOINT=<path>)"
SWEEP_CHECKPOINT="$OUTPUT/sweep_checkpoint.pt"
cp "$CHECKPOINT" "$SWEEP_CHECKPOINT"
ok "copied $CHECKPOINT -> $SWEEP_CHECKPOINT ($(du -h "$SWEEP_CHECKPOINT" | cut -f1))"
stage_done 5

# ── STAGE 6: generation sweep ────────────────────────────────────────────────
stage 6 "Generation sweep"
MAX_SLOTS="$(printf '%s' "$SWEEP_SLOTS" | tr ',' '\n' | sort -n | tail -1)"
if [ "$GAMES" -le "$MAX_SLOTS" ]; then
  warn "GAMES=$GAMES is not above the largest slot count ($MAX_SLOTS): that point"
  warn "cannot refill its pool, so it measures activation rather than throughput."
  warn "Raise GAMES above $MAX_SLOTS, or drop the top of SWEEP_SLOTS."
fi
log "grid: slots=$SWEEP_SLOTS caps=$SWEEP_CAPS inflight=$SWEEP_INFLIGHT workers=$SWEEP_WORKERS"
log "solver: $SOLVER_THREADS per shard (x workers = the total at each point)"
cd "$SWEEP_REPO"
"$PY" -m games.seven_wonders_duel.f4_phase_d_sweep \
  --checkpoint "$SWEEP_CHECKPOINT" \
  --output "$OUTPUT/generation" \
  --games "$GAMES" \
  --repetitions "$REPETITIONS" \
  --warmup-games "$WARMUP_GAMES" \
  --slots "$SWEEP_SLOTS" \
  --caps "$SWEEP_CAPS" \
  --inflight "$SWEEP_INFLIGHT" \
  --workers "$SWEEP_WORKERS" \
  --solver-threads "$SOLVER_THREADS" \
  --device cuda \
  --precision "$PRECISION" \
  || die "generation sweep did not complete; nothing was measured"
ok "generation: $OUTPUT/generation/phase_d_sweep.json"
stage_done 6

# ── STAGE 7: gate sweep (different harness, different shape) ─────────────────
stage 7 "Gate sweep"
if [ -z "$GATE_RUNG" ]; then
  warn "GATE_RUNG empty; skipping. measured_env.sh will not be written, since"
  warn "sweep_launch_env needs both halves."
else
  "$PY" -m games.seven_wonders_duel.w5_gate_slots_sweep \
    --checkpoint "$SWEEP_CHECKPOINT" \
    --work-dir "$OUTPUT/gate_$GATE_RUNG" \
    --output "$OUTPUT/gate_$GATE_RUNG.json" \
    --games "$GATE_RUNG" \
    --slots ${SWEEP_GATE_SLOTS:-96 144 256} \
    --caps ${SWEEP_GATE_CAPS:-1024 2048} \
    --sims "${GATE_SIMS:-64}" \
    --scheduler-workers "${GATE_WORKERS:-4}" \
    --precision "$PRECISION" \
    || die "gate sweep at rung $GATE_RUNG failed"
  ok "gate: $OUTPUT/gate_$GATE_RUNG.json"
fi
stage_done 7

# ── STAGE 8: the numbers, as flags ───────────────────────────────────────────
stage 8 "Result"
if [ -n "$GATE_RUNG" ]; then
  "$PY" "$SWEEP_REPO/games/seven_wonders_duel/sweep_launch_env.py" \
    --sweep-dir "$OUTPUT" --gate-rung "$GATE_RUNG" \
    || die "could not summarise the sweeps"
  echo
  log "measured settings ($OUTPUT/measured_env.sh):"
  cat "$OUTPUT/measured_env.sh"
fi

"$PY" - "$OUTPUT/generation/phase_d_sweep.json" <<'PYSUM' || true
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
rows = payload.get("summary") or []
if rows:
    print("\n  slots  cap   inflight workers  batch   games/h   vs best")
    best = rows[0]["median_games_per_hour"]
    for row in rows:
        print(
            f"  {row['slots']:<6} {row['global_batch_cap']:<5} "
            f"{row['max_inflight_batches']:<8} {row.get('scheduler_workers', 1):<8} "
            f"{row.get('median_batch_size', 0):<7.0f} "
            f"{row['median_games_per_hour']:<9.0f} "
            f"{row['median_games_per_hour'] / best:.2f}x"
        )
PYSUM

cat <<EOF

Nothing was launched and nothing in the run directory was touched.

To adopt these numbers, stop training and restart it with them, e.g.:

  kill \$(pgrep -f phase_d | head -1)
  source $OUTPUT/measured_env.sh
  SKIP_SWEEPS=1 bash $RUN_REPO/setup_cloud_7wd.sh

That RESUMES (do not delete the run directory). It resumes only while
$RUN_REPO stays at ${RUN_COMMIT:0:12} -- pulling new code there makes Phase D
refuse the resume.
EOF
stage_done 8
