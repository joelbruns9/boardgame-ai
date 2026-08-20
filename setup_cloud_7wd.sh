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
#   DRAFT_PRIOR_GAMES=10000 CURRICULUM_ANNEAL_GAMES=15000 HOF_START_GAMES=50000
#                   staggered scaffold schedule -- see the block below. Run 04
#                   ended all four at 10,000 games at once and the value head
#                   collapsed nine iterations later.
#   ANCHOR_GATE_EVERY_PROMOTIONS=0  bot anchors off; they saturate by iteration 10
#   PACK_THREADS=0  pack pool size; 0 = derive from cgroup/cpuset/affinity
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
# Capped, not $(nproc): process workers only do anything on the PYTHON
# generation backend, and this launch is Rust on both paths. On a 192-thread
# EPYC the bare core count would spawn 192 processes that each import torch, for
# a stage that is either inert or a seed buffer.
PROCESS_WORKERS="${PROCESS_WORKERS:-$(n=$(nproc); [ "$n" -gt 16 ] && echo 16 || echo "$n")}"
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
# ── Scaffold schedule: STAGGERED. Run 04 came off all of these at once.
#
# Three supports were removed and one new opponent distribution introduced at
# exactly 10,000 games -- the curriculum bot mix, the seed corpus's share of
# training, the wonder-draft prior, and HOF switching on. All four were keyed to
# the same clock, three of them by parser defaults the launcher never passed.
# The replay window then smeared the shift over ~7 iterations, so it surfaced at
# iteration 19 as a value-head collapse (value_acc 0.729 -> 0.639, train/val gap
# +0.03 -> +0.41) that the gate rejected twice, at 0.360 and 0.350, while
# policy_top1 kept improving.
#
# Each knot now clears the replay window before the next arrives, so a
# recurrence is attributable to one cause instead of four.

# Earliest and least entangled: it touches search priors, not the opponent
# distribution, so removing it cannot change what the value head is trained on.
DRAFT_PRIOR_GAMES="${DRAFT_PRIOR_GAMES:-10000}"

# Also moves seed retention, which shares this duration by design.
#
# NOT stretched. The bots are spent by ~10k: run 04's iteration-10 anchor gates
# scored 0.945-0.985 against all four archetypes and 1.000 against greedy, and
# the code's own note calls 10,000 "measured, not chosen" for exactly that
# reason. Holding them open longer would spend 15% of generation on decided
# games and feed them to training through the seed corpus. 15,000 buys clear
# separation from the draft prior and nothing more.
CURRICULUM_ANNEAL_GAMES="${CURRICULUM_ANNEAL_GAMES:-15000}"

# Was 10,000, which put league play against the *bootstrap* checkpoint. Run 04's
# league opponent was hof_iter_0000 at iterations 10, 11, 13, 14, 15, 16, 17, 19,
# 21 and 23 -- the pool only held promotions from iterations 0, 5 and 10, so
# sampling kept returning the weakest. That made 15% of every iteration's games
# lopsided wins, i.e. value targets pinned near +1, which is a live suspect for
# the value head degrading while the policy head improved.
#
# HOF earns its keep against opponents strong enough to punish forgetting, and
# the model improves too fast early for a frozen checkpoint to stay relevant.
# Note the pool is filled by *promotions*: run 04 managed four in thirty
# iterations, so a late start only helps if promotions have happened by then.
HOF_START_GAMES="${HOF_START_GAMES:-50000}"
GATE_LADDER="${GATE_LADDER:-200 600 1000 1500}"
GATE_LADDER_FLOOR_GAMES="${GATE_LADDER_FLOOR_GAMES:-10000}"
PROMOTION_EVERY="${PROMOTION_EVERY:-5}"
BOOTSTRAP_POLICY="${BOOTSTRAP_POLICY:-auto_first_trained}"
PROBATION_RESET_AFTER="${PROBATION_RESET_AFTER:-4}"
REVERT_RESET_AFTER="${REVERT_RESET_AFTER:-3}"
ANCHOR_GAMES="${ANCHOR_GAMES:-200}"
# Off. The bot anchor suite saturates within a few iterations -- run 04's only
# firing, at iteration 10, scored 0.945-1.000 across all five opponents -- so it
# costs five gates x ANCHOR_GAMES per promotion to re-measure a known ceiling.
# Set to 3 to restore the default cadence if out-of-distribution strength ever
# needs tracking again.
ANCHOR_GATE_EVERY_PROMOTIONS="${ANCHOR_GATE_EVERY_PROMOTIONS:-0}"
# Threads for the Rust feature-packing pool. 0 derives it from the cgroup quota,
# cpuset and affinity. Set it explicitly after running f4_pack_sweep on the box:
# on a hybrid P/E-core part the plateau measured on a homogeneous laptop does not
# necessarily transfer, and the sweep costs seconds.
PACK_THREADS="${PACK_THREADS:-0}"

# ── Search geometry, targets and the endgame solver ─────────────────────────
#
# NONE of this was in the script before 2026-08-19, so a launch from here took
# the PARSER's defaults: 16-24 cheap sims against cloud6's 100, 64-128 full
# against 1600, Gumbel where the plan ships PUCT, no Dirichlet, and the endgame
# solver off. cloud6's own command carried these by hand, which is why the gap
# went unnoticed -- the script has never actually launched the shipped
# configuration.
CHEAP_SIMS="${CHEAP_SIMS:-100}"
FULL_SIMS="${FULL_SIMS:-1600}"
FULL_SEARCH_FRACTION="${FULL_SEARCH_FRACTION:-0.25}"
TOP_K="${TOP_K:-16}"
AGE_DEAL_SAMPLES="${AGE_DEAL_SAMPLES:-32}"

# PUCT for generation and EVALUATION, Gumbel for cheap moves. The eval mode is
# the one that is easy to get wrong: the advisor deploys under PUCT, so a gate
# run under Gumbel promotes on a number nobody will ever see again.
SELFPLAY_SEARCH_MODE="${SELFPLAY_SEARCH_MODE:-puct}"
CHEAP_SEARCH_MODE="${CHEAP_SEARCH_MODE:-gumbel}"
EVAL_SEARCH_MODE="${EVAL_SEARCH_MODE:-puct}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
# 1.8, not KD's 0.3: the branching factor differs and alpha scales with it.
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-1.8}"
FORCED_PLAYOUT_K="${FORCED_PLAYOUT_K:-1.0}"

# The two head/readout switches. They change which parameters exist, so they
# must be set at launch and never mid-run: a checkpoint records them and a
# resume rebuilds from that record.
POOLED_READOUT="${POOLED_READOUT:-1}"
REPLY_HEAD="${REPLY_HEAD:-1}"

# Every Nth game searches every move at the full budget. Diversity: a
# wholly-full game is coherent end to end and carries Dirichlet noise and forced
# playouts on every ply rather than a quarter of them. ~3.4x a mixed game, so 25
# is about +10% generation compute.
FULL_SEARCH_EVERY_GAMES="${FULL_SEARCH_EVERY_GAMES:-25}"

# ── Endgame solver ──────────────────────────────────────────────────────────
#
# The node budget is the real cutoff and the depth knob. The cost model decides
# WHICH positions are attempted, replacing --endgame-solver-max-cards: a cap
# cannot attempt a cheap 11-card position or skip a dear 8-card one, and cost is
# driven far more by how much of the board is face down than by how many cards
# remain.
ENDGAME_SOLVER_MAX_NODES="${ENDGAME_SOLVER_MAX_NODES:-4500000}"
ENDGAME_COST_MODEL="${ENDGAME_COST_MODEL:-games/seven_wonders_duel/endgame_cost_model.json}"
SOLVER_FALLBACK_RESEARCH="${SOLVER_FALLBACK_RESEARCH:-1}"

# Seconds are a SAFETY NET, never the budget. A constant generous at one node
# budget binds at another -- a 3-second clock censored 11.3% of solves on the
# 2026-08-18 shakedown and made which positions got a proof depend on machine
# load. Derived at stage 6b from the node budget and this box's measured rate;
# set it explicitly only to override that.
ENDGAME_SOLVER_MAX_SECS="${ENDGAME_SOLVER_MAX_SECS:-}"

# Scheduler shards: one cooperative scheduler thread each, and one solver pool
# each. This is the "generation cores" side of the split, and its right value is
# a MEASUREMENT, not a default -- stage 1 of the sweep in PRE_RETRAIN_PLAN.md
# finds the smallest number that saturates the GPU, and everything above it is
# waste that could be solving instead.
#
# cloud6 ran 12. That is not carried forward as the default, because cloud6 ran
# with the solver off and so had nothing to trade against; on a 20-core box, 12
# generation threads leave 8 for solving and the plan wants the opposite ratio.
# 4 is a placeholder chosen to leave room, and the launch says so out loud.
RUST_SCHEDULER_WORKERS="${RUST_SCHEDULER_WORKERS:-4}"

# Solver threads are PER SHARD: the pool is built once per scheduler loop, so
# the total is this times --rust-scheduler-workers. Derived at stage 6b from the
# core count so the product is deliberate rather than accidental.
SOLVER_THREADS="${SOLVER_THREADS:-}"
GENERATION_THREADS="${GENERATION_THREADS:-}"
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

# Deliberately NOT a common:: helper, and this is the reason: on a box that
# already has a checkout, the common library is sourced from THAT checkout --
# which stage 2 has not updated yet. So anything this script calls before stage 2
# must be defined in this file, or a freshly curl'd script will call a function
# the deployed library does not have. (It did: "command not found", first run
# after the helper was added.)
require_operator_files() {
  # Operator-supplied paths, checked before anything is built. These arrive by
  # scp from another machine, so "not uploaded yet" is the ordinary failure and
  # it should cost seconds rather than rustup, torch and a crate build. Each
  # argument is "NAME=path"; an empty path is skipped, since all are optional.
  local entry name path missing=0
  for entry in "$@"; do
    name="${entry%%=*}"
    path="${entry#*=}"
    [ -z "$path" ] && continue
    if [ -r "$path" ]; then
      ok "$name: $path"
    else
      warn "$name points at $path, which does not exist or is not readable."
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || die "Upload the missing file(s) and re-run; nothing has been built yet."
}

require_operator_files \
  "PRECISION_ARENA_CHECKPOINT=${PRECISION_ARENA_CHECKPOINT:-}" \
  "SWEEP_CHECKPOINT=${SWEEP_CHECKPOINT:-}" \
  "LAUNCH_FLAGS_JSON=${LAUNCH_FLAGS_JSON:-}"

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
# A successful import proves the crate loaded, not that its native dependencies
# work. mimalloc is compile-time (building at all is the gate), but rayon has to
# spawn threads, and the CPU-limit detection that sizes the pack pool has only
# ever run on Windows -- where it takes the fallback path. Both are exercised
# here so a misconfigured slice is visible at provisioning rather than as lost
# throughput 20 hours in.
"$PY" - <<'PYNATIVE' || die "native dependency check failed"
import sys

import seven_wonders_rust as swr

from games.seven_wonders_duel.cloud_preflight import container_limits

limits = container_limits()
effective = int(limits.get("effective_cpus") or 0)
print(f"cpu limits: {limits}")
if effective < 1:
    sys.exit("effective_cpus resolved to 0; pack pool cannot be sized")

# The visible count is what rayon would have taken by default. Reporting the gap
# is the point: a slice that sells 12 of 192 cores would otherwise spawn 192
# packing threads and oversubscribe.
import os

visible = os.cpu_count() or 1
if effective < visible:
    print(f"NOTE: {visible} CPUs visible but only {effective} usable -- "
          f"pack pool will be sized to {effective}, not {visible}")

actual = swr.set_pack_threads(effective)
if actual != effective:
    sys.exit(f"pack pool requested {effective} threads, got {actual}")
print(f"pack pool: {actual} threads")

# Exercise the pool for real: a pool that builds but cannot run work is a
# failure mode an import check cannot see.
from games.seven_wonders_duel.rust_bridge import rust_game_for_self_play

corpus = [rust_game_for_self_play(seed) for seed in range(8)]
seconds = swr.bench_pack_routed(corpus, 4, actual)
if not (seconds > 0.0):
    sys.exit("rayon pack pool produced no measurable work")
print(f"rayon pack pool OK ({seconds*1000:.1f} ms for 4 x 8 rows)")
print("native dependencies verified: mimalloc built, rayon runs, limits detected")
PYNATIVE
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
# ── STAGE 6b: derive the solver's clock and thread split from THIS box ──────
stage 6b "Solver sizing (node rate, safety clock, thread split)"

# Node counts are machine-independent; the RATE is the only machine-specific
# term, so it is measured here rather than assumed. One minute.
NODE_RATE="$("$PY" - <<'PYRATE'
try:
    from games.seven_wonders_duel.endgame_trigger_study import measure_node_rate
    print(int(measure_node_rate()))
except Exception:
    print(0)
PYRATE
)"

if [ "${NODE_RATE:-0}" -gt 0 ]; then
  ok "Solver rate on this box: $((NODE_RATE / 1000000))M nodes/s"
else
  NODE_RATE=1200000
  warn "Could not measure the solver's node rate; assuming a conservative ${NODE_RATE}."
fi

if [ -z "$ENDGAME_SOLVER_MAX_SECS" ]; then
  # (nodes / rate) x 5. Generous enough never to bind, so the node budget stays
  # the cutoff and a decline remains a property of the POSITION rather than of
  # how busy the box was.
  ENDGAME_SOLVER_MAX_SECS="$(( (ENDGAME_SOLVER_MAX_NODES / NODE_RATE + 1) * 5 ))"
  ok "Derived --endgame-solver-max-secs ${ENDGAME_SOLVER_MAX_SECS}s from a ${ENDGAME_SOLVER_MAX_NODES}-node budget."
fi

# One thread solves one position; there is no intra-tree parallelism. So the
# total solver thread count IS the number of concurrent solves, and the split is
# simply: cores that are not feeding the GPU go to solving.
CORES="$(nproc)"
: "${GENERATION_THREADS:=$RUST_SCHEDULER_WORKERS}"
if [ -z "$SOLVER_THREADS" ]; then
  _spare=$(( CORES - GENERATION_THREADS ))
  [ "$_spare" -lt 1 ] && _spare=1
  # Per shard, so divide by the shard count. Rounded down: overshooting
  # oversubscribes, which inflates wall time per solve and pushes a run back
  # toward the clock this stage exists to keep slack.
  SOLVER_THREADS=$(( _spare / GENERATION_THREADS ))
  [ "$SOLVER_THREADS" -lt 1 ] && SOLVER_THREADS=1
fi
_total_solver=$(( SOLVER_THREADS * GENERATION_THREADS ))
if [ -z "${RUST_SCHEDULER_WORKERS_MEASURED:-}" ]; then
  warn "--rust-scheduler-workers=$RUST_SCHEDULER_WORKERS is a PLACEHOLDER, not a"
  warn "measurement. Stage 1 of the sweep finds the smallest value that saturates"
  warn "the GPU; every core above it is one that could have been solving. Read"
  warn "gpu= on the heartbeat and re-launch with the measured number."
fi
ok "Solver threads: $SOLVER_THREADS per shard x $GENERATION_THREADS shards = $_total_solver concurrent solves, on $CORES cores."
if [ "$(( _total_solver + GENERATION_THREADS ))" -gt "$CORES" ]; then
  warn "$_total_solver solver + $GENERATION_THREADS generation threads exceed $CORES cores."
  warn "Solves will run slower in WALL time, which pushes them toward the deadline --"
  warn "and a deadline decline makes which positions got a proof depend on machine load."
fi
stage_done 6b

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
    && _arena_status=0 || _arena_status=$?
  # 3 is the arena's own "ran, and the precisions disagree". Every other
  # non-zero exit (including 1, which is what an uncaught exception gives) means
  # it never reached a verdict.
  if [ "$_arena_status" -eq 3 ]; then
    die "bf16 differs from fp32 beyond its interval — relaunch with PRECISION=fp32."
  elif [ "$_arena_status" -ne 0 ]; then
    die "Precision arena could not run (exit $_arena_status) — that is NOT a verdict on bf16. See the error above; the box is not implicated."
  fi
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
    --precision "$PRECISION" \
    || die "Generation sweep did not complete - see the error above. Nothing was measured, so this says nothing about the settings."
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
# Switches that change which PARAMETERS exist, so they are set at launch and
# never mid-run: the checkpoint records them and a resume rebuilds from that
# record. Passing them as flags rather than baking them in keeps a run that
# wants the old architecture possible.
ARCH_FLAGS=()
[ "$POOLED_READOUT" = "1" ] && ARCH_FLAGS+=(--pooled-readout)
[ "$REPLY_HEAD" = "1" ] && ARCH_FLAGS+=(--reply-head)

SOLVER_FLAGS=()
if [ "$ENDGAME_SOLVER_MAX_NODES" -gt 0 ]; then
  SOLVER_FLAGS+=(
    --endgame-solver-max-nodes "$ENDGAME_SOLVER_MAX_NODES"
    --endgame-solver-max-secs "$ENDGAME_SOLVER_MAX_SECS"
    --solver-threads "$SOLVER_THREADS"
  )
  if [ -n "$ENDGAME_COST_MODEL" ] && [ -f "$REPO_DIR/$ENDGAME_COST_MODEL" ]; then
    SOLVER_FLAGS+=(--endgame-cost-model "$ENDGAME_COST_MODEL")
  elif [ -n "$ENDGAME_COST_MODEL" ]; then
    # Refuse rather than fall back to the card cap: they select different
    # positions, so a silent fallback would produce a run whose solver
    # configuration is not the one anybody chose.
    die "ENDGAME_COST_MODEL=$ENDGAME_COST_MODEL not found under $REPO_DIR."
  fi
  [ "$SOLVER_FALLBACK_RESEARCH" = "1" ] && SOLVER_FLAGS+=(--solver-fallback-research)
fi

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
  --cheap-sims-min "$CHEAP_SIMS" --cheap-sims-max "$CHEAP_SIMS"
  --full-sims-min "$FULL_SIMS" --full-sims-max "$FULL_SIMS"
  --full-search-fraction "$FULL_SEARCH_FRACTION"
  --full-search-every-games "$FULL_SEARCH_EVERY_GAMES"
  --top-k "$TOP_K"
  --age-deal-samples "$AGE_DEAL_SAMPLES"
  --selfplay-search-mode "$SELFPLAY_SEARCH_MODE"
  --cheap-search-mode "$CHEAP_SEARCH_MODE"
  --eval-search-mode "$EVAL_SEARCH_MODE"
  --dirichlet-epsilon "$DIRICHLET_EPSILON"
  --dirichlet-alpha "$DIRICHLET_ALPHA"
  --forced-playout-k "$FORCED_PLAYOUT_K"
  --rust-scheduler-workers "$RUST_SCHEDULER_WORKERS"
  "${ARCH_FLAGS[@]}"
  "${SOLVER_FLAGS[@]}"
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
  --curriculum-anneal-games "$CURRICULUM_ANNEAL_GAMES"
  --draft-prior-games "$DRAFT_PRIOR_GAMES"
  --anchor-games "$ANCHOR_GAMES"
  --anchor-gate-every-promotions "$ANCHOR_GATE_EVERY_PROMOTIONS"
  --pack-threads "$PACK_THREADS"
  --self-anchor-games "$SELF_ANCHOR_GAMES"
  --self-anchor-lag-games "$SELF_ANCHOR_LAG_GAMES"
  --self-anchor-every-games "$SELF_ANCHOR_EVERY_GAMES"
  --intervention-window-games "$INTERVENTION_WINDOW_GAMES"
  --replay-window-cap-games "$REPLAY_WINDOW_CAP_GAMES"
  --example-cache-gb "$EXAMPLE_CACHE_GB"
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
