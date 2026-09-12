#!/usr/bin/env bash
# =============================================================================
# setup_cloud_7wd.sh — first-login setup + launch for 7 Wonders Duel training.
#
# Brings a fresh Linux/CUDA box to "training launched", in one command:
#   1. Rust toolchain (rustup >= 1.85)
#   2. clone (or update) the repo at ~/boardgame-ai, then hand over to the
#      updated copy of this script if the pull replaced it
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
# Self-updating: stage 2 pulls, and if the pull changed this script or the
# common library, it re-executes the new copy rather than continuing on the code
# bash read at startup. Without that, the first re-run after any change to the
# launcher configures the run from the OLD script while recording the NEW
# commit in the manifest -- which is how a 200k-game run was started on a stale
# command line, missing three flags, with nothing failing to say so.
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
#   REPO_BRANCH=<branch>  clone/checkout this branch instead of the default.
#                   REQUIRED when the code being launched is not on main: a
#                   plain clone lands on main, the sentinel check still passes
#                   because those files exist there too, and the box launches
#                   without the flags you think you are running.
#   RUN_DIR_REL=runs/seven_wonders_duel/cloud
#   VERBOSE_STAGES=0  set 1 to put the noisy stages back on the terminal. By
#                   default pip, the crate build, the equivalence suite and the
#                   plumbing smoke write to $RUN_DIR/setup/*.log so the stage
#                   banners stay readable; a failing stage prints its last 40
#                   lines either way.
#   SETUP_LOG=$HOME/setup_cloud_7wd.log  transcript of this whole script,
#                   appended per invocation and copied to
#                   $RUN_DIR/setup/setup.log on exit. It starts outside the repo
#                   because `git clone` refuses a target directory that already
#                   has files in it.
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
#   SWEEP_SIMS_DIVISOR=4  run each sweep point at 1/N of the run's simulations,
#                   with the solver's node budget divided by the same N. Buys
#                   grid points with fidelity: the search algorithm and the
#                   solver's share of occupancy survive, absolute games/hour
#                   does not. Set 1 to measure the run's own search exactly.
#   SWEEP_STAGE_A_INFLIGHT=1 / SWEEP_STAGE_A_WAIT_MS=0  where the staged sweep
#                   pins the batching axes while it ranks geometry. Both must be
#                   values that also appear in the corresponding axis.
#   SWEEP_GENERATION_GAMES  games per sweep point. DERIVED from SWEEP_SLOTS_CSV
#                         as 3x the largest slot count, because a point cannot
#                         hold more games live than it is given: 200 games
#                         against 512 slots measured 200 slots wearing a 512
#                         label. Override only upward.
#   SWEEP_REPETITIONS=1
#   SKIP_SWEEPS=0   set 1 to launch on defaults rather than this box
#   SWEEP_INFERENCE_WAIT_CSV  evaluator coalescing waits in ms (default 0,1,2).
#                         0 is a real point, not "off": the worker always merges
#                         what is already queued, and a positive wait only buys
#                         width by blocking for arrivals that have not happened.
#                         It can only pay where there is more than one shard to
#                         merge across, so it is read with --workers, and the
#                         sweep drops single-shard points that carry one.
#   SWEEP_SOLVER_THREADS_CSV  TOTAL solver threads to sweep (e.g. "0,4,8,16").
#                   The generation/solver core split, as an axis. Unset keeps
#                   the previous behaviour of measuring one fixed split.
#   ALLOW_UNMEASURED_LAUNCH=0  set 1 to launch on defaults even though this box
#                   has an unsourced measured_env.sh
#
#   Workstream switches -- ALL default off, and none was reachable before:
#   SLOT_EMBEDDING=0        W1 learned tableau positions
#   GRAPH_MODULE=0          W2 graph-aware tableau encoding
#     GRAPH_LAYERS / GRAPH_BASES / GRAPH_ALPHA
#   SWD_CONTROL_FEATURES=1  W3 control channels (on by default; pinned here)
#   HIERARCHICAL_VALUE=0 HIER_VALUE_WEIGHT=0 HIER_VALUE_DETACH=1   W4
#   ACTION_RESIDUAL=0 ACTION_EXPOSES=0 ACTION_POLICY_WEIGHT=0      W5
#   SPECIALISTS=""          W7, name:share:LAMBDA, e.g. "science:0.15:3".
#                         Lambda 3 is MEASURED, not chosen: pursuit peaks there
#                         and declines above. Needs HIERARCHICAL_VALUE=1 -- the
#                         bias reads `hier_joint7`, and a net without that head
#                         makes every biased leaf a hard error, not an unbiased
#                         search.
#     SPECIALIST_BOOTSTRAP_GAMES=0 SPECIALIST_FLOOR_EVERY=5
#     SPECIALIST_REANALYSIS=0
#   DOUBLE_REVEAL_OFFSETS=3  offsets per first-reveal stratum on pure double
#                         card-reveal edges, FULL searches. Double reveals are
#                         54.5% of ALL forced chance children, and forced rows
#                         were 35.5% of every network row -- the largest single
#                         throughput lever in the search. Stratified, not
#                         truncated: probability mass is preserved. 0 restores
#                         exhaustive expansion.
#     REANALYSIS_BACKEND=rust_coalesced REANALYSIS_SLOTS=256
#                         Pinned rather than inherited: both were measured on a
#                         LAPTOP 3070 (2539 -> 45 ms/position), and a default
#                         nobody chose on a rented box is the --train-steps
#                         mistake again. 256 concurrent searches is a memory
#                         choice; lower it if the box is tight.
#                   SPECIALISTS requires HIERARCHICAL_VALUE=1: the leaf bias
#                   reads W4's outlook head.
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
  COMMON_SH="$SCRIPT_DIR/setup_cloud_common.sh"
  # shellcheck source=setup_cloud_common.sh
  source "$COMMON_SH"
elif [ -f "$REPO_DIR/setup_cloud_common.sh" ]; then
  COMMON_SH="$REPO_DIR/setup_cloud_common.sh"
  # shellcheck source=setup_cloud_common.sh
  source "$COMMON_SH"
else
  curl -fsSL "https://raw.githubusercontent.com/joelbruns9/boardgame-ai/main/setup_cloud_common.sh" \
    -o /tmp/setup_cloud_common.sh || {
      echo "[FATAL] could not fetch setup_cloud_common.sh" >&2; exit 1; }
  COMMON_SH=/tmp/setup_cloud_common.sh
  # shellcheck source=/dev/null
  source "$COMMON_SH"
fi

RUN_DIR_REL="${RUN_DIR_REL:-runs/seven_wonders_duel/cloud}"

# ── Record this invocation ───────────────────────────────────────────────────
# Every stage banner, warning and derived number used to exist only in the
# operator's scrollback: nothing on disk said which stages ran, what stage 6b
# measured, or what stage 10 launched. A run whose setup is unreproducible is
# one whose configuration cannot be audited afterwards.
#
# The log starts outside the repo because $REPO_DIR may not exist yet, and
# `git clone` refuses a non-empty target -- creating $RUN_DIR first would break
# the clone this file is trying to record. It is copied into the run directory
# on exit, once that directory is real.
SETUP_LOG="${SETUP_LOG:-$HOME/setup_cloud_7wd.log}"
if [ "${SETUP_LOGGING:-0}" != "1" ]; then
  export SETUP_LOGGING=1 SETUP_LOG
  mkdir -p "$(dirname "$SETUP_LOG")"
  printf '\n===== setup invocation %s =====\n' "$(date -Is)" >>"$SETUP_LOG"
  exec > >(tee -a "$SETUP_LOG") 2>&1
fi
setup::copy_log() {
  local dest="$REPO_DIR/$RUN_DIR_REL/setup/setup.log"
  if [ ! -d "$REPO_DIR/$RUN_DIR_REL" ]; then
    return 0   # setup died before the clone; the copy at $SETUP_LOG is all there is
  fi
  # tee writes asynchronously, so the last lines may still be in flight when the
  # shell exits. A short pause costs nothing at the end of a multi-minute setup
  # and keeps the final stage banner in the copied log.
  sleep 1
  mkdir -p "$(dirname "$dest")"
  cp "$SETUP_LOG" "$dest" 2>/dev/null || true
}
trap setup::copy_log EXIT

# The self-update trap: this script sources setup_cloud_common.sh at its top but
# does not `git pull` until stage 2, so the FIRST re-run after a code change
# pulls the new code and then keeps executing the old copy bash already read.
# That is not hypothetical -- it launched a 200k-game run on a stale command
# line whose manifest recorded the new commit. Checksum both files before the
# pull; if the pull moved either, hand over to the updated copy.
#
# The sum covers what bash actually READ -- this file and whichever copy of the
# common library was sourced -- so it is comparable against the repo's copies
# after the pull whether this was launched from the checkout or curled onto a
# bare box.
SETUP_SELF_SUM="$(common::self_checksum "${BASH_SOURCE[0]}" "$COMMON_SH")"
SETUP_ARGV=("$@")

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

# ── Training-side parameters cloud6 tuned ───────────────────────────────────
#
# These were absent until 2026-08-20, so a launch took the parser's defaults and
# trained a materially different run from cloud6 without anyone choosing to:
# weight decay 0.0001 against 0.5, a replay-window coefficient of 16 against
# 1,000, no value bootstrap, no minimum buffer, and a lower temperature floor --
# i.e. LESS exploration, which is the opposite of what this run wants.
#
# Not "cloud6 was right about all of these", but "cloud6 chose them and nothing
# has since argued otherwise", which is a better starting point than argparse.
WEIGHT_DECAY="${WEIGHT_DECAY:-0.5}"
VALUE_BOOTSTRAP="${VALUE_BOOTSTRAP:-0.5}"
MIN_BUFFER_POSITIONS="${MIN_BUFFER_POSITIONS:-200000}"
REPLAY_WINDOW_COEFFICIENT="${REPLAY_WINDOW_COEFFICIENT:-1000}"
REPLAY_WINDOW_EXPONENT="${REPLAY_WINDOW_EXPONENT:-0.6}"
# Temperature is a DIVERSITY control: the floor is how random move selection
# stays once annealing finishes, and the anneal length is how long it takes to
# get there. Both defaults are tighter than cloud6's, which narrows exploration.
TEMPERATURE_FLOOR="${TEMPERATURE_FLOOR:-0.35}"
TEMPERATURE_ANNEAL_MOVES="${TEMPERATURE_ANNEAL_MOVES:-30}"
CHEAP_DOUBLE_REVEAL_OFFSETS="${CHEAP_DOUBLE_REVEAL_OFFSETS:-3}"
# The same cap on FULL searches, which was gated off for years because those
# carry the training targets. CHANCE_ENUMERATION_PLAN.md Step 2 measured the
# approximation over 600 searches and found X=3 dominates X=2 on every quality
# metric (Q MAE 1.6e-4, action disagreement 4.8%, regret 0.007); its verdict is
# that "approximation quality is not the blocker".
DOUBLE_REVEAL_OFFSETS="${DOUBLE_REVEAL_OFFSETS:-3}"
GATE_SIMS="${GATE_SIMS:-64}"
OPPONENT_FRACTION="${OPPONENT_FRACTION:-0}"

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
# 4.5M was a LAPTOP number and leaves a rented box almost idle. Measured on the
# cloud corpus at the shipped margin, a 4.5M budget costs 2.26M solver nodes per
# game; at ~0.33 games/s that is 0.75M nodes/s of demand against roughly 36M
# available from 12 solver threads -- about 2% utilisation, on the subsystem this
# run is built around.
#
# 40M is where the same corpus lands (44.8M nodes/game unfiltered), which puts
# the pool in a sensible range and makes 10- and 11-card positions affordable
# rather than mostly declined. Raising it also widens the cost model's own
# trigger, since it admits anything whose prediction plus margin fits the
# budget -- so this buys DEPTH, not merely a longer leash on the same positions.
#
# The clock follows automatically: stage 6b derives max_secs from this number.
ENDGAME_SOLVER_MAX_NODES="${ENDGAME_SOLVER_MAX_NODES:-320000000}"
# The ATTEMPT BAR, separate from the timeout above.
#
# Measured on cloud2 iteration 96, where one number served both: 8,013 solves
# attempted, 7,754 answered, and 45.9% of ALL solver nodes went to the 259 that
# answered nothing -- each of which spent the full 40M before giving up. A
# decline costs exactly the timeout, so the timeout is the price of being wrong
# about a position and the bar is what decides how often that happens.
#
# 0 would keep them equal, which is what every run before the split did. They
# are deliberately 8x apart instead, and BOTH numbers move together or not at
# all -- raising the timeout alone widens admission by the same factor, which is
# the coupling the split exists to break.
#
# 40M bar / 320M timeout, priced on `solver_corpus.json` at cloud2's measured
# rate and 12 solver threads:
#
#     bar   timeout  proofs  nodes/iter  wasted   stall
#     40M       40M   7,636      16.60B   39.3%     47s   <- one number, before
#     40M      320M   7,786      28.49B   14.6%    373s   <- here
#     40M     1280M   7,798      32.05B    4.0%   1494s
#
# 40M is the widest bar this corpus can price: cloud2 ran there, so anything it
# refused is ABSENT from the corpus rather than recorded as expensive. Narrowing
# the bar is possible and costs hard proofs -- it filters on predicted cost, so
# it drops the expensive positions first, which are the ones worth proving.
#
# The 373s stall is accepted knowingly. The scheduler cannot end an iteration
# while a game is parked on a solve, but across 97 cloud2 iterations the drain
# tail below 25% occupancy was already a median 16.5% of generation wall (~730s)
# with the solver contributing 3s of post-batch idle. A 373s solve lands inside
# a window the run is already idling through. `solver_sizing` flags it anyway --
# it exceeds half the drain -- and that flag is why this is a decision rather
# than a default.
#
# The effective bar is this divided by 10^margin_decades (0.4 in the shipped
# model), so 40M here is really a 15.9M predicted-node bar.
ENDGAME_SOLVER_ATTEMPT_NODES="${ENDGAME_SOLVER_ATTEMPT_NODES:-40000000}"
# A parked slot's --rust-slots token goes back to the pool for the duration of
# the solve. ON here, where phase_d defaults it off: it shipped behind a flag and
# nobody flipped it, so until now a slot parked on a solve held its token AND
# counted toward `active_count`, which divides the batch cap and narrows the row
# allowance of every slot that IS working.
#
# Cheap to turn on because it cannot change what is learned: records are
# byte-identical either way, and `test_excluding_parked_slots_changes_no_record`
# is the gate. It is a throughput change and nothing else.
#
# It DOES change what --rust-slots means -- concurrent games becomes concurrent
# SEARCHING games -- so slot counts are not comparable across the two. That is
# why it is set BEFORE stage 8b rather than after: the sweep must tune the slot
# axis under the regime the run will use, or its optimum belongs to the other one.
EXCLUDE_PARKED_FROM_BUDGET="${EXCLUDE_PARKED_FROM_BUDGET:-1}"
# How much of the box's measured solver capacity the caps are allowed to commit.
#
# Not 100%: the corpus that prices the workload was built at one net's strength,
# and a run reaches different endgames as it improves. Solves also land unevenly
# across an iteration. The headroom is for both -- sized to the last node of a
# measured capacity, the first iteration whose endgames run rich is a late one.
SOLVER_TARGET_SHARE="${SOLVER_TARGET_SHARE:-0.80}"
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
# Leaf batching, adopted 2026-08-20.
#
# LEAF_BATCH must equal `ADVISOR_LEAF_BATCH` in advisor_adapter.py: the gate
# certifies the advisor, and if the two batch differently they run different
# algorithms at the root. test_advisor_leaf_batch pins the pair.
#
# 6 rather than 8 follows Kingdomino, whose notes record 6 as the best known
# quality setting with degradation "around 8". The paired A/B here put 8 at
# 0.482 [0.434, 0.531] against 1 -- no detectable harm, but too loose a bound
# to adopt 8 on, and 6 sits inside it.
# Resume a run whose checkout has moved. OFF by default and warned about
# loudly: the guard exists because a pull landing mid-run otherwise splits the
# run across two engines with nothing recording that it happened. Setting this
# accepts that, so the rows before and after are not interchangeable.
#
# It changes ONLY the code-identity refusal. Weights, buffer, games ledger and
# every schedule position carry on exactly as a normal resume.
ALLOW_RESUME_CODE_DRIFT="${ALLOW_RESUME_CODE_DRIFT:-0}"
# 1, not 6. Leaf batching and the cross-shard coalescer exist to fill the same
# forward pass, and the coalescer -- which postdates this default -- fills it
# from INDEPENDENT slots, exactly, with no algorithmic change. Within-tree
# batching is the approximate version of the same thing: it needs virtual loss
# at a PUCT root, and on full moves the root's visit distribution IS the policy
# target. It was also never swept, unlike RUST_SLOTS. Measured on a laptop 3070:
# --leaf-batch 6 delivered a realized wave width of 3.33 with 75,643 conflict
# cuts, while batch size came overwhelmingly from slot count.
#
# Fill the GPU with SLOTS (swept) rather than with leaf batches (not swept).
LEAF_BATCH="${LEAF_BATCH:-1}"
# Kept at 1 even so: it is INERT for generation at LEAF_BATCH=1 (Rust gates it
# on `leaf_batch > 1`), and EVAL_LEAF_BATCH below is refused without it, because
# evaluation runs a PUCT root. Setting it to 0 makes the launch invalid five
# stages after the decision.
VIRTUAL_LOSS_ROOT="${VIRTUAL_LOSS_ROOT:-1}"
# The cheap path batches under conflict-free waves instead: exact rather than a
# virtual-loss approximation. 16 matches top_k so round one is never the
# limiter, though sequential halving caps the realized mean near 2.6 regardless.
WAVE_FLAGS="${WAVE_FLAGS:-1}"
# Derived from WAVE_FLAGS, not fixed. A cheap leaf batch above 1 REQUIRES the
# cheap wave flags -- without them the conflict-free rule cuts every wave to
# width 1 and Phase D refuses the combination as inert. Pinning 16 here meant
# that turning waves off produced a config the run rejects at launch, five
# stages after the decision, which is how a relaunch died twice.
if [ "$WAVE_FLAGS" = "1" ]; then
  CHEAP_LEAF_BATCH="${CHEAP_LEAF_BATCH:-16}"
else
  CHEAP_LEAF_BATCH="${CHEAP_LEAF_BATCH:-0}"
fi
# Evaluation matches the ADVISOR, not training. The gate certifies the advisor,
# so they should run the same search -- and a leaf batch is a fraction of a
# budget, so sharing training's number across three budgets that differ by an
# order of magnitude would not make the searches alike. Pinned against
# ADVISOR_LEAF_BATCH by test_advisor_leaf_batch.
EVAL_LEAF_BATCH="${EVAL_LEAF_BATCH:-16}"
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
# Provenance for pass 2. `sweep_launch_env.py` exports this; nothing else does.
# `SKIP_SWEEPS` cannot serve the purpose -- an operator sets that by hand to
# skip measuring altogether, which is exactly the case the guard below must
# catch.
SWEEP_MEASURED="${SWEEP_MEASURED:-0}"
SWEEP_MEASURED_FROM="${SWEEP_MEASURED_FROM:-}"
# Set 1 to launch on defaults even though this box has an unsourced sweep. The
# guard exists because "I forgot to source measured_env.sh" and "I chose the
# defaults" produce identical command lines otherwise.
ALLOW_UNMEASURED_LAUNCH="${ALLOW_UNMEASURED_LAUNCH:-0}"

# ── What stage 8b's sweep is allowed to cost ────────────────────────────────
#
# The sweep now runs the RUN's search (`--emit-config` plus
# `--config-from-manifest`) instead of PhaseDConfig's laptop defaults. That was
# the point -- simulations per move set the leaf arrival rate, which is what the
# slot and worker axes act on -- and it made each grid point cost roughly what
# an iteration of the run costs. These two knobs buy the box hours back, and
# each one gives up something that can be stated.
#
# SWEEP_SIMS_DIVISOR runs every point at 1/N of the run's simulations, with the
# solver's node budget divided by the same N so the solver keeps its share of
# slot occupancy (measured per point as `parked_slot_fraction`). What survives:
# the search algorithm, the cheap/full mix, top_k, the chance fan-out. What does
# not: absolute throughput. games/hour from a divided sweep is about N times the
# run's rate and is not a prediction of anything; the RATIOS between points are
# the result. 1 measures the run's own search, at four times the cost.
SWEEP_SIMS_DIVISOR="${SWEEP_SIMS_DIVISOR:-4}"
# Where stage A sits on the stage-B axes while it ranks geometry. Both must
# appear in SWEEP_INFLIGHT_CSV / SWEEP_INFERENCE_WAIT_CSV, which the driver
# enforces: stage B then re-measures stage A's winning point, and the two
# stages can be compared instead of merely concatenated.
#
# The wait pin is the one with a real cost. The coalescing wait is what recovers
# cross-shard fragmentation, so at 0 a high shard count looks worse than it is,
# and stage A is where the shard count is decided. Raise it to the wait the run
# intends if the shard count is the decision you care about; leave it at 0 to
# rank shards on their own merits.
SWEEP_STAGE_A_INFLIGHT="${SWEEP_STAGE_A_INFLIGHT:-1}"
SWEEP_STAGE_A_WAIT_MS="${SWEEP_STAGE_A_WAIT_MS:-0}"

# ── Workstream switches (W1/W2/W3/W4/W5/W7) ─────────────────────────────────
#
# Every one of these defaults OFF in `phase_d`, and until now none was reachable
# from this launcher -- so a run that intended to carry the new architecture
# would have carried none of it, silently, while the manifest recorded a commit
# that contained all of it.
#
# W3 is the exception: `encoder` reads SWD_CONTROL_FEATURES and defaults it on.
# It is pinned explicitly here anyway, so the run records a decision rather than
# inheriting a default that could change.
SLOT_EMBEDDING="${SLOT_EMBEDDING:-0}"          # W1
GRAPH_MODULE="${GRAPH_MODULE:-0}"              # W2
GRAPH_LAYERS="${GRAPH_LAYERS:-}"
GRAPH_BASES="${GRAPH_BASES:-}"
GRAPH_ALPHA="${GRAPH_ALPHA:-}"
SWD_CONTROL_FEATURES="${SWD_CONTROL_FEATURES:-1}"   # W3
export SWD_CONTROL_FEATURES
HIERARCHICAL_VALUE="${HIERARCHICAL_VALUE:-0}"  # W4
HIER_VALUE_WEIGHT="${HIER_VALUE_WEIGHT:-0}"
HIER_VALUE_DETACH="${HIER_VALUE_DETACH:-1}"
ACTION_RESIDUAL="${ACTION_RESIDUAL:-0}"        # W5
ACTION_EXPOSES="${ACTION_EXPOSES:-0}"
ACTION_POLICY_WEIGHT="${ACTION_POLICY_WEIGHT:-0}"
# W7. Empty is HOF-only league play with no biased search, which is what every
# run before this one did.
SPECIALISTS="${SPECIALISTS:-}"
SPECIALIST_BOOTSTRAP_GAMES="${SPECIALIST_BOOTSTRAP_GAMES:-0}"
SPECIALIST_FLOOR_EVERY="${SPECIALIST_FLOOR_EVERY:-5}"
SPECIALIST_REANALYSIS="${SPECIALIST_REANALYSIS:-0}"
# Only read when SPECIALIST_REANALYSIS=1. Named here anyway so the launch line
# records what the run used instead of whatever the parser default was that day.
REANALYSIS_BACKEND="${REANALYSIS_BACKEND:-rust_coalesced}"
REANALYSIS_SLOTS="${REANALYSIS_SLOTS:-256}"
# Scheduler geometry. Empty meant "let the parser decide", and the parser's
# defaults are LAPTOP scale -- 16 slots against cloud6's 256, a 256-row global
# batch against 2,048. On a rented GPU that is not a conservative default, it is
# an underfed one: the box spends its money waiting for 16 games to produce
# enough leaves to make a batch worth submitting.
#
# cloud6's measured values are the defaults instead. The sweep replaces them
# (stage 8b writes sweeps/measured_env.sh; source it and re-run), which is the
# documented path -- but an operator who skips the sweep now gets a cloud-shaped
# configuration rather than a laptop one.
GATE_SLOTS="${GATE_SLOTS:-144}"
RUST_SLOTS="${RUST_SLOTS:-256}"
RUST_GLOBAL_BATCH_CAP="${RUST_GLOBAL_BATCH_CAP:-2048}"
RUST_MAX_INFLIGHT_BATCHES="${RUST_MAX_INFLIGHT_BATCHES:-1}"
GATE_GLOBAL_BATCH_CAP="${GATE_GLOBAL_BATCH_CAP:-1024}"
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
common::reexec_if_updated \
  "$REPO_DIR/setup_cloud_7wd.sh" "$REPO_DIR/setup_cloud_common.sh" \
  "$SETUP_SELF_SUM" ${SETUP_ARGV+"${SETUP_ARGV[@]}"}
stage_done 2

stage 3 "Python dependencies (cu128 torch first)"
common::quietly "$REPO_DIR/$RUN_DIR_REL/setup/python_deps.log" "pip install" -- \
  common::python_deps
stage_done 3

stage 4 "Build seven_wonders_rust"
common::quietly "$REPO_DIR/$RUN_DIR_REL/setup/build_crate.log" "maturin build" -- \
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
  --specialists "$SPECIALISTS" \
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
# One thread solves one position; there is no intra-tree parallelism. So the
# total solver thread count IS the number of concurrent solves, and the split is
# simply: cores that are not feeding the GPU go to solving.
# PHYSICAL cores, not $(nproc). The solver is compute-bound alpha-beta, which
# scales poorly across SMT siblings -- measured 4.37x from 16 logical CPUs, a
# ceiling attributed to all-core clock and SMT rather than bandwidth. Sizing to
# the logical count therefore buys contention rather than solves: on a 16-core /
# 32-thread part it would put 28 solver threads on 16 cores AND stay silent,
# since the oversubscription check compares against the same inflated number.
_physical_cores() {
  local n=""
  if command -v lscpu >/dev/null 2>&1; then
    n="$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
  fi
  if [ -z "$n" ] || [ "$n" -lt 1 ] 2>/dev/null; then
    n="$(awk -F: '/^core id/{ids[$2]=1} /^physical id/{pk[$2]=1}
          END{c=0; for (i in ids) c++; p=0; for (i in pk) p++;
              print (c>0 && p>0) ? c*p : 0}' /proc/cpuinfo 2>/dev/null)"
  fi
  if [ -z "$n" ] || [ "$n" -lt 1 ] 2>/dev/null; then n="$(nproc)"; fi
  echo "$n"
}
CORES="$(_physical_cores)"
_LOGICAL="$(nproc)"
if [ "$CORES" -lt "$_LOGICAL" ]; then
  ok "Sizing to $CORES physical cores ($_LOGICAL logical); SMT siblings are left as headroom."
fi
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
# The count the rate is measured at: the run's TOTAL concurrent solves. Measuring
# at any other number would price a contention level this run never reaches.
SOLVER_RATE_THREADS="${SOLVER_RATE_THREADS:-$_total_solver}"

# THE THREAD SPLIT FIRST, because the rate below is measured AT it.
#
# This used to run after the rate measurement, which was fine while that
# measurement was single-threaded and the count did not matter. It measures
# contention now, so it needs to know how many threads to contend.

# CONTENDED, not single-thread.
#
# `measure_node_rate` returns "this machine's single-thread solver rate", and
# says so. A run does not solve that way: it runs --solver-threads x
# --rust-scheduler-workers of them beside the generation shards, and the
# per-thread rate falls under that contention. The re-solve study measured
# 857,015 nodes/s/thread across 8 threads. Sizing a node budget off the
# uncontended figure overstates this box's capacity by whatever contention costs,
# and the budget is what decides the solver's caps at stage 8c.
#
# Measured at the thread count this run will actually use, on one shared set of
# real Age III endgames so the curve is a property of the box rather than of
# which positions each point drew.
NODE_RATE="$("$PY" - "$SOLVER_RATE_THREADS" <<'PYRATE'
import sys
try:
    from games.seven_wonders_duel.endgame_trigger_study import (
        measure_node_rate_contended,
    )
    wanted = max(1, int(sys.argv[1]))
    out = measure_node_rate_contended(wanted)
    # Per-thread, which is what `budget = threads x wall x rate` multiplies.
    print(int(out["nodes_per_second_per_thread"]))
    print(
        f"  {out['threads']} threads: "
        f"{out['nodes_per_second_per_thread']:,.0f} nodes/s/thread, "
        f"{out['nodes_per_second_total']:,.0f} total, "
        f"parallel efficiency {out['parallel_efficiency']:.2f}",
        file=sys.stderr,
    )
except Exception as exc:
    print(0)
    print(f"  rate measurement failed: {exc}", file=sys.stderr)
PYRATE
)"
NODE_RATE="$(printf '%s' "$NODE_RATE" | head -1)"

if [ "${NODE_RATE:-0}" -gt 0 ]; then
  ok "Solver rate on this box: $((NODE_RATE / 1000))k nodes/s/thread at $SOLVER_RATE_THREADS threads"
else
  NODE_RATE=1200000
  warn "Could not measure the solver's node rate; assuming a conservative ${NODE_RATE}."
  warn "Stage 8c will size the solver caps against a rate nobody measured."
fi

if [ -z "$ENDGAME_SOLVER_MAX_SECS" ]; then
  # (nodes / rate) x 5. Generous enough never to bind, so the node budget stays
  # the cutoff and a decline remains a property of the POSITION rather than of
  # how busy the box was.
  ENDGAME_SOLVER_MAX_SECS="$(( (ENDGAME_SOLVER_MAX_NODES / NODE_RATE + 1) * 5 ))"
  ok "Derived --endgame-solver-max-secs ${ENDGAME_SOLVER_MAX_SECS}s from a ${ENDGAME_SOLVER_MAX_NODES}-node budget."
fi

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
  common::quietly "$REPO_DIR/$RUN_DIR_REL/setup/equivalence.log" "equivalence suite" -- \
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

# ── The run's command line, assembled BEFORE the sweep that measures it ─────
#
# This used to live inside stage 10, after the sweeps. That order made stage 8b
# structurally unable to measure the run: `f4_phase_d_sweep
# --config-from-manifest` takes the RUN's architecture and search budget from a
# manifest, the run's manifest does not exist until the run starts, and the only
# other description of the run was the flag array sixty lines further down this
# file. So the sweep fell back to `PhaseDConfig`'s defaults for everything the
# launcher did not name -- Gumbel at 128 full simulations, to configure a PUCT
# run at 1600.
#
# `phase_d --emit-config` closes that by building the config from these flags
# and writing it in the manifest shape. Which means the flags have to be
# assembled first, and that is all this move is.
#
# What stays in stage 10 is everything that depends on the sweep's OUTPUT: the
# pass-2 guard, TUNED_FLAGS and GATE_TUNED_FLAGS. The split is exactly "does
# this come from the operator's choices, or from this box's measurement".

read -r -a LADDER_RUNGS <<< "$GATE_LADDER"

# W7b ships present-but-disabled: detection reports either way, and enabling
# the response is a deliberate choice made at launch, not mid-run.
# Switches that change which PARAMETERS exist, so they are set at launch and
# never mid-run: the checkpoint records them and a resume rebuilds from that
# record. Passing them as flags rather than baking them in keeps a run that
# wants the old architecture possible.
# Boolean flags as an array: passing `--virtual-loss-root 1` would be a parser
# error, and passing nothing at all is how the launcher lost --weight-decay.
LEAF_BATCH_FLAGS=()
[ "$ALLOW_RESUME_CODE_DRIFT" = "1" ] && LEAF_BATCH_FLAGS+=(--allow-resume-code-drift)
[ "$VIRTUAL_LOSS_ROOT" = "1" ] && LEAF_BATCH_FLAGS+=(--virtual-loss-root)
# CHEAP-path waves, not global. The two batching mechanisms are mutually
# exclusive: virtual loss discourages collisions so width approaches
# --leaf-batch, conflict-free waves forbid them and cut the wave short. Under
# the PUCT root of a full move that collapses width to ~1.19, so the global flag
# would silently disable the batching on the path carrying 1600 simulations.
# Measured: wave 1.88 per-path against 1.56 global, same leaf batch.
[ "$WAVE_FLAGS" = "1" ] && LEAF_BATCH_FLAGS+=(--cheap-conflict-free-waves --cheap-round-robin-candidates)

ARCH_FLAGS=()
[ "$POOLED_READOUT" = "1" ] && ARCH_FLAGS+=(--pooled-readout)
[ "$REPLY_HEAD" = "1" ] && ARCH_FLAGS+=(--reply-head)
# W1/W2/W4/W5. Each is a MODEL SHAPE change, so a resume that flips one is
# refused by Phase D's own contract check rather than silently loading a
# checkpoint the weights no longer fit.
[ "$SLOT_EMBEDDING" = "1" ] && ARCH_FLAGS+=(--slot-embedding)
if [ "$GRAPH_MODULE" = "1" ]; then
  ARCH_FLAGS+=(--graph-module)
  [ -n "$GRAPH_LAYERS" ] && ARCH_FLAGS+=(--graph-layers "$GRAPH_LAYERS")
  [ -n "$GRAPH_BASES" ] && ARCH_FLAGS+=(--graph-bases "$GRAPH_BASES")
  [ -n "$GRAPH_ALPHA" ] && ARCH_FLAGS+=(--graph-alpha "$GRAPH_ALPHA")
fi
if [ "$HIERARCHICAL_VALUE" = "1" ]; then
  ARCH_FLAGS+=(--hierarchical-value --hier-value-weight "$HIER_VALUE_WEIGHT")
  [ "$HIER_VALUE_DETACH" = "0" ] && ARCH_FLAGS+=(--no-hierarchical-value-detach)
fi
[ "$ACTION_RESIDUAL" = "1" ] && ARCH_FLAGS+=(--action-residual)
[ "$ACTION_EXPOSES" = "1" ] && ARCH_FLAGS+=(--action-exposes)
ARCH_FLAGS+=(--action-policy-weight "$ACTION_POLICY_WEIGHT")

# W7 specialist league. Separate from ARCH_FLAGS because these change the
# TRAINING ARRANGEMENT rather than the model shape: the same weights, played and
# trained differently.
SPECIALIST_FLAGS=()
if [ -n "$SPECIALISTS" ]; then
  # A biased search reads W4's seven-way outlook, and a leaf without one is a
  # hard error rather than a silent zero bias. Phase D refuses the combination
  # at launch, but failing here names the launcher knob rather than the flag.
  if [ "$HIERARCHICAL_VALUE" != "1" ]; then
    die "SPECIALISTS is set but HIERARCHICAL_VALUE=0. The specialist leaf bias
reads W4's outlook head; without it every biased leaf raises. Set
HIERARCHICAL_VALUE=1 and a positive HIER_VALUE_WEIGHT."
  fi
  SPECIALIST_FLAGS+=(
    --specialists "$SPECIALISTS"
    --specialist-bootstrap-games "$SPECIALIST_BOOTSTRAP_GAMES"
    --specialist-floor-every "$SPECIALIST_FLOOR_EVERY"
  )
  if [ "$SPECIALIST_REANALYSIS" = "1" ]; then
    SPECIALIST_FLAGS+=(
      --specialist-reanalysis
      --reanalysis-backend "$REANALYSIS_BACKEND"
      --reanalysis-slots "$REANALYSIS_SLOTS"
    )
  fi
fi

SOLVER_FLAGS=()
if [ "$ENDGAME_SOLVER_MAX_NODES" -gt 0 ]; then
  SOLVER_FLAGS+=(
    --endgame-solver-max-nodes "$ENDGAME_SOLVER_MAX_NODES"
    --endgame-solver-max-secs "$ENDGAME_SOLVER_MAX_SECS"
    --endgame-solver-attempt-nodes "$ENDGAME_SOLVER_ATTEMPT_NODES"
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
  # Inert with the solver off -- nothing parks -- so it rides inside this gate
  # rather than putting a knob on the line that changes nothing.
  [ "$EXCLUDE_PARKED_FROM_BUDGET" = "1" ] &&
    SOLVER_FLAGS+=(--exclude-parked-from-budget)
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
  --weight-decay "$WEIGHT_DECAY"
  --value-bootstrap "$VALUE_BOOTSTRAP"
  --min-buffer-positions "$MIN_BUFFER_POSITIONS"
  --replay-window-coefficient "$REPLAY_WINDOW_COEFFICIENT"
  --replay-window-exponent "$REPLAY_WINDOW_EXPONENT"
  --temperature-floor "$TEMPERATURE_FLOOR"
  --temperature-anneal-moves "$TEMPERATURE_ANNEAL_MOVES"
  --cheap-double-reveal-offsets "$CHEAP_DOUBLE_REVEAL_OFFSETS"
  --double-reveal-offsets "$DOUBLE_REVEAL_OFFSETS"
  --gate-sims "$GATE_SIMS"
  --derive-backend rust
  # Explicit 0, not omitted: the parser DEFAULTS this to 0.15, so leaving it out
  # sends 15% of games against a non-HOF opponent -- which cloud6 turned off and
  # nobody here decided to turn back on. --hof-opponent-fraction is the knob
  # that governs archived opponents.
  --opponent-fraction "$OPPONENT_FRACTION"
  "${ARCH_FLAGS[@]}"
  "${SPECIALIST_FLAGS[@]}"
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
  --leaf-batch "$LEAF_BATCH"
  --cheap-leaf-batch "$CHEAP_LEAF_BATCH"
  --eval-leaf-batch "$EVAL_LEAF_BATCH"
  "${LEAF_BATCH_FLAGS[@]}"
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
  # TUNED_FLAGS and GATE_TUNED_FLAGS are appended at STAGE 10, not set here.
  # They come OUT of the sweep this command is assembled to feed, so they
  # cannot exist yet on the pass that measures them -- and on that pass the
  # geometry left in place is the launcher's own default, which is exactly the
  # baseline `f4_phase_d_sweep` should rank its grid against.
)

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

  # ── THE RUN'S CONFIGURATION, so the sweep measures the run ────────────────
  #
  # `f4_phase_d_sweep --config-from-manifest` exists precisely for this, and its
  # own docstring records what happens without it: "roughly 50 simulations a
  # move instead of the run's measured 522, under a different search algorithm
  # ... the optimum found that way belongs to a machine nobody is running".
  # Simulations per move set the leaf arrival rate, and the leaf arrival rate is
  # what the slot and worker axes act on.
  #
  # The obstacle was never the flag, it was the ORDER: the run's manifest does
  # not exist until the run starts, and the sweep runs first. `--emit-config`
  # builds the config from the launch flags assembled above and writes it in the
  # manifest shape, so the sweep reads the run that is about to be launched
  # rather than a dataclass.
  #
  # Validated on the way out by Phase D itself. Emitting a config Phase D would
  # refuse is worse than emitting none: the sweep would spend box hours ranking
  # a geometry for a run that cannot start.
  #
  # `[@]:3` drops $PY, -m and the module name: the flags, not the interpreter.
  # No `cd` because stage 2's clone left the shell in $REPO_DIR, which is the
  # same thing the sweep invocations below rely on.
  RUN_CONFIG_JSON="$SWEEP_DIR/run_config.json"
  common::quietly "$REPO_DIR/$RUN_DIR_REL/setup/emit_config.log" "emit run config" -- \
    "$PY" -m games.seven_wonders_duel.phase_d \
    --emit-config "$RUN_CONFIG_JSON" "${TRAIN_CMD[@]:3}" \
    || die "Could not build this run's configuration, so the sweep would fall back
to PhaseDConfig's defaults and measure a search this run does not run.
Fix the knobs above; nothing was measured and nothing was launched."
  ok "Run configuration for the sweep: $RUN_CONFIG_JSON"

  # The generation/solver CORE SPLIT, as an axis rather than a constant.
  #
  # Generation and the endgame solver compete for the same physical cores, and
  # the solver runs SYNCHRONOUSLY inside a scheduler shard -- so a thread given
  # to it is a thread taken from leaf production, and the best split is a
  # property of this box's core count, not of the algorithm. Until now this
  # stage passed a single fixed --solver-threads, which measures one split and
  # reports it as the answer.
  #
  # Swept as a TOTAL, because --solver-threads is per shard and the worker count
  # is itself an axis: holding "threads per shard" fixed across worker counts
  # would silently vary the total, and the total is what competes with
  # generation. `f4_phase_d_sweep --solver-threads-total` divides at each point.
  SWEEP_SOLVER_ARGS=()
  if [ "$ENDGAME_SOLVER_MAX_NODES" -gt 0 ]; then
    if [ -n "${SWEEP_SOLVER_THREADS_CSV:-}" ]; then
      SWEEP_SOLVER_ARGS=(--solver-threads-total "$SWEEP_SOLVER_THREADS_CSV")
    elif [ -n "$SOLVER_THREADS" ]; then
      SWEEP_SOLVER_ARGS=(--solver-threads "$SOLVER_THREADS")
    fi
    # THE NODE BUDGET, without which threads measure nothing.
    #
    # `f4_phase_d_sweep` gates solving on `solver_threads > 0 AND
    # solver_max_nodes > 0`, and this stage passed only the first. So every
    # point ran with `solver_wants` refusing every position: the core-split axis
    # measured a solver that never ran, which is the defect THROUGHPUT_LEVERS.md
    # section 3.1 records -- on a run where the solver took 22-37% of generation
    # wall. `rehearse_sweep_laptop.sh` asserts against it; this script
    # reintroduced it.
    #
    # The RUN's budget, not a sweep-specific one: a split measured against a
    # cheaper solver is a split for a run nobody is launching.
    SWEEP_SOLVER_ARGS+=(
      --solver-max-nodes "$ENDGAME_SOLVER_MAX_NODES"
      --solver-max-secs "$ENDGAME_SOLVER_MAX_SECS"
      # THE BAR TOO. The harness falls back to the manifest only for values it
      # was not given, and this stage gives it an explicit timeout -- so passing
      # the timeout alone left the bar defaulting to it, and the split vanished
      # for the sweep. The sweep would then measure admission at the timeout
      # while the run admits at the bar: a solver load no run carries.
      --solver-attempt-nodes "$ENDGAME_SOLVER_ATTEMPT_NODES"
    )
  else
    # The solver is off for this run, so a split has nothing to divide.
    SWEEP_SOLVER_ARGS=(--solver-threads 0)
  fi

  # Games must outnumber the largest slot count, or that point never fills its
  # slots and the slot axis is measured at an occupancy no run has. The default
  # used to be a flat 200 against slot values up to 512 -- so the two largest
  # points on the axis were measuring 200 slots under someone else's label, and
  # `f4_phase_d_sweep` now refuses that outright.
  #
  # Derived from the slot list rather than pinned, so raising SWEEP_SLOTS_CSV
  # cannot silently reintroduce it. 3 games per slot is the steady-state
  # threshold; ramp and drain otherwise dominate and they favour small slot
  # counts.
  # Workers is an AXIS, not the shipped value. Defaulting it to
  # $RUST_SCHEDULER_WORKERS swept ONE point and reported it as the optimum --
  # the same shape as measuring one solver split and calling it the answer.
  # Shard count decides whether the CPU can walk trees fast enough to keep the
  # coalesced batch full, and post-coalescer it no longer fragments batches, so
  # there is no longer a reason to hold it fixed.
  #
  # Centred on the shipped value, so the current setting is always IN the grid:
  # a sweep that cannot return today's configuration cannot tell you it was right.
  SWEEP_WORKERS_DEFAULT="${SWEEP_WORKERS_DEFAULT:-$(
    printf '%s,%s,%s' \
      "$(( RUST_SCHEDULER_WORKERS / 2 > 0 ? RUST_SCHEDULER_WORKERS / 2 : 1 ))" \
      "$RUST_SCHEDULER_WORKERS" \
      "$(( RUST_SCHEDULER_WORKERS * 2 ))"
  )}"

  SWEEP_MAX_SLOTS="$(printf '%s' "${SWEEP_SLOTS_CSV:-128,256,512}" | tr ',' '\n' \
    | sort -n | tail -1)"
  SWEEP_GENERATION_GAMES="${SWEEP_GENERATION_GAMES:-$((SWEEP_MAX_SLOTS * 3))}"
  if [ "$SWEEP_GENERATION_GAMES" -lt "$SWEEP_MAX_SLOTS" ]; then
    die "SWEEP_GENERATION_GAMES=$SWEEP_GENERATION_GAMES cannot fill $SWEEP_MAX_SLOTS slots; the slot axis would measure nothing above the game count."
  fi
  say "Generation sweep: $SWEEP_GENERATION_GAMES games/point against max $SWEEP_MAX_SLOTS slots ($(( SWEEP_GENERATION_GAMES / SWEEP_MAX_SLOTS )) per slot)"

  # ── THE COST OF MEASURING THE RUN, and what is done about it ──────────────
  #
  # Two changes above made each grid point ~10x more expensive: the run's own
  # search instead of PhaseDConfig's defaults, and the run's own solver budget.
  # Both were fixes -- but 6 axes at 3x2x2x3x3 points, twice over, at 1600
  # simulations a move is a sweep that costs more than the run it configures.
  # Two knobs buy that back, and each gives up something SAYABLE:
  #
  # 1. `f4_staged_sweep` replaces the cartesian product with two stages:
  #    geometry (slots x caps x workers x solver split) is ranked first, then
  #    inflight x wait is swept at the winner. 18 + 6 points against 108. The
  #    cost is that stage B cannot reorder stage A; SWEEP_STAGE_A_WAIT_MS is the
  #    knob for the one interaction where that is known to bite (the coalescing
  #    wait rewards high shard counts, and stage A ranks shards without it).
  # 2. SWEEP_SIMS_DIVISOR runs every point at 1/N of the run's simulations, with
  #    the solver's node budget divided by the same N so the solver keeps its
  #    share of slot occupancy. The search ALGORITHM, the cheap/full mix and
  #    top_k all survive; ABSOLUTE throughput does not. games/hour out of a
  #    divided sweep is not the run's rate, and `measured_env.sh` says so.
  #
  # Set SWEEP_SIMS_DIVISOR=1 to measure the run's own search exactly, at four
  # times the box hours.
  #
  # Both harnesses take COMMA-separated axes and an --output DIRECTORY (the
  # staged driver writes phase_d_sweep.json inside, in the same shape).
  # w5_gate_slots_sweep takes space-separated axes and an --output FILE. They
  # are different harnesses; test_setup_cloud arg-parses both invocations so
  # this cannot drift again.
  "$PY" -m games.seven_wonders_duel.f4_staged_sweep \
    --checkpoint "$SWEEP_CHECKPOINT" \
    --output "$SWEEP_DIR/generation" \
    --games "$SWEEP_GENERATION_GAMES" \
    --repetitions "${SWEEP_REPETITIONS:-1}" \
    --config-from-manifest "$RUN_CONFIG_JSON" \
    --sims-divisor "$SWEEP_SIMS_DIVISOR" \
    --slots "${SWEEP_SLOTS_CSV:-128,256,512}" \
    --caps "${SWEEP_CAPS_CSV:-1024,2048}" \
    --inflight "${SWEEP_INFLIGHT_CSV:-1,2}" \
    --workers "${SWEEP_WORKERS_CSV:-$SWEEP_WORKERS_DEFAULT}" \
    --inference-wait-ms "${SWEEP_INFERENCE_WAIT_CSV:-0,1,2}" \
    --stage-a-inflight "$SWEEP_STAGE_A_INFLIGHT" \
    --stage-a-wait-ms "$SWEEP_STAGE_A_WAIT_MS" \
    ${SWEEP_SOLVER_ARGS[@]+"${SWEEP_SOLVER_ARGS[@]}"} \
    --device cuda \
    --precision "$PRECISION" \
    || die "Generation sweep did not complete - see the error above. Nothing was measured, so this says nothing about the settings."
  ok "Generation sweep: $SWEEP_DIR/generation/phase_d_sweep.json"

  # LIVENESS, not configuration. Asserting that solver threads were CONFIGURED
  # is what let the missing node budget go unnoticed: every point reported a
  # split and none of them solved anything. Assert the solver did work.
  if [ "$ENDGAME_SOLVER_MAX_NODES" -gt 0 ]; then
    "$PY" - "$SWEEP_DIR/generation/phase_d_sweep.json" <<'PYSOLVES'       || die "The generation sweep measured a solver that never ran. Its core-split and slot numbers describe a configuration this run will not use."
import json, sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
# THE UNION OF BOTH STAGES, not `summary`.
#
# `summary` is the BATCHING stage's, and that stage pins whatever the geometry
# stage won -- including the solver split. So if the solver-off point wins the
# geometry stage, every row here reads zero and this check would report "no
# point ran with the solver on" about a sweep that measured the split
# thoroughly. The split is measured in stage A; the assertion has to look there.
staged = payload.get("staged")
if staged:
    summary = [row for stage in staged["stages"] for row in stage["summary"]]
else:
    summary = payload["summary"]
on = [row for row in summary if row.get("solver_threads_total", 0) > 0]
if not on:
    print("  no sweep point ran with the solver on; nothing to check")
    raise SystemExit(0)
attempted = sum(row.get("solves_attempted", 0) for row in on)
# PREDICTIONS, not just attempts. Only the cost model produces one, so a nonzero
# count is what says the model was installed rather than the card cap having let
# something through -- and with `max_cards = 0` the card cap admits nothing at
# all, which is how every point came to attempt zero solves while dutifully
# reporting its thread count.
predicted = sum(row.get("solves_with_prediction", 0) for row in on)
if attempted and not predicted:
    print(
        f"  {attempted} solves attempted and NONE carried a cost-model "
        "prediction: the model was not installed in the sweep process",
        file=sys.stderr,
    )
    raise SystemExit(1)
if attempted == 0:
    print(
        f"  {len(on)} points configured solver threads and attempted ZERO "
        "solves",
        file=sys.stderr,
    )
    raise SystemExit(1)
answered = sum(row.get("solves_answered", 0) for row in on)
print(f"  solver LIVE across {len(on)} points: {attempted} attempted, "
      f"{answered} answered, {predicted} with a model prediction")
PYSOLVES
  fi

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

  # ── SOLVER CAPS, sized from this box rather than assumed ──────────────────
  #
  # The last input arrives here and not earlier: the solver's budget is
  # `threads x GENERATION WALL x rate x share`, and the generation wall is what
  # the sweep above just measured. Stage 6b supplied the other two.
  #
  # Everything else was priced once, off the box: `solver_corpus.json` holds the
  # true node cost of every position a real run attempted, and node counts are a
  # property of positions rather than of hardware. So this step solves rather
  # than searches -- no solving happens here at all.
  #
  # Skipped, loudly, when the corpus is absent: sizing the caps off nothing would
  # produce numbers indistinguishable from measured ones.
  SOLVER_CORPUS="${SOLVER_CORPUS:-$REPO_DIR/games/seven_wonders_duel/solver_corpus.json}"
  if [ "$ENDGAME_SOLVER_MAX_NODES" -le 0 ]; then
    # The run turned the solver OFF, and the sweep above measured a geometry
    # without it. Sizing caps anyway would write positive numbers into
    # measured_env.sh, and pass 2 would source them and launch WITH a solver, on
    # a geometry measured without one. An operator who disabled it must not have
    # it handed back by the stage that was supposed to tune it.
    warn "ENDGAME_SOLVER_MAX_NODES=0: the solver is off for this run, so no caps"
    warn "are sized. The sweep measured a geometry without it."
  elif [ ! -f "$SOLVER_CORPUS" ]; then
    warn "No solver corpus at $SOLVER_CORPUS; leaving the solver caps at the"
    warn "launcher's defaults (attempt bar $ENDGAME_SOLVER_ATTEMPT_NODES, timeout"
    warn "$ENDGAME_SOLVER_MAX_NODES). Build one with solver_corpus.py."
  else
    # ── The WINNER's geometry, not stage 6b's guess ──────────────────────────
    #
    # `_total_solver` and `NODE_RATE` came from stage 6b, which ran BEFORE the
    # sweep varied the worker and solver axes. Sizing against them budgets for a
    # thread count `measured_env.sh` will not launch -- a 12-thread allocation
    # followed by a 4-thread winner still receives a 12-thread budget, three
    # times the capacity the run will actually have.
    #
    # The wall is read at the FULL simulation budget for the same reason. The
    # sweep ran at SWEEP_SIMS_DIVISOR, and this file says in three places that a
    # divided sweep's games/hour is not the run's rate -- then used it as one.
    # At the default divisor that supplies roughly a quarter of production
    # generation wall, so the budget comes out a quarter of the truth and
    # affordable candidates are rejected, or setup aborts.
    read -r _WIN_WORKERS _WIN_SOLVER_TOTAL <<<"$("$PY" - "$SWEEP_DIR/generation/phase_d_sweep.json" <<'PYWIN'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
best = (payload.get("staged") or {}).get("winner") or payload["summary"][0]
print(int(best.get("scheduler_workers") or 0),
      int(best.get("solver_threads_total") or 0))
PYWIN
)"
    if [ "${_WIN_SOLVER_TOTAL:-0}" -le 0 ]; then
      warn "The sweep's winner ran no solver threads, so there is no split to"
      warn "size caps against. Leaving the solver caps at the launcher's values."
      _GEN_WALL=0
    else
      say "Sizing against the WINNING geometry: $_WIN_WORKERS shards, $_WIN_SOLVER_TOTAL solver threads"
      # Re-measure the rate at the thread count that won. Contention is what
      # this measures, and it is a function of the thread count -- stage 6b's
      # figure belongs to a different one.
      NODE_RATE_WIN="$("$PY" - "$_WIN_SOLVER_TOTAL" <<'PYRATE2'
import sys
from games.seven_wonders_duel.endgame_trigger_study import measure_node_rate_contended
out = measure_node_rate_contended(max(1, int(sys.argv[1])))
print(int(out["nodes_per_second_per_thread"]))
print(f"  {out['threads']} threads: {out['nodes_per_second_per_thread']:,.0f} "
      f"nodes/s/thread (efficiency {out['parallel_efficiency']:.2f})",
      file=sys.stderr)
PYRATE2
)"
      NODE_RATE_WIN="$(printf '%s' "$NODE_RATE_WIN" | head -1)"
      [ "${NODE_RATE_WIN:-0}" -gt 0 ] || NODE_RATE_WIN="$NODE_RATE"

      # GENERATION WALL AT FULL SIMULATIONS. One confirmation point at the
      # winning geometry with the divisor off -- a measurement, not an
      # extrapolation from a divided one.
      # ONE point, at the winner's exact geometry, divisor off. It costs a
      # full-cost iteration and it is the only honest source of a production
      # generation wall. A failure degrades to the extrapolation below rather
      # than losing the whole setup, which is why this warns and does not die.
      say "Confirmation point at FULL simulations (one point, the winner's geometry)"
      "$PY" -m games.seven_wonders_duel.f4_phase_d_sweep \
        --checkpoint "$SWEEP_CHECKPOINT" \
        --output "$SWEEP_DIR/full_sims" \
        --games "$SWEEP_GENERATION_GAMES" \
        --repetitions 1 --warmup-games 0 \
        --config-from-manifest "$RUN_CONFIG_JSON" \
        --sims-divisor 1 \
        --slots "$_WIN_SLOTS" --caps "$_WIN_CAP" \
        --inflight "$_WIN_INFLIGHT" --workers "$_WIN_WORKERS" \
        --inference-wait-ms "$_WIN_WAIT" \
        --solver-threads-total "$_WIN_SOLVER_TOTAL" \
        --solver-max-nodes "$ENDGAME_SOLVER_MAX_NODES" \
        --solver-attempt-nodes "$ENDGAME_SOLVER_ATTEMPT_NODES" \
        --device cuda --precision "$PRECISION" \
        || warn "The full-simulation confirmation point did not complete; the wall will be EXTRAPOLATED."
      _GEN_WALL="$("$PY" - "$SWEEP_DIR/full_sims/phase_d_sweep.json" "$SWEEP_DIR/generation/phase_d_sweep.json" "$GAMES_PER_ITERATION" "$SWEEP_SIMS_DIVISOR" <<'PYWALL'
import json, pathlib, sys
# Seconds of GENERATION per iteration. Not the whole iteration: solving happens
# during generation, and charging the solver for training time would inflate its
# budget by however long the learner runs.
#
# Preference order: a point measured at the FULL simulation budget, else the
# divided sweep scaled by the divisor -- which is an EXTRAPOLATION and says so,
# because simulations are not the only per-game cost and the scaling is not
# exact.
full = pathlib.Path(sys.argv[1])
games, divisor = int(sys.argv[3]), max(1, int(sys.argv[4]))
if full.is_file():
    payload = json.loads(full.read_text(encoding="utf-8"))
    best = (payload.get("staged") or {}).get("winner") or payload["summary"][0]
    rate = float(best["median_games_per_hour"])
    print(int(games / rate * 3600) if rate > 0 else 0)
    print("measured at the full simulation budget", file=sys.stderr)
else:
    payload = json.loads(open(sys.argv[2], encoding="utf-8").read())
    best = (payload.get("staged") or {}).get("winner") or payload["summary"][0]
    rate = float(best["median_games_per_hour"])
    print(int(games / rate * 3600) * divisor if rate > 0 else 0)
    print(f"EXTRAPOLATED from a {divisor}x divided sweep, not measured",
          file=sys.stderr)
PYWALL
)"
      _GEN_WALL="$(printf '%s' "$_GEN_WALL" | head -1)"
    fi
    if [ "${_GEN_WALL:-0}" -gt 0 ]; then
      "$PY" -m games.seven_wonders_duel.solver_sizing \
        --corpus "$SOLVER_CORPUS" \
        --rate "$NODE_RATE_WIN" \
        --threads "$_WIN_SOLVER_TOTAL" \
        --generation-wall-seconds "$_GEN_WALL" \
        --games "$GAMES_PER_ITERATION" \
        --target-share "$SOLVER_TARGET_SHARE" \
        --output "$SWEEP_DIR/solver_env.sh" \
        || die "Could not size the solver caps from this box's measurements."
      # Appended to the same file pass 2 sources, so there is ONE thing to
      # source and no way to take the geometry without the caps it was
      # measured beside.
      cat "$SWEEP_DIR/solver_env.sh" >> "$SWEEP_DIR/measured_env.sh"
      ok "Solver caps sized: $SWEEP_DIR/solver_env.sh"
    else
      warn "Could not read a generation rate from the sweep; solver caps left at defaults."
    fi
  fi

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
  common::quietly "$REPO_DIR/$RUN_DIR_REL/setup/plumbing_smoke.log" "plumbing smoke" -- \
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
# ── The pass-2 guard: were this box's measured numbers actually applied? ─────
#
# `RUST_SLOTS` and friends carry cloud6 defaults, so pass 2 without sourcing
# `measured_env.sh` launches on those and prints "Measured generation flags",
# which is a lie: they came from a default, not from this box. The two cases
# produce identical command lines, and the run is 24 hours long.
#
# `SWEEP_MEASURED` is exported only by `sweep_launch_env.py`. If a sweep exists
# on this box and that marker is absent, the operator forgot to source it.
if [ "$SWEEP_MEASURED" = "1" ]; then
  ok "Using this box's measured sweep: ${SWEEP_MEASURED_FROM:-unknown}"
elif [ -f "$REPO_DIR/$RUN_DIR_REL/sweeps/measured_env.sh" ]; then
  if [ "$ALLOW_UNMEASURED_LAUNCH" = "1" ]; then
    warn "This box has a measured sweep that was NOT sourced, and"
    warn "ALLOW_UNMEASURED_LAUNCH=1, so the run proceeds on defaults."
  else
    die "This box has a measured sweep that was not sourced, so the launch would
use built-in defaults while reporting them as measured. Run:

  source $REPO_DIR/$RUN_DIR_REL/sweeps/measured_env.sh && bash \$0

Set ALLOW_UNMEASURED_LAUNCH=1 to launch on defaults deliberately."
  fi
fi

TUNED_FLAGS=()
[ -n "$RUST_SLOTS" ] && TUNED_FLAGS+=(--rust-slots "$RUST_SLOTS")
[ -n "$RUST_GLOBAL_BATCH_CAP" ] &&
  TUNED_FLAGS+=(--rust-global-batch-cap "$RUST_GLOBAL_BATCH_CAP")
[ -n "$RUST_MAX_INFLIGHT_BATCHES" ] &&
  TUNED_FLAGS+=(--rust-max-inflight-batches "$RUST_MAX_INFLIGHT_BATCHES")
# The evaluator coalescing wait. Only ever set from a sweep that VARIED it --
# `sweep_launch_env` refuses to emit it otherwise -- so an unset value here
# means the run takes the 0 default, which still merges everything already
# queued. It does NOT mean coalescing is off.
[ -n "${RUST_INFERENCE_WAIT_MS:-}" ] &&
  TUNED_FLAGS+=(--rust-inference-wait-ms "$RUST_INFERENCE_WAIT_MS")
if [ ${#TUNED_FLAGS[@]} -gt 0 ]; then
  if [ "$SWEEP_MEASURED" = "1" ]; then
    ok "Measured generation flags: ${TUNED_FLAGS[*]}"
  else
    warn "Generation flags (NOT measured on this box): ${TUNED_FLAGS[*]}"
  fi
elif [ -n "${LAUNCH_FLAGS_JSON:-}" ]; then
  read -r -a TUNED_FLAGS <<< "$(
    "$PY" -m games.seven_wonders_duel.f4_launch_flags "$LAUNCH_FLAGS_JSON"
  )" || die "Could not translate $LAUNCH_FLAGS_JSON into Phase D flags."
  ok "Measured launch flags: ${TUNED_FLAGS[*]}"
else
  warn "LAUNCH_FLAGS_JSON unset; launching on Phase D defaults rather than this "
  warn "box's measured sweep."
fi

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

# The measured geometry, onto the command assembled before stage 8b. Appended
# rather than interpolated because this is the ONLY part of the launch line that
# comes from the box rather than from the operator, and it has to land after the
# sweep that produced it. Last also means last wins, which is what a measurement
# should do to a default.
TRAIN_CMD+=("${TUNED_FLAGS[@]}" "${GATE_TUNED_FLAGS[@]}")

if [ "$LAUNCH" != "1" ]; then
  warn "LAUNCH=$LAUNCH; verified but not launching. Launch manually with:"
  warn "  cd $REPO_DIR && nohup ${TRAIN_CMD[*]} >> $LOG_FILE 2>&1 &"
  stage_done 10
  ok "Setup complete."
  exit 0
fi

# Validate the assembled command BEFORE detaching.
#
# Every knob combination this script can emit has to be one Phase D accepts, and
# checking that here costs two seconds. Three relaunches died at stage 10
# instead -- after the toolchain, the preflight, the equivalence suite and the
# smoke -- each on a pairing decided in this file and rejected in that one.
log "Validating the assembled training command"
( cd "$REPO_DIR" && "$PY" -m games.seven_wonders_duel.phase_d --validate-config     "${TRAIN_CMD[@]:3}" ) >"$RUN_DIR/setup/validate_config.log" 2>&1   || { warn "The assembled command is not a configuration Phase D accepts:"
       tail -n 5 "$RUN_DIR/setup/validate_config.log" >&2
       die "Fix the knobs above and re-run. Nothing was launched."; }
ok "assembled command validates"

if [ "$ALLOW_RESUME_CODE_DRIFT" = "1" ]; then
  warn "ALLOW_RESUME_CODE_DRIFT=1: resuming across a code change. The rows"
  warn "before and after this point were generated by different engines, and"
  warn "nothing downstream distinguishes them. Keep the SEARCH settings"
  warn "identical to what the earlier iterations ran, or the buffer mixes"
  warn "targets from two different algorithms."
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
