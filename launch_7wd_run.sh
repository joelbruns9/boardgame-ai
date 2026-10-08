#!/usr/bin/env bash
# =============================================================================
# launch_7wd_run.sh — the RUN DECISION for one 7 Wonders Duel training run.
#
# This file holds what you decided; `sweeps/measured_env.sh` holds what the box
# measured. Keeping them apart is the point: the decision is portable and
# belongs in git, the measurement belongs to one rented instance and must be
# re-taken on the next one. Baking a slot count in here would silently carry a
# dead box's geometry onto a live one.
#
# Two passes, and the second one is where the run actually starts:
#
#   # PASS 1 — set up the box and MEASURE it. Ends without launching.
#   SWEEP_CHECKPOINT=/path/to/L_checkpoint.pt bash launch_7wd_run.sh
#
#   # PASS 2 — launch on this box's numbers.
#   source ~/boardgame-ai/runs/seven_wonders_duel/run08/sweeps/measured_env.sh
#   bash launch_7wd_run.sh
#
# Pass 2 REFUSES to launch if this box has a sweep that was not sourced. Before
# that guard existed, forgetting to source it produced a run on built-in
# defaults that reported them as "measured", and the two cases had identical
# command lines. Override with ALLOW_UNMEASURED_LAUNCH=1 only to launch on
# defaults deliberately.
#
# Everything below is an override of a `setup_cloud_7wd.sh` default. That script
# is the authority on what each knob means and why it is set where it is; this
# one records the choices for THIS run and nothing else.
#
# THIS RUN: run08, the final run of MODEL_GROWTH_PLAN.md ("Final run
# preparation", owner decisions 2026-10-06/07). run07's decisions are in git
# history. In one line: warm start from the encoder-8 laptop pretrain, empty
# buffer, every move searched at 1,200 sims, no promotion gate, no HOF, the
# specialists on a fixed cycle from iteration 0, G12 restarts at 25%.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Run identity ────────────────────────────────────────────────────────────
# The launcher keys everything -- buffers, checkpoints, sweeps, the manifest --
# off RUN_DIR_REL, so naming the run here is what keeps two runs on one box from
# resuming into each other.
export RUN_DIR_REL="${RUN_DIR_REL:-runs/seven_wonders_duel/run08}"

# The branch the box builds. Everything this run carries -- W1-W7, S2b, the
# warm-start knob, this file itself -- lives on this branch and is NOT on main,
# and a clone without a branch lands on main, builds, and launches none of it
# while every sentinel check still passes. Change this when the work is merged.
export REPO_BRANCH="${REPO_BRANCH:-sevenwd-w9-prototype}"
export ITERATIONS="${ITERATIONS:-200}"
export GAMES_PER_ITERATION="${GAMES_PER_ITERATION:-1000}"

# ── No cold-start scaffolding ───────────────────────────────────────────────
#
# The bots and the draft prior exist to get a RANDOM net to a playable level:
# scripted bots teach the instant-win conditions fast, and the draft prior blends
# a published Wonder tier list (the ZeusAI paper's converged preferences) into
# draft-node priors. A warm start from candidate_0085 is past both. Removing all
# three also leaves the league opening as the ONLY scaffold change at 10k games --
# run 04's value-head collapse followed four changes landing on one knot.
export SEED_GAMES="${SEED_GAMES:-0}"                          # no bot seed corpus
export DRAFT_PRIOR_GAMES="${DRAFT_PRIOR_GAMES:-0}"            # no tier-list draft prior
export CURRICULUM_ANNEAL_GAMES="${CURRICULUM_ANNEAL_GAMES:-0}"  # no bot games in the mix
# ── Empty replay buffer (owner 2026-10-06) ──────────────────────────────────
#
# The pretrain already absorbed run07 (iterations 41-100, G2b relabelled, G8.2
# overlay), so the run starts from an EMPTY buffer and learns only from games
# its own search produced. The cloud2 import below is run07's, kept for the
# record; leaving WARM_BUFFER unset is now the decision, not an omission.
ALLOW_COLD_BUFFER="${ALLOW_COLD_BUFFER:-1}"   # read below, not by the launcher

# ── (run07) Warm buffer: cloud2's last 20 iterations ────────────────────────
#
# A fixed 190x512 step budget over a replay buffer that starts empty
# over-presents early positions: at a 50k warm-up the first iteration's
# positions are seen 11.7x against a steady state of 5.6x. Importing cloud2's
# final 20 iterations (20,000 games, target_version 3, Rust-derived cleanly)
# fills the 20-iteration window from iteration 0, so the run's own positions
# see the steady-state rate from the start, and cloud2's age out one iteration
# per iteration. Those are the games candidate_0085's lineage trained on, so
# they are close to on-policy for the warm net.
#
# Build it from the cloud2 archive (gitignored, ~700 MB) and upload it:
#   cat runs/seven_wonders_duel/cloud2/7wd_cloud_20260825T005745Z/buffers/iter_00{77..96}.jsonl \
#     > runs/seven_wonders_duel/warm_buffers/cloud2_iter0077-0096.jsonl
#
# The rows carry no search outlook (`root_outlook`), so a victory-type soft
# target falls back to the final result on them.
export WARM_BUFFER="${WARM_BUFFER:-}"
if [ -z "$WARM_BUFFER" ] && [ "${ALLOW_COLD_BUFFER:-0}" != "1" ]; then
  echo "[FATAL] WARM_BUFFER is unset. This run imports cloud2's last 20 iterations;" >&2
  echo "        set it to the uploaded JSONL's absolute path, e.g." >&2
  echo "        WARM_BUFFER=\$HOME/cloud2_iter0077-0096.jsonl" >&2
  echo "        (ALLOW_COLD_BUFFER=1 starts from an empty buffer deliberately.)" >&2
  exit 1
fi
# With an empty buffer this is the training warmup. Every move is now a full
# search and every full move is a training row, so a game records several
# times run07's ~15 rows and 100k is reached within the first iteration or two.
# The laptop dry run measures rows per game; TRAIN_STEPS below follows it.
#
# run08: 350k (owner 2026-10-07). Training from the first iteration on an empty
# buffer draws the earliest games ~12x over the run against ~5x at steady state
# (review of ebc70c0). Waiting for ~6 iterations of rows (~59 per game in the
# dry run) brings the first cohort to ~6x. The first ~6 iterations are then
# generated by the unchanged pretrain at full search -- good data, no updates.
export MIN_BUFFER_POSITIONS="${MIN_BUFFER_POSITIONS:-350000}"

# ── Warm start ──────────────────────────────────────────────────────────────
#
# This run carries the trained net forward. The encoder's new features were
# APPENDED so an existing checkpoint migrates additively, and W1/W2/W4/W5 all
# start inert: seeding `candidate_0085.pt` into a model with every switch below
# reproduced its moves on 68 real BGA positions (top move 100% identical, policy
# within 0.05%, value within 0.0002). Without this the run starts from random
# weights, so it is REQUIRED rather than defaulted. Upload the checkpoint and
# give its absolute path on the box.
#
# run08: `runs/seven_wonders_duel/prep/final_41_100_g10a/pretrained.pt` (laptop,
# 7f90cbbc...): candidate_0100 migrated to encoder-8 (G10a consequence
# features) and pretrained on run07 iterations 41-100. Ties the encoder-7
# pretrain on sealed G0 and keeps every gain over candidate_0100.
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
if [ -z "$INIT_CHECKPOINT" ] && [ "${ALLOW_COLD_START:-0}" != "1" ]; then
  echo "[FATAL] INIT_CHECKPOINT is unset. This run warm-starts; set it to the" >&2
  echo "        uploaded checkpoint's absolute path, e.g." >&2
  echo "        INIT_CHECKPOINT=\$HOME/final_41_100_g10a.pt bash launch_7wd_run.sh" >&2
  echo "        (ALLOW_COLD_START=1 launches from random weights deliberately.)" >&2
  exit 1
fi

# ── Architecture: the workstreams this run carries ──────────────────────────
#
# ALL of these default OFF in `phase_d`, and until this file existed none was
# reachable from the launcher -- so a run meant to carry the new architecture
# would have carried none of it while the manifest recorded a commit that
# contained all of it. State every one explicitly, including the zeros, so the
# run's shape is readable here rather than inferred from a chain of defaults.
export SLOT_EMBEDDING="${SLOT_EMBEDDING:-1}"            # W1
export GRAPH_MODULE="${GRAPH_MODULE:-1}"                # W2
export SWD_CONTROL_FEATURES="${SWD_CONTROL_FEATURES:-1}"  # W3
# CUDA-graph replay: on a 5090 one thread dispatching ~520 kernels per forward
# capped generation while the shards idled 60%. See cuda_graphs.py.
export CUDA_GRAPHS="${CUDA_GRAPHS:-1}"
# Short-term value targets: carry endgame proofs and later, closer-to-the-end
# search values back along each game (dataset.short_term_values).
export SHORT_TERM_VALUE_WEIGHT="${SHORT_TERM_VALUE_WEIGHT:-0.25}"
# run08: 0.75, not 0.5. Every move is searched, so each game's outcome label is
# shared by ~51-59 rows instead of ~15.5 and presented ~3x as often. Target mix:
# outcome 18.75% / own search 56.25% / short-term 25%. Offline A/B
# (value_target_ab.py, run07 91-100 from candidate_0080, ~358 presentations per
# game): train/held-out outcome gap 0.113 -> 0.083, held-out log loss 0.475 ->
# 0.468, proof error 0.168 -> 0.166, G0 tactics unchanged. The distance-scaled
# schedule (OUTCOME_SHARE_DECAY/FLOOR) tied it exactly, so the simpler flag
# ships. OUTLOOK_BOOTSTRAP (W4) stays 0.5: untested (owner, 2026-10-07).
export VALUE_BOOTSTRAP="${VALUE_BOOTSTRAP:-0.75}"
export ACTION_RESIDUAL="${ACTION_RESIDUAL:-1}"          # W5
export ACTION_EXPOSES="${ACTION_EXPOSES:-1}"
export ACTION_POLICY_WEIGHT="${ACTION_POLICY_WEIGHT:-0.5}"
# W5 affects play once held-out evidence says it should. The weight starts at 0
# (an untrained scorer at 0.5 changed the warm net's top move on 8.8% of real
# positions), is refitted after every training step to the value that best
# predicts held-out search targets, moves at most 0.1 per iteration, and may
# reach 2 -- W5 outvoting the flat head. `alpha=` on the heartbeat is the run's
# own verdict; sustained above 1, run the W5-only arena
# (`arena --policy-source-a action`).
export FIT_ACTION_ALPHA="${FIT_ACTION_ALPHA:-1}"
export ACTION_ALPHA_MAX="${ACTION_ALPHA_MAX:-2.0}"
export ACTION_ALPHA_STEP="${ACTION_ALPHA_STEP:-0.1}"

# W4. Required by W7 below: the specialist leaf bias reads the seven-way
# outlook, and a leaf without one is a hard error rather than a silent zero
# bias.
#
# ATTACHED and REPLACING joint7 for this run, so the trunk learns victory type
# through W4 -- whose seven classes sum to its own win probability -- instead of
# the free flat head. The weight is joint7's own coefficient (value_weight 1 x
# aux_weight 0.2), so the change is the head's structure and its target, not how
# hard the trunk is pushed toward victory type. The flat joint7 goes stale; the
# advisor switches its victory-type read to W4 for these checkpoints.
export HIERARCHICAL_VALUE="${HIERARCHICAL_VALUE:-1}"
export HIER_VALUE_WEIGHT="${HIER_VALUE_WEIGHT:-0.2}"
export HIER_VALUE_DETACH="${HIER_VALUE_DETACH:-0}"
export HIER_VALUE_REPLACES_JOINT7="${HIER_VALUE_REPLACES_JOINT7:-1}"
# W4's target: the realised victory type blended 50/50 with search's seven-way
# root outlook, as VALUE_BOOTSTRAP does for win/loss. Ramped in over the first
# 10k games because the outlook is averaged from W4's own leaf predictions and
# the head starts untrained on a warm start. cloud2's imported rows carry no
# outlook and keep the hard label.
export OUTLOOK_BOOTSTRAP="${OUTLOOK_BOOTSTRAP:-0.5}"
# run08: no ramp. It existed because W4 started untrained on run07's warm
# start; the pretrain trained it on 60 iterations of run07 targets.
export OUTLOOK_BOOTSTRAP_GAMES="${OUTLOOK_BOOTSTRAP_GAMES:-0}"

# ── W7: the specialist league ───────────────────────────────────────────────
#
# Shares are fractions of ALL games and share one budget with HOF_FRACTION:
# 0.15 HOF + 0.15 science + 0.10 military is 40% league play, 60% pure
# self-play, and one class is drawn per iteration.
#
# ⚠ LAMBDA IS UNMEASURED. S0a (`specialist_probe.py`) has never been run against
# a trained checkpoint, and on an untrained net the bias moved the root value by
# exactly lambda x outlook while changing ZERO moves -- a constant bonus does
# not move an argmax. If a trained net behaves the same way, these games are
# played against an opponent identical to the general and nothing in the run
# says so. Run the probe first:
#
#   python -m games.seven_wonders_duel.specialist_probe \
#     --checkpoint <current_best.pt> --buffer <iter_NNNN.jsonl> \
#     --lambda 0.5 --victory scientific --positions 200 --sims 256
#
# and read `moved_fraction` and `credible_fraction_of_moved` before trusting
# the value below.
# name:share:LAMBDA. Lambda 3, measured -- see COALESCER-era probe results in
# SPECIALIST_LEAGUE_REVIEW_REQUEST.md 10. On a cloud2-trained net the pursuit
# gain peaks at lambda ~= 3 and DECLINES above it, while credibility falls
# monotonically, so 3 is a bracket to calibrate around at bootstrap rather than
# a ceiling to raise. The previous 0.5 / 0.4 moved 6% of decisions against 15%
# at 3, for +2.2% pursuit against +6.2%.
#
# Expect a strong net to want LESS than 3, not more: lambda's bite scales with
# how sharply the outlook head separates sibling moves, and this was measured
# on a 5.2M-param net at joint7_acc 0.512.
export SPECIALISTS="${SPECIALISTS:-science:0.15:3,military:0.10:3}"
# run08: a FIXED cycle, S M S M S, every iteration a specialist iteration.
# run07 drew at random: its first twenty league iterations went 8 military,
# 1 science, and 7 iterations fell through to plain self-play unannounced.
export LEAGUE_SCHEDULE="${LEAGUE_SCHEDULE:-cycle}"
# run08: HOF OFF. Without a gate nothing is promoted, and promotions are the
# only thing that fills the HOF, so its share would silently become self-play.
# Forgetting is watched by the self-anchor instead (vs the learner 20k games
# back); the specialists carry the strategic diversity.
export HOF_FRACTION="${HOF_FRACTION:-0}"
# 10k, not setup's 50k. The 50k was a COLD-start decision: run 04 opened the
# league against its own near-random bootstrap checkpoint, so league games were
# lopsided wins pinning value targets near +1. A warm start's iteration-0 best is
# candidate_0085, so the archive starts strong. 10k gives the new modules about
# ten iterations to train before specialists (which follow this clock, since
# SPECIALIST_BOOTSTRAP_GAMES=0) and S2b reanalysis start shaping the general.
#
# run08: 0. The league opens at iteration 0 -- the specialists seed from the
# pretrained start, which already carries every module.
export HOF_START_GAMES="${HOF_START_GAMES:-0}"
export SPECIALIST_BOOTSTRAP_GAMES="${SPECIALIST_BOOTSTRAP_GAMES:-0}"   # 0 = follow HOF_START_GAMES
export SPECIALIST_FLOOR_EVERY="${SPECIALIST_FLOOR_EVERY:-5}"
# S2b: on, the general also learns to EXECUTE the attacks the specialists find;
# off, it only learns to defend them. ON for this run -- the attacking half is
# the one the human losses to ZeusAI point at, and the coalesced backend brought
# reanalysis from 2539 to 45 ms/position. The cost is that this run cannot also
# be S5's defence-only vs defence+transfer comparison.
export SPECIALIST_REANALYSIS="${SPECIALIST_REANALYSIS:-1}"

# ── run08: lifecycle and search ─────────────────────────────────────────────
#
# No promotion gate (owner 2026-10-06): the learner always generates and
# progress is read from the self-anchor (the stopping rule). PROMOTION_EVERY=0
# is what actually removes the gate -- `latest` alone still schedules one, and
# a REJECT would reset the learner to the starting checkpoint.
export GENERATOR_MODE="${GENERATOR_MODE:-latest}"
export PROMOTION_EVERY="${PROMOTION_EVERY:-0}"
# Both reset the learner to current_best; the controller refuses them outside
# soft_gate (caught by the laptop dry run, 2026-10-07).
export REVERT_RESET_AFTER="${REVERT_RESET_AFTER:-0}"
export PROBATION_RESET_AFTER="${PROBATION_RESET_AFTER:-0}"
# Every move a full search at 1,200 sims (decision 11; budget 2026-10-07).
export CHEAP_SIMS="${CHEAP_SIMS:-1200}"
export FULL_SIMS="${FULL_SIMS:-1200}"
export FULL_SEARCH_FRACTION="${FULL_SEARCH_FRACTION:-1.0}"
# G12 restart archive: a quarter of games replay a decisive position from an
# earlier game's deal with an untried first move.
export RESTART_FRACTION="${RESTART_FRACTION:-0.25}"

# Train steps, RE-DERIVED for every-move-full search (laptop dry run
# 2026-10-07, run08 settings, 64 sims): ~51 general policy rows per game with
# 25% restarts and 25% league games, against run07's ~15.5 -- so setup's
# 0.19 x games would be ~1.9 samples per new row instead of run07's ~6. 550
# steps x 512 over ~51k rows per 1,000 games is ~5.5, back on target. Warmup a
# third, derived by setup from whatever TRAIN_STEPS ends up being.
#
# DERIVED from GAMES_PER_ITERATION (0.55 per game), not fixed: a fixed 550
# survived an override of GAMES_PER_ITERATION for a smoke run (review of
# ebc70c0). 550 at the planned 1,000.
export TRAIN_STEPS="${TRAIN_STEPS:-$(( (GAMES_PER_ITERATION * 55 + 99) / 100 ))}"

# Exact tactics (G4, incl. 2b reveal strata) and G2b tactic relabelling: both
# phase_d defaults, PINNED here so a default flipped before launch cannot
# silently change the run (review of ebc70c0). G2b comes off only on the
# phase-out census (`g2b_census.py`) over the first buffers.
export EXACT_TACTICS="${EXACT_TACTICS:-1}"
export TACTIC_LABELS="${TACTIC_LABELS:-1}"

# ── Scheduler geometry ──────────────────────────────────────────────────────
#
# Deliberately NOT set here. Pass 1 measures them and writes measured_env.sh;
# pass 2 sources it. The one exception is the sweep grid itself, which says what
# to measure rather than what to use.
export SWEEP_SLOTS_CSV="${SWEEP_SLOTS_CSV:-128,256,512}"
export SWEEP_CAPS_CSV="${SWEEP_CAPS_CSV:-1024,2048}"
export SWEEP_INFLIGHT_CSV="${SWEEP_INFLIGHT_CSV:-1,2}"
export SWEEP_WORKERS_CSV="${SWEEP_WORKERS_CSV:-2,4,8}"
# The generation/solver CORE SPLIT, as TOTAL threads. Generation and the solver
# compete for the same cores and the solver runs synchronously inside a shard,
# so a thread given to it is a thread taken from leaf production; 0 measures
# with the solver off, which is the baseline the others are read against.
export SWEEP_SOLVER_THREADS_CSV="${SWEEP_SOLVER_THREADS_CSV:-0,4,8,16}"
# The coalescing wait. Swept, never pinned: its effect is on batch WIDTH, which
# no wall-clock total reports, so a value carried over from another box would
# look harmless and change what the GPU sees on every forward.
export SWEEP_INFERENCE_WAIT_CSV="${SWEEP_INFERENCE_WAIT_CSV:-0,1,2}"
# SWEEP_GENERATION_GAMES is left to setup_cloud_7wd.sh, which derives it as 3x
# the largest slot count. A pin here (200) went stale against 512 slots.

# ── Hand over ───────────────────────────────────────────────────────────────
#
# `setup_cloud_7wd.sh` self-updates on pull and re-execs, so it must be invoked
# rather than sourced.
exec bash "$HERE/setup_cloud_7wd.sh" "$@"
