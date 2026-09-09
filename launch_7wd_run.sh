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
#   source ~/boardgame-ai/runs/seven_wonders_duel/run07_bundle/sweeps/measured_env.sh
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
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Run identity ────────────────────────────────────────────────────────────
# The launcher keys everything -- buffers, checkpoints, sweeps, the manifest --
# off RUN_DIR_REL, so naming the run here is what keeps two runs on one box from
# resuming into each other.
export RUN_DIR_REL="${RUN_DIR_REL:-runs/seven_wonders_duel/run07_bundle}"
export ITERATIONS="${ITERATIONS:-200}"
export GAMES_PER_ITERATION="${GAMES_PER_ITERATION:-1000}"
export SEED_GAMES="${SEED_GAMES:-5000}"

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
export ACTION_RESIDUAL="${ACTION_RESIDUAL:-1}"          # W5
export ACTION_EXPOSES="${ACTION_EXPOSES:-1}"
export ACTION_POLICY_WEIGHT="${ACTION_POLICY_WEIGHT:-0.5}"

# W4. Required by W7 below: the specialist leaf bias reads the seven-way
# outlook, and a leaf without one is a hard error rather than a silent zero
# bias. `HIER_VALUE_DETACH=1` keeps the head out of the trunk's gradient, which
# is what makes enabling it a bounded change rather than a second experiment.
export HIERARCHICAL_VALUE="${HIERARCHICAL_VALUE:-1}"
export HIER_VALUE_WEIGHT="${HIER_VALUE_WEIGHT:-0.5}"
export HIER_VALUE_DETACH="${HIER_VALUE_DETACH:-1}"

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
export HOF_FRACTION="${HOF_FRACTION:-0.15}"
export SPECIALIST_BOOTSTRAP_GAMES="${SPECIALIST_BOOTSTRAP_GAMES:-0}"   # 0 = follow HOF_START_GAMES
export SPECIALIST_FLOOR_EVERY="${SPECIALIST_FLOOR_EVERY:-5}"
# S2b is an ARM of the pilot, not a setting. On, the general also learns to
# execute the attacks; off, it only learns to defend them. Leave it off unless
# this run is the transfer comparison.
export SPECIALIST_REANALYSIS="${SPECIALIST_REANALYSIS:-0}"

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
export SWEEP_GENERATION_GAMES="${SWEEP_GENERATION_GAMES:-200}"

# ── Hand over ───────────────────────────────────────────────────────────────
#
# `setup_cloud_7wd.sh` self-updates on pull and re-execs, so it must be invoked
# rather than sourced.
exec bash "$HERE/setup_cloud_7wd.sh" "$@"
