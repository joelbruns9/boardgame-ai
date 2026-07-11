#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run10 = run9 loop + the ADVERSARIAL DRAFTING package (review before running).
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai).
#
# Thesis (RUN10_PLAN.md; evidence from the kylechu20 post-mortems): the loop is
# healthy but self-play equilibrium never teaches the PICK interaction — the
# policy prior starves squeeze lines (4.7%-prior game-losing reply got 0.62%
# of 3200 sims) and score heads assume benign continuations. Run10 makes the
# search and the opponent pool adversarial in the drafting dimension; nothing
# from run9 is reverted (its null result showed no harm, and the diversity
# knobs attack an independent axis):
#
#   1. Pick-group visit floors (6a4af7b): at tree depths 1-2 (never the root),
#      every pick-group (joint_idx % 5) is guaranteed >=8% of child visits —
#      the search must LOOK at every draft branch before PUCT concentrates.
#      Training-only; advisor/gates untouched. Read the new per-iteration
#      "pf: minshare=" diagnostic: it should sit near 0.08 when the floor
#      binds (baseline under a trained, concentrated prior was ~0.01).
#   2. SPITE personality in the HOF pool (7cae42e): --hof_style_choices adds
#      alpha=1 + margin_gain=0.2 (linear-tanh region) — a point-differential
#      maximizer for whom denying the learner 8 pts = gaining 8. Draws
#      uniformly among {win-max 0:2.0, score-max 1:2.0, spite 1:0.2}.
#
# Pre-registered reading (RUN10_PLAN.md): (a) promotions vs the banked avg on
# the usual ratchet; (b) bga_score_audit err_you on logged losses shrinks
# toward +/-10; (c) row-29 prior check — opponent's d34-pick-d8 at table
# 880439726 decision 29 should no longer sit at 4.7% prior / 0.6% visits.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run10

# ── pre-flight ───────────────────────────────────────────────────────────────
mkdir -p "$RUN"
# Warm start / gate baseline = the ratchet's latest banked best. run9 never
# promoted, so this is still the run8 banked average (sha 4bf07b); if run9's
# dir is gone, substitute runs/kingdomino/cloud_80x6_run8/current_best.pt.
cp -n runs/kingdomino/cloud_80x6_run9/current_best.pt "$RUN/current_best.pt"
# Pool: continue hof_run7 (recency-weighted; hof_add keeps feeding it).
ls -1 runs/kingdomino/hof_run7/hof_index.jsonl runs/kingdomino/hof_run7/*.pt

# ── launch ───────────────────────────────────────────────────────────────────
nohup python -m games.kingdomino.self_play \
  --engine batched_open_loop --device cuda --async_solve --game_cpus 4 \
  --channels 80 --blocks 6 \
  --batch_slots 86 --leaf_batch 6 --virtual_loss 1 --amp_inference \
  --exact_endgame_max_secs 3.0 --batch_size 256 \
  \
  --warm_start "$RUN/current_best.pt" \
  --current_best_path "$RUN/current_best.pt" \
  --checkpoint_dir "$RUN" \
  --save_buffer "$RUN/buffer_final.pkl" \
  --buffer_autosave_every 10 \
  \
  --iterations 120 --games_per_iter 400 --train_steps 300 \
  --sims 4800 \
  --lr_schedule 0:1e-4 \
  --buffer 200000 \
  --min_buffer 180000 \
  --temp_moves 30 \
  --endgame_oversample 1.0 \
  \
  --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 \
  --policy_weight 1.0 --grad_clip 1.0 --weight_decay 1e-4 \
  --margin_gain 2.0 --alpha 0.5 --c_puct 1.5 --fpu -0.2 \
  \
  --playout_cap_randomization --full_search_fraction 0.25 \
  --fast_move_sims 200 --fast_move_dirichlet_epsilon 0.0 --fast_move_temp_moves 0 \
  --policy_target_pruning \
  \
  --pick_floor_frac 0.08 --pick_floor_depth 2 \
  \
  --random_opening_fraction 0.5 \
  --random_opening_plies_min 2 --random_opening_plies_max 8 \
  \
  --hof_dir runs/kingdomino/hof_run7 \
  --hof_fraction 0.3 --hof_start_iter 1 --hof_sample_weights recency \
  --hof_style_choices "0:2.0,1:2.0,1:0.2" \
  --hof_sims 400 --hof_temp_moves 8 --hof_add_every 10 \
  \
  --selfplay_generator_mode soft_gate \
  --promotion_every 5 --promotion_games 2500 --promotion_sims 300 \
  --promotion_min_win_rate 0.51 --promotion_min_lcb 0.50 \
  --soft_gate_revert_win_rate 0.48 \
  --promotion_average_k 8 \
  --revert_reset_after 2 \
  \
  --benchmark_every 0 --elo_every 0 \
  --exact_fallback_positions "$RUN/exact_fallback_positions.jsonl" \
  --seed 10 \
  > "$RUN/nohup.log" 2>&1 &

echo $! > "$RUN/run.pid"
echo "run10 launched: PID $(cat "$RUN/run.pid")  ->  $RUN/nohup.log"
echo "graceful stop: touch $RUN/STOP   (buffer saves; do NOT kill -INT)"
