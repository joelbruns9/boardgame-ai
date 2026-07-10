#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run9 = run8 loop + the DIVERSITY package (FOR REVIEW — do not run unedited).
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai).
#
# Thesis (from the run7/run8 campaign): capacity is not binding (bake-off),
# loop hygiene is solved and self-healing (run8/8b), search depth is saturated
# (sims sweep) — three independent trajectories equilibrated at parity with
# the banked average. The remaining ceiling is the POSITION DISTRIBUTION.
# Run9 changes what the learner sees, not how it learns:
#
#   1. Random openings: half of self-play games start with k ~ U[2,8]
#      uniformly-random UNRECORDED plies (4 picks + up to 2 low-stakes early
#      placements per player — perturbs without ruining). Diversifies the
#      midgame, the worst-fit phase (bake-off CE: early 1.78, MID 1.98,
#      end 1.21).
#   2. HOF value-personalities: each opponent draw also draws alpha from
#      {0,1} for ITS evaluator — win-maximizer vs score-maximizer styles from
#      the same nets. Pool sampling switches uniform -> recency so ongoing
#      hof_add promotions dominate and stale entries fade (no manual curation).
#   3. Ops: STOP-file shutdown (touch $RUN/STOP — do NOT kill -INT) and
#      buffer autosave every 10 iters (bounds loss on hard kills).
#
# Pre-registered reading: resumed promotions by gate ~55-65 confirm the
# diversity thesis; parity again means 80x6@4800 self-play is at its practical
# ceiling without external data (-> BGA-seeded starts or ship it).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run9

# ── pre-flight ───────────────────────────────────────────────────────────────
mkdir -p "$RUN"
# Warm start / gate baseline = run8's banked average (top of the measured
# chain: beat run7-peak 52.1%, which beat run5 53.1% / run3 54.9%).
cp -n runs/kingdomino/cloud_80x6_run8/current_best.pt "$RUN/current_best.pt"
# Pool: continue hof_run7 (now recency-weighted; old entries fade on their own).
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
  --random_opening_fraction 0.5 \
  --random_opening_plies_min 2 --random_opening_plies_max 8 \
  \
  --hof_dir runs/kingdomino/hof_run7 \
  --hof_fraction 0.3 --hof_start_iter 1 --hof_sample_weights recency \
  --hof_alpha_choices 0.0,1.0 \
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
  --seed 9 \
  > "$RUN/nohup.log" 2>&1 &

echo $! > "$RUN/run.pid"
echo "run9 launched: PID $(cat "$RUN/run.pid")  ->  $RUN/nohup.log"
echo "graceful stop: touch $RUN/STOP   (buffer saves; do NOT kill -INT)"
