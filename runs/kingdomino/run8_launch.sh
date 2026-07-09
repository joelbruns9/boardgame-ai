#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run8 = run7 + the loop fixes (FOR REVIEW — do not run unedited).
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai).
#
# Evidence base (run7 post-mortem + capacity bake-off):
#   - Capacity NOT binding (80x6 matched 96x6/80x10 on held-out fit) → stay 80x6.
#   - Replay ratio ~32 samples/example drove win-head overfit drift
#     (brier_diag 0.10→0.25) → --train_steps 300 (ratio ~10). Watch the
#     brier_diag-vs-train gap: flat = ok, opening = cut further.
#   - Snapshots are ±5% noise around the trajectory mean → gate/bank a rolling
#     average of the last 8 checkpoints (--promotion_average_k 8).
#   - A diverged learner never recovers unaided (9 straight reverts) → reset
#     learner weights + optimizer moments after 2 consecutive reverts
#     (--revert_reset_after 2). One revert alone only swaps the generator.
#   - Warm start = run7 peak (iter_0025 of run7), CERTIFIED vs run5 iter_0005
#     (WR 53.1%, LCB 51.2%, 2500 games @300) and vs run3_iter80 (54.9%/52.4%)
#     — the missing transitivity link is closed.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run8

# ── pre-flight ───────────────────────────────────────────────────────────────
mkdir -p "$RUN"
# Warm start / gate baseline = the certified run7 peak, copied into run8's own
# dir so in-run promotions never touch the run7 artifact.
cp -n runs/kingdomino/cloud_80x6_run7/run7_peak_iter25.pt "$RUN/current_best.pt"
# HOF pool: continue hof_run7 (near-peers incl. the peak itself via the run7
# hof_add snapshots). Fresh buffer — no --warm_buffer, same rationale as run7.
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
  \
  --iterations 140 --games_per_iter 400 --train_steps 300 \
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
  --hof_dir runs/kingdomino/hof_run7 \
  --hof_fraction 0.2 --hof_start_iter 1 --hof_sample_weights uniform \
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
  --seed 8 \
  > "$RUN/nohup.log" 2>&1 &

echo $! > "$RUN/run.pid"
echo "run8 launched: PID $(cat "$RUN/run.pid")  ->  $RUN/nohup.log"
echo "kill cleanly (saves buffer): kill -INT \$(cat $RUN/run.pid)"
