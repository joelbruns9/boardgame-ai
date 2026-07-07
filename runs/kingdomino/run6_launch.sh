#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run6 = diversity package on the 80x6 net (FOR REVIEW — do not run unedited).
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai). Fully isolated from
# run5: its own checkpoint_dir + current_best, so it never touches the file the
# run5 gate owns.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run6

# ── pre-flight ───────────────────────────────────────────────────────────────
mkdir -p "$RUN"
# run6 starts from run5's promoted net (iter_0005 == current_best), copied into
# run6's own dir so run6 owns its gate baseline. iter_0005.pt is immutable, so
# this is unambiguous regardless of what best_checkpoint/current_best.pt points
# at right now.
cp -n runs/kingdomino/cloud_80x6_run5/iter_0005.pt "$RUN/current_best.pt"
# HOF pool (3 entries + index) — you have already uploaded these 4 files:
ls -1 runs/kingdomino/hof_run6/hof_index.jsonl runs/kingdomino/hof_run6/*.pt

# ── launch ───────────────────────────────────────────────────────────────────
nohup python games/kingdomino/self_play.py \
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
  --iterations 120 --games_per_iter 400 --train_steps 1000 \
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
  --hof_dir runs/kingdomino/hof_run6 \
  --hof_fraction 0.2 --hof_start_iter 1 --hof_sample_weights uniform \
  --hof_sims 200 --hof_current_sims 200 --hof_temp_moves 8 --hof_add_every 10 \
  \
  --selfplay_generator_mode soft_gate \
  --promotion_every 5 --promotion_games 1032 --promotion_sims 100 \
  --promotion_min_win_rate 0.53 --promotion_min_lcb 0.50 \
  --soft_gate_revert_win_rate 0.48 \
  \
  --benchmark_every 0 --elo_every 0 \
  --exact_fallback_positions "$RUN/exact_fallback_positions.jsonl" \
  --seed 6 \
  > "$RUN/nohup.log" 2>&1 &

echo "run6 launched: PID $!  ->  $RUN/nohup.log"
