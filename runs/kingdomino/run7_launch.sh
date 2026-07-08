#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run7 = run6 diversity package + three fixes (FOR REVIEW — do not run unedited).
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai).
#
# Deltas vs run6:
#   1. Warm start = the Item-1 banked net (best_checkpoint/current_best.pt after
#      the high-power run6 gate). FRESH buffer — run6's buffer holds exactly the
#      400-sim-HOF / drifted-learner data run7 removes; do not import it.
#   2. HOF asymmetric deep targets: the learner seat in HOF games now searches
#      at full --sims with playout-cap (recording only full-search moves, same
#      as normal self-play); --hof_sims caps ONLY the frozen opponent seat
#      (engine hof_opponent_seat override). Pool curated to near-peers
#      (hof_run7: run3_iter80 + run5_avg + pre_run7 if Item 1 banked).
#   3. Promotion ratchet: promote iff confidently >51% — LCB 0.51 on 2500 games
#      @ 300 sims (was WR wall 0.53 / LCB 0.50 on 1032 @ 100). This unlocks
#      banking of real micro-gains; it does not by itself create strength.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run7

# ── pre-flight ───────────────────────────────────────────────────────────────
mkdir -p "$RUN"
# run7 starts from the Item-1 winner: best_checkpoint/current_best.pt as banked
# by run7_item1_bank.py (or the unchanged run5 net if nothing cleared the gate).
# Copied into run7's own dir so run7 owns its gate baseline.
cp -n runs/kingdomino/best_checkpoint/current_best.pt "$RUN/current_best.pt"
# Curated HOF pool (near-peer entries + index) — synced from the laptop:
ls -1 runs/kingdomino/hof_run7/hof_index.jsonl runs/kingdomino/hof_run7/*.pt

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
  --hof_dir runs/kingdomino/hof_run7 \
  --hof_fraction 0.2 --hof_start_iter 1 --hof_sample_weights uniform \
  --hof_sims 400 --hof_temp_moves 8 --hof_add_every 10 \
  `# --hof_sims now caps ONLY the frozen opponent seat; the learner seat in` \
  `# HOF games searches at --sims with playout-cap (asymmetric deep targets).` \
  \
  --selfplay_generator_mode soft_gate \
  --promotion_every 5 --promotion_games 2500 --promotion_sims 300 \
  --promotion_min_win_rate 0.51 --promotion_min_lcb 0.51 \
  --soft_gate_revert_win_rate 0.48 \
  \
  --benchmark_every 0 --elo_every 0 \
  --exact_fallback_positions "$RUN/exact_fallback_positions.jsonl" \
  --seed 7 \
  > "$RUN/nohup.log" 2>&1 &

echo "run7 launched: PID $!  ->  $RUN/nohup.log"
