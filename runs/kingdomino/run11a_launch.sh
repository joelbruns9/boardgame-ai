#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run11a = EXPLOITER (PSRO-lite) — train a best-response to the frozen best.
# Runs on the CLOUD BOX from repo root (/root/boardgame-ai).
#
# Thesis (runs 9-10 nulls): self-play never GENERATES squeeze positions —
# exploits are self-erasing in mirror play (the victim adapts next iteration)
# but self-REINFORCING against a frozen victim. The exploiter starts as a
# weight-for-weight clone of the best and learns deviations against a
# stationary target; discovered blind spots stay exploitable and compound.
#
# What changes vs run10: hof_fraction 1.0 (every game learner-vs-frozen),
# --exploiter_frozen_baseline (gates measure, never promote/revert/reset;
# best candidate banked to exploiter_best.pt), exploration up (eps 0.30),
# smaller buffer (stationary target -> faster feedback), avg_k 4.
#
# READINGS (the gate WR curve in training_log.jsonl IS the deliverable):
#   - WR climbs to ~55%+  -> exploitable structure exists; post-mortem the
#     exploiter's wins (draft matrix / swing tools), then run11b: fold
#     exploiter checkpoints into the HOF pool and retrain the main line.
#   - WR pinned at ~50% through iter ~40 -> best is locally unexploitable at
#     this capacity; pivot to distributional score head / BGA seeding.
#   - Verify any claim H2H at 300 AND 1600 sims before believing it.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /root/boardgame-ai

RUN=runs/kingdomino/cloud_80x6_run11a
FROZEN_SRC=runs/kingdomino/cloud_80x6_run10/current_best.pt  # ratchet best at run10 end
HOF=runs/kingdomino/hof_run11a

# ── pre-flight: freeze the victim + build the single-net pool ────────────────
mkdir -p "$RUN"
cp -n "$FROZEN_SRC" "$RUN/frozen_best.pt"
sha256sum "$RUN/frozen_best.pt" | tee "$RUN/frozen_best.sha256"
python - <<'PY'
from games.kingdomino.hof import add_hof_entry, read_hof_index
e = add_hof_entry("runs/kingdomino/cloud_80x6_run11a/frozen_best.pt",
                  hof_dir="runs/kingdomino/hof_run11a", tag="frozen_best",
                  metadata={"seeded_for": "run11a", "role": "frozen victim"})
print("pool:", [(x.tag, x.sha256[:12]) for x in read_hof_index("runs/kingdomino/hof_run11a")])
PY

# ── launch ───────────────────────────────────────────────────────────────────
nohup python -m games.kingdomino.self_play \
  --engine batched_open_loop --device cuda --async_solve --game_cpus 4 \
  --channels 80 --blocks 6 \
  --batch_slots 86 --leaf_batch 6 --virtual_loss 1 --amp_inference \
  --exact_endgame_max_secs 3.0 --batch_size 256 \
  \
  --warm_start "$RUN/frozen_best.pt" \
  --current_best_path "$RUN/frozen_best.pt" \
  --exploiter_frozen_baseline \
  --checkpoint_dir "$RUN" \
  --save_buffer "$RUN/buffer_final.pkl" \
  --buffer_autosave_every 10 \
  \
  --iterations 60 --games_per_iter 400 --train_steps 300 \
  --sims 4800 \
  --lr_schedule 0:1e-4 \
  --buffer 100000 \
  --min_buffer 50000 \
  --temp_moves 30 \
  --endgame_oversample 1.0 \
  \
  --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 \
  --policy_weight 1.0 --grad_clip 1.0 --weight_decay 1e-4 \
  --margin_gain 2.0 --alpha 0.5 --c_puct 1.5 --fpu -0.2 \
  --dirichlet_epsilon_schedule 0:0.30 \
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
  --hof_dir "$HOF" \
  --hof_fraction 1.0 --hof_start_iter 1 --hof_sample_weights uniform \
  --hof_sims 800 --hof_temp_moves 0 --hof_dirichlet_epsilon 0.0 \
  --hof_add_every 0 \
  \
  --selfplay_generator_mode soft_gate \
  --promotion_every 5 --promotion_games 2500 --promotion_sims 300 \
  --promotion_min_win_rate 0.55 --promotion_min_lcb 0.52 \
  --promotion_average_k 4 \
  --revert_reset_after 0 \
  \
  --benchmark_every 0 --elo_every 0 \
  --exact_fallback_positions "$RUN/exact_fallback_positions.jsonl" \
  --seed 11 \
  > "$RUN/nohup.log" 2>&1 &

echo $! > "$RUN/run.pid"
echo "run11a launched: PID $(cat "$RUN/run.pid")  ->  $RUN/nohup.log"
echo "exploitability curve: grep promotion_win_rate $RUN/training_log.jsonl"
echo "graceful stop: touch $RUN/STOP   (buffer saves; do NOT kill -INT)"
