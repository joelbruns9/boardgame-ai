# Run7 Item-5 verification: tiny local job with run7-style flags at reduced
# scale (sims 600 / fast 100 / hof 100 mirror run7's 4800/200/400 asymmetry).
# Confirms (a) HOF learner-seat moves searched at full --sims, (b) only
# learner-seat moves recorded, (c) prune total_visits = frontier sims,
# (d) run7 gate flags flow into the config (read back from the checkpoint).
$ErrorActionPreference = "Continue"
$Python = "C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe"
Set-Location "C:\Users\joeld\projects\boardgame-ai"

# soft_gate requires the gate baseline to exist (mirrors run7_launch.sh's cp).
New-Item -ItemType Directory -Force runs\kingdomino\run7_verify_tiny | Out-Null
if (-not (Test-Path runs\kingdomino\run7_verify_tiny\current_best.pt)) {
    Copy-Item runs\kingdomino\best_checkpoint\current_best.pt `
        runs\kingdomino\run7_verify_tiny\current_best.pt
}

& $Python -u -m games.kingdomino.self_play `
    --engine batched_open_loop --device cuda --async_solve --game_cpus 2 `
    --channels 80 --blocks 6 `
    --batch_slots 8 --leaf_batch 6 --virtual_loss 1 --amp_inference `
    --exact_endgame_max_secs 3.0 --batch_size 64 `
    --warm_start runs/kingdomino/best_checkpoint/current_best.pt `
    --current_best_path runs/kingdomino/run7_verify_tiny/current_best.pt `
    --checkpoint_dir runs/kingdomino/run7_verify_tiny `
    --save_buffer runs/kingdomino/run7_verify_tiny/buffer.pkl `
    --iterations 2 --games_per_iter 8 --train_steps 5 `
    --sims 600 `
    --lr_schedule 0:1e-4 `
    --buffer 20000 --min_buffer 100 `
    --temp_moves 30 --endgame_oversample 1.0 `
    --lambda_score 0.5 --lambda_w 0.25 --score_scale 160.0 `
    --policy_weight 1.0 --grad_clip 1.0 --weight_decay 1e-4 `
    --margin_gain 2.0 --alpha 0.5 --c_puct 1.5 --fpu -0.2 `
    --playout_cap_randomization --full_search_fraction 0.25 `
    --fast_move_sims 100 --fast_move_dirichlet_epsilon 0.0 --fast_move_temp_moves 0 `
    --policy_target_pruning `
    --hof_dir runs/kingdomino/hof_run7 `
    --hof_fraction 0.25 --hof_start_iter 1 --hof_sample_weights uniform `
    --hof_sims 100 --hof_temp_moves 8 --hof_add_every 10 `
    --selfplay_generator_mode soft_gate `
    --promotion_every 5 --promotion_games 2500 --promotion_sims 300 `
    --promotion_min_win_rate 0.51 --promotion_min_lcb 0.51 `
    --soft_gate_revert_win_rate 0.48 `
    --benchmark_every 0 --elo_every 0 `
    --seed 7 *>&1 |
    Out-File -Encoding utf8 runs/kingdomino/run7_verify_tiny/run.log
Write-Output "[verify] exit code: $LASTEXITCODE"
Write-Output "VERIFY TINY RUN COMPLETE" |
    Out-File -Encoding utf8 -Append runs/kingdomino/run7_verify_tiny/run.log
