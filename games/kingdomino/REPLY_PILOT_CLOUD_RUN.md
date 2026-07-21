# Opponent-reply pilot: cloud runbook

This is the executable handoff for the pilot in
`AZ_SECONDARY_PICK_REPLY_PILOT_PLAN.md`. It does not update `current_best`.
Control and treatment checkpoints remain isolated until every behavior, BGA,
fixed-suite, and strength gate has been reviewed.

## 1. Files that must reach the cloud box

In addition to the repository checkout, copy these local artifacts without
renaming them:

```text
runs/kingdomino/best_checkpoint/current_best.pt
runs/kingdomino/cloud_80x6_run10/buffer_final.pkl
runs/kingdomino/denial_search/signal_positions.jsonl
runs/kingdomino/denial_search/secondary_seed/tree_seed20260717.jsonl
runs/kingdomino/denial_search/secondary_seed/tree_seed21260720.jsonl
runs/kingdomino/denial_search/secondary_seed/tree_seed22260723.jsonl
runs/kingdomino/bga_game_log/                         # for the BGA gate
```

Verify the baseline checkpoint SHA-256 is
`4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`.
The programs independently hash the checkpoint, frozen positions, searched
references, shards, merged labels, splits, and generated arm checkpoints.

## 2. Box setup and hard validation gate

On a fresh Linux RTX 5080/5090 box:

```bash
# Substitute the committed pilot branch/ref and its exact 40-character SHA.
REPO_REF=<pilot-branch-or-tag> EXPECTED_COMMIT=<pilot-commit-sha> \
  bash setup_reply_pilot_cloud.sh
```

If the complete working tree was securely synchronized to the instance rather
than cloned, use:

```bash
SKIP_GIT_UPDATE=1 bash setup_reply_pilot_cloud.sh
```

The pilot-specific setup installs the pinned CUDA dependency pair, builds the
Rust extension, verifies all required artifact paths and the baseline
checkpoint hash, runs the focused and shared-Rust tests, then runs the one-root
full-shape Python/Rust/Rayon comparison automatically. It writes
`runs/kingdomino/reply_pilot/cloud/setup_report.json`.

Stop if action selection differs, if any numerical delta exceeds `1e-6`, or if
the CUDA/build/source/artifact gates in `setup_reply_pilot_cloud.sh` fail.
Choose the fastest measured
Rayon count and set it below; local testing favors four threads, but the cloud
CPU/GPU balance may differ.

```bash
export PILOT_THREADS=4
export PILOT_SHARDS=4
export BASE_CKPT=runs/kingdomino/best_checkpoint/current_best.pt
export PILOT_DIR=runs/kingdomino/reply_pilot/cloud
mkdir -p "$PILOT_DIR"
```

## 3. Training-only calibration

Freeze 40 fresh calibration roots. They are checked against the frozen-50
evaluation set.

```bash
python -m games.kingdomino.reply_pilot \
  --mode freeze --checkpoint "$BASE_CKPT" \
  --positions-path "$PILOT_DIR/calibration_roots.jsonl" \
  --positions 40 --seed 20260719 --trajectory-sims 3200
```

Generate four resumable calibration shards sequentially. Running several GPU
processes concurrently is optional and should only be done after measuring
dollars per accepted label and confirming VRAM headroom.

```bash
for i in 0 1 2 3; do
  python -m games.kingdomino.reply_pilot \
    --mode generate --engine rust --rayon-threads "$PILOT_THREADS" \
    --checkpoint "$BASE_CKPT" \
    --positions-path "$PILOT_DIR/calibration_roots.jsonl" \
    --shards-dir "$PILOT_DIR/calibration_shards" \
    --num-shards "$PILOT_SHARDS" --shard-index "$i" \
    --pick-plies 8 --chance-k 16 --search-sims 3200 --calibration
done

python -m games.kingdomino.reply_pilot \
  --mode merge --shards-dir "$PILOT_DIR/calibration_shards" \
  --output "$PILOT_DIR/calibration_labels.jsonl"

python -m games.kingdomino.reply_pilot \
  --mode summarize --input "$PILOT_DIR/calibration_labels.jsonl" \
  --target-retention 0.75 \
  --output "$PILOT_DIR/calibration_summary.json"
```

The summary proposes locked numerical thresholds by allocating the allowed
rejection tail across margin, Monte Carlo error, and target entropy, then
reports the actual joint retention after top-rank ties are rejected. Inspect
the distributions before continuing. A zero or implausibly small retained set
is a stop condition, not a reason to relax thresholds using frozen-50 results.

Load the proposed values for the production run:

```bash
export MIN_MARGIN=$(python -c "import json; print(json.load(open('$PILOT_DIR/calibration_summary.json'))['proposed_filter']['min_top_two_margin'])")
export MAX_STDERR=$(python -c "import json; print(json.load(open('$PILOT_DIR/calibration_summary.json'))['proposed_filter']['max_mc_standard_error'])")
export MAX_ENTROPY=$(python -c "import json; print(json.load(open('$PILOT_DIR/calibration_summary.json'))['proposed_filter']['max_target_entropy'])")
export MAX_SEED_SD=$(python -c "import json; print(json.load(open('$PILOT_DIR/calibration_summary.json'))['proposed_filter']['max_searched_seed_sd'])")
export MIN_TOP_AGREE=$(python -c "import json; print(json.load(open('$PILOT_DIR/calibration_summary.json'))['proposed_filter']['min_top_pick_agreement'])")
```

## 4. Production labels

Freeze 500 fresh production roots. The calibration roots are passed as a
reserved set and excluded during trajectory collection; collection continues
until it has the requested number of unique, disjoint roots. The production
seed is deliberately far from the calibration seed to avoid wasting trajectory
work traversing the same games.

```bash
python -m games.kingdomino.reply_pilot \
  --mode freeze --checkpoint "$BASE_CKPT" \
  --positions-path "$PILOT_DIR/training_roots.jsonl" \
  --positions 500 --seed 20270720 --trajectory-sims 3200 \
  --reserved-test-path "$PILOT_DIR/calibration_roots.jsonl"
```

Generate, resume, merge accepted examples only, validate, and split by root
state so no root leaks between reply training and reply validation:

```bash
for i in 0 1 2 3; do
  python -m games.kingdomino.reply_pilot \
    --mode generate --engine rust --rayon-threads "$PILOT_THREADS" \
    --checkpoint "$BASE_CKPT" \
    --positions-path "$PILOT_DIR/training_roots.jsonl" \
    --shards-dir "$PILOT_DIR/production_shards" \
    --num-shards "$PILOT_SHARDS" --shard-index "$i" \
    --pick-plies 8 --chance-k 16 --search-sims 3200 \
    --min-top-two-margin "$MIN_MARGIN" \
    --max-mc-standard-error "$MAX_STDERR" \
    --max-target-entropy "$MAX_ENTROPY" \
    --max-searched-seed-sd "$MAX_SEED_SD" \
    --min-top-pick-agreement "$MIN_TOP_AGREE" --reject-ties
done

python -m games.kingdomino.reply_pilot \
  --mode merge --shards-dir "$PILOT_DIR/production_shards" \
  --accepted-only --output "$PILOT_DIR/reply_labels.jsonl"

python -m games.kingdomino.reply_pilot \
  --mode validate --output "$PILOT_DIR/reply_labels.jsonl"

python -m games.kingdomino.reply_pilot \
  --mode split --input "$PILOT_DIR/reply_labels.jsonl" \
  --validation-fraction 0.20 --split-seed 20260719 \
  --train-output "$PILOT_DIR/reply_train.jsonl" \
  --validation-output "$PILOT_DIR/reply_validation.jsonl"
```

Expect roughly 1,000–1,300 accepted examples from 500 roots, not 2,000. The exact count is a
measured outcome; quality and disjointness gates take precedence over volume.

## 5. Equal-step control and treatment

This uses one pre-registered treatment setting: 15% reply batch fraction and
`lambda_reply=0.15`. Both arms receive the exact same 1,000 ordinary replay
batches, including the same D4 transforms; sampled indices and transforms are
logged. Only treatment receives grouped reply loss.

```bash
python -m games.kingdomino.reply_training \
  --checkpoint "$BASE_CKPT" \
  --reply-train "$PILOT_DIR/reply_train.jsonl" \
  --reply-validation "$PILOT_DIR/reply_validation.jsonl" \
  --replay-buffer runs/kingdomino/cloud_80x6_run10/buffer_final.pkl \
  --output-dir "$PILOT_DIR/training" \
  --device cuda --steps 1000 --batch-size 256 \
  --reply-fraction 0.15 --lambda-reply 0.15 \
  --validation-batch-size 256 --buffer-capacity 200000 \
  --sample-workers 4 --lr 1e-4 --weight-decay 1e-4
```

Review `pilot_training_report.json` before search evaluation. It contains the
fixed ordinary holdout, grouped reply loss, within-group placement entropy,
and KL-to-generation-baseline for both arms. Neither output checkpoint is a
promotion candidate yet, and this program has no code path that writes
`current_best`.

## 6. Frozen-reference behavior gate

Rerun only the root ladder for each arm. The baseline eight-ply references are
loaded from the overnight artifacts and are never rebuilt or moved with the
student.

```bash
python -m games.kingdomino.reply_pilot_evaluation \
  --mode run --arm control \
  --checkpoint "$PILOT_DIR/training/control.pt" \
  --output-dir "$PILOT_DIR/evaluation"

python -m games.kingdomino.reply_pilot_evaluation \
  --mode run --arm treatment \
  --checkpoint "$PILOT_DIR/training/treatment.pt" \
  --output-dir "$PILOT_DIR/evaluation"

python -m games.kingdomino.reply_pilot_evaluation \
  --mode report \
  --control-ladder "$PILOT_DIR/evaluation/control_root_ladder.jsonl" \
  --treatment-ladder "$PILOT_DIR/evaluation/treatment_root_ladder.jsonl" \
  --output "$PILOT_DIR/evaluation/behavior_report.json"
```

Continue only when the report route is `proceed_to_bga_and_strength`. The
report enforces rank-specific median/p90 improvements, stable-flip limits,
rank-1 median-fragility and mean-Q anti-deflation guards, missing-Q and tie
guards, and a bounded seed-SD increase.

## 7. BGA, fixed-suite, and equal-compute strength gates

These commands are intentionally after the cheap anti-deflation gate:

```bash
python -m games.kingdomino.bga_denial_anchor \
  --checkpoint "$PILOT_DIR/training/control.pt" --device cuda \
  --out "$PILOT_DIR/evaluation/bga_control.json"

python -m games.kingdomino.bga_denial_anchor \
  --checkpoint "$PILOT_DIR/training/treatment.pt" --device cuda \
  --out "$PILOT_DIR/evaluation/bga_treatment.json"

python scripts/run_eval_suite.py \
  --checkpoint "$PILOT_DIR/training/control.pt" --device cuda \
  --out "$PILOT_DIR/evaluation/fixed_control.json" \
  --details_out "$PILOT_DIR/evaluation/fixed_control_details.jsonl"

python scripts/run_eval_suite.py \
  --checkpoint "$PILOT_DIR/training/treatment.pt" --device cuda \
  --out "$PILOT_DIR/evaluation/fixed_treatment.json" \
  --details_out "$PILOT_DIR/evaluation/fixed_treatment_details.jsonl"

python -m games.kingdomino.round_robin_eval \
  --checkpoints "$PILOT_DIR/training/control.pt" "$PILOT_DIR/training/treatment.pt" \
  --names control treatment --engine batched --open_loop \
  --device cuda --sims 300 --seeds_per_pair 1250 \
  --batch_slots 86 --leaf_batch 6 --amp_inference \
  --output "$PILOT_DIR/evaluation/control_vs_treatment.csv" \
  --leaderboard_output "$PILOT_DIR/evaluation/leaderboard.json" \
  --game_log_output "$PILOT_DIR/evaluation/games.jsonl"
```

Final routing remains the plan's rule: pass only if behavior gates pass, BGA
top-1 agreement is at least 70%, BGA median played-pick prior is at least 0.77,
the fixed suite does not regress, and the paired strength result is
non-inferior with a positive signal. Otherwise classify as measurement-only or
fail. Promotion is a separate, explicit future action.

## 8. Cost telemetry and shutdown

Record wall time and accepted-label count from each manifest. Compute:

```text
dollars per accepted label = hourly rental price * generation hours / accepted labels
```

The local real-shape benchmark was 111.56 seconds/tree in Python versus 6.29,
5.97, and 5.86 seconds with Rust at 1, 2, and 4 threads, respectively. That is
17.7–19.0× faster; the modest 1-to-4-thread gain shows that GPU inference is
already the dominant residual cost. Shut the rental down immediately after all
artifacts and manifests are copied off-box.
