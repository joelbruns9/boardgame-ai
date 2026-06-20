# Kingdomino Session Notes

## 2026-06-08 - Phase 0 Baseline V1 Locks

- Added `baseline_v1.json` as the frozen local baseline for M9 model/search
  and training pilots.
- Purpose: keep future experiments clean by changing one variable at a time.
- Locked architecture family:
  - `KingdominoNet`
  - shared my/opp board trunk
  - GroupNorm
  - avg+max pooled context
  - current bilinear full policy head
  - current value head
  - `bilinear_dim=64`
  - model-size variables are `channels` and `blocks`
- Locked scout model:
  - `32x4`
- Locked target/data semantics:
  - value target: `tanh(score_margin / 30)`
  - policy target: MCTS visit distribution
  - legal-only masked policy loss
  - current-actor perspective
  - D4 augmentation enabled
  - sparse policy/legal target storage
- Locked local inference defaults:
  - CUDA
  - TF32 allowed
  - AMP inference enabled
  - compile off
  - fixed-shape padding off
  - pinned transfer off
  - eval timing profiling off
- Locked self-play/search baseline:
  - engine: `batched`
  - `n_simulations=800`
  - `n_determinizations=1`
  - `batch_slots=86`
  - `leaf_batch=6`
  - `virtual_loss=1`
  - `c_puct=1.5`
  - `fpu=0.0`
  - `dirichlet_alpha=0.3`
  - `dirichlet_epsilon=0.25`
  - `temp_moves=20`
  - Rayon slot parallelism enabled
- Locked training defaults:
  - Adam
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `batch_size=256`
  - `train_steps_per_iteration=200`
  - `buffer_capacity=50000`
  - `value_weight=1.0`
  - `policy_weight=1.0`
  - `grad_clip=1.0`
  - augmentation enabled
  - checkpoint every iteration when `checkpoint_dir` is set
- Current built-in benchmark remains sanity-only:
  - greedy opponent
  - `benchmark_seeds=10`
  - `benchmark_sims=50`
- M10A-lite eval defaults are specified as a to-build lock:
  - fixed paired seeds
  - `eval_sims=800`
  - `games_per_match=200`
  - promotion requires convincing head-to-head win under fixed eval settings
- Default-off switches to retest on cloud:
  - compile
  - fixed-shape padding
  - pinned transfer
  - larger slots
  - larger leaf batch
  - legal/readback reductions

## 2026-06-09 - Baseline V1 32x4 Training Run

- Completed exploratory baseline training run:
  - command: `python -m games.kingdomino.self_play --device cuda --engine batched --channels 32 --blocks 4 --bilinear_dim 64 --sims 800 --batch_slots 86 --leaf_batch 6 --amp_inference --games_per_iter 192 --iterations 80 --train_steps 200 --batch_size 256 --lr 0.001 --buffer 50000 --checkpoint_dir checkpoints_baseline_v1_32x4_s800`
  - checkpoint dir: `checkpoints_baseline_v1_32x4_s800`
  - checkpoints: 80
  - wall time from first to last checkpoint: about 9.24 hours
  - mean time per iteration: about 7.02 minutes
- Extracted checkpoint history to:
  - `baseline_v1_32x4_s800_history.csv`
- Throughput:
  - iteration 1: 0.52 games/s, 18,473 evals/s
  - final iterations: about 0.59-0.60 games/s, about 21,650-21,730 evals/s
  - final timing split: roughly step 17%, eval 76%, update 7%
  - buffer reached capacity by iteration 6 and stayed at 50,000
- Loss trend:
  - value loss: 0.2348 at iter 1 -> 0.1622 at iter 80
  - best value loss: 0.1618 at iter 78
  - policy loss: 2.5533 at iter 1 -> 1.9102 at iter 80
  - best policy loss: 1.8926 at iter 78
- Built-in greedy sanity benchmark:
  - first >=90% win rate: iter 23
  - first 100% win rate: iter 35
  - final 10 iterations saturated at 95-100%
- Interpretation:
  - Full training loop is healthy.
  - Greedy benchmark is saturated and no longer useful for promotion.
  - Candidate checkpoints for M10A-lite evaluation should include iter 35,
    iter 50, iter 60, iter 70, iter 75, iter 78, and iter 80.

## 2026-06-07 - M6 BatchedMCTS N=1 Gate

- Fixed the partial BatchedMCTS Rust implementation so `cargo check` passes.
- Exported `BatchedMCTS`, `batched_det_seed`, and `batched_new_game` from the
  `kingdomino_rust` Python module.
- Added `test_batched_mcts_n1.py`.
- M6 N=1 mock gate passed:
  - `n_sims=50`: 288/288 positions bit-identical
  - `n_sims=200`: 288/288 positions bit-identical
  - `n_sims=800`: 288/288 positions bit-identical
- Regression results:
  - `test_rust_board_equiv --games=2000`: PASS
  - `test_rust_game_equiv --games=2000`: PASS
  - `test_rust_encode_equiv --games=1000`: PASS
- `test_rust_mcts_equiv` was not runnable in the available Python environment
  because importing `mcts_az.py` requires `torch`, which is not installed there.
  The stricter M6 N=1 test avoids that dependency by carrying its own md5 mock.

## 2026-06-07 - M7 BatchedMCTS N=32 Recycling Gate

- Added `test_batched_mcts_n32.py`.
- Deterministic recycling gate uses `n_slots=32`, `n_games=64`,
  `dirichlet_eps=0`, `temp_moves=0`, and compares every completed game's
  emitted policy targets against a `RustMCTS.search` replay.
- M7 deterministic results:
  - `n_sims=50`: PASS, 3328 positions, mean batch 151.3, max batch 192
  - `n_sims=200`: PASS, 3328 positions, mean batch 170.0, max batch 192
- Production-settings smoke uses `dirichlet_eps=0.25`, `temp_moves=20`,
  `n_sims=50`, and checks example shapes, finite targets, policy sums, and
  policy/legal-index consistency.
- Production-settings smoke result: PASS, 64 games, 3328 examples, mean batch
  152.1.

## 2026-06-07 - M8 Batched Engine Wiring

- Added `SelfPlayConfig.batch_slots` and `--engine batched --batch_slots N`.
- Added `play_selfplay_games_batched`, which drives one Rust `BatchedMCTS`,
  converts finished-game examples into the existing `Example` format, and
  reports batch stats (`mean_batch`, `fill_ratio`, `evals/s`, `max_batch_seen`).
- Wired `self_play.py --engine batched`.
- Wired `parallel_self_play.py --engine batched` as a single-worker path
  (`--workers 1`) using one synchronized `BatchedMCTS`.
- Added `throughput_bench.py --run batched --batch_slots N`.
- Added `production_probe.py --engine batched --batch_slots N`.
- Verification in this environment:
  - Python parse checks for the four modified scripts: PASS
  - `cargo check`: PASS
  - `test_batched_mcts_n32`: PASS
- Real-net smoke/performance gates still need to run in the project `.venv` or
  GPU environment because this tool environment cannot launch that venv and its
  bundled Python does not include `torch`.

### M8 Performance Probe Follow-up

- User-run `self_play.py --engine batched --sims 50 --games_per_iter 4` passed
  integration smoke on CUDA.
- User-run `throughput_bench.py --run batched --sims 800 --games 64` after
  release rebuild measured:
  - games/s: 0.258
  - mean_batch: 175.7/192
  - evals/s: 9121
- Interpretation: batch fill is healthy, but evals/s is below the M8 target.
  The likely overhead was copying the full `(B, 3390)` policy tensor to CPU each
  tick before slicing legal logits.
- Optimized `make_rust_evaluator` to gather legal logits on-device and transfer
  only the ragged legal logits back to CPU.
- Follow-up timing showed the on-device gather did not improve throughput and
  the run was forward-bound:
  - step: 29.5s
  - eval: 196.3s
  - update: 11.9s
  - evaluator split: h2d 7.8s, forward 175.0s, readback 12.3s
- Reverted the gather experiment and added CUDA inference knobs:
  - TF32 enabled by default for CUDA (`--no_tf32` disables it)
  - optional `--amp_inference` for float16 autocast during self-play inference
- Pure forward ceiling for 64ch/6b on the user's CUDA device:
  - TF32, batch 192: 13,440 evals/s
  - TF32, batch 256: 13,714 evals/s
  - AMP, batch 192: 15,549 evals/s
  - AMP, batch 256: 15,849 evals/s
- Conclusion: the M8 `evals/s >= 30,000` and `games/s >= 1.5` gates are not
  reachable with this net/GPU. BatchedMCTS is filling batches and running near
  the pure-forward ceiling; further gains require a faster/smaller/compiled
  model or different hardware, not more batching.
- Added `KingdominoNet.forward_legal` and a `forward_bench --legal_counts`
  option to test whether computing only legal policy logits is a meaningful
  inference-speed lever before wiring it into MCTS.

### Smaller-Net Batched Throughput Runs

- `48ch/4b`, `batch_slots=64`, `games=128`, `sims=800`, AMP:
  - games/s: 0.312
  - mean_batch: 348.7/384
  - evals/s: 10,939
  - timing: step 108.8s, eval 257.1s, update 43.4s
- `32ch/6b`, `batch_slots=64`, `games=128`, `sims=800`, AMP:
  - games/s: 0.311
  - mean_batch: 352.7/384
  - evals/s: 11,037
  - timing: step 118.2s, eval 250.9s, update 41.9s
- `32ch/6b`, `batch_slots=86`, `games=192`, `sims=800`, AMP:
  - games/s: 0.324
  - mean_batch: 352.0/516
  - evals/s: 11,481
  - timing: step 175.7s, eval 361.8s, update 55.5s
- Conclusion: smaller nets improve pure forward ceiling but live throughput only
  modestly improved because the single-threaded BatchedMCTS step/update cost
  grows with slot count and now accounts for roughly 39% of wall time.

### M8 Benchmark Matrix Runner

- Added `batched_matrix_bench.py` to run repeatable BatchedMCTS throughput
  matrices across model sizes, slot counts, and sim counts.
- The script writes CSV rows containing games/s, mean batch, fill, evals/s,
  max batch, ticks, and step/eval/update timing.
- Smoke check passed with a tiny CUDA run:
  - `32x4`, `slots=4`, `sims=5`, `games=2`, `leaf_batch=2`, AMP

### M8 Small Matrix Results

- Ran `batched_matrix_bench.py --models 32x6,48x4,64x6 --slots 32,64,86
  --sims 800 --games 64 --leaf_batch 6 --amp_inference`.
- Best row:
  - `48x4`, `batch_slots=64`: 0.372 games/s, 13,163 evals/s,
    mean batch 353.8/384, fill 92%.
- Ranking by games/s:
  - `48x4 slots=64`: 0.372 games/s
  - `48x4 slots=86`: 0.324 games/s
  - `32x6 slots=86`: 0.304 games/s
  - `48x4 slots=32`: 0.303 games/s
  - `32x6 slots=64`: 0.293 games/s
  - `32x6 slots=32`: 0.272 games/s
  - `64x6 slots=64`: 0.269 games/s
  - `64x6 slots=32`: 0.224 games/s
  - `64x6 slots=86`: 0.222 games/s
- Note: `slots=86` underfilled in the 64-game matrix because it can only start
  64 active games, so the slot cap is larger than the workload. Larger-slot
  tests need `games >= batch_slots` and preferably `games >= 2 * batch_slots`.

## 2026-06-07 - M8A Fair Benchmark Matrix

- The tool environment can run the project `.venv` when commands are launched
  with escalated permissions. Verified:
  - Python: 3.12.10
  - Torch: 2.12.0+cu126
  - CUDA available: true
- Ran the fair slot sweep:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 48x4,32x6 --slots 64,86,128 --sims 800 --games 192 --leaf_batch 6 --amp_inference --out m8_matrix_slots_fair.csv`
- Fair slot sweep results:
  - `32x6 slots=128`: 0.4575 games/s, 15,935 evals/s, mean batch 515.1/768, fill 67.1%
  - `32x6 slots=86`: 0.4504 games/s, 16,009 evals/s, mean batch 354.1/516, fill 68.6%
  - `32x6 slots=64`: 0.4346 games/s, 15,481 evals/s, mean batch 354.6/384, fill 92.4%
  - `48x4 slots=128`: 0.4236 games/s, 14,898 evals/s, mean batch 519.9/768, fill 67.7%
  - `48x4 slots=64`: 0.3973 games/s, 14,029 evals/s, mean batch 352.0/384, fill 91.7%
  - `48x4 slots=86`: 0.3897 games/s, 13,713 evals/s, mean batch 351.1/516, fill 68.0%
- Ran the missing-size sweep:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 32x4,48x6 --slots 64 --sims 800 --games 128 --leaf_batch 6 --amp_inference --out m8_matrix_missing_sizes.csv`
- Missing-size results:
  - `32x4 slots=64`: 0.4926 games/s, 17,411 evals/s, mean batch 352.1/384, fill 91.7%
  - `48x6 slots=64`: 0.3337 games/s, 11,793 evals/s, mean batch 351.2/384, fill 91.5%
- Combined ranking from these M8A runs:
  - `32x4 slots=64`: 0.4926 games/s
  - `32x6 slots=128`: 0.4575 games/s
  - `32x6 slots=86`: 0.4504 games/s
  - `32x6 slots=64`: 0.4346 games/s
  - `48x4 slots=128`: 0.4236 games/s
  - `48x4 slots=64`: 0.3973 games/s
  - `48x4 slots=86`: 0.3897 games/s
  - `48x6 slots=64`: 0.3337 games/s
- Timing split remains mostly neural forward:
  - Eval: roughly 64-74% wall time
  - Step/update combined: roughly 26-36% wall time
- Interpretation:
  - `32x4 slots=64` is the fastest raw self-play generator so far.
  - `32x6` scales better with larger slot counts than `48x4` in the fair run.
  - Step/update is large enough to justify M8C Rust slot parallelism testing,
    especially for smaller nets where forward is less dominant.

## 2026-06-07 - M8B Compile/Inference Optimization

- Added `--compile_net`, `--compile_backend`, and `--compile_mode` to:
  - `batched_matrix_bench.py`
  - `throughput_bench.py`
- Added `SelfPlayConfig.eval_pad_to_batch` and `--eval_pad_to_batch` so fixed
  shape CUDA graph experiments can pad live inference batches.
- Default `torch.compile` / Inductor is blocked in this Windows environment:
  - Failure: `torch._inductor.exc.TritonMissing`
  - Cause: no working Triton installation available to PyTorch.
- `torch.compile(backend="cudagraphs")` initially failed because live
  BatchedMCTS sends variable batch sizes. Padding to `slots * leaf_batch`
  makes CUDA graph replay shape-stable enough to run.
- Ran eager comparison:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 32x6,48x4,64x6 --slots 64 --sims 800 --games 64 --leaf_batch 6 --amp_inference --warmup 1 --out m8_compile_eager_slots64.csv`
- Ran CUDA graphs comparison:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 32x6,48x4,64x6 --slots 64 --sims 800 --games 64 --leaf_batch 6 --amp_inference --warmup 1 --compile_net --compile_backend cudagraphs --out m8_compile_cudagraphs_slots64.csv`
- Live BatchedMCTS compile comparison:
  - `32x6`: eager 0.3297 -> cudagraphs 0.3444 games/s (+4.5%)
  - `48x4`: eager 0.3159 -> cudagraphs 0.2788 games/s (-11.7%)
  - `64x6`: eager 0.2269 -> cudagraphs 0.2279 games/s (+0.5%)
- Recommendation:
  - Do not make compile the default.
  - CUDA graphs are experimental and model/config sensitive.
  - Re-test Inductor only after installing a working Triton stack or moving to a
    Linux/cloud environment where PyTorch CUDA compile is fully supported.
  - Next project-plan item is M8C: parallelize Rust BatchedMCTS per-slot
    step/update with Rayon and keep the M6/M7 correctness gates passing.

## 2026-06-07 - M8C Rust Slot Parallelism

- Added Rayon to the Rust extension.
- Parallelized BatchedMCTS per-slot work:
  - `step()` now runs slot-local descents, virtual-loss application, legal move
    generation, and state encoding in parallel.
  - Per-slot batch fragments are merged back in ascending slot order, preserving
    deterministic row order for evaluator inputs.
  - `update()` now runs slot-local root expansion, leaf expansion, virtual-loss
    removal, backup, and move finalization in parallel.
  - Finished-game recycling remains sequential in ascending slot order so new
    seed assignment and returned finished-game ordering match the old driver.
- Rebuilt and installed the updated Rust extension:
  - `maturin develop --release`
- Correctness gates after Rayon:
  - `python -m games.kingdomino.test_batched_mcts_n1`: PASS
    - `n_sims=50`: 288/288 bit-identical positions
    - `n_sims=200`: 288/288 bit-identical positions
    - `n_sims=800`: 288/288 bit-identical positions
  - `python -m games.kingdomino.test_batched_mcts_n32`: PASS
    - `n_sims=50`: PASS, 3328 positions, ticks=1040, mean batch 151.3, max batch 192
    - `n_sims=200`: PASS, 3328 positions, ticks=3640, mean batch 170.0, max batch 192
    - production smoke: PASS, 64 games, 3328 examples, ticks=1040, mean batch 152.1
- Performance gate:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 48x4 --slots 64,86 --sims 800 --games 192 --leaf_batch 6 --amp_inference --out m8c_rayon_48x4_slots64_86.csv`
- Post-Rayon comparison against `m8_matrix_slots_fair.csv`:
  - `48x4 slots=64`: 0.3973 -> 0.4319 games/s (+8.7%)
    - step: 99.7s -> 56.1s
    - update: 38.3s -> 24.8s
    - eval: 344.8s -> 363.0s
  - `48x4 slots=86`: 0.3897 -> 0.3920 games/s (+0.6%)
    - step: 109.9s -> 77.7s
    - update: 38.1s -> 30.0s
    - eval: 344.1s -> 381.2s
- Interpretation:
  - M8C succeeded as a step/update optimization.
  - `slots=64` shows a meaningful live throughput improvement.
  - `slots=86` still does not improve much overall because the run is now more
    strongly eval-bound and lower fill remains a drag.
  - With Rayon, the next bottleneck for strong configs is again neural forward;
    for model/search selection, prioritize `32x4`, `32x6`, and `48x4 slots=64`
    candidates before spending more time on CPU-side slot work.

## 2026-06-07 - M8D Fixed-Shape Padding Test

- Tested whether padding live inference batches to a fixed shape helps even
  without `torch.compile`.
- Initial three-model A/B:
  - No padding:
    - `32x4 slots=64`: 0.4421 games/s
    - `32x6 slots=64`: 0.3759 games/s
    - `48x4 slots=64`: 0.3635 games/s
  - `eval_pad_to_batch=384`:
    - `32x4 slots=64`: 0.4965 games/s
    - `32x6 slots=64`: 0.3886 games/s
    - `48x4 slots=64`: 0.4047 games/s
- That first A/B was confounded by run-to-run drift: padding should mostly
  affect evaluator time, but `step_sec` and `update_sec` also changed sharply.
- Ran back-to-back confirmation checks:
  - `32x4`, no padding: 0.5561 games/s, eval 180.6s, step+update 49.1s
  - `32x4`, pad 384: 0.5064 games/s, eval 204.5s, step+update 47.9s
  - Result: padding was -8.9%.
  - `48x4`, no padding: 0.4824 games/s, eval 222.2s, step+update 42.7s
  - `48x4`, pad 384: 0.3873 games/s, eval 284.4s, step+update 45.8s
  - Result: padding was -19.7%.
- Conclusion:
  - Do not use fixed-shape padding by default without compile.
  - Shape stability did not pay for the dummy-row forward cost.
  - Keep padding only as an enabling option for CUDA graph experiments.
  - The next throughput item to test is pinned CPU/GPU transfer, ideally using
    `32x4 slots=64` and `48x4 slots=64` back-to-back to avoid the drift seen here.

## 2026-06-08 - M8E Pinned CPU/GPU Transfer Test

- Added evaluator option `pin_transfer`:
  - Stages `mb`, `ob`, and `flat` tensors in pinned host memory.
  - Copies to CUDA with `non_blocking=True`.
- Added explicit `profile_eval_timing` option:
  - Synchronizes around evaluator H2D, forward, and readback only when requested.
  - Kept profiling off by default so normal training/benchmark runs do not pay
    extra synchronization overhead.
- Added CLI flags and CSV columns:
  - `--pin_transfer`
  - `--profile_eval_timing`
  - `pin_transfer`, `profile_eval_timing`, `eval_h2d_sec`,
    `eval_forward_sec`, `eval_readback_sec`, `eval_calls`
- Profiling A/B showed pinned transfer does reduce measured H2D:
  - `32x4`: H2D 12.6s -> 10.3s
  - `48x4`: H2D 12.1s -> 10.3s
- However, no-profile live throughput did not improve reliably:
  - `32x4 slots=64`: no pin 0.6146 games/s, pinned 0.5891 games/s (-4.1%)
  - `48x4 slots=64`: no pin 0.4607 games/s, pinned 0.4636 games/s (+0.6%)
- Interpretation:
  - H2D transfer is visible but not the dominant limiter.
  - The extra CPU pinned staging copy and run noise offset the H2D savings.
  - Do not enable pinned transfer by default for current BatchedMCTS.
  - Keep `--pin_transfer` as a diagnostic/architecture option, especially for
    future larger batches, different hardware, or if the evaluator starts
    building pinned buffers directly from Rust without the extra CPU copy.

## 2026-06-08 - M8 Final Post-Rayon Matrix and Repeat Check

- Ran final post-Rayon mini-matrix:
  - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 32x4,32x6,48x4 --slots 64,86,128 --sims 800 --games 192 --leaf_batch 6 --amp_inference --out m8_final_post_rayon_matrix.csv`
- Matrix ranking by games/s:
  - `32x4 slots=86`: 0.5708 games/s, 20,298 evals/s
  - `32x4 slots=128`: 0.5674 games/s, 20,218 evals/s
  - `32x4 slots=64`: 0.5512 games/s, 19,682 evals/s
  - `32x6 slots=128`: 0.4521 games/s, 15,877 evals/s
  - `32x6 slots=86`: 0.4380 games/s, 15,431 evals/s
  - `32x6 slots=64`: 0.4364 games/s, 15,361 evals/s
  - `48x4 slots=128`: 0.4216 games/s, 15,006 evals/s
  - `48x4 slots=86`: 0.4122 games/s, 14,587 evals/s
  - `48x4 slots=64`: 0.4057 games/s, 14,462 evals/s
- Ran repeat stability checks on top two exact configs:
  - `32x4 slots=86`: `--repeat 3 --warmup 1`, output `m8_final_repeat_32x4_slots86.csv`
  - `32x4 slots=128`: `--repeat 3 --warmup 1`, output `m8_final_repeat_32x4_slots128.csv`
- Repeat summary:
  - `32x4 slots=86`:
    - reps: 0.5652, 0.5759, 0.5676 games/s
    - mean 0.5696, median 0.5759, CV 0.8%
    - mean evals/s 19,990
    - mean timing: step 45.2s, eval 270.8s, update 20.6s
  - `32x4 slots=128`:
    - reps: 0.5719, 0.5725, 0.5413 games/s
    - mean 0.5619, median 0.5725, CV 2.6%
    - mean evals/s 19,874
    - mean timing: step 56.7s, eval 264.6s, update 20.1s
- M8 conclusion:
  - Best current throughput baseline is `32x4`, `batch_slots=86`,
    `leaf_batch=6`, `sims=800`, AMP inference.
  - `slots=128` is close but less stable and has higher step cost.
  - `slots=64` is slightly slower but remains a simple, high-fill fallback.
  - M8 throughput engineering is good enough to move into M9 model/search
    scaling. The open question is now strength per wall-clock hour, not raw
    self-play throughput.

## 2026-06-09 - M10A-lite Mixed-Architecture Evaluation Ladder

- Confirmed prior checkpoints can be evaluated against the new 32x4 baseline
  even when model sizes differ, as long as each checkpoint is loaded with its
  own architecture.
- Verified checkpoint configs:
  - `checkpoints_baseline_v1_32x4_s800/iter_0080.pt`: `32x4`,
    `bilinear_dim=64`, engine `batched`, `batch_slots=86`.
  - `checkpoints/iter_0030.pt`: `64x6`, `bilinear_dim=64`.
  - `checkpoints_scratch_s800_64x6/iter_0040.pt`: `64x6`,
    `bilinear_dim=64`.
  - `checkpoints_warm_s800_64x6/iter_0035.pt`: `64x6`,
    `bilinear_dim=64`.
  - `checkpoints_rust_s800_64x6/iter_0050.pt`: `64x6`,
    `bilinear_dim=64`, engine `rust`.
- Updated `round_robin_eval.py`:
  - Added `--manifest`.
  - Manifest CSV columns: `name,path,channels,blocks,bilinear_dim`.
  - Per-row architecture values are optional but supported, so mixed-size
    checkpoints do not need unsafe global `--channels/--blocks` overrides.
  - Relative manifest paths are resolved relative to the manifest first, then
    relative to the current working directory.
- Added initial ladder manifest:
  - `eval_manifests/m10a_baseline_v1_ladder.csv`
  - Initial broad draft included seven 32x4 checkpoints and four older 64x6
    checkpoints.
  - Trimmed active ladder after review:
    - 32x4 baseline: `i035`, `i050`, `i065`, `i078`, `i080`.
    - 64x6 prior: rust `i050`.
- Validation:
  - `python -m py_compile games/kingdomino/round_robin_eval.py`: PASS.
  - Smoke mixed-architecture eval:
    - `python -m games.kingdomino.round_robin_eval --manifest eval_manifests/m10a_baseline_v1_ladder.csv --device cuda --sims 1 --games_per_pair 2 --seed 424242 --output eval_results/m10a_manifest_smoke_pairs.csv --leaderboard_output eval_results/m10a_manifest_smoke_leaderboard.csv --game_log_output eval_results/m10a_manifest_smoke_games.csv`
    - PASS: 11 participants, 55 pairings, 110 games.
    - Output files:
      - `eval_results/m10a_manifest_smoke_pairs.csv`
      - `eval_results/m10a_manifest_smoke_leaderboard.csv`
      - `eval_results/m10a_manifest_smoke_games.csv`
- Trimmed manifest validation:
  - `python -m games.kingdomino.round_robin_eval --manifest eval_manifests/m10a_baseline_v1_ladder.csv --device cuda --sims 1 --games_per_pair 2 --seed 424242 --output eval_results/m10a_manifest_smoke_trimmed_pairs.csv --leaderboard_output eval_results/m10a_manifest_smoke_trimmed_leaderboard.csv --game_log_output eval_results/m10a_manifest_smoke_trimmed_games.csv`
  - PASS: 6 participants, 15 pairings, 30 games.
  - Output files:
    - `eval_results/m10a_manifest_smoke_trimmed_pairs.csv`
    - `eval_results/m10a_manifest_smoke_trimmed_leaderboard.csv`
    - `eval_results/m10a_manifest_smoke_trimmed_games.csv`
- Interpretation:
  - Smoke standings are not strength evidence because `sims=1` and only two
    games per pair.
  - The important gate is that old 64x6 and new 32x4 checkpoints can now share
    the same paired-seed evaluation harness.
  - Next strength step should use this manifest with a smaller candidate subset
    or a low/medium-sim ladder before spending time on a full `sims=800`
    tournament.

## 2026-06-09 - M10A Top-3 Strength Ladder

- Ran a focused top-3 paired round-robin:
  - `baseline32_i080`: `checkpoints_baseline_v1_32x4_s800/iter_0080.pt`
  - `baseline32_i065`: `checkpoints_baseline_v1_32x4_s800/iter_0065.pt`
  - `prior64_rust_i050`: `checkpoints_rust_s800_64x6/iter_0050.pt`
- Command:
  - `python -m games.kingdomino.round_robin_eval --checkpoints checkpoints_baseline_v1_32x4_s800/iter_0080.pt checkpoints_baseline_v1_32x4_s800/iter_0065.pt checkpoints_rust_s800_64x6/iter_0050.pt --names baseline32_i080 baseline32_i065 prior64_rust_i050 --device cuda --sims 100 --games_per_pair 300 --seed 20260609 --output eval_results/m10a_top3_s100_g300_pairs.csv --leaderboard_output eval_results/m10a_top3_s100_g300_leaderboard.csv --game_log_output eval_results/m10a_top3_s100_g300_games.csv`
- Pair results:
  - `baseline32_i080` beat `baseline32_i065`: `184-109-7`,
    62.5% score, +5.60 avg margin.
  - `baseline32_i080` beat `prior64_rust_i050`: `208-87-5`,
    70.2% score, +10.94 avg margin.
  - `baseline32_i065` beat `prior64_rust_i050`: `189-109-2`,
    63.3% score, +7.37 avg margin.
- Leaderboard:
  - `baseline32_i080`: 600 games, `392-196-12`, 66.3%, +8.27 avg margin,
    124.09 avg score.
  - `baseline32_i065`: 600 games, `298-293-9`, 50.4%, +0.89 avg margin,
    120.66 avg score.
  - `prior64_rust_i050`: 600 games, `196-397-7`, 33.2%, -9.15 avg margin,
    116.65 avg score.
- Runtime:
  - 900 games in 17,201.9s, 0.052 games/sec.
- Interpretation:
  - `baseline32_i080` is the current promoted best under the `sims=100`
    paired evaluation setting.
  - The smaller 32x4 model has convincingly surpassed the prior 64x6 rust
    checkpoint at this evaluation level.
  - `baseline32_i065` also beats the prior 64x6 checkpoint, but is clearly
    behind `baseline32_i080`.
  - Next evaluation work should be a focused high-sim confirmation match, not
    another broad ladder.

## 2026-06-09 - Promotion Bookkeeping and Batched Eval

- Promotion bookkeeping:
  - Promoted current best:
    `checkpoints_baseline_v1_32x4_s800/iter_0080.pt`.
  - Canonical current-best copy:
    `checkpoints_best/kingdomino_current_best.pt`.
  - Metadata:
    `checkpoints_best/kingdomino_current_best.json`.
  - SHA256 for source and promoted copy:
    `F6597A44CC551B9386ECDB59C8EB66D9AB56CC9FCA111748F9D1976175FA6305`.
  - Added quick manifest:
    `eval_manifests/current_best_vs_prior.csv`.
  - Updated `web_app.py` checkpoint autodiscovery to prefer
    `checkpoints_best/kingdomino_current_best.pt` when present.
- Batched round-robin eval implementation:
  - Added Rust `BatchedMCTS.row_actors()` accessor.
    - Non-breaking: existing `step()` return shape is unchanged.
    - Purpose: after `step()`, Python can route each eval row to the network
      for actor 0 or actor 1.
  - Added `round_robin_eval.py --engine batched`.
    - Supports checkpoint-vs-checkpoint pairs.
    - Uses Rust `BatchedMCTS` for each orientation.
    - Preserves paired seeds and existing pair/leaderboard/game-log CSV
      outputs.
    - Current constraints: `determinizations=1`, `temperature=0`, checkpoint
      participants only.
  - Added batched CLI knobs:
    - `--batch_slots`
    - `--leaf_batch`
    - `--amp_inference`
- Validation:
  - `cargo check`: PASS, existing warnings only.
  - `maturin develop --release`: PASS.
  - `python -m py_compile games.kingdomino.round_robin_eval.py games.kingdomino.web_app.py`: PASS.
  - Rust accessor smoke:
    - `BatchedMCTS.row_actors()` is available from Python.
  - Batched eval smoke:
    - `python -m games.kingdomino.round_robin_eval --manifest eval_manifests/current_best_vs_prior.csv --engine batched --device cuda --sims 1 --games_per_pair 2 --batch_slots 4 --leaf_batch 2 --amp_inference --seed 20260611 --output eval_results/m10a_batched_smoke_pairs.csv --leaderboard_output eval_results/m10a_batched_smoke_leaderboard.csv --game_log_output eval_results/m10a_batched_smoke_games.csv`
    - PASS: 2 games, current best vs prior.
  - Batched timing probe:
    - `python -m games.kingdomino.round_robin_eval --manifest eval_manifests/current_best_vs_prior.csv --engine batched --device cuda --sims 100 --games_per_pair 20 --batch_slots 16 --leaf_batch 6 --amp_inference --seed 20260611 --output eval_results/m10a_batched_timing_s100_g20_pairs.csv --leaderboard_output eval_results/m10a_batched_timing_s100_g20_leaderboard.csv --game_log_output eval_results/m10a_batched_timing_s100_g20_games.csv`
    - Result: current best beat prior `14-5-1`, 72.5%, +5.45 avg margin.
    - Throughput: 20 games in 18.9s, 1.061 games/sec.
- Interpretation:
  - Serial top-3 eval ran at about 0.052 games/sec.
  - Batched checkpoint-vs-checkpoint eval timing probe ran at about
    1.061 games/sec, roughly a 20x improvement for this focused matchup.
  - Batched eval is now good enough to use for M9 size/search scaling
    promotion checks.

## 2026-06-09 - M9 Equal-Wall-Clock Calibration Benchmarks

- Goal:
  - Compare future model-size pilots by wall-clock time, not equal iteration
    count.
  - Calibrate `32x6` and `48x4` against the promoted `32x4` baseline.
- BatchedMCTS throughput matrix:
  - Command:
    - `python -m games.kingdomino.batched_matrix_bench --device cuda --models 32x4,32x6,48x4 --slots 86 --sims 800 --games 192 --leaf_batch 6 --amp_inference --out m9_equal_wallclock_calibration.csv`
  - Results:
    - `32x4`: 0.5515 games/sec, 19,621 evals/sec, elapsed 347.8s.
      - timing: step 55.6s, eval 269.6s, update 22.5s.
    - `32x6`: 0.4388 games/sec, 15,521 evals/sec, elapsed 437.2s.
      - timing: step 53.2s, eval 363.6s, update 20.3s.
    - `48x4`: 0.4161 games/sec, 14,672 evals/sec, elapsed 461.1s.
      - timing: step 48.3s, eval 392.6s, update 20.0s.
  - Relative to `32x4`:
    - `32x6` self-play throughput is about 79.6% as fast.
    - `48x4` self-play throughput is about 75.5% as fast.
- One-iteration training smokes:
  - `32x6` command:
    - `python -m games.kingdomino.self_play --device cuda --engine batched --channels 32 --blocks 6 --bilinear_dim 64 --sims 800 --batch_slots 86 --leaf_batch 6 --amp_inference --games_per_iter 192 --iterations 1 --train_steps 200 --batch_size 256 --lr 0.001 --buffer 50000 --checkpoint_dir checkpoints_m9_smoke_32x6`
  - `32x6` result:
    - total wall time: 567.4s.
    - self-play: 192 games at 0.43 games/sec.
    - batched timing: step 47.5s, eval 375.2s, update 20.5s.
    - train: value_loss 0.2209, policy_loss 2.6105.
  - `48x4` command:
    - `python -m games.kingdomino.self_play --device cuda --engine batched --channels 48 --blocks 4 --bilinear_dim 64 --sims 800 --batch_slots 86 --leaf_batch 6 --amp_inference --games_per_iter 192 --iterations 1 --train_steps 200 --batch_size 256 --lr 0.001 --buffer 50000 --checkpoint_dir checkpoints_m9_smoke_48x4`
  - `48x4` result:
    - total wall time: 563.0s.
    - self-play: 192 games at 0.42 games/sec.
    - batched timing: step 43.8s, eval 390.7s, update 20.5s.
    - train: value_loss 0.2122, policy_loss 2.6065.
- Equal-time schedule against existing promoted `32x4` run:
  - Promoted `32x4` run was 80 iterations from about 2026-06-08 21:51 to
    2026-06-09 07:05, approximately 9.24 hours.
  - Using one-iteration smoke wall times:
    - `32x6`: 9.24h / 567.4s = about 58.6 iterations.
    - `48x4`: 9.24h / 563.0s = about 59.1 iterations.
  - Recommended equal-wall-clock pilots:
    - `32x6`: 58 or 59 iterations.
    - `48x4`: 59 iterations.
  - If using a fresh 10-hour budget instead:
    - `32x6`: about 63 iterations.
    - `48x4`: about 64 iterations.
- Interpretation:
  - Do not train `32x6` or `48x4` for 80 iterations if the comparison target
    is the existing `32x4` current best; that would give them materially more
    wall-clock compute.
  - `32x6` is still the cleaner next single-variable scout, but the pilot
    length should be about 59 iterations for a fair comparison to current best.

## 2026-06-09 - M9 Training Queue

- Added queued PowerShell runner:
  - `scripts/run_m9_training_queue.ps1`
- Purpose:
  - Run `32x6` and `48x4` pilots back-to-back without manual intervention.
  - Each run uses 80 iterations, but `iter_0060` should be treated as the
    approximate equal-wall-clock comparison point against the promoted `32x4`
    current best.
  - `iter_0080` is an extra-compute continuation checkpoint, useful for seeing
    whether the larger/deeper model keeps improving with more wall-clock time.
- Output folders:
  - `checkpoints_m9_32x6_s800`
  - `checkpoints_m9_48x4_s800`
- Logs:
  - `logs/m9_32x6_s800.log`
  - `logs/m9_48x4_s800.log`
- Expected runtime:
  - About 12.5 hours per 80-iteration run.
  - About 25 hours total if both complete.
- Added dedicated second-model-size runner:
  - `scripts/run_m9_48x4_training.ps1`
  - Runs the `48x4` M9 pilot.
  - Defaults to 80 iterations and `checkpoints_m9_48x4_s800`.
  - Refuses to overwrite an existing checkpoint folder unless `-Force` is
    passed.

## 2026-06-10 - M9 Training Run Logs Received

- User provided logs:
  - `logs/m9_32x6_s800.log`
  - `logs/m9_48x4_s800.log`
- Checkpoint availability:
  - `checkpoints_m9_32x6_s800` contains checkpoints through `iter_0077.pt`.
  - `checkpoints_m9_48x4_s800` contains checkpoints through `iter_0080.pt`.
- Note on logs:
  - `m9_32x6_s800.log` text output is truncated during `Iteration 71/80`,
    likely from restart/log interruption.
  - The `32x6` checkpoint metadata confirms `iter_0077.pt` is valid and has
    77 value-loss, policy-loss, and benchmark entries.
  - `m9_48x4_s800.log` is complete through `Iteration 80/80`.
- `32x6` checkpoint metadata:
  - `iter_0060.pt`:
    - last losses: value 0.1670, policy 1.9347.
    - min losses to that point: value 0.1627 at iter 59, policy 1.9347 at
      iter 60.
    - benchmark vs Greedy: 100%.
  - `iter_0077.pt`:
    - last losses: value 0.1628, policy 1.9092.
    - min losses to that point: value 0.1585 at iter 72, policy 1.9023 at
      iter 76.
    - benchmark vs Greedy: 100%.
- `48x4` checkpoint metadata:
  - `iter_0060.pt`:
    - last losses: value 0.1565, policy 2.0115.
    - min losses to that point: value 0.1551 at iter 59, policy 2.0115 at
      iter 60.
    - benchmark vs Greedy: 100%.
  - `iter_0080.pt`:
    - last losses: value 0.1502, policy 1.9866.
    - min losses to that point: value 0.1489 at iter 72, policy 1.9739 at
      iter 74.
    - benchmark vs Greedy: 100%.
- Evaluation candidates:
  - Added `eval_manifests/m9_size_search_candidates.csv`.
  - Includes:
    - promoted current best `32x4 i080`
    - `32x6 i060` fair-wall-clock checkpoint
    - `32x6 i077` late/extra-compute checkpoint
    - `48x4 i060` fair-wall-clock checkpoint
    - `48x4 i080` late/extra-compute checkpoint
- Interpretation:
  - Fair wall-clock model-size comparison should use `i060` for both larger
    candidates.
  - Late checkpoints are useful as extra-compute continuation signals.
  - Greedy benchmark is saturated for both runs and should not guide promotion.
- Manifest smoke:
  - `python -m games.kingdomino.round_robin_eval --manifest eval_manifests/m9_size_search_candidates.csv --engine batched --device cuda --sims 1 --games_per_pair 2 --batch_slots 8 --leaf_batch 2 --amp_inference --seed 20260610 --output eval_results/m9_size_search_smoke_pairs.csv --leaderboard_output eval_results/m9_size_search_smoke_leaderboard.csv --game_log_output eval_results/m9_size_search_smoke_games.csv`
  - PASS: 5 participants, 10 pairings, 20 games.
  - Smoke standings are not strength evidence.

## 2026-06-10 - Overnight Eval-Then-Train Runner

- Added script:
  - `scripts/run_m9_eval_then_train.ps1`
- Behavior:
  - Runs the full M9 size-search batched eval using
    `eval_manifests/m9_size_search_candidates.csv`.
  - Default eval settings:
    - `sims=100`
    - `games_per_pair=300`
    - `batch_slots=86`
    - `leaf_batch=6`
    - AMP inference enabled.
  - Parses the pair CSV after eval.
  - Compares fair-wall-clock checkpoints only against current best:
    - `candidate32x6_i060`
    - `candidate48x4_i060`
  - If either fair checkpoint scores at least 55% against current best, starts
    the next training pilot using the best passing architecture.
  - If neither passes, starts the next training pilot with the current-best
    `32x4` architecture.
  - Default next training:
    - `sims=1600`
    - 40 iterations
    - 192 games/iter
    - same batch/training defaults as baseline.
- Rationale:
  - Lets the size-search eval finish unattended and immediately move into the
    next planned search-scaling training pilot.
  - Keeps the auto-decision based on equal-wall-clock checkpoints, not the
    extra-compute late checkpoints.

## 2026-06-11 - Overnight M9 Eval and Sims-1600 Pilot Results

- M9 size-search eval completed:
  - Pair CSV: `eval_results/m9_size_search_s100_g300_pairs.csv`
  - Leaderboard: `eval_results/m9_size_search_s100_g300_leaderboard.csv`
  - Game log: `eval_results/m9_size_search_s100_g300_games.csv`
  - Log: `logs/m9_size_search_s100_g300.log`
  - Settings:
    - 5 participants, 10 pairings, 300 games/pair.
    - `sims=100`, batched eval, `batch_slots=86`, `leaf_batch=6`.
  - Runtime:
    - 3000 games in 1036.9s, 2.893 games/sec.
- Eval leaderboard:
  - `current_best_32x4_i080`: 1200 games, 59.0%, +4.823 avg margin,
    121.952 avg score.
  - `candidate32x6_i077`: 1200 games, 53.0%, +1.597 avg margin,
    119.389 avg score.
  - `candidate48x4_i080`: 1200 games, 51.3%, +1.038 avg margin,
    117.943 avg score.
  - `candidate32x6_i060`: 1200 games, 45.2%, -3.152 avg margin,
    116.133 avg score.
  - `candidate48x4_i060`: 1200 games, 41.4%, -4.305 avg margin,
    115.268 avg score.
- Current best pair results:
  - Current best beat `32x6_i060`: 179-119-2, 60.0%, +6.583 margin.
  - Current best beat `32x6_i077`: 175-119-6, 59.3%, +4.250 margin.
  - Current best beat `48x4_i060`: 188-106-6, 63.7%, +7.100 margin.
  - Current best beat `48x4_i080`: 157-139-4, 53.0%, +1.360 margin.
- Interpretation:
  - No larger-model fair checkpoint cleared the auto-promotion threshold.
  - `32x4_i080` remains current best and remains best per equal wall-clock.
  - Late `48x4_i080` got closest to current best, but still lost at this eval
    setting.
  - `32x6_i077` was the strongest larger-model challenger overall and beat
    both `48x4` checkpoints head-to-head.
- Automatic next training:
  - The script selected the current-best `32x4` architecture for the sims sweep.
  - Output folder:
    `checkpoints_m9_autonext_32x4_s1600_20260610_224322`
  - Log:
    `logs/m9_autonext_32x4_s1600_20260610_224322.log`
  - Completed `40/40` iterations.
  - Settings:
    - `32x4`, `sims=1600`, batched self-play, `batch_slots=86`,
      `leaf_batch=6`, AMP inference.
  - Checkpoint:
    - `checkpoints_m9_autonext_32x4_s1600_20260610_224322/iter_0040.pt`
- Sims-1600 pilot summary:
  - Iteration 1:
    - value_loss 0.2518, policy_loss 2.5466, Greedy 0%.
  - Iteration 40:
    - value_loss 0.1749, policy_loss 1.9524, Greedy 95%.
  - Min losses:
    - value_loss 0.1725 at iter 8.
    - policy_loss 1.9524 at iter 40.
  - Greedy sanity benchmark:
    - first >=90% at iter 19.
    - first 100% at iter 23.
  - Throughput:
    - self-play about 0.27-0.28 games/sec.
    - typical batched eval throughput about 19.5k-20.0k evals/sec.
  - Interpretation:
  - Fixed `sims=1600` from scratch is viable but roughly halves self-play
    throughput compared with `sims=800`.
  - It reached Greedy saturation earlier than the original `32x4 @ sims=800`
    run, but Greedy is not a promotion metric.
  - Next required check is head-to-head batched eval:
    - `sims1600_32x4_i040` vs `current_best_32x4_i080`.

## 2026-06-11 - Sims-1600 Pilot Eval

- Ran focused batched eval:
  - Current best:
    `checkpoints_best/kingdomino_current_best.pt`
  - Candidate:
    `checkpoints_m9_autonext_32x4_s1600_20260610_224322/iter_0040.pt`
  - Command:
    - `python -m games.kingdomino.round_robin_eval --checkpoints checkpoints_best/kingdomino_current_best.pt checkpoints_m9_autonext_32x4_s1600_20260610_224322/iter_0040.pt --names current_best_32x4_s800_i080 candidate32x4_s1600_i040 --engine batched --device cuda --sims 100 --games_per_pair 300 --batch_slots 86 --leaf_batch 6 --amp_inference --seed 20260611 --output eval_results/m9_s1600_i040_vs_current_s100_g300_pairs.csv --leaderboard_output eval_results/m9_s1600_i040_vs_current_s100_g300_leaderboard.csv --game_log_output eval_results/m9_s1600_i040_vs_current_s100_g300_games.csv`
- Result:
  - `current_best_32x4_s800_i080` beat `candidate32x4_s1600_i040`:
    `179-115-6`, 60.7%, +7.49 avg margin.
  - Runtime: 300 games in 90.0s, 3.332 games/sec.
- Interpretation:
  - The scratch `32x4 @ sims=1600` 40-iteration pilot did not beat the
    current `32x4 @ sims=800` best.
  - At this checkpoint, higher sims did not compensate for fewer generated
    training iterations/examples under wall-clock constraints.
  - Current best remains `checkpoints_best/kingdomino_current_best.pt`.
  - Next sim-count work should not assume fixed `sims=1600` from scratch is
    better; if testing higher sims further, prefer either longer/equal-example
    comparisons or a staged/ramped continuation experiment.

## 2026-06-11 - M9 Hyperparameter Scout Queue

- Added script:
  - `scripts/run_m9_hparam_scouts.ps1`
- Purpose:
  - Run two controlled `32x4 @ sims=800` hyperparameter scouts back-to-back.
  - Keep one variable changed per run.
- Run 1: larger replay buffer
  - Checkpoint dir: `checkpoints_m9_hparam_buffer100k_i40`
  - Log: `logs/m9_hparam_buffer100k_i40.log`
  - Settings:
    - `buffer=100000`
    - `train_steps=200`
    - 40 iterations
    - all other baseline settings unchanged.
- Run 2: more training steps
  - Checkpoint dir: `checkpoints_m9_hparam_trainsteps400_i40`
  - Log: `logs/m9_hparam_trainsteps400_i40.log`
  - Settings:
    - `buffer=50000`
    - `train_steps=400`
    - 40 iterations
    - all other baseline settings unchanged.
- Script behavior:
  - Stops if a run fails.
  - Refuses to reuse a checkpoint directory that already contains checkpoints
    unless `-Force` is passed.
- Expected runtime:
  - Buffer test: roughly 4.5-5 hours.
  - Train-steps test: likely somewhat longer because training work doubles;
    estimate roughly 5.5-6.5 hours.

## 2026-06-11 - M9 Hyperparameter Scout Results

- Completed both queued `32x4 @ sims=800` hparam scout runs.
- Run 1: larger replay buffer
  - Checkpoint:
    `checkpoints_m9_hparam_buffer100k_i40/iter_0040.pt`
  - Log:
    `logs/m9_hparam_buffer100k_i40.log`
  - Settings:
    - `buffer=100000`
    - `train_steps=200`
  - Iteration 40:
    - self-play `0.55` games/sec.
    - mean batch `354.2/516`, fill `69%`.
    - `19998` evals/sec.
    - timing: step `70.2s` / eval `256.0s` / update `24.6s`.
    - value loss `0.1935`.
    - policy loss `2.1881`.
    - Greedy benchmark `90.0%` (`18-0-2`).
  - First Greedy `>=90%`: iter 25.
  - First Greedy `100%`: none by iter 40.
- Run 2: more train steps
  - Checkpoint:
    `checkpoints_m9_hparam_trainsteps400_i40/iter_0040.pt`
  - Log:
    `logs/m9_hparam_trainsteps400_i40.log`
  - Settings:
    - `buffer=50000`
    - `train_steps=400`
  - Iteration 40:
    - self-play `0.44` games/sec.
    - mean batch `354.3/516`, fill `69%`.
    - `16154` evals/sec.
    - timing: step `75.2s` / eval `330.6s` / update `28.6s`.
    - value loss `0.1517`.
    - policy loss `2.1319`.
    - Greedy benchmark `95.0%` (`19-0-1`).
  - First Greedy `>=90%`: iter 35.
  - First Greedy `100%`: iter 38.
- Same-iteration baseline reference:
  - Checkpoint:
    `checkpoints_baseline_v1_32x4_s800/iter_0040.pt`
  - value loss `0.1752`.
  - policy loss `2.0815`.
  - Greedy benchmark `90.0%`.
  - First Greedy `>=90%`: iter 23.
  - First Greedy `100%`: iter 35.
- Added eval manifest:
  - `eval_manifests/m9_hparam_scout_candidates.csv`
- Ran batched hparam eval:
  - Command settings:
    - `sims=100`
    - `games_per_pair=300`
    - `engine=batched`
    - `batch_slots=86`
    - `leaf_batch=6`
    - AMP inference
    - seed `20260611`
  - Pair CSV:
    `eval_results/m9_hparam_scout_s100_g300_pairs.csv`
  - Leaderboard:
    `eval_results/m9_hparam_scout_s100_g300_leaderboard.csv`
  - Game log:
    `eval_results/m9_hparam_scout_s100_g300_games.csv`
  - Runtime:
    - `1800` games in `486.3s`.
    - `3.701` games/sec.
- Eval leaderboard:
  - `current_best_32x4_i080`: 900 games, `78.9%`, +20.71 avg margin,
    127.55 avg score.
  - `baseline32x4_i040`: 900 games, `49.6%`, -0.20 avg margin,
    113.68 avg score.
  - `trainsteps400_i040`: 900 games, `37.0%`, -9.56 avg margin,
    106.22 avg score.
  - `buffer100k_i040`: 900 games, `34.6%`, -10.95 avg margin,
    107.90 avg score.
- Key pair results:
  - Current best beat baseline iter 40:
    `220-76-4`, `74.0%`, +16.41 margin.
  - Current best beat buffer100k iter 40:
    `248-46-6`, `83.7%`, +24.25 margin.
  - Current best beat trainsteps400 iter 40:
    `235-61-4`, `79.0%`, +21.48 margin.
  - Baseline iter 40 beat buffer100k iter 40:
    `178-117-5`, `60.2%`, +7.02 margin.
  - Baseline iter 40 beat trainsteps400 iter 40:
    `186-111-3`, `62.5%`, +8.78 margin.
  - Trainsteps400 iter 40 beat buffer100k iter 40 narrowly:
    `157-142-1` from trainsteps400 perspective, `52.5%`, +1.57 margin.
- Interpretation:
  - Neither `buffer=100000` nor `train_steps=400` improved same-iteration
    strength in this 40-iteration scout.
  - `train_steps=400` improved value loss and slightly beat the 100k-buffer
    scout, but it lost clearly to the original baseline iter 40.
  - `buffer=100000` looks worse for short wall-clock learning because the
    larger buffer keeps more early weak data and diluted recent improvements.
  - The original locked baseline settings remain preferred for now:
    `buffer=50000`, `train_steps=200`.

## 2026-06-11 - Planned Warm-Start Higher-Sims Run

- Added script:
  - `scripts/run_m9_currentbest_s1600_warm_14h.ps1`
- Purpose:
  - Train from the current best checkpoint with stronger search.
  - This is a warm start from model weights, not a full optimizer/replay
    continuation.
- Planned config:
  - Warm start:
    `checkpoints_best/kingdomino_current_best.pt`
  - Model:
    `32x4`, `bilinear_dim=64`
  - Search:
    `sims=1600`
  - Replay buffer:
    `100000`
  - Train steps:
    `300`
  - LR:
    `0.001`
  - Self-play:
    `192` games/iteration
  - Batch/search:
    `batch_slots=86`, `leaf_batch=6`, AMP inference
  - Default iterations:
    `60`
- Time estimate:
  - Prior scratch `32x4 @ sims=1600` run took about `9.2` hours for `40`
    iterations from checkpoint timestamps.
  - `60` iterations should land around `13.8` hours with the old training
    step count, plus modest overhead from `train_steps=300`.
  - Expected runtime is approximately `14` hours, with thermals/system load
    as the main uncertainty.
- Outputs:
  - Checkpoint dir:
    `checkpoints_m9_warm_currentbest_32x4_s1600_b100k_t300_i60`
  - Log:
    `logs/m9_warm_currentbest_32x4_s1600_b100k_t300_i60.log`
- After completion:
  - First eval should compare the final checkpoint against current best.
  - If final checkpoint looks strong, also evaluate an intermediate checkpoint
    such as iter 40 or 50 to see whether the run peaked before the end.

## 2026-06-12 - Warm-Start Higher-Sims Run Results

- Completed run:
  - Script:
    `scripts/run_m9_currentbest_s1600_warm_14h.ps1`
  - Checkpoint dir:
    `checkpoints_m9_warm_currentbest_32x4_s1600_b100k_t300_i60`
  - Log:
    `logs/m9_warm_currentbest_32x4_s1600_b100k_t300_i60.log`
  - Final checkpoint:
    `checkpoints_m9_warm_currentbest_32x4_s1600_b100k_t300_i60/iter_0060.pt`
- Config:
  - Warm start:
    `checkpoints_best/kingdomino_current_best.pt`
  - Model:
    `32x4`, `bilinear_dim=64`
  - Search:
    `sims=1600`
  - Replay buffer:
    `100000`
  - Train steps:
    `300`
  - LR:
    `0.001`
  - Self-play:
    `192` games/iteration
  - Batch/search:
    `batch_slots=86`, `leaf_batch=6`, AMP inference
- Runtime:
  - `60/60` iterations completed.
  - First checkpoint:
    2026-06-11 10:07 PM.
  - Final checkpoint:
    2026-06-12 10:26 AM.
  - Checkpoint timestamp span:
    `12.31` hours across 59 intervals.
  - Mean interval:
    `12.52` minutes/checkpoint.
- Throughput summary:
  - Average self-play throughput:
    `0.297` games/sec.
  - Late self-play throughput, iterations 51-60:
    `0.295` games/sec.
  - Average neural eval throughput:
    `21078` evals/sec.
  - Late neural eval throughput:
    `21076` evals/sec.
  - Average batch:
    `352.3/516`, about `68%` fill.
  - Final iteration timing:
    - step `19%`
    - eval `74%`
    - update `7%`
- Training curve:
  - Greedy benchmark was already `100%` from iteration 1 because this was a
    warm start from the current best.
  - Iteration 1:
    - value loss `0.1246`
    - policy loss `1.7955`
    - Greedy `100%`
  - Iteration 10:
    - value loss `0.1571`
    - policy loss `1.8513`
    - Greedy `100%`
  - Iteration 30:
    - value loss `0.1502`
    - policy loss `1.7654`
    - Greedy `100%`
  - Iteration 50:
    - value loss `0.1454`
    - policy loss `1.7602`
    - Greedy `100%`
  - Iteration 60:
    - value loss `0.1464`
    - policy loss `1.7425`
    - Greedy `100%`
  - Minimum value loss:
    - `0.1246` at iteration 1.
  - Minimum policy loss:
    - `1.7425` at iteration 60.
- Interpretation from the log:
  - The run was stable.
  - Throughput was better than expected and did not degrade materially late in
    the run.
  - Larger buffer filled by iteration 10, then remained at `100000`.
  - Policy loss improved to its best value at the final checkpoint, so the log
    does not indicate an obvious earlier peak.
  - Greedy is saturated and no longer useful for distinguishing late strength.
- Focused eval against current best:
  - Command settings:
    - `sims=100`
    - `games_per_pair=300`
    - `engine=batched`
    - `batch_slots=86`
    - `leaf_batch=6`
    - AMP inference
    - seed `20260612`
  - Pair CSV:
    `eval_results/m9_warm_s1600_i060_vs_current_s100_g300_pairs.csv`
  - Leaderboard:
    `eval_results/m9_warm_s1600_i060_vs_current_s100_g300_leaderboard.csv`
  - Game log:
    `eval_results/m9_warm_s1600_i060_vs_current_s100_g300_games.csv`
  - Result:
    - `warm_s1600_b100k_t300_i060` beat current best:
      `182-112-6`, `61.7%`, +5.89 avg margin.
    - Runtime:
      `300` games in `85.6s`, `3.506` games/sec.
- Current interpretation:
  - The warm-start higher-sims run produced a strong candidate that appears to
    beat the previous current best.
  - This is the first positive result for higher sims: scratch `sims=1600`
    underperformed, but warm-start `sims=1600` with stronger data improved.
  - Recommended next step is either promote this checkpoint after the existing
    300-game result or run a larger confirmation eval before promotion.

## 2026-06-12 - BGA Kingdomino Advisor Scaffold

- Added a separate extension folder:
  - `extension_kingdomino/`
- Purpose:
  - Build a Firefox-first BGA Kingdomino advisor overlay without overwriting
    the existing Can Stop advisor in `extension/`.
- Files:
  - `extension_kingdomino/manifest.json`
    - Firefox-oriented WebExtension manifest.
  - `extension_kingdomino/manifest.chrome.json`
    - Chrome Manifest V3 variant using a background service worker.
  - `extension_kingdomino/background.js`
    - Localhost POST fallback for BGA/CORS/CSP issues.
  - `extension_kingdomino/popup.html`
  - `extension_kingdomino/popup.js`
    - Manual capture/recommend and debug-capture controls.
    - Configurable engine, sims, and checkpoint path.
  - `extension_kingdomino/content.js`
    - BGA page-context scraper.
    - On-page overlay.
    - Recommendation rows with visit share, prior, value, raw visits, and legal
      index when present.
    - Sims dropdown and Think deeper button.
    - Debug overlay when BGA state cannot yet be normalized into engine JSON.
  - `extension_kingdomino/README.md`
- Local server target:
  - `http://127.0.0.1:8000/api/recommend`
- Advisor payload defaults:
  - engine `nn`
  - sims `800`
  - device `cuda`
  - architecture `32x4`, `bilinear_dim=64`
  - blank checkpoint path uses the server default checkpoint discovery.
- Current limitation:
  - The extension can capture `gameui.gamedatas` and DOM samples from BGA, but
    the exact Kingdomino BGA state mapping still needs live BGA payload/HTML
    validation.
  - If normalization fails, the overlay asks for a copied debug capture. That
    capture is also stored as `kingdomino_last_capture` in extension storage.

## 2026-06-12 - BGA Kingdomino Opening Capture Mapping

- Reviewed first live BGA debug capture from table `867153920`.
- BGA capture details:
  - `gamestate.name`: `chooseDomino`
  - active BGA player: `89146710`
  - `playerorder`: `[89146710, 84634030]`
  - current visible dominoes:
    - `12`
    - `17`
    - `21`
    - `43`
  - all visible dominoes had `location="FUTURE"` and no owner.
- Added first normalization pass in:
  - `extension_kingdomino/content.js`
- Supported mapping so far:
  - BGA `chooseDomino` -> engine `INITIAL_SELECTION`
  - unowned BGA `FUTURE` dominoes -> engine `current_row`
  - BGA `playerorder` -> engine player indices
  - BGA active player + opening pick count -> engine `start_player` and
    current actor inference
  - empty starting boards -> engine boards with centered castles
- Validation:
  - Constructed the normalized opening state from the capture and imported it
    through `state_from_debug_json`.
  - Result:
    - phase `INITIAL_SELECTION`
    - current actor `0`
    - current row `[12, 17, 21, 43]`
    - legal actions:
      - `Pick domino 12`
      - `Pick domino 17`
      - `Pick domino 21`
      - `Pick domino 43`
- Current limitation:
  - Placement-phase board reconstruction is not complete yet.
  - Need another live capture after at least one selected/current domino exists,
    ideally during BGA `placeDomino`, to map BGA `location`, `x`, `y`, and
    `rotation` into engine board cells and pending claims.

## 2026-06-12 - BGA Advisor Hidden Deck Fix

- Observed server failure after CORS was fixed:
  - `OPTIONS /api/recommend` returned `200 OK`.
  - `POST /api/recommend` returned `500`.
  - MCTS error:
    `_evaluate received a non-terminal state with no legal actions
    (phase=PLACE_AND_SELECT)`.
- Root cause:
  - The extension sent `deck_count` but not `debug.deck`.
  - `state_from_debug_json` imports only `debug.deck`, so the engine state had
    an empty hidden deck.
  - During MCTS simulation from the opening pick phase, after the four initial
    picks the engine advanced to `PLACE_AND_SELECT` with no next row, producing
    no legal actions.
- Fix:
  - Updated `extension_kingdomino/content.js` to compute hidden deck IDs from
    `dominoesDescription` minus visible BGA `dominoes`.
  - Added this list to `state.debug.deck`.
  - Set `deck_count` to the hidden deck length.
- Validation:
  - Recreated the opening capture state with hidden deck length `44`.
  - Imported through `state_from_debug_json`.
  - Legal actions remained the expected four opening picks.
  - Ran a tiny `nn` recommendation with `nn_sims=4`; it returned
    recommendations without crashing.

## 2026-06-12 - BGA Opponent First Placement Capture

- Reviewed live BGA capture for opponent `placeDomino` turn.
- BGA details:
  - `gamestate.name`: `placeDomino`
  - active BGA player: `84634030`
  - engine player index: `1`
  - BGA current domino from args: `12`
  - current position: `1`
  - visible domino locations:
    - `FUTURE`: 4
    - `CURRENT`: 4
  - future row:
    - `4`
    - `14`
    - `40`
    - `44`
  - current/pending claims:
    - player 1: domino `12`
    - player 0: domino `17`
    - player 0: domino `21`
    - player 1: domino `43`
- Normalized state validation:
  - phase `PLACE_AND_SELECT`
  - current actor `1`
  - actor index `0`
  - current row `[4, 14, 40, 44]`
  - pending claims `[(1,12), (0,17), (0,21), (1,43)]`
  - hidden deck count `40`
  - legal action count `48`
  - first legal actions place domino `12` adjacent to the centered castle and
    choose one of the four future-row tiles.
- Interpretation:
  - First `placeDomino` state is usable because both boards are still empty
    except the castle.
  - Claim/current-row/actor mapping looks correct.
  - Board reconstruction after placements still needs exact BGA cell mapping.
- Added richer debug extraction in:
  - `extension_kingdomino/content.js`
- New scraper diagnostics:
  - `normalization.debug.bga_current_domino`
  - `normalization.debug.bga_current_position`
  - `normalization.debug.kingdom_summary`
    - counts BGA kingdom cells.
    - includes up to 80 non-empty raw cell samples from
      `gamestate.args.kingdom`.
  - DOM sampler now ignores the advisor overlay itself.
- Next needed capture:
  - Capture after at least one domino has actually been placed on a board, so
    `kingdom_summary.samples` reveals BGA's terrain/crown/domino cell encoding.

## 2026-06-12 - Advisor Position Audit: Pick 4 vs Pick 44

- User noticed a likely strategic error:
  - Opponent/player 1 has current domino `12` (`SWAMP/SWAMP`) and already owns
    domino `43` (`WHEAT/SWAMP+2 crowns`).
  - Future row contains:
    - `4`: `FOREST/FOREST`
    - `14`: `WHEAT/WATER`
    - `40`: `MINE+1/WHEAT`
    - `44`: `GRASS/SWAMP+2 crowns`
  - Human intuition favors picking `44` to continue swamp/crown synergy.
  - NN/MCTS advisor instead strongly recommended picking `4`.
- Added reusable audit script:
  - `scripts/audit_kingdomino_advisor_position.py`
- Script behavior:
  - Loads a copied extension capture JSON or state JSON.
  - Prints normalized state summary:
    - phase
    - current actor
    - current row tile descriptions
    - pending/next claims
    - hidden deck count
    - legal action counts by pick
  - Runs recommendation sweeps for selected sim counts.
  - Optional greedy baseline with `--include_greedy`.
- Audit capture:
  - `C:\Users\joeld\.codex\attachments\fb9a556a-0b67-4e16-adfe-c385e597ec3b\pasted-text.txt`
- State audit result:
  - phase `PLACE_AND_SELECT`
  - current actor `1`
  - pending claims correctly show player 1 owns/current dominoes `12` and
    `43`.
  - current row correctly shows `[4, 14, 40, 44]`.
  - legal action count `48`.
  - legal actions by pick:
    - each of `4`, `14`, `40`, `44` has 12 legal placement variants.
- Greedy heuristic result:
  - Top 12 actions all pick `44`.
  - Interpretation: the heuristic agrees with human swamp/crown intuition.
- NN/MCTS result:
  - At `sims=200`:
    - top-k visit share by pick:
      - `4`: `0.440`
      - `14`: `0.140`
    - `44` absent from top 12.
  - At `sims=800`:
    - top-k visit share by pick:
      - `4`: `0.551`
      - `40`: `0.022`
    - `44` absent from top 12.
  - At `sims=1600`:
    - top-k visit share by pick:
      - `4`: `0.630`
      - `40`: `0.106`
    - `44` absent from top 16.
  - At `sims=3200`:
    - top-k visit share by pick:
      - `4`: `0.783`
      - `40`: `0.052`
    - `44` absent from top 16.
- Interpretation:
  - This is not a UI parsing issue; state mapping correctly includes the
    swamp/crown context.
  - This is not a low-sim artifact; higher sims made the preference for `4`
    stronger.
  - Current leading hypotheses:
    - learned policy/value blind spot around early terrain/crown synergy.
    - self-play/search may overvalue low-number turn-order control or board
      flexibility.
    - possible training/eval rule mismatch around BGA rules or tile-order
      implications still worth auditing.
  - This position should become a named regression/probe position for future
    checkpoints.

## 2026-06-12 - BGA Pending-Claim Mapping Tightening

- Observed later advisor error from Uvicorn log:
  - `POST /api/recommend` returned `500`.
  - MCTS error:
    `_evaluate received a non-terminal state with no legal actions
    (phase=PLACE_AND_SELECT)`.
- Likely scraper cause:
  - The extension previously treated every owned domino with
    `location != FUTURE` as a remaining pending/current claim.
  - After the round starts, BGA may move already-placed dominoes out of
    `CURRENT`, so non-`FUTURE` can include completed/placed dominoes.
  - Pending claims for the remaining part of a round should use owned
    `location == CURRENT` dominoes, with active `gamestate.args.domino` as a
    fallback.
- Extension fix:
  - Updated `extension_kingdomino/content.js`.
  - `currentOwned` now includes only owned `location == CURRENT` dominoes.
  - For `placeDomino`, pending claims use `CURRENT` owned dominoes.
  - If the active `gamestate.args.domino` is missing from that set, it is added
    as a fallback active claim.
  - Debug note now says the mapper is using CURRENT owned dominoes as remaining
    pending claims.
- Server diagnostic improvement:
  - Updated `games/kingdomino/web_app.py`.
  - NN/MCTS `ValueError` from `run_pimc` now returns HTTP `400` with structured
    state diagnostics instead of a raw `500` traceback:
    - phase
    - current actor
    - current row
    - pending claims
    - next claims
    - deck count
    - root legal action count
- Validation:
  - `node --check extension_kingdomino/content.js` passed.
  - `from games.kingdomino.web_app import app` passed in the project venv.
- Next if error persists:
  - Capture Debug Only from the failing BGA state after reloading the extension.
  - The new server response should also include structured diagnostics in the
    overlay/error body.

## 2026-06-12 - BGA Board Reconstruction Mapping

- Reviewed debug capture after the pending-claim fix succeeded.
- Capture details:
  - active BGA player: `89146710`
  - engine current actor: `0`
  - `gamestate.name`: `placeDomino`
  - BGA current domino: `21`
  - current row: `[14, 44]`
  - pending claims:
    - player 0: domino `21`
    - player 1: domino `43`
  - next claims:
    - player 1: domino `4`
    - player 0: domino `40`
  - BGA domino locations:
    - `FUTURE`: 4
    - `KINGDOM`: 2
    - `CURRENT`: 2
- Useful BGA board evidence:
  - `kingdom_summary.samples` for active player showed:
    - `(0,0)` -> Castle
    - `(-1,0)` -> forest
    - `(-1,1)` -> lake
  - BGA domino `17` was:
    - owner player `89146710`
    - location `KINGDOM`
    - rotation `3`
    - x `-1`
    - y `0`
  - This confirms:
    - BGA `(0,0)` is the castle.
    - engine coordinate = BGA coordinate + `(7,7)`.
    - BGA `x,y` is the left/A half anchor.
    - rotation `3` maps the right/B half to `(x, y+1)`.
  - Earlier opponent placement with domino `12`, rotation `1`, x `-1`, y `0`
    plus player kingdom bounds supports rotation `1` -> `(x, y-1)`.
- Added board reconstruction in:
  - `extension_kingdomino/content.js`
- Mapping implemented:
  - Start every board with castle at engine `(7,7)`.
  - For every BGA domino with `location == KINGDOM`:
    - map `owner_player` through BGA `playerorder`.
    - use `dominoesDescription` left/right half terrain and crowns.
    - place left/A half at `(7 + x, 7 + y)`.
    - place right/B half using rotation offset:
      - `0`: `(x+1, y)`
      - `1`: `(x, y-1)`
      - `2`: `(x-1, y)`
      - `3`: `(x, y+1)`
  - terrain name mapping:
    - `field/wheat -> WHEAT`
    - `forest -> FOREST`
    - `lake/water -> WATER`
    - `grassland/grass -> GRASS`
    - `swamp -> SWAMP`
    - `mountain/mine -> MINE`
    - `Castle -> CASTLE`
- Fixed debug DOM sampler:
  - Replaced injected reference to outer `OVERLAY_ID` constant with literal
    `kingdomino-advisor-overlay`.
- Validation:
  - `node --check extension_kingdomino/content.js` passed.
  - Reconstructed the capture manually in Python:
    - player 0 board imported with castle, forest at `(6,7)`, water at `(6,8)`
      for domino `17`.
    - player 1 board imported with castle and swamp cells for domino `12`.
    - phase `PLACE_AND_SELECT`, current actor `0`.
    - legal action count became `32`, lower than the empty-board assumption,
      confirming placed cells constrain legal placement.
- Remaining uncertainty:
  - Rotation `0` and `2` are inferred but not yet directly confirmed from a
    live capture.
  - More midgame captures should confirm all four rotations and board mapping
    for mixed-terrain dominoes.

## 2026-06-12 - BGA Legality Check: Placement at `(6,8)->(7,8)`

- User questioned advisor recommendation:
  - `Place domino 43 at (6,8) -> (7,8) flipped; pick 14`
  - Concern: opponent had already placed double swamp at `(6,7)/(6,8)`.
- Reviewed exact debug capture:
  - active BGA player: `84634030`
  - engine current actor: player `1`
  - active pending domino: `43`
  - active player board cells reconstructed as:
    - castle `(7,7)`
    - swamp `(6,7)`
    - swamp `(6,6)`
  - BGA `kingdom_summary` for active player confirms:
    - castle `(0,0)`
    - swamp `(-1,0)`
    - swamp `(-1,-1)`
  - Therefore engine `(6,8)` is empty for the active player.
- Verified with Python importer/legal actions:
  - legal action count: `16`
  - both non-flipped and flipped placements at `(6,8)->(7,8)` are legal.
- Conclusion:
  - The questioned recommendation is legal in the captured state.
  - The confusion came from the earlier assumption that double swamp occupied
    `(6,7)/(6,8)`; BGA/current normalized state shows it occupies
    `(6,7)/(6,6)`.
- Found and fixed a separate scraper bug while inspecting the capture:
  - JavaScript `Number(null)` produced `0`, so BGA `KINGDOM` dominoes with
    missing `x/y` could be reconstructed incorrectly on the castle.
  - Updated `extension_kingdomino/content.js`:
    - `asInt(null|undefined|"", fallback)` now returns the fallback.
  - This prevents future missing-coordinate domino records from being placed at
    BGA `(0,0)` accidentally.

## 2026-06-12 - Coordinate/Board Audit Improvements

- User asked whether:
  - rotation causes grid numbers to vary from turn to turn.
  - the opponent board is being read when it is opponent's turn vs player 0
    board.
- Answer from current mapping:
  - Rotation changes which second cell a placed BGA domino occupies relative to
    its BGA anchor, but engine coordinates are stable:
    - BGA castle `(0,0)` -> engine `(7,7)`.
    - engine coordinate = BGA coordinate + `(7,7)`.
  - Each player's board has its own independent coordinate system.
    - `(6,8)` on player 0's board is not the same physical board cell as
      `(6,8)` on player 1's board.
  - The extension maps BGA `owner_player` through `playerorder`, so placed
    dominoes are assigned to the owning player's board, not simply the active
    player.
- Latest capture audit:
  - Current actor was player `1`.
  - Player 0 board:
    - `(6,7)` FOREST domino `17`
    - `(6,8)` WATER domino `17`
    - `(7,7)` WHEAT crown from domino `21` in the old capture, revealing the
      pre-null-fix issue for missing coordinate records.
    - `(8,7)` GRASS domino `21`
  - Player 1 active board:
    - `(6,6)` SWAMP domino `12`
    - `(6,7)` SWAMP domino `12`
    - `(7,7)` CASTLE
  - Therefore the questioned active-player placement using `(6,8)` was legal
    because player 1's `(6,8)` was empty.
- Added richer debug output to future captures:
  - `state.debug.reconstructed_placements`
  - For each BGA `KINGDOM` domino:
    - BGA owner id
    - engine player index
    - BGA anchor
    - BGA rotation
    - rotation offset
    - BGA cells
    - engine cells
    - terrain/crowns per half
- Updated audit script:
  - `scripts/audit_kingdomino_advisor_position.py`
  - Now prints reconstructed boards and, for future captures, the BGA
    reconstructed placement mapping.

## 2026-06-12 - BGA Active Claim Selection Fix

- User questioned whether the board state updates within a BGA turn, especially
  when multiple dominoes are placed/picked during the same round.
- Current advisor architecture:
  - The browser extension is stateless with respect to game moves.
  - Every recommendation injects a page-context reader into BGA and rebuilds
    engine JSON from current `gameui.gamedatas`.
  - It does not locally apply the previous recommendation.
  - Therefore the second placement in a round only sees the first placement once
    BGA has updated that domino to `location == KINGDOM` with `x`, `y`, and
    `rotation`.
- Found one concrete ambiguity:
  - During `placeDomino`, BGA exposes the exact active domino as
    `gamestate.args.domino`.
  - The extension was choosing `actor_index` by first remaining `CURRENT` claim
    owned by the active player.
  - In 2-player/Mighty Duel, the same player can own multiple current dominoes,
    so ownership alone can choose the wrong active claim.
- Fix in `extension_kingdomino/content.js`:
  - For `placeDomino`, prefer an exact match on:
    - active BGA player mapped through `playerorder`
    - `gamestate.args.domino`
  - Fall back to active-player ownership only if BGA does not expose an active
    domino id.
  - Added debug fields:
    - `active_claim`
    - `actor_claim`
    - `current_owned_claims`
    - `next_owned_claims`
- Validation:
  - `node --check extension_kingdomino/content.js` passed.
- Operational note:
  - Requires reloading the extension and refreshing the BGA page.
  - Does not require restarting the Uvicorn server.

## 2026-06-12 - BGA Server Import Cache Fix for False Discards

- User reported a clearly wrong recommendation:
  - Advisor showed only discard actions for active domino `2`.
  - Human-visible board had legal placements.
  - Example UI output:
    - `Discard domino 2; pick 7`
- Debug capture:
  - `gamestate.name`: `placeDomino`
  - active BGA player mapped to engine player `1`
  - active claim:
    - player `1`
    - domino `2`
  - current row:
    - `[7, 11, 16, 26]`
  - pending claims:
    - `(1, 2)`
    - `(0, 15)`
    - `(1, 36)`
    - `(0, 47)`
  - active board reconstruction showed relevant wheat-adjacent placement space.
- Reproduced with:
  - `scripts/audit_kingdomino_advisor_position.py`
  - Before fix:
    - legal action count: `4`
    - all legal actions were discards, one per future pick.
    - greedy and NN/MCTS both had no placement choices because root legal action
      generation thought none existed.
- Root cause:
  - `games/kingdomino/web_app.py::state_from_debug_json` rebuilt optimized
    `Board` objects from browser JSON.
  - It rebuilt:
    - `terrain`
    - `crowns`
    - `domino_id`
    - `_occupied`
    - bounding box
  - It did **not** rebuild `Board._cell`.
  - `Board._cell` is the optimized terrain cache used by
    `Board.half_connects()` and placement legality.
  - Result:
    - board display/debug looked correct.
    - legality checks only saw stale/empty terrain cache state, often just the
      castle, so legal terrain connections were missed.
- Fix in `games/kingdomino/web_app.py`:
  - Clear `b._cell` when rebuilding an imported board.
  - Add every imported cell to `b._cell[(x, y)] = terrain_id`.
  - Re-add castle to `_cell` in the empty-board fallback.
- Validation on the exact pasted capture:
  - After fix:
    - legal action count: `32`
    - legal actions by pick:
      - `7`: 8
      - `11`: 8
      - `16`: 8
      - `26`: 8
    - greedy and NN/MCTS recommendations now include real placements such as:
      - `Place domino 2 at (9,8) -> (10,8); pick 26`
- Operational note:
  - This is server-side.
  - Requires restarting Uvicorn:
    - `.\.venv\Scripts\python.exe -m uvicorn games.kingdomino.web_app:app --host 127.0.0.1 --port 8000`
  - Extension reload is not required for this specific fix, though it is still
    required to pick up the active-claim selection fix above.
