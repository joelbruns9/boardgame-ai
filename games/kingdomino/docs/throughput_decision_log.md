# Kingdomino Throughput Decision Log

Purpose: track throughput experiments as decisions, not folklore. Every item
should say what bottleneck it targets, what was measured, what decision we made,
and what evidence would reopen it.

This file is the source of truth for throughput work going forward. Older notes
in `Throughput_Backlog.txt` are useful history, but may be stale.

---

## How To Use This Log

For every throughput idea, add or update an experiment card:

```md
## Item Name

Status: planned | testing | implemented | rejected | deferred | superseded
Priority: high | medium | low | situational
Owner/date:

Hypothesis:
- What bottleneck should this improve?

Applies when:
- Hardware/config assumptions.

Baseline:
- Hardware:
- Command:
- games/s:
- step/eval/update:
- mean batch/fill:
- evaluator busy:
- exact fallback:
- eval timing: h2d / forward / readback:

Result:
- Measured numbers.

Decision:
- Implement / reject / defer / retest later.

Reopen if:
- What changed enough to make the decision stale?
```

Keep failed experiments. A measured regression is valuable because it prevents
retesting the same idea without a new reason.

---

## Current Bottleneck Snapshot

Update this section after any representative run.

| Field | Current value |
|---|---|
| Date | 2026-06-29 |
| Hardware | Laptop, 8 physical cores; cloud target is 5090 with 16-32 vCPUs |
| Engine/config | `batched_open_loop`, 100 games, 1600 sims, exact endgame solver on, 3s solve budget |
| games/s | 0.139 async + `--solver_cpus 6` (== solver-off 0.140); 0.1216 single-buffer solver-on (solver on critical path) |
| step/eval/update | async: step 62.7s, eval 643.7s, update 13.4s (eval is concurrent-solver inflated; profile pending) |
| batch fill | 72% async@32 solver_cpus=6; 78% sync solver-on |
| total evals | about 6.9M solver-on |
| exact solve success | 59/100 (solver_cpus=6); 70/100 (full-machine, but contended) |
| exact fallback | 41/100 (solver_cpus=6) |
| exact solver time | ~210s (fully hidden under eval on the dedicated pool) |
| eval timing split | **forward-bound**: forward 601.7s (90.5%), h2d 26.7s (4%), readback 29.6s (4.5%), 50923 calls (async@32, solver_cpus=6) |
| Forward bench (48ch/6b, cudnn_benchmark) | batch 32/64/128/137/192/256 -> 2.85/5.29/11.91/12.46/17.63/23.56 ms; µs/sample ~89/82.6/93/91/91.8/92; peak 12.1k evals/s at batch 64. Scales linearly with batch, µs/sample ~constant => **GPU-compute-bound, NOT launch-bound**. In-loop forward (11.8 ms @137) == bench (12.46 ms), so the forward is NOT solver-inflated. |
| Throughput ceiling note | Laptop self-play is **GPU-forward-compute-bound at ~0.14 games/s**; solver is free. Floor = ~90µs/sample x positions. Double-buffer marginal (hides only ~75s CPU, ceiling ~0.15). CUDA graph / torch.compile low value (compute-bound, little launch overhead). channels-last is the one real forward lever (modest). Forward most efficient at batch ~64 (82.6µs) vs ~137 (91µs) -> batch_slots sweep could be retested. Real win is the 5090. |

Important distinction: `eval_sec` is the whole evaluator call, not just neural-net
math. It can include NumPy contiguity/padding, H2D transfer, forward pass,
postprocess, D2H readback, and CPU legal-logit gathering. Use
`--profile_eval_timing` to split H2D / forward / readback before prioritizing GPU
kernel work vs transfer slimming.

---

## Active Priorities

| Item | Hypothesis | Status | Next evidence needed |
|---|---|---|---|
| Async exact solve jobs + overbooked search slots | Hide exact solver CPU time behind GPU eval without reducing active batch size | Planned | Implement and gate on games/s, fallback, solve-queue depth, batch fill, evaluator busy |
| Profile evaluator timing | Determine whether `eval_sec` is forward-bound or transfer/readback-bound | Planned | Short solver-on run with `--profile_eval_timing` |
| Channels-last inference benchmark | Improve CUDA convolution throughput for board planes | Planned | Forward benchmark comparing NCHW vs channels-last under TF32/AMP settings |
| Slim legal descriptors | Reduce legal-index traffic and conversion overhead | Planned | Measure current idx dtype/bytes and run an int32/uint16 transport benchmark |
| Throughput backlog cleanup | Replace stale double-buffer priority with measured decision | Planned | Update or archive `Throughput_Backlog.txt` after this log is adopted |

---

## Implemented Wins

## Rust BatchedMCTS Open-Loop Port

Status: implemented
Priority: historical high

Hypothesis:
- Move open-loop self-play into Rust and batch leaf evaluations across slots.

Result:
- About 48x speedup over Python open-loop.
- Roughly 0.02 games/s to 0.962 games/s at 200 sims.
- Roughly 0.27 games/s at 1600 sims in the cloud run.

Decision:
- Keep. This is the core training path.

## Direct Batch Buffer Encoding (`encode_arrays_into`)

Status: implemented
Priority: historical high

Hypothesis:
- Avoid per-leaf array allocation/copy by encoding directly into batch buffers.

Result:
- Removed 192 intermediate Array3/Array1 allocation+copy pairs per tick at
  `batch_slots=32`, `leaf_batch=6`.
- Measured around +5% on GPU-dominated workload.

Decision:
- Keep. More valuable as CPU pressure rises.

## Allocation Reduction In Open-Loop Selection

Status: implemented
Priority: historical medium

Hypothesis:
- Replace per-node HashMap merge with zero-allocation two-pointer merge.

Result:
- Removed one HashMap allocation per tree node per descent.
- Measured around +2-5% on GPU-bound workload; likely larger when CPU-bound.

Decision:
- Keep.

## Double-Clone Elimination In `ol_descend`

Status: implemented
Priority: historical low

Hypothesis:
- Remove redundant `RustGameState` clone per simulation.

Result:
- Small but free CPU reduction.

Decision:
- Keep.

## Rust Training-Batch Augmentation

Status: implemented
Priority: historical medium

Hypothesis:
- Move D4 augmentation, mask transform, and legal-mask construction out of Python
  hot loops.

Result:
- Rust `d4_augment`: about 3x speedup on augmentation path.
- Rust `augment_mask`: sample_batch about 16.5 ms to 9.4 ms.
- Rust legal-mask construction eliminated Python `encode_action` loop per sample.

Decision:
- Keep.

## GPU-Side Legal Logit Gather And f32 Readback

Status: implemented
Priority: historical high

Hypothesis:
- Avoid reading back full policy logits and avoid f64 D2H transfer.

Result:
- Legal gather reduced D2H from `K * 3390 * 4` bytes to roughly
  `K * n_legal * 4` bytes.
- f32 readback halved values/logits transfer compared with f64.

Decision:
- Keep. Especially important on slower PCIe cloud VMs.

## `torch.inference_mode()`

Status: implemented
Priority: historical low

Hypothesis:
- Reduce PyTorch inference bookkeeping versus `no_grad()`.

Result:
- No major measured throughput change, but correct for pure inference.

Decision:
- Keep.

## Rust Benchmark Player

Status: implemented
Priority: historical medium

Hypothesis:
- Make benchmark games cheap enough for frequent evaluation.

Result:
- About 8.9x benchmark speedup.
- Made `benchmark_every=1` viable.

Decision:
- Keep.

## Batch-Generated Per-Simulation Shuffle Seeds

Status: implemented
Priority: historical low

Hypothesis:
- Prepare for future cross-simulation parallelism.

Result:
- No throughput impact in current serial-per-slot path.

Decision:
- Keep as groundwork.

## `batch_slots=32` Sweep

Status: implemented
Priority: historical medium

Hypothesis:
- Find best synchronized batch population.

Result:
- Confirmed 32 was optimal in the measured regime.

Decision:
- Keep as current default, but retest on 5090 or after async solve overbooking.

## IPC Batched-Send Protocol

Status: superseded
Priority: historical medium

Hypothesis:
- Batch leaves across Python parallel workers through one inference service.

Result:
- 1.74x on old Python parallel path, about 0.081 to 0.141 games/s.

Decision:
- Superseded by Rust batched engine, but useful if the multi-worker IPC topology
  becomes relevant again.

---

## Rejected Or Regressed

## AMP Inference On 3070, 32ch/4b Net

Status: rejected
Priority: do not retry without new hardware/model evidence

Hypothesis:
- fp16 autocast would speed GPU inference.

Result:
- 0.74x; a slowdown.
- The measured net/batch regime appeared latency-bound, and cast overhead
  exceeded compute savings.

Decision:
- Keep disabled by default.

Reopen if:
- Larger net, different GPU, channels-last, or Tensor Core utilization changes
  the forward benchmark.

## Fine-Grained `sample_batch` Threading

Status: rejected
Priority: do not retry in same form

Hypothesis:
- Use `ThreadPoolExecutor` to parallelize sample densify/augmentation.

Result:
- 0.44-0.50x; about a 2x regression.
- Fine-grained GIL-held small-array work dominated.

Decision:
- Keep `--sample_workers` opt-in/off by default.

Reopen if:
- The entire batch preparation is moved into coarse Rust or process-level work.

## Double-Buffer Two `BatchedMCTS` Instances

Status: rejected for current solver-on workload; may be retested under larger
batches/hardware
Priority: not the preferred overlap design

Hypothesis:
- Run two independent game batches A/B so one uses GPU eval while the other does
  CPU step/update/solve. Wall time approaches `max(CPU, GPU)` instead of
  `CPU + GPU`.

Baseline:
- Single-buffer solver-on: 0.1216 games/s, batch fill about 78%.

Result:
- Double-buffer solver-on: 0.1185 games/s, batch fill about 74%.
- Timing sums exceeded 100%, so some overlap was real.
- Net was negative because phase synchronization caused both instances to enter
  endgame solve bursts together, and splitting the batch reduced GPU efficiency.

Decision:
- Do not use two-instance double-buffer as the next overlap vehicle.
- Prefer one-instance async exact solve jobs plus overbooked active search slots.

Reopen if:
- Each half-buffer remains near the GPU throughput knee, A/B phases can be
  intentionally staggered, and a profile shows CPU solve/update time is hideable
  without reducing batch fill.

Update 2026-06-29 (still rejected; profile in): async exact solve removed
reopen-blocker #1 (solving is off the critical-path step). But `--profile_eval_timing`
now shows eval is ~90% forward and **GPU-compute-bound** (forward_bench: µs/sample
constant, in-loop == bench). So the only thing double-buffer could hide is step
(~63s) + update (~13s) ~= 75s of CPU behind the ~600s forward -> ceiling ~0.15, an
~8% best case, and it doubles the background solver threads. Not worth it: the
forward compute is the floor, and double-buffer cannot touch it. Prefer
forward-compute levers (channels-last / smaller net / fewer sims / the 5090).

---

## Planned / Future Experiments

## Async Exact Solve Jobs + Dedicated Solver Pool

Status: implemented (in-place overbooking; behind `--async_solve`, default off)
Priority: high
Owner/date: 2026-06-29

Hypothesis:
- Detach exact endgame solving from the GPU-facing step() so solving overlaps GPU
  eval instead of blocking the critical path.

Applies when:
- Solver-on exact endgame targets are required for training quality.

Result (laptop, 8 cores; 100 games, 1600 sims, 3s budget, batch_slots=32):
- Built: background solver thread + `SolvingInBackground` slot state +
  dispatch/harvest + in-place fallback resume (slot keeps its game; a timed-out
  solve resumes MCTS in place). Correctness: deterministic, identical per-seed
  games vs sync; sync path 38/38.
- KEY NEGATIVE THEN FIX: a single shared (global) Rayon pool made it WORSE
  (async@32 0.118 vs sync 0.122) because the within-solve YBW's long,
  non-preemptible subtree-solves head-of-line-blocked the descent/backup
  `par_iter` that feeds the GPU -- inflating step, eval, AND update together.
  A DEDICATED solver pool (`--solver_cpus N`) confines the solver's `par_iter`;
  generation gets the rest of the cores via the global pool.
- With solver_cpus=6: games/s 0.139 ~= solver-off 0.140 (solver now
  throughput-free) with exact targets for 59/100 games. step 62.7s ~= solver-off
  61.2s; update 13.4s ~= 12.8s (the shared-pool update blow-up to 46s was
  contention, not a bug).
- solver_cpus sweep (8-core box): 5 -> 0.140 / 54 solved; 6 -> 0.139 / 59;
  7 -> 0.141 / 59. Throughput flat across 5-7 (GPU-bound, solver fully hidden);
  solve-success peaks at 6 then plateaus. Recommend solver_cpus ~ physical-2.
- CEILING: games/s sits at ~0.14 = the solver-off / GPU-forward-compute floor
  (see Profile Evaluator Timing). The eval is ~90% forward and compute-bound, and
  the in-loop forward is NOT solver-inflated. The small async-vs-sync eval
  difference is mostly MORE calls at smaller batches (fill 72% vs 78%, because
  slots are out solving), not forward contention. So there is no ~0.155 "left on
  the table" from the solver; the laptop is GPU-bound.

Decision:
- Keep. Default off; use `--async_solve --solver_cpus N` (N ~ physical-2) when the
  exact solver is on. Overbooking via a larger `--batch_slots` is optional and
  unnecessary once the dedicated pool is used.
- The dedicated solver pool is also the Step-2 admission-control primitive.

Reopen if:
- `--profile_eval_timing` shows eval is forward-bound (then the solver/eval CPU
  contention is small and other levers dominate), or the cloud 5090 changes the
  core balance.

## Stagger Game Generation (desynchronize endgame arrival)

Status: deferred (contingent)
Priority: low (currently)
Owner/date: 2026-06-29

Hypothesis:
- All 32 slots start at tick 0 and games are ~constant length, so endgames arrive
  in synchronized bursts. Staggering the start (each slot at a different stage)
  would desynchronize them, keeping batch fill high and the solve queue shallow.

Why deferred:
- The big cost of clustering (a serial solver burst on the critical-path step) is
  GONE now that solving is async on the dedicated pool. What remains is a milder
  batch-fill dip during clusters (72% vs 78%) and a bursty solve queue.
- async@32 + solver_cpus=6 is ALREADY at the solver-off throughput ceiling
  (0.139 vs 0.140), so there is little throughput headroom for staggering to
  capture right now.
- The actual cap is the **GPU forward compute** (eval is 90% forward,
  compute-bound; see Profile Evaluator Timing), which staggering does not touch.
- Queue delay does not hurt solve-success (the 3s budget starts when a solve
  begins, not when it is queued).

Reopen if:
- A forward-compute lever (channels-last, smaller net, fewer sims, or the 5090)
  lifts the GPU floor so throughput can grow past ~0.14, AND batch fill (currently
  72% async vs 78% sync, from slots out solving) then shows up as the next limiter.
  Staggering is the cheaper fill fix vs overbooking (no extra slots / overhead).

## Profile Evaluator Timing

Status: implemented
Priority: high
Owner/date: 2026-06-29

Hypothesis:
- `eval_sec` may be dominated by forward, H2D, readback, or CPU packaging. The
  next throughput choice depends on which one dominates.

Result (async@32, solver_cpus=6, `--profile_eval_timing`):
- eval 664.9s = forward 601.7s (90.5%) + h2d 26.7s (4%) + readback 29.6s (4.5%),
  50923 calls. **Forward-dominated; transfers/packaging negligible.**
- forward_bench (48ch/6b, cudnn_benchmark): ms/fwd scales linearly with batch and
  µs/sample is ~constant (~90µs) => **GPU-compute-bound, NOT launch-bound**.
  In-loop forward (11.8ms @ batch 137) == bench (12.46ms) => the forward is NOT
  solver-inflated. Peak 12.1k evals/s at batch 64 (82.6µs/sample); ~91µs at 137.

Decision:
- The laptop is GPU-forward-compute-bound; the solver is free. So:
  - Forward-dominated AND compute-bound: **channels-last** is the one real forward
    lever (targets conv compute). torch.compile / CUDA graph are LOW value here
    (little launch overhead to remove). Double-buffer marginal.
  - H2D / readback are tiny -> int8 transfer, pinned H2D, slim descriptors all
    deprioritized for this config.
- Biggest real win is the 5090 (more forward compute). On the laptop, ~0.14
  games/s is near the floor at 1600 sims.

Reopen if:
- Net size, sim count, or GPU changes (rerun the profile + forward_bench), or
  channels-last lands and shifts the forward cost.

## Channels-Last Inference

Status: planned
Priority: medium-high

Hypothesis:
- CUDA convolutions may run faster when board tensors and model weights use
  `channels_last` memory format.

Test:
- Add benchmark support for channels-last in `forward_bench.py`.
- Compare against current NCHW under TF32 and any AMP/compile variants.

Risks:
- The board tensors enter as NCHW NumPy arrays; conversion cost can erase forward
  gains unless handled carefully.

## Slim Legal Descriptors

Status: planned
Priority: medium-high

Hypothesis:
- Legal action ids fit in 12 bits (`NUM_JOINT_ACTIONS` around 3390), but some
  paths transport them as int64. Use `uint16` or `int32` across Python/Rust
  boundaries and cast to `torch.long` only at the final gather.

Success criteria:
- Same policies/results.
- Lower evaluator packaging time and memory traffic.

## `torch.compile` Inference Network

Status: planned
Priority: medium-high

Hypothesis:
- Fuse or reduce launch overhead for small conv/head network.

Known caveat:
- Requires Triton for meaningful CUDA inductor speedups.
- Dynamic batch sizes can cause compile/cache behavior surprises.

Decision rule:
- Keep only if steady-state timed reps improve after warmup and no shape
  instability appears.

## CUDA Graph / `eval_pad_to_batch`

Status: planned
Priority: medium

Hypothesis:
- Fixed-shape batches reduce kernel launch overhead.

Risks:
- Padding can waste forward work if live batch sizes vary too much.
- Works best after batch-size and overbooking behavior are stable.

## Single Prefetch Thread For Training `sample_batch`

Status: planned
Priority: medium

Hypothesis:
- Coarse prefetch of training batch N+1 can overlap CPU preparation with GPU
  train step, avoiding the fine-grained threading failure.

Decision rule:
- Keep if end-to-end training iteration time improves without queue/memory bloat.

## Overlap Self-Play With Training

Status: deferred
Priority: medium

Hypothesis:
- Generate iteration N+1 games while training on iteration N.

Risk:
- Changes on-policy/off-policy staleness. This is a training-quality decision,
  not just throughput engineering.

Decision rule:
- Evaluate after current self-play/training quality is stable.

## Pin Transfer + Non-Blocking H2D

Status: deferred
Priority: situational

Hypothesis:
- Pinned staging and non-blocking copy can reduce/overlap H2D.

Decision rule:
- Only prioritize if `--profile_eval_timing` shows H2D is material.

## int8/uint8 Board Transfer

Status: future idea
Priority: situational

Hypothesis:
- Board planes are mostly small/binary features; transporting as 1-byte ints
  could reduce H2D bandwidth.

Risks:
- Flat features remain float-like.
- Conversion to float on GPU may add overhead.
- Existing H2D may already be small relative to forward.

Decision rule:
- Only test if H2D is a meaningful fraction of `eval_sec`.

## Allocator Experiment (`mimalloc` / `jemalloc`)

Status: future idea
Priority: situational

Hypothesis:
- Rust MCTS tree operations allocate/free many small objects; a better allocator
  may help, especially on Windows or CPU-heavy cloud runs.

Decision rule:
- Profile allocation pressure first. Test allocator swap only if allocation shows
  up in CPU profiles or tree descent is the bottleneck.

## Backup / Update Pipeline

Status: future idea
Priority: medium

Hypothesis:
- Move tree backup/update/finalize work off the GPU lane so it overlaps with
  subsequent evaluator work.

Decision rule:
- Prioritize if `update_sec` or CPU post-eval work becomes material after solve
  overlap and forward optimizations.

## Eval Cache For Repeated Shallow Nodes

Status: deferred
Priority: medium

Hypothesis:
- Cache encoded-state to eval near roots where open-loop public state repeats.

Risks:
- Open-loop redeterminization and cache key correctness are non-trivial.

Decision rule:
- First measure hit rate on real self-play traces.

## Rayon Cross-Simulation Parallelism Within A Slot

Status: deferred
Priority: low

Hypothesis:
- Parallelize leaf-batch simulations within one slot.

Risks:
- Requires shareable or split arena design.
- High correctness risk for likely limited gain while GPU-bound.

Decision rule:
- Revisit only if CPU descent dominates on rented hardware.

## Multi-Worker IPC For CPU-Bound Regime

Status: situational
Priority: situational

Hypothesis:
- On CPU-rich/GPU-modest hardware, multiple worker processes with a shared
  inference server may outperform the single synchronized batched engine.

Decision rule:
- Test only for that hardware profile. Current `batched_open_loop` path expects
  workers=1.

## Closed-Loop `encode_arrays_into`

Status: deferred
Priority: low

Hypothesis:
- Apply direct-buffer encoding to closed-loop RustMCTS path.

Decision:
- Low priority because closed-loop is no longer the training path.

---

## Measurement Commands To Fill In

Representative commands should be recorded here after the exact run configs are
chosen.

```powershell
# Forward-only ceiling by batch size.
python -m games.kingdomino.forward_bench --device cuda --channels 64 --blocks 6 --cudnn_benchmark

# Batched self-play with evaluator timing.
python -m games.kingdomino.throughput_bench --run batched_open_loop --device cuda --games 100 --sims 1600 --profile_eval_timing
```

When a command becomes the accepted benchmark, paste the full command and result
into the relevant experiment card above.
