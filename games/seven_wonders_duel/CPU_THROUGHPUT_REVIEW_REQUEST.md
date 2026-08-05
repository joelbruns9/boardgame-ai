# CPU throughput work — review request

**Date:** 2026-08-04, revised 2026-08-05 after review. **Scope:** generation
throughput for 7 Wonders Duel, laptop development only. **Outcome:** **1.521× on
generation, ~1.39× at the training-loop level**, all committed and gated.

**What this document is for.** The wins are measured and I believe them. What I
want checked is §4 — the assumptions and leaps underneath them, several of which
are load-bearing and unverified. §3 lists what was tried and failed, because the
negative results cost real time and shouldn't be repeated.

Detail lives in `CORE_UTILIZATION_PLAN.md`; this is the reviewable summary.

---

## 1. Wins

| step | gain | 95% CI | cumulative |
|---|---:|---|---:|
| mimalloc global allocator | 1.2165× | [+19.2%, +24.1%] | 1.2165× |
| `TokenBuf` (remove per-token allocations) | 1.0351× | [+1.23%, +5.79%] | 1.2592× |
| row-parallel packing, 16 threads | **1.2078×** | [1.1776, 1.2379] | **1.521×** |

**1.061 → 1.588 games/s** at S/fp32, 192 slots, cap 2048.

> **§4.6 RETIRED.** The parallel-packing figure was re-measured paired — one
> build, `RAYON_NUM_THREADS` 1 vs 16 (the one-thread guard routes to the serial
> loop), same five seed sets, work identical every repetition. **1.2078×
> [1.1776, 1.2379]**, so the unpaired 1.189× was *conservative*. Cumulative
> corrected 1.497× → **1.521×**.

At the loop level: generation was 82.3% of measured phase time across the 30
preserved cloud iterations (replay derivation 9.8%, gates 6.0%, training 1.9%),
so 1.497× on generation is **~1.38× on the loop**. Quote the loop figure.

### Diagnostic findings that changed the direction of the work

* **The plan's core diagnosis was wrong about which thread.** It held that one
  scheduler thread doing per-row MCTS work was the serial resource. Measured, that
  thread is **idle 82.6%** of the time. The critical path is the *worker* thread,
  which does pack → Python call → extract strictly in sequence. `encode_pack`
  runs there (`eval.rs:580` via `eval.rs:885`), not on the scheduler.
* **All prior profiling was accidentally fp32 while production runs bf16.**
  `load_evaluator` (`phase_e.py:503`) defaults `precision="fp32"` and the bench
  never passed one — including against a checkpoint whose own config says bf16.
  bf16 measures **2.19× end-to-end at L** and a **1.01× null at S**.
* **86.7% of transmitted feature values are zeros.** `FEATURE_COUNTS` is
  `[130, 1, 26, 1, 8, 4, 1, 79, 14]`, but the wire format writes 130 floats per
  token regardless.

  > **CORRECTED (review finding 3).** An earlier version of this line said the
  > network "projects each type with `nn.Linear(FEATURE_COUNTS[type], d_model)`
  > and reads no further". That is the **unfused** path (`net.py:272`).
  > Production inference uses `_forward_fused` (`net.py:243-247`), which applies
  > one 130-wide projection producing **all nine type outputs for every token**
  > and gathers the selected one afterwards. So the network *does* multiply
  > through the zeros, deliberately, to avoid the per-type loop's host syncs.
  >
  > The 86.7% figure stands as **wire, copy and H2D waste**. It does **not**
  > describe projection arithmetic, and compact-upload-plus-dense-GPU-rebuild
  > would not remove any of that arithmetic. The stronger avenue is type-aware
  > projection that never builds the dense rectangle — pre-grouped fixed-slice
  > GEMMs, or a kernel using per-type width directly — which is plausibly worth
  > more at L than rebuilding zeros on the device.
* **Sequence padding is worse than the tracked metric said**: quadratic 0.309
  against the linear 0.180 the plan had been quoting. Attention is quadratic in
  length.
* **Extraction is not a cost**: 25.95 s of `extract_ns` against 6,404 s of
  `encode_pack_ns` across the whole real cloud run — 0.4% of packing.

### Instrumentation added

Scheduler-thread partition (6 timers, closes to 100.2%); worker-thread partition
(4 timers, 99.7%); `--inference-precision`; both padding dimensions;
`f4_pack_sweep.py`, a cgroup-driven packing microbenchmark.

---

## 2. Tests and gates

Every accepted change passed **49 tests** — `test_encoder.py` (Rust vs Python
encoder, bit-exact), `test_f4_boundary.py`, `test_f4_phase0_telemetry.py`.

Additionally, for each accepted change, **work identity per repetition**:
`simulations`, `moves`, `global_batches` and `global_batch_leaves_mean` matched
exactly against the pre-change build. This is the check that separates "same work,
faster" from "less work".

**Statistical protocol** (after review flagged that every early profile used
`--repetitions 1` against a contract requiring 5): `f4_throughput_bench` with ≥5
repetitions, mean and CI on the ratio. Baseline CV is **1.63%**.

---

## 3. Measured failures — do not repeat

| attempt | result | why |
|---|---|---|
| **Cheap zero-write** — keep the 130-wide row but write the real prefix and bulk-append a zeroed tail | unmeasurable | Same bytes still reach memory; only instruction count fell. The write is bandwidth-bound, not issue-bound. |
| **Compact wire format with CPU-side reassembly** | **−12.2%** [−13.9%, −10.5%] | Savings were real and as predicted (payload −83%, pack −36%, ≈8.5 s). PyTorch advanced indexing on CPU cost ≈11.8 s to rebuild the rectangle. Reconstruction must be on the **device**. |
| **Conflict-free waves**, `leaf_batch` 2/4/8 | **−8 to −9%** | Wave width saturates at **1.18** asking for 8: Gumbel sequential halving narrows the live root-candidate set, producing 425,654 conflict cuts. Batches widen only 9%; bookkeeping costs more. Bit-identity held. |
| `--pinned-memory` | not run | Prior negative result in the Kingdomino implementation (user). |
| `torch.compile` | **cannot be tested on this laptop** | `TritonMissing` — Triton is unavailable on Windows. Linux-only; belongs on the box. |

**Consequence of the waves result — CORRECTED (review finding 2).** The earlier
claim that this "closes the overlap avenue" and that the 1.21× cap is
"algorithmic, not hardware" was wrong on both halves:

* **Waves only test surplus leaves *within one search tree*.**
  `run_many_pipelined_sharded` already spawns one scheduler thread per worker, so
  **independent games across shards can produce concurrent requests at
  `leaf_batch = 1`.** `--rust-scheduler-workers` is an existing knob that creates
  surplus work, and it remains **unmeasured**.
* **1.21× is not a fixed bound.** It is
  `(scheduler + worker) / max(scheduler, worker)` for one measured timing split,
  and it moves whenever packing, GPU time or scheduler time moves — all three of
  which this work changed.
* **Perfect-overlap arithmetic assumes independent resources.** After
  row-parallel packing, scheduler shards and rayon contend for the same cores, so
  the attainable gain is likely below the arithmetic ceiling.

**Correct conclusion:** conflict-free waves are a laptop-regime regression and
should be retired. **Scheduler sharding is untested and must be swept jointly
with pack-thread count and inflight depth**, since all three contend for the same
CPUs.

---

## 4. Assumptions and leaps — **this is what I want checked**

### 4.1 The headline number is laptop S/fp32; production is L/bf16 on a 5090

Composition differs sharply: this laptop is now **56.8% device**, the box
extrapolates to **~22%**. Per-term gains should transfer; the 1.497× ratio should
not be quoted for production. **Nothing here has been verified on the target
hardware.**

### 4.2 mimalloc — the largest single win — may not transfer, and has never been built on Linux

1.2165× was measured on **Windows**, whose default heap is a weak baseline;
glibc's malloc is considerably stronger. The gain could be materially smaller on
the box. Worse, **mimalloc compiles C**, and neither it nor rayon has ever been
built on Linux. A provisioning-time build failure on a rented box is the failure
mode. **Highest-priority item to verify.**

### 4.3 The bf16 production extrapolation is a 3070 measurement applied to a 5090

The "~22% device share on the box" comes from scaling the box's fp32 forward by a
**3070-measured 2.88×**. It matches the ~23% seen on the live dashboard, which is
corroboration but not measurement. Every production-regime statement rests on it.

### 4.4 Thread-count conclusions rest on partially-known topology

`os.cpu_count()` = 16 **logical**; physical cores were never checked. I inferred
SMT from the 8→16 efficiency drop (62% vs 73%), which is a curve-shape argument,
not a measurement — 8P+8E would look similar for a different reason.
**RESOLVED by review:** the laptop is an i7-11800H, **8 physical / 16 logical**,
so the 8→16 falloff is SMT as inferred. The inference was right; it was still an
inference. The box's
topology and cgroup quota are unknown, and `f4_pack_sweep`'s cgroup path **has
never executed** (it fell back to `os.cpu_count()` on Windows).

### 4.5 `f4_pack_sweep`'s corpus is synthetic

It walks `legal[(index + step) % len]` for 40 plies. Token-type distribution, and
therefore per-row packing cost, may not match self-play states.

### 4.6 The 1.189× parallel-packing figure carries a repetition-count confound

The thread sweep ran **3** repetitions against a **5**-repetition serial baseline,
so the two averaged different seed sets. Per-repetition work identity was
verified, and the per-term movements are coherent, but **the headline 1.497%
inherits this**. Re-measuring parallel packing at matched repetitions is the
cheapest way to firm up the top-line number.

### 4.7 Padding ratios are single-run

0.180 / 0.309 / 0.867 come from one 100-game, 1-repetition run. They are
structural ratios so noise should be low, but they are unreplicated and 0.867 is
the basis for the largest remaining opportunity.

### 4.8 Loop-level 1.38× uses pre-optimisation phase shares

Generation at 82.3% was measured before any of this work. The arithmetic is
correct for that input, but the phase shares came from one run at one
configuration.

### 4.9 The advisor claim is untested

`advisor_adapter.py` takes `load_evaluator`'s fp32 default, so bf16 should be
~2× of live inference — **but I never confirmed the advisor runs on CUDA**, and
`Evaluator.autocast` is a deliberate no-op on CPU. If it runs on CPU the claim is
void.

### 4.10 Option B's savings are assumed to survive relocation to the GPU

Payload −83% and pack −36% were measured in the configuration that lost overall.
The Rust side would be unchanged in a device-side version, so they should carry,
but that is an inference.

---

### 4.11 The measured pack thread count does not control production (review finding 1)

**`f4_pack_sweep` installs a *scoped* pool; production packing uses rayon's
*global* pool** (`eval.rs:427`). Nothing converts the sweep's recommendation into
a production setting, and `RAYON_NUM_THREADS` is not recorded in the benchmark
manifest. The laptop happens to expose 16 CPUs so the default matched the winning
rung — **coincidence, not control.** A rental exposing 192 host CPUs while
selling a smaller quota would oversubscribe badly and erase the gain.

`container_limits` (`cloud_preflight.py:282-322`) compounds this: it reads CFS
quota but **not `cpuset.cpus.effective` or the process affinity mask**. A
cpuset-limited container with no CFS quota falls back to the host count.
Effective parallelism should be the minimum of quota, cpuset/affinity count and
visible count.

**Fix:** a persistent explicitly-sized pack pool behind `--pack-threads`, with
requested *and actual* thread count, affinity, quota and cpuset recorded per run.

### 4.12 Packing still has a serial duplicate-copy tail (review finding 4)

Only `RowPack::fill` is parallel. Every row-local buffer is then copied into the
global payload by a **serial** loop (`eval.rs:447-468`), which holds both copies
simultaneously and puts a one-thread memory-bandwidth ceiling on scaling as cores
rise. A two-pass design — parallel encode into retained `TokenBuf`s, prefix-sum
the token counts, pre-size the final buffers, then flatten in parallel into
disjoint slices — removes both the row-local buffers and the serial copy.

---

## 4b. Post-review work — four items, two wins and two negatives

| item | result |
|---|---|
| **Paired re-measure** (§4.6) | **1.2078× [1.1776, 1.2379]**. Retires the confound; top line up to 1.521×. |
| **Pack-thread control** (finding 1) | **Landed.** `set_pack_threads()` installs a dedicated sized rayon pool; packing uses it when present, global pool otherwise. Python owns detection — CFS quota, `cpuset.cpus.effective` (v1/v2) and `sched_getaffinity`, combined as a **minimum**. `--pack-threads 0` derives it. Manifest records **requested, actual, and every limit**. Costs nothing: 0.9846×, CI [−5.1%, +2.0%]. |
| **`scheduler_workers`** (finding 2) | **Negative.** 2 workers flat (1.0176 [0.985, 1.050]); 4 workers **0.8933 [0.852, 0.935]**. Batch width collapses 240.7 → 149 → 84, ~inversely with shard count. Work identical throughout. |
| **fp16 on the wire** | **Negative, −9.7%.** Savings landed as predicted (payload −37.4%, tensor build −7.6%) but the f32→f16 conversion cost **+41.2%** on pack. Reverted. |
| **`-C target-cpu=native`** | **Null on the shipped config**: 1.0097× [0.958, 1.062]. Mixed per-term (pack +6.6%, `sched_collect` −8.0%). Not adopted. |

**Sharding needs the merge first — this corrects both sides of the review
exchange.** My claim that waves "closed the overlap avenue" was wrong: sharding
*does* create surplus work at `leaf_batch = 1`. But `--rust-scheduler-workers` is
not a standalone route to overlap either, because N shards produce N proportionally
narrower batches and that cancels the gain. **The merging worker is the
prerequisite, not a companion** — merge first, then shard.

**fp16 is dead even with hardware conversion.** `target-cpu=native` did what it
should — F16C cut the conversion, pack 12.05 → 9.91 s — but fp16+native is still
**0.8995× [0.879, 0.921]** against fp32+generic. Halving the buffer does not pay
for converting into it.

**A note on `target-cpu=native` if it is ever revisited:** it bakes the *build*
machine's ISA into the artifact. Harmless while the cloud script builds on the
box, but a cached or moved binary would fault on an older CPU — a crash, not a
slowdown.

---

## 5. Open, in priority order

**Re-ordered after review.** The five items below were raised by the reviewer and
are cheaper and better-founded than what this document originally ranked first.

1. **Transmit features as fp16.** Rust writes f32 and `rust_bridge.py:322`
   immediately does `f32 → fp16 → f32` — **the extra precision is discarded
   anyway**. Packing IEEE fp16 directly halves the dominant buffer, removes a
   conversion, and keeps the format rectangular. Laptop-testable now; gate exact
   rounding against the current path. *This is the best remaining laptop item.*
2. **Fix pack-thread control** (§4.11): persistent sized pool, `--pack-threads`,
   cpuset/affinity-aware detection, thread count in the manifest. Correctness of
   every future sweep depends on it.
3. **Sweep `pack_threads × scheduler_workers × inflight` jointly** (§ waves
   correction). Scheduler sharding is untested and is the live route to overlap.
4. **Remove the `PyByteArray` copy** — ~4-5% of wall. Lifetime-safe view over the
   locked Rust scratch, or reusable Python-owned staging.
5. **Keep removing encoder allocations.** `TokenBuf` took the per-token feature
   vectors; each row still allocates `UnseenPool` vectors, obtainable/relevant
   lists, per-player vectors, tableau feature vectors and two cost vectors per
   pool. Fixed arrays / `SmallVec` / retained row scratch.
6. **Build with `-C target-cpu=native`** and compare against the generic release
   build. Do not assume the rented build uses available SIMD. PGO later.
7. **Reusable pinned staging slabs** — the Kingdomino negative was
   `pin_memory=True` allocating fresh tensors per batch, which does not rule out
   persistent slabs for this payload and batch regime.
8. **Parallelise the serial flatten tail** (§4.12).
9. **GPU-side reassembly of the compact format.** Savings measured (≈8.5 s,
   6-7% of wall); only the CPU-side reconstruction sank it. **Demoted:** per the
   §1 correction it removes wire/copy/H2D waste but *not* fused projection
   arithmetic, so type-aware projection is the stronger version of this idea.
2. **Scheduler shards + merging worker** — the only mechanism that reaches the
   **1.21×** overlap ceiling. Correctly sized now; the plan originally hoped 2-3×.
3. **`sched_collect`, 8.1%** — MCTS descent, masks, expansion. Never examined. On
   the critical path despite the scheduler thread being idle-blocked, because
   nothing overlaps.
4. **Unpartitioned 4.7%** of the scheduler loop, still unexplained.
5. **Box-only:** `torch.compile`, thread-count confirmation, and a real L/bf16
   profile.
6. **`load_evaluator`'s fp32 default** in `advisor_adapter.py` — separate from
   throughput, likely ~2× of advisor latency, gated on 4.9.

---

## 6. Required measurements before this is a cloud plan (from review)

* Rerun serial vs final row-parallel packing on the **same five seed sets** and
  report a **paired** CI — this retires §4.6.
* Save a **representative corpus of real self-play states** for the pack sweep,
  replacing the synthetic walk (§4.5).
* Add a **full record/trajectory digest** to A/B gates. Aggregate simulations,
  moves and batch counts prove equal work *quantity*, not identical work.
* Record pack-thread count, physical/logical topology, affinity/cpuset, NUMA
  placement, CPU frequency, CPU time, RSS and per-stage timing in every run.
* Measure `pack_threads × scheduler_workers × inflight` **jointly**.
* On the first rental, run a short L/bf16 matrix and **recompute the overlap
  ceiling from that profile** rather than carrying 1.21× forward.
* **Add a Linux CI build now** for mimalloc/rayon. A rental is needed for
  performance, not for catching a provisioning-time compile failure (§4.2).

---

## 7. Process notes worth keeping

Three false results were produced and caught during this work, all from the same
error class — **comparing runs whose conditions differed**:

* averaging a 5-repetition reference against a 3-repetition sweep and reading the
  different seed sets as a behavioural divergence;
* an apparent **+25.6%** from `f4_pack_sweep` for a change with no real effect —
  five runs of one *unchanged* build spanned **26%** (123,431 to 155,360 rows/s),
  because its `spread` metric is within-process while builds are compared across
  processes;
* an early conclusion that the packing optimum depended on batch width, when the
  argmax was simply wandering inside the noise floor.

`f4_pack_sweep` is valid for ranking thread counts **within one process** and
invalid for comparing builds. Cross-build A/Bs belong in `f4_throughput_bench` at
≥5 repetitions.
