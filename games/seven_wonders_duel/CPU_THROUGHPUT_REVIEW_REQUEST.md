# CPU throughput work — review request

**Date:** 2026-08-04. **Scope:** generation throughput for 7 Wonders Duel, laptop
development only. **Outcome:** 1.497× on generation, ~1.38× at the training-loop
level, all committed and gated.

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
| row-parallel packing, 16 threads | 1.189× | — (see §4.6) | **1.497×** |

**1.061 → 1.588 games/s** at S/fp32, 192 slots, cap 2048.

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
* **86.7% of the feature payload is zeros.** `net.py` projects each token type
  with `nn.Linear(FEATURE_COUNTS[type], d_model)` — counts are
  `[130, 1, 26, 1, 8, 4, 1, 79, 14]` — but the wire format writes 130 floats per
  token regardless.
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

**Consequence of the waves result:** the overlap ceiling is unreachable by any
existing knob. The two threads alternate (~22 s scheduler, ~104 s blocked), so
wall is their sum and overlapping caps at **1.21×** — but overlap needs surplus
ready work, and waves were the only remaining mechanism that could create it at
`leaf_batch = 1` (more slots was already null past 96). The cap is algorithmic,
not hardware, so it will not differ on the box.

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

### 4.4 Thread-count conclusions rest on unknown topology

`os.cpu_count()` = 16 **logical**; physical cores were never checked. I inferred
SMT from the 8→16 efficiency drop (62% vs 73%), which is a curve-shape argument,
not a measurement — 8P+8E would look similar for a different reason. The box's
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

## 5. Open, in priority order

1. **GPU-side reassembly of the compact format.** Savings measured (≈8.5 s,
   6-7% of wall); only the CPU-side reconstruction sank it. Needs
   `build_device_batch` restructured to upload compact and expand on device.
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

## 6. Process notes worth keeping

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
