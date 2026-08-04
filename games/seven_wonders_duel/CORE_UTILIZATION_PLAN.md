# Spending the cores: a throughput plan for 7 Wonders Duel generation

**Status (2026-08-04):** Phase 0 complete. The pipeline is fully profiled, the
original diagnosis is superseded, and **nothing has been built yet**. The work
below is re-ordered against measurements, not arithmetic.

**Headline:** generation runs on two threads. One of them is **idle 82.6% of the
time**; the other does three jobs strictly in sequence, and only one of those
three is the GPU. Every previous revision of this plan proposed adding
parallelism to the *idle* thread.

**Build order:** parallelise the pack → move extract/validation off the worker →
padding → (only then) reconsider scheduler threads.

This document states the current understanding first. **Appendix A** records how
it changed and what was wrong, because five of the corrections came from
measurements that contradicted plausible reasoning, and the reasoning is worth
not repeating.

---

## 1. The pipeline, as it actually runs

Generation runs `run_many_pipelined_sharded` (`self_play.rs:1882`) with
`leaf_batch = 1`, `max_inflight_batches = 1`, and **one scheduler thread** and
**one worker thread**.

**Scheduler thread** — owns the games. Per loop iteration it refills activation
slots, advances every ready game's search until it needs a network evaluation
(`collect_ready_groups`), assembles those pending positions into one batch,
submits it, **blocks until results return**, then scatters values and priors back
into the trees.

**Worker thread** (`spawn_py_flat_worker`, `eval.rs:878`) — the only thread that
attaches to Python. Per batch, strictly in sequence:

1. **Pack** — `pack_routed` (`eval.rs:377`) turns N `GameState`s into flat byte
   buffers. Pure Rust CPU.
2. **Python call** — build tensors over those bytes, H2D, transformer forward,
   gather over legal actions, D2H.
3. **Extract + validate** — convert the returned nested Python lists into
   `Vec<(f64, Vec<f64>)>`, then check every prior is finite, non-negative and
   sums positive.

**Nothing overlaps.** `max_inflight_batches = 1` means the scheduler submits one
batch and immediately waits; its own work happens only after results return. The
two threads alternate, so the wall clock is their *sum*, not their maximum.

---

## 2. The measurement that matters

**S width, fp32, 400 games, 192 slots / cap 2048 / inflight 1, `--cuda-events`.
Wall 362.2 s, 1.104 games/s.** Six scheduler-thread timers added for this
(Phase 1a); the partition closes to 100.2% and instrumentation cost 0.4%.

| | s | % wall | thread |
|---|---:|---:|---|
| **scheduler blocked in `ticket.wait()`** | **299.13** | **82.6%** | scheduler |
| ├─ pack | 104.02 | 28.7% | worker |
| ├─ Python call | 180.36 | 49.8% | worker |
| │  ├─ device (fwd 126.30, h2d, gather, d2h) | 148.76 | 41.1% | GPU |
| │  └─ tensor build | 23.49 | 6.5% | worker |
| └─ extract + validation | 14.76 | 4.1% | worker |
| `collect_ready_groups` (tree, masks, expansion) | 29.86 | 8.2% | scheduler |
| scatter + extract *(combined, see §7)* | 7.89 | 2.2% | both |
| batch assemble (clone into Vecs) | 6.72 | 1.9% | scheduler |
| unpartitioned loop bookkeeping | 19.07 | 5.3% | scheduler |
| refill / retire / submit | 0.05 | 0.0% | scheduler |

**Read it as two columns.** The worker's serial chain is 299.13 s. The
scheduler's own work is ~63 s. They do not overlap, so wall = 299 + 63 = 362.

### What each thread is actually doing

* **The scheduler thread is idle 82.6% of the time.** It is not the bottleneck
  and never was.
* **The worker thread is the critical path**, and only 41% of it is the GPU. The
  rest — packing, tensor build, extraction, validation — is CPU work that happens
  to sit in the same queue as the forward.
* **Packing alone is 28.7% of wall**, over two-thirds the size of the entire
  device term.

---

## 3. Where the time can come from

Sized against the S fp32 measurement. Each is independent of the others.

### 3.1 Parallelise the pack *(largest, and genuinely spends cores)*

`pack_routed` is a `for` loop over rows calling `encoder::encode(state)` on
independent `GameState`s. **No cross-row dependency exists.** The only thing
forcing sequence is that rows append into shared output buffers — and each row's
output size is computable before encoding, so rows can encode in parallel into
pre-sized slices.

* **Prize: 104.02 s, 28.7% of wall**, on the critical path.
* **This is the first item in this plan's history that uses more than one core.**
  Everything previously proposed added threads to the idle scheduler.
* **Determinism is the easy kind**: independent rows written into pre-sized
  slices produce byte-identical output. Gate with a direct comparison of the
  packed buffers, which is far stronger and cheaper than the search-level
  equivalence a batch-merging change would need.
* **It grows in production.** Packing is CPU work, so bf16 does nothing for it,
  while bf16 roughly halves the device term (§5). The faster the GPU side gets,
  the more packing dominates.

### 3.2 Move extract + validation off the worker

* **Prize: 14.76 s, 4.1% of wall.** A seventh of the pack, so not worth doing
  alone — but it is the same refactor and the same thread, so do it in the same
  pass.
* Two independent wins available:
  * **Flatten the return path.** `out.extract::<Vec<(f64, Vec<f64>)>>()`
    allocates one `Vec<f64>` per row: ~440 per batch × ~9,000 batches ≈ 4 M heap
    allocations, each filled element-by-element through PyO3's generic
    conversion. The *outbound* direction already solved this with flat buffers
    and offsets; the return can mirror it.
  * **Move validation to the scheduler thread**, which is idle. It is
    O(total priors) and currently sits on the busiest thread in the system.
* **Its share grows as the bigger items shrink**: 7.6% of the worker's critical
  path once packing is parallelised, ~10% once padding is also fixed.

### 3.3 Padding

`padding_ratio` is **0.187-0.263** depending on configuration (§5). It multiplies
the device term, so at S fp32 it is worth up to ~10% of wall — but see §5 for why
it must be quoted per configuration rather than as a constant.

### 3.4 Overlap the two threads

With `inflight = 1` the scheduler's ~63 s never overlaps the worker's 299 s.
Fixing that is worth **at most 1.21×** on its own (362 → 299). Note §6 records
`max_inflight_batches > 1` as a *measured null* — because with `leaf_batch = 1`
there is no surplus ready work to pipeline. Overlap only becomes available
alongside a change that creates surplus work.

### 3.5 Stacked estimate

At S fp32, with pack parallelised ~8×, extract/validation moved, padding fixed,
and the threads overlapped: worker ≈ 155 s, scheduler ≈ 78 s, wall ≈ **155 s**
against today's 362 s — roughly **2.3×**.

**This is arithmetic, not a measurement.** Arithmetic has been wrong three times
in this document's history (Appendix A). Treat it as an ordering argument, not a
forecast.

---

## 4. What this retires

**More scheduler threads (`--rust-scheduler-workers`) is no longer the plan.**
Even making the scheduler infinitely fast caps at **1.21×**, and the extra games
it would activate still queue behind the same single worker. The original plan's
Phases 1 and 2 were both aimed here.

**The merging eval worker is demoted, not deleted.** Its stated rationale was
partly latency — but `queue_wait_seconds` is **0.22 s**, so the scheduler never
waits to hand work over. It becomes relevant only if multiple scheduler threads
are ever introduced, which §3.4 makes unlikely to be worth it.

**"Build tensors in Rust" (old Phase 3a) is capped at 6.5%** — the measured
`pyo3_tensor_seconds` share.

---

## 5. Regime: width and precision change the answer

**Every throughput claim must state model width and precision.** The same code is
CPU-bound at S and GPU-bound at L on identical hardware.

| run | wall | games/s | device share | pack | padding |
|---|---:|---:|---:|---:|---:|
| box L fp32, 400g | 396.1 s | 1.010 | 41.6% | 81.8 s | 0.261 |
| laptop L fp32, 200g | 774.2 s | 0.258 | 87.1% | 47.4 s | 0.187 |
| laptop L bf16, 200g | 353.1 s | 0.566 | 68.8% | 54.0 s | 0.192 |
| laptop M fp32, 400g | 765.8 s | 0.522 | 70.2% | 113.8 s | 0.261 |
| laptop S fp32, 400g | 360.7 s | 1.109 | 41.3% | 104.0 s | 0.259 |
| laptop S bf16, 400g | 356.9 s | 1.121 | 41.4% | 103.2 s | 0.263 |

S/M runs are 400 games and L runs 200: **compare shares and ratios, never raw
seconds.**

**bf16 is 2.19× end-to-end at L and a 1.01× null at S.** A 1.03 M model cannot
occupy tensor cores; a 14.9 M one can. Asking at S first returned a null that
would have closed the question wrongly had it stopped there.

**Production runs L at bf16.** Scaling the box's forward by the measured 2.88×
puts its device share at **~22%**, which independently corroborates the ~23% GPU
seen on the live dashboard. So in production the CPU-side terms — packing above
all — dominate more than any table here shows directly.

**Padding is not a constant.** It varies with the games-per-call to slots ratio:
200 games on 192 slots keeps games phase-aligned and token lengths uniform (0.19);
400 games spreads them across phases (0.26). It is a property of generation
geometry, not only of the encoder.

**Development venue.** Laptop S at fp32 reproduces the *box's fp32* balance
closely (device/Rust 1.05 vs 1.16; Amdahl 2.00× vs 2.01×; ceiling 2.42× vs
2.41×). It does **not** reproduce production's bf16 balance (~0.47), and no
laptop configuration does — S is the smallest model available and bf16 is a null
there. **Develop and correctness-gate on the laptop at S; do not quote laptop
speedups as production speedups.** The throughput number needs a bf16 L box.

---

## 6. Rejected, with the measurement that killed each

| lever | result |
|---|---|
| `OMP_NUM_THREADS` tuning | null: 4,032 vs 4,187 unset, noise band ~4% |
| `max_inflight_batches` 2 / 4 | null: 1.03× / 0.99×. At `leaf_batch = 1` one batch drains every waiting leaf |
| more slots at fixed cap | 144 slots underperformed 96 at cap 1024 |
| more scheduler threads | §4: caps at 1.21×, and the single worker still serialises |
| build tensors in Rust | capped at 6.5% of wall |
| multi-box sharding | out of scope; one GPU is not the limit |
| **renting more cores** | ~~prefer clock over count~~ **see §8** |

---

## 7. Instrumentation: what is trustworthy and what is not

**Trustworthy.** The Phase 1a scheduler timers (`sched_refill_ns`,
`sched_collect_ns`, `sched_retire_ns`, `sched_assemble_ns`, `sched_submit_ns`,
`sched_wait_ns`) are all taken on one thread and partition the loop to 100.2%.
`--cuda-events` gives true device time without perturbing the pipeline.

**Three traps, each of which produced a confident wrong answer here:**

1. **`gather_d2h_seconds` is a decoy.** It read 703 s against a true
   `device_gather_seconds` of 5.4 s. The gather is the first host operation to
   touch the result, so it absorbs the entire asynchronous forward wait. **Read
   `device_*`, never the host timers.**
2. **`py_call_ns` is measured on the worker thread**, and the Rust work counters
   on the scheduler. Adding them and calling the remainder "unaccounted" is not a
   partition of anything — it was how the old 18-25% residual was computed.
   `sched_wait_ns` is the honest version.
3. **`--diagnostic-sync` is the wrong instrument.** It perturbs what it measures.
   `--cuda-events` queries after the run.

**Known gaps.**

* `scatter_seconds` in the bench sums `scatter_ns` (scheduler) and `extract_ns`
  (worker) — two threads in one field. Split it when next touching this.
* **5.3% of the scheduler loop is unpartitioned.** Harmless today because the
  thread is idle anyway, but it is not explained.
* `sched_collect_ns` is *inclusive* of per-slot tree and chance work.

---

## 8. Hardware

**Instance selection: a fast 8-16 core part with strong single-thread
performance. The GPU is already ahead of what the pipeline can feed.**

* The GPU is not the bottleneck in any configuration measured: 22% (production,
  extrapolated) to 42% (box fp32).
* **Cores now matter, because §3.1 finally has something to spend them on.**
  Packing parallelises across rows; how far is an open question (§9).
* Single-core clock still matters for the serial remainder — the tensor build and
  extraction are effectively single-threaded, and the Python call is GIL-bound.
* Core *count* beyond what packing can use buys nothing.

**Caveat:** production's bf16 device time is extrapolated from a 3070-measured
2.88× forward speedup, not measured on a 5090. A 5090's bf16 advantage is likely
larger, which strengthens rather than weakens the case for CPU-side work.

---

## 9. Open questions

1. **How far does packing parallelise?** Rows are independent, but the win
   depends on per-row cost against thread-dispatch overhead at ~440 rows/batch.
   Measure before assuming linear.
2. **What is the unpartitioned 5.3%?**
3. **What is the extract/scatter split?** Currently one field over two threads.
4. **bf16 on a real 5090** — every production estimate here rests on a 3070
   extrapolation.
5. **Does `load_evaluator`'s fp32 default bite elsewhere?** `advisor_adapter.py`
   (`:399`, `:537`) takes it. If the BGA advisor runs on CUDA at L width, it is
   leaving ~2× inference on the table, which converts directly into search depth
   inside its 30 s budget. The quality tools taking fp32 is defensible.

---

## 10. Gates any change here must pass

* **Byte-identical packed buffers** for the pack change, mock evaluator, single
  thread. This is the strong, cheap gate parallel-independent-rows makes
  available.
* **Never claim "byte-identical records" on a real net.** Batch shape can change
  search choices through float ties; the sweep campaign observed 6 distinct
  trajectory sets across 6 geometries. Fingerprint the discrete trajectory and
  report divergence as a measurement.
* **Beat the ~4% noise band**, established from a repeated 200-game point
  (4,187 then 4,032).
* **State width and precision** in every claim.
* **Metrics attribution**: if a counter changes meaning, say so in its doc
  comment, or numbers stop being comparable across the change.

---

# Appendix A — how this document changed, and what was wrong

Five revisions in one day, each from a measurement contradicting plausible
reasoning. Recorded so the reasoning is not repeated.

**A0. The original theory.** 48 cores, generation using ~1, GPU ~23%. Inferred
from slot-scaling that per-row MCTS work — descent, make/unmake, 1,202-action
masks, expansion, backprop, "plus `encode_pack`" — was "all on one scheduler
thread", making that thread the serial resource. Proposed: a merging eval worker,
then more scheduler threads. Hoped 2-3×.

**Right that the GPU was not the limit. Wrong about which thread.** `encode_pack`
runs on the *worker* thread (`eval.rs:580`, called from `eval.rs:885`), and the
scheduler thread is idle 82.6% of the time.

**A1. "The run will emit this telemetry for free" — false.** `phase_d.py`
consumes none of the boundary counters. Waiting for the training run would have
yielded nothing.

**A2. "Phase 0 needs building" — false.** `f4_throughput_bench.py:576` already
carried a `# --- Phase 0: host/device separation ---` block, gated by
`test_f4_phase0_telemetry.py`. It was a run and a read. The original term list
also omitted `rust_tree_ns`, the one counter that would have tested its own
hypothesis.

**A3. The first profile ran on the laptop at L and was GPU-bound at 84%**, so it
could not answer a question about what limits a machine whose GPU is not the
limit. Corrected by profiling the box before it was destroyed.

**A4. "M width reproduces the box's regime" — wrong, measured 66.9% device.** The
error was reusing the S-vs-L inference ratio (7.4×) to justify M, which is only
2.26×. S was the right width, and measured 1.05 against a predicted 1.06.

**A5. Every figure was fp32 while production is bf16.** `load_evaluator`
(`phase_e.py:503`) declares `precision: str = "fp32"` and the bench never passed
one, so the forward was always fp32 — including against a checkpoint whose own
config says bf16. The lock meanwhile *asserted* `"inference_precision":
"float32"` into the manifest rather than recording it. Same failure class as two
pre-rental defects: a parameter nobody passed taking a default nobody chose.
Fixed with `--inference-precision`.

**A6. The 18-25% "unaccounted" residual was partly an artifact of mixing
threads.** It was `wall − (rust terms + py_call_ns)`, adding scheduler-thread and
worker-thread clocks. Phase 1a replaced it with a true single-thread partition,
which closes to 100.2% and shows the real structure.

**Estimates over time**, all superseded: 1.56× → 2.01× → ~3.0× → the §3 model.
Each was arithmetic on an incomplete partition. The current numbers rest on a
partition that closes.

**What stayed true throughout:** the GPU is not the bottleneck; something
CPU-side is; and **nothing was built** — five revisions, zero wasted engineering,
which is what Phase 0 existed for.

---

# Appendix B — the operational failure that freed the box

The 200k-game run started 05:03 UTC and stopped at **13:45 UTC during iteration
30**, having completed 0-29 (~30,000 games). `run.log` ends immediately after
iteration 30's league-play banner: no traceback, no `Killed`. Clean SIGHUP.

**Cause: launched from the vast.ai Jupyter terminal, which has no tmux.** SSH
sessions on that image get auto-tmux and would have survived. The tunnel dying,
the SSH reverse port-forward being refused, and the run stopping were one event.

**Rules for the next rental:**

* **Launch long runs over SSH, never the Jupyter terminal.** Confirm the tmux
  banner first. Do not `touch ~/.no_auto_tmux`.
* `trycloudflare.com` names are ephemeral; a restarted `cloudflared` mints a new
  one, so a dead bookmark says nothing about the run.
* **Liveness check that works from any container or PID namespace:**
  `date -u; stat -c '%y %s' <run-dir>/run.log`. `pgrep`/`top` mislead — the
  Jupyter terminal cannot see the run's processes even when it is healthy.
* `phase_d.py` has **no shebang and absolute imports**: use
  `python -m games.seven_wonders_duel.phase_d` from the repo root, with
  `/venv/main` activated (an SSH session does not inherit it).

**Not resumed.** It was one iteration short of its iteration-30 gate and its
second self-anchor, but a separate diagnosis — all four training scaffolds
(curriculum, draft prior, seed retention, HOF start) annealing out at exactly
10,000 games — calls for a fresh run with staggered knots regardless. That
diagnosis belongs with the training plan, not here.
