# Spending the cores: a throughput plan for 7 Wonders Duel generation

**Status (2026-08-04):** Phase 0 complete. The pipeline is fully profiled, the
original diagnosis is superseded, and **nothing has been built yet**. The work
below is re-ordered against measurements, not arithmetic.

**Headline:** generation runs on two threads. One of them is **idle 82.6% of the
time**; the other does three jobs strictly in sequence, and only one of those
three is the GPU. Every previous revision of this plan proposed adding
parallelism to the *idle* thread.

**Build order** (revised 2026-08-04 after external review):

1. ~~Add the missing worker sub-timers.~~ **DONE 2026-08-04.** Partition closes
   to 99.7%; the residual was `payload` (bytearray copies) at 3.7% of wall, and
   the return path is retired at 0.27% (§3.2).
2. **Rust→Python data movement, 32.2% of wall.** Three changes, benchmarked
   separately — do not assume the win is parallelism:
   **(a)** direct flat writer, skipping `encode()`'s intermediate `Vec<Token>`
   and per-token `Vec<f64>`;
   **(b)** row-parallel encoding over a persistent bounded pool, sweeping
   1/2/4/8/16 physical threads;
   **(c)** avoid the `PyByteArray` copy — reuse buffers across batches, or hand
   Python a view over Rust-owned memory that lives for the call.
   **Establish a ≥5-repetition baseline first** (see below).
3. Gate **byte-identical packed buffers**; A/B at laptop S/fp32 with **≥5
   repetitions**, matched seeds, interleaved.
4. **Re-profile.** Pursue the return path only if its newly isolated components
   justify it.
5. Instrument padding's **linear and quadratic** waste, then benchmark a concrete
   strategy (§3.3).
6. Recompute the overlap ceiling from the new profile. **Only then** revisit
   scheduler shards + request merging — which, having become the only candidate
   mechanism for creating surplus work, may return (§3.4, §4).
7. Next 5090 rental: **L/bf16 CUDA-event preflight and a 1/2/4/8/16 pack-thread
   sweep before starting training.**

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
independent `GameState`s. **No cross-row dependency exists.**

* **104.02 s, 28.7% of wall** at laptop S/fp32; **25.7% of generation wall** on
  the real cloud run at L/bf16. Both on the critical path.
* **This is the first item in this plan's history that uses more than one core.**
  Everything previously proposed added threads to the idle scheduler.
* **It grows as the GPU side is fixed.** Packing is CPU work, so bf16 does
  nothing for it while roughly halving the device term (§5).

### Result: allocator throughput was the cost. **1.2165× measured.**

Before restructuring anything, the allocation hypothesis was probed by swapping
the global allocator to mimalloc — three lines, no logic change, no
bit-exactness risk. **S/fp32, 5 repetitions each side, 200 games per rep:**

| metric | baseline | mimalloc | Δ |
|---|---:|---:|---:|
| **games/s** | **1.061** | **1.290** | **+21.6%** |
| pack | 53.14 s | 37.79 s | −28.9% |
| `sched_collect` (tree search) | 15.12 s | 9.23 s | **−39.0%** |
| wall | 188.58 s | 155.04 s | −17.8% |
| payload | 6.36 s | 6.99 s | +9.9% |

**Speedup 1.2165×, 95% CI [+19.2%, +24.1%]** — far outside the 1.63% CV.

**The work is identical**, which is the check that makes this trustworthy:
`global_batches` 5546.2, `global_batch_leaves_mean` 247.2, `simulations`
659,521.2 and `moves` 14,044.8 are unchanged to the decimal on both sides. Only
the rate moved. 49 tests pass, including the encoder bit-exactness suite.

**Two things this revealed beyond the hypothesis:**

* **Allocation pressure is not confined to the encoder.** `sched_collect` — tree
  search on the *scheduler* thread — fell 39%, more than packing did. The MCTS
  node and mask allocations were costing as much proportionally.
* **`device_forward` fell 9.3% on identical GPU work.** Not a measurement error
  and not less work: a GPU fed faster holds higher boost clocks. A consequence of
  the speedup, not a cause.
* **`payload` got 9.9% *worse*** — mimalloc is not better for the eleven large
  `PyByteArray` allocations per batch. Small in absolute terms (0.6 s).

**Open decision: keep the dependency or spend the restructure.** This crate has
deliberately carried only `pyo3` (see the Cargo.toml comment). mimalloc buys
1.22× for three lines; the `TokenBuf` restructure below would *remove* the
allocations rather than make them cheaper, and the two partly overlap. Not
mutually exclusive, but the restructure's marginal value is now unknown rather
than assumed.

### Result: `TokenBuf` adds 1.0351× on top. Mostly redundant, slightly additive.

`TokenBuf` retains the token vector and every token's feature buffer across
calls, so after the first row an encode allocates nothing. All 13 construction
sites now write into it; `encode()` survives as an allocating wrapper for cold
callers. **S/fp32, 5 repetitions, measured against the mimalloc build:**

| metric | mimalloc | + `TokenBuf` | Δ |
|---|---:|---:|---:|
| **games/s** | 1.290 | 1.336 | **+3.5%** |
| pack | 37.79 s | 34.20 s | −9.5% |
| wall | 155.04 s | 149.77 s | −3.4% |
| `sched_collect` | 9.23 s | 9.31 s | +0.8% |

**Speedup 1.0351×, 95% CI [+1.23%, +5.79%].** Real, but small. Work identical
again on both sides (batches 5546.2, simulations 659,521.2, moves 14,044.8), and
the encoder bit-exactness gate passes — 49 tests.

**Cumulative: 1.2592× (1.061 → 1.336 games/s).**

**The honest reading.** An earlier revision argued removing allocations "should
beat making them cheaper". Directionally true — it is additive and outside
noise — but mimalloc had already captured ~85% of the available win, and
`TokenBuf` cost ~80 lines across 13 sites for the remaining 3.5%. On its own
merits it was marginal.

**What justifies it is what comes next.** Packing is still **34.20 s / 22.8% of
wall**, and the remaining cost is now genuine encoder computation —
`unseen_pool`, `obtainable_cards`, `compute_symbols`, `minimum_payment` per row —
not allocator traffic. That is the case for parallelism, and per-row reusable
buffers are exactly the structure row-parallel encoding needs.

### Still open: parallelism

**Benchmark separately — do not assume the win is parallelism.**

**(a) Direct flat writer.** `encode()` (`encoder.rs:71`) is not a thin append: it
builds a dynamic `Vec<Token>`, pushing through eight builders, with a `Vec<f64>`
per token, and only then flattens and pads every token to the fixed float
feature width. A writer that emits straight into the flat output, skipping the
intermediate token and feature vectors, may recover much of the cost **with no
threading at all** — and it composes with (b).

**(b) Row-parallel encoding**, over a persistent bounded pool rather than a
per-batch spawn.

**Structure it as row-local buffers plus a prefix-sum copy**, not as rows writing
directly into pre-sized global slices. A previous revision asserted the latter
was safe; it is not obviously so, because token counts are only known *after*
encoding. Row-local then copy is simpler, keeps output order exact, and is the
version whose determinism argument is trivial.

**Sweep 1, 2, 4, 8 and 16 physical threads. Do not bake in 8×** — the win depends
on per-row cost against dispatch overhead at ~440 rows/batch, and is unmeasured.

**Gate: byte-identical packed buffers** against the serial build, mock evaluator,
single thread. Far stronger and cheaper than the search-level equivalence a
batch-merging change would need.

### 3.2 The worker residual — **measured 2026-08-04, and it retires a target**

Build-order step 1 is done: `attach_ns`, `payload_ns`, `validate_ns` and
`metrics_ns` were added, and the worker partition now closes to **99.7%**.

| region | s | % wall | % of wait |
|---|---:|---:|---:|
| pack | 105.92 | 28.5% | 34.5% |
| Python call (device 152.64, tensor build 24.12) | 185.67 | 49.9% | 60.4% |
| **payload — `PyByteArray` copies** | **13.88** | **3.7%** | 4.5% |
| extract | 0.79 | 0.2% | 0.3% |
| validate | 0.21 | 0.1% | 0.1% |
| attach (GIL) | 0.05 | 0.0% | — |
| metrics lock | 0.00 | 0.0% | — |
| unpartitioned (channel/reply) | 0.87 | 0.2% | 0.3% |

**`payload_ns` is 13.88 s of the 14.76 s residual — 94% of it.** The outbound
bytearray copying was the answer.

**The return path is retired as a target.** Extract plus validate is **1.0 s,
0.27% of wall**, and the laptop now agrees with the cloud log (`extract_ns`
25.95 s against `encode_pack_ns` 6,404 s). Flattening it would be unmeasurable.
An earlier revision proposed both relocating validation *and* flattening the
return; both are dead — the first because relocation is not overlap, the second
because there is nothing there to win.

**`payload` folds into §3.1 rather than standing alone.** `pack` builds flat
`Vec<u8>` buffers and `payload` copies those into Python-owned `PyByteArray`s:
the same problem, Rust→Python data movement, **119.8 s / 32.2% of wall combined.**
Scope the §3.1 work to the whole path, not just `encode()`.

<details><summary>Superseded: the reasoning before it was measured</summary>

### The 14.76 s worker residual — composition unknown, measure it first

An earlier revision called this "extract + validation" and sized a 4.1% prize
from it. **Both the label and the prize were wrong**, and external review caught
it.

**It is a subtraction, not a timer**: `299.13 wait − 104.02 pack − 180.36 call`.
The only thing actually timed in there is `out.extract()` (`extract_ns`).
Everything else in `evaluate_batch_prepared_routed` (`eval.rs:484`) falls outside
all three timers, so the residual is some mixture of:

* outbound `PyByteArray` allocation and copying during payload construction —
  eleven buffers per batch, built *before* `call_start`;
* PyO3 result extraction;
* Rust validation of every prior;
* the metrics mutex;
* channel / reply overhead.

**The real cloud run says extraction is not the term.** Over the preserved 30
iterations: `encode_pack_ns` **6,404.27 s**, `extract_ns` **25.95 s**,
`queue_wait_ns` **7.55 s**. Extraction is **0.4% of packing** and ~0.10% of
generation wall. Whatever the residual is, it is almost certainly the outbound
bytearray copies, not the return path.

**Action, not estimate:** add timers around payload construction, adapter call,
extraction, validation, metrics lock and reply, then re-read. No prize is claimed
here until that partition closes.

**Do not "move validation to the scheduler thread".** A previous revision
proposed this on the grounds that the scheduler is idle. At `inflight = 1` it
gains nothing: the scheduler waits, receives, scatters, and only then builds the
next batch, so relocating work from before the reply to after it leaves it on the
same serial path. **Relocation is not overlap.** Making the work cheaper (or
removing it) still helps; moving it does not.

</details>

### 3.3 Padding — **metric is wrong before the remedy is worth designing**

`padding_ratio` is **0.187-0.263** depending on configuration (§5), and it is
`1 − ΣL / (N × Lmax)` — a **token-linear** waste measure
(`f4_throughput_bench.py:553`). But the network (`net.py:355`) has both
token-linear cost *and* attention cost quadratic in sequence length, so the
linear ratio understates the compute actually wasted.

**Record both**, then design against the second:

* linear: `1 − ΣL / (N × Lmax)`
* quadratic: `1 − ΣL² / (N × Lmax²)`

**And benchmark the actual remedy, not the waste.** Length bucketing is the
obvious fix but it splits one forward into several, trading padding for launch
overhead and changing batch shapes — which perturbs float reductions and so needs
a strength argument, not an identity one. No prize is claimed here until a
concrete strategy is measured end to end.

### 3.4 Overlap the two threads

With `inflight = 1` the scheduler's ~63 s never overlaps the worker's 299 s.
Fixing that is worth **at most 1.21×** on its own (362 → 299). Note §6 records
`max_inflight_batches > 1` as a *measured null* — because with `leaf_batch = 1`
there is no surplus ready work to pipeline. Overlap only becomes available
alongside a change that creates surplus work.

### 3.5 No stacked forecast

A previous revision projected ~2.3× by stacking an assumed 8× pack speedup, the
whole residual vanishing, padding falling proportionally, and thread overlap.
**Removed.** It compounded four unmeasured mechanisms, and the fourth does not
exist: §3.4 shows nothing in items 3.1-3.3 *creates* surplus work, so overlap
cannot be assumed to arrive with them.

**Two grounding facts to size anything against, both measured:**

* **Packing is 25.7% of generation wall** across all 30 cloud iterations at
  L/bf16 (23.8% over the late 20). The laptop diagnosis transfers to production.
* **Generation is only 82.3% of the training loop.** Measured phase shares over
  the same 30 iterations: generation 82.3%, replay derivation 9.8%, gates 6.0%,
  training 1.9%. **A generation speedup dilutes**: an 8× pack alone is ~1.29× on
  generation and ~1.22× on the loop.

Quote loop-level numbers, not generation-level ones, when justifying effort.

---

## 4. What this retires

**More scheduler threads (`--rust-scheduler-workers`) is no longer the plan.**
Even making the scheduler infinitely fast caps at **1.21×**, and the extra games
it would activate still queue behind the same single worker. The original plan's
Phases 1 and 2 were both aimed here.

**The merging eval worker is demoted, not deleted — and may return.** Its stated
rationale was partly latency, and that is dead: `queue_wait_seconds` is **0.22 s**
on the laptop and **7.55 s across the entire cloud run**, so nothing waits to
hand work over. But §3.4 leaves overlap unreachable because nothing in items
3.1-3.3 creates surplus ready work, and **scheduler shards plus request merging
is the only identified mechanism that would.** So it is deferred behind the pack
work, not retired: revisit at build-order step 6, once the profile has changed.

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

**Trustworthy.** The scheduler timers (`sched_refill_ns`, `sched_collect_ns`,
`sched_retire_ns`, `sched_assemble_ns`, `sched_submit_ns`, `sched_wait_ns`)
partition the scheduler thread to 100.2%; the worker timers (`attach_ns`,
`payload_ns`, `validate_ns`, `metrics_ns`, with `encode_pack_ns`, `py_call_ns`,
`extract_ns`) partition the worker to 99.7%. Both are single-thread partitions —
never add across the two. `--cuda-events` gives true device time without
perturbing the pipeline.

**Instrumentation is not free, and single runs cannot price it.** Wall on the
same S/fp32 configuration went 360.7 → 362.2 → 372.2 s across the two
instrumentation rounds, ~3.2%. That is inside the old ~4% band but trending, and
one run cannot separate the two. **Establish a ≥5-repetition baseline on the
current build before any A/B**, per §10.

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

* **On the cloud geometry the GPU is not the dominant component**: ~22%
  (production, extrapolated) to 42% (box fp32). This is *not* true in general —
  laptop L runs 69-87% device (§5), which is exactly why laptop L could not
  answer the question. An earlier revision said "not the bottleneck in any
  configuration measured", contradicting its own table.
* **Cores now matter, because §3.1 finally has something to spend them on.**
  How far packing parallelises is unmeasured (§9).
* Single-core clock still matters for whatever remains serial on the worker.
* Core *count* beyond what packing can use buys nothing.

**Two claims withdrawn as unsupported:**

* ~~"the Python call is GIL-bound"~~ — only the worker attaches to Python and no
  GIL contention was ever demonstrated; much of that call is Torch C++/CUDA work
  that releases the GIL anyway.
* ~~"a 5090's bf16 advantage is likely larger"~~ — direction depends on kernel
  shapes, launch overhead and the relative fp32/bf16 paths. Unknown, not likely.

**Production's bf16 device time is extrapolated from a 3070-measured 2.88×
forward speedup, not measured on a 5090**, and the laptop's CUDA runtime cannot
reproduce the 5090's. Blackwell `sm_120` support begins with CUDA 12.8, so the
cloud image's cu128 choice is right — but **pin the exact Torch build** rather
than installing the latest cu128 wheel, or the next rental is not comparable to
this one.

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
* **Statistical acceptance — the single-run habit in §2 and §5 does not meet it.**
  Two 200-game observations do not establish a noise band, and every profile in
  this document used `--repetitions 1` while `f4_contract_v2.json` requires
  **`minimum_repetitions: 5`** at `minimum_measured_games_per_repetition: 100`.
  Any A/B claim needs **matched seeds, interleaved baseline/treatment, ≥5
  repetitions, and a confidence interval on the ratio** — not a point estimate
  against the old ~4% band.
* **State width and precision** in every claim.
* **Report CPU topology** alongside throughput: physical cores vs vCPUs,
  affinity, sustained frequency, NUMA distance to the GPU, context switches, and
  **per-core** utilisation. The normalised `cpu_utilization` field hides "one
  busy core among 48", which is the exact condition this plan exists to fix.
* **Cover both workload regimes on cloud acceptance.** Measured generation fell
  from 1.56 to ~1.11 games/s across the run as the mix changed, so accept against
  both early (curriculum/bot-mixed) and late (pure self-play) traffic.
* **Metrics attribution**: if a counter changes meaning, say so in its doc
  comment, or numbers stop being comparable across the change.
* **Preserve raw benchmark artifacts** (`rows.jsonl`, manifests), not just
  summaries. Several corrections in Appendix A were only possible because the raw
  rows survived.

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
