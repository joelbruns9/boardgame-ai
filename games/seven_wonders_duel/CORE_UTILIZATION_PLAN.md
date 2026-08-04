# Spending the cores: a throughput plan for the cloud box

**Status:** **Phase 0 done and gated (§11, §13); Phases 1-2 confirmed as the
build, ceiling 2.41×, realistic 2.01×.** Nothing else built. Written 2026-08-04
after the first cloud sweep campaign on a rented RTX 5090 / EPYC 9654 slice;
§10-§13 added the same day.
**Trigger:** the box is 48 cores and generation uses about one of them, because
every scheduler axis available today is exhausted at **1.18×** end to end.

**Build order: Phase 1's merging worker, then the Phase 2 sweep, then padding.**
3a is not the fallback -- it is capped at 6.9% of wall. Develop at **S** width on
the laptop (§7), verify at L on the next rental.

*An earlier revision put padding first, on the strength of its 26% of tokens
against a device share of 41.6%. Two later corrections both moved share away from
the device and onto the scheduler thread -- the unaccounted remainder (§11 caveat
1) and the fp32/bf16 confound (§11 caveat 4) -- so padding's ~10%-of-wall payoff
is an upper bound that shrinks under bf16, while Phase 1+2's grew from 1.56× to
2.01×. Padding is still worth doing and still carries no determinism risk; it is
simply no longer the best first move.*

**The constraint that shaped the original plan no longer binds.** It said none of
this could land in the run that was starting, because `_refuse_changed_code`
(W6.5) refuses a resume on a changed commit. That run died at iteration 30 (§12)
and is not being resumed -- the schedule-cliff diagnosis calls for a fresh run
with staggered knots anyway. So the throughput work is no longer racing a live
run; it gates the *next* launch.

---

## 1. What was measured

All figures from the rented box, 2026-08-03/04: RTX 5090 (32 GB), 48 cores of a
shared EPYC 9654, 64 GB, L = 384x8x6 at bf16, the W0 `sweep_L_lr5e-05_seed0`
checkpoint, 200 games per point, `--record-fast-moves` off, no curriculum bots.

### Generation, games/hour

| slots | cap 256 | cap 1024 | cap 2048 |
|---:|---:|---:|---:|
| 48 | 3,766 | 3,899 | -- |
| 96 | 2,913 | **4,187** | -- |
| 144 | 3,485 | 4,080 | 4,334 |
| 192 | -- | -- | **4,427** |

### The three nulls

These matter more than the wins, because each one removes an explanation:

| lever | result | reading |
|---|---|---|
| slots beyond 96 at cap 1024 | 4,187 → 4,080 (144) | past the knee; concurrency is not the limit |
| `max_inflight_batches` 1 / 2 / 4 | 4,032 / 4,138 / 3,980 | pipelining depth buys nothing |
| `OMP_NUM_THREADS=12` | 4,032 vs 4,187 unset | thread oversubscription costs nothing |

The repeated 96/1024/1 point (4,187 then 4,032) puts the **noise band at roughly
4%** on single 200-game runs, which is the bar any future claim here must clear.

### The gate path, for contrast

0.833 games/s at 96 slots / 1024 cap, **setup 0 s** -- the 193 s fixed cost the
plan was built around is gone, and gate cost is now linear in games. The gate is
not the problem.

### Derived, not measured

At 4,187 games/h, ~70 moves/game and ~39 average sims: **~3,200 NN rows/s**. At
observed batch means near 150 that is **~20 batches/s, i.e. ~50 ms per batch
cycle**, against a 155-row forward through a 14.9 M model on a 5090 that should
cost single-digit milliseconds. GPU utilisation sat at ~18% throughout.

**So roughly 90% of each batch cycle is boundary cost, not compute.** This is the
central claim of the plan and it is an *inference from arithmetic*, not a
profile. Phase 0 exists to confirm or kill it before anything is built.

---

## 2. The read

**Not the GIL.** An earlier draft of this plan blamed GIL contention between
packing and the forward. That is wrong: `lib.rs:2313` runs the whole scheduler
under `py.detach`, and `eval.rs:601` states the discipline -- Rust scheduling
runs detached, only the worker thread attaches to Python. Packing and the forward
already overlap. Recorded because it is a plausible-sounding story that survives
until someone greps for `detach`.

What the evidence actually supports:

**Why `inflight > 1` is null.** With `leaf_batch = 1`, every active game
contributes at most one leaf per cycle, and slots (96-192) sit far below the cap
(1024-2048). So a single batch already contains *every* leaf that is waiting, and
no game can produce another until that batch's results come back. Batch N+1 has
nothing to pack. Pipelining depth only pays when ready work exceeds one batch,
which is precisely the regime the cap was widened out of.

**Why more slots stop paying.** If a fixed per-batch cost dominated, doubling the
rows per batch would nearly double throughput. Measured, 48 → 96 slots at cap
1024 bought **7%** (3,899 → 4,187), and 144 was worse. So at these widths the
cycle is dominated by **per-row** cost, not per-batch overhead.

**Where per-row cost lives, and why it is serial.** Every row is one MCTS leaf:
a PUCT descent, a make/unmake, a legality mask over 1,202 actions, an expansion
and a backprop -- plus `encode_pack` -- all on **one scheduler thread**. Slots
add games to that thread's queue; they do not add threads. That is the serial
resource, and it is the one thing none of the three levers touched.

This also explains the 5090 being only ~1.6× the laptop 3070 when W0's
arithmetic predicted 3-4×: if the cycle is CPU-side per-row work at ~18% GPU
utilisation, a faster GPU buys proportionally little.

**The correction this implies for §1.** "~90% of the cycle is boundary cost" was
derived by comparing the cycle time against an *idealised forward*, which
silently attributes everything non-GPU to the boundary. The slot-scaling evidence
says much of it is per-row Rust work instead. Both readings agree the GPU is not
the limit; they disagree about what is, and they imply different fixes -- §3
decides between them with a profile instead of arithmetic.

`--rust-scheduler-workers` would add packers -- `run_many_pipelined_sharded`
already spawns one scheduler thread per worker -- but it is pinned to 1 because
of what sits behind it. `spawn_py_flat_worker` (`eval.rs:878`) is a single thread
running `while let Ok(request) = request_rx.recv()`: one request, one forward,
one reply, **never merging**. Two scheduler threads therefore produce two
half-width batches evaluated back to back, trading the only thing currently
helping (batch width) for parallelism the GPU does not need.

**That is the thing to fix.**

---

## 3. Phase 0 -- confirm the read before building anything *(S)*

The instrumentation already exists and nobody has read it at this scale.
`BoundaryMetrics` (`eval.rs:324`) counts `encode_pack_ns`, `queue_wait_ns`,
`py_call_ns` and `extract_ns`; the Python adapter separately times tensor build,
H2D, forward and gather.

**Correction (2026-08-04, on running it).** Three claims in this section as first
written were wrong, and §10 records what the run actually said.

* **"W3 records the Rust metrics in every iteration's stats block, so the run
  starting now will emit this for free" is false.** `phase_d.py` consumes none of
  them: it has no reference to `total_metrics`, `last_metrics`, `forward_seconds`
  or any `device_*` field, and `encode_pack_ns` appears nowhere in it. The
  counters exist in `lib.rs:1436-1449` and are read by `f4_throughput_bench.py`
  alone. The 200k-game run produces no boundary telemetry; waiting for it would
  have yielded nothing.
* **Phase 0 is not a thing to build.** `f4_throughput_bench.py:576` already
  carries a `# --- Phase 0: host/device separation ---` block emitting every term
  below plus steady-state windowing, and `test_f4_phase0_telemetry.py` gates it.
  Phase 0 is a run and a read.
* **The term list below omitted `rust_tree_ns`**, which is the counter measuring
  the PUCT descent, make/unmake, masks and expansion -- i.e. the exact quantity
  §2 nominates and row 1 of the gate table acts on. As first written this phase
  would have measured everything except its own hypothesis. The bench already
  sums it as `rust_tree_seconds` (`:498`), with `rust_chance_seconds`,
  `rust_record_seconds` and `scatter_seconds` beside it.

Deliverable: a breakdown of the ~50 ms cycle that **adds up to the wall clock**,
per phase, at the shipped geometry. Specifically, how much is:

1. `encode_pack` in Rust,
2. queue wait,
3. the Python call itself (tensor build + H2D),
4. the forward,
5. the gather,
6. unaccounted.

**The trap that would invalidate this:** `rust_bridge._sync()` is a no-op unless
`--diagnostic-sync` is set, so `forward_seconds` times an *asynchronous enqueue*
and reads far too low.

**Correction: the prescribed fix was the wrong flag.** This said any Phase 0
measurement "must run with diagnostic sync on". It must not: `--diagnostic-sync`
perturbs the pipeline it is measuring. `--cuda-events` (`:1111-1117`) brackets
H2D / forward / gather / D2H with CUDA events queried *after* the run, giving
true device time without synchronising. The bench already says so at `:562-564`,
where `gpu_busy_fraction` is annotated "NOT evidence of a GPU-bound pipeline --
`device_forward_fraction` and `nvml_utilization_mean` are."

**The trap is real but lands somewhere else.** Measured (§10): the async forward
wait is absorbed by the *gather*, because the gather is the first host operation
that touches the result. Host `gather_d2h_seconds` read 703 s of an 870 s wall --
81%, and a completely convincing story about the gather path being the
bottleneck. Device gather was 10.0 s. **Anyone reading host timers alone would
have concluded the gather was the problem and built the wrong thing.** Read
`device_*`, not the host terms.

**The gate, and what each outcome implies.** The two candidate bottlenecks want
different fixes, so this measurement chooses the phase that follows:

| dominant term | fix | phase |
|---|---|---|
| scheduler-side per-row work (descent, make/unmake, masks, `encode_pack`) | more scheduler threads | 1 + 2 |
| `py_call_ns` (tensor build + H2D) | build tensors in Rust; more threads will not help | 3a |
| the forward itself | nothing here helps; the model is simply expensive | stop |
| a large unaccounted remainder | the model in §2 is wrong; re-derive before building | stop |

Phases 1-2 and 3a are **not** interchangeable, and the ordering below assumes the
first row. If Phase 0 says otherwise, reorder rather than proceeding by habit.

**Correction: the table assumes a dominant term, and there isn't one.** Measured
on the box (§11): device 41.6%, Rust scheduler thread 35.7%, Python tensor build
6.9%, unaccounted 14.5%. No row wins. The question the table should have asked is
not "which term is biggest" but **"which terms fail to overlap"** -- and the
answer is that the single scheduler thread alternates between Rust work and
blocking inside the Python call, strictly, never both. Read the gate as:

| observation | fix | phase |
|---|---|---|
| `queue_wait` ≈ 0 **and** `pyo3_call` + Rust work ≈ wall | the thread serialises what could overlap: more scheduler threads, plus a merge to keep batches wide | 1 + 2 |
| `pyo3_tensor` a large share of `pyo3_call` | build tensors in Rust | 3a |
| device total ≈ wall | the model is simply expensive | stop |
| unaccounted ≫ 15% | re-derive before building | stop |

---

## 4. Phase 1 -- make the eval worker merge requests *(M)*

The change, in `spawn_py_flat_worker`:

1. `recv()` one request, as today.
2. `try_recv()` in a loop, draining whatever else is already queued, while the
   accumulated row count stays under `max_rows`.
3. Concatenate the drained requests' `states`, `actors`, `legals` and `net_ids`.
4. One `evaluate_batch_prepared_routed` call.
5. Split the returned rows back by each request's length and reply to each
   request's own channel.

Roughly 150 lines with the error path done properly.

### Traps, named in advance

* **`net_ids` must survive the merge, per row.** This is exactly where W1's
  league routing broke: routing worked in `run_many` and silently did nothing in
  `run_many_pipelined` because `WorkerRequest` dropped the field at the worker
  boundary. A merge that concatenates states but rebuilds `net_ids` from the
  wrong request produces a player that is neither checkpoint, and every game
  still completes and validates. **Test league routing through the merged path
  specifically**, not just through the direct scheduler.
* **Error fan-out.** Today a failure replies to the one waiting request and
  breaks the loop. Merged, *every* drained request must receive the error, or
  its scheduler thread blocks until the timeout with no explanation. W2's
  postmortem is the precedent: the run died reporting a symptom, and the real
  error was two layers down.
* **`max_rows` is a hard bound, not a target.** `validate_leaf_batches_fit`
  already refuses jobs whose leaf batch exceeds the cap; the drain must respect
  the same ceiling or it will build a batch the adapter rejects.
* **Metrics attribution.** `queue_wait_ns` and `py_call_ns` are per request
  today. After merging, one Python call serves many requests; decide explicitly
  whether `batches` counts merged calls or source requests, and say so in the
  field's doc comment, or the throughput numbers become incomparable across the
  change.

### Determinism gate

Use the shape the throughput programme settled on, which exists because a
previous version of this reasoning was wrong:

1. **Mock evaluator, `scheduler_workers = 1`:** records byte-identical to the
   pre-change build. Merging must be a no-op when there is nothing to merge.
2. **Mock evaluator, `scheduler_workers = 4`:** the *set* of games is identical
   and every game is individually legal, but do **not** assert byte-identity of
   targets -- batch composition changes float reductions.
3. **Real net:** fingerprint the discrete trajectory (actions, digests, sims)
   and report divergence as a measurement. `self_play.rs` documents that batch
   shape can change search choices through float ties; the sweep campaign
   already observed 6 distinct trajectory sets across 6 geometries.

**Never claim "byte-identical records" on the real net.** That claim has failed
review here before.

---

## 5. Phase 2 -- sweep `scheduler_workers` and re-fit the geometry *(S)*

`--rust-scheduler-workers` is already plumbed through Phase D, so once Phase 1
lands this is a sweep, not a build. Axes: workers ∈ {1, 2, 4, 8}, slots ∈ {96,
192, 384}, cap ∈ {1024, 2048, 4096}.

Two things to watch:

* **Slots are a global budget, not a per-worker one.** `SlotBudget::new`
  (`self_play.rs:764`) holds one shared atomic pool of `max_active_slots`,
  reserving one slot per shard and letting the rest compete; it refuses outright
  when `max_active_slots < scheduler_workers`. So raising workers does not raise
  total concurrency -- it divides the same concurrency among more packers, and
  the average width per scheduler falls as ~slots/workers. Sweep slots *and*
  workers jointly, for the same reason cap and slots had to be.
* **Merged batch width is the metric that matters**, not games/s alone. Record
  mean rows per Python call. If it falls as workers rise, the merge is not
  working and games/s is winning for some other reason.

**Acceptance:** a measured configuration beating 4,427 games/h by more than the
4% noise band, with the determinism gates green. ~~Target 2-3×; anything under
1.3× means the boundary cost is per-row rather than per-batch, and Phase 3 is the
answer instead.~~

**Corrected by §11.** The Amdahl arithmetic is now measured, not hoped for:
removing *all* Rust scheduler-thread work gives **1.56×**, and perfect overlap of
Rust work with device work gives a hard ceiling of **2.41×**. So target
**1.5-1.8×**, treat anything above 2× as suspicious enough to re-check the
determinism gates, and do not read a 1.6× result as failure -- it is the
arithmetic maximum for this phase. 3a is *not* the fallback: `pyo3_tensor` is
6.9% of wall, so moving the tensor build to Rust cannot pay more than that.

---

## 6. Phase 3 -- if Phase 2 disappoints *(L, contingent)*

Two candidates, in order of preference.

**3a. Move the tensor build into Rust.** If the cost is per-row rather than
per-batch, more packers will not help; the fix is to hand the adapter tensors it
does not have to construct. The features are already packed into bytes in Rust;
the Python side unpacks them into torch tensors every call. Building them in
Rust and passing them as buffers removes the largest per-row Python cost. This is
a bigger change to the boundary contract and needs its own equivalence gate
against the current encoder.

**3b. Multi-process generation on one GPU.** N processes, each with its own
scheduler thread and CUDA context, each generating a slice of the iteration's
games, merged into one buffer. `phase_e/launch_shards.sh` is the precedent (12
shards took the trap suite from 9.6 h to 1.7 h) and `run_jobs_in_processes`
already ships model state to workers for the Python backend.

Preferred *last*, because it multiplies what Phases 1-2 keep singular:

* **the routing indices.** Curriculum bots and league opponents are derived from
  `job.index` (`(job.index // 2) % len(CURRICULUM_BOT_TYPES)`), so a shard that
  renumbers its jobs locally silently re-routes opponents. Pass global indices.
* **the buffer.** Shards must merge in job order, atomically, or iteration
  buffers stop being reproducible and the games ledger miscounts.
* **the journal.** The pending-iteration recovery assumes a single writer.
* **memory.** A CUDA context plus a model copy per process, against a 64 GB
  slice.

---

## 7. Sequencing

```
[run 200k games on the current commit]        <- do not disturb
   |
   +-- Phase 0: read the run's own boundary telemetry     (free, during)
         |
         +-- Phase 1: merging eval worker      (laptop, gated)
               |
               +-- Phase 2: sweep workers x slots x cap   (laptop, then next box)
                     |
                     +-- Phase 3a/3b only if Phase 2 misses its bar
```

Phase 0 during the run, Phases 1-2 on the laptop while it runs, Phase 2's cloud
half at the next rental.

**Correction: "the laptop is sufficient for everything except the final
throughput number" is false for Phase 0 specifically.** The laptop's binding
constraint is its own GPU (§10: 84% device-forward, NVML 84%), which the box does
not share -- the box sweep measured ~18%. The laptop cannot answer a gate
question about what dominates when the GPU is *not* the limit, because on the
laptop it is. Phase 0's gate needs box data.

**But model width is the knob that fixes this, and it is why laptop development
of Phases 1-2 is still sound.** The CPU-side work -- `encode_pack`, PUCT descent,
make/unmake, the 1,202-action masks -- is **independent of `d_model`**, while the
forward is not: `w0_sizing.py:55-57` measures bf16 inference at S 32,182 rows/s
against L 4,356, a **7.4×** spread across a **14.5×** parameter range
(S 1.03 M, L 14.9 M). So running L on a 3070 buries the CPU terms under an
expensive forward, and dropping to **M** collapses the device share and
reproduces the box's regime locally.

~~**Develop Phases 1-2 at M width on the laptop; verify at L on the next
rental.**~~ Throughput claims must state their width, because the same code is
CPU-bound at S and GPU-bound at L on identical hardware.

**Corrected 2026-08-04 by measuring it (§13): M is not enough -- use S.** The M
run came back at **66.9% device** against the box's 41.6%, so it does not
reproduce the regime. The error was using the S-vs-L ratio (7.4×) to justify M:
M-vs-L is only **2.26×**, and the 3070 is 4.4× slower than the 5090, so M does
not close the gap. Device-to-Rust ratios, measured:

| run | device / Rust |
|---|---:|
| box L | **1.16** |
| laptop M | 3.47 |
| laptop L | 9.2 |

S is a further 3.27× cheaper than M, putting laptop S at ≈**1.06** -- the box's
regime. **Develop Phases 1-2 at S width; verify at L on the next rental.**

This also explains why the cloud run *felt* slower than the laptop pilots despite
better hardware: the pilots ran S and the cloud run ran L. Measured at identical
model and lock, the 5090 is **4.4×** the 3070 (0.230 → 1.010 games/s, §10 vs
§11). But L costs 7.4× more per NN row than S, and 7.4 > 4.4, so games/s fell.
That is the capacity-for-speed trade working as intended, not a regression.

---

## 8. What this is worth

At 4,427 games/h a 200k-game run is ~50 h of generation. At 2× it is 25 h, which
is a day of rental per run and, more importantly, doubles the games a fixed
budget buys -- the science gap (13.0% against ZeusAI's 21.4%) is plausibly a
volume problem, and volume is what this buys.

Against that: Phase 1 is a day of Rust with a determinism gate, and the payoff
rests on a claim (§1, "90% boundary cost") that is currently arithmetic rather
than a profile. **Phase 0 is cheap and decides whether the rest is worth
starting.**

**Updated by §11, which ran it.** The "90% boundary cost" claim is dead: device
work is 41.6% of wall. The realistic payoff is **1.5-1.8×**, not 2×, so a 200k
run goes from ~50 h of generation to ~30 h rather than 25 h. Phase 1+2 remains
worth doing on that arithmetic, but it is a day of Rust for ~20 h of rental per
200k-game run, not a doubling. **The padding lead (26% of tokens, ~10% of wall,
no determinism exposure) has a better effort-to-payoff ratio than Phase 1 and
should land first.**

---

## 9. Rejected, recorded so they are not re-proposed

* **`OMP_NUM_THREADS` tuning** -- measured null on the box (4,032 vs 4,187, and
  the noise band is ~4%). Torch's pool is not the bottleneck.
* **`max_inflight_batches > 1`** -- measured null (1.03× at 2, 0.99× at 4). §2
  explains why: at `leaf_batch = 1` with slots well under the cap, one batch
  drains every waiting leaf and there is nothing to pack for the next one.
  Revisit only if `leaf_batch` rises or slots approach the cap, which are the
  conditions that create surplus ready work.
* **More slots at a fixed cap** -- 144 slots underperformed 96 at cap 1024. Slots
  only pay alongside a cap that can carry them.
* **Multi-box sharding** -- out of scope in `CLOUD_TRAINING_PLAN.md` and still
  out of scope: one GPU is 18% busy.
* **Renting more cores** -- the current architecture cannot spend the 48 already
  rented. ~~Until Phase 2 lands, prefer single-core clock and GPU over core count
  when choosing an instance.~~

  **Corrected by §11.** "Prefer single-core clock" was written before the profile
  and does not survive it. The Rust scheduler work is 141.6 s of a 396.1 s wall
  on one thread, so a **1.5× faster core** (a 9950X-class part against this
  EPYC 9654) buys `396.1 → 348.9`, i.e. **1.14×** -- and it buys *nothing* once
  Phase 1+2 lands, because 2+ scheduler threads put Rust work at ~70 s, already
  below the 164.7 s device floor. Paying a premium for clock buys a benefit the
  software work is about to make redundant.

  **Instance selection, revised: GPU first, and little else matters.** After
  Phase 1+2 the device time *is* the wall (164.7 s / 400 games → ~8,700 games/h;
  ~11,800 with the padding fixed), so every further gain is GPU. Cores: enough
  for 2-4 scheduler threads, so 8-16. Clock: do not pay for it.

  **The one thing that would overturn this** is the 14.5% unaccounted (§11). If
  it is serial CPU work, Rust-side totals ~199 s -- *above* the device floor --
  and clock keeps mattering after Phase 1+2. Resolve it before buying hardware on
  the strength of this bullet.

---

## 10. Phase 0, run on the laptop -- 2026-08-04

RTX 3070, `sweep_L_lr5e-05_seed0` (the box's checkpoint), `leaf_batch = 1`,
96 slots / cap 1024 / inflight 1, `--games-per-call 200` so the window is steady
state rather than one drain per call, `--cuda-events --resource-sample-hz 20`,
**no** `--diagnostic-sync`. 200 games, **1 repetition** --
`sample_minimums_met: false`, so this is one unreplicated point.

Wall 870.5 s, 0.230 games/s, 8,967 batches, mean 152.4 rows/batch (the box's
~150, so the geometry is comparable).

| term | s | % wall |
|---|---:|---:|
| **`device_forward_seconds`** | **732.6** | **84.2%** |
| `rust_encode_pack_seconds` | 59.6 | 6.8% |
| `rust_tree_seconds` | 16.0 | 1.8% |
| `pyo3_tensor_seconds` | 13.9 | 1.6% |
| `device_h2d_seconds` | 7.6 | 0.9% |
| `device_gather_seconds` | 10.0 | 1.1% |
| `device_d2h_seconds` | 1.3 | 0.2% |
| `scatter_seconds` | 4.4 | 0.5% |
| `queue_wait_seconds` | 0.17 | 0.0% |
| unaccounted | 23.2 | 2.7% |

It **adds up to the wall clock**, which was §3's deliverable. `nvml_utilization_
mean` = 84.4% independently corroborates `device_forward_fraction` = 0.84.

**Verdict on this hardware: gate row 3, "the forward itself -- stop."** Not
scheduler-side per-row work (`rust_tree` 1.8%), not tensor build + H2D (2.5%),
not a large remainder (2.7%). §2's per-row hypothesis is not supported here.

**But this does not transfer, and the reason matters.** 0.230 games/s = 827
games/h against the box's 4,187 -- a **5.1×** ratio, matching the raw 3070→5090
bf16 gap, and flatly contradicting §2's "the 5090 being only ~1.6× the laptop
3070", which was the evidence for the cycle being CPU-side. On the laptop the
GPU is simply the wall. On the box at ~18% utilisation it is not. **Phase 0's
gate is still open and needs the same command on the next rental.**

That §2 figure should be treated as suspect until someone re-measures it at a
stated geometry; the inference built on it (per-row CPU cost) currently has one
measurement against it and none for it.

**Free lead, independent of all the above:** `padding_ratio` = 0.249. A quarter
of every GPU token batch is padding. In any GPU-bound regime that is ~25%
throughput for no algorithmic risk and no determinism exposure -- cheaper than
Phase 1 and not mentioned anywhere above. `boundary_tokens` vs
`boundary_padded_tokens` already measure it per batch.

---

## 11. Phase 0, run on the box -- 2026-08-04. **This is the gate result.**

Same rented RTX 5090 / EPYC 9654 slice, taken during the gap left by the training
run's crash (§12). `current_best.pt` (iteration 15, L 384x8x6 bf16),
`--exploratory-leaf1`, the shipped generation geometry: 192 slots / cap 2048 /
inflight 1, `--games-per-call 400`, `--cuda-events --resource-sample-hz 20`, no
`--diagnostic-sync`. 400 games, **1 repetition**.

Wall **396.1 s**, 400 games, **1.010 games/s = 3,635 games/h**.

| group | s | % wall |
|---|---:|---:|
| device total (forward 151.4, h2d 7.3, gather 5.4, d2h 0.6) | 164.7 | 41.6% |
| Rust scheduler thread (`encode_pack` 81.8, `tree` 43.7, `scatter` 14.7, chance/record 1.0) | 141.6 | 35.7% |
| Python tensor build (`pyo3_tensor`) | 27.3 | 6.9% |
| unaccounted | 57.4 | 14.5% |

`device_forward_fraction` 0.382 against `nvml_utilization_mean` 0.371 -- two
independent instruments, same answer. Host `gather_d2h_seconds` read 128.9 s
against a true `device_gather_seconds` of 5.4 s, so the §3 decoy reproduced
exactly as the laptop run predicted it would.

### The finding

**No term dominates. The serialisation does.** `pyo3_call_seconds` = 197.2 s and
Rust scheduler work = 141.6 s, together 85.5% of wall, with
**`queue_wait_seconds` = 0.22 s -- effectively zero.** The one scheduler thread
alternates strictly between doing Rust work and sitting blocked inside the Python
call, and never overlaps them. While it is blocked, no packing or tree work
happens; while it packs, the GPU is idle. §2 reached the right conclusion (the
single scheduler thread is the serial resource) from slot-scaling inference; this
is the same claim in measured seconds, and it names `encode_pack` (81.8 s) rather
than tree work (43.7 s) as the largest thing that thread does.

### What it decides

* **Phases 1+2 confirmed.** Removing all Rust scheduler work: 396.1 → 254.6 s,
  **1.56×**. Perfect Rust/device overlap: `max(164.7, 141.6)` = 164.7 s,
  **2.41× ceiling**. Both clear the bench's own `decision_rule`
  (`minimum_removable_wall_fraction` 0.15, `minimum_amdahl_speedup_upper_bound`
  1.1) with room.
* **3a is dead as a primary.** `pyo3_tensor` is 6.9% of wall.
* **Phase 1's stated rationale needs amending.** §4 justifies the merging worker
  partly as a latency fix; `queue_wait` ≈ 0 says the scheduler never waits on the
  worker. The merge is still required, but only for §5's reason -- to stop N
  scheduler threads producing N narrow batches. Current batches are 295 mean /
  335 p50 against a 2048 cap, so the headroom to fill is real.
* **Padding is now a first-class lead, not a footnote.** `padding_ratio` = 0.261
  sits on top of a term that is 38% of wall, so removing it is worth ~10% of wall
  by itself.
* **Slots are not the problem.** `steady_rows_per_batch` 430 vs
  `drain_rows_per_batch` 100, `steady_drain_seconds_ratio` 4.0 -- 192 slots fill
  batches well in steady state.

### Caveats, not to be dropped when this is cited

1. ~~**14.5% unaccounted**, five times the laptop's 2.7%.~~ **Resolved by §13:
   it is not box-specific and it is not noise.** Normalised against *non-device*
   time it is a consistent **18-25% on both machines at both widths** (laptop L
   19.5%, laptop M 17.9%, box L 24.8%). It tracks the Rust side, not the GPU, so
   it is scheduler-thread work the existing timers do not cover. **Folding it in
   raises the Phase 1+2 estimate from 1.56× to 2.01×** (Rust-side total 199.0 s =
   50.2% of wall; 396.1 → 197.1 s). The 2.41× ceiling is unchanged, being set by
   device time. *Measured:* the three-point ratio. *Inferred:* that it is
   scheduler-thread work -- timers would settle it, and Phase 1 should add them.
2. **One unreplicated point.** The 4% noise band from §1 was established on
   200-game runs; this is a single 400-game run.
3. **Geometry is near but not identical to the run's.** The lock forced 128 sims
   where production uses 39; effective sims/move landed at 46.6 vs 39.1 and
   sims/s at 3,315 vs ~3,044, so NN load is close.

4. **OPEN, and it moves the numbers: every Phase 0 figure here is fp32, while
   production is bf16.** `f4_throughput_bench.py:783` hard-codes
   `"inference_precision": "float32"` into the `--exploratory-leaf1` lock with no
   CLI override, and both sweep checkpoints carry `precision: fp32`; the cloud
   run's `current_best.pt` is `bf16`. This is the better explanation for
   `device_forward_fraction` 0.38 here against the ~23% observed on the live
   dashboard -- bf16 uses tensor cores and fp32 largely does not.

   **Direction of the error is known, magnitude is not.** A cheaper forward moves
   device share down and scheduler share up -- the same direction as caveat 1 --
   so both corrections make the scheduler thread *more* dominant than the table
   above says. **What it does not change: Phases 1+2 remain the build under
   either precision.** What it could change is §9's hardware advice (if the Rust
   side exceeds the device floor after parallelising, clock and cores start
   mattering again) and whether padding still deserves to go first, since padding
   only pays on the device share. Measuring it needs a patch letting the lock
   record a precision instead of asserting one.

---

## 12. Why the box was free: the run died of SIGHUP

Recorded because it cost an hour of rental and will recur otherwise.

The 200k-game run started 05:03 UTC and stopped at **13:45 UTC during iteration
30**, 8h42m in, having completed iterations 0-29 (~30,000 games). `run.log` ends
immediately after iteration 30's league-play banner: no traceback, no `Killed`,
no error. GPU idle at 0% / 2 MiB. `dmesg` was unreadable in the container (no
`CAP_SYSLOG`), but the clean cut plus the absence of any exception is SIGHUP.

**Cause: it was launched from the vast.ai Jupyter terminal, which has no tmux.**
SSH sessions on this image get auto-tmux and would have survived; the Jupyter
terminal dies with its cloudflared tunnel. The tunnel dying, the SSH reverse
port-forward being refused, and the run stopping were one event, not three.

**Rules for the next rental:**

* **Launch long runs over SSH, never the Jupyter terminal.** Confirm the tmux
  banner before starting anything. Do not `touch ~/.no_auto_tmux`.
* `trycloudflare.com` hostnames are ephemeral quick-tunnel names -- a restarted
  `cloudflared` mints a new random name, so a bookmarked URL is permanently dead
  and its failure says nothing about the run.
* **Liveness check that works from any container and any PID namespace:**
  `date -u; stat -c '%y %s' <run-dir>/run.log`. Iterations are ~17 min, so an
  mtime older than ~20 min means it has stopped. `pgrep`/`top` mislead here --
  the Jupyter terminal cannot see the run's processes even when it is healthy.
* `phase_d.py` has **no shebang and absolute imports**, so `./phase_d.py` fails
  with `Permission denied` and `chmod +x` would not help either. Use
  `python -m games.seven_wonders_duel.phase_d` from the repo root, with
  `/venv/main` activated -- an SSH session does not inherit it.

**Not resumed.** The run was ~1 iteration short of both its iteration-30 gate and
its second self-anchor (~31k games), but both are head-to-head matches between
checkpoints already on disk and can be replayed offline at any time. With the
schedule-cliff diagnosis (all four scaffolds annealing out at 10,000 games)
calling for staggered knots, and `_refuse_changed_code` blocking a resume across
the throughput work anyway, the next run is a fresh one rather than a resume.

---

## 13. Phase 0 at M width, laptop -- 2026-08-04

Run to test two things §11 left open: whether the unaccounted remainder was
box-specific, and whether dropping model width reproduces the box's regime on the
laptop. Identical to §11 in every parameter except the checkpoint
(`sweep_M_lr5e-05_seed0`, 256x6x4, 5.21 M params): 192 slots / cap 2048 /
inflight 1, `--games-per-call 400`, `--cuda-events`, 400 games, 1 repetition.

Wall **765.8 s**, 400 games, **0.522 games/s**, `device_forward_fraction` 0.669
against `nvml_utilization_mean` 0.680.

| term | s | % wall |
|---|---:|---:|
| device total (forward 512.3, h2d 13.2, gather 10.3, d2h 1.3) | 537.2 | 70.2% |
| Rust scheduler thread (`encode_pack` 113.8, `tree` 32.2, `scatter` 8.3, rest 0.3) | 154.7 | 20.2% |
| Python tensor build | 24.3 | 3.2% |
| unaccounted | 40.9 | 5.3% |

### What it settled

**The unaccounted is structural, not a box artifact.** See §11 caveat 1: 18-25%
of non-device time across both machines and both widths. Phase 1+2's estimate
rises to **2.01×**.

**M does not reproduce the box's regime; S should.** See the correction in §7.

**Two things transfer exactly, which is what licenses laptop development at all:**

* `padding_ratio` **0.2606** here vs **0.2608** on the box. Model-independent, so
  it is a property of the encoder and can be fixed at any width on any machine.
* Batch geometry at matched slots: `steady_rows_per_batch` **435** vs the box's
  **430**, mean rows 289 vs 295. The batching regime transfers even though the
  device balance does not.

### Method note

`encode_pack` scaled with width (59.6 s at L/200 games → 113.8 s at M/400 games,
i.e. 0.298 → 0.285 s/game) while `rust_tree` did not move much per game. Both are
nominally width-independent CPU work, so **treat per-game normalisation as the
comparison unit here, not per-run totals** -- §10 ran 200 games and §11/§13 ran
400, and comparing raw seconds across them is a trap.
