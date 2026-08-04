# Spending the cores: a throughput plan for the cloud box

**Status:** proposed, nothing built. Written 2026-08-04, immediately after the
first cloud sweep campaign on a rented RTX 5090 / EPYC 9654 slice.
**Trigger:** the box is 48 cores and generation uses about one of them, because
every scheduler axis available today is exhausted at **1.18×** end to end.

**The constraint that shapes the whole plan:** none of this can land in the run
that is starting now. `_refuse_changed_code` (W6.5) refuses a resume whose commit
differs from the one the run began on, deliberately, so a mid-run pull would stop
the run rather than speed it up. Everything here is for the *next* rental, built
and gated on the laptop while the 200k-game run proceeds.

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

One scheduler thread walks the games, packs leaves, and calls Python. The Python
call builds tensors, runs the forward, and gathers results. The Rust side
releases the GIL for tree work, but **packing and the forward both need the
interpreter**, so they cannot overlap each other. That single mechanism explains
all three nulls at once:

* **more slots** widen batches but do not add a second packer;
* **`inflight > 1`** cannot build batch N+1 while batch N holds the GIL;
* **more OMP threads** speed up something that is not the bottleneck.

And it explains why the 5090 is only ~1.6× the laptop 3070 at this width when
W0's arithmetic predicted 3-4×: the GPU is idle most of the cycle, so a faster
GPU buys proportionally little.

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
H2D, forward and gather. W3 records the Rust metrics in every iteration's stats
block, so **the run starting now will emit this for free**.

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
and reads far too low. Any Phase 0 measurement must run with diagnostic sync on,
or the forward will look free and the conclusion will be wrong in exactly the
direction that flatters this plan.

**Gate:** if the breakdown shows the forward dominating, or a large unaccounted
remainder, stop -- the model in §2 is wrong and Phases 1-2 are aimed at nothing.
If packing plus the Python call dominate, proceed.

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
4% noise band, with the determinism gates green. Target 2-3×; anything under
1.3× means the boundary cost is per-row rather than per-batch, and Phase 3 is the
answer instead.

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
half at the next rental. The laptop is sufficient for everything except the final
throughput number: the whole original throughput programme was developed there.

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

---

## 9. Rejected, recorded so they are not re-proposed

* **`OMP_NUM_THREADS` tuning** -- measured null on the box (4,032 vs 4,187, and
  the noise band is ~4%). Torch's pool is not the bottleneck.
* **`max_inflight_batches > 1`** -- measured null (1.03× at 2, 0.99× at 4), and
  §2 explains why: the GIL prevents the overlap it exists to create. Revisit only
  *after* Phase 1, when packing may no longer hold the interpreter.
* **More slots at a fixed cap** -- 144 slots underperformed 96 at cap 1024. Slots
  only pay alongside a cap that can carry them.
* **Multi-box sharding** -- out of scope in `CLOUD_TRAINING_PLAN.md` and still
  out of scope: one GPU is 18% busy.
* **Renting more cores** -- the current architecture cannot spend the 48 already
  rented. Until Phase 2 lands, prefer single-core clock and GPU over core count
  when choosing an instance.
