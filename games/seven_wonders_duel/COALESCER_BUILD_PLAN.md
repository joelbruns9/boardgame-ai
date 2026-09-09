# Evaluator coalescing, and the box-side work that follows it

**Status: rev 3 (2026-09-09). PHASE 0 CODE COMPLETE; the box-side baseline
(§0.3) is not taken. The coalescer itself is NOT started.**

Rev 1 was reviewed against the working tree and **seven of its claims did not
survive**. The most important: it concluded the endgame solver cost no
generation wall, from a counter that does not aggregate. That conclusion is
withdrawn, and a **Phase 0 instrumentation repair now blocks everything else** —
including renting a box.

Companion to `THROUGHPUT_ACTION_PLAN.md`, which took generation 1,661 → 3,332
games/hour and closed at Phase 4. Classification in `THROUGHPUT_LEVERS.md`
terms: the coalescer is **class B**; the geometry that follows is class A; leaf
batching is class C and priced in strength.

---

## 0. Phase 0 — repair the instrumentation, then re-baseline

**Nothing below is measurable until this lands.** Three counters are wrong or
missing, and two of them decide the questions this plan exists to answer.

### 0.1 `sched_solve_wait_ns` does not aggregate — **FIXED**

`SchedulerMetrics::merge` (`self_play.rs`) sums every `sched_*` field except
this one. `sched_solve_wait_ns` is declared on the struct and never merged, so a
multi-shard run reports **zero regardless of how long generation blocked on
solves**. cloud2 ran four shards.

`THROUGHPUT_LEVERS.md` §3.3 documents this exact defect as having already
happened once in this project. It is still here.

Two further problems with the same counter, both of which must be fixed before
its value means anything:

* **It times more than blocking.** The span covers harvesting and collecting
  solver results, not only the wait in `wait_one()`. Separate *blocked on a
  solve* from *processing a completed solve*; only the first is a cost that
  generation pays.
* **`SOLVER_THREADS=0` bypasses it entirely.** With no pool, `finish_move`
  solves **inline on the shard thread**, so the cost lands outside this timer
  altogether. An inline run and a fully-parallel run both report near zero, for
  opposite reasons.

### 0.2 Batch-width metrics count requests, not forwards — **FIXED**

`metrics.global_batches += 1` is incremented **in the scheduler shard**, once
per submitted request, before anything could merge it. So
`global_rows / global_batches` is *rows per request* and will not move when
coalescing lands — rev 1 proposed it as the headline success metric, and it
would have been blind to its own subject.

What is needed:

* the **worker** exports its own `(forwards, evaluated_rows)`;
* the shard keeps its request counters, renamed so the two cannot be confused
  (`requests`, `request_rows`);
* **adapter calls are not model forwards.** `rust_searcher_routed_flat_batch_adapter`
  splits a batch by network id and runs one forward per network, so under W7
  league play one adapter call is two forwards. Export both, and label them.

### 0.2b What landed, and what it measured

Committed to `self_play.rs` / `lib.rs`, verified by
`test_f4_phase0_telemetry.py` (5 new cases):

* **`sched_solve_wait_ns` is in `merge`**, and now spans only the block in
  `wait_one()`. Measured on a live solver: 1 shard 330 ms, 4 shards 618 ms — it
  scales with shards where it previously reported **zero**.
* **`sched_solve_pump_ns`** carries the harvest/resume/re-collect/retire work
  the old span silently included (≈1.2 ms of the 618 ms above, so the
  mis-attribution was small in magnitude and total in effect).
* **`sched_solve_inline_ns`** makes `--solver-threads 0` visible. Measured
  **725 ms** on the same workload where wait and pump are both zero — the
  largest of the three, and previously invisible to every timer.
* **`boundary_forwards` / `boundary_forward_rows`** come from the worker;
  `global_batches` / `global_rows` are documented in place as REQUESTS.
* **A compile-time coverage guard** in `merge` destructures every field of
  `SchedulerMetrics` with no `..` rest pattern. Mutation-tested: adding a field
  without handling it fails with `error[E0027]: pattern does not mention field`.
  This matters more than any test — `THROUGHPUT_LEVERS.md` §3.3 records the same
  omission happening once before, and a test cannot assert on a field it does
  not yet know about. A full scan confirms **70 of 70 fields merge**.

**The fragmentation mechanism, now measured rather than inferred.** rev 2
withdrew the "256 ÷ 4 = 64 slots per shard" derivation and was left with no
explanation for cloud2's 46.9 rows per call. On the production flat path,
identical seeds and identical total work:

| shards | requests | rows | rows/request |
|---|---|---|---|
| 1 | 400 | 2,981 | **7.45** |
| 4 | 1,558 | 2,981 | **1.91** |

The same 2,981 rows arrive as 3.9× more, 3.9× narrower calls. That is the
quantity the coalescer is meant to recover, and it is now pinned by
`test_shards_fragment_batches_rather_than_pooling_them` before the change rather
than argued about after it.

`boundary_forwards == global_batches` today, confirming one forward per request.
**That equality is the pre-coalescer baseline, and breaking it is the point** —
the test asserting it is expected to be inverted by the build, not preserved.

### 0.2c The sweep measured a solver that never ran — **FIXED**

Found by the laptop rehearsal, and it is the *same* defect
`THROUGHPUT_LEVERS.md` §3.1 records as having already cost a day of rented box:

> *"the sweep called the thread-pool sizer for the exact solver but never the
> function that installs its node budget -- and the budget defaults to zero,
> which disables solving entirely. Every point was measured with the solver
> switched off... The sweep dutifully printed `solver threads: 6 per shard x 2
> shards = 12 total`. Twelve threads were spawned. They sat blocked on an empty
> queue."*

**It was documented and not fixed.** `f4_phase_d_sweep.run_point` called
`pd.configure_solver_threads` and never `pd.configure_endgame_solver`, so
`endgame_solver()` reported `max_nodes = 0`, `solver_wants` refused every
position, and the solver axis added in rev 2 was measuring a solver that could
not run. The laptop rehearsal's first result — 264 games/h with four solver
threads against 261 with none — is what that looks like from outside.

Fixed:

* `run_point` installs the node budget and sets `endgame_solver_max_nodes`, the
  same field `_generate_iteration_rust` derives its per-call grant from, so the
  sweep and the run cannot disagree about whether solving is permitted;
* `--solver-max-nodes`, defaulting to the value in `--config-from-manifest`,
  because the budget is the solver's **admission threshold** and a sweep with a
  different one measures a different workload (class D);
* the harness **refuses** a solver split with no budget available, rather than
  running the axis against a disabled solver;
* every summary row carries **`solves_attempted` and `solves_answered`** —
  liveness, not configuration.

Verified: 4 solver threads → 3 attempted, 3 answered; 0 threads → 0 attempted.

**And the check that was supposed to catch this had the same bug.** The
rehearsal script asserted `solver_threads_total > 0` — a configured value — and
passed happily while nothing solved. It now asserts `solves_attempted > 0` on
solver-on points and `== 0` on solver-off points. A liveness check that reads a
config field is not a liveness check.

### 0.3 A warmed, repeated baseline

Rev 1 reasoned from a single iteration of one run. Before and after the
coalescer, take the same measurement on the same box and checkpoint:
**discard the first iteration after any restart** (`THROUGHPUT_LEVERS.md` §3.4
measured 8–15% cold), then at least three warm repetitions, reporting the median
and the spread.

**The baseline needs the box; the pipeline that produces it does not.**
`rehearse_sweep_laptop.sh` runs the whole of stage 8b at toy scale on a laptop
and asserts the infrastructure: that the harness completes with every axis
including the solver split, that every subsystem is live rather than merely
configured, that the repaired counters move, and that `measured_env.sh` is
written, sources cleanly and carries the `SWEEP_MEASURED` marker the launcher's
pass-2 guard refuses without.

It deliberately proves nothing about speed. A laptop measuring while it also
runs an editor and a test suite is the contended case `RENTING_A_BOX.md` §6 says
invalidates a throughput sweep, and the script says so in its own output rather
than leaving the reader to remember it.

**Phase 0 exit:** ~~the counters are correct and mutation-tested~~ **DONE**
(§0.2b). A warm baseline with an interval (§0.3) still requires the box, and is
the remaining gate before the coalescer can be judged.

---

## 1. What cloud2 actually measured, and what it did not

From `runs/seven_wonders_duel/cloud2/7wd_cloud_20260825T005745Z`, last
iteration, 1,000 games, `--generation-backend rust`:

| quantity | value |
|---|---|
| generation wall | 4,421 s (0.226 games/s) |
| `py_call_ns` — the evaluator worker | **4,056 s = 91.8% of scheduler wall** |
| requests submitted | 781,803 |
| **per request** | **5.19 ms, 46.9 rows** |
| `--rust-global-batch-cap` | 2,048 |
| `queue_wait_ns` | 11,906 s → **15.2 ms mean wait against 5.19 ms service** |
| CPU side | tree 15.0%, encode+pack 5.8%, scatter 4.3% |
| token padding | 718,909,460 wasted of 2,699,213,542 = **26.6% of slots** (36.3% on top of real tokens) |

Geometry: `rust_slots=256`, `rust_scheduler_workers=4`, `leaf_batch=1`,
`rust_max_inflight_batches=1`.

**Supported conclusions:**

1. **Generation is evaluator-bound.** One worker, 92% busy, 47 rows a call, with
   CPU-side work at a quarter of the wall.
2. **The cap was never reached.** 46.9 rows against 2,048.
3. **Requests queue ~3 deep** (15.2 ms wait / 5.19 ms service), so there is
   material already waiting that a coalescer could merge.
4. **Slot count has stopped paying.** `THROUGHPUT_ACTION_PLAN.md` measured 27
   rows/batch at 16 slots; cloud2 got 46.9 at 256.

**Withdrawn from rev 1:**

* ~~"The solver cost nothing (`sched_solve_wait_ns = 0`)."~~ The zero is an
  artifact of §0.1, now fixed. **cloud2's solver share remains unknown and is
  not recoverable** — the counter was broken while that run was recorded, so the
  number does not exist in its logs. It becomes measurable on the next run, and
  no core-split decision should be taken before then.
* ~~"256 ÷ 4 = 64 slots per shard, observed 46.9 = 73% occupancy."~~ `SlotBudget`
  is a **shared atomic pool**, so there is no static per-shard division and the
  arithmetic was a coincidence fit. The mechanism is now measured directly
  instead — see §0.2b: shards fragment the same work into ~3.9× more and
  narrower calls.
* ~~"36.3% of token slots are padding."~~ 26.6% of slots are padding; padding
  adds 36.3% on top of real tokens. Both exceed the ~20% trigger
  `THROUGHPUT_ACTION_PLAN.md` set for token bucketing, so the trigger fires
  either way — but the two numbers are not interchangeable.

### 1.1 Why `--inference-wait-ms` did nothing

`CoalescingEvaluator` (`loop_inference.py`) is constructed only on the **Python**
generation branch (`phase_d.py:4003`) and in `f4_throughput_bench`. The Rust
path uses `eval::spawn_py_flat_worker`, which ignores it.

And **`setup_cloud_7wd.sh` never passes `--inference-wait-ms` at all**
(rev 1 claimed it did). It takes argparse's 2.0 — the silent-default pattern of
`RENTING_A_BOX.md` §1.2, on a flag that was inert anyway.

### 1.2 Why the `max_inflight_batches` null was correct and uninformative

It was measured against a serial consumer. `max_inflight_batches` controls how
many requests a shard may have outstanding; the worker services them one at a
time, so extra outstanding requests lengthen a queue already ~3 deep. Re-take it
after the coalescer, when inflight becomes what *supplies* the merge.

---

## 2. What is missing

`eval::spawn_py_flat_worker` runs a single-consumer loop — `recv()`, one
`evaluate_batch_prepared_routed`, reply — and `submit_prepared_routed` sends
exactly one `WorkerRequest`. `max_rows` is a validation ceiling on a single
request; `timeout` is a client-side deadline on the ticket. Nothing merges.

Batch width is `max` over shards where it should be `sum`.

**`loop_inference.CoalescingEvaluator` is a partial model only.** It settles the
shape — drain, one forward, slice back, fan errors — but its contract differs
from the one needed here in four ways, and rev 1 wrongly cited it as covering
them:

| | Python version | Rust worker needs |
|---|---|---|
| zero wait | `deadline = now + 0` → breaks immediately, **never drains** | drain whatever is already queued without waiting |
| cap | `while positions < max_batch` then appends a whole request — **can overshoot** | never exceed `max_rows`; carry the excess request forward |
| routing | none | `net_ids`, including empty-means-zeros |
| errors | catches per batch and **continues** | fan out to every member, then latch and stop |

---

## 3. The build — a vertical slice, not one file

Rev 1 scoped this to `eval.rs`. That is where the loop lives, but the feature is
not usable without the rest of the path: the wait cannot be configured and the
new counters cannot be read.

| layer | change |
|---|---|
| `eval.rs` | the drain loop, carry-over, reply scatter, error fan-out, worker-side `(forwards, rows)` counters |
| `lib.rs` | plumb the wait through `self_play_many_flat_net` / `search_many_flat_net`; export the new counters in the metrics dict |
| `rust_bridge.py` | surface adapter-call vs model-forward counts from the routed adapter |
| `phase_d.py` | pass `--inference-wait-ms` to the Rust path; carry the counters into the training-log row and the heartbeat |
| `setup_cloud_7wd.sh` | pass the flag explicitly; add the sweep axis (§5) |
| `f4_phase_d_sweep.py` | the wait as a swept axis; report engagement per point |
| `sweep_launch_env.py` | **emit the winning wait**, or production runs a configuration the sweep never measured |

The loop:

```rust
while let Ok(first) = request_rx.recv() {
    let mut batch = vec![first];
    let mut rows = batch[0].states.len();
    let deadline = wait.map(|w| Instant::now() + w);
    let mut carried = None;
    loop {
        if rows >= max_rows { break }
        let next = match deadline {
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) => request_rx.recv_timeout(left).ok(),
                None => request_rx.try_recv().ok(),
            },
            None => request_rx.try_recv().ok(),
        };
        let Some(next) = next else { break };
        if rows + next.states.len() > max_rows { carried = Some(next); break }
        rows += next.states.len();
        batch.push(next);
    }
    // one forward; scatter replies IN RECEIPT ORDER; `carried` heads the next batch
}
```

### 3.1 Traps

* **Reply slicing** — record each request's row count; slice by running offset.
* **Error fan-out** — every member of a merged batch gets the error; the
  terminal-error latch still stops the worker. A carried-over request must also
  be answered, not stranded, when the worker is shutting down.
* **`net_ids`** — empty means "every row on network 0". Merging an empty request
  with a routed one must **expand the empty to zeros**; concatenating a short
  vector silently shifts rows onto the wrong network.
* **Ordering** — reply in receipt order, so a shard with `inflight > 1` keeps
  its sequencing.
* **Carry-over** — an `mpsc::Receiver` cannot un-receive. A request that would
  exceed `max_rows` is held and heads the next batch.

### 3.2 Wait policy

`try_recv`-only (wait = 0) is the intended default and is genuinely different
from the Python version's zero-wait behaviour: it drains what is *already*
queued without blocking, which at a ~3-deep queue should be enough. A positive
wait trades latency for width; it is an axis to sweep, not a default to assume.

---

## 4. Correctness gates (laptop, before the box)

The coalescer changes batch *composition*, which changes float reductions on
CUDA. So the contract splits:

1. **Deterministic evaluator, driven through the real flat worker.** The
   existing `MockEval` entry points (`closed_search`, `closed_search_resumable`)
   call `tree::search_closed` directly and **never touch the worker**, so they
   cannot gate this. Use a deterministic adapter through
   `self_play_many_flat_net` / `search_many_flat_net` (the `_row_eval` shape in
   `test_f4_scheduler`), with submissions synchronised so a merge is forced
   rather than hoped for.
2. **Bit-identical** on that path: digests, actions, visit counts and float
   targets, with coalescing forced on.
3. **Trajectory fingerprints** on the real net path — actions, digests, visit
   counts only. Float targets legitimately drift ~1e-5 with batch shape, the
   same contract `--rust-global-batch-cap` already lives under.
4. **Unit tests**, each written to fail on a wrong merge *and* on the
   pre-coalescer path:
   * members with different row counts scatter correctly;
   * an evaluator error reaches **every** member, including a carried-over one;
   * empty `net_ids` merged with a routed request puts each row on its own
     network — assert on *which evaluator saw which row*, not on row counts;
   * `max_rows` is never exceeded and the held-back request is served next;
   * replies arrive in receipt order;
   * a positive wait times out and still issues the partial batch;
   * a dropped receiver (abandoned ticket) does not wedge the loop;
   * shutdown drains or answers outstanding requests rather than hanging.
5. **Mechanism-engaged test**: two shards, synchronised, assert
   `requests_per_forward > 1`. Without it, a silent revert to one-per-forward
   passes everything above.

---

## 5. What folds into `setup_cloud_7wd.sh`

### 5.1 Knobs

```
INFERENCE_WAIT_MS=0            # NOW passed explicitly; the launcher passes nothing today
SWEEP_INFERENCE_WAIT_CSV       # e.g. "0,1,2"
```

Add `--inference-wait-ms` to the explicit-values block so the run records a
decision rather than argparse's 2.0.

### 5.2 The grid must be budgeted, and the workload must fit it

Rev 1's grid was **972 points per repetition** (3 slots × 3 caps × 3 inflight ×
3 workers × 3 wait × 4 solver) at 200 games each. Two problems:

* **200 games cannot fill 256 slots**, let alone 1,024. The harness only warns
  (`games <= max slots` measures activation, not throughput). Require
  `games >= k × max(slots)` — `k = 4` gives the pool time to refill — and
  **report realised occupancy** per point so the requirement is checked rather
  than assumed.
* Screen, then confirm. **Stage A**: a coarse one-repetition screen over the
  axes most likely to interact (slots × workers × wait), everything else pinned.
  **Stage B**: 3–5 repetitions on the 3–5 finalists, order reversed on alternate
  repetitions (`THROUGHPUT_LEVERS.md` §3.4).

The shard axis **inverts** with this change — shards fragmented batches before
and pool after — so no previous `measured_env.sh` carries over.

### 5.2b `SWEEP_CHECKPOINT` must carry the run's ARCHITECTURE

Found by rehearsing the pipeline on a laptop (`rehearse_sweep_laptop.sh`), and
it invalidates the current guidance.

`W2_W3_W5_CLOUD_ACCEPTANCE.md` says: *"Copy one W0 L checkpoint to the cloud
host. Its playing strength is irrelevant to gate timing; its `384x8x6` tensor
shapes are required."* **The width is no longer sufficient.** W1 slot
embeddings, W2's graph module, W4's hierarchical head and W5's action residual
each change the model, so a sweep against a plain 384x8x6 transformer measures a
different network from the one the run trains -- `THROUGHPUT_LEVERS.md` 3.1
("measure the configuration you are configuring") in a new dress. The
architecture-complete model measures **18,059,006 parameters**.

Two consequences for stage 8b:

* **A checkpoint from a previous run is usually unusable**, and fails loudly
  rather than silently: the encoder signature moves with the workstreams, and
  `load_checkpoint` refuses with *"checkpoint migration required -- encoder
  signature changed since this model was trained"*. cloud2's `current_best.pt`
  does exactly this against the current tree. Good -- but it means the box needs
  a checkpoint built for the run, not carried from the last one.
* **Build the sweep checkpoint from the run's own config**, so the two cannot
  drift. `rehearse_sweep_laptop.sh` does this with `PhaseDLoop._new_model()`,
  the same constructor the run uses; stage 8b should either do the same or
  refuse a `SWEEP_CHECKPOINT` whose contract does not match the launch flags.

Playing strength really is irrelevant, so an untrained checkpoint is fine. Its
*shape* is not.

### 5.3 Report, do not refuse

Rev 1 proposed refusing any point with `requests_per_forward <= 1.01`, width
below 46.9, or `sched_solve_wait_ns > 0`. All three reject legitimate outcomes:
a geometry with nothing queued to merge, a narrower-but-faster configuration,
and a run where some solver blocking is simply the right trade.

Instead:

* **Refuse** on malformed or missing instrumentation — a point whose
  worker-side counters are absent, whose fingerprints diverge, or whose
  occupancy shows the pool never refilled. That is an invalid measurement.
* **Report** engagement (`requests_per_forward`), realised width, realised
  occupancy and solver blocking as columns on every point, and select on
  **games per hour**.
* Force coalescing in the deterministic test of §4.5, where "did the mechanism
  engage" is a correctness question with a right answer, not a tuning outcome.

### 5.4 The solver split

Unchanged in shape, but **it cannot be interpreted until §0.1 lands**. Keep
`SWEEP_SOLVER_THREADS_CSV` and leave `--endgame-solver-max-nodes` alone (class
D — do not move it to absorb headroom the coalescer frees).

⚠️ `SOLVER_THREADS` is **per shard**, so it moves with the shard axis. Express it
as a total and divide at each point; `f4_phase_d_sweep --solver-threads-total`
does this.

### 5.5 Memory — a stage, not an assumption

Rev 1 claimed no memory change because the requests were already resident. That
ignores what merging *creates*: larger packed payloads on the Rust side, larger
padded tensors on host and device, and larger model intermediates. The sweep
also raises slots and cap together, and the preflight's VRAM floor is a fixed
function of `d_model`, not of batch geometry.

**Measure peak host RSS and device memory at the largest intended geometry
before running the full grid**, and drop grid points above what fits.

---

## 6. Order

1. **Phase 0** — instrumentation repair, mutation-tested, plus a warm repeated
   baseline on the box and checkpoint the comparison will use.
2. **Coalescer alone**, at the cloud2 geometry. Report **games/hour** with an
   interval, not batch width — `THROUGHPUT_LEVERS.md` §4.7.
3. **Re-sweep class A**, screened then confirmed (§5.2).
4. **Re-evaluate the boundary.** Padding gets *more* valuable as batches widen
   (26.6% of slots today, above the 20% trigger); per-call fixed cost gets
   *less* valuable, since coalescing already amortises it. Re-measure rather
   than pre-committing.
5. **Leaf batching — class C.** Multiplies with coalescing rather than competing.
   `conflict_free_waves` forbids collisions while virtual loss discourages them;
   under the PUCT root cloud2 ran, forbidding collapses realised wave width to
   ~1 and disables the mechanism under test. Set it from **realised** wave width
   and the histogram, not from branching: 7WD's median 4 / mean 5.6 legal actions
   bound fan-out, but root concentration binds first, and this project has
   already measured a requested leaf batch of 8 realising 1.97. cloud2's
   `leaf_batch=1, collisions: 0` is a clean baseline, not evidence.

---

## 7. Out of scope

* **The solver's admission set.** `--endgame-solver-max-nodes` is the admission
  threshold (`solver_wants` → `model.affordable(features, max_nodes)`), class D.
  Separately, the cloud2 buffers show **42.2% of solver nodes spent on
  declines**, and only the cheapest band (<10k nodes) yields the `exact` regime
  that earns a value-weight bonus. That is a target-quality question worth its
  own document, and it is not a throughput lever.
* **The gate path** inherits the coalescer through the same boundary, but has
  its own cost regime and is swept separately (`THROUGHPUT_LEVERS.md` §5.1).

---

## 8. Success criteria

* **Phase 0**: ~~each repaired counter fails a mutation test~~ **met** — the
  `merge` guard fails to compile on an unhandled field, and five telemetry tests
  pin the repaired counters. A warm baseline with a stated interval remains
  outstanding and needs the box.
* `requests_per_forward > 1` in the forced deterministic test — the mechanism
  engages.
* Worker-reported rows per **forward** materially above the request-level 46.9.
* Discrete fingerprints unchanged; bit-identical on the deterministic path.
* **Games per hour up, with an interval that excludes no change.** If width
  rises and games/hour does not, the bottleneck is not where this document says,
  and the honest outcome is to say so rather than bank the width.
