# Throughput action plan (7WD generation)

**Date:** 2026-07-26 (rev 2, after review). **Status:** Phases 0, 1, 2, 2b and 3b
built and run; Phase 3 (gather) and Phase 4 (sweep) not started.

**Generation is 1.50× faster than the Phase-1 baseline** — 1,661 → 2,495
games/hour at 32 slots — and the whole of that came from Phase 3b, a change to
how the network's embedder is *dispatched*, not from any of the scheduler work.

Four results reroute the programme:

1. The pipeline is **launch-bound** (Phase 0), so Phase 3b is unconditional and
   precedes Phase 4.
2. **Slot count is worth far more than the scheduler work** (Phase 1: +48% for
   16 → 32 slots, at no memory cost), so sweep the Phase 4 slots axis early.
3. **Wave widening needed a schedule change, not a batching change** (Phase 2 /
   2b). The conflict-free rule is exact but realized width was only 1.19, because
   the *blocked* halving order repeats a candidate on consecutive simulations.
   Interleaving the round takes width to 2.15 and throughput +19% — but see (4):
   that margin nearly vanishes once the forward is cheap, so it stays off.
4. **The win was in the embedder** (Phase 3b): fusing `TokenEmbedder`'s per-type
   loop into one gather plus one matmul removes ~18 host syncs and ~50 kernel
   launches per forward, for **1.50×** end to end. It changes no algorithm, keeps
   every checkpoint loadable, leaves training bit-identical, and provably flipped
   no search decision across 210k simulations.

Baseline is **1,661 games/hour** at 32 slots (mean of two runs; run-to-run spread
~9%), up from 1,301; **2,495** with the fused embedder, and 2,628 with the
interleaved schedule also on — the +5% that no longer justifies its cost.

**Rev 2 changed the ordering**: instrumentation now precedes all implementation,
because the review showed the plan's own gates could not have measured its own
work. Several rev-1 claims are withdrawn below rather than quietly edited.

Context: `CHANCE_ENUMERATION_PLAN.md` Step 5 measured where generation time goes
while A/B-ing chance capping. Those numbers still say the remaining wins are in
the scheduler and the boundary, not in more approximation — but they say it less
precisely than rev 1 claimed.

---

## 0. Ledger: what is measured, what is inferred, what is withdrawn

### Measured and trustworthy

| fact | value | source |
|---|---|---|
| per-batch cost is ~flat in rows | 7.27 ms at 27 rows/batch, 7.46 ms at 38.5 | Step 5 paired A/B |
| batch fill is far below the cap | mean 27 rows (slots 16), 38.5 (slots 24), cap **256** | same |
| `run_many` never replaces a finished slot | one `GameSlot` per job; `phase_d` chunks by `rust_slots` | `self_play.rs:990`, `phase_d.py:1161` |
| chance capping | +7.4% games/s (slots 16), +14.5% (slots 24) | Step 5 |
| baseline | 1,301 games/hour, 58% "GPU busy", 12% padding | Step 5 |
| conflict-free leaf-wave width, real corpus | **2.58** cheap / **2.03** full | corpus replay, 3,892 decisions |
| cheap moves with `per_action == 1` in round 1 | **49%** | same |
| leaf waves per search under the conflict rule | 20 → **8.0** cheap (2.5×), 96.6 → 47.8 full (2.0×) | same |
| decisions with ≤ 4 legal actions | **52%** | same |
| the eval worker is single-threaded and does not merge requests | `recv()` one at a time | `eval.rs:723` |

### Inferred, not yet established

* **The 7.3 ms is probably host dispatch, not device execution.** `_sync()` is a
  no-op unless `--diagnostic-sync` is passed (`rust_bridge.py:182`), which the
  Step 5 runs did not pass — so `forward_seconds` times an *asynchronous*
  enqueue and the real device wait lands later, in the blocking `.cpu()`.
  Independent plausibility check: 27 rows × ~60 tokens through a 4-layer d128
  transformer is tens of microseconds of compute, not 7 ms. **The per-batch
  dominance conclusion survives; its attribution does not.**
* Wall-clock shares below are **timer categories, not device attribution**:
  "forward" 58%, gather+D2H 15%, `encode_pack` 8%, tensor+H2D 6%, tree 2%.
* Per-row costs (these are safe, they scale with rows either way): `gather_d2h`
  **72 µs/row**, `encode_pack` 39, `pyo3_tensor` 21, `h2d` 7 — ~29% of wall.

### Withdrawn from rev 1

| claim | why |
|---|---|
| "~1.7× on games/hour" | two-point fit of a mis-attributed timer |
| "GPU busy > 0.85 → GPU-bound, stop" | `gpu_busy_fraction` is `forward_seconds / wall` with no synchronisation — a host timer |
| "each game contributes 16 rows per wave" | measured mean width is **2.58** |
| "leaf waves 20 → 2" | measured **20 → 8** |
| "at cheap budgets the conflict condition never triggers" | false for **51%** of cheap moves; `candidates = min(top_k, n_legal)` and most positions have few legal actions |
| "18% of capacity lost to idle slots" | `scheduler_idle_slot_cycles` counts loop iterations, not time |
| "re-run the A/B at 16 games / 16 slots" | `jobs == slots` cannot refill; the gate would have measured nothing |

### The correction that matters most

Rev 1 said leaf-batch quality loss was confined to deep halving rounds at full
budgets. Corpus measurement says **51% of cheap moves already start with
`per_action ≥ 2`**, because half of all decisions have ≤ 4 legal actions. So
today's `leaf_batch > 1` is unsafe on the *majority* case, which is the likeliest
explanation of the quality degradation previously measured — and it makes the
conflict-free rule the fix for that majority rather than a refinement for the
tail. The rev-1 bit-identity result (48/48 at `sims=20`, `leaf_batch` 2/4/8/16)
is real but was measured on six mid-Age-I positions with 12–18 legal actions,
i.e. the unrepresentative half.

---

## Phase 0 — instrumentation (blocks everything else)

Without this we cannot tell success from noise, and one outcome could reroute the
whole programme.

1. **Make the benchmark able to exercise refill.** Remove Python-side chunking in
   `f4_throughput_bench.py` and submit `jobs >> slots` in one Rust call — at
   least 8–16 windows (128–256 games at 16 slots). Report **steady state and
   final drain separately**; a single mean over both is what produced "27 rows".
2. **Separate the four times.** CPU enqueue, device execution, synchronisation,
   transfer — CUDA events around the forward, plus `--diagnostic-sync` runs as a
   cross-check and sampled NVML utilisation. Retire `gpu_busy_fraction` as a
   decision input.
3. **Time-weight the scheduler metrics.** Integrate live-slot count over wall
   time and record live slots per batch, instead of counting loop iterations.
4. **Fix memory telemetry.** CUDA peak *reserved* (not allocated), sampled RSS,
   and Rust arena nodes/bytes per active game — the last is what bounds
   `max_active_slots`.
5. **Controlled batch-size / token-length sweep** with a fixed evaluator and no
   search: rows ∈ {1, 8, 27, 64, 128, 256} × representative token lengths. This
   is what actually establishes the per-batch/per-row cost model.

**Exit criteria.** A cost model of the form *(fixed per batch, marginal per row,
marginal per token)* with device and host separated, plus a benchmark whose
steady-state window is at least 4× the drain window.

**Branch immediately after Phase 0:**

| what Phase 0 shows | consequence |
|---|---|
| the ~7.3 ms is mostly **host dispatch** | add CUDA graphs / `torch.compile` / static-shape buckets / `inference_mode` / buffer reuse as **Phase 3b**, and consider it *before* Phase 4 — it may substitute for much of the concurrency work |
| the ~7.3 ms is mostly **device execution** | batch widening is the only lever; Phases 1–2 keep their priority and Phase 3b is dropped |
| per-row costs already dominate at 27 rows | Phase 3 moves ahead of Phase 1 |

---

## Phase 0 results (2026-07-26, built and run)

### What was built

| item | where |
|---|---|
| single-call submission + steady/drain reporting | `f4_throughput_bench.py` (`--games-per-call`, `window_split`) |
| repeated-pass grid with per-cell medians | `f4_cost_model.py` (`--passes`) |
| host/device/sync separation via lazily-drained CUDA events | `rust_bridge.py` (`--cuda-events`) |
| time-weighted slot occupancy, per-batch live-slot and timestamp series | `self_play.rs` (`Occupancy`), exported in `lib.rs` |
| arena node/deep-byte peaks per active game | `tree_resumable.rs`, `state.rs::heap_bytes` |
| memory telemetry: CUDA **reserved** peak, sampled RSS, sampled NVML | `f4_throughput_bench.py` (`--resource-sample-hz`) |
| controlled rows × token-length sweep with a cost-model fit | `f4_cost_model.py` (new) |
| launch-bound vs device-bound discriminator | `f4_cost_model.py::queue_depth_probe` |
| gates | `test_f4_phase0_telemetry.py` (11 tests) |

Optional deps `psutil` and `nvidia-ml-py` are now in `requirements.txt`. Without
them the sampled metrics report `null` and an explicit `*_available: false` —
never `0`, because an unmeasured utilisation and a zero utilisation argue for
opposite decisions.

### The branch point is resolved: **launch-bound**

`queue_depth_probe` enqueues 50 forwards back to back with no synchronisation
and times the host loop and the device span independently. RTX 3070 laptop,
production checkpoint, widest token bucket:

| rows | host enqueue ms/forward | device span ms/forward | verdict |
|---|---|---|---|
| 1 | 6.46 | 6.46 | launch-bound |
| 8 | 6.36 | 6.36 | launch-bound |
| 27 | 6.61 | 6.61 | launch-bound |
| 64 | 7.33 | 7.33 | launch-bound |
| 128 | 10.46 | 10.54 | launch-bound |
| 256 | 16.15 | 16.35 | launch-bound |

The host loop matches the device span to within 1% at every width: the host
cannot get ahead of the device even when free to run 50 forwards ahead. The
device span is therefore *not* evidence of device work — it is the host's own
dispatch rate, observed on the device timeline. **The rev-2 inference was
right: the ~7 ms is host dispatch.** So Phase 3b is live and takes priority over
Phase 4, per the branch table above. A second run agreed within 10% (6.9–8.5 ms
at 1–27 rows, same verdict at every width).

Corroborating: a single row costs ~6.5 ms, which cannot be compute for a
4-layer d128 transformer; and 256× the rows costs only ~2.5× the time.

### Where the dispatch time goes (measured, 27×60 on the same GPU)

| section | ms/forward | share |
|---|---|---|
| full forward | 6.26 | 100% |
| `TokenEmbedder` | 4.59 | **73%** |
| 4-layer `TransformerEncoder` | 1.33 | 21% |

`TokenEmbedder.forward` loops over all 9 token types doing a boolean-masked
gather and scatter per type. Of its 4.59 ms, the 9 `mask.any()` host
synchronisations account for ~0.4 ms (measured separately: 0.50 ms with the
`.any()` sync versus 0.10 ms for the masks alone) — real but only ~9%. The
remaining ~4 ms is the launch load of the per-type dynamic-shape gather/scatter
kernels themselves. Phase 3b should target the embedder loop first; that
attribution is measured, but which specific rewrite recovers the time is not yet.

### Cost model

18 cells (3 token buckets × 6 widths), each the median of 3 passes of 25 calls.
**Single passes were not fittable**: two independent one-pass runs disagreed by
2× on the per-row terms and 20% on the fixed terms, which is why `--passes`
exists and defaults to 3. Even so, per-cell spread across passes is 8–47%, so
these coefficients are good to roughly ±15%, not better.

| target | fixed ms | per row ms | per padded token ms | R² |
|---|---|---|---|---|
| host total | **7.61** | 0.0460 | 0.00059 | 0.994 |
| device forward span | **6.30** | 0.0121 | 0.00029 | 0.963 |
| device total span | 7.07 | 0.0304 | 0.00037 | 0.988 |
| gather (per-row softmax loop) | 0.04 | **0.0229** | 0.00039 | 0.991 |
| tensor build | 0.34 | −0.0025 | 0.00016 | 0.996 |

Three things follow.

1. The fixed per-batch term dominates everything below ~100 rows, confirming
   rev 2's measurement and — with the probe above — identifying it as dispatch.
2. The gather is the cleanest fit in the set and has essentially **no** fixed
   cost: at the ~55 tokens/row of the middle bucket it is ~0.044 ms/row, which
   reaches the 7.6 ms batch cost near **170 rows**. That is inside the measured
   range, not an extrapolation: at 128 rows the gather is 4.1–6.6 ms against a
   device forward of 8.8–10.4 ms, and by 256 rows it is 10.3–13.6 ms against
   12.5–15.3 ms. **Phase 3's ordering argument survives**: widening batches
   without fixing the gather converts one bottleneck into another.
3. The host `forward_ms` section alone fits badly (R² 0.26) — as it should,
   since it times an asynchronous enqueue. Its poor fit is itself evidence for
   the launch-bound verdict.

### Benchmark honesty check

A 32-game single call (slots 16, cap 256) now reports its two regimes apart:

| window | batches | seconds | rows/batch |
|---|---|---|---|
| steady | 1,189 | 23.7 | **69.4** |
| drain | 2,654 | 45.1 | **39.7** |
| blended (the single mean this phase replaces) | 3,843 | 68.8 | 48.9 |

Steady/drain seconds ratio **0.53**, against the exit criterion of ≥ 4. The
benchmark cannot yet meet it: without the Phase 1 slot pool a call has no
refill, so every call is mostly drain. The apparatus to *measure* it is in
place, which is what Phase 0 owed; the criterion becomes checkable in Phase 1.

And the retired metric, measured against a real one — a separate 16-game run,
because NVML was not installed when the 32-game run above was taken:
`gpu_busy_fraction` **0.59** versus NVML utilisation **34.5%** (849 samples).
The host timer overstates GPU occupancy by ~1.7×. Do not use it.

### Memory, i.e. what actually bounds `max_active_slots`

From the same 32-game run: CUDA peak **reserved** 237 MB against peak
*allocated* 52 MB — the old metric understated the real device footprint by
4.6×, though at this scale neither is near a ceiling. On the Rust side the
arenas peaked at 9,297 live nodes summed across the 32 slots, and the deepest
single slot sampled held **4.8 MB** of arena (node structs at 1,040 B each plus
each node's cloned `GameState` heap). Sampled host RSS peaked at 1.2 GB in the
16-game run.

So the binding constraint on a rolling pool is host memory at single-digit MB
per active game, not device memory. That leaves Phase 1 substantial headroom to
raise `max_active_slots`; the exact ceiling should be read off these counters
during Phase 1 rather than assumed from this one configuration.

### Open from Phase 0

* `--games-per-call` gives the benchmark one call, but concurrency still equals
  the call's game count; genuine refill needs Phase 1's `max_active_slots`.
* Timestamps are loop-relative per shard, so the steady/drain split is exact
  only at `scheduler_workers = 1` (which Phase 4 mandates anyway).
* `arena_deep_bytes_slot_peak` is sampled round-robin, one slot per scheduler
  cycle: a lower bound on the true peak, not an exact one.

---

## Phase 1 — bounded rolling active-game pool

**Why before the sweeps:** the drain tail does not merely cost throughput, it
*contaminates* the measurements used to tune everything else. Mean batch 27 is an
average over concurrency decaying from 16 live games to 1, so a slots/cap sweep
run against it measures a blend of well-fed and starving regimes.

**Development.**

* `run_many` / `run_many_pipelined` accept `jobs >> slots` with an explicit
  **`max_active_slots`** parameter, distinct from job count.
* Queued jobs stay **lightweight** (seed + config); instantiate `GameSlot` only
  on activation, since `GameSlot::new` builds game state eagerly.
* The slot budget is **global**, not per scheduler worker.
* `phase_d.py` stops chunking and passes the whole group.
* Records must return in **input order**, independent of completion order.

**Two gates, because byte identity is not available on the net path.**

| evaluator | gate | rationale |
|---|---|---|
| **Mock** | records **byte-identical** to the chunked path for the same seeds | proves scheduler semantics; nothing but the schedule changed |
| **Real net** | (a) row-wise output invariance across batch shapes, then (b) measured record/action divergence | `self_play.rs:1402` already documents that batch-shape sensitivity can change search choices through CUDA float ties |

If (a) fails, the choice is explicit: enforce batch-invariant inference, or
classify rolling as a **numerical refactor** needing a strength/non-regression
gate rather than an identity gate.

Plus: every existing F4 gate stays green; active slots hold at
`min(max_active_slots, jobs_remaining)`; time-weighted live-slot occupancy rises.

### Phase 1 results (2026-07-26, built and run)

**What was built.** `SlotPool` in `self_play.rs` replaces `Vec<GameSlot>` with
per-job entries that are `Queued` (state + config only), `Active` (a real slot
with its arena) or `Finished` (a record). Both schedulers accept
`max_active_slots`; `0` preserves the old "everything active" behaviour. A
completed game is retired to its record the cycle it ends — freeing its arena and
its activation — and the lowest-indexed queued job takes its place. Records are
emitted by job index, so input order is independent of activation and completion
order. `SlotBudget` is a shared atomic, so sharded schedulers draw from **one**
global ceiling (each shard keeps one reserved activation, or a shard could starve
and the run would deadlock); a budget below `scheduler_workers` is now an error
rather than being silently widened. `phase_d.py` submits whole groups.

**Gate 1 (mock): passed, exactly.** Records are byte-identical to the unpooled
path at `max_active_slots ∈ {1, 2, 3, 5, 12, 64}`, and byte-identical to the
**chunked** path it replaces (three calls of three games versus one pooled call
of nine at a ceiling of three). Under a batch-shape-independent evaluator the
schedule provably cannot matter, and it doesn't.

**Gate 2 (real net): passed.** (a) Row-wise outputs are invariant across batch
shapes — the same position evaluates identically alone, in a group, and in a
padded group. (b) Divergence is then *measured*, not assumed: **0 of 432 moves**
across 6 games between pooled and unpooled play. Reported this way on purpose —
CPU float determinism does not license the same claim on CUDA.

**Occupancy: criterion met.** 64 games, ceiling 16, RTX 3070 laptop:

| | chunked (4 calls × 16) | pooled (1 call, ceiling 16) |
|---|---|---|
| time-weighted live slots | 13.47 | **14.62** |
| steady ÷ drain seconds | 1.40 | **3.98** |
| drain seconds | 83.8 | **41.6** |
| global batches | 14,438 | **13,146** |
| mean rows/batch | 26.6 | **29.3** |
| games/hour | 1,145 | 1,113 |

The remaining gap to 16.0 is the genuine end-of-queue tail: the last 16 games
have nothing left to refill from. It shrinks as games-per-call rises.

**Throughput at matched concurrency: no gain, and Phase 0 explains why.**
1,145 → 1,113 games/hour is a 2.8% regression, within the ~10% run-to-run spread
Phase 0 measured, so the honest reading is *flat*. This corrects rev 2's framing:
the drain tail was **not** costing meaningful throughput. Batch count fell only
9%, and on a launch-bound pipeline that caps the available win at ~9% before the
slightly wider batches give some of it back. Mean batch width barely moved
(26.6 → 29.3) because width is set by *games × rows-per-wave* (~2.6), not by how
well-fed the pool is — 16 games can only ever produce ~40 rows.

**But the sweep the pool made safe to run found the real lever: concurrency.**
Same 64 games, same cap, pooled, ceiling raised from 16 to 32:

| | pooled, 16 slots | pooled, 32 slots |
|---|---|---|
| games/hour | 1,113 | **1,650** |
| global batches | 13,146 | **7,433** |
| mean rows/batch | 29.3 | **51.7** |
| time-weighted live slots | 14.6 | 28.7 |
| peak host RSS | 1.20 GB | 1.21 GB |
| CUDA peak reserved | 237 MB | 237 MB |

**+48% for doubling the slot count, at no measurable memory cost.** Halving the
batch count on a launch-bound pipeline is exactly the win Phase 0's cost model
predicts, and the memory telemetry says there is plenty of headroom to go further.

**Attribution, measured rather than assumed.** The +48% is *not* the pool's doing:
a chunked run at 32 slots reaches **1,630** games/hour, statistically the same as
the pooled 1,650. Higher concurrency was always available; it simply had never
been swept, because until Phase 0 the measurements could not distinguish a
well-fed scheduler from a starving one. The pool's own throughput contribution is
~0 at matched concurrency (1,145 vs 1,113 at 16 slots; 1,630 vs 1,650 at 32).

So Phase 1's value is what its own success criteria claimed, and not more:

1. **The measurement contamination is gone** — steady ÷ drain went 1.40 → 3.98 at
   16 slots and 0.92 → 2.29 at 32, so Phase 4's sweeps no longer measure a blend
   of fed and starving regimes. This is what made the concurrency finding above
   trustworthy.
2. **Concurrency is decoupled from call size.** The benchmark could always fake it
   by choosing chunk size = concurrency; real generation cannot, because
   `phase_d` group sizes are whatever the curriculum mix produces. A 20-game
   group at `rust_slots=16` used to run 16 games and then 4 — a badly starved
   second window. Now it holds 16 throughout.
3. **Memory is bounded and measured** — a pool of 2 over 16 games peaks at a
   fraction of the arena of 16 (gated in `test_f4_phase1_pool.py`), and the
   32-slot run above shows the headroom that bound leaves.

**Not converted:** `f4_strength.py` and the `phase_d.py` arena/eval paths still
chunk. That is deliberate — they are gates, and changing their batch shapes
would perturb the results they exist to measure.

---

## Phase 2 — conflict-free leaf waves

**Rule:** never allow two in-flight simulations in the same root candidate's
subtree — drain before selecting a candidate already represented in the pending
wave. The taper falls out of the invariant rather than being configured.

**Realistic expectation, from the corpus:** wave width **2.58** cheap / 2.03
full, giving **2.5× / 2.0× fewer leaf batches**. Not the 10× rev 1 implied.
Because the rule preserves scalar order, it also batches adjacent simulations
from *different* candidates during repeated-candidate phases — which is where the
width-2 mode comes from — so no RNG-changing reordering is required.

> **Withdrawn by measurement.** The realized width is **1.19**, not 2.58 — see
> "Phase 2 results" below. The `per_action` repetition in the Gumbel schedule
> makes 90% of waves width 1, so the batch-count reduction is 1.13×, not 2.5×.

**Gates.**

* Bit-identity to `leaf_batch=1` across a **stratified corpus**: all
  `sims ∈ 16..24` and full budgets, stratified by legal count (≤2, 3–4, 5–8,
  9–16, >16), not just wide mid-Age positions.
* **The strong claim to test:** with the rule in place, *every* budget and every
  stratum should be bit-identical, because the rule forces a cut wherever a
  candidate would repeat. If that holds, leaf batching needs no quality gate at
  all. If it fails anywhere, the failing stratum names the mechanism.
* Assert the invariant directly (no two in-flight sims share a root candidate),
  not merely its consequence.
* Instrument the **realized wave-width distribution** in production runs; do not
  infer it from `top_k`.

### Phase 2 results (2026-07-26, built and run)

**What was built.** `SearchConfig::conflict_free_waves` (default off, so nothing
existing moved). A wave records the root edge of every member; before selecting,
if the upcoming candidate is already in the wave, the wave is cut and issued. The
cut happens *before* `select_leaf`, so the schedule and the RNG stream are
untouched and the next call re-derives exactly the same simulation — the rule
never reorders, it only cuts. The invariant is asserted directly on every wave
(Rust errors if an edge appears twice), and the realized width distribution is
recorded as a histogram all the way out to the benchmark rows.

**The strong claim holds.** Bit-identical to `leaf_batch = 1` across all five
legal-count strata (≤2, 3–4, 5–8, 9–16, >16) × budgets {16, 17, 20, 24, 67} ×
`leaf_batch` {2, 4, 8, 16}: action, visits, top-k, policy, action value, root
value and the full tree digest, at zero tolerance. That includes the ≤2 and 3–4
strata — the 52% of decisions the rev-1 identity result never covered. A negative
control is gated too: with the rule *off*, wide `leaf_batch` does diverge on this
corpus, so the identity result is evidence of something.

**A second reordering source, which the plan did not anticipate.** The first run
of the gate failed on exactly one quantity — `root_value`, in its last two bits,
only when `force=True`. Cause: simulations whose value is known at selection time
(terminal leaves, and forced children whose evaluation was cached) were backed up
*immediately*, jumping ahead of earlier simulations still waiting on the network.
Every backup adds into the shared root sum and float addition is not associative,
so the sum differed. Under the rule such simulations now join the wave carrying
their own value and back up in launch order; a wave that turns out to need no
rows at all is settled inline rather than submitted as a zero-row batch. Without
this the exactness claim is false, and `root_value` is a training target, so it
is not a difference one may wave through.

**The width prediction is withdrawn.** Realized mean wave width is **1.19**, not
2.58. The distribution (production run, 32 slots, `leaf_batch = 16`):

| width | 1 | 2 | 3–8 | 9–16 |
|---|---|---|---|---|
| share of waves | **90.1%** | 8.1% | 1.4% | 0.4% |

**Mechanism.** The Gumbel schedule spends `per_action` *consecutive* simulations
on one candidate before advancing, so whenever `per_action ≥ 2` the next
simulation always repeats the current candidate and the wave is cut at width 1.
Only a round with `per_action == 1` can produce a wide wave, and that is round 1
alone: at cheap budgets `per_action = sims / (rounds × candidates)` is 1 in round
1 and ≥ 2 in every later round, and full budgets have `per_action ≥ 2`
throughout. 138,955 of 148,291 waves — 94% — were cut by the conflict rule. The
rev-2 corpus estimate of 2.58 cheap / 2.03 full does not survive contact with the
realized distribution the plan itself asked for.

**Therefore no throughput gain.** Matched arms, 64 games, 32 slots, cap 256:

| | `leaf_batch=1` | conflict-free, `leaf_batch=4` | conflict-free, `leaf_batch=16` |
|---|---|---|---|
| leaf waves | 168,213 | 150,869 | **148,291** (1.13× fewer) |
| mean wave width | 1.000 | 1.163 | **1.188** |
| global batches | 7,433 | 6,884 | **6,808** (1.09× fewer) |
| mean rows/batch | 51.7 | 55.9 | 56.5 |
| games/hour | 1,560 | 1,548 | **1,590** |

+1.9% is inside the ~10% run-to-run spread (the same configuration measured 1,650
earlier in the session), so the honest reading is **flat**. It could not have been
otherwise: 90% of waves are width 1, which *is* `leaf_batch = 1`.

**What Phase 2 is worth.** Not throughput. It converts leaf batching from an
approximation into an exact transformation, which retires the quality question
that made `leaf_batch > 1` unusable — the plan's own conditional ("if it holds,
leaf batching needs no quality gate at all") is now discharged. That matters if
anything later needs wide waves. It is not a generation speedup.

**Consequence for the programme.** Wave width cannot be widened without
reordering simulations *within* a halving round so that consecutive ones come
from different candidates. That reordering was prototyped immediately — see
below — and it works, so the earlier "do not pursue" recommendation is withdrawn.

### Phase 2b — interleaved halving rounds (2026-07-26, prototyped and measured)

**The blocked order was never required.** `search.py:518` reads
`for action in candidates: for _ in range(per_action)`. Sequential halving fixes
only *how many* simulations each surviving candidate receives per round, never
the order they are issued in, so cycling the candidate list `per_action` times is
equally faithful. The blocked order is a reference-implementation choice that the
Rust port inherited by being bit-identical to it.

`SearchConfig::round_robin_candidates` (default **off**, both languages) switches
the interleaving. It is not a refactor: a round visits different leaves, so every
search output changes. It is a different, equally valid sample.

**Why "batch deeper after each halving" cannot work instead.** Exact batching
requires that no two in-flight simulations make a *statistics-based* decision at
a shared node. The root is uniquely safe because the Gumbel root reads its edge
from the fixed schedule and consults no visits or Q at all. Every node below it
selects by PUCT. Two simulations entering the same root candidate therefore both
decide at that candidate's child, where the second sees the first's virtual loss
in place of its real backup — and even if their paths diverge below that node,
the divergence decision was already contaminated. Root-level disjointness is not
merely sufficient for bit-identity, it is **necessary**; the batching boundary
cannot be moved deeper. Interleaving is the only lever.

**Gates (`test_f4_round_robin.py`), all passing.** Identity has to be
re-anchored, not reused:

| gate | result |
|---|---|
| Rust round-robin ≡ Python reference under round-robin (force on and off, 3 budget/top-k combos) | pass |
| Phase 2 exactness carries over: conflict-free `leaf_batch` 2/8/16 ≡ `leaf_batch=1` under the new order, incl. tree digest at zero tolerance | pass |
| allocation unchanged — same sims, same total root visits, same Gumbel top-k — while outputs *do* differ | pass |
| realized width rises (corpus of full searches): **1.28 → 3.34**, 2.6× fewer waves | pass |

**Production measurement.** 64 games, 32 slots, cap 256, `leaf_batch=16`,
conflict-free on, two runs per arm back to back:

| | blocked | interleaved |
|---|---|---|
| games/hour | 1,590 / 1,732 → **mean 1,661** | 2,005 / 1,949 → **mean 1,977** |
| global batches | 6,808 | **4,562** (1.49× fewer) |
| mean rows/batch | 56.5 | **85.1** |
| mean wave width | 1.19 | **2.15** |
| leaf waves | 148,291 | **87,951** |
| simulations run | 210,810 | 209,454 |

**+19%**, and the same work is being done (simulations within 0.6%; the
deterministic counters are identical across repeats). Run-to-run spread was 8.9%
blocked and 2.9% interleaved, so the effect is well outside noise. This is the
first phase in the programme to actually move throughput on the scheduler side,
and it lands close to the ~+20% predicted from the schedule arithmetic beforehand.

Realized width is 2.15 rather than the corpus figure of 3.34 because 44% of waves
are still width 1: positions with few legal actions halve their candidate set down
to **one** survivor, and a single-candidate round can only ever issue width-1
waves. Cheap searches also already had `per_action = 1` under the frozen config
(`top_k=16`, 16–24 sims), so interleaving changes nothing for them — the entire
gain comes from full searches, which are ~94% of all leaf waves.

**Not free, and off by default.** Every recorded gate and quality lock is anchored
to the blocked order, and the generated training distribution shifts. Turning this
on in production needs a **strength/arena justification, not an identity one**,
plus re-anchoring the F3 fixtures. What Phase 2 bought is that the batching on top
of it is provably exact, so the only open question is the reordering itself.

**Remaining headroom here:** mean rows/batch is 85 against a 256 cap (p95 134),
so the cap is not yet binding; and 55% of all NN rows are forced root-chance rows
that no wave work touches.

---

## Phase 3 — vectorise the legal-policy gather

**Moved ahead of the sweeps, for a dependency reason.** `rust_bridge.py:267` runs
a Python loop calling `torch.softmax` **once per row**. At 27 rows that is
1.9 ms/batch against 7.3 ms for the forward; it grows linearly with rows while
the per-batch cost is flat, so it overtakes the forward somewhere near 100 rows.
Widening batches before fixing it would convert the current 65/29 split into a
gather-dominated one and make Phase 4 measure a pipeline that degrades as it
widens.

**Development.** Padded masked softmax or a segmented softmax over the compacted
logits, compaction on GPU, then a **single** D2H transfer. Consider pinned
memory (the flag exists) and a narrower transfer dtype.

**Gate.** Row-wise outputs identical to the loop within tolerance, and the same
`gather_seconds` per row measured at 27 and 256 rows to prove it no longer scales
per row.

**Phase 3b — now unconditional, and promoted ahead of Phase 4.** Phase 0 showed
the pipeline is launch-bound at every batch width, with 73% of the forward's
dispatch in `TokenEmbedder`'s per-type masked gather/scatter loop. Levers, in
the order the measurement suggests: collapse the embedder's per-type loop
(single gathered index pass, or per-type padding removed), then CUDA graphs /
`torch.compile` / static shape buckets / `inference_mode` / buffer reuse. Any
change here alters the numerics of the forward, so it needs the same
non-regression treatment as Phase 1's real-net path — it is not an exact
refactor.

### Phase 3b results (2026-07-26, built and measured) — **1.50×**

**Root cause, precisely.** The loop's cost is not arithmetic, it is *shape*. Each
of the 9 token types does `mask = type_ids == t`, `mask.any()`, and then
**boolean-mask indexing** (`entity_ids[mask]`, `per_type[mask] = rows`), whose
output shape is data-dependent — so each type forces host synchronisations on top
of its handful of small kernels. Nine types, ~18 syncs and ~50 launches, to
compute something the GPU does in tens of microseconds.

**What was built.** `TokenEmbedder.fuse()` replaces the loop with:

* the 9 entity tables concatenated into one, with per-type id offsets, so the
  lookup is a **single gather**;
* the 9 feature projections as one `[n_types × d_model, MAX_FEATURES]` matmul,
  zero-padded beyond each type's own `in_features` so unused columns contribute
  exactly nothing, then a gather picking each token's own type slice.

Zero syncs, ~6 kernels. Crucially the **parameters are not renamed or moved**:
`entity`/`feature` remain canonical, so every existing checkpoint loads unchanged
and the training path keeps its original numerics bit for bit. The fused cache is
a detached snapshot used only in eval mode, and it self-invalidates on
`train()`, `load_state_dict()` and `_apply()` (`.to()`/`.float()`/…) — building
those hooks was prompted by a gate failure showing the cache would otherwise go
*partially* stale (it copies `entity`/`feature` but reads `type_embedding`/`aux`
live), which is a silently wrong forward rather than a loud one.

**Micro-measurement** (RTX 3070 laptop, production checkpoint, host ms/forward):

| rows × tokens | per-type loop | fused | speedup | max deviation |
|---|---|---|---|---|
| 1 × 60 | 6.79 | 2.19 | **3.10×** | 7e-07 |
| 27 × 60 | 7.24 | 2.03 | **3.57×** | 2e-06 |
| 64 × 60 | 7.39 | 2.33 | 3.17× | 2e-06 |
| 128 × 74 | 10.63 | 5.29 | 2.01× | 2e-06 |
| 256 × 74 | 15.94 | 10.02 | 1.59× | 2e-06 |

**End-to-end** (64 games, 32 slots, cap 256, `leaf_batch=16`, conflict-free on):

| arm | games/hour (runs) | mean | vs base | batches |
|---|---|---|---|---|
| blocked | 1590, 1732 | 1,661 | 1.00× | 6,808 |
| blocked **+ fused** | 2517, 2395, 2572 | **2,495** | **1.50×** | 6,808 |
| round-robin | 2005, 1949 | 1,977 | 1.19× | 4,562 |
| round-robin + fused | 2610, 2647 | **2,628** | **1.58×** | 4,562 |

`gpu_forward_seconds` falls 61.5 s → 18.9 s; device forward span 78.4 s → 31.9 s;
`device_forward_fraction` 0.54 → 0.35.

**The search is untouched.** Batch counts (6,808) and simulation totals (210,810)
are *identical* between fused and unfused arms — so a ~2e-6 output shift flipped
no search decision across 210k simulations. That is much stronger evidence than a
tolerance check, and it is gated (`test_f4_phase3b_fused.py`, 10 tests: agreement
on every head, padding exactness, offset mapping asserted directly, zero-padded
columns proven irrelevant, checkpoint layout unchanged, cache invalidation, no
training through a snapshot, and identical scheduler work end to end).

### This retires the Phase 2b recommendation

Round-robin's marginal value **collapses from +19% to +5%** once the embedder is
fused (2,495 → 2,628). It was only ever buying fewer forwards, and forwards are
now 2.5–3.5× cheaper. Since interleaving costs a re-anchoring of every recorded
gate and a shift in the training distribution, **+5% does not justify it.**
Recommendation: ship the fused embedder, leave `round_robin_candidates` off.

**Next on this axis, unmeasured:** `torch.compile` / CUDA graphs / static shape
buckets on top of the fused path (the flag already exists), and the gather's
per-row softmax loop (Phase 3 proper) — now a larger share of what remains, at
26–28 s of an 88 s run.

---

## Phase 4 — joint sweep

**Promoted by the Phase 1 measurement.** A single point on this sweep (slots
16 → 32) was worth **+48%**, more than Phases 1–2 were expected to deliver
together, and the memory telemetry says 32 is nowhere near a ceiling. Run the
slots axis early and properly rather than last.

Only now, and **jointly**, because the axes interact: once waves widen, 16 slots
× width 2.6 already changes what the 256 cap means.

* slots ∈ {8, 16, 24, 32, 48, 64} × `global_batch_cap` ∈ {128, 256, 512},
  subject to the memory ceiling Phase 0 measured.
* `max_inflight_batches` after that.
* Keep `scheduler_workers = 1`: sharded schedulers each submit their own batch to
  a single serial worker that does not merge them (`eval.rs:723`), so batch
  fragmentation is structural — that fully explains the historical pathology
  where 7 workers dropped mean batch to 5.2 rows and halved games/s. Tree work is
  ~2% of wall, so extra scheduler threads have little upside until batching is
  centralised across shards.

---

## Measurement matrix

Four arms, measured **independently**, so interactions and regressions are
attributable:

| arm | pool | conflict-free waves |
|---|---|---|
| baseline | off | off |
| pool-only | on | off |
| waves-only | off | on |
| combined | on | on |

Each at `jobs >> slots`, reporting steady state and drain separately, with the
Phase 0 timing split.

---

## Later, gated on results

* **Full-move round-robin reordering.** ~~Only if full moves remain a batch-count
  bottleneck after Phase 2.~~ **Done and measured — see Phase 2b: +19%.** What
  remains is the decision, not the code: it changes every search output, so it
  needs an arena/strength result and re-anchored F3 fixtures before production
  use. It is off by default until then.
* **Token-count bucketing** if padding exceeds ~20% once batches widen.
* **Chance capping (`cheap_double_reveal_offsets`, X=3).** Re-measure *after* the
  exact pipeline is tuned: it touches only the per-row ~29%, and it makes batches
  thinner, so its value moves in both directions. Independently gated on arena
  strength.

**Explicitly not next:** product chains (17.2% of forced children) and full-move
capping. Both add approximation risk to chase the same per-row share, while
Phases 1–4 are exact.

---

## Success criteria

Baseline: **1,301 games/hour**, 27 rows/batch, 12% padding, at slots 16 on the
pinned laptop configuration.

* Phase 0: a cost model with device and host separated, and a benchmark that
  actually refills slots. **No throughput target — this phase buys knowledge.**
  **Met, except refill**: the cost model exists with host and device separated
  and the launch-bound verdict established; the benchmark can submit one call
  and reports steady/drain apart, but true refill needs Phase 1's pool, so the
  "steady window ≥ 4× drain" criterion is carried into Phase 1.
* Phase 1: time-weighted live-slot occupancy ≈ `max_active_slots` through steady
  state; mock-path records byte-identical. **Met**: 14.62 of a 16 ceiling,
  byte-identical at every pool size and against the chunked path, real-net
  divergence measured at 0/432 moves. Throughput itself is flat — see the Phase 1
  results section; the gain the plan hoped for was not there to take.
* Phase 2: leaf batches per search down ~2.5× cheap / ~2.0× full, bit-identical
  across the stratified corpus. **Split verdict**: bit-identity **met** in full
  (every stratum, every budget, zero tolerance, plus a negative control); the
  batch-count target **missed** — 1.13×, not 2.5×, because realized wave width is
  1.19. Throughput flat. See the Phase 2 results section.
* Phase 3: `gather_seconds` per row flat in batch width.
* Phase 4: rows/batch and games/hour reported against the Phase 0 cost model —
  **the expected multiplier is deliberately left blank until that model exists**,
  since rev 1's estimate came from the timer this revision retired.
