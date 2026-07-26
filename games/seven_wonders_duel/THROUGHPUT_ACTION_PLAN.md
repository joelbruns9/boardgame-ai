# Throughput action plan (7WD generation)

**Date:** 2026-07-26 (rev 2, after review). **Status:** nothing built.
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

**Phase 3b (conditional on Phase 0):** launch-overhead work — CUDA graphs,
`torch.compile`, static shape buckets, `inference_mode`, buffer reuse.

---

## Phase 4 — joint sweep

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

* **Full-move round-robin reordering.** Only if full moves remain a batch-count
  bottleneck after Phase 2. Sound (order within a halving round is arbitrary) but
  it permutes RNG consumption and changes full-move outputs, so it needs its own
  arena justification. Measure Phase 2's actual full-move benefit (2.0×) first.
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
* Phase 1: time-weighted live-slot occupancy ≈ `max_active_slots` through steady
  state; mock-path records byte-identical.
* Phase 2: leaf batches per search down ~2.5× cheap / ~2.0× full, bit-identical
  across the stratified corpus.
* Phase 3: `gather_seconds` per row flat in batch width.
* Phase 4: rows/batch and games/hour reported against the Phase 0 cost model —
  **the expected multiplier is deliberately left blank until that model exists**,
  since rev 1's estimate came from the timer this revision retired.
