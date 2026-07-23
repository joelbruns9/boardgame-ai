# Phase F4 — batched inference bridge + throughput (design & plan)

Working design doc for F4, the Phase F exit. `PHASE_F.md` keeps the milestone
summary + running log; this holds the detailed plan. Charter: `AZ_PROJECT_PLAN.md`
§8.4.

> **Production-search revision (2026-07-22):** the original F4 implementation is
> complete, but its registered 64-simulation, non-forced `leaf_batch` sweep found
> quality degradation beginning at batch 2. Section 9 therefore supersedes the
> original production-selection assumptions: full policy searches use 128 Gumbel
> simulations with complete tractable root-chance expansion, and `leaf_batch=1`
> plus many concurrent games is the production baseline. The original contract
> and results remain immutable historical evidence; the revised decision gets a
> versioned contract and a new forced-vs-forced quality run.

## 1. Goal / conjunctive exit gate

Make the complete self-play hot path fast enough to train at scale while keeping
the search approximation demonstrably safe. F4 exits only when **all** of these
are green:

1. **Exact refactor gate:** the arena/phase-split searcher at `leaf_batch=1`
   reproduces the sequential F3.3 oracle (structure and discrete outputs exact;
   floating outputs within 1e-9).
2. **Production-search quality gate:** the selected concurrency configuration
   passes the paired statistical, consequential-trap, regret, and
   playing-strength non-inferiority gates in §4. `leaf_batch=1` is the exact
   baseline; any `leaf_batch>1` remains an optional, separately gated algorithm.
3. **Inference/scheduler correctness:** batching preserves request alignment to
   defined floating tolerance; failures cannot strand requests; concurrent games
   are complete, valid, replayable, and buffer-schema compatible.
4. **Complete Rust hot path:** Rust owns game advancement, chance handling,
   search, action sampling, and move/game recording. Python is not re-entered per
   move; it remains only the control plane and the narrow Torch inference bridge.
5. **Local comparative throughput:** **≥20× aggregate self-play games/s vs the
   Python loop at identical locked forced/128 semantics on the development
   laptop**. Section 9g supersedes the original baseline semantics and defines the
   qualifying rerun; the earlier non-forced result and Kingdomino's ~28× are
   historical port evidence only. The Python loop is not run on rented cloud
   time.
6. **Cloud production calibration:** the checked-in Rust-only cloud sweep (§5c)
   selects and confirms the highest-throughput eligible configuration on the
   target high-end consumer GPU.

Games/s is the authoritative throughput and configuration-ranking metric.
Policy-eligible targets/s measures training-data yield: completed self-play moves
with `policy_excluded=false` whose improved root policy is written to the replay
record, divided by steady-state wall time. Moves/s, scheduled simulations/s,
forced outcome evaluations/s, unique ordinary NN evaluations/s, total NN rows/s,
tokens/s, and collision rate explain them. No CPU cores are reserved for an exact
endgame solver: 7WD has no planned exact tail solver, and scheduler/core count is
a cloud tuning variable.

## 2. Why F4 is different from F1–F3

F1–F3 were equivalence ports. F4 deliberately changes simulation ordering:

- The F3.3 Gumbel root performs sequential halving. Each completed simulation
  changes deeper PUCT statistics before the next descent and completed-Q before
  the next root reduction.
- Batched NN inference requires several descents to be outstanding before their
  evaluations and backprops return, so `leaf_batch>1` cannot be bit-identical to
  the sequential reference.

The sequential F3.3 searcher therefore remains the permanent oracle. F4 first
proves that its structural refactor is exact at `leaf_batch=1`, then treats the
batched path as a quality-gated parallel-search algorithm.

### 2a. WU-style pending visits below the Gumbel root

The conditional `leaf_batch>1` design uses **Watch the Unobserved-style
incomplete visit counts**, adapted from UCT to the existing PUCT formula. For
every edge:

```text
O(s,a) = simulations launched through (s,a) but not yet backed up
N_eff(s,a) = N_completed(s,a) + O(s,a)

score(s,a) = Q_completed_actor(s,a)
             + c_puct * P(s,a)
               * sqrt(N_completed(s) + O(s))
               / (1 + N_completed(s,a) + O(s,a))
```

Only completed evaluations contribute to Q. Pending simulations alter the
exploration counts, not the value estimate. This is a better fit than synthetic
virtual loss for 7WD's probability-weighted chance edges: it does not invent a
player-relative result or perturb an expected chance Q.

Known consequence: when pending visits greatly outnumber completed visits, the
exploration term can over-spread work among poorly measured edges. That is the
quality/fill failure mode measured by the collision, regret, and policy gates and
the reason a pending-edge cap remains a separately gated fallback.

The Gumbel root still owns the exact candidate schedule. WU-PUCT operates only on
PUCT descents below the forced root candidate. Build pending simulations in the
same order as the sequential schedule, permit a leaf wave to cross candidate
boundaries within a round, and **never cross a halving-round boundary**. Before a
reduction, every pending result must be backed up and every `O` counter must be
zero.

Pending-path rules:

- Record stable `(node_id, edge_id, chance_child_id)` steps in an arena.
- Increment the traversed node/edge incomplete counts immediately after each
  descent so the next descent sees them.
- Preserve scalar RNG consumption and chance-materialization order.
- Deduplicate identical pending `leaf_id`s for NN evaluation; expand once, but
  backprop once per scheduled simulation/path, including collisions.
- Remove incomplete counts before real backup. A guard must remove them on
  evaluator error, cancellation, or shutdown.
- Terminal leaves need no NN call but still complete one scheduled simulation.

The published WU-UCT theory is not a proof for Gumbel PUCT or 7WD; it is the
principled starting point. Kingdomino-style synthetic virtual loss remains a
fallback/control only if WU produces unacceptable collision/fill behavior, and
would have to pass the same laptop quality gates before becoming eligible.

Full BU-UCT-style aggregation is also out of initial scope. Its pending-count
threshold and same-node aggregation can reduce redundant inference, but
collapsing several simulations into an aggregate backup would complicate exact
Gumbel round budgets, visit accounting, and per-simulation trajectory
bookkeeping. If WU still permits too much concentration, test a separately
gated pending-edge cap before considering aggregated backups.

Primary references:

- WU-UCT: <https://arxiv.org/abs/1810.11755>
- WU/BU theory and incomplete-count analysis: <https://arxiv.org/abs/2006.08785>

### 2b. Batching priority: independent games before pending leaves

F4 exposes two batching dimensions, but they are no longer equally preferred:

1. **Cross-game global batching is primary.** Many resumable Rust game/search
   slots each contribute one causally complete ordinary leaf, while complete
   forced root outcomes contribute natural multi-row bursts.
2. **Intra-search leaf waves are conditional.** A game may contribute a
   quality-approved `leaf_batch>1` using WU pending visits only if practical
   cross-game concurrency cannot reach the GPU throughput knee.

Independent games preserve exact sequential Gumbel semantics and cannot collide
in one tree. Self-play optimizes aggregate games/s, so higher per-game latency is
acceptable. Logical game slots should substantially outnumber persistent CPU
workers because most slots wait for inference; do not create one OS/Rayon task per
simulation. Use long-lived scheduler shards that cooperatively advance many
slots, and introduce Rayon only for measured coarse-grained work.

Kingdomino's `leaf_batch≈6`, ~32 slots, and RTX-5090 measurements remain useful
starting evidence, not 7WD defaults. Cloud calibration may tune scheduler workers,
game slots, global batch cap, wait window, and buffering. It may not tune
`leaf_batch`, forced-chance semantics, or the pending-visit policy after the
quality lock.

## 3. System architecture

### 3a. Cooperative Rust scheduler + Python/Torch inference worker

Use a dynamic ready/pending scheduler rather than fixed A/B game banks:

```text
                    complete game records
                           ^
                           |
 ready Rust slots -- selection/materialization/encode --> pending eval queue
      ^                                                       |
      |                                                       v
 backprop/scatter <-- completed eval queue <-- Python/Torch GPU worker
```

Each slot is a complete self-play game and is in one of:

```text
ReadyForCpu -> WaitingForRoot -> NeedForcedChildren -> WaitingForForced
                                                    -> ReadyForSearch
ReadyForSearch -> WaitingForLeaf -> ReadyForBackprop -> SearchComplete
                                                     -> GameComplete
```

While one GPU batch executes, the Rust scheduler prepares leaves for other ready
slots and/or the next host batch. Double buffering is an implementation/tuning
option, not a requirement that games belong permanently to two sets.

Start with one scheduler thread. The cloud sweep may add scheduler shards, each
cooperatively advancing a subset of slots, if one CPU thread cannot feed the GPU.
No fixed core budget and no process per game. Rayon is not assumed either way;
use it only if a measured, coarse-grained scheduler workload benefits from it.

The Python entry point releases the GIL (`py.detach`) while the Rust scheduler and
game/search work run. A dedicated Python inference thread owns Torch/the GPU and
performs one Python call per **global** batch. `py.detach` surrounds Rust work; it
does not mean the Python GPU forward itself runs without Python.

### 3b. Complete Rust self-play hot path

Rust owns:

- game setup/state and all action/chance application;
- full/cheap simulation schedule and Gumbel/search seeds;
- arena tree, WU pending paths, evaluation scatter, and backprop;
- draft-prior blending;
- temperature action sampling;
- move statistics, policy eligibility, and complete replayable game records;
- concurrent slot lifecycle and deterministic result ordering by game index.

Python owns the cold/control work: checkpoint/model lifecycle, training,
promotion, manifests, and buffer persistence. It must not regain control after
each move. Conceptually the generation call is:

```text
run_self_play(seeds, config, evaluator) -> completed Phase-D-compatible records
```

### 3c. Phase-split resumable searcher

Refactor search into explicit resumable phases:

1. **Selection/materialization:** follow the current Gumbel schedule's forced root
   candidate, then WU-PUCT below it; sample/materialize chance children; record a
   stable path; stop at a terminal or unevaluated leaf.
2. **Prepare pending evaluation:** collect at most
   `min(leaf_batch, simulations remaining in the current halving round)`, never
   launch a path from the next round, deduplicate leaf IDs, and encode non-terminal
   leaves into reusable flat numeric buffers with cached actor/legal metadata.
3. **Apply evaluation/backprop:** validate aligned results, expand each unique
   leaf, clear pending counts, and backprop once per scheduled path.
4. **Yield/finish:** return another eval request, a completed search result, or a
   completed game record to the scheduler.

`leaf_batch=1` bypasses pending-count behavior and is the exact oracle path.

### 3d. Transformer inference boundary

Default boundary:

```text
Rust flat buffers -> one Python/Torch global-batch call -> aligned compact results
```

Requirements:

- Reuse numeric input buffers; do not rebuild Python `Token`/`Encoding` objects.
- Pass cached actor/legal data; do not recompute it in Python.
- Transfer only the search value and gathered legal policy entries back to Rust;
  do not copy the full policy vector or unused auxiliary heads to CPU.
- Measure token-length distribution and padding ratio; length bucketing is a
  profile-driven cloud option.
- Instrument Rust encode/pack, queue wait, PyO3 attach/call/extract, tensor
  construction, H2D, forward, legal gather, D2H, and scatter separately.
- Use coarse asynchronous counters in production runs and a separate synchronized
  CUDA-event diagnostic mode; synchronization itself distorts throughput.

Only if the optimized Python/Torch boundary remains a material critical-path
bottleneck do we consider tch/ONNX/native inference.

### 3e. Complete tractable root chance without semantic drift

Force expansion currently evaluates a child but discards its priors; an ordinary
first visit evaluates that child again and stops there. Naively storing priors and
marking the child expanded would make the first visit descend an extra ply and
change search semantics.

The safe optimization is a cached-pending expansion:

1. Batch-evaluate forced children and cache value + priors.
2. Preserve their current seeded value/visit accounting.
3. On the first ordinary visit, consume the cached evaluation, materialize edges,
   stop at that node, and backprop the cached value exactly as the old re-eval did.

This optimization gets its own `leaf_batch=1` equivalence gate.

Production full searches force-expand **every outcome of every tractable
immediate root chance event for every legal root action**, not only the eventual
Gumbel winner. The edge value is the exact network-weighted expectation
`sum_o P(o) * V(s[a,o])`; ordinary simulations reuse the cached child evaluation.
This removes immediate reveal luck from early sequential-halving reductions but
does not claim exact game-theoretic value: deeper chance remains sampled and the
child value is still a network estimate.

Forced children must be yielded to the cooperative scheduler and globally
coalesced across games. A GPU batch cap may split a complete outcome set into
chunks but may never truncate or sample it. Actions with the same chance signature
(for example build/discard/wonder using one pyramid card) may share CPU outcome
enumeration, but their child states and NN evaluations remain distinct.

`AgeDeal` remains the explicit exception because complete deal/layout enumeration
is combinatorial. Starting-player alternatives use a separately gated paired
sampler: both actions are evaluated against the same sampled deal set. This
reduces comparative noise without representing the sample as exhaustive.

Here **tractable** is a versioned semantic allowlist, not a runtime performance
decision. Under the current rules it means a chance chain containing only
`CardReveal`, `GreatLibraryDraw`, and `WonderGroupReveal`; any chain containing
`AgeDeal` or a future unregistered chance kind is intractable. Current invariant
bounds are:

| Chance chain | Maximum outcomes per action |
|---|---:|
| One pyramid card reveal | 23 |
| Two same-back pyramid reveals, without replacement | `23 * 22 = 506` |
| Great Library: choose three of five off-board progress tokens | `C(5,3) = 10` |
| Two pyramid reveals plus Great Library | `506 * 10 = 5,060` |
| Second wonder group: choose four of eight unseen wonders | `C(8,4) = 70` |
| Age deal/layout | sample-only; outside the allowlist |

The play-age action generator has at most six accessible slots and at most six
actions per slot (build, discard, and four wonders). Treating every slot as if it
simultaneously attained the worst reveal geometry gives a conservative whole-root
envelope of `6 * (5,060 + 5 * 506) = 45,540` forced rows. Reachable roots should be
far smaller because topology, affordability, and wonder ownership constrain those
maxima, so F4-R1 must record the empirical mean/p50/p95/max `F` distribution.

Assert these per-kind and whole-root bounds. Exceeding one is a contract/schema
error requiring an explicit design revision; never silently truncate, sample, or
disable forced expansion based on load. GPU caps may only chunk the complete set.

## 4. Validation strategy

### 4a. Exact structural/refactor gate

At `leaf_batch=1`, require:

- chosen action, top-k, visits, fingerprints, tree topology, chance-child order,
  and RNG consumption exact;
- values, node/edge Q, and policy target within 1e-9;
- coverage across draft, Ages I–III, pending choices, terminal leaves, sampled
  chance, force expansion off/on, and varied sims/top-k/seeds;
- the same gates before and after arena conversion, phase splitting, full-game
  Rust ownership, and forced-child cache optimization.

### 4b. WU + `leaf_batch` quality lock

The original registered run used 64 simulations with
`force_expand_root_chance=false` and found no eligible `leaf_batch>1`; batch 2
already failed multiple quality bounds. That result is retained and not retuned.
The production revision reruns the same frozen thresholds at 128 simulations with
complete tractable root expansion enabled for both baseline and candidates. See
§9 for the exact matrix and contract-versioning rules.

Quality calibration runs on the laptop, not in the cloud throughput sweep. Use a
stratified corpus containing all phases/chance mechanisms, random production
states, the full consequential Phase-E corpus, baseline-clean trap positions, and
positions with both close and large sequential action gaps.

For paired position/search seeds, measure:

- chosen-action agreement, stratified by sequential action gap;
- root-policy divergence and root-value error;
- sequential-tree regret of the fast choice,
  `Q_seq(best) - Q_seq(fast_choice)`;
- new consequential blunders and aggregate consequential trap-rate delta;
- collision/duplicate-leaf rate;
- WU-vs-sequential playing strength in paired, seat-swapped games.

Calibrate natural search variance with sequential-vs-sequential independent-seed
comparisons. At F4.0, preregister the numerical non-inferiority margins and sample
sizes before observing the `leaf_batch` sweep. Hard requirements:

- zero new blunders on baseline-clean consequential fixtures;
- a one-sided non-inferiority bound on aggregate consequential trap-rate delta;
- a declared playing-strength non-inferiority margin (initial proposal: no worse
  than -20 to -25 Elo; freeze the actual margin at F4.0).

`curriculum_seed:35:63` identifies Phase-E fixture game index 35 at move 63. It is
reported as a known 100%-baseline-blunder sentinel, not used as a vacuous “no
worse than sequential” gate.

Select the largest `leaf_batch` satisfying every quality bound. Check in the
approved algorithm contract as `f4_quality_lock.json` (pending policy,
`leaf_batch`, any pending-edge cap, force-expansion setting, gate version/results).
The cloud sweep consumes it and rejects quality-sensitive overrides.

### 4c. Inference/coalescer correctness

- Scalar and batched results align within explicit CPU/CUDA tolerances; do not
  claim CUDA bit identity across batch shapes.
- Request grouping, permutation, mixed token lengths/legal counts, deduplication,
  and scatter preserve exact row ownership.
- A deterministic fingerprint evaluator makes sequential and cooperatively
  scheduled full games identical at `leaf_batch=1`.
- Adapter errors, OOM, timeout, cancellation, and shutdown wake every pending
  slot, clear incomplete counts, and surface the original operational error.

### 4d. Full-game harness correctness

- Rust records replay through the Python engine to the same terminal fingerprint.
- Actions, chances, starting player, seeds, per-move visits/policy/root value,
  full/cheap flag, `policy_excluded`, agents metadata, and victory fields preserve
  the Phase D schema.
- Concurrent completion order does not change deterministic output ordering.
- Long stress runs have no incomplete games, lost requests, deadlocks, or
  unbounded queue/memory growth.

### 4e. Conjunctive performance gates

Pass both the local comparison (§5a) and Rust-only cloud calibration (§5c), after
all quality/correctness gates above are green.

## 5. Benchmark contract

### 5a. Laptop comparative gate (Python vs Rust)

The ≥20× comparison is run on the development laptop. Section 9g supersedes
the original baseline semantics. The qualifying Python reference and Rust
candidate both use closed search, `sims=128`, `top_k=16`, complete tractable root
expansion, the locked paired `AgeDeal` sampler, and the same full/cheap schedule,
checkpoint, seeds, inference precision, draft prior, and temperature. Freeze and
record:

- complete config: net/checkpoint hash, sims/full-cheap schedule, top_k, closed
  mode, force flag, draft prior, temperature schedule, seeds/first-player order;
- device and inference precision, Torch/CUDA versions, CPU thread settings;
- warmup, completed-game count, steady-state window, and repetitions.

Compare the frozen Python loop to the quality-approved Rust production algorithm
on those identical semantics and laptop settings. Require the lower bound of the
repeated-run games/s speedup estimate to clear 20×. A sampled/non-forced Python
reference cannot qualify. Report games/s plus policy-eligible targets/s, moves/s,
total NN rows/s, and simulations/s so game-length and forced-work differences are
visible.

Laptop profiles also establish:

- Rust mock-evaluator CPU ceiling;
- WU/`leaf_batch` quality lock;
- end-to-end inference-boundary breakdown;
- basic cooperative-scheduler scaling.

### 5b. Component metrics

Report at minimum:

- games/s, policy-eligible targets/s, moves/s, scheduled simulations/s;
- forced outcome states/search and forced evaluations/s, split by chance kind;
- requested vs unique ordinary NN leaves/s, total NN rows/s, forced-cache hits,
  terminal leaves, collisions and dedupe ratio;
- mean/p50/p95 leaves and tokens per global batch, padding ratio;
- scheduler ready/waiting/idle fractions and queue wait;
- Rust tree, chance, encode/pack, and record time;
- PyO3/tensor, H2D, GPU forward, gather/D2H, and scatter time;
- CPU utilization, GPU busy fraction, peak host/GPU memory, and OOMs.

Concurrent component timers overlap; report worker/scheduler and inference-service
timelines separately rather than pretending every component sums to wall time.

### 5c. Cloud production calibration (Rust only)

Before renting the target box, check in:

- `f4_throughput_bench.py` — benchmark/instrumentation driver;
- `run_f4_cloud_sweep.sh` — reproducible staged production sweep.

The shell script does **not** run the Python self-play loop and does **not** sweep
`leaf_batch`, WU/KD policy, virtual-loss magnitude, sims, mode, force-expansion, or
unapproved inference precision. Those are quality/algorithm inputs locked on the
laptop.

It captures git/checkpoint/config hashes, CPU/GPU, driver, CUDA/Torch, clocks/power
limit, and thread environment, then performs staged sweeps rather than one giant
Cartesian matrix:

1. correctness smoke + environment manifest;
2. transformer forward geometry by global batch/token distribution;
3. concurrent game slots × scheduler threads/shards;
4. global batch cap × coalescer wait window;
5. single/double buffering, pinned memory, token bucketing, and approved Torch
   compile modes;
6. long repeated confirmation of the top configurations.

Every row emits JSONL/CSV with the §5b metrics and exact command. Selection is
highest steady-state games/s among quality-eligible, non-OOM configurations. The
winning production config and box spec are recorded in `PHASE_F.md`.

## 6. Historical implemented sub-sequence

This section records how F4.0–F4.7 were implemented. It is not a second live
production plan: Section 9 supersedes its search semantics, measurement baseline,
and remaining execution order.

- **F4.0 — contracts + instrumentation.** Freeze §4 quality thresholds, §5 laptop
  comparison, boundary timers, cloud sweep schema, and the quality-lock format.
- **F4.1 — arena + resumable phase split.** Stable pending paths and
  selection/eval/backprop state machine. Gate: `leaf_batch=1` exact (§4a).
- **F4.2 — WU-PUCT leaf waves.** Incomplete counts, round-boundary assertions,
  collision/dedupe behavior, error cleanup, local batch evaluator. Gate: §4b
  quality sweep; check in `f4_quality_lock.json`.
- **F4.3 — complete Rust self-play slot.** Move/game RNG policy, draft prior,
  action sampling, game advancement, and Phase-D-compatible records. Gate: exact
  `leaf_batch=1` full-game oracle + replay/schema checks.
- **F4.4 — cooperative scheduler + global coalescer.** Dynamic ready/pending slots,
  dedicated Python/Torch inference worker, deterministic scatter/order, failure
  and stress gates (§4c–d).
- **F4.5 — transformer fast boundary + semantic-safe forced cache.** Flat reusable
  buffers, cached actor/legal, compact GPU gather/D2H, padding instrumentation,
  forced-child cached-pending expansion. Gate: boundary tolerance + `leaf_batch=1`
  equivalence.
- **F4.6 — performance exit.** Laptop ≥20× comparison, checked-in Rust-only cloud
  sweep, and target-box calibration/confirmation. F4 exits only when the full
  conjunctive gate in §1 is green.
- **F4.7 *(conditional)* — native NN.** tch/ONNX/other native inference only if
  §3d profiling shows the optimized Python/Torch boundary materially limits the
  cloud result.

F4.0–F4.7 describe the implemented original design. The post-measurement
production revision is sequenced as F4-R0–F4-R5 in §9.

## 7. Locked decisions / reopen conditions

1. **No exact endgame solver and no reserved CPU cores.** Reopen only if later
   trained checkpoints show systematic, consequential public-tail blunders that
   more search/forced expansion cannot resolve; any solver must not inspect true
   buried identities.
2. **Complete hot path in Rust; Python/Torch stays at the inference/control
   boundary initially.** Reopen native inference only from measured boundary cost.
3. **Cooperative dynamic scheduling**, not fixed A/B banks or a process per game.
   Scale scheduler shards from cloud measurement.
4. **Complete tractable root-chance expansion is production semantics.** Deeper
   chance remains sampled; `AgeDeal` uses a separately gated paired sampler.
5. **`leaf_batch=1` plus many concurrent games is the production baseline.** WU
   incomplete visits remain the implementation for conditional `leaf_batch>1`;
   KD synthetic virtual loss is a fallback/control only.
6. **Any `leaf_batch>1` is a laptop quality decision.** It is eligible only after
   passing the forced/128 gate and is locked before cloud calibration, never tuned
   on cloud throughput alone.
7. **Cloud rental time optimizes absolute Rust self-play games/s.** The Python
   comparison belongs to the laptop gate.

## 8. Kingdomino reuse, with 7WD-specific changes

Reuse KD's lessons and scaffolding where they fit: GIL-detached Rust work,
in-process GPU ownership, request/result alignment, flat reusable buffers,
component instrumentation, and support for both intra-search and cross-game
batching even though 7WD prioritizes the latter.

Do not inherit KD-only constraints: no endgame-solver CPU reservation, no assumed
1–2-core budget, no fixed `leaf_batch=6`, and no claim that KD synthetic virtual
loss is automatically best for 7WD's chance-weighted Gumbel tree.

## 9. Production-search revision: actionable plan

### 9a. Locked target semantics and cost model

The initial training target is:

```text
mode                         closed
full policy search          128 Gumbel simulations, top_k=16
root chance                 complete expansion when tractable
deeper chance               sampled closed search
AgeDeal                     paired common-outcome sampling (separate gate)
production baseline         leaf_batch=1
cheap policy-excluded move  existing 16–24 simulation schedule, force enabled
```

`closed` means each node represents the public information state and branches on
public-consistent hidden outcomes from the unseen pool; it does not determinize
one hidden deck for the entire search. Section 3e defines the forced root subset
of that chance handling.

The 128 simulations are the sequential-halving decision budget, not a complete NN
budget. Record each search as:

```text
total NN work = 1 root row
              + F forced-outcome rows
              + U unique ordinary-leaf rows
```

`F` is state-dependent and is not charged against the 128 scheduled simulations,
but it is bounded and asserted by Section 3e. Cached forced children avoid
duplicate ordinary inference when first visited. Training capacity estimates and
throughput reports use total NN rows and tokens, not nominal simulations alone.

### 9b. F4-R0 — version the revised experiment contract

Preserve `f4-contract-1`, its 64/non-forced run, and all frozen results. Create a
new contract/addendum that:

- fixes full-search `sims=128`, `top_k=16`, closed mode, and
  `force_expand_root_chance=true`;
- retains the existing corpus, seeds, sample sizes, statistical method, and every
  numerical non-inferiority threshold unchanged;
- defines the exact baseline as forced `leaf_batch=1`;
- registers candidate batches `{2,4,6,8}`, with 8 diagnostic if resource limits
  require prioritizing the requested target of 6;
- records the AgeDeal paired-sampler configuration separately;
- adds forced/ordinary/total NN-work metrics to the quality lock and throughput
  manifests.

Gate: schema/manifest tests reject mixed v1/v2 results and any force, sims,
checkpoint, corpus, or batching mismatch.

### 9c. F4-R1 — make forced expansion a global scheduler producer

Replace synchronous per-slot forced evaluation with a resumable forced phase:

1. Materialize all tractable root outcomes and stable ownership metadata.
2. Yield forced rows to the same global coalescer used by ordinary leaves.
3. Combine forced rows from multiple games up to the global row/token cap.
4. Scatter results, verify probability mass, seed child value/visit, and retain
   priors for cached-pending first visits.
5. Permit chunking only for memory/batch caps; complete every chunk before Gumbel
   search proceeds.

Also cache outcome enumeration by chance signature within a root. This saves CPU
work only: action-specific child construction and evaluation remain separate.

Gates:

- exact result and canonical topology digest versus current forced
  `leaf_batch=1` for varied phases, seeds, sims, and top-k;
- mixed forced/ordinary global-batch ownership and permutation tests;
- cancellation/OOM/adapter failures leave no stranded slot or partial
  probability-weighted edge;
- chunked and unchunked forced evaluation agree within the existing boundary
  tolerance.

### 9d. F4-R2 — establish the leaf-1 concurrency frontier

Before reopening intra-tree batching, sweep the exact baseline:

```text
leaf_batch       1
game slots       32, 64, 128, 256
CPU workers      1, approximately half physical cores, physical cores minus 1
force            true
full sims        128
```

Use persistent coarse-grained workers, each owning many logical games. Do not
spawn a Rayon task per game or simulation. Add scheduler shards only when one
worker cannot feed the global queue.

For each row report games/s, policy-eligible targets/s, total NN rows/tokens/s,
GPU busy fraction, forward-only utilization, queue starvation, CPU utilization,
RAM/VRAM, padding, forced-burst geometry, and p50/p95 move latency.

Define the practical GPU knee as both:

- at least 90% GPU busy time in the synchronized diagnostic; and
- at least 95% of the isolated-forward rows/s at the observed token distribution.

If an eligible leaf-1 configuration reaches that knee without OOM or unstable
queues, `leaf_batch>1` has no production justification unless its repeated-run
one-sided 95% lower confidence bound improves games/s by at least 10%, its
policy-eligible-targets/s ratio lower bound is at least 1.00, and it passes every
quality gate. Games/s is the authoritative 10% override metric; target yield is a
non-regression guardrail, not an alternate ranking rule.

### 9e. F4-R3 — rerun quality at forced/128 semantics

Run two deliberately separated experiments. The chance diagnostic and `AgeDeal`
calibration are required. The `leaf_batch>1` batching arm is required only if
F4-R2 shows that practical leaf-1 concurrency misses the GPU knee; otherwise it
is optional research and is not an F4 exit gate.

**Batching gate (production decision):**

```text
reference   force=true, sims=128, leaf_batch=1
candidates  force=true, sims=128, leaf_batch=2,4,6,8
```

Use the unchanged position, natural-variance, consequential-trap, regret,
policy-divergence, value-error, and paired strength gates. Compare forced to
forced so the result isolates pending-leaf approximation. No candidate advances
to strength testing unless it first clears every position gate.

**Chance diagnostic (algorithm understanding):**

```text
sampled reference  force=false, sims=128, leaf_batch=1
forced candidate   force=true,  sims=128, leaf_batch=1
```

Report action/policy/value changes, consequential outcomes, playing strength,
total NN work, and wall-clock cost. This diagnostic does not weaken the decision
to use complete tractable root expansion, but can expose systematic network-value
errors hidden by chance averaging.

For AgeDeal, test paired sample counts `{4,8,16}` against a paired-32 diagnostic
reference and choose the smallest count whose action agreement, value error, and
playing-strength bounds pass the same relevant thresholds. The selected count is
then locked; it is not a cloud tuning variable.

### 9f. F4-R4 — test leaf-X only against equal GPU supply

If F4-R2 shows leaf-1 underfilling, compare configurations that offer roughly
equal maximum ordinary rows per scheduler cycle, for example:

```text
leaf=1, slots=192
leaf=2, slots=96
leaf=4, slots=48
leaf=6, slots=32
```

Prefer causal/root-candidate-distinct pending paths before unrestricted WU waves;
never cross a sequential-halving boundary. Rank only quality-eligible rows. A
larger leaf batch wins only if its repeated-run lower confidence bound improves
games/s by at least 10% over the best practical leaf-1 row, its
policy-eligible-targets/s ratio lower bound is at least 1.00, and it does not
increase failed/cancelled games or memory instability. These are the same
authoritative metric and guardrail as F4-R2.

### 9g. F4-R5 — final production confirmation

Run the laptop Python-vs-Rust comparison and RTX-5090 sweep with identical locked
forced/128 semantics. The earlier non-forced 20x+ result demonstrates port speed,
but it is not the final production measurement. Confirm:

- the existing >=20x laptop lower-confidence-bound gate at equal settings;
- absolute games/s and policy-eligible targets/s on the 5090;
- stable long-run memory and queue behavior;
- no quality-sensitive cloud override;
- whether the measured F4.7 native-inference condition is triggered.

The production selection order is therefore:

1. complete tractable root chance;
2. cross-game forced-child batching;
3. `leaf_batch=1` with oversubscribed logical game slots;
4. coarse scheduler-worker scaling;
5. quality-approved causal leaf batching only when the exact configuration cannot
   feed the GPU efficiently.
