# Review request — 7WD generation throughput (Phases 0–3b)

Companion to `THROUGHPUT_ACTION_PLAN.md`, which is the measurement and decision
**record**: what was predicted, what was measured, and which predictions were
withdrawn. This document is the **review brief** — what changed, which invariants
must hold, and where a second pair of eyes is worth most.

**Two changes are live by default** (fused embedder, vectorised gather); three
are built but off (`conflict_free_waves`, `round_robin_candidates`, and the
rolling pool's `max_active_slots` beyond what `phase_d` now passes). The live pair
is where review effort should go — see §5.

Headline: **1.99× generation throughput in the F4 benchmark configuration**,
1,633 → 3,250 games/hour (64 games, 32 slots, cap 256, `leaf_batch=1`), with
*identical* batch counts and simulation totals before and after — and, since
2026-07-27, **1.89× on the real Phase D generation path**, 1,361 → 2,578
games/hour (`f4_phase_d_ab.py`, three interleaved repetitions, identical discrete
trajectories in every arm; §8).

The uncomfortable summary: **none of the throughput came from the scheduler work
the plan was built around.** Phases 1 and 2 are correctness and measurement work.
All of the speed is Phases 3b and 3, both of which are about *dispatch* — kernel
launches, host syncs, and per-element Python — not about search or arithmetic.

## 1. Scope

| Phase | What | Commit | Primary files |
|---|---|---|---|
| 0 | Instrumentation: host/device split, time-weighted occupancy, arena bytes, cost model | `7eefb37` | `f4_throughput_bench.py`, `rust_bridge.py`, `self_play.rs`, `lib.rs`, `state.rs`, `tree_resumable.rs`, `f4_cost_model.py` (new) |
| 1 | Bounded rolling active-game pool (`SlotPool`, `SlotBudget`) | `2f9b556` | `self_play.rs`, `lib.rs`, `phase_d.py` |
| 2 | Conflict-free leaf waves (`conflict_free_waves`) | `0356489` | `tree.rs`, `tree_resumable.rs`, `self_play.rs`, `lib.rs` |
| 2b | Interleaved halving rounds (`round_robin_candidates`) | `4ebc639` | `search.py`, `tree.rs`, `tree_resumable.rs`, `lib.rs` |
| — | Correction: 2b risk claim withdrawn | `07c17f7` | plan only |
| 3b | Fused token embedder | `894374e` | `net.py`, `f4_throughput_bench.py` |
| 3 | Vectorised legal-policy gather | `d1b790e` | `rust_bridge.py`, `f4_throughput_bench.py` |
| — | Fix: unsound regime verdict in the Phase 0 probe | `c816789` | `f4_cost_model.py` |
| — | **Ship the live pair by default** | `398febf` | `inference.py`, `rust_bridge.py`, `f4_throughput_bench.py` |
| — | Review 1 response: fusion envelope, cache staleness, budget leaks | `8819012` | `net.py`, `inference.py`, `self_play.rs` |
| — | Phase D A/B (the missing end-to-end measurement) | uncommitted | `f4_phase_d_ab.py` (new) |
| — | Review 2 response: `.data` contract, abort accounting, gate holes | `2de8976` | `net.py`, `self_play.rs`, `f4_phase_d_ab.py`, `test_f4_phase3b_fused.py` |
| 4 | Free-axis sweep (slots/cap/inflight) on the real path | uncommitted | `f4_phase_d_sweep.py` (new) |
| 4b | Per-game curriculum-bot routing (+28%) | uncommitted | `lib.rs`, `phase_d.py`, `test_f4_bot_routing.py` (new) |

New test files: `test_f4_phase0_telemetry.py` (13), `test_f4_phase1_pool.py` (13),
`test_f4_phase2_waves.py` (5), `test_f4_round_robin.py` (4),
`test_f4_phase3b_fused.py` (11), `test_f4_phase3_gather.py` (8),
`test_f4_bot_routing.py` (9). Suite: **527 Python + 16 cargo, green**
(2026-07-27; was 517 + 8 at first submission). New tools: `f4_phase_d_ab.py`,
`f4_phase_d_sweep.py`.

## 2. What changed (design)

### Live by default

**Phase 3b — fused token embedder** (`net.py::TokenEmbedder.fuse`). The per-type
loop ran, for each of 9 token types, `mask = type_ids == t`, `mask.any()`, and
boolean-mask indexing. Boolean-mask indexing has a *data-dependent output shape*,
so each type forces host synchronisations on top of its small kernels: ~18 syncs
and ~50 launches per forward, to compute what the GPU does in tens of
microseconds. Fusing gives one gather (9 entity tables concatenated with per-type
id offsets) and one matmul (9 feature projections as a single
`[n_types × d_model, MAX_FEATURES]` weight, zero-padded past each type's own
`in_features`), then a gather to select each token's slice.

Parameters are **not** renamed or moved. `entity`/`feature` remain canonical, so
every checkpoint loads unchanged and training keeps its numerics bit for bit. The
cache is a detached, eval-only snapshot; `Evaluator.__init__` builds it.

**Phase 3 — vectorised gather** (`rust_bridge.py`). Three costs that scale per
row while the per-batch cost is flat: one `torch.softmax` launch per row, two D2H
transfers, and a Python `float()` per legal action (~5.8M per run). Replaced by a
scatter into a padded `[rows, max_legal]` matrix plus one row-wise softmax, one
concatenated transfer, and one bulk `tolist()`.

> **Worth knowing before you review it.** The first implementation used
> `scatter_reduce_(amax)` + `index_add_`. `index_add_` accumulates with atomics on
> CUDA, so it is **not run-to-run deterministic** (~8e-7 relative over 50
> repeats) — that would have made generation irreproducible at a fixed seed. It
> was caught while writing §5 of this document, not by the gates, and is now
> fixed and gated. Treat the rest of §5 as a list of things that have *not* had
> that scrutiny.

### Built but off

**Phase 1 — rolling pool.** `SlotPool` holds each job as `Queued` (state+config
only), `Active` (a slot with its arena) or `Finished` (a record); a game is
retired the cycle it ends and a queued one activates in its place. `SlotBudget` is
a shared atomic so sharded schedulers draw from **one** global ceiling, each shard
keeping one reserved activation (without it a shard can starve and the run
deadlocks). `phase_d` now submits whole groups with `max_active_slots=rust_slots`
instead of chunking.

**Phase 2 — conflict-free waves.** Never two in-flight simulations in one root
candidate's subtree; the wave is cut *before* selecting, so the schedule and RNG
stream are untouched. Distinct root edges own disjoint arena nodes, so members
cannot see each other's virtual loss — which makes `leaf_batch > 1` an exact
batching of `leaf_batch = 1`.

**Phase 2b — interleaved rounds.** `for action in candidates: for _ in
range(per_action)` becomes a cycle. Same allocation, different order.

## 3. What is already gated (please don't re-verify by hand)

* **Fused embedder ≡ loop**: every head within 1e-4 (measured 2e-6); padding
  exactly zero; entity offsets asserted against the per-type tables directly;
  zero-padded feature columns proven irrelevant *by polluting them* and asserting
  bit-equality; checkpoint state-dict keys and values unchanged; cache
  invalidated on `train()`, `load_state_dict()`, `_apply()`; no training through a
  snapshot; identical scheduler work end to end.
* **Vectorised gather ≡ loop**: **run-to-run deterministic** (repeated identical
  calls bit-identical); row-wise to 1e-6 across widths — measured exactly 0 on the
  verification corpus; each row's policy normalised over *its own* legal actions;
  zero-legal terminal rows produce no NaN; identical scheduler work **and final
  fingerprints** end to end.
* **Pool**: records byte-identical to the unpooled path at pool sizes
  {1,2,3,5,12,64} *and* to the chunked path it replaces; real-net divergence
  measured at 0/432 moves; queued jobs hold no arena; the global budget is not
  multiplied by shards.
* **Conflict-free waves**: bit-identical to `leaf_batch=1` across all five
  legal-count strata × 5 budgets × `leaf_batch` {2,4,8,16}, tree digest included,
  zero tolerance — plus a **negative control** proving unconstrained batching
  diverges on the same corpus.
* **Round-robin**: Rust ≡ the Python reference under the new order; Phase 2
  exactness carries over; allocation provably unchanged while outputs differ.

## 4. Findings that changed the plan (the record, for context)

Five predictions in the plan did not survive measurement. They are documented in
place rather than edited away; the short version:

1. **Launch-bound, not device-bound.** Host enqueue matched device span within 1%
   at every width 1–256.
2. **The drain tail cost no throughput.** Phase 1 met its occupancy criterion
   (13.5 → 14.6 of a 16 ceiling) and moved games/hour not at all. What it bought
   was uncontaminated measurement — and that immediately found the real lever:
   **+48% for 16 → 32 slots**, which had simply never been swept.
3. **Wave width is 1.19, not the predicted 2.58.** The blocked halving order
   repeats a candidate on consecutive simulations, so 90% of waves are width 1.
4. **Interleaving fixes that (+19%) — but only +5% once the embedder is fused.**
   Its value was always "fewer forwards", and forwards got 3× cheaper.
5. **A bigger model leaves the launch-bound regime.** Device µs/row is flat in
   width for d256L8 and d384L12 on this GPU, so batch width buys ~nothing there.
   Relevant to the cloud config; see §6.

## 5. Focus areas (highest value)

### Correctness of the live pair

* **`TokenEmbedder.fuse` zero-padding argument.** The claim is that
  `feature_weight[t][:, in_features_t:] = 0` makes the wide matmul equivalent to
  the narrow one *regardless of what the feature tail contains*. Gated by
  pollution, but the argument is the thing to check.
* **Cache invalidation completeness.** `train()`, `load_state_dict()` (post-hook)
  and `_apply()` all unfuse. **Is there a fourth path that mutates parameters in
  place while in eval mode?** An optimizer step without `.train()`, an EMA/weight
  averaging routine, `torch.no_grad()` in-place edits. A missed path gives a
  *partially* stale cache — the copied `entity`/`feature` go stale while
  `type_embedding`/`aux` stay live — which is a silently wrong forward. This is
  the single highest-value thing to attack.
* **Segmented softmax on empty segments.** Terminal rows have zero legal actions,
  so their denominator is zero. The argument is that `legal_rows` never refers to
  them, so the division never reads it. Gated, but please check the reasoning
  holds for *every* caller, not just self-play.
* **~~`scatter_reduce_(amax)` determinism~~ — found and fixed before review.**
  `amax` is deterministic; `index_add_` (the denominator) was not. Replaced by the
  padded formulation, which measured bit-identical to the loop and is gated by
  `test_gather_is_run_to_run_deterministic`. Flagged here because it is the
  *class* of bug worth hunting for: **which other kernels on the changed paths use
  atomics or otherwise vary run to run?** The fused embedder's
  `nn.functional.embedding` backward does, but inference-only use never reaches
  it; please confirm.

### Rust scheduler

* **`SlotBudget` accounting.** `holds_reserved` + `budget_held` must satisfy
  `active_count == holds_reserved + budget_held` on every path, including the
  error path in `refill` (which calls `release_activation` after a failed
  `GameSlot::new`). A leak here strands capacity; an over-release lets shards
  exceed the global ceiling.
* **Progress guarantee.** Both schedulers `continue` when `active_count == 0` and
  work remains, on the assumption that retiring freed budget the next refill will
  spend. Convince yourself this cannot spin.
* **`drain_immediate_wave` reachability.** It errors if a member lacks
  `immediate_value`. Reachable only when `unique_leaf_ids` is empty — is that
  invariant actually maintained at all three call sites?

### Measurement honesty (this work's recurring failure mode)

* Two predicted magnitudes and one regime verdict were wrong before being caught.
  **Assume more remain.** In particular: the 1.99× is a single machine, one
  checkpoint, two runs per arm, with ~9% run-to-run spread on the slower arm.
* `f4_cost_model.py --passes` defaults to 3 because single passes disagreed by 2×
  on per-row terms. If you re-measure, use it.

### Assumptions worth an explicit sign-off

* **Shipping ~2e-6 numerical drift into generation by default.** Justification:
  identical batch counts, simulation totals and final fingerprints across 210k
  simulations; it is the same class of change as a different GPU. Reviewer should
  agree this needs no strength gate, or say what would.
* **`Evaluator` fusing on construction** means gates and arena runs also use the
  fused path. Deliberate — they should measure what production runs — but it
  moves recorded numbers by ~2e-6.

## 6. Known limitations / out of scope

* **Phase 4 (joint sweep) not done.** One point on the slots axis was worth +48%;
  the axis is unswept, and `global_batch_cap=256` will start binding (p95 batch
  was 134 at 32 slots before Phase 3).
* **The Phase D A/B is one box, one checkpoint, one slot count.** 1.89× is
  measured on an RTX 3070 laptop at `rust_slots=16` with the d128 L4 production
  checkpoint. It does not decompose the shortfall against the benchmark's 1.99×
  into the slot count and the small bot groups, and it says nothing about cloud
  hardware (below).
* **Cloud config unmeasured.** A faster GPU pushes toward launch-bound (width
  pays), a bigger model pushes toward device-bound (width buys nothing); measured
  here, the model effect dominates on fixed hardware. Run `f4_cost_model.py` on
  the box — it now prints the regime and its implication directly.
* **Likely next bottleneck is CPU-side.** `encode_pack` is 15.6 s of a 67.6 s run
  (23%) and tensor build 5.0 s (7%); neither shrinks with a better GPU, and both
  are serial on one scheduler thread feeding one eval worker.
* **55% of NN rows are forced root-chance rows**, untouched by any of this.
* **Not converted to the pool:** `f4_strength.py` and the `phase_d` arena/eval
  paths still chunk, deliberately — they are gates, and changing their batch
  shapes would perturb what they measure.
* **Not done:** fp16 transfer dtype (~1e-3 into PUCT, wrong class of change),
  CUDA graphs / `torch.compile` on the fused path (flag exists, untried),
  pinned-memory D2H.

## 7. Running the gates

```bash
# This work's own gates (~30 s)
python -m pytest games/seven_wonders_duel/test_f4_phase0_telemetry.py \
  games/seven_wonders_duel/test_f4_phase1_pool.py \
  games/seven_wonders_duel/test_f4_phase2_waves.py \
  games/seven_wonders_duel/test_f4_round_robin.py \
  games/seven_wonders_duel/test_f4_phase3b_fused.py \
  games/seven_wonders_duel/test_f4_phase3_gather.py -q

# Full 7WD regression (~11 min) + Rust
python -m pytest games/seven_wonders_duel -q
cd games/seven_wonders_duel/seven_wonders_rust && cargo test

# Reproduce the headline, production settings, ~4 min
python -m games.seven_wonders_duel.f4_throughput_bench --mode rust \
  --exploratory-leaf1 --checkpoint <ckpt> --output /tmp/after --device cuda \
  --games 64 --games-per-call 64 --repetitions 1 --warmup-games 2 --slots 32 \
  --global-batch-cap 256 --max-inflight-batches 1 --scheduler-workers 1 \
  --python-workers 1 --cuda-events --allow-underfilled
# ...and the before, same command plus:
#   --no-fused-embedder --no-vectorized-gather

# Reproduce the Phase D A/B, real generation path, ~26 min
python -m games.seven_wonders_duel.f4_phase_d_ab \
  --checkpoint <ckpt> --output /tmp/phase_d_ab --games 64 --repetitions 3

# Regime / cost model on any box (~3 min)
python -m games.seven_wonders_duel.f4_cost_model \
  --checkpoint <ckpt> --device cuda --output /tmp/cost_model
```

## 8. Review response (2026-07-26)

Every finding was checked against the code and, where it was a measurement claim,
re-measured. **Four accepted and fixed, one accepted and corrected in wording, one
accepted with a data-driven amendment, one accepted in part.** Nothing was
dismissed.

### P0 — fusion enabled beyond its measured envelope — **accepted, amended by measurement**

Correct that it was shipped universally on one measurement. Measured the rest
(RTX 3070 laptop, unfused → fused, host ms/forward):

| device | model | 8 rows | 64 rows | 256 rows |
|---|---|---|---|---|
| CPU | d128 L4 | 1.31× | **0.90×** | — |
| CPU | d256 L8 | 1.00× | 0.98× | — |
| CUDA | d128 L4 | 3.17× | 2.52× | 1.17× |
| CUDA | d256 L8 | 2.06× | 1.13× | 1.01× |
| CUDA | d384 L12 | 1.76× | 1.07× | 1.02× |

**The CPU half of the concern is confirmed** — 0.90× at width, no launch overhead
to recover. **The larger-model half is not**: on CUDA, bigger models converge
towards *neutral* (1.01–1.02×), never a loss, because a deeper model adds kernels
as fast as it adds arithmetic.

So the gate is on **device, not model size** (`net.fusion_is_profitable`): fuse on
CUDA, decline on CPU, `force=True` to override. Gating by model size would have
cost real speedup on cloud for no measured benefit.

The **memory** concern is separate and real, and is handled independently:
`TokenEmbedder.MAX_PROJECTION_BYTES` (512 MB) makes `forward` fall back to the
per-type loop when `[rows, tokens, n_types, d_model]` would exceed it. No measured
configuration reaches it; it is a valve, not a path.

Not taken: the gathered-weight `bmm`/`einsum` alternative. It would help CPU, but
CPU is now off by default, and materialising per-token weights is *larger* than
the temporary it replaces. Recorded as a follow-up if CPU inference ever matters.

### P1 — cache not completely self-invalidating — **accepted, fixed; and the fix's own claim was then wrong (see below)**

The class-level claim was false, exactly as described. `fuse()` now records the
autograd `_version` of all 27 copied parameters, and `forward` drops the cache if
any changed. That covers optimizer steps in eval mode and EMA/SWA
`copy_`/`lerp_`. `fuse()` additionally raises if called while training.

**Correction (2026-07-27, second review).** This response then claimed the guard
covered "direct `.data` writes — every in-place path", and that was wrong.
`parameter.data.add_(x)` does **not** bump `parameter._version`: unlike
`.detach()`, which shares the version counter, every `.data` access mints a fresh
one over the same storage. The reviewer demonstrated a retained cache and a stale
output error of ~1.0. Verified here directly:

```
p = torch.nn.Parameter(torch.zeros(3))
with torch.no_grad(): p.add_(1.0)   # _version 0 -> 1   (caught)
p.data.add_(1.0)                    # _version 1 -> 1   (NOT caught)
p.detach().add_(1.0)                # _version 1 -> 2   (caught)
```

Two changes rather than one:

* **Strengthened where it is cheap.** The snapshot now records
  `(version, data_ptr)` per parameter, which additionally catches *rebinding* —
  `parameter.data = other` swaps storage while leaving the counter untouched,
  so that path was silently uncovered too. 54 pointer/integer reads, no sync.
* **Narrowed where it cannot be.** In-place writes routed through `.data` are
  invisible to any counter-based guard, and the only thing that would see them is
  comparing parameter values — the reads and the device sync that fusing exists to
  avoid. So the contract is now explicit: **mutating a fused module through
  `.data` is unsupported; call `unfuse()` after such a write.** The completeness
  claim is withdrawn from the docstring.

`test_fused_cache_invalidation_contract` pins all four cases, including asserting
that a `.data` write is *not* detected — so a torch change or a stronger guard
fails the test loudly instead of quietly widening what the docs claim. In this
codebase nothing writes through `.data`: training builds its own model, and
`Evaluator` fuses a model it then only reads.

Also worth stating plainly: the version guard shipped in the first response with
**no test of its own** — `train()`, `load_state_dict()` and `_apply()` had tests,
the new guard did not. That is how a false completeness claim survived to review.

### P1 — the "no decision changed" gate did not prove that — **accepted, fixed; claim refined**

Correct, and the refinement matters. Replaced with a full paired-trajectory
comparison (`assert_records_identical`) that splits the two things I had
conflated:

* **discrete decisions asserted exactly equal** — action, legal mask, visits,
  sims, Gumbel top-k, chance log, winner, scores, final fingerprint;
* **recorded continuous targets compared within tolerance**, because they inherit
  the network drift, with the observed maximum reported rather than assumed.

Run on the **production checkpoint on CUDA** with the real search (128-sim full
budgets, `top_k=16`, `force=True`), 12 games / **836 moves**:

* every discrete decision identical — **zero action changes**;
* max policy-target drift **7.7e-05**, max root-value drift **1.4e-06**.

So the original claim was right in substance and wrong in what it had evidence
for. "No search decision changed" now holds on 836 real moves; "records are
identical" was never true and is no longer implied — policy targets move at 1e-5,
which is far below the noise a training target already carries.

The vectorised gather, separately, is **bit-identical** on records (0.0 drift).

### P1 — `SlotBudget` is not leak-free — **accepted, both leaks fixed**

1. Confirmed. `release_activation` decremented `budget_held` without
   `budget.give_back()`, so a failed `GameSlot::new` stranded a token. It now
   takes the budget and returns the borrowed token.
2. Confirmed, and fixed rather than documented: `donate_reservation_if_finished`
   hands a shard's reservation back once its queue is empty and its last slot
   retires, so usable concurrency no longer shrinks as shards finish.

Four Rust unit tests cover the accounting directly
(`self_play::budget_tests`) — the paths are not reachable from Python.

**Third leak, found by the second review and now fixed.** The first response
documented rather than fixed the error path, on the reasoning that "the run is
aborting anyway". That reasoning was wrong about the mechanism: shards run under
`std::thread::scope` and a failing shard's error is only collected after the
scope joins, so its siblings keep working at reduced concurrency for the rest of
the run — and `cancel_all` restored none of the three quantities (`budget_held`,
the shard reservation, the global `live` count).

`cancel_all` is replaced by `abort(&budget)`, which cancels pending work and then
performs exactly the accounting `retire` does for each active slot — one
`on_retire` and one `release_activation` — before donating the now-dead
reservation. It is idempotent, and all 17 error paths route through it. Two Rust
tests cover it: full restoration plus idempotency, and the invariant that matters
(after one shard aborts, a sibling can reach the *full* budget). Rust suite now
15 tests.

Still true, and now the only accepted residue: a shard that panics rather than
returning `Err` unwinds without running `abort`. Closing that would take an RAII
guard holding `&SlotBudget` in the pool; the pool is per-shard and dropped at the
scope boundary, so the exposure is a panicking run only.

### P1 — the headline is not an end-to-end production measurement — **accepted**

Correct. The 1,633 → 3,250 figure is the F4 benchmark configuration, not a Phase D
iteration, and the differences listed (16 slots, randomised 64–128 budgets, draft
prior, ~15% curriculum-bot games in eight groups) are all real. The plan header
now says so explicitly and states that the **relative** improvement is what should
transfer.

**Update (2026-07-27): the `_generate_iteration_rust` A/B has now been run**, so
this finding is closed with a measurement rather than a caveat.
`f4_phase_d_ab.py` calls `_generate_iteration_rust` with exactly the jobs
`generate_iteration` builds — 16 slots, randomised 64–128 budgets, annealed draft
prior, curriculum-bot grouping — and toggles the two features at the `phase_d`
module boundary, leaving production code untouched. 64 games (55 neural + 9 bot
in 6 groups), three repetitions, arms interleaved `A B B A A B` so drift on a
laptop GPU cannot be read as an effect:

| arm | seconds (3 runs) | median | games/hour |
|---|---|---|---|
| before | 170.7 / 168.5 / 169.3 | 169.3 | 1,361 |
| after | 89.4 / 88.1 / 91.3 | 89.4 | **2,578** |

**1.89×**, against 1.99× in the benchmark configuration. Within-arm spread 1.3%
and 3.6%; the ranges do not overlap. All six runs produced **identical discrete
trajectories** — winner, chained trajectory digest, final digest, action
sequence, per-move sim counts — which is the sanity check that the arms did the
same work rather than less of it.

Two corrections to how that check was first written and reported (second review):

* It is **not** "byte-identical records". The fingerprint deliberately excludes
  policy targets and root values, which drift at ~1e-5 under the fused embedder
  (§8 P1 above), and several other record fields. Record-level equivalence is
  gated separately by `assert_records_identical`; this is the discrete-trajectory
  check, and the report and the JSON key now say so.
* The cross-arm comparison **could false-positive**: it took one arbitrary
  fingerprint per arm, so an arm that disagreed with *itself* across repetitions
  could still report agreement. It now unions every run's fingerprint and
  requires exactly one distinct value, exiting non-zero otherwise. The recorded
  run satisfies the stricter rule without a re-run: both arms were internally
  consistent (`self_consistent_within_arm` true for each, i.e. one fingerprint
  apiece) *and* the arms agreed, which together imply a single distinct
  fingerprint across all six runs. The benchmark was not re-timed.

The shortfall is in the expected direction and its magnitude is not decomposed:
16 slots instead of 32 and ~14% of games in bot groups too small to fill the pool
both dilute a per-batch saving, but this run does not separate the two.

The observation that bot groups average ~9 games and so cannot fill 16 slots is a
finding about *this work*: the Phase 1 pool cannot combine them because the Python
wrapper applies one bot configuration per call. Recorded as the per-job bot
routing follow-up.

### P2 — "launch-bound" wording too strong — **accepted**

The tool was corrected but the narrative was not. Now stated as **fixed
launch/synchronisation overhead dominates**, with an explicit note that the
evidence does not separate host dispatch from device-side launch latency or fixed
device work, and that the host≈device equality cannot separate them either.

### P2 — round-robin rationale — **accepted**

"Milder than a reseed" bounds the perturbation magnitude; it does not establish
equal strength or absence of bias, and a reseed changing Gumbel keys is exactly
why its disagreement is larger. Reworded, and the reasons for leaving it off are
now the ones that hold: two runs inside noise, unmeasured on the fully live path,
Phase 4 may change it, fixtures anchored to the blocked order. Recorded as a
Phase 4 arm.

### Not adopted

* **Empty-row sentinel logit** in the padded softmax. The reviewer confirms the
  current code is safe for all callers and offers this defensively. It would add a
  write on every batch to remove NaNs that are provably never read, and the
  "provably never read" property is already gated
  (`test_rows_with_no_legal_actions_do_not_poison_the_batch`). Declined as a cost
  with no failure mode behind it — happy to add it if a future caller ever indexes
  the padded matrix directly.

## 9. Sign-offs — status after two reviews

| # | Request | R1 | R2 | Now |
|---|---|---|---|---|
| 1 | Cache invalidation complete | **No** | **No** — `.data` writes bypass `_version` | Claim **withdrawn, not re-asserted**: guard strengthened with `data_ptr` (catches rebinding), `.data` in-place writes declared unsupported, contract pinned by `test_fused_cache_invalidation_contract`. **Sign-off now sought on the narrowed contract, not on completeness.** |
| 2 | ~2e-6 drift needs no strength gate | **Conditional** | not revisited | Gate added and run: 836 moves, **zero** decision changes, targets drift 7.7e-05; the gate's `None`-parity hole (R2) is closed. **Condition believed met — please confirm.** |
| 3 | `SlotBudget` leak-free | **No** | **No** — `cancel_all` strands everything | Third leak fixed: `abort(&budget)` restores tokens, reservation and `live`; 17 call sites; 2 new Rust tests. **Panic-path residue remains and is stated, so this is "leak-free on every `Err` path", not unconditionally. Re-requested on that wording.** |
| 4 | Padded softmax safe for every caller | **Yes** | — | No change; sentinel-logit suggestion declined with reasons (§8). |
| 5 | Leave round-robin off | **Yes, provisionally** | — | Rationale reworded; recorded as a Phase 4 arm. Agreed. |
| 6 | Phase D A/B validates the headline | — | "credible" | 1.89×; equivalence check tightened after R2 (union across runs, non-zero exit). |

Still unreviewed, both introduced by the first response:

* the **device gate** (`fusion_is_profitable`) — is CUDA-vs-CPU the right axis,
  given larger CUDA models measured neutral rather than negative?
* the **projection budget** fallback — a second numerical path selected by batch
  size. It is unreachable in every measured configuration, but it does mean
  results could in principle depend on batch shape at extreme sizes.

**Pattern worth naming across both rounds.** Three of the four R2 findings are the
same failure: a *guard* was asserted to be complete without a test that could
have falsified it — the version guard shipped untested, the A/B's equivalence
check compared one sample per arm, and `assert_records_identical` skipped absent
targets. The measurements themselves have held up under both reviews; the claims
about what the checks prove have not. Every fix in this round adds the test that
would have caught the overstatement.
