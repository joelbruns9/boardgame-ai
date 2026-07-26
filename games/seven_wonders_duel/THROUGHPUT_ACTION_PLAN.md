# Throughput action plan (7WD generation)

**Date:** 2026-07-26. **Status:** nothing built yet; ordering and gates agreed.
**Context:** `CHANCE_ENUMERATION_PLAN.md` Step 5 measured where the time goes
while A/B-ing chance capping. Those numbers say the remaining wins are in the
scheduler, not in more approximation.

## The finding this all rests on

**A GPU forward costs ~7.3 ms per batch almost regardless of how many rows are
in it.**

| slots | rows/batch | ms/batch | µs/row |
|---|---:|---:|---:|
| 16 | 27.0 | 7.27 | 270 |
| 24 | 38.5 | 7.46 | 194 |

Rows per batch rose 43%; per-batch cost rose 2.6%. The forward is latency-bound:
~7.2 ms fixed + ~16 µs/row. We pay full price for batches 11% full (27 of a
256 cap).

Wall-clock split at slots 16 (44.3 s per 16-game repetition):

| | share | scales with |
|---|---:|---|
| GPU forward | 58% | **batch count** |
| Python adapter overhead (inside `pyo3_call`, beyond its sub-timers) | 6.5% | **batch count** |
| `gather_d2h` | 15% | rows (72 µs/row) |
| `encode_pack` | 8% | rows (39 µs/row) |
| `pyo3_tensor` + `h2d` | 6% | rows (28 µs/row) |
| Rust tree + chance | 2% | tree work |

**≈65% per-batch, ≈29% per-row.** The taxonomy predicts the capping result
(18.5% fewer rows × 29% ≈ 5.4%, plus tree savings; measured 6.9% wall) which is
why it is trusted to rank the work. It also sets a ceiling: **every
row-reduction idea is competing for a share of 29%.** Batch-count reduction
attacks 65%.

Two structural causes, both confirmed in code:

1. `self_play.rs::run_many` creates **one slot per job and never replaces a
   finished one**; `phase_d.py` submits fixed chunks of `rust_slots` games. Each
   chunk's concurrency decays monotonically to zero — 18-20% of slot-cycles in
   the Step 5 runs were completed-and-idle.
2. `leaf_batch=1` in production, so each game contributes **one row per wave**.
   16 slots therefore produce ~16-row leaf batches.

---

## Item 1 — rolling active-game pool

**Why first:** it is structural, and the drain tail does not merely cost
throughput, it *contaminates* every measurement we would use to tune anything
else. "Mean batch 27" is an average over a concurrency profile decaying from 16
live games to 1, so a slots/inflight sweep run today measures a blend of
well-fed and starving regimes and its optimum need not be the real one.

**Development.** Accept `jobs > slots` in `run_many` / `run_many_pipelined`:
keep an active window of `slots` games and start the next queued game the moment
one completes. `phase_d.py` then stops chunking and passes the whole group.

**The risk to design around:** `run_many` returns records positionally and sets
`metrics.games = slots.len()`. A pool needs explicit slot↔job bookkeeping to
preserve input order. Per-game determinism is *not* at risk — each game keeps
its own seed and its own search session, independent of what else is in flight.

**Tests.**

* Records are **byte-identical** to the chunked path for the same seeds. This is
  the real gate: a pool may change only the schedule, never a game's result.
* Every existing F4 gate stays green (digest equality, `self_play_many_mock`
  determinism, replay).
* Active-slot count stays at `min(slots, jobs_remaining)` until the queue drains.
* `scheduler_idle_slot_cycles` falls to ~0 outside the final window.

**Expected:** removes the 18-20% idle slot-cycles and holds rows/batch at its
well-fed value instead of averaging down.

---

## Item 2 — cheap-only leaf batching

**New evidence (2026-07-26).** Leaf batching is **already bit-identical** to
`leaf_batch=1` at cheap budgets. 144 comparisons over 6 positions, checking
action, visits, policy target and the full canonical digest:

| sims | `leaf_batch` 2/4/8/16 vs 1 |
|---|---|
| **20** (cheap) | **48/48 bit-identical** |
| 64 | policy target differs (action occasionally) |
| 128 | visits and action differ |

The mechanism is `per_action`, recomputed after every halving reduction:

```
per_action = max((sims - completed) / (rounds_remaining * candidates), 1)
```

| | round 1 | after 1st reduction |
|---|---|---|
| sims=20 | 16 × 1 | (20−16)/(3×8) = 0 → **1** |
| sims=64 | 16 × 1 | (64−16)/(3×8) = **2** |
| sims=128 | **2** | 4 |

While `per_action == 1`, consecutive simulations go to *different* root
candidates and waves are cut at round boundaries, so every in-flight leaf sits
in a different subtree and the WU virtual loss never perturbs a selection that
matters. At `per_action >= 2` two in-flight leaves share a subtree and the
approximation begins to bias the deeper descent. **The known leaf-batch quality
loss is confined to the deep halving rounds; it is not inherent to batching.**

**Development.** One rule: **cut the wave before any simulation that would enter
a subtree already in flight.** At cheap budgets the condition never triggers, so
`leaf_batch` up to `top_k` is exact; in deeper rounds the wave shrinks
automatically — the taper, falling out of the invariant rather than being
configured.

**Tests.**

* Promote the ad-hoc experiment to a gate: `leaf_batch ∈ {2,4,8,16}` is
  bit-identical to 1 at cheap budgets, force on and off.
* **Stronger claim to check after the rule lands:** *every* budget becomes
  bit-identical, because the rule forces waves of 1 wherever
  `per_action >= 2`. If that holds, leaf batching needs no quality gate at all.
* Assert the invariant directly (no two in-flight simulations share a root
  candidate), not just its consequence.
* `leaf_waves` per cheap search drops ~`leaf_batch`× (~20 → ~2).

**Expected:** each game contributes 16 rows per wave instead of 1, so 16 slots
reach the 256-row cap. This is probably the single largest lever, and it is
exact.

---

## Item 3 — slots sweep 32 / 48 / 64

Only after 1 and 2, so it measures a scheduler that stays fed and produces wide
waves. GPU headroom is large (peak 51 MB); the unknown is host memory per live
game tree, which the sweep finds. Use `f4_throughput_bench.py --mode rust` (one
output directory per configuration) and `f4_frontier.py` to analyse.

---

## Decision gates and conditional work

Re-run the paired A/B (8 repetitions, slots 16) after items 1+2, then branch:

| observation | conclusion | next |
|---|---|---|
| rows/batch > 100 **and** games/s up | producer fixed | Item 3, then re-measure capping |
| rows/batch still < 50 | the **coalescer** is the limit, not the producer | instrument `EvalWorker` batch assembly: are waves arriving but not merging? |
| GPU busy > 0.85 | GPU-bound at last | stop; further work needs a faster net or lower precision |
| padding > 20% | wide batches now mixing token counts | token-count bucketing |
| cpu_util ~0.5 **and** GPU busy < 0.75 | host-side serialisation | `max_inflight_batches` and worker sweep — but first explain the historical pathology where 7 python workers *dropped* mean batch to 5.2 rows and halved games/s |
| per-row share now dominates | expected by construction: cutting batch count raises the per-row share | attack `gather_d2h` (72 µs/row): GPU-side fused gather then a single D2H, pinned memory, narrower dtypes |

**Optional, only if its gate opens:**

* **Full-move leaf batching (round-robin reorder).** If full moves remain a
  batch-count bottleneck after Item 2, reorder the schedule round-robin within a
  round so no two in-flight leaves ever share a subtree. Sequential halving does
  not care about order within a round, so it is sound — but it permutes RNG
  consumption and therefore changes full-move outputs, so it needs an arena
  justification of its own. Not worth it unless the measurement asks.
* **Chance capping (`cheap_double_reveal_offsets`, X=3).** Re-measure *after* the
  batch work: it buys +7.4% (slots 16) / +14.5% (slots 24) today, but it only
  touches the per-row 29%, and it makes batches thinner, so its value moves in
  both directions. Still gated on arena strength independently.

**Explicitly not next:** product chains (17.2% of forced children) and full-move
capping. Both are more approximation risk chasing the same 29%, while Items 1-3
are exact.

---

## Target

Baseline (slots 16, uncapped, pinned laptop configuration): **1,301 games/hour**,
27 rows/batch, 58% GPU busy, 18% idle slot-cycles.

Success for Items 1-3: **rows/batch ≥ 100, idle slot-cycles < 5%, GPU busy
> 0.75**, at byte-identical game records. If the ~16 µs/row marginal cost holds,
that implies roughly **1.7×** on games/hour before any approximation is
considered — the estimate the sweep is there to confirm or refute.
