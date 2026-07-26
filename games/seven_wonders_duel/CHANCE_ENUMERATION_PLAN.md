# Chance-node enumeration: measurement, design, and build plan

**Status:** measured; design revised after review; **Steps 1-2 built
(2026-07-25)**, Steps 3-5 outstanding.
**Date:** 2026-07-25.

## The problem

`_force_expand_root` materializes *every* enumerable chance outcome of *every*
root edge, evaluates each with the network, and weights them by exact
probability. The rationale is catastrophe coverage: a rare instant-loss reveal
is guaranteed probability-weighted rather than possibly unsampled.

The suspicion was that this is expensive. It is much more expensive than
expected: **forced expansion, not search, is the majority of neural evaluations
in self-play generation.**

This document records what was measured, what the numbers mean, the design that
follows, and what to build.

---

## Vocabulary

| term | meaning |
|---|---|
| **node** | a board state in the search tree; the **root** is the position being decided |
| **root edge** | one legal action from the root (12-20 per position) |
| **child** | the state after taking an edge. Deterministic action → 1 child. Chance action → **one child per outcome** |
| **afterstate** | the state after an action but before chance resolves (ZeusAI's term) |
| **forced expansion** | materializing and evaluating every enumerable chance child of every root edge, once, before simulation 1 |

Two clarifications that caused confusion and are easy to re-derive wrongly:

* **Root expansion is ONE network call**, not one per legal move. It returns one
  value for the root plus a **prior** per legal action. Move values come from
  search, not from that call.
* **Forced-expansion children ARE individual network calls** — one row each,
  batched into one GPU call but 110 rows of work for a 110-outcome edge.
* Forced expansion covers **only chance-triggering edges, one ply deep**.
  Deterministic edges are skipped (`if not edge.specs: continue`) and evaluated
  lazily. Everything below the root is built by sampling.

---

## Measurements

### Method

**Level 0 (analytic, no network).** Replay 150 real games from
`runs/laptop_training_10h_02/buffer_final.jsonl`. At every searched decision,
take `chance_signature` for each legal action and count
`len(enumerate_chains(...))`. Exact child counts at zero evaluation cost, over
the *real* self-play position distribution. Bot moves (`sims == 0`) excluded.
**9,830 roots analysed.**

**Level 1 (measured network rows).** Run `search_many_flat_net` with
`force=True` on 24 mid-game positions using `latest.pt`, and read the existing
counters: `nn_work.forced_rows`, `nn_work.forced_cache_hits`,
`metrics.requested`, `metrics.unique`.

> **Caveat on Level 1:** its positions come from the test harness's random-play
> trajectories, which average **92** forced children/root against Level 0's
> **40** on real self-play. Level 1's *percentages* are therefore inflated;
> Level 0 is the authoritative distribution. Rescaled shares are given below.

### Level 0 — children per root

| | cheap (≤24 sims) | full (≥25 sims) |
|---|---:|---:|
| roots | 7,211 | 2,619 |
| forced children/root, mean | 40.0 | 35.8 |
| median | **0** | **0** |
| p90 | 100 | 96 |
| max | 2,710 | 2,710 |
| worst single edge, mean | 16.8 | 14.7 |
| worst single edge, max | 900 | 900 |
| roots with a 2-reveal edge | 14.6% | 14.2% |

**The median root has zero forced children.** This is not a uniform tax; it is
cheap most of the time and occasionally enormous.

Chance kinds seen: `card_reveal` 18,965, `great_library_draw` 1,519,
`wonder_group_reveal` 140. Card reveals per edge cap at **k=2**.

### Level 0 — where the cost actually lives

Total forced children across all roots: **382,246**.

| combination on one action | edges | children | share | mean/edge |
|---|---:|---:|---:|---:|
| card_reveal + card_reveal | 3,784 | 208,244 | **54.5%** | 55.0 |
| card_reveal | 10,989 | 86,072 | 22.5% | 7.8 |
| cc + great_library_draw | 98 | 50,970 | 13.3% | 520.1 |
| card_reveal + great_library_draw | 212 | 15,070 | 3.9% | 71.1 |
| great_library_draw | 1,209 | 12,090 | 3.2% | 10.0 |
| wonder_group_reveal | 140 | 9,800 | 2.6% | 70.0 |

**The cost is not in the extremes.** The 900-child edges are 13.3% of the total
across 98 edges. The bulk is the unremarkable double reveal: 3,784 edges
averaging 55 children, **54.5%** of everything.

### Level 0 — cap curve (per-edge cap, all chain kinds)

| cap | children retained |
|---:|---:|
| 16 | 43.4% |
| 24 | 51.1% |
| 32 | 57.4% |
| 48 | 69.4% |
| 64 | 78.8% |
| 110 | 89.5% |
| none | 100% |

A cap at 110 ("just kill the extremes") saves 10%. A cap at **24 roughly halves
total forced work.**

### Level 1 — measured network rows

| age_deal_samples | sims | forced rows | net (− cache hits) | unique leaves | net forced share |
|---:|---:|---:|---:|---:|---:|
| 0 | 20 | 2,081 | 1,894 | 293 | 86.6% |
| 0 | 64 | 2,081 | 1,667 | 1,122 | 59.8% |
| 32 | 20 | 2,209 | 1,989 | 260 | 88.4% |
| 32 | 64 | 2,209 | 1,715 | 1,042 | 62.2% |

Two findings:

* **`forced_rows` is identical at 20 and 64 sims** (2,081 either way). Forced
  expansion is a **fixed per-root cost**, so its share is worst exactly where
  the budget is smallest — and ~75% of self-play moves run `cheap_sims` 16-24.
* **`age_deal_samples=32` adds only 128 of 2,209 rows (~6%).** The cost is
  overwhelmingly chance-chain enumeration, not AgeDeal.

Rescaled to Level 0's real distribution: roughly **65% of network rows at cheap
sims** and **~35% at 64 sims**.

### Network calls per move

```
NN calls per move  =  1 (root)  +  F (forced children)  +  S (leaf evals ≤ sims)
```

`S < sims` because descents that hit terminal nodes or already-cached forced
children cost nothing (494 cache hits at 64 sims across 24 positions).

Scaled to the real distribution (F ≈ 40):

| move type | nominal sims | forced | leaf evals | actual NN calls | vs nominal |
|---|---:|---:|---:|---:|---:|
| cheap | 20 | ~40 | ~11 | **~52** | 2.6× |
| full | 64 | ~40 | ~43 | **~84** | 1.3× |
| weighted (75/25) | — | — | — | **~60** | — |

---

## Explaining the extremes

The 900-child edge decomposes as:

```
900 = 90 (two card reveals, 11×10 → here 10×9) × 10 (great library, C(5,3))
action: CONSTRUCT_WONDER slot (4,3) — the Great Library
```

Building the Great Library **from the pyramid** triggers two chance things at
once: taking the card uncovers two face-down cards, *and* the wonder triggers
its draw. `enumerate_chains` walks specs recursively, so counts **multiply**.

2,710 is one root's sum across 18 legal actions — a couple of ~900
wonder-construct edges plus a dozen 110s.

**Confirmed:** a single reveal maxes at exactly **11** children and a double at
exactly **110 = 11×10**. The unseen-pool bound of 11 holds; the extremes come
from stacking, not from a larger pool.

`wonder_group_reveal` is the **second group of 4 wonders being flipped during
the draft**, not initial setup: C(8,4) = 70, roughly once per game.

---

## Relationship to Gumbel search

A natural worry: does forced expansion defeat the point of Gumbel + sequential
halving, which exists to reduce simulations?

**No — but it erodes it, and the right unit is network calls, not "sims".**

* ZeusAI: **1,000 simulations** per training move.
* This implementation: **~60 network calls** per training move.

Still roughly **17× cheaper per move**. Gumbel bought ~15×; forced expansion
gives back ~2×.

More importantly, the two are **synergistic, not opposed**. The Gumbel policy
target is `softmax(log_prior + σ(completed_Q))`. At 64 sims across 16
candidates each root edge gets ~4 visits — a 4-sample estimate of a
distribution over up to 110 outcomes, i.e. near-noise. Forced expansion
replaces that with the exact expectation. Gumbel's premise is a small
simulation budget, and a small budget is precisely the regime where
chance-heavy edges are worst estimated.

The genuine tension is **allocation at a fixed network budget**: given ~84
calls for a full move, is it better to spend 40 on exact root chance and 43 on
simulations, or 0 on chance and ~84 on simulations? Exact chance buys root
accuracy; more simulations buy depth. See *Future tests*.

### ZeusAI comparison

| | ZeusAI | this implementation |
|---|---|---|
| scope | **every** afterstate in the tree | **root only** |
| limit | 11 children | none (exhaustive) |
| at the limit | continue randomly into an existing child | n/a |
| probabilities | implicit via sampling | **exact**, mass asserted = 1.0 |
| widening | limit grows with visits (non-training games) | none |
| training sims | 1,000 | 16-24 cheap / 64-128 full |
| non-training sims | 5,000 | advisor, unbounded |

Their fallback ("continue randomly with one of the existing child states") can
miss a hidden card entirely and re-weights toward whichever children happened
to be created first. The balanced-coverage design below is strictly better at
equal budget.

---

## Review corrections (2026-07-25)

The first draft of the design below was reviewed and three blocking defects were
found and **verified against the code**. They are recorded here because each one
is a trap that is easy to walk back into.

### [P0, confirmed] A capped edge breaks the probability-mass invariant

Force expansion asserts `mass == 1.0` **once**, at expansion time
(`tree_resumable.rs` `materialize_forced_root`). Nothing re-checks it later.

But ordinary descent still samples from the **complete** chance distribution in
`_closed_child` / `closed_child`: it draws an outcome, looks for a child with
that observable key, and **appends a new child carrying its original
probability** if none is found. With exhaustive expansion every key already
exists, so this never fires. Under a cap, a descent that samples an omitted
outcome silently appends it, the edge mass exceeds 1, and `q_p0` then computes a
probability-weighted sum over more than unit mass with no error raised.

**A capped edge must therefore have closed support**: later descents sample only
among retained children, re-normalised. Rust already has exactly this mechanism
for AgeDeal -- `paired_sampled` edges (renamed `fixed_support` in Step 1) sample among existing children by
cumulative empirical probability and never create new ones. The cap should reuse
that pattern rather than invent one.

Three edge classes must be kept distinct:

| class | support | later descent |
|---|---|---|
| `probability_weighted` | exhaustive, exact | always finds an existing child |
| **approximate fixed-support** (new) | retained subset, re-normalised | samples **only** among retained |
| ordinary sampled | grows lazily | may materialise a new child |

**Mandatory test:** thousands of descents through a capped root edge must
materialise no new child and leave mass at exactly 1.

### [P0, confirmed] The Great Library collapse is NOT a drop-in identity

The mathematics is right -- E[max over the drawn 3-subset] = .6*V1 + .3*V2 +
.1*V3 -- but it does not describe the states this tree actually holds.

Verified in `engine.py`: `CHOOSE_UNUSED_PROGRESS` draws the 3-token subset,
emits the `GREAT_LIBRARY_DRAW` event, then calls `_set_pending_if_options` with
`PendingChoiceKind.CHOOSE_UNUSED_PROGRESS` and `consume_all_options=True`. The
pick is resolved **later**, by a separate action through
`resolve_pending_choice`.

So:

* The 10 forced children today are **pending-choice states**, and the network
  evaluates those. The proposed 5 "token taken" states are **one ply deeper**.
* A singleton token is **not a valid Great Library chance outcome** -- the engine
  requires a 3-token subset.
* Three synthetic children would not match the 10 real chance keys, so later
  sampling could not find them, producing new children -- the P0 above, again.
* The resumable path caches both value **and priors** for the real
  pending-choice node; priors from a post-choice state cannot be attached to it.
* Search can currently refine the 3-option decision by descending into it. A
  frozen ranking removes that.

It may still be a worthwhile **one-ply Great Library resolver**, possibly
stronger than what exists. But it is a **search-semantics change**, not a
zero-cost identity, and it invalidates the claim that full-move policy targets
stay bit-identical.

**Removed from the first implementation.** See *Future tests*.

### [P0, confirmed] "A per-edge cap across all chain kinds" was not a design

The balanced first-reveal / cyclic-second construction is defined **only for a
two-card reveal**. It is undefined for:

* `wonder_group_reveal` -- 4-subsets, no first-reveal stratum, `n` has no meaning;
* Great Library alone -- 10 unordered 3-subsets, not ordered pairs;
* **product chains** -- capping the reveal component of `reveal + GL` to 22 and
  then enumerating GL still yields 220 rows. A per-component cap is not a
  per-edge cap.

Consequently the cap curve above (`min(outcome_count, cap)`) **does not describe
the balanced design** and overstates its saving. Corrected numbers below.

### Other corrections accepted

* **Unbiasedness needs a randomness contract.** A deterministic position-derived
  offset is reproducible but not *conditionally* unbiased. The offset must be
  uniform over directed-pair distances `1..n-1` (naive `mod n` can select
  `(card, same card)`), drawn from a **domain-separated** seed built from the
  search seed, the canonical public-state fingerprint, and the chance signature
  -- and it must **not consume the main search RNG stream**. That exact bug class
  already bit the PUCT root port, where drawing Gumbel keys from the shared RNG
  offset every downstream chance sample. Edges sharing a signature should
  deliberately share support: common random numbers reduce variance when
  comparing actions.
* **Rows are not GPU calls.** The `NN calls per move` model counts *rows*.
  Production coalesces rows across games and chunks forced requests, so cutting
  rows does not translate linearly into games/hour and may push batches below
  the utilisation knee. All throughput figures here are **row-count upper
  bounds** until measured.
* **"Bit-identical targets" was too broad.** Correct claim: for a *fixed*
  full-search root state and seed, the target is unchanged. Earlier cheap moves
  still shift the trajectory, so the *set* of recorded positions differs -- fine,
  and not a version bump. The Great Library collapse *would* change full-move
  outputs, so that conclusion never covered it.
* **The catastrophe guarantee is narrower than claimed.** Balanced coverage
  gives **exact marginal coverage** -- every hidden card appears in each revealed
  slot. It does *not* cover every dangerous **pair interaction**. Much better
  than IID sampling; not a general catastrophe guarantee.
* **Terminal-child filtering measures at nothing.** Proposed as an exact,
  semantics-free saving: skip forced children that are already terminal, whose
  values are known exactly. Verified correct -- `materialize_forced_root` pushes
  *every* child with no terminal check. But measured on the replay corpus it is
  **8 terminal children out of 163,712 (0.005%)**. Correct in principle, not
  worth building.

---

## Design (revised)

### Scope: pure double card-reveal edges, on cheap moves only

Everything else stays exhaustive in v1 -- single reveals, Great Library and its
products, wonder flips, and every chain with more than one chance component.
Those are the cases where the balanced construction is undefined or where the
semantics are unsettled.

### Construction

For a pure double-reveal edge with `n` unseen cards and `X` offsets:

```
directed pairs = n * X            (exhaustive once X >= n-1)
weight         = 1 / (n * X)      -> mass exactly 1
```

* **Stratify on the first reveal**: every hidden card appears in first position
  exactly once. This is the marginal-coverage guarantee.
* **Second reveal by cyclic block**: stratum *i* takes seconds `i + d` for `X`
  offsets `d` drawn from `1..n-1`, so every card also appears exactly `X` times
  in second position.
* **Offsets from a domain-separated seed** (search seed + public-state
  fingerprint + chance signature), never from the search RNG stream.
* **Closed support**: the edge is marked approximate fixed-support; later
  descents sample only among the `n * X` retained children.

The parameter is `cheap_double_reveal_offsets = X`, not a generic cap -- the
retained row count is `n * X`, which is what the design actually produces.

### Corrected savings

Measured on the replay corpus (382,246 forced children total; pure double-reveal
edges are 3,784 edges / 208,244 children, of which 156,110 sit on cheap roots;
`n` ranges 5-11):

| X | retained on cheap pure-cc | total forced children | vs current |
|---:|---:|---:|---:|
| 1 | 21,698 of 156,110 (13.9%) | 247,834 | 64.8% |
| **2** | **43,396 (27.8%)** | **269,532** | **70.5%** |
| 3 | 65,094 (41.7%) | 291,230 | 76.2% |
| 4 | 86,792 (55.6%) | 312,928 | 81.9% |

Translated to network **rows** per 100 self-play moves:

| configuration | rows/100 moves | saving |
|---|---:|---:|
| current | 6,550 | -- |
| X = 2 (recommended) | ~5,380 | **~18%** |
| X = 1 | ~5,155 | ~21% |

**This is materially less than the first draft claimed (26-35%)**, for two
reasons the review identified: that projection applied `min(count, cap)` to
*all* edge kinds including full moves, and `min(c, 24)` is more aggressive than
`n * X`. The honest figure for this safe scope is **~18% fewer rows**, and that
is a row-count upper bound on throughput, not a measured speedup.

The larger savings remain available -- they require either capping full moves or
handling product chains -- but both carry costs this scope deliberately avoids.

### Settings

| setting | value |
|---|---|
| `cheap_double_reveal_offsets` | **2** (flag; sweep 1/2/3) |
| full moves | **exhaustive, unchanged** |
| single reveals, GL, GL products, wonder flip | **exhaustive, unchanged** |
| AgeDeal samples | 32, unchanged |
| advisor | exhaustive, unchanged |

---

## Build plan (revised)

### Step 1 -- the approximate fixed-support edge class -- **DONE 2026-07-25**

Infrastructure before any approximation. No behaviour change yet.

1. Add the third edge class alongside `probability_weighted` and ordinary
   sampled, modelled on Rust's existing `paired_sampled`.
2. `closed_child` in both languages: on a fixed-support edge, sample among
   retained children by their stored weights; **never** materialise a new one.
3. Tests: thousands of descents through a fixed-support edge materialise no new
   child and leave mass at exactly 1; `q_p0` matches a hand-computed weighted
   sum.

**What was built.** `fixed_support` is now a first-class edge flag in all three
searchers: `search.py::_Edge`, `tree.rs::Edge` (scalar) and
`tree_resumable.rs::Edge`. Rust's `paired_sampled` was **renamed** into it
rather than left alongside -- the paired AgeDeal sampler always was a member of
this class, and one flag keeps the closed-support rule in a single place.

* Selection rule shared by both languages: `search.py::fixed_support_index` /
  `tree::fixed_support_index` -- first child whose cumulative weight exceeds one
  `next_float()` draw, last child absorbing the float residue. Pinned by an
  identical golden table on both sides.
* Closing an edge goes through `_Edge.close_fixed_support` /
  `Edge::close_fixed_support`, which refuses any support that is not
  re-normalised to mass 1 -- so a truncation cannot be half-applied.
* `closed_root_exact_value` now **refuses** to descend a fixed-support edge:
  its children are a subset, so an "exact" value through one would be a lie.
* Nothing sets the flag except the AgeDeal sampler yet, so search output is
  unchanged; the whole 7WD suite (440 tests) and every Rust/Python equivalence
  gate stay green.
* Tests: `test_search.py` -- golden table, 5,000 direct draws plus 200 full
  descents materialise no child and hold mass at exactly 1, draws follow the
  re-normalised weights, `q_p0` equals a hand-computed weighted sum (uniform and
  skewed), partial mass is rejected, exact value refuses the edge.
  `test_f4_boundary.py::test_f4_r0_fixed_support_edges_never_grow_and_keep_unit_mass`
  parses the Rust tree digest and asserts the same no-growth/unit-mass invariant
  on the Rust path, against a legacy-sampled contrast run that does grow.
  `tree.rs::fixed_support_tests` covers the selection rule under `cargo test`.

### Step 2 -- balanced double-reveal support -- **DONE 2026-07-25**

1. Build the `n * X` directed-pair support with the cyclic construction and the
   domain-separated offset seed.
2. Rust mirror; the parameter must reach both scalar and resumable paths.
3. Tests: every card once in first position and `X` times in second; offsets
   drawn from `1..n-1` (never a self-pair); mass exactly 1 at several `n`, `X`;
   exhaustive once `X >= n-1`; **support identical for two edges sharing a
   chance signature** (common random numbers); Python/Rust equivalence.
4. Unbiasedness as a statistical test: mean root value over many random offsets
   converges to the exhaustive value on a fixed mock evaluator.

**What was built.** `search.py::balanced_double_reveal_chains` and its Rust
mirror `chance.rs::balanced_double_reveal_chains` return the balanced support in
the same `(outcomes, probability, key)` shape as `enumerate_chains`, or `None`
when the construction does not apply (wrong signature, `X = 0`, or an `X` that
would retain the whole space). Force expansion consumes it in all three
searchers -- `_force_expand_root`, `tree.rs::force_expand_root`,
`tree_resumable.rs::materialize_forced_root` -- and closes the edge through the
Step 1 `close_fixed_support`.

* **Parameter.** `double_reveal_offsets` on `SearchConfig` in both languages,
  default 0 = exhaustive, plumbed to the scalar and resumable Rust paths and
  exposed on `closed_search`, `closed_search_resumable` and
  `search_many_flat_net`. Step 3 adds the driver-level cheap-move gating
  (`cheap_double_reveal_offsets`); self-play still passes 0.
* **Different backs, which the design did not cover.** The two uncovered slots
  can carry different backs (Age III mixes guilds in). Those pools are disjoint,
  so no exclusion applies and `n * (n-1)` is the wrong count. Handled as the
  natural generalisation: the cycle runs over the second pool with distances
  `0..n2-1`. First-position coverage is still exact; second-position incidence
  is even to within one, since `n1` strata spread over `n2` residues.
* **Offset draw.** FNV-1a over (domain tag, search seed, chance signature,
  both reveal pools) seeds a private `PortableRng` / `Rng`; a partial
  Fisher-Yates takes `X` distinct offsets, returned ascending so the support does
  not depend on draw order. The action index is deliberately NOT in the seed, so
  edges sharing a signature share their support.
* **Cross-language equivalence is bit-exact, not just structural.** A seed-derived
  support could diverge by one element and still look valid, so the gate compares
  whole trees: `test_rust_engine_equiv.py::test_balanced_double_reveal_support_matches_python_and_the_resumable_path`
  runs Python, the scalar Rust oracle and the resumable searcher at `X` in
  {1,2,3} and asserts identical canonical digests.
* **Other tests.** `test_search.py` -- balance/mass/distinctness/no-self-pair at
  several `X` (and the mixed-back variant), fallback cases, offset-seed
  separation (search seed / position / signature), subset-uniformity of
  `distinct_offsets`, mean-unbiasedness through the real construction against a
  fixed leaf oracle, and force expansion capping *only* pure double reveals,
  closing them, and holding under 2,000 descents.
  `test_rust_engine_equiv.py::test_balanced_double_reveal_support_shrinks_the_forced_tree`
  asserts the capped support is a strict subset and that every other edge keeps
  exactly its exhaustive children. Full suite green (452 tests).
* **Saving, sanity check only.** 180 mid-game roots from random-play
  trajectories, total forced children: X=1 50.0%, **X=2 57.8%**, X=3 65.5% of
  exhaustive. That is *better* than the 70.5% projected in this document, and
  should not be believed as the self-play figure: this corpus averages ~92 forced
  children per root against real self-play's 40, exactly the Level-1 caveat
  above. It confirms the cap bites; Step 5 measures throughput for real.

### Step 3 -- apply to cheap moves only

1. `cheap_double_reveal_offsets` on the configs; hook where the driver already
   chooses cheap vs full sims.
2. **Test that a full-search move at a fixed state and seed produces a
   bit-identical `policy_target`.** This is the claim that keeps buffers
   compatible; assert it, do not assume it.

### Step 4 -- validate approximation quality, not just throughput

Mean-unbiasedness is not sufficient. On the replay corpus, compare capped
against exhaustive root output:

* Q mean-absolute error and p95 error, broken down by edge size and signature;
* selected-action disagreement rate;
* Gumbel top-k survivor disagreement;
* policy-target KL divergence;
* missed terminal / catastrophe outcomes.

### Step 5 -- measure real throughput

Rows are not GPU calls. Record games/hour, global batches, rows per batch, GPU
utilisation, padded tokens, forced-phase wall time and scheduler idle time,
before and after.

---

## Future tests

### 1. Equal-network-budget allocation (highest value)

At a fixed budget, is it better to spend on exact root chance or on depth?
`--full-sims 64` with force against `--full-sims 128` with
`--no-force-root-chance`, matched at ~84 rows/move, 200-400 games through
`--eval-search-mode puct`. Nothing measured so far bears on this, and it decides
strength per unit compute. ~90 minutes.

### 2. Offset sweep

X in {1, 2, 3} against arena strength and games/hour, using the Step 4 quality
metrics as the leading indicator.

### 3. One-ply Great Library resolver (was "collapse")

Re-scoped from a free identity to a **search-quality experiment**. Resolving the
3-option choice one ply early may be stronger than evaluating the pending-choice
state, and halves those rows. But it changes full-move outputs, so it needs a
`TARGET_VERSION` bump, its own story for the cached value+priors on the
pending-choice node, and an arena A/B against exhaustive.

### 4. Product chains and a genuine per-edge cap

A generic cap must operate on the **whole Cartesian chain** with balanced
incidence across every component -- an orthogonal / systematic product design.
Capping one component and multiplying by the rest is not a per-edge cap. This is
what unlocks the `reveal + GL` cases (17.2% of children).

### 5. Capping full moves

Worth roughly another 9 points. A capped edge is a **stratified** estimate over
correctly-weighted outcomes, far closer to exact than the ~6 random visits an
unforced edge would get, so it may be nearly free. Requires a `TARGET_VERSION`
bump. Run only after a working baseline exists.

### 6. Advisor deep-search child growth

Deep afterstates have **no** cap; `closed_child` accumulates one child per
distinct sampled outcome, bounded only by visits. Self-limiting at 64 sims. At
advisor scale (thousands of simulations on a narrow principal variation) a deep
node could approach its full outcome count. Measure child counts per depth with
the same `nn_work` counters; add visit-scaled widening if they balloon.

### 7. Progressive widening

ZeusAI grows the afterstate limit with visits in non-training games. A
visit-scaled limit spends children only where search concentrates -- a better
shape than either a flat cap or exhaustive, and cheap once the fixed-support
class exists.

### 8. Re-measure on the trained distribution

Level 0 used run 02's buffer. Chance-edge frequency depends on how the net
plays. Re-run on run 03's buffer and confirm the sizing still holds.
