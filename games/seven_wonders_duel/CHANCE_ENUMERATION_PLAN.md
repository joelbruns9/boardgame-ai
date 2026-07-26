# Chance-node enumeration: measurement, design, and build plan

**Status:** measured, designed, not yet implemented.
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

## Design

### Decision 1 — Great Library collapse (exact, no accuracy cost)

The Great Library draws 3 of the 5 offboard progress tokens; the player keeps
the best one. So the outcome is determined entirely by **which token is best in
the drawn subset**:

```
P(rank-i token is best) = C(5−i, 2) / C(5,3)
  rank 1 → 6/10 = 60%
  rank 2 → 3/10 = 30%
  rank 3 → 1/10 = 10%
  rank 4, 5 → 0   (can never be best of three)
```

Therefore, with V₁ ≥ V₂ ≥ V₃ the sorted values of *holding* each token:

> **E[max over the drawn 3-subset] = 0.6·V₁ + 0.3·V₂ + 0.1·V₃**

This is an **exact identity, not an approximation**. It replaces 10 subset
evaluations with **5 token evaluations** — and on the stacked wonder case,
900 → 450.

Design points:

* **Hardcode 60/30/10.** The Great Library is the only route to the offboard
  five (tokens acquired in play come from the board's face-up five via science
  pairs), and it is a single wonder built at most once. The math cannot change.
* **Assert the offboard pool is exactly 5** and fail loudly otherwise. If some
  future change touches that pool, a loud failure beats silently wrong
  probabilities weighting a training target.
* **Do not collapse to a single value.** Keep **three children** with
  probabilities .6/.3/.1 pointing at the top-3 token states, preserving tree
  structure, `q_p0` probability weighting, and the `mass == 1.0` check.
* Known limitation: the ranking freezes at expansion. If search would later
  refine which token is best, the collapse cannot. Same character as the
  existing one-shot evaluation of each subset.

### Decision 2 — the wonder draft is NOT collapsible

The identity above works because **one of three is taken**, which makes it a
max. In the wonder draft **all four revealed wonders get drafted** between both
players, so each of the 70 subsets leads to a genuinely different game. There
is no max to collapse, and no static ranking recovers that — independent of the
correlation issues (military wonders, Mausoleum/discard, Great Library/tokens).

At 2.6% of children and once per game, it is not worth special handling. The
per-edge cap covers it for free.

> Keep distinct: a static wonder ranking as a **policy prior** (e.g. seeding
> ZeusAI's published preferences via `--draft-prior-iterations`) is sound and
> useful. A static ranking to **collapse chance outcomes** is not.

### Decision 3 — per-edge cap with balanced coverage

For an edge whose chain exceeds cap `Y`, with `n` distinct first-reveal
candidates:

```
X = max(1, Y // n)          second-reveals per stratum
chains = n × X              (exhaustive when n·(n−1) ≤ Y)
weight = 1 / (n × X)        → mass still sums to exactly 1.0
```

* **Stratify on the first reveal** — every hidden card appears in first
  position, which is the coverage guarantee. The first reveal is also the more
  decision-relevant card.
* **Second reveals via a cyclic block**: stratum *i* takes seconds
  `{i+1, …, i+X} mod n`, so every card also appears exactly **X** times in
  second position. Total **2X reveals per card**, perfectly balanced.
* **Systematic sampling with a random start**: seed the cyclic offset from the
  node's chance signature. Balanced at every node *and* unbiased in expectation
  over the offset, because which pairs get chosen is independent of their
  values. Deterministic and reproducible, since the seed comes from the
  position.

Why balanced beats random: it removes variance from single-card effects (the
"trap card sampled 7 times" problem) at the cost of sampling pair interactions
in a structured pattern — which the random start de-biases.

**A cap ≥ 11 makes single reveals exhaustive automatically** (they max at 11),
so no special case is needed for them.

| cap Y | X at n=11 | chains | reveals per card |
|---:|---:|---:|---:|
| 16 | 1 | 11 | 2 |
| 24 | 2 | 22 | 4 |
| 32 | 2 | 22 | 4 |

### Decision 4 — cap cheap moves only; full moves stay exhaustive

**Cheap-search moves record no policy target.** They only advance the game. So
exact chance on a cheap move buys a marginally better *move*, not a better
*label*, while costing the same fixed ~40 forced children.

This is the same principle already accepted for simulations via playout cap
randomization: **spend accuracy where it becomes a label, not where it only
advances the game.**

Per 100 self-play moves:

| configuration | NN calls | saving |
|---|---:|---:|
| current (exhaustive everywhere) | 6,550 | — |
| **cap cheap only, full exhaustive** | **4,855** | **26%** |
| cap everywhere | 4,290 | 35% |

**Cheap-only captures ~74% of the available saving** and — decisively —
**leaves every recorded policy target bit-identical**. No `TARGET_VERSION`
bump, no buffer incompatibility, and if the run underperforms the labels are
not in the suspect list. Capping full moves would put target quality under
suspicion for the whole run, and subtle label corruption is the hardest failure
to diagnose after the fact.

### Settings

| setting | value | note |
|---|---|---|
| Great Library collapse | **everywhere** | exact; no accuracy cost on either move type |
| cap on cheap moves | **Y = 24** (X=2 at n=11) | flag, sweepable 16/24/32 |
| cap on full moves | **none** (exhaustive) | preserves label definition |
| cap scope | **per-edge, all chain kinds** | covers stacked GL-wonder and wonder flip with one rule |
| AgeDeal samples | **32**, unchanged | only ~6% of forced rows |
| advisor | **exhaustive**, unchanged | differentiation vs ZeusAI's cap-11 |

Expected: **~1.35× more games per hour** in generation, converting directly
into training data.

---

## Build plan

Each step is independently testable and independently revertable. Steps 1 and 2
touch search semantics, so both need Python/Rust equivalence to the F3
standard.

### Step 1 — Great Library collapse

1. `pool.py`: add the ranked-outcome helper beside `enumerate_great_library`.
   Assert `len(offboard_progress) == 5`; raise loudly otherwise.
2. `search.py`: in `_force_expand_root`, when a chain contains
   `GREAT_LIBRARY_DRAW`, evaluate the 5 "token taken" states, sort by
   **actor-relative** value, and emit 3 children at .6/.3/.1 instead of 10.
3. Rust: mirror in `chance.rs` / `tree.rs` force-expansion.
4. Tests:
   - weights sum to 1.0 and match C(5−i,2)/C(5,3);
   - the loud failure fires when the pool is not 5;
   - collapsed expectation equals the exhaustive 10-subset expectation on a
     fixed mock evaluator (this is the exactness claim — it must be an equality
     test, not an approximation test);
   - Python/Rust equivalence on real positions.
5. Re-run Level 0 to confirm GL-containing combinations drop ~2×.

### Step 2 — per-edge cap with balanced coverage

1. `search.py`: add the capped chain builder — stratify first reveal, cyclic
   block for seconds, offset seeded from the chance signature.
2. Preserve `probability_weighted` semantics and the `mass == 1.0` assertion.
3. Rust: mirror in `chance.rs`; the cap must be part of `SearchConfig` so it
   reaches both the scalar and resumable paths.
4. Tests:
   - every hidden card appears exactly once in first position and X times in
     second;
   - mass sums to exactly 1.0 at several n and Y;
   - exhaustive when `n·(n−1) ≤ Y` (the cap must not perturb small edges);
   - single reveals untouched at any Y ≥ 11;
   - unbiasedness: mean over random starts ≈ exhaustive expectation on a mock
     evaluator;
   - Python/Rust equivalence.

### Step 3 — tie the cap to the cheap/full split

1. `SelfPlayConfig` / `PhaseDConfig`: `chance_cap_cheap` (default 24) and
   `chance_cap_full` (default 0 = exhaustive).
2. Hook at the point the self-play driver already chooses cheap vs full sims —
   the chance policy rides along with the sim budget being selected there.
3. CLI: `--chance-cap-cheap`, `--chance-cap-full`.
4. Test that a full-search move produces a **bit-identical** `policy_target` to
   the current code. This is the claim that lets buffers stay compatible; it
   must be asserted, not assumed.

### Step 4 — validate

1. Re-run Level 0 and Level 1 and confirm the predicted reductions.
2. Measure games/hour on a short generation run, before and after.
3. Confirm `samples_per_new_position` is unchanged (the cap changes throughput,
   not the sample-per-position accounting).

---

## Future tests

### 1. Equal-network-budget allocation (highest value)

The central unanswered question: at a **fixed** network-call budget, is it
better to spend on exact root chance or on depth?

Compare `--full-sims 64` + force-root-chance against `--full-sims 128` +
`--no-force-root-chance`, matched at ~84 calls/move, 200-400 games through the
**fixed PUCT evaluation path** (`--eval-search-mode puct`). This directly
answers "what maximizes strength per unit compute", which is the question that
matters for a world-best player. ~90 minutes.

### 2. Cap sweep

Y ∈ {16, 24, 32} on cheap moves, arena strength plus games/hour. Establishes
whether the accuracy/throughput curve is flat (take 16) or steep (take 32).

### 3. Capping full moves too

Worth ~9 additional points of throughput. A capped edge is still a
**stratified** estimate over Y correctly-weighted outcomes — far closer to
exact than the ~6 random visits an uncapped-but-unforced edge would get — so
this may be nearly free. Run it as an A/B **after** a working baseline exists,
never folded into a first run where it would be confounded. Requires a
`TARGET_VERSION` bump.

### 4. Advisor deep-search child growth

Deep afterstates have **no** cap; `closed_child` accumulates one child per
distinct sampled outcome, bounded only by visit count. At 64 sims that is
self-limiting. At advisor scale (thousands of simulations concentrated on a
narrow principal variation) a deep node could approach its full outcome count —
up to 110 children each needing an evaluation, exactly on the line that matters
most.

Measure child counts per depth at a few thousand sims on real positions using
the same `nn_work` counters. If they stay modest, this is ZeusAI-beating
accuracy for free. If they balloon, add visit-scaled widening deeper in the
tree.

### 5. Progressive widening

ZeusAI grows the afterstate limit with visits in non-training games. A fixed
cap pays the same everywhere; a visit-scaled limit spends children only where
search concentrates. Better shape than either a flat cap or exhaustive, and
cheap to add once the cap exists.

### 6. Re-measure on the trained distribution

Level 0 used run 02's buffer. Chance-edge frequency depends on how the net
plays — a stronger net that avoids or seeks pyramid positions differently will
shift the distribution. Re-run Level 0 on run 03's buffer and confirm the cap
is still sized correctly.
