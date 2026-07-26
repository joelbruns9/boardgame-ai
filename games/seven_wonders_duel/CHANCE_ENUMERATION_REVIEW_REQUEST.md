# Review request — chance-enumeration capping (fixed-support edges)

Companion to `CHANCE_ENUMERATION_PLAN.md`, which is the measurement and decision
**record**: why forced root expansion was worth attacking, what was measured, and
which designs were withdrawn. This document is the **review brief** — what code
changed, which invariants must hold, and where a second pair of eyes is worth
most.

**The feature is off by default** (`cheap_double_reveal_offsets = 0`). Nothing in
generation, evaluation, or the advisor behaves differently until someone sets it,
so this is a review of machinery and of the case for turning it on — not of a
live behaviour change.

## 1. Scope

| Step | What | Commit | Primary files |
|---|---|---|---|
| 1 | `fixed_support` edge class: closed support, shared selection rule | `d3ee15f` | `search.py`, `tree.rs`, `tree_resumable.rs` |
| 2 | Balanced `n*X` double-reveal support + `double_reveal_offsets` | `d3ee15f` | `search.py`, `chance.rs`, `tree.rs`, `tree_resumable.rs`, `lib.rs` |
| 3 | `cheap_double_reveal_offsets`: cheap generation moves only | `055115f` | `self_play.rs`, `lib.rs`, `phase_d.py` |
| 4 | Approximation-quality measurement | `941f232` | `chance_cap_quality.py` |
| — | Mixed-back construction withdrawn | `bed527b` | `search.py`, `chance.rs` |
| 5 | Throughput A/B | `f91d7a7` | `f4_throughput_bench.py` |
| — | Review brief | `9e1af56` | this document |
| — | Review response: guards, per-seat routing, corrected measurement | *this change* | `search.py`, `tree_resumable.rs`, `self_play.rs`, `lib.rs`, `phase_d.py`, `chance_cap_quality.py` |

Python: `search.py` (`_Edge.fixed_support`, `close_fixed_support`,
`fixed_support_index`, `double_reveal_offset_seed`, `distinct_offsets`,
`balanced_double_reveal_chains`, `_closed_child`, `_force_expand_root`,
`closed_root_exact_value`).

Rust: `chance.rs` (`mix64`, `double_reveal_offset_seed`, `distinct_offsets`,
`balanced_double_reveal_chains`), `tree.rs` (`fixed_support_index`,
`Edge::fixed_support`, `closed_child`, `force_expand_root`),
`tree_resumable.rs` (`Edge::close_fixed_support`, `closed_child`,
`materialize_forced_root`), `self_play.rs` (`cheap_offsets`).

## 2. What it does (design)

Forced root expansion materialises and evaluates **every** enumerable chance
child of every root edge before simulation 1 — one network row each. Measured, it
is the majority of network rows in generation, and it is a *fixed* per-root cost,
so it hurts most at the small budgets 75% of self-play moves use.

The change replaces the exhaustive support of **one** edge kind with a smaller,
exactly-weighted, **stratified** one:

* **Which edges.** Only a pure double card-reveal whose two slots share a back —
  54.5% of all forced children. Single reveals, Great Library and its products,
  wonder flips, every multi-kind chain, and **different-back** double reveals all
  stay exhaustive.
* **The support.** `n` unseen cards, `X` offsets: stratify on the first reveal
  (each card leads exactly one stratum), take seconds by cyclic block
  `names[(i + 1 + t) % n]` over `X` offsets `t` drawn from `1..n-1`. Every card
  appears exactly `X` times in each position, self-pairs are unreachable, weights
  are `1/(n*X)` so mass is exactly 1. Falls back to exhaustive when `X >= n-1`.
* **Offsets** come from a domain-separated FNV-1a seed (search seed + chance
  signature + reveal pools, **not** the action index) on a **private** RNG, so
  edges sharing a signature share their support (common random numbers) and the
  main search stream is untouched.
* **A capped edge is CLOSED** (`fixed_support`): later descents sample only among
  retained children by their re-normalised weights. This is the load-bearing
  part. Ordinary descent samples the COMPLETE distribution and appends whatever
  observable key it cannot find, carrying that outcome's original probability —
  on a truncated edge that pushes mass past 1 and `q_p0` then weights a
  super-unit sum with nothing raising.
* **Cheap moves only.** Both self-play drivers gate on the `full` flag they
  already compute, so a full-search move always passes 0.

## 3. What is already gated (please don't re-verify by hand)

* **Cross-language bit-identity of the capped tree.**
  `test_rust_engine_equiv.py::test_balanced_double_reveal_support_matches_python_and_the_resumable_path`
  compares Python, the scalar Rust oracle, and the resumable searcher at
  `X in {1,2,3}` × 2 seeds × 6 games on full canonical digests. A seed-derived
  support that diverged by one element would still look valid, so structural
  checks were not considered enough.
* **No-growth / unit-mass under descent**, both languages:
  `test_search.py::test_fixed_support_edge_never_grows_and_keeps_unit_mass`
  (5,000 direct draws + 200 full descents) and
  `test_f4_boundary.py::test_f4_r0_fixed_support_edges_never_grow_and_keep_unit_mass`
  (parses the Rust digest; includes a legacy-sampled contrast arm that *does*
  grow).
* **Construction properties**: exact per-position balance, distinctness, no
  self-pair, mass, fallbacks, offset-seed separation, subset-uniformity of the
  offset draw, mean-unbiasedness against a fixed leaf oracle, common random
  numbers across edges sharing a signature (`test_search.py`).
* **Full-move targets unchanged**: at `full_search_fraction = 1.0`, self-play
  records are byte-identical at `X in {1,2,3}` versus `X = 0`; at 0.0 they differ
  (`test_f4_boundary.py`).
* **The selection rule** is one shared function with an identical golden table in
  `test_search.py` and `tree.rs::fixed_support_tests`.

* **The guards from the 2026-07-26 review** (§4.1): exhaustive expansion and
  forced re-entry both refuse a closed edge without mutating it, and support mass
  is validated before materialisation.
* **Per-seat arena routing** and the `fixed_support_edges` runtime witness
  (`test_f4_boundary.py`).

Full suite: 464 tests green. `cargo test --lib`: 8 green. No new clippy warnings
beyond two `type_complexity` matching the existing `enumerate_chains` signature.

## 4. Focus areas (highest value)

### 4.1 Closure completeness -- **gap found and fixed 2026-07-26**

Production descent was closed correctly, but two non-descent paths were not:

* `expand_exhaustive` appended every omitted outcome to a closed edge, corrupting
  its unit mass in place before `closed_root_exact_value` could refuse the tree.
  It now raises on `fixed_support`.
* Re-entering forced expansion with a different seed appended members of a
  *second* support and only failed at the closing mass check, leaving the tree
  mutated if the exception were caught. Forced expansion now refuses an
  already-closed edge before touching anything, in Python and Rust.
* Support mass is now validated **before** any child is materialised, in both
  languages, so an evaluator failure part-way through cannot leave an edge whose
  mass is neither 1 nor recoverable.

Gated by `test_search.py::test_exhaustive_expansion_refuses_to_complete_a_closed_support`,
`::test_forced_expansion_refuses_to_re_enter_a_closed_edge`, and
`::test_forced_expansion_validates_mass_before_materializing` -- each asserts the
tree is *unmutated* after the refusal, not merely that it raised.

### 4.2 RNG contract

Offsets are drawn on a private `PortableRng`/`Rng` seeded by FNV-1a over
(domain tag, search seed, chance signature, both pools), so they never advance
the search stream. **But** a fixed-support descent draws one `next_float` where
an uncapped edge draws two `randrange`s, so the downstream stream legitimately
diverges between `X=0` and `X=2`. Please confirm nothing depends on cross-config
stream identity — buffer replay and `chance_log` use the *game* RNG rather than
the search RNG, which is the reason I believe this is safe, but I would like that
checked rather than assumed.

Related: is per-signature the right **common-random-numbers** granularity? The
seed includes the reveal slots, so three actions on the same slot (build /
discard / wonder) share a support, but two actions uncovering *different* slots
with the same pools do not. Sharing more would cancel more comparison noise.

### 4.3 Gumbel amplification of the Q error -- **WRONG, retracted 2026-07-26**

This section originally argued that sigma multiplies a Q error by
`(c_visit + max_visits) * c_scale` ~ 5-7, so 1.9e-4 lands at ~1e-3 in logit
space, three orders below log-prior differences.

**That omitted the min-max normalisation and was wrong by three orders of
magnitude.** Sigma is `scale * (q - low) / span`, so a perturbation is *divided
by the completed-Q span*; where the root's actions are nearly tied, the same
absolute Q error saturates the whole sigma range. It also couples one edge's
error into every action's sigma, and moving an endpoint rescales all of them.

Measured (600 comparisons, `chance_cap_quality.py` now reports it):

| | control (re-seed) | X = 2 | X = 3 |
|---|---:|---:|---:|
| max Δsigma, mean | 1.393 | 0.457 | 0.382 |
| max Δsigma, max | 6.40 | 6.40 | 6.40 |
| normalisation endpoint moved | 25.8% | 9.2% | 7.7% |
| Δsigma exceeds top-2 logit margin | 37.5% | 15.5% | 14.8% |

So the logit effect is ~0.46 on a scale whose logit margins average 1.8, and it
exceeds the decisive margin in 15.5% of positions. No bound on the logit
perturbation can be stated. What survives is the *comparison*: every figure is
smaller than re-seeding the same search produces.

### 4.4 Training-target exposure, and whether `TARGET_VERSION` needs a bump

Cheap moves record `policy_excluded`, and `dataset.py` (`is_fast_search_move`,
`has_policy=not move.policy_excluded`) means **no cheap-move policy target is
ever trained on**. So capping cheap moves cannot corrupt a policy target at all;
it changes which action gets played, hence the trajectory and the value labels
attached to those positions. Combined with full-move byte-identity, I concluded
no `TARGET_VERSION` bump is needed. Is that reasoning complete?

### 4.5 Can the flag leak into evaluation? -- **two fixes 2026-07-26**

* **Runtime witness, not source inspection.** `fixed_support_edges` is now a
  search and self-play metric counting root edges left holding an approximate
  support. Exact-chance runs assert it is zero; `chance_cap_quality.py` asserts
  zero on its uncapped arm, and `test_f4_boundary.py` asserts it on the arena
  configuration. This replaces the source-text check.
* **The Python generation backend silently ignored the flag.** `_search_move`
  builds its `SearchConfig` without force expansion at all, so
  `cheap_double_reveal_offsets > 0` with `generation_backend="python"` produced
  uncapped games while the run manifest recorded a cap. `PhaseDConfig.validate`
  now rejects that combination outright. (Separately and pre-existing: the Python
  generator also ignores `force_root_chance`. Not fixed here -- it is a different
  bug with its own blast radius, and it deserves its own change.)

The gate/arena path still deliberately receives no offsets, so the arena is
routed instead through new per-seat settings -- see §4.7.

### 4.6 Are the measurements strong enough to justify turning it on?

**Both soft spots were confirmed by the review and both are now fixed:**

* **One seed for every root** (Level A and B). Level A now enumerates the
  estimator's *entire* realization space -- `C(n-1, X)` <= 45 subsets, 12,261 at
  X=2 across 529 edges -- so no seed choice can flatter it. Result: signed bias
  ~1e-19 (exact unbiasedness to float precision, previously only bounded by a
  4,000-seed test), MAE over all draws 2.1e-4 against the single seed's 2.0e-4,
  and **zero** terminal children dropped by any support of any edge. Level B now
  derives a distinct seed per root, shared by both arms, with 3 paired replicates
  (600 comparisons).
* **The vacuous survivor metric.** "Actions with visits > 0" cannot see an
  elimination -- anything visited in round 1 keeps visits forever, which is why it
  read 1.000 everywhere. The searcher now exports the candidate set at every
  sequential-halving reduction. Capping changes an elimination in **1.8%** of
  positions (X=2) against the re-seed control's **59.2%**.
* **The X=3 KL anomaly** (mean above p95) was a heavy tail, now reported: p99
  2.71, max 6.28. Source identified -- roots whose completed-Q span is ~0, where
  min-max normalisation is degenerate. The control shows it worse.

**Still open:** the control re-draws the Gumbel keys as well as the chance stream,
so it perturbs strictly more than capping does and remains a *loose* upper bound
on the noise floor. A tighter control needs a diagnostic chance stream, or matched
RNG consumption, so that only the chance draw differs. Not built.

**Throughput is geometry-dependent**: +7.4% at slots 16 (8 paired reps, CI
[+5.5, +9.4]) and +14.5% at slots 24. And see §4.8: the batches are so far below
the cap that fixing the scheduler is worth more than any further approximation.

### 4.7 Per-seat routing for the strength gate (added 2026-07-26)

The production gate deliberately passes no offsets, which left the proposed offset
arena unroutable. `cheap_double_reveal_offsets_p0/_p1` now exist on
`self_play_many_mock/_net/_flat_net`, mirroring `age_deal_samples_by_player`, so
capped can play exhaustive on one shared checkpoint with seats swapped. Gated by
`test_f4_boundary.py::test_cheap_double_reveal_offsets_route_per_seat_for_the_arena`,
which also asserts the two seats are actually distinguished and that a one-sided
setting is rejected.

Two distinct questions, now separated in the plan (*Future tests 2* and *2c*):
**search** strength (same net, capped vs exhaustive) and **learning** strength
(train with capped generation, evaluate both checkpoints exhaustively). The second
is the shipping gate; the first is a cheap precondition.

### 4.8 Scheduler utilisation is the bigger prize (added 2026-07-26)

Confirmed in code: `self_play.rs::run_many` creates one slot per job and never
replaces a finished one, and `phase_d.py` submits fixed chunks of `rust_slots`
games, so concurrency falls monotonically as each chunk drains -- 18-20% of
slot-cycles in the Step 5 runs were completed-and-idle. Mean batch is 27 rows at
slots 16 (38.5 at slots 24) against a `global_batch_cap` of **256**, i.e. ~1.6
rows per slot.

A rolling active-game pool plus a slot sweep past 24 would turn row savings into
*eliminated* batches instead of thinner ones, at exact search semantics. Recorded
as *Future test 2b* and recommended ahead of any further approximation scope.

## 5. Known limitations / out of scope

* **Marginal coverage, not pair coverage.** Every hidden card appears in every
  revealed slot, but the specific worst *pair* is retained only at the rate a
  random subset of that size would (16/33/51% for X=1/2/3). Zero terminal
  children were dropped in 529 measured edges, and the mean worst-case shortfall
  is 0.0037 at X=2, which is why this was accepted rather than fixed.
* **Not attempted, deliberately:** product chains (`reveal + Great Library`,
  17.2% of forced children), capping full moves, the one-ply Great Library
  resolver, progressive widening, advisor deep-node growth. Each has an entry in
  the plan's *Future tests*.
* **Different-back double reveals** were built and then withdrawn (`bed527b`);
  the reasoning is a standalone section in the plan. The block-stratified-offset
  variant that would recover that 2.9% unbiasedly is described but not built.
* **Arena strength is unmeasured.** It is the only remaining gate before
  `cheap_double_reveal_offsets = 2` should ship on.

## 6. Running the gates

```bash
# whole suite (~10 min) and the Rust unit tests
python -m pytest games/seven_wonders_duel -q
cd games/seven_wonders_duel/seven_wonders_rust && cargo test --lib

# the capping gates alone
python -m pytest games/seven_wonders_duel -q -k \
  "fixed_support or balanced or capped or offset or cheap_double or chance_cap"

# regenerate the quality measurement (~4 min, needs CUDA + a checkpoint)
python -m games.seven_wonders_duel.chance_cap_quality \
  --buffer  games/seven_wonders_duel/runs/laptop_training_10h_02/buffer_final.jsonl \
  --checkpoint games/seven_wonders_duel/runs/laptop_training_10h_02/checkpoints/latest.pt \
  --games 150 --roots 200 --offsets 1 2 3

# regenerate one throughput arm (~6 min each; one directory per X)
python -m games.seven_wonders_duel.f4_throughput_bench --mode rust \
  --exploratory-leaf1 --checkpoint <ckpt> --output <dir>/x2 \
  --games 16 --repetitions 8 --warmup-games 2 --slots 16 \
  --global-batch-cap 256 --max-inflight-batches 1 --scheduler-workers 1 \
  --python-workers 1 --cheap-double-reveal-offsets 2
```

## 7. Sign-offs -- requested, and the 2026-07-26 resolution

| # | Claim | Outcome |
|---|---|---|
| 1 | Closure is complete | **Was not.** `expand_exhaustive` and forced re-entry could corrupt a closed edge; both now refuse before mutating, with tests asserting the tree is unmutated (§4.1). |
| 2 | RNG contract holds | **Signed off.** Replay and `chance_log` use the game RNG; cross-config search-stream identity is not required for correctness. It does contaminate the A/B, which §4.6 leaves open. |
| 3 | No `TARGET_VERSION` bump | **Signed off.** Cheap moves are dropped from training by default and full-search target semantics are unchanged; the trajectory distribution moves but no retained label's definition does. |
| 4 | The flag cannot reach evaluation | **Signed off with two additions**: a runtime `fixed_support_edges` metric replaces source inspection, and the Python generation backend -- which silently ignored the flag -- is now rejected outright (§4.5). |
| 5 | Enable X = 2 | **Not yet, and not X = 2.** After the sigma and seed corrections, X = 3 dominates X = 2 on every quality metric while throughput cannot separate them, so X = 3 is the conservative default. Capping stays off until the same-net arena and the paired training A/B pass (*Future tests 2, 2c*). |

Reviewer's closing judgement, recorded because it sets the priority: *"The core
support estimator and closed-edge implementation look sound. The best next work is
strengthening the measurement and filling GPU batches -- not expanding
approximation scope."* The measurement work is done; the batch work is §4.8.
