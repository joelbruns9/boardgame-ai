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

Full suite: 460 tests green. `cargo test --lib`: 8 green. No new clippy warnings
beyond two `type_complexity` matching the existing `enumerate_chains` signature.

## 4. Focus areas (highest value)

### 4.1 Closure completeness — the one that can corrupt a tree silently

Every path that appends to `edge.children` must be unable to do so on a
fixed-support edge. Known appenders: `_closed_child` / `closed_child` (scalar and
resumable — all three branch on the flag first), `expand_exhaustive` (test
helper), `materialize_forced_root`, `materialize_paired_age_deals`.

**The question I could not fully answer myself:** can force expansion ever run on
a root that *already holds children*, so that two supports mix? `make_root`
(advisor) and `begin_search_from_root_forced` each expand once per handle today,
and `_force_expand_root` skips keys already present — but that skip is what would
silently merge a stale support into a new one. Is there a resume, retry, or
re-entry path that reaches it twice?

Secondary: `closed_root_exact_value` now *raises* on a fixed-support edge. Is
refusing right, or should it fall back to the sampled branch?

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

### 4.3 Gumbel amplification of the Q error

Capped edge Q enters `completed_q`, which sigma rescales by
`(c_visit + max_visits) * c_scale` ≈ 5–7 at these budgets. A measured 1.9e-4 mean
Q error therefore lands at ~1e-3 in logit space against log-prior differences of
1–3, i.e. ~3 orders down. That arithmetic is the whole basis for "the policy
target barely moves" — please check it rather than trust it.

### 4.4 Training-target exposure, and whether `TARGET_VERSION` needs a bump

Cheap moves record `policy_excluded`, and `dataset.py` (`is_fast_search_move`,
`has_policy=not move.policy_excluded`) means **no cheap-move policy target is
ever trained on**. So capping cheap moves cannot corrupt a policy target at all;
it changes which action gets played, hence the trajectory and the value labels
attached to those positions. Combined with full-move byte-identity, I concluded
no `TARGET_VERSION` bump is needed. Is that reasoning complete?

### 4.5 Can the flag leak into evaluation?

The gate/arena path deliberately does **not** receive it: arena games run
`full_search_fraction = 0.0`, so every arena move takes the *cheap* branch and
passing it there would cap chance in the games meant to measure capping.
`test_phase_d.py` asserts the flag reaches the generation call and not the gate
call, but that test reads source text — a better check is welcome. The advisor
builds `SearchConfig` from `req.options` for `c_puct` and `force_expand` only, so
a request cannot inject offsets; `eval_suite.py` and `f4_strength.py` never set
it. Please confirm there is no fourth path.

### 4.6 Are the measurements strong enough to justify turning it on?

Both soft spots are known and I would rather they be attacked than accepted:

* **Level B's control is loose.** Re-seeding perturbs the Gumbel keys as well as
  the chance stream (top-k identical: 15.5% control vs 100% capped), so "capping
  disagrees less than re-seeding does" is bounded by a floor that is wider than
  the thing being bounded. The clean evidence is the seed-free Level A error.
* **Throughput is geometry-dependent.** +7.4% at the pinned slots-16 geometry
  (8 paired reps, CI [+5.5, +9.4]) and +14.5% at slots 24, because an 18.5% row
  cut mostly *thins* batches rather than removing them (batches −1.8%, GPU
  forward −0.9%). A 3-rep A/B had said +15%; it was noise.

Everything is also conditional on the ~16% of cheap roots that carry a pure
double-reveal edge at all.

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

## 7. Sign-offs requested

1. **Closure is complete** — no reachable path appends a child to, or re-expands,
   a fixed-support edge (§4.1).
2. **The RNG contract holds** and nothing depends on cross-config stream
   identity (§4.2).
3. **No `TARGET_VERSION` bump is needed** given full-move byte-identity plus
   cheap-move policy exclusion (§4.4).
4. **The flag cannot reach evaluation, the gate, or the advisor** (§4.5).
5. **The evidence supports enabling `X = 2`** subject to an arena-strength gate —
   or a statement of what additional measurement would be needed (§4.6).
