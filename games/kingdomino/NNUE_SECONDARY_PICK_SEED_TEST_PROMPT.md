# Implementation prompt — Secondary-pick fragility: systematic overvaluation vs. sampling noise (overnight sim-ladder × 5-seed)

Context: `games/kingdomino/denial_search.py`, its frozen set
`runs/kingdomino/denial_search/signal_positions.jsonl` (50 real midgame positions), and the sweep
harness `games/kingdomino/denial_signal_sweep.py`. Read the memory note
`kingdomino_denial_rescore_verdict` for the full arc. This is a **diagnosis** task designed for an
unattended **~8–9 GPU-hr overnight run**. It does **NOT** retrain, does **NOT** change the label
schema or the search's value semantics, and does **NOT** touch throughput/concurrency code.

## The question (precise)

The net appears to overvalue its **2nd/3rd** picks (advisor fragility concentrates on prior ranks
2–4, median ~0.14; rank-1 median 0.037). The BGA anchor (`bga_denial_anchor.py`) refuted the
**broad, on-path** version — the policy predicts top-30 human picks 76% exactly (median prior 0.82).
Secondary-pick fragility is **off-path** (refutations to moves you consider but reject; real games
never exercise them), so the anchor cannot see it and it stays open. Two indistinguishable causes:

- **(a) Systematic overvaluation** — the net genuinely overrates 2nd/3rd picks because it starves the
  opponent's refutation inside those subtrees. **Persists as sims rise; seed-stable. Learnable.**
- **(b) Sampling noise** — secondary picks receive few sims, so their root Q is a high-variance
  estimate; "fragility" then partly measures MCTS sampling variance + selection bias. **Shrinks as
  sims rise; seed-unstable. NOT learnable by training — only more search fixes it.**

**This test adjudicates (a) vs (b) at and beyond the deployed operating point, and measures whether
the effect changes the move played.** All prior fragility work compared the 8-ply search against a
**32-sim** root (≈ raw prior) — near-tautological. This is the first test at the real deployed sim
count (3200) and above (10000).

## Two design facts this run exploits

1. **The 8-ply tree (`searched_value_actor`) is sim-independent.** It is driven by `pick_plies` and
   `chance_k`, **not** by `root_search_sims`. So once a tree is computed, evaluating `root_Q`
   (`headline_edge`) at *any* number of sim counts is cheap (root searches only, reusing the same
   `searched` reference). We therefore **factor the tree from the root search**: pay for the
   expensive tree a few times, then sweep the cheap `root_Q` over a sim ladder × many seeds.
2. **`chance_k` does not touch the discriminator.** The root open-loop search
   (`advisor_open_loop_search`, `denial_search.py:432`) takes no `chance_k` argument — it does its own
   deck sampling internally. `chance_k` (`_chance_key`, `:378`) governs only the offline 8-ply tree,
   i.e. only the `searched` **reference**. So `chance_k=16` is fixed throughout; do **not** raise it
   hoping to move the verdict — the verdict lives in `root_Q`, which `chance_k` cannot affect. (The
   one place `chance_k` matters is the tie side-probe, Phase 3.)

`SearchConfig.seed` (`:352`) feeds **both** the root-search seed (`_root_seed`, `:385`, → the Rust
search `seed`, `:435`) **and** the chance-node CRN seed (`:378`). The open-loop root search **samples
deck draws**, so at finite sims its per-pick Q is a *sampled* estimate → genuinely seed-dependent.
That seed-dependence, and its shrinkage with sims, is the signal (b) predicts and (a) does not.

## Method — reuse `DenialSearch.search_position` and `denial_signal_sweep.load_frozen_positions` unchanged

Add `games/kingdomino/secondary_pick_seed_test.py`. Do not perturb the validated search. Same
current-best checkpoint (record sha; expect `4bf07b0c…`). Same frozen 50 every phase. Hold tree knobs
locked: `pick_plies=8`, `chance_k=16`, `placement_top_k=2`, `policy_temperature=0.10`,
`tie_tolerance=1e-6`, `uncertainty_z=1.0`, `starved_prior=0.10`.

Seeds: `TREE_SEEDS = {S0,S1,S2}` (3), `ROOT_SEEDS = {S0..S4}` (5, superset). Sim ladder:
`SIMS = {800, 3200, 10000}` (3200 = deployed operating point; 10000 = 4× headroom to see a plateau).

### Phase 0 — equivalence gate (run first, fail loud)

On one fixed position, re-derive each pick's `root_Q` from a standalone
`_root_search(state, seed_override=s, cache_namespace=…)` by replicating `_root_candidates`'
representative selection (`:467–482`: per pick, choose the representative action by
`(group_visits, raw_prior, -idx)` and take its root Q). Assert it is **byte-identical** to the
`headline_edge` inside `search_position`'s `per_pick` at the same seed and same `root_search_sims`.
**Abort the whole run if this fails** — the cheap factoring is only valid when this holds.

### Phase 1 — reference trees (expensive core, ~5–6 GPU-hr)

For each `seed ∈ TREE_SEEDS`, build `SearchConfig(root_search_sims=3200, chance_k=16, seed=seed, …)`
and call `search_position(state)` on all 50 frozen positions, **retaining full `per_pick` rows**
(`pick_domino_id, raw_prior, group_visits, headline_edge, searched_value_actor, policy_target,
fragility`). Stream each position to `runs/kingdomino/denial_search/secondary_seed/tree_seed{seed}.jsonl`
incrementally (bounded memory). This yields, per pick: `searched(d, seed)` at 3 seeds **and**
`root_Q(d, 3200, seed)` at 3 seeds (computed inside the call).

Define the **stable searched reference** `searched_ref(d) = median_{seed∈TREE_SEEDS} searched(d,seed)`.

### Phase 2 — root_Q sim-ladder × 5 seeds (cheap, ~1.5–2.5 GPU-hr)

For each `seed ∈ ROOT_SEEDS` and each `sims ∈ SIMS`: set `config.root_search_sims = sims` and call
`_root_search(state, seed_override=seed, cache_namespace=f"s{sims}_seed{seed}")` on all 50 positions,
re-deriving per-pick `root_Q(d, sims, seed)` via the Phase-0 logic. (The `(3200, S0/S1/S2)` cells may
reuse Phase-1 values.) Stream to `…/root_ladder.jsonl`. The 10000 rung dominates Phase 2 cost;
**if wall-time runs long, drop 10000 → 6400** rather than cutting seeds.

`fragility(d, sims, seed) = root_Q(d, sims, seed) − searched_ref(d)`.

### Phase 3 — tie side-probe (~1 GPU-hr)

On ~15 positions (fixed sample, 1 seed), recompute the tree at `chance_k=32` and compare to
`chance_k=16`: report the count of within-position **bit-identical searched-value ties** at each k,
and whether specific ties dissolve. Ties dissolve → chance-sampling artifact (higher-k labels would
fix). Ties persist → structural (horizon/transposition) → the flip metric's tie-guard is load-bearing
and flips are under-powered on those positions.

## Metrics (track picks by DOMINO ID; classify `d` *secondary* if `searched_ref(d)` ranks ≥ 2)

1. **Sim-ladder monotonicity — PRIMARY (a)/(b) discriminator.** For secondary picks, report median
   (and p90) `fragility` at 800 / 3200 / 10000. **(b): monotone decrease toward ~0 by 10000. (a):
   plateau above 0.** Also report per-secondary-pick the slope `fragility(10000) − fragility(3200)`.
2. **Seed-SD of `root_Q` across the 5 seeds, at each sim count.** For secondary picks, distribution of
   `SD_seed[root_Q(d,sims,·)]`. (b): SD large and shrinking with sims. (a): SD small at 3200 already.
3. **Move-flip, per sim count.** Per position/seed/sims: `root_top = argmax_d root_Q`,
   `search_best = argmax_d searched_ref`. A **flip** = `root_top ≠ search_best` **and**
   `searched_ref(search_best) − searched_ref(root_top) > tie_tolerance` (tie-guard — mandatory, see
   the 44/50 tie degeneracy in the verdict note). `value_at_risk` = that difference.
4. **Flip stability & sim-trend.** Per position/sims, in how many of the 5 seeds does a tie-guarded
   flip occur → buckets 0/5…5/5. Report the count of positions with a **≥4/5 stable flip** at each
   sim count, and how that count **changes across the ladder** (collapses toward 10000 → (b);
   persists → (a)).
5. **Searched-reference stability check.** From Phase 1, per pick report `SD_{TREE_SEEDS}[searched]`
   vs `SD_seed[root_Q@3200]`. The factoring assumes `SD(searched) ≪ SD(root_Q)`; if it does not hold,
   flag it (the fixed reference is then suspect and Phase-2 fragility inherits reference noise).
6. **Tie counts** k=16 vs k=32 from Phase 3.

## Report — `runs/kingdomino/denial_search/secondary_seed_test.json`

Provenance (checkpoint sha, frozen path+hash, TREE_SEEDS/ROOT_SEEDS, SIMS, per-phase config, Phase-0
result, elapsed/GPU-hr per phase). Distributions for metrics 1–2 and 5. The sim-ladder fragility
table (median/p90 per rung). The flip table: per sim count, count by stability bucket (0/5…5/5) and
the ≥4/5-flip positions with `position_index, root_top, search_best, value_at_risk`. The
ladder-trend of the ≥4/5-flip count. Metric-6 tie counts. One-paragraph verdict against routing.

## Pre-registered routing

- **SYSTEMATIC (a) → real, strength-relevant, learnable.** Concretely: secondary-pick fragility
  **plateaus** (median at 10000 ≥ 0.08 and slope `fragility(10000)−fragility(3200)` small), seed-SD of
  secondary `root_Q@3200` is **< 0.05**, **AND ≥ 8/50** positions show a **≥4/5 seed-stable
  tie-guarded flip at 3200 that persists at 10000**, median `value_at_risk ≥ 0.10`. **Route:** build
  the **ply-1 opponent-reply label channel** (the forced search already computes 7-ply-backed
  opponent-reply values and discards them; emitting `denial_policy_target` over a ply-1 node's pick
  edges is zero extra search), then a small pilot retrain (≤2k positions, fine-tune from
  current_best, measure fragility drop on a held-out frozen set). Downweight overrated secondary
  picks; do **not** upweight starved picks (that criterion was inverted).
- **NOISE (b) → not learnable, close this lever.** Secondary fragility falls monotonically toward ~0
  by 10000, **or** the ≥4/5-flip count collapses as sims rise, **or** seed-SD is comparable to the
  fragility mean. **Route:** no curriculum. Report the **sim count at which the flips die** — that is
  the operational fix (raise advisor sims to there), not a retrain. Consistent with the routing
  pressure (banked-best net + queued 7WD).
- **BETWEEN → default to NOISE/close.** A marginal, partly-stable effect does not justify the
  ~100 GPU-hr curriculum against the alternatives. Record magnitude; move on unless the user
  overrides.

## Invariants / scope / caveats

- Same checkpoint (record sha); reserved test split closed; frozen 50 identical across all phases.
  No retrain, no label-schema change, no throughput code.
- **`search_sims` leaks weakly into the tree** via representative-action selection
  (`_root_candidates`, `:467–482`): at 3200/10000 the chosen representative placement per pick may
  differ from the 32-sim validation run, so `searched` here will **not** reproduce `validation.json`
  — expected, not a bug. Since `searched_ref` is computed at 3200 (Phase 1) but `root_Q` is also
  evaluated at 800/10000, the representative at those rungs may differ from the reference's; treat
  `searched_ref` as the pick-level 8-ply value keyed by DOMINO ID (group), which is stable across
  representative choice — do **not** re-pair by representative action index.
- **Circularity note (intentional):** `searched`/`value_at_risk` are in the AZ value head's own units
  (stipulated adequate). This is a net-internal consistency probe; the external, non-circular check
  already exists and is clean (BGA anchor on the policy prior). Do not re-open value-head adequacy.
- The **44/50 bit-identical tie degeneracy** is still unexplained; handled here only by the
  `tie_tolerance` flip-guard plus the Phase-3 k-probe. If most would-be flips are killed by the
  guard, say so — the flip metric is under-powered there.

## Tests

Extend `games/kingdomino/tests/test_denial_search.py` (or add `tests/test_secondary_pick_seed_test.py`):
- **Phase-0 equivalence:** re-derived `root_Q` from standalone `_root_search` equals the
  `headline_edge` inside `search_position`'s `per_pick` at the same seed and `root_search_sims`.
- **Frozen-set seed independence:** the 50 loaded `GameState`s are identical regardless of
  `config.seed` and `root_search_sims` (neither changes *which* positions are scored).
- **Root-Q seed & sim sensitivity:** on one fixed position, `_root_search` with two `seed_override`
  values at the same sims can differ, and at two sim counts can differ, while `public_state_key` is
  unchanged (confirms both discriminator axes are live).
- **Tie-guard:** a synthetic position with `searched_ref(search_best) − searched_ref(root_top) ≤
  tie_tolerance` is **not** counted as a flip.
- **`chance_k` independence of `root_Q`:** changing `chance_k` leaves `_root_search` output unchanged
  on a fixed position/seed/sims (guards the Phase-2 factoring assumption).
