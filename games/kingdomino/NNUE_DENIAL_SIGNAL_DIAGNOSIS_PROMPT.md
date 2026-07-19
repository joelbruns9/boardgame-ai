# Implementation prompt — Denial-search signal diagnosis (is 0/13 a measurement artifact?)

Context: `games/kingdomino/denial_search.py` and its report
`runs/kingdomino/denial_search/validation.json`. Read
`games/kingdomino/NNUE_DENIAL_SEARCH_BUILD_PROMPT.md` (search contract) and
[[kingdomino_denial_curriculum]] for the committed direction. This is a **diagnosis** task: it
determines whether the null denial finding (`high_fragility_starved_picks_upweighted = 0/13`) is a
noisy-measurement artifact that de-noising fixes, or a real absence that blocks the retrain. It does
**NOT** retrain, does **NOT** touch the throughput/concurrency code, and does **NOT** change the
label schema or the search's value semantics beyond the two swept knobs.

## What is (and is not) being tested

- **Stipulated, do NOT re-verify:** the AZ **value head is adequate** — forced search reliably
  recognizes losing positions. The pathology is entrenched weak **policy** priors on
  adversarial/denial lines. We therefore trust that a forced-exploration tree scored by the value
  head can detect a *policy-only* flaw. **Do not spend compute on exact-solver / head-to-head
  circularity checks** — that gate is explicitly waived.
- **The actual question:** the search already up-weights starved picks 8/50 times
  (`starved_picks_upweighted = 8`), but 0 of them land in the high-fragility (≥0.2) bucket
  (`high_fragility_starved_picks_upweighted = 0/13`). Is that because the **fragility gate is noisy**
  and mis-selects positions, not because the search fails to find denial?

## Why de-noising is the hypothesis (mechanism)

Two independent noise sources plausibly produce the 0/13, and each has a dedicated knob:

1. **`fragility = headline_edge − searched_value_actor`** (`denial_search.py:605`). `headline_edge`
   is the **root-search Q** for the representative (`denial_search.py:424`), computed at only
   `search_sims = 32`. A noisy `headline_edge` scatters fragility, so the "high-fragility" set (≥0.2)
   does not reliably contain the genuine denial positions. → **Knob: `search_sims` up.**
2. **`searched_value`** comes from the expectiminimax tree, whose chance node **samples** `chance_k`
   draws when `C(remaining,4) > k`. At `chance_k = 4` this value carries large sampling variance →
   inflated `stderr` → the robust softmax in `denial_policy_target` subtracts
   `uncertainty_z * combined_stderr` from every gap (`denial_search.py:122`), **flattening** the
   distribution so a genuinely-denied low-prior pick is not up-weighted decisively. → **Knob:
   `chance_k` up** (monotonically reduces sampling variance; exact when `C(n,4) ≤ k`).

Success looks like: as these knobs de-noise the gate, the **existing 8 starved-upweights migrate
into the high-fragility bucket**, and `high_fragility_starved_picks_upweighted` climbs off 0.

## Method

Build a thin sweep harness (e.g. `games/kingdomino/denial_signal_sweep.py`) that reuses
`DenialSearch.search_position` — do **not** perturb the validated `denial_search.run_validation`.

### 1. Freeze the position set (critical — this is the #1 confound)

`generate_az_midgame_positions` currently walks trajectories using `_root_search` at
`config.root_search_sims`, so raising `search_sims` would change **which** positions are sampled,
confounding the sweep. Decouple them:

- Sample the position set **once**, at the original settings (`seed=20260716`,
  trajectory root sims = 32, `min_deck=8`, `max_deck=28`, `positions=50`), and **serialize the
  `GameState`s to disk** (e.g. `runs/kingdomino/denial_search/signal_positions.jsonl`).
- Every sweep cell loads that identical frozen set. The trajectory-sampling sim count must be a
  **separate, fixed** parameter from the denial-search `search_sims` being swept.
- These must be the same 50 positions as the original run — the reference cell (below) proves it.

### 2. Reference-cell anchor

Run one cell at the **original config** (`search_sims=32, chance_k=4`) on the frozen set and confirm
it reproduces the published report: `high_fragility_starved_picks_upweighted=0`,
`starved_picks_upweighted=8`, `high_fragility_positions=13`, fragility median ≈ 0.037. If it does
not reproduce, the frozen set or code path differs — stop and fix before sweeping. (Note: unlike the
TT task there is **no byte-identical gate** here — the sweep is *supposed* to change outputs; the
reference cell is the only fixed anchor.)

### 3. Sweep grid (two one-at-a-time ablations sharing a center)

Hold everything else at the invariant-locked config (checkpoint + sha, official cascade,
`policy_temperature=0.10`, `material_margin=0.03`, `starved_prior=0.10`, `placement_top_k=2`, band).

- **Vary `chance_k`** at fixed `search_sims=128`: `chance_k ∈ {4, 16, 64}`.
- **Vary `search_sims`** at fixed `chance_k=16`: `search_sims ∈ {32, 128, 400}`.

That is 5 distinct cells (center `128 / 16` shared). Do **not** run a full 3×3 — `chance_k=64`
multiplies the second-round subtree ~16× and is the cost driver; keep it to the isolating ablations.
If wall-time is prohibitive, drop the frozen set to 40 positions (re-anchor the reference cell
accordingly) — but keep the set identical across all cells. Concurrency is deferred
([[kingdomino_denial_curriculum]]); run sequentially, or manually shard disjoint position ranges
across processes if convenient — do not build a scheduler here.

**Staged execution (do NOT run the grid in cell order).** The `chance_k=64` cell costs ~16× the
reference in BOTH compute (~8 GPU-hr at 50 pos) and peak RAM (see the memory bound below), and it is
the whole 3×3-avoidance rationale. So run it **last and conditionally**:

1. **Stage A — cheap cells first:** run the four cells with `chance_k ≤ 16`
   (`128/4`, `32/16`, `128/16`, `400/16`) plus the reference `32/4`. These are small-RAM and total
   ~5–6 GPU-hr. They already establish the **migration trend** on the 8 baseline starved-upweight
   positions (step 4).
2. **Decision gate:** inspect the Stage-A migration table. If migration is already climbing toward
   the PRIMARY criterion by `chance_k=16` (baseline positions rising toward `fragility ≥ 0.2`), the
   routing decision may not need `chance_k=64` at all — record that and treat Stage B as
   confirmatory. If Stage A is flat, `chance_k=64` becomes the decisive NULL-vs-signal test.
3. **Stage B — the `128/64` cell** only after Stage A, with the memory bound applied.

**Peak-memory bound (required — the naive run OOMs on `chance_k=64`).** The eval caches are already
size-capped (`max_policy_cache`, `max_leaf_cache`, `_node_tt_max` clear-on-overflow) and are NOT the
leak. The blowup is the **materialized leaf/node frontier**: `values_p0(n.state for n in leaves)`
(`denial_search.py:581`) collects the whole expectiminimax leaf frontier — each leaf a full
`GameState` — into memory at once, and at `chance_k=64` that frontier is ~16× the reference working
set. Before running Stage B, bound peak RAM without changing any emitted value:

- **Chunk the leaf-frontier evaluation** — evaluate leaves in fixed-size slices (reuse the existing
  `batch_size` chunking) instead of materializing the entire frontier list; this changes batch
  composition only, not the per-leaf value, so labels are unaffected (assert this against a small
  `chance_k=16` position: identical `policy_target`/`searched_value` chunked vs unchunked).
- **Stream sweep results to disk** — write each position's row to `signal_sweep.json`/JSONL
  incrementally rather than buffering all cells × positions × seeds (with policy vectors) in a list.
- Prefer the **40-position** frozen set for Stage B, and you may lower `_node_tt_max` — cross-position
  reuse is structurally ~0 ([[kingdomino_denial_curriculum]]), so shrinking it costs nothing.
- If it still will not fit, shard the 40 positions across processes **for the `128/64` cell only**;
  each worker holds one position's frontier, so keep the worker count small.

### 4. Migration metric (the primary deliverable)

Identify the **8 baseline starved-upweight positions** (from the reference cell). For every cell,
report for those same 8 positions: their `fragility`, whether each now clears the 0.2 gate, and
their `policy_target − raw_prior` on the corrected-best pick. The headline number per cell is
`high_fragility_starved_picks_upweighted`, but the **per-position migration table** (do the same 8
positions rise past 0.2?) is what actually answers the question — an aggregate could move for the
wrong reason.

### 5. Stability evidence (secondary)

Re-run the reference cell and the **`search_sims=400 / chance_k=16` cell** each with **3 root-search
seeds** (varying the `_root_search` seed for `headline_edge`; and the CRN seed for `searched_value`).
**Use the `400/16` cell, NOT the `128/64` cell, for the stability sub-study** — it is ~⅓ the cost
(no ~16× `chance_k` multiplier tripled across seeds, which would dominate the whole run), and
`search_sims` feeds `headline_edge`, the noisier gate term this study is meant to characterize.
Report the variance of `fragility` per position and of the high-fragility set membership. The
de-noising claim predicts: `headline_edge`/fragility variance **shrinks** as `search_sims` rises, and
the corrected-best `policy_target` **sharpens** (moves further above `raw_prior`) as `chance_k` rises.
(If Stage A leaves `chance_k=64` as the decisive cell, add a single-extra-seed check there — one
re-run, not three — to bound its variance without paying 3× its cost.)

### 6. Negative-fragility characterization (resolve in passing)

The report shows fragility min −0.397 (`headline_edge < searched_value` → search finds the position
**better** than the shallow root Q thought; "anti-fragile"). Confirm both terms are in the same
actor frame (they are: `headline_edge` = actor-frame root Q, `searched_value_actor` = actor frame —
verify). Then report whether `|negative fragility|` **shrinks as `search_sims` rises** (→ it was
noise) or **persists** (→ genuine search-found upside, or a residual sign/frame issue to flag).

## Report — `runs/kingdomino/denial_search/signal_sweep.json`

- Provenance: checkpoint sha, frozen-positions path + hash, per-cell config, seeds.
- Per cell: `high_fragility_positions`, `starved_picks_upweighted`,
  `high_fragility_starved_picks_upweighted`, `material_corrections`, fragility distribution
  (min/median/p90/max), mean `stderr`, mean corrected-best `policy_target − raw_prior`, negative-
  fragility count, elapsed / positions-per-hour.
- The **migration table** for the 8 baseline positions across all cells.
- The stability variances from step 5.
- A one-paragraph verdict against the pre-registered criteria below.

## Pre-registered success + routing

- **PRIMARY (confirms mechanic):** as the gate de-noises, the starved-upweights migrate into the
  high-fragility bucket — concretely, at the most de-noised cell **≥ 4 of the 8** baseline starved-
  upweight positions now have `fragility ≥ 0.2`, and `high_fragility_starved_picks_upweighted` is
  clearly off 0. **Route:** mechanic confirmed on the net's own terms → proceed to the equal-compute
  **control-vs-treatment curriculum retrain**, using the de-noised `(search_sims, chance_k)` as the
  label-generation config (record the throughput cost of that config for the concurrency work).
- **SECONDARY (supporting):** fragility/`headline_edge` variance falls with `search_sims`;
  corrected-best `policy_target` sharpens with `chance_k`; negative-fragility count shrinks.
- **NULL (blocks retrain):** if migration does **not** occur even at `search_sims=400, chance_k=64`
  — the starved-upweights stay out of the high-fragility bucket — then 0/13 is **not** mere gate
  noise. Do not retrain. Investigate, in order: (a) the fragility definition itself (is
  `headline_edge` the right baseline, or should fragility be measured against the *raw policy's*
  pick rather than the representative's root-Q?); (b) the `denial_policy_target` up-weight math
  (does a real denial gap survive the robust-softmax subtraction at all?); (c) as a last lever,
  `placement_top_k` (opponent placement fidelity) — the only tree knob deliberately held out of this
  sweep. Report which is implicated.

## Invariants / scope

- Same current-best checkpoint (record sha); reserved test split stays closed; frozen set identical
  across every cell.
- No retrain, no mixture construction, no throughput/concurrency code, no label-schema change beyond
  the additive sweep report.
- Do not re-verify the value head (stipulated adequate). Do not attempt leaf-TT reuse (structurally
  ~0, documented).

## Tests

Extend `games/kingdomino/tests/test_denial_search.py`:
- Frozen-set determinism: loading `signal_positions.jsonl` yields identical `GameState`s across
  runs; the reference cell on the frozen set reproduces `starved_picks_upweighted=8`,
  `high_fragility_starved_picks_upweighted=0`.
- Trajectory sims and denial `search_sims` are independent: changing `search_sims` does not change
  the frozen set.
- `chance_k` monotonicity on a fixed position: increasing `chance_k` does not increase the corrected
  leaf-value `stderr` (variance is non-increasing), and hits exact enumeration when `C(n,4) ≤ k`.
- Chunked leaf-frontier equivalence: on a fixed `chance_k=16` position, evaluating the leaf frontier
  in slices yields byte-identical `policy_target`/`searched_value` to the unchunked path (the memory
  bound must be behavior-preserving).
