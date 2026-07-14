# Step 3 — Sparse Feature Accumulator: what to include & how

**Rev 3** (supersedes Rev 2; see §Changelog). Recommendation for the NNUE accumulator,
the step that makes the eval *fast*. Grounded in `encoder.py`, `board.py` scoring
(exact), `dominoes.py` (verified counts), and the `search::Game`/`Eval` boundary.

## Build status (2026-07-14)

At-a-glance progress; commits carry the granular history. Both encoders are
frozen-ready — the schema hashes (`core_schema_hash`, `summary_schema_hash`) must be
stored in every derived-feature artifact / `.knnue` / trained model.

- [x] **D4 augmentation bug fix** — flat `width`/`height` swap under 90°/270° (numpy
      `_transform_flat` + Rust `transform_flat`); had mis-augmented even the dense net.
- [x] **Official-outcome cascade labels + conservation** — self-play `win_target`
      routes through `official_outcome_i8` (score→largest territory→total crowns);
      genuine draws stay 0.5. 48-domino conservation ledger as an engine gate.
- [x] **Data harness** `nnue/datagen.py` — GPU-free CPU self-play → replayable-source
      buffer (features DERIVED by replay, no encoder-lock). Strict loader hard-fails on
      engine/format/catalog/rules mismatch. 50k pilot clean (0 replay failures).
- [x] **Sparse core encoder (5,710)** `nnue/sparse_encoder.py` — lossless perspective-
      relative Markov core. Frozen. Gated: seat-swap, lossless decode, completeness,
      inventory, hidden-deck-order, boundary, RAW golden fixtures, semantic schema hash.
- [x] **Summary encoder (171)** `nnue/summary_encoder.py` (commit `1f4093a`) — base 50 +
      extension 78 + global 43. Hardened: all extension/global norms are TRUE
      combinatorial bounds → provably 0 clip; only base `score/160` & claim `legal/64`
      saturate by design (measured max 94/160, 38/64, 0 clips on random play — re-audit
      on stronger-play corpora). Semantic schema hash; golden fixtures. Frozen.
- [x] **D4 `transform_state` + augmentation-identity gate** `nnue/d4.py` (commit
      `d611810`) — `encode(transform(s)) == Dmap(encode(s))` over 25 trajectories × 8 D4
      elements for BOTH sparse and summary. `coord_fwd` drives the state transform and
      the index perms (no drift). Sparse/summary augmentation now unblocked.
- [x] **Rust reference derivation from `RustGameState` + Python/Rust parity** — isolated
      `kingdomino_rust/src/nnue_features.rs`; `RustGameState.nnue_features` returns the
      sorted 5,710-core indices + 171-value summary, including terminals. Exact sparse
      and bit-identical float32 summary parity hold at every ply, both perspectives,
      forced discards, all four rules configurations, golden-derived states, and all D4
      transforms. Rust exposes and gates both frozen schema hashes. The shared
      Python→Rust converter now preserves discard counts (previously defaulted them to
      zero, which would corrupt Harmony/progress features after a forced discard).
- [x] **Fixed-weight network-output seat-swap** — the full sparse+summary network
      preserves actor-frame outcome/margin exactly under a seat swap and negates both
      after conversion to the P0 frame. This is an untrained, fixed-weight structural
      gate, so training cannot hide a frame error.
- [x] **Packed sparse trainer + two-head/aux pilot** — `nnue/sparse_data.py` replays
      source games into CSR active-index lists + summaries and stamps both frozen schema
      hashes; D4 is applied by feature permutation at batch time. `nnue/sparse_net.py`
      uses `EmbeddingBag(sum)` for the exact accumulator column sum; auxiliary targets
      are final per-player territory/largest/crowns + Harmony/Middle, derived from the
      terminal Rust state. On the validation split (41,340 train / 4,940 val), the 50k
      pilot reached best Brier **0.2115** vs **0.2500** base rate and margin MAE **16.62**
      vs **17.81** points at epoch 8; later epochs overfit. The reserved 5,720-position
      test split remains unopened.
- [x] **Sparse v3 export/loader + Rust oracle + reversible dual accumulators** — the
      `KNSP` v3 artifact is feature-major and omits training-only auxiliary heads.
      Stateless Rust forward matches PyTorch; incremental search matches stateless
      values/actions and node counts across deterministic and sampled-chance trees.
      The composite search state makes game undo + accumulator rollback one indivisible
      operation; full playout/unwind and injected-error cleanup restore both exactly.
      Semantic placement additions come from the move undo; compact dynamic banks are
      re-derived after each move/chance result, so sampled rows and round promotion cannot
      drift from the frozen reference encoder.
- [x] **v3.1 profile-driven float inference** — same v3 artifact and frozen schemas; no
      retraining. The loader transposes dense-tail weights to input-major layout once,
      making each scalar input an SIMD-friendly update across contiguous outputs and
      skipping zero activations. Summary construction now derives base + extension from
      one region traversal per board, reuses its flood stack, uses fixed frontier bitsets,
      and shares unresolved-claim legal counts. On the same six-position depth-3
      `choose_action` gate, incremental throughput rose from **33.5k to 94.3k nodes/s
      (2.81×)**; stateless reaches **87.1k** and the accumulator adds **1.08×**. Component
      cost is now approximately **2.58 µs sparse derive+sum, 3.60 µs summary, 1.21 µs
      float tail**, down from 3.8/10.5/14.2 µs. Python/Rust feature and forward parity,
      stateless/incremental values/actions/node counts, sampled chance, and unwind gates
      remain green.
- [x] **v3.2 guarded quantization + SIMD** — `sparse_nnue_q` quantizes the unchanged v3
      float artifact at load: per-channel power-of-two int16 accumulator scales, int8
      tail weights (separate accumulator/summary scales in tail0), dynamic int16 tail
      activations, float biases/dequantization, and AVX2 integer dot with scalar parity.
      The conservation-derived 112-active-feature guard gives a conservative pilot bound
      of **19,958 < 32,767**; transition removals precede additions so every intermediate
      remains a bounded subset. On 100 real positions, expected-score MAE/max are
      **0.0028/0.0109** and margin MAE/max **0.13/0.43 points**. Search action agreement
      is 20/20 at depth 2 and 6/6 at depth 3. The six-position depth-3 gate is **102.3k
      nodes/s** vs 98.8k float (**1.04×**) with roughly half the inference-weight memory.

- [x] **Native operational fast search** — `choose_action_timed` adds deadline-safe
      iterative deepening with last-complete-depth fallback, full-width cheap heuristic
      ordering with root/PV promotion, root window reuse, aspiration re-search,
      depth-aware TT reuse at canonical round roots, bounded Star1 chance pruning, and
      exact official-outcome extensions through deterministic tails. Timeout/error
      paths unmake both game and accumulator.
      Telemetry exposes completed depth, timeout, total/final-iteration nodes, Star/TT
      cutoffs, re-searches, and exact extensions. Native release suite: 47/47. On one
      real sampled-chance depth-5 gate, the whole iterative search was **9.321s / 8.79M
      nodes** vs fixed-depth **10.115s / 9.79M**; final-iteration nodes fell 14%.

The isolated release-wheel operational/sparse/generation Python suite passes
**32/32**, including the bot adapter and timed quantized path, without replacing the
extension loaded by the web server. Six-position `sparse_nnue_q` timed gates completed
depth 3 on every 0.5s move, depth 3/4 in a 5/1 split at 1.0s, and depth 3/4 in a 3/3
split at 2.0s, at approximately 101k-126k nodes/s.

Full-width ordering is independently gated from selective pruning and is now the NNUE
operational default. It never truncates the action list. At completed depth 3 over 20
positions it preserved root values 20/20 while reducing nodes 15.7% and wall time
11.5%. A same-artifact, equal-clock, paired-seat gate was neutral on outcomes (32-32
over 64 games), positive on average score margin (+7.14), and timing-balanced. The
underlying Rust API retains an unordered default for explicit oracle comparisons.

**Deferred (non-blocking):** independent Python-oracle full-replay gate; align legacy AZ
`terminal_search_value` with the official cascade before AZ/hybrid data.

## The one principle everything follows

NNUE is **sparse binary features + a big first layer, updated incrementally**, where the
incrementally-maintained quantity is the **pre-activation `z`**, never the post-ReLU
value:

```
  z = b1 + Σ W1[:, f]   over active features f          ← maintained incrementally
  a = ReLU(z)                                            ← computed ONLY at leaf eval
  value = tail(concat[a | summary(state)])
```

ReLU is nonlinear, so `a' = a + added − removed` is **wrong**. Column add/subtract is
valid only on the *linear* pre-activation. A move that flips a handful of features
updates `z` by a few column adds — O(flips × width) instead of recomputing the
`~5.7k × width` layer. That was the design hypothesis for removing the **dominant**
per-node cost. v3.0 profiling answered it more precisely: once the stateless evaluator
also sums only the ~110 *active* rows, sparse derive+sum is ~3.8 µs while summary is
~10.5 µs and the float tail ~14.2 µs. The accumulator is correct but only neutral-to-
modestly faster in deep search; hundreds of k nodes/s requires work on the tail/summary,
not more first-layer delta engineering (§Phasing).

So the feature set is designed around one question: **does each move flip only a few
features?** Everything below is chosen so the answer is yes.

## Design target: a lossless public-state core with a decode fingerprint

The accumulator features must be a lossless encoding of the **future-relevant public
Markov state** — *not* byte-lossless for every stored engine field:

> **Completeness criterion (Markov quotient).** From the active sparse-index set alone,
> one can reconstruct an exact **canonical public-state fingerprint** (all public fields:
> both boards cell-by-cell, current row, all unresolved claims with owners, bag
> membership, phase/terminal, actor, turn slot, discard/harmony status, and the rules
> configuration). Two reachable states encode identically **iff** they have the same
> fingerprint.

**Intentionally omitted engine history** (a *quotient*, not a loss — none of these can
change legal actions, chance children/probabilities, future transitions, or terminal
score/outcome, and each must be *tested* to that effect):

- **Deck order** (hidden information — see the info-set gate in §Verification).
- **Resolved claim entries before `actor_index`** (already placed → on the board).
- **Original placed domino IDs**, once only their two board cells matter.
- **Exact discard count beyond zero/nonzero.** A discard is *invisible on the board* yet
  permanently kills **Harmony** (which needs occupied==49). It does **not** kill Middle
  Kingdom — that depends only on a 7×7 centered bounding box (`width==height==7`, castle
  centered), *not* on all 49 cells being filled, so Middle Kingdom eligibility is fully
  determined by board geometry already captured in the cell features. Hence only the
  **binary** discard/harmony-lost flag is future-relevant and retained; the exact count is
  derivable from round/turn counters + board when a summary aid (e.g. progress) needs it.
- **`start_player`** after the initial selection is complete.

Note: **no "modulo symmetry" qualifier.** We do *not* canonicalize orientation; we
augment over D4 at training time, and augmented orientations have *different* permuted
indices. Equal feature sets therefore mean the *same* state, full stop. (If we later
canonicalize to a representative orientation, revisit this.)

The schema also **encodes the rules-configuration flags** (`harmony_enabled`,
`middle_kingdom_enabled`) and an explicit **terminal** marker, so it is not silently
bound to one rules variant or ambiguous at game end (the phase bank otherwise lists
only the three in-play phases). Terminal states are still *scored exactly* by the
searcher (`terminal_value_p0`), not by the NNUE — the terminal bit is for completeness
and robustness.

## Two shared-weight perspective accumulators (frame)

Maintain **two** pre-activation accumulators over **one shared weight table**:
`z_from_P0` and `z_from_P1`. At a leaf, select the actor's perspective,
`a = ReLU(z_from_actor)`, predict the **actor-relative** expected score, then flip to P0
(`2·sigmoid−1`, negate if actor==P1) exactly as Step 2b already does.

Why this over a single player-absolute accumulator: buffer-frame compatible
(training data is actor-relative and doesn't record the actor); player-swap symmetry is
free; and a placement is cheap in both (below).

## Feature banks & dimensions (the lossless core)

Verified from `dominoes.py`: 48 unique dominoes, **16 distinct `(terrain,crowns)`
half-types**, 33 distinct compositions. **Active/flip counts below are stated explicitly
as per-accumulator vs total-across-both.**

| Bank | Encoding | Dim | Active (typ. → max, per accum.) | Flips/placement |
|---|---|---|---|---|
| **Board cells** | 2 roles × 169 cells × 16 half-types | **5,408** | ≤96 (my+opp, ≤48 each) | **+2 per accum. (4 total)** |
| Current row | domino-ID **membership** | 48 | ≤4 | ~2 pick / +4 deal (per accum.) |
| Pending-to-place | owner × domino-ID, unresolved only | 96 | ≤4 | ~2 / round |
| Claimed-for-next | owner × domino-ID | 96 | ≤4 | ~1 / pick |
| Bag remaining | domino-ID membership | 48 | ≤44 early | −4 / deal |
| Phase | one-hot incl. **terminal** | 4 | 1 | ~1 |
| Actor | one-hot | 2 | 1 | 1 / ply |
| Turn slot | one-hot (pick order in round) | 4 | 1 | ~1 |
| Discard/harmony-lost flag | per player | 2 | ≤2 | rare |
| Rules-config flags | harmony_enabled, middle_kingdom_enabled | 2 | ≤2 | 0 (fixed/game) |
| **Total** | | **5,710** | **~110 mid-game (loose bound ≲157)** | **~4–12 total** |

The counts corrected from Rev 2: board is ≤96 active **per perspective** (both roles),
and unresolved pending can total **4**. The ~110 figure is a mid-game representative. The
per-bank maxima do **not** sum to a *reachable* maximum: max bag occupancy is early-game
while max board occupancy is late-game, and they cannot coexist — so ≲157 is a loose
upper bound, not an attainable state. A placement adds **2 board features per accumulator
= 4 total**.

Encoding rationale (unchanged from Rev 2, still valid): all 169 cells (7×7 board reaches
offset 6; `MAX_BOARD_CELLS=48`, `49−occupied` feature); joint `(cell, half_type)` = 2
flips/placement/accum.; current row as ID membership (unique IDs, row sorted by ID →
membership determines order, no slot churn); pending only the *unresolved* claims
(resolved ones are on the board); bag membership is public (only order is hidden); no
ternary scalars in the accumulator (`pick_pos`, fill-ratios live in the summary).

First layer ≈ `5,710 × width`; at width 256 ≈ **1.46M** params. **Train it as an
embedding-column sum over the packed active index lists (CSR / `EmbeddingBag(mode=sum)`),
never a dense `N × 5,710` matrix** — a 500k-row dense input is ~11.4 GB and pointless.

## Summary vector (real-valued, recomputed per node, concat at tail) — EXACT schema

This block **requires retraining if changed**, so it is pinned exactly here. Definitions
are chosen to be computable in the **single** connected-component traversal that already
computes score (`board.score`). **All definitions below are APPROVED/frozen** (2026-07-13).
**Normalization rule (approved):** every denominator is a **fixed schema constant** derived
from the catalog/rules — *never* computed from the pilot data — and is **included in the
feature-schema hash**. Add explicit range assertions on each raw value and **measure clip
frequency** (a clipped feature silently collapses distinct states).

**Perspective-relative — hard contract.** The summary is `summary(state,
perspective=actor)`, **not** `summary(state)`. Every per-player block is ordered
**`[my (actor) block, opponent block]`**, never `[P0 block, P1 block]`. Otherwise the
sparse half would be seat-symmetric while the 171-value tail silently reintroduced
absolute-seat asymmetry — defeating the whole two-perspective design. Seat-independent
blocks (unresolved claims in fixed domino/action order; bag aggregates; game_progress)
stay global, but any **owner indicator** on a claim must be encoded **my/opp**, never
absolute P0/P1.

**Base block — reuse `_encode_board_summary` logic per perspective (25 each, ordered
[my, opp], already exact & normalized):** `score.total`(/SCORE_SCALE), `score_by_terrain[6]`(/SCORE_SCALE),
`largest_by_terrain[6]`(/48), `total_crowns`(/MAX_TOTAL_CROWNS), harmony one-hot[3]
(`[awarded, still_possible, impossible]`), middle one-hot[3], `width`(/7), `height`(/7),
`remaining = 49−occupied`(/48), `legal_count` for next domino(/MAX_LEGAL_PLACEMENTS),
`forced_discard` flag. = **25 × 2 = 50**.

**Extension block (per perspective, ordered [my, opp]), exact definitions:**

| Feature | Dim | Definition | Norm |
|---|---|---|---|
| terrain_cell_count | 6 | placed cells of each placeable terrain | /48 |
| terrain_crown_count | 6 | Σ crowns on each terrain | **/ per-terrain catalog crown max** (`MAX_CROWNS_PER_TERRAIN[t]`, NOT MAX_TOTAL_CROWNS — each terrain saturates on its own ceiling) |
| largest_region_crowns | 6 | crowns in the largest-area region of each terrain; **tie rule (approved):** among that terrain's *maximal-area* regions, take the maximum crown count | **/ `MAX_CROWNS_PER_TERRAIN[t]`** |
| global_largest_territory | 1 | `ScoreBreakdown.largest_territory_size` (terrain-agnostic official tiebreaker) | /48 |
| crownless_region_count | 1 | # connected same-terrain regions with 0 crowns | **/48** (`MAX_BOARD_CELLS`; ≤ one region per cell — a genuine bound, not the old empirical /24) |
| stranded_crowns | 1 | Σ crowns in area-1 regions (isolated crowned tiles) | **/39** (`TOTAL_CATALOG_CROWNS` = Σ all catalog crowns — true bound) |
| open_frontier | 6 | **(approved)** per terrain, # of **unique empty cells** that are empty + in-bounds + **bbox-admissible** (keeps bbox ≤7×7) AND orthogonally adjacent to ≥1 placed cell of that terrain. "Legal expansion" means bbox-admissible, **not** that a whole domino can be placed there. A cell may count for two terrains (e.g. Wheat *and* Forest) if adjacent to both — counted once per terrain | **/48** (`MAX_BOARD_CELLS`; frontier ⊆ empty in-bbox cells — a genuine bound, not the old empirical /24) |
| enclosed_single_holes | 1 | **(approved, renamed to avoid overclaiming)** # of empty cells inside bbox with **all 4** orthogonal neighbors occupied — detects fully-enclosed **single-cell** holes only, *not* every possible unfillable cavity | /48 |
| gaps | 1 | bbox_area − occupied (empty cells inside the bounding rectangle) | /48 |
| castle_extent L/R/U/D | 4 | `castle_x−min_x`, `max_x−castle_x`, `castle_y−min_y`, `max_y−castle_y` — how far the kingdom reaches past the castle each way (all four = 3 ⇒ Middle-Kingdom geometry). **Replaces** the redundant `bbox_room` (which collapsed to `7−width`/`7−height`). **D4:** permutes/swaps like directions | /6 |
| largest_crownless_region | 6 | per terrain, size of the **largest 0-crown** connected region (dormant potential a single crown would activate — distinct from `largest_region` which may be crowned). ⚠ **first ablation candidate** (see §Ablation) | /48 |
| **per-player subtotal** | **39** | | |

**Global block, exact:**

| Feature | Dim | Definition | Norm |
|---|---|---|---|
| bag_terrain_halfcount | 6 | # remaining half-tiles of each terrain in the bag | **/ `MAX_HALVES_PER_TERRAIN[t]`** (per-terrain catalog half-count) |
| bag_terrain_crowns | 6 | Σ crowns of each terrain remaining in the bag | **/ `MAX_CROWNS_PER_TERRAIN[t]`** (per-terrain catalog crown max) |
| unresolved claims, **self-identifying** (fixed action order, ≤4) | 24 | per claim (6 each): presence, `legal_placement_count`(/64 = `MAX_LEGAL_PLACEMENTS`, a saturating scale — see clip note), `forced_discard` flag, `owner_role` (+1 my / −1 opp / 0 absent), `draft_priority_rank` = **`(domino_id − 1) / 47`** (zero-based tempo signal, *not* content: id→tile is non-monotonic), `turn_distance` = **`min(k, 3) / 3`** where k = slot offset from the actor (actions until it resolves). Makes each slot a self-contained unit so the tail needn't re-bind a summed accumulator to the right legal-count. **D4-invariant** | — |
| next-round draft order `pick_pos[4]` | 4 | sort `next_claims` by domino id; element k = owner of next-round pick k (+1 my / −1 opp / 0 unassigned). Exposes tempo / consecutive turns / first-vs-last choice — a *sort* an EmbeddingBag sum cannot reconstruct. **D4-invariant** | — |
| game_progress, fill_ratio my/opp | 3 | **(approved)** `game_progress = ((occupied_non_castle_cells / 2) + total_discards) / 48` (reaches 1.0 at terminal even with discards; the old `placed_cells/96` did not and duplicated fill_ratio). **`fill_ratio` (per role) = placed_non-castle_halves / bbox_area** — castle **excluded** from the numerator but **included** in bbox_area (occupied density of the reachable rectangle) | — |

**v3.0 summary total = 50 (base) + 78 (2×39 ext) + 43 (global) = 171.**

> **Clip audit (recorded 2026-07-13, `test_summary_encoder.py`).** Every extension +
> global feature is now normalized by a **true combinatorial/catalog bound** (`/48`,
> `MAX_CROWNS_PER_TERRAIN`, `TOTAL_CATALOG_CROWNS=39`, per-terrain half/crown maxima),
> so their raw values *cannot* exceed the denominator — **provably 0 clips** (asserted
> over 30 random games + a dense fill-out board). The only two saturating features are
> **inherited from the base block / claims**: `score.total /160` and
> `legal_placement_count /64` — these are training *scales*, not rules maxima. Measured
> over random self-play: score max **94/160**, legal max **38/64**, **0 clips observed**.
> Saturation is therefore accepted by design; a stronger-play corpus must be re-audited
> (the test prints the live max + clip counts so this stays visible).

Tail: `[a (width) | s (171)] → 32 → 32 → {outcome_logit, margin, + auxiliary heads}`
(auxiliary heads below; still small vs the 256-wide accumulator).

> Scope note: 171 summary inputs widen the tail and the dense net already overfit by
> epoch 4. The **schema** is frozen for training-stability, but the additions past the
> lossless core are learnability aids — see §Auxiliary heads (prefer regularizing via
> outputs) and §Ablation (measure which handcrafted inputs are load-bearing).

### Auxiliary training heads (outputs, not inputs — dropped at export)

Prefer shaping the shared trunk with auxiliary **targets** over piling on handcrafted
**inputs**: extra heads regularize the representation toward the quantities the official
outcome cascade uses, cost ~nothing, and are **not exported to `.knnue`** (zero inference
cost — the searcher only reads outcome + margin). All predict game-final quantities from
the current state, like the outcome/margin heads. Add heads for, **per player** (not just
differences — differences are linearly derivable by the tail, and both per-player values
are already in the buffer):

- final **territory score**, final **largest-territory size**, final **total crowns**;
- **Harmony achieved** (bit), **Middle Kingdom achieved** (bit).

Temper the two bonus heads: Harmony/Middle occur in a small fraction of games, so they are
class-imbalanced and low-signal — keep their loss weight modest; their value is
representation-shaping, not accuracy. **Data dependency:** each derived training artifact
must carry the game's final `ScoreBreakdown` targets per player (territory,
largest-territory, total-crowns, `harmony_bonus>0`, `middle_bonus>0`). The replayable
source is sufficient: these are now derived from the terminal Rust state while features
are materialized, so the existing pilot does not need regeneration or a format bump.

### Ablation (pre-registered, on a reserved untouched split)

"Not a kitchen sink" is a claim to *measure*, not assert. Freeze the schema with all
additions, but pre-commit to an ablation on a **test split never touched during model
selection**: drop `largest_crownless_region` (first candidate); drop the claim-slot
extras; drop summary breadth carried by the aux heads. Priors: `pick_pos` and claim
self-ID earn their place (they fix the sort/binding the accumulator sum can't); item 4 and
some summary breadth may be redundant once the aux heads exist. This is the honest test of
input-vs-output regularization.

## How to include it (the mechanism)

The accumulator is **search-path state**, so the evaluator is no longer the current
immutable `Eval::eval(&state)`.

1. **Semantic, chance-aware deltas on `Game`.** Deltas are emitted as **game-level
   semantic changes**, independent of the network's perspective/index layout; the
   evaluator maps each into *both* accumulators. A `FeatureDelta` with a single index
   list is **wrong** — a P0 placement is a `my` feature in `z_from_P0` but a *different*
   `opp` index in `z_from_P1`.

   ```
   enum SemChange {
     Cell   { owner: P, cell: CellId, half: HalfType, on: bool },
     RowId  { id: DominoId, present: bool },
     Claim  { owner: P, id: DominoId, present: bool, kind: Pending|Next },
     BagId  { id: DominoId, present: bool },
     Scalar { field: Phase|Actor|TurnSlot|DiscardFlag..., ... },
   }
   struct FeatureDelta { changes: Vec<SemChange>, summary_dirty: bool }
   ```

   Chance-awareness mirrors the existing `make`/`make_with_chance` split — feature
   deltas come from the same place the transition does, and a deal's row/bag changes
   depend on the **sampled** row:
   - deterministic → `action_delta(state, action) -> FeatureDelta`
   - stochastic → `chance_action_delta(state, action, chance_outcome) -> FeatureDelta`

   Cleanest: have `make`/`make_with_chance` **return `(Undo, FeatureDelta)`**.

2. **A stateful, cloneable `IncrementalEval` threaded through search.**
   - `push(delta)` → for each `SemChange`, add/subtract the mapped column in **both**
     `z_from_P0` and `z_from_P1`; snapshot for rollback.
   - `pop()` → restore (copy-on-make first; `z` is `width` values × depth ≤ a few KB).
   - `eval(state)` → `tail(ReLU(z_actor), summary(state, perspective=actor))` — the
     summary is perspective-relative too (see the summary §hard contract), not just `z`.
   - `refresh(state)` → rebuild both `z` from scratch (root + equivalence test).
   - **Cloneable per parallel root child** (each rayon branch owns its accumulator stack).

3. **Error-safe pairing is a hard contract (gate).** *Every* successful `make` must be
   paired with **both** `unmake` **and** accumulator `pop`, on every exit path —
   including recursion that returns `Err` (empty legal actions / non-finite eval) and a
   failed accumulator update. Use an **RAII guard** (a scope struct whose `Drop` runs
   `unmake`+`pop`) rather than trusting each branch to remember both. Tested by injecting
   errors mid-search and asserting the accumulator/undo stacks are balanced.

4. **Weight layout:** `W1` **feature-major** (`input_dim × width`), each feature's column
   contiguous → a delta is a straight vector add. The `.knnue` exporter transposes and
   bumps the format version.

5. **Leaf → heads** unchanged from 2b.

## Data: store replayable SOURCE, not encoded features

The run10 buffer cannot be converted (33 < 48 compositions → domino ID unrecoverable;
no actor; no move log to replay). Generate fresh — **but do not repeat run10's mistake of
storing only an encoder's output.** Storing active sparse indices would lock the buffer
to *this* encoder, the same trap. Store **replayable source**; derive everything else.

**Per game, persist:**
- initial ordered deck / seed, start player, **rules configuration**;
- the **complete action + chance (dealt-row) trajectory**;
- final **official outcome, margin, and tiebreak components** (score, largest territory,
  total crowns);
- **engine/rules version + domino-catalog hash**;
- **generator provenance**: policy, depth, randomness/temperature, teacher eval, seed.

Per-position canonical state and cached active indices are **derived conveniences**
(regenerable by replay). This guarantees a future v4 feature schema needs **no new
self-play** — just re-derive from the stored trajectories.

**Stale-buffer rejection (IMPLEMENTED — `datagen.load_records`).** The loader hard-fails
(`StaleBufferError`) on any mismatch of **engine_version**, **format_version**, or
**catalog_hash** vs the current code, on a buffer that **mixes rules configs**, or (if an
expected config is passed) on a rules mismatch. Records also carry **git commit +
dirty-state/diff hash** so the exact producing engine is identifiable even if
`ENGINE_VERSION` was not bumped. So no incompatible buffer can silently train a model.

## Data generation plan — enhanced Option A (CPU, no GPU)

Measured full-game path (not just raw search throughput): depth-2 self-play ≈
**0.143 s/game mean** (52 decisions/game) → **~23 min for 500k positions on one core**,
before encoding/serialization. Parallelize with **independent processes** (the
Python-exposed search does not release the GIL). So generation is CPU-only and cheap.

But **pure deterministic depth-2 self-play is a narrow, biased dataset** — it teaches the
*return under a shallow heuristic policy*, not optimal position value, and fixed depth
has uneven round-awareness (decisions near a deal see the next row; earlier ones don't).
For the first *real* buffer:
- **vary** seeds, start player, shallow depths, and teacher evaluation;
- **controlled exploration**: ε-random or temperature/top-k root choices, especially
  early; some **random openings** and deliberately awkward placements;
- **split train/val/test by whole game** (no position leakage);
- **preserve generator config per game** (provenance above);
- if affordable, **deeper search for a minority** of games / selected anchor positions;
- **exact-solver relabeling** for tractable late positions.

**Labels — one authoritative outcome path (Priority 2).** Use **exact-tiebreak
outcomes**: `score → largest territory → total crowns → draw`. The Rust cascade already
exists (`official_outcome`/`official_outcome_i8`, lib.rs:1547/1764) but the self-play
label-fill paths (lib.rs:5765/6352) still write *score-only* targets with a stale "lacks
determine_winner" comment. **Every** label path must route through the one official
function before we generate the buffer. Precise rule (this does *not* reverse "leave draws
in" — it sharpens it):

- Score ties **resolved** by largest territory or total crowns become decisive
  wins/losses (score-only labeling wrongly called these 0.5).
- **Genuine** ties, still tied after the full cascade, stay **0.5** and **remain in the
  dataset**.

*[Confirmed: exact-tiebreak labels; genuine post-cascade draws stay 0.5 and remain in the
data.]*

> **AZ terminal alignment (deferred to before AZ/hybrid data).** The buffer labels are now
> official, and the CPU `RustSearch` teacher's terminal value already uses the official
> cascade (`terminal_value_p0`). But the legacy AZ MCTS leaf `terminal_search_value`
> (lib.rs) still treats a raw-score tie as a draw. That is internally consistent for *this*
> CPU pilot, but before generating AZ/hybrid data its terminal value must be aligned with
> the cascade, or the AZ search could call a tiebreak win neutral while the labels call it
> decisive. (Left as-is now: it is under a documented bit-identical-to-Python contract.)

**Actor attribution (Priority 2).** Kingdomino does **not** alternate seats reliably (a
player can take consecutive turns). Labels must use the **recorded actor**, never ply
parity or "the other player after `step`". Test attribution explicitly across
consecutive same-player turns.

Strength caveat: Option A validates the **pipeline** end-to-end. Before any strength
claim or AZ challenge, move to a **hybrid** dataset or **iterative search-guided**
generation — "NNUE self-play → automatically stronger NNUE" is *not* guaranteed. Retain
older/diverse data and measure each generation against **fixed** opponents.

## Phasing (retire the fiddliest engineering last)

- **v3.0 — COMPLETE: float accumulator, recomputed rich summary, no union-find.** Schema,
  enhanced-Option-A pilot, packed `EmbeddingBag` training, v3 export, stateless Rust
  oracle, reversible dual accumulator, chance/error cleanup gates, and profiling landed.
  The composite state provides atomic game+accumulator unwind without a separate mutable
  evaluator stack.
- **v3.1 — COMPLETE: profile-driven float inference.** Input-major tail weights plus
  fused/allocation-light summary derivation delivered 2.81× end-to-end throughput on the
  fixed depth-3 gate. Same schema and artifact → **no retrain**.
- **v3.2 — COMPLETE: guarded quantization + SIMD.** Int16 accumulator, int8 tail,
  explicit AVX2/scalar-parity kernel, conservative overflow proof, float-oracle numeric
  gates, and search-level action/performance gates. Same artifact; no retrain.
- **Operational search — NATIVE COMPLETE.** Deadline-safe iterative deepening,
  root/PV/TT ordering and reuse, aspiration windows, bounded Star1 chance pruning,
  deterministic-tail exact extension, and completed-depth/timeout telemetry. Native
  and isolated-wheel Python gates are complete; matched-clock strength measurement
  remains.

## Verification discipline (float-aware; the completeness gate is redesigned)

Enumerating all reachable states is infeasible, and random-state search almost never
hits an accidental omission. Replace it with:

- **Fingerprint reconstruction.** Decode the active-index set → canonical public-state
  fingerprint; assert it equals the engine's fingerprint for that state. (This is the
  operational completeness gate.)
- **One-field mutation pairs.** For states differing in *exactly one* field — a bag/row
  domino ID, claim owner, actor, turn slot, phase/terminal, discard flag, rules
  config — assert **different** encodings. This catches omissions directly.
- **Seat-swap invariant (build this BEFORE generating data).** Construct
  `swap_players(state)` that swaps boards, claim owners (pending *and* next), actor, start
  player, discard flags, and every other player-owned field. Then require, over targeted
  and random states, `encode(state, perspective=P0) == encode(swap_players(state),
  perspective=P1)` — for **both** the sparse index set **and** the 171-value summary (this
  is the gate that actually enforces the perspective-relative summary contract). Coverage
  must include: initial selection after 0/1/2/3 picks; every `actor_index`; interleaved
  ownership (P0/P1/P0/P1); two unresolved claims owned by one player; a player with no
  pending claim but existing next claims; round promotion next→pending; forced discard;
  final placement.
- **Network-output seat-swap gate.** Extend the equality through *evaluation* with
  arbitrary **fixed** weights (tests structural framing, not trained behavior):
  `actor_value(state) == actor_value(swap_players(state))` and
  `p0_value(state) == −p0_value(swap_players(state))`. This catches a wrong `z_actor`
  selection or a mis-applied P0 sign flip even when both *encodings* are already correct.
- **Inventory invariant + 48-ID ledger.** Unplaced public dominoes are exactly
  `current_row ∪ pending_claims[actor_index..] ∪ next_claims ∪ bag` — the engine keeps
  resolved pending entries *before* `actor_index`, which are already on the board, so the
  full pending list would double-count. Each such ID appears **exactly once** across those
  containers and **exactly once** per perspective accumulator under the right role. But
  container-duplicate checks are partly tautological and can't catch an ID that *silently
  disappeared*, so drive a **test-only 48-ID ledger from the trajectory**: classify every
  domino as bag / row / pending / next / placed / discarded and reconcile all 48 against
  the engine **after every transition**.
- **Transition truth table / round-boundary conservation (Priority 3).** For *every* move
  type — initial pick; ordinary placement+pick; the 4th action + chance deal; promotion
  next→pending; final placement; forced discard — verify: inventory conservation (above)
  holds across the transition; every claim has an owner; picked IDs never disappear during
  promotion; placement *atomically* removes the pending claim and adds its two board
  halves; both perspective accumulators receive the role-swapped changes; and **refresh,
  incremental update, unmake, and player-swap all commute**. Round boundaries and forced
  discards get disproportionate coverage.
- **Information-set / chance equivalence (Priority 4).** The encoder must never leak deck
  order. For many states, shuffle *only* the hidden deck and require identical sparse
  indices, summary, and NNUE output. For the chance children: exact child-multiset
  enumeration only for **manageable late bags**; for large bags (a fresh deal is
  `C(44,4)≈135k`) compare the **analytic** row distribution, or deterministically-seeded
  sampled results, rather than enumerating per state. This proves bag *membership* is
  sufficient for the searcher's public belief state without the combinatorial blowup.
- **Summary normalization audits (Priority 5).** `legal_placement_count` must use the
  engine's *deduplicated* placement semantics for symmetric dominoes; compute exact
  normalization maxima from the catalog where possible; **measure clip frequency**
  (`min(score,160)` silently collapses all scores >160 into one value); confirm
  `open_frontier` counts *unique empty cells*, not adjacency edges (current name vs
  definition differ); confirm every directional/per-player field has an explicit D4 rule.
- **Correctness fixtures vs dataset coverage — keep them separate (Priority 5).** Rare
  cases (each tiebreak level, ±6 offsets in every direction, particular forced-discard
  configs, two unresolved claims owned by one player, bag sizes 44/4/0, harmony/middle
  each possible *and* impossible) must be **mandatory correctness gates on targeted,
  hand-constructed states** — never gated on whether a pilot happened to sample them,
  otherwise a *correct* encoder fails just because a rare crown-tiebreak example wasn't
  drawn. Separately, use **dataset histograms as coverage gates** on the pilot buffer
  (all 48 IDs / 16 half-types / every actor slot & phase / consecutive same-player turns /
  every round-boundary pattern), with explicit **injection or oversampling** if a bucket
  is empty. Correctness is proven on constructed states; coverage is measured on the data.
- **refresh vs make/unmake fuzz.** Over large random walks, delta-maintained active set
  = `refresh` **exactly** (set equality); pre-activation `z` within tight numeric
  tolerance (float add-order ULPs — *not* byte-for-byte).
- **Independent Python-oracle replay (deferred; closes the source contract).** Current
  replay verification is *self-consistent* (generation and replay both use Rust
  `step`/official outcome — proves the serialized source reproduces under the same engine,
  which the pilot passed on all 1000 games). To fully verify the trajectory matches the
  Python rules oracle, replay the whole pilot through Python `GameState`, checking actor,
  legal action, phase, public-state fingerprint, final scores, and official outcome at
  every ply. Needs a Rust→Python action mapping; existing Rust/Python differential tests
  (search-equiv, encoder bit-exactness, `official_outcome` parity) cover the risk, so this
  is not encoder-blocking — but it closes the loop.
- **Bounded exhaustion.** Fully enumerate small late-game subsets where tractable.
- **Snapshot restore: exact.** After `push` then `pop`, `z` restores bit-for-bit.
- **RAII balance test.** Inject mid-search errors; assert undo/accumulator stacks balanced.
- **Rust ≈ PyTorch** on real positions within f32 tolerance (extend `test_nnue_eval_equiv`).
- **Quantized path (v3.2): exact.**
- **D4 transform correctness (Priority 1) — and an existing bug to fix.** The hard test is
  `encode(transform_actual_state(state)) == transform_encoded_features(encode(state))`
  (encode a *genuinely rotated state* and compare — not merely "rotating an encoded tensor
  stays structurally valid", which is all the current test checks). Under D4:
  **transforms** — board-cell sparse indices permute; `width ↔ height` under 90°/270°;
  `bbox_room L/R/U/D` permutes under every rotation/reflection. **Invariant** — scores,
  region counts, gaps, holes, bag, claims, legal-placement counts.
  ✅ **Existing bug FIXED** (was: `flat` declared wholly invariant, but it carries
  `width`/`height`, so 4 of 8 D4 transforms mis-augmented the dense net too). Fix landed
  in the shared augmenter — Rust `transform_flat` + numpy `_transform_flat` swap the two
  per-player bbox `(width,height)` slots iff `k` is odd; all three call paths (`augment`,
  Rust `d4_augment`, `self_play._apply_augment`) now route through it. Covered by
  `test_augment_flat_parity` (Rust==NumPy + real swap exercised) and `test_augmentation`
  TEST 3. **Remaining for Step 3:** every *new* directional/per-player field
  (`bbox_room L/R/U/D`, board-cell sparse indices) needs its own explicit transform rule
  — none may default to "unchanged" — validated by the full `encode(rotate_state(·))`
  harness. This harness is a **hard blocker before any sparse-feature D4 augmentation is
  enabled** (deferred only in the sense that it lands with the sparse encoder, where
  `rotate_state` is also needed for the cell-index D4 maps — not optional).
- **Overflow tests** before int16 (v3.2).

## Open decisions to pin (recommended defaults in bold)

- **Summary definitions — APPROVED/frozen** (2026-07-13): largest-region-crowns tie rule
  (max crowns among maximal-area regions); open-frontier = unique bbox-admissible empty
  cells per terrain; `enclosed_single_holes` (renamed); `game_progress =
  ((occupied_non_castle/2)+discards)/48`; normalizations are fixed catalog/rules constants
  in the schema hash, with range assertions + clip-frequency measurement.
- Board feature join: **joint `(cell, half_type)`** vs separate `(cell,terrain)+(cell,crowns)`.
- Accumulator width: **256** (sweep 128–512).
- Frame: **two shared-weight perspective accumulators**.
- Rollback: **copy-on-make** first.
- Labels: **exact-tiebreak** (pending your confirmation).
- Numeric: **float first**, quantize in v3.2 with overflow tests.

## Changelog vs Rev 2

- **Deltas are semantic** (`SemChange{owner,cell,half,…}`), mapped into both accumulators;
  the single-index-list `FeatureDelta` was wrong for dual perspectives.
- **Summary schema pinned** (143 dims) with exact definitions, dims, norms — anchored on
  the existing exact `_encode_board_summary` (25/perspective) plus a specified extension;
  loose "holes/stranded/mobility/~80" removed. **Three definitions remain flagged ⚠**
  (largest-region-crowns tie rule, open-frontier, enclosed-holes) pending confirmation
  before the schema is frozen.
- **Data stores replayable source** (deck/seed, full trajectory, official outcome +
  tiebreaks, engine/catalog hash, provenance); active indices/canonical state are derived.
  Prevents the run10 encoder-lock trap for future schemas.
- **Completeness gate redesigned**: fingerprint reconstruction + one-field mutation pairs
  + refresh/delta fuzz + bounded exhaustion. Dropped the infeasible "enumerate reachable
  states" and the incorrect "modulo symmetry" qualifier.
- **Rules-config flags + explicit terminal** added to the core (was silently config-bound;
  phase bank had only 3 phases). Core 5,707 → **5,710**.
- **Summary made perspective-relative** (`summary(state, perspective=actor)`, blocks
  ordered `[my, opp]`, claim owners my/opp) — closes the seat-asymmetry hole where the
  171-value tail could reintroduce absolute-seat bias the sparse half avoids.
- **Seat-swap invariant + inventory invariant** added as gates (build `swap_players`
  before generating data; `encode(s,P0)==encode(swap_players(s),P1)` for sparse *and*
  summary, with the enumerated coverage cases).
- **Count table fixed**: board ≤96 active/perspective (was ≤48), pending ≤4 (was ≤2);
  ~110 mid-game with ≲157 stated as a **loose, non-coexisting** upper bound (was quoted as
  the max); flip counts labeled per-accumulator vs total.
- **RAII cleanup contract** added as a gate (make ↔ unmake+pop on every exit path).
- **Perf headline softened** ("removes the dominant cost; throughput TBD by profiling").
- **Training pipeline note**: packed/CSR + `EmbeddingBag(sum)` first layer; never a dense
  N×5,710 matrix (~11.4 GB).
- **Data plan upgraded to "enhanced Option A"** (measured 0.143 s/game; exploration,
  whole-game splits, provenance, deeper-search minority, solver relabeling) with an
  explicit hybrid/iterative path before any strength claim.

### Rev 3 patch (post-freeze-audit review)

- **Completeness reframed as a Markov quotient**: lossless for the *future-relevant public
  state*, not byte-lossless; intentional omissions (deck order, pre-`actor_index` resolved
  entries, placed domino IDs, exact discard count beyond binary, `start_player`)
  documented + each gets an invariance test.
- **D4 transform correctness made a real gate** with the encode-a-rotated-state test, and
  a **verified existing bug** flagged: `augmentation.py` treats `flat` (incl.
  `width`/`height`) as invariant, mis-augmenting even the dense net under 90°/270°.
  `bbox_room` added to the transforming fields.
- **Official-outcome routing (score→largest territory→total crowns→draw) + actor
  attribution** made pre-generation gates; noted the cascade already exists
  (`official_outcome*`) but self-play label paths write score-only.
- **game_progress redefined** to `(placed+discarded)/48` (was `placed_cells/96`, never
  reaches 1.0 with discards + duplicates fill_ratio).
- **Added gates**: transition truth table / round-boundary conservation; information-set /
  chance equivalence (shuffle hidden deck → identical encoding + chance multiset); summary
  normalization audits (dedup legal counts, clip-frequency, unique-cell frontier); rare-
  state coverage pilot. Priority order: D4 → labels/actor → conservation → info-set →
  normalization/coverage.

### Rev 3 patch 2 (post-D4-fix review)

- **D4 bug FIXED and shipped** (Rust `transform_flat` + numpy `_transform_flat`, all three
  call paths, tests updated). Rust `d4_augment` now **validates board/flat/policy lengths**
  (raises `ValueError` instead of panicking on malformed direct input — closes a small
  regression the fixed-offset swap introduced).
- **Middle Kingdom correction**: a discard kills **Harmony only** (needs occupied==49), not
  Middle Kingdom (7×7 centered bbox, no full-fill requirement — captured by cell geometry).
- **Inventory invariant tightened** to `row ∪ pending[actor_index..] ∪ next ∪ bag` and
  upgraded to a **48-ID trajectory ledger** (bag/row/pending/next/placed/discarded,
  reconciled every transition) — container-duplicate checks alone can't catch a vanished ID.
- **Network-output seat-swap gate** added (`actor_value(s)==actor_value(swap(s))`,
  `p0_value(s)==−p0_value(swap(s))` with fixed weights) — catches z_actor/sign-flip bugs.
- **Draws rule sharpened** (not "reversed"): cascade-resolved score ties become decisive;
  genuine post-cascade ties stay 0.5 and stay in the dataset.
- **Chance-equivalence gate bounded** (analytic/seeded for big bags; exact enumeration only
  for late bags). **Correctness fixtures separated from dataset coverage** (rare cases are
  mandatory gates on constructed states; coverage measured via histograms + injection).

### Rev 3 patch 3 (feature-addition review — summary 143 → 171 + aux heads)

- **`pick_pos[4]`** next-round draft order restored (owner of each next-round pick;
  D4-invariant) — a sort the accumulator sum can't reconstruct.
- **Claim slots made self-identifying** (+12 → 24): added `owner_role`,
  `draft_priority_rank` (tempo, not content), `turn_distance` per slot — fixes the
  accumulator-to-legal-count binding in the area that corrupted run10.
- **`bbox_room` replaced by castle-relative extents** (dimension-neutral): `castle−min` /
  `max−castle` per axis carry the split width/height lose; better for Middle-Kingdom &
  denial; clean D4 permutation rule.
- **`largest_crownless_region[6]`/player** added (dormant potential) — flagged **first
  ablation candidate**.
- **Auxiliary training heads** (per-player final territory/largest/crowns + Harmony/Middle
  bits): regularize via outputs, dropped at export (zero inference cost); bonus heads kept
  low-weight (rare-event imbalance). Requires storing final `ScoreBreakdown` per game.
- **Pre-registered ablation** on a reserved untouched split to measure which handcrafted
  inputs are load-bearing vs carried by the aux heads.
- Endorsed the reviewer's **defer** (legal-count enumeration would wreck leaf speed; rest
  derivable) and **do-not-add** (info-set/symmetry/history/generator-artifact) lists in
  full; **generator provenance stays metadata, never a model input**.
