# Kingdomino NNUE — Project Plan

Goal: build a CPU-only NNUE evaluation + alpha-beta/expectiminimax searcher for
2-player Kingdomino, then use it to (a) **train** the AlphaZero net with
higher-quality / independent targets, (b) **combine** it into the HOF gauntlet as
an architecturally-independent exploiter, and (c) **help** as an analysis tool.

## Current status (2026-07-14)

The authoritative frozen encoder/inference detail is in `NNUE_STEP3_FEATURES.md`.
Phase 0 search, mutable Rust make/unmake, generic search traits, dense representability,
the 5,710+171 sparse encoder, replayable data harness, packed `EmbeddingBag` training,
v3 sparse export, stateless Rust forward, and reversible dual accumulators are complete.
The 50k pilot learned above baseline (validation Brier 0.2115 vs 0.2500; margin MAE
16.62 vs 17.81), with the reserved test split still unopened.

The v3.1 float-inference pass is complete without changing the schema, artifact, or
trained weights. Dense-tail weights are transposed to input-major layout once at load
for SIMD-friendly output updates; summary construction now shares one region traversal
per board, fixed frontier bitsets, and cached legal-placement counts. On the same
six-position depth-3 `choose_action` gate, incremental throughput rose from **33.5k to
94.3k nodes/s (2.81×)**; the stateless oracle reaches 87.1k. Per-eval profiling fell
from roughly 3.8/10.5/14.2 µs to **2.58 µs sparse derive+sum, 3.60 µs summary, and
1.21 µs float tail**. The accumulator itself is now a modest 1.08× over stateless;
rollback union-find remains unjustified.

v3.2 is also complete. `sparse_nnue_q` derives an int16 accumulator + int8 tail from
the unchanged v3 float artifact at load time, retains the float path as its oracle, and
uses an AVX2 integer-dot kernel with a scalar fallback. A top-112-feature proof bounds
the pilot accumulator at **19,958 < 32,767**. Across 100 real positions, quantized vs
float expected-score MAE/max are **0.0028/0.0109** and margin MAE/max are
**0.13/0.43 points**. Root actions agree 20/20 at depth 2 and 6/6 at depth 3. The fixed
six-position depth-3 gate reaches **102.3k nodes/s** vs 98.8k float (**1.04×**) while
roughly halving inference-weight storage.

The native operational searcher is now implemented. `RustSearch.choose_action_timed`
uses one shared wall-clock deadline, iterative deepening, last-complete-depth fallback,
root/PV-first ordering, root-sibling windows, aspiration re-search, a depth-aware
bound/exact TT at Kingdomino's canonical round roots, bounded Star1 chance pruning,
and exact horizon extension through deterministic deck-in-{0,4}/final tails. The tail
extension searches the generic official-outcome objective to GAME_OVER rather than
calling the legacy raw-margin solver, so score ties retain the largest-territory/crowns
cascade. Telemetry reports completed depth, timeout, total/final-iteration nodes,
Star cutoffs, TT hits/cutoffs, aspiration re-searches, and exact extensions. A bot_match
adapter (`OperationalRustSearchBot`) makes this the playable path.

Native release gates are green (**45/45 Rust tests**): deterministic node-budget
timeout/unwind, aspiration failure re-search, Star1, TT reuse, exact-tail ply counts on
real games, and fixed-vs-operational move equivalence. On the real sampled-chance gate
at depth 5, operational search used **8.79M total nodes / 9.321s** (including depths
1-4) vs fixed depth-5's **9.79M / 10.115s**; its final iteration was 8.55M nodes
(14% below fixed). The isolated release-wheel Python gate now passes **26/26** tests,
including the bot adapter and a timed `sparse_nnue_q` move matched to fixed-depth.
The live web-server extension was not replaced. On six representative positions the
quantized timed path sustained roughly **101k-126k nodes/s**: every 0.5s move completed
depth 3; at 1.0s, five completed depth 3 and one depth 4; at 2.0s, three completed depth
3 and three depth 4. All intentionally timed out while returning the last complete
iteration. Next: matched-clock NNUE-vs-AZ strength measurement. The original phase
descriptions below remain planning history; these measurements supersede their
bottleneck predictions.

Motivated by: the Rzepecki 2025 Azul MSc thesis (alpha-beta + tiny NNUE beat MCTS
and reached superhuman 2p play) and our own run5/run10/run11 verdict — data
exhaustion + a monoculture of AlphaZero-lineage agents. An NNUE searcher attacks
both: it searches differently, evaluates differently, and is *exact* in endgames
where MCTS approximates.

---

## What NNUE actually is (the four ideas)

1. **Sparse binary features + a giant first layer.** Describe the position as a
   huge list of yes/no features (e.g. "my board has a wheat+1-crown tile at cell
   (3,4)?"). Input is huge but only a few dozen features are "on."
2. **The accumulator (the defining trick).** The first layer maps that sparse
   vector to a modest dense vector (say 256-wide). Because inputs are one-hot,
   the matmul = "sum the W1 columns of active features." When a move flips a few
   features, you **add/subtract just those columns** from the previous
   accumulator instead of recomputing — a few vector ops per move. "Efficiently
   Updatable."
3. **Tiny dense tail.** After the accumulator: 256→32→32→1 or smaller. Cheap.
4. **Integer quantization + SIMD (AVX2).** int8/int16 weights & activations →
   tens of millions of evals/sec/core. **Kingdomino caveat (Azul lesson): NO
   clipped ReLU** — clipping destroys point-magnitude info in scoring games. Use
   plain ReLU with careful scaling (+ tanh on the output for [-1,1]).

Why it wins *for search*: alpha-beta with a cheap eval searching deep can
out-calculate MCTS with a rich-but-slow eval searching shallow. Depth × cheap
beat shallow × smart in chess, shogi, and (per the thesis) Azul.

**Kingdomino wrinkle — connectivity scoring.** Score = Σ region_size × crowns is
a *global* function a shallow per-cell MLP represents poorly. Mitigation: engine
maintains region sizes via **incremental union-find** (placements only ever merge
regions → cheap on make; unmake via union-by-rank-without-path-compression +
undo stack). Feed "current exact score margin" as an input feature so the net
only learns the *potential* on top of a known base.

---

## Assets we already have

- **Rust engine** with `encode_state` (bit-exact ported). **CORRECTION (review
  2026-07-12): `step` is FUNCTIONAL — deep `cloned()` per child, NO make/unmake**
  (lib.rs docstring). So rollback union-find + accumulator undo are NOT hooks into
  an existing mechanism; they need a state-mutation/undo refactor. That refactor is
  also likely needed for the searcher itself (deep-clone-per-node is too slow at
  target node rates), so it's load-bearing, not just an accumulator nicety — but
  **profile first** (≤49 cells: clone/hash/legal-gen/chance-expansion may dominate
  the flood-fill; let the measured bottleneck order the work).
- **Solver-restructure branch**: `ExactPolicyMode` + transposition table — the
  alpha-beta skeleton. NOTE: `solve_endgame_ab` optimizes RAW MARGIN (s0−s1) and
  calls the winner by margin sign — **exact for margin, NOT for the official
  outcome** (`determine_winner` cascade: total → largest_territory → crowns). At
  equal total score it mis-ranks. **CORRECTION (Phase-0 review): a single scalar
  `outcome·BIG + margin` is NOT lexicographic under chance** — expectiminimax
  averages it to `BIG·E[outcome] + E[margin]`, a weighted tradeoff, so a small
  expected-outcome edge can be overturned by expected margin. For strongest
  standalone play the objective is **expected match outcome**: terminals return the
  official result {+1,0,−1} and the horizon eval is a *bounded* proxy (`tanh margin`
  in (−1,1)) so proven results always dominate. Margin is then a training signal /
  optional documented blend (`margin_weight`), not a silent tiebreak. (`expectiminimax.py`
  implements this.)
- **Labeled buffers**: run10 = 200k positions (balanced 49.7/1.8/48.5 W/D/L, `z`
  144-valued, margin std≈19 range ±80, own/opp scores present, HOF game_types).
  run8/run9/local_run10 add ~1M more.
- **ELO round-robin / gating harness + BGA ELO scraping.**
- **AlphaZero policy head** (3390-wide) — a ready-made move-orderer for alpha-beta.

---

## What each phase de-risks (risk-retirement ordering)
The project is sequenced so the CHEAPEST experiment retires the most uncertainty
first and REDIRECTS later work — you don't build the expensive, project-specific
parts (encoder/accumulator, Rust) until a cheap experiment says they're worth it.
So "we haven't built the encoder or Rust yet" is BY DESIGN, not an omission.

| Risk | Retired by | Cost |
|---|---|---|
| Can we do CORRECT expectiminimax over Kingdomino's chance structure without clairvoyance? (the thesis's flagged bottleneck) | **Phase 0** (done) | days, pure Python |
| Where does strength come from — eval content vs raw depth? | **Phase 0 probe** (done: eval dominates at feasible depth) | free (a control run) |
| Can a small STATIC learned eval represent Kingdomino value well enough? (connectivity scoring, pick value) | **Phase 1** — dense MLP on run10 buffer, dropped into the Phase-0 harness | hours, pure PyTorch, NO Rust |
| Can the incremental accumulator be made bit-exact AND fast? | **Phase 2 complete** — correct; only ~1.02× at depth 3 because tail/summary dominate | weeks, Rust |
| Does the whole thing beat / complement AZ? | Phases 4–5 | ongoing |

**Phase 0 is NOT a feasibility test of the NNUE representation or performance.** It
tests the SEARCH layer (correctness + chance handling) and is the harness every
later eval — including the eventual NNUE — plugs into to be measured. Its "eval >
depth" finding is depth-2-limited and uses hand-crafted evals, so it does NOT prove
a *learned* eval will work.

### NNUE Rust engine vs the existing AZ engine (reuse vs new work)
They share the RULES layer but are DIFFERENT EXECUTION MODELS for opposite search
paradigms. AZ self-play = THROUGHPUT: bottleneck is the GPU net, modest nodes/move
each expensive, so clone-per-node is fine and you batch many parallel game slots +
coalesce leaf evals + release the GIL. NNUE alpha-beta = LATENCY/DEPTH: bottleneck
is per-node CPU cost, millions of nodes/move each a cheap CPU eval, so clone-per-node
is fatal. So it's a NEW path on the SAME rules, not a tweak (hence "refactor").

REUSE (substantial — de-risks it):
- Rules/board/legality/scoring (`RustBoard`, `legal_placements`, `place`, `scores`).
- **The endgame alpha-beta is the seed**: `solve_endgame_ab` + TT (128-bit xxh3) +
  move ordering + YBW parallel (`solve_endgame_ab_parallel`) + wall-clock budget.
  The NNUE searcher GENERALIZES it from the no-chance tail to the whole game.
- **Incremental scoring partly exists**: `placement_score_delta`/`terrain_group_delta`
  already compute the connectivity score DELTA on placement.
- Encoding machinery; the Python-equivalence test methodology.

NEW work (the actual differences):
- **make/unmake + undo stack** — even `solve_endgame_ab` clones (`state.step()`);
  ~1μs/node is fine for the budgeted endgame tail, fatal whole-game. No undo path
  exists. Needs reversible place/deck-draw/claim + score-undelta (persistent
  union-find w/ rollback, vs today's per-call local flood-fill).
- **In-tree chance in Rust** — today the Rust AB is NO-CHANCE only (`_is_rust_no_chance_state`,
  deck∈{0,4}); chance is enumerated Python-side. NNUE needs sampled chance nodes +
  Star1/Star2 INSIDE the Rust search.
- **CPU heuristic eval in-search** — AZ evals via GPU net on batched leaf tensors;
  the existing engine has NO heuristic-CPU-eval-in-search path. NNUE = incremental
  accumulator (SIMD), the accumulator-last piece.
- **Concurrency** — parallel game slots + GPU leaf coalescing → single deep search /
  Lazy-SMP-YBW alpha-beta (partly prototyped in `solve_endgame_ab_parallel`).
- Leverage note: the endgame solver's ~1μs/node is `step()` (clone) + ordering
  (legal-gen), NOT scoring (already delta'd) → confirms make/unmake + fast legal-gen
  are the leverage, accumulator secondary. Reinforces engine-first below.

### REVISED SEQUENCING (2026-07-12): Rust-first, accumulator-last
Decision (user, agreed): a Python dense-eval Phase 1 can't answer the real
question, because NNUE's value is *deep search × cheap eval* and depth-2 Python
tests neither. Representability is also partly pre-answered by the existing AZ value
head. And the Rust engine is (a) the MEASURED bottleneck (Phase 0: 6.7k nodes/s,
engine-bound), so it's on the critical path regardless, and (b) reusable
infrastructure for future board-game projects even if this experiment fails. So:
1. **Rust ENGINE first** — make/unmake, fast legal-gen, incremental scoring. This is
   the bottleneck; it lets the EXISTING `ExpectiminimaxBot` + a hand-crafted eval
   reach depth 4–6 and get a first "does deep search approach/beat AZ?" signal.
2. **Eval next** — train on the EXISTING dense encoding first (no new sparse feature
   design needed for a first pass); run it in the fast Rust searcher vs AZ.
3. **Sparse NNUE accumulator LAST** — it is a *performance optimization* (max
   nodes/s), not a prerequisite for a strength signal. Build it once steps 1–2 show
   eval×depth is worth maximizing. This defers the fiddliest engineering (rollback
   union-find, quantization, overflow) until there's evidence, while the reusable
   engine + a strength signal land early. NB: "large input layer" isn't the point —
   *sparse + incrementally-updatable + expressive* is; bigness is incidental.

## Phases

### Phase 0 — Search harness + trivial eval  (~1 wk)
- **Chance model is settled: composition known, only ORDER uncertain** (engine's
  `redeterminize` preserves the deck multiset; encoder is info-set safe). So this
  is a *stochastic perfect-information* game (backgammon-class), NOT a POMDP — the
  tool is **expectiminimax**, not belief states / PIMC-over-full-orders (which
  suffers strategy fusion). `endgame_solver.py` already IS exact expectiminimax
  (`itertools.combinations(deck,4)`, `expected += p*solve(child)`) — extend that
  pattern from the tail to the whole game with a depth horizon.
- Chance node at each round boundary = uniform draw of next 4-subset; branching =
  C(remaining,4). **CORRECTION (review): 48-domino game, so the OPENING is far
  wider than earlier stated** — after the setup row, 44 unseen: C(44,4)=135,751 →
  91,390 → 58,905 → 35,960 → 20,475 → 10,626 → 4,845 → 1,820 → 495 → 70 → 1. Exact
  enumeration is a LATE-game advantage; the opening is genuinely hard and sampling-
  bound (reinforces that chance handling is the dominant difficulty).
- **Depth-limited expectiminimax**: explicitly branch chance nodes only within the
  search horizon; below it the NNUE eval integrates over future orders. **CORRECTION
  (review): the eval is not literally `E[value | public state]`** — info-set safety
  only blocks deck-ORDER leakage; a net trained on realized AZ games *approximates*
  expected value under its training distribution / policy / loss, and is unreliable
  off that distribution (exactly where alpha-beta wanders). So the leaf absorbs the
  deep uncertainty only approximately.
- Wide early chance nodes: **sample k of C(N,4)** with common random numbers
  across sibling moves; prune with **Star1/Star2 \*-minimax** (our bounded eval
  gives the L/U bounds Star2 needs). Late nodes: enumerate exactly.
- Eval = raw score margin (+ trivial potential). Wire into round-robin vs a fixed
  AZ checkpoint → **baseline number**. Reuse solver-restructure TT + endgame_solver.
- **Deliverable:** a measurable expectiminimax searcher with a dumb eval.

**STATUS (2026-07-12): Phase 0 BUILT + tested + review-corrected.** `expectiminimax.py`
(`ExpectiminimaxBot`), `test_expectiminimax.py` (9 pass). Depth-limited
expectiminimax, player-0 frame, **expected-outcome objective** (terminals {+1,0,−1};
bounded `tanh margin` horizon proxy), alpha-beta on decision layers, chance nodes
enumerated when ≤enum_cap (**exact chance handling in the endgame** — the search is
still depth-limited unless depth reaches GAME_OVER) else MC-sampled with
deck-multiset-seeded CRN (explicit stable hash). Plugs into `bot_match.run_match`.
- **Two tests caught real bugs.** (1) clairvoyance: wide-chance sampler `random.sample`d
  the *unsorted* deck → value leaked hidden order (fixed: sample sorted bag). (2) the
  `outcome·BIG+margin` scalar was a blend not a lexicographic tiebreak (fixed:
  outcome-only utility). `test_alphabeta_equals_unpruned_reference` now proves
  alpha-beta == full expectiminimax; deck-order-invariance + constructor-validation
  guarded too.
- **Perf (confirms review):** pure-Python ~6.7k nodes/s → **depth 2 is the ceiling;
  depth ≥3 needs the Rust mutation refactor.** depth1 0.01s, depth2 ~1s, depth3 ~23s/move.
- **Baseline (pick-blind margin eval):** beats Random 18–2; loses to Greedy at both
  depth 1 (~33% WR) and depth 2 (~28% WR, avg 86 vs 99). **Caveat (review finding 3):
  small unpaired samples, and this only shows raw board score is an inadequate leaf
  eval — NOT that a learned eval specifically is needed.** The proper control is a
  trivial PICK-AWARE eval (`pick_aware_p0`): `run_phase0_control.py` runs blind vs
  aware PAIRED (same seeds, both seats).
- **CONTROL RESULT (paired, depth 2, N=24 per config):** pick-BLIND 5/24 (~21%,
  avg 91 vs 105); pick-AWARE **20/24 (~83%, avg 104 vs 85), symmetric 10–2 in BOTH
  seats.** One crowns-on-claimed-dominoes term flips a lopsided loss into a lopsided
  win. **Verdict (review finding 3 vindicated): the bottleneck was leaf-eval
  pick-blindness, NOT search depth — and a *trivial* pick-aware eval already beats
  Greedy, so a learned NNUE is NOT required merely to clear the Greedy bar.**
- **Revised recommendation:** Greedy is now too weak a bar. `pick_aware_p0` becomes
  the competent Phase-0 baseline; the real question for Phase 1 is whether a learned
  eval beats **the AZ agent** (and generalizes past hand-crafted pick heuristics).
  Cheap next measurement BEFORE committing to Phase 1: pit pick-aware-EMM(d2) vs the
  AZ agent to get a real reference point.

### Phase 1 — Distilled NNUE eval, dense prototype  (~1 wk, pure Python)
- PyTorch MLP on existing buffer tensors → **two SEPARATE heads** (review: these
  are NOT interchangeable): **official-outcome head** ← `win_target` (1/0.5/0), and
  **margin head** ← `own_score − opp_score`. `z` is itself a tanh(margin) blend, so
  don't treat it as the outcome label.
- Validate *before* any Rust work: **outcome head by Brier / log-loss; margin head
  by MAE + ranking** (not ranking vs `z` alone). Does it beat raw-margin eval in
  Phase 0?
- **Splits by whole game / trajectory (ideally by run+seed)** — random-position
  splits leak dozens of correlated states from one game across train/val.
- **Deliverable:** a validated eval function, hours of training.

### Phase 2 — Sparse feature set + Rust accumulator  (~2–3 wk, the real work)
- **Feature set is mostly designed already** — steal `encoder.py`'s proven layout:
  per-cell terrain/crowns (→ sparse one-hots for the accumulator's cheap add path)
  + `board_summary` (total score, score_by_terrain, largest_by_terrain,
  total_crowns, harmony, middle, bbox, legal-placement count) + bag membership +
  pick-interleaving (pick_pos_0..3) + phase/progress/fill_ratio.
- **THE connectivity fix (see below): `_board_component_facts` currently
  flood-fills from scratch every encode.** At alpha-beta speeds that's fatal.
  Convert it to an **incremental union-find** maintained on make/unmake (placements
  only ever MERGE → cheap+exact forward; unmake via undo stack, union-by-rank w/o
  path compression). This is the core new engineering.
- Keep summaries accumulator-friendly: either a small real-valued feature block
  (scalar-multiply updates, few features → negligible) OR one-hot/bucket them
  (Azul-thesis style). Both preserve incrementality.
- **Additions worth sweeping** (not in current encoder): per-terrain open-frontier
  / expansion room, deck aggregates by (terrain × crown-level), crownless-region &
  stranded-crown mismatch, per-terrain region count (fragmentation).
- Implement accumulator with make/unmake; quantize (int16 acc, int8 tail; plain
  ReLU). **Verify** Rust eval ≈ PyTorch net within quant tolerance on a large batch
  — same discipline as the encode_state / MCTS bit-exactness milestones.
- **Deliverable:** fast CPU eval matching the trained net.

### Phase 3 — Fast searcher  (~1 wk)
- Accumulator eval into Phase 0 searcher.
- Move ordering (TT hash move → AZ policy head), iterative deepening, aspiration
  windows.
- **Endgame exactness:** once the final round's dominoes are revealed the tree is
  exact — solve, don't estimate. (A structural edge over MCTS.) Already done today
  via `solve_endgame_ab` at deck=4 (400/400 game endings exact-solved).
- **NNUE pushes the exact-solve frontier EARLIER — via ORDERING, not bounds.**
  CORRECTION (review): a heuristic NNUE eval is NOT an admissible bound, so it
  CANNOT justify exact alpha-beta cutoffs; exactness still needs full enumeration
  to terminal. What NNUE legitimately buys: better **move ordering** (→ more cutoffs
  → the same exact solve fits in budget from a slightly larger/earlier root) and
  **extended approximate** depth-limited search. The doc defers "policy-prior
  ordering" — NNUE value/policy fills it; with the 62–86% transposition re-entry,
  ordering+speed could move the *practical* exact frontier toward the last 2–3
  rounds. The eval extends *approximate* search; it never *certifies* an earlier
  exact frontier.
- **Deliverable:** re-measured ELO; ideally > the AZ checkpoint at matched wall-clock.

**STATUS (2026-07-14): operational search machinery complete and gated.** The
remaining Phase-3 deliverable is experimental, not engine construction: play the
paired matched-clock NNUE-vs-AZ bar and re-measure strength.

### Phase 4 — Training loop (self-generated data)  (ongoing)
- Azul methodology: searcher self-plays → positions labeled with deep-search
  values → retrain → select if stronger → repeat.
- **This is what breaks past run10's coverage ceiling** (200k ≈ only a few
  thousand correlated trajectories on one policy's support; alpha-beta walks
  off-distribution where a static-buffer eval is blind).
- Reuse existing self-play orchestration + gating.

### Phase 5 — Combine with AlphaZero (the payoff)
- **TRAIN (highest strategic value):** relabel/augment AZ training targets with
  deep NNUE-alphabeta search values — higher quality than 800-sim MCTS *and* from
  a different search process → genuinely new signal into an exhausted pipeline.
- **COMBINE:** enter NNUE-alphabeta as an independent HOF/exploiter agent. run11a
  said "locally unexploitable, but all attackers share AZ lineage." An
  independent searcher is the honest exploiter the monoculture lacked — holes
  found = real; none found = a much stronger unexploitability claim.
- **HELP:** position eval / endgame solver / puzzle generator / blunder explainer
  (falls out for free).
- **Hybrids (later, don't front-load):** NNUE exact endgames + MCTS midgame; NNUE
  value as MCTS leaf eval; AZ policy as alpha-beta orderer.

---

## Chance handling — what the Azul thesis actually did (directly transferable)
Azul faced our exact problem (known bag composition, unknown future dispersement).
It did NOT do exact expectiminimax. At each chance node its `branching`
enhancement:
- **Pre-generates** `branchingFactor` sampled draws *before* the search → the tree
  is deterministic and TT-consistent, and sibling moves are compared against the
  *same* futures (= common random numbers, free variance reduction).
- Aggregates child values with `branchingMethod`: `arithmeticMean` (unbiased E),
  or `median`/`truncatedMean`/`winsorizedMean` (robust, lower-variance, slightly
  biased) to tame outlier draws.
- **Exploits known composition for variance reduction** via
  `prepareNewRoundDataGenerationPolicy`: `random` (true uniform, high variance) vs
  `semirandom` (pin per-color *quantities* to the bag's expectation, then randomize
  positions) vs `evenly` (also match per-slot distribution to expectation). The
  latter two cut variance hard by sampling *near the bag's mean* instead of freely.

They flagged this as their **suspected bottleneck** (future work). We can beat it:
our chance branching C(remaining,4) *collapses* (10626→…→70→1) so we get **exact
expectiminimax in the endgame** (Azul's 20-from-big-bag couldn't), sample +
variance-reduce early, and add **Star2 pruning** they never tried.

**Calibration guardrail (we already hit this): a deterministic WORLD is fine; a
deterministic FUTURE inside the search is not.** The world (self-play game) must
have a fixed shuffle — you need concrete games. The danger is letting the search
*condition on* that one order → strategy fusion / clairvoyance → over-confident
values → a net that learns false confidence (our early deterministic attempt).
Three separable safeguards, all required:
1. **Encoder order-blind** (have it: reads bag membership, never order).
2. **The invariant is STRUCTURAL, not "K>1"** (review sharpened this): decisions at
   identical public states must NOT depend on which sampled hidden scenario led
   there. Sampling must be *chance nodes inside one public-state tree*, with
   identical info-states sharing one decision. Solving K complete sampled deck
   orders separately and averaging root values is PIMC and fuses strategies for ANY
   K (2 or 200). Azul is safe because its `branchingFactor` samples are consumed AT
   the chance nodes of one expectiminimax tree, not as K full-info solves. K>1 +
   common random numbers is good variance reduction, but the tree structure is what
   prevents fusion.
3. **Targets = honest outcome samples or expectiminimax values, NEVER a
   future-revealed solve.** *Where you average matters*: at chance nodes inside one
   tree = correct info-set value; at the root over fully-solved deterministic
   worlds = PIMC = fuses strategies = false confidence. Our buffer's z/win_target/
   own_score are the safe kind (a realized game is one honest draw from the outcome
   distribution; the dataset averages them out).

## Two distinct problems — don't conflate
- **(a) Calibration / clairvoyance** (our early deterministic attempt → false
  confidence): a *correctness* bug, fixed by order-blind encoder + averaging at
  chance nodes. SOLVED by the guardrail above.
- **(b) Azul's "draw simulation is the bottleneck"**: a *variance/efficiency*
  ceiling. Their final system already did correct sampled in-tree expectiminimax
  (variance-reduced draws, pre-generation) — it wasn't miscalibrated; it was that
  covering a huge draw space with few samples is noisy/expensive, and robust means
  are a biased band-aid. Being calibrated does NOT solve (b).
- **Our methodology ≈ Azul's for the general case.** What softens (b) for
  Kingdomino is *game structure*, not a cleverer algorithm: C(remaining,4)
  collapses → exact enumeration late + exact endgame (Azul's 20-from-big-bag
  never could); draws are 4-tile objects (lower per-node variance); + Star2 chance
  pruning they likely lacked; + delegating draw-averaging to the learned eval. The
  OPENING still samples → the bottleneck shrinks, doesn't vanish.
- **Real edge = measurement.** They *suspected* draw handling was the bottleneck
  but didn't isolate it. We make draw-handling an explicit swept axis
  (branchingFactor, enum threshold, sampling scheme, Star2 on/off, eval-delegation
  depth) measured on the round-robin — suspicion → knob.

## References
- **Yu Nasu, "NNUE" (2018)** — original Shogi paper; the foundational text.
- **Stockfish `nnue-pytorch` + `docs/nnue.md`** — canonical trainer + the clearest
  written spec of accumulator/quantization: github.com/official-stockfish/nnue-pytorch
- **Bullet** (Jamie Whiting) — modern fast **Rust** NNUE trainer, simpler than
  Stockfish's; best fit for our Rust stack.
- **dhbloo/pytorch-nnue-trainer** — NNUE for **Gomoku/Renju**, a rare *non-chess
  board-game* reference.
- **asdfjkl/nnue** + **beuke.org/nnue** — minimal/educational implementations.
- **"Neural Networks for Chess" (arXiv 2209.01506)** — free book; NNUE + AZ + search.
- **"Study of the Proper NNUE Dataset" (arXiv 2412.17948, Dec 2024)** — directly
  relevant to our buffer/data-coverage question.
- **chessprogramming.org/NNUE** — practical reference. Small clean Rust engines to
  read for the accumulator: Viridithas, Carp, Stormphrax.
- NOTE: **Stockfish = alpha-beta + NNUE, NOT AlphaZero.** The AlphaZero-style chess
  engine is **Leela Chess Zero (Lc0)**. Our project is literally "bring the
  Stockfish recipe (cheap CPU eval + deep search) alongside our existing Lc0-style
  agent" — the two architectures our own pipeline has been missing.

## Long-horizon denial + honest success criteria
**Why AZ likely missed 3+ round opponent denial:** a self-reinforcing MCTS blind
spot. "Take a tile you don't need, to deny the opponent" looks bad to the policy
prior → low prior → few PUCT visits → never reinforced → self-play never generates
denial → value head never learns it. This IS the monoculture/data-exhaustion story.
**Alpha-beta breaks the loop:** it examines the move to refute-or-confirm regardless
of prior, so if denial is actually best the search returns it. That's the single
strongest argument for the project.

**But you don't get 3+ round denial for free — it's grown by bootstrapping, not
learned directly.** Three requirements:
1. **Opponent-need features:** opponent's incomplete/crownless regions, which
   remaining tiles complete them, contested tiles in the row, pick-order pressure.
   No eval can price denial it can't see.
2. **Iterated search→train ratchet:** search finds 1-round denial at its horizon →
   bake into eval → next iteration's search sees 1-round denial at horizon and
   discovers 2-round → … The horizon extends by TD/bootstrapping across loop
   iterations (how chess NNUE learns long concepts). Phase 4 is the mechanism.
3. **Adversarial/exploiter data** (Azul's "duel agents with move preferences/
   penalties" idea; our run11 exploiter machinery) to force denial into the buffer.

**Honest prediction (asked directly):** likely does NOT surpass the mature AZ net
as a *solo* player initially — Kingdomino's high joint branching + heavy chance
layer blunt alpha-beta's depth edge exactly in the opening/midgame where most of
the game is decided, and the eval it falls back on there is trained on the same
data as AZ's value head (no inherent edge). It WILL be stronger in the endgame
(exact) and will surface lines AZ is blind to. ~1-in-3 it eventually overtakes
outright with the full loop (Azul precedent — tempered: their MCTS was an
afterthought, ours is mature). **Solo dominance is the wrong bar anyway.**

**Generator bar (how strong is "useful"?) — much lower than "stronger than AZ":**
- **Endgame relabeler (near-free):** exact endgame values beat AZ approximations at
  ANY overall strength. Useful the moment the deep solver runs.
- **Midgame value relabeler:** useful once its search+eval gives *lower-error value
  targets* than N-sim AZ-MCTS on the same positions — value-target accuracy, NOT
  game strength (measure vs deep-solve ground truth on a probe set). Achievable
  well below AZ's playing strength.
- **Diversity generator / exploiter:** useful once it plays *competently* (positions
  realistic, ~within a couple hundred Elo) AND *diverges* from AZ's policy (covers
  under-visited lines) OR finds one repeatable exploit. Difference > dominance.
All three are directly measurable on the existing round-robin + run11 exploiter
harness (Elo gap, exploit win-rate, value-error vs ground truth).

## Review additions (2026-07-12) — architecture, ordering, Phase 5, success bar
**Architecture specs to pin down (a position-indexed MLP does NOT inherit the CNN's
symmetry/perspective handling):**
- **Evaluation frame:** fixed player-0-frame vs actor-relative (encoder is
  actor-relative `(my,opp)`) — decide and state it.
- **Two perspective accumulators?** (chess NNUE maintains side-to-move + opponent).
- **D4 symmetry:** explicit augmentation over all 8 orientations OR canonicalization
  — the MLP won't get it for free from convolutions.
- **Equivalence tests:** full accumulator refresh == incremental update after
  arbitrary make/unmake sequences.
- **Overflow tests BEFORE committing to int16 accumulators** — plain (non-clipped)
  ReLU + score features can grow large; Azul flagged quantization overflow risk.

**Move-ordering tension (review):** using the AZ policy head per node (a) makes
GPU calls that break the CPU-only premise, and (b) makes the agent not fully
architecturally independent (weakens the monoculture/exploiter test). → Evaluate
BOTH an **NNUE-only ordering** (honest independence test) and an **AZ-assisted**
variant (may be strongest). If AZ ordering helps, distill it into a tiny CPU
move-orderer rather than calling the GPU net in the search loop.

**Phase 5 — value relabeling alone will NOT fix policy-prior starvation** (the
denial blind spot is a POLICY problem). Alpha-beta must ALSO emit **sparse root
policy targets** from searched child values, with explicit tie / incomplete-search
/ uncertainty / temperature handling. Reuse the `ExactPolicyMode` machinery.
**Teacher labels carry provenance + quality** — {exact official-outcome solve |
complete sampled expectiminimax | depth-limited | static NNUE} + (chance samples,
completed depth) — and training weights exact/high-confidence labels higher.
NNUE-search values are NOT automatically better than 800-sim MCTS; only exact /
high-confidence ones are.

**Success bar (partial pushback accepted):** "solo dominance is the wrong bar" was
mis-framed. Standalone NNUE > current AZ agent is a **first-class success, possibly
the biggest** — just not the *only* early continuation bar and not my modal
prediction. Four independent wins: (1) better exact/endgame labels, (2) competent
divergent exploiter, (3) better AZ value+policy targets, (4) standalone player >
AZ. Judge (4) rigorously: paired seeds, both seats, CIs, **full HOF (not one
checkpoint)**, matched wall-clock AND deployment; report completed depth, nodes/s,
eval share, chance-node share, and AZ-assisted-ordering effect.

## Risks / decision points
- **Chance-node handling** is the biggest search-design risk — the thesis flagged
  its own naive chance handling as *the* bottleneck. Budget iteration
  (determinization count, variance reduction, Star2).
- **Connectivity nonlinearity** — mitigate with union-find features; measure
  whether the net needs them or learns around them.
- **Branching factor** — the joint pick+place space is large (3390-wide policy);
  alpha-beta depth depends on good ordering → lean on the AZ policy head.
- **Encoding bridge** — buffer stores AZ dense planes `(9,13,13)+flat(333)`, not
  the sparse NNUE layout. Convertible (planes → one-hot deltas), but Phase 1 can
  train on the dense tensors directly while Phase 2 designs the sparse set.
