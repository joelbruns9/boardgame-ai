# Kingdomino NNUE — Project Plan

Goal: build a CPU-only NNUE evaluation + alpha-beta/expectiminimax searcher for
2-player Kingdomino. The **ideal / stretch outcome** is a standalone NNUE search
agent stronger than the trained AlphaZero agent at an honestly matched clock. The
**practical primary outcome** is a competent, strategically different agent that
generates adversarial and underrepresented positions which, after high-quality
reanalysis, improve AlphaZero. Independent HOF exploitation, exact endgame labels,
and analysis tooling remain valuable supporting outcomes.

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

Quantized overflow safety is already **per artifact**, not a one-time pilot assumption:
`QuantizedSparseWeights::from_float` chooses per-channel power-of-two scales, recomputes
the conservative top-112 active-feature bound from that artifact's weights/biases, and
rejects a bound above `i16::MAX`; runtime construction also rejects too many active
features and checked full-refresh overflow. The **19,958** figure is the pilot's reported
instance, not a permanent constant. Every promoted artifact must instantiate
`sparse_nnue_q`, record `quantization_info`, and pass float-vs-quantized parity before
gameplay. Track headroom, but do not impose an arbitrary 20% rejection margin: the
dynamic scale is designed to use safe int16 range, and exact bound enforcement—not raw
weight magnitude—is the correctness condition. Score scalars live in the separately
bounded summary block, so higher game scores do not directly enlarge the sparse binary
accumulator.

The native operational searcher is now implemented. `RustSearch.choose_action_timed`
uses one shared wall-clock deadline, iterative deepening, last-complete-depth fallback,
full-width cheap heuristic ordering with root/PV promotion, root-sibling windows,
aspiration re-search, a depth-aware bound/exact TT at Kingdomino's canonical round
roots, bounded Star1 chance pruning,
and exact horizon extension through deterministic deck-in-{0,4}/final tails. The tail
extension searches the generic official-outcome objective to GAME_OVER rather than
calling the legacy raw-margin solver, so score ties retain the largest-territory/crowns
cascade. Telemetry reports completed depth, timeout, total/final-iteration nodes,
Star cutoffs, TT hits/cutoffs, aspiration re-searches, and exact extensions. A bot_match
adapter (`OperationalRustSearchBot`) makes this the playable path.

Native release gates are green (**47/47 Rust tests**): deterministic node-budget
timeout/unwind, aspiration failure re-search, Star1, TT reuse, exact-tail ply counts on
real games, and fixed-vs-operational move equivalence. On the real sampled-chance gate
at depth 5, operational search used **8.79M total nodes / 9.321s** (including depths
1-4) vs fixed depth-5's **9.79M / 10.115s**; its final iteration was 8.55M nodes
(14% below fixed). The isolated release-wheel operational/sparse/generation gate now
passes **32/32 Python tests**, including the bot adapter and a timed `sparse_nnue_q`
move matched to fixed-depth.
The live web-server extension was not replaced. On six representative positions the
quantized timed path sustained roughly **101k-126k nodes/s**: every 0.5s move completed
depth 3; at 1.0s, five completed depth 3 and one depth 4; at 2.0s, three completed depth
3 and three depth 4. All intentionally timed out while returning the last complete
iteration.

Full-width move ordering is now the NNUE operational default and remains completely
separate from selective pruning: every legal action is retained, ordering uses only the
game's cheap public-state heuristic, and telemetry distinguishes scored actions from
actual eval probes or pruned actions. On 20 representative positions at completed
depth 3, ordered and unordered search produced identical values **20/20** while ordering
reduced nodes **588,834 -> 496,342 (15.7%)** and wall time **6.506s -> 5.758s
(11.5%)**. Root actions agreed 19/20; the lone difference had exactly the same root
value and was an equal-valued traversal tie. The same-artifact, equal-0.1s-clock,
seat-swapped gameplay gate finished **32-32 over 64 games**, with ordered search at
**+7.14 average score margin** and effectively identical decision timing. The lower-
level Rust API still defaults to unordered search so it remains an explicit oracle and
benchmark baseline; the NNUE bot, match, and operational data-generation entry points
default to full-width ordering and record the choice in provenance.

The first clock-matched strength floor is now frozen: four paired seeds / both seats,
pilot NNUE at a 0.1s deadline versus the current AZ checkpoint at 13 simulations.
Actual non-forced decision means were **85.0ms NNUE vs 83.5ms AZ**. The pilot NNUE
lost **0-8**, average margin **-88.4**. This is a small baseline, not a strength CI,
but the gap is too large to explain by clock mismatch.

The Phase-4 MVP loop is also operational: incumbent-guided replayable self-play,
immutable shard mixing, warm-start training, frozen validation, export, paired
candidate gating, restart guards, and atomic run-local promotion. It never opens the
reserved test split. A 256-game pilot improved held-out Brier **0.2115 -> 0.2091** and
margin MAE **16.62 -> 16.31**, but lost **5-11** to its incumbent at equal 0.1s clocks;
the gate correctly rejected it. The result is important: static validation loss is not
a promotion proxy, and scaling shallow self-imitation is not the next move.

An opt-in selective-search experiment is now implemented and explicitly labeled in
telemetry. It keeps the root full-width by default, NNUE-orders deterministic upper-tree
actions, caps only eligible non-exact subtrees, keeps the tactical bottom full-width,
and never prunes exact deterministic tails. At a 1s clock, internal width 2 / bottom
one ply full completed **depth 4-6** versus full-width **depth 3-4**; an explicit top-8
root cap completed **depth 5-7**. Root widths 1-2 can display depth 8-12, but those are
principal-line/beam rollouts, not comparable to full-width alpha-beta. More importantly,
the honest same-net strength confirmation rejected the first full-root width-2 policy:
after an initial 11-5 result, it lost the disjoint 32-game confirmation 11-21; combined
**22-26 (45.8%, -2.9 average margin)** at identical 0.1s clocks. Selective mode therefore
remains a research/data-diversity knob, not the gameplay or training default. Next:
run the matched-clock scaling curve and the primary-goal disagreement/curriculum pilot
described below. Defer the larger stochastic-search and NNUE-training packages until
those cheap gates report. The original phase descriptions below remain planning
history; these measurements supersede their bottleneck predictions.

## Approved direction after the search/training review (2026-07-14)

The project now has **two connected tracks**, not an all-or-nothing bet on standalone
strength. Engineering and data artifacts should serve both tracks so that a negative
standalone result does not strand the work.

### Outcome hierarchy

1. **Stretch success — standalone superiority.** NNUE plus its CPU search defeats the
   current AlphaZero/open-loop-MCTS agent, then holds up against the full HOF, at paired
   seeds, both seats, and matched practical clocks. This remains the ideal result.
2. **Primary practical success — AlphaZero curriculum.** NNUE is competent and
   strategically divergent enough to generate realistic positions that AlphaZero
   self-play rarely reaches. Those positions are reanalyzed before use; AlphaZero is
   never asked to imitate a weaker NNUE action blindly.
3. **Independent success — exactness and exploitation.** NNUE supplies exact or
   higher-confidence endgame labels, a non-AZ-lineage HOF opponent, disagreement
   probes, puzzles, and analysis even if its whole-game Elo remains below AlphaZero.

The 0-8 pilot result does **not** decide feasibility. It compares a mature AlphaZero
agent with a 50k-position pilot NNUE before a serious self-play/reanalysis ratchet. A
fair standalone verdict requires both (a) a Kingdomino-appropriate stochastic-search
sweep and (b) at least two promotion-capable training iterations using materially
stronger and broader targets than the rejected 256-game self-imitation shard.

The Azul result is motivation, not a direct algorithm verdict. Its final strength came
from a fast integer/AVX2 evaluator, a heavily optimized alpha-beta engine, million-scale
datasets, many candidate models and duels, and repeated training rounds. Its abandoned
MCTS baseline used random rollouts, not a trained AlphaZero policy/value network with
open-loop MCTS. Conversely, our pilot is far too small to prove that NNUE cannot work.
The relevant experiment is our mature AZ against a fairly trained NNUE under
Kingdomino-safe chance handling—not paper labels or nominal depth comparisons.

### Risk ranking after implementation evidence

Separate **chance correctness** from **chance efficiency**:

- Chance correctness is implemented and gated: the encoder is hidden-order-blind;
  sampled rows come from the sorted bag with a bag-keyed seed; sibling decisions with
  the same remaining bag see the same sampled rows; chance outcomes are averaged inside
  one public-state expectiminimax tree; and late rows/exact tails are enumerated when
  feasible. Do not rebuild this as a belief-state or independent-determinization system.
- Chance efficiency remains measurable: sample count, variance, and where the horizon
  falls can still consume depth. Treat scenario tries and boundary evaluation as
  compute-allocation experiments, not correctness repairs.
- The **joint pick × placement branching factor is now the leading structural risk for
  standalone strength**. The current full-width ceiling (usually depth 3-4 at practical
  clocks), weak narrow-deep confirmation, and strong AZ policy prior all point here.
  Measure legal-action distributions and effective alpha-beta branching by phase rather
  than relying on a rough `100^4` estimate, which ignores legality contraction,
  ordering, transpositions, and pruning.

The selective result (22-26, -2.9 margin) is negative evidence for that one width-2
policy, not a proof that every selective or learned-ordering design fails. The 0-8 AZ
floor is likewise a directional small-sample result, not evidence that the gap must
widen with time. A clock-scaling curve is the correct cheap discriminator.

### Immediate work package P0 — validate the primary AZ-curriculum bet first

This package moves ahead of the standalone-oriented A/B/C packages. It uses the pilot
NNUE as a **disagreement hypothesis generator**, not as an authoritative teacher.

1. **Clock-scaling characterization.** Run matched practical clocks at
   `0.1 / 0.5 / 2 / 10s` for NNUE and AZ, with paired seeds/both seats and actual
   decision-time telemetry. Use a small screening set at all clocks, then a disjoint
   confirmation only where the trend could change a decision. Report NNUE completed
   depth/nodes/chance share and AZ simulations, plus win rate and margin. The question
   is whether the gap narrows, stays flat, or widens—not which agent wins one tiny set.
2. **Competence floor for generation.** At a generous offline NNUE budget, benchmark
   the pilot against Greedy, pick-aware search, and a mid-strength AZ checkpoint. Also
   report illegal/replay failures, score/discard distributions, and phase coverage.
   Whole-game superiority to current AZ is not required; positions must be legal,
   coherent, and nontrivial.
3. **Collect fresh replayable AZ trajectories.** The old run10-style encoded buffers
   cannot seed this miner: they have no move trajectories and lose exact domino ID
   information (33 compositions for 48 unique tiles). Extend/reuse the replayable-source
   harness to record AZ games as initial deck/row + actions + model/search provenance.
   Do not attempt lossy reconstruction of the million-position legacy buffers.
4. **Mine disagreements on AZ-supported states.** Replay fresh AZ trajectories and run
   offline NNUE full-width search at selected positions. Rank candidates by action
   disagreement, NNUE-estimated improvement over AZ's most-visited move, AZ entropy,
   value swing, novelty, phase, and exact-tail availability. Calling these candidates
   "refutations" is provisional: NNUE disagreement nominates a hypothesis; it does not
   prove the AZ move is wrong.
5. **Reanalyze both actions fairly.** On the top candidates, run high-budget AZ MCTS
   and exact search where feasible. A normal AZ rerun can reproduce policy starvation,
   so guarantee the NNUE candidate and AZ candidate adequate root-child evaluation
   (forced first-action searches, a temporary root prior floor, or equivalent explicit
   child-value probes). Keep ordinary high-budget AZ visits separate from forced-probe
   evidence in provenance. Reject candidates whose apparent edge disappears.
6. **Freeze 1-2k validated disagreement examples.** Split by whole source game/seed.
   Keep a frozen disagreement/exact probe subset out of training. Store the public state,
   both candidate actions, original and reanalysis visits/values, realized official
   outcome, exact status, and all teacher hashes/budgets.
7. **Run the falsifiable AZ curriculum experiment.** Fine-tune two AZ candidates with
   identical initialization, examples, updates, optimizer settings, and compute:
   control = ordinary replay replacement; treatment = the tagged 75/10/10/5 mixture
   below. Compare HOF/gauntlet strength, frozen disagreement/exact error, and closure of
   repeatable exploits. Lower loss alone is not success.

If the treatment helps, the project has delivered its primary practical goal and earns
scaling investment even if NNUE remains weaker head-to-head. If it does not, inspect
whether the failure is generator competence, disagreement precision, reanalysis label
quality, or replay mixing before producing the 5k-game teacher corpus.

### Standalone work package A — stochastic allocation (after P0)

The Azul result does not justify selecting one permanent representative four-domino
draw in Kingdomino. Azul tiles are exchangeable within colors and a refill can be
summarized by color quantities. Kingdomino dominoes are mostly unique; ID determines
draft order, and terrain/crowns/placement feasibility can make two superficially
similar rows strategically opposite. Our earlier deterministic-deck AlphaZero failure
is direct evidence that collapsing the future distribution can teach brittle policy.

Required invariants for every experimental mode:

- The encoder remains blind to hidden deck order.
- Every root action is compared against the same sampled public futures (common random
  numbers), so sampling noise is not mistaken for action quality.
- Decisions at identical public information states are shared. Never solve complete
  deck orders independently and average their clairvoyant root values; that is PIMC
  strategy fusion even when many deck orders are used.
- Samples respect without-replacement draws and the sorted four-domino row rule.
- Scenario seeds, draw probabilities, completed depth, sample count, and aggregation
  method are recorded in telemetry and training provenance.

Evaluate in this implementation order after P0; do not build every mode up front:

1. **Current sampled expectiminimax + existing late exact baseline.** Independently
   sample/enumerate chance children inside the public-state tree (`chance_samples=8`
   today), retaining the already-built switch to full row enumeration and official-
   outcome exact tails when feasible. Audit and measure this path; late exact is not a
   new subsystem to rebuild.
2. **Current-round search plus boundary evaluation.** Fully search the visible round
   and stop before the next unknown row, using an order-blind boundary value trained to
   integrate over the remaining bag. This sacrifices concrete next-round tactics but
   never invents a representative unique-domino row. This is the first material search
   experiment because it can change the depth/accuracy trade. The engine currently
   fuses dealing into the round transition, so scope an explicit pre-deal boundary/eval
   state, make/unmake behavior, hashing rule, and parity tests before implementation;
   do not install an arbitrary row and pretend it is unknown.
3. **K=1 representative future — diagnostic only.** Use one legal sampled future to
   measure the maximum possible depth gain and instability. Do not promote it to a
   gameplay or data-generation default merely because it reaches a larger nominal
   depth.
4. **K={2,4,8} sampled scenario tree — deferred unless modes 1/2 expose a measured
   gap.** Pre-sample complete remaining-deck scenarios,
   organize them as a trie of revealed row sequences, and branch only when a row becomes
   public. Scenarios sharing a public history must share the same decision node. This
   preserves non-anticipativity while avoiding an independent K-way resample at every
   later round. It may still be too broad when sampled rows rarely share prefixes; that
   is a measurement question. K=4 is the leading candidate, not a predetermined winner.

The first gate is an offline stochastic-search probe, stratified by opening, middle,
late, placement-heavy, denial-heavy, and forced-discard positions. Build a high-sample
reference (`K=64` or `K=128`, and exact enumeration wherever feasible), then measure:

- root-action agreement and expected-score error;
- **regret**, defined as reference-best value minus the reference value of the selected
  action (more informative than action agreement when several moves tie);
- action entropy and value variance across scenario seeds;
- completed full-width depth, nodes, chance nodes, and wall time;
- stability of ordering and TT reuse;
- paired same-net playing strength against the current `chance_samples=8` baseline.

No stochastic approximation becomes the default unless it improves matched-clock play
on a disjoint confirmation set. A displayed depth increase alone is insufficient.

### Scale-up work package B — stronger, source-separated teacher data (conditional)

Run this after P0 shows useful curriculum signal, or when standalone work specifically
needs broader labels. Do not scale the rejected shallow self-imitation recipe. Generate
positions cheaply, then spend teacher compute only on a stratified subset. Scale-up
target:

- roughly 5,000 replayable games from a mixture of full-width NNUE, pick-aware search,
  randomized openings, NNUE-vs-AZ games, and the best gated stochastic-search mode;
- sample approximately 4-8 positions per game across rounds/phases rather than labeling
  all correlated plies equally;
- emphasize NNUE/AZ disagreements, high AZ policy entropy, rare board geometry, unusual
  draft rows, forced discards, value swings, and exact-solvable tails;
- retain ordinary positions as a control so the selector does not create only exotic
  outliers.

Store **separate targets**, never an undocumented blend:

- honest official final outcome and actor-relative margin from the realized game;
- exact official outcome/action/value when the endgame solver finishes;
- high-budget AlphaZero MCTS root value and visit distribution on selected positions;
- deep NNUE-search value plus sparse root child values only for exact/late/tactical
  positions, or after a frozen probe proves lower error than the available AZ teacher.
  Do not use weak opening/midgame NNUE values merely because they are available;
- training-only score/territory/crowns/bonus auxiliaries already supported.

Every label carries: teacher type and artifact hash, search algorithm, clock/node/sim
budget, completed depth, chance method and K, exact/complete/timeout status, actor frame,
official-cascade version, and source trajectory ID. Exact and high-confidence labels
may receive greater training weight, but source targets remain independently auditable.
The reserved test split stays unopened.

### Standalone work package C — controlled NNUE training experiment (conditional)

Train at least three same-architecture candidates with matched examples/optimizer
budget so the source of improvement is identifiable:

1. **Outcome control:** final outcome + margin + current auxiliaries only.
2. **Reanalysis value:** control targets plus exact/deep/AZ value supervision, with
   confidence-aware weighting.
3. **Value + ordering — standalone-only, deferred:** candidate 2 plus a small CPU
   ordering/ranking head distilled
   from exact root actions, deep child values, or AZ visits. The head orders the complete
   legal set and does not authorize pruning. Do not build it for the primary curriculum
   pilot; require evidence that ordering, rather than evaluation/width, is the next
   standalone bottleneck.

The ordering experiment must reuse the frozen state features or live in a separately
versioned action-ranking artifact. It does not silently reopen the v3 state encoder.

Model selection remains two-stage: frozen validation chooses epochs; paired gameplay
chooses promotions. Each promoted candidate must be tested against the incumbent NNUE,
the current AZ checkpoint, the HOF subset, and the frozen stochastic/exact probe suites.
Report results by phase and label source, not only an aggregate win rate.

### Standalone continuation and pivot rule

Continue optimizing the standalone agent while candidates show repeatable promotion,
the AZ gap narrows, or added clock produces a meaningful strength curve. After the
stochastic sweep and at least two materially stronger training iterations, treat
standalone superiority as unlikely **for the current design** if all are true:

- no candidate earns a statistically credible promotion over the NNUE incumbent;
- matched-clock results remain decisively below AZ with no narrowing trend;
- deeper/longer search mainly increases stochastic variance or nominal depth rather
  than reference regret and playing strength;
- the best chance allocation cannot cross useful horizons without unsafe selective or
  clairvoyant approximations.

That is a design pivot, not project failure. Freeze the strongest competent NNUE and
make the AlphaZero curriculum/exploiter track primary. Revisit standalone dominance
only when a material change arrives (better labels, learned ordering, new chance model,
or a substantially faster engine).

### AlphaZero curriculum deliverable

NNUE-generated positions are **off-policy curriculum**, not NNUE policy labels. The
safe pipeline is:

```
NNUE / NNUE-vs-AZ trajectory
    -> novelty + disagreement + decision-importance filter
    -> high-budget AZ MCTS and/or exact reanalysis
    -> AZ policy target + official/exact value target
    -> source-tagged replay mixture
    -> controlled AZ training and HOF evaluation
```

On AZ turns, existing MCTS visits are usable if their budget meets the quality bar. On
NNUE turns, re-run AZ MCTS after the game; never encode the NNUE's selected action as
the AlphaZero policy target by default. A weaker NNUE can therefore generate valuable
positions without teaching weaker moves. Replay the stored source trajectories through
the current AZ encoder to create dense AZ inputs; do not add another encoded-only
buffer. Split and deduplicate by whole trajectory/source seed so related positions do
not leak across train/validation or dominate the mixture.

Start the AZ experiment with a conservative tagged mix (subject to ablation): **75%
normal AZ self-play, 10% NNUE-generated/AZ-reanalyzed positions, 10% NNUE-vs-AZ
disagreement positions, and 5% exact/near-exact endgames**. Compare with an
equal-example, equal-update control. Success means improved HOF/gauntlet strength,
reduced error on frozen exact/disagreement suites, or closure of repeatable NNUE
exploits—not merely lower aggregate training loss.

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
matched-clock AZ floor and first selective-search sweep are complete. The pilot NNUE
lost 0-8 to AZ at a matched clock. Selective beam search can expose longer nominal
horizons, but the first credible full-root width-2 configuration did not beat the
full-width incumbent over 48 paired-seat games. Do not report selective completed depth
as full-width depth; telemetry exposes `selective`, widths, ordering probes, and pruned
actions specifically to prevent that mistake.

### Phase 4 — Training loop (self-generated data)  (ongoing)
- Searcher self-plays → replayable trajectories → stratified position selection →
  source-separated outcome/exact/reanalysis targets → retrain → select if stronger →
  repeat.
- **This is what breaks past run10's coverage ceiling** (200k ≈ only a few
  thousand correlated trajectories on one policy's support; alpha-beta walks
  off-distribution where a static-buffer eval is blind).
- Mix policies and opponents deliberately: incumbent NNUE, pick-aware search, the best
  gated stochastic mode, randomized openings, and NNUE-vs-AZ games. A training round
  should cover new public states, not merely repeat the incumbent's principal lines.
- Keep final outcome/margin, exact labels, deep NNUE values, and AZ MCTS reanalysis as
  distinct target channels with provenance and confidence. Never silently replace an
  honest outcome with a bootstrapped estimate.
- Train outcome-control, reanalysis-value, and value+ordering candidates under matched
  data/optimizer budgets. Validation selects epochs; gameplay selects promotions.
- Reuse existing self-play orchestration + gating, extending artifacts rather than
  creating encoder-locked buffers.

**STATUS (2026-07-14): MVP complete and first candidate rejected.**
`nnue/generation_loop.py` runs one restartable generation; `nnue/match.py` provides
the shared paired, seat-swapped clock-accounted gate. `train_sparse.py` can mix
immutable replay shards while keeping validation frozen and warm-starting from the
incumbent. Bootstrap targets are still honest final outcomes, not yet deep-search
values. The 256-game result above says to improve the teacher/search before producing
a large corpus. Promotion remains game-strength-gated; Brier selects an epoch only.
The next action is immediate package P0: clock scaling, competence floor, fresh
replayable AZ trajectories, disagreement mining/reanalysis, and the equal-compute AZ
curriculum control. Work package A follows only for standalone chance-allocation work;
the 5k-game / 25k-40k-position package B is conditional on P0 or a demonstrated
standalone label need. Do not launch a 500k-position homogeneous self-imitation run.

### Phase 5 — Combine with AlphaZero (the payoff)
- **SEQUENCING UPDATE:** the small disagreement/curriculum falsification pilot is now
  immediate package P0, ahead of standalone A/B/C. Full-scale mixing remains Phase 5.
- **TRAIN (highest strategic value):** use NNUE as a position/curriculum generator,
  then relabel selected positions with high-budget AZ MCTS, exact official-outcome
  search, or demonstrably higher-confidence NNUE search. NNUE search values are not
  presumed better than 800-sim MCTS; prove label quality on exact/high-sample probes.
- **POLICY SAFETY:** a weaker NNUE action is not an AZ policy target. On NNUE turns,
  reanalyze the state and train the AZ policy head from AZ visits or exact/deep child
  values. The realized official outcome remains a valid value sample.
- **SELECT:** prioritize AZ/NNUE action disagreements, high AZ entropy, AZ losses,
  large value swings, rare geometries/rows, forced discards, and exact-solvable tails.
  Retain ordinary controls so filtering does not create an exotic-only distribution.
- **MIX + ABLATE:** begin with a conservative tagged fraction of NNUE-derived data and
  compare against an equal-example/equal-update AZ control. Promote only on HOF,
  exact-suite, disagreement-suite, and repeatable-exploit improvement.
- **COMBINE:** enter NNUE-alphabeta as an independent HOF/exploiter agent. run11a
  said "locally unexploitable, but all attackers share AZ lineage." An
  independent searcher is the honest exploiter the monoculture lacked — holes
  found = real; none found = a much stronger unexploitability claim.
- **HELP:** position eval / endgame solver / puzzle generator / blunder explainer
  (falls out for free).
- **Hybrids (later, don't front-load):** NNUE exact endgames + MCTS midgame; NNUE
  value as MCTS leaf eval; AZ policy as alpha-beta orderer.

---

## Chance handling — what transfers from Azul and what does not

**Kingdomino-specific correction (2026-07-14): do not assume a representative
four-domino row exists.** The Azul thesis found that, under its clock, a branching
factor of one often beat spending the same time on several refill samples; its final
agent also benefited from a semirandom refill whose color quantities were near their
expectation. That is evidence for sweeping accuracy versus depth, not permission to
hard-code one Kingdomino future. Azul has repeated, exchangeable colors. Kingdomino
has mostly unique domino IDs whose rank, crowns, terrains, placement feasibility, and
removal from later rounds jointly determine value. A row that is average on one axis
can be an extreme tactical draw on another.

Our deterministic-deck AlphaZero experiment already showed the failure mode: when the
environment/search collapses hidden order, the learner can specialize to artificial
future regularities and fail to generalize. Therefore:

- K=1 is a depth/variance diagnostic, not an approved default.
- There is no hand-designed "median ID/crowns/terrain" row in the plan.
- A fixed `state_hash -> representative row` mapping is forbidden for training data.
- Any reduced-sample method must be compared with a high-sample/exact reference for
  regret and seed stability, then beat the current method in disjoint paired games.
- The preferred reduced-cost design is one sampled **public scenario tree/trie** with
  K complete without-replacement scenarios, not K independent full-information solves.
- Current-round search plus an order-blind boundary value is the conservative fallback
  if unique-domino scenario variance remains too high.

Azul faced a related problem (known bag composition, unknown future dispersement),
but repeated colors make its refill distribution more compressible than Kingdomino's.
It did NOT do exact expectiminimax. At each chance node its `branching`
enhancement:
- **Pre-generates** the sampled draws used when round refills are reached, avoiding
  traversal-order-dependent RNG and allowing sibling moves to be compared against the
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
our chance branching C(remaining,4) *collapses* (135751→…→70→1) so we get **exact
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
   K (2 or 200). Our required design consumes sampled rows AT chance nodes in one
   public-state tree (or a scenario trie with shared public histories), not as K
   full-information solves. K>1 + common random numbers is good variance reduction,
   but the tree structure is what prevents fusion.
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
  ceiling. It used pre-generated draws and tuned refill policies, often favoring a
  branching factor of one under a fixed clock. Covering a huge draw space with few
  samples is noisy; using a representative draw or robust mean reduces variance by
  accepting bias. Being calibrated does NOT solve (b).
- **Our current methodology is more expectation-oriented than Azul's default.** What
  softens (b) for Kingdomino is *late-game structure*: C(remaining,4)
  collapses → exact enumeration late + exact endgame (Azul's 20-from-big-bag
  never could); draws are 4-tile objects (lower per-node variance); + Star2 chance
  pruning they likely lacked; + delegating draw-averaging to the learned eval. The
  OPENING still samples → the bottleneck shrinks, doesn't vanish.
- **Real edge = measurement.** They *suspected* draw handling was the bottleneck
  but didn't isolate it. We make draw-handling an explicit swept axis
  (branchingFactor, enum threshold, sampling scheme, Star2 on/off, eval-delegation
  depth) measured on the round-robin — suspicion → knob.

## References
- **Mateusz Rzepecki, "Implementing superhuman AI for Azul board game with a
  variation of NNUE" (2025)** — primary comparison for the search/training loop,
  branching-factor experiments, refill simulation, integer/AVX2 inference, and
  million-scale iterative training:
  https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf
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
as a *solo* player initially — Kingdomino's high joint pick×placement branching is
the leading structural limiter on alpha-beta depth; sampled chance cost compounds it
but is not a correctness gap. This matters in the opening/midgame where most of the
game is decided, and the eval it falls back on there is trained on the same data as
AZ's value head (no inherent edge). It WILL be stronger in the endgame
(exact) and will surface lines AZ is blind to. ~1-in-3 it eventually overtakes
outright with the full loop (Azul precedent — tempered: their MCTS was an
afterthought, ours is mature). **Solo dominance is the ideal bar, but not the only
bar that determines whether the project creates strategic value.**

**Generator bar (how strong is "useful"?) — much lower than "stronger than AZ":**
- **Endgame relabeler (near-free):** exact endgame values beat AZ approximations at
  ANY overall strength. Useful the moment the deep solver runs.
- **Midgame value relabeler:** useful once its search+eval gives *lower-error value
  targets* than N-sim AZ-MCTS on the same positions — value-target accuracy, NOT
  game strength (measure vs deep-solve ground truth on a probe set). Achievable
  well below AZ's playing strength.
- **Diversity generator / exploiter:** useful once it plays *competently* (positions
  are realistic) AND diverges from AZ's policy, reaches under-visited lines, or finds
  one repeatable exploit. The generated positions are reanalyzed; NNUE need not be
  within a fixed Elo distance or provide the final AZ policy label. Difference plus
  label quality matters more than dominance.
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
denial blind spot is a POLICY problem). Selected NNUE positions must ALSO receive a
credible policy target: high-budget AZ visits, exact root actions, or sparse root
targets from completed deep child values, with explicit tie / incomplete-search /
uncertainty / temperature handling. Reuse the `ExactPolicyMode` machinery.
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
- **Joint pick×placement branching** is the leading standalone-search risk. The
  3390-wide action encoding is only a ceiling, not the effective branching factor,
  so record legal actions, searched children, cutoffs, and completed depth by phase
  before choosing an intervention. Use the matched-clock curve to distinguish an
  ordering/engineering problem from a structural depth ceiling. Keep NNUE-only
  ordering as the independence baseline; any AZ-assisted or distilled ordering is a
  separately reported standalone enhancement and must never silently prune moves.
- **Chance efficiency, not chance correctness** remains a secondary search risk.
  The current engine already samples public rows from the sorted remaining bag,
  reuses a bag-keyed scenario set, expands chance inside the tree, and enumerates
  late chance exactly. Kingdomino's unique dominoes still make K=1 unsafe as a
  strength method. Audit the existing baseline, then test current-round boundary
  evaluation before paying for a K-scenario trie; retain K=1 only as a diagnostic.
- **Strategy fusion / hidden-order leakage** — a hard correctness failure, not a
  strength trade. Reject independent full-deck solves, order-dependent encoding, and
  future-revealed training targets with structural tests.
- **Teacher circularity** — deeper NNUE labels can distill its own mistakes. Keep
  outcome, exact, AZ, and NNUE targets separate; use exact/high-sample probes and
  gameplay to establish confidence before weighting them heavily.
- **Weak-generator policy contamination** — NNUE-generated states are useful, but
  its chosen action is not automatically an AZ target. Reanalyze NNUE turns.
- **Novelty filter bias** — selecting only disagreements can create an exotic replay
  distribution. Retain ordinary controls, source tags, and phase/geometry coverage.
- **Statistical power** — early 8- or 16-game results are directional only. Use
  paired seeds/both seats, disjoint confirmation sets, confidence intervals, and
  equal-compute AZ ablations for decisions.
- **Connectivity nonlinearity** — the frozen encoder includes exact region/score and
  expansion summaries, but training must still prove they support long-horizon value.
- **Quantized-artifact safety** — every promoted checkpoint must instantiate the
  existing Rust quantized path, record its selected scales and accumulator bound,
  and pass float/quantized parity. Do not replace the exact per-artifact overflow
  rejection with a fixed heuristic margin.
- **Compute allocation** — CPU trajectory generation is cheap; high-budget AZ/exact
  reanalysis is not. Stratify and filter positions first, then spend teacher compute.
- **Frozen-schema discipline** — trajectories remain the source of truth. Do not add
  encoder-locked buffers or reopen the reserved test split during tuning.
