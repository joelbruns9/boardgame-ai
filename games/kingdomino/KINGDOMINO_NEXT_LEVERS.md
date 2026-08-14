# Kingdomino: what is left to try

- **Status:** Living decision record. Placement headroom and
  explicit/progressive chance modeling are resolved negative. Corrected
  deep-target reanalysis passed its development gate and now requires the
  untouched confirmation split. The contextual open-loop offline gate remains
  a separate low-compute search question.
- **Date:** 2026-08-13
- **Current model:** `runs/kingdomino/best_checkpoint/current_best.pt`
  (sha `4bf07b0c…`, 80x6), placed **3rd in a three-month BGA arena**.
- **Purpose:** capture credible remaining levers and, more importantly, the
  reasons the obvious ones are already closed — so a future session does not
  re-run a null.

## 1. What is already closed, and how firmly

The completed nulls and closures matter more than any idea below: the cheap and
obvious directions have been tried.

| Attempt | Result |
|---|---|
| run9 — diversity package | null |
| run10 — pick-group visit floors (depth 1-2) | null |
| run10 — spite personalities in the HOF pool | null |
| run11a — exploiter loop (PSRO-lite) | null: **locally unexploitable at equal capacity** |
| 2026-08 — tile action-value head (M0-M2.5) | closed, see `AZ_TILE_Q_HEAD_PLAN.md` |
| 2026-08 — exact late-placement headroom audit | closed: model no worse than strong humans on confirmation; see `PLACEMENT_LATE_AUDIT_FINDINGS.md` |
| 2026-08 - selective 30k replay-reanalysis qualification | development passed after root-mask correction; confirmation pending; see `DEEP_TARGET_STAGE3_FINDINGS.md` |
| 2026-08 - exact/progressive post-reveal chance exposure | closed: the production progressive treatment scored 48.24% at 800 simulations and 49.46% at 4,800 against ordinary open loop; see `CHANCE_PROGRESSIVE_GATE0_FINDINGS.md` |

The run11a result is the strongest: an exploiter warm-started as a clone of the
banked net, trained specifically to beat it, plateaued at ~48.5% over ~15,000
games with no trend across ~8,700 training steps. It never learned to want the
squeeze position it was pointed at.

**Read the qualifier carefully: "at equal capacity."** Every null above was run
at 80x6. That is the loophole none of them tested.

Independent external evidence that the policy is already strong: the BGA anchor
(606 clean opponent decisions, 36 games, top-30 opponents) found the net's top
first claim matched the human's **76.4%** of the time, top-2 **95.5%**, with only
**3.6%** of human claims being moves the net rated below 5%. In Mighty Duel these
are claim-order statistics, not necessarily disagreements about the completed
two-tile bundle; see `BGA_TWO_DISAGREEMENT_CASES.md`.

## 2. Recommended next measurement

Placement is closed. Deep-target reanalysis has earned its predeclared
confirmation measurement, but not yet a training experiment. Separately, one
cheap search-mechanic gate can reuse existing chance traces: test whether a
compact context recovers the small observation-conditioned signal without
recreating the failed exact-row tree. Contextual open loop and CORAL are one
abstraction family, not separate levers.

If that gate fails, the next distinct question is whether starting self-play
from verified BGA states improves state-distribution coverage. That is not
target reanalysis and needs its own sampling, value-label, mixture-weight, and
hidden-information contract.

### 2.1 Contextual open-loop abstraction — offline gate only

**Question.** Can strategically similar revealed states share conditional pick
Q statistics while strategically opposite states stop sharing them?

Current open-loop MCTS stores one value per slot-relative action history. It
filters legality in each concrete determinization, but the surviving edge value
still averages across different revealed rows, dominoes in hand, and resulting
boards. Full chance/information-set branching avoids that aliasing but fragments
the budget over thousands of possible rows.

The 50-position A-1 audit found a small local upper-bound signal: depending on
the chance-sampling arm, fully observation-conditioned continuation changed the
backed root pick on 2-5 positions, with three reasonably stable cases. The later
practical search probe did not convert that signal into a qualified search
improvement. At 128 simulations on 50 positions, mean regret was 0.03936 for no
added chance exposure and 0.04082 for exposures 1, 2, and 4. At 4,800
simulations on 12 positions, the preferred exposure depended on whether the
ordinary or hybrid 10,000-sim reference scored it; those references agreed on
the exact top action in only 7/12 positions.

The subsequent fixed-network Gate 0 was decisive. The implemented cap-16
progressive treatment scored 48.24% over 2,048 games at 800 simulations and
49.46% over 4,096 games at 4,800 simulations against ordinary open loop. Its
high-budget active and mature widths were both effectively 16/16, so failure to
activate does not explain the null. It failed the low-budget gate and was
inconclusive while converging toward parity at the teacher budget. See
`CHANCE_PROGRESSIVE_GATE0_FINDINGS.md`.

Therefore exact/progressive reveal branching is closed at the tested compute.
The remaining experiment is an offline context-abstraction comparison using a
single shared reference:

1. Constant context: current open-loop baseline.
2. Exact revealed row: adaptive upper reference, not an early-game deployment
   candidate.
3. Enriched edge identity: public current/picked-domino signatures in addition
   to the already-encoded rank slot.
4. CORAL-style intent: a cheap actor-relative heuristic's preferred pick slot.
5. Intent plus one coarse own-fit/denial/turn-order pressure class.

CORAL is a special case of contextual open loop in which the context is a
heuristic preferred action. Its Thompson-sampling and warm-up machinery are
deferred; first test whether the intent abstraction itself preserves useful
information. No candidate may inspect the sampled order of unrevealed tiles,
and conditioning starts only after a simulated reveal.

Only a compact context that recovers stable adaptive value under both balanced
and IID chance samples, retains adequate bucket support at a 4,800-sim budget,
and improves a shared decision reference earns an advisor-only Rust PUCT
treatment. Failure closes contextual open loop and CORAL together. Passing
earns an equal-wall-time advisor comparison, then a small paired-game gate; it
does not directly authorize self-play or training. See
`KINGDOMINO_CONTEXTUAL_OPEN_LOOP_PLAN.md`.

### 2.2 Deep-target qualification - development positive, confirmation pending

**Question.** On realistic decision states, does the 4,800-sim search choose
actions with material regret relative to a stable, deeper information-set-safe
teacher? This is the gate for selective replay reanalysis. It is not a claim
that 30,000-sim self-play is automatically compute-efficient.

**Corpus status (2026-08-13).** `bga_reanalysis_corpus.py` joined every kept
placement reconstruction back to its original BGA decision snapshot and emitted
1,400 canonical, reconstructable roots from all 36 frozen games: 940 development
and 460 confirmation. The human action is legally encoded on 1,393 roots; 311
late roots are flagged as exact candidates. See `BGA_REANALYSIS_CORPUS_V1.md`.

**Stage-1 screen complete (2026-08-13).** Two independent 800-sim searches were
run on all 940 development roots and none of the 460 confirmation roots. Tile-pick
agreement between repeats was 99.0% (9 disagreements); exact joint-action
agreement was 90.9% (86 disagreements), so most low-budget instability is within
a tile's placements. The deliberately high-recall 0.05 Q-gap rule selected too
many roots to be the Stage-2 cut: 382 were close in either repeat and 359 in
both. See `DEEP_TARGET_SCREEN_STAGE1.md` before choosing the 4,800-sim subset.

**Stage-2 screen complete (2026-08-13).** The cohort was frozen at 172
development roots: 122 live roots with a top-two pick Q gap <=0.03 in both
800-sim repeats, all 9 tile-unstable roots, 33 easy controls, and 11
deck-stratified starvation controls (with overlaps counted once). Paired
4,800-sim searches changed the tile in at least one repeat on 31 roots and
changed a stable two-seed consensus on 11. None of the easy or starvation
controls changed tile. Thirteen roots still disagreed by seed at 4,800.

A matched forced-pick check searched every tile at the 18 roots where ordinary
4,800 MCTS still gave a group zero visits. The first output was invalid because
the Rust root mask was bypassed during missing-child recovery. After the fix,
all 144 restricted searches used exactly 800 visits inside the requested group.
None of the 20 formerly unvisited groups ranked best or was within 0.03 Q; only
three were within 0.05. The corrected deep cut is therefore 24 roots: 11 stable
consensus changes and 13 unresolved 4,800 roots. See
`DEEP_TARGET_STAGE2_FINDINGS.md`.

**Corrected Stage-3 development gate passed (2026-08-13).** Every tile on the
24 frozen roots received two properly restricted 10,000-simulation searches;
all 136 searches received the exact requested budget. The ordinary
30,000-simulation searches were unaffected by the bug and were reused.

Cross-seed validation selected on one seed and scored on the other. Matched
teacher uplift over the 4,800 tile averaged +0.00868 Q decision-weighted and
+0.00741 game-weighted, with a game-clustered 95% interval of
[+0.00038, +0.01847]. Ordinary 30,000-simulation choice uplift was similar at
+0.00888 Q, with interval [+0.00050, +0.01863]. Three gains exceeded +0.03 and
none was below -0.01, but two source positions dominate the signal.

The positive lower bound passes the development gate. The 460 confirmation
positions remain frozen and must now be tested with the method unchanged.
Selective reanalysis is qualified for confirmation, **not yet for training**;
a higher general self-play budget is not implied. See
`DEEP_TARGET_STAGE3_FINDINGS.md`.

This is a qualification corpus, not training data. The current KD replay buffer
cannot itself be reanalyzed because it stores encoded tensors rather than a
reconstructable `GameState`. The BGA suite asks whether building reconstructable
self-play sidecars and selective relabeling is worth that engineering cost.

**Completed design.** On a frozen development subset, ordinary 4,800-sim actions
were compared with repeated 30,000-sim searches using common random numbers.
Every pick group received a matched conditional probe. Cross-seed evaluation
distinguished stable value improvement from determinization noise. Because the
corrected development gate passed, the planned confirmation pass is now the
next decision point and has not yet been run.

The primary statistic is not action disagreement. It is the deep teacher's
estimated value loss after forcing the 4,800-sim action, with uncertainty
clustered by source game. Also report where regret concentrates: phase, pick
entropy, visit gap, secondary-pick starvation, discards, and exact eligibility.

**Predeclared gate.** Negligible regret closes broad reanalysis. Concentrated regret earns a
selective deep-target dataset. Broad material regret earns a compute-matched
training pilot, but still does not justify 30,000 sims on every self-play move;
the likely treatment is 4,800-sim self-play plus selective 20k–30k reanalysis.

**Gate result.** Development found concentrated, reproducible regret and earned
confirmation. No training pilot is warranted until confirmation independently
passes.

**Caveat.** BGA positions test real strong-human support, not the full current
self-play distribution. A positive result qualifies reanalysis; it does not by
itself determine the final training mixture or prove a gameplay gain.

### 2.3 Placement headroom audit — resolved

The 36-game BGA corpus was reconstructed with complete ordered 24-domino
sequences for both players. Whole-game exact optimization was intractable and
wide beams did not converge, so no beam result was used as ground truth.

Freezing the actual board after placement 16 made the remaining eight-tile
suffix exactly enumerable. For every reconstructable placement 17–24 decision,
the audit held the logged next pick fixed, enumerated every legal placement,
and exactly optimized the remaining actual claimed-tile suffix. Human, raw
network placements were evaluated against the same clairvoyant offline
reference; the network never saw future claims. The originally reported
searched-placement comparison is withdrawn because the root pick restriction
was not enforced throughout open-loop selection.

Confirmation results (10 games, 49 decisions; game-weighted regret points per
decision):

| Actor | Mean regret | Zero-regret decisions |
|---|---:|---:|
| Strong-human opponent | 1.018 | 71.4% |
| `current_best` raw policy | 0.835 | 77.6% |

Raw-policy minus human was -0.183 points with a paired game-clustered 95%
interval of [-0.713, +0.333]. The pre-frozen confirmation criterion required a
positive lower bound, so it failed. Late placement is not a demonstrated
relative weakness and **does not earn board-auxiliary supervision**. This
conclusion rests on the valid raw-policy comparison and does not require the
withdrawn search result.

Placements 3–16 remain technically unmeasured, but that limitation is not a
reason to keep the lever open after a negative confirmation result in the exact
region. Do not revisit without new independent evidence. See
`PLACEMENT_LATE_AUDIT_FINDINGS.md` and `placement_late_audit_protocol_v1.json`.

## 3. Other credible levers

Ranked by my estimate of expected value, with the case against each.

**BGA-seeded self-play or search distillation.** Number one in the recorded
pivot order and still undone. The 1,400 verified roots supply state-distribution
coverage that mirror self-play did not generate. Two uses are valid: begin
self-play continuations from the public roots, or add root-only policy
distillation examples labeled by production search. Either path must
redeterminize the unordered hidden bag and predeclare sampling, value targets,
mixture weight, and whole-game splits. Do not insert one-hot human first claims
as authoritative labels: Mighty Duel claim order does not identify the final
two-tile bundle. *Against:* needs start-from-state or masked/root-only training
support. BGA-seeded self-play remains a distribution experiment; the separate
deep-target result says selected 20k-30k labels may be useful if confirmation
passes.

**Capacity.** Every null was at 80x6, and run11a's unexploitability was
explicitly *at equal capacity*. The recorded pivot order already says "rerun
capacity bake-off on a squeeze-containing buffer before any 80x10 talk."
*Against:* run5's verdict was **data exhaustion**, so more capacity on the same
data may only overfit. This is why the bake-off is conditioned on better data
rather than run standalone.

**Endgame exactness.** The solver is exact at deck <= 4
(`endgame_solver.py`). Pushing the frontier to deck <= 8 would make the final
rounds provably optimal, and endgame errors translate directly into final score.
Its great virtue is that it is **verifiable against ground truth** — no proxy
metric, no gate to mis-specify. *Against:* the exact region is small, so the
win may be small too; measure the frontier cost before committing.

**Distributional score head.** Third in the recorded pivot order. Never
attempted; no evidence either way.

**Ship.** Fourth in the recorded pivot order, and the record calls an
"exploiter-certified plateau a defensible stopping point." Third place in a
three-month arena against elite opponents is consistent with a genuinely strong
player in a game with real variance. This is not a failure state.

## 4. Levers I would NOT revisit without new evidence

- **Paired-seat simulation-budget sweep.** A 10k-vs-800 match can describe how
  a particular search-budget gap survives game variance, but it does not
  distinguish luck from search saturation and does not select an efficient
  self-play label budget. Revisit only if outcome-variance calibration itself
  becomes the question.

- **Full information-set/chance MCTS and progressive exact-row exposure.** The
  A-1 audit found a small adaptive upper-bound signal, but the practical search
  probe did not establish an advantage over ordinary open loop and was
  reference-dependent at 4,800 simulations. Do not add more exact reveal
  branches or exposure levels. The only permitted follow-up is the compact
  offline abstraction gate in `KINGDOMINO_CONTEXTUAL_OPEN_LOOP_PLAN.md`.
- **CORAL as a separate lever.** CORAL's intent is one possible context key for
  contextual open loop. Do not run a separate Thompson-sampling/warm-up project
  unless intent-conditioned PUCT first passes the shared-reference and game
  gates.
- **Board-auxiliary placement supervision.** The exact placement-17+ confirmation
  audit found no positive model-minus-human regret gap. Earlier flexibility is
  unmeasured, not affirmative evidence of weakness.
- **Pick-side denial / secondary-pick sharpening.** Closed after M0-M2.5 (see
  `AZ_TILE_Q_HEAD_PLAN.md`). The premise held — searched values really do
  disagree with the raw value head, monotonically in policy rank — but every
  attempt to convert that into a better *decision* failed.
- **Pick-group visit floors as a shipped setting.** Measured a wash at 10,000
  sims (fixes 3 positions, breaks 3). They remain useful as a **label-generation
  device** at low sim counts, not as an inference setting.
- **More offline analysis against the 8-ply forced reference.** Its own
  reliability is unverifiable on the frozen 50: the exact solver needs deck <= 4
  and that set's minimum deck is 8, so it cannot check its own judge.

## 5. Methodological lessons that should govern future work

These cost real time to learn and are the most transferable thing here.

1. **Measure the objective, not a proxy for it.** The last arc repeatedly
   measured *calibration* when the objective was *selection*. Raw child value
   cut MAE 3.2x versus a zero constant while scoring identically on top-1 tile
   choice, because the decision gaps (median 0.033 between best and second-best
   tile) sit well inside the estimator's error.

2. **Every gate needs a control cohort.** An absolute magnitude on a treated
   group cannot show the effect is specific to it. This was gotten wrong twice.

3. **Compare paired quantities with paired statistics.** Subtracting one arm's
   marginal confidence bound from another's point estimate bounds nothing. This
   error was identified, fixed, and then rebuilt two milestones later — it
   produced an invalid "proceed" verdict.

4. **Cluster by the unit that generated the data.** Positions from one game are
   correlated; treating them as independent understates every interval.

5. **Check the baseline is not trivially achievable.** A "25% better than raw"
   gate turned out to be three-quarters clearable by a single constant per rank.

6. **Prefer arbiters with no network in them.** Everything scored against the
   8-ply reference inherits that reference's value head. The exact solver, real
   game outcomes, and the placement optimiser do not.

7. **Do not let each treatment define its own judge.** In the progressive
   chance probe, the preferred exposure changed with the reference family.
   Future search arms need one shared evaluator or independently paired game
   outcomes; agreement with a deeper version of itself is not evidence that the
   treatment is better.

## 6. Reference numbers

Measured 2026-08-06/07 unless noted, on `current_best` (sha `4bf07b0c…`).

| Quantity | Value |
|---|---|
| Policy top-1 agreement with 8-ply reference (held-out games) | 69.0% |
| Policy pairwise ordering accuracy | 82.7% |
| Policy mean searched-Q regret | 0.0076 |
| Median searched-Q gap, best vs second-best tile | 0.033 |
| Secondary tile visit share @3,200 / @10,000 sims | 2.19% / 1.16% |
| Best-prior placement == searched-best placement | 42.0% |
| BGA anchor: net top pick == top-30 human's pick | 76.4% (top-2: 95.5%) |
| Reconstructable BGA corpus | 1,400 roots: 940 development / 460 confirmation |
| run11a exploiter plateau vs banked net | ~48.5% over 15,000 games |
| Exact late placement, confirmation: raw/search minus human regret | -0.183 points; clustered 95% CI includes zero (2026-08-13) |
| Progressive chance probe @128 sims, mean regret X=0 / X=1,2,4 | 0.03936 / 0.04082 (50 positions) |
| Progressive chance probe @4,800 sims | no reference-independent winning exposure (12 positions) |
| Progressive Gate 0 @800 sims | 48.24%; one-sided 95% interval [47.34%, 49.15%] (2,048 games) |
| Progressive Gate 0 @4,800 sims | 49.46%; one-sided 95% interval [48.69%, 50.23%] (4,096 games) |

Related: `AZ_TILE_Q_HEAD_PLAN.md` (closed experiment),
`SECONDARY_PICK_FRAGILITY_FINDINGS.md`, `RUN10_PLAN.md`,
`PLACEMENT_LATE_AUDIT_FINDINGS.md`, `BGA_REANALYSIS_CORPUS_V1.md`,
`BGA_TWO_DISAGREEMENT_CASES.md`, `CHANCE_PROGRESSIVE_GATE0_FINDINGS.md`, and
`KINGDOMINO_CONTEXTUAL_OPEN_LOOP_PLAN.md`.
