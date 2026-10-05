# 7WD Model Growth Plan

**Status:** DRAFT r2, 2026-10-02. Nothing below is built. r1 (`d8a620e`) was
written after run07's resume on `2fe60a7` and the RICCP 923216750 review. r2
folds in the external review `reviews/sevenwd-growth-plan-d8a620e.md` (all eight
findings verified against code and saved results), the Braun chess paper
(arXiv:2609.37447), and the owner decisions of 2026-10-01/02.

**What changed in r2:** the measurement instrument comes first (gates ran at 64
sims); the diagnosis is a coupled data / learning / search problem, not "the
value head"; G0b's clairvoyant oracle is replaced; G1's size is measured
(+43%) and needs a sampling design; G2 gets a per-head target contract; G4 has a
positive causal result; G9 is a heuristic, not exact; an all-moves fixed-budget
arm leads G11; new workstreams for restart archives, exploiters, plasticity, and
gating.

## Where we are

- **run07 ENDED 2026-09-30 at iter 100** (revert_reset; best = iter 60, gate
  0.553 [0.517, 0.588] n=1500 vs iter 20). Gates vs 60 afterwards 0.49-0.53;
  self-anchor STAGNANT. All 101 buffers, candidates 0-100, logs, specialists and
  HOF are local in `runs/seven_wonders_duel/run07_bundle` (main folder). No
  `learner_*.pt` -> no warm-optimizer continuation is possible.
- **Every strength number from run07 was measured at 64 sims/move.** Gates AND
  the self-anchor (`self_anchor_gate` reuses the gate path, `gate_sims=64`).
  Full training search is 1,600; advisor investigations used 4k-16k. A plateau
  at 64 sims does not establish a plateau at the deployed budget (chess: ~81 Elo
  per doubling 200->1,600; teacher/student gaps grew with search depth). The
  iter-60 promotion and the stall are both 64-sim facts.
- **Policy fit improved, strength did not (at 64 sims).** W5 held-out CE ~0.92
  vs flat ~1.31. This is agreement with a search teacher that shares the blind
  spots; it does NOT show the policy is not a bottleneck. Measured tail priors
  on refutations are 0.012-0.077.
- **Rare decisive tactics are mis-learned.** RICCP: an immediate Mausoleum
  science win valued 3.9%; buffer audit: such positions 47% predicted vs ~94%
  target; ordinary sixth-symbol wins 82%; 20 retained Mausoleum-win rows in
  297k. Science threat prior under-predicted by .4-.77. BGA: revealing moves
  overrated (Port -25 pts).
- **Proof retention is broken at scale** (review audit, iters 60/100): only
  21-22% of solver-labelled positions survive derivation; ~530/iter excluded
  proofs come from **1,600-sim** league moves (policy-excluded, so caught by
  `is_fast_search_move`). Retaining them adds +43-44% rows.
- **Search cannot compensate behind chance nodes.** Great Library edge frozen at
  754 visits / Q 6.94% from 4k to 16k sims (70-way split).
- **Exact leaf values help causally.** `immediate_guard_findings.md`: a
  value-only override at winning retrievals moves the Library edge 6.94% ->
  46.36% at 16k sims (6.73% -> 32.89% mean at ~4k, 3 seeds), policy untouched,
  no measurable slowdown. It did NOT change the earlier University decision at
  ~4k sims.

## Diagnosis (r2)

A **coupled** problem; no single component is established as the cause of the
plateau:

1. **Coverage / retention** -- decisive positions are filtered out of training
   (G1) and rarely reached at all by self-play (G12).
2. **Learning absorption** -- even correctly labelled decisive rows are fit
   poorly (47% vs 94%): sparse exposure, features, interference, optimizer
   history and capacity are not yet separated (G14).
3. **Search allocation** -- wrong values/priors behind chance nodes are not
   funded enough to be corrected (G4, G8, G9).
4. **Label contamination** -- cheap moves miss killer replies, labelling the
   earlier positions as safe (G11).
5. **Measurement** -- the 64-sim gate may hide or misrank all of the above (E0).

Evidence table (development cases, not population frequencies):

| case | value / prior error | search corrects? |
|---|---|---|
| RICCP retrieval node | 3.9% on a 100% win | yes (764 sims) |
| RICCP before Great Library | same values, one level down | no: Q frozen at 6.94% to 16k sims; exact-leaf override -> 46.36% |
| RICCP before University (the losing decision) | -- | not changed by the override at ~4k |
| 908370787 ToA refutation | prior .012-.077, value 36 pts wrong | no: ~160 visits/world vs 165-400 needed |
| science threat prior | opponent science -.4 to -.77, worst early | only near the end |
| BGA Port | revealing move ~25 pts optimistic | no: fan-out |
| W11 | 2 forced plies recover 19% | value needs real search depth |

Failure labels (multi-label allowed; **unresolved** is a category, not F5 by
default):

- **F1** own decisive win misvalued -- value.
- **F2** opponent decisive threat under-predicted -- value/prior.
- **F3** refutation unfunded behind a chance node -- prior x allocation.
- **F4** cheap-move label contamination -- data.
- **F5** strategic -- only when the reference computations are adequate and still fail.
- **U** unresolved -- reference insufficient, assumptions invalid, or several errors at once.

## Rules for this plan

1. Every intervention gets its own measurement before it is combined. A
   combined win cannot attribute learning vs search: use 2x2 (old/new weights x
   old/new search) where both change.
2. Strength is judged at **representative budgets and equal wall-clock time**,
   with the 64-sim gate as a cheap screen only. Paired setups/seats, intervals
   clustered by setup pair, fresh confirmation seeds; validate that paired hidden
   deals actually match after trajectories diverge.
3. Predeclared **non-inferiority** for broad strength, not "no significant loss".
4. Measure magnitudes on the real buffer, not reasoned ones (r1 claimed G1
   growth "small"; it is +43%).
5. Three evaluation sets: development cases (reviewed BGA games, the 267-episode
   corpus -- they shaped the design), a **sealed** family-level tactical test,
   and representative fresh games. Split by whole game / restart ancestor /
   tactical family; report unique ancestors, not rows.
6. An intervention must improve its intended class without a meaningful broad
   regression; it need not move every class.
7. Laptop-scale first; one matched online A/B at a time on the box.

## Final run preparation (owner, 2026-10-05) -- supersedes the orders below

Goal: ONE more cloud run (hopefully the last), warm-started, not random.

**Cost split.** Game generation (a full search with network calls for every
move of every game) is the only step that needs a box. Everything done to the
existing run07 data is laptop work: correcting all 101 buffers (G1/G2/G2b) is
~2.5 s per 1,000-game buffer (~4 min total); training on corrected rows ran
~25 min per 2,000 steps; a full pretrain is a few hours.

**What correction cannot fix** (the gap the cloud run must close):

1. Tactics more than one move deep -- a move that walks into a forced loss two
   or more moves later (the predecessor class; RICCP's University) keeps
   run07's possibly wrong target. G2b sees one move ahead only.
2. Reveal gambles -- G2b corrects a revealing move only when EVERY reveal
   loses; partial-loss reveals keep run07's fan-out-starved targets (~20 pts
   optimistic in the reviewed games).
3. Strategic quality -- drafts, wonders, economy, plans: no exact check
   exists; labels are as good as run07's search, and a network cannot beat its
   teacher from the same targets.
4. Values outside proofs -- most value targets are still the result blended
   with run07's search value; only proof and certain-win rows are exact.
5. Coverage -- positions run07 never reached carry no label (G12's job).

**Laptop sequence** (user launches the long steps):

1. Correct the buffers: derivation with G1 + G2 + G2b (all default on).
2. Targeted reanalysis (G8.2, laptop-sized): re-search a SUBSET of buffer
   positions with today's search (G4 on, full budget) and replace their
   targets -- revealing moves, cheap-search moves, and decisions a few moves
   before decisive positions. Shrinks gaps 1 and 2; doing every position would
   cost as much as generation.
   **BUILT 2026-10-05** (`targeted_reanalysis.py`; `--reanalysis-overlay` in
   `g3_offline_ab.py`; `dataset.apply_reanalysis`, both backends; resumable
   JSONL overlay; G0-sealed games never selected). Classes: pre_decisive
   (within 4 plies before a forced-win/loss/must_block position), reveal (the
   played move uncovered a card), cheap (sampled). Measured on run07 iter 100:
   ~4,800 targets per 1,000 games at a cap of 6 per game; 0.42 s per position
   at 1,600 sims (coalesced, laptop) -> the full 41-100 selection is ~34 h, so
   cap 2 over 81-100 (~32k, ~4 h) or cap 3 over 61-100 (~96k, ~11 h). On a
   204-position probe (78-85% were cheap searches in run07) the re-search
   moved a third of the move target (TV 0.30) and changed the top move in 28-29%;
   value |change| 0.16 (pre_decisive) / 0.07 (reveal). Caveat: re-searched with
   iter 60's network, so part of each change is a different network.
3. G14 initialisation check on identical corrected data, equal steps and
   holdouts: (a) the base (candidate_0100, decision 13) + fresh optimizer, (b) random init, optional
   (c) candidate_0060 with value heads reset. Judge on sealed G0 + held-out
   validation. (a) best -> warm start; (b) catches up -> the old weights hurt
   and random init earns its compute; (c) best -> partial reset.
4. Pretrain from the G14 winner over iterations 41-100 in RAM-sized windows
   (early iterations' weaker targets down-weighted or left out), withholding
   G0's sealed games.
5. Score the pretrained checkpoint on sealed G0 (and against candidate_0060)
   before renting anything.

**Cloud run:** self-play from the pretrained checkpoint with G4 exact tactics
on (labels tactics-aware natively), every move fully searched (owner decision
11; budget still open), G2b on until its phase-out criterion is met, plus
whichever chance-reveal items are ready. G3 off.

**Open decisions:** (a) sims per move -- one compute-neutral budget (~500,
run07 averaged ~519) vs 1,600 everywhere (~3x compute per game); G0 evidence:
more sims help must_block tactics (7.7% blunders at 64 -> 5.7% at 800 with
G2b) but not reveal traps (~30% at both). (b) The chance-reveal programme, in
the order recommended 2026-10-05: G8.2 targeted reanalysis (attacks the
unfunded-refutation cause W9 identified; feeds step 2 above) -> partial-proof
bounds at reveals (G4 layer 2b: k of n worlds proven lost bounds the move's
value; reuses `losing_mass`) -> the chance-capping A/B (already built, off) ->
G8.3 afterstate value head (only after G8.2 supplies correct targets). W9-style
reply sharing across reveal worlds is closed (four nulls). G8.4 root
verification for the advisor any time.

## Execution order

Steps 1-4 are laptop work and can overlap; 5+ need a box.

**Revised 2026-10-05** after the first G3 offline A/B (see G3): step 4 ran on
buffers corrected only by solver proofs and certain wins, so it could only move
the endgame. The order is now:

1. **G2b tactical relabelling** + a three-arm offline A/B (uniform without G2b
   [done], G2b uniform, G2b + G3 priority) on the sealed G0 split. [steps 2 + 4]
2. G4 / G4b / G8.0 search work -- DONE. [step 3]
3. **G11 measurement** -- reanalyse cheap moves at full budget to size the
   contamination BEYOND one move ahead, which G2b cannot see. [step 1]
4. **Short online confirmation** on a box: self-play with G4 on (tactics-aware
   labels natively) + G2b + G3. [step 5]
5. **G12 restart archive / G8.2 targeted reanalysis** -- more deep and
   predecessor-class positions. [step 6]

Original order, kept for reference:

1. **E0 measurement contract** + G0/G0b instruments + G11 measurement.
2. **G1/G2 retention and target repair** -> re-derived buffer artifacts.
3. **G4 exact tactics and G4b pending-choice expansion** with frozen weights,
   measured at the losing predecessor decision. G4b's Library half needs G8.0's
   token afterstates; its Mausoleum half does not and can go first.
4. **Offline absorption experiments** (G3 sampling, G10a features, G14 resets) on corrected buffers.
5. **Short online confirmation** (2x2), including the G11 fixed-budget arm.
6. **G12 restart archive and G8.2 targeted reanalysis**, vs the same compute spent on ordinary games.
7. **G8.0 exact Library sharing**, then heuristic allocation (G9, G8.1) separately.
8. **G13 continuous exploiter evaluation.** Capacity (W8) only when corrected data carries learnable signal the model demonstrably cannot absorb.

## E0 -- Measurement contract (first)

- **Declare the deployment target:** ruleset, setup/draft regime, hardware,
  thinking time per move (advisor and BGA play).
- **Checkpoint budget matrix:** reference (`candidate_0085`), run07 iters 20,
  60, 80, 100. Screen at 64 sims; finalists at a representative budget and equal
  time. A box is needed for game-level resolution (2-3 pp needs ~2,000+ games at
  any budget); rent a few hours rather than infer from 64-sim gates.
- **Laptop proxy (paired, position-level):** the run07 buffers hold ~8.5k
  solver-proven positions per iteration. For each checkpoint at 64 and 800 sims,
  measure action regret vs the proof on a fixed sealed sample. Each position
  is paired across checkpoints, so hundreds of positions suffice. Endgame-biased
  -- a proxy, not a strength claim.
- **Gate budget for the next run:** raise `gate_sims` (e.g. ~400, ~6x gate cost)
  or keep 64 as a screen with a periodic deep confirmation. Decide from the matrix.
- Keep a fixed pool of different lineages and exploiters as opponents; a single
  incumbent hides non-transitive weaknesses.

## G0 -- Tactical suite (instrument)

Classes: own immediate win (build, Mausoleum retrieval, military, Wonder, Law
token); opponent immediate threat (must-block); solver-proven endgames (exact /
expectimax); chance-node cases (win one move below a chance node); **predecessor
decisions** that lead into or prevent those threats; **unplayed alternatives**
encountered during search; quiet negative controls (same visible motif,
harmless), science symbol variety, military, Wonder retirement and tempo,
economy, age transitions, draft.

Report per class: actor-relative utility error vs proof, calibration on ordinary
states, catastrophic overconfidence, policy mass/rank on critical actions,
within-Wonder burial correctness, **action regret**, and simulations needed to
establish a refutation.

Scale caveat: 54 retrieval-win rows exist in 17,000 games; "a few hundred per
class" needs legal replays/restarts for diversity, and oversampled copies are
not new cases. The audit also found every one of those 54 wins was **taken**, so
the real benefit must come from earlier evaluations and decisions -- the
predecessor class matters most.

**BUILT 2026-10-04** (`tactical_suite.py`, `test_tactical_suite.py`, 7 tests):
`harvest` files buffer positions into own_win / forced_loss / must_block (exact
per-action labels from `RustGame.classify_actions`, reference
`tactics.classify_actions`), reveal_trap (`phase_e.analyze_position`), solver
(recorded proofs), predecessor (the mover's previous decision before walking
into a forced loss), quiet (motif in reach, nothing forced) and ordinary
(uniform); games sealed 20% by hash. `evaluate` scores the raw net and/or search
at given budgets (exact tactics on by default): value MAE / bias,
overconfident-wrong on proven results, found-win / blunder rates and policy
mass, trap picks and expected losing mass, ECE for the realized-outcome
classes; rows AND unique games, each class also split near_end (<= 2 plies
to the end: mostly the game-ending move) vs deep. Reveal traps use the Rust
`losing_mass` (the Python `phase_e.analyze_position` cost ~2.4 h per buffer);
harvest runs ~45 s per 1,000-game buffer. Smoke (iters 99-100, 60 cases per
class, run07 iter 60): must_block/deep blunders 23.5% raw / 11.8% at 64 sims;
reveal_trap picks 42% / 33%; predecessor value bias +0.24 / +0.08. Not yet:
the unplayed-alternatives class,
sims-to-refute, burial correctness, family-level sealing beyond whole games.

## G0b -- Controlled attribution study (replaces the clairvoyant oracle)

r1's "actual reveal known in advance" axis gives the player information it did
not have; an unlucky correct decision looks "fixed". Instead, keep the root's
information set and without-replacement reveal probabilities identical, and
intervene on one factor at a time:

- **allocation:** normal search vs adequately funded probability-weighted search
  over the same legal chance support (exhaustive when small; fixed stratified
  sample otherwise, separating sampling error from leaf error);
- **prior:** force-fund known critical replies, or substitute a stronger
  reference policy, values unchanged;
- **value:** substitute exact values on leaves that can be solved.

Regret for stochastic positions: `V*(s) - Q*(s, chosen)`, both before the
unknown reveal -- never the realized child. Multi-label, with U. Selected
losses give failure composition conditional on selection; add a representative
sample of all decisions to estimate cost in normal play.

Sources: solver-proof blunders in the run07 buffers (endgame-biased), gate games
(`results.jsonl` if detailed), BGA losses via `bga_review`.

## G1 -- Retention repair

- **Separate value eligibility from policy eligibility.** `is_fast_search_move`
  (`dataset.py:636`, `policy_excluded and sims > 0`) also drops 1,600-sim
  league moves. Keep their proofs as value rows; do NOT re-enable their policy
  labels to do it.
- Retain proofs, immediate wins, and immediate opponent threats as value rows
  regardless of search budget.
- **Size is +43-44% of base rows from proofs alone** (iters 60/100), before
  threats. At 190 x 512 = 97,280 presentations/iteration, uniform admission
  dilutes everything else. Use a controlled mixture (uniform component, per-game
  cap on correlated proof rows) and record **rows actually sampled** per class.
  Adjust training steps from measured absorption, not the old budget.
- Re-derive the 101 run07 buffers into **new artifacts**; raw records stay.
- Corroboration (Braun 6.1): excluding late-game rows and searching them cheaply
  created self-reinforcing endgame-conversion failure in chess.

## G2 -- Target contract per head and proof type

The flat value head already takes solver overrides (`73a3504`) and the
short-term targets (`2fe60a7`), and run07 search read the flat head -- so the
mid-run change did reach search. The **hierarchical (W4) head does not**: its
loss (`train.py` hierarchical branch) blends the realized class with search
outlook, with no solver override, no short-term term, no per-row value weights.
Specialist bias reads W4's `hier_joint7`, and `--value-source hierarchical`
would silently drop both improvements.

Contract:

- Certain immediate science/military win -> exact actor-relative outcome AND
  victory type, applied to every applicable head and same-turn pending choices.
- Scalar expectimax proof -> **exact expected utility** only (may be fractional);
  not a win, no victory type. The current `(win, 0, loss)` mapping is
  utility-equivalent, not exact WDL: either propagate real WDL/type under a
  declared tie rule, or supervise only the proven quantity and mask the type.
- Opponent threat -> a reason to inspect, never a proof of loss.
- Acceptance: "effective targets equal the proved quantity in the correct
  player's frame" (not "> 0.95"), checked through Python derivation, Rust
  derivation, packed batches, loss construction, and deployed head selection.

**BUILT 2026-10-04** (`--value-target-contract g2`, default; `legacy` = run07's
objective, the G5 arm). `train.value_targets` builds every head's target in one
place; tests in `test_g2_value_contract.py`.

- W4's loss is split into its factors, outcome + type-given-outcome (exactly the
  old joint NLL under `legacy`). The outcome factor now takes the proofs, the
  short-term term and the solver row weights.
- Expectimax proof -> expected utility only, both heads: BCE of `(1+v)/2`
  against `P(win) + P(draw)/2`, indifferent to draw mass.
- **Certain win** = `dataset.certain_win_moves`: every remaining move is the
  winner's and none but the last triggered chance. Exact outcome and recorded
  type, nothing blended. Computed from the record by both backends (Rust derive
  now returns per-move chance counts), so re-derived run07 buffers get it.
  Missed immediate wins are NOT detected (audit: all 54 were taken); certain-win
  fast moves are not newly retained (G1's cap only).
- Deviation from the bullet above: a proof does not MASK W4's type. The
  realised type given the realised outcome is still a valid sample of that
  conditional; masking would drop type supervision on ~26% of rows. **Qualified
  after review (6ab4342):** that conditional is the BEHAVIOUR policy's
  P(type | outcome), not the type distribution under optimal play; joined to a
  proof-trained outcome marginal it mixes policies (on a proved win the
  realised loss-type conditional may have zero true mass). Kept as auxiliary
  supervision -- do not call the joint "proof-calibrated". Likewise a
  certain-win type label is ROUTE supervision (a type the winner could force
  along the recorded line), not proof that every winning line has that type,
  and must not be read as an exact specialist utility.
- Measured on run07 iters 60 / 100 (1,000 games each, G1 cap 4): 18,384 /
  18,491 rows; exact proofs 1,186 / 1,298; expectimax 3,804 / 3,909; certain
  wins 306 / 347 (types civ/sci/mil 93/121/92 and 133/132/82; 86-90% also have a
  proof); short-term newly reaches W4 on ~13.3k rows.
- Validation numbers are unchanged by the contract (proofs off there; a certain
  row's exact target equals its realised one).

## G2b -- Tactical relabelling (owner decision 2026-10-05)

G4's proof service applied to TRAINING targets, the "reusable" half of its
title. At derivation, every buffered position gets `classify_actions` (+1 the
move forces a win this turn, -1 it loses by force, 0 unknown) and its targets
become a third proof type in the G2 contract:

- **forced win available** -> exact value +1 (an exact proof row: replaces the
  blend, pinned by G3); move target restricted to the winning moves;
- **every move loses** -> exact value -1;
- **some moves lose by force (must_block)** -> those moves get zero move
  target, the rest renormalised (uniform over the non-losing moves if search
  put all its mass on losing ones);
- victory type untouched (no-masking decision, G2).

Why: the G3 A/B trained on run07 labels from a search WITHOUT G4, so a missed
immediate win or a walked-into immediate loss kept its wrong target; only the
solver's endgame proofs and certain wins were exact, and G3 could only amplify
those (near-end must_block blunders halved, nothing deeper moved). G2b supplies
the exact answer at every position the games passed through.

Scope limits: one move ahead only -- a move after which the opponent has a
REPLY that leaves the mover lost (the predecessor class) is not labelled;
reveal traps' partial losing mass is not used in v1. A partial fix for F4
(cheap-move label contamination); deeper misses stay with G11. Once self-play
runs with G4 on, new labels are tactics-aware anyway: G2b matters most for the
run07 buffers and for cheap moves whose searches are too short for the check
to fire everywhere it should.

**Three-arm offline A/B (2026-10-05), sealed G0, paired, 300 cases/class.**
uniform -> G2b-uniform: must_block blunders 19.0 -> 13.3% raw, 14.0 -> 7.7% at
64 sims, 12.0 -> 5.7% at 800 (19 fixed / 0 broken at both search budgets);
near-end 31.5 -> 7-9%; deep 10.2 -> 7.3% (64) and 7.7 -> 5.3% (800), p 0.016 /
0.031; immediate wins taken 89-92 -> 92-95% (0 broken); reveal traps,
predecessor, solver, quiet, ordinary unchanged. G2b-uniform -> G2b-priority:
nothing significant under search (raw deep must_block 14.2 -> 11.0%, raw
forced-loss value error +0.02). candidate_0060 -> uniform (plain retraining on
run07 iters 91-100): values better almost everywhere, but near-end must_block
blunders DOUBLED, 16.7 -> 31.5% (2 / 10, p 0.012-0.039) -- training imitated
run07's contaminated move targets (F4 directly observed); G2b more than undoes
it (7-9%, below the starting checkpoint). Hypothesis, untested: the same
contamination is part of why run07 stopped promoting after iteration 60.

**Owner decisions 2026-10-05:** G2b ON by default. It is a bridge: G4 in
self-play (plus the proven-loss guard below) should produce clean labels
natively. **Phase-out criterion:** on the first new run's buffers, measure how
often G2b changes a target; near zero -> remove it, otherwise the cheap moves
still need it. G3 DROPPED from the plan (code kept, off by default): it adds
frequency, not information, and showed no gain under search.

**Proven-loss guard (built 2026-10-05, `tree_resumable.rs`):** with tactics on,
a root move PROVEN lost (one-move `classify_actions` on the root, or a proof
search found) gets zero target mass and is never
played while a move without a proven loss exists. Needed because the Gumbel
improved policy blends Q with the prior, so a strong prior could keep target
mass -- and the move -- on a proven -1. Applies to self-play targets/moves,
arena and G0; the advisor's live panel reads visits and is not changed.

**BUILT 2026-10-05** (`--tactic-labels`, ON by default after the A/B; `dataset.apply_tactic_labels`,
Rust `derive_records(tactic_labels=True)`, `test_tactic_labels.py`; part of the
example-cache key). Measured on run07 iter 100 (1,000 games, G1 cap 4): 6.4% of
18,491 rows labelled -- 396 forced wins, 345 forced losses, 438 must_block.
New information: must_block move targets had >5% mass on a proven-losing move
in 107 of 255 policy rows (mean 16% moved); forced-win targets moved in 57 of
231 (11%); 55 new exact values and 37 corrected (expectimax -> exact); every
forced win in these games was won and every forced loss lost. Derivation 1.9 ->
2.5 s per buffer.

## G3 -- Controlled decisive-pattern sampling

Recipe to start from (Braun 4.2.4, KataGo policy-surprise weighting): 70% of
draws by priority, capped at 2x; 30% uniform; every sampled row keeps unit loss
weight -- priority changes frequency only. Priority signals: policy surprise,
search-vs-network value correction, proof membership. A/B against G1+G2 alone;
success = G0 decisive classes improve with ordinary-state calibration unchanged.

**BUILT 2026-10-04** (unit-tested, no training run yet; `priority_sampling.py`, `--priority-sampling`
off by default, `g3_offline_ab.py`, `test_priority_sampling.py`): priority =
mean of mean-normalised policy surprise (KL target || model policy) and value
correction (|root - model value|) from one no-gradient pass of the model about
to train; solver / certain-win / retained rows pinned at the cap; 30% uniform,
cap 2x; `train_steps(sample_weights=)` draws by it, loss weights untouched, and
counts `sampled_proof_rows`. Phase D recomputes per training call and stores
the report on the training row. Offline A/B: `g3_offline_ab.py --arm
uniform|priority` warm-starts candidate_0060 on a re-derived window with run07
loss settings; judge with `tactical_suite.py evaluate`.

**First offline A/B (2026-10-05)**: candidate_0060, iterations 91-100 (1,987
sealed games withheld), 2,000 steps per arm; proof rows drawn 1.53x (384k vs
252k presentations), max row 1.7x, effective sample 81%. Sealed G0, 300 cases
per class, PAIRED (`tactical_suite.py compare`, game-clustered bootstrap):
must_block near the end halved at every budget (31.5% -> 16.7-18.5%, 8-9 fixed
vs 0-1 broken, p 0.008-0.039); must_block overall at 64 sims 14.0 -> 10.3%
(p 0.007); deep must_block, reveal traps, predecessor, deep own_win: no change;
quiet value error slightly worse (+0.01-0.03, CI excludes 0); ordinary
calibration unchanged. Reading: G3 adds no information, only frequency -- it
amplified the exact labels that existed (endgame proofs, certain wins), which
are near the end of the game. Hence G2b. G3 stays OFF by default; re-test it
with G2b.

## G4 -- Exact tactics in search (reusable proof service)

Already shown causal (Library 6.94% -> 46.36%). Build natively:

- side-to-move ownership; same-actor pending choices (Mausoleum build ->
  retrieval does not flip the sign); terminal action sets; explicit solved results;
- a **persistent solved-node** marker, tested separately from the one-time leaf
  override -- a proved node must stop averaging in old NN estimates;
- then **bounded exact tactical search** returning proof / bound / unknown:
  max/min in a consistent frame on deterministic nodes, probability-weighted
  bounds on chance nodes (unresolved mass `m` widens the interval by at most
  `2m`; a win in one world is not a proof at the chance parent).

**Layer 1 BUILT 2026-10-04** (`--exact-tactics`, OFF by default; `tactics.rs`,
`tree_resumable.rs`, `test_exact_tactics.py`): proven-node marker for an
immediate guaranteed win, a port of `phase_e.guaranteed_win_now` (own pending
chains, every chance outcome, military/science only). A proven node is searched
like a terminal -- never expanded or evaluated, never averaged with NN values.
Gate: 0 mismatches vs the Python reference on 11,566 bot-game positions (192
positives, 3 inside pending choices). RICCP, run07 iter 60
(`riccp_923216750_review/g4_native_tactics.py`):

| | baseline | native G4 | prototype override |
|---|---|---|---|
| Library Q, ~4k sims, 3 seeds | 6.6-6.9% | 49.2-50.5% | 32.9% |
| Library Q, 16k | 6.9% | 51.1% (true 51.4%) | 46.4% |
| University root value, ~4k | 88-89% | 64-66% | (76% at 16k) |
| University top move, ~4k | Build University x3 | changed in 2/3 seeds | unchanged |
| seconds to ~4k sims | 4.2-5.0 | 2.0-3.2 | -- |

**Default ON 2026-10-04 (owner).** Laptop self-play A/B: -4.5% sims/s (CPU-bound).

**Layer 1b BUILT 2026-10-04: proven LOSSES** (`phase_e.guaranteed_loss_now` ->
`tactics::guaranteed_loss_now`; on with the flag, own switch
`set_exact_tactics_losses`). Gate: 0 mismatches / 10,463 positions, 95 losses
(28 inside the mover's pending choice). RICCP ~4k: University abandoned 3/3
seeds (win-only 2/3), already at ~1k; root 59-62%. Self-play vs win-only:
+7.5..+13% sims/s in 3/3 interleaved pairs (noisy) -- no measurable cost.
Advisor host: `SWD_ADVISOR_EXACT_TACTICS`, default on.

**Remaining G4 BUILT 2026-10-04.** Reference moved to `tactics.py`: extra-turn
wins (one replay, Theology included) and civilian last-card wins, which also
strengthen proven losses. Layer 2 = MCTS-Solver proof propagation in
`tree_resumable.rs` (exact winning edge, or all edges exact; chance edges only
with complete support; never the root; skipped under a specialist leaf bias).
Gate 0 / 16,676 (438 wins, 223 losses; +141 wins beyond `phase_e`, 77
civilian). First build cost -28% sims/s: every candidate's whole reveal support
was applied before checking, and the extra-turn screen let everything through.
Fixed by checking outcomes lazily and noting that play-again wonders carry no
shields or science (only Theology widens those reaches): check mean 91 -> 4 us,
self-play ~-2% vs off with bit-identical games to the slow build. RICCP
unchanged from win+loss (neither addition fires in those two trees).

Not yet: no broad suite or game-level measurement; bounded search with partial
[lo, hi] intervals (score-bounded MCTS) not built -- propagation covers exact
proofs only.

Measure at the **predecessor decision that loses the game** (University), not
only the tactical leaf. Validate correctness broadly and measure native
throughput before default-on. Difference from W11 (null): exact values, never a
rolled-forward NN evaluation.

## Review of G2 / G4 / G4b / G0 / G3 at 6ab4342 (2026-10-05)

`reviews/sevenwd-growth-g0-g4-6ab4342.md`: eight findings, all verified and
fixed; response in `GROWTH_G0_G4_REVIEW_REQUEST.md`. What changed in meaning:
combine nodes now pass up a coherent outlook (first expansion) or none (later
backups); a proven node is authoritative for stale in-flight settlements; the
Library offer's synthetic seed is replaced at attach; proofs are off under a
specialist leaf bias; G0 "trap"/"blunder" are exposure diagnostics, not regret;
G3's cap is a probability-space bound (max 1.7x uniform at the defaults);
G3's offline A/B withholds G0-sealed games; the expectimax utility loss uses
logsumexp. RICCP numbers were unchanged by the fixes (none of the defect paths
fired in those two trees).

## G4b -- Pending-choice expansion for Great Library and Mausoleum (owner decision 2026-10-02)

The net values **pending-choice** positions poorly (RICCP retrieval: 3.9% on a
won position) but values the resulting ordinary positions much better. So every
time search expands a node where a Great Library or Mausoleum choice is pending,
it evaluates **all** options at once instead of one NN call on the pending node:

- **Mausoleum (decision, no chance):** expand every legal retrieval from the
  public discard; one batched NN call (terminal/proven children take exact
  values via G4); node value = **max over children's current values** in the
  chooser's frame.
- **Great Library (chance then decision), with G8.0:** for the sampled card
  reveal, expand the 5 token afterstates; combine sorted values as
  `0.6*best + 0.3*second + 0.1*third` (the expected best of a random 3-of-5
  offer); reveals stay probability-weighted.

**Implementation contract -- what makes the optimism self-correcting:**

1. First visit: expand all options, cache their NN values, pass up the combined value.
   **This counts as ONE simulation, not 5 or N.** The batched evaluation of every
   option happens inside that single visit; the simulation budget is unchanged.
2. Later visits: descend into the currently best option(s) and search **below**
   them; never re-take the max of the same cached NN values (the net is
   deterministic, so that would lock the inflation in).
3. The node's value is always recomputed from current child values: refined Q
   for visited options, cached NN value for the rest.

**Risk -- max of noisy estimates (winner's curse).** With independent child
errors of spread sigma and truly equal options, the expected max is inflated by
~0.85 sigma (3 options), 1.16 (5), 1.54 (10), 1.87 (20); the Library rule equals
a fair best-of-3, +0.85 sigma. Smaller when one option is clearly best or errors
are correlated; zero for exact children. Biased toward the chooser, so it could
make building these Wonders look too good. Under the contract above, optimism
pulls visits to the inflated option and search below it corrects it; the
residual risk is **low budgets** (100-sim cheap moves) ending before correction.

**BUILT 2026-10-04** -- both halves,
with G8.0's sharing, in `tree_resumable.rs`; rides `--exact-tactics`, skipped
under a specialist leaf bias and at the root:

- When a Mausoleum retrieval or a Great Library token choice is first reached
  as a leaf, every option child is built and sent in the SAME request (exact
  children -- terminal or G4-proven -- need no row). The leaf is ONE
  simulation: each option is seeded with one visit at its value, its priors
  cached for its first ordinary visit, and the leaf passes up the best.
- The node becomes a `max_node`: every later backup through it passes up the
  best option's CURRENT Q (refined below visited options, cached elsewhere),
  so the max is recomputed, never locked in (contract points 2 and 3).
- Library (first version): the per-offer max sat under the build's sampled
  chance edge, whose noisy running average only converged to the formula.
  **Replaced 2026-10-04 by the exact formula:** the build
  edge samples only its card reveal; each reveal gets ONE `LibraryOffer` node
  whose choice holds the whole pool (built through the engine with any valid
  draw, options then widened), expanded over every token, reporting
  `sum_k w_k v_(k)`, `w_k = C(n-k, d-1)/C(n, d)` (0.6/0.3/0.1 for 3 of 5), from
  current token values on every backup. Fan-out: reveals x 10 offers ->
  reveals. Never G4-proven from its own (non-real) state; solved only when all
  tokens are exact. Forced root expansion enumerates reveals only; an offer
  node drops its forced network seed on first visit. Not representable (and
  left to sampling): empty pool, or the Library on the last card of Age I/II.
  With one offer node per reveal, G8.0 sharing has little left to share.
  Inside an offer node, selection skips options whose value is already exact
  (the first build kept re-visiting a proven Law token: 55% of sims hit exact
  leaves while the 0.3/0.1-weighted tokens went unrefined). RICCP Library Q:
  ~1k 50.9-52.0%, ~4k 51.4-51.8%, 16k 51.5% (true 51.4%; sampled draws gave
  45-53% at 1k); University corrected 3/3; ~6.8 s per 4k sims there (dense in
  Library/Mausoleum nodes). Self-play vs off: ~-2.5% sims/s (-1.2 / -3.8 /
  -2.3% per pair; sampled-draw G4b was ~-4%).
- Metrics: `option_expansions`, `option_rows`, `shared_afterstates` (self-play
  and `RustPuctSearch.tactics_metrics()`).
- Tests: `test_exact_tactics.py` (G4b / G8.0 sections), passing.
- RICCP (run07 iter 60, vs full G4 without G4b): Library Q at ~1k 45-53% (was
  47-48%), ~4k 51.9-53.1% (was 50.5-50.8%; ~1 pt over the true 51.4% --
  attributed at the time to the predicted max optimism; the review showed
  bookkeeping could also contribute, see the review-fix note), 16k 51.8%; University root ~4k 56-58% (was 59-62%),
  Build University top in 0/3 seeds; seconds per ~4k sims 3.8-5.2 (was
  1.8-2.6, baseline 4.8-6.5) -- each expansion costs one row per option.
- Self-play vs off (laptop, 3 x 32 games per arm): ~-4% sims/s (+6.5 / -6.0 /
  -3.9% per pair; full G4 without G4b was ~-2%); 8-11k expansions and 28-48k
  extra network rows per 32 games.

**Measure** (extend `riccp_923216750_review/test_immediate_guard.py` with an
arm next to "terminal override only"): Library edge Q at ~4k and 16k; the
University decision; quiet negative controls for inflation (expanded value vs a
deep-search reference); Mausoleum/Library build frequency at cheap and full
budgets; throughput at **equal time** (one visit now costs N evaluations).

## G5 -- Matched A/B of the value-target changes

Attribute iter 60: short-term targets + expectimax labels vs old targets. Now
part of step 5's online confirmation. Note the short-term target is blended into
the main target; KataGo instead uses separate short-horizon heads -- a different
experiment, worth a separate arm (G15).

## G6 -- Policy head simplification (DEFERRED)

r1 proposed making W5 the sole head. Deferred until refutation metrics (G0 tail
priors, sims-to-refutation) and deep-budget strength support it. Not
permanently dual-headed either.

## G7 -- Correction corpus

The 267-episode W9 corpus is a **development** set (it shaped the design), and a
fixed corpus is a starting point for robustness, not a closure (G13). Train it
in after G1-G3.

## G8 -- Chance fan-out

Problem: a revealing move splits visits ~10 ways; a chance-independent
refutation (prior ~0.03-0.08) needs ~165-400 visits at the reply node; the move
reads ~20 pts optimistic. Closed: W9, W10, W11. Chance capping is a throughput
change, not a fan-out fix.

- **G8.0 Great Library token afterstates (exact, owner decision).** Returned
  tokens go to the box and nothing draws from it again. For reveal `r`:
  `V(r) = (1/10) * sum_offers max_{t in O} V(s[r,t])`, i.e. with token values
  sorted in the chooser's frame `0.1*v3 + 0.3*v4 + 0.6*v5` (unit test). 70 ->
  7 reveals x 5 shared token subtrees. Reuse, not a guaranteed 6x gain (Braun:
  incidental transpositions in chess saved 0.02-0.18%; this sharing is
  structural, but measure the realized reuse). The engine removes every offered
  token from `unused_progress_tokens`, so post-pick states differ by offer:
  canonicalize only after proving those fields cannot affect future play, and
  keep chosen token, reveal, actor, extra-turn and pending effects in the key.
  **BUILT 2026-10-04** with G4b: Library token afterstates are
  canonicalized (unused pool cleared -- read only by the Library draw itself,
  invisible to the encoder) and shared by digest across every offer, so each
  "took `t` after reveal `r`" node is created and evaluated once. Realized
  reuse is `shared_afterstates` in `RustPuctSearch.tactics_metrics()` (a test
  confirms reuse happens; not aggregated in self-play metrics, not measured).
- **G8.1 Hybrid widening (heuristic).** Represent every outcome (one NN value
  each), deepen a selected few, back up with **true chance probabilities**, not
  allocation frequencies; keep an exploration floor. One evaluation per world
  does not cure shared confident errors (the Walls world shows replies can
  depend on the reveal). Only after G8.0 and G11.
- **G8.2 Targeted chance-aware reanalysis (training).** Reanalyse revealing
  moves with per-world budget; train the reply prior and pre-reveal value. Step 6.
- **G8.3 Afterstate value head.** Borrow Stochastic MuZero's separation of
  deterministic effects and chance -- not its learned dynamics (we have an exact
  engine). Only a carrier for G8.2's labels; trained on today's targets it learns
  the same optimism.
- **G8.4 Root verification (advisor only).** Per-world dedicated search for
  revealing top-k candidates; report mean and worst world. Any time.
- **G8.5 Thin-world penalty.** Last resort; risks under-playing necessary reveals.

## G9 -- Grouped Wonder/discard statistics (heuristic, owner decision)

Group node per idea (`Wonder X`, `discard`), per-card children keep their own
statistics; card builds ungrouped. **Not exact:** a group's mean Q depends on its
mix of good and bad burials, and changed selection can starve the one correct
burial. Mausoleum makes the within-group choice decisive (discards are
revivable, burials are not). Controls: group cardinality, misleading sibling
averages, prior aggregation, within-group exploration floor. Test on
Mausoleum-live positions (wrong within-group choice rate vs ungrouped) and the
corpus; hold to the W9 standard.

## G10 -- Action features and action-value supervision (W5b, owner decision)

- **G10a (early, cheap, falsifiable):** engine-derived, route-consistent
  immediate-effect features -- wins now, completes a science pair / sixth symbol
  (tableau AND discard routes; the asymmetry is real), military zone, extra turn
  with a legal follow-up, ends the age. "The net can derive it" shows
  information, not efficient learning. Test on held-out symbol/card/Wonder
  combinations with reachable negative controls. Moved to step 4.
- **G10b:** supervise action consequences across ALL legal actions where cheap
  to prove; for a sample, solved/deep-reanalysed critical alternatives as action
  values or preference inequalities; set-valued policy loss when the optimal set
  is known; partial proofs stay partial. Measure gradient interference before
  adding many losses.
- Rest of W5b (Wonder ordinal/retirement, pending-choice target token,
  payment/chain, W1/W2/W3 into the scorer): after G10a shows features move tail
  priors. Judge by the prior on the exact refutation vs a control, never by CE.

## G11 -- Search budget per move (fast/full)

The bias runs both ways; the dangerous direction is optimistic (a cheap
opponent misses the killer reply, the earlier position is labelled safe).
Policy targets are protected, value targets are not.

- **Measure (step 1):** reanalyse a buffer sample of cheap moves at full budget;
  count decision changes by reveal / threat / quiet.
- **Lead arm -- all moves at a fixed staged cap, compute-neutral.** Current mix
  averages ~519 sims/move (25% x 1,600 + 75% x 100), so every move at ~500 costs
  about the same, gives ~4x policy targets and removes cheap-move contamination.
  Braun dropped fast/full in chess (short games, clear outcomes; cheap moves were
  43% of search yet discarded as targets and still decided outcomes) and uses a
  staged cap (300 -> 800). 7WD games (~70 plies) resemble chess more than Go.
  Batching: a full-only workload removes the cheap/full tail-underfill problem.
- Other options: raise cheap sims (100 -> 300, ~+20%); triggered upgrades
  (top-k reveal AND live threat; root chance children disagree). **Caution
  (Braun 4.1.3):** a learned allocator improved deep-policy agreement yet online
  training trailed by 60-100 Elo -- shallow targets near the prior teach little.
  Judge any trigger by online learning and regret reduction per added
  millisecond, include randomized upgrades to expose trigger misses, and
  distinguish world uncertainty (reveals genuinely differ) from model
  uncertainty (repeated-search disagreement, budget-to-budget correction).

## G12 -- Restart archive from search states (Go-Exploit)

Retaining recorded wins cannot recover positions neither self-play side
reaches. Recipe to start from (Braun 4.2.1, Trudeau & Bowling 2023): half of
games start from archived states with an unresolved plausible alternative
(not near the end, not decided); archive priority by search-vs-network value
correction, 30% uniform; restart chooses an untried branch; exhausted states
leave. Also archive exact discoveries, policy surprise and predecessors of
missed opportunities; restart **before** the decisive choice as well as at it;
resample hidden deals conditional on the public history (historical hidden
identities never enter policy input or pre-reveal selection). Deduplicate
families, cap per ancestor, keep ordinary starts. Compare against the same
compute spent on ordinary games. Infrastructure partly exists (BGA log restarts).

## G13 -- Exploiters

Distinct from science/military specialists: maximize actual wins against a
frozen deployed player (civilian, economic, draft, tempo, reveal, combined).
Retain exploiters that expose a weakness even if weak in cross-play; feed real
unshaped outcomes to the general player, decide separately which policy labels
are safe. Evaluate repairs against **fresh** attackers not used in the repair
(Tseng et al. 2025: defenses against known attacks failed against new ones).
Cost: a training loop per attacker -- start with one short exploiter against
frozen candidate_0060.

## G14 -- Plasticity / initialization diagnostic (owner's random-init idea)

On identical corrected data, holdouts and row presentations: candidate_0060 +
fresh optimizer; value heads reinitialized; small final-trunk reset; fully
fresh init (with a learning curve, not a tiny fine-tune budget). No warm-optimizer
arm (no `learner_*.pt`). Stages: (1) can each arm fit a small corrected slice?
(2) does it survive interleaved ordinary replay? (3) does it generalize to new
families? Failure to fit -> optimization / plumbing / representation; fit then
loss -> interference; fit but no generalization -> features / coverage.
Down-weight early iterations' weaker targets. Primacy-bias evidence supports
the test, not a diagnosis.

## G15 -- Gating and target-structure questions

- **Within-run gating.** Braun publishes the latest weights every 500 steps
  within a model size and gates only size changes (a loss-based gate once
  promoted a model ~270 Elo weaker). Our soft gate was measured at 64 sims and
  may be throttling the generator. Decide after E0.
- **Separate short-horizon value heads** (KataGo) vs the current blend into the
  main target -- separate arm, after G2.

## Owner decisions (2026-10-01/02), as carried

1. Mausoleum: correct targets, then seed (G1/G2 -> G3/G12).
2. Great Library: 5 token afterstates (G8.0).
3. Grouped Wonder/discard statistics with individual values kept (G9, heuristic).
4. W5b re-explored (G10).
5. Random init after the buffer correction (G14).
6. Fast/full re-opened; throughput may be worth paying at the performance edge (G11).
7. Triggered budgets, not "reveals always full" (G11).
8. W3 board-control reliance check: candidate_0060 on the 907773062 ladder and
   threat corpus with real / zeroed / shuffled channels. Measures reliance, not
   benefit. Laptop, any time.
9. Great Library and Mausoleum: every time one is built, evaluate all options
   (G4b) as ONE simulation, not 5 or N. Library 0.6/0.3/0.1 over token values; Mausoleum best child.
10. (2026-10-04) G2: do NOT mask W4's victory type on proof rows. Missed
    certain wins (e.g. an extra-turn wonder uncovering a face-up winning card,
    then not taking it) stay undetected: strong players take them and the
    audit found all 54 taken. Revisit only if G0 shows otherwise.
11. (2026-10-05) Every move in the next run is fully searched -- no fast/full
    split. The G11 measurement step is dropped. **Rationale corrected the same
    day:** the contaminated targets G2b found, and the near-end blunders plain
    retraining taught, came from run07's FULL 1,600-sim searches -- cheap moves
    were never trained on as move targets (derivation drops them). The cause
    was search without exact tactics, not a low budget. The decision stands on
    other grounds: cheap moves still shape which positions the games reach
    while contributing no targets, and searching every move gives ~4x the move
    targets at a compute-neutral budget. The budget per move is still open.
13. (2026-10-05) Base checkpoint for the final run's preparation is
    **candidate_0100**, chosen without the arena. G0 (biased toward 100: it
    trained on the iter 61-100 games G0's cases come from) showed much lower
    must_block blunders under search (64 sims 11.7 -> 4.3%, deep 11.0 -> 2.4%)
    and worse raw value in lost positions (forced_loss error 0.184 -> 0.253).
12. (2026-10-05) G2b on by default with a measured phase-out; G3 dropped (code
    kept, off); the next run warm-starts from a laptop-pretrained checkpoint
    (option 3), subject to the G14 check.

## Review of WORLD_CLASS_MODEL_EVOLUTION_PLAN workstreams (2026-09-30)

| WS | What | Status | Verdict |
|---|---|---|---|
| W1 | slot embedding | on in run07 | keep; unattributed |
| W2 | tableau graph module | on in run07 | keep; unattributed |
| W3 | control channels | on; offline arms inconclusive | keep; reliance check only |
| W4 | hierarchical value | on, replaces joint7 | keep; fix its targets (G2) |
| W4b | distributional backup | recording only | park (linear collapse) |
| W5 | action scorer (+ `exposes`) | on in run07 | keep both heads for now (G6 deferred); W5b -> G10 |
| W6 | search integration | not started | superseded by G4/G8/G11; per-action telemetry still useful |
| W7 | specialist league | running in run07 (plan header stale) | keep; add exploiters (G13) |
| W8 | scale | policy | only per step 8 |
| W9/W10/W11 | chance-sibling stats / afterstate clustering / tactical extension | NULL | closed |

All representation workstreams shipped together into run07 unattributed; all
three search workstreams were nulls; none addressed retention or absorption.

## Deferred

- Full MuZero dynamics model; CFR/ReBeL-style rewrite (hidden cards are shared
  uncertainty here; public-history chance search is the natural model); generic
  mixture of experts; indiscriminate scaling; permanent risk penalties on
  revealing moves.
- Per-discard sixth-symbol input feature as a pure input -- covered by G10a's
  route-consistent effect feature.
- Revisit only after a specific limitation is demonstrated.

## References

Published AlphaZero-family results run at compute scales that hide rare-pattern
blind spots, mostly in deterministic perfect-information games, and report Elo
rather than blind-spot audits. The late-stage fixes live in engine write-ups.

- **Braun**, *Engineering Efficient Self-Play Chess: Search, Replay, and
  Throughput Under Limited Compute*, arXiv:2609.37447 (27 Sep 2026). Read in
  full. Used for: fast/full non-transfer (G11), learned-allocation failure
  (G11), late-game target poisoning (G1), restart archive (G12), policy-surprise
  sampling (G3), no within-size gating and loss-gate failure (G15), search-depth
  scaling (E0), sparse exact reuse (G8.0).
- **KataGo** -- Wu, *Accelerating Self-Play Learning in Go* (arXiv:1902.10565);
  `docs/KataGoMethods.md`: playout cap randomization, policy-surprise
  weighting, short-term value heads, predicted search error.
- **Leela Chess Zero** -- WDL head, moves-left head, tablebase rescoring (G2).
- **Go-Exploit** -- Trudeau & Bowling, AAMAS 2023 (arXiv:2302.12359) (G12).
- **Prioritized Level Replay** -- Jiang et al., ICML 2021 (G12 sampling).
- **Adversarial policies** -- Wang, Gleave et al., ICML 2023
  (arXiv:2211.00241); Tseng et al., AAAI 2025 (arXiv:2406.12843) (G13).
- **Plasticity** -- Nikishin et al., *The Primacy Bias in Deep RL*, ICML 2022;
  Dohare et al., *Maintaining Plasticity in Deep Continual Learning*
  (arXiv:2306.13812) (G14).
- **Chance nodes** -- Antonoglou et al., Stochastic MuZero, ICLR 2022 (G8.3);
  Couetoux et al., double progressive widening, LION 2011 (G8.1).
- **Hidden information** -- Schmid et al., *Student of Games* (2023); Brown et
  al., *ReBeL* (NeurIPS 2020) -- deferred.
- **Gumbel search** -- Danihelka et al., ICLR 2022 (cheap-search mode).
- **Retrospectives** -- Minigo write-ups; Cazenave et al., *Polygames* (2020).

Citations other than Braun are from memory or the external review; verify
titles and venues before quoting externally.
