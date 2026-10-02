# 7WD Model Growth Plan

**Status:** DRAFT 2026-09-29. Nothing below is built. Written after run07's
resume on `2fe60a7` (expectimax proof labels + short-term value targets) and the
RICCP game 923216750 review.

## Where we are

- run07 promoted once after the resume (iter 60, gate 0.553 [0.517, 0.588]
  vs iter 20), then stalled again: gates vs iter 60 read 0.533, 0.488, 0.520;
  self-anchor STAGNANT. The gain cannot be attributed to the new targets yet --
  there is no matched arm.
- **Policy is not the bottleneck.** W5 already beats the flat head by ~0.3 nats
  held-out (0.95 vs 1.29); mixed adds only ~0.02 over W5 alone. Policy fit
  improved a lot while strength stayed flat.
- **The value head smooths over rare, decisive tactics** -- for both seats.
  Evidence, all consistent:
  - RICCP review (`runs/seven_wonders_duel/riccp_923216750_review/`): an
    immediate Mausoleum science win valued at 3.9%; across the buffer such
    positions average 47% predicted vs ~94% target; ordinary sixth-symbol wins
    82%.
  - Coverage: 20 retained Mausoleum immediate-win rows in 297k (1 in ~15k).
    The cheap-search example filter (`record_fast_moves=False`) drops 34/54 of
    them and 2,412/3,156 ordinary immediate science wins.
  - Science threat prior: opponent science under-predicted by .4-.77, worst
    early (`sevenwd_science_threat_prior`).
  - BGA reviews: search overrated card-revealing moves (Port -25 pts).
- **Search cannot compensate across chance nodes.** Great Library edge frozen at
  754 visits / Q 6.94% from 4k to 16k sims; visits spread over 70 outcomes, one
  terminal science node reached once. Four W9 visit-reallocation mechanisms
  already failed -- the loop is prior/value-seeded, so only training breaks it.

**Thesis:** the next strength comes from getting decisive tactical positions into
training with exact targets, and from exact (not heuristic) terminal checks in
search. Not from more policy representation, more sims, or more search
bookkeeping.

## Rules for this plan

1. Every intervention gets its own measurement before it is combined.
2. The tactical suite (G0) is the fast instrument; gates and the self-anchor are
   the strength instrument. A change must move the suite AND not lose the gate.
3. Measure magnitudes on the real buffer, not reasoned ones.
4. One matched A/B at a time on the box; laptop-scale where possible.

## G0 -- Tactical test suite (instrument, build first)

Held-out positions with a known answer, split by class:

- own immediate win available (build, Mausoleum retrieval, military, wonder);
- opponent threatens an immediate win next turn (must-block);
- solver-proven endgame results (exact / exact_expectimax);
- chance-node cases like the Great Library tree (win one move below a chance node).

Source from run buffers held out of the training window, the BGA game log, and
the equivalence corpus; label with the engine and endgame solver. Report per
class: value error vs proven result, policy mass on winning/blocking moves, and
search Q at fixed sims. Target a few hundred positions per class; runs in
minutes against any checkpoint.

Deliverable: `tactical_suite.py` + frozen suite file + baseline table for
iter 20, iter 60, and the latest checkpoint.

## Failure-mode diagnosis (2026-10-02)

**Root cause is the value head (and tail priors) on rare decisive positions;
search is the amplifier.** Search corrects a bad leaf value wherever it funds
the node -- the RICCP retrieval node reads 3.9% raw and Q=100% after 764 sims.
Games are lost where the wrong value/prior sits BEHIND A CHANCE NODE, so the
correcting visits are split 10-70 ways:

| case | value / prior error | search corrects? |
|---|---|---|
| RICCP retrieval node | 3.9% on a 100% win | yes (764 sims) |
| RICCP before Great Library | same values, one level down | no: Q frozen 6.94% from 4k to 16k sims, 70-way split |
| 908370787 ToA refutation | prior .012-.077, value 36 pts wrong | no: ~160 visits/world vs 165-400 needed; ~8 corrects once funded |
| science threat prior | opponent science -.4 to -.77, worst early | only near the end |
| BGA Port | revealing move ~25 pts optimistic | no: fan-out |
| W11 | 2 plies of forced line recover 19% | value is fixed by real search depth |

Fixing values makes the fan-out matter less; fixing the fan-out lets search
correct values. Both are attacked (G1-G3 values, G8/G9 fan-out, G11 labels).
Caveat: three games and one buffer audit. Frequencies are unmeasured -> G0b.

Working taxonomy:

- **F1** own decisive win misvalued (Mausoleum retrieval, sixth symbol) -- value.
- **F2** opponent decisive threat under-predicted (science threat) -- value/prior.
- **F3** refutation unfunded behind a chance node -- prior x search allocation.
- **F4** cheap-move label contamination -- data (G11).
- **F5** strategic: not fixed by exact leaves or a known reveal.

## G0b -- Oracle-swap attribution study (scope first, with G0)

On positions where the model played a losing move, rerun the decision four ways:

| | NN leaf values | exact leaf values (solver / engine) |
|---|---|---|
| normal chance fan-out | baseline | fixed here -> value failure (F1/F2) |
| actual reveal known in advance | fixed here -> fan-out failure (F3) | fixed only here -> both |

None fixes it -> F5. Sources of decisive mistakes:

- endgame-solver proofs in the run07 buffers (~8.5k solved positions per
  iteration; a played move that changes a proven result is a labelled blunder;
  endgame-biased);
- run07 gate games (`results.jsonl`, if detailed enough);
- BGA losses via `bga_review` (separates luck from play).

Output: frequency table over F1-F5 that sets the effort split between G1-G3,
G8/G9 and G11. Solver-proof source alone should be a laptop day.

## G1 -- Keep decisive positions (data coverage)

Always emit training examples for positions with an available immediate win,
an immediate opponent threat, or a solver proof, regardless of cheap/full
search. Value target always; policy target only where search quality is
adequate (or use the exact winning-move set).

- Measure first: fraction of raw decisive positions currently retained, per class.
- Cost: buffer growth should be small (decisive positions are rare); measure it.
- Success: G0 value error on the retained classes drops; no gate loss.

## G2 -- Exact targets for proven positions

Where a win/loss is proven (immediate terminal, solver proof), the proof replaces
the bootstrapped target entirely -- extend what `73a3504` did for endgame proofs
to immediate tactical wins anywhere in the game. RICCP audit found a retained
immediate win with a 55% blended target.

- Check interaction with `--short-term-value-weight` (proofs already replace all).
- Success: no proven position has a target below ~0.95 in the buffer audit.

## G3 -- Oversample rare decisive patterns

Even with correct labels the net reaches 47% on Mausoleum wins: rarity, not
only label quality. Weighted sampling of G1 rows (cap the multiplier; watch the
value calibration on ordinary positions).

- A/B against G1+G2 alone. Success: G0 improves beyond G1+G2 with value
  calibration on ordinary positions unchanged.

## G4 -- Exact terminal check in search

Before accepting an NN leaf value, check whether the side to move has an
immediate winning action (including pending choices such as Mausoleum
retrieval); if so, back up the terminal value. Exact, cheap, and different in
kind from the failed W9 mechanisms.

**Closest prior work is W11 (NULL), so state the difference up front.** W11
rolled forward a *non-terminal* threatening build and asked the net about the
result; it recovered 7 of 37 points because the error lived in the opponent's
replies, not in the rolled plies. G4 returns an exact terminal value and never
asks the net. Its expected reach is limited, though: in the Library tree the
win sits behind Joel's reply (RICCP answers every reply with Mausoleum ->
retrieve), so a 1-ply check fires only at leaves after Joel's reply, and the
Mausoleum build + retrieval is a two-step same-actor chain the check must
follow (no chance, forced pending choice). Measure how many Library-tree leaves
it actually converts before building further.

- Measure on the RICCP Library tree and G0 chance-node class; measure the
  throughput cost (must be small -- evaluator-bound generation).
- Optional follow-on: one-ply opponent-threat check (must-block) under the same
  rules.

## G5 -- Matched A/B of the value-target changes

Attribute iter 60's promotion: short-term targets + expectimax labels vs old
targets, same init, same seeds, same games. Decides whether to push that line
(e.g. tune `short_term_value_weight`, decay) or drop it.

## G6 -- Simplify the policy head (next fresh run)

Make W5 the sole policy head; drop the flat head. Simplification, not new
capability. Do NOT build the rest of W5b (W1/W2/W3 feeds into the scorer) until
G0-G4 show the policy is limiting again.

## G7 -- Correction corpus training (already built)

The 267-episode W9 correction corpus targets underrated refutations -- the same
failure family. Train it in after G1-G3 so its effect is measured on G0.

## G8 -- Chance fan-out: optimistic Q on card-revealing moves

**Problem** (`sevenwd_root_second_move_underpriced`, W9-W11): a move that
reveals a card splits its visits over ~10 worlds; a chance-INDEPENDENT
refutation with prior ~0.03-0.08 needs ~165-400 visits at the reply node to be
discovered, and each world gets ~160. The move's Q reads ~20 pts optimistic.
Closed: sibling statistic sharing (W9), afterstate clustering (W10), tactical
leaf extension (W11). Note chance capping (`cheap_double_reveal_offsets`) is a
throughput change for cheap moves, not a fan-out fix -- it keeps 3n children.

Ordered candidates:

1. **Progressive widening at chance nodes** (double progressive widening,
   Couetoux et al. 2011). Expand a new world only when visits exceed
   `k * N^alpha`; otherwise re-descend an existing world. With ~3 live worlds
   instead of 10, the reference edge's 1,649 visits give ~550 per world --
   above the measured discovery threshold. The bias is bounded because the
   refutation is chance-independent, so finding it in any world is the point;
   the cost is higher variance in Q for chance-DEPENDENT moves. Different from
   W9: it moves visits, it does not share statistics seeded from wrong values.
   Measure on the 908370787 reference case (`sims_to_promote_refutation`) and
   the 267-episode corpus before any arena.
2. **Targeted chance-aware reanalysis (training).** For card-revealing moves in
   the buffer, reanalyse each world with the budget a no-chance reply would get,
   and train on the result: the reply prior for the refutation and the value of
   the pre-reveal position. This attacks the root cause (the 0.03 prior and the
   36-pt-wrong value), which every W9-W11 measurement pointed at. The S2b
   coalesced reanalysis path already exists; the new part is selecting revealing
   moves and giving per-world budget. Pairs with G7's correction corpus.
3. **Afterstate value head** (Stochastic MuZero, Antonoglou et al. 2022). Have
   the net predict the expected value of the pre-reveal afterstate directly,
   trained from (2), and use it at chance nodes as a prior/blend while children
   are thin. Larger change (new head + search hook); only after (2) shows the
   labels are learnable.
4. **Root verification pass (deployment only).** When a top-k root candidate
   reveals a card, run a dedicated search per sampled world at a non-chance
   budget and report the mean (and worst world). Fixes the advisor and gate
   play without touching training; cost is paid only on revealing candidates.
   Cheap to build, directly addresses the BGA Port-style overrating.
5. **Thin-world uncertainty penalty** (heuristic, last). Penalise Q by
   `c * sigma / sqrt(min world visits)` for revealing moves. Risk: systematically
   under-plays necessary reveals. Only if 1-4 fail.

Instrument: G0's chance-node class plus the reference case. Success = the
refutation is promoted at a simulation budget comparable to the same reply with
no chance node in front of it (WORLD_CLASS "definition of success").

## Owner decisions 2026-10-01

1. **Mausoleum: correct, then seed.** Re-derive the 101 run07 buffers with
   G1/G2 rules (decisive positions kept, proven positions get exact targets)
   BEFORE any seeding or oversampling, so the added volume carries correct
   labels. Then seed: G3 oversampling plus self-play starts from decisive
   positions (KataGo side positions; BGA log restarts).
2. **Great Library: 5 token afterstates (G8.0, exact).** The returned tokens go
   to the box and nothing draws from it again, so the post-pick state is the
   same whichever offer contained the token: V(offer) = max of its three token
   values, offer probabilities analytic. Library goes 7 x 10 = 70 outcomes ->
   7 reveals x 5 shared token subtrees (each ~6x the visits). Check that
   `unused_progress_tokens` does not make post-pick encodings offer-dependent;
   if it does, key the transposition on value-relevant state.
3. **Grouped action statistics, individual values kept (G9).** Two-level
   selection: a group node per idea (`Wonder X`, `discard`) aggregates visits
   and Q; children are the individual burial/discard cards with their own
   statistics. Card BUILDS stay ungrouped (the card matters). Mausoleum caution:
   discarded cards are revivable, buried ones are not, so the within-group
   choice can decide games -- the group only allocates visits to the idea, it
   never replaces the per-card value. Test specifically on Mausoleum-live
   positions: rate of wrong within-group choice vs ungrouped. Retest with G8.0
   on the corpus, not one reference case (W9 m2 was judged on one).
4. **W5b re-explored (G10).** Unbuilt features, first two prioritised:
   immediate effect (wins / completes science pair / military zone),
   extra-turn has a legal follow-up, ends-the-age, Wonder ordinal and
   retirement, pending-choice target token, affordability/payment/chain,
   W1/W2/W3 outputs into the scorer. Target is the TAIL prior on refutations
   (0.012-0.077 measured), not average CE. Test: prior on the exact refutation
   vs a flat-policy control on the same data.
5. **Random-init run comes AFTER the buffer correction** (1), as an offline
   comparison: random init vs fine-tune of candidate_0060 on the corrected
   buffers, scored on G0. Down-weight or drop early iterations.
6. **Fast/full search re-opened (G11).** The bias runs BOTH ways, and the
   optimistic direction is the dangerous one: when a cheap-searched opponent
   misses the killer reply (Mausoleum science build, extra-turn Wonder), the
   earlier position is labelled good, so the value head learns that allowing
   rare refutations is safe -- self-reinforcing with the blind spot. Policy is
   protected (cheap moves carry no policy target); value targets are not.
   Current mix: 25% full at 1600, 75% cheap at 100 -> mean ~519 sims/move;
   all-full is ~3x cost per game. At the performance edge the throughput hit
   may be worth it, but measure first: reanalyse a buffer sample of cheap
   moves at full budget, count decision changes, split by reveal / threat /
   quiet. Options by cost: raise cheap sims (100 -> 300 ~ +20%); triggered
   upgrades; all-full.
7. **Triggered full budget, not "reveals always full".** Most boards have a
   revealing move, so a reveal trigger is ~all-full. Narrower triggers: a top-k
   candidate reveals AND an opponent threat is live (science pair or five
   symbols, military near a zone, Mausoleum with a live discard, extra-turn
   Wonder available); or the root's chance children disagree widely in value.
   Progressive widening is orthogonal (allocation within a budget, not the
   budget) and helps cheap moves most (100 sims over 10 worlds = 10 each), but
   it drops unexpanded worlds -- use the hybrid (expand every world once for
   its NN value, deepen a risk-weighted few, probability-weighted backup).
   Order: exact restructurings (G8.0, G9) first, then the G11 measurement, then
   choose triggers / widening.
8. **W3 board-control check on targeted positions.** candidate_0060 on the
   907773062 ladder + threat corpus with real / zeroed / shuffled control
   channels. Measures reliance, not benefit.

## Review of WORLD_CLASS_MODEL_EVOLUTION_PLAN workstreams (2026-09-30)

| WS | What | Status | Verdict for this plan |
|---|---|---|---|
| W1 | learned slot embedding | built; on in run07 (`--slot-embedding`) | keep; no isolated strength evidence |
| W2 | tableau graph module | built; on in run07 (`--graph-module`) | keep; no isolated strength evidence |
| W3 | public tableau-control engine + input channels | shipping; offline arms inconclusive (placebo best) | keep on; do not reopen offline testing |
| W4 | hierarchical win x victory-type value | on in run07, replaces joint7 | keep. RICCP: the flat win head is equally pessimistic, so W4 is not the cause |
| W4b | distributional backup | recording only | park: linear collapse means it cannot change move selection |
| W5 | action-token scorer (+ `exposes`) | on in run07 | succeeded as a policy predictor; becomes sole head (G6). Rest of W5b parked |
| W6 | search integration | not started | items 2-3 (threat-triggered extension, exact control resolution) are superseded by G4's narrower exact check; item 5 (per-action telemetry) still worth doing for review work |
| W7 | specialist league | **built and running in run07** (the plan's "NOT STARTED" header is stale) | keep; no attribution yet |
| W8 | scale after representation | policy | unchanged: capacity only if G1-G3 move the suite but not the gates |
| W9 | chance-sibling statistics | NULL | closed |
| W10 | afterstate clustering | NULL | closed |
| W11 | tactical leaf extension | NULL | closed; see G4 for the difference |

Pattern across the program: the representation workstreams (W1-W5) all shipped
**together** into run07 with none individually attributed, and all three search
workstreams (W9-W11) were nulls. The one thing no workstream addressed is what
the RICCP audit found -- decisive positions are filtered out of training and
under-learned when kept. That is this plan's G0-G3.

## Deprioritised

- More sims / bigger search budgets (16k sims left the Library edge frozen) --
  except G11's targeted full-budget upgrades, which are a label-quality change.
- New heuristic search bookkeeping (four W9 nulls). The exact restructurings
  G8.0 and G9 are not in this class.
- Per-discard sixth-symbol feature (net can derive it from card identity;
  revisit only if G1-G3 fail to close the gap). W5b itself is re-opened as G10.
- Capacity increase -- revisit only if G1-G3 improve the suite but not the gates.

## Sequencing

0. run07 ENDED 2026-09-30 at iter 100 (revert_reset; best = iter 60). All 101
   buffers, candidates 0-100 and logs are local in
   `runs/seven_wonders_duel/run07_bundle` (main folder; no `learner_*.pt`).
1. G0 suite + baselines and G0b attribution study (laptop).
2. G1 coverage measurement, then G1 + G2 build (laptop, unit + buffer audit).
3. G4 build + tree measurement (laptop).
4. Box run: G1+G2+G4 vs control, G0 tracked per promotion. G5 folds in here if
   the box budget allows a third arm.
5. G3, then G7, each as its own arm.
6. G6 at the next fresh run.
7. Fan-out, per owner decisions: exact restructurings first (G8.0 Library
   afterstates, G9 grouped Wonder/discard statistics), measured on the corpus
   and G0 chance class; then G11's cheap-move measurement; then choose
   triggered full budget and/or hybrid widening (G8.1). G8.4 root verification
   any time for the advisor. G8.2 reanalysis as a box arm after G1-G3; G8.3
   afterstate head only after G8.2.
8. G10 (W5b features) and the W3 control-reliance check can run alongside on
   the laptop; random-init vs fine-tune on the corrected buffers after step 2.

## References

Why this plan exists: published AlphaZero-family results run at compute scales
that hide rare-pattern blind spots, mostly in deterministic perfect-information
games, and report Elo rather than blind-spot audits. The late-stage fixes live
in engine write-ups:

- **KataGo** -- D. J. Wu, *Accelerating Self-Play Learning in Go* (2019,
  arXiv:1902.10565), and `docs/KataGoMethods.md` in the KataGo repository.
  Playout cap randomization (origin of our cheap-move policy filter), auxiliary
  targets, post-superhuman fixes. Most relevant single source.
- **Leela Chess Zero** -- lczero.org blog and GitHub discussions: WDL value
  head, moves-left head, **tablebase rescoring** of training targets (the
  model for G2 and the expectimax labels).
- **Adversarial blind spots** -- T. T. Wang, A. Gleave et al., *Adversarial
  Policies Beat Superhuman Go AIs* (ICML 2023): a superhuman value net blind to
  a rare pattern; KataGo's follow-up adversarial training. Model for G0/G3.
- **Chance nodes** -- I. Antonoglou et al., *Planning in Stochastic
  Environments with a Learned Model* (Stochastic MuZero, ICLR 2022): afterstates
  and chance codes (G8.3). A. Couetoux et al., *Continuous Upper Confidence
  Trees* (LION 2011): double progressive widening (G8.1).
- **Hidden information** -- M. Schmid et al., *Student of Games* (Science
  Advances 2023); N. Brown et al., *ReBeL* (NeurIPS 2020).
- **Gumbel search** -- I. Danihelka et al., *Policy Improvement by Planning
  with Gumbel* (ICLR 2022); already the cheap-search mode.
- **Engineering retrospectives** -- Minigo project write-ups; T. Cazenave et
  al., *Polygames: Improved Zero Learning* (ICGA Journal 2020).

Citations are from memory (knowledge cutoff mid-2026); verify titles and venues
before quoting them externally.

## Open questions

- Is value error on decisive positions the main cause of lost games? Classify the
  decisive error in recent gate/BGA losses (underrated opponent reply / misjudged
  quiet position / luck) with `bga_review`.
- Does the stall persist once G0 is fixed? If so, optimization or capacity is
  the remaining suspect (`sevenwd_cloud6_stall_diagnosis`).
