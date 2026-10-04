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

## Execution order

Steps 1-4 are laptop work and can overlap; 5+ need a box.

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
  conditional; masking would drop type supervision on ~26% of rows.
- Measured on run07 iters 60 / 100 (1,000 games each, G1 cap 4): 18,384 /
  18,491 rows; exact proofs 1,186 / 1,298; expectimax 3,804 / 3,909; certain
  wins 306 / 347 (types civ/sci/mil 93/121/92 and 133/132/82; 86-90% also have a
  proof); short-term newly reaches W4 on ~13.3k rows.
- Validation numbers are unchanged by the contract (proofs off there; a certain
  row's exact target equals its realised one).

## G3 -- Controlled decisive-pattern sampling

Recipe to start from (Braun 4.2.4, KataGo policy-surprise weighting): 70% of
draws by priority, capped at 2x; 30% uniform; every sampled row keeps unit loss
weight -- priority changes frequency only. Priority signals: policy surprise,
search-vs-network value correction, proof membership. A/B against G1+G2 alone;
success = G0 decisive classes improve with ordinary-state calibration unchanged.

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

Measure at the **predecessor decision that loses the game** (University), not
only the tactical leaf. Validate correctness broadly and measure native
throughput before default-on. Difference from W11 (null): exact values, never a
rolled-forward NN evaluation.

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
