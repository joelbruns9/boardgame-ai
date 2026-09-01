# World-Class 7 Wonders Duel Model Evolution Plan

## Objective

Build the strongest 7 Wonders Duel player in the world.

The existing model is already strong. This program must preserve what it has
learned while adding the structural, tactical, and strategic capabilities that
currently limit it. No model change is promoted merely because it improves its
own auxiliary metric or fixes one reviewed game. It must improve playing
strength under representative search and compute constraints.

The target system combines:

- a strong general neural policy/value model;
- evolving science- and military-specialist league opponents;
- an explicitly position- and graph-aware tableau representation;
- public-information tableau-control analysis;
- action-conditioned policy scoring;
- one internally consistent distribution over winner and victory type; and
- MCTS with exact or extended tactical handling where public terminal lines can
  be proved cheaply.

## Non-negotiable principles

### Preserve the current strength

`candidate_0085.pt` is the frozen reference model. It has approximately 15.8M
parameters, an eight-layer, 384-dimensional Transformer with six attention
heads, pooled readout, and encoder version `7wd-encoder-5`.

Its SHA-256 is
`0198a5f787d8b34c92373b30c3ff2f4e903821bda5703b28436603bd0f0c05ad`.
Its embedded metadata records iteration 85 and 86,000 generated games, but the
producing run directory, promotion history, and absolute-strength record do not
survive locally. Treat its historical provenance as unknown rather than
inferring a successful promotion from its filename.

Every compatible expansion should begin from this checkpoint. New paths should
initially be neutral or run in shadow mode:

- Append new scalar features rather than inserting them into existing tuples.
- Initialize projections for appended features to zero.
- Keep graph and action-policy additions behind residual gates.
- Keep the current flat policy and WDL head authoritative until their proposed
  replacements pass playing-strength gates.
- Keep the current production checkpoint deployable throughout the program.

Appending is a checkpoint-compatibility rule, not a claim that feature order
improves the neural computation. A linear projection has no preference for
whether related columns are adjacent. Inserting a column would shift the
meaning of every later checkpoint weight unless migration explicitly remapped
columns by feature name. That extra migration risk offers no playing-strength
benefit. Append related additions as documented blocks at the end of the
relevant token tuple.

The current padded feature width is 132 and is set by the GLOBAL token. The
remaining-pool and tableau tuples are 79 and 26 columns respectively. Several
appended tableau features therefore should not increase the stored tensor
width, while a GLOBAL append widens every padded token row. Put target-specific
control facts on tableau tokens and action-specific consequences on action
tokens whenever their semantics allow it.

Any GLOBAL-width change must update the Python schema, Rust
`FEATURE_COUNTS`, and the duplicate `FEATURE_WIDTH` literal in
`f4_cost_model.py`. Measure the actual fused inference path whenever any feature
group changes the schema.

Exact-zero residual gates are for migration-equivalence tests, not the default
training initialization. At `alpha = 0`, the gated module's own parameters
receive zero gradient; only the gate initially moves. After equivalence is
proved, begin training with a normalized small nonzero gate such as `1e-3`, or
give the new branch a direct warm-up loss while its deployment contribution
remains disabled.

### Optimize playing strength, not proxy accuracy

Validation loss, policy top-1, outcome accuracy, and tactical-corpus performance
are diagnostics. Promotion depends on arena strength, robustness across setup
regimes, and performance at equal wall-clock search cost.

### Keep public reasoning legal

Tableau-control features may use the public layout, revealed cards, legal
remaining-card pools, turn order, Wonder status, and game rules. They must never
condition on the actual identity of a face-down card that the player cannot
know. Unknown reveals are distributions over the legal pool.

### Generalize terminal reasoning

Science exposed the current weakness, but military can produce the same class
of multi-turn forced loss. Structural features, control calculations, outcome
heads, evaluation slices, and search extensions should treat science and
military as parallel instant-win channels.

### Correct confidently wrong priors deliberately

A strength-preserving warm start can also preserve a blind spot. Under PUCT, an
action with a near-zero network prior may require an impractical number of
simulations before it is examined. Training only from those search targets can
create a loop in which the action is never explored and therefore never earns a
better target.

Do not rely on additional simulations alone. The training system must support:

- root noise and forced playouts on selected full-search self-play moves;
- guaranteed examination of every legal action on targeted tactical positions;
- deep reanalysis or public-control oracle labels for critical alternatives;
- counterfactual ranking targets for proven blocking versus greedy actions;
- ordinary-position distillation to preserve existing play; and
- reduced or disabled old-policy distillation on correction positions.

Table `908370787` measured this loop concretely: the game-swinging refutation
carried a prior between 0.012 and 0.086. Note that a chance node multiplies the
problem, because the simulation requirement applies per chance world rather than
per action; correcting priors and sharing chance statistics (Workstream 9) are
complementary, and neither substitutes for the other.

**Amended 2026-09-01.** They are complementary in principle, but they are not
peers here. Sharing chance statistics was prototyped and measured a null at
every gain, for a reason that belongs in this section: the shared statistic is
seeded from the same confidently wrong evaluations, so a bonus strong enough to
act on it freezes the error instead of correcting it -- this very loop, one
level up. Prior correction does not substitute for statistic sharing; it is a
precondition for it.

Evaluation search remains clean and deterministic. Exploration mechanisms are
for creating better evidence, not for concealing a weak policy at deployment.

## Relationship to the plateau findings

`PLATEAU_FINDINGS.md` remains active prior evidence. Its aggregate measurements
found the plateau-era policy head within approximately 0.19 nats of its target-
entropy floor, found no broad value-head bottleneck, and explicitly recommended
measuring the encoder before investing in it.

The current candidate is not the plateau-era cloud5 checkpoint: all 139 shared
tensors differ, it adds pooled-readout and reply-head parameters, and it uses a
newer encoder signature. That difference does not prove it broke the promotion
plateau because no matching gate history survives.

The reviewed forced-science game supplies specific new evidence about rare
tableau topology, reveal control, and Wonder tempo; it does not invalidate the
aggregate plateau measurements. The second reviewed game, below, points
somewhere the plateau measurements never looked: its failure is in search
statistics, not in policy entropy or value-head capacity, so no aggregate
head-quality metric would have detected it.

Resolve these by front-loading inexpensive measurement and net-only
prototypes. Do not treat the plateau as proof that
structural work cannot help rare high-regret decisions, and do not pay for a
full structural cloud run before measuring whether those decisions carry a
meaningful prize.

## Second reviewed game: chance fan-out on a chance-independent reply

BGA table `908370787`, Age II, the human's move 31 (decision row 17 of
`runs/seven_wonders_duel/bga_game_log/table_908370787.jsonl`). Measured on the
frozen `candidate_0085.pt`. This game was won 60-53; the finding is about the
advice, not the result.

### Position

The Age II structure's right edge was a single cover chain:

```text
Shelf Quarry (r5c7) -> r4c8 -> Caravansery (r3c9) -> r2c10 -> School (r1c11)
```

`School` was **face-up from the start of the Age**. The opponent already held
`Apothecary`, so `School` completed their identical-symbol pair and yielded
`Theology`, converting their three remaining Wonders into extra-turn Wonders.
Removing `Caravansery` exposes `r2c10`; the opponent's unbuilt
`The Temple of Artemis` then takes `r2c10` and `School` in a single turn.

No hidden information gates the threat. `School` is public, the pair is public,
and the extra-turn Wonder is public.

### Measured failure

Root tree walk at 3000 simulations (`search_impl='python'`), the
`Discard for coins: Caravansery` edge held 1649 visits distributed over **ten
chance children**:

| world visits | Temple of Artemis visits | its prior | that world's top reply |
|---|---|---|---|
| 184 | 7 | 0.086 | Great Lighthouse |
| 182 | 54 | 0.044 | Build Brewery |
| 180 | 2 | 0.027 | Build Parade Ground |
| 169, 161, 159, 159, 155 | 3-7 | 0.039-0.082 | Great Lighthouse |
| 156 | 85 | 0.077 | **Temple of Artemis** |
| 154 | 2 | 0.012 | Build Walls |

The card at `r2c10` is face-down, and the opponent *buries* it under the Wonder.
Its identity cannot affect the legality of the Artemis-extra-turn-School
sequence or activate the buried card's effect. It can affect the opportunity
cost and the value of alternative replies -- `Walls` is the observed example --
so the chance outcomes must remain separate. The engine correctly models that
identity as a chance event, but search consequently splits one strategic reply
into ten statistically independent copies, each receiving roughly 160 visits.
At a prior near 0.03 the refutation requires on the order of a thousand visits in
a single world to be promoted. Eight of the ten worlds funded it at 2-7 visits,
which is to say they looked without learning anything. The edge's reported Q is
therefore a ten-way average in which eight terms are wrong.

> **Corrected 2026-09-01 by `w9_reference_case.py`.** An earlier draft said
> those eight worlds "never examined it at all". They do examine it -- the table
> above already shows 2-7 visits in each -- so the accurate statement is
> *examined but never funded*. The distinction matters: a mechanism that only
> guarantees the action is *looked at* fixes nothing here, and that is exactly
> what the Workstream 9 measurements went on to find.

More simulations eventually resolve this asymptotically, but not at a practical
root budget. At 45,000 root simulations the same edge held 2944 visits, or
approximately 294 per world. Isolated single-world probes at 6000 simulations,
where no splitting occurs, found `The Temple of Artemis` in **five of six**
sampled worlds at 5395-5853 visits.

> **Corrected 2026-09-01.** An earlier draft said six of six at 4700-5800.
> Re-measured with a deterministic slice across the enumerated ten-world support
> (`w9_reference_case.py --stages probes`), the sixth world is `Walls`, where the
> opponent correctly prefers `Build: Walls` (3589 visits) over Artemis (2371).
> The earlier sample evidently missed that world. This *sharpens* rather than
> weakens the finding: `Walls` is precisely the world this plan requires
> statistic sharing not to corrupt, and it is now a measured negative control
> instead of an assertion.

### Cost of the error

Deep-search reference estimates are 6000 dedicated simulations at the child
position, repeated across independent determinizations. They are not solved
game-theoretic values, but are sufficient to expose the advisor's large error.

> **Open question, 2026-09-01.** "Repeated across independent determinizations"
> is probably the wrong axis for a chance-node action. Determinization is which
> hidden cards exist; the chance worlds are which card the reveal turns up, and
> this plan already established that single-world determinization is *not* the
> defect (the searcher resamples chance itself). For the played move the
> decision-relevant number is also a probability-weighted mean over all ten
> worlds, not a min-max range over a sampled few. `w9_reference_case.py`
> implements the chance-world axis; settle this before spending the run.

| Action at the decision | Measured | Shown at 45k | Root visits |
|---|---|---|---|
| `Circus Maximus (using Caravansery)` | 59-70% | 66.9% | 41738 |
| `Build: Aqueduct` (keeps the chain closed) | 53-62% | 62.5% | 72 |
| `Discard: Caravansery` (played) | **42-51%** | **63.0%** | 2944 |

At the approximately 4000 simulations a 24-second turn affords, the losing move
was displayed **first** at 1000, 2000, and 3000 simulations, and second by no
more than about two points thereafter. The advisor recommended the losing move
for most of the available thinking time, and never expressed the roughly
twenty-point gap that separates it from the best move.

### What this evidence does and does not implicate

Correct in this position:

- Rules modelling. `Theology`, the extra turns, and the science pair are all
  handled; `The Temple of Artemis` is legal after every chain-opening move.
- The value head. The position after the opponent holds `Theology` reads
  approximately 43%, which is accurate.
- Determinization at the root. Six processes with six different injected hidden
  cards produced identical root output, because the searcher resamples chance
  itself. Single-world determinization is **not** the defect.

Implicated:

- **Search.** The reply is partitioned across **ten** chance siblings -- one
  exact refutation edge per world -- each competing inside a **two-way** Wonder
  burial-target branch. See Workstream 9.

  (Corrected twice on 2026-09-01. Earlier drafts said "roughly four" variants
  and "about forty buckets"; the opponent has exactly two accessible cards here,
  the newly revealed slot and `Aqueduct`. And "twenty buckets" is still the
  wrong shape: the two burial targets are not two copies of one reply. Burying
  the exposed slot removes the coverer AND buys the extra turn, so the terminal
  card falls in one turn; burying `Aqueduct` buys the turn and leaves the
  coverer standing, handing the card back. Ten exact edges with a two-way
  branch above each -- not twenty copies of the same idea.)
- **Encoder.** No feature expresses that the actor's own move brings a known
  threatening card within reach, and none pairs reach-distance with an
  opponent's unbuilt extra-turn Wonder. See Workstream 2.
- **Policy training.** The refutation carried a prior of 0.012 to 0.086. This is
  the self-reinforcing low-prior loop described under
  *Correct confidently wrong priors deliberately*.
- **Advisor presentation.** Per-action win percentages are read as comparable
  across actions and are not, because visit allocation is extremely unequal.

This complements rather than duplicates the earlier forced-science review. That
game concerned recognizing an already-public terminal threat. This one concerns
a threat the actor **creates** by its own move, where the punishing reply sits
behind a chance node that multiplies its discovery cost.

### Two defects found alongside -- both FIXED 2026-09-01

1. **FIXED.** `advisor_scrape.py` `_unseen()` built its candidate list by
   iterating a `frozenset`, so determinization depended on `PYTHONHASHSEED` and
   `resample_seed` reproduced nothing across processes. Now sorts by `CARD_IDS`
   as `pool.enumerate_card_reveal` already did.

   A **second** hash-seed leak in the same function was found while verifying
   the first: `_filler = next(iter(BACK_UNIVERSES[...]))` seeds the placeholder
   name for tableau slots, and that placeholder *survives* on ABSENT slots --
   only face-down PRESENT slots are overwritten. Also pinned, by `min(...,
   key=CARD_IDS.__getitem__)`.

   Measured before and after: every field of the reconstructed state (present
   cards, removed piles, future decks) differed under every `PYTHONHASHSEED`
   before, and is byte-identical after.
   `test_advisor_scrape.py::test_determinization_is_reproducible_across_hash_seeds`
   pins it and fails on the old code.

2. **FIXED.** `extension_7wd/README.md` now names
   `extension_7wd/candidate_0085.pt`. Its `cpu`-is-deliberate rationale was also
   stale: it was measured at 1.03M parameters and does not describe the 15.8M
   checkpoint, which has not been re-measured for device. That is now stated in
   the README rather than left to mislead.

## Baseline package: establish before model changes

Create a permanent baseline bundle containing:

1. The frozen `candidate_0085.pt`, its hash, embedded configuration, encoder
   signature, and an explicit record that its producing run/gate provenance is
   unavailable.
2. A fresh strength record against cloud5/cloud6 survivors, useful HOF
   checkpoints, deterministic baselines, and strong BGA/human play where
   available. Internal arenas are relative; top-human or external-engine play
   is the closest available absolute-strength evidence.
3. A target-mass-weighted prior/search disagreement report over existing
   buffers. This distinguishes cheap argmax flips from large policy-target
   disagreements using one network forward pass per position.
4. Actual action regret wherever per-action searched Q or exact proof exists.
   Target-mass disagreement is not value regret and must not be presented as an
   Elo estimate.
5. The corrected BGA representation for table `907773062`.
6. Search ladders for the end-of-Age-II decision and the public Age III forced
   loss.
7. The Law-conditioned replay measurements.
8. The prior science-threat corpus and immediate-block measurements.
9. A military-threat corpus built with the same definitions.
10. Throughput, batch-size, GPU-memory, and advisor-latency measurements.
11. Fixed arena seeds and opponents for repeatable promotion gates.
12. **BUILT 2026-09-01: `w9_reference_case.py`.** The table `908370787` Age II
    decision: the root tree walk with per-chance-child visit and prior
    breakdown, the single-world reference values, and the simulation ladder
    showing the rank flip. This is the reference case for Workstream 9 and must
    be re-measured after any search change.

    The harness loads the position through the same
    extractor -> determinizer chain the live advisor uses and *refuses* to run
    if decision row 17 no longer resolves to the reviewed decision, so a
    re-scrape or an extractor change cannot be silently measured as a search
    regression. Stages are selectable (`walk`, `ladder`, `probes`,
    `ref-values`); `--sweep-arms all` runs the mechanism arms against one model
    load and one position. Headline gate metric is
    `sims_to_promote_refutation`.

    Artifacts: the full report belongs under the gitignored `runs/`; the small
    diffable summary (`--summary-out`, carrying checkpoint hash, search config
    and position digest) is what a later change is compared against. Baseline
    summary committed at `w9_reference_baseline.json`.

    It reproduces the reviewed measurement exactly -- 1649 visits over ten
    worlds at 3000 sims, 172 tracked visits, per-world counts 7/54/2/85/2 --
    which is what licenses using it as the reference. `ref-values` is
    implemented but UNRUN; see the note under *Cost of the error* about which
    axis it should vary.
13. A per-action visit-allocation report for the advisor's displayed top-k, so
    the gap between a move's rank and its evidence is visible.

Required tactical slices include:

- Age I economy versus unique-symbol denial;
- Law present, Law absent, and Law absent/no-Great-Library setups;
- science attacks that succeed without Law;
- early and late military pressure;
- end-of-Age extra-turn timing;
- fifth-, sixth-, and seventh-Wonder retirement races;
- visible terminal cards with one or more coverers;
- positions where the actor's own move uncovers a visible threat, separated by
  chain distance and by whether the opponent holds an unbuilt extra-turn Wonder;
- positions whose punishing reply sits behind a chance node that does not change
  the replying player's action set;
- critical hidden locations evaluated only from their public remaining pool;
- preventable, unpreventable, and already-lost terminal threats; and
- positions where survival makes a civilian win highly likely.

For each tactical position, record the best action, alternatives, action value,
regret, preventability, oracle source, search budget, and whether hidden chance
was enumerated or sampled.

## Workstream 1: learned tableau positions

### Change

Add a learned identity for every structural tableau location, separate by Age.
A tableau token should include both component and location identity:

```text
card/back embedding
+ token-type embedding
+ learned AGE_AND_SLOT embedding
+ existing rule-feature projection
```

Retain the current row/x, accessibility, coverer-count, and per-player card
features. The learned slot does not replace them.

### Why

The current encoder provides numeric row/x features but no learned positional
embedding. A stable slot identity lets attention learn that a location has the
same structural role across every game. This is the least expensive board-
awareness improvement and is comparable to the input representation described
in the ZeusAI paper.

### Strength-preserving migration

Add a dedicated slot-embedding table initialized to zero. Loading the current
checkpoint should initially reproduce the old computation within numerical
tolerance. Train the new table while retaining all existing weights.

### Gate

- Migration-equivalence test before training.
- Improved tableau-control probes after training.
- No throughput or arena regression at equal search cost.
- Promotion only after general and tactical arenas pass.

## Workstream 2: graph-aware tableau encoding

### First implementation

Add a small tableau-only graph module before the main Transformer. Nodes are the
present tableau slots. Directed edge types include:

- directly covers;
- directly covered by;
- shared child or sibling branch;
- ancestor/descendant at distance 2, 3, 4, or 5; and
- no structural relation.

Begin with two or three lightweight residual message-passing layers. Feed the
updated tableau tokens into the existing Transformer with every non-tableau
token unchanged.

### Why not rely only on learned slots

Learned slot identities make the cover graph learnable, but the model still has
to discover every relationship statistically. Explicit graph edges make the
relationship available in one operation. Transitive-distance edges prevent a
five-step cover chain from requiring five neural layers merely to move the
relevant information between locations.

Table `908370787` is direct evidence for the transitive edges. The decisive
relationship was a four-step chain from the actor's own accessible card to a
face-up terminal threat, and the actor's move consumed one link of it. Add
reach-distance to accessibility as a slot feature, and pair it with the
availability and affordability of each player's unbuilt extra-turn Wonders. Do
not encode a rule that an extra-turn Wonder simply halves the distance: the
actual tempo depends on its burial target, affordability, accessible choices,
and seventh-Wonder retirement. The offline tableau-control oracle should instead
label the minimum player decisions required to reach a public target with zero
and with one available tempo use.

### Strength-preserving migration

Use a residual form:

```text
tableau_output = tableau_input + alpha_graph * graph_update
```

Use `alpha_graph = 0` to prove exact migration equivalence. For training, switch
to a normalized small nonzero value such as `1e-3` so the graph parameters learn
from the first step. Retain an ablation mode that forces the gate back to zero.

### Later option

If the lightweight module demonstrates value, test per-attention-head relation
biases in the main Transformer. Do this after measuring whether a custom
attention path reduces optimized inference throughput.

## Workstream 3: public tableau-control engine

### Purpose

Calculate the deterministic consequences of the public removal graph that a
strong human can count, especially:

- who must reveal or receive a critical card;
- whether the line is preventable;
- whether one tempo change flips control; and
- whether Wonder retirement removes that tempo resource first.

### Control state

Use a memoizable abstraction based on:

```text
age
remaining tableau-slot mask
player to move
publicly usable extra-turn wonders
total wonders built / next build ordinal
end-of-age starter-control state
critical target or terminal channel
```

The control engine is not a replacement for the full game solver. It answers
narrow ownership, reveal, and preventability questions on the public tableau.
The existing exact endgame solver is capped at six present tableau cards and
cannot label ordinary mid-Age forced-control positions. This is a separate,
narrow solver with its own correctness tests.

### Outputs

Per critical target and, where feasible, per legal action:

- `can_force_target`
- `opponent_can_force_target`
- `target_control_owner`
- `forced_revealer`
- `forced_taker`
- `decisions_until_exposed`
- `preventable_now`
- `preventable_within_k`
- `requires_extra_turn`
- `control_if_tempo_spent`
- `control_if_tempo_preserved`
- `forced_science_win_in_k`
- `forced_military_win_in_k`

For hidden locations, output probabilities calculated from the legal remaining
pool and card back:

- probability of revealing a missing science symbol;
- probability of revealing lethal effective shields; and
- expected terminal hazard of opening that branch.

### Cost-control levels

1. **Precomputed structural lookup:** fixed-layout facts indexed by Age,
   remaining mask, player, and simplified tempo state.
2. **Bounded online analysis:** search only the next 4–8 decisions during
   ordinary encoding.
3. **Exact threat-gated extension:** run deeper only when a public science or
   military terminal condition is active.
4. **Offline oracle:** compute expensive exact labels for training and tactical
   evaluation without paying that cost during every neural inference.

Cache aggressively and measure hit rate. Do not add the control calculation to
every MCTS leaf until its amortized cost has been measured against the value of
additional neural simulations it displaces.

### Integration sequence

1. Build the control engine and validate it against hand-labeled positions.
2. Run it offline on the tactical corpus.
3. Use the offline labels to decide which outputs have real predictive and
   action-regret value.
4. Only then add the successful bounded outputs as appended, zero-projected
   encoder features with Python/Rust parity.
5. Measure a threat-gated tactical search extension in shadow mode.
6. Promote neural features and search extension independently.

## Workstream 4: consistent winner/victory-type value

### Current limitation

The existing WDL and seven-way (`joint7`) heads are independent. MCTS backs up
only:

```text
P(win) - P(loss)
```

from WDL. The seven-way distribution is discarded by search, and the advisor's
victory-type display is a static raw-root evaluation.

### Target architecture

Use one hierarchical distribution:

```text
P(win, draw, loss)
P(civilian, science, military | win)
P(civilian, science, military | loss)
```

Derive the seven outcomes:

```text
P(my science)       = P(win)  * P(science | win)
P(opponent science) = P(loss) * P(science | loss)
```

and likewise for civilian and military. Search value remains:

```text
V = P(win) - P(loss)
```

This preserves the correct overall objective while making winner and victory
type consistent by construction.

### Migration

1. Retain the existing WDL head as authoritative.
2. Add conditional type heads initialized from or distilled against the current
   `joint7` head where useful.
3. Train hierarchical outputs in shadow mode.
4. Compare WDL calibration, seven-way calibration, and tactical action regret.
5. Switch search to the hierarchical WDL only after it beats the old head.

A KL-consistency loss between the two current heads may be tested as an interim
ablation, but it is not the target architecture: two wrong heads can be made
consistently wrong.

### Distributional MCTS backup

Treat this as a separate production integration from the hierarchical neural
head. The current Rust evaluator boundary returns only `(value_p0, priors)`.
Backing up seven outcomes requires a cross-language API change plus additional
storage and perspective/chance handling in both normal and resumable Rust
trees. Running it as telemetry prevents selection changes but does not remove
that plumbing cost.

`inference.py` already produces `joint7`, margin, military, and science outputs
for each neural evaluation. No additional network forward is required; the
missing work is carrying the distribution across the Rust boundary and storing
and backing it up through the tree.

Back up all seven outcome probabilities at each evaluated or terminal leaf.
Canonicalize actor-relative outputs to a fixed player perspective before
backup. Store per-edge probability sums alongside scalar value and visits.

Selection continues to use the scalar value. Because the scalar is a linear
function of the backed-up distribution, the displayed outcome vector and Q
remain consistent at every search depth.

The BGA advisor should show, for every recommended move:

- searched Q and visit count;
- your civilian/science/military win probabilities;
- opponent civilian/science/military win probabilities;
- draw probability; and
- change from the raw root or from the top alternative.

Initially run vector backup as telemetry only so it cannot change move
selection. Promote it to the authoritative displayed outlook after calibration
and perspective/chance-node tests pass.

## Workstream 5: legal-action tokens

### Current limitation

The Transformer encodes state components, but the policy is a flat linear head
over 1,202 fixed action identities. It must learn card × use × Wonder
combinations separately and infer each action's structural consequence through
one pooled state vector.

### Target design

Encode the state once. Construct a token or candidate representation for each
legal action from:

- action-use embedding;
- contextual source-card token;
- learned tableau-slot embedding;
- contextual Wonder token when applicable;
- pending-choice target token when applicable;
- affordability, payment, chain, and immediate-effect features;
- cards made accessible by the action;
- whether the action ends the Age;
- whether an extra turn has a legal follow-up;
- resulting Wonder build ordinal and retirement consequences; and
- bounded tableau-control changes from Workstream 3.

Score legal actions with a shared function over the global state and action
representation. Preserve the existing 1,202-index codec for replay, engine,
wire, and policy-target compatibility; action tokens change how logits are
computed, not how actions are identified externally.

### Strength-preserving hybrid

Do not replace the flat policy abruptly. Begin with:

```text
final_logit(action) = old_flat_logit(action)
                    + alpha_action * action_token_logit(action)
```

Initialize `alpha_action` to zero for equivalence testing. For actual training,
use a normalized small nonzero gate, or train the action scorer with its own
direct policy loss while its contribution to served logits remains disabled.
Use existing MCTS targets and selective teacher distillation to preserve
ordinary play before allowing the residual to become authoritative.

Do not train only to reproduce the existing priors. Add targeted reanalysis or
oracle-backed correction positions, and relax teacher distillation on those
rows; otherwise the residual can faithfully inherit the low-prior blind spots
it exists to fix.

Test three promoted forms:

1. Flat policy plus learned action residual.
2. Learned interpolation between flat and action policy.
3. Pure action-token policy, only if it proves stronger and retains throughput.

### Expected benefit

This is the path that should let the policy represent concepts such as:

> Sacrifice this accessible card to Sphinx, not Mausoleum, because the usable
> extra turn changes control of the final coverer while preserving the seventh-
> Wonder race.

It also generalizes learned Wonder effects across many source cards rather than
requiring every card/Wonder action index to receive sufficient direct examples.

## Workstream 6: search integration

The neural changes should reduce the simulations needed to see forced lines,
but a world-class player should also exploit exact public tactics directly.

Add search capabilities in this order:

1. ~~Shared action statistics across chance siblings, and Wonder-construction
   action abstraction (Workstream 9). Sequence these first: they are cheap,
   they are the measured cause of the table `908370787` blunder, and a budget
   extension spent on a forty-way partition is largely wasted.~~

   **Superseded 2026-09-01.** They were cheap, and they were prototyped first.
   They are not the cause: the partition is real but not binding, and neither
   mechanism moved `sims_to_promote_refutation`. The measured cause is the
   ~0.03 prior together with the network's -0.6 misvaluation of a reply that
   revises +0.56 the moment it is searched. Prior correction should lead; a
   budget extension is still largely wasted on a partition, but so is a
   bookkeeping fix on a prior. See the verdict box at the head of Workstream 9.
2. Threat-triggered search-budget extension when science or military hazard
   rises sharply.
3. Exact public tableau-control resolution when the bounded solver proves a
   terminal line.
4. Shadow distributional outcome backup after the hierarchical head is
   calibrated and the Rust cost is justified.
5. Per-action searched victory-type telemetry in the advisor, including
   per-action visit counts so an under-searched move is legible rather than
   presented as a confident near-tie.
6. Optional action-conditioned auxiliary Q prediction trained from searched
   action values, used first for diagnostics and move ordering rather than as a
   replacement for MCTS backup.

Measure strength at equal wall-clock time as well as equal simulation count.
An exact tactical extension that reduces blunders may be worthwhile even if it
lowers simulations per second; a graph module that consumes substantial GPU
time must earn back that cost in compute-normalized arenas.

## Workstream 7: evolving specialist opponents

### Timing and purpose

The early scripted science and military bots efficiently teach the rules of the
two instant-win conditions. Retain them for curriculum and smoke testing. Later,
they become too weak to create useful defensive experience and should be
replaced by neural specialists that remain strategically competent while
favoring a particular victory type.

Train the neural specialists only after the structural encoder, outcome model,
and action-policy foundation has been promoted. Otherwise, cloud compute would
be spent teaching specialists an architecture that is about to be replaced.

### Specialist training

Initialize each specialist from the promoted general checkpoint. Train with the
ordinary terminal result plus a victory-type bonus:

```text
science reward  = game result + lambda_science  * is_science_win
military reward = game result + lambda_military * is_military_win
```

Do not reward raw green-card count, shield count, or forward military movement
as the main objective. Those dense proxies can produce agents that accumulate
the resource while losing the game. Auxiliary predictions may use them, but
specialist promotion must be based on completed victories and playing strength.

Train a small population of reward multipliers rather than selecting one in
advance. Retain specialists on a practical Pareto frontier:

- intended victory-type frequency;
- overall score rate against current-best and HOF opponents;
- exploitability by the general population; and
- strategic diversity from existing specialists.

### League integration

Maintain separate general, science, and military archives. A general-model
self-play iteration should draw opponents from:

- current best;
- general HOF;
- active science specialists;
- science-specialist HOF;
- active military specialists; and
- military-specialist HOF.

The exact mixture is an experimental parameter. Report results by opponent
class rather than hiding them in one aggregate score.

Keep shaped specialist rewards in specialist-only buffers. Games injected into
the general replay buffer must retain the real unshaped WDL result. Exclude the
specialist seat's policy target when it should not be imitated; retain the
general seat's search target and the game result so the general model learns
strong defense and exploitation.

### Promotion gate

A specialist is useful only if it both:

1. creates materially more credible attacks of its intended type; and
2. remains strong enough that the general model cannot dismiss its moves as
   obvious strategic errors.

Reject specialists that increase victory-type frequency only by sacrificing too
much overall strength.

## Workstream 8: scale only after representation improvements

Do not assume that matching ZeusAI's approximately 92M parameters is the first
solution. The current 15.8M model is more rule-aware, but its tableau structure
and action representation are indirect.

After the structural and action changes are individually validated, run a
capacity sweep over depth and width. Larger models cannot generally be made
exactly equivalent to the old model as easily as zero-gated residual features,
so use:

- trunk warm starts where shapes permit;
- teacher distillation from the strongest smaller model;
- the same general replay window and specialist mixture;
- equal-training-compute and equal-inference-compute comparisons; and
- full cross-play rather than only candidate-versus-parent matches.

Promote scale only if it improves the strongest complete architecture.

## Workstream 9: shared action statistics across chance siblings

> ## PROTOTYPED AND MEASURED, 2026-09-01 -- READ THIS BEFORE THE SECTION BELOW
>
> Both mechanisms are built in `search.py` behind flags defaulting to off
> (`chance_sibling_bias`, `wonder_group_selection`), with exact off-equivalence
> verified and 170 existing search/chance/parity tests plus 26 new ones passing.
> The design as specified below is preserved for the record; these are the
> measured verdicts.
>
> ### Mechanism 1 (chance-sibling progressive bias): does not solve this case
>
> **Retracted 2026-09-01 (second revision): the first write-up of this box
> called it a "structural null" and said "there is no gain at which it works".
> Both were wrong**, and wrong for a reason worth recording. The first
> measurement matched the refutation by Wonder NAME, so it summed
> `Artemis using <the exposed slot>` -- the actual refutation -- with
> `Artemis using Aqueduct`, which buys the extra turn but leaves the coverer
> standing and hands the terminal card back. Those are near-opposite moves. The
> harness reported the wrong one as the group's best variant in 4 of 10 worlds,
> and the sharing key merged them, so the mechanism was never given the chance
> to share the right thing. See *The two keys are not the same key* below.
>
> Re-measured with the exact refutation tracked by exposed slot, and with
> sharing keyed structurally:
>
> | arm | refutation visits | promote in any / half |
> |---|---|---|
> | closed | **152** | 2000 / never |
> | closed + sibling, signed advantage | 181 | 3000 / never |
> | closed + sibling, positive-only | 208 (+37%) | 2000 / never |
> | closed + both | 223 (+47%) | 2000 / never |
> | closed + wonder | 240 (+58%) | 2000 / never |
>
> The baseline's true figure is **152, not the 172 first reported** -- 20 of
> those visits were the wrong burial target.
>
> **What survives.** No arm promotes the refutation in half the worlds at any
> budget tested, and none changes the root recommendation. The mechanisms do not
> solve this position.
>
> **What does not survive.** "Cannot work" and "no gain works" are withdrawn.
> Keyed correctly the sharing funds the refutation 37% better, and
> `closed+both` reaches 2 of 10 worlds at **6000** sims where baseline needs
> **12000** -- a 2x budget saving at that coverage. One position, one seed:
> suggestive, not evidence.
>
> ### The signed advantage was a real defect, and is fixed
>
> The original bonus clamped symmetrically, so a stale sibling estimate could
> argue *against* examining an action. Since those estimates begin as the raw
> network values that caused the neglect, a large gain then buried exactly the
> move the mechanism exists to surface: at gain 30 the refutation fell from 1st
> to 20th of 26 groups and its visits collapsed 172 -> 5.
>
> `chance_sibling_bias_positive_only` now defaults ON, and the measurement above
> confirms it matters: signed reaches 181 visits and needs 3000 sims to promote
> anywhere, positive-only reaches 208 and needs 2000. **A discovery mechanism
> must only ever add exploration pressure.** The signed form is retained as an
> arm purely so the difference stays attributable.
>
> The high-gain collapse remains the same self-reinforcing loop this plan
> describes under *Correct confidently wrong priors deliberately*, reproduced
> one level up inside the mechanism meant to break it -- that diagnosis stands,
> but it indicts the signed formulation, not sharing as such.
>
> ### The two keys are not the same key
>
> The mechanisms need different notions of "the same thing", and conflating them
> was the defect above:
>
> - **Mechanism 2** groups actions that should compete as one decision. Burying
>   any card under Artemis is "build Artemis", so the Wonder alone is right
>   (`canonical_action_group`).
> - **Mechanism 1** transfers evidence about the same TACTICAL action between
>   chance worlds. There the burial target is the whole point, and what
>   corresponds across reveal worlds is the structural action -- what you do, to
>   which SLOT, with which Wonder (`structural_action_key`). The card identity in
>   that slot is exactly what the chance event varies, so a card-derived action
>   index does not correspond; the slot does.
>
> ### A candidate variant: share the REVISION, not the mean
>
> | group | Q where barely searched | Q where searched | gap |
> |---|---|---|---|
> | Great Lighthouse (incumbent) | -0.392 | -0.370 | **+0.022** |
> | Temple of Artemis (refutation) | -0.504 | +0.056 | **+0.559** |
>
> A 25x discrimination that a mean throws away by construction, and revision is
> structurally immune to the freezing failure: an action nobody has searched has
> zero revision and so gets no bonus, rather than a confidently negative one.
>
> **Read this table narrowly.** It compares *different worlds* at one snapshot --
> lightly-searched worlds against heavily-searched ones -- and both columns were
> computed under the Wonder-name grouping, so they mix the refutation with the
> other burial target. It is NOT a within-action trace, and it does not license
> saying the value "revises the instant it is searched". The measurement that
> would license that does not exist yet:
>
> - the first simulation on which the exact refutation edge is visited;
> - its Q after 1, 2, 4, 8, 16... visits;
> - the simulation at which it becomes the top reply;
> - its raw prior and first leaf value.
>
> That trace is what separates a policy-prior failure from a value-depth
> failure, and it should be built before any revision-based variant is tried.
> **UNTESTED and, as measured, not yet properly evidenced.**
>
> ### Mechanism 2 (Wonder-action factorization): WORKS, NARROWER THAN CLAIMED
>
> It does what it says, absolutely, at the nodes it governs: multi-variant
> Wonder groups holding an unexplored burial target while another is revisited
> go from 146/2142 to **0/2056** (Gumbel root) and 180/1895 to **0/2246** (PUCT
> root). Baseline violates 7-9% of such groups.
>
> But it reaches **interior nodes only**. The Gumbel root runs top-k plus
> sequential halving and calls `simulate(action_index)` directly, so root edges
> never pass through `_select_closed`; under the PUCT root it does govern, but
> changes nothing, because an unvisited edge's prior term already guaranteed the
> look.
>
> **This contradicts the claim below that mechanism 2 is "the durable half"
> because it reshapes visit-count policy targets.** Those targets are built from
> ROOT visits, and the training root is the Gumbel one. Targets would still move
> indirectly -- a different interior tree yields different completed-Q -- but the
> direct mechanism described below is not there. Mechanism 2 also has **no
> strength evidence** yet, only a correctness property; judging it needs a
> corpus of multi-variant Wonder decisions rather than this one position.
>
> ### What this implies for sequencing
>
> The execution order below puts Workstream 9 first on the grounds that it is
> "the measured cause" of the blunder. The partition is real, but on this
> evidence it is **not the binding constraint**: correcting the key and the
> clamp buys 37-58% more funding for the refutation and still does not change
> the recommendation. What remains is the ~0.03 prior and the network's -0.6
> evaluation of a move that scores around +0.1 once actually searched. Prior and
> value correction should lead -- Workstream 5 and *Correct confidently wrong
> priors deliberately*.
>
> Note the earlier framing "it takes ~5400 visits to discover" is withdrawn:
> 5400 is where the isolated 6000-simulation probe *converged*, not what
> discovery cost. Discovery cost is unmeasured until the trace above exists.
>
> Nothing here is shippable to the advisor: this is the Python searcher and the
> advisor defaults to Rust. Per the sequencing below a Rust port waits on
> measured benefit, and mechanism 1 has none.

### Current limitation

One strategic reply is partitioned across many independent search buckets, so
its discovery cost is multiplied by the size of that partition. Table
`908370787` measured the partition at twenty buckets for a single idea
(corrected 2026-09-01 from "roughly forty"), each receiving about four visits
where roughly a thousand in one bucket were required.

The partition has two independent dimensions.

**Chance-sibling split.** Taking a covering card uncovers a face-down slot,
which fires a `CardReveal` on the *actor's own move*. The searcher creates one
child per possible identity. Measured on the `Discard for coins: Caravansery`
edge:
1649 visits over ten children of roughly 160 visits each.

**Action-variant split.** Constructing a Wonder is exposed as a separate action
per burial target. In each world the opponent held **two**
`Wonder: The Temple of Artemis (using X)` edges, plus the same variants for
every other unbuilt Wonder. (Corrected 2026-09-01 from "roughly four": the
opponent has exactly two accessible cards here.)

They "differ only in which card is buried" in the codec's sense and in no other:
only the variant burying the newly exposed slot is the refutation. Treating the
pair as interchangeable is what made the first Workstream 9 measurement wrong.

Combined, `The Temple of Artemis` received 172 of 1639 opponent visits, spread
so thinly that eight of the ten worlds funded it at only 2-7 visits -- enough to
look, never enough to learn (corrected 2026-09-01 from "never examined it at
all"). It does not
resolve at a practical budget: the per-world budget grows linearly while the
requirement is per-world, and at 45,000 root simulations the same edge held only
about 294 visits per world.

### Why the existing chance coalescing cannot help

`chance.rs` already coalesces chance children, but on an *observability* key:
two outcomes that look identical to the player are the same child, which off
`AGE_DEAL` means no coalescing at all. That rule is correct and must stay. The
ten revealed cards here are plainly distinguishable and genuinely change the
opponent's options and values -- in the `Walls` world the opponent simply builds
`Walls` for 141 of 154 visits. Merging these nodes would be unsound.

The problem is therefore not that distinguishable outcomes are kept apart. It is
that an action which is *correspondent across* those outcomes has to be
rediscovered independently inside each one, from a prior near 0.03.

### Change

Two independent mechanisms. Prototype both in the Python searcher first; they
compose but must be measured separately.

1. **Chance-sibling progressive bias.** Keep per-world child nodes and keep value
   backup exactly as it is, per outcome and probability-weighted. Add a bounded
   *selection-only* statistic, accumulated per
   `(parent edge, canonical action key)` across chance siblings, that biases
   exploration inside a world toward actions its siblings found strong. In the
   measured position `Wonder: The Temple of Artemis (using Aqueduct)` carries
   the same action index in every world, so it shares trivially; variants that
   differ only by burial target need the canonical key from mechanism 2.

   Use the shared statistic as a decaying progressive-bias or first-play-urgency
   term, not as raw shared visits: importing visit counts into PUCT can suppress
   rather than promote exploration. Aggregate only across siblings in which the
   canonical action group is legal, cap its influence, and decay it toward zero
   as direct local visits accumulate. Local evidence must override sibling
   evidence.

   Selection-only sharing does not algebraically mix incompatible world values
   into `q_p0`, but it still changes the sampled tree and can therefore affect
   finite-budget Q estimates and policy targets. Treat that as a measured bias,
   not as a claim of harmlessness.

2. **Hierarchical Wonder-action factorization.** Select the Wonder group first,
   then select the burial target within the promoted group, instead of exposing
   every `(Wonder, buried card)` pair as a flat sibling action. Preserve separate
   statistics, original action indices, and final policy targets for every full
   action. The sacrificed card's effect is not activated, but its slot and
   identity are not low-value: the target determines denial, tableau topology,
   and subsequent reveals.

   Initialize a Wonder group's prior from the sum of its legal member priors and
   guarantee exploration among the burial targets after the group is selected;
   otherwise the same low-prior failure can recur one level lower. This
   factorization also supplies the canonical action key mechanism 1 needs, and
   is useful on its own even where no chance node is involved.

### Why not rely on more simulations or a better prior

The practical-budget simulation argument is arithmetic and is given above. The
prior argument is circular: the prior is low precisely because self-play search
never funded the move, so the visit-count target never carried mass, so the
prior stayed low. This is the loop described under *Correct confidently wrong
priors deliberately*. Sharing discovery evidence changes the arithmetic instead
of assuming the prior improves.

### Where this lands

Both mechanisms are searcher changes, in `search.py` and the
`seven_wonders_rust` crate, and the searcher is shared:

- **Advisor.** Immediate effect at inference, with no retraining.
- **Training.** `self_play_many_net` uses the same crate, so visit-count policy
  targets change too. This was expected to be the durable half: what allows the
  next training run to break the low-prior loop rather than paying to overcome
  it every turn.

  **Not supported as written (2026-09-01).** Visit-count policy targets are
  built from ROOT visits, and the training root is the Gumbel one, which selects
  by top-k plus sequential halving and never passes root edges through
  `_select_closed`. Mechanism 2 therefore cannot reach the target's root
  allocation; mechanism 1 is a null regardless. Targets would still move
  indirectly through a different interior tree, but that is a weaker and
  unmeasured claim. Reaching root targets would require factorizing the Gumbel
  candidate set itself, which is a separate design.

Sequence the advisor benefit first for cheap validation, but do not treat it as
the deliverable.

### Interaction with existing chance work

`CHANCE_ENUMERATION_PLAN.md` distinguishes probability-weighted, fixed-support,
and ordinary sampled edges. Both mechanisms must preserve each class's
invariants: the mass accounting behind `q_p0` is untouched because sharing is
selection-only, and a fixed-support edge must remain closed against growth.
Mechanism 2 changes the action-edge layout rather than the chance layer, so
verify it against the same parity tests.

### Revisit open-loop search as a diagnostic, not the default architecture

The existing Python open-loop searcher automatically shares action-path
statistics across determinizations, so table `908370787` is a reason to include
it in the Workstream 9 experiment. It is not a reason to replace closed-loop
search by default.

Open-loop nodes alias publicly distinguishable reveal outcomes even though the
correct continuation can depend on the revealed identity. The current reference
implementation also caches the policy and value context from the first world to
expand an action-path node, then only re-masks legality in later worlds. Phase E
measured the resulting stale-prior signature and worse consequential trap
coverage, although its eleven consequential positions were too few to settle
equal-wall-clock playing strength.

**This arm would not measure what this section expects (2026-09-01).** Open loop
aggregates on the action path, and the refutation's action index is
card-derived: `Artemis (using Sawmill)` in one world, `Artemis (using Drying
Room)` in the next. The index therefore changes with the reveal, so open loop
does **not** naturally aggregate the exact refutation across worlds -- the very
sharing this section recruits it to demonstrate. It would still alias
publicly distinguishable outcomes and carry its stale-prior defect. Run it only
with a structural (slot-based) path key, or not at all; as written the arm is
mis-specified rather than merely expensive. De-prioritised.

Use open loop as the aggressive statistic-sharing control. The intended
production design remains a closed-loop hybrid: retain outcome-conditioned
nodes, priors, values, and probability-weighted backup while transferring only
bounded action-discovery evidence across siblings. If open loop solves the
reference case but regresses cases such as the `Walls` world, that is positive
evidence for the hybrid rather than evidence to switch architectures. Consider
an open-loop production path only if it passes the expanded conditional-outcome
suite and equal-wall-clock arenas after its stale-prior defect is addressed.

### Strength-preserving migration

Ship both behind flags defaulting to off. Off must reproduce current search
output exactly, verified by the existing Python/Rust parity, perspective, and
chance-node tests. Port to Rust only after the Python prototype shows measured
benefit, and keep the flags separable so a regression can be attributed.

### Gate

- Exact output equivalence with both features disabled.
- On a corpus of positions where the actor's move uncovers a known threat,
  report simulations required to promote the refutation, for each mechanism
  alone and for both together. Table `908370787` is the reference case.
- Run a five-arm Python diagnostic with the same frozen network and seeds:
  current closed loop, current open loop, closed plus sibling bias, closed plus
  Wonder factorization, and closed plus both. Report both equal-simulation and
  equal-wall-clock results.
- Confirm no regression where a chance outcome genuinely changes the correct
  reply or its value -- the `Walls` world above is the case that must not be
  corrupted. Re-run the prior consequential trap corpus and expand it with
  actor-created public threats before drawing an architecture conclusion.
- Measure refutation-discovery latency and action regret against shallow-exact or
  deep-search references; do not use only final action agreement.
- Arena strength at equal wall-clock cost, not only equal simulation count,
  since both mechanisms change work per simulation.
- After a training run with the searcher enabled, re-measure the policy prior on
  the reference refutation. The prior rising is the evidence that the loop
  broke; advisor-side improvement alone is not.

### Gate status, 2026-09-01

| gate item | status |
|---|---|
| Exact output equivalence, both features disabled | **PASS.** Bit-identical action, visits, policy target and completed Q on 6 positions; no added rng draw; no allocation on the off path. Each flag alone changes output on 5 of 6, so the equivalence is not passing vacuously. |
| Sims required to promote the refutation, per mechanism | **DONE; no arm solves the case.** All arms promote in some world at 2000 sims (signed sibling: 3000) and in half the worlds never. Refutation funding does improve: 152 baseline -> 208 (+sibling) / 223 (+both) / 240 (+wonder). Reference case only, not a corpus. |
| Five-arm diagnostic | **PARTIAL.** Five flag arms run (closed, +sibling positive-only, +sibling signed, +wonder, +both). The open-loop arm is excluded and now believed mis-specified -- its action-path key is card-derived, so it would not aggregate the refutation across worlds at all. Equal-simulation reported; equal-wall-clock not yet meaningful, because the first arm in a sweep pays warm-up costs and the recorded per-rung times reflect that, not the mechanisms. |
| Exact refutation tracked, not the Wonder group | **FIXED, was wrong.** The first measurement matched on Wonder name and summed both burial targets, reporting the wrong action as best variant in 4 of 10 worlds. Now matched by exposed slot, with `worlds_whose_best_variant_is_wrong` reported so the contamination stays visible. All amended numbers postdate this fix. |
| No regression in the `Walls` world | **NOT RUN as a regression test.** The `Walls` world is now characterised (isolated probe: `Build: Walls` 3589 vs Artemis 2371, so the reply genuinely differs there), which makes it usable as the negative control. With mechanism 1 a null there is nothing yet to regress. |
| Refutation-discovery latency and action regret vs deep references | **NOT RUN.** `ref-values` implemented but unrun, and its axis is unsettled -- see the note under *Cost of the error*. |
| Arena strength at equal wall-clock | **NOT RUN.** Premature: mechanism 1 is a null and mechanism 2 has no strength evidence, only a correctness property. |
| Prior rising after a training run | **NOT RUN.** No training run with these mechanisms is justified by the above. |

The corpus item is the honest gap in this verdict: everything here is one
position. Mechanism 1's null is diagnosed well enough that a corpus is unlikely
to reverse it -- the failure is in the mechanism's own feedback structure, not
in this position's specifics -- but mechanism 2 has *only* been shown to hold a
correctness property, and whether that buys playing strength is untested and
needs a corpus of multi-variant Wonder decisions.

## Experiment and promotion framework

### Cost-aware bundling with switchable additions

This is a personal strength-engineering project, not a publication. Perfect
causal attribution is less valuable than reaching a strong answer efficiently.
Test correctness separately, but allow interdependent foundational changes to
share one expensive training run.

Preserve inexpensive diagnostic control through independently switchable paths:

- frozen baseline;
- new component present but forced neutral/off;
- new component active;
- new component active without its auxiliary loss; and
- combined model.

For example, keep separate gates or masks for learned slots, graph updates,
public-control features, and action-policy residuals, while retaining both old
and new policy/value heads during transition. Post-training switch-off tests are
not perfect scientific ablations because the components learned together, but
they are adequate for deciding what to serve.

Pay for separate full training arms only when:

- the combined candidate fails and the responsible component is unclear;
- a component has a substantial throughput or memory cost;
- two designs are mutually exclusive; or
- the decision is large enough to justify the additional cloud run.

### Training and cost funnel

Use three levels of evidence.

1. **Laptop rejection testing:** correctness, migration equivalence, frozen-
   trunk/new-parameter training, short replay fine-tuning, tactical regression,
   small arenas, and throughput profiling. Overnight runs should reject broken
   or unpromising designs; they are not expected to prove a small global Elo
   gain.
2. **Combined architecture absorption:** warm-start the frozen checkpoint,
   train the bundled new paths on existing replay plus targeted reanalysis,
   freeze most of the trunk initially, then unfreeze it at a lower learning
   rate. Use ordinary-position distillation to control forgetting without
   suppressing tactical corrections.
3. **Cloud strength run:** generate enough new self-play for the expanded model
   to change its strategic distribution, then run broad and compute-normalized
   arenas. Reserve the RTX 5090 cloud box for candidates that already passed the
   first two levels.

The intended budget is one justified foundational cloud run, not one cloud run
per feature. Specialist and final mixed-league runs occur only after that
foundation is promoted.

Existing per-row `policy_weight` and `value_weight` should be reused to
emphasize derived correction rows. They do not themselves implement pairwise
action ranking or old-policy distillation. Those require action-level targets
and an explicit ranking or teacher-KL loss.

### Evaluation hierarchy

Every candidate passes:

1. **Correctness:** engine, encoder, Python/Rust parity, perspective, chance,
   terminal, and migration tests.
2. **Equivalence:** neutral initialization reproduces baseline policy/value
   within documented numerical tolerance.
3. **Tactical corpus:** action regret, calibration, preventability, and search
   depth required to identify forced lines.
4. **General arenas:** current best, multiple general HOF checkpoints, and both
   seats across fixed and fresh seeds.
5. **Specialist arenas:** active and archived science/military opponents across
   setup strata.
6. **Cross-play:** candidate families against one another to expose cyclic
   weaknesses.
7. **Compute-normalized arenas:** equal simulations and equal wall-clock search.
8. **Advisor canary:** real BGA observations with representation parity and a
   one-command rollback to the frozen checkpoint.

### Required reporting slices

- overall score rate and confidence interval;
- first-player and second-player score rate;
- civilian, scientific, and military results;
- Law/Great Library setup strata;
- science and military attack and defense rates;
- preventable terminal-threat block rate;
- end-of-Age and Wonder-retirement decisions;
- raw, shallow-search, and full-search calibration;
- simulations needed to reverse an incorrect raw preference; and
- games/second, neural rows/second, memory, and advisor latency.

### Promotion rule

No single tactical fix is allowed to trade away greater general strength. A
candidate becomes the new best only when the combined evidence supports a real
playing-strength improvement and no critical slice shows an unexplained
regression. Inconclusive candidates remain experimental and the frozen parent
continues serving.

Before an arena begins, name its primary metric, smallest effect worth
detecting, maximum game budget, and decision thresholds. Size the game count to
that effect rather than reusing 600–1,500 games automatically. Do not make every
reporting slice an independent hard veto; most are diagnostics and become
blocking only when a regression is large, repeated, or strategically critical.

Primary foundational gates are:

1. overall score at equal wall-clock search cost;
2. the public, preventable/unpreventable terminal-threat tactical gate; and
3. no material throughput or correctness regression.

If a bundled cloud candidate is statistically tied at its initial budget:

1. run switch-off diagnostics for graph, control, outcome, and action paths;
2. extend the arena only if the larger sample can resolve a worthwhile effect
   at acceptable cost;
3. reanalyze positions where candidate and parent disagree;
4. allow one targeted absorption fine-tune when new paths are visibly
   undertrained; and
5. keep the frozen parent authoritative if the candidate remains tied without a
   strategically important tactical gain.

## Recommended execution order

### Stage 0: freeze evidence and infrastructure

- Record the candidate hash, embedded metadata, missing provenance, and a fresh
  relative/absolute-strength baseline.
- Reconcile the program explicitly with `PLATEAU_FINDINGS.md`; do not claim the
  current candidate broke the plateau without gate evidence.
- Run target-mass-weighted prior/search disagreement over existing buffers.
- Measure actual regret wherever per-action Q or proof data exists.
- Create the permanent tactical corpus.
- Add the military analogues of the science-threat tests.
- Establish fixed, fresh, and compute-normalized arenas.
- Specify detectable effect sizes, maximum samples, and tied-candidate actions.
- Preserve the corrected BGA state fixtures.

Fix the two defects recorded under *Second reviewed game* before any
measurement depends on them: the `_unseen()` iteration order, which makes
determinization studies irreproducible, and the stale advisor checkpoint path in
`extension_7wd/README.md`.

**Status 2026-09-01.** Both defects are FIXED (plus a second hash-seed leak in
the same function, found while verifying the first) and the reference-case
measurement infrastructure is BUILT (`w9_reference_case.py`, baseline item 12).
The rest of Stage 0 is untouched: no fresh strength baseline, no permanent
tactical corpus, no military analogues, no arena specification. Those were
deferred deliberately -- they are days of arena time and did not gate the
Workstream 9 prototype -- and remain the largest outstanding block of Stage 0.

### Stage 1: cheap structural, action, and search prototypes

- Add zero-initialized learned Age/slot embeddings.
- Add the small-gated lightweight tableau graph module.
- Add a minimal legal-action residual using existing state tokens and legal
  action IDs; do not wait for the online control engine to test whether
  compositional action scoring has value.
- Keep the evaluator's returned scalar/prior boundary unchanged.
- Prototype the public tableau-control oracle offline in parallel, without
  placing it on every Rust leaf.
- **DONE 2026-09-01, and the result is a null.** Prototype the two Workstream 9
  mechanisms in the Python searcher behind separate off-by-default flags. Both
  are net-free and independently testable, so they do not wait on any
  representation change. See the verdict box at the head of Workstream 9:
  mechanism 1 is a structural null at every gain, mechanism 2 works but reaches
  interior nodes only. Exact-off equivalence verified; flags stay off; nothing
  ported to Rust.
- **NOT RUN.** Include the existing Python open-loop mode as a diagnostic arm on
  the same cases, but do not port or promote it merely for winning the single
  reference position. Deliberately excluded from the four-arm sweep: open loop
  is `mode="open"`, a different searcher with a known stale-prior defect, not a
  flag, so folding it in would confound a mechanism comparison with an
  architecture change. It should be run as its own arm against the same harness.
- Verify exact-off equivalence and small-nonzero training locally.

### Stage 2: consistent outcome and offline control evidence

- Train hierarchical outcome heads in shadow mode.
- Keep current WDL authoritative until the replacement passes calibration and
  arena gates.
- Hand-validate the public tableau-control oracle.
- Generate offline control labels and determine which proposed outputs improve
  prediction or action regret before adding them to the encoder.
- Expand the minimal action representation with contextual slot, graph, Wonder,
  and proven-useful offline control outputs.
- Train action scoring as a small-gated residual on the existing flat policy.
- Keep old flat policy logits available for authority, blending, and rollback.

### Stage 3: laptop absorption and rejection gate

- Train new parameters first with most of the existing trunk frozen.
- Use existing replay, targeted reanalysis, and forced exploration of tactical
  alternatives.
- Unfreeze conservatively at a lower learning rate only after the new paths
  begin learning useful corrections.
- Run tactical regression, small arenas, migration checks, and throughput
  profiling overnight.
- Re-run the table `908370787` reference case and report simulations required
  to promote the refutation, for each Workstream 9 mechanism alone and for both
  together. (`w9_reference_case.py --sweep-arms all`. Already run on the frozen
  net at Stage 1: every arm returned 2000 / never / never, i.e. no mechanism
  changed it. The value of re-running here is to see whether prior correction
  moved it, not whether the search mechanisms did.)
- Drop or revise failures before renting cloud compute.

### Stage 4: combined foundational cloud run

- Warm-start from `candidate_0085.pt`.
- Train the compatible structural, proven-useful control, outcome, and action
  paths together.
- Preserve separately switchable gates and old authoritative heads.
- Use ordinary-position distillation for retention and targeted labels for
  correction.
- Generate new general/HOF self-play so the architecture can move beyond the
  old replay distribution.
- Promote only after broad and compute-normalized arenas support a real gain.

### Stage 5: justified production Rust integrations

- Add only control features that earned value offline, with Python/Rust encoder
  parity and placement on tableau/action tokens where possible.
- Test the threat-gated public-control search extension at equal wall-clock
  cost.
- Keep the hierarchical neural head separate from full distributional backup.
- Implement seven-way Rust tree backup and searched per-action victory telemetry
  only if its diagnostic/playing value justifies the evaluator and tree-plumbing
  cost.

### Stage 6: neural specialist population

- Retain scripted science/military bots for future early curriculum.
- Fine-tune science and military specialists from the newly promoted general
  architecture.
- Train several victory-type reward multipliers and keep strong, diverse
  specialists rather than the most extreme ones.
- Add specialist/HOF mixture support without contaminating general rewards.
- Establish specialist promotion, exploitability, and diversity reporting.

### Stage 7: final mixed league and scale

- Retrain the general model with the full general/HOF/specialist league.
- Run depth/width scaling experiments.
- Select by world-class playing strength at the intended advisor search budget.

## Definition of success

This program is successful when the promoted model:

- preserves or exceeds the current model's general economic and civilian play;
- recognizes **public and unpreventable** forced science and military losses at
  raw or shallow-search depth;
- blocks preventable terminal routes without indiscriminate overdefense;
- deliberately preserves and spends extra-turn Wonders to control critical
  future cards, and recognizes when its own move hands the opponent that
  control;
- finds a punishing reply that sits behind a chance node at a simulation budget
  comparable to the same reply with no chance node in front of it;
- understands fifth-through-seventh Wonder retirement timing;
- remains robust against general, science-specialist, and military-specialist
  populations;
- reports searched per-action victory modes consistent with its Q values;
- improves at equal wall-clock search cost; and
- demonstrates sustained advantage through broad cross-play and top-human/BGA
  evaluation rather than one narrow benchmark.

## Primary implementation locations

- `encoder.py`: learned slots, appended public-control features, encoder schema.
- `f4_cost_model.py`: duplicate flat feature-width literal that must track any
  new GLOBAL maximum until it is eliminated or schema-derived.
- `net.py`: graph module, hierarchical outcome heads, action-token scorer.
- `dataset.py`: slot/relation/action tensors and hierarchical targets.
- `train.py`: conditional outcome losses, new ranking/distillation losses, and
  migration; existing row weights can emphasize but not define new targets.
- `search.py`: vector outcome backup, tactical extensions, and Workstream 9.
  Sharing is selection-only and must not touch `_Edge.q_p0`; the shared
  progressive-bias statistics hang off the parent `_Edge` beside its `children`,
  and the Wonder hierarchy factors selection while preserving full-action
  statistics and policy targets. **Built 2026-09-01 exactly as described**
  (`_Edge.sibling_stats`, `ClosedNode.selection_groups`,
  `canonical_action_group`, `_select_within_group`), behind
  `chance_sibling_bias` and `wonder_group_selection`, both off. Note
  `_select_closed` governs interior nodes and the PUCT root only -- the Gumbel
  root bypasses it.
- `w9_reference_case.py`: the reference-case harness (baseline item 12) and the
  mechanism arm sweep. `test_w9_mechanisms.py` and `test_w9_reference_case.py`
  hold the off-equivalence gate and the scope limits.
- `seven_wonders_rust/src/chance.rs`: the observability-keyed coalescing that
  Workstream 9 must leave intact, and its `enumerate_chains` key.
- `advisor_scrape.py` and `pool.py`: determinization; `_unseen()` must sort by
  `CARD_IDS` as `enumerate_card_reveal` already does.
- `CHANCE_ENUMERATION_PLAN.md`: the edge-class invariants that statistic sharing
  must preserve.
- `rust_bridge.py` and `seven_wonders_rust/`: inference/search parity and
  performance path.
- `phase_d.py`: 7WD-specific self-play population, league assignment, and
  replay-mixture integration.
- `games/az_loop/`: game-agnostic promotion lifecycle, soft gates, probation,
  ratchet, HOF archive primitives, and run control.
- `advisor_adapter.py`: searched per-action victory-mode reporting.
- `endgame_corpus.py`: existing exact solver's six-present-card limit; not the
  proposed public tableau-control solver.
- `PLATEAU_FINDINGS.md`: prior plateau measurements and cheap-test rationale
  that constrain this program.
- `SCIENCE_BLOCKING_AND_WONDER_TEMPO_REVIEW.md`: evidence and design rationale
  that motivated this plan.
