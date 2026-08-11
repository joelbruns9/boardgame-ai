# Chance-aware Kingdomino search and full-training review brief

Status: prepared for third-party review; no implementation or new training run
is authorized by this document.
Branch under review: `codex/kingdomino-chance-correct`
Current production checkpoint:
`runs/kingdomino/best_checkpoint/current_best.pt`
Checkpoint SHA-256:
`4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`

## 1. Review request

We would like an independent review of the chance-aware search work completed
so far and a concrete plan for the next implementation and training cycle.

The project owner is now convinced that chance-aware methodology deserves a
full training cycle. The question is no longer whether to abandon chance-aware
search after the first pilot. The question is how to run a credible full cycle
without repeating the pilot's likely training-distribution problems or wasting
most of the added inference work at the chance boundary.

Please review:

1. whether the evidence below justifies continuing to a full cycle;
2. whether the proposed batched, depth-gated progressive-widening search is
   information-set safe and statistically sound;
3. whether the proposed parent/child visit and backup semantics are correct;
4. how much conditional search should be guaranteed after a reveal before
   activating additional reveal rows;
5. whether deck 12 should join deck 8 in the first full cycle;
6. how replay continuity and chance-example weighting should be handled;
7. what minimum gates should precede the full run without turning the project
   back into an indefinite sequence of proxy experiments; and
8. whether the proposed cloud execution and promotion design are sufficient.

The preferred outcome is a reviewed implementation plan and frozen full-run
specification, not another broad exploratory test programme.

## 2. Executive summary

The evidence supports the following conclusions.

- The current open-loop search is information-safe before a reveal, but it
  aliases distinct public states after the reveal. Explicit observation
  splitting materially changes root policies.
- At deck 8, search reliably reaches the reveal boundary from the first
  selection root. Lack of causal reach is not the bottleneck.
- The structural observation split, not a 70-row one-shot network bootstrap,
  accounts for almost all of the measured policy change.
- A forced-action continuation duel was inconclusive on all disagreements but
  directionally positive; the preregistered pick-changing subgroup showed a
  positive raw-margin interval. This is credible evidence that at least some
  changed decisions are better, not merely different.
- A ten-iteration deck-8/deck-12 training pilot worked mechanically and moved
  the network toward sampled-split preferences on held-out disagreements.
- The selected iteration-2 checkpoint nevertheless showed a confirmed
  whole-game regression. Giving both sides training-aligned chance search did
  not rescue it.
- A matched two-iteration ordinary open-loop training control also trended
  weaker. Its result was inconclusive rather than a confirmed regression, but
  it establishes that fresh-buffer short warm-start training is a credible
  common cause. The chance mechanism cannot be declared dead from the pilot.
- Earlier A1c progressive-widening work produced a small `X=4` oracle signal
  concentrated in one of eight positions; `X=8` regressed. That screen was
  correctly classified as inconclusive. It warns against assuming that more
  chance width is always better.

Our current hypothesis is that the next search should reduce early chance
volatility with a small batched mini-panel, retain distinct observation
subtrees, and spend most subsequent work deepening the informed decisions
after the reveal. Chance width should grow only after the active conditional
subtrees have received enough real search.

The current training recommendation is a staged real cycle: establish replay
continuity with ordinary open-loop self-play, then enable the reviewed chance
search at natural replay frequency (initially 1x, not 2x), run long enough to
outgrow fresh-buffer burn-in, and promote only through a preregistered paired
match. A run expected to exceed 12 laptop hours should move to a cloud GPU only
after a short end-to-end throughput and recovery probe.

## 3. Terminology and notation

### 3.1 Correct decision semantics

Let `x` be a decision made before the next row is known, `R` the public row
revealed by chance, and `y` a decision made after observing `R`. The target is:

```text
max_x E_R [ max_y Q(x, R, y) ]
```

The search must avoid both incorrect extremes:

```text
E_R [ max_x max_y Q(x, R, y) ]   # pre-reveal clairvoyance
max_x max_y E_R [ Q(x, R, y) ]   # post-reveal action aliasing
```

The action `x` is shared across reveal outcomes. Conditional action `y` may
adapt after the row becomes public.

### 3.2 Historical `X` versus proposed width/depth notation

The existing roadmap and A1c documents use `X` as *per-tile exposure*. A
tile-balanced cycle for a bag of `n` tiles contains `n/4` rows, so historical
`X=4` at deck 8 corresponds to eight panel rows, not four.

Recent design discussion has sometimes used “sample four rows” colloquially.
To prevent a serious implementation misunderstanding, this document uses:

- `W`: number of active reveal rows at a chance node;
- `W0`: initial active width;
- `D_r`: real conditional visits/search work for active row `r`;
- `D_min`: minimum conditional work required before widening; and
- `M`: full support size (`70` at deck 8, `495` at deck 12).

Any implementation specification must keep this notation or define an equally
unambiguous replacement. Do not reuse `X` for both exposure and row count.

## 4. Current search and training behavior

### 4.1 Production sampled split implemented on the branch

The current opt-in training treatment applies to all four `PLACE_AND_SELECT`
roots at deck 8 and deck 12. Before the reveal, PUCT commits to the action
without reading hidden order. At the first stochastic transition, the action
becomes an afterstate/chance node. Public reveal rows route to separate lazy
observation subtrees.

Current semantics are:

- fixed complete support identity (`70` or `495` rows);
- one balanced sampled row per chance crossing;
- persistent row-specific observation subtrees;
- ordinary sampled backup;
- zero initialization/bootstrap rows; and
- only the first stochastic reveal in one simulation is explicitly split.

The network remains frozen during each MCTS search. It supplies leaf priors and
values. Chance-aware structure changes conditional subtree statistics, backed
values, root visits, and ultimately the MCTS policy target. Optimizer updates
occur only after self-play examples enter replay.

Internal counterfactual observation nodes are not independently written to
replay. They influence the root target and selected trajectory indirectly. A
post-reveal state becomes a direct training example only when it is actually
reached in the real self-play game and searched as a new root.

This distinction matters: the pilot distilled an aggregated pre-reveal policy
signal, not every conditional policy discovered in counterfactual branches.

### 4.2 Why the current approach can be volatile

One chance crossing returns one row's continuation value. A bad early row is
an unbiased sample, but it can lower the pre-reveal action's finite-budget Q,
reduce subsequent PUCT allocation, and prevent the chance node from receiving
enough visits to correct the estimate. Balanced traversal improves coverage
only after the action is revisited.

Conversely, evaluating many rows once can overinvest in breadth. A raw network
evaluation at the first post-reveal state gives priors and a value, but it does
not substantially search the consequential placement/pick choices after the
reveal. The completed 70-row panel work suggests this breadth-only allocation
has little incremental value.

## 5. Evidence completed so far

### 5.1 Chronological summary

| Work | Main result | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| A1/A1b and A1c oracle work | Observation splitting has decision headroom; sealed A1c `X=4` signal was small and concentrated in 1/8 positions; `X=8` regressed | Information-set topology matters; width/depth tradeoff is real | Whole-game strength or a globally valid widening schedule |
| Exhaustive deck-8 panel | 70-row exact immediate expectation implemented correctly | Full immediate chance support can be batched without fake visits | Stronger play |
| 400-unit whole-game panel A/B | 63-65, 49.22%, margin -0.766; intervals crossed null; 89.1% of control throughput | Valid weak-negative/inconclusive screen and real cost measurement | Harm, equivalence, or training potential |
| Causal reach/leverage sweep | Every one of 64 roots reached a reveal even at 800 sims; sampled/panel policies increasingly diverged from control | Deep reveal is reachable and can influence root allocation | Direction or quality of the changed actions |
| Sampled-split ablation | At 6,400 sims, control-to-sampled TV 0.383; sampled-to-panel TV 0.015; only 1/64 panel top-action change | Explicit split creates almost all policy movement; enumeration adds little | Playing strength |
| Forced-move duel | Primary: +1.523 margin `[-0.056,+3.162]`; pick-changing subgroup: +2.884 `[+0.059,+5.665]` | Directionally positive action-quality evidence, especially for tile changes | Whole-game Elo or a definitive primary positive result |
| Ten-iteration chance-aware pilot | 4,000 games; 3,000 steps; natural treated buffer share 9.18%; 2x weighting produced ~17% treated optimizer draws | Production integration, observability, and network absorption all work | Strength gain |
| Chance iteration-2 confirmation | 42.77% open-loop; 42.58% under shared aligned search; both upper bounds below 50% | Iteration 2 is genuinely weaker; inference mismatch does not rescue it | That chance search caused the regression |
| Matched open-loop training control | 45.31%, margin -2.24, interval `[39.33%,51.43%]` | Ordinary two-iteration fresh-buffer fine-tuning also trends weak | A confirmed baseline regression or a paired chance-vs-control treatment effect |

### 5.2 Exhaustive chance panel

The first production mechanism evaluated all `C(8,4)=70` public rows after
PUCT selected an eligible reveal-triggering action. Bootstrap rows created no
visits, and the direct probability mean initialized chance Q. At most one
panel was admitted per real move, and 70 rows were charged to the move budget.

The paired 128-game screen was weak negative/inconclusive: treatment scored
49.22% with a raw-score margin of -0.766. Throughput fell to 89.1% and peak
allocated GPU memory rose from about 70 MB to 713 MB. This did not justify
enabling exhaustive enumeration in training.

Primary references:

- [Deck-8 chance-correct MCTS](DECK8_CHANCE_CORRECT_MCTS.md)
- [Deck-8 chance-panel whole-game A/B](DECK8_CHANCE_AB_TEST.md)

### 5.3 Reach, leverage, and attribution

The 64-position causal experiment rejected the hypothesis that the deck-8
first-selection search cannot reach the reveal. Median first reach was
simulation 29; all positions reached a reveal even at 800 simulations. At
6,400 simulations about 90.2% of paths crossed a reveal action.

Production-like panel search changed root policy much more than a single-node
value pulse. Charged and extra-work panels were almost identical, so 70-unit
compute displacement did not explain the policy movement at the tested
budgets. The result still conflated observation splitting and bootstrap values.

The sampled-split ablation resolved that attribution. At 6,400 simulations:

- control-to-sampled mean root-policy TV was `0.383`;
- 33/64 top joint actions and 17/64 top picks changed;
- sampled-to-panel-extra TV was only `0.015`; and
- the panel changed the sampled top action in only 1/64 positions.

Thus, separating public observations is the operative mechanism. One-shot
enumeration does not materially improve the resulting policy.

Primary references:

- [Deck-8 causal reach/leverage experiment](DECK8_CAUSAL_LEVERAGE_EXPERIMENT.md)
- [Deck-8 sampled-split ablation](DECK8_SAMPLED_SPLIT_ABLATION.md)

### 5.4 Decision quality

The forced-move duel concentrated evaluation on the 33 positions whose
6,400-simulation top actions differed. It used common continuation seeds,
player-label mirrors, forced actions, and the incumbent open-loop continuation
engine as a conservative arbiter.

The all-position primary result narrowly missed the positive gate. The
preregistered 17-position pick-changing subgroup had a positive raw-margin
interval but an unresolved official-points interval. This is insufficient for
a strength claim, yet it is the best evidence that split-preferred actions can
be better rather than merely different.

Reference:

- [Deck-8 forced-move disagreement duel](DECK8_FORCED_MOVE_DUEL.md)

### 5.5 Earlier progressive-widening evidence

The A1c advisor prototype atomically initialized balanced or matched-IID
panels after a visit threshold, charged all NN rows, capped initialization
work, and widened in complete cycles. The sealed eight-position, 384-search
equal-work screen produced:

| Arm | Exact-best | Mean regret | P90 regret | Exact pairwise |
| --- | ---: | ---: | ---: | ---: |
| incumbent `x0` | 68.75% | 0.005153 | 0.014127 | 81.42% |
| lazy `X=1` | 62.50% | 0.006970 | 0.018488 | 76.87% |
| A1c `X=4` balanced | 70.31% | 0.004698 | 0.013094 | 79.75% |
| A1c `X=4` IID | 71.88% | 0.004243 | 0.013094 | 78.12% |
| A1c `X=8` balanced | 62.50% | 0.006970 | 0.018488 | 77.96% |
| A1c `X=8` IID | 62.50% | 0.006970 | 0.018488 | 79.03% |

All `X=4` gains came from one position; the other seven tied. `X=8`
regressed. The conservative final classification was inconclusive/weak signal,
not target-positive. This evidence is highly relevant to the proposed work:
more active chance width can reduce conditional depth and make search worse.

References:

- [A1c overnight implementation and experiment plan](A1C_OVERNIGHT_IMPLEMENTATION_PLAN.md)
- [Kingdomino next levers, A1c result and execution history](KINGDOMINO_NEXT_LEVERS.md)
- [Chance-search follow-up review request](CHANCE_CORRECT_REVIEW_REQUEST_2.md)

### 5.6 Chance-aware training pilot and control

The ten-iteration pilot warm-started `current_best`, used a fresh replay
buffer, applied sampled splits at all deck-8/deck-12 selection roots, and
weighted treated examples 2x. It completed 4,000 games and 3,000 optimizer
steps. Treated examples were 9.18% of the final buffer and approximately 17%
of optimizer draws. Deck/slot accounting was balanced and bootstrap rows were
zero.

On 33 held-out disagreement positions, the raw-network probability margin
`P(split top) - P(control top)` moved from `-0.0577` to `+0.0138`. The paired
mean change was `+0.0716`, with a position-bootstrap 95% interval of
`[+0.0241,+0.1219]`; 25 positions moved toward the split preference and eight
moved away. The split-relative fit measure also improved by `+0.1822`, with a
95% interval of `[+0.0845,+0.2960]`. This is direct evidence that the learner
absorbed a *relative* chance-aware policy signal. It did not fully imitate the
frozen split target: absolute split KL and top-action agreement did not
improve.

Iteration 2 was selected after an exploratory positive 32-pair screen, then
failed a preregistered 256-pair confirmation:

| Evaluation | Pair score | Mean margin | 95% interval |
| --- | ---: | ---: | ---: |
| chance iter 2, shared open-loop search | 42.77% | -2.81 | `[36.86%,48.90%]` |
| chance iter 2, shared deck-8/deck-12 split | 42.58% | -3.24 | `[36.67%,48.70%]` |
| matched open-loop control iter 2 | 45.31% | -2.24 | `[39.33%,51.43%]` |

The original exploratory result reproduced exactly and remained invariant when
embedded in a larger batch; it was simply a favorable small seed sample.

The chance checkpoint is confirmed weaker. The matched control is
directionally weak but inconclusive, and it was evaluated on a different seed
block. The 2.73-point difference between the two trained candidates is not a
paired treatment contrast. The defensible conclusion is:

- the particular pilot training recipe did not produce a promotable model;
- open-loop inference mismatch does not explain iteration 2's weakness;
- fresh-buffer short warm-start training is a credible common contributor;
- chance-specific harm remains possible but is not isolated; and
- chance-aware methodology remains a live candidate for a properly staged
  full cycle.

Reference:

- [Chance-aware deck-8/deck-12 training pilot](CHANCE_AWARE_TRAINING_PILOT.md)

## 6. Findings considered settled

1. **Information safety is non-negotiable.** Pre-reveal actions must never see
   the future row. Row selection occurs only after the action is committed.
2. **Reveal reach is adequate at deck 8.** More root simulations are not needed
   merely to make the first-selection tree touch the chance boundary.
3. **Explicit public-observation separation is the primary mechanism.** It
   corrects post-reveal aliasing and strategy fusion.
4. **One raw NN evaluation for every possible row is not the main lever.** The
   70-row panel adds little to sampled-split policy while consuming work and
   memory.
5. **Chance width has diminishing and potentially negative returns.** Historical
   `X=4` was more promising than `X=8` on the small exact screen.
6. **Changed actions have some quality evidence.** The forced duel is not a
   whole-game proof, but it is incompatible with dismissing every changed
   decision as noise.
7. **The learner can absorb the signal.** Held-out disagreement preferences
   moved toward sampled-split actions.
8. **The first training pilot cannot adjudicate the mechanism.** The shared
   training recipe itself plausibly caused early weakness.
9. **No trained checkpoint from this work is promotable.** `current_best.pt`
   remains authoritative and unchanged.

## 7. Important unresolved questions

1. How much of current sampled-split target variance comes from the first few
   reveal rows at a chance node?
2. Are conditional observation subtrees too shallow to choose the first
   informed placement/pick well?
3. Should the first chance-node admission return a mean of raw NN values, or
   should it guarantee real conditional searches in every admitted row before
   returning one parent result?
4. Should parent PUCT consume the current direct mean of row Q values or a
   historical mean of sampled returns?
5. How should inactive support mass be estimated before the active sample
   widens? At deck 8 rows are uniform; deck 12 remains a much larger sample.
6. Does deck 12 benefit from the same treatment, or does its 495-row support
   create unacceptable fragmentation?
7. Is natural replay frequency sufficient, or is modest weighting needed only
   after the buffer reaches a stable size?
8. Would writing selected counterfactual post-reveal search targets to replay
   improve learning, or create correlated/pseudoreplicated examples? The first
   full cycle should probably leave this disabled.
9. Can the new search batch initialization/widening rows across nodes and game
   slots without small-call fragmentation?

## 8. Proposed search: batched depth-gated progressive widening

### 8.1 Intended behavior

The proposed design combines four ideas:

1. a small initial mini-panel to avoid a single unlucky reveal;
2. separate persistent observation subtrees;
3. real conditional search after the reveal; and
4. widening only after active rows have adequate conditional depth.

Provisional deck-8 parameters for review:

- initial active width `W0=4` rows;
- balanced, value-independent sampling without replacement;
- minimum real conditional work `D_min=4` per active row before widening;
- a second candidate with `D_min=8` if implementation cost is small;
- widths `4 -> 8 -> 16 -> 32 -> 64 -> 70`; and
- one direct uniform mean over active row estimates.

These are starting specifications, not evidence-backed production constants.
The historical `X=4` result must not be misread as direct support for `W0=4`.

### 8.2 Admission and batched evaluation

After PUCT commits to a reveal-triggering action:

1. select `W0` rows from a node-local fixed random schedule;
2. materialize the corresponding public post-reveal states;
3. gather admission requests across chance nodes and game slots;
4. flatten all newly admitted rows into as few GPU evaluator calls as
   practical;
5. scatter returned priors/values to the owning observation subtrees;
6. publish a panel only if its entire admission group succeeds; and
7. charge every row as NN work while adding no fake root or policy-target
   visits.

The pre-reveal parent receives one probability-weighted result, not `W0`
independent visits. Otherwise selecting an action once would artificially
inflate its visit count and corrupt the root policy target.

### 8.3 Conditional depth and widening

The key correction to breadth-only bootstrapping is that newly active rows must
not trigger immediate further widening while their post-reveal decision nodes
remain one-evaluation stubs.

Subsequent real paths route evenly among active rows and continue through the
post-reveal decision tree. A node widens only when a depth condition such as
the following is satisfied:

```text
min(real_conditional_visits[row] for row in active_rows) >= D_min
```

At a widening event, only the new rows are batch-evaluated. Existing row trees
and statistics remain intact. Widening order must depend only on the frozen
random schedule and visit/depth state, never row value, policy, or downstream
legality information unavailable before the reveal.

For deck 12, blindly widening toward all 495 rows is unlikely to be useful.
The reviewer should recommend whether to use a conservative maximum width such
as 16 or 32, a common power-law schedule, or to exclude deck 12 from the first
implementation.

### 8.4 The unresolved admission-depth choice

There are two defensible implementations. This review should choose one
explicitly.

**Persistent outer-search widening (simpler):**

- admission batch gives `W0` bootstrap estimates;
- the triggering parent simulation backs their mean once;
- later ordinary outer simulations revisit and deepen active rows; and
- widening waits for `D_min` real row visits.

This is close to existing A1c machinery and easiest to integrate with
`BatchedMCTS`. Its weakness is that a parent action depressed by its initial
mini-panel may not be revisited enough to reach meaningful conditional depth.

**Nested conditional macro-search (stronger, more complex):**

- one outer action selection opens `W0` rows;
- each row receives `D0` inner conditional simulations;
- searched row values are averaged; and
- the outer parent receives one result and one visit.

This directly spends work on the first informed placement/pick, but creates a
nested budget, multi-wave completion, virtual-loss, cancellation, and visit-
invariant problem. Internal visits must not leak into outer root policy counts.

A possible compromise is `W0=4` with one guaranteed conditional action descent
per row at admission, followed by persistent widening and `D_min=4`. The
reviewer should assess whether that additional complexity is justified or
whether existing PUCT exploration is sufficient.

### 8.5 Chance value and backup

At deck 8, active rows form a uniform sample without replacement. The chance
estimate should be the direct mean of current row estimates:

```text
Q_chance = mean(Q_row for row in active_rows)
```

Row search visit counts must not become chance probabilities. A row searched
more deeply may have a better estimate, but not greater probability mass.

The reviewer should verify whether ancestors should receive the current panel
mean on every later backup, or whether the parent edge should query the chance
node's dynamic mean directly. Historical shallow returns remaining forever in
ancestor averages can dilute improvements made by later conditional search.

### 8.6 Compute accounting

Report both logical work and wall time:

- ordinary leaf NN rows;
- initial admission rows;
- widening rows;
- inner conditional rows if macro-search is used;
- evaluator calls and batch occupancy;
- maximum batch and GPU memory;
- real outer simulations;
- conditional visits per active row; and
- useful recorded training examples per hour.

Admission rows consume budget even when evaluated in one GPU forward. A fair
comparison must use matched total NN work and also report the strongest
wall-time-feasible configuration.

## 9. Proposed implementation plan

### Phase I - freeze semantics before editing

Write a short implementation specification resolving:

- persistent versus nested admission;
- `W0`, `D_min`, width schedule, and deck-specific caps;
- direct-mean backup semantics;
- first-reveal-only scope;
- budget charge and parent visit rules;
- bootstrap prior/value storage;
- interaction with exact deck-4 handling; and
- whether counterfactual rows remain excluded from replay.

Do not tune parameters while implementing the data structures.

### Phase II - Rust search core

Prefer extending the existing chance/A1c structures rather than creating a
third unrelated chance engine.

Required elements:

- node-local immutable support/schedule identity;
- active-width and widening state;
- per-row real conditional-visit/depth counters;
- pending atomic admission groups;
- cross-node row gathering and result scattering;
- direct probability/sample-mean chance Q;
- reversible virtual loss across pending work;
- one parent visit per completed outer simulation or macro evaluation;
- hard NN budget accounting; and
- zero-cost, bit-stable disabled behavior.

### Phase III - batched production boundary

`BatchedMCTS.step()` should gather ordinary leaves plus complete admission or
widening groups from all active slots. It may split a very large combined batch
for memory, but a logical group must commit atomically only after every row
returns successfully.

`update()` must validate ownership, row identity, expected group width and
probability mass before publishing any group. Failure must leave the affected
chance node unmodified and remove any virtual loss safely.

### Phase IV - self-play and replay integration

Add opt-in configuration with explicit names for width and depth. Preserve the
old sampled-split and exhaustive modes as reviewable ablations, and reject
invalid combinations.

Training records should retain:

- root deck and selection slot;
- whether a chance node was actually crossed;
- admission/widening rows;
- active width at search end;
- conditional visits/depth distribution;
- whether the root target used the new direct mean; and
- replay draw accounting.

The first full cycle should use chance replay weight `1.0`. The pilot's natural
treated share was already about 9%; 2x weighting raised optimizer exposure to
about 17% and is an avoidable confound.

### Phase V - required correctness tests

At minimum:

1. pre-reveal action selection is invariant to hidden order;
2. rows are selected only after the action is committed;
3. active support is unique, probability-correct, and value-independent;
4. one action selection cannot add `W` parent visits;
5. bootstrap/admission rows never enter policy visit counts;
6. row visit counts never alter chance probability weights;
7. depth gates prevent widening while any active row is undersearched;
8. widening preserves all existing subtree statistics;
9. cross-node batching scatters every result to its correct owner;
10. partial or failed groups never commit;
11. NN budget cannot overshoot;
12. virtual loss is exactly reversible on success and failure;
13. actor/player-0 value frames remain correct across chance boundaries;
14. serial and batched small deterministic cases agree; and
15. disabled incumbent search remains bit-identical on a nontrivial golden
    vector.

### Phase VI - minimal pretraining gates

The goal is to prevent a broken day-long run, not reopen unlimited search
tuning.

1. Run the focused correctness suites.
2. Run a small mechanical CUDA smoke confirming batched admissions, widening,
   depth gates, zero fake visits, and replay/log compatibility.
3. On a frozen deck-8 set, compare only a small preregistered family such as:
   incumbent, current sampled split, `W0=4/D_min=4`, and optionally
   `W0=4/D_min=8`, at matched NN work. The purpose is to reject clear oracle or
   depth regressions, not to prove strength.
4. Run a short end-to-end self-play throughput probe using the intended cloud
   configuration.

If the new mode is mechanically valid and does not clearly regress the frozen
quality screen, proceed to the full cycle. Do not demand another whole-game
search A/B before training; previous work already shows why small whole-game
tests dilute sparse decision changes.

## 10. Proposed full training cycle

### 10.1 Objective

Train a real challenger long enough to move beyond fresh-buffer burn-in and
allow the new policy targets to influence a mature replay distribution. This
is a model-strength run, not a clean one-variable causal experiment.

### 10.2 Staged replay plan

Preferred plan when no audited compatible incumbent buffer is available:

**Phase A - replay stabilization**

- warm-start weights from unchanged `current_best`;
- run 8-10 ordinary open-loop iterations;
- 400 games and 300 optimizer steps per iteration;
- preserve the incumbent 4,800/200 playout-cap schedule;
- fresh buffer, atomically saved every iteration; and
- no promotion or early checkpoint selection during burn-in.

**Phase B - chance-aware training**

- continue from the Phase-A checkpoint and replay buffer;
- run 20-30 additional iterations with the reviewed search;
- use natural `1.0x` treated-example replay weight initially;
- retain the latest learner as generator unless the reviewer recommends an
  existing soft-gate mode;
- save every checkpoint and buffer; and
- keep `current_best.pt` read-only.

If an audited, encoder-compatible replay buffer generated by `current_best` is
available, the reviewer may recommend shortening Phase A rather than rebuilding
the same distribution.

### 10.3 Deck scope decision

Deck 8 has the strongest causal, oracle, and forced-move evidence. Deck 12 was
included in the pilot to provide more natural treated examples but was an
explicit extrapolation and has 495 reveal rows.

Options for review:

- **Conservative:** first full cycle treats deck 8 only.
- **Broader:** treat deck 8 and 12 with the same `W0/D_min` rule but a smaller
  deck-12 width cap.
- **Staged:** begin Phase B with deck 8, then enable deck 12 after a fixed
  mechanical/throughput milestone without inspecting strength.

Our provisional preference is the broader option only if progressive widening
keeps deck-12 active width and conditional depth healthy in the smoke. Otherwise
use deck 8 alone rather than importing another poorly constrained treatment.

### 10.4 Evaluation and promotion

Do not repeat the 32-pair checkpoint-ranking process that selected iteration 2
from noise.

Before training, freeze:

- the primary checkpoint-selection rule (preferably final Phase-B checkpoint
  or a fixed milestone, not the best observed small screen);
- a fresh paired seed block;
- match size and confidence rule;
- deployed search configuration; and
- whether a second final confirmation block is reserved.

Recommended gates:

1. **Network gain under common new search:** challenger versus unchanged
   `current_best`, both using the reviewed chance-aware search.
2. **Complete-system gain:** challenger plus new search versus deployed
   `current_best` plus incumbent open-loop search.

Promotion requires the established same-deck, seat-swapped pair statistic and
pair-score lower confidence bound above 50%. No exploratory screen, training
loss, exact suite, or policy-absorption metric promotes a checkpoint.

## 11. Cloud execution

The laptop is an RTX 3070 Laptop GPU. Recent iterations required roughly
46-52 minutes each; the proposed 28-40 total iterations exceed the owner's
12-hour local-run threshold.

Use the existing [cloud runbook](CLOUD_RUN.md) as an operational starting point,
but update its historical cold-start assumptions for this warm-start,
encoder-compatible run.

Before renting:

- commit or otherwise freeze the reviewed code and run configuration;
- prepare Linux/Rust/Python setup and checkpoint transfer;
- use persistent storage or frequent off-instance synchronization;
- preserve atomic per-iteration checkpoints and replay buffers;
- define a maximum spend/duration; and
- make the STOP/resume procedure explicit.

During the first paid hour, run a 20-40 game end-to-end probe with the real
search and exact solver. Synthetic forward throughput is insufficient. Require
approximately 0.30 games/second for a 30-iteration run or 0.38 games/second for
a 40-iteration run to target a roughly 12-hour window. Verify that available
CPU cores do not bottleneck Rust game search or exact solving.

Reference training behavior and replay/gating semantics are documented in
[training parameters](training_parameters.md).

## 12. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Initial sampled rows are unlucky | Good action suppressed before correction | `W0>1`, balanced value-independent sample, direct mean |
| Too much breadth | Conditional placement/pick remains shallow | Depth-gated widening; track visits/depth per active row |
| Too much depth on a biased small sample | Chance expectation remains volatile | Monotone widening and frozen random nested support |
| Child visit allocation changes chance probability | Biased expectation | Probability/sample mean independent of visit counts |
| Bootstrap work inflates root policy | Corrupt training targets | Zero fake visits; one parent result per outer selection |
| Small evaluator calls | Cloud GPU underutilized | Cross-node/slot admission batching |
| Deck-12 explosion | Memory/depth collapse | Conservative width cap or deck-8-only first cycle |
| Fresh-buffer burn-in | Early checkpoint regression | Open-loop stabilization phase or audited warm buffer |
| 2x replay weighting | Global policy interference | Begin at natural 1x frequency |
| Counterfactual replay pseudoreplication | Overweighted correlated states | Exclude internal branches from first full cycle |
| Small-match checkpoint selection | False positive, as observed for iteration 2 | Freeze candidate/milestone and fresh confirmation seeds |
| Cloud preemption/data loss | Lost day-long run | Persistent volume, atomic saves, off-box sync |

## 13. Specific questions for the reviewer

1. Is `W0=4` with `D_min=4` a sound first production candidate, or should the
   initial width follow historical tile-balanced cycles instead?
2. Should admission be bootstrap-only with later persistent deepening, or
   should one outer action selection guarantee inner conditional work before
   returning its averaged value?
3. Is a direct active-row mean sufficient, or should an estimator account for
   nested sampling and unequal row maturity differently?
4. How should updated chance-node means influence ancestor Q without retaining
   excessive stale shallow returns?
5. Should width be gated by minimum row visits, a lower quantile, total node
   visits, conditional decision depth, or a combination?
6. Should deck 12 be included, capped, or deferred?
7. Is a 1x natural replay frequency appropriate, and when—if ever—should it be
   increased?
8. Is the proposed open-loop replay-stabilization phase the best response to
   the matched-control result, or should generator gating/warm replay be used
   instead?
9. Is evaluating only a fixed final checkpoint adequate, or should the full
   cycle use a preregistered sequential promotion design?
10. Which implementation invariants or failure modes are missing from this
    brief?

## 14. Recommended source reading order

1. [Kingdomino next levers](KINGDOMINO_NEXT_LEVERS.md) — information-set model,
   early chance work, completed A1c result, and roadmap history.
2. [Deck-8 chance-correct MCTS](DECK8_CHANCE_CORRECT_MCTS.md) — production
   exhaustive-panel semantics and invariants.
3. [Deck-8 chance-panel whole-game A/B](DECK8_CHANCE_AB_TEST.md) — frozen-net
   gameplay and throughput result.
4. [Deck-8 causal reach/leverage experiment](DECK8_CAUSAL_LEVERAGE_EXPERIMENT.md)
   — proof that chance nodes are reached and can influence root allocation.
5. [Deck-8 sampled-split ablation](DECK8_SAMPLED_SPLIT_ABLATION.md) — evidence
   that topology, not enumeration, drives policy changes.
6. [Deck-8 forced-move disagreement duel](DECK8_FORCED_MOVE_DUEL.md) — direct
   changed-action quality evidence.
7. [A1c overnight implementation and experiment plan](A1C_OVERNIGHT_IMPLEMENTATION_PLAN.md)
   — original progressive-widening specification; read with the later result
   in `KINGDOMINO_NEXT_LEVERS.md`.
8. [Chance-search follow-up review request](CHANCE_CORRECT_REVIEW_REQUEST_2.md)
   — prior correctness and estimator concerns.
9. [Chance-aware training pilot](CHANCE_AWARE_TRAINING_PILOT.md) — production
   integration, ten-iteration run, confirmation, aligned evaluation, and
   matched training control.
10. [Training parameters](training_parameters.md) and
    [cloud runbook](CLOUD_RUN.md) — replay, generator, promotion, throughput,
    and operational guidance.

## 15. Requested reviewer deliverable

Please return:

1. a correctness verdict on the proposed chance-node estimator and visit
   semantics;
2. a choice between persistent and nested admission-depth designs;
3. recommended initial deck-specific width/depth parameters;
4. an implementation sequence identifying reusable existing A1c/panel code;
5. a focused test plan and minimum pretraining gates;
6. a frozen full-cycle configuration, including replay strategy, deck scope,
   cloud target, runtime, checkpoint selection, and promotion design; and
7. explicit stop conditions that would prevent wasting the full training run.

The project owner is prepared to fund a real cloud training cycle once this
design has received a credible external review and passed its mechanical and
short quality gates.
