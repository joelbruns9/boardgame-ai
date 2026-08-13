# Chance-progressive v1: training results and next-run plan

Status: post-run record and proposed v2 specification
Date: 2026-08-12
Branch: `codex/kingdomino-chance-correct`
Promotion status: **no checkpoint promoted**

> **Amendment 2026-08-12 (owner sign-off):** Sections 2 and 7-11 reflect the
> post-review owner decisions: (a) Gate 0 — the fixed-network
> progressive-versus-open-loop match — is a hard precondition and now carries
> the deck-16/deck-20 question as additional match arms; (b) the only v2
> training-recipe changes are the learning-rate and replay-reuse reductions;
> (c) the KL term is dropped outright, and the protected teacher reservoir and
> 1.5x chance weighting move to a one-at-a-time contingency list. Sections 3-6
> remain the v1 factual record except where marked.

## 1. Purpose and document authority

This document records the first full cloud training cycle using production
progressive chance search, its subsequent high-powered checkpoint evaluations,
and the currently recommended design for a second cycle. It separates observed
facts from interpretations and proposals so a reviewer can audit the evidence
without treating a design discussion as a completed result.

The design history and implementation contracts remain in:

1. [Full-training review brief](CHANCE_AWARE_FULL_TRAINING_REVIEW_BRIEF.md)
2. [Reviewer response and frozen v1 specification](CHANCE_AWARE_FULL_TRAINING_REVIEW_RESPONSE.md)
3. [Cloud runbook](CHANCE_PROGRESSIVE_CLOUD_RUN.md)
4. [Earlier chance-aware training pilot](CHANCE_AWARE_TRAINING_PILOT.md)

Where this document differs from the v1 frozen specification, it describes a
**proposed v2 amendment**, not a retroactive reinterpretation of v1.

## 2. Executive conclusion

The v1 cycle was mechanically successful but did not produce a promotable
network. Iteration 20, the most promising rolling-gate checkpoint, scored
48.14% against the unchanged incumbent under symmetric open-loop search and
47.80% under symmetric progressive search over 2,048 games in each condition.
Iteration 10 subsequently scored 49.07% under the same symmetric progressive
configuration and seed set. All paired confidence intervals crossed 50%, and
all point estimates and mean margins favored the incumbent.

The symmetric progressive result rules out the simple explanation that the
candidate only looked weak because it was evaluated under the open-loop search
used by the incumbent. It does **not** determine whether progressive search is
stronger than open-loop search for a fixed network; both sides used the same
search in each completed evaluation.

No v2 training engineering should begin until a fixed-network search A/B shows
that incumbent plus progressive search plays better than the identical network
plus open-loop search. This is the premise of teacher-search distillation and is
now Gate 0, not merely pending evidence. Per the 2026-08-12 owner decision,
Gate 0 also carries the deck-16/deck-20 question: wider treatment scopes enter
as additional Gate-0 match arms, so the search-value question is answered in
its most definitive form — does this search family improve play, and does
treating more of the game improve it further? — at match cost rather than
training-cycle cost.

Conditional on Gate 0 passing, the amended v2 cycle is deliberately a
two-change experiment:

- reduce learning rate from `1e-4` to `3e-5`;
- reduce steady-state replay reuse from about 5.6x to about 2.8x (96 optimizer
  steps per 400 games);
- keep everything else — single-FIFO replay, 1.0x chance-treated weight,
  losses, search constants — identical to v1 so a second null remains
  attributable;
- run self-play chance treatment at the scope Gate 0 selects (deck 8/12 at
  minimum, up to deck 8/12/16/20);
- evaluate the advisor deployment configuration at 800 simulations;
- retain random reveal panels and cap 16 at every treated deck;
- replace the 384-game/55% rolling gate with a preregistered sequential paired
  gate that can recognize smaller real improvements without treating seat-
  swapped games as independent, plus fixed-iteration candidate nomination.

The first v2 cycle drops the proposed KL term outright. Teacher search targets
already anchor ordinary teacher rows, while KL toward the teacher's raw network
prior would pull those rows away from their stronger searched policy targets.
The protected teacher reservoir and the 1.5x chance-treated weighting are
likewise **not** in the launch recipe; they are preregistered contingency
levers (section 7.5), applied one at a time only if the amended v2 nulls
despite a positive Gate 0.

## 3. V1 experimental identity

### 3.1 Incumbent and candidate

| Artifact | Path | SHA-256 |
|---|---|---|
| Unchanged incumbent | `runs/kingdomino/best_checkpoint/current_best.pt` and the Phase-B run-local copy | `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3` |
| Evaluated candidate | `runs/kingdomino/chance_progressive_cloud_v1/phase_b/iter_0020.pt` | `c1f128c514c4f26b721506bc90d3c6f45c3f3ac8d515ea3b6ae5fdd59c063434` |

The global incumbent was not overwritten. Iteration 20 was selected for the
larger diagnostic because it had the strongest rolling-gate point estimate;
that selection did not establish that it was the true strongest checkpoint.

### 3.2 Machine and calibrated runtime settings

- GPU: NVIDIA RTX 5090.
- CPU: AMD Ryzen 9 9950X3D.
- Calibrated network: 80 channels x 6 blocks, AMP enabled.
- `batch_slots=48`, `leaf_batch=6`, `game_cpus=2`.
- G4 feasibility throughput: 0.805 games/second.
- Sustained prefill/training throughput: approximately 0.87-0.90 games/second.

### 3.3 Search treatment

- Engine: batched open-loop MCTS with progressive explicit chance separation.
- Treated public deck counts: 8 and 12.
- Full-search budget: 4,800 NN rows/simulations.
- Fast-move budget: 200; full-search fraction 25%; fast moves not recorded.
- `W0=4`, `N_init=2`, `D_min=4`.
- Width schedule `4,8,16,32,64,70`, with terminal cap 16 at both treated decks.
- Maximum initialization fraction: 25%.
- Reveal support selection: uniform random public-state-derived permutation;
  no hidden deck order is read.
- Chance-treated replay weight: 1.0x.

### 3.4 Training treatment

- Architecture: 80x6, bilinear dimension 64.
- Replay capacity: 200,000 examples.
- Batch size: 256.
- Learning rate: `1e-4`.
- Optimizer steps: 192 per 400 new games.
- Approximate steady-state reuse: 49,152 draws / about 8,700 new positions =
  5.6x.
- Exact endgame solver and existing exact targets remained enabled.

## 4. Cloud execution results

### 4.1 G4 feasibility probe

G4 passed before the full run:

- 32 games at 0.805 games/second;
- 20 deck-8 and 31 deck-12 treated recorded examples;
- 6,645 admissions and 2,405 widenings;
- 26,206 bootstrap rows, 1.6% of evaluations;
- no progressive initialization guardrail trip;
- exact endgame solved normally with no fallback;
- finite optimizer step and valid saved buffer.

This established that the production mechanism, accounting, solver, replay,
training, and persistence paths all operated together on the rented machine.

### 4.2 Phase A stop

The original Phase A trained immediately after crossing its 5,000-example
minimum. At iteration 5, after 1,500 optimizer steps on a young buffer, its
measurement match scored 42.2% with pair-Wilson interval `[35.4%, 49.3%]`.
The preregistered upper-bound stop fired.

Interpretation: this rejected the immediate 300-step, approximately 8.8x reuse
recipe. It did not adjudicate a fully prefilled lower-reuse cycle. The Phase-A
checkpoint and buffer were excluded from Phase B.

### 4.3 Phase B frozen-incumbent prefill

Phase B prefill completed 23 iterations x 400 games with:

- original `current_best` controlling both seats;
- progressive search active at every eligible deck-8/deck-12 position;
- zero optimizer steps;
- a fresh buffer reaching the 200,000-example capacity;
- approximately 8,600-8,800 retained examples per iteration;
- approximately 0.87-0.90 games/second;
- typical progressive initialization cost of 1.5-1.8% of all NN rows;
- typical end-of-search active width near 11 and mature width near 8.1-8.5;
- low exact-solver fallback counts and rare guardrail blocks.

The completed buffer was then loaded by the separate Phase-B training run.

### 4.4 Infrastructure restart before the clean Phase B

The first Phase-B launch reached its iteration-5 promotion gate and then failed
because FastAPI was absent from the cloud environment. Its directory was moved
intact to:

`runs/kingdomino/chance_progressive_cloud_v1/phase_b_failed_missing_fastapi_iter5`

FastAPI/uvicorn were installed, the fixed suite was checked (17 rows), and
Phase B was restarted from the original incumbent plus the unchanged completed
prefill buffer. Results below refer to this clean restart, not a continuation
of the failed process. A v2 readiness build must either declare these runtime
dependencies in setup or remove the fixed-suite evaluator's unnecessary web
application import.

### 4.5 Phase B training

The clean run completed all 24 iterations:

- 24 x 400 = 9,600 learner self-play games;
- 24 x 192 = 4,608 optimizer steps;
- full 200,000-example warm buffer at launch and throughout training;
- valid completion according to the Phase-B validator (`rows=24`,
  `status=complete`);
- progressive search active at both requested decks throughout;
- approximately 750-860 treated recorded examples per 400-game iteration;
- progressive bootstrap rows typically 1.5-1.8% of total NN evaluations;
- active width remained approximately 11 and mature width approximately
  8.2-8.6;
- exact solving remained healthy with only isolated fallbacks.

Representative training diagnostics improved numerically:

| Iteration | Policy loss | Win loss | Training Brier | Diagnostic Brier |
|---:|---:|---:|---:|---:|
| 5 | 1.347 | 0.443 | 0.148 | 0.137 |
| 10 | 1.334 | 0.417 | 0.138 | 0.128 |
| 20 | 1.321 | 0.398 | 0.132 | 0.138 |

The loss improvement did not translate into demonstrated head-to-head strength.
The diagnostic Brier metric was noisy enough that checkpoint selection from it
alone was not reliable.

### 4.6 Rolling open-loop gates

Every fifth training iteration used 384 shared-open-loop games at 400 sims
against the run-local incumbent. Promotion required both raw score at least 55%
and pair-Wilson lower bound above 50%, plus no fixed-suite veto.

| Iteration | W-L-D | Candidate score | Reported LCB | Result |
|---:|---:|---:|---:|---|
| 5 | 196-188-0 | 51.0% | 44.0% | Probation; no promotion |
| 10 | 200-184-0 | 52.1% | 45.0% | Probation; no promotion |
| 15 | 190-194-0 | 49.5% | 42.5% | Probation; no promotion |
| 20 | 204-180-0 | 53.1% | 46.1% | Probation; no promotion; no fixed-suite regression |

These gates showed why 384 games and the 55% floor could not recognize a small
improvement. They did not prove iteration 20 was stronger: the subsequent
2,048-game match reversed its favorable 384-game point estimate.

## 5. High-powered iteration-20 evaluations

Both evaluations used:

- candidate `iter_0020.pt` and the unchanged incumbent hashes in section 3.1;
- 2,048 games / 1,024 same-seed seat-swapped pairs;
- seed base `20330000`;
- 400 simulations/NN rows;
- `batch_slots=48`, `leaf_batch=6`;
- `c_puct=1.5`, `fpu=-0.2`, `margin_gain=2.0`, `alpha=0.5`.

### 5.1 Results

| Search used by both models | Candidate score | Pair-Wilson 95% interval | Mean margin | Runtime |
|---|---:|---:|---:|---:|
| Open loop | 48.1445% | `[45.0968%, 51.2061%]` | -0.8594 | 1,260.1 s |
| Progressive at deck 8/12 | 47.8027% | `[44.7571%, 50.8648%]` | -1.0181 | 1,262.4 s |

Open-loop pair outcomes were 177 candidate wins, 215 candidate losses, and 632
pair draws. The 986-1,062 individual-game result contained no individual draws.

The progressive result was valid and could not have silently fallen back to
open loop. Across both orientations it recorded:

- 8,192 deck-8 and 8,192 deck-12 progressive searches;
- 2,337,539 deck-8 and 2,328,289 deck-12 crossed paths;
- 265,682 admissions and 69,821 widenings;
- 900,600 bootstrap rows;
- mean active width 12.343 and mean mature width 9.726;
- 2,247 budget blocks and only 9 guardrail blocks.

The bootstrap cost was about 55 rows per treated root, or about 13.7% of a
400-row treated-root budget. Despite doing materially different search work,
the progressive evaluation changed the candidate's relative point estimate by
only -0.34 percentage points.

### 5.2 Conclusions supported by these evaluations

1. Iteration 20 must not replace `current_best`.
2. There is no evidence that iteration 20 is stronger than the incumbent under
   either completed 400-sim deployment profile.
3. Open-loop evaluation mismatch does not explain the iteration-20 result.
4. Mechanism inactivity does not explain the progressive result.
5. The run demonstrates optimization and search activity, but not a network
   strength gain.

### 5.3 Conclusions not supported

1. These evaluations do not compare progressive search against open loop for
   one fixed network. They therefore do not establish the intrinsic playing
   value of the search mechanism.
2. They do not establish iteration 20 as the strongest or weakest checkpoint
   in the 24-iteration trajectory.
3. They do not evaluate the intended 800-sim advisor configuration.
4. They do not prove that deck-16/deck-20 treatment is useful or harmful.

### 5.4 Iteration-10 follow-up result

A 2,048-game iteration-10 versus incumbent evaluation completed under the same
symmetric progressive 400-sim configuration and seed base:

`runs/kingdomino/chance_progressive_cloud_v1/eval_iter10_vs_baseline_progressive_2048`

| Candidate | Candidate score | Pair-Wilson 95% interval | Mean margin | Runtime |
|---|---:|---:|---:|---:|
| Iteration 10 | 49.0723% | `[46.0195%, 52.1320%]` | -0.1846 | 1,263.5 s |
| Iteration 20 | 47.8027% | `[44.7571%, 50.8648%]` | -1.0181 | 1,262.4 s |

Iteration 10 was descriptively 1.27 percentage points and 0.83 score-margin
points better than iteration 20 against the common incumbent. This is not a
confidence interval for the iteration-10-minus-iteration-20 contrast: the two
matches reused the same seed set, and their game-level artifacts would need to
be joined by seed and orientation to calculate the correlated difference.

Iteration 10 still did not demonstrate superiority to the incumbent. Its point
estimate and mean margin favored the incumbent, while its interval included
50%. The result also reversed iteration 10's favorable 52.1% point estimate
from the earlier 384-game rolling gate, illustrating the rolling gate's noise.

The mechanism was fully active and closely matched the iteration-20 run:

- 8,192 searches at each treated deck;
- 2,325,981 deck-8 and 2,332,770 deck-12 crossed paths;
- 268,987 admissions and 69,435 widenings;
- 905,782 bootstrap rows;
- mean active width 12.395 and mean mature width 9.758;
- 2,205 budget blocks and 27 guardrail blocks.

The nearly identical runtime and search diagnostics make mechanism activation
an implausible explanation for the checkpoint difference. The result is
consistent with useful learning early in the cycle followed by drift, but does
not prove that trajectory because the direct correlated contrast has not been
computed.

Because iteration 20 and iteration 10 reuse seed block `20330000`, that block
is diagnostic-only and must not be described as fresh promotion evidence in
v2.

### 5.5 Iteration-10 fixed-suite artifact

The local artifact
[chance-progressive iteration-10 fixed-suite comparison](../../runs/kingdomino/chance_progressive_iter10_fixed_suite/COMPARISON.md)
evaluated a checkpoint labeled iteration 10 on the 17-position fixed suite:

- candidate SHA-256
  `d64713c8eafb62460f7961079372d3fdae0e14947e56995cc07f3f5c0eeae176`;
- exact-value MAE increased by 0.00705, within the 0.05 veto tolerance;
- the top action changed on 1 of 17 positions;
- one exact position dominated the small aggregate MAE regression;
- neither of the two chance-relevant suite positions has an exact action or
  value oracle.

This supports only the statement "no broad fixed-suite regression detected."
It is not playing-strength evidence and is not the 2,048-game iteration-10
evaluation. Before treating both artifacts as observations of the same file,
compare this SHA against the checkpoint hash recorded in the cloud evaluation's
`manifest.json`; the pasted evaluation summary did not include that hash.

## 6. Interpretation of the v1 outcome

The leading hypothesis is signal dilution plus policy drift, not a proven
failure of explicit chance modeling:

- Only full-search moves were recorded. The run typically produced about two
  treated examples per game across deck 8 and deck 12.
- Chance-treated examples had neutral 1.0x replay weight.
- The learner generated rolling replacement data immediately after training
  began. By iteration 20, roughly 174,000 new examples had entered a 200,000-
  example FIFO buffer, so most frozen-incumbent prefill had likely been evicted.
- An already strong warm-start network was updated at `lr=1e-4` with about 5.6x
  steady-state reuse and no behavioral anchor to the incumbent on positions
  where chance search supplied no new evidence.
- The prior forced-move duel showed value in concentrated disagreement
  positions, but ordinary games changed far fewer played moves. That is a
  plausible fine-tuning signal, not a license for broad aggressive rewriting.

Iteration 10 finishing closer to the incumbent than iteration 20 on the same
progressive seed set strengthens—but does not prove—the drift hypothesis. Under
the 2026-08-12 amendment, drift is treated first with the two lowest-risk
levers (learning rate and reuse); the protected teacher reservoir is the first
contingency lever if drift persists (section 7.5).

This explanation remains a hypothesis. More fundamentally, v1 never measured
whether progressive search improves the incumbent's play relative to open-loop
search. Until that premise passes Gate 0 below, there is no demonstrated
teacher-search improvement to protect or distill.

## 7. Proposed v2 training cycle

### 7.1 Gate 0: prove the teacher-search signal exists

Before any v2 training engineering, extend the evaluator so search
configuration can differ by seat while network weights remain identical. Every
Gate-0 arm is a same-network, same-seed, seat-swapped match:

- treatment: unchanged incumbent plus progressive search at the arm's treated
  decks;
- control: the same incumbent checkpoint plus open-loop search;
- 800 simulations/NN rows per move, matching the intended advisor budget;
- cap 16 and the existing frozen progressive parameters (per-deck admission
  for decks 16/20 as in section 7.6);
- fresh seeds unused by v1 training, gates, or diagnostics, with a distinct
  preregistered block per arm;
- at least 2,048 games / 1,024 complete seat pairs per arm;
- pair-aware uncertainty from actual per-seed scores, not individual games.

Per the 2026-08-12 owner decision, wider treatment scope enters here — as
Gate-0 match arms, not training scope — so the underlying question is answered
in its most definitive form before any cloud training spend: does this search
family improve play, and does treating more of the game improve it further?

| Arm | Treated decks | Prerequisite | Order |
|---|---|---|---|
| A | 8, 12 | Asymmetric per-seat harness (implemented) | First |
| B | 8, 12, 16 | Rust per-deck mask + deck-16 admission | Second |
| C | 8, 12, 16, 20 | Deck-20 support | Third |

Arm A runs first because it needs no new Rust search work and gives the
cheapest early read. Arms B and C run before the final scope decision even if
Arm A passes; if Arm A fails, arms B and C decide whether the direction
survives at wider scope. Mechanical health telemetry (per-deck width,
conditional depth, initialization fraction, budget and guardrail blocks) is
recorded for every arm and is part of the scope decision.

The original symmetric evaluator could not run this contrast because it applied
one search configuration to both seats. Gate-0 implementation now adds a
seat-restricted progressive mode and a dedicated fixed-network harness. The
harness loads the checkpoint once, supplies the same evaluator object to both
players, assigns progressive search to seat 0 in the first orientation and seat
1 in the second, and fails closed unless the per-deck search counts prove that
exactly one seat received the treatment. The cloud run itself remains pending.

Implementation verification on 2026-08-12:

- 37 focused Python Gate-0, symmetric-evaluation, promotion, and production
  boundary tests passed;
- 8 adjacent progressive-cloud and G3 regression tests passed;
- all 4 focused Rust progressive-search tests passed;
- a 4-game, 400-row local checkpoint smoke completed both orientations with one
  checkpoint load and valid artifacts. Each orientation recorded exactly 4
  treated searches at deck 8 and 4 at deck 12, plus nonzero admissions,
  widening, and bootstrap rows. This was mechanism verification only and is not
  strength evidence. The final harness labels any explicitly allowed run below
  2,048 games `smoke_only` and refuses to emit a Gate-0 verdict from it.

Hard decision rule:

- at least one arm's one-sided lower confidence bound above 50%: premise
  passes; v2 training uses the widest treated-deck scope that is statistically
  non-negative and mechanically healthy;
- every arm's one-sided upper confidence bound at or below 50%: premise fails;
  do not build or run v2 training;
- otherwise inconclusive at 2,048 games per arm: do not start a full training
  cycle. First tune or reassess the search itself, or extend the preregistered
  matches with valid repeated-look error control.

This gate establishes whether a searched teacher signal exists and at what
scope. It does not guarantee that a student can absorb it.

### 7.2 Conditional v2 objective

Produce a network that internalizes useful progressive-search decisions while
retaining the incumbent's behavior elsewhere, then select it under the actual
800-sim advisor search profile. The v2 run should be treated as conservative
expert iteration rather than a fresh broad retraining cycle.

### 7.3 Replay architecture: unchanged from v1

Per the 2026-08-12 owner decision, v2 launches with v1's replay architecture:
a single 200,000-example FIFO buffer, frozen-incumbent progressive prefill to
capacity at the Gate-0-selected scope, then FIFO replacement by learner
self-play. Chance-treated replay weight stays 1.0x, and the existing
exact-endgame weighting semantics are unchanged.

The pre-amendment two-reservoir design and 1.5x chance weighting are not
adopted; they are the first two contingency levers in section 7.5. Keeping the
replay path untouched preserves the two-change identity of the run: if the
amended v2 also fails to promote, the result cleanly localizes to either the
optimization-pressure hypothesis or a missing/weak teacher signal, rather than
an untested interaction with new replay machinery.

The distillation rationale is unchanged: the teacher network supplies priors,
but progressive MCTS supplies a refined policy target the student can absorb.
Gate 0 exists to prove that refinement is real before v2 relies on it.

### 7.4 Optimization

| Parameter | V1 | Amended v2 |
|---|---:|---:|
| Learning rate | `1e-4` | `3e-5` |
| Optimizer steps / 400 games | 192 | 96 |
| Approximate steady-state reuse | 5.6x | 2.8x |
| Chance-treated replay weight | 1.0x | 1.0x (unchanged) |
| Replay architecture | Single FIFO | Single FIFO (unchanged) |

Keep architecture, batch size, weight decay, augmentation, value losses,
policy pruning, exact solving, 4,800/200 playout-cap schedule, and other search
constants unchanged unless a preflight test identifies an incompatibility. The
training configuration diff against v1 must be exactly two lines: learning
rate and optimizer steps.

### 7.5 Contingency levers (preregistered, not in the launch recipe)

Applied one at a time, and only if the amended v2 cycle fails to promote
despite a positive Gate 0:

1. **Protected teacher reservoir:** 50/50 split of the logical pool, the
   protected half immutable frozen-incumbent prefill sampled 50/50 against a
   rolling learner half. The iteration-10-versus-20 trajectory (section 5.4)
   makes drift the leading explanation, so this is the first lever. If
   invoked, revisit or anneal the protected share after a first successful
   promotion so it does not permanently cap policy movement.
2. **Chance-treated replay weight 1.5x**, logging both the raw treated share
   and the resulting minibatch share, without reducing exact-target priority.
3. **Policy-preservation KL on learner-generated rolling rows only.** Never
   apply KL toward the teacher's raw prior on teacher-generated rows: their
   replay targets are already the frozen incumbent's searched policy, and the
   raw-prior pull would conflict with search distillation. Any KL experiment
   needs separate loss/gradient telemetry before cloud spend.

Each lever, if invoked, gets its own preflight checks (reservoir persistence
and eviction protection; minibatch ownership audit; weighting draw-rate audit)
and its own preregistered amendment note in this document.

### 7.6 Chance-search scope and width

Retain uniform random reveal sampling. Earlier random-versus-structured tests
slightly favored random sampling; v2 should not overturn that empirical result.

The four-tile chance support sizes are:

| Hidden deck count | Possible sorted four-tile reveals |
|---:|---:|
| 8 | 70 |
| 12 | 495 |
| 16 | 1,820 |
| 20 | 4,845 |

Support size alone does not require width proportional to the population. The
sampling error of an expectation is governed primarily by reveal-value variance
and sample count, while every additional row reduces conditional search depth.

The v2 self-play treatment scope is whatever Gate 0 selects (section 7.1):

| Deck | Width cap | Admission | In v2 if |
|---:|---:|---|---|
| 8 | 16 | Existing `N_init=2`, `D_min=4` | Always, given Gate 0 passes |
| 12 | 16 | Existing `N_init=2`, `D_min=4` | Always, given Gate 0 passes |
| 16 | 16 | Initial proposal `N_init=4`, `D_min=8` | Arm B non-negative and mechanically healthy |
| 20 | 16 | Strictly later admission than deck 16 | Arm C non-negative and mechanically healthy |

Later admission at deeper decks protects conditional depth at earlier chance
nodes while keeping the agreed cap constant. This requires extending Rust
support beyond the current deck-8/deck-12 mask to a generic per-deck
configuration with per-deck `N_init`/`D_min` — work Gate-0 arms B and C
already require, so it lands before the training launch either way.

### 7.7 Generator policy

- Start from the unchanged incumbent and a fresh frozen-incumbent progressive
  prefill at the Gate-0-selected scope.
- Learner self-play replaces buffer contents FIFO, as in v1.
- Keep the incumbent as the deployment model until a candidate passes the full
  advisor-aligned promotion gate.
- A learner that fails an interim gate is not automatically discarded; it may
  continue training unless it crosses a preregistered regression/futility bound.

## 8. Proposed evaluation and promotion

### 8.1 Advisor-aligned search

The advisor is the deployment target and can afford more search than the v1
400-sim gate. Use this final comparison profile for both models:

- 800 simulations/NN rows;
- symmetric progressive search;
- cap 16;
- the Gate-0-selected treated-deck scope;
- identical seeds, seat swaps, network/value constants, and row accounting.

At the observed 21-minute runtime for 2,048 games at 400 sims, budget roughly
40-45 minutes for 2,048 games at 800 sims. Calibration should replace this
linear estimate if measured throughput differs.

The persistent Rust advisor handle must receive the same progressive search
configuration before system promotion. A symmetric evaluator alone does not
change the live advisor.

### 8.2 Sequential paired gate

The 384-game v1 gate plus a hard 55% score floor could not promote a small real
improvement. Replace it with a preregistered sequential match over one fresh
seed block:

1. Look after 512 games / 256 complete seat pairs.
2. If inconclusive, extend the same match to 1,024 games / 512 pairs.
3. If still inconclusive, extend to 2,048 games / 1,024 pairs.
4. At the maximum, an inconclusive result means no promotion.

Compute uncertainty from the actual per-seed scores `{0, 0.5, 1}` using a
paired bootstrap, paired-score t interval, or a reviewed confidence-sequence
implementation. Do not treat the two games in a seat pair as independent.

Remove the separate 55% raw-score floor. Promotion requires a one-sided lower
confidence bound above 50%, plus the existing fixed-suite non-regression veto.
Early promotion must use preregistered alpha
spending or an anytime-valid confidence sequence; repeatedly applying an
ordinary 95% interval at three looks would inflate false promotions.

Each promotion event must use a new preregistered seed block. Seed block
`20330000` is already consumed by the v1 diagnostic evaluations and is not
eligible.

### 8.3 Candidate nomination

Preregister candidate checkpoints rather than selecting whichever noisy rolling
gate looks best. Nominate fixed iterations 5, 10, 15, 20, and 24. After training
completes:

1. Evaluate every nominated checkpoint for 512 games / 256 pairs at the 800-sim
   symmetric progressive profile.
2. Use one shared diagnostic seed block for all five candidates so common
   random numbers improve trajectory comparisons.
3. Rank by pair score, then mean margin, then prefer the earlier checkpoint on
   an exact tie.
4. Freeze the selected candidate before opening any final confirmation seeds.
5. Run the sequential final gate in section 8.2 on a wholly fresh seed block.

The nomination matches are selection data, not promotion evidence. This permits
noisy checkpoint screening without contaminating the final candidate-versus-
incumbent interval or repeating v1's iteration-20 selection error.

### 8.4 Evaluation ladder

1. **Training verdict:** candidate versus incumbent, both using the v2
   progressive 800-sim search. Failure means no network promotion.
2. **System verdict:** after persistent-advisor integration, candidate plus v2
   advisor search versus incumbent plus incumbent advisor search. This requires
   asymmetric per-seat search plumbing and tests the deployable system.
3. **Open-loop fallback:** optional only if preserving an open-loop deployment
   remains a product goal. It is not a substitute for the advisor-aligned gate.

## 9. Required implementation before v2 launch

0. **Harness implemented; cloud result pending.** Asymmetric per-seat
   evaluation search, then the fixed-network Gate 0 in section 7.1 (all arms).
   No remaining v2 training implementation proceeds unless it passes.
1. Generic per-deck Rust progressive support: treated-deck mask beyond 8/12,
   per-deck `N_init`/`D_min` for decks 16 and 20, plus per-deck search
   counters and width/depth histograms (required by Gate-0 arms B and C, and
   by any widened training scope).
2. A treated/exact overlap sanity counter. The categories should remain
   structurally disjoint at every candidate scope: progressive roots are at
   deck 8/12/16/20, while exact-policy roots are detected at deck 4/0. Assert
   zero overlap so future scope changes cannot silently violate that
   assumption.
3. 800-sim symmetric evaluation configuration and audited artifacts.
4. Fixed-checkpoint nomination harness and its shared diagnostic seed schedule.
5. Sequential pair-aware gate with valid repeated-look error control.
6. Persistent Rust advisor progressive configuration and asymmetric system
   evaluation before deployment.
7. Fix the FastAPI/uvicorn cloud dependency or decouple fixed-suite evaluation
   from the web application module.
8. New Gate-0, nomination, and confirmation seed schedules excluding all v1
   diagnostic blocks.

Explicitly excluded from the v2 launch recipe (contingency levers, section
7.5): two-reservoir replay, 1.5x chance weighting, and any KL loss.

## 10. Preflight acceptance checks

Before renting or launching a full v2 cycle:

- fixed-network Gate 0 passes under the exact 800-sim advisor profile, and its
  scope decision is recorded in this document;
- asymmetric evaluation swaps search treatments and seats while loading the
  identical checkpoint hash on both sides;
- the training configuration diff against v1 is exactly two lines: learning
  rate and optimizer steps;
- deck-16/deck-20 panels (if in scope) are hidden-order invariant, use uniform
  random support, and charge bootstrap work to the hard NN-row budget;
- per-deck admission parameters are applied to the correct decks;
- treated/exact overlap is recorded and equals zero;
- cap 16 remains terminal at every treated deck;
- G4-style smoke reaches every in-scope treated deck with admissions and
  widening;
- initialization stays below the 25% guardrail;
- 800-sim evaluation applies identical search to both models;
- checkpoint nomination uses the fixed list and does not inspect confirmation
  seeds;
- the sequential gate passes simulation/null tests at its declared type-I
  error rate;
- the cloud environment reaches the fixed-suite gate without manual package
  installation.

## 11. Decisions and remaining decision points

### Agreed direction (owner sign-off 2026-08-12)

- Continue investigating progressive chance modeling.
- Require the fixed-network progressive-versus-open-loop Gate 0 before v2
  training engineering or cloud rental.
- Run deck 16 and deck 20 as Gate-0 match arms; the v2 training scope follows
  the Gate-0 outcome.
- The only v2 training-recipe changes are learning rate `3e-5` and 96
  optimizer steps per 400 games (about 2.8x reuse).
- Keep v1's single-FIFO replay and 1.0x chance weight; the protected teacher
  reservoir and 1.5x weighting are one-at-a-time contingency levers.
- Omit KL from v2; any later KL experiment applies only to learner-generated
  rolling rows.
- Use 800 simulations for final advisor-oriented evaluation.
- Retain cap 16 rather than scaling width with chance-support size.
- Retain random rather than structured reveal sampling.
- Nominate candidates at fixed iterations before fresh confirmation.

### Proposed defaults requiring final review

- Deck-16 admission `N_init=4`, `D_min=8`; deck-20 admission timing.
- Exact Gate-0 confidence method, arm ordering, and extension rule if 2,048
  games per arm is inconclusive.
- Pair-aware sequential gate implementation and alpha-spending rule.
- Fresh Gate-0, nomination, and promotion seed bases.

### Evidence required before v2

- Direct fixed-network progressive-versus-open-loop playing-strength evidence
  (Gate-0 arms A-C).

### Evidence still useful but not blocking

- 800-sim cap-16 advisor-aligned checkpoint result.
- Direct seed-and-orientation-paired iteration-10-minus-iteration-20 analysis
  from the two saved `games.jsonl` artifacts.
- Verification that the local fixed-suite iteration-10 SHA matches the cloud
  evaluation's iteration-10 checkpoint SHA.
