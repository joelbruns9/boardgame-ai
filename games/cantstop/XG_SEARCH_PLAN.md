# XG-inspired search and training plan for Can't Stop

Date: 2026-09-29
Status: proposed implementation and experiment plan; no runs started by this document.

## Objective and starting point

Implement progressive candidate evaluation, short rollouts, and variance reduction as one configurable search framework. First establish whether it improves decisions and arena strength, then use it to improve training targets and self-play. Keep the existing advisor unchanged until offline evidence supports deployment.

The baseline already solves the current turn exactly given NN values for end-of-turn boards. Additional search must improve those continuation estimates or directly improve the choice between current actions. Exact current-turn calculation does not mean that its NN boundary values are exact.

The existing selective whole-turn search remains a comparison method. Its four-expansion, depth-one confirmation scored 989/2,000 (49.45%; approximate 95% interval 47.26%-51.64%). This does not demonstrate an advantage or rule out a small advantage. Avoid tuning to those confirmation seeds.

Reference checkpoint: `runs/p4_pilot/iter_0080.pt`. Start with two players, five columns, nonblocking; extend to all supported variants after correctness and cost are understood.

## Design contract

- A candidate is an actual legal decision: a dice selection or stop/roll. Deduplicate equivalent dice selections by resulting state. A dice candidate may be evaluated through its subsequent stop/roll decision; do not inadvertently force the baseline stop/roll choice when evaluating an improved continuation.
- Preserve exact terminal wins, legal moves, blocking rules, required-column counts, and absolute-seat value vectors. In multiplayer, each acting player maximizes their own win probability.
- Dice outcomes are chance branches with their true probabilities. End-of-turn boards are consequences, not independently selectable actions.
- Define rollout horizon H as completion of the current player's turn plus H additional player turns. Stopping or busting completes the current turn immediately; rolling must finish its remainder. H=0 ends at the current turn boundary. In two-player games, H=2 includes the opponent's next turn and then our next turn. Stop early at a terminal win.
- Compare contenders at the same defined horizon and continuation strength before final selection. Deeper estimates may screen candidates, but do not silently rank a shallow estimate against a deeper estimate as if their errors were interchangeable.
- Keep the current solver as the baseline continuation policy. Stronger early continuation decisions are a separate configuration, with bounded nested search and an explicit baseline fallback.
- Record full configuration, checkpoint and code hashes, seeds, rule variant, value perspective, horizon, and estimator version in every result and training dataset.

## Build order and acceptance tests

### 1. Decision interface, fixtures, and reference evaluator

Build a common interface for baseline, old selective search, and the new search. Expose all legal root actions and their baseline values; support forcing one action before continuing play. Define deterministic tie-breaking and independent RNG streams for game dice, search, and training.

Create a versioned position suite covering opening, midturn, three active runners, near-claims, terminal banking, blocking, tied choices, and 2/3/4 players. Include both close and clearly separated choices. Separate development positions from held-out evaluation positions.

Tests and gate:

- Zero extra budget reproduces baseline values and actions.
- Forced-action transitions match the game engine and preserve the input board.
- Seat permutation and column reflection transform values and actions correctly.
- Exact terminal win vectors, no illegal roll after a completed game, correct fifth-column banking.
- Extra search cannot consume or alter the arena's actual dice RNG.
- Existing current-turn, old-search, and advisor regression tests pass.

Deliverable: saved decision reports and a deterministic comparison harness.

### 2. Equal-budget rollout evaluator

For every legal root action, run the same number of simulations with the same continuation policy and horizon. Support both finite-horizon NN evaluation and full-game outcomes. Begin with fixed budgets and no pruning so the reference behavior is easy to audit.

Evaluate NN endpoints only at supported turn boundaries. Batch independent endpoint evaluations. Record per-action means, sample counts, variances, elapsed time, and the action selected. Full-game estimates describe the specified continuation policy, not perfect play.

Tests and gate:

- Correct horizon counting after stop, bust, continued rolling, and early terminal wins.
- On small enumeratable fixtures, sampled estimates agree with exhaustive expected values within predeclared Monte Carlo tolerances.
- Repeated fixed-seed runs reproduce trajectories and estimates.
- H=0 agrees statistically with the baseline forced-action expectation when using the baseline continuation policy.
- Full-game mode never substitutes NN evaluation for an unfinished trajectory. Safety-limit hits are explicit failures, not silently discarded samples.

Deliverable: a working, unfiltered rollout search and independent long-rollout reference mode.

### 3. Variance reduction

Add two independently switchable methods:

1. Common random numbers across candidates. Index dice by simulation, player-turn, and roll index so different turn lengths do not accidentally shift every subsequent comparison. Preserve each candidate's correct marginal dice distribution.
2. Dice-luck control variates. For each sampled chance event, compute a predeclared approximate value g for the realized outcome and its probability-weighted expectation over possible outcomes. Correct the final payoff vector with `expected(g) - realized(g)`. The conditional mean of this correction must be zero; g need not be a perfect value estimate.

Use existing turn-solver quantities where their perspective and policy match this definition. Freeze the correction rule before drawing the corresponding dice. Do not use decision-visitation weights as outcome probabilities. Do not clip individual adjusted samples to [0,1], because that biases their mean. Retain raw estimates and report any downstream projection separately.

Tests and gate:

- Exhaustive dice enumeration proves the correction has zero expected value on small fixtures.
- Plain and corrected estimates have the same mean across many independent seed batches, within declared statistical tolerance.
- Test unbiasedness with deliberately inaccurate g, not only a near-perfect evaluator.
- Check absolute-seat perspectives, terminal branches, vector sums, and correlated-sample bookkeeping.
- Measure paired-difference variance and variance times runtime. Keep a method enabled by default only if it improves useful precision per second on the position suite.

Deliverable: validated estimators and measured precision-per-second improvements; no assumed XG speedup factor.

### 4. Progressive evaluation and deeper early decisions

Rank all candidates with the baseline. Give every candidate a minimum rollout allocation before aggressive filtering in the initial version. Progress through increasing future-turn horizons, carrying forward contenders within a configurable win-probability window of the leader. Make candidate caps optional; the root action count is small enough to retain all candidates initially.

For deeper deterministic lookahead, enumerate chance outcomes where affordable and apply acting-player optimization at decision nodes. Full expansion across several whole turns can be enormous: expose sampled chance coverage explicitly rather than labeling a partial calculation exact expectiminimax.

Add stronger play for the first one or two future turns, then revert to the baseline for the rest of a rollout. Bound the nested search budget explicitly to prevent recursive rollout explosion. Cache reusable board evaluations and turn solves with checkpoint/rules/perspective/configuration-aware keys.

Tests and gate:

- Exhaustive small-tree reference checks for chance weighting, opponent choice, depth, and propagation into current decisions.
- No-pruning mode agrees with the equal-budget reference at matching settings.
- Horizon changes, pruning, tie handling, and cache reuse are reproducible.
- Report how often filtering eliminates the winner of a substantially stronger independent evaluation and the estimated cost of those errors.
- Include fixtures where an initially inferior action becomes best after future-turn evaluation.

Deliverable: all three XG-inspired components integrated, with independent ablation switches.

### 5. Adaptive allocation, throughput, and failure handling

After fixed-budget behavior is validated, allocate additional rollout batches to contenders whose ordering remains uncertain. Use uncertainty in paired action differences. Use a sequentially valid confidence method or predeclared evaluation stages; repeatedly checking an ordinary fixed-sample 95% interval is not a valid stopping guarantee.

Separate sampling uncertainty from NN endpoint bias and continuation-policy error. If the budget expires unresolved, use a documented fallback, initially the baseline action among unresolved contenders.

Batch across candidates and independent positions on the GPU; move hot simulation/control paths to Rust after profiling. Reuse identical turn solves with bounded caches. Record GPU utilization, NN rows, completed rollouts, cache hits, peak memory, and median/p95 decision latency. Provide fixed-work budgets for reproducibility and soft time budgets with measured overshoot.

Tests and gate:

- Adaptive behavior on known-gap toy problems, including exact ties and a misleading initial ranking.
- Fixed-seed serial/batched agreement within documented numeric tolerances.
- Resume preserves RNG state, sample counts, and paired statistics without duplication.
- Cancellation, inference failure, memory limits, and exhausted budgets leave explicit results or a valid fallback.
- Profile early/middle/late boards across variants before projecting training time.

Deliverable: offline arena-ready implementation with measured throughput.

## Search experiments before training

All counts below are proposed budgets, not claims of adequate power for every effect size. Keep development and confirmation seeds separate. Compare both equal wall-clock cost and equal simulation counts; warm inference before timing.

| Run | Purpose | Initial budget |
| --- | --- | --- |
| S0 | Correctness and timing | 100 diverse saved decisions, all supported variants |
| S1 | Horizon/sample/variance-reduction ablation | 300 development decisions; H=1,2,4; 32 and 128 simulations per candidate; baseline continuation |
| S2 | Filtering and stronger early-play ablation | Best S1 settings; filtering off/on; baseline versus stronger first future turn; fixed matched compute budgets |
| S3 | Independent decision check | 300 held-out decisions; longer independent full-game rollouts, initially 1,024 per contender with predeclared extensions for close cases |
| S4 | Arena screening | 400 seat-balanced games per shortlisted setting, two-player extended nonblocking |
| S5 | Primary confirmation | One locked configuration, 10,000 fresh seat-balanced games against the same checkpoint without extra search |
| S6 | Generalization | 3,000 games in 3-player/4-column/blocking, then 2,000 games per remaining variant for regression screening |

Run S1/S2 in stages rather than executing the entire Cartesian product blindly. S3 must contain ordinary positions as well as search disagreements; report disagreements separately. Independent long rollouts remain uncertain estimates under a specified policy.

Measure all changed decisions, including later stop/roll and dice choices, not just opening dice selections. Report estimated decision improvement, harmful-change rate, unresolved fraction, filtering errors, and precision per second.

For arenas, rotate the search seat and report seat-stratified results. In multiplayer, specify one challenger versus baseline opponents; its null rate is 1/player_count, not 50%. Keep opponent checkpoints fixed. If paired/reused game seeds are introduced, analyze seed blocks rather than pretending all games are independent.

The 10,000-game primary test has roughly 80% power for a 1.4 percentage-point departure from 50%, under an independent-game normal approximation with a two-sided 5% test. Smaller gains need more games. Predeclare the primary comparison; exploratory sweeps do not each confer an independent 95% strength claim.

Gate for expensive training: correct implementation, measured usable throughput, and independent evidence that the proposed teacher improves decisions. Prefer arena confirmation; if arena strength remains unresolved, restrict training to a small labeled exploratory pilot.

## Training integration and runs

### Target semantics and integration work

The current network predicts end-of-turn board values. Preserve that input/output contract initially. For a stored boundary board, search starts before the next player's opening dice. Integrate or sample those dice correctly; do not attach a favorable observed-roll value to a pre-roll input.

Separate two interventions:

- Search targets/reanalysis: keep baseline data-generation behavior and replace selected board targets with stronger search estimates.
- Search behavior: let search choose actions during data generation, changing which positions are visited.

This distinction lets us find whether better labels, better trajectories, or both cause improvement. Build explicit target metadata: teacher checkpoint, search configuration, target version, uncertainty, and generation iteration. Reanalyze selected replay rows with the frozen current teacher rather than silently reusing obsolete targets. Do not mix target modes without recording their proportions.

Freeze the teacher within an iteration. Use separate random samples to select a teacher action and estimate its training value, limiting optimistic selection bias. Preserve exact terminal targets. If adjusted rollout targets leave the probability simplex, define and test the training loss/projection policy and measure its bias; never silently clip raw rollout samples.

Retain the existing 70% ordinary / 20% conservative / 10% aggressive game mix for initial matched experiments. Personas affect behavior; target evaluation uses the declared unbiased teacher policy. Hold augmentation, replay window, TD settings, optimizer, rule mixture, and row quotas constant between arms. Random starting boards are a separate future experiment, not bundled into these runs.

Required tests:

- Pre-roll target perspective, dice expectation, and value-vector alignment.
- Teacher frozen within iteration; checkpoint/version changes invalidate relevant caches.
- Reanalysis replaces only intended rows and preserves provenance.
- Resume restores optimizer, replay, RNG, teacher identity, and search settings.
- Zero search fraction reproduces the existing training path.
- Reflection and seat transformations apply consistently to stronger targets.

### Staged training schedule

Use fresh output directories under `runs/xg_search/`. Names and settings here are experiment specifications, not already implemented CLI commands.

| Run | Configuration | Initial size and gate |
| --- | --- | --- |
| T0: plumbing smoke | Search target mode, then search behavior mode | 2 iterations, 200 rows per variant; check finite losses, complete provenance, resume, and terminal values |
| T1-C: matched control | Continue reference checkpoint with original targets/behavior | 20 iterations, 4,000 rows per variant per iteration, all existing generalist variants |
| T1-R: reanalysis | Same as control; independently select 25% of new/replay target rows for search reanalysis | Same row and optimizer-update budgets; one training seed for screening |
| T1-P: search behavior | Search controls 25% of generated games; original target method | Same budgets; isolates trajectory effect |
| T1-RP: combined | Same reanalysis fraction and search-game fraction | Run after the single-intervention arms establish cost and correctness |
| T2: replication | Best T1 intervention versus matched control | 3 fresh training seeds each, 40 iterations, matched rows/updates; also report strength at matched elapsed-compute checkpoints |
| T3: scale-up | Winning intervention across the existing full variant mix | Up to 80 iterations; raise search fraction to 50% only if quality and cost justify it |

Choose learning rate and inherited training settings from the actual reference run metadata before launching T1. Use the same settings in every matched arm; do not accidentally restart the fresh-network high-learning-rate schedule. Record whether optimizer/replay state is inherited or reset, and apply the same choice across arms.

Evaluate every 5 pilot iterations for monitoring, but preselect the final checkpoint or a separate validation-based selection rule before confirmation. Validation games used to choose a checkpoint cannot also be its final confirmation.

Evaluate each trained checkpoint in two distinct modes:

1. No extra search versus the frozen reference and the matched training control. This tests whether stronger play has been distilled into the NN.
2. Identical extra-search budgets on both trained and control checkpoints. This tests the strength of the complete playing system.

Use 400-game development screens, then a separately seeded 10,000-game two-player confirmation for the selected primary comparison. Report per-training-seed results as well as an aggregate; many games on one checkpoint do not establish robustness across training seeds. Screen all other variants for regression and retain the fixed heuristic/persona evaluation suite as secondary diagnostics.

Scale up only after replicated improvement at an acceptable compute cost. Persist intermediate checkpoints and complete reports even when a run is stopped for weak results.

## Timing and promotion criteria

Estimate runtime from measured end-to-end pilot throughput, including generation, reanalysis, optimization, and evaluation. A useful forecast is baseline generation time plus searched decisions times measured incremental search cost plus reanalyzed rows times measured target cost; use actual batched throughput rather than multiplying serial microbenchmarks. Publish measured hours per iteration before T1 and T2.

Promote search into training only after teacher quality and cost are understood. Promote a trained checkpoint only after fresh evaluation supports improvement without material variant regressions. Advisor integration is a later task with its own latency and cache design; decision-specific rollouts cannot automatically inherit the existing once-per-turn cache guarantees.

## Documentation basis

The three mechanisms are directly applicable; their numerical budgets and benefits must be calibrated for this game.

- [XG search intervals](https://www.extremegammon.com/Searchinterval.aspx): progressively deeper evaluation of promising moves and stronger early rollout decisions. Much of the article describes XG1, with a separate XG2 update; do not copy historical thresholds as Can't Stop probability margins.
- [XG engine FAQ](https://www.extremegammon.com/support.aspx): short rollouts, finite horizons, and different early/late evaluation strengths.
- [XG2 manual](https://www.extremegammon.com/extremegammon2.pdf): rollout seeds, truncation, variance reduction, minimum samples, precision goals, and time/sample limits.

Implementation details above, including random-stream coupling, training arms, estimator tests, and promotion gates, are our proposed adaptations rather than claims about undisclosed XG internals.

## Step 1 implementation (2026-09-29)

`decision_search.py` now defines immutable actions/results, a `DecisionBackend`
protocol, and `TurnTableBackend` adapters for the existing baseline and selective
search. Future rollout backends implement the same `evaluate(state)` entry point.
Each result includes all distinct decision outcomes, sorted by the actor's value,
and the existing solver's selected action. Exact ties retain the solver's choice;
reflection can choose another equally valued action because canonical tie order
is not reflection invariant. Only cyclic seat relabeling preserves turn order.

`force_action` clones its input. A roll choice returns an awaiting-roll state
without drawing dice; callers supply dice through the engine. Winning banks use
the solver's stop-only convention: the engine technically permits declining a
win, but the solver does not evaluate that dominated continuation. A terminal
state returns its exact winner vector without invoking the evaluator.

`rng_stream(seed, domain, index)` provides separately derived portable game,
search, and training streams. The decision interface never receives game RNG
state. This is infrastructure for future rollouts; no stochastic search is
implemented in step 1.

The versioned fixture file `tests/fixtures/search_decisions_v1.json` contains
50 development cases across all ten variants and 10 reserved legal-play cases.
These are correctness fixtures, not a representative strength benchmark. Flat
mock evaluation exercises ties; the NN reports retain all candidate scores for
close-choice analysis. Enlarge the independently sampled held-out set before
strength tuning and the S3 experiments.

Run from the Can't Stop checkout with the project virtual environment:

```powershell
python -m games.cantstop.decision_compare --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --out runs/decision_zero_budget.json
python -m games.cantstop.decision_compare --expansions 1 --depth 1 --out runs/decision_selective_heuristic.json
python -m pytest games/cantstop/tests/test_decision_search.py -q
```

Zero expansions is the default. Omitting the checkpoint uses the deterministic
progress heuristic. Reports include every candidate value, selected actions,
input snapshots, source/checkpoint/suite hashes, and completion status. Output
paths must be new. Development is the default split; `--split heldout` explicitly
opens the reserved cases. Comparison values/actions are reproducible; run metadata
contains timestamps and command lines and therefore is not byte-identical.

Initial CUDA verification on `p4_pilot/iter_0080.pt` matched all 50 development
positions exactly between baseline and zero-budget selective search, including
all candidate value vectors. No search-strength or training claim follows from
this equivalence check. Next build step: equal-budget rollouts (step 2).

Step 1 acceptance checks completed: 86 new decision-interface tests plus 337
existing engine, current-turn solver, selective-search, and advisor checks passed
(423 total). The final NN comparison report is
`runs/decision_step1_nn_final_20260929.json`: 50/50 development positions identical,
zero changed choices. Step 1 is complete; step 2 has not started.

## Step 2 implementation (2026-09-30)

`rollout_search.py` implements `RolloutBackend.evaluate(state)` with the common
step-1 result interface. Every distinct legal candidate gets exactly `samples`
simulations. Each candidate is forced once, then the baseline Rust turn solver
chooses subsequent dice and stop/roll decisions. The root table is shared across
samples; future turns get fresh baseline tables, reused throughout that turn.
There is no pruning, adaptive allocation, stronger nested search, or variance
reduction yet.

Finite-horizon mode finishes the current turn and then H additional player
turns. H=0 finishes only the current turn; H=1 also finishes the next player's
turn. A forced stop/bust counts as completing the current turn. Nonterminal
endpoints are evaluated before the next opening dice, in bounded batches grouped
by rules and player-to-move so NN outputs retain the correct absolute-seat
perspective. Terminal wins always use exact one-hot vectors.

Full-game mode follows the baseline policy to an actual winner. It still uses
the NN inside baseline turn solves, but never substitutes an NN endpoint value
for a game result. Turn/roll safety limits raise an explicit failure, mark the
report incomplete, and produce no final comparison summary. No failed sample is
silently discarded. Endpoint probabilities are checked for shape, range,
finiteness, and normalization.

Search dice use uniform rejection sampling on the portable RNG. Streams are
independently derived from the configured seed, board, candidate, and simulation
index. This preserves reproducibility without consuming arena game randomness
or accidentally pairing candidates. Common random numbers belong to step 3.

Per-action reports include means, sample variances, standard errors, sample and
terminal counts, completed turns, dice rolls, and generation time. Batched
endpoint time and total elapsed time are also recorded. Standard errors describe
sampling uncertainty only, not NN/policy error; one sample reports unavailable
variance/standard error. Exact sampled-value ties follow canonical engine action
order. Selecting the highest sampled mean is not an unbiased estimate of the
selected action's true value; independent selection/value samples are still
required for the later training integration.

### Commands

Run from the Can't Stop checkout using the project virtual environment. Use a
new output filename for each run. Start with selected positions: full-game
rollouts from early boards can be expensive.

```powershell
# Finish this turn plus the next player's turn, 32 simulations per choice.
python -m games.cantstop.decision_compare --backend rollout --samples 32 --horizon 1 --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --position 2p5n_three_runners --out runs/rollout_h1_32.json

# Full-game reference, 32 simulations per choice.
python -m games.cantstop.decision_compare --backend rollout --samples 32 --full-game --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --position 2p5n_three_runners --out runs/rollout_full_32.json

python -m pytest games/cantstop/tests/test_rollout_search.py games/cantstop/tests/test_decision_search.py -q
```

`--position` is repeatable; omitting it evaluates the selected fixture split.
`--seed` controls search randomness. `--max-turns` (default 1000) and `--max-rolls`
(default 10000) apply separately to each simulated trajectory. `--full-game`
and `--horizon` are mutually exclusive. Omitting `--backend rollout` retains
the original selective-search comparison mode.

### Acceptance results

- 52 rollout tests passed, covering horizon counting for all phases and 2/3/4
  players, root-action forcing, endpoint batching/perspective, full-game outcomes,
  exact terminal wins, safety-limit failure reports, invalid settings/values,
  reproducibility, RNG isolation, and sample statistics.
- H=0 rollout means agree with an exhaustive 1,296-dice reference and exact
  baseline action values on compact positions, using 4,096 samples per action
  and a predeclared absolute Monte Carlo tolerance of 0.022. Additional dice-choice
  checks preserve optimal subsequent stop/roll decisions.
- 86 decision-interface tests and 139 existing search/advisor checks passed
  (277 checks total including the new rollout tests).
- Actual-checkpoint CUDA finite-horizon checks completed for two-, three-, and
  four-player fixtures: `runs/rollout_step2_h1_cuda_20260930.json`.
- Actual-checkpoint full-game check completed four simulations each for stop and
  roll from a two-player/five-column midturn board. All eight reached winners;
  together they simulated 171 completed turns and 947 dice rolls. Report:
  `runs/rollout_step2_full_cuda_20260930.json`.

These are correctness and integration checks, not arena strength evidence. The
full-game smoke evaluation took about 2.13 seconds for this one position and tiny
budget; it is not a general latency forecast. Step 2 is complete. Next: step 3,
variance reduction. No training or advisor integration has been enabled.

## Step 3 implementation (2026-10-01)

Both variance-reduction methods are implemented as independent, opt-in rollout
settings. `--common-random-numbers` shares dice between competing actions by
simulation index, completed player-turn index, and roll index within that turn.
A longer first turn cannot shift the dice assigned to a later turn. Each
trajectory retains the correct marginal dice distribution.

`--dice-luck` adds `expected(g) - realized(g)` after sampled rolls, using the
baseline turn table for g. Expected value is computed before drawing dice, with
the true dice probabilities. A bust uses the table's original bust-leaf vector;
all values and corrections are in absolute-seat order. `RustTurnSolver.bust_value`
returns a copy of that vector without a new NN call or native rebuild.

The root table is already available. For future turns, the first opening roll
uses a zero control variate; the usual table is built after observing its dice,
and later rolls use that table for correction. This deliberate choice avoids a
costly extra pre-roll solve, retains the step-2 continuation policy and trajectories,
and remains unbiased. It leaves some opening-roll noise for shared dice or larger
sample budgets to address. An initial pre-roll-table implementation was abandoned
after timing; its two interrupted reports are explicitly marked incomplete.

Corrections are accumulated across the trajectory. Raw and adjusted means are
both recorded, along with correction means and the count of adjusted samples
outside [0,1]. Adjusted samples are never clipped or normalized. They are unbiased
estimators of the declared rollout payoff, not individually valid probability
vectors. NN endpoint and continuation-policy bias remain.

Reports now include every pairwise difference between root candidates: mean,
sample variance, standard error, raw difference variance, and whether dice were
paired. Paired uncertainty uses the observed differences, including their
covariance; it does not add independent-action variances. One-sample uncertainty
is unavailable. These are descriptive fixed-budget standard errors, not an
adaptive-stopping confidence guarantee.

### Commands

```powershell
# Both methods on a short-horizon comparison.
python -m games.cantstop.decision_compare --backend rollout --samples 32 --horizon 1 --dice-luck --common-random-numbers --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --position 2p5n_three_runners --out runs/rollout_h1_vr.json

# Four-way precision/time ablation: plain, shared dice, luck correction, both.
python -m games.cantstop.benchmark_rollout_variance --checkpoint runs/p4_pilot/iter_0080.pt --samples 8 --batches 4 --horizon 1 --out runs/rollout_variance_pilot.json

python -m pytest games/cantstop/tests/test_rollout_variance.py -q
```

The comparison switches also work with `--full-game`. They require
`--backend rollout`; omitting both preserves uncorrected, independent-dice
rollouts. Keep both opt-in until larger quality/cost experiments select defaults.

The benchmark uses independent seed batches, rotates configuration execution
order, warms inference, and records both within-batch estimated variance and
between-batch variance of the comparison mean. It reports variance of the mean
multiplied by elapsed time: lower means better estimated precision per second.
Candidate pairs on the same position are correlated and must not be counted as
independent experiments.

### Acceptance results

- 292 targeted tests passed: 15 variance/benchmark/CLI checks, 52 step-2 rollout
  checks, 86 decision-interface checks, and 139 search/advisor regressions.
- Exhaustive enumeration of all 1,296 ordered dice outcomes verifies zero-mean
  corrections with flat and deliberately inaccurate hashed values, mirrored
  positions, and 2/3/4-player absolute-seat vectors.
- Full-game tests across 24 independent seed batches check mean preservation,
  while retaining adjusted samples outside [0,1]. Tests also cover shared-dice
  coordinate stability, sample statistics, reproducibility, and CLI validation.
- H=0 correction removes essentially all sampling variance in the compact exact
  reference case, as expected when the control variate matches the endpoint
  evaluator and continuation policy.
- Real-model CUDA pilot: six development positions, four independent batches,
  eight samples per candidate, H=1; all 96 evaluations completed. Across 12
  candidate pairs, median variance-times-runtime ratios relative to plain were:

| Configuration | Median relative variance x runtime |
| --- | ---: |
| Plain | 1.000 |
| Shared dice | 0.454 |
| Dice-luck correction | 0.004 |
| Both | 0.001 |

Report: `runs/rollout_variance_step3_final_cuda_20261001.json`. These are small
exploratory measurements, not general speedup guarantees or strength evidence.
The large short-horizon reduction is consistent with the control variate using
the same NN-backed turn calculations as the endpoint policy. It does not measure
or remove NN evaluation bias. Plain and corrected actual-model endpoint/count
fingerprints matched for every corresponding position/seed batch.

Both switches also completed real-model full-game rollouts, eight samples per
candidate, from a nonterminal two-player/five-column decision:
`runs/rollout_variance_step3_full_cuda_20261001.json`.

Step 3 is complete. Next: progressive evaluation and deeper early decisions
(step 4), followed by adaptive allocation and throughput work. Advisor and
training defaults remain unchanged.

## Step 4 implementation (2026-09-30)

`progressive_search.py` implements staged comparisons through the common decision
interface. Baseline scores rank the initial candidates, but every candidate gets
the first configured rollout stage. Between stages, an optional margin and/or
candidate cap selects survivors. All survivors advance to the same next horizon,
using the same continuation policy. Final selection uses only the final-stage
values; earlier, pruned estimates are retained in labeled stage diagnostics and
are not mixed into the final ranking.

The default stages are H=0 with 8 samples per action and H=1 with 32. Filtering
is disabled by default. Horizons must strictly increase; `full` is allowed only
as the last stage. One remaining survivor still receives the final-stage budget.
Stages use the same configured seed and stable per-action sample indexing for
reproducibility. This is fixed-stage screening, not statistical confidence pruning
or an unbiased estimate after adaptive selection.

### Stronger early continuation

`--early-turns N --early-expansions K --early-depth D` enables the existing bounded
selective whole-turn search for the first N future player turns of each rollout.
The current turn remainder uses the baseline. Later turns also use the baseline.
K caps additional turn solves per stronger table, across that table's search tree;
D limits additional player-turn depth. The budget is per simulated turn solve,
not a single global budget for the entire root decision. Samples and candidates
therefore still multiply cost.

This reuses the validated chance-weighted turn search; it does not recursively
launch additional rollouts. Current-turn backups are exact given their frontier
values, while selective future-turn coverage remains approximate. This is not
full N-turn expectiminimax. `WholeTurnSearch.bust_value` exposes its updated bust
continuation, so dice-luck correction uses the same backed-up values on both sides
of its expected-minus-realized calculation.

A per-evaluation LRU reuses identical tables across candidates, simulations, and
stages. Keys include the full state (rules, seats, board, phase, runners, dice)
and baseline/stronger mode. Evaluator and search configuration are fixed for the
cache lifetime; each new decision gets a fresh cache, so no tables survive a
checkpoint change. Both entry count and retained solver-position count are
bounded. Oversize solves run without retention: these are cache bounds, not hard
limits on a single solve's transient memory or elapsed time. Cache hits, solve
counts, total expansions, and peak retained positions/entries are reported.

### Commands and filtering audit

```powershell
# Two stages, retaining choices within two estimated win-percentage points.
python -m games.cantstop.decision_compare --backend progressive --stage-horizons 0 1 --stage-samples 8 32 --margin 0.02 --early-turns 1 --early-expansions 1 --early-depth 1 --dice-luck --common-random-numbers --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --position 2p5n_opening --out runs/progressive_h0_h1.json

# Same comparison plus an independently seeded full-game reference for all actions.
python -m games.cantstop.decision_compare --backend progressive --stage-horizons 0 1 --stage-samples 8 32 --margin 0.02 --early-turns 1 --early-expansions 1 --dice-luck --common-random-numbers --audit-samples 128 --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --position 2p5n_opening --out runs/progressive_filter_audit.json

python -m pytest games/cantstop/tests/test_progressive_search.py -q
```

For progressive runs, `--stage-horizons` and `--stage-samples` set the actual stage
budgets; the single-stage `--horizon`/`--samples` settings do not replace them.
Use `--stage-horizons 0 1 full` with three matching sample budgets for a full-game
last stage. Omit margin/cap for the unfiltered reference. `--max-candidates` caps
survivors after the initial allocation; `--cache-entries` and `--cache-positions`
set retention bounds. Zero early turns/expansions leaves baseline continuation.

`--audit-samples` uses a separate seed, full-game outcomes, all legal candidates,
and the same declared early/late continuation policy. Reports identify whether
the reference's selected action was pruned, the estimated value lost through
filtering, and the estimated value lost by the actual selection. The reference's
sampling statistics are preserved. These estimates are not ground truth, and
small reference budgets can disagree through noise. Aggregate them on the larger
independent position suite before choosing filtering defaults.

### Acceptance results

- 24 new progressive-search tests passed, plus 292 existing targeted checks
  (316 total).
- With filtering disabled, the final result exactly matches a direct rollout at
  the matching final horizon/sample budget, with variance reduction and stronger
  early continuation independently enabled or disabled.
- Small 2/3/4-player stronger-turn solves agree with the independent Python
  one-expansion backup reference. Existing tests cover deeper ancestor propagation.
- Exhaustive dice enumeration confirms zero-mean luck corrections after updated
  search frontier values, including the refined bust continuation.
- Tests cover candidate subsets, first-stage allocation, margin/cap filtering,
  fixed budgets, cache eviction/oversize behavior/evaluator identity, full-game
  final stages, incomplete failures, and independent audit CLI output.
- A saved regression construction initially prefers stop (about 72.8% versus
  68.8% for roll), but H=1 evaluation reverses that ranking. A one-candidate cap
  incorrectly prunes roll; a separately seeded, larger reference detects that
  filtering loss. This demonstrates current-decision impact and why pruning
  remains optional, not a strength result for the trained network.
- Real-checkpoint CUDA checks completed for 2p/5-column opening, 3p/4-column
  blocking midturn, and 4p/3-column midturn positions with one stronger future
  turn. Report: `runs/progressive_step4_cuda_20260930.json`.
- An actual-model independent full-game audit completed 16 simulations for each
  of three opening choices: `runs/progressive_step4_audit_cuda_20260930.json`.
  No choice was pruned in that case. The short-horizon selection differed from
  the audit selection; that small budget is not sufficient to establish which
  is better. Its purpose was verifying the diagnostic path end to end.

Step 4 is complete. Next: step 5, adaptive allocation, throughput, and failure
handling. No advisor or training integration has been enabled.

## Step 5 implementation (2026-09-30)

The offline framework now includes adaptive allocation, resumable checkpoints,
cooperative budgets/cancellation, bounded inference batching, profiling, and an
arena accepting fixed-rollout, progressive, or adaptive decision backends.

### Adaptive allocation and uncertainty

`AdaptiveBackend` compares all legal root actions at one fixed horizon and fixed
continuation policy. At predeclared cumulative sample counts (default 32, 128,
512), it removes demonstrably inferior actions and spends subsequent batches only
on survivors. It does not certify a set previously pruned by heuristic margins;
progressive filtering remains a separate optional controller.

Elimination uses paired differences of the RAW rollout payoffs. Each difference
is bounded in [-1,1]. For P original candidate pairs and J planned checks, the
Hoeffding half-width at n samples is:

`radius = sqrt(2 * log(2 * P * J / alpha) / n)`

The union bound covers every original pair and planned check for this decision.
Sharing dice inside a pair is allowed; simulation indices are independent. No
repeated ordinary 95% interval or normal-distribution assumption is used. The
scope is the declared fixed rollout policy, not perfect play or NN accuracy.

Variance-corrected estimates and their observed sampling statistics remain in
reports. The stopping rule deliberately does not treat corrected samples as
bounded probabilities, nor assume their variance alone justifies a normal
interval. Consequently, allocation is conservative and does not yet capture all
of variance reduction's potential for early stopping. Small budgets often leave
close choices unresolved. A future tighter valid bound can improve efficiency.

If multiple choices survive, the action is the baseline's preferred surviving
choice, explicitly marked `fallback=true`. It may differ from the largest noisy
corrected sample mean. One survivor is marked resolved. No early stopping claim
is made about NN or continuation-policy bias.

### Resume and failure handling

Simulation streams are addressed by explicit sample index. Complete aligned
batches are committed atomically; partial/interrupted batches are discarded and
replayed from their original indices on resume. Checkpoints contain raw and
corrected samples, counts, stage history, survivors, seeds/configuration, board,
model identity, and source/native-library hashes. Resume rejects incompatible
boards, models, code, configurations, and malformed sample prefixes.

`--seconds` is a soft per-position allowance, including the mandatory baseline
solve. In-flight native solves and inference finish before cancellation can be
observed; overshoot is measured. Cooperative cancellation is checked during
trajectories and before new solves/inference. Ctrl+C in the comparison driver
signals its workers to stop and preserves completed checkpoints. A hard process
kill can leave a running label, but only committed checkpoint batches are resumed.

Inference failures and sample-storage limits return a valid baseline fallback
when baseline evaluation succeeded, with explicit error/status fields. If even
the mandatory baseline fails, the position/run reports failure rather than
inventing advice. No failed rollout is silently treated as a completed sample.

The configured numeric sample-payload budget, turn-table cache entry/position
bounds, GPU batch row cap, and queued inference row cap limit their respective
resources. They are not a hard cap on total process memory: Python objects,
serialization, Torch, and an in-flight native solve add overhead. Process and
CUDA peak memory are measured separately.

### Throughput and profiling

`BatchingEvaluator` combines concurrent requests from different candidates and
positions into a single NN forward, including mixed player counts and active
seats. Results are rotated back separately for each request. A single inference
worker owns model calls; bounded queues provide backpressure and propagate errors
to all affected waiters. Parallel candidates share the immutable root turn table;
future-turn caches remain worker-local.

The profile showed substantial work in native turn construction/feature export,
not the Python simulation loop. No additional Rust rewrite was justified by this
small profile. One measured overhead was removed: the live-seat mask now uses a
strided tensor view instead of allocating/transferring a four-element GPU index
tensor on every inference. Outputs match the prior index-select implementation
on CPU/CUDA, including noncontiguous inputs. Model/checkpoint formats are unchanged.

The comparison driver records NN rows, committed rollouts, cache statistics,
median/p95 latency, process/CUDA peak memory, sampled GPU utilization when NVML is
available, and inference batch sizes/counts. GPU utilization is device-wide and
can include other processes. Parallel candidate cache counters at the controller
level describe the shared root cache; worker-local future-cache activity is
reflected in total NN rows and inference requests, not that root-cache counter.

Three alternating-order repeats on early/middle/late 2/3/4-player boards, budgets
4 then 8, H=1, both variance-reduction methods enabled:

| Mode | Median total decision-work wall time | NN rows per run |
| --- | ---: | ---: |
| Serial | 0.803 s | 539,238 |
| Three parallel positions and candidates | 1.087 s | 575,355 |

Both modes chose the same actions; all raw/corrected sample values agreed within
1.28e-7 (test tolerance 2e-6). Batching combined up to five requests, but the extra
scheduling and reduced sharing of future caches outweighed the gain on this
workload. Serial remains the default. These are small fixed-work measurements,
not a general speedup forecast. The initial profiled serial run is under
`runs/adaptive_step5_serial_20260930.states/*.prof`; final timing reports are
`runs/adaptive_step5_final_{serial,batched}_r{1,2,3}_20260930.json`.

### Commands

```powershell
# Fixed-horizon adaptive comparison with a soft per-position time allowance.
python -m games.cantstop.adaptive_compare --checkpoint runs/p4_pilot/iter_0080.pt --budgets 32 128 512 --horizon 1 --dice-luck --common-random-numbers --position 2p5n_three_runners --seconds 2 --out runs/adaptive_pilot.json

# Resume the same committed samples with a larger allowance per invocation.
python -m games.cantstop.adaptive_compare --checkpoint runs/p4_pilot/iter_0080.pt --budgets 32 128 512 --horizon 1 --dice-luck --common-random-numbers --position 2p5n_three_runners --seconds 10 --resume --out runs/adaptive_pilot.json

# Optional batching and per-position CPU profiles; benchmark before enabling.
python -m games.cantstop.adaptive_compare --checkpoint runs/p4_pilot/iter_0080.pt --budgets 4 8 --horizon 1 --dice-luck --common-random-numbers --parallel-candidates 3 --parallel-positions 3 --profile --out runs/adaptive_batch_profile.json

# Small exploratory progressive-search arena, ordinary starting boards.
python -m games.cantstop.adaptive_arena --checkpoint runs/p4_pilot/iter_0080.pt --backend progressive --players 2 --extended --games 20 --stage-horizons 0 1 --stage-samples 8 32 --dice-luck --common-random-numbers --out runs/progressive_arena_pilot.json

python -m pytest games/cantstop/tests/test_adaptive_search.py -q
```

The arena also accepts `--backend rollout` with `--samples`/`--horizon`, or
`--backend adaptive` with `--budgets`/`--horizon`. Soft per-decision `--seconds`
currently applies to the adaptive arena backend. It rotates the challenger seat,
uses independent recorded game dice seeds, and records fallback counts/costs.
A supplied `--start-fixture` is an integration smoke test and disables strength
verdicts. Full strength matches should use ordinary starts and fresh fixed-size
confirmation seeds after choosing the configuration.

Comparison checkpoints default to a sibling `.states` directory beside the
output; `--state-dir` overrides it. Use one writer per output/state directory.
Code or checkpoint changes intentionally invalidate resume, even if filenames
remain the same. The comparison driver's full-game mode is `--full-game`.

### Acceptance results and next work

- 354 targeted checks passed: 22 adaptive/batching/arena/CLI checks, 316 earlier
  search/advisor checks, and 16 model checks. Tests cover known gaps, exact ties,
  misleading baseline rankings, aligned bounds, serial/chunked sample equality,
  cancellation and resume, inference failures, corrupt checkpoints, memory
  limits, mixed-seat batching, and all three arena backends.
- Actual-model soft-budget interruption committed zero samples, then resumed to
  eight samples per candidate without duplication:
  `runs/adaptive_step5_resume_20260930.json`.
- Two seat-balanced fixture games verified arena integration:
  `runs/adaptive_step5_arena_smoke_20260930.json`. No strength verdict was issued.
- The small throughput runs remained unresolved and correctly used baseline
  fallback; their purpose was verifying cost and equivalence, not improving play.

Step 5 is complete. The next work is the planned search-quality/strength
experiments (S0-S6), beginning with budget/horizon comparisons and independent
position audits. Training integration/runs (T0 onward) and advisor integration
remain gated on that evidence; neither has been started.
