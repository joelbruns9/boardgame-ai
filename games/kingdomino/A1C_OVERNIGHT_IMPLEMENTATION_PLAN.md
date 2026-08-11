# Kingdomino A1c Overnight Implementation and Experiment Plan

Status: proposed, not yet executed
Branch: `codex/kingdomino-chance-correct`
Parent roadmap: `games/kingdomino/KINGDOMINO_NEXT_LEVERS.md`
Primary objective: determine whether fully initialized, progressively widened
chance panels produce better deck-8 search decisions than the current open-loop
search at matched neural-network work.

## 1. Decision this work must support

This tranche is not a model-training run and is not a promotion match. It is a
target-quality screen for A1c.

The experiment must answer:

> On positions whose post-reveal continuations are exactly solved, does A1c use
> the current frozen network to rank root actions closer to the exact answer
> than incumbent open-loop search and the existing lazy one-reveal treatment?

The answer determines the next engineering investment:

- If A1c has a credible exact-oracle signal, optimize cross-node bootstrap
  batching and prepare the design for `BatchedMCTS` integration.
- If A1c is clearly negative, stop optimizing it and move to the broader frozen
  position corpus needed to investigate other chance schedules and BGA failure
  modes.
- If the result is mixed or too noisy, retain A1c as a tuning candidate but do
  not integrate it into self-play. The next evidence must come from more
  independent positions, not more seeds on these same eight positions.

No result from eight calibration positions is sufficient to promote a model or
claim an increase in playing strength.

## 2. Current implementation baseline

The branch already contains:

- the incumbent open-loop search (`X=0`);
- the lazy `X=1`, Hajek/balanced observation-splitting treatment;
- A1c balanced and matched-width IID panel construction;
- delayed, wave-safe, atomic whole-cycle admission;
- a global 25% initialization-work guard;
- a hard total-NN-row budget that includes root, ordinary-leaf and bootstrap
  evaluations;
- evaluator-row accounting independently checked in Python and Rust;
- rotated arm execution order;
- exact deck-8 boundary oracles; and
- schema-v5 per-node admission diagnostics.

The 513-row preflight passed accounting. It was intentionally too small and too
easy to establish strength: every arm selected the exact-best action on all six
seeds. It did establish that current A1c bootstrap evaluation is per node and
therefore adds several small evaluator calls. That is a throughput issue, not an
equal-NN validity issue.

## 3. Scope and non-goals

### In scope

1. A resumable multi-oracle comparison harness.
2. Frozen, versioned experiment inputs and configuration.
3. A one-case schema-v5 GPU smoke.
4. A sealed eight-position, six-arm, equal-NN comparison.
5. Position-clustered analysis and an explicit outcome classification.
6. Conditional cross-node bootstrap batching development if the oracle result
   is sufficiently positive.
7. A fallback corpus-manifest scaffold if A1c is negative.

### Out of scope

- Neural-network training or checkpoint modification.
- A model promotion decision.
- A long head-to-head match.
- `BatchedMCTS` self-play integration before A1c earns it.
- Equal-time or maximum-strength claims from the unoptimized advisor path.
- Tuning against the future 120-position confirmation set.
- Calling any deck>=12 search reference an oracle.
- Additional exact solving unless an existing oracle fails validation.

## 4. Frozen inputs

All paths are relative to the `boardgame-ai-kd` worktree unless shown otherwise.

### 4.1 Network and public positions

- Checkpoint:
  `runs/kingdomino/best_checkpoint/current_best.pt`
- Position source:
  `runs/kingdomino/denial_search/signal_positions.jsonl`

Every output must record the resolved paths and SHA-256 hashes of both inputs.
The resolved manifest freezes the checkpoint hash before the smoke and verifies
it again before every resumed position. The checkpoint stays frozen for every
arm and position even if the `current_best.pt` path later changes.

### 4.2 Exact deck-8 boundary oracles

Use exactly these eight completed summaries, ordered by position index:

| Position | Legal actions | Oracle summary |
|---:|---:|---|
| 5 | 12 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_05_x0/summary.json` |
| 11 | 7 | `runs/kingdomino/chance_correct_a1/deck8_oracle_pos11_x0_line/summary.json` |
| 17 | 9 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_17_x0/summary.json` |
| 23 | 15 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_23_x0/summary.json` |
| 29 | 7 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_29_x0/summary.json` |
| 35 | 2 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_35_x0/summary.json` |
| 41 | 9 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_41_x0/summary.json` |
| 47 | 7 | `runs/kingdomino/chance_correct_a1/deck8_oracle_calibration/position_47_x0/summary.json` |

The suite manifest must record each oracle ID and file hash. Before any GPU
search, validate that every summary:

- has `status == "complete"`;
- has `complete_actions == legal_actions`;
- has 70 solved reveal rows for every legal action;
- matches its position index and stored prefix actions;
- reconstructs the same boundary state from the frozen position source; and
- passes the existing exact-value and actor-frame consistency checks.

Any failure stops the entire sealed run. Do not silently omit a position.

### 4.3 Calibration-set limitation

These boundaries were selected along incumbent/X=0 lines from a small frozen
position set. They are useful because every legal root action has an exact
post-reveal expectation, but they are not an unbiased sample of Kingdomino play.
In particular, they can underrepresent states reached preferentially by a
different search policy, early-game flexibility cases and BGA-specific failure
modes. Report this selection path in the suite artifact and do not extrapolate a
positive result directly to whole-game strength.

## 5. Frozen search design

### 5.1 Arms

Run exactly six arms:

| Arm | Exposure | Backup | Panel mode | Sampling |
|---|---:|---|---|---|
| `x0` | 0 | Hajek | lazy/incumbent | balanced default |
| `x1_hajek_balanced` | 1 | Hajek | lazy | balanced |
| `a1c_x4_balanced` | 4 | sampled panel mean | A1c | balanced |
| `a1c_x4_iid` | 4 | sampled panel mean | A1c | matched-width IID |
| `a1c_x8_balanced` | 8 | sampled panel mean | A1c | balanced |
| `a1c_x8_iid` | 8 | sampled panel mean | A1c | matched-width IID |

The A1c arms force `chance_enum_max_rows=1` at deck=8. The exact 70-row
calculation is the external target and must not leak into a candidate arm.

### 5.2 Shared parameters

- NN evaluation budget per search: `4801` rows.
- Simulation ceiling: `9600` simulations.
- Search repeats: `8` seeds per position.
- `leaf_batch=8` for all arms.
- `fpu=-0.2`.
- `cpuct=1.5`.
- A1c `chance_init_visits=32`.
- A1c `chance_widening_c=0.25`.
- A1c `chance_init_max_fraction=0.25`.
- Device: CUDA on the laptop GPU.
- No root noise.
- Frozen current-best weights for all arms.

This produces 8 positions x 8 seeds x 6 arms = 384 searches and a nominal
maximum of 1,843,584 counted NN rows. A row evaluated inside a GPU batch still
counts as one unit of work.

The simulation ceiling is not a second resource entitlement. It is deliberately
loose so every valid search stops on the 4,801-row NN budget. If any arm reaches
the ceiling with unused NN work, the suite is invalid and stops. The diagnostic
must be examined for ordinary-leaf production versus terminal saturation before
changing the ceiling. Do not keep a partial result as if it were comparable.

### 5.3 Seeds and execution order

Within a position/repeat pair, every arm receives the same search seed. Different
positions receive disjoint deterministic seed blocks. Proposed derivation:

```text
seed(position_ordinal, repeat) = 2026084000 + 100 * position_ordinal + repeat
```

Arm order rotates over the global case index, not from zero independently at
every position:

```text
rotation = (position_ordinal * seed_count + repeat) mod arm_count
```

With 64 cases and six arms, execution slots differ by at most one occurrence per
arm. The suite manifest stores every derived seed and execution order.

## 6. Development deliverables

### 6.1 Suite manifest

Add a committed configuration file, proposed path:

`games/kingdomino/configs/deck8_oracle_a1c_suite_v1.json`

It must contain:

- a schema version;
- ordered oracle-summary paths and expected oracle IDs;
- checkpoint and positions paths;
- all six immutable search specifications;
- NN budget, simulation ceiling and repeat count;
- base seed and seed-derivation rule;
- arm-order rotation rule;
- selection reason; and
- the experiment classification rules from Section 10.

The executable records a resolved snapshot with hashes in the run directory.
It also records the Git commit and hashes of the suite runner, single-position
comparator and Rust search source. A globally clean worktree is not required,
because unrelated local files exist, but the relevant executable source must be
identifiable exactly.

### 6.2 Multi-position runner

Add `games/kingdomino/deck8_oracle_suite_compare.py`.

The runner should reuse `deck8_oracle_compare.py` rather than duplicate search,
oracle or scoring logic. A small refactor is acceptable to share one loaded
network/evaluator across positions, but correctness and resumability are more
important than avoiding eight checkpoint loads.

Required behavior:

1. Validate the complete manifest and all oracle summaries before loading CUDA.
2. Refuse an output directory containing incompatible provenance.
3. Warm the evaluator outside measured arms.
4. Execute one position/repeat at a time with common random seeds across arms.
5. Rotate arm order using the global case index.
6. Atomically persist each completed position artifact.
7. Resume only artifacts whose schema, oracle hash, checkpoint hash, position
   hash and complete search configuration match the current manifest.
8. Fail closed on malformed, partial or mismatched artifacts. Never overwrite
   them automatically.
9. Aggregate only after all eight position artifacts are complete.
10. Atomically write the final suite summary.

Proposed output directory:

`runs/kingdomino/chance_correct_a1/deck8_oracle_a1c_suite_v1/`

Proposed files:

```text
manifest.resolved.json
positions/position_05.json
positions/position_11.json
...
positions/position_47.json
summary.json
```

The runner must support a validation-only mode that performs all CPU-side input
and resume checks without importing or initializing CUDA.

### 6.3 Comparator extension

Extend `deck8_oracle_compare.py` only where needed:

- allow an arm-order rotation offset while preserving the current default;
- expose a callable comparison seam usable by the suite runner;
- preserve schema-v5 per-node admission rows;
- retain the hard no-overshoot and exact-budget checks; and
- keep single-position CLI behavior backward compatible.

Do not change A1c search semantics during this deliverable.

### 6.4 Suite aggregation

The suite summary must retain all raw per-position records by reference and
report, for every arm:

- exact-best selection rate, treating all exact ties as rank one;
- mean exact joint-action regret;
- median and p90 exact regret;
- exact pairwise ordering accuracy;
- mean selected-action exact rank;
- selection agreement across repeated seeds;
- mean and maximum NN evaluations;
- mean simulations completed;
- mean wall time;
- evaluator calls and batch sizes;
- initialization NN rows, calls and fraction;
- reached, initialized and never-initialized chance nodes;
- pre-admission visits and their per-node fractions; and
- simulation-ceiling or unused-budget violations.

For every candidate, report paired deltas against both `x0` and
`x1_hajek_balanced` at the same position and seed. Aggregate the repeated seeds
within each position first, then aggregate the eight position-level values.
Repeated seeds characterize search variance; they are not independent strategic
samples.

Produce position-clustered bootstrap intervals by resampling the eight positions
with replacement and retaining each selected position's complete repeat set.
Use a fixed bootstrap seed and at least 20,000 resamples. Because there are only
eight clusters, label the intervals as calibration diagnostics, not promotion
confidence bounds.

## 7. Required tests before GPU execution

Add focused CPU tests covering:

1. Manifest validation accepts the frozen eight-oracle configuration.
2. Duplicate position IDs or oracle IDs are rejected.
3. Missing, incomplete or hash-mismatched oracle summaries are rejected.
4. Global arm rotation is balanced and deterministic.
5. Common seeds are identical across arms and disjoint across positions.
6. Resume accepts a byte-valid matching position artifact.
7. Resume rejects schema, hash or search-configuration drift.
8. Partial JSON is rejected and not overwritten.
9. Aggregation computes paired deltas after averaging within position.
10. The bootstrap resamples positions, not individual repeats.
11. Exact ties receive rank one and zero regret.
12. Any unmet NN budget invalidates the suite.
13. Controls emit empty A1c node arrays without affecting aggregation.
14. Late and never-initialized A1c nodes survive the suite artifact.
15. Single-position CLI defaults remain unchanged.

Validation commands:

```powershell
C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m pytest `
  games\kingdomino\tests\test_deck8_oracle.py `
  games\kingdomino\tests\test_deck8_oracle_suite.py -q

cargo test a1c_ --lib
cargo fmt --check
git diff --check
```

Project Python/Torch/CUDA commands must follow the repository `AGENTS.md` and
run with the project virtual environment and escalated permission.

## 8. Execution stages

### Stage R0 - CPU validation only

Run the suite in validation-only mode. Confirm all eight oracles, hashes, prefix
actions, exact rows, seeds, arm orders and output targets. Expected time: less
than five minutes.

Stop on any mismatch.

### Stage R1 - schema-v5 GPU smoke

After confirming no other compute process is using the GPU, run one position,
one seed, all six arms at 513 NN rows and a 1,024-simulation ceiling. Use a new
output path and do not overwrite the existing v4 preflight.

The smoke passes only if:

- every arm uses exactly 513 NN rows;
- no arm reaches the simulation ceiling;
- Rust and Python evaluator-row accounting match;
- A1c output contains structured per-node diagnostics;
- initialized plus uninitialized reached nodes reconcile;
- evaluator-call and initialization-call counts are present; and
- the output is valid, atomic schema-v5 JSON.

This smoke validates plumbing only. Its action choices are not evidence.

### Stage R2 - sealed eight-position comparison

Run the frozen suite exactly once. Do not inspect partial action results while it
is active. Health monitoring may check only process existence, elapsed time, GPU
process ownership and whether atomic completed artifacts appear.

Expected laptop runtime is approximately 20-60 minutes. The estimate should be
reported as uncertain because A1c uses extra small evaluator calls and the eight
positions differ in legal-action width.

If interrupted, resume only from verified complete position artifacts. Do not
restart completed positions under a different configuration in the same suite.

### Stage R3 - sealed analysis

After every position completes:

1. Validate all budgets and provenance again.
2. Build the suite summary without rerunning search.
3. Inspect primary metrics and paired deltas.
4. Inspect whether null results coincide with late or refused A1c admission.
5. Inspect evaluator-call fragmentation separately from target quality.
6. Classify the result using Section 10.
7. Update `KINGDOMINO_NEXT_LEVERS.md` with the sealed result and next action.

Do not add seeds after seeing the result. More repeats of these positions cannot
repair a lack of independent strategic coverage.

## 9. Primary and diagnostic metrics

### Primary target-quality metrics

1. Mean exact joint-action regret.
2. Exact-best selection rate.

These are evaluated against the complete 70-row deck-8 oracle. Mean regret is
the main continuous discriminator; exact-best rate prevents a small average
improvement from hiding frequent wrong winners.

### Secondary target-quality metrics

- p90 regret;
- pairwise ordering accuracy;
- mean exact rank;
- position-level win/tie/loss counts for paired regret;
- seed-to-seed action stability; and
- performance separated by exact top-action gap and legal-action count.

### Mechanism diagnostics

- initialization fraction;
- number of reached/initialized/uninitialized chance nodes;
- visits before first admission per node;
- initialized cycles and active rows;
- ordinary versus bootstrap NN rows;
- evaluator calls, mean/max batch size and initialization calls;
- simulations completed and effective search depth where available; and
- wall-clock time.

Mechanism diagnostics explain a result but cannot override a negative exact
oracle result.

## 10. Preregistered result classification

The eight positions are a calibration screen. Use these classifications rather
than treating the run as a promotion gate.

### Target-positive

At least one A1c arm must satisfy all of the following:

1. Lower mean paired exact regret than both `x0` and lazy `X=1`.
2. Exact-best selection rate no lower than both controls.
3. Improvement, rather than regression, on at least one of p90 regret or
   pairwise ordering.
4. A candidate-minus-`x0` mean regret delta at or below zero on a majority of
   the eight positions, counting exact equality as a tie rather than a win.
5. No validity failure and no evidence that the apparent gain is confined to
   searches whose panel never initialized.

A position-clustered 95% upper bound below zero against `x0` is strong evidence
but is not required for the calibration classification because there are only
eight clusters.

### Inconclusive/weak signal

Use this classification when an A1c arm improves the regret point estimate but
loses exact-best rate, metrics disagree, gains are concentrated in one or two
positions, or the clustered interval is broad. Preserve the candidate for the
240-position tuning corpus; do not optimize or integrate it yet.

### Target-negative

Use this classification when no A1c arm lowers mean exact regret versus `x0`
and no A1c arm improves exact-best selection, or when any apparent aggregate
gain is explained by a large regression on multiple independent positions.

### Invalid

The suite is invalid if any input hash drifts, an oracle is incomplete, an arm
does not exhaust exactly 4,801 NN rows, Python/Rust accounting disagrees, an
output is malformed, or arms within a paired case use different seeds or
weights.

The classification must be made before choosing a preferred A1c exposure or
sampling method. Balanced versus IID remains an empirical ablation.

## 11. Conditional development after the result

### 11.1 If target-positive: cross-node bootstrap batching

Current A1c evaluates each admitted node's bootstrap rows in a separate call.
Refactor only the advisor diagnostic path first:

1. Gather all deduplicated admission requests after a wave's backups.
2. Preserve visit-priority and seeded tie-breaking.
3. Select only complete cycles that fit both the 25% initialization guard and
   remaining hard NN budget.
4. Flatten selected rows from all nodes into one evaluator request when
   practical.
5. Scatter returned values back to the owning node/cycle.
6. Commit every selected cycle atomically only after the full evaluator call
   succeeds.
7. Preserve bootstrap values as priors, not MCTS visits.
8. Preserve the exact total-NN accounting and no-overshoot proof.

Add diagnostics for requested nodes, selected nodes, rows per admission wave,
cross-node batch size, evaluator calls saved and rejected cycles.

Required tests:

- at least two nodes admitted in the same wave produce fewer initialization
  calls than committed cycles;
- flattened/scattered values go to the correct node and row;
- deterministic zero-evaluator search is action/visit identical to the
  per-node implementation;
- a cycle never partially commits on budget exhaustion;
- evaluator failure leaves every affected node unmodified;
- initialization fraction remains at or below 25%; and
- disabled/incumbent behavior remains bit-identical.

Then repeat the 513-row smoke and compare call counts and latency to the preserved
v4 artifact. Do not rerun the sealed eight-position target-quality suite merely
because batching got faster; target semantics should be unchanged. Stop before
`BatchedMCTS` integration and request review of the batching commit.

### 11.2 If target-negative: broader corpus groundwork

Do not optimize A1c. Instead, implement the manifest and validation layer for
the roadmap's 240-position corpus:

- 120 tuning and 120 untouched confirmation positions;
- stratification by every reachable bag size;
- ordinary self-play, advisor/BGA losses, flexibility/draft-order and defensive
  blocking tags;
- public-state identity and hidden-order invariance checks;
- immutable split assignment before schedule tuning; and
- explicit provenance for any BGA-derived state.

Only build the machinery and inventory available sources overnight. Do not fill
missing BGA strata with convenient self-play positions, open the confirmation
split during tuning, or start a global high-`X` sweep.

### 11.3 If inconclusive

Stop after writing the result and the corpus scaffold. Do not use throughput
optimization to turn an uncertain target-quality result into a pass.

## 12. Commit and review boundaries

Keep changes reviewable:

1. **Commit A - suite harness and tests.** Manifest, runner, aggregation,
   comparator seam and CPU tests. No search-semantic change.
2. **Run artifact - smoke and sealed suite.** Results stay under `runs/`; update
   the roadmap with hashes, configuration and conclusions.
3. **Commit B - result documentation.** Analysis/reporting fixes and roadmap
   update, if code changes are needed.
4. **Commit C - conditional cross-node batching.** Only after a target-positive
   result; separate from the harness so logic and throughput can be reviewed in
   isolation.

The most important review point is after Commit A and the sealed suite result,
before Commit C. The batching review should focus on atomicity, row ownership,
budget enforcement, estimator semantics and actual evaluator-call reduction.

Do not push automatically unless already authorized for the new commits. Leave
unrelated worktree files untouched.

## 13. Overnight schedule and stopping rules

Approximate schedule:

| Work | Expected time |
|---|---:|
| Suite runner, manifest and aggregation | 45-90 minutes |
| Focused tests and review pass | 15-30 minutes |
| CPU validation and GPU smoke | 5-10 minutes |
| Sealed eight-position GPU suite | 20-60 minutes |
| Analysis and roadmap update | 20-40 minutes |
| Conditional batching prototype and tests | 90-180 minutes |

Total useful work is approximately 3-6 hours. Variance is mainly in batching
development, not the sealed oracle run.

Stop immediately if:

- another compute process owns the GPU;
- frozen input validation fails;
- a completed output has incompatible provenance;
- Rust/Python NN accounting disagrees;
- any arm overshoots or fails to exhaust its budget;
- an evaluator error risks a partially committed panel; or
- the sealed result is negative and the next task would merely optimize the
  rejected mechanism.

At handoff, report:

- commits created and whether they were pushed;
- exact test commands and outcomes;
- smoke and suite artifact paths and hashes;
- per-arm primary metrics and paired deltas;
- mechanism/throughput diagnostics;
- the preregistered result classification; and
- the single recommended next step.
