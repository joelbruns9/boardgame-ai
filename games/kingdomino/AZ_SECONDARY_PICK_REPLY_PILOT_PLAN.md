# Secondary-Pick Opponent-Reply Training Pilot and Search Acceleration Plan

- **Status:** Proposed; requires explicit approval before training
- **Date:** 2026-07-19
- **Scope:** Kingdomino AlphaZero policy fine-tuning, denial-label generation, and offline-search throughput
- **Baseline checkpoint:** `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`
- **Primary evidence:** `runs/kingdomino/denial_search/secondary_seed_test.json`

## Executive decision

The overnight experiment did not produce a clean pre-registered `SYSTEMATIC` verdict, so it does
not authorize a full denial curriculum. It did, however, show a large, persistent, seed-stable
secondary-pick inconsistency that is poorly explained by ordinary root-MCTS sampling noise. The
formal result was `BETWEEN`, which defaults to closing the lever, because one systematic gate
missed narrowly and the supposedly stable searched reference was noisier than root Q.

The recommended next step, if the user explicitly overrides the conservative route, is a small
control-versus-treatment pilot. The treatment teaches the policy head to select strong opponent
refutations at ply 1. It does not directly penalize all secondary picks, alter the value-head
semantics, or upweight low-prior picks. The plan of record is 1,200–1,600 accepted opponent-reply
states, with 2,000 as a hard ceiling, and it cannot update `current_best` automatically.

Offline label search is the dominant compute cost. The implementation therefore stages throughput
work in increasing order of risk:

1. Deterministic position sharding and artifact merge.
2. Multiple CPU search workers feeding one shared GPU evaluator.
3. A serial Rust port of the forced denial tree with strict Python equivalence gates.
4. Deterministic Rayon expansion and backup over tree levels.

The pilot proceeds to training only after the label path passes correctness gates. It proceeds to a
larger curriculum only if held-out fragility improves, the BGA anchor does not regress, and equal-
compute game strength is at least non-inferior.

## 1. Overnight experiment: completed baseline

### 1.1 Question tested

The test separated two explanations for high fragility among policy ranks 2–4:

- **Systematic overvaluation:** root search consistently overrates secondary picks because opponent
  refutations are starved inside those subtrees. The effect should persist with more simulations
  and remain stable across root-search seeds.
- **Sampling noise:** secondary picks receive too few visits, making their root Q estimates noisy.
  The effect should shrink toward zero and become less seed-sensitive with more simulations.

For domino pick `d`, simulation count `s`, and root seed `r`, the measured quantity was:

```text
fragility(d, s, r) = root_Q(d, s, r) - searched_ref(d)
```

`searched_ref(d)` was the median actor-frame value of three independent 8-ply trees. A pick was
secondary when its competition rank under `searched_ref` was at least 2; picks tied for best
remained rank 1.

### 1.2 Frozen inputs and configuration

- Frozen positions: 50 real midgame states.
- Frozen-set SHA-256: `d38ebb5f1e0430f2f78dfaa7998a1cb48ece0584b27ebb42ba8d144737768f9a`.
- Rank-1 picks: 59 across the 50 positions, including searched-value ties for best.
- Secondary picks: 141 across the 50 positions.
- Tree seeds: 3.
- Root seeds: 5.
- Root simulation ladder: 800, 3,200, and 10,000.
- Tree search: 8 pick plies, `chance_k=16`, opponent placement top-2.
- Tie tolerance: `1e-6`.
- Phase-0 factored-root equivalence: passed byte-identically for every pick on the gate position.
- Reserved test split: not opened.

### 1.3 Runtime

| Phase | Work | GPU/wall hours |
|---|---|---:|
| Phase 0 | Equivalence gate | 0.030 |
| Phase 1 | 150 reference trees | 5.904 |
| Phase 2 | 750 root-ladder cells | 0.499 |
| Phase 3 | 15 `chance_k=32` tie probes | 1.088 |
| **Total** | Full experiment | **7.52** |

The machine used an RTX 3070 Laptop GPU with 8 GB VRAM. The process frequently showed low GPU
utilization during serial Python tree expansion, and CPU time closely tracked wall time.

### 1.4 Primary results

The median below pools available secondary-pick/root-seed observations. There were at most
`141 × 5 = 705` observations per rung; lower-simulation searches sometimes did not visit a
representative action and therefore emitted no Q for that cell.

| Root sims | Observations | Median fragility | p90 fragility | Median root-Q seed SD | Positions with ≥4/5 stable flips |
|---:|---:|---:|---:|---:|---:|
| 800 | 630 | 0.1701 | 0.4245 | 0.00187 | 20 |
| 3,200 | 685 | 0.1514 | 0.4167 | 0.00266 | 18 |
| 10,000 | 695 | 0.1476 | 0.3616 | 0.00308 | 12 |

The increase in the raw root-Q seed SD (`0.00187 → 0.00308`) should not be read as evidence that
more simulations destabilize Q. Observation coverage rises from 630 to 695 cells, and the number of
secondary picks observed under all five seeds rises from 126 to 139. Previously unvisited hard
picks contribute zero/no seed variation until they enter the sample. The complete-pick median SD
also rises modestly (`0.00206 → 0.00310`), so composition explains part, but not necessarily all,
of the trend; either way, every value remains tiny relative to the fragility effect.

Additional results:

- Median per-secondary-pick slope from 3,200 to 10,000: `-0.01137`.
- Positions with a ≥4/5 flip at both 3,200 and 10,000: 10.
- Median 3,200-sim value at risk among those persistent positions: `0.09816`.
- Tie-guard-killed would-be flips: 28 at 800, 24 at 3,200, and 20 at 10,000.
- Secondary searched-reference seed SD, median: `0.00881`.
- All-pick searched-reference seed SD, median: `0.00850`.
- All-pick root-Q seed SD at 3,200, median: `0.00277`.
- Picks where searched SD was not lower than root-Q SD: 161 of 200.
- Factoring/reference-stability warning: **triggered**.

#### Rank-1 method-offset baseline

The original report conditioned fragility on secondary picks. A zero-search-compute reanalysis of
the same artifacts now measures rank-1 picks as a method-offset control. Since root MCTS Q is a
visit-weighted estimate while the forced 8-ply tree backs values up under different search
semantics, rank-1 fragility estimates the common offset between the two estimators.

| Root sims | Rank-1 observations | Rank-1 median | Secondary median | Secondary-minus-rank-1 median | Rank-1 p90 | Secondary-minus-rank-1 p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 295 | -0.0007 | 0.1701 | 0.1708 | 0.1884 | 0.2361 |
| 3,200 | 295 | 0.0158 | 0.1514 | 0.1356 | 0.1627 | 0.2540 |
| 10,000 | 295 | 0.0228 | 0.1476 | 0.1248 | 0.1547 | 0.2069 |

The rank-1 offset is small at the two decision rungs. It does **not** explain most of the secondary
fragility: the secondary-specific median excess remains `0.1356` at 3,200 and `0.1248` at 10,000.
This strengthens the interpretation that the effect is rank-specific, while still not proving that
the forced-tree teacher is objectively correct. The generated report records this analysis under
`rank_conditioned_fragility` and preserves the original pre-registered routing result unchanged.

### 1.5 Tie side-probe

| Metric | `chance_k=16` | `chance_k=32` |
|---|---:|---:|
| Probe positions with a bit-identical tie | 13/15 | 13/15 |
| Total tied pick pairs | 13 | 13 |

Three `k=16` ties dissolved at `k=32`, but three new ties appeared. Raising `chance_k` did not
reduce the aggregate tie count. Most tie degeneracy therefore looks structural or horizon-related,
not like a problem that a blanket `chance_k=32` increase will solve.

### 1.6 Pre-registered routing result

| Systematic gate | Threshold | Observed | Result |
|---|---:|---:|---|
| High-sim median fragility | ≥0.08 | 0.1476 | Pass |
| Median absolute high-minus-3,200 slope | ≤0.02 | 0.01137 | Pass |
| Median root-Q seed SD at 3,200 | <0.05 | 0.00266 | Pass |
| Persistent ≥4/5 flip positions | ≥8 | 10 | Pass |
| Persistent median value at risk | ≥0.10 | 0.09816 | **Fail** |

No pre-registered sampling-noise condition fired: fragility did not approach zero, the stable-flip
count did not halve, and root-Q seed SD was not comparable to mean fragility. Nevertheless, missing
one systematic gate produced the formal classification:

```text
BETWEEN → default to NOISE / close the curriculum lever
```

## 2. Interpretation

### 2.1 What the experiment supports

The simple sampling-noise explanation is weak:

- Median fragility remained near 0.15 at 10,000 simulations.
- Almost all of the median improvement occurred by 3,200 simulations; the curve then plateaued.
- Root-Q seed variation was approximately two orders of magnitude smaller than the median effect.
- Ten of 50 positions retained a seed-stable move disagreement at both 3,200 and 10,000 sims.
- Rank-1 median fragility was only `0.0158` at 3,200 and `0.0228` at 10,000, leaving a
  secondary-specific excess of `0.1356` and `0.1248`, respectively.

The observed root search is therefore consistently optimistic about many secondary picks. Extra
root simulations improve the upper tail and eliminate some move flips, but do not remove the
central discrepancy. The rank-1 control makes a generic root-Q-versus-forced-tree method offset an
unlikely explanation for most of that discrepancy.

### 2.2 What the experiment does not establish

The test is net-internal and does not prove that the deeper searched choice is objectively better.
Its teacher uses the same network's policy and value heads. In addition:

- The searched reference was less seed-stable than root Q, violating the intended factoring
  assumption.
- Exact tie degeneracy persisted at higher `chance_k`.
- The persistent median value-at-risk gate missed by `0.00184`.
- The clean BGA anchor covers on-path human picks and cannot validate off-path refutations.

Training could reduce measured fragility merely by teaching the student to imitate a noisy teacher.
It could also reduce absolute fragility by deflating root Q globally, including rank-1 picks,
without improving decisions. The pilot therefore targets the secondary-minus-rank-1 excess and
includes explicit rank-1/root-Q drift guards rather than accepting an absolute fragility decrease
by itself.
An internal-metric improvement without external policy retention and game-strength improvement is
not a success.

### 2.3 Working conclusion

The result is strong enough to justify a tightly bounded pilot, but not a full curriculum. The
pilot's purpose is to answer a new question:

> Does teaching the opponent's missed ply-1 refutations reduce held-out root overvaluation and
> improve or preserve equal-compute playing strength?

## 3. Pilot hypothesis and invariants

### 3.1 Mechanism

The forced 8-ply tree already computes 7-ply-backed values for every opponent pick at the ply-1
state reached after each candidate root pick. Those values are currently discarded. The treatment
will emit a policy target over the opponent's pick groups at that state.

After fine-tuning, ordinary MCTS should assign more prior to strong opponent refutations, explore
them sooner, lower the Q of vulnerable root picks, and stop overplaying those picks. The model learns
the refutation pattern; it does not receive a hard-coded penalty for being policy rank 2 or 3.

### 3.2 Locked invariants

- Start from the current-best checkpoint identified above.
- Do not change value-head targets, frames, or semantics.
- Do not upweight a root pick merely because its prior is low.
- Do not train on the frozen 50 or the reserved test split.
- Do not overwrite `current_best` automatically.
- Keep control and treatment at equal optimizer steps, batches, replay mixture, and evaluation
  compute.
- Preserve the original Python denial search as the correctness oracle until the Rust path passes
  every equivalence gate.

## 4. Reply-label construction

### 4.1 Training unit

One accepted example contains:

- Encoded ply-1 opponent state.
- Legal action indices and their pick-group mapping.
- Pick IDs.
- Backed actor-frame value and standard error for each pick.
- `denial_policy_target` probability for each pick.
- Parent root pick, parent raw prior, parent searched rank, and parent fragility.
- Tree seed, chance configuration, state key, checkpoint hash, and source provenance.
- Quality flags for exact ties, top-two margin, target entropy, and cross-seed agreement.

The official self-play/replay schema remains unchanged. Reply labels live in a separate artifact and
enter training through an auxiliary grouped-pick loss.

### 4.2 Target calculation

At a ply-1 state, group complete legal actions by selected domino. Retain the existing placement
delegation semantics within each group, convert backed player-0 values into the ply-1 actor frame,
and call the existing uncertainty-aware `denial_policy_target`.

For the network's complete-action probabilities `p(a|s)`, define its probability for pick `d` as:

```text
P(d | s) = sum of p(a | s) over legal actions a whose selected domino is d
```

The reply loss is cross-entropy over these grouped probabilities:

```text
L_reply = -sum_d target(d | s) * log P(d | s)
```

This allows the auxiliary target to train the existing full action policy head without inventing a
new deployment head.

### 4.3 Data volume and split

- Generate 400–500 fresh training-only root positions.
- Each root can yield up to four ply-1 reply states.
- Plan for approximately 1,200–1,600 accepted reply examples after ambiguity filtering.
- Treat 2,000 as a storage/training ceiling, not an acceptance target; 400–500 roots cannot produce
  more than 1,600–2,000 pre-filter candidates in the first place.
- Hold out a fresh, disjoint reply-label validation shard for loss/target diagnostics.
- Retain the original frozen 50 exclusively for the pre/post search-behavior comparison.

### 4.4 Label-quality filter

Because searched-reference noise exceeded root-Q noise, label quality is a first-class gate.

Initial generation uses `chance_k=16` and one tree seed. Additional seeds or `chance_k=32` are
computed selectively for ambiguous examples rather than for the whole dataset. An example is
ambiguous when it has an exact/near tie, a small top-two backed-value margin, high Monte Carlo
standard error, unstable parent fragility, or high target entropy.

Before the production generation run, use a training-only calibration shard to lock numeric
thresholds for:

- Minimum top-two reply-value margin.
- Maximum searched-value seed SD.
- Maximum acceptable target entropy.
- Required cross-seed top-pick agreement.

After those thresholds are locked, do not tune them on the frozen 50. Drop ambiguous labels that do
not stabilize; do not try to rescue the whole dataset with blanket `chance_k=32`.

## 5. Fine-tuning experiment

### 5.1 Arms

Run two arms from identical copies of current-best:

- **Control:** ordinary replay fine-tuning only.
- **Treatment:** the same ordinary replay batches and optimizer steps plus reply examples and
  `L_reply`.

Use fixed random seeds and record exact sampled indices so the only intended difference is the reply
loss. Neither arm participates in automatic promotion.

### 5.2 Loss and mixture

Use the existing AlphaZero loss unchanged for ordinary samples. For treatment batches:

```text
L_total = L_existing_AZ + lambda_reply * L_reply
```

Start conservatively:

- Reply examples: 10–20% of training samples or steps.
- Initial `lambda_reply`: 0.10–0.20.
- Short fixed-step fine-tune from current-best.
- Existing optimizer hardening, finite-loss checks, and gradient clipping remain enabled.

Select one treatment setting on the training-only reply validation shard. Do not grid-search against
the frozen 50 or game-strength suite.

The grouped loss intentionally constrains only total pick mass. For each legal pick group, monitor
the conditional placement distribution

```text
q(a | d, s) = p(a | s) / P(d | s)
```

at reply states. Report the median and p90 within-group placement entropy and
`KL(q_arm || q_baseline)` for both control and treatment, plus their deltas. This detects accidental
redistribution among placements even though reply states contain no placement target. It is a
diagnostic, not a new loss term; the low reply mixture and low `lambda_reply` remain the primary
containment measures.

### 5.3 Pilot compute envelope

Fine-tuning should take less than 1–2 GPU-hours. Search-label generation and evaluation dominate.
Using the overnight Phase-1 measurement:

```text
5.904 hours / 150 trees = 2.36 minutes per root tree per seed
```

At the local baseline rate, 500 roots require about 19.7 hours for one seed or 59 hours for three
seeds. Selective confirmation should keep the pilot closer to the one-seed cost than the three-seed
cost.

At the quoted cloud prices, before accounting for the much faster cloud GPU/CPU:

| Workload | RTX 5080 at $0.20/h | RTX 5090 at $0.36/h |
|---|---:|---:|
| 25–35 hour fast pilot | $5–$7 | $9–$12.60 |
| 65–80 hour robust pilot | $13–$16 | $23.40–$28.80 |

The 5090 must be at least `0.36 / 0.20 = 1.8×` faster end to end to beat the 5080 on cost. Select the
box by measured dollars per accepted label, not nominal GPU specifications.

## 6. Pre-registered evaluation and routing

### 6.1 Internal search-behavior gates

Evaluate control and treatment on the unchanged frozen 50 with the same checkpoint-independent
position set, seeds, and simulation ladder. Use fixed baseline searched references for the primary
comparison so the target does not move with the student. A secondary analysis may rebuild student
trees but must be reported separately.

Treatment must beat control on all primary behavior gates:

- Reduce the median secondary-minus-rank-1 fragility excess at 3,200 by at least 20%. The overnight
  reference is `0.1356`, corresponding to `0.1085` or lower if control reproduces baseline exactly;
  the actual gate is treatment versus the equal-training control.
- Reduce the p90 secondary-minus-rank-1 fragility excess at 3,200 by at least 10%. The overnight
  reference is `0.2540`, corresponding to `0.2286` or lower if control reproduces baseline exactly.
- Also report the original absolute secondary metrics (`0.1514` median and `0.4167` p90), but do not
  allow a global shift in Q to satisfy the rank-specific gates.
- Reduce ≥4/5 stable flips at 3,200 from 18 to at most 14.
- Reduce the 3,200-to-10,000 persistent stable-flip count from 10 to at most 8.
- Do not materially increase root-Q seed SD, missing-Q rate, or tie-guard dependence.

Global-Q-deflation guard, required before spending the game-evaluation budget:

- Treatment rank-1 median fragility at 3,200 must remain within `±0.02` of control.
- Treatment mean rank-1 root Q across the common frozen pick/seed cells must remain within `±0.02`
  of control.
- Report the mean root-Q shift over all common cells as a diagnostic. A broad pessimistic shift is
  not evidence that the reply labels fixed secondary-pick evaluation.

Run these cheap guards first. If either rank-1 guard fails, classify the treatment as a behavior
failure and do not spend the full BGA/game-strength evaluation budget on it.

The primary evaluation reuses the already frozen searched references; it reruns only the root sim
ladder. The overnight root ladder cost about `0.50` hours total, so budget roughly `0.5` GPU-hours
per arm on the original machine, not a Phase-1-scale forced-tree rebuild. Rebuilding student trees
is optional secondary analysis with a separate, explicitly approved budget.

These thresholds are pilot gates, not claims that the fixed teacher is objectively correct.

### 6.2 External and strength gates

- Re-run the BGA anchor. Preserve at least 70% exact agreement on the top-30 human picks and a
  median human-pick prior of at least 0.77, versus the observed baseline of 76% and 0.82.
- Run control versus treatment at equal search compute, paired seeds, and both seats.
- Run the repository's unchanged promotion/fixed-suite gates before any candidate can replace
  current-best.
- Report score margin and confidence intervals, not only win-rate point estimates.

An internal fragility improvement accompanied by a BGA or game-strength regression is a failed
pilot.

### 6.3 Routing

- **PASS:** behavior gates pass, BGA is non-inferior, and treatment is strength-non-inferior with a
  positive strength signal. Proceed to a larger reply curriculum and a second held-out study.
- **MEASUREMENT-ONLY:** fragility improves but strength is flat/uncertain. Do not promote or scale;
  retain the artifact for analysis and consider one replication.
- **FAIL:** behavior gates miss, BGA regresses, or equal-compute strength regresses. Close the lever
  and keep current-best.

## 7. Throughput optimization plan

### 7.1 Baseline bottleneck

The current harness iterates positions and seeds serially. The open-loop root search is implemented
in Rust and releases the Python GIL, but its simulation loop is serial; `leaf_batch=8` batches
inference rather than using eight CPU threads. The forced tree's level expansion, transposition
construction, and backup are serial Python. Raising `RAYON_NUM_THREADS` does not accelerate this
path today.

### 7.2 Stage A — deterministic sharding

Add first-class sharding to the new reply-label generator:

```text
--num-shards N --shard-index I --shard-name NAME
```

Requirements:

- Partition by frozen position index, never by completion order.
- Write one manifest and JSONL artifact per shard.
- Include checkpoint, input-set, config, seed, and code-version hashes.
- Resume each shard independently.
- Merge only when indices are complete, unique, and provenance-identical.
- Prove that a merged N-shard run equals a serial run after canonical sorting.

This enables multiple cloud boxes immediately. Two 5080 boxes should approximately halve wall time
at roughly unchanged total GPU rental cost, without any shared-GPU concurrency risk.

### 7.3 Stage B — multiple search workers, one GPU evaluator

On a single box, run independent CPU tree workers and centralize network inference:

```text
CPU worker 0 ─┐
CPU worker 1 ─┼─> shared inference queue ─> one GPU model ─> responses
CPU worker N ─┘
```

Reuse the repository's inference-service/batched-self-play patterns where practical. The service
should combine policy and leaf-value requests across active trees into large forwards. Search
workers must never mutate another worker's tree or cache.

Benchmark worker counts 1, 2, and 4 before attempting more. Record:

- Accepted labels/hour and trees/hour.
- Mean and p90 GPU batch size.
- GPU utilization and VRAM.
- CPU utilization by worker.
- Peak RAM.
- Cache hit rate.
- Dollars per accepted label.

Stop increasing workers when throughput falls, GPU latency dominates, or memory/caching costs erase
the gain.

### 7.4 Stage C — serial Rust forced tree

Port the validated Python tree to a new Rust API without changing `DenialSearch` semantics. Use a
flat arena of nodes and edges indexed by integers. Keep the algorithm level-synchronous:

1. Gather all live nodes at one depth.
2. Batch-evaluate their policies through one Python/PyTorch callback.
3. Expand legal pick/placement edges in Rust.
4. Apply deterministic chance rows and build the next level.
5. Batch-evaluate all leaves.
6. Back up from deepest level to root.
7. Emit root and ply-1 grouped-pick values, errors, and provenance.

Reuse existing Rust components for `GameState`, legal actions, action encoding, network encoding,
and chance/deck handling. The serial Rust version must pass correctness before any Rayon work begins.

### 7.5 Stage D — deterministic Rayon expansion and backup

Parallelize only independence-safe work:

- Expand nodes at the same depth using `par_iter` into thread-local edge/child buffers.
- Merge thread-local results in canonical parent/action/chance order.
- Deduplicate children through a deterministic merge or sharded table with canonical final IDs.
- Back up nodes at the same reverse depth with `par_iter` after all children are final.
- Preserve per-edge floating-point summation order; do not use nondeterministic parallel reductions.

The shared transposition table is the main design risk. Prefer thread-local generation plus a stable
merge over fine-grained concurrent mutation. Exact tie behavior is part of correctness, not merely a
diagnostic.

### 7.6 Cloud benchmark matrix

Run a fixed 10–20-position training-only benchmark on the candidate rental hardware:

| Dimension | Values |
|---|---|
| GPU | RTX 5080, RTX 5090 if available |
| Search workers | 1, 2, 4 |
| Rayon threads per worker | 1, 2, 4, 8, 16 subject to CPU count |
| Tree path | Python reference, serial Rust, Rayon Rust |
| `chance_k` | 16 |

Use 8–16 fast modern CPU cores, 32 GB RAM minimum, and 64 GB preferred. Optimize for cost per
accepted label subject to a wall-time target. The Rust/Rayon path ships only if it is label-equivalent
and improves end-to-end trees/hour by at least 2× or reduces cost per accepted label by at least 35%.

## 8. Correctness and test plan

### 8.1 Reply-label tests

- Pick-group probabilities sum to one over legal picks.
- Complete legal actions map to exactly one pick group.
- Actor-frame conversion is correct for both players.
- Per-pick placement delegation matches the validated tree.
- Uncertain/exact ties are filtered or share target mass as registered.
- Low prior alone never causes an upweight label.
- Training, reply-validation, frozen-50, and reserved-test state keys are disjoint.

### 8.2 Sharding tests

- Serial and merged-shard artifacts are identical after canonical sorting.
- Duplicate, missing, out-of-range, or provenance-mismatched rows fail loudly.
- Interrupted shards resume without duplicating labels.
- Worker count and completion order do not change labels.

### 8.3 Rust equivalence tests

On fixed states covering pre-chance, chance-crossing, forced-pick, and tie cases, compare:

- Legal pick IDs and representative action indices.
- Chance rows and weights.
- Node, edge, leaf, and transposition counts.
- Leaf state keys.
- Per-edge player-0 and actor-frame values.
- Monte Carlo standard errors.
- Root and ply-1 policy targets.
- Corrected best pick, exact tied pairs, and tie-guarded flips.

Require bit identity for discrete structure and serialized float outputs where summation order is
preserved. Any relaxed tolerance must be explicit, at most `1e-12`, and must not alter a target,
argmax, tie, or filter decision.

### 8.4 Parallel determinism tests

- Rayon thread counts 1, 2, 4, 8, and 16 emit identical artifacts.
- Search-worker counts 1, 2, and 4 emit identical artifacts.
- Repeated runs with the same seeds are byte-identical apart from elapsed time.
- Threading never changes cache namespace, root representative, or chance CRN behavior.

## 9. Work breakdown and deliverables

### Milestone 0 — baseline archive (complete)

- Preserve the overnight report and JSONL artifacts.
- Record the exact checkpoint and frozen-set hashes in every future manifest.
- Treat the frozen 50 as closed evaluation data.
- Compute the zero-search rank-1 fragility baseline from the existing artifacts and record
  secondary-minus-rank-1 excess at every simulation rung. **Complete:** the 3,200/10,000 median
  excess is `0.1356`/`0.1248`.

### Milestone 1 — Python reply-label oracle

- Add a separate reply-label emitter that exposes ply-1 backed pick edges from the validated tree.
- Define the standalone reply artifact schema and quality flags.
- Add grouped-pick target and actor-frame tests.
- Run a 10-position smoke set and manually inspect labels.

**Gate:** no training until labels are normalized, frame-correct, provenance-complete, and free of
frozen-set leakage.

### Milestone 2 — sharded cloud generator

- Add deterministic shard arguments, manifests, merge, and resume.
- Benchmark one and two boxes/processes.
- Implement the shared GPU inference service only if one-process GPU utilization remains poor.

**Gate:** merged output equals the serial oracle and throughput/cost improves.

### Milestone 3 — Rust serial tree

- Implement the flat Rust arena and batched evaluator boundary.
- Emit both root summaries and ply-1 reply targets.
- Pass the full Python/Rust equivalence suite.

**Gate:** zero semantic divergence and a meaningful single-thread throughput gain.

### Milestone 4 — Rayon tree

- Parallelize level expansion and reverse-depth backup.
- Add deterministic transposition merge and thread-count equivalence tests.
- Benchmark 1/2/4/8/16 threads and 1/2/4 workers on the rental box.

**Gate:** at least 2× end-to-end throughput or 35% lower dollars per accepted label versus the
Python serial baseline.

### Milestone 5 — pilot dataset

- Generate 400–500 fresh root trees.
- Selectively confirm ambiguous labels with extra seeds or `chance_k=32`.
- Filter, deduplicate, and freeze an expected 1,200–1,600 accepted reply examples, with 2,000 as a
  hard ceiling rather than a target.
- Publish dataset statistics: acceptance rate, margins, entropy, seed stability, ranks, and sources.

**Gate:** accepted labels meet the locked quality thresholds and dataset splits are disjoint.

### Milestone 6 — equal-compute control/treatment fine-tune

- Snapshot current-best into control and treatment arms.
- Use identical ordinary replay batches and fixed optimizer steps.
- Add the grouped-pick auxiliary loss only to treatment.
- Monitor within-group placement entropy and KL-to-baseline for both arms at reply states; do not
  add placement-target loss machinery in this pilot.
- Persist every training argument, RNG seed, batch index, and checkpoint hash.

**Gate:** finite/stable training, no abnormal gradient norms, treatment improves reply-validation
loss without degrading ordinary holdout losses materially, and placement-distribution drift is
reported before held-out search evaluation.

### Milestone 7 — held-out evaluation and final route

- Re-run the frozen-50 root sim ladder for control and treatment against the existing fixed searched
  references (about 0.5 GPU-hours per arm on the overnight machine; no forced-tree rebuild).
- Apply the rank-1/global-Q guard before the expensive external and game-strength evaluations.
- Re-run the BGA anchor.
- Run paired, equal-compute control-versus-treatment games and unchanged promotion suites.
- Produce one report containing all internal, external, strength, cost, and throughput metrics.

**Gate:** apply the PASS / MEASUREMENT-ONLY / FAIL routing in Section 6.3. No automatic promotion.

### 9.1 Implementation readiness record (2026-07-19)

The local pre-rental implementation is complete:

- Python oracle extraction, self-contained float16 reply-state artifacts, deterministic shards,
  append/resume, strict manifest/hash merging, accepted-only filtering, and root-disjoint splits.
- A release Rust forced tree with deterministic Rayon planning/merge and bounded GPU-evaluator
  batches. The full 8-ply, `chance_k=16` benchmark measured `111.56 s/tree` in Python versus
  `6.29`, `5.97`, and `5.86 s/tree` at 1/2/4 Rust threads: `17.7â€“19.0Ã—` faster. Maximum backed
  value, target, and standard-error deltas were below `3e-9`, with zero action mismatches.
- Three-seed fixed-reply confirmation. The primary seed must exactly reproduce the reply embedded
  in the parent tree; additional CRN seeds record maximum searched-value seed SD and top-pick
  agreement. Production confirms only candidates that first pass the cheaper margin, within-tree
  error, entropy, and tie filters.
- Equal-step control/treatment training with exact ordinary replay indices/D4 transforms, grouped
  pick-mass cross-entropy, a fixed ordinary holdout, reply validation, placement entropy, and
  KL-to-generation-baseline. Saved checkpoints retain their original architecture and never update
  `current_best`.
- Frozen-reference control/treatment evaluation with rank-specific excess-fragility gates,
  stable-flip gates, rank-1 fragility/mean-Q anti-deflation guards, missing-Q/tie/seed-SD checks,
  and an automatic stop-before-strength route on failure.

A two-root artifact pipeline and a two-step CUDA control/treatment run completed end to end. Both
generated 80x6 checkpoints reloaded successfully; the training report recorded
`current_best_updated=false`. A final production-shape calibration root (`8` plies, `chance_k=16`,
three confirmation seeds) completed in `30.20 s`; its four reply labels had maximum searched-value
seed SD near `0.010`, and the agreement diagnostic correctly separated one `2/3` top-pick case from
three `3/3` cases. The executable cloud sequence is in `REPLY_PILOT_CLOUD_RUN.md`.

## 10. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Noisy/self-referential searched teacher | Fragility falls without strength gain | Filter unstable labels; require BGA and game gates |
| Global value/Q deflation | Absolute fragility falls without better decisions | Gate secondary-minus-rank-1 excess and rank-1 Q stability before game evaluation |
| Structural exact ties | Arbitrary targets and flip instability | Preserve ties, share mass, or drop ambiguous examples |
| Parallel floating-point reordering | Labels change with thread count | Canonical merge and fixed reduction order |
| Shared transposition contention | Poor Rayon scaling | Thread-local expansion plus deterministic merge |
| Multiple CUDA contexts | Higher VRAM and lower throughput | One shared inference service; benchmark worker count |
| Sharding loses cross-position cache reuse | Less-than-linear scaling | Measure cache hits and dollars per accepted label |
| Auxiliary loss overwhelms normal policy | On-path/BGA regression | Low reply mixture and loss weight; control arm |
| Grouped loss moves placement mass | Reply picks improve while placements drift | Monitor within-group entropy and KL-to-baseline for both arms |
| Frozen-set leakage | Invalid evaluation | Hash/state-key disjointness gate |
| Optimizing internal metrics only | No practical value | Equal-compute games and unchanged promotion gates |

## 11. Final decision boundary

This plan deliberately separates evidence that the inconsistency is real from evidence that the
proposed remedy improves play. The overnight run supplies the former only partially. The pilot must
supply the latter.

Do not scale beyond the 1,200–1,600-example pilot (and never beyond the 2,000-example hard ceiling
within this pilot) unless treatment:

1. Passes the rank-specific held-out behavior gates and both anti-deflation guards.
2. Preserves the external BGA anchor.
3. Is at least non-inferior in equal-compute games with a positive strength signal.
4. Passes the unchanged promotion/fixed-suite checks.
5. Produces labels deterministically across the selected worker/thread configuration.

If any external or strength gate fails, close the curriculum lever regardless of the measured
fragility reduction.
