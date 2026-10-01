# Review request — S2 replay window, Rust loader, league, and promotion

This request covers the lifetime of S2 evidence and models: selecting immutable
positions across iterations, sampling a fixed optimizer budget, preserving a
stable holdout, comparing candidate and incumbent on paired games, and atomically
installing only a passing candidate. The framework follows 7WD's growing replay
window and fixed-step training controller, adapted to Welcome To's WTS shards and
multiplayer endpoint.

## 1. Scope

| Area | Status | Main files |
|---|---|---|
| base S2 optimizer/evaluation/checkpoint | committed `51beba6` | `s2_train.py`, `tests/test_s2_train.py` |
| durable growing replay window and manifest | reviewed/remediated | `s2_replay.py`, `tests/test_s2_replay.py` |
| fixed-step uniform random training and stable holdout | reviewed/remediated | `s2_train.py`, `self_play.py`, tests |
| bulk Rust WTS index/decode loader | reviewed/accepted | `welcome_to_rust/src/samples.rs`, Python bridge and tests |
| paired candidate/incumbent gate and atomic install | reviewed/remediated | `s2_promotion.py`, `tests/test_s2_promotion.py` |
| bounded current/recent/HOF opponent pool | reviewed/accepted; throughput tuning deferred | `s2_league.py`, `self_play.py` league-selection slice, tests |

Plan-target schema and legacy head/optimizer expansion are reviewed separately
in `PLAN_SIGNAL_REVIEW_REQUEST.md`. Generation/search behavior is out of scope.

## 2. Intended lifecycle

1. Every completed iteration leaves immutable `.wts` shards plus generation
   metrics and a generation manifest under `iter_NNNN`.
2. `GrowingReplayWindow` selects newest whole iterations until its games target
   is met, bounded by floor and cap. The exact selection and manifest hashes are
   written beside the candidate.
3. Validation is assigned by a stable hash of iteration, game seed, fraction,
   and salt. Changing window membership or input order does not move an existing
   game between train and validation.
4. Training performs exactly `train_steps` full minibatches. Rows are sampled
   uniformly with replacement from the train positions; work does not scale
   accidentally with buffer size.
5. The paired gate gives candidate and incumbent the same seeds, seat schedule,
   frozen incumbent opponents, deterministic temperature, and search settings.
6. Promotion requires the lower confidence bound on paired
   `own score - best opponent score` improvement to exceed zero. The secondary
   endpoint rejects only when its confidence interval establishes a normalized-
   rank regression beyond tolerance.
7. Promotion first writes a durable intent containing pinned candidate and
   incumbent hashes. Archive, install, and league registration are idempotent;
   resume finishes an interrupted intent without rerunning the gate. A rejection
   never changes `current_best`.
8. Future generation samples actual opponents from current best, recent promoted
   archives, and a deterministic bounded HOF sample, with history ramped in over
   early promotions.

## 3. What is already gated

* Replay discovery assembles multiple iterations, checks selected game/position
  counts against shards, rejects duplicate seeds, records window metrics, and
  drives five fixed Rust random batches.
* The Rust loader's sequential full-corpus decode is array-equal to the Python
  row oracle and deterministic under both shuffled and random resets.
* The train/validation split is by complete game, stable under input reversal,
  and carries a directly tested hash function.
* Checkpoint continuation preserves Adam state and monotone optimizer/run
  counters; requested learning rate and weight decay replace stale saved values.
* Identical candidate/incumbent networks produce an exact paired null and reject.
* Promotion tests cover passing atomic replace/archive/league registration and a
  rejection that leaves current best untouched.
* League tests cover bounded recent/HOF selection, deterministic selection,
  early-history ramp, missing/changed archive rejection, and strict manifest ABI.

## 4. Review focus

### 4.1 What is the replay clock?

`assemble_replay` uses the sum of games in every discovered iteration through
the requested point as `total_games`, then selects whole iterations newest first.
It permits numbering gaps. Confirm this is the intended cumulative clock when a
run directory is copied, pruned, or partially restored. If deleted old
iterations should not make the power-law window shrink, a separate durable game
clock is required.

### 4.2 Manifest and shard integrity

Generation-manifest identity is checked across selected iterations and its hash
is recorded. Shards are structurally indexed and their declared game/position
counts are checked, but shard content hashes are not written into the replay
manifest. Decide whether atomic WTS files plus structural validation is enough
for reproducibility, or whether the exact shard hashes belong in the candidate
ledger.

Also check that a mutable metrics JSON cannot select a plausible but unintended
window: game and position totals are cross-checked, but those numbers still
drive the games clock before content is read.

### 4.3 Uniform position sampling

The Rust loader indexes only rows belonging to the Python-selected train game
seeds and draws row indices with replacement. Verify that every indexed position
has exactly equal probability, modulo RNG range sampling, independent of shard,
game length, iteration, file order, and final partial batch. Confirm duplicate
paths/seeds and missing requested seeds fail rather than silently changing the
distribution.

### 4.4 Stable validation versus newest-iteration diagnostics

Validation assignment includes iteration and seed, so it remains stable as the
window grows. Review the small-corpus fallback that forces non-empty train and
validation sets, and confirm it does not make production membership depend on
input order. `pretrain_newest_metrics` evaluates all newest games, including
training games; it is an adaptation diagnostic, not a held-out metric, and must
not be read as one.

### 4.5 Update/reuse budget

`train_steps × batch_size / newest_positions` is reported as reuse of newly
generated evidence, while `training_samples / train_positions` reports passes
over the full buffer. Neither automatically targets 5×. Confirm these are the
right denominators and decide which metric the run controller should hold near
the historical 5× target.

### 4.6 Paired statistical endpoint

The primary endpoint is a fixed-N normal confidence interval over paired margin
deltas. Normalized rank is only a mean non-regression check, not its own
confidence-bound check. Review tie handling, multiplayer seat normalization,
the use of best-opponent margin, the 300-game default, and whether a fixed-N
normal interval is adequate for the observed heavy-tailed score distribution.

The candidate and incumbent arms run sequentially. Determinism should remove
machine drift, but check that seed, seat, opponent assignment, search, and action
temperature are truly common across arms and that no mutable scheduler or model
state crosses from the first arm to the second.

### 4.7 Atomicity across archive, league, install, and report

The operations are individually atomic but not one filesystem transaction.
Review failure after archiving, after league registration, after replacing
`current_best`, and before writing the report. Decide what the resume/reconcile
procedure is for each partial state. Concurrent promotion or league writers are
not locked; confirm single-controller ownership is an acceptable contract.

### 4.8 League weighting and memory bound

Missing category mass stays on current best; recent entries receive linear
recency weights; older HOF entries are sampled deterministically by iteration.
Confirm that duplicate current-best hashes are excluded, history ramp uses the
right count, weights remain normalized for zero-sized categories, and the number
of simultaneously loaded models is acceptable on the target GPU.

## 5. Coverage gaps to resolve or accept explicitly

* The replay module currently has one integrated test. There is no focused test
  for numbering gaps, mixed generation ABI, corrupted metrics, changed manifest,
  missing requested seed, or shard-hash reproducibility.
* Deterministic random sampling is tested; statistical uniformity is not.
* Legacy WTS decoding through the Rust bulk loader is not directly tested; that
  belongs jointly with the plan-signal migration review.
* Promotion tests do not inject failures between filesystem steps and do not
  test concurrent controllers.
* The paired null gate is strong for common-random-number wiring, but no
  synthetic non-null test pins the direction and threshold of both endpoints.
* No top-level 20-iteration controller invokes generation, training, gating,
  league registration, and resume reconciliation as one tested state machine.

## 6. Running the gates

The pre-review snapshot below is retained for provenance. Post-remediation
results are recorded in section 8 and `REVIEW_BACKLOG.md`.

```powershell
.\.venv\Scripts\python.exe -m pytest `
  games/welcome_to/tests/test_s2_train.py `
  games/welcome_to/tests/test_s2_replay.py `
  games/welcome_to/tests/test_s2_league.py `
  games/welcome_to/tests/test_s2_promotion.py `
  games/welcome_to/tests/test_self_play.py -q

Push-Location games/welcome_to/welcome_to_rust
cargo test
Pop-Location
```

## 7. Sign-offs requested

1. Is discovered-game count the correct durable replay clock, including after
   archival or partial restore?
2. Must the replay manifest pin WTS shard hashes, not only generation manifests
   and structural counts?
3. Does the Rust loader sample uniformly over eligible positions and fail on
   every incomplete selection?
4. Which reported reuse metric should control the historical 5× target?
5. Is the paired fixed-N margin/rank rule statistically adequate for promotion?
6. Is the multi-step promotion protocol recoverable from every partial failure,
   and is single-writer ownership explicit enough?
7. Are league selection and GPU residency bounded as intended?

## 8. External review response and disposition — 2026-08-28

The review reported R1–R12. The findings were valid except where explicitly
classified as an accepted semantic or a deferred performance policy.

| Finding | Assessment | Disposition |
|---|---|---|
| R1 resumed position count | correctness defect | Fixed generation metrics so both game and searched-root counts are cumulative across resumed shards. Added a regression gate for combined old/new position counts. |
| R2 shrinking replay clock | correctness/lifecycle defect | Added an atomic `replay_ledger.json`. Completed-iteration game counts remain on the cumulative clock after old data directories are archived. |
| R3 incomplete iteration directory | recovery defect | Never-completed directories are ignored. An incomplete iteration already recorded complete in the ledger is treated as corruption and fails loudly. |
| R4 unvalidated clock inputs | integrity defect | Every newly visible complete iteration is fully decoded and its game/position counts validated before it enters the ledger or affects the clock. Later metrics/manifest changes are compared with the pinned ledger entry. |
| R5 gate Dirichlet noise | evaluation defect | Promotion search now explicitly disables all root-noise settings. Generation defaults are unchanged. |
| R6 asymmetric internal opponent | evaluation bias | The evaluator now accepts a separate POLICY-row model. Both gate arms use the incumbent for simulated-opponent POLICY rows while learner LEAF rows still use the arm under test. |
| R7 secondary coin flip | statistical defect | The secondary now rejects only when the upper confidence bound is below `-secondary_tolerance`; an uncertain near-zero mean is not called a regression. |
| R8 record written last | durability defect | The report path is now a write-ahead intent and commit record. Paths and hashes are pinned, each later step is idempotent, CLI resume reconciles `installing` records before any new gate, and failure injection covers a crash after candidate installation. Single-writer ownership is explicit but not OS-lock-enforced. |
| R9 evaluation grows with buffer | bounded-cost defect | Validation and newest-iteration diagnostics now each use a deterministic, order-independent hash-ranked cap (`--max-eval-games`, default 256). All validation-pool games remain excluded from training, so the cap does not create leakage. |
| R10 league pool splits batches | valid throughput tradeoff | Deferred. The existing pool is bounded and its counts are configurable. Narrowing it per iteration changes opponent diversity and should be measured on the 5090 rather than silently changing the training distribution in a correctness remediation. |
| R11 position-uniform sampling | correct as specified | Accepted with no code change. Longer games contribute more positions by design. `samples_per_new_position`, not buffer passes, remains the controller metric for the historical 5x target. |
| R12 positional optimizer migration | migration hardening defect | New checkpoints store optimizer parameter names/order and resume requires an exact match. Legacy widening is restricted to the final per-seat output layer; old version-1 checkpoints retain the positional compatibility path needed for this migration. |

Shard hashes were not added: atomic WTS files, structural/header validation, and
content-derived counts provide integrity, while the durable per-iteration ledger
pins exactly the values that affect selection. The primary fixed-N paired rule
is retained; its real paired standard error should be measured after the R5–R7
repairs before changing the 300-game default.

Post-remediation verification: **62 focused Python tests passed**; the complete
Welcome To suite is **588 passed, 1 skipped, 1 pre-existing test warning**; the
Rust crate is **25 passed**.
