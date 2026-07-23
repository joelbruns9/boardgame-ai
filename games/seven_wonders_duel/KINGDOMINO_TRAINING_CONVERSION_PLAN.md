# Seven Wonders Duel — Kingdomino-Style Training Conversion Plan

## 1. Objective

Build a game-agnostic training controller in `games.az_loop`, modeled on the
proven Kingdomino lifecycle, and convert the Seven Wonders Duel Phase D loop
into its first client. Kingdomino remains on its existing production pipeline
without code changes, adapter work, or new test requirements.

The shared controller must preserve the Seven Wonders Duel-specific components:

- Rust game generation, search, bots, and evaluation
- Gumbel search and explicit chance handling
- SPRT promotion testing
- Transformer model and existing training losses
- Replayable JSONL game records
- Existing curriculum, seed buffer, HOF, Elo, manifests, and checkpoints

The conversion is complete when training can improve cumulatively without
requiring every intermediate learner to beat `current_best.pt`, while
`current_best.pt` remains a conservative, promotion-protected artifact. It must
also leave behind a reusable contract so a future modeling project supplies
game-specific generation, training, evaluation, checkpoint payload, and replay
operations instead of rebuilding the control loop.

The target lifecycle is:

```text
latest learner generates frontier self-play
              |
              v
latest learner trains cumulatively
              |
              v
periodic latest-vs-current_best SPRT
       |              |             |
     reject        continue        accept
       |              |             |
best generates     latest stays   latest promoted
recovery data      probationary   to current_best
```

This directly fixes the failure observed in `laptop_training_10h_01`, where all
eight candidates independently restarted from iteration `-1` because an
inconclusive gate discarded the learner.

### Architectural decision

Kingdomino is a behavioral reference, not a migration target.

```text
Kingdomino production pipeline (frozen, unchanged)
                    |
                    | behavior/reference only
                    v
          games.az_loop shared controller
                    |
                    +---- Seven Wonders Duel adapter
                    |
                    +---- future game adapter
```

The new shared code must not import `games.kingdomino`, read Kingdomino
checkpoints, or require Kingdomino tests to run. Kingdomino may adopt the shared
controller in a separate future project, but that is explicitly outside this
conversion.

## 2. Non-goals

This project does not:

- Modify, wrap, migrate, or test the Kingdomino production training pipeline.
- Require Kingdomino to implement the new adapter contract.
- Make Kingdomino depend on new `games.az_loop` controller modules.
- Replace the Rust hot path with Kingdomino's engine.
- Change Seven Wonders Duel search semantics or approve `leaf_batch > 1`.
- Replace Gumbel search with Kingdomino PUCT.
- Add Kingdomino board encodings, action codecs, NNUE, or endgame solver.
- Invent GPU, queue, entropy, Brier, or timing metrics that are not measured.
- Change model architecture or scale the network as part of the conversion.
- Select a historical candidate from the failed laptop run without a separate
  evaluation.
- Add reanalyze in this project.

## 2.1 Shared ownership boundary

The conversion must put only genuinely game-independent policy in
`games.az_loop`.

| Concern | Owner |
|---|---|
| Generator modes and state transitions | `games.az_loop` |
| Bootstrap/probation/revert/promote policy | `games.az_loop` |
| Promotion cadence | `games.az_loop` |
| Checkpoint role and lineage metadata | `games.az_loop` |
| Atomic rolling-checkpoint lifecycle | `games.az_loop` |
| Resume-state reconstruction contract | `games.az_loop` |
| Per-iteration log envelope and synchronization | `games.az_loop` |
| Human-readable console tee and `run.log` lifecycle | `games.az_loop` |
| Autosave scheduling and failure policy | `games.az_loop` |
| Game generation and search | Game adapter |
| Model construction and inference | Game adapter |
| Training data construction and SGD | Game adapter |
| Candidate-vs-best and anchor execution | Game adapter |
| Replay serialization and filtering | Game adapter |
| Game-specific metrics and summaries | Game adapter |
| HOF/Elo storage implementation | Existing shared helpers plus adapter calls |

The controller orchestrates. It does not understand cards, tiles, legal actions,
network heads, Rust engines, replay record classes, or model architectures.

## 2.2 Proposed shared modules

Extend the existing `games.az_loop` package rather than creating another
Seven Wonders Duel utility package:

```text
games/az_loop/
  training_control.py       # modes, state, actions, transition policy
  checkpoint_lifecycle.py   # roles, hashes, atomic rolling artifacts
  training_log.py           # stable envelope, append/backfill rules
  run_log.py                # human-readable, line-flushed console tee
  run_controller.py         # iteration orchestration and resume
```

Use fewer files if the final interfaces remain cohesive. Existing
`manifest.py`, `hof.py`, `elo.py`, `schedule.py`, and `sprt.py` remain shared
building blocks.

Seven Wonders Duel supplies an adapter, initially in:

```text
games/seven_wonders_duel/training_adapter.py
```

`phase_d.py` remains the CLI/composition entry point and delegates lifecycle
decisions to the shared controller.

## 2.3 Minimum adapter contract

The implementation should use typed request/result objects rather than passing
an unstructured dictionary through the controller. The exact names may change,
but a future game must be able to provide equivalents of:

Name this Protocol `LifecycleAdapter` (not `TrainingAdapter`). `games.az_loop`
already defines `core.GameAdapter` — the small engine/match boundary used by
orchestration and evaluation (`new_game`/`step`/`terminal`/`outcome`). That is a
different, unrelated concern; reusing "adapter" unqualified for the training
lifecycle would collide with it in imports and reading. The lifecycle Protocol
below composes with, and does not replace, the existing `core.GameAdapter`.

```python
class LifecycleAdapter(Protocol):
    def initialize_learner(...) -> CheckpointArtifact: ...
    def load_learner(...) -> LearnerHandle: ...
    def generate(...) -> GenerationResult: ...
    def assemble_replay(...) -> ReplayResult: ...
    def train(...) -> TrainingResult: ...
    def evaluate_promotion(...) -> PromotionResult: ...
    def evaluate_anchors(...) -> AnchorResult: ...
    def promote(...) -> CheckpointArtifact: ...
    def save_replay(...) -> ReplayArtifact: ...
    def load_replay(...) -> ReplayArtifact: ...
```

The shared controller owns when these operations happen and what lifecycle
transition follows. The adapter owns how they happen and which existing metrics
they return.

## 3. Current Problem

The current Phase D loop treats `current_best.pt` as all three of:

1. The protected best model.
2. The next self-play generator.
3. The initialization point for the next candidate.

Its iteration is:

```text
load current_best
generate games with current_best
train candidate from current_best
gate candidate against current_best
promote only on SPRT accept
```

An SPRT `continue` leaves `current_best` unchanged. The next iteration therefore
reloads the same model and loses the previous candidate's learning.

In the laptop run:

- `current_best.pt` remained the random iteration `-1` checkpoint.
- All eight candidates were trained independently from it.
- Every 50-game gate returned `continue`.
- None of the candidates became the next learner or generator.

The gate protected the best artifact correctly, but it also prevented
bootstrapping and cumulative learning.

## 4. Target State Model

Introduce three explicit identities.

### 4.1 Learner

The learner is the model being updated by SGD. It persists across iterations.
Its durable checkpoint is:

```text
checkpoints/latest.pt
```

After each successful training phase, `latest.pt` is atomically replaced with
the newly trained learner. An inconclusive promotion test does not roll it back.

### 4.2 Self-play generator

The generator is the checkpoint used by Rust self-play for the next iteration.
It can be either `latest.pt` or `current_best.pt`, depending on control mode and
the last gate action.

The generator identity must be logged explicitly. It must never be inferred
from filenames after the fact.

### 4.3 Protected best

`current_best.pt` changes only after an SPRT `accept`. Before replacement, the
old best is copied into the HOF. A `continue` or `reject` must not overwrite it.

### 4.4 Candidate snapshots

Continue writing immutable per-iteration snapshots:

```text
checkpoints/candidate_0000.pt
checkpoints/candidate_0001.pt
...
```

The candidate snapshot and `latest.pt` should contain identical model weights
immediately after training. Candidate snapshots provide auditability while
`latest.pt` is the rolling continuation point.

## 5. Generator Modes

Add a Seven Wonders Duel equivalent of Kingdomino's control modes:

```text
--selfplay-generator-mode latest
--selfplay-generator-mode current_best
--selfplay-generator-mode strict_gate
--selfplay-generator-mode soft_gate
```

### `latest`

- Self-play always uses `latest.pt`.
- Training continues from `latest.pt`.
- Promotion may be disabled or run only for measurement.
- Useful for research runs where frontier learning matters more than a protected
  production checkpoint.

### `current_best`

- Self-play always uses `current_best.pt`.
- The learner can continue independently, but generation does not follow it.
- Useful for controlled data generation and ablations.

### `strict_gate`

- Preserves current Phase D behavior.
- Self-play uses `current_best.pt`.
- A candidate affects generation only after SPRT `accept`.
- Retained for compatibility and gate testing, not recommended for new
  bootstrap training.

### `soft_gate`

- Recommended production mode.
- Self-play normally uses `latest.pt`.
- Training continues from `latest.pt`.
- SPRT `accept` promotes latest to best.
- SPRT `continue` keeps latest as a probationary generator.
- SPRT `reject` switches generation to `current_best.pt` for recovery data.
- A reject does not immediately destroy `latest.pt`; the learner may recover.

Initially keep the CLI default as `strict_gate` for backward compatibility.
After the conversion gates pass, change the documented training command to
`soft_gate`. Change the code default only in a separate, explicit commit.

## 6. Bootstrap Policy

An untrained iteration `-1` checkpoint is not a meaningful incumbent. Add an
explicit bootstrap state rather than asking the ordinary promotion gate to
solve initialization.

Recommended policy:

1. Initialize `latest.pt` and `current_best.pt` from the same random weights.
2. Mark both checkpoints with `training_state: "untrained"`.
3. Generate the configured seed and first self-play data.
4. Train the first learner successfully.
5. Atomically install it as both `latest.pt` and `current_best.pt`.
6. Add manifest/log action `bootstrap_promote`.
7. Begin normal soft gating on subsequent promotion checks.

Bootstrap promotion requires:

- A non-empty training set.
- Finite training and validation metrics.
- A successfully written and reloadable checkpoint.
- No replay, encoder-signature, or model-contract error.

It does not require a strength result against random initialization.

Add:

```text
--bootstrap-policy auto_first_trained
--bootstrap-policy gate
```

`auto_first_trained` is the recommended default for new runs. `gate` preserves
the old behavior for exact compatibility tests.

## 7. Soft-Gate State Transitions

Seven Wonders Duel already has a paired SPRT with three decisions. Reuse it
directly instead of adding a second raw-win-rate decision system.

| SPRT result | Soft-gate action | Next generator | Learner | Best |
|---|---|---|---|---|
| `accept` | `promote` | latest | keep latest | replace with latest |
| `continue` | `probation` | latest | keep latest | unchanged |
| `reject` | `revert` | current best | keep latest initially | unchanged |

This is the Seven Wonders Duel analogue of Kingdomino's promote/probation/revert
policy. SPRT already encodes `gate_alpha`, `gate_beta`, and the indifference
band, so no duplicate confidence formula is required.

### Recovery after reject

Track consecutive reverts.

Add:

```text
--revert-reset-after 3
```

The counter is measured in **gate checks, not iterations** (matching
Kingdomino, whose comment is explicit: "after N CONSECUTIVE failed-below-revert
gate checks"). This matters once `--promotion-every > 1` (Section 8): an
iteration that logs `promotion_action: "not_scheduled"` ran no gate and must
**not** touch the counter — it neither increments it nor resets it. Only an
actual gate decision moves it.

Initial behavior:

- Each gate `reject` increments the consecutive-revert counter.
- First and second consecutive rejects: generate with current best, but continue
  training the latest learner on the resulting recovery data.
- At the configured consecutive-reject threshold: reset learner weights to
  current best before the next training phase.
- Reset any persisted optimizer state at the same time.
- Log `revert_reset`.
- Any gate `probation` (`continue`) or `promote` (`accept`) resets the
  consecutive-revert counter to zero. A `not_scheduled` iteration leaves it
  unchanged.

Use `0` to disable automatic learner reset.

## 8. Promotion Cadence

The current loop gates every iteration. Add:

```text
--promotion-every 1
```

Semantics:

- `0`: do not run automatic promotion matches.
- `N`: gate after every N completed training iterations.
- Non-gate iterations continue with the current generator policy and log
  `promotion_action: "not_scheduled"`.

Keep the existing paired SPRT parameters:

```text
--gate-sims
--gate-max-games
--gate-alpha
--gate-beta
--gate-indifference
```

Do not use `gate_max_games=50` as a production strength conclusion. It is a
short control check and often returns `continue`. Soft gate makes that safe:
`continue` retains frontier learning instead of discarding it.

## 9. Persistent Training State

Implement this in two stages so model-lifecycle correctness is isolated from
optimizer-policy changes.

### Stage A: persistent weights, existing epoch trainer

- Load `latest.pt` at the start of each iteration.
- Train using the existing `train_epochs`, batch size, optimizer, scheduler,
  losses, early stopping, and game-honest split.
- Save candidate and rolling latest checkpoints.
- Recreate AdamW each iteration as the current trainer does.

This is the minimum safe conversion and fixes the failed ratchet immediately.

### Stage B: optional persistent optimizer and fixed-step training

After Stage A passes learning and resume gates:

- Store optimizer and scheduler state in `latest.pt`.
- Add an optimizer schema/version field.
- Restore optimizer state only when model and optimizer contracts match.
- Clear optimizer state on `revert_reset`.
- Add an optional fixed optimizer-step budget:

```text
--train-budget epochs
--train-budget steps
--train-steps N
```

Keep `epochs` as the compatibility default. Fixed steps are valuable once the
replay buffer grows because epoch-based work increases every iteration. Do not
change this policy in the same milestone as soft-gate lifecycle correctness.

## 10. Replay Buffer Conversion

Seven Wonders Duel already has:

- Per-iteration JSONL game records
- A replay window
- Atomic `--save-buffer`
- `--warm-buffer`
- Warm-record aging

Preserve the replayable JSONL format; do not copy Kingdomino's pickle format.

Add Kingdomino-style operational controls:

```text
--buffer-autosave-every 0
--warm-buffer-max-staleness 20
```

### Autosave

- `0`: save only on exit.
- Positive `N`: atomically save every N completed iterations and on exit.
- A failed autosave warns but does not terminate training.
- A partial write must never replace the last valid export.

### Warm-buffer staleness

- Apply an explicit iteration-age filter when importing.
- Default to the active `replay_window`.
- Record loaded, dropped, and retained game counts using actual counts.
- Do not renumber source iteration metadata.
- Continue validating record schema and replay compatibility at load.

### Capacity

Do not add a game or position capacity until measured memory/disk growth makes
the iteration window insufficient. `replay_window` is currently the real
retention mechanism and should remain authoritative during this conversion.

## 11. Checkpoint Contract

Every learner/best/candidate checkpoint should contain:

- Model name and architecture
- Encoder signature
- Iteration
- Training history
- Training state: `untrained` or `trained`
- Role at write time: `latest`, `candidate`, or `current_best`
- Generator source used to produce the newest data
- Source checkpoint path and SHA-256
- Best checkpoint path and SHA-256
- Optional optimizer state/schema after Stage B
- Existing baselines and game-honest split metadata

Writes to `latest.pt` and `current_best.pt` must use temporary files plus atomic
replace. Candidate snapshots remain immutable.

On resume:

1. Validate the run manifest.
2. Validate model/encoder contracts.
3. Load `latest.pt` as learner.
4. Load `current_best.pt` as protected baseline.
5. Restore generator state from the last completed log/manifest row.
6. Verify referenced checkpoint hashes.
7. Refuse to silently substitute random weights for a missing established
   checkpoint.

## 12. HOF and Elo

Keep the current Seven Wonders Duel HOF and Elo implementations.

### HOF

- Add the previous `current_best.pt` to HOF immediately before promotion.
  This is a deliberate change from the current `promote()` in `phase_d.py`,
  which replaces `current_best` and then adds the *new* best to HOF tagged
  `promoted`. The new behavior archives the *outgoing* best instead, so HOF
  accumulates the lineage of superseded champions rather than the current one.
  Any test or tooling that assumes the old HOF-gets-new-best behavior must be
  updated in the same change.
- Never add an `untrained` checkpoint to HOF. Because bootstrap (Section 6)
  initializes `current_best.pt` from random weights, the first promotion would
  otherwise archive the iteration `-1` random model into the protected HOF.
  Skip the pre-promotion archive whenever the outgoing best is still
  `training_state: "untrained"`.
- Do not add probationary or rejected candidates to the protected HOF.
- Candidate snapshots remain available for diagnostics independently of HOF.

### Elo

- Continue recording promotion games.
- Run anchor gates only at the configured successful-promotion cadence.
- Do not treat disconnected candidate-only Elo values as reliable absolute
  ratings when no fixed-anchor games were played.
- Optionally add `--smart-elo-on-promote` later, but do not block the lifecycle
  conversion on it.

## 13. Training Log Contract

Retain `training_log.jsonl` as one append-only JSON object per completed
iteration. Continue synchronizing missing rows from `run_manifest.json`.

Two complementary logs are required:

```text
<run-dir>/training_log.jsonl   # structured, one JSON row per iteration
<run-dir>/run.log              # human-readable console transcript
```

Neither replaces the other. `training_log.jsonl` is the analysis/API surface;
`run.log` is the operational record used to follow a live run and diagnose
warnings, stalls, early stopping, gates, exceptions, and shutdown behavior.

### Human-readable `run.log`

Implement a shared `RunLog`/tee utility in `games.az_loop.run_log`. It must:

- Default to `<run-dir>/run.log`.
- Append on resume rather than truncate the previous invocation.
- Mirror output to the interactive console and the file.
- Capture both normal progress and warnings/errors written by the Phase D entry
  point.
- Use UTF-8 and normalized newlines.
- Be line-buffered or explicitly flushed after every complete line so a crash
  loses at most an incomplete line.
- Be written **only from the orchestrator (parent) process.** Generation and
  gate work runs through `core.run_jobs_in_processes`, which uses the `spawn`
  start method, so workers are separate OS processes; a `threading.Lock` in the
  parent serializes nothing across them and a worker cannot be handed the tee.
  Workers must therefore not write `run.log` directly — they return results, and
  the parent logs their aggregated progress. Within the parent, serialize the
  tee's own threads (`run_jobs` thread pool, trainer callback) with a lock so
  concurrent lines cannot interleave bytes. Do not specify or rely on any
  cross-process file-locking scheme.
- Work on Windows PowerShell and Linux without `Tee-Object`, `tee`, `nohup`
  redirection, or shell-specific quoting.
- Close and flush on clean completion, Ctrl+C, and ordinary exceptions.
- Preserve the original traceback and exit code; logging must not swallow a
  training failure.
- Warn to the console if the file cannot be opened, then continue training with
  console output rather than failing the run.

The CLI boundary should install the tee before model/run initialization so the
log captures configuration validation, warm-load messages, checkpoint warnings,
and initialization failures. Route trainer progress through the same logger
callback. Do not maintain a separate second set of hand-formatted training
metrics.

Each process invocation appends a clearly delimited header:

```text
============================================================
Run invocation started: <UTC timestamp>
Run directory: <absolute path>
Command: <exact argv>
Resume iteration: <number or new run>
Generator mode: <mode>
Structured log: <absolute training_log.jsonl path>
Manifest: <absolute run_manifest.json path>
============================================================
```

Each iteration should follow the useful Kingdomino narrative pattern, populated
only with Seven Wonders Duel values already produced by the pipeline:

```text
============================================================
Iteration 5/20
============================================================
  generator: latest_iter_0004 (probation)
  self-play: 350 games (0.35 games/sec), mixed=24, neural=326
  victories: civilian=319 military=25 scientific=6
  replay: 2350 games, 162319 examples
  epoch 0: train total ... | val total ... policy_top1 ... value_acc ...
  promotion: 22/50 score=44.0% decision=continue action=probation
  checkpoint: candidate_0005.pt; latest updated; current_best unchanged
iter 005 | sp: 350 games 0.35/s | replay: 2350 games | gate: probation
```

The exact wording may evolve, but the ordering and field meanings must remain
stable. If an underlying value does not exist, omit that clause rather than
printing a fabricated zero.

At shutdown, append:

```text
Run completed: <UTC timestamp>
Completed iterations: <count>
Latest checkpoint: <path>
Current best: <path>
Final buffer: <path or disabled>
```

For interruption or failure, append the corresponding termination status and
traceback before closing the log.

Do not call this file `nohup.log`; `run.log` is the platform-independent
canonical name. Users may still redirect the outer process if desired, but the
pipeline-owned log remains authoritative and consistently located.

### Lifecycle fields

Add state metadata that the new loop necessarily knows:

```text
iteration
generator_mode
generator_source
generator_checkpoint
generator_sha256
learner_source
latest_checkpoint
latest_sha256
current_best_checkpoint
current_best_sha256
current_best_iteration
bootstrap_state
promotion_scheduled
promotion_action
consecutive_reverts
```

These are control-state facts, not new performance metrics.

### Existing generation data

Log the existing values without renaming their meaning:

```text
generated_games
generation_performance.seconds
generation_performance.games_per_second
generation_performance.mode
generation_performance.rust_games
generation_performance.rust_bot_games
generation_performance.python_bot_games
generated_summary.moves
generated_summary.game_kinds
generated_summary.victory_types
generated_summary.policy_eligible_moves
generated_summary.policy_eligible_fraction
generated_summary.searched_moves
generated_summary.average_sims
```

### Existing replay/training data

```text
training_games
training_performance.examples
training_performance.train_examples
training_performance.validation_examples
training_performance.train_games
training_performance.validation_games
training_performance.newest_iteration
training_performance.pretrain_newest_metrics
training_performance.epochs
training_summary
```

Per-epoch rows already contain training loss parts, validation metrics, and
epoch seconds. Keep them nested rather than manufacturing a flattened metric
with different semantics.

### Existing gate data

```text
promotion_gate
anchor_gates
phase_gate_passed
promoted
gates
```

Add `promotion_action` to distinguish:

```text
not_scheduled
bootstrap_promote
promote
probation
revert
revert_reset
```

### Explicitly unavailable

Do not log placeholders for:

- GPU utilization
- GPU busy time
- H2D/forward/readback split
- CPU utilization
- Queue wait
- Padding fraction
- NN rows/s unless the evaluator actually exposes it
- Policy entropy or Brier scores unless computed by the trainer

Missing metrics should be absent, not set to zero.

## 14. CLI Changes

Add:

```text
--selfplay-generator-mode
--bootstrap-policy
--promotion-every
--revert-reset-after
--buffer-autosave-every
--warm-buffer-max-staleness
--run-log
--no-run-log
```

`--run-log PATH` overrides the default `<run-dir>/run.log`.
`--no-run-log` is intended for tests and embedding; normal training should
always keep the human-readable log.

Stage B optionally adds:

```text
--train-budget
--train-steps
```

Retain:

```text
--generation-backend
--gate-backend
--save-buffer
--warm-buffer
--replay-window
--gate-*
--anchor-gate-every-promotions
```

Update `training_parameters.md` in the same change as each CLI addition. A
mechanical parser-to-document coverage test must continue to report no missing
arguments.

## 15. Implementation Milestones

### Milestone 1 — State and checkpoint separation

Deliverables:

- `games.az_loop` controller package with no Kingdomino dependency
- `GeneratorState` or equivalent typed shared state object
- Typed adapter requests/results
- Separate `latest.pt` and `current_best.pt`
- Atomic checkpoint helpers
- Resume reconstruction
- Lifecycle metadata in manifest/log

Acceptance:

- A candidate can become latest without changing best.
- Continue/reject cannot overwrite best.
- Restart loads the exact learner, generator, and best state.
- Generic controller tests use a fake in-memory adapter.
- Existing Seven Wonders Duel strict-gate tests still pass.
- No Kingdomino files or tests are changed.

### Milestone 2 — Bootstrap and generator modes

Deliverables:

- Seven Wonders Duel implementation of the shared adapter contract
- Four generator modes
- Automatic first-trained bootstrap
- Mode-specific generator selection
- CLI validation and documentation

Acceptance:

- A new soft-gate run escapes iteration `-1` after its first valid training.
- Iteration 1 initializes from iteration 0 latest weights.
- Strict mode reproduces the old lifecycle.
- Rust receives the selected generator model at every generation call.

### Milestone 3 — Soft-gate transitions

Deliverables:

- Accept/promote
- Continue/probation
- Reject/revert
- Consecutive-revert tracking and reset
- Promotion cadence

Acceptance:

- Synthetic accept/continue/reject tests assert exact checkpoint hashes and
  next-generator identity.
- Paired SPRT remains unchanged.
- Probation continues learning from latest.
- Revert generates with best without immediately deleting latest.
- Revert-reset restores best weights and clears applicable learner state.

### Milestone 4 — Replay operations

Deliverables:

- Periodic atomic autosave
- Warm-buffer staleness filtering
- Load/save counts in existing operational output
- Resume tests

Acceptance:

- Hard interruption during temporary save leaves the previous export readable.
- Warm data ages out deterministically.
- Saved then loaded records preserve trajectory digests.
- No duplicate records appear solely because a run resumed.

### Milestone 5 — Logging parity

Deliverables:

- Stable training-log schema/version
- Lifecycle fields
- Existing nested metrics preserved
- Manifest/log synchronization
- Shared platform-independent `run.log` tee
- Stable startup, iteration, and shutdown narrative
- Compact console summary

Acceptance:

- Exactly one log row per completed iteration.
- Resume backfills missing rows without duplicates.
- Every logged metric traces to an existing measured value.
- Golden-row test covers bootstrap, probation, revert, and promote.
- `run.log` contains startup, epoch, gate, checkpoint, and shutdown messages.
- Resume appends a new invocation header without truncating prior output.
- A simulated exception appears in `run.log` and still propagates to the caller.
- Disabling the run log affects only the human transcript, not JSONL/manifest
  persistence.

### Milestone 6 — Optional persistent optimizer/fixed steps

Deliverables:

- Versioned optimizer checkpoint state
- Epoch/step training budget selection
- Revert-reset optimizer clearing

Acceptance:

- Interrupted/resumed training restores optimizer state exactly.
- Epoch mode remains behaviorally compatible.
- Step mode performs exactly the configured number of completed updates.
- This milestone is independently revertible if it does not improve operations.

### Milestone 7 — End-to-end validation

Run in order:

1. CPU plumbing smoke with one bootstrap and one subsequent iteration.
2. Forced synthetic SPRT transition tests.
3. Rust CUDA smoke with three iterations and tiny budgets.
4. Resume the same smoke for two more iterations.
5. Short real run with enough games to observe probation.
6. Only then launch a multi-hour laptop run.

## 16. Test Matrix

### Unit tests

- Mode parsing and validation
- Generator selection
- Bootstrap eligibility
- Promotion action mapping
- Consecutive-revert counter
- Checkpoint atomicity
- Warm-record staleness
- Log serialization
- Run-log tee, flush, append, and failure fallback

### State-transition tests

| Starting state | Gate result | Expected action | Next generator | Best changes? |
|---|---|---|---|---|
| untrained | not run | bootstrap promote | latest | yes |
| trained | accept | promote | latest | yes |
| trained | continue | probation | latest | no |
| trained | reject | revert | best | no |
| repeated reject | reject | revert reset | best | no |

### Integration tests

- Two soft-gate iterations prove cumulative weight lineage.
- Strict-gate compatibility proves old behavior remains available.
- Rust generation uses latest during probation.
- Rust generation uses best after revert.
- Rust model-v-model gate uses the intended two checkpoint hashes.
- HOF receives only pre-promotion best checkpoints.
- Save/warm replay works across distinct run directories.

### Regression tests

- Existing Phase D suite
- Bot parity and seed-buffer trajectory parity
- Rust/Python record conversion
- F4 exactness at `leaf_batch=1`
- Full Seven Wonders Duel test suite

Kingdomino is not part of the conversion regression matrix because its files
and runtime path are unchanged. The implementation may read Kingdomino source
as a behavioral reference, but completion does not require editing or rerunning
its pipeline.

## 17. Operational Acceptance Gates

The conversion is ready for a real laptop run only when:

- All automated tests pass.
- The first trained model bootstraps exactly once.
- A `continue` result leaves latest active and best unchanged.
- A `reject` switches only the generator unless reset threshold is reached.
- An `accept` archives old best and atomically installs new best.
- Resume preserves checkpoint hashes and generator state.
- The training log identifies which model generated every iteration.
- `run.log` provides a complete, flushed human-readable transcript at a stable
  path.
- Final and autosaved buffers reload and replay successfully.
- Rust throughput with `leaf_batch=1` does not regress materially in a matched
  smoke.

The first real-run success criteria are:

- More than one iteration descends from a trained predecessor.
- `current_best` is no longer iteration `-1`.
- Training and validation remain finite.
- Promotion actions are visible and internally consistent.
- Military/science victory coverage does not collapse unnoticed.
- The run can stop, save, resume, and continue without cold-starting.

## 18. Migration and Compatibility

### Existing runs

Do not silently reinterpret old Phase D runs as soft-gate runs.

- A run manifest without generator-state schema remains `strict_gate`.
- It may be resumed only under strict compatibility behavior.
- To start soft-gate training from an old artifact, use a new run directory,
  an explicitly selected compatible checkpoint, and `--warm-buffer`.
- Do not automatically choose one of the failed laptop candidates.

### New runs

Recommended initial command shape after conversion:

```powershell
.\.venv\Scripts\python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir games\seven_wonders_duel\runs\<run_name> `
  --selfplay-generator-mode soft_gate `
  --bootstrap-policy auto_first_trained `
  --promotion-every 1 `
  --revert-reset-after 3 `
  --generation-backend rust `
  --gate-backend rust `
  --leaf-batch 1 `
  --save-buffer games\seven_wonders_duel\runs\<run_name>\buffer_final.jsonl
```

Fill in game, search, model, and scheduler budgets from the measured hardware
configuration. Do not copy Kingdomino's numeric parameters.

## 19. Recommended Implementation Order

Implement Milestones 1–5 before another training run. Milestone 6 is optional
and should follow evidence that epoch-based training or optimizer resets are an
operational limitation.

The safest sequence is:

```text
shared controller contract and fake adapter
  -> shared checkpoint/state separation
  -> Seven Wonders Duel adapter
  -> bootstrap
  -> generator modes
  -> soft-gate transitions
  -> replay autosave/staleness
  -> log contract
  -> CPU/Rust/CUDA/resume validation
  -> short learning run
  -> laptop run
```

Do not combine this conversion with model scaling, WU-UCT approval, new training
targets, reanalyze, or cloud throughput tuning. The first objective is to prove
that one learned iteration becomes the parent of the next while the protected
best checkpoint remains correct. Kingdomino remains frozen throughout.
