# Review request — dense plan-completion and end-mode learning signal

The model is weak at the behavior that ends high-level games: completing all
three City Plans. This change adds dense terminal labels that tell the shared
trunk whether each plan will complete, who completes it first, and which
independent end condition fires. The heads are auxiliary only; search still
uses policy, rank, and score.

This request includes storage and checkpoint migrations because a target-schema
change is not complete if old replay or the current S1/S2 checkpoint becomes
unusable.

## 1. Scope

All changes are currently uncommitted on top of `d8509b3`.

| Area | Files |
|---|---|
| terminal outcome derivation and target schema | `training.py`, `AUX_TARGETS_SPEC.md`, `tests/test_training.py` |
| binary output heads, BCE loss, group weights, compatible model load | `network.py`, `tests/test_network.py` |
| S0/S2 binary evaluation metrics | `train.py`, `s2_train.py`, their tests |
| WTS v2 writer and v1 readers/upgrader | `welcome_to_rust/src/samples.rs`, `self_play.py`, tests |
| S2 checkpoint v2 and legacy model/Adam expansion | `s2_train.py`, `network.py`, `tests/test_s2_train.py` |

Replay-window selection and the general Rust loader are reviewed in
`S2_REPLAY_PROMOTION_REVIEW_REQUEST.md`; only their schema-migration behavior is
in scope here.

## 2. New labels and semantics

Nine binary values are appended to each valid seat's supervised head outputs:

* `will_complete_plan_0..2`: terminal truth for each plan slot, trained on every
  valid seat at every recorded position;
* `plan_0_first..2`: whether that seat's completion turn equals the earliest
  completion turn for the slot. Ties are positive for every tied earliest seat;
  non-completers carry sentinel `-1` behind a zero mask;
* `end_trigger_full_sheet`, `end_trigger_all_plans`,
  `end_trigger_max_permit`: independent terminal clauses. They are not a
  mutually exclusive end-reason softmax; multiple labels may be one.

All nine network outputs are raw logits and use masked
binary-cross-entropy-with-logits. The existing `plans_completed` and
`turns_to_plan_*` regressions retain the `plan_race` group at weight 0.3.
Completion and first-finisher labels use a separate `plan_outcome` group at
weight 0.3; end-mode labels form `outcome_mode` at weight 0.2. Group weights are
applied once to the mean of members with support in the current batch.

## 3. Compatibility contract

### WTS shards

New writers emit WTS version 2 with 32 per-seat target columns. Version-1 shards
with 20 columns remain readable in Python and in the Rust bulk loader:

* completion is recovered exactly from `turns_to_plan_k_mask`;
* first-finisher order did not exist, so its value is sentinel `-1` and mask 0;
* full-sheet, all-plans, and max-permit clauses are recovered from the normalized
  terminal `houses`, `plans_completed`, and `permits` targets;
* every old column keeps its original order and value.

### Checkpoints

S2 checkpoint version 2 accepts version 1. The new outputs are appended to the
final shared per-seat linear layer: old weight/bias rows remain byte-equal and
new rows initialize to zero logits. On optimizer resume, first and second Adam
moments for appended rows also initialize to zero while historical rows and
scalar step values are retained. The generic `train.load` uses the same model
migration.

Zero logits mean a neutral 0.5 prediction before the new heads learn. Since the
new heads are auxiliary and score retains its old row, loading an old checkpoint
does not change inference outputs used by search.

## 4. What is already gated

* Fifteen completed three-seat games compare all dense labels against terminal
  `PlayerOutcome`, including encoder seat order, ties, masks, sentinels, and all
  three independent end clauses.
* Every active binary label is asserted to be exactly zero or one.
* Every binary head is checked against an independently computed masked BCE with
  mask-sum normalization.
* Evaluation reports support, BCE, Brier score, accuracy, and positive rate for
  binary heads and does not mislabel them as R² regressions.
* Current WTS v2 rows are covered by the existing Rust-capture versus Python
  oracle comparison.
* A focused v1 target-upgrade test verifies exact completion/end recovery and
  refuses to invent first-finisher order.
* Version-1 checkpoint tests verify old model rows and Adam moments remain
  exact, appended rows and moments are zero, scalar optimizer state survives,
  both loaders agree, and resumed optimization completes a finite step.

## 5. Review focus

### 5.1 Target truth and timing

These are terminal labels copied back to every recorded state. Confirm that this
is the intended auxiliary task rather than a claim about current legality or
probability under optimal play. Check plan slot identity, absolute completion
turns, ties, and the seat-axis transform. A seat-order mistake preserves every
shape and teaches the wrong player.

### 5.2 End clauses are simultaneous predicates

The engine stops on the first terminal transition, but more than one predicate
can be true in that final state. Confirm `not sheet.has_free_box()`, all three
recorded plan completions, and `permits >= PERMIT_BOXES` are exactly the rule
predicates we want to supervise, including a move that satisfies two at once.

### 5.3 Masking and group scale

`plan_k_first` is conditioned on eventual completion; `will_complete_plan_k`
is not. Verify padded seats and non-completer sentinels cannot reach BCE, and
that group-mean weighting does not let six plan binary heads silently dominate
policy/value because of member count. The selected 0.3/0.2 weights are design
defaults, not strength evidence.

### 5.4 Two independent legacy WTS upgrade paths

Python `_decode_wts_targets` and Rust `append_training_targets` implement the
same v1→v2 map separately. The current focused test directly exercises only the
Python helper; the Rust random loader test uses v2 shards. Please require a
synthetic v1 WTS shard decoded through both paths and compared column-for-column,
or explicitly accept the duplicate unpaired implementation.

### 5.5 Optimizer migration

The legacy checkpoint test covers model weights, not Adam moments. Audit the
parameter-id-to-current-parameter zip, final-layer identification, moment shape
expansion, scalar state preservation, and param-group loading. A test should
resume a real version-1 optimizer state, prove old rows/moments are unchanged,
new moments are zero, and complete one finite optimizer step.

### 5.6 Schema completeness guards

Target names are duplicated across Python and Rust, and the ordering is an ABI.
Confirm import-time checks compare exact names, not only counts; version-1 and
version-2 header readers reject all other combinations; and adding a future
target fails loudly until writer, both readers, head mapping, loss grouping, and
tests move together.

## 6. Known limitations

* No loss-weight ablation has been run. The change improves supervision density;
  it does not yet prove better plan completion.
* `plan_k_first` supervision is unavailable in historical WTS data by design,
  so its effective support grows only with new v2 positions.
* Accuracy at a 0.5 threshold is diagnostic and can look good on imbalanced
  outcomes; BCE, Brier, positive rate, and support are the useful companions.
* The heads are not used by MCTS. Any search-time use would be a separate
  strength change.

## 7. Running the gates

Current focused result on 2026-08-27, as part of the combined review suite:
**138 Python tests passed with one test-only warning; 24 Rust tests passed.**

```powershell
.\.venv\Scripts\python.exe -m pytest `
  games/welcome_to/tests/test_training.py `
  games/welcome_to/tests/test_network.py `
  games/welcome_to/tests/test_s2_train.py `
  games/welcome_to/tests/test_self_play.py -q

Push-Location games/welcome_to/welcome_to_rust
cargo test
Pop-Location
```

## 8. Sign-offs requested

1. Are completion, tied-first, and independent end-clause labels semantically
   correct and aligned to the encoded seat axis?
2. Are masks and group reductions correct for both valid and padded seats?
3. Is the v1→v2 WTS mapping exact for every recoverable label and honest about
   unavailable first-finisher order?
4. Must a cross-language legacy-shard parity test be added before sign-off?
5. Does legacy model and Adam migration preserve all old state and initialize
   only appended rows?
6. Are exact target-name/order guards present at every storage boundary?

## 9. External review response and disposition — 2026-08-28

| Finding | Assessment | Disposition |
|---|---|---|
| Q1 plan-loss dilution | correctness/objective defect | Fixed. Historical plan progress retains its independent 0.3 group, while the six new terminal plan labels receive a separate 0.3 group. Masked targets with zero support no longer occupy a group denominator, so legacy first-finisher placeholders cannot dilute informative losses. |
| Q2 unenforced legacy ordering | migration guard defect | Fixed. Rust exports the exact 20-name legacy schema, Python asserts it at import and at shard admission, Rust upgrade offsets are named constants, and a Rust test pins those constants to the exported names. This takes the review's explicit-export option; a duplicate synthetic shard builder is unnecessary for detecting reorder drift. |
| Q3 Adam migration untested | coverage defect | Fixed. A real populated AdamW state is converted to the legacy final-head shape, loaded through the migration, and checked for byte-equal old moments, zero new moments, retained scalar step state, exact model widening, and one finite resumed update. Optimizer name/order validation from replay review R12 remains active. |
| Q4 low-variance diagnostics | valid reporting concern | Added `support_*`, `target_mean_*`, and `target_std_*` beside every regression and binary diagnostic in both S0 and S2. Binary base rate remains available as `positive_rate_*`. R² is retained as a mathematically defined diagnostic but can now be identified immediately as unstable when target standard deviation is tiny. Outcome-mode weight 0.2 is unchanged pending an ablation; class imbalance alone is not evidence that the representation target is useless. |

The strength correction in the review is accepted: the later continuation
learner is stronger than GreedyBot on both score and plans per seat-game. The
remaining behavioral gap is specifically completing all three plans, with the
hardest headroom in plan slots 0 and 1. Reachability features, capacity-credit
instrumentation, and non-uniform row sampling are potential follow-up
experiments, not correctness fixes in this review.

Post-remediation verification: **96 focused Python tests passed**; the complete
Welcome To suite is **590 passed, 1 skipped, 1 pre-existing test warning**; the
Rust crate is **26 passed**.
