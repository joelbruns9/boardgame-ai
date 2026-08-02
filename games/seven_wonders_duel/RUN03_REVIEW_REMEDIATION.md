# Remediation plan — external review of the run 03 fixes

**Status: the cloud run is blocked until items 1–3 land.** Items 4–6 should land
before `--intervention-ladder` or `--probation-reset-after` is enabled on any
long run.

This document is written to be picked up cold. Each item states what is wrong,
where, why it matters, what to change, and how to prove the change worked. Line
numbers are as of commit `98494c1` plus the uncommitted `stagnation.py` edit;
verify them with the quoted search strings rather than trusting the numbers.

## Background in one paragraph

`laptop_training_03_w7` went 85 iterations without promoting. Diagnosis: the
learner was **regressing** (at iteration 135 the candidate scored 0.335 against
the net it started from), and a 200-game gate could not resolve anything between
0.402 and 0.598, so nothing arrested it. Commits `7a2d769` and `98494c1`
addressed that with a gate ladder, a probation-driven reset, a retargeted
self-anchor (W7c), a HOF training-target filter, and a manifest rewrite. An
external review then found nine defects in that work, all verified. Two of the
original claims were also wrong and need retracting (items 7 and 10).

Terminology: **`latest.pt`** is the rolling learner; **`current_best.pt`** is the
promoted champion; **`candidate_NNNN.pt`** is the raw training output of
iteration `NNNN`, written *before* the lifecycle decides what to do with it.

---

## 1. [BLOCKER] The cloud launcher does not run the configuration we documented

**Where:** `setup_cloud_7wd.sh`, `TRAIN_CMD=(` at line 289, and the
`GATE_LADDER` default at line 89.

**What is wrong.** The launch command passes only
`--selfplay-generator-mode soft_gate --revert-reset-after 2`. It never passes
`--promotion-every`, `--bootstrap-policy`, or `--probation-reset-after`, so the
cloud run silently uses the parser defaults `4`, `gate`, and `0`. The ladder
default is still `100 200 400 800`, not the `200 600 1000 1500` that
`training_parameters.md` documents and that the gate-resolution analysis calls
for. There are no environment variables for any of them.

**Why it matters.** Every run-03 fix that is configuration rather than code is
absent from the run we are about to pay for. `--probation-reset-after 0` in
particular means the learner-arrest mechanism is off.

**Fix.**

1. Add env knobs beside the existing ones (around line 89):
   `PROMOTION_EVERY="${PROMOTION_EVERY:-5}"`,
   `BOOTSTRAP_POLICY="${BOOTSTRAP_POLICY:-auto_first_trained}"`,
   `PROBATION_RESET_AFTER="${PROBATION_RESET_AFTER:-4}"`,
   `REVERT_RESET_AFTER="${REVERT_RESET_AFTER:-3}"`.
2. Change the `GATE_LADDER` default to `"200 600 1000 1500"`.
3. Pass all four in `TRAIN_CMD`, replacing the hardcoded `--revert-reset-after 2`.
4. Document the new knobs in the header comment block (lines 35–52).

**Watch out.** `--probation-reset-after` and `--revert-reset-after` are
soft-gate-only; `ControllerConfig.validate()` raises if either is non-zero in
another mode. The script already forces `soft_gate`, so this is safe, but keep
them together if anyone adds a mode switch.

**Verification.** `games/seven_wonders_duel/test_setup_cloud.py` already checks
the launch command against the parser — extend it to assert each of the four
flags is present with the intended value, and that the ladder default matches
the one in `training_parameters.md`. A doc/launcher drift test is the thing that
would have caught this.

---

## 2. [BLOCKER] The memory-stability acceptance gate is dead

**Where:** `tools/validate_az_memory_stability.py`, `_rows()` at lines 10–12; and
`games/seven_wonders_duel/run_w2_w3_w5_cloud_acceptance.sh` line 33.

**What is wrong.** Both read `run_manifest.json`'s `iterations` list. The
manifest rewrite (item 8 of `7a2d769`) stopped writing rows there, so the list is
permanently empty on a fresh run and frozen at the pre-migration tail on a
migrated one. The validator's `--minimum-iterations 60` check can therefore never
be satisfied, and the acceptance script prints
`memory stability deferred: 0/60 iterations complete` forever.

**Why it matters.** This is the check that would catch an RSS problem on the
cloud box. It is currently a no-op that reports success-by-deferral. This is a
regression introduced by our own change.

**Fix.** Give both the same log-first loading that `tools/az_report.py::load_rows`
already implements: prefer `training_log.jsonl`, fall back to the manifest for a
run whose log was never written. The cleanest version is to import `load_rows`
from `tools.az_report` rather than duplicating it a third time; if that creates
an unwanted dependency, extract it to a small shared helper.

**Verification.** A test that builds a run directory with rows only in
`training_log.jsonl` and asserts the validator sees them; a second with rows only
in the manifest (legacy) and asserts the same. Then re-run the validator against
`runs/laptop_training_03_w7`, which has 210 rows in the log and none in the
manifest — it should report 210, not 0.

---

## 3. [BLOCKER] An allowed HOF change is compared against the original manifest forever

**Where:** `games/seven_wonders_duel/phase_d.py`,
`_refuse_changed_schedules` at line 2448 (specifically
`stored_config = manifest_payload.get("config")`), and `_record_hof_change` at
line 2515.

**What is wrong.** `_record_hof_change` appends to `schedule_changes` but never
updates the effective baseline. The guard always reconstructs `stored` from
`manifest["config"]`, which still holds the run's *original* HOF settings. So
after an accepted `0.0 → 0.15`:

- the next resume with the same, unchanged `0.15` is seen as another change —
  refused without `--allow-hof-change`, and double-recorded with it (a spurious
  second amnesty knot);
- a later genuine `0.15 → 0.30` is logged as `0.0 → 0.30`.

**Why it matters.** Turning HOF on is a one-way door as it stands: every
subsequent resume of that run needs the override flag and pollutes the
provenance the flag exists to protect.

**Fix.** Before comparing, fold the recorded changes into the stored identity:
read `manifest["schedule_changes"]` in clock order and apply each entry's `to`
values over `stored`. Keep `manifest["config"]` untouched — it is the record of
how the run *started*, and overwriting it would destroy the provenance. Then
`changed` reflects a genuine delta against the current effective regime, an
unchanged resume is a no-op, and a second change records `0.15 → 0.30`.

**Watch out.** `_reload_schedule_change_knots` (search that name) already parses the
same list for `at_games`; reuse one parse rather than adding a second reader with
a different tolerance for malformed entries.

**Verification.** Extend the tests added in `test_phase_d.py` under the
`W1.5` heading:

- resume twice with `0.15`: the second is a no-op, records nothing, adds no knot;
- resume `0.0 → 0.15` then `0.15 → 0.30`: the second entry reads `from: 0.15`;
- an unchanged resume **without** `--allow-hof-change` after an accepted change
  must not raise.

---

## 4. The self-anchor's reference series is corrupted at every reset

**Where:** `games/seven_wonders_duel/phase_d.py`, `anchor_reference` at line
1774; `anchor_subject` at line 1831. Related:
`games/az_loop/run_controller.py` line 489 (`if transition.reset_learner:`).

**What is wrong.** W7c indexes the anchor on `candidate_NNNN.pt`, on the premise
that the candidate series *is* the learner's history. It is not. The candidate is
written before the lifecycle transition; on `revert_reset` (and on the new
probation reset) `latest.pt` is then overwritten with `current_best.pt`, so the
candidate for that iteration was **never** the learner in force. Verified on all
three resets in `laptop_training_03_w7`:

```
iter 135  revert_reset  candidate=effcaac7d5f2  latest after=7bf851d5055e
iter 165  revert_reset  candidate=70ecef9a567f  latest after=51343f951469
iter 190  revert_reset  candidate=1421b5c6247b  latest after=bf6b2f01cff0
```

**Why it matters.** The false points are the *rejected, degraded* candidates, so
when the lagged reference lands on one the anchor compares against an
artificially weak opponent and reports an **inflated** score. The observed w7
series (`… 0.760 0.790 0.725 0.570 …`) cannot be read as a strength curve until
this is fixed, and the "is the learner still improving" question the anchor
exists to answer is unreliable near every reset.

**Fix — one of two options.**

*Option A (preferred): record the post-transition learner.* After the lifecycle
resolves, write a small durable record of what `latest.pt` actually is — either a
copy (`learner_NNNN.pt`) or, cheaper, the digest plus a pointer to the file it
equals. Index `anchor_reference` on that record. Cost: one digest per iteration
if pointers are used; one 4 MB copy per iteration if not.

*Option B: reconstruct from the log.* Every row already carries
`latest_sha256`. Walk `training_log.jsonl`, and for each iteration resolve the
checkpoint whose digest matches — `candidate_NNNN.pt` normally,
`current_best.pt` (or the HOF archive of it) after a reset. Cheaper on disk, but
depends on the older `current_best` still existing, which is not guaranteed once
HOF pruning is on.

Take Option A unless disk is tight.

**Watch out.** `anchor_caught_up` (line 1836) guards the degenerate
subject-equals-reference case. After a reset, `latest.pt` *is* `current_best.pt`,
so a reference resolving to the same weights becomes reachable again — the exact
W7a failure mode W7c was meant to remove, re-entered through resets. Whatever
option is chosen, keep that guard and add a test for the post-reset case.

**Verification.** A test that drives a synthetic run through a reset and asserts
the reference for that clock resolves to the restored learner, not the rejected
candidate. Then re-derive the w7 anchor series offline under the new rule and
compare — the three reset-adjacent points should move.

---

## 5. Promotion-gate reuse can substitute stale or pre-reset evidence

**Where:** `games/seven_wonders_duel/phase_d.py`, `anchor_duplicates_gate` at
line 1863.

**What is wrong.** The predicate compares only the *opponent* digest against
`current_best`. It does not check:

- that the gate ran **this** iteration (`last_promotion_gate_iteration` is
  written and never read);
- that the gate's **subject** still matches `anchor_subject()`;
- that `report.games == self_anchor_games`.

Because `run_controller.py:489` resets `latest.pt` **before**
`_measure(iteration)` at line 500, a reused gate on a reset iteration describes
the *rejected pre-reset candidate* while the anchor claims to measure the
restored learner. On non-gate iterations, `last_promotion_gate` still holds an
older gate and can be reused wholesale.

**Why it matters.** This is worse under the cloud configuration than it was on
the laptop. In w7 the gate and anchor cadences coincided (both every 5
iterations / 2,000 games), so the stale case never arose. The cloud defaults are
`--self-anchor-every-games 10000` against gates every 2,500 games — they do not
align, so the anchor will routinely run on non-gate iterations.

**Fix.** Require all four conditions before reuse: same iteration, subject digest
equal to the current `anchor_subject()`, `games` equal to `self_anchor_games`,
and opponent digest equal to the reference. Record the gate's subject digest
alongside `last_promotion_gate` at the point it is stored (`self.last_promotion_gate = report`, line 4002). If any condition fails, play the match.

**Alternative worth considering.** Reuse is an optimisation whose value fell
sharply once the anchor was retargeted — the collision is no longer structural.
Deleting `anchor_duplicates_gate` entirely is a defensible fix and removes a
class of bug; the cost is one extra fixed-N match on the rare coinciding
iteration.

**Verification.** Tests for each rejected condition — stale iteration, mismatched
subject, mismatched games count — plus one that a reset iteration never reuses a
gate whose subject was the pre-reset candidate.

---

## 6. Revert amnesty can trigger the probation reset it was meant to prevent

**Where:** `games/az_loop/run_controller.py` line 462
(`gate_decision = promotion.decision`) into `decide_transition` at line 466;
decision produced in `games/seven_wonders_duel/phase_d.py`
`wilson_pair_decision`, line 1511 (`revert_suppressed_knot`).

**What is wrong.** When a knot suppresses a revert, the decision returned is a
plain `"continue"`; the distinguishing `stop_reason="revert_suppressed_knot"`
survives only in `promotion.metrics`. The controller passes only the decision
string, so the lifecycle counts the gate as an ordinary probation and increments
`probations_since_decisive`. At `N-1` probations, the amnestied gate immediately
performs `REVERT_RESET`.

**Why it matters.** The amnesty exists so a distribution shift — the curriculum
ending, the draft prior ending, HOF switching on — cannot punish the learner.
As written it can *cause* the harshest lifecycle action instead. This lands
directly on the HOF-enablement path, which is the scenario item 3 unlocks.

**Fix.** Thread the stop reason into the transition. Either add an optional
`gate_stop_reason` parameter to `decide_transition`/`gate_transition`, or
introduce a distinct decision value for a suppressed revert. On that outcome:
take the probation *action* (do not promote, do not roll the generator back) but
leave **both** counters unchanged — it is explicitly a non-measurement.

**Verification.** A test asserting that a suppressed revert at
`probations_since_decisive == N-1` does not reset the learner and does not
advance either counter, and that the row still records the stop reason so the
amnesty is visible.

---

## 7. Remove the slope trigger from lifecycle decisions

**Where:** `games/az_loop/stagnation.py` line 174
(`if slope is not None and slope <= self.slope_epsilon:`), and
`slope_epsilon` on `StagnationDetector`.

**What is wrong — this is a design error, not a tuning error.** For a fixed lag
`L`, the anchor measures `S(t) − S(t−L)`. A learner improving at a *steady* rate
has a **constant** difference, therefore a flat anchor score, therefore a slope
of approximately zero — and is declared stagnant. Real learning curves
decelerate, so a healthy run trends *negative*. The trigger tests acceleration,
not whether learning continues. Because it is OR'd with the interval trigger, a
run whose anchor is confidently above 0.5 is still declared stagnant.

Observed in `laptop_training_03_w7` iterations 150–209: **every** STAGNANT
verdict came from this trigger while the interval trigger never fired once.

**Why it matters.** With `--intervention-ladder` on, this escalates schedule
changes on a statistic that fires on healthy runs.

**Fix.** Drop the slope from the stagnation verdict. Keep the **interval
trigger**, which is self-calibrating: it uses Wilson bounds and so already
accounts for sample size. Continue to *compute and report* the slope in the row
and heartbeat — it is useful telemetry — but it must not contribute to
`stagnant`.

If a second trigger is wanted, the right shape is an **absolute lagged-advantage
test**: is the anchor score's lower bound above 0.5 by a meaningful margin,
sustained over several measurements. That tests "still ahead of my past self",
which is the intended question.

**Also correct the documentation this invalidates.** The cadence/noise analysis
currently in `stagnation.py`'s `slope_epsilon` docstring and in
`training_parameters.md`'s Detection section is a *variance* argument about a
statistic that should not be in the decision at all. Replace it rather than
leaving both explanations side by side.

**Verification.** A test constructing measurements from a steadily-improving
learner (constant score well above 0.5, near-zero slope) and asserting the
verdict is **not** stagnant — that test fails today.

---

## 8. Retract the HOF policy-filter claim

**Where:** claim text in the `7a2d769` commit message,
`training_parameters.md` under `--hof-opponent-fraction`, and
`RUN03_FIXES_REVIEW_REQUEST.md` §5. Code:
`games/seven_wonders_duel/seven_wonders_rust/src/self_play.rs:1461`;
`games/seven_wonders_duel/dataset.py` (`archive_policy_seats` usage);
`games/seven_wonders_duel/buffer.py` (`archive_policy_seats`).

**What is wrong.** We claimed that `--hof-opponent-fraction 0.15` would have
trained the learner to imitate the archive on ~7.5% of positions. **That is false
for every path that can produce a league game.** The Rust scheduler already sets

```rust
policy_excluded: !meta.full || self.cfg.net_by_player[meta.actor] != 0,
```

in `finish_move`, with a comment making the same argument. Path check:
`_tag_league_opponents` is called only from `_generate_iteration_rust`, which
uses `self_play_many_flat_net` → `run_many_pipelined_sharded` → `finish_move`.
The one unguarded site (`self_play.rs:478`, `fn run`) is reachable only from two
single-game entry points in `lib.rs`, one of them `MockEval`.

**What to do.** Keep the code — `archive_policy_seats` is correct, and is genuine
defence for imported/legacy buffers and the Python generation backend. Change the
*description* everywhere: it is defence in depth for non-Rust and imported
records, **not** a fix to the live generation path. Add a line to the docstring
naming `self_play.rs:1461` as the primary enforcement point so the next reader
does not re-derive this.

**Process note worth recording.** The claim was made after reading `dataset.py`,
`phase_d.py`, `buffer.py` and `training_adapter.py`, and stopping at the
language boundary. Any future claim about generation behaviour has to read the
Rust.

---

## 9. Rename `lr_warm_restart`, and consider a reset rung

**Where:** `games/az_loop/stagnation.py` line 225, `DEFAULT_LADDER` rung 3.

**What is wrong.** The rung multiplies the configured LR by 3 and nothing else.
`train_steps` reloads the existing AdamW moments, treats the optimizer as warm
and therefore **skips warmup**, and Phase D never enables `cosine_decay`. So it
is not a warm restart and does not test "the optimiser is stuck" — it tests
whether tripling the LR on top of accumulated moments helps.

**Fix.** Either rename it (`lr_jump`) and correct the rationale string, or
implement a real restart by explicitly clearing optimizer state and re-enabling
warmup for that rung. Renaming is the honest minimum.

**Related, and worth a decision.** Reset-to-`current_best` is the only
intervention in this codebase with direct evidence of working — three times in
w7 (iterations 135, 165, 190), each followed by a promotion within five
iterations. It is not on the ladder at all. Consider adding it as an early rung,
ahead of the LR jump. Note the confound before relying on it: the post-reset gate
compares a learner that was just re-initialised *from* its opponent, so some of
the effect may be a measurement artifact.

---

## 10. Soften the claims that outran their evidence

Documentation only; no code.

**Divergence vs cause.** 0.335 over 100 paired observations supports "the
candidate was materially worse than the best it started from" and nothing about
*why*. Monoculture (curriculum, HOF, seed retention and draft prior all zero from
iteration ~45) and the soft-gate feedback loop are **hypotheses**, never tested.
Update `RUN03_FIXES_REVIEW_REQUEST.md` §1 and the commit narrative accordingly.

**Memory attribution.** We claimed the manifest transient caused the ~8.7 MiB/iter
RSS creep. Newly measured: `store.iterations()` still allocates a **+261 MB**
transient per iteration against the old manifest path's **+320 MB** — comparable —
yet the creep fell to +0.70 MiB/iter. That undercuts the "large transient raises
the RSS floor" mechanism. The more likely cause is the ~380 MB/iteration of
**file I/O** the manifest also performed. State the fix as empirically effective
with the mechanism unresolved.

**HOF level-vs-position.** `hof_start_games` is plainly positional and was
allowed under a different argument than the level one. All three fields create a
forward regime boundary; recording it does not make metrics comparable across it.
Say so in the `--allow-hof-change` documentation, and note that consumers must
segment at the boundary.

**HOF value targets.** Keeping the archive's value labels is a plausible
experiment, not a correctness result — the league trajectory follows a mixed
policy. The settling experiment is three arms: all league values / learner-turn
values only / no league values.

---

## 11. [DEFER] `store.iterations()` is still O(n²)

**Where:** `games/az_loop/run_controller.py` — called at lines 150, 359, 373 and
930; implemented by `_PhaseDRunStore.iterations()` in `phase_d.py`, which parses
the whole of `training_log.jsonl`.

**Measured, so the trade is explicit:** 1.02 s and a +261 MB transient per call
on the 100 MB / 210-row w7 log — about **0.13%** of a 12.6-minute iteration, and
~1.8 min across a 210-iteration run. Real, quadratic, and currently negligible.

**Defer the performance fix**, but note the transient is the interesting part —
see item 10. If it is done: load once in `initialize()` and append committed rows
to an in-memory list, or keep only the aggregates the four call sites need
(resume point, totals, promotion counts). Any change must preserve the
crash-recovery property that the log on disk is the source of truth.

---

## Suggested order

| # | Item | Blocking? | Rough size |
|---|---|---|---|
| 1 | Cloud launcher flags | **yes** | config + one test |
| 2 | Memory validator row source | **yes** | small |
| 3 | HOF baseline folding | **yes** | small, 3 tests |
| 4 | Post-transition learner record | before trusting the anchor | medium |
| 5 | Gate-reuse guards (or delete reuse) | before trusting the anchor | small |
| 6 | Amnesty must not count as probation | before enabling probation reset | small |
| 7 | Remove slope trigger | before enabling the ladder | small + doc rewrite |
| 8 | Retract HOF claim | doc only | small |
| 9 | Rename `lr_warm_restart` | doc/naming | small |
| 10 | Soften overreaching claims | doc only | small |
| 11 | `store.iterations()` | no | deferred |

Items 4 and 5 are the same underlying problem — no durable record of what the
learner actually was after the lifecycle ran — and are best done together.

Until 6 and 7 land, keep `--intervention-ladder` off and
`--probation-reset-after 0` on any long run: the ladder escalates on a trigger
that fires on healthy runs, and the reset can be triggered by the amnesty
designed to prevent it.
