# Independent decision audit

The audit freezes the pilot’s decisions and checks their value with independently
seeded **terminal full-game rollouts** under the unchanged baseline continuation
policy. The same checkpoint supplies continuation decisions; no NN endpoint
value is used in the final payoff. This is a small policy-conditioned decision
audit, not arena evidence.

The 2026-09-30 audit includes every position changed by any pilot arm (two),
every additional position unresolved at the largest ceiling (one), and one fresh
control matching each focus position’s rule variant, claim-based stage, and
decision type. Controls come from newly seeded ordinary baseline games and are
checked against the entire pilot suite for duplicate snapshots.

The plan is written to the report before drawing audit samples:

- Initially 1,024 samples per legal action for all six positions.
- Extend uncertain **focus** positions to 4,096 per action. A changed position
  extends if any frozen changed action’s interval versus baseline spans zero.
  An unresolved focus position extends if an alternative versus baseline is
  uncertain. Controls stay at 1,024.
- All actions are evaluated with aligned common dice and dice-luck correction.
  Fresh position-specific audit seeds are checked against all pilot search
  seeds; audit streams never consume pilot samples.
- Confidence uses approximate paired Student-t intervals on corrected samples,
  with Bonferroni adjustment over all six planned action pairs and both planned
  looks. No classification is made at intermediate checkpoint batches.
- Raw binary terminal payoffs, raw paired intervals, and conservative raw
  Hoeffding intervals are retained as diagnostics. Corrected samples are not
  clipped; sample ranges and counts outside [0,1] are recorded. Approximate
  t coverage is not guaranteed for arbitrary nonnormal game distributions.
- Corrected interval above zero: `helps`; below zero: `hurts`; spanning zero
  (or degenerate observed variance): `uncertain`. These labels concern only the
  declared fixed continuation policy. Control alternatives are diagnostic
  comparisons, not frozen pilot changes.

Run from the Can’t Stop worktree:

```powershell
Set-Location C:\Users\joeld\projects\boardgame-ai-cantstop
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.cantstop.decision_audit --pilot runs/decision_pilot_20260930.json --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --budgets 1024 4096 --batch 128 --seed 2026100101 --out runs/decision_audit_20260930.json
```

The report records checkpoint and code hashes, the input pilot’s content hash,
frozen choices, seed/estimator settings, control collection, per-pair uncertainty,
and batch costs. Full per-action raw and corrected vectors are in sibling
`.samples` JSON files; the saved audit positions are in `.positions.json`.
A human-readable table is updated in `.md` after each planned look.

Samples commit transactionally after each batch. A failed or interrupted batch
does not advance the saved sample prefix. The report becomes incomplete and
retains the current batch before raising; no failed rollout is silently dropped.
Resume using the identical command plus `--resume`. Completed samples and looks
are skipped. Changes to inputs, settings, source, or checkpoint invalidate resume.
Use one writer per output/sample directory.

Validation includes oriented help/harm contrasts, multiplicity, degenerate and
nonfinite streams, terminal-outcome checks, focus selection, extension decisions,
and failure/resume prefix preservation. Existing full-game and variance-reduction
regressions also pass. The initial throughput measurement was 16 samples per
action on the two-player blocking opening: 32 terminal games in 5.88 seconds.
Actual 128-sample batches have taken approximately 50–60 seconds there; later
positions and controls can have different game lengths.

## Completed 2026-09-30 result

The run completed and its saved samples and intervals were independently
recomputed and verified. All 30,720 rollouts reached terminal outcomes; no
failed samples were excluded. Search took 3,754.92 seconds and total run wall
time was 3,756.80 seconds (62.6 minutes), evaluating 1,736,204,226 NN rows.

Both frozen pilot changes remain uncertain at 4,096 samples per action:

| Focus comparison | Estimated gain, percentage points | Approximate corrected interval |
| --- | ---: | --- |
| 2p blocking: roll vs stop | +0.059 | [-0.292, +0.410] |
| 4p nonblocking: column 9 vs 6 | -0.243 | [-0.724, +0.237] |
| 4p blocking unresolved case: column 7 vs 10 | +0.124 | [-0.135, +0.383] |

The three fresh controls favor their baseline actions by 4.659, 1.549, and
0.812 percentage points; their corrected intervals exclude zero. Raw outcomes
and corrected estimates agree that neither frozen change is established helpful.
The audit does not establish harm either. These are policy-conditioned,
approximate sampling statements, not arena strength results.

Evidence from this small audit is insufficient to promote either search setting
or justify a large arena. A broader decision pilot or a bounded stronger early
continuation experiment at selective depth one is the next useful quality test.

Results: `runs/decision_audit_20260930.md` and `.json`; verification summary:
`runs/decision_audit_20260930.validation.json`. All samples and control snapshots
are retained. Eight audit tests and 67 existing rollout/variance checks passed.


## Batched full-game continuation

The audit now supports `--backend pool`. It preserves the root table, v1 shared
dice, terminal payoffs, within-turn correction and all-or-nothing batch failures.
The default remains serial; stronger selective continuation is not supported by
the pool. See `AUDIT_POOL.md` for implementation, validation and the measured
3.39x speedup.

Because the adapter required rebuilding the native extension, an audit of the
original pilot also needs `--allow-native-rebuild`. This permits only a recorded
native hash change and rechecks every pilot baseline before drawing samples.
Python source/model/suite checks and resume identities remain strict. All 60
original choices and values matched exactly in the end-to-end CLI check.
Use a new output for the new backend. Previous audit results are unchanged.
