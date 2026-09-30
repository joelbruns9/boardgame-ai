# Decision pilot

`decision_pilot.py` compares adaptive search at separate horizon/budget settings
on the same saved decisions. This measures decisions and cost; changed choices
still need independent audits before any claim of stronger play.

## Run the pilot

Run from the Can’t Stop worktree. It shares the main checkout’s Python environment:

```powershell
Set-Location C:\Users\joeld\projects\boardgame-ai-cantstop
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.cantstop.decision_pilot --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --budgets 32 128 512 --horizons 1 2 --suite runs/decision_pilot_timing_20260930.positions.json --out runs/decision_pilot_20260930.json
```

This uses the 60 ordinary positions already collected on 2026-09-30: six
stage/phase buckets across all ten variants. It runs 360 comparisons. A measured
sample of ten positions (all variants, all six settings) took 234.86 seconds of
search work, or about four minutes including startup and collection. Linear
projection is about 24 minutes; allow **20–40 minutes** on the same GPU. Close
decisions and other GPU activity can change this. No full pilot or strength
claim is implied by that timing sample.

To collect fresh positions, omit `--suite` and use a new output name:

```powershell
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.cantstop.decision_pilot --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --budgets 32 128 512 --horizons 1 2 --positions-per-bucket 1 --games-per-variant 2 --out runs/decision_pilot_fresh.json
```

Collection plays complete games with the unmodified baseline, reusing one turn
table per turn. It took 4.85 seconds for twenty games in the timing run. Sampling
has its own stream so changing reservoir size does not change the game dice.
Each variant is sampled separately into early/middle/late × dice/stop-roll
buckets. Early means no saved claims, late means somebody needs one more claim
to win, and middle means between these. Terminal and single-action decisions
(including winning banks) are excluded. Duplicate boards are deduplicated before
sampling. Missing buckets are reported, never filled using constructed fixtures;
increase `--games-per-variant` if coverage is short. The sample is stratified
baseline play, not an estimate of natural position frequencies or arena strength.

## Settings and comparisons

- `--budgets 32 128 512` creates **three independent budget-ceiling arms**.
  Their adaptive schedules are `(32,)`, `(32,128)`, `(32,128,512)`. The paired-t
  correction accounts for each arm’s own declared looks, so a larger-ceiling
  arm can continue past a stage where the smaller arm resolved.
- Horizons share position-specific seeds and sample prefixes, as do budget
  arms. H finishes the current turn and then H additional player turns.
- Dice-luck correction and common random numbers are enabled by default.
  Paired-t intervals remain approximate for game payoffs.
- Continuation defaults to the baseline. Optional stronger continuation uses
  `--early-turns 1 --early-expansions 1`. Selective depth stays fixed at one;
  rollout H=2 is supported independently. The timing estimate is for baseline
  continuation.
- Serial execution, rotated arm order, and model warmup precede decision timing.
  Omit `--seconds` for the measured fixed-work experiment. A supplied soft time
  limit defines a different budgeted policy and is recorded with its fallbacks.

## Saved results and resumption

The output JSON includes baseline and selected actions, changed choices,
unresolved/fallback reasons, committed samples, NN rows, per-arm median/p95
latency, summaries by variant/stage/phase, paired horizon and adjacent-budget
choice differences, and changed-position IDs for later audits. It records model
and code hashes, settings, seeds, suite hash, and invocation times. Raw samples
and per-pair interval diagnostics remain in sibling `.states` checkpoints.
Generated snapshots and collection coverage are saved to `.positions.json`.
Supplied suites use development rows only and exclude trivial decisions.

The runner saves after each comparison and marks errors incomplete before
raising. Use one writer per output/state directory. Resume with the **same
command plus `--resume`**; completed comparisons are skipped and an interrupted
comparison can continue its committed stages. Changes to code, checkpoint,
settings, or suite invalidate resume. Resumed comparison latency is explicitly
marked as excluding work done in earlier invocations; invocation wall times are
retained separately.

For a bounded timing sample, add `--max-cells 60`. Resume without that option to
finish, or set another limit on newly completed comparisons. `--collect-only`
saves the positions and an empty report; resume without it to start searching.
`--position ID` can restrict a supplied suite and is repeatable.

The timing report is `runs/decision_pilot_timing_20260930.json`. It deliberately
contains only 60 of 360 comparisons. Reporting-only additions were finalized
after timing; its recorded source identity is retained. The command above
starts a new report with the final code while reusing its saved positions.

Validation: 10 focused pilot tests and 210 existing targeted search tests passed.
A final-code CUDA smoke run completed all six settings by resuming after one
comparison, with no duplicate completed cells.
