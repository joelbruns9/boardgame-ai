# Implementation prompt — Depth-conversion probe (fitness discriminator)

Context: `games/kingdomino/NNUE_PROJECT_PLAN.md`, section "P0 results + eval-quality pivot
(2026-07-15)". Read it and the two completed reports
(`runs/kingdomino/nnue_loop/clock_scaling_p0_step1.json`,
`competence_floor_p0_step2.json`) before starting. This is **step 1 of the re-sequenced
critical path** and runs **before** the eval diagnosis and any retrain.

## The one question this answers

Does **search depth convert to playing strength at all in Kingdomino**, for an eval that is
merely *decent* (not broken)? Everything downstream forks on the answer:

- **If depth converts** (deeper clearly beats shallower for a fixed decent eval) → the
  regime is searchable, the pilot's 0-8/not-generation-ready failure is *purely an
  eval-quality problem*, and a retrain is worth it.
- **If depth barely converts** → Kingdomino is structurally eval-dominated (scoring horizon
  beyond reachable depth), standalone depth-based superiority is out for this design, and
  effort routes fully to eval-as-generator + exact endgames + curriculum. That is a
  project-shaping result worth the cheap probe.

## Why a FIXED decent eval, not the NNUE

The clock-scaling test (step 1) already varied budget for the *NNUE* eval and saw a flat
curve — but that's uninformative about depth, because the pilot eval is below trivial and
**deeper search over a broken eval walks toward the positions that eval misjudges most**
(search amplifies eval error). To measure depth's value you must hold a *directionally
correct* eval fixed. Use `pick_aware` (margin + claimed-crown potential) — the same eval
that beat the pilot 25%-for-NNUE under identical search in step 2, and beat Greedy in Phase
0. It is weak but points the right way, so added depth has a fair chance to help.

This is a controlled ablation: **independent variable = search depth; held constant = the
eval (`pick_aware`); measured = playing strength via paired games.** Any strength difference
across depths is attributable to depth alone.

## What "depth" is here (so the runs are set up right)

In `RustExpectiminimax._value(eng, depth, ...)` each decision layer decrements `depth` by 1;
depth 1 = one ply of lookahead (greedy-on-eval, no opponent reply), depth 2 adds the
opponent's reply, depth 3 adds my counter. Note (from step 1 telemetry) depth ~3 does **not
cross a round boundary** (`chance_share ~1.5%`), so this measures *within-row tactical
depth*. State that caveat in the report: if depth doesn't help even within a revealed row,
the structural verdict is strong; if the benefit would only appear across round boundaries
(depth we can't reach), that itself answers the fitness question.

## Experiment

Use **fixed depth**, not a wall clock, so depth is the clean controlled variable (not
confounded by nodes/second). Two feasible drivers — pick one and record it:
- `RustExpectiminimax(depth=N, eval_fn=pick_aware)` (fixed-depth oracle), or
- `OperationalRustSearchBot(eval="pick_aware", max_depth=N, max_secs=<generous>)` driven so
  it always completes depth N (report `completed_depth` to confirm it did).

Reuse the paired, seat-swapped runner (`nnue/match.py` `run_paired` / `Participant` /
`round_robin_eval.play_game`) — do not write a new match loop.

Pairings (all paired seeds, both seats):

1. **`pick_aware`@d1 vs @d2 vs @d3** (self-play across depths). The core measurement: does
   more depth beat less depth for the same eval? Report each higher-vs-lower pairing's points
   rate + Wilson interval and avg margin. (Add @d4 only if affordable.)
2. **`pick_aware`@d(best) vs AZ** (current-best checkpoint from step-1 settings, at a couple
   of sim budgets, e.g. a mid and a high one). This reads the *ceiling* of "cheap decent eval
   + feasible depth" against the strong reference — how much of the AZ gap can depth + a
   decent eval close.
3. **(optional) `pick_blind`@d1..d3** (margin only) as a second directional eval. If depth
   helps `pick_aware` but not `pick_blind`, that localizes which eval content depth can
   leverage.

Use enough paired seeds for usable intervals (step 1's 4-8 was fine for a directional flat
call but too few to *rank* depths); target ~24-50 paired seeds per pairing and record the
count. `pick_aware` is trivial to compute so d3 over dozens of seeds is affordable.

## Deliverable

A driver (e.g. `games/kingdomino/nnue/depth_conversion.py`) + JSON report
(`runs/kingdomino/nnue_loop/depth_conversion.json`) that includes:

- Per-pairing points rate + Wilson LCB/UCB, avg margin, seed count, and the completed depth
  actually reached (confirming fixed-depth completion).
- A **fitness verdict** field with an explicit classification, e.g.
  `depth_converts` / `weak_conversion` / `no_conversion`, with the reasoning and the
  monotonicity of the depth curve.
- The vs-AZ ceiling result and how much of the gap depth+decent-eval closes.
- The within-row-depth caveat above, stated explicitly.
- Provenance mirroring prior steps (eval names, driver/depths, AZ checkpoint sha + sims,
  seed starts, `reserved_test_split_opened: false`).

## Invariants / scope

- Encoder order-blind; chance handled inside the tree; no K=1 / representative-row shortcut;
  full-width (no selective pruning — that's the already-rejected knob and would confound the
  measurement). Reserved test split stays closed.
- Do **not** train anything, run AZ reanalysis, do Step 0 cascade alignment, or start the
  eval diagnosis in this task. This produces a fitness read only.
- Add a focused test for the driver/aggregation logic in the style of the step-1/step-2
  telemetry tests.

## What the result routes to (state in report, don't act on it here)

- `depth_converts` → proceed to the eval-quality diagnosis
  (`NNUE_P0_EVAL_DIAGNOSIS_PROMPT.md`), then a retrain gated on beating `pick_aware` under
  matched search. Standalone strength remains live.
- `weak_conversion` / `no_conversion` → standalone depth-based superiority is structurally
  unlikely for this design; re-weight to eval-as-generator + exact endgames + AZ curriculum.
  An eval retrain still matters (for the generator/relabeler roles), but not in pursuit of
  out-searching AZ.

Report the verdict and the recommended fork; do not begin either branch in this task.
