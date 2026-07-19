# Implementation prompt — Eval-quality diagnosis (static/shallow-evaluator focus)

Context: `games/kingdomino/NNUE_PROJECT_PLAN.md`, section "P0 results + eval-quality pivot
(2026-07-15)" including the depth-conversion UPDATE block. Read it and the three completed
reports before starting:
`runs/kingdomino/nnue_loop/{clock_scaling_p0_step1,competence_floor_p0_step2,depth_conversion}.json`.

This is the **NEXT** step of the re-sequenced critical path. It runs after the
depth-conversion probe (DONE, verdict `no_conversion`) and before any retrain.

## What is already settled — and how it reframes this step

- **Step 1 (flat vs AZ), Step 2 (pilot not_generation_ready): the failure is EVAL CONTENT.**
  Under identical search the pilot loses to the trivial `pick_aware` eval.
- **Depth-conversion probe: `no_conversion` — depth is COUNTERPRODUCTIVE** (d3 < d1 even for
  the decent `pick_aware` eval; minimax pathology). So the eval's value is **NOT** as a
  deep-search enabler. Its value is as a **strong STATIC / 1-ply evaluator** for the roles
  that survive: generator, relabeler, and AZ curriculum, plus feeding shallow play.

Therefore this diagnosis judges the eval **as a static/1-ply evaluator**, not as something
that has to enable deep search. Do **not** re-run depth-conversion experiments — that question
is answered. The gate any retrain must clear is now **shallow**: clear Greedy and be
competitive with `pick_aware` at d1-d2.

## The question this answers

**Why is the pilot NNUE eval below a trivial heuristic, and which fix does that imply?** The
cause must be localized among four non-interchangeable possibilities before spending on a
retrain (a retrain aimed at the wrong cause is wasted):

1. **Bug / scaling / frame** — quantized-vs-float divergence in the deployed path, an eval
   sign/frame error, or an output-range/target-scaling mismatch. Fix: correct it; no retrain.
2. **Undertraining (scale)** — directionally right but noisy from only 50k positions. Fix:
   retrain on the full existing buffers (~1M positions), same distribution.
3. **Off-distribution generalization** — fine on AZ-realized states, unreliable on
   search-reached states. Fix: training data must cover search-reached states, not just more
   AZ-distribution data. (This is the leading hypothesis given the static-val-vs-play gap.)
4. **Capacity / representation** — the architecture can't represent Kingdomino value well
   enough (last resort; require evidence, and note AZ's value head proves a strong static
   evaluator exists).

## Experiments (cheap, no training)

Build a small reusable eval-accuracy probe (e.g. `games/kingdomino/nnue/eval_diagnosis.py`).
Reuse: NNUE eval via `RustSearch(eval="sparse_nnue_q"/"sparse_nnue")`, `pick_aware`/`pick_blind`
via the same `RustSearch`; exact ground truth via the endgame solver; the pilot artifact + AZ
checkpoint paths/shas are in the prior reports' `settings`. There is no eval-accuracy harness
yet (`benchmark_sparse.py` is throughput only) — this builds it.

### A. Quantized-vs-float parity in the *deployed* path (rule out bug #1 first — cheapest)

On a few hundred real positions spanning phases, compare `sparse_nnue_q` (what steps 1-2/‌the
depth probe would use) against the float `sparse_nnue` oracle for the pilot artifact:
expected-score and margin MAE/max, and root-action agreement at d1-d2. Parity was claimed at
build; **confirm it still holds for this exact artifact in this deployment.** A parity break
here would by itself explain the collapse. Also spot-check the eval sign/frame is consistent at
non-root (minimizing) nodes.

### B. Static (depth-0/1) eval accuracy vs ground truth, stratified — the CORE of this step

Assemble a labeled probe set with ground-truth values, split by whole game/seed, never
touching the reserved test split:
- **Exact-labeled:** late/endgame positions the exact solver finishes (official-outcome cascade
  value — see Step-0 note below).
- **Outcome-labeled:** positions from realized games with their honest final outcome/margin.
- Stratify each by phase (opening/mid/end) and by **on-distribution vs off-distribution**:
  AZ-realized states vs states reached by NNUE/pick_aware operational search (replay a handful
  of prior games and sample the search-reached states actually evaluated).

For the pilot NNUE eval, `pick_aware`, and the AZ value head, report per stratum, **as static
evaluators**:
- value error (Brier for outcome, MAE for margin/expected-score) vs ground truth;
- **ranking quality**: on positions with several legal moves, does the eval rank moves
  correctly vs the exact/high-budget reference (rank correlation / top-1 agreement)? This is
  the property that actually matters for a 1-ply generator.

The decisive contrast: **does the pilot NNUE beat `pick_aware` as a static evaluator on
ON-distribution positions but lose OFF-distribution?** If yes → cause #3 (off-distribution),
the leading hypothesis. If it loses even on-distribution at depth 0/1 → cause #2 or #4 (weak
content / capacity). Also compare against the AZ value head as the strong-static reference: how
far below AZ is the pilot, and is the gap uniform or concentrated off-distribution / by phase?

### C. (small, optional) confirm the eval doesn't rescue itself with depth

One quick check only: verify the NNUE eval, like `pick_aware`, does not improve d1→d3 (expected
given `no_conversion`). This is a sanity confirmation, not a focus — do not expand it into a
depth study.

## Step-0 dependency note (for the exact-label stratum)

Ground-truth endgame labels must use the official largest-territory/crowns cascade. The legacy
`solve_endgame_ab` optimizes RAW margin and mis-ranks score ties; the NNUE-side exact search
already uses the official-outcome objective. Use the cascade-correct exact path for labels here,
and record which terminal rule produced each label. (Full Step-0 alignment across the AZ side is
still separately required before label PRODUCTION for the curriculum, but for this diagnosis just
route exact labels through the cascade-correct search.)

## Deliverable

A JSON report (`runs/kingdomino/nnue_loop/eval_diagnosis.json`) + printed summary that:
- states the parity result (A), the per-stratum static accuracy/ranking tables (B), and the
  optional depth sanity check (C);
- gives an explicit **cause classification** (bug / undertraining / off-distribution / capacity)
  with the evidence;
- gives a **concrete recommended fix** naming the exact next action, e.g.:
  - bug → the fix, then re-run the Step-2 competence floor;
  - undertraining → retrain on the full ~1M AZ buffer, whole-game splits, gate = clear Greedy +
    competitive vs `pick_aware` at d1-d2;
  - off-distribution → generate search-reached states (self-play of the current search), label
    with honest outcome + exact tails, mix with existing buffers, retrain, same shallow gate;
  - capacity → the specific architecture/feature experiment and why.
- provenance mirroring prior steps (artifact/checkpoint shas, seeds, budgets,
  `reserved_test_split_opened: false`).

## Invariants / scope

- Same pilot artifact; encoder order-blind; no K=1 shortcut; reserved test split stays closed.
- Do **not** train, run AZ curriculum reanalysis, or start the retrain in this task. The
  pre-registered gate for the retrain that follows is a **shallow-play** competence gate (clear
  Greedy + competitive vs `pick_aware` at d1-d2), never lower validation loss alone.
- Add a focused test for the probe/aggregation logic, in the style of the prior telemetry tests.

Report the cause and recommended fix; do not begin the fix in this task.

---
Note: the **exact endgame relabeler** is an eval-independent parallel win elevated by the
`no_conversion` result; it is gated on Step-0 cascade alignment, not on this diagnosis, and can
be pursued concurrently (see the plan's "Parallel, eval-independent track").
