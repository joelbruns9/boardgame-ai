# Implementation prompt — NNUE Package P0, Step 2: competence floor for generation

Context: `games/kingdomino/NNUE_PROJECT_PLAN.md` ("Immediate work package P0", step 2) and
the completed step-1 report `runs/kingdomino/nnue_loop/clock_scaling_p0_step1.json`. Read
both before starting. This prompt implements **P0 step 2 only**.

## Why this is next (read before scoping)

Step 1 is done and the NNUE-vs-AZ gap is **flat**: the pilot lost every paired game at
0.1/0.5/2/10s, margin pinned near −80 to −90, while mean completed depth rose only
1.93→3.27. Crucially, `nnue_chance_share ≈ 1.7–1.9%` — chance handling is **not** the
bottleneck, so Work Package A (stochastic allocation) is not the lever and is not what
this step is about. The flat curve is the plan's expected modal outcome and it **shifts
the project onto the primary practical track (AZ curriculum), which does not require NNUE
to beat AZ.** What it requires is a generator that is *competent and divergent* enough to
produce **legal, coherent, nontrivial** positions worth reanalyzing.

Therefore the urgent open question the 0-8 result raises is: **is the pilot even competent
enough to be a curriculum generator at all?** That is exactly step 2. Whole-game
superiority to current AZ is explicitly **not** required here; the bar is much lower.

Do **not** start: step-0 cascade alignment (it does not gate this gameplay measurement),
steps 3–7, Work Packages A/B/C, the Middle-Kingdom LMR experiment, or any retraining. This
step produces a competence read, nothing downstream.

## What exists already (reuse, do not rebuild)

- `games/kingdomino/round_robin_eval.py` — `Participant`, `play_game`,
  `build_open_loop_checkpoint_participant(...)` (use it for both a **mid-strength** and,
  if useful as a ceiling reference, the current-best AZ checkpoint), `build_participants`.
- `games/kingdomino/bots.py` — `RandomBot`, `GreedyBot`.
- Pick-aware search baseline = `OperationalRustSearchBot(eval="pick_aware", ...)`
  (`rust_expectiminimax.py`) — the plan's "pick-aware search" opponent, i.e. the same
  operational searcher but with the hand-crafted `pick_aware` eval instead of the NNUE.
- NNUE pilot participant = `nnue_participant(...)` in `nnue/match.py`; the pilot artifact
  and AZ checkpoint paths/shas are recorded in the step-1 report `settings` block —
  **use the same pilot artifact** (`sparse_v3_pilot.knnue`).
- The paired, seat-swapped, clock-accounted runner `run_paired` / `TimedBot` in
  `nnue/match.py` (now with per-move telemetry aggregation) — reuse it; do not write a new
  match loop.

## Deliverable

A driver (e.g. `games/kingdomino/nnue/competence_floor.py`) plus a JSON report answering:
**at a generous offline NNUE budget, is the pilot a competent generator** — does it clearly
beat Greedy and pick-aware search, stay respectable against a mid-strength AZ checkpoint,
and produce legal, coherent, phase-diverse positions?

### Match design

- Give the NNUE a **generous offline budget** (this is the generation budget, not a
  matched-clock test — e.g. 2s/move or a fixed generous node/depth cap; record it). The
  opponents run at their natural settings. Clock-matching is *not* the point here; note
  each side's actual decision time for the record but do not gate on equality.
- Opponents, each a separate paired series (both seats, paired seeds):
  1. `RandomBot` — sanity floor; the pilot must dominate.
  2. `GreedyBot` — the plan's basic bar.
  3. Pick-aware operational search (`eval="pick_aware"`, same search settings/budget as the
     NNUE side so the comparison isolates eval quality, not search).
  4. A **mid-strength** AZ checkpoint (not the mature current-best). Pick an earlier
     checkpoint from the run history; record which. Optionally also include current-best as
     a ceiling reference, clearly labeled — but the competence bar is the mid-strength one.
- Use enough paired seeds that Greedy/pick-aware results have a usable Wilson interval
  (step 1's 4–8 games was fine for a directional flat call but is too few to *rank*
  competence). Target on the order of 24–50 paired seeds per opponent; record the count.

### Coherence / legality instrumentation (required — this is half the step)

The plan requires reporting "illegal/replay failures, score/discard distributions, and
phase coverage." From the NNUE side across all its games, aggregate:

- **Legality/replay integrity:** count any illegal action attempts, action-resolution
  failures (`OperationalRustSearchBot` raising on returned action), or replay mismatches.
  This should be **zero**; surface it explicitly rather than assuming it.
- **Score and discard distributions:** final own/opp score histograms; forced-discard
  frequency and where in the game they occur.
- **Phase coverage:** distribution of visited phases / rounds, so it's clear the generator
  reaches midgame and endgame states, not just openings.
- Reuse the per-move telemetry already surfaced on `last_report` (completed depth, nodes,
  timeout rate, chance share) so the generation-budget behavior is documented.

### Report must include

- Per-opponent: paired points rate + Wilson LCB/UCB, avg margin, seed count, both sides'
  mean decision time and the NNUE generation budget used.
- The coherence block above.
- A **competence verdict** field with an explicit classification, e.g.
  `competent_generator` / `borderline` / `not_generation_ready`, plus the reasoning. The
  intended positive outcome: **beats Random and Greedy clearly, at least competitive with
  pick-aware and mid-strength AZ, with zero legality failures and broad phase coverage** —
  i.e. its positions are legal, coherent, and nontrivial even though step 1 showed it does
  not beat mature AZ.
- Provenance mirroring step 1: pilot + all opponent artifact paths/shas, budgets, seed
  starts (screening vs any confirmation), `reserved_test_split_opened: false`.

### Invariants

- Same pilot artifact as step 1; encoder order-blind; chance handled inside the tree; no
  K=1/representative-row shortcut; full-width ordering on the NNUE side (no selective
  width). Never open the reserved test split.

## How to know it's done

- One command produces the competence JSON and prints a per-opponent summary table plus the
  competence verdict and the legality-failure count.
- A focused test (extend `games/kingdomino/tests/`) covering the coherence-aggregation
  logic (e.g. a stub game stream → correct legality/phase/score aggregation), in the style
  of the step-1 telemetry test.

## What the result decides (state in the report, don't act on it here)

- If **competent** → the curriculum track (P0 steps 3–7) is viable; the immediate
  follow-on becomes **step 0 (AZ terminal cascade alignment)**, which is blocking for the
  label-producing steps 5–7, then step 3 (collect fresh replayable AZ trajectories).
- If **not generation-ready** (loses to Greedy/pick-aware, or produces incoherent/illegal
  positions) → the generator itself must be strengthened before spending any teacher
  compute; that is a training-quality problem, not a search-allocation one, and would
  redirect to improving the pilot before P0 steps 3+.

Report the verdict and the recommended fork; do not begin either branch in this task.
