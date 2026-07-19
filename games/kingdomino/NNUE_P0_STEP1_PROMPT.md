# Implementation prompt — NNUE Package P0, Step 1: matched-clock scaling characterization

Context: `games/kingdomino/NNUE_PROJECT_PLAN.md`. Read the "Immediate work package P0"
section (steps 0–7) and the "Current status (2026-07-14)" block before starting. This
prompt implements **P0 step 1 only**. It is the cheap discriminator the plan puts ahead
of every standalone A/B/C package and ahead of any further NNUE training.

## Why this is next (do not re-scope)

- The pilot NNUE lost 0-8 to AZ at a single matched 0.1s clock (four paired seeds, both
  seats, −88.4 avg margin). The plan is explicit that this "does **not** decide
  feasibility" and that a **clock-scaling curve is the correct cheap discriminator**:
  the question is whether the gap **narrows, stays flat, or widens** as both agents get
  more time — not who wins one tiny set.
- Step 0 (cascade-align the AZ terminal value) is **blocking for P0 steps 5–7** (label
  production) but explicitly **not** for "the pure gameplay measurements in steps 1–2."
  Do **not** touch `terminal_search_value` / `solve_endgame_ab` in this task.
- Do not start work packages A (stochastic allocation), B (teacher data), or C (training),
  the Middle-Kingdom LMR experiment, or any new training run. Those are deferred until
  this gate and step 2 report.

## What exists already (reuse, do not rebuild)

- `games/kingdomino/nnue/match.py` — the paired, seat-swapped runner with per-agent
  clock accounting. Its `az-baseline` subcommand (`_main_az`) already pairs
  `nnue_participant(...)` against `build_open_loop_checkpoint_participant(...)` at a
  **single** `--nnue-move-secs` / `--az-sims`, and `TimedBot` already separates forced
  from non-forced decision wall-time. This is the harness to extend — do not write a new
  match loop.
- `games/kingdomino/rust_expectiminimax.py::OperationalRustSearchBot` — the playable NNUE
  path. `choose_action` stores the per-move telemetry object on `self.last_report`.
- `OperationalSearchReport` (Rust, `kingdomino_rust/src/lib.rs` ~line 2348) exposes per
  move: `completed_depth`, `timed_out`, `elapsed_secs`, `nodes`, `last_iteration_nodes`,
  `aspiration_researches`, `star_cutoffs`, `exact_extensions`, `tt_hits`, `tt_cutoffs`,
  `ordering_evals`, `ordering_actions`, `selective_pruned`, `selective`.

## The two gaps this task closes

1. **Telemetry is dropped.** `TimedBot` (match.py) wraps only `.choose_action` and never
   reads `last_report`, so the plan-required NNUE **completed depth / nodes / timeout
   share** never reach the report. Extend `TimedBot` (or a subclass used only for the
   NNUE side) to capture `getattr(self.bot, "last_report", None)` after each non-forced
   decision and aggregate it.
2. **No chance-node share exists.** The plan's step-1 deliverable asks for NNUE "chance
   share," but `OperationalSearchReport` has **no chance-node counter**. Add a
   `chance_nodes: u64` field to the Rust report (increment wherever the operational search
   expands a chance node — the same place `make_with_chance` / row-sampling happens in the
   operational path, not the Python `rust_expectiminimax` prototype), rebuild the wheel,
   and surface it. If adding the Rust counter is out of scope for this pass, **say so
   explicitly in the report and record chance share as `null`** rather than silently
   omitting it — do not fake it from Python.

## Deliverable

A driver (e.g. `games/kingdomino/nnue/clock_scaling.py`) plus a JSON report that answers:
**does the NNUE-vs-AZ gap narrow, stay flat, or widen across matched clocks?**

### Clocks and matching

- Sweep NNUE deadlines `0.1 / 0.5 / 2 / 10s` (plan step 1).
- For each clock, match AZ to comparable **actual decision wall-time**, not a guessed
  sim count. AZ cost per move is not linear in sims and depends on device; so calibrate:
  at each clock, pick `--az-sims` such that AZ's measured `decision_mean_seconds`
  (from `TimedBot.summary()`) lands within a tolerance (e.g. ±15%) of NNUE's. Do a short
  calibration pass (a couple of seeds) to choose sims, record the chosen sims **and the
  realized mean decision times for both agents** in the report, then run the real match.
  The plan demands "actual decision-time telemetry" — a nominally matched clock that hides
  unequal search time is exactly what `TimedBot` was built to expose.
- Paired seeds, **both seats** (the runner already plays A/B and B/A per seed).
- Screening set at all four clocks; then a **disjoint confirmation set only at the
  clock(s) where the trend could change a decision** (plan step 1: "a disjoint
  confirmation only where the trend could change a decision"). Use a different
  `--seed-start` for confirmation so the sets don't overlap.

### Per-clock report must include

- Win/points rate for NNUE with its Wilson LCB (already in `run_paired`'s `pair` block)
  and average margin.
- NNUE aggregates from `last_report`: mean/median **completed_depth**, mean **nodes** and
  **last_iteration_nodes**, **timeout rate** (`timed_out` fraction), and **chance share**
  (or `null` per gap 2).
- AZ: chosen sims and realized mean/p95 decision seconds.
- Both agents' `decision_mean_seconds` / `p95` / forced counts (already in
  `TimedBot.summary`), so the clock match is auditable.
- Settings/provenance already emitted by `_main_az`'s `settings` dict (artifact + AZ
  checkpoint sha256, chance_samples, ordering flags). Add the sweep index / clock /
  screening-vs-confirmation tag.

### Invariants (correctness — enforce, don't skip)

- Keep the encoder order-blind and chance handled inside the public-state tree; this task
  changes **measurement only**, not the search's chance model. Do not add any K=1 /
  representative-row shortcut.
- Same NNUE artifact and same AZ checkpoint across all clocks (only the budgets vary).
- Full-width ordering stays the NNUE default (`full_width_ordering=True`); do **not**
  enable selective width — that is a separate, already-rejected research knob.
- Never open the reserved test split.

## How to know it's done

- One command produces the sweep JSON; a short summary prints per-clock NNUE points-rate
  (+LCB), avg margin, mean completed depth, and both agents' realized decision times.
- The report makes the narrow/flat/widen verdict legible: e.g. a small table of clock →
  NNUE points-rate with confidence intervals, plus completed depth so a "more time only
  buys nominal depth, not strength" outcome is visible.
- Add a focused test (extend `games/kingdomino/tests/`) that the extended `TimedBot`
  captures and aggregates `last_report` correctly (a stub bot with a canned report is
  enough); if the Rust `chance_nodes` field is added, extend a Rust test asserting it is
  non-zero on a state with a live chance node and zero on a no-chance endgame.

## Explicitly out of scope

- Step 0 cascade alignment, steps 2–7, packages A/B/C, Middle-Kingdom LMR, any retraining
  or new data generation, any change to the search's chance model or eval. Report the
  curve; interpretation and the next step are decided after the numbers land.
