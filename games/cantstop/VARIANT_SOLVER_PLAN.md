# Can't Stop — One Net for All Variants, Trained Through a Turn Solver

Plan of record, drafted 2026-09-14. This is the single Can't Stop build: it
replaces the legacy 2-player MCTS pipeline in this folder rather than living
beside it (the legacy trained net was lost, so there is nothing worth
comparing against).

## Goal

A single value network that plays every player count and rule variant, with
decisions made by an exact **turn solver** instead of MCTS. Target game of
interest: 3 players, extended columns, blocking on.

## Decisions (made)

| # | Decision |
|---|---|
| D1 | **One net** covers all variants. Fallback (only if the generalist test fails): two nets split by player count. |
| D2 | **Turn solver** replaces MCTS for decision-making and self-play. |
| D3 | **Start from a random net.** No MCTS-trained or heuristic warm start in the baseline run. |
| D4 | **Value-only net.** No policy head; the solver checks every option itself. |
| D5 | **Rust port** is in scope, sequenced after the Python solver is validated and profiled. |
| D6 | **Single build.** Refactor/replace the legacy code; no parallel legacy engine or legacy baseline. Removed legacy files remain recoverable from git history. |
| D7 | **Laptop first.** Rent cloud compute only if laptop runs become too slow. |

## The variant space

| Players | Columns to win (base) | Columns to win (extended) |
|---|---|---|
| 2 | 3 | 5 |
| 3 | 3 | 4 |
| 4 | 3 | 3 (same as base) |

× blocking {off, on} → **10 distinct rule sets** (4-player base and extended
are identical).

**Blocking rule** (confirmed): if any of the active player's runners is on
the same space as a different player's saved progress marker, the turn
cannot end by stopping; the player must roll again. Runners may land on and
pass those spaces freely mid-turn. Invariant: two players' saved progress
never share a space. Bust is unaffected.

## How the turn solver works

Within one turn, nobody else moves, so the only uncertainty is the dice, and
their probabilities are exactly known.

1. **Enumerate** every in-turn position reachable from the current one:
   runner placements and heights. Different roll orders that reach the same
   runners share one entry.
2. **End-of-turn leaves:**
   - *Stop* → commit runners → a board with the opponent to move → **net**
     scores it (per-seat win probabilities). A stop that wins the game is
     scored exactly (1 for the winner), with no net call.
   - *Bust* → the board as it was at turn start, opponent to move → **one**
     net call, shared by every bust in the turn.
3. **Work backwards.** Every non-bust roll moves a runner up, and columns have
   tops, so the turn is finite even though stopping is optional. Solve
   the most-advanced positions first (there, rolling is a certain bust), then
   step down:
   `value(pos) = max( stop value [if stopping is legal],
                      Σ_roll P(roll) × best option's value, or bust value )`
4. **Decide.** At the actual position: take the option / stop-or-roll with the
   highest win probability for the active seat.

The net carries everything strategic (column races, who is ahead, how
dangerous an opponent's marker is). The solver carries the dice exactly.
Risk appetite (roll more when behind, less when ahead) falls out of maximizing
win probability, because win probability is not linear in progress.

**Roll outcomes as menus.** A roll is represented by the set of options it
offers (after all runners are placed: a subset of
{a, b, c, aa, bb, cc, ab, ac, bc}), with exact probability from the 126
four-dice multisets. `mcts.py` and `chance_fanout_probe.py` already bucket
rolls by legal-move set; reuse that.

**Known blind spot:** no lookahead into opponents' turns. The net's reading of
the end-of-turn board stands in for it. A shallow opponent-turn search on top
is a later option, not part of the baseline.

## Phases and gates

### Phase 0 — Rule-parameterized engine (Python)

- `RuleSet(num_players, columns_to_win, blocking)`, with the variant table
  above as the only constructors. No module-level `COLUMNS_TO_WIN`.
- Stop legality under blocking; a blocked position offers roll only.
- Tests: every one of the 10 rule sets plays random games to completion;
  targeted tests for blocking (stop illegal on an opponent marker, legal once
  moved past, bust unaffected, saved markers never share a space), extended
  win thresholds, 3/4-player turn order.
- Carry over the legacy engine's correctness fixes and tests where they still
  apply (e.g. the runner-cap partial-move fix).
- **Gate:** tests green on all 10 rule sets.

### Phase 1 — Turn solver prototype with a fixed evaluator

- Solver as specified above, taking any `evaluate(boards) -> per-seat win
  probs` callable. Develop against `heuristic_value` as the leaf so no net is
  needed yet.
- **Correctness:** on small hand-built positions, solver value equals brute
  force; on real positions, solver value agrees with long Monte-Carlo rollouts
  of the solver's own policy within sampling noise.
- **Measure (do not assume):** in-turn positions and distinct leaf boards per
  decision, split by runners placed (0/1/2/3) and by variant. The pre-cap
  (fewer than 3 runners) turn is the cost risk.
- Dominance pruning (e.g. "ab" ⊇ "a") is an **optional, measured**
  optimization, off by default. It is expected to fail under blocking.
- **Gate:** correctness checks pass; per-decision cost is acceptable for
  self-play (target set once measured).

### Phase 2 — Value net and training loop from random

- **Encoding:** end-of-turn boards only (no runner features). Seat-relative,
  padded to 4 seats. Per seat: saved progress per column, claimed columns,
  **columns still needed to win** (so progress means the same thing across
  variants). Global: blocking flag; seat presence encodes player count.
- **Output:** per-seat win-probability vector (softmax over seats).
- **Loop:** solver self-play with the current net → one training row per turn
  end, labelled with the game winner → train → repeat. Reuse
  `games/az_loop` (checkpointing, run log, gating) where it fits a value-only
  learner.
- **Sampling:** balance by **rows per variant**, not games (4-player and
  5-column games are longer).
- **Targets:** baseline is game outcome. Blending in the solver's own
  root value is a later, measured change.
- **Exploration:** baseline relies on dice variety alone. Watch for collapse
  early; add noise only if measured necessary.
- **Baselines:** heuristic-leaf solver (all variants); earlier checkpoints.
- **Gate:** win rate vs the heuristic-leaf solver rises across iterations in
  every variant, not just the common ones.

### Phase 3 — Rust port

- Engine + solver in Rust, pyo3 bindings. Read the KD port memory first
  (pyo3 bytes, f64-not-f32, pyo3 0.28 attach/detach).
- **Before porting, profile Phase 2**: if net evaluation dominates, Rust buys
  little (the 7WD lesson). Port what the profile says is hot.
- **Gate:** equivalence test vs the new Python engine and solver on thousands of
  seeded games across all 10 rule sets (WT M1 pattern: portable RNG).

### Phase 4 — Generalist vs specialist

- Train a specialist on the target variant (3 players, 4 columns, blocking).
  Compare it to the generalist twice: at equal compute on that variant, and
  at equal total compute. Report both.
- **Gate / verdict:** generalist within noise of the specialist → D1 holds.
  Otherwise fall back to per-player-count nets and re-test.

### Later (not scheduled)

- Shallow opponent-turn lookahead on top of the solver.
- Bootstrapped value targets.
- BGA advisor on the shared `games/advisor` host (see `bga_recon.md`).

## Resolved questions

- **Q1 — Blocking details:** resolved; see the blocking rule above.
- **Q2 — Code location:** single build replacing the legacy code (D6).
- **Q3 — Compute:** laptop first, cloud if needed (D7).

## Notes on the legacy code

- Legacy `engine.py` hard-codes `COLUMNS_TO_WIN = 3` and 2-player shortcuts in
  `clone()`.
- Claiming a column leaves other players' progress entries on it; they are
  inert because the column is in `all_claimed`. The new encoding should drop
  them.
- The legacy trained model was deleted in `c8be893`.
