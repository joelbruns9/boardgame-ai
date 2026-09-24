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
| D5 | **Rust port** is in scope and (decided 2026-09-24) **moved ahead of the full training run**: Python proves the loop learns on one rule set, Rust does the real training. Multicore is a separate milestone *after* single-threaded equivalence, with rayon over games. |
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

**Status: DONE (2026-09-14).** `engine.py` + `tests/test_engine.py`, 81 tests;
mutation-checked (removing blocking, the runner cap, or claim-clears-markers
each fails tests). Legacy pipeline removed in `053d612`.

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

**Status: BUILT (2026-09-14), cost gate awaiting sign-off.** `solver.py`,
`tests/test_solver.py`, `turn_size_probe.py`.

- Correctness: matches a brute-force expectimax on all 10 rule sets, at the
  runner cap and before it; matches Monte-Carlo rollouts of its own policy.
  Mutation-checked (ignoring blocking, missing wins, dropping the bust value,
  menu-signature collision each fail tests).
- Real bug caught by the pre-cap brute force: a runner at the top of its
  column shared a menu signature with a claimed column, but still holds a
  runner slot. Fixed (signature code 5).
- Roll menus cached by an 11-column status signature: 26 s -> 8 s on the
  empty board. 109 distinct roll classes (not 126 multisets).
- **Solve after the first roll, not at turn start**: identical decisions
  (tested), table rooted at the rolled options only.

Measured cost, Python, heuristic self-play, 1 game per rule set
(heuristic games are short and aggressive -- 8 to 42 turns -- so the board
distribution is not what trained play will see):

| Solve rooted at | positions p50 (range over variants) | positions max | seconds p50 | seconds max |
|---|---|---|---|---|
| turn start | 8.9k – 46k | 69,038 | 1.3 – 7.0 | 11.0 |
| after first roll | 1.0k – 13k | 36,839 | 0.1 – 1.9 | 5.2 |

Leaf (evaluator) calls are ~75–100% of positions. After the first move the
remaining turn is small: 2 runners p50 0.8k–4.3k, 3 runners p50 ~200–460.
One solve per turn suffices (the table answers every later decision), so
Python self-play is roughly 1 s per turn on one core.

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
- **Gate: SIGNED OFF 2026-09-24.** Correctness checks pass. Per-decision cost
  (~1 s/turn, one core) is **accepted as too slow for the real run and not
  fixed in Python** — it is enumeration cost, not evaluator cost, so it is
  exactly what the Rust port addresses. Python continues only as far as the
  Phase 2 MVP and as the equivalence oracle for Phase 3.

### Phase 2 — MVP learning test (Python, deliberately narrow)

**Status: BUILT (2026-09-24), the run itself is the remaining work.**
`encoder.py` (8c8641a), `model.py` (ef359dd), `self_play.py` + `arena.py` +
`train.py` (4f43832). 182 tests, every module mutation-checked.

Launch (from the worktree root, `boardgame-ai-cantstop`):

    python -m games.cantstop.train --out runs/mvp --iterations 10 --games 20

At the measured ~10 s/game for 2p base that is ~35 min of generation plus the
arena checks; single-process.

Purpose: prove the **loop learns at all** — encoder, labels, training step and
the net-as-evaluator contract — at the smallest scale that can show it. This is
*not* the full multi-variant gate; that moves to Phase 4, on Rust.

- **Scope:** one cheap rule set (2 players, base, blocking off) for training.
  The encoder is still built to full width (4 padded seats, blocking flag,
  columns-still-needed) so nothing has to be reshaped later.
- **Encoding:** end-of-turn boards only (no runner features). Seat-relative,
  padded to 4 seats. Per seat: saved progress per column, claimed columns,
  **columns still needed to win** (so progress means the same thing across
  variants), seat-present flag (this is how player count is encoded). Global:
  blocking flag. **No probability features** — the solver owns the dice
  exactly, so feeding it an approximation of its own job is duplication, and
  per-column difficulty is a constant per feature slot that the first layer
  absorbs for free.
- **Output:** per-seat win-probability vector (softmax over seats), absent
  seats masked.
- **Evaluator adapter:** wrap the net to the *existing* contract
  `evaluate(boards) -> (N, num_players)` that `ProgressHeuristic` already
  satisfies, so `TurnSolver` needs no changes. Slice the 4-seat output to the
  live seat count and renormalize.
- **Loop:** standalone script, **not** `games/az_loop` yet — az_loop is built
  around policy+value AlphaZero learners and wiring it is real work that would
  gate the MVP on integration rather than on learning. Self-play with the
  current net → one row per turn end, labelled with the eventual game winner
  (one-hot over seats) → cross-entropy → repeat.
- **Multicore:** NOT YET BUILT -- `generate()` in `train.py` is the seam it
  slots into, and is deliberately free of execution detail. Allowed here
  (`GameState` pickles cleanly; process pool over *games*). Keep the fan-out logic — variant scheduling, per-variant row
  budgets, result collection, checkpoint cadence — in its own module, separate
  from the pool mechanism, because the fan-out survives the Rust port and the
  `multiprocessing` mechanism does not. **Smoke/correctness tests stay
  single-core**: parallel workers add per-worker RNG seeding and result-order
  nondeterminism to exactly the tests whose job is to be a clean signal (WT's
  search-seed collision is the cautionary case).
- **Gate (MVP):** training loss falls; the trained net beats both its own
  random init and the heuristic-leaf solver on the training rule set, over
  enough games to clear noise. Watch turn-length distribution in early
  iterations — collapse to always-stop or always-roll is the signal that D3's
  "dice variety is enough exploration" bet failed and noise is needed.
- **Also produced here:** a profile confirming the split below, and the
  reference corpus the Rust equivalence gate will replay.

**Why the port is expected to pay (unlike 7WD).** `TurnSolver._evaluate_leaves`
collects every stoppable leaf plus the one shared bust board and calls the
evaluator **once per turn** (1k–13k rows). So a turn is one batched forward
against ~1 s of Python enumeration: enumeration dominates, the evaluator does
not. That is the mirror image of the 7WD measurement (98.2% evaluator) that
killed *that* port. Confirm it with the Phase 2 profile before writing Rust —
but the prediction is strong, which is why the port moves ahead of the full
training run.

### Phase 3 — Rust port and multicore (moved ahead of the full run)

Read [[project_kingdomino_rust]] and [[welcome_to_rust_m1]] first. Crate at
`games/cantstop/cantstop_rust`, pyo3 0.28, `maturin develop --release`.

**Boundary:** Rust owns the engine, turn enumeration and backward induction.
Python keeps the training loop, replay buffer and torch. Only the per-turn leaf
batch crosses back.

- **M0 — portable RNG + snapshot.** Port the RNG so a seed reproduces a game in
  both languages, and have the snapshot carry the RNG state so a divergence in
  the *number of draws* fails on the step it happens, not several boundaries
  later.
- **M1 — engine equivalence.** Thousands of seeded games across all 10 rule
  sets, lockstep. **A uniform-random driver is not a gate** (WT trap): cycle
  drivers and add constructed positions for blocking-blocked stops, the runner
  cap, and claim-clears-markers.
- **M2 — solver equivalence.** Same positions, compare the full decision table,
  not just the chosen move, against the Python solver under a deterministic
  **mock** evaluator — that isolates enumeration and backup from torch FP noise.
  Use **f64 throughout**, not f32: f32 diverges after tens of accumulations and
  breaks bit-identity because Python accumulates in float64.
- **M3 — multicore (separate milestone, after M1/M2 are green).** Land
  single-threaded-correct first, then parallelize — this ordering is the one
  explicit "do it again" from the KD port. Parallelism goes over **games**
  (rayon), not inside a single turn solve. Release the GIL (`py.detach` in
  0.28 — `allow_threads` no longer exists) and coalesce the per-turn leaf
  batches from all in-flight games into one torch forward. Can't Stop's shape
  makes this much cleaner than KD's per-leaf funnel: there is exactly one
  evaluator call per turn, at a known point between enumeration and backup, so
  N parallel games give an N× larger batch with no per-leaf round-trip to
  amortize.
- **Gate:** equivalence green on all 10 rule sets; measured speedup reported as
  games/s end-to-end, not just engine microbenchmarks (WT's microbenchmark read
  ~27× while the whole-path ceiling was ~15×).

### Phase 4 — Full training run, all variants (the original Phase 2 gate)

Now on the Rust engine, and now worth wiring into `games/az_loop` for
checkpointing, run log and gating.

- **Sampling:** balance by **rows per variant**, not games (4-player and
  5-column games are longer, so equal games silently trains a 4-player
  specialist).
- **Targets:** baseline is game outcome. Blending in the solver's own root
  value is a later, measured change.
- **Baselines:** heuristic-leaf solver (all variants); earlier checkpoints.
- **Gate:** win rate vs the heuristic-leaf solver rises across iterations in
  **every** variant, not just the common ones.

### Phase 5 — Generalist vs specialist

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
