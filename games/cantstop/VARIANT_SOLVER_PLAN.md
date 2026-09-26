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

External review brief for M0-M3: `PHASE3_RUST_PORT_REVIEW_REQUEST.md`. For
the training targets, search options and personas built after it:
`TRAINING_SEARCH_REVIEW_REQUEST.md`.

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
  **GREEN (2026-09-24).** `src/engine.rs`, `snapshot.py` (the shared 8-tuple
  both engines emit and load), `rust_equiv.py` (lockstep harness + CLI),
  `tests/test_rust_engine_equiv.py`. Full gate
  `python -m games.cantstop.rust_equiv --games-per-cell 100`: 4,000 games
  (10 rule sets x 4 drivers: uniform / pusher / cautious / climber), 1.395M
  steps, 0 divergences, 114 s. After every step it compares the snapshot, the
  dice *and the RNG state*, legal moves, `can_stop`, `stop_blocked`; every
  25th step all 126 dice multisets; every 5th step every transition on
  clones, illegal ones included (same `ValueError`, state untouched).
  Coverage: 169k blocked decisions, 68k busts, 90k cap-split rolls, 4k
  one-space doubles, 21k claims that wiped an opponent's marker. 16
  constructed positions, each fixture asserted to hit what it names. Gate
  goes red on 6/6 Rust mutations (cap off-by-one, double room, claim keeps
  markers, strict win, unsorted dice, blocking scans only two seats) and on 6
  Python-side ones in the test file. One surviving mutant is *equivalent*,
  not a gap: counting the mover's own marker as a blocker changes nothing,
  because a runner always sits above its own saved marker.
- **M2 — solver equivalence.** Same positions, compare the full decision table,
  not just the chosen move, against the Python solver under a deterministic
  **mock** evaluator — that isolates enumeration and backup from torch FP noise.
  Use **f64 throughout**, not f32: f32 diverges after tens of accumulations and
  breaks bit-identity because Python accumulates in float64.
  **GREEN (2026-09-24), bit-exact:** full gate `--positions-per-ruleset 10`
  = 400 solves (10 rule sets x 10 positions x 4 evaluators), 3.60M
  configurations, 0 divergences, 18 min. `src/solver.rs`, `rust_solver.py`
  (`RustTurnSolver`, a drop-in for `TurnSolver`, plus the gate and CLI
  `python -m games.cantstop.rust_solver`), `tests/test_rust_solver_equiv.py`.
  The Rust solver is **two-phase**: construct (enumerate) ->
  `leaf_snapshots()` -> evaluate in Python -> `set_leaf_values` (backup) ->
  query. That split is the M3 coalescing seam. The gate compares every
  configuration's stoppable / winning / bust prob / menu (children and
  probabilities) / stop, roll and decision values / stop flag with `==`,
  then `value` / `choose_move` / `should_stop` on sampled configurations
  under all 126 dice. Four mock evaluators; **`mover_wins`** (every leaf,
  bust included, a win for the seat that just moved) puts every stop-or-roll
  choice within 1.3e-15 of a tie with both answers common, so the summation
  order and the prefer-stopping tie-break decide -- it is what makes
  "bit-exact" a real claim. (`flat` never exercises the stop tie: winning
  leaves make rolling strictly better.) Mutations: 6/7 caught (stop tie
  strict, backup last-max, fused multiply-add, win off-by-one, query
  last-max, empty root stoppable); the survivor (`w/1296` -> `w*(1/1296)`) is
  equivalent -- class weights are only {1,4,6,8,12,18,24}, all exact.
  **Measured, 2p base, AWAIT_MOVE roots (the self-play entry point), 15
  positions, mean 3.7k configurations:** warm-cache Python 256 ms/solve vs
  Rust enumerate + backup 12.4 ms -> **~21x**. ⚠ The M2 boundary now
  dominates: `leaf_snapshots` 3.8 ms + rebuilding Python `GameState`s 29.7 ms
  per solve, before any encoder runs. **M3 must encode leaves in Rust
  straight into a float32 buffer** (port `encoder.py`, gate it bit-exact on
  f32 against the Python encoder) rather than ship boards across.
  **DONE (2026-09-24):** `src/encoder.rs`; `TurnSolver.leaf_features()`
  returns LE float32 bytes, `NetEvaluator.evaluate_features(features,
  reference)` takes them, and `RustTurnSolver` uses that path whenever the
  evaluator has it (mocks still go through boards).
  `tests/test_rust_encoder_equiv.py` compares raw bytes on every leaf of
  sampled solves in all 10 rule sets and on ~1.2k whole-game boards.
  Mutations: 2/3 caught (columns-needed scaled by the rule set's own
  threshold -- the known trap -- and reversed seat rotation); the survivor
  (divide in f32 instead of f64-then-cast) is provably equivalent for one
  division. **Measured end to end with the (untrained) net on CPU, warm
  caches, 15 AWAIT_MOVE roots:** 2p base Python 416 ms vs Rust 24.5 ms per
  solve (**17x**; net forward 3.5 ms of it); 4p blocking 257 vs 12.1 ms
  (**21x**).
  **SUPERSEDED the same day -- that 17-21x was my implementation, not the
  port's ceiling.** Profiling the Rust path showed enumeration at 64% and
  ~4 us per configuration. Two causes, both fixed: (1) the menu cache was
  rebuilt per solve while Python's is module-level and warm -- now one per
  thread for the process (14.5 -> 5.8 ms); (2) per-node `Vec<Vec<usize>>`
  menu groups sorted per group, ~150k small allocations per solve -- now
  flat storage (one `kids` pool, `len x players` f64 slabs, FxHash maps,
  the sorted-first-max tie-break done as "larger value, or equal value and
  smaller key" with no sort) -> 1.4 ms. Values now cross as f64 bytes
  (`set_leaf_values_bytes`). **Re-measured, same 15 positions, warm caches
  both sides:** 2p base with net 410.7 -> 6.80 ms (**60x**), solver alone
  357 -> 1.62 ms (**220x**); 4p blocking with net 257 -> 3.87 ms (**66x**),
  solver alone 222 -> 1.12 ms (**198x**). All equivalence tests re-run green
  and the backup mutations re-checked on the new code (4/4 caught).
  ⚠ **With the solver this fast the net forward is now ~half of a Rust
  solve (3.1 of 6.5 ms)** -- i.e. self-play becomes evaluator-bound, the
  7WD shape. That is what M3's cross-game batching (and a GPU forward) is
  for; further Rust tuning of enumeration has little left to buy.
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

**M3 BUILT (2026-09-24).** `src/selfplay.rs` (`Game`, `Pool`),
`PySelfPlayPool`, `rust_pool.py` (driver: `run_pool`, `generate`,
`play_match`), `rust_pool_equiv.py` (gate + `bench` CLI),
`tests/test_rust_pool.py`. `train.generate` and `arena.play_match` take
`backend="auto"|"rust"|"python"`; auto uses the pool when built;
`train.py --backend`.
- **Per-game seeds** (behaviour change, both backends): `generate` and
  `play_match` draw one seed per game up front (`rust_pool.game_seeds`) and
  each game rolls from its own `PortableRng`. Before, one generator ran
  through all games in sequence, so game N's dice depended on games 0..N-1
  -- impossible to parallelize reproducibly.
- **Gate:** every `GameResult` field, training rows byte for byte, equals
  `play_game(rules, ev, PortableRng(seed))`, at 1 thread and all cores,
  with every game in flight or with an in-flight window of 1/4/7 (refill),
  mixed rule sets in one pool; under `hashed` and `mover_wins` feature
  mocks. Arena: pool and Python backends give identical wins with two
  different evaluators. Relative-output fast path == absolute path.
- **Measured** (RTX 3070 laptop, untrained net, 256 games/rule set, 64 in
  flight, GPU forward): 2p base **34.5k games/h** vs Python 144 (**~240x**),
  4p blocking 31.9k vs 134, 2p extended 16.9k vs 207. CPU forward ~9-10k
  games/h. 128 in flight was SLOWER on this 8 GB GPU (19.6k), so the default
  is 64. Arena vs the heuristic 61 s -> 3.7 s per 20 games.
- **Four fixes the measurements forced**, each worth not re-deriving:
  (1) *refill* -- with every game started at once, long games trailed
  alone: median round had 6 of 64 games waiting; now `in_flight` slots are
  refilled from the schedule; (2) *leaf encoding* was the pool's largest
  cost -- now the bust board is encoded once and each leaf patches only the
  mover's runner columns (`encode_leaves_into`), written in parallel straight
  into a Python-owned `bytearray` (no copies); (3) **`seat_mask_tensor` was
  copying the entire feature tensor to the CPU** to read 4 columns -- 64 of
  123 ms of a 900k-row GPU forward; now indexed on device (identical mask;
  also speeds training); (4) `to_absolute` per block moved into Rust
  (`resume_relative`, bit-identical, gated). `ProgressHeuristic` is now
  vectorized and scores boards and features through one function; it
  differs from the old per-board loop by <= 4e-16 (Python 3.12's `sum` is
  compensated), so heuristic numbers shift by rounding only.
- **Where the time goes now (GPU):** Rust ~17 s vs eval ~8.5 s per 256
  games, i.e. Rust-bound again; within Rust, writing ~350 MB of features per
  round is memory-bandwidth work. Next levers if ever needed: overlap Rust
  and the forward (two half-pools), or send compact leaves (runner keys +
  one template per game) and expand features on the GPU.

**External review of M0-M3 (2026-09-24), outcome.** Brief:
`PHASE3_RUST_PORT_REVIEW_REQUEST.md`. No soundness blocker; all four
sign-offs given (GIL release in `pending`, `write_f32_le`, `Pool::resume`;
per-game seeding without a compatibility flag; the three equivalent
mutants; proceed to Phase 4 with 64 in flight, overlap and GPU expansion
deferred). Both findings were reproduced as stated and fixed:
- **[P2] `backend="auto"` broke plain callable evaluators** (AttributeError:
  no `evaluate_features`). Fixed with `rust_pool.BoardEvaluator`, which wraps
  any callable by decoding features back to boards. Decoding is exact, so
  plain callables run on the pool and play the same games as the Python
  backend. (The Python path reports `evaluator_rows = 0` for them, because it
  reads an optional `rows` counter; the pool counts rows itself.)
- **[P2] mismatched schedule inputs silently dropped games** (`zip`
  truncation). `run_pool` now raises on unequal lengths and on a seating
  that names a missing evaluator; the remaining zips are `strict=True`.

Corrections accepted:
- The third mutant proof was worded too broadly. It is replaced by an
  exhaustive check of every (position, height) and (columns, 5) division.
- **The menu cache does not live for the whole process.** Pool workers
  belong to a rayon pool created per `run_pool`, so their caches die with
  it; the M2 speed text above says otherwise and is wrong for the pool.
- NumPy's short-row reduction order is an implementation detail;
  `test_rust_rotation_is_to_absolute_bit_for_bit` stays in the suite as the
  upgrade guard.

**Measuring the reviewer's cache question found a real problem.** The
original M2 "warm cache" speed-up re-solved the same 15 positions, so its
hit rate was flattering. In real self-play (320 all-variant games, GPU):
- the cache hits 98.4% of 94M lookups;
- turning it off makes self-play **6.6x slower** (264 s vs 40 s);
- but each worker thread builds its own copy, and the caches grow with
  every game a pool plays: 1.5M menus (~1 KB each), **3.27 GB peak RSS**.
  An earlier 640-game probe peaked at 4.8 GB, and the process did not
  return that memory afterwards.

Fix: a per-thread cap (`solver::MENU_CACHE_CAP`, default **20,000**,
settable with `cantstop_rust.set_menu_cache_cap`). The cache is cleared when
a miss would exceed the cap. 20k ran in the same time (40.0 s) at **2.00 GB
peak**; 100k saved nothing; 2k cost 32%. `menu_cache_stats()` reports
menus built. A test shows a cap of 50 plays exactly the same games.
`train.py` also gains `--threads` and `--in-flight`.

### Phase 4 — Full training run, all variants (the original Phase 2 gate)

**Phase 4 run plan (drafted 2026-09-26, after the 2p-base plateau).**

Why now: every lever tried on 2p base lands at the same strength (~57-58%
vs the heuristic):
- LR step-down: +1 pt;
- exact targets at λ=0.7: 0;
- λ=0 + exact + personas: 0 to +2 (the definitive final-vs-start match was
  49.8%);
- stop-bias recalibration: 0;
- k=4 lookahead: 0.

2p base looks close to saturated at one-turn search depth. The open
questions -- one net across variants, and whether interaction-heavy
variants reward more value accuracy or more search -- live in the other
nine variants. The user's own table game is 3 players, 4 columns,
blocking, which is also the Phase 5 target.

*Staying on `train.py`, not `games/az_loop`:* the standalone loop already
has the replay window, passes, TD targets, search options, personas, LR
schedule and the Rust pool. az_loop is built around policy+value AlphaZero
learners; wiring it is cost without a measured benefit.

**Measured sizing** (2026-09-26, pool on the RTX 3070 laptop, the 2p net
playing every variant off-distribution -- lengths and costs, not
strength):

| variant | rows/game | turns/game | net rows/game | plain games/h | exact games/h |
|---|---|---|---|---|---|
| 2p, 3 col | 9.8 | 10.8 | 121k | 52.6k | 10.3k |
| 2p, 3 col, blocking | 9.8 | 10.8 | 114k | 50.7k | 11.3k |
| 2p, 5 col | 20.5 | 21.5 | 143k | 39.4k | 9.8k |
| 2p, 5 col, blocking | 20.1 | 21.1 | 138k | 38.6k | 9.1k |
| 3p, 3 col | 29.0 | 30.0 | 450k | 17.4k | 3.5k |
| 3p, 3 col, blocking | 29.3 | 30.3 | 455k | 17.3k | 3.4k |
| 3p, 4 col | 41.2 | 42.2 | 324k | 16.9k | 4.8k |
| 3p, 4 col, blocking | 50.0 | 51.0 | 300k | 15.2k | 4.2k |
| 4p, 3 col | 29.1 | 30.1 | 452k | 17.0k | 3.4k |
| 4p, 3 col, blocking | 28.2 | 29.2 | 437k | 17.7k | 3.3k |

Rows per game span 5x (2p base ~10, 3p 4-col blocking ~50), so
equal-games sampling would give the 3-player extended variants 5x the 2p
base share. But the COST per row is nearly flat: ~0.5-0.8M rows/hour plain
and ~0.1-0.2M exact in every variant. Balancing by rows therefore also
roughly balances compute.

**Decisions proposed** (the user's to confirm):
1. **Start fresh, from a random net.** It is the clean test of the recipe
   as Phase 4 means it, and the plan's original decision. The 2p specialist
   (`runs/td0_personas/iter_0120.pt`) becomes the Phase 5 comparison on
   2p base. Warm-starting from it is the fallback if the fresh run learns
   too slowly.
2. **Targets: λ=0, sampled first, exact later** (PureTD's staging). Exact
   costs ~5x in every variant and showed no measured gain on 2p. Sampled
   λ=0 is effectively variable-depth on opening busts (review), which is
   acceptable. Switch to exact when the heuristic matches flatten.
   *Needs `--exact-from ITER`.*
3. **Rows, not games, per variant:** ~4,000 rows per variant per
   iteration, 40k total -- the same per-variant volume as the 2p runs.
   Games per variant adapt each iteration from the previous one's
   rows/game, seeded from the table above. *Needs `--rows-per-variant`.*
4. **Replay:** 10-iteration window (400k rows), 5 lifetime passes, so
   ~200k samples ~ 780 steps of 256 per iteration.
5. **Net:** 512-512-256-256 (0.51M). The capacity probe tied 92k-5.4M on
   2p data, but ten variants are richer. Re-run `capacity_probe` on
   Phase 4 data after ~20 iterations before changing size.
6. **Personas:** keep 20% conservative / 10% aggressive (user's choice).
   The 2p evidence is neutral (+1.6 / +2.5 pts, within noise). Measure per
   variant against fixed persona opponents at checkpoints.
7. **LR:** `--lr-schedule 1:1e-3 40:3e-4 80:1e-4 120:5e-5`.

**Evaluation** (all per variant):
- **vs heuristic:** the challenger against n-1 heuristic copies, seats
  rotated, even = 1/n, Wilson interval; 200 games per variant every 5
  iterations (`--arena-every 5`).
- **vs a frozen reference:** the iteration-0 net early, then a fixed
  checkpoint (e.g. iteration 40) as the review's frozen-opponent monitor.
- **Probe-set monitor** (review): a fixed set of ~50 boards per variant,
  sampled uniformly once. Each iteration logs the net's value against an
  exact one-turn search from the same net (RMSE and bias per variant). A
  growing residual means λ=0 bootstrapping is drifting.
- Per-variant log columns: games, rows, turns, turn length, seat wins.

**Gate (Phase 4 passes when):**
- in all 10 variants the heuristic match's lower 95% bound clears 1/n,
  and the trend is still rising or flat, not falling;
- the probe residual stays bounded;
- the 2p-base generalist is carried into Phase 5 against the 2p
  specialist.

**Throughput estimate:** 40k rows per iteration ~ 4-5 min generation plain,
plus <1 min training, plus arena every 5 iterations (~3-4 min). About 5-6
min per iteration, so ~90-100 iterations per overnight run while sampled;
about 5x slower after the switch to exact.

**Code work before the first run** (each with tests, Python reference where
it touches the pool):
1. `--rows-per-variant`: the adaptive per-variant game schedule, and the
   persona seating over the mixed schedule.
2. Per-variant evaluation: heuristic and frozen-reference matches for
   every variant, n-player seating, `--arena-every`.
3. Probe-set monitor, logged per variant.
4. `--exact-from ITER`: staged sampled-to-exact targets.
5. Per-variant log columns.

Rough size: a day of work, mostly (2) and (3).

**Earlier Phase 4 notes** (targets, search options, diagnosis):

- **Sampling:** balance by **rows per variant**, not games (4-player and
  5-column games are longer, so equal games silently trains a 4-player
  specialist).
- **Targets:** ~~baseline is game outcome; blending in the solver's own root
  value is a later, measured change~~ **BUILT 2026-09-24 (moved up after the
  mvp2 plateau, TD-Gammon lineage):** a lambda-return, `self_play.td_targets`.
  Row i (the board after turn i) takes G_i = (1-lam) v_{i+1} + lam G_{i+1}.
  v_{i+1} is the solver's value at the NEXT turn's opening roll: a one-turn-
  deeper estimate of row i, sampled over that roll. G at the end is the
  one-hot winner. An opening bust has no solve, and G passes through. lam is
  applied when rows enter the buffer, from the net that played the game
  (`--td-lambda`, default **0.7**, TD-Gammon's classic value; 1.0 = the old
  outcome labels). The pool records `turn_values` and the M3 gate compares
  them to Python. Loss: `masked_soft_cross_entropy`. **Not yet measured
  against lam = 1.**
- **Search options (`self_play.Search`, BUILT 2026-09-24; user's call:
  cross-turn search in self-play is worth throughput).**
  - `exact_root` (`--td-target exact`): solve BEFORE the opening roll, so
    the turn value is the exact expectation over every roll (PureTD's exact
    1-ply backup). Games are unchanged; only the recorded value differs.
  - `lookahead_k` (`--lookahead-k`): selective 2-turn lookahead,
    `lookahead.py` / `src/lookahead.rs`.
    1. Refine the bust board plus the top k-1 stop leaves by reach under
       the current policy.
    2. Solve the next player's whole turn from each, from its turn start.
    3. Swap those values in; with `lookahead_offset`, shift the other
       leaves by the reach-weighted mean refinement; back up again.
  - Search is set **per seat** (`search_seating`), so an arena can pit
    depth 2 against depth 1 with the same net.
  - Gate: per turn from mid-game positions, bit-exact vs Python, 3 rule
    sets x 2 mocks x {exact, k2, k3 raw, k3 exact}, plus mixed-seat turns.
    3/3 planted bugs caught. The first version of that gate checked
    NOTHING (a size filter no rule set met); it now fails loudly.
  - **Measured cost (2p base, untrained net, GPU, 64 in flight):**

    | search | games/h | net rows per turn |
    |---|---|---|
    | plain | 30.2k | 14.7k |
    | exact | 6.0k | 67k |
    | k=4 | 1,260 | 287k |
    | k=16 | 189 | 1.1M |

    The pre-build estimate ("exact at most ~2x, k=16 5-10x") was wrong:
    it used the LARGEST table sizes. A typical turn-start solve has ~4.6x
    the leaves of a post-roll one, and every refinement is a turn-start
    solve.
  - A k=16 + exact run exhausted host memory: 64 games x 17 live solver
    tables. `max_rows` (default 1M rows per round) bounds the forward, not
    the tables.
  - `NetEvaluator` now chunks forwards at 262k rows by default. One 16M-row
    forward asked for 16.8 GB.
- **Value-net diagnosis (2026-09-25), 2p base.** overnight1 (0.51M net,
  TD 0.7 sampled, window 30, passes 5): beats the heuristic ~55.5% but
  plateaued by iteration ~60 (1 hour of 8).
  - Learning rate: a 1e-4 step-down (runs/lr1e4) bought ~1 pt (51.2% vs
    its start, 8,000 games). Not the main limit.
  - Capacity (`capacity_probe.py`, 20k games, split by game): 92k, 0.51M,
    3.3M and 5.4M nets all reach held-out 0.520 by epoch ~5; bigger ones
    then overfit. Not capacity-bound at this data volume.
  - Accuracy against 400-rollout truth (`value_accuracy.py`, 60 boards):
    net RMSE 5.0 pts, one turn of exact search 2.9, k=4 2.4, k=16 2.3.
    The direction is clear but not significant at 60 boards.
  - Exact targets at lam 0.7 (runs/exact1): no gain (48.2% vs its start,
    57.1% vs heuristic). At lam 0.7 the outcome's dice luck dominates the
    target.
  - Decision: lam = 0 (PureTD; Tesauro found lam 0 about as good as small
    lam), exact targets, window 10.
- **Self-play personas (BUILT 2026-09-25):** `Search.stop_bias`, per seat.
  A persona stops when stop + bias >= roll: +0.03 conservative, -0.03
  aggressive. Purpose: human opponents mostly stop too early, and pure
  self-play never shows the net those positions.
  - `train.py --conservative 0.2 --aggressive 0.1 --persona-bias 0.03`:
    one persona seat per chosen game, rotating seat; draws no random
    numbers when both fractions are 0.
  - **Personas play biased, but record BEST-PLAY values as TD targets**
    (the same table backed up without the bias). Otherwise ~15% of rows
    would teach the net that the side to move plays worse than it does;
    the net cannot see who is a persona.
  - The net therefore values positions as if both sides play best.
    Exploiting a cautious human is opponent modelling -- not built.
- **LR schedule (BUILT):** `--lr-schedule 1:1e-3 60:3e-4 ...`, a step
  schedule by iteration; `lr` is logged each iteration.
- **External review of 3d72ef3..403ad7d (2026-09-25)**, brief
  `TRAINING_SEARCH_REVIEW_REQUEST.md`; notes in the main checkout,
  `reviews/cantstop-training-search-403ad7d.md`. Next recipe approved as a
  controlled k=0 trial. All four findings reproduced as stated and fixed:
  - **[P1] Offset lookahead produced negative targets.** A common shift
    keeps a row's sum but not its signs. A hashed-mock 3-player root came
    out [0.310, -0.005, 0.696], and 2,579 of 6,029 stop vectors had a
    negative entry; negative soft-CE targets are unbounded below.
    - Shifted rows are now projected onto the simplex (clip, renormalise),
      bit-identical in Rust and Python.
    - `check_targets` refuses invalid targets at the training boundary.
    - A 3-player end-to-end test.
    - **Lookahead stays parked for training:** the offset is an
      unvalidated heuristic.
  - **[P2] The arena measured a different search.** Dropping `exact_root`
    is only move-neutral without lookahead; with k > 0 it changes which
    leaves are refined. `train.arena_search_for` keeps it when k > 0.
  - **[P2] `value_accuracy.py` took the LOWEST indices** (mean percentile
    19%, not 50%): it sorted before truncating.
    - Now uniform without replacement, with source game ids kept and a
      game-cluster bootstrap SE.
    - **The 60-board accuracy numbers above are therefore unreliable
      until re-run.** They are also scoped to calibration against plain
      play, not a search-investment criterion.
  - **[P2] Persona targets with k > 0 still depend on the persona**:
    refinement is chosen under the biased policy. Documented as an
    approximation; k = 0 runs are unaffected.
  - Corrections accepted:
    - Terminal anchors are NOT confined to the last played turn: the
      solver scores every enumerated winning stop exactly.
    - Sampled and exact targets differ on opening busts. Exact uses the
      net's value of the passed-turn board; sampled forwards the next
      search, so sampled is effectively variable-depth there.
    - The capacity tie supports only "no demonstrated benefit for this
      dataset and procedure"; `capacity_probe` now reports CE - H(target).
    - The PureTD comparison is qualified: it trains on fresh data for one
      epoch, and we use a replay window.
    - A weights-only resume resets Adam and the buffer. Pass the rate
      explicitly (the next run's `--lr-schedule 1:1e-4` does).
  - Monitoring suggested for the λ=0 run, not built yet: frozen-opponent
    win rates, calibration on independent rollout boards, and exact-backup
    residuals on a fixed probe set.
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

- ~~Shallow opponent-turn lookahead on top of the solver.~~ Built
  (selective, `lookahead_k`), parked for training: no gain on 2p base at
  k=4. Retest in 3-4p blocking once a Phase 4 net exists.
- ~~Bootstrapped value targets.~~ Built (TD(λ), exact root).
- Opponent modelling (exploiting cautious humans explicitly).
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
