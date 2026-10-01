# Review request: Can't Stop Phase 3, the Rust port (M0-M3)

This asks for a review of the whole Phase 3 Rust port of the Can't Stop
variant solver:

- a portable RNG;
- the rules engine;
- the exact turn solver;
- the value-net leaf encoder;
- a multicore self-play pool that batches net calls across games.

It also covers the Python changes the port required in the existing training
and arena code.

The design and the measurement history live in `VARIANT_SOLVER_PLAN.md`
(Phase 3 section). That document is long and records dead ends; this brief is
what you need to review. Phases 0-2 (Python engine, Python solver, encoder,
net, MVP loop) were built earlier. They are **out of scope** except where this
work changed them (§1, last table).

Branch `cantstop-variant-solver`, reviewed range `0b0216f..6d64a25` (26 files,
~5k lines added).

## 1. Scope

| Milestone | What | Commit | Primary files |
|---|---|---|---|
| M0 | Portable SplitMix64 RNG, Python + Rust | `231816b` | `portable_rng.py`, `cantstop_rust/src/rng.rs` |
| (fix) | M0 test that could never fail | `0e24425` | `tests/test_portable_rng.py` |
| M1 | Rules engine + lockstep equivalence gate | `644fea7` | `src/engine.rs`, `snapshot.py`, `rust_equiv.py` |
| M2 | Exact turn solver, two-phase API | `644fea7` | `src/solver.rs`, `rust_solver.py` |
| M2+ | Leaf encoder (f32 features built in Rust) | `644fea7`, `6d64a25` | `src/encoder.rs`, `TurnSolver::encode_leaves_into` |
| M3 | Self-play pool: many games, rayon, batched forwards, refill | `6d64a25` | `src/selfplay.rs`, `rust_pool.py`, `rust_pool_equiv.py` |
| all | pyo3 bindings | both | `src/lib.rs` |

**Python code this work changed** (in scope):

| File | Change |
|---|---|
| `train.py` | `generate()` gains `backend=auto/rust/python`; **one seed per game**; `--backend` flag |
| `arena.py` | `play_match()` likewise; per-game seeds |
| `model.py` | `NetEvaluator.relative_probs` / `evaluate_features`; **`seat_mask_tensor` now indexes on device** |
| `solver.py` | `ProgressHeuristic` **vectorized**, with one scoring function for boards and features |
| `encoder.py` | `decode_features` (the exact inverse of `encode_board`) |

Gates: `tests/test_rust_engine_equiv.py`, `test_rust_solver_equiv.py`,
`test_rust_encoder_equiv.py`, `test_rust_pool.py`, `test_portable_rng.py`.

## 2. What it does

- **Boundary.** Rust owns the engine, turn enumeration, backward induction,
  leaf encoding and whole-game play. Python owns torch, training and the
  replay buffer. The only thing that crosses per turn is the leaf batch:
  f32 features out, values back.
- **Engine** (`engine.rs`). A transcription of `engine.py`. Columns are
  indexed by dice sum in arrays of length 13, and runners are an array with
  0 meaning "no runner". Moves are normalized to their effect and ordered like
  Python tuples.
- **Solver** (`solver.rs`). The same backward induction as `solver.py`, in
  two phases:
  1. `new` enumerates every reachable runner configuration, using a
     thread-local menu cache keyed by column signature. *(Review
     correction: pool workers belong to a rayon pool created per
     `run_pool`, so their caches die with it; they do not live for the
     whole process. Each is now also capped, see the plan.)*
  2. the caller evaluates the leaves (the shared bust board plus every
     stoppable, non-winning configuration);
  3. `set_leaf_values` runs the backup.

  Storage is flat: one `kids` pool, and `len × players` f64 value slabs. The
  tie-break is "first maximum over children sorted by key". Rust implements
  it without sorting, as "larger value, or equal value and smaller key".
- **Leaf encoder** (`encode_leaves_into`). The bust board is encoded once as a
  template. Each stop leaf is the template with the mover's runner columns
  patched: progress, or a claim that zeroes every seat's progress on that
  column and recomputes the mover's columns-needed.
- **Pool** (`selfplay.rs`, driven by `rust_pool.run_pool`).
  - Each `Game` plays `self_play.play_game` turn for turn, rolling from its
    own `PortableRng`.
  - A game runs until its turn solve needs values. `pending()` then writes
    every waiting game's leaves, in parallel, into one Python-owned
    `bytearray`.
  - Python runs one forward per evaluator. `resume` / `resume_relative`
    supply the values, and each game advances to its next wait.
  - At most `in_flight` games (default 64) are live at once. Finished games
    are replaced from the schedule, in schedule order.
  - For batched nets, Rust rotates seat-relative output back to absolute
    seats (`absolute_from_relative`, a port of `encoder.to_absolute`).

## 3. What is already gated (please don't re-verify by hand)

Python is the reference throughout. Every comparison is **exact**:
`==` on float64, or on raw f32 bytes.

- **RNG.** Bit-identical to 7WD's SplitMix64 stream and to Rust for
  `next_u64`, `float`, `randrange`, dice and shuffle.
- **M1 engine.** 4,000 lockstep games: all 10 rule sets × 4 drivers
  (uniform / pusher / cautious / climber), 1.395M steps, 0 divergences.
  - After every step it compares the snapshot, the dice **and the RNG
    state**, legal moves, `can_stop` and `stop_blocked`.
  - On sampled steps it also compares legal moves for all 126 dice
    multisets, and tries every transition on clones, illegal ones included:
    both sides must raise the same `ValueError` and leave the state
    unchanged.
  - Coverage floors are asserted: 169k blocked decisions, 68k busts, 90k
    rolls where the runner cap forced a single move, 21k claims that wiped an
    opponent's marker.
  - 16 constructed positions, each fixture asserted to hit what it names.
- **M2 solver.** 400 solves, 3.60M configurations: 10 rule sets ×
  3 phases × 4 mock evaluators.
  - For every configuration it compares: stoppable, winning, bust
    probability, the menu (children and probabilities), stop, roll and
    decision values, and the stop flag.
  - It also compares `value` / `choose_move` / `should_stop` on sampled
    configurations under all 126 dice.
  - One mock, **`mover_wins`**, puts every stop-or-roll choice within
    1.3e-15 of a tie, with both answers common. That is what makes the
    summation order and the prefer-stopping tie-break observable.
- **Encoder.** Raw f32 bytes equal `encode_batch` on every leaf of sampled
  solves in all 10 rule sets, and on ~1.2k whole-game boards.
  `decode_features` round-trips exactly.
- **M3 pool.** Every `GameResult` field equals `play_game(rules, ev,
  PortableRng(seed))`, training rows byte for byte. That holds:
  - at 1 thread and at all cores;
  - with in-flight windows of 1, 4, 7 and all;
  - with mixed rule sets in one pool;
  - under the `hashed` and `mover_wins` feature mocks.

  Arena wins are identical between the pool and the Python backend with two
  distinct evaluators. The relative fast path equals the absolute path, and
  the Rust rotation equals `to_absolute` byte for byte.
- **Mutation checks.** Bugs were planted by hand in the Rust code (they are
  not in the test suite), and the gates went red on 14 of 17:
  - engine: 6 of 7;
  - solver: 6 of 7, plus 4 of 4 re-run after the flat-storage rewrite
    changed the tie-break code;
  - encoder: 2 of 3.

  The three survivors were each **proven equivalent**, not assumed:
  - counting the mover's own marker as a blocker changes nothing, because a
    runner always sits above its own saved marker;
  - `w/1296` and `w*(1/1296)` agree because the roll-class weights are only
    {1, 4, 6, 8, 12, 18, 24};
  - an f32 division and an f64 division cast to f32 agree. *(Review: the
    original reason, "double rounding is exact for a single operation", was
    too broad. It is now checked exhaustively over every division the
    encoder can perform; see the review outcome in the plan.)*

So the algorithms are shown equivalent under identical evaluations. Please
spend your time on what the gates **cannot** see.

## 4. Focus areas

### Soundness (Rust)

1. **`PySelfPlayPool.pending`** (`lib.rs`). It creates a
   `PyByteArray::new_with` and, *inside* its initializer closure, calls
   `py.detach` so rayon threads can write into the Python-owned buffer
   without the GIL. We believe this is sound: the object is not yet reachable
   from Python. Please confirm against pyo3 0.28's contract for `new_with`
   and `detach`.
2. **`write_f32_le`**. It uses `unsafe { bytes.align_to_mut::<f32>() }` with
   a fallback path, plus a `compile_error!` for big-endian targets.
   - Is the alignment reasoning (CPython's allocator gives ≥8-byte alignment)
     sufficient?
   - Is the fallback correct?
3. **`Pool::resume`**. It pairs each game with its value slice through a
   `by_game: Vec<Option<&[f64]>>` zipped against `games.par_iter_mut()`.
   Please check for aliasing or ordering assumptions. Blocks are built in
   ascending game order, and nothing enforces that beyond construction.

### Correctness the gates may miss

4. **Refill + failure.** A game that hits `max_turns` becomes `Failed`, and
   the pool keeps playing the others. `run_pool` raises only after
   `running` is false, whereas Python's `play_game` raises immediately.
   - Is anything lost, or double-counted, when a failure happens mid-schedule
     with queued games behind it?
   - `promote()` fills slots in schedule order. Does anything depend on
     *which* game fills a slot? We believe not, because games are independent.
5. **Per-game seeding** (`rust_pool.game_seeds`). Seeds are drawn with
   `rng.randrange(2**64)` from either `random.Random` (train) or
   `PortableRng` (tests), and each game rolls from `PortableRng(seed)`.
   - Any correlation concern with SplitMix64 streams seeded this way?
   - This is a deliberate **behaviour change**: the same seed no longer
     reproduces pre-M3 runs. Please say if that should have been gated
     behind a flag instead.
6. **`ProgressHeuristic` rewrite** (`solver.py`). Top-k climbs are
   zero-padded, sorted descending and cumsum'd. The result is within 4e-16 of
   the old per-board loop, but not identical (we attribute the difference to
   Python 3.12's compensated `sum`). Check the edge cases:
   - a seat needing more columns than remain open;
   - `k == 0`;
   - claimed columns excluded for every seat.
7. **`seat_mask_tensor`** (`model.py`) now indexes on device. It is verified
   identical on CPU and CUDA, batch and single row. It is also used in
   training (`train_steps` builds the mask for the whole buffer with it).
   Any path where `x` is not float32 features?
8. **The equivalent-mutant arguments in §3.** Do you accept all three?
9. **`absolute_from_relative`** sums the row in absolute seat order and then
   divides. It is gated byte-equal to numpy's `out.sum(axis=1)` on sampled
   data. Is numpy's axis-1 reduction guaranteed sequential for n ≤ 4, or did
   the gate get lucky? If it isn't guaranteed, the fast path could diverge
   from the Python path by an ulp on some row.

### Design

10. **Menu cache growth.** One cache per thread, never evicted. The
    signature space is bounded (6 codes × 11 columns), but the reachable
    count across a long training run is **not measured**. Is unbounded-but-
    bounded acceptable, or should it be capped?
11. **Where the remaining time goes.** On the GPU, Rust takes ~17 s and the
    net ~8.5 s per 256 games. Most of the Rust share is writing ~350 MB of
    features per round. Two levers are listed but not built: overlap Rust
    with the forward (two half-pools), or ship compact leaves and expand
    them on the GPU. Is either worth it before the Phase 4 run, or is the
    port done?
12. **`in_flight = 64` default.** 128 was *slower* on the 8 GB laptop GPU
    (19.6k vs 34.5k games/h). That is one run on one machine, and the cause
    isn't diagnosed (memory pressure is a guess). Should the default be
    per-device?

## 5. Known limitations (not asking you to find these)

- **Benchmarks** are single runs on a laptop (RTX 3070 8 GB), with an
  **untrained** net. Game length, and so leaves per round, will change once
  the net learns. Treat games/hour as about ±15%.
- **The Phase 2 learning test has not passed.** The last MVP run *regressed*
  (iteration 2: 0/20 against the heuristic; turns collapsed from 9.6 to 5.4
  steps). The port makes that run cheap to repeat at scale; it does not fix
  it.
- **`Cargo.lock` is ignored repo-wide** (`.gitignore:24`), so the `pyo3` and
  `rayon` patch versions are not pinned.
- **The test suite takes ~18 minutes**, dominated by the Python reference
  games in the pool gate. Smoke and correctness tests run single-threaded
  on purpose.
- **Mutation checks were done by hand** (sed, rebuild, re-run). Only their
  outcomes are recorded, in the plan; they are not automated.

## 6. How to run

From the `boardgame-ai-cantstop` worktree, with the main folder's venv:

```
set VIRTUAL_ENV=C:\Users\joeld\projects\boardgame-ai\.venv
cd games\cantstop\cantstop_rust && maturin develop --release && cd ..\..\..
python -m pytest games/cantstop/tests -q                 # ~18 min, 318 tests
python -m games.cantstop.rust_equiv --games-per-cell 100 # M1 full gate, ~2 min
python -m games.cantstop.rust_solver --positions-per-ruleset 10  # M2, ~18 min
python -m games.cantstop.rust_pool_equiv gate --games 4  # M3 vs Python
python -m games.cantstop.rust_pool_equiv bench --games 256 --in-flight 64
cargo test --lib  (in cantstop_rust)                     # Rust unit tests
```

## 7. Sign-offs requested

1. **Soundness** of the three `unsafe` / GIL-release patterns (§4.1-4.3).
2. **The per-game seeding behaviour change** (§4.5): accept it as is, or
   require a compatibility flag.
3. **The equivalent-mutant proofs** (§4.8).
4. **Port complete for Phase 4**: either sign off that throughput is
   sufficient to proceed to the full all-variant run, or name what must
   change first (§4.10-4.12).
