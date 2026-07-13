# Rust engine work item #1: make/unmake for the NNUE searcher

> **STATUS — Step 1 DONE (2026-07-12).** `make`/`unmake` + the differential gate
> shipped in `lib.rs` (plain `impl RustGameState` before the `#[pymethods]` block;
> `PlaceUndo` / `RoundSnapshot` / `UndoRecord`). Tests in `mod make_unmake_tests`
> all PASS: `make_matches_step_and_round_trips`, `full_playout_unwinds_to_start`,
> `make_rejects_illegal_actions_without_mutating`. The `make==step` and
> `make+unmake` round-trip properties hold across 200 random games via a full-field
> `fingerprint` (deck in order + board bbox/occupied), with asserts that forced
> discards (phase-gated, NOT bare `p.is_none()` — opening picks are also
> placement-None) and ≥50 round boundaries were actually exercised; the full-playout
> test asserts `GAME_OVER` is reached; atomicity (illegal action → `Err`, state
> unchanged) is covered.
> Note: 4 `solver_restructure_tests` fail under `cargo test --lib` — a PRE-EXISTING
> debug-build timeout (they use a 60s wall-clock deadline; debug solves take ~125s;
> confirmed identical failure on the stashed baseline). Run the crate's heavy
> solver tests in `--release`. Next: Step 2 (Python-hosted smoke).

> **STATUS — Step 2 DONE (2026-07-12).** `make`/`unmake` exposed to Python via a
> `#[pyclass] SearchEngine` (owns a `RustGameState` + an internal `Vec<UndoRecord>`
> stack; `make` / `unmake` / `make_with_row` + read-delegates + `official_outcome`,
> which mirrors `determine_winner`'s `(total, largest_territory, crowns)` cascade
> in Rust). Finding from the build: the Phase-0 bot is PURE PYTHON on the Python
> `GameState`, so Step 2 was a real port, not a verbatim reuse. Deliverables, all
> green: `test_search_engine.py` (make==step + unwind at the Python boundary,
> `official_outcome` vs `determine_winner`, `make_with_row` chance-child mechanics);
> `rust_expectiminimax.py` (`RustExpectiminimax`, the make/unmake-driven search
> mirroring `expectiminimax.py`); `test_rust_expectiminimax_equiv.py` (byte-identical
> search VALUES vs the pure-Python `ExpectiminimaxBot` with chance fully enumerated,
> both evals, depths 2–3). **Depth/speed read** (`bench_rust_expectiminimax.py`,
> release, deck-12 PLACE_AND_SELECT, pick_aware): pure-Python ~5.5k nodes/s vs
> Python-hosted make/unmake 130k–220k nodes/s = **24–40×**, identical node counts &
> values. Depth 4 in 0.18s (was 4.3s); depth 5 ≈ a few s; depth 6 stays out of
> reach Python-hosted → the per-node FFI is now the bottleneck, confirming Step 3
> (host the recursion in Rust) is what unlocks the depth-6 vs-AZ measurement.
>
> **Step 2 review hardening (2026-07-12):** (1) `make_with_row` now VALIDATES the
> row before mutating — exactly 4 distinct tiles, all in the pre-deal bag, and the
> action must actually deal; failure leaves the engine + undo depth unchanged
> (was: silently accepted malformed/empty rows, a clairvoyance risk). (2)
> `rust_expectiminimax._chance_rows` uses the reference's blake2 `_stable_seed`, so
> sampled (wide) chance nodes now match the Python bot byte-for-byte and are
> cross-version reproducible (was: Python builtin hash → divergent samples). (3)
> the Python search wraps make/unmake in `try/finally`, so a raising eval_fn
> unwinds the shared engine. (4) the Python-boundary test now fingerprints the full
> public state (boards + discards + scalars), not a spot-check. Tests: 7 green
> (added `make_with_row` negative tests, sampled-row equivalence, evaluator-
> exception unwind). Next: Step 3.


**Goal.** Give the expectiminimax searcher a mutable, reversible engine so a single
search walks one `RustGameState` down and back up the tree instead of cloning a
fresh state per node. This is the measured critical-path bottleneck (Phase-0
Python peaked at ~6.7k nodes/s → depth-2 ceiling; the endgame solver's ~1µs/node
is dominated by `step()`'s clone + legal-gen, not scoring). Get this and the
existing `ExpectiminimaxBot` + hand eval reaches depth 4–6 → first
"approaches/beats AZ?" signal.

Non-goals here: the NNUE accumulator, sparse features, in-tree Rust chance,
Star2. Those come after this proves out (see NNUE_PROJECT_PLAN.md sequencing).

## Decision: add alongside, do not replace `step`

Keep the existing functional `step` (lib.rs:1428) exactly as is. Add a parallel
mutable path: `make(action) -> UndoRecord` and `unmake(record)`. Reasons:

1. `step` stays the correctness oracle — the differential test below asserts
   `make` reproduces `step` byte-for-byte, and `make`+`unmake` round-trips to the
   original bytes. We already have `solver_state_bytes` (lib.rs:2484) to serialize
   for that comparison; no new infra needed.
2. `step` is still the Python-boundary entry point for self-play / bot_match.
3. Zero risk to the shipped AZ pipeline — this is purely additive.

## The state-transition surface (verified in lib.rs)

`RustGameState` fields that any action can touch: `boards[2]`, `deck`,
`current_row`, `pending_claims`, `next_claims`, `phase`, `actor_index`,
`initial_pick_count`, `discards[2]`. (`start_player`, `harmony`,
`middle_kingdom` are immutable within a game.)

Three action shapes, from `step`:

- **INITIAL_SELECTION pick** — `current_row.remove(pos)`; `next_claims.push`;
  `initial_pick_count += 1`. On the 4th pick only: sort `next_claims` → move to
  `pending_claims`, `deal_row()`, reset `actor_index`, `phase → PLACE_AND_SELECT`.
- **PLACE_AND_SELECT / FINAL_PLACEMENT move** — either `boards[p].place(...)` or
  `discards[p] += 1`; (PLACE only) `current_row.remove(pos)` + `next_claims.push`;
  `actor_index += 1`. On wrap: `FINAL → GAME_OVER`, else `advance_round()`.

### Reversibility analysis (the crux is the board bbox)

- **`place` (lib.rs:245) is NOT cleanly self-inverse.** It writes 2 cells
  (`terrain`, `crowns`), `occupied += 2`, and updates `min_x/max_x/min_y/max_y`
  **monotonically**. Clearing the two cells and `occupied -= 2` is trivial, but
  the bbox cannot be recomputed cheaply on undo. → **The undo record must snapshot
  the pre-move bbox (4 × i8) and restore it verbatim.** This is the one place a
  naive "just clear the cells" undo is wrong.
- **`current_row.remove(pos)`** shifts the tail. Undo = re-insert the removed
  domino id at `pos`. Record stores `(pos, domino_id)`.
- **`next_claims.push`** → undo is `pop`. Free.
- **Scalar bumps** (`initial_pick_count`, `actor_index`, `discards`, `phase`) →
  record the old value, restore.
- **Round boundaries (`deal_row`/`advance_round`, the 4th-initial-pick branch)**
  rewrite `deck`, `current_row`, and reshuffle `pending_claims`/`next_claims`.
  Don't try to invert the deal logic. These vectors are tiny (deck shrinks by 4,
  the claim vecs are ≤4 each), so the undo record just **snapshots the affected
  vectors** and restores them. Still vastly cheaper than a full `cloned()` (which
  copies two 225-cell boards every node).

The hot path — the millions of interior placement/pick nodes — pays only:
2 cell writes + 1 `occupied` bump + bbox snapshot + one `remove`/`push` pair.
The heavy round-boundary undo fires ~12 times per game line, not per node.

## Proposed shape

```rust
enum UndoRecord {
    InitialPick { removed_pos: usize, domino_id: u16, boundary: Option<RoundSnapshot> },
    Move {
        player: u8,
        placed: Option<PlaceUndo>,   // None = forced discard
        pick: Option<(usize, u16)>,  // None in FINAL_PLACEMENT
        boundary: Option<RoundSnapshot>,
    },
}
struct PlaceUndo { i1: usize, i2: usize, bbox: (i8,i8,i8,i8), /* occupied implicit -2 */ }
struct RoundSnapshot { deck: Vec<u16>, current_row: Vec<u16>,
                       pending_claims: Vec<(u8,u16)>, next_claims: Vec<(u8,u16)>,
                       actor_index: usize, phase: u8 }
```

(As implemented, `RoundSnapshot` deliberately OMITS `initial_pick_count`: no
boundary transform mutates it, so the per-move `-= 1` in `unmake` fully reverses
it. It is snapshotted nowhere.)

`make` mirrors `step`'s branches but mutates `self` and returns the record;
`unmake` replays it in reverse. Search keeps records on its own stack (or the
Rust recursion frame), so no allocation churn beyond the rare `RoundSnapshot`.

## Equivalence testing (the gate)

Byte-exact comparison uses a dedicated full-field `fingerprint` (test-local), NOT
`solver_state_bytes`. The latter sorts the deck and omits the board bbox /
`occupied` — it would mask exactly the two failure modes this gate exists to
catch (a deck-ORDER leak and the bbox-restore hazard). `fingerprint` serializes
every field, deck in order, including each board's bbox and `occupied`.

Three properties (as implemented in `mod make_unmake_tests`):

1. **make == step.** For random legal action sequences from `new_game`, at every
   ply assert `fingerprint(clone.make(a))` == `fingerprint(orig.step(a))`.
2. **make/unmake round-trips.** Snapshot fingerprint, `let u = s.make(a);
   s.unmake(u);` assert unchanged. Run across 200 full random games (all three
   phases, placements AND forced discards AND round boundaries AND terminal), with
   asserts that forced discards and ≥50 boundaries were actually exercised. NOTE:
   the discard counter must require `phase != INITIAL_SELECTION` — opening picks
   also carry `placement=None`, so counting bare `p.is_none()` is a false positive.
3. **atomicity.** Illegal actions (invalid pick, illegal placement, wrong-phase
   shape, terminal `make`) return `Err` and leave the fingerprint unchanged.

A `full_playout_unwinds_to_start` test also walks one mutable state to
`GAME_OVER` via `make` (asserting the phase reached), cross-checks each ply
against a parallel `step` reference, then unwinds the whole stack via `unmake`
back to the byte-identical start — the real search-walk discipline. This is the
same differential methodology that validated the AZ engine port.

## Then what (defines "done" for the strength signal)

Once make/unmake is green, wire `ExpectiminimaxBot`'s node recursion to it (or a
thin Rust search shell over it), push depth to 4–6 with `pick_aware_p0`, and run
vs the AZ agent to set the real Phase-1 bar (Greedy is already beaten at depth 2,
per the paired control). No accumulator needed for that measurement.

## Open questions for you

- **Search host:** keep the tree walk in Python calling Rust `make`/`unmake` per
  node (simplest, but ~1 FFI hop/node caps us ~10⁵–10⁶ nodes/s), or move the
  alpha-beta recursion itself into Rust (generalize `solve_endgame_ab` from the
  no-chance tail to the whole game) for full CPU speed? I lean Rust-hosted search
  since the whole point is depth, but the Python-hosted version is a faster first
  light and reuses `ExpectiminimaxBot` verbatim.
- **Legal-gen cost:** `legal_actions_indexed` rebuilds each node. If profiling
  shows it dominates once cloning is gone, incremental move-gen is a follow-up —
  but not in this work item.
