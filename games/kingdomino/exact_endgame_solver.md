# Exact Endgame Solver — Reference Document

## Purpose

This document tracks the design, implementation status, empirical results, and
optimization roadmap for the exact endgame solver in the Kingdomino AlphaZero
project. Update as results come in.

---

## Why This Exists

AlphaZero MCTS never plays to terminal nodes — it stops at leaves and uses the
network's value estimate. In terminal-adjacent positions (deck=0 or deck=4) the
network estimate is miscalibrated and does **not** improve with more simulations.
More sims just averages more instances of the same wrong estimate.

The fix: detect terminal-adjacent positions during search and replace the network
value with exact minimax scoring by playing out all remaining moves exhaustively.

**Training leverage is higher than advisor leverage.** Every self-play game that
reaches an endgame position now produces a perfectly-grounded value target. This
compounds across all future training. Reanalysis (Milestone 9) can retroactively
relabel buffer positions that were played before the solver existed.

---

## Game Structure Facts

In 2-player Mighty Duel Kingdomino, 4 tiles are revealed per round and 2 players
each act once. This means `deck.len()` is always a multiple of 4:

```
deck=0  → FINAL_PLACEMENT (last 4 tiles are pending_claims, no more draws)
deck=4  → one hidden draw remains; next current_row = sorted(deck)
deck=8+ → out of scope for exact solving
```

**deck=1,2,3 never occur.** Any code handling those is dead code.

### Chance-Node Collapse for deck=4

When `deck.len() == 4`, those 4 tiles are hidden but will deterministically become
the next `current_row` via `deal_row()` which sorts them (`row.sort_unstable()`).

All 4! = 24 orderings of those 4 tiles produce an **identical** sorted
`current_row`. Therefore:

- There is NO chance branching for the deck=4 case
- The endgame tree is pure minimax, not expectimax
- The solver enumerates the 4 tiles as a known (sorted) next row

This collapses the naïve 24-branch expectimax (~19M nodes) to a plain minimax
(~3M nodes median). The collapse is real and correctly implemented — it falls
out naturally from `step()` → `advance_round()` → `deal_row()` without any
special handling in the solver.

---

## Current Implementation Status

### What Exists (current)

| Component | File | Status |
|-----------|------|--------|
| Rust minimax solver (reference) | `lib.rs` → `exact_solve_no_chance` | ✅ kept as reference (`#[allow(dead_code)]`) |
| Rust **alpha-beta** solver (hot path) | `lib.rs` → `solve_endgame_ab` | ✅ single-pass, budgeted, pruning (OPT-2/3) |
| Rust move ordering | `lib.rs` → `order_legal_for_solver` / `placement_score_delta` / `pick_order_score` | ✅ OPT-4b (exact score delta) |
| Rust **parallel** solver (YBW) | `lib.rs` → `solve_endgame_ab_parallel` | ✅ OPT-6 (root-only; mixed — see results) |
| Rust budget counter | `lib.rs` → `exact_count_no_chance_bounded` | ✅ full-tree count (diagnostics/tests) |
| Rust `is_no_chance_endgame_state` | `lib.rs` | ✅ deck ∈ {0,4} |
| Rust `terminal_search_value` | `lib.rs` | ✅ Matches Python exactly |
| Python-callable wrapper | `lib.rs` → `exact_endgame_value_no_chance` | ✅ now single-pass alpha-beta; returns (value, solved, pruned_nodes) |
| Rust state injection | `lib.rs` → `RustGameState.from_parts` | ✅ Working |
| Python bridge | `endgame_solver.py` | ✅ Routes Rust-first, Python fallback |
| MCTS hook | `mcts_az.py` → `OpenLoopMCTS._simulate` | ✅ + OPT-1 solve-once cache + root gate |
| Default threshold | `mcts_az.py` | `max_hidden_tiles=4`, `max_nodes=500_000` |
| Test suite | `test_endgame_exact.py` | ✅ 21 tests passing |

> **Scope note:** exact solving now runs in both the Python `OpenLoopMCTS` path
> and the Rust `BatchedMCTS` training-generation path. The Rust batched hook is
> implemented in `BatchedMCTS::resolve_exact_slots` and is the routine training
> path for `engine=batched_open_loop`.

**Current status update:** BatchedMCTS training is now wired into the exact
solver. `BatchedMCTS::resolve_exact_slots` solves terminal-adjacent roots once,
skips GPU/MCTS for those moves, emits exact child-value policy targets, and
exposes `exact_solve_count` / `exact_fallback_count` plus split attempt/fallback
counters for `deck4_initial`, `deck4_retry`, and `deck0`. The routine self-play
default is `exact_endgame_max_secs=3.0`, exposed as a CLI/run setting and shared
by all three solve entry points; `0.0` disables exact solving for ablation and
higher values are available for quality-first/reanalysis runs. The focused suite
is 38 tests passing, including BatchedMCTS integration tests.

1600-sim smoke result over 32 games, 48x6 CUDA: `off=0.0742 games/s`,
`500k=0.0816`, `2M=0.0769`, `5M=0.0760`. 500k gave the best throughput in that
routine-training comparison while still solving most exact roots.

**Note:** The Python `exact_endgame_value_no_chance` fallback only handles deck=0
and deck=1 (which never occurs). For deck=4, Python is too slow — Rust only.

### API Shape (actual, not the original prompt spec)

The solver budget is **wall-clock time** (`max_secs`), not a node count. The
node-count budget was replaced because throughput is what matters: a node budget
times out at a roughly fixed wall-clock cost regardless of position complexity,
whereas a time budget bounds the thing we actually care about. `count_endgame_nodes`
and the Rust `count_endgame_nodes_no_chance` still report full unpruned tree sizes
for diagnostics, but the solver itself no longer exposes a pruned-node count.

```python
# Python-side bridge
from games.kingdomino.endgame_solver import exact_endgame_value

value, solved = exact_endgame_value(
    state,
    max_secs=3.0,      # per-position wall-clock budget; <= 0 disables
    rng=...,           # unused in deterministic minimax; API compat only
    score_scale=100.0,
    margin_gain=2.0,
    alpha=0.0,         # use 0.0 for eval/advisor; 0.8 for training
)

# Rust direct call (for benchmarking/testing) — note it returns elapsed_secs,
# not a node count.
value, solved, elapsed_secs = kr.exact_endgame_value_no_chance(
    rust_state,
    3.0,               # max_secs
    100.0,             # score_scale
    2.0,               # margin_gain
    0.0,               # alpha
)
```

---

## Empirical Results

### Throughput baseline

Rust solver speed: **~700k nodes/sec** (constant; alpha-beta wins by visiting far
fewer nodes, not by going faster per node).

### deck=0 (FINAL_PLACEMENT)

- Solves reliably in <1ms for any reasonable budget
- ~200x faster than Python solver
- Solve rate: ~100% at max_nodes=50_000
- **Training impact: immediate and reliable. This is already delivering value.**

### deck=4 (PLACE_AND_SELECT, one hidden draw)

**Full (unpruned) tree size** — counted over 20 real positions, cap 50M:
median ≈ **9.9M nodes**, p75 ≈ 37M, max > 50M. (The original prompt's ~810k
estimate was wrong by ~12×.) 85% of positions exceed 500k nodes unpruned.

#### Before optimization (plain minimax, count-then-solve)

| Budget | Solve rate | Median time (solved) |
|--------|-----------|---------------------|
| 50,000 | 0% | — |
| 500,000 | 17% | ~0.6s |
| 2,000,000 | 42% | ~2.9s |

#### After OPT-2 + OPT-3 + OPT-4 (single-pass alpha-beta + move ordering)

Same 30 real positions:

| Budget | Solve rate | Median **pruned** nodes | Median time | Max time |
|--------|-----------|-------------------------|-------------|----------|
| 50,000 | 60% | — | — | — |
| 500,000 | **97%** | **~44k** | **~73ms** | ~580ms |
| 2,000,000 | **100%** | ~44k | ~73ms | ~1.1s |

**Headline:** at the 500k budget, solve rate went **17% → 97%**, and the median
traversal dropped from ~9.9M full-tree nodes to **~44k pruned nodes** (~225×
fewer nodes visited). Alpha-beta + the crowns/region move-ordering heuristic do
the work; per-node throughput is unchanged.

### Full tail distribution (for the always-solve decision)

Measured by `bench_endgame_tail.py` → `RustGameState.measure_endgame_tree`
(benchmark-only method: same `solve_endgame_ab`, plus timing and a 50M safety
ceiling). **n=300 real deck=4 positions, alpha=0.8 (training frame), cap 50M:**

| | nodes (pruned) | single-core time |
|---|---|---|
| p50 | 111,856 | 193 ms |
| p75 | 287,256 | 480 ms |
| p90 | 712,421 | 1.25 s |
| p95 | 1,192,836 | 2.12 s |
| p99 | 3,661,963 | 6.9 s |
| max | 14,700,596 (seed=8) | 25.2 s |

**100% solved within the 50M ceiling; 0% hit it.** Throughput ~601k nodes/sec.
Only one position (seed=8, 0.33%) exceeds 10M nodes; the next worst is 7.3M
(seed=155). Nothing approaches the 50M "unsolvable" regime. Reproduce the
heaviest positions via their seeds (`--base-seed 8 --n 1`, etc.).

**Pruning depends on alpha** (same 50 positions, cap 10M):

| alpha | p50 | p90 | p99 | seed=8 (worst) |
|---|---|---|---|---|
| 0.0 (eval/advisor) | 54.5k | 205k | 7.9M | 7.9M |
| 0.8 (training) | 106k | 503k | >10M | 14.7M |

alpha=0.0 prunes ~2× better: coarse win-value leaves ({-1, 0, +1}) produce many
exact ties → frequent alpha-beta cutoffs, whereas the continuous tanh-margin
leaves at alpha=0.8 rarely tie → fewer/later cutoffs. **Training's alpha=0.8 is
the worst case**, so the n=300 alpha=0.8 numbers above are the binding figures;
an alpha=0.0 advisor sees roughly half the tail.

### Always-solve readiness (read)

The tail is bounded and rare — nothing near unsolvable — so OPT-6 is **not** a
prerequisite for always-solve in **training**: ship it now with a ~20M budget
(catches 100% of observed positions with margin). The cost is a rare per-root
stall (~1% of deck=4 roots ≈ 7 s, ~0.3% ≈ 25 s, single-core), acceptable in a
throughput generator and paid once per root via OPT-1. **The interactive advisor
is different**: a 7 s (p99) to 25 s (max) wait is unacceptable UX, so there either
keep the budgeted network fallback (500k → 97%, ~73 ms) or build OPT-6 (8-core
YBW → p99 ~1 s, max ~3-4 s) before enabling always-solve. A cheaper lever than
full Rayon for the training tail: a stronger OPT-4 move-ordering heuristic, since
alpha=0.8's poor pruning is largely an ordering/leaf-granularity problem.

### OPT-4b results — better move ordering (exact score delta)

Replaced the crowns × neighbour-count proxy with the **exact immediate territory
score delta** of each placement (`placement_score_delta`, via scoped per-region
BFS) plus a terrain-weighted pick heuristic (`pick_order_score`). Values unchanged
(advisory only). **n=300, alpha=0.8, serial:**

| | OPT-4 (before) | OPT-4b (after) | gain |
|---|---|---|---|
| p50 | 111,856 | 97,691 | 1.15× |
| p95 | 1,192,836 | 951,627 | 1.25× |
| p99 | 3,661,963 | **1,905,650** | **1.9×** |
| max | 14,700,596 (25.2 s) | **4,567,099** (8.6 s) | **3.2×** |

The old worst case (seed=8) fell 14.7M → 2.45M (6×). The alpha gap narrowed: at
the tail, alpha=0.8/alpha=0.0 went from ~1.9× to ~1.56× (both improved 5–6× at the
worst case). **OPT-4b alone moves training into the always-solve zone (serial, 15M budget)** (p99 1.9M,
max 4.57M) with a clean, predictable serial cost — no parallelism needed there.

### OPT-6 results — Rayon / Young Brothers Wait (mixed)

Root-only YBW: first (best-ordered) child solved serially to set a bound, the rest
in parallel with that bound, each given the FULL budget. Values identical to serial
(equivalence-tested). **n=300, alpha=0.8, time percentiles:**

| | serial (OPT-4b) | parallel (OPT-6) |
|---|---|---|
| p50 | 176 ms | 130 ms |
| p95 | 1.85 s | 1.29 s |
| p99 | 4.02 s | 3.01 s |
| max | 8.60 s | **3.71 s** (2.3×) |

**The speedup is real but modest and non-uniform** — far from 8×/16× ideal — because
root-only parallelism sacrifices alpha-beta's *sequential* bound tightening (each
sibling normally improves the bound for the next; in parallel they all share only
the first child's bound). Per-position diagnostics (16 cores):

| seed | shape | serial | parallel | speedup | node inflation |
|---|---|---|---|---|---|
| 155 | 64 root children, well-fanned | 9.2 s | 2.8 s | **3.3×** | 4.57M→5.2M (1.1×) |
| 8 | critical-path-dominated | 4.8 s | 3.5 s | 1.4× | 2.45M→5.38M (2.2×) |
| 68 | 20 children, pruning-dependent | 2.8 s | 3.4 s | **0.82× (slower!)** | 1.38M→4.72M (3.4×) |

**OPT-6 can be a net loss** on positions where serial alpha-beta relied on
progressive sibling-bound tightening: weak parallel bounds inflate node counts
(up to 3.4×), occasionally beyond what the extra cores recover. It helps the
worst-case wall-clock (max 8.6 s → 3.7 s) but does **not** reach the advisor's
sub-second-p99 goal (p99 still ~3 s). True sub-second p99 needs deeper parallelism
(recursive YBW with a depth cutoff), a future OPT — root-only YBW is not enough.

### Current defaults / recommendation (post BatchedMCTS integration)

The budget is now **wall-clock seconds** (`exact_endgame_max_secs`, default `3.0`),
not a node count. The earlier node-budget tuning (500k best, 2M/5M slower) is
historical; see the smoke-result tables below for context.

- **Routine training: serial exact solves with `exact_endgame_max_secs=3.0`.** The
  per-game `exact_unsolvable` sentinel means a hard endgame costs at most one
  timeout per game (≤ `max_secs`) instead of repeatedly re-solving every remaining
  move, so the previous fallback-retry overhead is gone.
- **Quality/reanalysis: use a larger `max_secs` selectively.** This is the right
  place to pay for hard endgames if the goal is cleaner labels rather than maximum
  self-play throughput.
- **Advisor: parallel solver and a larger `max_secs`.** User-facing advisor mode
  should prefer coverage and wall-clock latency over training throughput. OPT-6
  improved the tail (max 8.6 s -> 3.7 s), but true sub-second p99 likely needs
  deeper/recursive YBW or additional pruning improvements.
- The production `exact_endgame_value_no_chance` pyfunction routes through the
  parallel solver and returns `(value, solved, elapsed_secs)`. The batched training
  path controls its own budget through `SelfPlayConfig.exact_endgame_max_secs` / CLI
  `--exact_endgame_max_secs`.


### Current MCTS hook behavior

The previous pre-OPT-1 behavior (retrying an expensive deck=4 solve every simulation) has been replaced.

Current Python `OpenLoopMCTS` behavior:

- Exact solving is enabled only when the search root is terminal-adjacent (`deck.len() <= max_hidden_tiles`). This avoids caching a deck=4 leaf reached from a deck>=8 root, where the hidden row depends on the determinization.
- On the first exact-solvable leaf, `_simulate` attempts the exact solve.
- If solved, the value is stored on `OpenLoopNode.exact_value`; later simulations that reach the same node return the cached value immediately.
- If the budget is exceeded, the node falls back to normal network evaluation.

Current Rust `BatchedMCTS` behavior:

- `BatchedMCTS::resolve_exact_slots` checks terminal-adjacent roots before GPU evaluation/MCTS expansion.
- When solvable, it builds an exact continuation plan, emits exact child-value policy targets, chooses the minimax move, and skips GPU evaluation for those moves.
- Plan reuse means one expensive exact tree solve can serve the deterministic continuation moves from that root; diagnostics expose tree solves, cache hits, total exact moves, and fallbacks.

### No-retry-after-timeout (retry-storm fix)

Both paths avoid re-attempting an expensive solve that has already timed out:

- **Rust `BatchedMCTS` — bounded retry sentinel.** When `solve_exact_plan`
  exceeds the per-position `exact_endgame_max_secs` budget, the slot sets an
  `exact_unsolvable` flag and falls back to MCTS. `finalize_move` then refuses
  to re-enter `ExactSolving` for the same full deck=4 root, but it still allows
  cheap `deck=0` exact solving and one later deck=4 retry after the current row
  has progressed to two or fewer remaining claims. A 20-position sample showed
  deck=4 after two random moves was ~15x faster at the median than the first
  deck=4 root. The flag is reset in `new_for_game`, so each new game gets a
  fresh initial attempt.

- **Python `OpenLoopMCTS` — three-state node cache.** `OpenLoopNode.exact_value`
  is one of three states: `None` (Unsolved), a `float` (Solved — returned in O(1),
  node stays unexpanded), or the `_EXACT_UNSOLVABLE` sentinel (timed out — the node
  was expanded once via the network and is never re-solved). On the first timeout
  the search also clears `_exact_endgame_active`, so no other leaf in that search
  re-attempts the solver. Counters reset per `search()`.

---

## Optimization Backlog

Historical optimization notes. Completed items are retained for design rationale; remaining future work is called out in the build-order retrospective below.

---

### OPT-1: Solve-once caching via `OpenLoopNode.exact_value`
**Status:** ✅ Implemented (with a correctness refinement — see below)
**Priority:** Highest — architectural fix, prerequisite for deck=4 viability

> **Correctness refinement (important — deviates from the original note below).**
> The original claim "all determinizations that reach a deck=4 node see the same
> concrete game state" is **only true when the search ROOT is terminal-adjacent**
> (`len(root.deck) <= max_hidden_tiles`). From a deck≥8 root, descending to a
> deck=4 leaf passes through a hidden-row reveal (which 4 of the 8+ tiles get
> dealt), so that leaf's board **does** depend on the determinization — caching a
> single value there would be wrong, and with the 500k budget, per-sim re-solving
> of such deep leaves would also wreck throughput.
>
> Fix: `OpenLoopMCTS.search()` sets `self._exact_endgame_active` once per search,
> true only when the root is terminal-adjacent. `_should_exact_solve()` requires
> it. In that regime the *only* hidden information is the ORDER of the ≤4 remaining
> tiles (irrelevant — they become a sorted row), so every node's concrete state is
> determinization-independent and the cache is always correct. From a
> non-terminal-adjacent root, exact solving is skipped entirely; the grounding
> happens later, when the game actually reaches a terminal-adjacent position and
> that move's search solves it. Verified by
> `test_exact_solving_gated_off_at_non_terminal_adjacent_root`.

**Problem:** The current hook fires inside `_simulate`, meaning for a 1600-sim
search, a deck=4 leaf is attempted 1600 times. Budget hit 1600 times = 1600 × 55ms
= ~88s overhead for zero benefit.

**Fix:** Add `exact_value: Optional[float] = None` to `OpenLoopNode.__slots__`.
On first simulation hitting an exact-solvable leaf:
- If `node.exact_value is not None`: use cached value, skip all evaluation (O(1))
- If `_should_exact_solve(state)`: attempt solve; if `solved`, store in `node.exact_value`
- If budget exceeded: fall through to `_expand_and_evaluate` as before

The node stays `is_expanded=False` when `exact_value` is set — the PUCT descent
loop `while node.is_expanded` won't descend below it, so every sim that reaches
this node returns immediately with the cached value.

**Correctness:** Valid because the exact solver uses only public information (boards,
current_row, pending_claims) — not deck order. All determinizations that reach a
deck=4 node see the same concrete game state, so the cached value is correct for
all of them.

**Expected gain:** 1600x reduction in solver calls per search. Deck=4 pays the
solve cost exactly ONCE per search, not per simulation.

**Files:** `mcts_az.py` (OpenLoopNode, `_simulate`)

---

### OPT-2: Single-pass budgeted solver (replace count-then-solve)
**Status:** ✅ Implemented (`solve_endgame_ab`)
**Priority:** High — easy win, no risk

> The solve path no longer pre-counts. `exact_endgame_value_no_chance` calls
> `solve_endgame_ab` directly, which aborts mid-traversal on budget. The standalone
> full-tree count (`count_endgame_nodes_no_chance`) is retained for diagnostics and
> equivalence tests. Returned `nodes` is now the pruned traversal count.

**Problem:** Current implementation counts the full tree first (traversal 1), then
solves it if within budget (traversal 2). On budget misses (83% of deck=4 positions
at 500k), the tree is traversed fully once and then aborted — paying full cost for
zero benefit.

**Fix:** Replace with a single-pass recursive solver that aborts mid-traversal when
the node counter hits the budget:

```rust
fn solve_endgame_budgeted(
    state: &RustGameState,
    nodes: &mut u64,
    max_nodes: u64,
    alpha: f64, beta: f64,  // alpha-beta bounds (see OPT-3)
    score_scale: f64,
    margin_gain: f64,
    alpha_param: f64,
) -> Result<f64, ()> {  // Err(()) = budget exceeded
    if state.phase == GAME_OVER {
        *nodes += 1;
        return Ok(terminal_search_value(state, score_scale, margin_gain, alpha_param));
    }
    if *nodes >= max_nodes {
        return Err(());
    }
    // ... minimax over legal actions, returning Err on budget exceeded mid-tree
}
```

**Expected gain:** ~2x on budget misses (eliminates the count pass). No gain on
solved positions. Combined with OPT-1 (solve-once), budget misses become rare
after the first simulation.

**Files:** `lib.rs` (replace `exact_count_no_chance_bounded` + `exact_solve_no_chance`
with single `solve_endgame_budgeted`)

---

### OPT-3: Alpha-beta pruning
**Status:** ✅ Implemented (in `solve_endgame_ab`)
**Priority:** High — multiplicative gain on node count

> Standard fail-soft alpha-beta. **The search runs on the RAW integer score margin
> `(s0 - s1)`, player-0 frame, range ~[-80, 80]**, with the window initialised to
> [-200, 200] (`MARGIN_LO`/`MARGIN_HI`). Integer margins have the widest spread →
> tightest bounds, and carry no training hyperparameters. The training value
> (`terminal_search_value`/`margin_to_training_value`) is a monotone transform
> applied AFTER the solve, so the minimax-optimal move is identical to a search run
> directly in value space (the monotonicity guarantee). Max/min layers are typed
> per node by `actor()` and need not strictly alternate. Value is bit-identical to
> the value-space search — verified against the Python expectiminimax
> (`test_rust_deck_four_matches_python`), via budget/order invariance
> (`test_alpha_beta_value_exact_and_public_consistent`), and via ranking invariance
> (`test_raw_margin_matches_training_value_ranking`).

**Problem:** Plain minimax evaluates every node in the tree. Alpha-beta prunes
subtrees that cannot affect the result.

**Correctness:** The terminal value formula `alpha * tanh(score_diff * margin_gain)
+ (1-alpha) * win_value` is monotone in score_diff, so alpha-beta applies correctly.
Standard alpha-beta with (alpha, beta) bounds in [-1, 1].

**Expected gain:** 1.5-3x node reduction on random move order, more with good move
ordering (see OPT-4). Effective throughput with OPT-3 alone: ~1-2M nodes/sec
effective (same 700k nodes/sec but fewer nodes visited).

**Implementation note:** Must be combined with OPT-2 (single-pass) to work — the
count-then-solve design cannot use alpha-beta in the count pass.

**Files:** `lib.rs` (add alpha/beta parameters to solver recursion)

---

### OPT-4: Move ordering (pairs with OPT-3)
**Status:** ✅ Implemented; superseded by **OPT-4b** (see results above)
**Priority:** High (only valuable alongside OPT-3)

> OPT-4 (original): light heuristic — placements scored by crowns × matching-terrain
> neighbours, picks by claimed-domino crowns. Cut the median deck=4 traversal to
> ~44k pruned nodes (from ~9.9M full).
>
> OPT-4b (current, `placement_score_delta` + `pick_order_score`): exact immediate
> territory score delta per placement + terrain-weighted pick value. Cut p99 1.9×
> and max 3.2× over OPT-4 (see "OPT-4b results" above). Advisory only — values
> unchanged. The serial solver (`solve_endgame_ab`) uses this ordering; the
> parallel solver (`solve_endgame_ab_parallel`, OPT-6) reuses the same
> `order_legal_for_solver`.

**Problem:** Alpha-beta's effectiveness is entirely dependent on move order. With
random ordering: ~1.5x pruning. With perfect ordering: ~sqrt(N) pruning (2x more
aggressive than even a good heuristic).

**Heuristic for Kingdomino:** Sort placements by immediate score contribution
descending — placements that connect more crowns to larger regions of matching
terrain come first. This is computable in O(placements) with a simple crown-count
heuristic using the existing board data.

Concretely, for each candidate placement:
1. Compute crowns contributed by the new tile halves
2. Count adjacently-connected terrain of matching type (approximation of region size)
3. Score = crowns × connected_area (higher = try first for maximizing player)

For the minimizing player (opponent), try high-score placements first too (they
are worst for the maximizing player).

Pick ordering: try the domino with the most crowns first (simple, O(1) per pick).

**Expected gain combined with OPT-3:** 2-4x over plain minimax. Together with
700k base throughput → effective ~1.4M-2.8M nodes/sec.

**Files:** `lib.rs` (sort legal actions before recursion)

---

### OPT-5: Undo/redo state mutation
**Status:** Not yet implemented  
**Priority:** Medium — significant memory/allocation reduction

**Problem:** Current `step_internal` calls `cloned()` which copies both RustBoard
terrain arrays (2 × 225 bytes = 450 bytes) plus Vec clones on every node. At 3M
nodes per tree: ~1.35GB of allocation/deallocation. Allocator pressure is real at
this scale.

**Fix:** Mutate state in-place and undo after recursion. For a placement:
- Save: 2 cells (x1,y1) and (x2,y2) terrain+crowns, old bbox, old occupied count
- Apply: write terrain/crowns, update bbox, increment occupied
- Undo: restore saved values exactly

For round transition (advance_round):
- Save: pending_claims, next_claims, current_row, actor_index, phase, deck
- Apply: promote next_claims → pending_claims, deal row if deck non-empty
- Undo: restore saved values

**Expected gain:** 20-40% reduction in solver time from eliminated allocation.
Combines multiplicatively with OPT-3/OPT-4.

**Risk:** Medium. Undo logic must be correct — a missed undo corrupts all
subsequent evaluations. Requires careful testing with equivalence checks against
the cloning solver.

**Files:** `lib.rs` (new solver variant with explicit undo stack)

---

### OPT-6: Rayon parallelism (Young Brothers Wait)
**Status:** ✅ Implemented (`solve_endgame_ab_parallel`) — mixed result, see
"OPT-6 results" above. Root-only YBW gives 2.3× on max wall-clock but can be a net
loss (0.82×) on pruning-dependent positions due to node inflation, and does not
reach the advisor's sub-second-p99 goal. **Recommendation: serial for training,
parallel for advisor; true sub-second p99 needs recursive-depth YBW (future).**
**Priority:** Medium — wall-clock speedup, not node-count reduction

**Problem:** The solver is serial. The machine has 16 logical cores sitting idle
during a solve.

**Approach:** Young Brothers Wait (YBW) — a parallelization strategy compatible
with alpha-beta:
1. Solve the first root child serially to establish an initial alpha bound
2. Solve remaining root children in parallel using Rayon, each with the alpha bound
3. Collect results, update the global best

Pure parallel without alpha-beta wastes work (solves subtrees that serial
alpha-beta would prune). YBW preserves most of the pruning while parallelizing.

**Expected gain:** Up to 8x wall-clock speedup on the 8-core machine. At 8x:
advisor deck=4 median solve at 2M nodes: 2.9s → ~360ms.

**Risk:** Medium-high. YBW adds complexity. Rayon is already imported in lib.rs.
Only build after OPT-3 alpha-beta is working and measured.

**Files:** `lib.rs` (Rayon par_iter over root children)

---

### OPT-7: Incremental score tracking
**Status:** Measure before building  
**Priority:** Low until profiled

**Problem:** `board.score()` runs a BFS flood-fill over the board at every terminal
leaf. At 700k nodes/sec with ~40% being leaves, that's ~280k flood-fills/sec.

**Fix:** Track score incrementally during placement — compute only the score delta
for the newly placed tile's connections instead of full board BFS.

**When a tile is placed at (x1,y1) and (x2,y2):**
- Check adjacent cells of matching terrain type
- If the new tile bridges two previously disconnected regions, score changes
  non-locally (old scores of both regions become invalid)
- If the new tile extends one region, delta = new_crowns × old_region_size +
  new_region_cells × old_region_crowns + new_crowns × new_cells

The bridging case is the tricky one — requires tracking connected components.

**Expected gain:** Unknown without profiling. If flood-fill is 30% of solver time,
this is ~30% speedup. If it's 5%, not worth building.

**Action:** Profile first. Run solver with `perf` or Rust `Instant` timing on
`score()` vs total to determine if this is a bottleneck before implementing.

**Files:** `lib.rs` (incremental board scoring)

---

### SKIP: Transposition table
**Reason:** Low expected hit rate. The (board0, board1, current_row, pending_claims,
actor_index) state tuple is nearly always unique — two paths producing the same
board layout requires the same tiles placed in the same cells in the same order,
which is rare. Implementation cost is high (hashing two 225-cell boards). Skip.

### DONE: Rust-side BatchedMCTS hook (solver called from `BatchedMCTS` tick)
**Status:** Done

`BatchedMCTS::resolve_exact_slots` now checks terminal-adjacent roots before normal batched GPU evaluation. When the root is solvable within budget it builds an exact continuation plan, emits exact child-value policy targets, selects the minimax move, and advances the slot without spending MCTS simulations or network forwards for that move.

Diagnostics exposed through `self_play.py`:

- `exact_solve_count`: exact moves produced;
- `exact_tree_solve_count`: expensive exact continuation plans built;
- `exact_cache_hit_count`: later deterministic continuation moves served from the plan;
- `exact_fallback_count`: exact attempts that exceeded budget and fell back to MCTS.
- `exact_attempt_deck4_initial_count`, `exact_attempt_deck4_retry_count`, and
  `exact_attempt_deck0_count`: attempts by solve entry point;
- `exact_fallback_deck4_initial_count`, `exact_fallback_deck4_retry_count`, and
  `exact_fallback_deck0_count`: fallbacks by solve entry point.

1600-sim, 32-game smoke result with 48x6 CUDA:

| Setting | Games/s | Exact moves | Trees | Cache hits | Fallbacks |
|---------|---------|-------------|-------|------------|-----------|
| exact off | 0.0742 | 0 | 0 | 0 | 0 |
| 500k | 0.0816 | 369 | 32 | 337 | 15 |
| 2M | 0.0769 | 379 | 32 | 347 | 5 |
| 5M | 0.0760 | 382 | 32 | 350 | 2 |

Conclusion: the Rust batched hook is implemented and useful. For routine training, `500_000` is the best measured default so far; higher budgets reduce fallbacks but lost throughput in this smoke.`r`n`r`n---

## Build Order Retrospective

The original implementation-order section served its purpose and is now historical. The actual build sequence was:

| Step | Result | Current status |
|------|--------|----------------|
| OPT-1 solve-once cache | Added `OpenLoopNode.exact_value` and root gating for correctness | Done |
| OPT-2 single-pass budgeted solve | Removed count-then-solve from the hot path | Done |
| OPT-3 alpha-beta | Added exact minimax pruning with value-equivalence tests | Done |
| OPT-4 move ordering | Added initial heuristic ordering | Done; superseded by OPT-4b |
| OPT-4b exact score-delta ordering | Improved p99 and max tail materially | Done |
| OPT-6 root-only Rayon/YBW | Reduced worst-case wall-clock but inflated nodes on some pruning-dependent positions | Done; mixed result |
| Rust BatchedMCTS hook | Wired exact solving into the batched open-loop training path | Done |

Remaining forward-looking items:

- **OPT-5 undo/redo state mutation:** likely allocation/time reduction, but higher correctness risk; build only with strong equivalence tests.
- **OPT-7 incremental score tracking:** profile first; only worth building if terminal `board.score()` is a meaningful share of solver time.
- **Recursive-depth YBW / deeper parallelism:** candidate for advisor p99 latency; root-only YBW was not enough for sub-second p99.
- **Fallback diagnostics / hard-position reanalysis:** log high-budget fallbacks and replay offline to distinguish true complexity from pruning failures.

---

## Measured Performance Summary

### Solver tail after OPT-4b and OPT-6

Deck=4 benchmark sample from development:

| Metric | OPT-4b serial | OPT-6 parallel |
|--------|---------------|----------------|
| p50 time | 176 ms | 130 ms |
| p95 time | 1848 ms | 1292 ms |
| p99 time | 4019 ms | 3011 ms |
| max time | 8597 ms | 3711 ms |
| wall-clock sample | 132.8 s | 94.5 s |

Node-count improvement from OPT-4 to OPT-4b:

| Metric | OPT-4 before | OPT-4b after |
|--------|--------------|--------------|
| p50 nodes | 111,856 | 97,691 |
| p95 nodes | 1,192,836 | 951,627 |
| p99 nodes | 3,661,963 | 1,905,650 |
| max nodes/time | 14,700,596 / 25.2 s | 4,567,099 / 8.6 s |

Interpretation:

- OPT-4b made the serial solver much more predictable and moved the observed tail into a practical solve budget for offline/advisor use.
- OPT-6 improved wall-clock tail latency, especially max time, but can inflate node counts because parallel siblings lose some serial alpha-beta bound tightening.
- For training, many concurrent games already provide parallelism, so serial exact solves with a capped budget are usually the safer default.
- For advisor use, parallel solving is still attractive because user-facing latency is wall-clock dominated.

### 1600-sim batched training smoke

32 games, 48x6 CUDA, `engine=batched_open_loop`, 1600 simulations:

| Setting | Games/s | Step sec | Eval sec | Exact moves | Trees | Cache hits | Fallbacks |
|---------|---------|----------|----------|-------------|-------|------------|-----------|
| exact off | 0.0742 | 33.523 | 385.800 | 0 | 0 | 0 | 0 |
| 500k | 0.0816 | 55.712 | 327.202 | 369 | 32 | 337 | 15 |
| 2M | 0.0769 | 77.405 | 329.244 | 379 | 32 | 347 | 5 |
| 5M | 0.0760 | 89.488 | 322.226 | 382 | 32 | 350 | 2 |

The table above is **historical** — it was measured under the old node-count
budget, where `500_000` was the best throughput point. The budget is now
wall-clock (`exact_endgame_max_secs`, default `3.0`); these node figures are kept
only for context.

### Viability thresholds

- **Training:** exact solving is a good investment when it saves more GPU/MCTS work than it spends in CPU solving. In the 1600-sim smoke, 500k met that bar.
- **Advisor:** the target is different: solve all eligible endgames if possible, even if that requires higher budgets, parallelism, caching, or a longer user-facing wait.

---

## Follow-ups, Testing, and Known Compromises

### 1. Separate true complexity from solver-pathology

Positions that exceed the node budget are not automatically the most strategically
complex positions. A high node count may mean the endgame is genuinely rich, but
it may also mean the pruning strategy failed to find strong bounds early.

Potential diagnostics to log for each exact-solver fallback:

| Metric | Why it matters |
|--------|----------------|
| `phase`, `deck_len`, actor index | Confirms the exact-solve window and game shape |
| legal action count at each remaining ply | Distinguishes real branching from bad pruning |
| counted/visited nodes before cutoff | Measures how hard the solver actually worked |
| shallow best-child margin | Identifies positions that look obvious but fail to prove |
| root policy entropy / child value spread | Helps separate tactical complexity from noise |
| placement counts per player | Finds geometry-driven blowups |
| alpha-beta cutoff count, if exposed | Direct signal of pruning effectiveness |

Useful follow-up: collect the top fallback positions from training, then replay
them offline with larger budgets and richer instrumentation. This should answer
whether the tail is mostly "worth solving" or mostly "solver needs better move
ordering / transpositions."

### 2. Training value of solved complex endgames

Skipping an exact solve does not corrupt every earlier training example in that
self-play game. Earlier examples keep the MCTS targets produced at their own
roots. The main loss is that the skipped endgame root falls back to approximate
MCTS instead of perfect minimax targets.

The caveat: exact values only propagate into earlier search if those earlier
searches reach exact-solver-eligible states during simulation and use the exact
value as a leaf. If the solver only fires once the real game root enters the
endgame window, the benefit is primarily better late-game play and cleaner
late-game examples, not retroactive correction of all earlier moves.

Known compromise for routine training:

- `500_000` nodes is a good default because the 1600-sim smoke showed the best
  throughput/solve-rate balance.
- Larger budgets may improve the hardest late-game labels, but they also spend
  time exactly where the solver tail is worst.
- It is reasonable to skip pathological fallbacks during routine training, then
  optionally reanalyze selected hard positions offline.

Potential enhancement: add a "quality/reanalysis" mode that samples or stores
hard fallback positions, solves them later with a larger budget, and compares
their exact policy/value against the in-training approximate target.

### 3. Advisor assumption changes the optimization target

For the BGA advisor, assume all eligible endgame positions should be solved
exactly. In that setting, the neural net does not need to be perfect inside the
exact-solve window because the solver is the authority.

Training still matters for:

- positions just before the exact-solve boundary;
- steering into favorable solvable endgames;
- move quality when a position is outside the advisor's exact budget/window;
- policy priors and value estimates used before exact solving starts;
- general representation learning.

This suggests separate operating modes:

| Mode | Solver budget policy | Goal |
|------|----------------------|------|
| Routine training | `exact_endgame_max_secs=3.0` | Bound per-position solve cost; sentinel avoids retry storms |
| Training ablation | `exact_endgame_max_secs=0.0` | Measure baseline without exact solves |
| Quality/reanalysis | larger `max_secs`, possibly offline | Produce cleaner labels for selected hard positions |
| Advisor | higher budget, parallel/cache-friendly | Solve all eligible user-facing endgames |

Known compromise: optimizing routine training throughput and optimizing advisor
coverage are related but not identical. Advisor strength should mainly be won by
making the exact solver faster and more complete; training solves should be used
where they improve pre-endgame judgment without dominating self-play wall-clock.

---

## Training vs Advisor Settings

```python
# Training (balance solve quality with throughput)
OpenLoopMCTS(
    ...
    exact_endgame_max_hidden_tiles=4,
    exact_endgame_max_secs=3.0,        # per-position wall-clock budget
    exact_endgame_enabled=True,
)

# Advisor (can afford more time; user waits a moment)
OpenLoopMCTS(
    ...
    exact_endgame_max_hidden_tiles=4,
    exact_endgame_max_secs=10.0,
    exact_endgame_enabled=True,
)
```

The batched training path is configured through `SelfPlayConfig.exact_endgame_max_secs`
/ CLI `--exact_endgame_max_secs` (default `3.0`; `0.0` disables for ablation).

Revisit these defaults after each optimization step is benchmarked.

---

## Endgame Oversampling in Training

Endgame positions (game_progress ≥ 0.75) carry the best labels in the buffer —
exact minimax values and exact-derived policy targets — but at natural frequency
they are only ~20-23% of training batches. `ReplayBuffer.sample_batch` accepts an
`endgame_oversample_weight` (config/CLI `--endgame_oversample`, default `2.0`) that
weights those examples `2×` relative to the rest, concentrating gradient where the
labels are most reliable.

- The per-example weight vector is O(buffer) to build, so it is cached and
  invalidated only when `add()` mutates the buffer.
- `1.0` recovers exact uniform sampling (fast `rng.integers` path, no weights).
- The realised endgame fraction is logged as `n_endgame_in_batch` on diagnostic
  iterations to confirm the oversampling is taking effect (≈33% at weight 2.0 on a
  20%-endgame buffer).

All exact values use `terminal_search_value` — the single source of truth:

```
own_norm  = score_p0 / score_scale
opp_norm  = score_p1 / score_scale
margin    = tanh((own_norm - opp_norm) * margin_gain)
win_value = +1.0 / -1.0 / 0.0
result    = alpha * margin + (1 - alpha) * win_value
```

**Always returns player-0 frame.** Caller negates if needed for player-1 context.

Config values: `score_scale=100.0`, `margin_gain=2.0`  
Training: `alpha=0.8` (margin-dominant, matches training leaf value)  
Eval/advisor: `alpha=0.0` (pure win probability, matches eval sweep winner)

The Rust `terminal_search_value` is verified bit-identical to the Python version.

---

## Policy Target Formula

Exact endgame roots do not use visit-count policy targets. Instead,
`exact_policy_target` derives a soft policy target from the exact minimax value
of each legal child.

Child values are stored in the player-0 value frame. For the acting player, first
identify the best and worst child values:

```
actor = player to move

if actor == player_0:
    v_best  = max(child_values)
    v_worst = min(child_values)
else:
    v_best  = min(child_values)
    v_worst = max(child_values)

range = abs(v_best - v_worst)
```

If all legal moves are effectively equal value (`range < 1e-9`), the policy
target is uniform across legal moves.

Otherwise, the target is an advantage-weighted softmax:

```
temperature = range / 3
advantage_i = abs(v_i - v_worst) / temperature
policy_i = exp(advantage_i - max_advantage) / sum_j exp(advantage_j - max_advantage)
```

Equivalently, the worst move has advantage `0`, the best move has advantage `3`,
and all other legal moves are placed proportionally between them.

Rationale:

- **Scale-invariant:** multiplying every child value gap by the same factor does
  not change the target distribution, because the temperature scales with the
  observed range.
- **Self-calibrating:** decisive positions produce sharper labels; genuinely
  close endgames produce softer labels.
- **No fixed temperature hyperparameter:** there is no global policy-temperature
  constant to retune when value scale, `alpha`, or score normalization changes.
- **Tie-safe:** equal-valued moves are represented as equal choices instead of
  arbitrary one-hot labels.

Design implication: exact policy targets are not "pick the single best move"
labels. They preserve ambiguity when several moves are close in exact value,
which should make late-game policy learning less brittle but may also make
policy-head curves look softer than one-hot expert supervision.

---

## Test Suite

File: `games/kingdomino/test_endgame_exact.py`

| Test | What it checks | Status |
|------|---------------|--------|
| `test_exact_vs_sampled_convergence` | MCTS root value converges toward exact value | ✅ |
| `test_encoder_hidden_order_independence` | encode_state ignores deck order | ✅ |
| `test_exact_solver_public_consistent` | same tiles different deck order → same value | ✅ |
| `test_no_exact_solve_above_budget` | budget=1 returns solved=False | ✅ |
| `test_game_over_state` | terminal state behavior | ✅ |
| `test_rust_deck_empty_vs_python` | Rust deck=0 matches Python | ✅ |
| `test_rust_deck_four_vs_python` | Rust deck=4 matches Python on solvable positions | ✅ |
| `test_mcts_deck_four_exact_solve` | MCTS hook fires and counter increments | ✅ |
| `test_minimax_beats_greedy` | solver does genuine minimax, not greedy | ✅ |

**Tests added as optimizations landed:**

| Test | Covers | Status |
|------|--------|--------|
| `test_solve_once_cache` | OPT-1: later sims served from cache, each leaf solved once | ✅ |
| `test_three_state_cache_no_retry` | Change 1: timed-out leaf marked Unsolvable; ≤1 solve attempt per search; resets per search | ✅ |
| `test_timeout_fires_quickly` | Change 2: 1ms budget on a deck=4 position returns unsolved within <0.1s | ✅ |
| `test_exact_solving_gated_off_at_non_terminal_adjacent_root` | OPT-1 correctness gate: no solve from deck≥8 roots | ✅ |
| `test_alpha_beta_solves_nontrivial_deck4` | OPT-3/4: solves real ≥100k-node deck=4 trees within the wall-clock budget | ✅ |
| `test_alpha_beta_value_exact_and_public_consistent` | OPT-3: budget- and deck-order-invariant exact value | ✅ |
| `test_solve_rate_deck4` | OPT-2+3+4: solve rate high (≥50% floor; ~100% within a few seconds) | ✅ |
| `test_solves_deck4_within_budget` | OPT-4b: median deck=4 solve time fits the wall-clock budget | ✅ |
| `test_oversample_increases_endgame_fraction` | Change 4: weight 2.0 lifts endgame fraction 20%→~33% | ✅ |
| `test_oversample_weight_cache_invalidates_on_add` | Change 4: cached weights rebuilt after `add()` | ✅ |
| `test_exact_stats_in_log_row` | Change 3: aggregate and split exact-solver keys present in every JSONL row | ✅ |
| `test_disabled_solver_returns_unsolved` | `max_secs<=0` disables solving (returns 0.0, unsolved) | ✅ |
| Rust `exact_retry_tests::deck4_timeout_allows_one_after_two_moves_retry_and_deck0` | BatchedMCTS retry policy: no same-root retry, one deck=4 retry after two moves, deck=0 still allowed | ✅ |
| `test_undo_redo_equivalent` | OPT-5: undo/redo solver gives same values as cloning solver | ⏳ pending |

---

## Key Correctness Invariants

1. **`terminal_search_value` is the single source** for all terminal backups —
   in the exact solver, in `_simulate`'s GAME_OVER branch, and in test assertions.
   Never compute scores inline.

2. **Solver uses public information only** — boards, current_row, pending_claims.
   Never reads deck order for the value computation. This is what makes OPT-1's
   cache correctness valid across determinizations.

3. **Exact-value nodes are not expanded** — a node with `exact_value` set stays
   `is_expanded=False`. PUCT descent stops there. Future sims return the cached
   value without descending.

4. **deck ∈ {0, 4} only** — `is_no_chance_endgame_state` correctly rejects all
   other deck sizes. deck=1,2,3 never occur in 2-player Mighty Duel.

5. **Alpha-beta correctness** (when OPT-3 is built) — the value formula is monotone
   in score_diff, so alpha-beta applies without modification. Bounds are in [-1, 1].

---

## Files Modified by This Feature

| File | Role |
|------|------|
| `games/kingdomino/kingdomino_rust/src/lib.rs` | Rust exact solver and integration: `solve_endgame_ab`, `solve_endgame_ab_parallel`, `exact_solve_no_chance` reference path, `exact_count_no_chance_bounded`, `is_no_chance_endgame_state`, `terminal_search_value`, `placement_score_delta`, `pick_order_score`, `order_legal_for_solver`, `exact_policy_target`, `ExactSolveResult`, exact continuation plan support, `BatchedMCTS::resolve_exact_slots`, `RustGameState.from_parts`, and PyO3 exports |
| `games/kingdomino/endgame_solver.py` | Python bridge: `exact_endgame_value`, `_python_state_to_rust`, and Python fallback for deck-empty/reference cases |
| `games/kingdomino/mcts_az.py` | Python `OpenLoopMCTS` exact hook: `exact_endgame_enabled`, `exact_endgame_max_hidden_tiles`, `exact_endgame_max_secs`, `_should_exact_solve`, solve counters (`_exact_solve_count`/`_exact_cache_hits`/`_exact_fallback_count`), root gate, and the three-state `OpenLoopNode.exact_value` cache (`None`/float/`_EXACT_UNSOLVABLE`) with give-up-on-timeout |
| `games/kingdomino/self_play.py` | Batched training wiring: `SelfPlayConfig.exact_endgame_max_secs`, CLI `--exact_endgame_max_secs`, pass-through into Rust `BatchedMCTS`; exact-solver aggregate stats and split attempt/fallback stats in the JSONL log row + `_compact_summary`; endgame oversampling (`ReplayBuffer.sample_batch(endgame_oversample_weight=...)` with cached weights, `SelfPlayConfig.endgame_oversample`, CLI `--endgame_oversample`, `n_endgame_in_batch` diagnostic) |
| `games/kingdomino/test_endgame_exact.py` | Exact solver test suite, currently 21 tests including Rust deck=0/deck=4 agreement, alpha-beta correctness, solve-once cache, BatchedMCTS integration, policy validity, fallback counters, and value sanity checks |
| `games/kingdomino/print_model_contract.py` | Milestone 0 contract checker; adjacent project artifact, not part of exact solving |


