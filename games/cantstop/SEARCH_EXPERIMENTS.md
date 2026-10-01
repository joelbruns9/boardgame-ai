# Offline whole-turn search experiments

This framework leaves the advisor and training policy unchanged. Ordinary play
already solves the current turn exactly; “no search” below means no **extra-turn**
search. Both competitors use the same checkpoint, evaluator and current-turn
solver. Only one seat gets extra search; it rotates equally through every seat.

## Algorithm

`turn_search.WholeTurnSearch(state, evaluator, TurnSearchConfig(...))` has the
same value/choose_move/should_stop query interface as the existing Rust wrapper.
The root is solved after the actual opening roll. Every expanded continuation
starts BEFORE the next player's opening roll: Rust averages the actual dice
probabilities and each player optimizes their own absolute-seat winning chance.
Terminal wins remain exact. No sampled outcomes replace these calculations.

Each expansion replaces a single NN continuation estimate and recalculates all
ancestors immediately. Branch priorities are then recomputed under the updated
policies. The first expansion examines bust. Normal expansions prioritize the
product of decision visitation weights along the path. Every fourth expansion
(default) instead checks the most optimistic unexpanded estimate for that node's
acting player, including zero-policy-reach alternatives. These are experimental
allocation heuristics, not probabilities of selected terminal outcomes or a
measured uncertainty score. The old offset correction is not used.

`--expansions` caps extra turn solves across the entire tree (root excluded).
`--depth 1` allows the next player's turn; `--depth 2` can also examine the player
after that. In a two-player game that is your next turn. A deeper limit permits,
but does not force, deeper expansions. `--explore-every 0` disables the alternative
priority. `--seconds` is an optional SOFT total search limit, checked between
expansions; root and in-flight solves cannot be interrupted. Fixed expansion
budgets are reproducible; time budgets are machine/load dependent.

The implementation is an offline reference controller around compiled Rust
turn solves, with a serial arena driver. It does not yet batch independent games
on the GPU or merge identical boards reached along different tree paths. Memory
and computation grow with the expansion budget; start small. Node parent links
are weak so each finished search can release its tables promptly.

## Run

Use the project virtual environment from the boardgame-ai-cantstop checkout.
The Rust binding needs `rebackup`, `best_candidate`, `leaf_reach_at`, and
`leaf_snapshot` (already installed
in the current environment). On another installation, build/install the Rust
extension using the project's usual maturin workflow first.

Two-player, five-column, nonblocking pilot:

```powershell
python -m games.cantstop.search_arena --checkpoint runs/p4_pilot/iter_0080.pt --players 2 --extended --games 200 --expansions 4 --depth 1 --out runs/search_k4_d1_200.json
```

Three-player, four-column, blocking pilot:

```powershell
python -m games.cantstop.search_arena --checkpoint runs/p4_pilot/iter_0080.pt --players 3 --extended --blocking --games 300 --expansions 4 --depth 1 --out runs/search_3p4b_k4_300.json
```

Use a fresh output path for each run. Games must be a positive multiple of the
player count. `--expansions 0` runs an identical-policy sanity match. CUDA is the
default; use `--device cpu` if required. No personas or heuristic opponents are
involved. Each game has its own recorded dice seed.

## Results and verification

The report records checkpoint SHA256, code identity/source hashes, rule/search
settings, individual seeds, winning seats, turn-level search costs, changed root
choices, average/p95 solve latency, evaluator rows and peak retained positions.
Inference is warmed up before competitor timings. Position counts are not byte
measurements, and search/control visit different boards, so their latency ratio
is descriptive rather than a matched-position benchmark.

Reports are saved after each game. An interrupted or failed run is marked
incomplete and receives no strength verdict; unfinished games are not silently
dropped. The final summary gives seat counts, challenger win rate and an
approximate Wilson 95% interval against the even rate 1/players. Small pilot
matches and multiple budget experiments are exploratory. Select a setting, then
run a separately seeded fixed-size confirmation before claiming improvement.

`python -m pytest games/cantstop/tests/test_turn_search.py -q` checks zero-budget
agreement, independent Python one-expansion agreement for 2/3/4 players, depth,
upward propagation, exact terminal wins, soft budgets/cancellation, deterministic
selection, invalid input, an actual changed stop decision and arena bookkeeping.

## Controller optimization (2026-09-29)

Candidate ranking now runs as vectorized NumPy reductions within each table,
retaining only one candidate per expandable table. The original depth-first,
first-leaf tie order is preserved, including zero/underflowed influence weights.
Reach weights are cached until that node is backed up with changed continuation
values. Tables at the maximum depth are skipped entirely during allocation.
Updates still back up exactly the affected ancestor chain. No native library
update is required for this optimization.

The frozen original controller is in tests/turn_search_reference.py. Run:

```powershell
python -m pytest games/cantstop/tests/test_turn_search.py games/cantstop/tests/test_turn_search_optimized.py -q
python -m games.cantstop.benchmark_turn_search --repeats 3 --out runs/controller_benchmark_new.json
```

The benchmark uses the CPU heuristic and the saved BGA fixture, alternates
execution order, and checks identical expansion traces, values and moves before
reporting speed. The first three-repeat measurement gave median full-search times:

| Budget | Original | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| 4 expansions, depth 1 | 0.318 s | 0.221 s | 1.44x |
| 8 expansions, depth 1 | 0.641 s | 0.441 s | 1.45x |
| 8 expansions, depth 2 | 1.266 s | 0.638 s | 1.98x |

These are one-board CPU-heuristic measurements, not GPU arena timing guarantees.
Fixed-budget behavior is preserved. With a wall-clock budget, faster execution
can complete more expansions and therefore produce different advice. Already
running arena processes retain their imported controller; new invocations use
the optimized implementation. The native update below completes candidate selection and selected-board export.
Backup conversion and GPU batching across games remain future opportunities.


## Native throughput follow-up

Both remaining throughput items are complete. Rust ranks candidates using cached
reach weights and cached acting-player continuation values, returning one candidate
per table. Only scalar reach weights for expanded children cross the boundary.
Successful value updates invalidate reach caches; failed updates preserve them.
`leaf_snapshot(index)` constructs only the selected continuation board. The NN
feature path never exports a full list of boards. Legacy callable evaluators
still require full boards for evaluation and reuse that existing list.

The extension is rebuilt and installed; other installations require a new Rust
build. Fixed-budget expansion traces, values, choices and evaluation counts remain
identical, including first-leaf ties and floating-point underflow behavior.

The CPU benchmark compares all controller versions on the SAME updated Rust
library. Even the frozen original controller benefits from native reach caching,
so these timings should not be compared directly with older binary measurements.

| Budget | Previous NumPy | Native | Incremental speedup |
| --- | ---: | ---: | ---: |
| 4 expansions, depth 1 | 0.197 s | 0.189 s | 1.04x |
| 8 expansions, depth 1 | 0.383 s | 0.381 s | 1.01x |
| 8 expansions, depth 2 | 0.542 s | 0.428 s | 1.27x |

Three-repeat medians on one saved board with CPU heuristic evaluation; arena
improvements may differ. Results: runs/controller_native_20260929.json.
139 targeted tests passed. A two-game CUDA smoke run exactly reproduced the
previous seeds, winners, turn counts, per-turn values, choices and evaluator row
counts. This is correctness evidence, not evidence of playing strength.
Native API tests: games/cantstop/tests/test_turn_search_native.py.
