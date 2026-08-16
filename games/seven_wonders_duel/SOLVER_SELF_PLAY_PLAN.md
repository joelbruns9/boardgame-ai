# Putting the exact endgame solver into 7WD self-play

**Written to be picked up cold.** Every number here was measured on this machine
(2026-08-15/16) and is reproducible from the committed gates. Where a number
contradicts an earlier claim elsewhere in the repo, this file is the later
measurement.

The solver itself is **done and gated** (NEXT_STEPS step 5). What remains is
using it during self-play, and the design below is deliberately *not* "replace
the search's targets with the solver's".

---

## 1. The one-paragraph version

At an endgame position the solver knows, exactly, which moves lose. It does not
know which of the surviving moves is better, because 7WD's value is win/draw/loss
and **77–88% of legal moves are proven equally optimal**. So: take the exact
value as the **value target**, and use the solver only as a **mask** on the
**policy target** — zero the provably-losing moves, renormalise the search's own
distribution over the survivors, and let the search keep its ranking among them.

---

## 2. What already exists

| thing | where |
|---|---|
| Rust solver | `seven_wonders_rust/src/solver.rs` |
| Python entry | `RustGame.solve_endgame(max_nodes, max_secs, policy_mode, chance_pruning)` in `seven_wonders_rust/src/lib.rs` |
| State injection | `rust_bridge.rust_game_from_state(game)` |
| Equivalence gate | `endgame_corpus.py` (86 positions) + `test_endgame_solver_rust.py` (15 tests) |
| Undo audit | `engine::journal_undo_audit`, exposed as `RustGame.journal_undo_audit(depth)` |
| Python reference | `advisor_endgame.solve_position` — slow, authoritative |

`solve_endgame` returns `None` (not solvable within budget, or at all) or a dict:

```
regime            "exact" | "exact_expectimax"     # did a chance edge occur?
root_value        float in [-1, 1], ACTOR-relative
best_index        action index, always a proven-optimal one
per_action_value  {action_index: float}            # see exact_per_action
exact_per_action  bool                             # False in value_only mode
nodes             u64
nodes_under_chance u64
```

**Modes.** `policy_mode="exact"` prices every root action on a full window, so
`per_action_value` is exact for all of them — **this is what the mask needs**.
`policy_mode="value_only"` narrows the root window as better actions are found,
so non-best entries are *bounds*, and ties are hidden. Cost of `exact` over
`value_only`: 1.17× nodes on the corpus. Cheap; take `exact`.

`chance_pruning="star1"` is the default and should stay on.

---

## 3. The mask, exactly

At a solved position with exact per-action values:

```
O   = { a : |v(a) - max_b v(b)| <= 1e-9 }        # the proven-optimal set
pi' = { a: pi(a) / sum_{b in O} pi(b)  if a in O
         0                              otherwise }
```

`pi` is whatever policy target that position would have had anyway. Two choices,
both defensible:

* **mask the search's improved policy** — best ranking, but you pay for a search
  *and* a solve at the same position;
* **mask the net's raw prior** — one forward, no search. This is the natural fit,
  because Kingdomino skips MCTS entirely at solved roots to save the forwards
  (`kingdomino/self_play.py`, `exact_endgame_max_secs`), and it is the cheaper
  place to start.

**The 1e-9 tolerance is load-bearing.** Expectimax sums probabilities in floating
point, so a true zero arrives as `-1.4e-17`. An exact comparison would split ties
that are not ties. The solver has the same constant (`TIE_EPSILON` in
`solver.rs`) for the same reason, after a bug where a bound displaced an exact
value and the solver named a move worth -0.3 as best.

**Do not** replace the policy target with a uniform-over-ties label. With 77–88%
of moves optimal, that flattens the distribution in exactly the region where the
search had learned discrimination. Kingdomino's `argmax_ties` label won its
ablation there (231–162–7 vs `soft_clamp`) because KD's value is a score
*margin*, so ties are rare and the label is genuinely sharp. That reasoning does
not transfer to a win/loss value.

---

## 4. The value target is a classification head, not a regression

`dataset.py` builds **W/D/L 3-way** (`_actor_value_class`) and **joint7**
(winner × victory type, `_joint7_class`). So the exact value does not drop in as
a scalar:

* `regime == "exact"` — the value is ±1 or 0, which maps cleanly onto a hard
  W/D/L class. For joint7 you also need the *victory type* of the proven line,
  which `solve_endgame` does **not** currently return. Either add it (the solver
  knows it at the terminal) or leave joint7 on the game's realised outcome.
* `regime == "exact_expectimax"` — the value is a probability in (-1, 1). There
  is no hard class; use a **soft** target over W/D/L (cross-entropy accepts it),
  or exclude these rows from the value-target substitution and keep the realised
  result. Decide deliberately; do not silently round.

`dataset.py` already carries `root_value` for `--value-bootstrap`, which is the
natural slot for a solver value if you want the softer route first.

---

## 5. Where to hook it

Self-play is native Rust: `seven_wonders_rust/src/self_play.rs`.

* `MoveRecord` (line ~194) already carries `legal`, `visits`, `policy_target`,
  `prior`, `root_value`, and — importantly — **`policy_excluded`**, a flag that
  withholds the policy label while keeping the value target. That flag is the
  precedent for shipping the two targets independently, and it is how a
  "solver value, no solver policy" first cut should be built.
* `finish_move` (line ~1611) is where a record is written; the solve belongs
  before it, on the pre-move state.
* `run` / `run_many_pipelined` are the loops that would own a solver budget.

---

## 6. Trigger rule and budgets — set these from HUMAN positions

Bot-played endgames are **not** the target distribution. Measured at 10 cards:
human positions cost ~3× the nodes of bot positions and offer far more legal
moves at equal card count (15 vs 3–4), because a human still holds wonders and
options a rush bot has spent.

After the history-ordering change, on real captured endgames:

| cards left (Age III) | time to solve exactly |
|---|---|
| ≤ 6 | milliseconds |
| 8 | 0.05–0.31 s |
| 10 | 3.4–4.1 s |
| 12 | ~60 s, or unsolved |

So a 3 s budget reaches roughly 8–10 cards on real positions. **Age I and II are
never solvable** at any budget: the next age's deal is a sample-only chance edge
and the solver refuses (`SolveStop::Unsolvable`).

Suggested trigger: Age III and ≤ N cards present, N chosen from the table above
against the budget, with the solve attempted and its failure treated as normal.

---

## 7. Knobs worth copying from Kingdomino (`kingdomino/self_play.py`)

* `exact_endgame_max_secs` (default 3.0) **plus its iteration schedule**
  (`exact_endgame_max_secs_schedule`);
* `exact_fallback_positions` — a JSONL sidecar of roots the solver declined, kept
  out of the replay buffer;
* `endgame_oversample` (2.0) — endgame rows carry exact labels, so concentrate
  gradient there;
* `async_solve` with `game_cpus` / `solver_cpus` — but see §8: beyond ~8 solver
  threads buys little on this machine.

---

## 8. Measured facts — do not re-derive these

* Solver vs the Python reference, same 86 positions: **~18,000×**
  (`value_only`+`star1`). Python reference is 1,361 nodes/s; Rust is ~1.7M.
* Journaled undo: 885 → 581 ns/node (**1.52×**). Undo is a delta replay, not a
  state copy (`GameState::apply_journaled` / `undo`).
* History move ordering: **1.95× fewer nodes, 1.73× faster** over 15 positions
  at 8–10 cards.
* Star1 chance pruning: 0.77× nodes in `value_only`, 0.96× in `exact` — it needs
  a *narrow* root window to bite, which only `value_only` produces.
* Parallelism: solves parallelise across positions (the binding releases the
  GIL). 2.89× at 4 threads, 3.77× at 8, 4.37× at 16 on 16 logical CPUs. The
  ceiling is the machine (all-core clock/SMT), **not** memory bandwidth — that
  hypothesis was tested and disproved.
* Tie structure: 77–88% of legal moves are proven optimal; values are
  win/draw/loss only in 100% (chance-free) and 72% (with chance) of positions.

## 9. Already tried and rejected — do not retry without new information

* **star2 probing**: 1.17–1.86× *worse*, degrading with depth. It assumes a
  cheap probe, which needs a depth-limited search plus a heuristic evaluation;
  this solver runs to terminal and has no evaluator, so every probe is a full
  search of one move.
* **Lexicographic margin / turns-to-win refinement** (rank wins by score margin,
  or supremacy wins by speed): sound in principle, **7.9× nodes on the corpus
  and 11–19× at 8–9 cards**, because a lexicographic objective cannot cut on
  equality and ties are the common case here. The affordable version is to
  propagate the refinement without pruning on it, accepting that it then
  describes the first line found rather than the best one.
* **Transposition table**: 1.13× on 7WD endgames. Kingdomino's TT win does not
  transfer.
* **Allocation work** (per-depth clone buffers, dropping chance-chain keys):
  flat, both times. The cost was the copy, which the journal removed.

## 10. Gates that must stay green

```
python -m pytest games/seven_wonders_duel/test_endgame_solver_rust.py        # 15 tests
python -m pytest games/seven_wonders_duel/test_endgame_solver_self_play.py   #  9 tests
cd games/seven_wonders_duel/seven_wonders_rust && cargo test --release        # 26 tests
python -m games.seven_wonders_duel.endgame_corpus                            # corpus self-check
```

Any change to the solver must leave all 86 corpus positions matching the Python
reference exactly — regime, every action's value, and the proven-optimal set.
Pruning and ordering may change node counts and nothing else.

## 11. Decisions (settled 2026-08-16) and what shipped

1. **Mask the search policy**, not the net prior. The mask's value is entirely
   in what it *removes*; with 77–88% of moves proven equal it says almost
   nothing about which survivor is better, and the search is the only thing in
   the system that ranks them. Masking a prior hands the net back a mask over
   its own opinion. Kingdomino can skip its search because *its* value label is
   a score margin, so ties are rare and the solver alone determines the label —
   that reasoning does not transfer. Cost is contained instead by solving only
   on **full-search moves**: cheap ones emit no example at all
   (`dataset.is_fast_search_move`), so a solve there buys nothing.
2. **Soft W/D/L for `exact_expectimax`, one-hot for `exact`**, both through a
   new `value_solver` channel that is separate from `value_soft`. The two mean
   different things: `value_soft` is the search's *opinion*, blended with the
   outcome in whatever proportion `--value-bootstrap` says; a proven value is
   the *answer*, so it **replaces** the outcome at full weight and ignores
   `value_bootstrap` entirely. Nothing is rounded: an expectimax 0.0 is balanced
   win/loss mass and stays `(0.5, 0, 0.5)`, while a chance-free 0.0 is a proven
   draw and becomes one-hot on it.
3. **No victory type, for now.** joint7 is an auxiliary head at `aux_weight`
   0.2; the proven line's victory type is only well-defined in the `exact`
   regime (under expectimax different outcomes end differently); and returning
   it means threading a terminal descriptor up through exactly the code the
   86-position gate covers. Bad ratio. joint7 stays on the realised outcome, and
   this is cheap to revisit for `exact` rows alone.
4. **Both, behind independent switches** — `--endgame-solver-max-nodes` turns
   the value target on, `--no-endgame-solver-mask-policy` leaves the policy
   alone. The plan's reason for sequencing them (opposite risk profiles) is a
   reason to be able to *attribute* a result to one of them, which two switches
   give without two deployments.

### Shipped

| piece | where |
|---|---|
| trigger, solve, mask | `self_play::endgame_overlay` (called from `run` and `finish_move`) |
| config (process-wide, like `set_cheap_top_k`) | `swr.set_endgame_solver(max_nodes, max_secs, max_cards, mask_policy)` |
| CLI | `phase_d.py --endgame-solver-*` |
| record fields | `MoveRecord.solver_value / solver_regime / solver_nodes / solver_masked` |
| value target | `dataset.collate` → `value_solver`, consumed in `train.compute_losses` |
| gate | `test_endgame_solver_self_play.py` (9 tests) + 3 Rust unit tests |

**Off by default** (`max_nodes = 0`), and an off run is the generator that
existed before this change — asserted, not assumed.

**Bound the solve by nodes, not seconds.** A deadline makes generation
irreproducible from `(seed, net)`: the same position solves on an idle machine
and times out on a loaded one, so the mask appears or does not and a different
move is played. `--endgame-solver-max-secs` is a safety net against one
pathological position holding a scheduler slot — the solve is synchronous and
does hold one — and should be set high enough never to bind.

`TARGET_VERSION` was deliberately **not** bumped. The masked rows carry a new
definition of `policy_target`, but `solver_masked` marks exactly those rows,
which is strictly more informative than a global bump — and a bump would
invalidate every buffer ever written to say less.

Still open: `exact_fallback_positions` (a sidecar of declined roots) and
`endgame_oversample` from §7 are not implemented; the per-move `solver_nodes`
and the absence of `solver_value` already identify both populations in the
buffer, so both are additive.
