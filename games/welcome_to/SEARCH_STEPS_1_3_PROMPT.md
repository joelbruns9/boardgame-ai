# Task: Welcome To search — steps 1–3 of `SEARCH_SPEC.md` §12

Self-contained brief. You do not need the design conversation that produced it.

## Context you need

`games/welcome_to/` is an AlphaZero-style build for the board game *Welcome To…*.
The engine (`game.py`) is transcribed from BoardGameArena's PHP and is the sole
authority on the rules. `macro_codec.py` defines the frozen 684-action search
vocabulary. `mcts.py` is the search. `SEARCH_SPEC.md` is the spec of record —
**read §2 (phase graph), §5.1 (dominance), and §12 (implementation order) before
starting.**

These three steps are **independent of the unresolved chance-boundary design**
(§6–§9 of the spec). Do not touch `_advance`, the observation key, or anything
about turn boundaries. Those are blocked on a separate review.

Current state: **347 tests pass, 1 skipped.**
Run with `python -m pytest games/welcome_to/tests -q` from the repo root.

---

## Step 1 — Search-only dominance pruning

Some legal actions are provably dominated and the search should not spend budget
on them. Prune these **from the search's action set only**:

| phase | prune | when |
|---|---|---|
| `ACTION_PARK` | `PASS_PARK` | a build is legal |
| `ACTION_POOL` | `PASS_POOL` | a build is legal |
| `ACTION_ESTATE` | `PASS_ESTATE` | any row is legal |
| `ROUNDABOUT_PLACE` | `PASS_ROUNDABOUT` | any placement is legal |

**Why these are dominated** (the argument is in §5.1; it is provable, not a
heuristic). Park, pool and estate only advance a scoring track —
`parks[x] += 1`, `pools[street] += 1`, `estate_marks[row] += 1`. They consume no
box, fence, number, turn or resource. `PARK_SCORES` and `POOL_SCORES` in
`constants.py` are strictly increasing. Every plan predicate that reads parks or
pools (`DECORATIVE`, `COMPLETE_STREET`) is monotone. Plans are **not**
auto-validated (`CHOOSE_PLAN` has `PASS_PLAN`), so taking one cannot force an
unwanted three-plan game end. For the roundabout: opening and then passing
reaches the same `CHOOSE_CARDS` state as never opening, minus the option — so not
opening weakly dominates it, and pruning makes `ROUNDABOUT_OPEN` mean "I will
place one", which is the real decision.

**`ACTION_BIS` and `ACTION_SURVEYOR` keep their passes.** Bis calls
`sheet.write(..., is_bis=True)` — it fills a box and takes a scoring penalty. The
surveyor fence partitions a street into estates and can destroy an `EstatePlan`'s
required sizes. Both are genuine decisions. Do not prune them.

### ⚠ The trap that will break the build if you get it wrong

**Do not put the pruning inside `macro_codec.legal_macros` or `legal_mask`.**
`datagen.replay` uses `legal_mask` to build the training legal mask, and
`test_macro_codec.py::test_replay_emits_one_sample_per_macro_and_labels_it_legally`
asserts every recorded label is legal under its own mask.

**Measured, over 75 GreedyBot games at 2/3/4 seats, the reference policy takes
the dominated actions anyway:**

| action | offered with an alternative | taken by GreedyBot |
|---|---|---|
| `PASS_PARK` | 1200 | 0 |
| `PASS_POOL` | 177 | 0 |
| `PASS_ESTATE` | 1435 | **78** |
| `PASS_ROUNDABOUT` | 2028 | **1775** |

So pruning in `legal_macros` would make 1,853 recorded labels illegal and break
replay. Add a **separate** function — e.g. `macro_codec.search_legal_macros(state)`
/ `search_legal_mask(state)` — and call it only from `mcts.py`. The engine's
`legal_actions()`, the 684 codec, and everything `datagen` touches stay
untouched.

### ⚠ The behavioural risk to measure

GreedyBot opens a roundabout and then passes **87% of the time** (1775 / 2028).
An S0 network cloned from it will therefore carry a strong `ROUNDABOUT_OPEN`
prior — and with `PASS_ROUNDABOUT` pruned, that prior now means *"build a
roundabout"*, each of which costs −3 or −8 points.

**Log roundabouts built per game before and after this change** (GreedyBot builds
1.28 per game in advanced play). If it spikes, this is the cause, and the fix is
not to un-prune — it is that the prior is stale and the search needs enough
simulations to override it. Record the number either way.

### Tests to add

- each pruned pass is absent from the search action set when its alternative is
  legal, and **present** when it is not (e.g. `PASS_PARK` must survive when no
  park street is available — though `_settle` usually skips that phase entirely);
- `ACTION_BIS` and `ACTION_SURVEYOR` passes always survive;
- `legal_macros` / `legal_mask` are **unchanged** — assert directly that the
  unpruned set still contains the passes;
- a full `datagen.replay` over several games still labels every sample legally
  (the existing test covers this; make sure it still passes).

---

## Step 2 — Collapse forced nodes inside simulations

`MCTS.play` already short-circuits when there is exactly one legal action at the
**external root**. `MCTS._simulate` does not: it creates a node, calls the
network, and runs PUCT over a single-element action array.

Make the descent skip forced nodes — apply the forced action and continue without
evaluating or storing a node. After step 1's pruning, park and pool nodes become
forced and disappear entirely, so the two steps compound.

**Measured baseline: 1.91 network decisions per turn** (distribution: 2 in 169
turns, 1 in 62, 7–8 in 8, over 239 sampled turns). Re-measure after steps 1–2 and
record the new figure in `SEARCH_SPEC.md` §4.

### Constraints

- A forced action must still be **applied** — skipping the node is not skipping
  the move.
- Do not skip across a turn boundary. If skipping forced nodes would run past the
  root player's turn, stop and let `_advance` handle it as it does now.
- The backup path must not include skipped nodes (they have no statistics).

---

## Step 3 — Deterministic within-turn re-rooting

`MCTS.play` currently discards the whole tree after choosing an action. Within a
turn this throws away exact work: **no chance occurs inside a turn**, so the
subtree under the chosen action is still valid for the next decision.

Re-root onto the selected child when the next decision is still the same player's
in the same turn, and search from there with the existing statistics.

### ⚠ The invariant that makes this legal — and its limit

Re-rooting is exact **only within a turn**, because the transition is
deterministic. **It is not valid across a turn boundary**, where a chance reveal
intervenes and the child's statistics were gathered under determinizations that
no longer apply. Do not generalise it; add a comment saying so, because it is the
kind of optimisation someone extends later without noticing.

### Design latitude

`search()` builds a fresh root node each call. You will need either an optional
node argument, or a small stateful wrapper that holds the retained subtree
between `play()` calls. Either is fine. Requirements:

- re-rooting must be **opt-in or automatic but provably safe** — verify the
  retained node corresponds to the state being searched;
- Dirichlet root noise (`SearchConfig.dirichlet_alpha`, currently `None`) must not
  be applied twice to the same node;
- an explicit reset when the turn changes.

### Tests to add

- searching, playing, then re-rooting gives the same action as searching the
  resulting state fresh with the combined budget — or, if exact equality is too
  strong, assert the retained subtree's visit counts are preserved and the value
  estimates match;
- the tree is **discarded** across a turn boundary;
- total simulations actually saved: log it.

---

## What must not change

- `game.py` rules and `GameState.legal_actions()` — BGA fidelity is the engine's
  job, and both the differential harness and the advisor depend on it;
- the 684-action `macro_codec` vocabulary and index layout;
- `datagen.replay` labels and legal masks;
- `_advance`, the observation key, or any turn-boundary behaviour (blocked on a
  separate review);
- the root-player contract in `mcts.py` — all four clauses, with tests in
  `tests/test_mcts.py`. Leaves are evaluated as `encode_state(state, root)` and
  never `state.actor`; opponents are sampled forward as transitions; the backed-up
  scalar is never negated.

## Definition of done

1. `python -m pytest games/welcome_to/tests -q` — 347 existing tests still pass,
   plus the new ones.
2. Network decisions per turn re-measured and written into `SEARCH_SPEC.md` §4,
   replacing the 1.91 baseline.
3. Roundabouts per game measured before and after step 1, and recorded.
4. Simulations preserved by re-rooting measured and recorded.
5. `SEARCH_SPEC.md` §12 steps 1–3 marked done, with the measured numbers.

## House style

Match the surrounding code: docstrings explain *why* a thing is the way it is,
especially where a plausible alternative is wrong. Where you make a measurement,
put the number in the docstring or the spec rather than a commit message alone.
If a measurement contradicts something this brief asserts, **trust the
measurement and say so** — several claims in the spec were corrected exactly that
way.
