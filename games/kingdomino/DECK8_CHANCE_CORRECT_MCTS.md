# Deck=8 chance-correct MCTS

Status: implemented behind the opt-in `--deck8_chance_enumeration` flag for
`--engine batched_open_loop`.

## Goal

At a Kingdomino decision with eight unseen dominoes, the next public row has
exactly `C(8,4) = 70` possible outcomes. The production open-loop search used to
sample one hidden deck order per simulation. This change lets an encountered
stochastic action evaluate all 70 next-row information states, so that action's
Q value starts from the exact uniform expectation over the complete reveal
support instead of a small-sample estimate.

This is the production self-play vertical slice discussed after the A1c oracle
work. It changes search, not the replay schema or learner target format.

## Search semantics

The implementation is active only when all of these are true:

1. `BatchedMCTS` is running open-loop.
2. `deck8_chance_enumeration` is enabled.
3. The real search root has exactly eight hidden dominoes.
4. PUCT has selected an action that triggers the next reveal.
5. At least 71 move-budget units remain: one real discovering path plus the
   complete 70-row panel.

The order is important. PUCT commits to the action before a reveal row is
selected, so counterfactual row values cannot leak into the preceding player
choice. The 70 support rows are sorted, unique, and each has probability `1/70`.
Each row is materialized as its own public observation subtree.

The network evaluates all 70 observation states through the existing
`BatchedMCTS.step()` batch. Panels from different action nodes and game slots are
therefore combined into the same GPU forward. `update()` validates that every
admitted group contains exactly 70 rows and unit probability mass before it
publishes any panel Q.

At most one chance node is admitted in each real move's search tree. The first
eligible node selected by PUCT receives the panel and all sibling actions keep
the ordinary unbiased sampled backup. The guard survives across successive
`step()`/`update()` waves and resets only after the real move is played (or the
tree is otherwise replaced). This bounds the exhaustive cost at 70 rows per
move instead of allowing a large budget to enumerate several sibling actions.

For an admitted chance node:

- `Q = sum_r P(r) * V(r)` over all 70 rows.
- Bootstrap values and policy priors initialize observation subtrees.
- Bootstrap work creates no MCTS visits.
- Real simulations refine the conditional subtree for the row they traverse.
- If an observation row already has real visits when its panel commits, its
  searched conditional mean takes precedence over the network bootstrap.
- When a conditional mean changes, the chance Q replaces only that row's
  probability-weighted contribution.
- Root visit counts, move selection, and replay policy targets still contain
  real search visits only.

## Budget decision

A 70-row panel is charged as 70 units of the move's existing search budget; the
real simulation path that discovered it is charged normally. Admission is
atomic. If the full panel does not fit, the node remains on unbiased sampled
backup over the same closed support. A partial panel is never treated as an
exact expectation.

This choice keeps configured search cost bounded and makes the compute tradeoff
explicit. A 100-unit move can spend 70 units on one exhaustive panel and has at
most 30 units left for real paths (including the path that discovered it). At
the recommended initial A/B setting of 400 units, the same fixed panel consumes
17.5% and leaves up to 330 real paths. PUCT determines which first eligible
action receives the expensive exact reveal mean; no second panel is admitted in
that move even when it would fit.

The flag defaults off. This allows matched self-play or strength comparisons
before changing a long training run.

## Configuration and diagnostics

Python:

```text
--engine batched_open_loop --deck8_chance_enumeration
```

Programmatic configuration uses
`SelfPlayConfig(deck8_chance_enumeration=True)`. Enabling it with a non-open-loop
batched engine fails immediately.

The batched statistics and iteration history expose:

- `deck8_chance_panel_count`: complete panels committed.
- `deck8_chance_bootstrap_rows`: network rows spent on those panels; always
  `70 * panel_count`.
- `deck8_chance_budget_blocked_count`: admission attempts that stayed sampled
  because fewer than 70 budget units remained. The same sampled node can appear
  again on a later wave, so this is an attempt count, not a unique-node count.

`max_batch_cap` includes the possible panel rows when the mode is enabled.

HOF and gating games always force this mode off. Those games deliberately give
the learner and frozen opponent different simulation budgets, so enabling a
fixed 70-row charge could change the two seats by different proportions and
confound the comparison. Closed-loop HOF configuration with the flag enabled is
rejected like ordinary self-play; open-loop HOF accepts the training config but
constructs its evaluation search with the chance mode disabled.

## Interpretation and determinization

The panel is exact only over the immediate reveal distribution under the
current leaf evaluator. It is not a fully expanded game tree. At commitment,
unvisited reveal rows use a one-ply network estimate while a row already reached
by real search uses its searched conditional mean; subsequent real simulations
can deepen those row subtrees. This creates a deliberate depth asymmetry versus
sibling actions, and the first PUCT-selected eligible action also receives more
network evaluation than its siblings. We are not blending the panel mean with
sampled estimates before measuring the clean mechanism. Whether that selective
refinement improves playing strength is an A/B question, not a correctness
claim.

Open-loop descent begins from the slot's cloned public state and samples a fresh
hidden-deck permutation for each real path. For a panel, the selected action is
first fixed by PUCT and then each of the 70 unordered four-tile reveals is
materialized as a public observation state. This is valid because pre-reveal
hidden order is not public information and the state encoder/search logic are
permutation-invariant over that hidden remainder; no counterfactual row is used
to choose the preceding action.

## Verification completed

- Rust compile-only test target: passed.
- Focused Rust regression
  `deck8_exhaustive_panel_is_closed_and_bootstraps_without_fake_visits`: passed.
  It checks 70 unique equal-probability rows, delayed admission, no fake visits,
  exact panel mean, precedence for an existing searched observation, and
  probability-aware replacement after a real backup.
- Python syntax compilation: passed.
- Python production-boundary regression: four tests passed. A two-slot mock
  evaluator drives complete games through the real `step()`/`update()` API and
  checks panel accounting plus the unchanged 12-field replay tuples. A
  nonconstant value regression checks the committed probability mean through
  the production API and observes at most one panel throughout a 400-unit move.
  Configuration tests cover invalid closed-loop use and method-neutral HOF
  construction.
- Existing self-play/replay/learner script suite: all 12 sections passed.
- CPU end-to-end self-play, one game, 100 simulations: 52 examples, two panels,
  140 bootstrap rows, maximum batch 72.
- CUDA end-to-end self-play, two games, 100 simulations: 104 examples, six
  panels, 420 bootstrap rows, maximum batch 144.
- CUDA learner smoke from generated replay: D4 augmentation, forward, backward,
  and optimizer step all completed with finite policy/score/win losses.

The pre-existing full Rust unit suite contains long-running tests and was not
used as the interactive gate; the test target compiled successfully and the
new focused regression was run directly.

## Reviewer map

- `kingdomino_rust/src/lib.rs`
  - `OLChanceConfig` / `ol_descend`: request a production full-panel admission
    only after a stochastic action is chosen.
  - `BatchedMCTS::step`: enforce budget, materialize 70 observation rows, and
    merge them into the cross-slot inference batch.
  - `BatchedMCTS::update`: validate/commit panels, preserve visit-free
    bootstraps, and use probability-aware `ol_backup_path`.
  - `BatchedMCTS` getters: cumulative panel diagnostics.
- `self_play.py`
  - `SelfPlayConfig` and CLI flag.
  - Both production `BatchedMCTS` construction paths.
  - Single-buffer, double-buffer, merged stats, console output, and history.

## Deliberate limitations / next experiment

- Deck sizes above eight retain the existing open-loop sampled search.
- Deck=4 remains handled by the existing exact deterministic endgame route.
- This change makes the admitted chance expectation exact with respect to the
  current network leaf evaluator; it does not enumerate the rest of the game.
- Strength and throughput should next be measured in a matched self-play A/B at
  400 budget units per move, comparing the flag off and on with identical game
  seeds, model, search settings, and total configured budget. Report decisive
  win rate / score margin with uncertainty, panels per game, bootstrap rows,
  positions per second, wall time, peak GPU memory, and batch-size distribution.
  The implementation tests above establish correctness and integration, not an
  Elo claim. Only after that clean A/B should alternatives such as blending or
  broader sibling admission be considered.
