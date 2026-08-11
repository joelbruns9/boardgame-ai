# Deck=8 forced-move disagreement duel

Status: harness implemented; the experiment has **not** been run.

## Question

Test 1 established that sampled explicit chance splitting causes almost all of
the root-policy movement previously attributed to the exhaustive 70-row panel.
It did not establish whether the changed actions are better. This duel asks the
quality question directly:

> At deck=8 first-selection roots where control and sampled split choose
> different actions, which forced action produces the better continuation?

This is a frozen-network evaluation. It does not train or modify the model.

## Frozen disagreements

The input is the completed 6,400-simulation Test 1 comparison:

- 64 frozen deck=8 first-selection roots;
- 33 roots where the top joint action differs;
- 17 of those 33 also change the selected tile;
- exact control and sampled-split action indices are read from the immutable
  artifacts, not searched again by this runner.

The runner verifies the position, checkpoint, base-artifact, and sampled-split
artifact hashes. It also fails unless the expected 33/17 counts are present.
The 17 pick-changing positions are a preregistered mechanistic subgroup; all 33
positions remain the primary sample.

## Continuation construction

For each disagreement position and each of 16 continuation seeds:

1. deterministically shuffle the eight-tile hidden bag;
2. create the original and exactly player-relabeled versions of the state;
3. force control's top action in one arm and sampled split's top action in the
   other arm;
4. play both resulting states to completion with the same current checkpoint,
   common search seeds, and the incumbent aliased open-loop search;
5. use 1,600 simulations per remaining searched move, no chance enumeration,
   no Dirichlet noise, and greedy move selection;
6. route the deck=4 tail through the existing exact solver when feasible.

The action is fixed before the future row is exposed. Both arms receive the
same hidden deck ordering. Search seeds are identical across forced arms and
label mirrors; arms and mirrors run as separate cohorts so duplicate paired
seeds never appear inside one Rust batch.

The Rust addition is a narrow `BatchedMCTS.from_states` constructor. It starts
the unchanged production engine from caller-supplied nonterminal states and
never recycles a finished slot into a standard opening. The constructor knows
nothing about control, sampled split, or forced actions.

## Metrics and uncertainty

The primary outcome is terminal score-margin difference in the original
chooser's frame:

`split-forced chooser margin - control-forced chooser margin`.

Official win points are secondary. Each continuation seed is averaged over its
two player-label mirrors. Continuation seeds are then averaged within each
position. Confidence intervals bootstrap the 33 position means, giving every
disagreement position equal weight and avoiding pseudoreplication from repeated
deck seeds.

The completed report will include:

- mean chooser-margin and chooser-points deltas with position-bootstrap 95%
  intervals;
- counts of positions favoring split, tied, and favoring control;
- the same metrics on the 17 pick-changing positions;
- mirror disagreement as a label-symmetry diagnostic;
- inference, exact-solver, and fallback accounting.

## Interpretation gate

- **Positive:** the all-position margin interval is above zero, its point
  estimate is positive, and the points estimate is nonnegative. This is strong
  evidence because the downstream arbiter is aligned with incumbent control;
  chance-aware training earns a separate design.
- **Negative:** the margin interval is below zero, its point estimate is
  negative, and the points estimate is nonpositive.
- **Otherwise:** inconclusive. Operationally, do not fund chance-aware training
  from a null result alone. A null does not prove the mechanism is valueless,
  because the incumbent continuation policy may not realize advantages that
  require chance-aware downstream decisions.

This estimand is conditional on positions where one frozen search seed produced
a disagreement. It is not a whole-game Elo estimate and does not measure how
often disagreements occur.

## Planned command — do not run before review

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.forced_move_duel `
  --output runs/kingdomino/chance_correct_a1/deck8_forced_move_duel_v1.json `
  --budget 6400 --continuation-seeds 16 --sims 1600 `
  --batch-slots 32 --leaf-batch 6 --solver-cpus 4 `
  --device cuda --amp-inference
```

The output is atomic and resumable after each batch. Progress logs contain only
completion counts, not partial outcomes. A smoke, if later authorized, must use
`--limit-positions`, `--allow-nonstandard-design`, and a separate output path so
it cannot collide with the frozen full run.
