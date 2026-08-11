# Deck=8 forced-move disagreement duel

Status: completed on 2026-08-10. The frozen primary screen is
**inconclusive**; the preregistered pick-changing subgroup is directionally
positive. All post-run integrity checks passed.

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

The completed report includes:

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

## Executed command

```powershell
..\boardgame-ai\.venv\Scripts\python.exe -m games.kingdomino.forced_move_duel `
  --output runs/kingdomino/chance_correct_a1/deck8_forced_move_duel_v1.json `
  --budget 6400 --continuation-seeds 16 --sims 1600 `
  --batch-slots 32 --leaf-batch 6 --solver-cpus 4 `
  --device cuda --amp-inference
```

The output is atomic and resumable after each batch. Progress logs contain only
completion counts, not partial outcomes. A smoke, if later authorized, must use
`--limit-positions`, `--allow-nonstandard-design`, and a separate output path so
it cannot collide with the frozen full run.

## Result

The run completed all 2,112 planned cells in 3,578.7 seconds (59m39s wall
time). Positive deltas favor the sampled-split forced action.

| Frozen cohort | Positions | Chooser-margin delta (95% position bootstrap) | Chooser-points delta (95% position bootstrap) | Positions split / tied / control |
| --- | ---: | ---: | ---: | ---: |
| All disagreements (primary) | 33 | +1.523 [-0.056, +3.162] | +0.0672 [-0.0152, +0.1553] | 18 / 1 / 14 |
| Pick-changing disagreements | 17 | +2.884 [+0.059, +5.665] | +0.1158 [-0.0368, +0.2702] | 11 / 1 / 5 |

The primary result is **inconclusive** under the frozen gate because the lower
margin bound is slightly below zero. The 17-position subgroup clears zero on
margin, but its official-points interval does not. Because that subgroup was
declared before outcomes were opened, it is valid mechanistic evidence that the
split may improve the consequential tile-changing choices; it does not override
the primary gate or establish a whole-game strength gain.

The result is materially more encouraging than the prior whole-game null. The
forced duel concentrates evaluation on decisions that actually changed, and
the incumbent open-loop continuation arbiter is conservative with respect to
split-preferred lines. However, the primary interval is unresolved and the
subgroup interval only narrowly clears zero. The documented decision is
therefore **do not start chance-aware training yet**. Run the exact
conditional-value audit next. Do not add more seeds to this already revealed
sample; any forced-duel extension must be separately preregistered and use a
disjoint seed set.

### Post-run integrity audit

- `completed=true`; 2,112 cells, 2,112 unique composite keys, 528 continuation
  quartets, 33 positions, and 17 pick-changing positions.
- Each arm/mirror cohort contains exactly 528 cells. Every position contains 64
  cells.
- All 528 quartets match on deck seed, game/search seed, root state, and exact
  shuffled hidden-deck order. Every forced action equals its frozen arm action.
- All 68 chunks report zero chance panels, confirming that the continuation
  arbiter stayed aliased open loop.
- Player-label mirrors differ by only 0.144 chooser-margin points on average.
- Exact-endgame accounting is balanced: 2,112 continuation trees and 24,864
  exact moves were produced. There were 240 intermediate exact-solver timeouts
  that fell back to MCTS, split 121 control versus 119 sampled-split. These are
  timed-out decision-level exact attempts, not missing terminal game results.
- Independent reconstruction of the summary from raw result cells matched the
  stored summary exactly.

### Reproducibility

The result artifact is
`runs/kingdomino/chance_correct_a1/deck8_forced_move_duel_v1.json` with SHA-256
`6b760350d0ef00ee279344dbbc7c4976ca4059753cddb1383a62dcdfd89a34a`.
It records source commit `13c8361d582ffb1908b8d69e8118c40caeed0289`.

The frozen checkpoint, positions, causal-leverage input, sampled-split input,
runner, and Rust source hashes all matched their recorded provenance at review
time. The raw run artifact remains under the ignored `runs/` directory; this
document is the reviewable result record.

## Outcome-blind post-run review checklist

These gates were written while the sealed run was in progress and before any
score, winner, or treatment-effect field was inspected. Counter terminology was
clarified afterward from its Rust definition; the gates were not changed.

### Completeness and identity

- `completed` is true and the artifact contains exactly 2,112 result cells:
  33 positions × 16 continuation seeds × 2 label mirrors × 2 forced arms.
- The composite `(position, continuation_index, mirror, arm)` key is unique.
- Each of the four arm/mirror cohorts contains 528 cells, and every position
  contains 64 cells.
- Checkpoint, position corpus, causal-leverage artifact, sampled-split artifact,
  runner, and Rust hashes match the frozen provenance.
- The recorded disagreement table still contains 33 joint-action changes and
  17 pick changes at the 6,400-simulation budget.

### Pairing and arbiter integrity

- Within each `(position, continuation_index)`, both arms and both mirrors use
  the same deck seed and search seed.
- Paired arms record the same shuffled hidden-deck ordering before the forced
  action; label mirrors preserve that ordering exactly.
- `forced_action_idx` equals the frozen control or split action for its arm and
  differs between arms at every included position.
- Every chunk reports zero chance panels. The continuation arbiter must remain
  aliased open loop throughout.
- Exact tree solves and exact-attempt fallbacks are reported by arm and mirror.
  A fallback is a decision-level exact attempt that exceeded its budget and
  returned to MCTS; it is not a missing or approximate terminal score.

### Analysis integrity

- Compute split-minus-control margin and points in the chooser frame.
- Average the two label mirrors inside each continuation seed, then average the
  16 seeds inside each position, then bootstrap the 33 position means.
- Do not bootstrap 1,056 continuation pairs as if they were independent.
- Report all 33 disagreements as primary and the 17 pick-changing positions as
  the preregistered subgroup; do not select a more favorable subset afterward.
- Report mirror disagreement and any arm imbalance in exact fallbacks alongside
  the treatment effect.

## Frozen next-step decision tree

- **Positive screen:** prepare—but do not automatically launch—a chance-aware
  training/fine-tuning design and its crossed old/new-network evaluation. The
  forced duel supplies the missing action-quality rationale.
- **Negative screen:** do not train this mechanism. A conditional-value audit
  may still distinguish genuinely inferior split actions from incompatibility
  with the incumbent downstream policy, but it is diagnostic rather than a
  reason to override the negative result.
- **Inconclusive screen:** do not extend these revealed seeds and do not start
  training. Run the exact conditional-value audit next, or preregister a
  disjoint forced-duel extension only if the observed position-level variance
  makes a practically relevant effect resolvable at acceptable cost.

No branch treats a null as proof of equivalence, and no branch enables sampled
split in self-play directly from this conditional disagreement experiment.
