# K3 Equivariant Network and Vector MCTS

Network version: `1`.

K3 implements the network and Python search oracle. Architecture and search
correctness gates are complete. The small heuristic strength pilot clears the
flexibility anchor in both player counts; promotion over the strongest
denial-aware anchor remains pending a trained evaluator or a larger preregistered
evaluation. This distinction is intentional—the pilot is not promoted into a
strength claim.

## Network contract

The network applies the same board CNN, domino encoder, player MLP, attention,
and value heads to all four padded player slots. Cross-player interaction is
masked self-attention followed by a masked mean. There are no seat embeddings,
slot-specific weights, ordered-opponent blocks, or flattening across players.

Outputs are:

| Output | Shape | Symmetry |
|---|---:|---|
| policy logits | `[batch, 2170]` | invariant to player relabeling; D4-equivariant |
| score | `[batch, 4]` | player-equivariant; D4-invariant |
| rank logits | `[batch, 4, 4]` | player-equivariant; D4-invariant |
| win logits/probabilities | `[batch, 4]` | player-equivariant; D4-invariant |

Absent-player score and rank rows are zero. The impossible fourth-place class
is negative infinity in 3p. Absent win logits are negative infinity and their
probabilities are zero. Present win probabilities form a simplex.

An ordinary CNN does not guarantee rotational equivariance. The forward pass
therefore applies a Reynolds projection: evaluate all eight D4 board images,
map each policy back through the exact K2 action permutation, and average.
Values are averaged without a policy transform. This makes the contracts hold
for arbitrary weights, rather than expecting training augmentation to learn
them approximately.

`NetworkConfig.manifest_fields()` stamps network, encoder, action-codec, and
policy-size compatibility into future checkpoints.

## Search contract

`VectorMCTS` backs up a four-component value at every edge. A decision node's
PUCT score reads the component belonging to that node's actor. Values are never
negated, and another player's node is not treated as a minimizing node.

The generic search adapter is exercised by a solved three-player tree: player 1
chooses its own 0.8 outcome even though it reduces player 0 from 0.8 to 0.1;
player 0 anticipates this and selects a separate guaranteed 0.6 action.

The Classic adapter returns padded official shared-win values at terminal
nodes. Network evaluation and lightweight K1-derived heuristic evaluation are
both supported. The tiered heuristic can use denial-aware root priors and
cheaper flexibility evaluation inside the tree.

## Chance handling

The completed mechanical findings from the Mighty Duel chance work are adopted
without copying its large two-player engine:

- hidden deck order is never read by evaluation or schedule construction;
- reveal rows come from a value-independent, uniform without-replacement
  schedule over four-domino combinations;
- the schedule seed is canonical under player relabeling and D4 geometry;
- a node starts with one lazy sampled outcome and admits four after two visits;
- persistent widening requires at least four real visits per active outcome;
- width doubles to the next prefix and is capped at 16;
- active outcomes have equal weight, including bootstrap-only outcomes;
- parent selection reads the chance node's current mean dynamically.

The cap-16, depth-before-width choice and current-mean semantics match the
approved chance experiment. Its Phase-B training result is still external to
this milestone, so K3 does not claim that chance-aware training has succeeded.

## Verification

Tests cover:

- every S3/S4 permutation for arbitrary network weights;
- all eight D4 transforms for policy and all value heads;
- masking, padded values, and present-player win normalization;
- end-to-end inference from an untrained network through vector MCTS;
- solved multiplayer actor-component selection with no sign flip;
- chance admission, widening, equal means, and public schedule invariance;
- complete Classic search commutation under player relabeling;
- hidden-deck-order and D4 invariance at reveal boundaries.

## Directional strength pilot

Configuration: Harmony and Middle Kingdom enabled, 16 simulations per move,
heuristic leaf evaluation, chance width capped at 8 for Python runtime. Each
seed rotates the challenger through all seats. These samples are screens, not
confidence-qualified promotions.

| Players | Challenger | Field | Seeds | Seat-games | Win share | Neutral | Mean rank |
|---:|---|---|---:|---:|---:|---:|---:|
| 3 | flexibility-prior MCTS | flexibility | 0-1 | 6 | 50.0% | 33.3% | 1.833 |
| 4 | flexibility-prior MCTS | flexibility | 0-2 | 12 | 33.3% | 25.0% | 2.250 |

The same configuration was neutral in 3p and below neutral in 4p on one seed
against denial-aware. A tiered denial-root variant won its first 4p screen but
lost its first 3p screen; a higher PUCT constant did not repair the mismatch.
Those exploratory seeds are not a gate. The next legitimate strength step is a
lightly trained symmetric evaluator and a frozen multi-seed suite against
denial-aware, not further manual tuning on this pilot.
