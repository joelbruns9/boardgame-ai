# Classic Kingdomino 3-4 Player Project

## Scope

Build one player-equivariant model for standard 5x5 Kingdomino across:

- three and four players;
- Harmony enabled or disabled;
- Middle Kingdom enabled or disabled.

Two-player Mighty Duel is explicitly out of scope. It remains in
`games/kingdomino` with separate engines, checkpoints, replay data, and ratings.
The Lost Treasures expansion is also out of scope for K0; the first rules oracle
covers the base game plus the Harmony and Middle Kingdom scoring toggles.

Artifact identity begins with:

```text
game_id: kingdomino_classic
rules_version: 1
configuration: {players}p-h{0|1}-m{0|1}
```

## Standing design decisions

1. Python is the rules oracle until profiling justifies native acceleration.
2. Three- and four-player games share a fixed four-player tensor layout.
3. The absent fourth player in 3p is represented by an explicit presence mask.
4. Every round exposes four draft slots in both player counts. In 3p, three
   dominoes are selected and the unclaimed fourth domino is discarded. In 4p,
   all four are selected and the final selection is forced.
5. Search values are vectors. Multiplayer search never uses two-player sign
   inversion.
6. Training samples carry their complete rules configuration and are balanced
   by configuration at replay-sampling time.
7. The running Mighty Duel chance experiment is integrated before shared search
   or self-play code is extracted. Concepts can transfer; experiment-specific
   implementation does not automatically become the multiplayer API.

## Milestones and gates

### K0 - Rules engine

- Deterministic seeded setup using all 48 dominoes over twelve rounds.
- Four-tile draft rows, including the 3p unclaimed-tile discard and the 4p
  forced final selection.
- 5x5 placement legality, forced discards, bonuses, and tiebreaks.
- Copy, serialization, terminal-state, and full-game inventory invariants.
- Scripted rules examples and BGA replay equivalence where available.

Gate: thousands of seeded random games complete without invariant violations,
and every scripted rules example agrees with the oracle.

### K1 - Baselines

Status: complete (2026-08-12). See `K1_BASELINES.md` for the held-out,
seat-balanced gate and reproduction commands.

- Random bot.
- Immediate-score greedy bot.
- Placement-flexibility bot.
- Pick-denial-aware bot.

Gate: each intended strength tier beats the preceding tier in seat-balanced
tournaments and becomes a stable evaluation anchor.

### K2 - Encoder and action codec

- Four padded player slots with presence masks.
- Shared per-player board representation.
- Player count and bonus configuration features.
- Fixed placement-by-four-pick policy space, with all four pick slots available
  at the start of a 3p or 4p draft.
- D4 board symmetry and seat-permutation transforms.

Gate: action round trips, legal-mask equivalence, symmetry commutation,
seat-permutation equivariance, and 3p padding invariance.

### K3 - Network and multiplayer MCTS

- Shared board encoder and player interaction layer.
- Masked four-player score/rank/win outputs.
- Vector-valued MCTS backup and actor-component selection.
- Chance handling informed by the completed Mighty Duel experiment.

Gate: toy multiplayer search games with known solutions, followed by search
that beats the K1 baselines using an untrained or lightly trained evaluator.

### K4 - Mixed self-play

- Explicit sampling over all eight rule configurations.
- Configuration-balanced replay sampling.
- Configuration-aware manifests and checkpoint compatibility checks.
- 3p and 4p specialists trained as reference models.

Gate: the unified model reaches competitive strength against both specialists
without a systematic regression on either player count.

### K5 - Evaluation and generalization

Report separately for every configuration:

- win rate and mean rank;
- normalized score and last-place rate;
- seat/start-position effects;
- performance against mixed baseline and checkpoint populations.

Run a leave-one-bonus-combination-out experiment only after joint competence is
established. Player-count transfer is measured by pretraining/fine-tuning, not
claimed from unsupported zero-shot evaluation.
## Directory responsibilities

| Path | Responsibility |
|---|---|
| `config.py` | Supported rules configurations and artifact identity |
| `board.py` | 5x5 placement and scoring oracle |
| `game.py` | Setup, draft, turns, terminal state, and returns |
| `action_codec.py` | Fixed action indices and legal masks |
| `encoder.py` | Tensor encoding and symmetry/seat transforms |
| `network.py` | Shared player encoder, policy, and value heads |
| `mcts.py` | Vector backup and stochastic multiplayer search |
| `self_play.py` | Balanced generation and replay records |
| `baselines.py` | Stable handcrafted evaluation anchors |
| `evaluation.py` | Seat-balanced tournaments and reports |
| `configs/` | Reviewed experiment configurations |
| `tests/` | Rules, equivalence, invariance, and regression gates |
