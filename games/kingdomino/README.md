# Kingdomino AlphaZero

An AlphaZero-style self-play AI for 2-player Mighty Duel Kingdomino,
built as a hobby and learning project. The goal is to beat top Board
Game Arena (BGA) players.

## Current Status

- **Architecture:** 32 channels, 4 residual blocks (32ch/4b)
- **Best Elo:** ~952 (vs anchor pool at sims=400)
- **vs GreedyBot:** 100% win rate, +74 score margin
- **Training:** ~300 iterations of self-play completed across 4 runs

## Architecture Overview

### Network (network.py)
Four-head residual network:
- **Shared trunk:** N residual blocks of 3×3 convolutions (C channels)
- **policy_head:** joint (placement × pick) logits via bilinear interaction
- **own_score_head:** predicts own final score (auxiliary training signal)
- **opp_score_head:** predicts opponent final score (blocking signal)
- **win_prob_head:** predicts win probability

Inputs:
- `my_board` (9, 13, 13) — own kingdom, castle-centered on 13×13 canvas
- `opp_board` (9, 13, 13) — opponent kingdom
- `flat` (333,) — global features (pending tiles, score/bonus summaries, pick positions, game phase, round, etc.)

### Search (lib.rs — Rust)
Open-loop BatchedMCTS: runs 32 game slots simultaneously, resampling
the hidden deck order per simulation rather than committing to one
determinized future. This is architecturally necessary for Kingdomino
because the future tile reveal order is hidden information.

Key constants:
- `FLAT_SIZE = 333` — flat encoder dimension (symmetric pending-tile + board-fact summaries + pick_pos_0..3 encoding)
- `checkpoint_version = 3` — symmetric pending-tile encoder format

### Training (self_play.py)
AlphaZero self-play loop:
1. Generate games using current network + BatchedMCTS (sims=1600)
2. Add positions to replay buffer (300k capacity, ~25 iter history)
3. Train network on sampled buffer positions
4. (Periodic) Rate checkpoint against Elo anchor pool
5. Save checkpoint

Leaf value formula:

    leaf_value = α·tanh((own-opp)·MARGIN_GAIN) + (1-α)·(2·win_prob-1)

MARGIN_GAIN = 2.0, α = 0.8

### Elo Rating (elo_rating.py)
Automated Elo rating using open-loop BatchedMCTS with two-network
routing (searcher-owns-network via `row_search_actors()`). Ratings
use one-shot MLE per checkpoint inline, with a global Bradley-Terry
re-solve (`--resolve`) at run end.

Anchor pool (sims=400, fpu=-0.2):
- greedy_bot: Elo 0 (reference floor, not played in routine rating)
- cloud_iter100: Elo ~510
- local_cont_iter070: Elo ~781
- local_cont_iter100: Elo ~876

## Project Structure

```
games/kingdomino/
├── self_play.py          # Main training loop — start here
├── network.py            # KingdominoNet (four-head ResNet)
├── encoder.py            # Board → tensor encoding (FLAT_SIZE=333)
├── mcts_az.py            # Python open-loop MCTS (reference/testing)
├── round_robin_eval.py   # Head-to-head checkpoint evaluation
├── elo_rating.py         # Automated Elo rating system
├── elo_anchors.csv       # Anchor pool configuration
├── elo_db.json           # Current Elo ladder
├── elo_games.jsonl       # Append-only game log (accumulates across runs)
├── bots.py               # GreedyBot, RandomBot baselines
├── game.py               # Game engine (GameState, Phase, dominoes)
├── board.py              # Board representation
├── dominoes.py           # Domino tile definitions
├── action_codec.py       # Action encoding/decoding
├── augmentation.py       # D4 board symmetry augmentation
├── inference_service.py  # Local inference server
├── correctness_oracle.py # Determinism/equivalence tests
├── policy_compare.py     # Statistical policy divergence harness
├── best_checkpoint/      # Best trained checkpoints
│   └── best_32x4.pt
├── docs/                 # Research papers and reference materials
└── kingdomino_rust/      # Rust engine (BatchedMCTS)
    └── src/lib.rs
```

### Deprecated / Legacy files
See individual file headers for details.
- `parallel_self_play.py` — superseded by Rust BatchedMCTS
- `threaded_self_play.py` — superseded by Rust BatchedMCTS
- `mcts.py` — pre-AlphaZero random-rollout MCTS
- `mcts_match.py` — legacy match runner
- `profile_actions.py`, `profile_selfplay.py` — pre-Rust diagnostics
- `sim_concentration_probe.py` — one-time diagnostic
- `supervised_validation.py` — early-development harness

## Setup

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (RTX 3070 or better recommended)
- Rust toolchain: https://rustup.rs

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd boardgame-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux

# Install Python dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy maturin

# Build Rust engine
cd games\kingdomino\kingdomino_rust
maturin develop --release
cd ..\..\..
```

### Linux setup (for cloud instances)
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Python deps
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy maturin

# Build Rust engine
cd games/kingdomino/kingdomino_rust
maturin develop --release
cd ../../..
```

## Training

### Canonical training command (soft-gated laptop overnight)
```powershell
.\.venv\Scripts\python.exe -m games.kingdomino.self_play --engine batched_open_loop --device cuda --warm_start_current_best --selfplay_generator_mode soft_gate --promotion_every 5 --promotion_games 384 --promotion_sims 100 --smart_elo --smart_elo_on_promote --smart_elo_games_per_anchor 32 --iterations 60 --games_per_iter 160 --train_steps 600 --sims 1600 --channels 48 --blocks 6 --batch_slots 32 --leaf_batch 6 --lr 1e-4 --buffer 300000 --lambda_score 0.5 --lambda_w 0.25 --score_scale 160 --margin_gain 2.0 --alpha 0.2 --fpu -0.2 --c_puct 1.5 --benchmark_every 10 --benchmark_sims 50 --benchmark_seeds 20 --checkpoint_dir runs\kingdomino\local_48x6_softgate --save_buffer runs\kingdomino\local_48x6_softgate\buffer_final.pkl --elo_every 0 --elo_sims 400 --elo_games_per_anchor 32 --elo_db elo_db.json --elo_games_log elo_games.jsonl --seed 0
```

See `training_parameters.md` for full parameter documentation.

### Key invariants
These must be preserved across all runs:
- `FLAT_SIZE = 333` (symmetric pending-tile + board-fact encoder)
- `checkpoint_version = 3` (symmetric pending-tile encoder)
- Open-loop MCTS is required — closed-loop (det=1) trains on a
  non-existent game with revealed future draws
- `terminal_search_value()` is the single source for terminal backup
- MARGIN_GAIN and ALPHA are bound into evaluator closures at
  construction time (not read from module globals)

## Elo Rating

### View current ladder
```bash
python -m games.kingdomino.elo_rating --leaderboard
```

### Rate a new checkpoint
```bash
python -m games.kingdomino.elo_rating --checkpoint <path> --name <name> --sims 400 --games_per_anchor 32 --device cuda --verbose
```

### Re-solve global ladder from full game log
```bash
python -m games.kingdomino.elo_rating --resolve --verbose
```

### Re-bootstrap anchor pool (after adding a new anchor)
```bash
python -m games.kingdomino.elo_rating --reanchor --sims 400 --games_per_anchor 32 --device cuda --verbose
```

## Cloud Training (Vast.ai)

### Merging Elo data from cloud and local runs
The `elo_games.jsonl` game log accumulates on whichever machine runs
training. After each cloud run:

```bash
# On cloud machine — download the game log
scp user@cloud:/path/to/elo_games.jsonl elo_games_cloud.jsonl

# On laptop — append cloud games to local log, deduplicate, re-solve
cat elo_games_cloud.jsonl >> elo_games.jsonl
# Remove duplicate lines (same checkpoint/opponent/seed/orientation/timestamp)
sort -u elo_games.jsonl -o elo_games.jsonl
python -m games.kingdomino.elo_rating --resolve --verbose
```

Checkpoint names are prefixed with the run directory name
(e.g. `checkpoints_ol_cloud_iter_0010`) ensuring no collision between
local and cloud runs in the shared game log.

### Cloud run preparation checklist
- [ ] Run throughput benchmarks first: bench_compile.py, bench_doublebuffer.py
- [ ] Profile eval breakdown: H2D / forward / readback split
- [ ] Re-sweep batch_slots on cloud hardware (optimal may differ from laptop's 32)
- [ ] Transfer best checkpoint and buffer: scp best_32x4.pt buffer_final.pkl cloud:
- [ ] Confirm elo_anchors.csv paths use forward slashes (already fixed)
- [ ] Test --compile flag (Triton available on Linux)

## Correctness Gates

Before any new engine goes into training:
```bash
python -m games.kingdomino.correctness_oracle  # determinism gate
python -m games.kingdomino.policy_compare      # statistical divergence harness
```

## Research Papers

The `docs/` folder contains the papers that informed this implementation:
- AlphaZero.pdf — DeepMind AlphaZero paper
- Open_Loop_MCTS.pdf — open-loop MCTS for hidden information games
- Stochastic_MCTS.pdf — MCTS for stochastic games
- King_Domino_MCTS.pdf — Kingdomino-specific MCTS work (Gedda 2018)
- MCTS_for_multiplayer.pdf, MCTSsurvey.pdf — MCTS background
- Backgammon_MCTS.pdf, Catan_MCTS.pdf — stochastic game references
- kingdomino_rules.pdf — official rules
- Kingdomino_tiles_8x6_With_Frequency.png — tile reference with frequencies
