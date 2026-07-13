"""Kingdomino-specific data loading for the NNUE trainer.

The ONLY game-specific module in the NNUE pipeline: it turns the self-play
`ReplayBuffer` pickle into the flat feature matrix + label vectors that the
game-agnostic net (`net.py`) and training loop (`train.py`) consume. A second game
would replace just this file.

Feature vector per position (the EXISTING dense encoding, flattened — Step 2 uses
no new features): [ my_board (1521) | opp_board (1521) | flat (333) ] = 3375, all
ACTOR-relative (the encoder is actor-relative). Labels are the actor's view too:
outcome = win_target (1 win / 0.5 draw / 0 loss); margin = own_score - opp_score.
"""
from __future__ import annotations

import pickle

import numpy as np

from games.kingdomino.encoder import FLAT_SIZE, NUM_BOARD_CHANNELS, CANVAS_SIZE

BOARD_SIZE = NUM_BOARD_CHANNELS * CANVAS_SIZE * CANVAS_SIZE  # 9*13*13 = 1521
INPUT_DIM = 2 * BOARD_SIZE + FLAT_SIZE                       # 3375


def load_examples(path):
    # The buffer was pickled while self_play ran as the entry point, so its
    # `Example` class was stored under the name `__main__.Example`. Bind the real
    # class onto whatever module is currently `__main__` so the unpickler resolves
    # it (self_play.ReplayBuffer.save writes the payload dict {'data', ...}).
    import sys
    from games.kingdomino.self_play import Example
    sys.modules["__main__"].Example = Example
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["data"] if isinstance(payload, dict) else payload


def build_arrays(examples):
    """-> (X float32 [N, INPUT_DIM], outcome [N] in {0,.5,1}, margin_raw [N],
    iteration [N] int64). Boards are stored float16; the copy upcasts to float32."""
    n = len(examples)
    X = np.empty((n, INPUT_DIM), np.float32)
    outcome = np.empty(n, np.float32)
    margin = np.empty(n, np.float32)
    iteration = np.empty(n, np.int64)
    b2 = 2 * BOARD_SIZE
    for i, e in enumerate(examples):
        X[i, :BOARD_SIZE] = e.my_board.reshape(-1)
        X[i, BOARD_SIZE:b2] = e.opp_board.reshape(-1)
        X[i, b2:] = e.flat
        outcome[i] = e.win_target
        margin[i] = float(e.own_score) - float(e.opp_score)
        iteration[i] = int(getattr(e, "iteration", 0))
    return X, outcome, margin, iteration


def iteration_split(iteration, val_frac, seed):
    """Game-honest split: hold out WHOLE iterations. Each self-play game is generated
    within one training iteration, so no game's correlated positions straddle the
    train/val boundary (unlike a random-position split). Holding out a RANDOM subset
    of iterations (not just the latest) keeps train/val on the same distribution.
    Returns (train_mask, val_mask, held_out_iterations)."""
    iters = np.unique(iteration)
    rng = np.random.default_rng(seed)
    k = max(1, round(len(iters) * val_frac))
    val_iters = rng.choice(iters, size=k, replace=False)
    val_mask = np.isin(iteration, val_iters)
    return ~val_mask, val_mask, sorted(int(v) for v in val_iters)
