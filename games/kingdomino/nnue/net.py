"""Game-agnostic NNUE-shaped two-head evaluator.

Architecture (the NNUE shape — bigness is incidental, sparse+incrementally-updatable
+ expressive is the point):

    feature vector (input_dim)
        -> [accumulator]  Linear(input_dim, acc_width) -> ReLU
        -> [tail]         Linear(acc_width, tail_hidden) -> ReLU
                          Linear(tail_hidden, tail_hidden) -> ReLU
        -> two heads:
             outcome_logit  (sigmoid -> win probability in (0,1)), trained on
                            win_target (1 win / 0.5 draw / 0 loss), ACTOR frame
             margin         normalized final score margin (own - opp), ACTOR frame

The first Linear is the "accumulator" that Step 3 will make incrementally
updatable over a SPARSE feature set; the rest of the architecture is unchanged, so
this net + its trainer carry forward verbatim. Plain (non-clipped) ReLU on the
accumulator per the Azul NNUE lesson: clipped ReLU destroys point-magnitude
information in a scoring game.

Frame: the net is ACTOR-relative (the encoder is actor-relative my/opp). The
player-0-frame flip the searcher needs happens in the eval wrapper that knows
whose turn it is, NOT here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TwoHeadNNUE(nn.Module):
    def __init__(self, input_dim: int, acc_width: int = 256, tail_hidden: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.acc_width = acc_width
        self.tail_hidden = tail_hidden

        self.accumulator = nn.Linear(input_dim, acc_width)
        self.tail = nn.Sequential(
            nn.Linear(acc_width, tail_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(tail_hidden, tail_hidden),
            nn.ReLU(inplace=True),
        )
        self.outcome_head = nn.Linear(tail_hidden, 1)
        self.margin_head = nn.Linear(tail_hidden, 1)

    def forward(self, x: torch.Tensor):
        """Returns (outcome_logit, margin_pred), both (B,). Apply sigmoid to the
        logit for a win probability; the margin is in normalized units (see the
        trainer's margin_scale)."""
        a = torch.relu(self.accumulator(x))
        h = self.tail(a)
        return self.outcome_head(h).squeeze(-1), self.margin_head(h).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor):
        """Inference convenience: (win_prob in (0,1), margin_pred normalized)."""
        logit, margin = self.forward(x)
        return torch.sigmoid(logit), margin


def config_of(net: "TwoHeadNNUE") -> dict:
    """Architecture hyperparameters to persist alongside the weights (so the Rust
    forward-pass port and any reloader can reconstruct the net exactly)."""
    return {
        "input_dim": net.input_dim,
        "acc_width": net.acc_width,
        "tail_hidden": net.tail_hidden,
    }
