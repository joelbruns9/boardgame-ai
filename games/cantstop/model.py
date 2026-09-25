"""Value network for Can't Stop, and the adapter that makes it a solver leaf.

The net is **value-only** (D4): no policy head, because the turn solver
enumerates and checks every option itself. It reads an end-of-turn board and
predicts who wins, as a probability per seat.

Output is over ``MAX_SEATS`` slots in *encoding* order (slot 0 is the seat to
move), with absent seats masked out of the softmax so a 2-player board never
spends probability mass on seats 3 and 4. ``NetEvaluator`` then rotates back
to absolute seat order, which is the contract ``TurnSolver`` expects and
``ProgressHeuristic`` defines.
"""

import numpy as np
import torch
import torch.nn as nn

from .encoder import (
    FEATURE_SIZE, MAX_SEATS, encode_batch, seat_mask, seat_present_index,
    to_absolute,
)

DEFAULT_HIDDEN = (256, 256)


class CantStopNet(nn.Module):
    """MLP over the flat board encoding, one win-probability head."""

    def __init__(self, hidden=DEFAULT_HIDDEN, input_size=FEATURE_SIZE):
        super().__init__()
        self.input_size = input_size
        self.hidden = tuple(hidden)

        layers = []
        prev = input_size
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, MAX_SEATS)

    def forward(self, x):
        """Raw per-slot logits, unmasked. Training uses these."""
        return self.head(self.trunk(x))

    def win_probs(self, x, mask=None):
        """Softmax over the live seats only.

        ``mask`` defaults to the seat-present flags carried in ``x`` itself,
        so the mask cannot drift out of step with the features it masks.
        """
        logits = self.forward(x)
        if mask is None:
            mask = seat_mask_tensor(x)
        return masked_softmax(logits, mask)

    def config(self):
        return {"hidden": list(self.hidden), "input_size": self.input_size}


def masked_softmax(logits, mask):
    """Softmax restricted to ``mask``; masked slots come out exactly zero."""
    neg_inf = torch.finfo(logits.dtype).min
    return torch.softmax(logits.masked_fill(~mask, neg_inf), dim=-1) * mask


def seat_mask_tensor(x):
    """Live-seat mask for a feature tensor, as a bool tensor on its device.

    Indexed on the tensor's own device -- the same ``> 0`` test as
    ``encoder.seat_mask``. The first version round-tripped the WHOLE feature
    tensor through numpy to read four columns: measured at 64 of 123 ms of a
    900k-row GPU forward in the M3 pool.
    """
    if x.ndim == 1:
        x = x[None, :]
    idx = torch.tensor([seat_present_index(s) for s in range(MAX_SEATS)],
                       device=x.device)
    return x.index_select(1, idx) > 0.0


def masked_soft_cross_entropy(logits, targets, mask):
    """Cross-entropy against a target DISTRIBUTION over encoding slots
    (N, MAX_SEATS), live seats only -- the TD targets. Absent seats carry
    zero target mass and are zeroed out of the log-probabilities, so their
    masked logits contribute nothing. With one-hot targets this equals
    ``masked_cross_entropy``."""
    neg_inf = torch.finfo(logits.dtype).min
    logp = torch.log_softmax(logits.masked_fill(~mask, neg_inf), dim=-1)
    return -(targets * logp.masked_fill(~mask, 0.0)).sum(dim=-1).mean()


def masked_cross_entropy(logits, target_slots, mask):
    """Cross-entropy against a one-hot winner, over live seats only.

    ``target_slots`` are encoding-slot indices (not absolute seats), which is
    what ``encoder.seat_to_slot`` produces for a training row.
    """
    neg_inf = torch.finfo(logits.dtype).min
    return nn.functional.cross_entropy(
        logits.masked_fill(~mask, neg_inf), target_slots)


class NetEvaluator:
    """Wraps a net as the ``evaluate(boards) -> (N, num_players)`` callable
    that ``TurnSolver`` takes, matching ``ProgressHeuristic``'s contract.

    The solver calls this **once per turn** with every stoppable leaf plus the
    single shared bust board, so batches are naturally large (1k-13k rows) and
    there is no per-leaf round trip to amortize.

    Every board in one call comes from the same turn, hence shares a rule set
    and seat to move, so one rotation serves the whole batch.
    """

    def __init__(self, net, device=None, batch_size=None):
        self.net = net
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net.to(self.device).eval()
        self.batch_size = batch_size
        self.calls = 0
        self.rows = 0

    @torch.no_grad()
    def __call__(self, states):
        if not states:
            raise ValueError("evaluator called with no boards")
        return self.evaluate_features(encode_batch(states), states[0])

    @torch.no_grad()
    def relative_probs(self, features):
        """Seat-relative win probabilities (N, MAX_SEATS), float32, for any
        mix of rule sets and seats to move -- one forward. The Rust pool
        batches every waiting game's leaves through here, then rotates each
        game's block back to absolute seats itself."""
        chunks = []
        # Chunked even by default: a lookahead round can hand over a million
        # rows, and one forward over all of them does not fit an 8 GB GPU.
        step = self.batch_size or 262_144
        for start in range(0, len(features), step):
            x = torch.from_numpy(features[start:start + step]).to(self.device)
            chunks.append(self.net.win_probs(x).cpu().numpy())
        probs = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        self.calls += 1
        self.rows += len(features)
        return probs

    def evaluate_features(self, features, reference):
        """The same, from already-encoded boards. ``reference`` is any board
        sharing the batch's rule set and seat to move -- only those two are
        read, to rotate the output back to absolute seats. This is the entry
        the Rust solver uses: it encodes leaves itself (``RustTurnSolver``)."""
        probs = self.relative_probs(features)
        # to_absolute widens to float64, which is the contract the solver
        # needs: backward induction sums these thousands of times, and the
        # Rust port is specified to accumulate in f64 for the same reason.
        return to_absolute(probs, reference)


def save_net(net, path):
    torch.save({"config": net.config(), "state_dict": net.state_dict()}, path)


def load_net(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=True)
    net = CantStopNet(**blob["config"])
    net.load_state_dict(blob["state_dict"])
    return net
