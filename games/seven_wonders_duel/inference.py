"""Batched-inference service API (plan §4: the search-facing boundary).

Phase C's searcher calls :meth:`Evaluator.evaluate` with any number of
positions; the service vectorizes, pads, and runs one batched forward. The
call signature — many requests in, aligned results out — is the coalescing
boundary: the Phase F in-process server (KD leaf-coalescing design) slots in
behind this exact interface, and the searcher never changes.

Results are actor-relative like everything else: ``policy`` is a probability
vector aligned to the request's legal-index list, ``wdl`` is win/draw/loss
from the actor's seat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .dataset import collate_inputs, vectorize
from .encoder import Encoding, encode
from .net import masked_policy_log_softmax


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: np.ndarray  # [L] probabilities aligned to the request's legal list
    wdl: np.ndarray  # [3] win/draw/loss probabilities (actor-relative)
    joint7: np.ndarray  # [7] winner x victory-type probabilities
    margin: float
    military: float
    science: np.ndarray  # [2] my/opp final symbol-count forecasts


class Evaluator:
    """Synchronous batched evaluator. Thread-safety and cross-caller
    coalescing arrive with the Phase F service; the API does not change."""

    def __init__(self, model, device: str = "cpu", max_batch: int = 512):
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = max_batch

    @torch.no_grad()
    def evaluate(
        self,
        encodings: Sequence[Encoding],
        legal_lists: Sequence[Sequence[int]],
    ) -> list[Evaluation]:
        if len(encodings) != len(legal_lists):
            raise ValueError("encodings and legal_lists must align")
        results: list[Evaluation] = []
        for start in range(0, len(encodings), self.max_batch):
            chunk = encodings[start : start + self.max_batch]
            legals = legal_lists[start : start + self.max_batch]
            batch = collate_inputs(
                [vectorize(e) for e in chunk], list(legals), self.device
            )
            outputs = self.model(batch)
            log_policy = masked_policy_log_softmax(
                outputs["policy"], batch["legal_mask"]
            )
            policy = log_policy.exp().cpu().numpy()
            wdl = torch.softmax(outputs["value"], dim=-1).cpu().numpy()
            joint7 = torch.softmax(outputs["joint7"], dim=-1).cpu().numpy()
            margin = outputs["margin"].cpu().numpy()
            military = outputs["military"].cpu().numpy()
            science = outputs["science"].cpu().numpy()
            for row, legal in enumerate(legals):
                legal_indices = np.asarray(list(legal), dtype=np.int64)
                results.append(
                    Evaluation(
                        policy=policy[row, legal_indices].astype(np.float32),
                        wdl=wdl[row].astype(np.float32),
                        joint7=joint7[row].astype(np.float32),
                        margin=float(margin[row]),
                        military=float(military[row]),
                        science=science[row].astype(np.float32),
                    )
                )
        return results

    def evaluate_states(self, games) -> list[Evaluation]:
        """Convenience for callers holding engine states rather than
        encodings; uses each state's actor observation and legal indices."""

        from .codec import legal_action_indices

        encodings = []
        legals = []
        for game in games:
            actor = (
                game.pending_choice.player
                if game.pending_choice is not None
                else game.active_player
            )
            encodings.append(encode(game.observation(actor)))
            legals.append(legal_action_indices(game))
        return self.evaluate(encodings, legals)
