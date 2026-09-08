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

import contextlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .dataset import collate_inputs, vectorize
from .encoder import Encoding, encode
from .net import fuse_for_inference, masked_policy_log_softmax


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: np.ndarray  # [L] probabilities aligned to the request's legal list
    wdl: np.ndarray  # [3] win/draw/loss probabilities (actor-relative)
    joint7: np.ndarray  # [7] winner x victory-type probabilities
    margin: float
    military: float
    science: np.ndarray  # [2] my/opp final symbol-count forecasts
    #: W4's hierarchical read of the same seven classes, or None when the model
    #: has no such head.
    #:
    #: Not a second opinion to average with `joint7`. The two differ in a
    #: specific way: `joint7` and `wdl` are independent projections, so their
    #: implied win probabilities can contradict each other, while this one is
    #: `P(outcome) * P(type | outcome)` and marginalises to `hier_wdl` exactly.
    #: A caller that needs the victory split and the win probability to be the
    #: same claim should read this pair; one that wants the historical numbers
    #: should read `joint7` and `wdl`.
    hier_joint7: np.ndarray | None = None
    hier_wdl: np.ndarray | None = None


class Evaluator:
    """Synchronous batched evaluator. Thread-safety and cross-caller
    coalescing arrive with the Phase F service; the API does not change."""

    #: Which head produces the W/D/L the SCALAR search value is read from.
    #:
    #: `"flat"` is the historical `value` head. `"hierarchical"` is W4's outcome
    #: factor, which is constrained to agree with its own victory-type split --
    #: the hypothesis being that a marginal forced into consistency is better
    #: calibrated than a free one. This governs `wdl` ONLY, so `joint7` still
    #: comes from the flat head; search does not read `joint7`, and leaving it
    #: alone keeps the arm to the one variable it is testing.
    #:
    #: A real strength arm, unlike the head's mere presence: every leaf value in
    #: every search changes.
    VALUE_SOURCES = ("flat", "hierarchical")

    def __init__(
        self,
        model,
        device: str = "cpu",
        max_batch: int = 512,
        fuse_embedder: bool = True,
        precision: str = "fp32",
        value_source: str = "flat",
    ):
        if value_source not in self.VALUE_SOURCES:
            raise ValueError(
                f"value_source must be one of {self.VALUE_SOURCES}, got {value_source!r}"
            )
        if value_source == "hierarchical" and not getattr(
            getattr(model, "_orig_mod", model), "hierarchical_value", False
        ):
            # Loud here rather than a KeyError at the first forward, or -- worse
            # -- a silent fall back to the flat head, which would report an arm
            # that never ran.
            raise ValueError(
                "value_source='hierarchical' needs a model built with "
                "hierarchical_value=True; this one has no such head"
            )
        self.value_source = value_source
        if precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = max_batch
        self.precision = precision
        # Fuse the token embedder's per-type loop where it is measured to pay --
        # CUDA only; on CPU there is no launch overhead to recover and the extra
        # arithmetic costs ~10% at width (`net.fusion_is_profitable`). Worth 1.5x
        # end to end on the production path (THROUGHPUT_ACTION_PLAN.md Phase 3b).
        # Must come after `.to()`, which invalidates the cache by design.
        self.fused_embedder = bool(fuse_embedder) and fuse_for_inference(self.model)

    def autocast(self):
        """Context for every forward through this evaluator's model.

        The production speedup is CUDA bf16. CPU stays in fp32: autocast is not
        faster for this model there, and keeping it out makes ``precision`` a
        performance choice rather than an avoidable CPU numerical change.
        """

        if self.precision == "bf16" and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

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
                [vectorize(e) for e in chunk],
                list(legals),
                self.device,
                contextual_actions=bool(
                    getattr(
                        getattr(self.model, "_orig_mod", self.model),
                        "action_residual",
                        False,
                    )
                ),
            )
            with self.autocast():
                outputs = self.model(batch)
            log_policy = masked_policy_log_softmax(
                outputs["policy"].float(), batch["legal_mask"]
            )
            policy = log_policy.exp().cpu().numpy()
            wdl = self.wdl_tensor(outputs).cpu().numpy()
            joint7 = (
                torch.softmax(outputs["joint7"].float(), dim=-1).cpu().numpy()
            )
            margin = outputs["margin"].float().cpu().numpy()
            military = outputs["military"].float().cpu().numpy()
            science = outputs["science"].float().cpu().numpy()
            # Already log-probabilities out of the head, so `exp`, not softmax.
            # Softmaxing them again would renormalise a normalised vector --
            # silently flattening it rather than failing.
            hier_joint7 = (
                outputs["hier_joint7"].float().exp().cpu().numpy()
                if "hier_joint7" in outputs
                else None
            )
            hier_wdl = (
                outputs["hier_value"].float().exp().cpu().numpy()
                if "hier_value" in outputs
                else None
            )
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
                        hier_joint7=(
                            None if hier_joint7 is None
                            else hier_joint7[row].astype(np.float32)
                        ),
                        hier_wdl=(
                            None if hier_wdl is None
                            else hier_wdl[row].astype(np.float32)
                        ),
                    )
                )
        return results

    def wdl_tensor(self, outputs: dict) -> "torch.Tensor":
        """The W/D/L this evaluator serves, per `value_source`.

        One place, because the scalar search value is derived from it in three
        of them -- here, the flat Rust batch path, and the scalar adapters --
        and an arm that switched only some would be measuring a mixture.
        """

        if self.value_source == "hierarchical":
            # Already log-probabilities; `exp`, never a second softmax.
            return outputs["hier_value"].float().exp()
        return torch.softmax(outputs["value"].float(), dim=-1)

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
