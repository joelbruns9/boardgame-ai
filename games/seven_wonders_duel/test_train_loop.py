"""`train_loop` must actually run.

It read `optimizer_name` as a module global while having no such parameter, so
every call raised `NameError` before the first batch. Nothing caught it because
nothing executed this entry point -- the self-play loop uses `train_steps`, and
the offline path is only reached by the Phase B gate and the `train.py` CLI.

These are smoke tests, not quality tests: they assert the function completes and
returns history, and that both optimizer names are wired. A one-line signature
regression should not be able to hide here again.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from .buffer import GameRecorder
from .codec import legal_action_indices
from .dataset import examples_from_record
from .game import Phase
from .net import SWDNet
from .train import train_loop


def _examples(seed: int = 91):
    recorder = GameRecorder(seed, agents={"p0": "test", "p1": "test"})
    rng = random.Random(seed)
    while recorder.game.phase is not Phase.COMPLETE:
        legal = legal_action_indices(recorder.game)
        choice = rng.choice(legal)
        recorder.play(choice, policy_target={choice: 1.0})
    return examples_from_record(recorder.finish())


@pytest.mark.parametrize("optimizer_name", ("adamw", "adam"))
def test_train_loop_runs(optimizer_name):
    """The regression: this raised NameError before reaching a single batch."""

    examples = _examples()
    assert len(examples) > 20
    model = SWDNet(d_model=32, layers=1)
    history = train_loop(
        model,
        examples[:24],
        examples[24:32],
        device="cpu",
        epochs=1,
        batch_size=8,
        optimizer_name=optimizer_name,
        log=lambda _m: None,
    )
    assert history, "train_loop returned no history"


def test_train_loop_rejects_an_unknown_optimizer():
    """The error path is reachable, rather than masked by the NameError."""

    examples = _examples()
    model = SWDNet(d_model=32, layers=1)
    with pytest.raises(ValueError, match="unknown optimizer"):
        train_loop(
            model, examples[:8], None, device="cpu", epochs=1, batch_size=8,
            optimizer_name="sgd", log=lambda _m: None,
        )


def test_train_loop_signature_has_no_free_globals():
    """Guards the whole class of bug: a body reading a name the signature does
    not define, which only fails when the function is actually called."""

    code = train_loop.__code__
    assert "optimizer_name" in code.co_varnames
    assert "optimizer_name" not in code.co_names
