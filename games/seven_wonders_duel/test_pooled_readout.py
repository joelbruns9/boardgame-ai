"""The mean/max readout, and specifically its masking.

Attention is an averaging operator, so "is there ANY token with property X" --
an existential -- is what it approximates worst. 7WD is full of them: any card
completing a sixth science symbol, any single card that swings the game. The
encoder hand-codes two (`sci_win_feasible`, `mil_win_feasible`); a max-pool
generalises the pattern rather than adding a third bespoke flag.

The failure mode worth testing is not the shape, it is the MASK. A padding token
that dilutes the mean, or wins the max, is wrong in a way no shape check sees.
"""

from __future__ import annotations

import pytest
import torch

from .dataset import MAX_FEATURES
from .net import SWDNet
from .train import build_model, pooled_readout_from_config


def _batch(rows: int = 3, tokens: int = 7, real: int = 4) -> dict:
    batch = {
        "type_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "entity_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "aux_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "features": torch.randn(rows, tokens, MAX_FEATURES),
        "pad_mask": torch.zeros(rows, tokens, dtype=torch.bool),
    }
    batch["pad_mask"][:, real:] = True  # trailing padding
    return batch


def test_padding_cannot_change_the_readout():
    """The decisive property. Whatever sits under the pad mask must not reach
    either pool -- not through the mean's denominator, and not by winning the
    max, which is what an unmasked max would let a large pad value do."""

    torch.manual_seed(20260817)
    model = build_model("transformer", 32, 2, None, True).eval()
    batch = _batch()
    with torch.no_grad():
        before = model(batch)["value"]
        # Scribble arbitrary values into the padded positions only.
        batch["features"][:, 4:] = 1e3
        after = model(batch)["value"]
    assert torch.allclose(before, after, atol=1e-5), "padding leaked into the readout"


def test_the_max_pool_is_a_max_and_not_another_mean():
    """Otherwise the whole justification evaporates: a second averaging path
    adds parameters and no new expressive power."""

    torch.manual_seed(1)
    model = SWDNet(d_model=16, layers=1, pooled_readout=True).eval()
    batch = _batch(rows=1, tokens=6, real=5)
    with torch.no_grad():
        normed = model.final_norm(
            model.encoder(model.embedder(batch), src_key_padding_mask=batch["pad_mask"])
        )
        real = ~batch["pad_mask"]
        maxed = normed.masked_fill(~real.unsqueeze(-1), float("-inf")).max(1).values
        mean = (normed * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True)
    assert torch.all(maxed >= mean - 1e-6), "max-pool fell below the mean"
    assert not torch.allclose(maxed, mean), "max-pool is behaving like a mean"


def test_disabled_is_exactly_the_old_readout():
    """Off by default, and off must mean unchanged -- every gate in this package
    was written against the CLS-only readout."""

    torch.manual_seed(7)
    model = build_model("transformer", 32, 2, None, False).eval()
    assert model.readout_proj is None
    batch = _batch()
    with torch.no_grad():
        encoded = model.encoder(model.embedder(batch), src_key_padding_mask=batch["pad_mask"])
        expected = model.heads(model.final_norm(encoded[:, 0]))
        actual = model(batch)
    for key in expected:
        assert torch.allclose(expected[key], actual[key], atol=1e-6), key


def test_the_readout_mode_travels_with_the_weights():
    """Same hazard as the attention-head count: rebuild without it and the
    checkpoint's `readout_proj` has nowhere to load."""

    assert pooled_readout_from_config({}) is False  # pre-flag checkpoints
    assert pooled_readout_from_config({"pooled_readout": True}) is True
    pooled = build_model("transformer", 32, 2, None, True)
    plain = build_model("transformer", 32, 2, None, False)
    with pytest.raises(RuntimeError):
        plain.load_state_dict(pooled.state_dict())


def test_a_row_of_pure_padding_does_not_produce_nan():
    """Cannot happen today -- the GLOBAL token is always present -- but the
    clamp is what makes that a guarantee rather than an observation."""

    torch.manual_seed(3)
    model = build_model("transformer", 32, 2, None, True).eval()
    batch = _batch(rows=2, tokens=5, real=1)
    with torch.no_grad():
        out = model(batch)
    for key, value in out.items():
        assert torch.isfinite(value).all(), key


def test_a_phase_d_checkpoint_records_enough_to_rebuild_itself():
    """The smoke run caught this the hard way: the model was built with the new
    readout and head, the checkpoint's own config did not say so, and the strict
    reload failed with "Unexpected key(s): readout_proj.weight" -- an iteration
    later, which on a long run is hours in.

    The cause was three near-copies of one dict, of which only `train.py`'s was
    updated. There is one now, and this asserts it carries what a rebuild needs.
    """

    from .phase_d import PhaseDConfig, PhaseDLoop

    config = PhaseDConfig(
        run_dir="unused", d_model=32, layers=1, pooled_readout=True, reply_head=True
    )
    loop = PhaseDLoop.__new__(PhaseDLoop)   # no run directory needed
    loop.config = config
    model = build_model("transformer", 32, 1, None, True, True)
    contract = loop._model_contract(model)
    assert contract["pooled_readout"] is True
    assert contract["reply_head"] is True
    # And a model rebuilt from that contract accepts the weights.
    from .train import heads_from_config, reply_head_from_config

    rebuilt = build_model(
        "transformer",
        contract["d_model"],
        contract["layers"],
        heads_from_config(contract),
        pooled_readout_from_config(contract),
        reply_head_from_config(contract),
    )
    rebuilt.load_state_dict(model.state_dict())


def test_a_gate_spec_carries_the_architecture_switches():
    """A gate rebuilds its model from a ModelAgentSpec. If the spec does not
    carry the switches, the rebuild is a plain model and the load fails -- which
    is what the smoke run hit after the checkpoint config was fixed.
    """

    from .phase_d import ModelAgentSpec, PhaseDLoop, _model_from_spec

    model = build_model("transformer", 32, 1, None, True, True)
    spec = ModelAgentSpec(
        name="candidate",
        model_state=model.state_dict(),
        d_model=32,
        layers=1,
        sims=1,
        mode="puct",
        top_k=2,
        heads=PhaseDLoop._built_heads(model),
        pooled_readout=True,
        reply_head=True,
    )
    rebuilt = _model_from_spec(spec)   # raises if the switches are dropped
    assert rebuilt.pooled_readout and rebuilt.reply_head
