"""W0.2: the attention-head count is configurable, width-scaled, and persisted.

The hazard this file exists for: `nn.MultiheadAttention` parameter shapes
(`in_proj_weight` [3D, D], `out_proj` [D, D]) do not depend on the head count,
so a state dict from a 4-head model loads into a 2-head model reporting "All
keys matched successfully" and then computes something else. A head count can
therefore only be trusted if it travels with the weights.
"""

from __future__ import annotations

import pytest
import torch

from .net import LEGACY_HEADS, SWDNet, default_heads
from .train import build_model, heads_from_config, make_checkpoint


def _num_heads(model: SWDNet) -> int:
    """Head count actually in effect, read from the attention module."""

    return model.encoder.layers[0].self_attn.num_heads


def _batch(rows: int = 4, tokens: int = 16) -> dict[str, torch.Tensor]:
    return {
        "type_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "entity_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "aux_ids": torch.zeros(rows, tokens, dtype=torch.long),
        "features": torch.randn(rows, tokens, 130),
        "pad_mask": torch.zeros(rows, tokens, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "d_model,expected",
    [
        (32, 4),  # test-sized nets: the floor keeps them legal (32 // 64 == 0)
        (64, 4),
        (128, 4),  # the existing baseline stays 4-head, NOT 128 // 64 == 2
        (256, 4),
        (384, 6),  # above the floor the 64-dim ratio governs
        (512, 8),
        (768, 12),  # ZeusAI's geometry
    ],
)
def test_default_heads_scales_with_width(d_model: int, expected: int) -> None:
    assert default_heads(d_model) == expected
    assert d_model % default_heads(d_model) == 0


def test_default_heads_preserves_the_run03_baseline() -> None:
    """d_model 128 must keep the head count every existing checkpoint used.

    The sizing experiment compares wider arms against this baseline; silently
    turning it into a different 2-head model would make that comparison
    meaningless and would reinterpret run 03's trained weights.
    """

    assert default_heads(128) == LEGACY_HEADS


def test_explicit_heads_override_the_default() -> None:
    assert _num_heads(build_model("transformer", 256, 2, 8)) == 8
    assert _num_heads(build_model("transformer", 256, 2)) == default_heads(256)


def test_indivisible_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        SWDNet(d_model=128, layers=1, heads=7)
    with pytest.raises(ValueError, match="not divisible"):
        SWDNet(d_model=128, layers=1, heads=0)


def test_head_mismatch_loads_silently_and_changes_the_output() -> None:
    """Pin the hazard: this is why the count is persisted, not re-derived."""

    four = build_model("transformer", 128, 2, 4).eval()
    two = build_model("transformer", 128, 2, 2).eval()

    # No error and no missing/unexpected keys -- the corruption is invisible.
    result = two.load_state_dict(four.state_dict())
    assert not result.missing_keys and not result.unexpected_keys

    batch = _batch()
    with torch.no_grad():
        difference = (four(batch)["value"] - two(batch)["value"]).abs().max()
    assert difference > 1e-4, (
        "identical weights under different head counts must diverge; if this "
        "ever passes trivially the hazard model here needs revisiting"
    )


def test_checkpoint_round_trip_preserves_a_non_default_head_count() -> None:
    model = build_model("transformer", 384, 2, 12)  # 12, not the default 6
    checkpoint = make_checkpoint(
        model, {"model": "transformer", "d_model": 384, "layers": 2, "heads": 12}
    )

    rebuilt = build_model(
        "transformer", 384, 2, heads_from_config(checkpoint["config"])
    )
    rebuilt.load_state_dict(checkpoint["model_state"])

    assert _num_heads(rebuilt) == 12 == _num_heads(model)
    batch = _batch()
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(model(batch)["value"], rebuilt(batch)["value"])


def test_pre_heads_checkpoints_resolve_to_the_historical_count() -> None:
    """A config written before `heads` existed means 4, not the derived value.

    These disagree from d_model 384 up, so resolving a legacy checkpoint with
    `default_heads` would silently rebuild it wrong.
    """

    assert heads_from_config({"d_model": 384, "layers": 8}) == LEGACY_HEADS
    assert heads_from_config({"d_model": 384, "layers": 8, "heads": 6}) == 6
