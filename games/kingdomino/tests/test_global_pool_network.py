from __future__ import annotations

import torch

from games.kingdomino.encoder import CANVAS_SIZE, FLAT_SIZE, NUM_BOARD_CHANNELS
from games.kingdomino.network import GlobalPoolResBlock, KingdominoNet
from games.kingdomino.self_play import SelfPlayConfig, load_generator_net


def _inputs(batch: int = 2):
    torch.manual_seed(11)
    my = torch.randn(batch, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
    opp = torch.randn(batch, NUM_BOARD_CHANNELS, CANVAS_SIZE, CANVAS_SIZE)
    flat = torch.randn(batch, FLAT_SIZE)
    return my, opp, flat


def test_global_pooling_blocks_preserve_contract_and_start_as_local_blocks():
    torch.manual_seed(7)
    local = KingdominoNet(channels=16, blocks=2, bilinear_dim=8).eval()
    pooled = KingdominoNet(
        channels=16, blocks=2, bilinear_dim=8, global_pooling=True
    ).eval()
    missing, unexpected = pooled.load_state_dict(local.state_dict(), strict=False)

    assert not unexpected
    assert missing == [
        "res_blocks.0.global_bias.weight",
        "res_blocks.0.global_bias.bias",
        "res_blocks.1.global_bias.weight",
        "res_blocks.1.global_bias.bias",
    ]
    assert all(isinstance(block, GlobalPoolResBlock) for block in pooled.res_blocks)

    inputs = _inputs()
    with torch.no_grad():
        local_out = local(*inputs)
        pooled_out = pooled(*inputs)
    for left, right in zip(local_out, pooled_out):
        torch.testing.assert_close(left, right)


def test_global_pooling_projection_receives_gradient():
    net = KingdominoNet(
        channels=16, blocks=2, bilinear_dim=8, global_pooling=True
    )
    outputs = net(*_inputs())
    sum(tensor.float().mean() for tensor in outputs).backward()

    for block in net.res_blocks:
        assert block.global_bias.weight.grad is not None
        assert block.global_bias.weight.grad.abs().sum() > 0


def test_pooled_checkpoint_round_trips_through_self_play_loader(tmp_path):
    net = KingdominoNet(
        channels=16, blocks=2, bilinear_dim=8, global_pooling=True
    )
    path = tmp_path / "pooled.pt"
    torch.save(
        {
            "model_state": net.state_dict(),
            "config": {
                "channels": 16,
                "blocks": 2,
                "bilinear_dim": 8,
                "global_pooling": True,
            },
        },
        path,
    )
    cfg = SelfPlayConfig(
        channels=16,
        blocks=2,
        bilinear_dim=8,
        global_pooling=True,
        device="cpu",
    )
    loaded = load_generator_net(str(path), cfg)

    assert loaded.global_pooling is True
    assert all(isinstance(block, GlobalPoolResBlock) for block in loaded.res_blocks)
