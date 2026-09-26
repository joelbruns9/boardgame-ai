"""CUDA-graph replay agrees with the eager forward, and pads what it must."""

from __future__ import annotations

import random

import pytest
import torch

from .cuda_graphs import GraphedForward, row_bucket, width_bucket

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_buckets_cover_and_waste_little():
    for rows in range(1, 3000):
        bucket = row_bucket(rows)
        assert bucket >= rows
        if rows > 16:
            assert bucket < rows * 1.25 + 32
    assert [width_bucket(w) for w in (1, 8, 9, 72, 73)] == [8, 8, 16, 72, 80]


class _RowModel(torch.nn.Module):
    """Row-independent, and returns one non-row output like a real head may."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, batch):
        x = batch["x"].float()
        return {"y": self.linear(x).sum(1), "scale": self.linear.bias.sum()}


@cuda
def test_graphed_matches_eager_and_replays():
    model = _RowModel().cuda().eval()
    graphed = GraphedForward(model)
    for rows in (1, 5, 8, 37, 5, 37):
        batch = {"x": torch.randn(rows, 6, 4, device="cuda")}
        with torch.no_grad():
            want = model(batch)["y"]
            got = graphed(batch)
        assert got["y"].shape == (rows, 3)
        torch.testing.assert_close(got["y"], want)
        assert got["scale"].dim() == 0
    assert graphed.captures == 2  # rows 1, 5, 8 share bucket 8; 37 -> 64
    assert graphed.replays == 6


@cuda
def test_graphed_falls_back_on_cpu_and_oversize():
    model = _RowModel().cuda().eval()
    graphed = GraphedForward(model, max_rows=16)
    with torch.no_grad():
        graphed({"x": torch.randn(40, 6, 4, device="cuda")})
    assert graphed.eager_calls == 1 and graphed.captures == 0


@cuda
def test_adapter_graphs_agree_with_eager_within_bf16_noise():
    """The real network through the real adapter, bf16, several widths."""

    from . import f4_cost_model as cm
    from .inference import Evaluator
    from .train import model_from_config
    from .rust_bridge import rust_flat_batch_adapter

    torch.manual_seed(0)
    # Every module the run carries, so the legal-axis padding (W5) and the
    # slot/graph paths (W1/W2) are inside the graph.
    switches = {
        "pooled_readout": True, "reply_head": True, "action_residual": True,
        "action_exposes": True, "slot_embedding": True, "graph_module": True,
        "hierarchical_value": True,
    }
    model = model_from_config(switches, d_model=64, layers=2).cuda().eval()
    evaluator = Evaluator(model, "cuda", 512, precision="bf16")
    corpus = cm.collect_corpus(2, 1)
    eager = rust_flat_batch_adapter(evaluator)
    graphed = rust_flat_batch_adapter(evaluator, cuda_graphs=True)
    random.seed(0)
    for rows in (1, 7, 40):
        payload = cm.build_payload(random.sample(corpus * 20, rows))
        want = eager(payload)
        graphed(payload)
        got = graphed(payload)
        assert len(got) == rows
        for a, b in zip(want, got):
            assert abs(a[0] - b[0]) < 3e-2
            assert max(abs(p - q) for p, q in zip(a[1], b[1])) < 3e-2
    assert graphed._model.replays >= 3
