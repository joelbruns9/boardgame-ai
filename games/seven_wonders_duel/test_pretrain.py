"""Final-run preparation: windowed pretrain and the G14 init arms."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("seven_wonders_rust")

from . import pretrain
from .buffer import append_records
from .phase_e import fresh_bot_records
from .train import build_model, make_checkpoint


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretrain")
    buffers = root / "buffers"
    buffers.mkdir()
    records = fresh_bot_records(8, seed=1212)
    append_records(buffers / "iter_0001.jsonl", records[:4])
    append_records(buffers / "iter_0002.jsonl", records[4:])
    base = root / "base.pt"
    torch.save(make_checkpoint(build_model("transformer", 32, 1),
                               {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2}),
               base)
    return root, buffers, base


def _args(root, buffers, base, init, out_name, extra=()):
    return pretrain.build_parser().parse_args([
        "--base", str(base), "--init", init, "--buffers-dir", str(buffers),
        "--iterations", "1-2", "--window", "1", "--steps-per-window", "2",
        "--presentations-per-row", "-1",
        "--validate-every", "1000", "--device", "cpu", "--precision", "fp32",
        "--out", str(root / out_name), *extra,
    ])


@pytest.mark.parametrize("init", pretrain.INITS)
def test_every_init_runs_window_by_window(setup, init):
    root, buffers, base = setup
    summary = pretrain.run(_args(root, buffers, base, init, f"out_{init}"))
    assert [w["window"] for w in summary["windows"]] == ["0001_0001", "0002_0002"]
    assert (root / f"out_{init}" / "pretrained.pt").is_file()
    assert all(w["rows"]["train"] > 0 for w in summary["windows"])


def test_inits_differ_where_they_should(setup):
    _root, _buffers, base = setup
    stored = torch.load(base, map_location="cpu", weights_only=False)["model_state"]
    kept, _ = pretrain.build_model(base, "checkpoint", 0)
    fresh, _ = pretrain.build_model(base, "random", 123)
    reset, _ = pretrain.build_model(base, "reset-value", 456)
    assert torch.equal(kept.state_dict()["heads.policy.weight"], stored["heads.policy.weight"])
    assert not torch.equal(fresh.state_dict()["heads.policy.weight"], stored["heads.policy.weight"])
    # reset-value: the trunk and policy kept, the value head redrawn.
    assert torch.equal(reset.state_dict()["heads.policy.weight"], stored["heads.policy.weight"])
    assert not torch.equal(reset.state_dict()["heads.value.weight"], stored["heads.value.weight"])


def test_resume_skips_completed_windows(setup):
    root, buffers, base = setup
    pretrain.run(_args(root, buffers, base, "checkpoint", "out_resume"))
    progress = json.loads((root / "out_resume" / "progress.json").read_text())
    assert len(progress["completed_windows"]) == 2
    summary = pretrain.run(_args(root, buffers, base, "checkpoint", "out_resume", ("--resume",)))
    assert len(summary["windows"]) == 2  # nothing re-trained or duplicated


def test_presentations_per_row_sets_the_step_count(setup):
    root, buffers, base = setup
    args = pretrain.build_parser().parse_args([
        "--base", str(base), "--buffers-dir", str(buffers), "--iterations", "1-1",
        "--window", "1", "--presentations-per-row", "2", "--validate-every", "1000",
        "--device", "cpu", "--precision", "fp32", "--out", str(root / "out_ppr"),
    ])
    window = pretrain.run(args)["windows"][0]
    expected = -(-2 * window["rows"]["train"] // 512)
    assert window["steps"] == expected
    assert window["presentations_per_row"] >= 2.0
