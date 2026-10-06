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


def test_every_arm_refits_w5_alpha_after_each_window(setup, monkeypatch):
    """Review of 8014a6c, #1: a random arm is built with W5's gate at zero and
    frozen; without the per-window fit it would serve W5-off for the whole
    comparison. Every arm fits, the jump is uncapped by default, and the fitted
    alpha is in the window checkpoint a resume starts from."""

    from .action_alpha import AlphaFit
    from .train import model_from_config

    root, buffers, _base = setup
    config = {"model": "transformer", "d_model": 32, "layers": 1, "heads": 2,
              "action_residual": True, "fit_action_alpha": True, "action_gate_max": 2.0}
    base = root / "w5_base.pt"
    torch.save(make_checkpoint(model_from_config(config), config), base)
    calls = []

    def fake_refit(model, examples, device, *, alpha_max, step, **_kw):
        calls.append((alpha_max, step))
        previous = model.action_scorer.alpha_value()
        applied = model.action_scorer.set_alpha(0.7)
        return AlphaFit(previous=previous, applied=applied, fitted=0.7, positions=len(examples),
                        loss_flat_only=1.0, loss_at_previous=1.0, loss_at_fitted=0.9,
                        loss_w5_only=1.1)

    monkeypatch.setattr(pretrain, "refit_alpha", fake_refit)
    for init in pretrain.INITS:
        calls.clear()
        out = f"out_alpha_{init}"
        summary = pretrain.run(_args(root, buffers, base, init, out))
        assert calls == [(2.0, 2.0)] * 2
        assert all(w["alpha_fit"]["applied"] == pytest.approx(0.7) for w in summary["windows"])
        for name in ("pretrained.pt", "window_0002_0002.pt"):
            saved = torch.load(root / out / name, map_location="cpu", weights_only=False)
            model = model_from_config(saved["config"])
            model.load_state_dict(saved["model_state"])
            assert model.action_scorer.alpha_value() == pytest.approx(0.7, abs=1e-5)


def test_grown_layers_start_as_an_exact_no_op_and_still_learn(setup):
    """Capacity probe: the grown net computes what the base did, and the new
    layers receive gradient (a fully zeroed block would not)."""

    from .dataset import collate, examples_from_record
    from .train import compute_losses, load_checkpoint, model_from_config

    _root, _buffers, base = setup
    stored = torch.load(base, map_location="cpu", weights_only=False)
    original = model_from_config(stored["config"])
    load_checkpoint(base, original, checkpoint=stored)
    grown, config = pretrain.build_model(base, "checkpoint", 7, grow_to=3)
    assert config["layers"] == 3 and len(grown.encoder.layers) == 3
    rows = examples_from_record(fresh_bot_records(1, seed=99)[0])[:16]
    batch = collate(rows)
    original.eval()
    grown.eval()
    with torch.no_grad():
        before, after = original(batch), grown(batch)
    for key, tensor in before.items():
        assert torch.allclose(tensor, after[key], atol=1e-5), key
    grown.train()
    loss, _ = compute_losses(grown(batch), batch)
    loss.backward()
    for name in ("encoder.layers.2.linear2.weight", "encoder.layers.2.self_attn.out_proj.weight"):
        grad = dict(grown.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    with pytest.raises(ValueError):
        pretrain.build_model(base, "random", 0, grow_to=3)


def test_a_grown_pretrain_saves_a_checkpoint_that_rebuilds(setup):
    from .train import model_from_config

    root, buffers, base = setup
    summary = pretrain.run(_args(root, buffers, base, "checkpoint", "out_grow",
                                 ("--grow-layers", "2")))
    assert summary["grow_layers"] == 2
    saved = torch.load(root / "out_grow" / "pretrained.pt", map_location="cpu",
                       weights_only=False)
    assert saved["config"]["layers"] == 2
    model_from_config(saved["config"]).load_state_dict(saved["model_state"])
    # And a resume rebuilds the same grown architecture.
    pretrain.run(_args(root, buffers, base, "checkpoint", "out_grow",
                       ("--grow-layers", "2", "--resume")))
