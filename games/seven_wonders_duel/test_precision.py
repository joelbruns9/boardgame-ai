"""W0.3 gates for explicit per-path model-call precision."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from .buffer import StaleSpecVersionError, read_records
from .dataset import collate, examples_from_record
from . import f4_phase_d_ab
from .f4_cost_model import build_payload, collect_corpus
from .inference import Evaluator
from .net import masked_policy_log_softmax
from .phase_d import PhaseDConfig, PhaseDLoop
from .rust_bridge import rust_flat_batch_adapter
from .train import build_model, heads_from_config


def test_phase_d_ab_forwards_declared_precision(monkeypatch, tmp_path: Path) -> None:
    seen: list[tuple[bool, bool, str]] = []

    def fake_run_arm(
        _loop,
        _model,
        _iteration,
        _jobs,
        _destination,
        fused,
        gather,
        precision,
    ):
        seen.append((fused, gather, precision))
        stats = {
            "wall_seconds": 1.0,
            "games_per_second": 1.0,
            "moves_per_second": 1.0,
            "observed_forward_dtypes": [
                "torch.bfloat16" if precision == "bf16" else "torch.float32"
            ],
            "rust_games": 1,
            "rust_bot_games": 0,
            "rust_chunks": 1,
        }
        return stats, (precision,), []

    monkeypatch.setattr(f4_phase_d_ab, "run_arm", fake_run_arm)
    settings = {
        "fp32": (True, True, "fp32"),
        "bf16": (True, True, "bf16"),
    }
    results, _ = f4_phase_d_ab.run_measured_arms(
        object(),
        object(),
        0,
        [],
        tmp_path,
        ["fp32", "bf16"],
        settings,
    )
    assert seen == [settings["fp32"], settings["bf16"]]
    assert [row["precision"] for row in results] == ["fp32", "bf16"]


def test_phase_d_ab_rebuilds_the_checkpoint_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wide.pt"
    torch.save(
        {"config": {"d_model": 384, "layers": 8, "heads": 6}},
        checkpoint,
    )
    assert f4_phase_d_ab.checkpoint_architecture(checkpoint) == {
        "d_model": 384,
        "layers": 8,
        "heads": 6,
    }


def _capture_value_dtype(evaluator: Evaluator, invoke) -> torch.dtype:
    seen: list[torch.dtype] = []
    handle = evaluator.model.heads.value.register_forward_hook(
        lambda _module, _inputs, output: seen.append(output.dtype)
    )
    try:
        invoke()
    finally:
        handle.remove()
    assert seen
    return seen[-1]


def test_default_evaluator_precision_is_fp32_on_cpu() -> None:
    evaluator = Evaluator(build_model("transformer", 32, 1), device="cpu")
    payload = build_payload(collect_corpus(1, 20260728, stride=100)[:1])
    dtype = _capture_value_dtype(
        evaluator, lambda: rust_flat_batch_adapter(evaluator)(payload)
    )
    assert evaluator.precision == "fp32"
    assert dtype is torch.float32


def test_bf16_evaluator_stays_fp32_on_cpu() -> None:
    evaluator = Evaluator(
        build_model("transformer", 32, 1),
        device="cpu",
        precision="bf16",
    )
    payload = build_payload(collect_corpus(1, 20260729, stride=100)[:1])
    dtype = _capture_value_dtype(
        evaluator, lambda: rust_flat_batch_adapter(evaluator)(payload)
    )
    assert dtype is torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_evaluator_path_really_autocasts() -> None:
    evaluator = Evaluator(
        build_model("transformer", 32, 1),
        device="cuda",
        precision="bf16",
    )
    payload = build_payload(collect_corpus(1, 20260730, stride=100)[:1])
    dtype = _capture_value_dtype(
        evaluator, lambda: rust_flat_batch_adapter(evaluator)(payload)
    )
    assert dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_regular_evaluator_path_really_autocasts() -> None:
    from .codec import legal_action_indices
    from .encoder import encode
    from .game import new_game

    game = new_game(20260731)
    evaluator = Evaluator(
        build_model("transformer", 32, 1),
        device="cuda",
        precision="bf16",
    )
    dtype = _capture_value_dtype(
        evaluator,
        lambda: evaluator.evaluate(
            [encode(game.observation(game.active_player))],
            [legal_action_indices(game)],
        ),
    )
    assert dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_real_net_is_batch_invariant_before_trajectory_comparison() -> None:
    """W0.4: batch composition may move floats, not discrete root choices."""

    from .codec import legal_action_indices
    from .encoder import encode
    from .game import new_game

    games = [new_game(20260801 + seed) for seed in range(8)]
    encodings = [
        encode(game.observation(game.active_player)) for game in games
    ]
    legals = [legal_action_indices(game) for game in games]
    torch.manual_seed(20260801)
    evaluator = Evaluator(
        build_model("transformer", 32, 1),
        device="cuda",
        precision="bf16",
    )
    batched = evaluator.evaluate(encodings, legals)
    scalar = [
        evaluator.evaluate([encoding], [legal])[0]
        for encoding, legal in zip(encodings, legals)
    ]
    for together, alone in zip(batched, scalar):
        assert together.policy.argmax() == alone.policy.argmax()
        assert abs(float(together.wdl[0]) - float(alone.wdl[0])) < 1e-3


def test_invalid_precision_is_rejected() -> None:
    config = PhaseDConfig(run_dir="unused", precision="f16")
    with pytest.raises(ValueError, match="precision"):
        config.validate()
    with pytest.raises(ValueError, match="precision"):
        Evaluator(build_model("transformer", 32, 1), precision="f16")


def test_resume_refuses_changed_precision(tmp_path: Path) -> None:
    common = dict(
        run_dir=str(tmp_path / "run"),
        device="cpu",
        d_model=32,
        layers=1,
        seed_games=0,
        promotion_every=0,
    )
    PhaseDLoop(PhaseDConfig(**common, precision="fp32")).initialize()
    resumed = PhaseDLoop(PhaseDConfig(**common, precision="bf16"))
    with pytest.raises(ValueError, match="changed precision"):
        resumed.initialize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_real_position_fidelity() -> None:
    """Guard the measured bf16 fidelity on 512 real run-03 positions."""

    root = Path(__file__).parent / "runs" / "laptop_training_03"
    checkpoint_path = root / "checkpoints" / "current_best.pt"
    buffer_paths = sorted((root / "buffers").glob("iter_*.jsonl"), reverse=True)
    if not checkpoint_path.is_file() or not buffer_paths:
        pytest.skip("run-03 checkpoint/buffers are not available")

    examples = []
    try:
        for path in buffer_paths:
            for record in read_records(path):
                examples.extend(examples_from_record(record, record_fast_moves=False))
                if len(examples) >= 512:
                    break
            if len(examples) >= 512:
                break
    except StaleSpecVersionError:
        # run-03 was played by the pre-2026-08-03 engine, whose Age deal came
        # after the starter choice; its positions cannot be re-derived here.
        # The subject of this test is bf16 numerics, not engine semantics.
        pytest.skip("run-03 buffers predate the current codec spec")
    examples = examples[:512]
    assert len(examples) == 512

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    model = build_model(
        config.get("model", "transformer"),
        config.get("d_model", 128),
        config.get("layers", 4),
        heads_from_config(config),
    ).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    batch = collate(examples, "cuda")

    with torch.no_grad():
        fp32 = model(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = model(batch)

    fp32_policy = masked_policy_log_softmax(
        fp32["policy"], batch["legal_mask"]
    ).exp()
    bf16_policy = masked_policy_log_softmax(
        bf16["policy"], batch["legal_mask"]
    ).exp()
    fp32_value = torch.softmax(fp32["value"], dim=-1)
    bf16_value = torch.softmax(bf16["value"], dim=-1)

    agreement = float(
        (fp32_policy.argmax(-1) == bf16_policy.argmax(-1)).float().mean()
    )
    max_win_delta = float((fp32_value[:, 0] - bf16_value[:, 0]).abs().max())
    assert agreement >= 0.99
    assert max_win_delta < 1e-2
