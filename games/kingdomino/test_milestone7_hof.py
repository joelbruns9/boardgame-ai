from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from games.kingdomino.hof import (
    add_hof_entry,
    load_hof_net,
    read_hof_index,
    sample_hof_entry,
)
from games.kingdomino.network import KingdominoNet
from games.kingdomino.self_play import (
    Example,
    ReplayBuffer,
    SelfPlayConfig,
    run_self_play_training,
    save_checkpoint,
)


def _tiny_example(*, trainable: bool, owner: str = "current") -> Example:
    return Example(
        np.zeros((9, 13, 13), dtype=np.float16),
        np.zeros((9, 13, 13), dtype=np.float16),
        np.zeros((261,), dtype=np.float16),
        np.array([0], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
        np.array([0], dtype=np.int32),
        0.0,
        10.0,
        8.0,
        1.0,
        owner=owner,
        trainable=trainable,
        game_type="current_vs_hof",
    )


def test_hof_entry_records_metadata_and_loads_own_architecture(tmp_path: Path) -> None:
    cfg = SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        device="cpu",
    )
    checkpoint = tmp_path / "current_best.pt"
    net = KingdominoNet(channels=8, blocks=1, bilinear_dim=8)
    save_checkpoint(str(checkpoint), net, cfg, 7, {"benchmark": []})

    entry = add_hof_entry(
        checkpoint,
        hof_dir=tmp_path / "hof",
        tag="seed_8x1",
        iteration=7,
        metadata={"why": "unit-test"},
    )
    entries = read_hof_index(tmp_path / "hof")
    assert len(entries) == 1
    assert entries[0].path == entry.path
    assert entries[0].channels == 8
    assert entries[0].blocks == 1
    assert Path(entry.path).exists()

    loaded = load_hof_net(entry.path, device="cpu")
    assert loaded.channels == 8
    assert loaded.blocks == 1


def test_hof_sampling_latest_and_replay_trainable_filter(tmp_path: Path) -> None:
    cfg = SelfPlayConfig(channels=8, blocks=1, bilinear_dim=8, device="cpu")
    net = KingdominoNet(channels=8, blocks=1, bilinear_dim=8)
    entries = []
    for i in range(2):
        checkpoint = tmp_path / f"best_{i}.pt"
        save_checkpoint(str(checkpoint), net, cfg, i, {"benchmark": []})
        entries.append(add_hof_entry(
            checkpoint, hof_dir=tmp_path / "hof", tag=f"seed_{i}", iteration=i))

    import random
    picked = sample_hof_entry(entries, rng=random.Random(0), weights="latest")
    assert picked == entries[-1]

    buffer = ReplayBuffer(capacity=10)
    buffer.add([
        _tiny_example(trainable=True, owner="current"),
        _tiny_example(trainable=False, owner="hof"),
    ])
    assert len(buffer) == 1
    assert buffer.data[0].owner == "current"


def test_hof_mixed_selfplay_smoke_logs_current_owned_samples(tmp_path: Path) -> None:
    cfg_for_ckpt = SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        device="cpu",
        exact_endgame_max_secs=0.0,
    )
    hof_source = tmp_path / "current_best.pt"
    net = KingdominoNet(channels=8, blocks=1, bilinear_dim=8)
    save_checkpoint(str(hof_source), net, cfg_for_ckpt, 1, {"benchmark": []})
    add_hof_entry(hof_source, hof_dir=tmp_path / "hof", tag="seed", iteration=1)

    run_dir = tmp_path / "run"
    cfg = SelfPlayConfig(
        channels=8,
        blocks=1,
        bilinear_dim=8,
        n_iterations=1,
        games_per_iteration=2,
        train_steps_per_iteration=0,
        min_buffer_to_train=999999,
        n_simulations=1,
        hof_sims=1,
        hof_fraction=0.5,
        hof_start_iter=1,
        engine="open_loop",
        device="cpu",
        exact_endgame_max_secs=0.0,
        benchmark_every=0,
        elo_every=0,
        checkpoint_dir=str(run_dir),
        log_path=str(run_dir / "training_log.jsonl"),
        hof_dir=str(tmp_path / "hof"),
    )
    result = run_self_play_training(cfg, verbose=False)
    row = json.loads((run_dir / "training_log.jsonl").read_text().splitlines()[-1])

    assert row["hof_games"] == 1
    assert row["hof_trainable_examples"] > 0
    assert row["hof_opponent"]
    assert any(ex.game_type in ("current_vs_hof", "hof_vs_current")
               for ex in result["buffer"].data)
    assert all(ex.trainable for ex in result["buffer"].data)
