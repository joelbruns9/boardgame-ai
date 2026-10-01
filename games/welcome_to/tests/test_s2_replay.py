"""Durable multi-iteration replay-window and random-loader gates."""

from __future__ import annotations

import json

import pytest
import torch

from games.welcome_to import network as nw
from games.welcome_to import s2_replay
from games.welcome_to import self_play


pytest.importorskip("welcome_to_rust")

_SMALL = nw.NetConfig(
    sheet_hidden=16,
    sheet_out=8,
    trunk_hidden=24,
    trunk_blocks=1,
    head_hidden=16,
)


def _net() -> nw.WelcomeToNet:
    torch.manual_seed(310)
    return nw.WelcomeToNet(_SMALL).eval()


def _write_iteration(root, iteration: int, seed: int) -> None:
    directory = root / f"iter_{iteration:04d}"
    directory.mkdir()
    prefix = directory / "trajectories.jsonl"
    writer = self_play.wr.RustSampleShardWriter(
        prefix, shard_games=1, queue_games=1
    )
    try:
        trajectories, metrics = self_play.generate(
            _net(),
            config=self_play.SelfPlayConfig(
                games=2,
                inflight=2,
                max_batch=2,
                seed=seed,
            ),
            search_config=self_play.default_search_config(simulations=2),
            device="cpu",
            on_captured=lambda _trajectory, captured: writer.add(captured),
        )
    finally:
        writer.close()
    (directory / "trajectories.jsonl.metrics.json").write_text(
        json.dumps(
            {
                "total_games": len(trajectories),
                "searched_roots": int(metrics["searched_roots"]),
            }
        ),
        encoding="utf-8",
    )
    (directory / "trajectories.jsonl.manifest.json").write_text(
        json.dumps(
            {
                "format": "welcome_to_s2_generation",
                "version": 1,
                "trajectory_version": self_play.FORMAT_VERSION,
                "table_signature": int(self_play.wr.table_signature()),
            }
        ),
        encoding="utf-8",
    )


def test_replay_window_assembles_multiple_iterations_and_records_selection(tmp_path):
    _write_iteration(tmp_path, 1, 31_000)
    _write_iteration(tmp_path, 2, 32_000)
    corpus = s2_replay.assemble_replay(
        tmp_path,
        coefficient=100.0,
        exponent=0.6,
        cap_games=4,
        floor_games=1,
    )
    assert corpus.selection.iterations == (1, 2)
    assert corpus.selection.realised_games == 4
    assert len(corpus.trajectories) == 4
    assert corpus.positions == sum(
        len(game.searches) for game in corpus.trajectories
    )
    assert corpus.newest_positions == corpus.iterations[-1].positions

    loader = self_play.rust_training_loader(
        corpus.trajectories, batch_size=7, shuffle_seed=None
    )
    assert loader is not None
    loader.reset_random(123, 5)
    batches = list(self_play.iter_rust_training_batches(loader))
    assert len(batches) == 5
    assert all(batch["policy"].shape == (7, 684) for batch in batches)

    manifest = s2_replay.write_replay_manifest(tmp_path / "candidate.replay.json", corpus)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["window_iterations"] == 2
    assert saved["realized_window_games"] == 4


def test_replay_clock_survives_archiving_an_old_iteration(tmp_path):
    for iteration in range(1, 4):
        _write_iteration(tmp_path, iteration, 33_000 + iteration * 10)
    first = s2_replay.assemble_replay(
        tmp_path, coefficient=0.01, exponent=0.6, cap_games=10, floor_games=1
    )
    assert first.games_before_selection == 6
    assert (tmp_path / s2_replay.LEDGER_NAME).is_file()

    (tmp_path / "iter_0001").rename(tmp_path / "archived_iter_0001")
    second = s2_replay.assemble_replay(
        tmp_path, coefficient=0.01, exponent=0.6, cap_games=10, floor_games=1
    )
    assert second.games_before_selection == 6
    assert second.selection.iterations == (3,)


def test_incomplete_new_iteration_is_ignored_but_ledgered_damage_fails(tmp_path):
    _write_iteration(tmp_path, 1, 34_000)
    (tmp_path / "iter_0002").mkdir()
    corpus = s2_replay.assemble_replay(tmp_path, floor_games=1)
    assert corpus.selection.iterations == (1,)

    metrics = tmp_path / "iter_0001" / "trajectories.jsonl.metrics.json"
    metrics.rename(metrics.with_suffix(".missing"))
    with pytest.raises(ValueError, match="completed replay iteration 1 is now incomplete"):
        s2_replay.assemble_replay(tmp_path, floor_games=1)


def test_every_iteration_is_content_validated_before_it_drives_clock(tmp_path):
    _write_iteration(tmp_path, 1, 35_000)
    _write_iteration(tmp_path, 2, 36_000)
    metrics_path = tmp_path / "iter_0001" / "trajectories.jsonl.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["searched_roots"] += 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics say .* positions"):
        s2_replay.assemble_replay(
            tmp_path, coefficient=0.01, exponent=0.6, cap_games=10, floor_games=1
        )
