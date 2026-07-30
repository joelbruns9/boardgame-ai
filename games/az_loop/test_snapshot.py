"""W6.7: a snapshot must be resumable, and cheap to repeat."""

from __future__ import annotations

import json

import pytest

from .snapshot import (
    iteration_in_flight,
    resumable_set,
    snapshot,
    wait_for_boundary,
)


def _run(tmp_path, iterations=2, pending=False):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "buffers").mkdir()
    (run / "elo").mkdir()
    (run / "_recovery").mkdir()
    (run / "checkpoints" / "latest.pt").write_bytes(b"latest")
    (run / "checkpoints" / "current_best.pt").write_bytes(b"best")
    (run / "checkpoints" / "optimizer_0001.pt").write_bytes(b"moments")
    (run / "_recovery" / "latest.pt").write_bytes(b"stale rollback copy")
    for index in range(iterations):
        (run / "buffers" / f"iter_{index:04d}.jsonl").write_text("{}\n")
    (run / "training_log.jsonl").write_text("{}\n")
    (run / "heartbeat.log").write_text("HEARTBEAT iter=0001\n")
    (run / "elo" / "elo_games.jsonl").write_text("{}\n")
    (run / "run_manifest.json").write_text(
        json.dumps({"iterations": [{"iteration": i} for i in range(iterations)]})
    )
    if pending:
        (run / "checkpoints" / "pending_iteration.json").write_text("{}")
    return run


def test_the_manifest_is_copied_last(tmp_path):
    run = _run(tmp_path)
    order = resumable_set(run)
    assert order[-1].name == "run_manifest.json", (
        "a manifest newer than the rows it describes is the inconsistency this "
        "helper exists to prevent"
    )


def test_recovery_state_is_not_carried_to_the_destination(tmp_path):
    run = _run(tmp_path)
    selected = resumable_set(run)
    assert "pending_iteration.json" not in {path.name for path in selected}
    # Compare parts under the run dir: the tmp_path itself is named after this
    # test, so a substring check on the whole path matches its own name.
    assert not any(
        "_recovery" in path.relative_to(run).parts for path in selected
    )


def test_snapshot_copies_the_resumable_set(tmp_path):
    run = _run(tmp_path)
    report = snapshot(run, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "checkpoints" / "latest.pt").read_bytes() == b"latest"
    assert (out / "checkpoints" / "current_best.pt").read_bytes() == b"best"
    assert (out / "buffers" / "iter_0000.jsonl").is_file()
    assert (out / "run_manifest.json").is_file()
    assert report["last_iteration"] == 1
    assert report["files_copied"] == len(resumable_set(run))


def test_a_repeat_snapshot_only_copies_what_is_new(tmp_path):
    run = _run(tmp_path)
    snapshot(run, tmp_path / "out")
    (run / "buffers" / "iter_0002.jsonl").write_text("{}\n")
    report = snapshot(run, tmp_path / "out")
    # The new buffer plus the manifest, which is rewritten every iteration.
    assert report["files_copied"] <= 2
    assert report["files_already_current"] >= 5


def test_an_iteration_in_flight_blocks_the_snapshot(tmp_path):
    run = _run(tmp_path, pending=True)
    assert iteration_in_flight(run) is True
    assert wait_for_boundary(run, timeout=0.0, poll=0.01) is False
    with pytest.raises(TimeoutError, match="still in flight"):
        snapshot(run, tmp_path / "out", timeout=0.0)


def test_a_boundary_is_detected_once_the_pending_marker_clears(tmp_path):
    run = _run(tmp_path, pending=True)
    (run / "checkpoints" / "pending_iteration.json").unlink()
    assert iteration_in_flight(run) is False
    assert wait_for_boundary(run, timeout=0.0) is True


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        snapshot(tmp_path / "empty", tmp_path / "out")
