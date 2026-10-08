"""The launcher's CUDA-graph readers (review of d8aa2e3, findings 1, 3, 4).

`testdata/graph_smoke_training_log.jsonl` is a REAL stage-9 smoke log: a local
`phase_d --plumbing-smoke --cuda-graphs` run on CUDA, soft-gate controller.
"""

import copy
import json
from pathlib import Path

import pytest

from . import graph_check
from .graph_check import FAILED, OK, UNAVAILABLE

SMOKE_LOG = Path(__file__).parent / "testdata" / "graph_smoke_training_log.jsonl"


@pytest.fixture
def real_row():
    return json.loads(SMOKE_LOG.read_text(encoding="utf-8").splitlines()[0])


def _with(row, iteration, **counters):
    row = copy.deepcopy(row)
    row["iteration"] = iteration
    row["generation_performance"]["performance"]["rust_boundary"].update(counters)
    return row


def _write(path, rows, tail=""):
    text = "".join(json.dumps(row) + "\n" for row in rows) + tail
    path.write_text(text, encoding="utf-8", newline="\n")


def test_the_real_smoke_row_is_read_at_its_controller_nesting(real_row):
    # Finding 1: the counters sit under generation_performance.performance.
    assert "rust_boundary" not in real_row["generation_performance"]
    counters = graph_check.boundary_counters(real_row)
    assert counters["graph_replays"] > 0
    assert graph_check.verdict(counters) == OK


def test_a_strict_gate_row_is_read_too(real_row):
    boundary = real_row["generation_performance"]["performance"]["rust_boundary"]
    flat = {"iteration": 0, "generation_performance": {"rust_boundary": boundary}}
    assert graph_check.boundary_counters(flat) == graph_check.boundary_counters(real_row)


def test_missing_counters_are_unavailable_not_a_failure():
    assert graph_check.verdict(graph_check.boundary_counters({"iteration": 0})) == UNAVAILABLE
    row = {"generation_performance": {"performance": {"rust_boundary": {"batches": 3}}}}
    assert graph_check.verdict(graph_check.boundary_counters(row)) == UNAVAILABLE


def test_the_real_smoke_passes(capsys):
    assert graph_check.smoke(SMOKE_LOG) == 0


def test_a_smoke_without_counters_does_not_pass(tmp_path):
    log = tmp_path / "training_log.jsonl"
    _write(log, [{"iteration": 0, "generation_performance": {"performance": {}}}])
    assert graph_check.smoke(log) == 1
    assert graph_check.smoke(tmp_path / "absent.jsonl") == 1


def test_a_smoke_with_a_capture_failure_does_not_pass(tmp_path, real_row):
    log = tmp_path / "training_log.jsonl"
    _write(log, [_with(real_row, 0, graph_capture_failures=1)])
    assert graph_check.smoke(log) == 1


class _Box:
    """Simulated time, process and signals: nothing real is signalled."""

    def __init__(self):
        self.now = 0.0
        self.killed = []
        self.on_sleep = []

    def run(self, log, guard="stop", offset=0, rows_wanted=3):
        return graph_check.watch(
            log, 4242, guard, offset, rows_wanted=rows_wanted,
            alive=lambda pid: not self.killed,
            terminate=self.killed.append,
            clock=lambda: self.now,
            sleep=self._sleep,
            deadline_seconds=3600,
        )

    def _sleep(self, seconds):
        self.now += seconds
        if self.on_sleep:
            self.on_sleep.pop(0)()


def test_a_healthy_run_is_never_stopped(tmp_path, real_row):
    # The reviewer's reproduction: 1,000 replays, no eager calls, no failures.
    log = tmp_path / "training_log.jsonl"
    _write(log, [_with(real_row, i, graph_replays=1000, graph_eager_calls=0) for i in range(3)])
    box = _Box()
    assert box.run(log) == 0
    assert box.killed == []


def test_rows_without_counters_never_stop_the_run(tmp_path):
    log = tmp_path / "training_log.jsonl"
    _write(log, [{"iteration": i, "generation_performance": {"performance": {}}} for i in range(3)])
    box = _Box()
    assert box.run(log) == 0
    assert box.killed == []


def test_a_confirmed_failure_stops_only_in_stop_mode(tmp_path, real_row):
    log = tmp_path / "training_log.jsonl"
    _write(log, [_with(real_row, 0, graph_replays=0, graph_eager_calls=50)])
    box = _Box()
    assert box.run(log, guard="warn", rows_wanted=1) == 1
    assert box.killed == []
    box = _Box()
    assert box.run(log, guard="stop", rows_wanted=1) == 1
    assert box.killed == [4242]


def test_a_resume_checks_the_new_rows_not_the_history(tmp_path, real_row):
    # Finding 3: three healthy historical rows, then a failing current one.
    log = tmp_path / "training_log.jsonl"
    history = [_with(real_row, i) for i in range(3)]
    _write(log, history)
    offset = log.stat().st_size
    _write(log, history + [_with(real_row, 3, graph_capture_failures=2)])
    box = _Box()
    assert box.run(log, offset=offset) == 1
    assert box.killed == [4242]


def test_historical_failures_do_not_stop_a_healthy_relaunch(tmp_path, real_row):
    log = tmp_path / "training_log.jsonl"
    history = [_with(real_row, i, graph_replays=0) for i in range(3)]
    _write(log, history)
    offset = log.stat().st_size
    # The startup backfill may append an old iteration after the offset.
    _write(log, history + [_with(real_row, 1, graph_replays=0)]
           + [_with(real_row, i) for i in range(3, 6)])
    box = _Box()
    assert box.run(log, offset=offset) == 0
    assert box.killed == []


def test_a_half_written_last_row_is_retried_not_fatal(tmp_path, real_row):
    # Finding 4: one complete row, then an incomplete append.
    log = tmp_path / "training_log.jsonl"
    rows = [_with(real_row, i) for i in range(3)]
    complete = json.dumps(rows[0]) + "\n"
    second = json.dumps(rows[1]) + "\n"
    log.write_text(complete + second[:40], encoding="utf-8", newline="\n")

    def finish():
        log.write_text(complete + second + json.dumps(rows[2]) + "\n",
                       encoding="utf-8", newline="\n")

    box = _Box()
    box.on_sleep = [finish]
    assert box.run(log) == 0
    assert box.killed == []


def test_a_malformed_complete_row_is_skipped_and_reported(tmp_path, real_row, capsys):
    log = tmp_path / "training_log.jsonl"
    good = [_with(real_row, i) for i in range(3)]
    text = json.dumps(good[0]) + "\n{not json\n" + "".join(json.dumps(r) + "\n" for r in good[1:])
    log.write_text(text, encoding="utf-8", newline="\n")
    box = _Box()
    assert box.run(log) == 0
    assert "malformed" in capsys.readouterr().out


def test_the_launcher_uses_this_module_for_both_checks():
    text = (Path(__file__).resolve().parents[2] / "setup_cloud_7wd.sh").read_text(encoding="utf-8")
    assert "graph_check smoke" in text
    assert "graph_check watch" in text
    assert 'get("rust_boundary")' not in text
    # The offset is taken BEFORE the launch, so the new process's rows follow it.
    assert text.index("GRAPH_LOG_OFFSET=") < text.index('common::launch_detached "$LOG_FILE"')
