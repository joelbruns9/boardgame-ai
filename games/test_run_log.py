"""Tests for the game-agnostic run.log tee."""

from __future__ import annotations

import pytest

from games.az_loop import RunLog


def test_run_log_mirrors_output_and_writes_header_and_footer(tmp_path, capsys):
    log = tmp_path / "run.log"
    with RunLog(log, header={"Generator mode": "soft_gate"}) as run_log:
        print("progress line")
        run_log.completion_fields = {"Completed iterations": 2}

    console = capsys.readouterr().out
    text = log.read_text(encoding="utf-8")

    for stream in (console, text):
        assert "Run invocation started" in stream
        assert "Generator mode: soft_gate" in stream
        assert "progress line" in stream
        assert "Run completed" in stream
        assert "Completed iterations: 2" in stream
    # Normalized newlines: the file must not contain CR bytes.
    assert "\r" not in log.read_bytes().decode("utf-8")


def test_run_log_appends_on_resume_without_truncating(tmp_path):
    log = tmp_path / "run.log"
    with RunLog(log, header={"Resume iteration": "new run"}):
        print("first invocation body")
    with RunLog(log, header={"Resume iteration": "1"}):
        print("second invocation body")

    text = log.read_text(encoding="utf-8")
    assert text.count("Run invocation started") == 2
    assert "first invocation body" in text  # prior content preserved
    assert "second invocation body" in text
    assert "Resume iteration: 1" in text


def test_run_log_records_exception_and_reraises(tmp_path):
    log = tmp_path / "run.log"
    with pytest.raises(RuntimeError, match="boom"):
        with RunLog(log):
            print("did some work")
            raise RuntimeError("boom")

    text = log.read_text(encoding="utf-8")
    assert "did some work" in text
    assert "Run failed (RuntimeError)" in text
    assert "boom" in text  # the traceback is captured
    assert "Traceback (most recent call last)" in text


def test_run_log_records_keyboard_interrupt(tmp_path):
    log = tmp_path / "run.log"
    with pytest.raises(KeyboardInterrupt):
        with RunLog(log):
            raise KeyboardInterrupt
    assert "Run interrupted (KeyboardInterrupt)" in log.read_text(encoding="utf-8")


def test_disabled_run_log_is_a_console_only_no_op(tmp_path, capsys):
    log = tmp_path / "run.log"
    with RunLog(log, enabled=False, header={"Generator mode": "soft_gate"}):
        print("hello")
    assert not log.exists()
    console = capsys.readouterr().out
    assert "hello" in console
    assert "Run invocation started" not in console  # no header when disabled


def test_run_log_warns_and_continues_when_file_cannot_be_opened(tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    log = blocker / "run.log"  # parent is a regular file -> mkdir fails
    with RunLog(log):
        print("still runs")

    captured = capsys.readouterr()
    assert not log.exists()
    assert "could not open run log" in captured.err
    assert "still runs" in captured.out  # training output continues on console
