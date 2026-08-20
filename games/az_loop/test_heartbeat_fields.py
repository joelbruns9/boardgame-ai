"""What the heartbeat line carries.

This is the line someone reconnecting over SSH greps for, so a number that is
collected but not printed is a number nobody reads. Two were: GPU utilisation
(sampled into `resources` all along) and the endgame solver's activity.
"""

from __future__ import annotations

import pytest

from .run_controller import RunController


def _line(row: dict) -> str:
    return RunController.heartbeat_line(row)


def _row(**overrides) -> dict:
    row = {
        "iteration": 7,
        "stats": {"generation": {"games_per_second": 1.5}, "resources": {}},
    }
    row.update(overrides)
    return row


def test_gpu_utilisation_is_printed_when_sampled():
    """The number that says whether generation is GPU-bound or core-bound."""

    row = _row()
    row["stats"]["resources"]["gpu_utilization_percent"] = 28.5
    assert "gpu=28%" in _line(row)


def test_a_box_without_a_gpu_reading_prints_no_field():
    """Absent is not zero: printing gpu=0% would read as an idle GPU rather
    than an unmeasured one."""

    assert "gpu=" not in _line(_row())


def test_the_solver_line_reports_masked_against_attempted():
    row = _row()
    row["generation_performance"] = {
        "summary": {"solver": {"attempted": 100, "masked": 92, "stops": {"nodes": 8}}}
    }
    assert "solved=92/100" in _line(row)
    assert "DEADLINE" not in _line(row)


def test_a_deadline_decline_is_shouted_about():
    """A deadline stop makes which positions got a proof depend on machine load,
    so the buffer stops being a function of its seeds. It should be impossible
    to miss on a reconnect."""

    row = _row()
    row["generation_performance"] = {
        "summary": {
            "solver": {"attempted": 100, "masked": 80, "stops": {"deadline": 12}}
        }
    }
    assert "DEADLINE=12" in _line(row)


def test_a_game_without_a_solver_prints_nothing_about_one():
    """Game-agnostic: Kingdomino and 7WD share this controller, and a game with
    no solver block must not print zeros for it."""

    assert "solved=" not in _line(_row())
    row = _row()
    row["generation_performance"] = {"summary": {"solver": {"attempted": 0}}}
    assert "solved=" not in _line(row)
