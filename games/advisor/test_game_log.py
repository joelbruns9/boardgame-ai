"""Gates for passive BGA game-log capture and reload.

Exercises `GameLogWriter` directly rather than through HTTP: the behaviour that
matters (validate, dedupe, append, reload) is transport-independent, and the
host's endpoint is a four-line wrapper over it.
"""

from __future__ import annotations

import json

import pytest

from games.advisor.game_log import (
    GameLogWriter,
    UnloggableState,
    load_positions,
    log_dir_for,
)
from games.seven_wonders_duel.advisor_adapter import SevenWondersAdvisor
from games.seven_wonders_duel.advisor_scrape import observation_to_wire
from games.seven_wonders_duel.game import Phase, new_game


def _wire(seed: int = 5) -> dict:
    """A real 7WD position on the scrape wire the advisor already accepts."""

    game = new_game(seed)
    while game.phase is Phase.WONDER_DRAFT:
        game.pick_wonder(game.legal_wonder_choices()[0])
    return {"observation": observation_to_wire(game.observation(game.active_player))}


@pytest.fixture()
def writer(tmp_path):
    return GameLogWriter(tmp_path), tmp_path


@pytest.fixture()
def adapter():
    return SevenWondersAdvisor()


def test_logged_position_is_written_and_reloads(writer, adapter):
    log, log_dir = writer
    result = log.append(
        adapter, table_id="899263451", state=_wire(), extra={"move": 12}
    )
    assert result["appended"] is True

    path = log_dir / "table_899263451.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["table_id"] == "899263451"
    assert row["extra"]["move"] == 12

    positions, report = load_positions(adapter, log_dir)
    assert report.loaded == 1
    assert positions[0].table_id == "899263451"
    # The whole point: it comes back as a position, not just bytes.
    assert positions[0].state.game.phase is Phase.PLAY_AGE


def test_identical_repost_is_deduped(writer, adapter):
    log, log_dir = writer
    wire = _wire()
    assert log.append(adapter, table_id="t1", state=wire)["appended"] is True
    # The extension re-posts while a search streams; the log must not grow.
    assert log.append(adapter, table_id="t1", state=wire)["appended"] is False
    lines = (log_dir / "table_t1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_a_different_position_appends(writer, adapter):
    log, log_dir = writer
    assert log.append(adapter, table_id="t1", state=_wire(5))["appended"] is True
    assert log.append(adapter, table_id="t1", state=_wire(6))["appended"] is True
    lines = (log_dir / "table_t1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_unloadable_state_is_refused_not_written(writer, adapter):
    log, log_dir = writer
    # Writing a row that cannot be rebuilt only defers the failure to whatever
    # consumes the log, by which time the game is long over.
    with pytest.raises(UnloggableState):
        log.append(adapter, table_id="t1", state={"nonsense": True})
    assert not list(log_dir.glob("*.jsonl"))


def test_final_record_needs_no_state(writer, adapter):
    log, log_dir = writer
    assert log.append(
        adapter, table_id="t1", kind="final", extra={"winner": 0}
    )["appended"] is True
    positions, report = load_positions(adapter, log_dir)
    # Default load keeps decision points only, so a terminal row is not a
    # restart candidate -- but it is still on disk.
    assert positions == [] and report.lines == 1


def test_stateless_rows_dedupe_on_their_own_content(writer, adapter):
    """Keying a stateless row on ``state`` made every one a duplicate of the last.

    Notification-packet batches carry their content in ``extra``; deduping them
    against ``state=None`` meant only the first batch of a game was ever
    written, and the loss was invisible until a replay came up short.
    """

    log, log_dir = writer
    first = log.append(adapter, table_id="t1", kind="packets", extra={"packets": [1]})
    second = log.append(adapter, table_id="t1", kind="packets", extra={"packets": [2]})
    repeat = log.append(adapter, table_id="t1", kind="packets", extra={"packets": [2]})
    assert [first["appended"], second["appended"], repeat["appended"]] == [
        True,
        True,
        False,
    ]
    lines = (log_dir / "table_t1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_positions_still_dedupe_when_extra_changes(writer, adapter):
    """``extra`` varies between re-posts of one board, so it must not key them."""

    log, log_dir = writer
    wire = _wire()
    assert log.append(adapter, table_id="t1", state=wire, extra={"ms": 10})["appended"]
    assert not log.append(adapter, table_id="t1", state=wire, extra={"ms": 900})[
        "appended"
    ]


def test_table_id_cannot_escape_the_log_directory(writer, adapter):
    log, log_dir = writer
    log.append(adapter, table_id="../../etc/passwd", state=_wire())
    written = list(log_dir.glob("*.jsonl"))
    assert len(written) == 1
    assert written[0].name == "table_etcpasswd.jsonl"


def test_load_reports_skips_instead_of_raising(writer, adapter):
    log, log_dir = writer
    log.append(adapter, table_id="t1", state=_wire())
    # A live log's last line can be a partial write.
    with (log_dir / "table_t1.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "decision", "state": {"trunc')
    positions, report = load_positions(adapter, log_dir)
    assert report.loaded == 1 and len(positions) == 1


def test_dedup_is_per_table(writer, adapter):
    log, log_dir = writer
    wire = _wire()
    assert log.append(adapter, table_id="t1", state=wire)["appended"] is True
    # Two tables can legitimately sit on the same position (same opening).
    assert log.append(adapter, table_id="t2", state=wire)["appended"] is True


def test_default_log_dir_is_per_game():
    assert log_dir_for("seven_wonders_duel").as_posix().endswith(
        "runs/seven_wonders_duel/bga_game_log"
    )
