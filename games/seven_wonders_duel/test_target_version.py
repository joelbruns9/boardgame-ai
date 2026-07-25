"""The target-definition guard.

``schema``/``spec_version`` version the codec -- what a state and an action
mean.  Nothing versioned what the LABELS mean, so the 2026-07-25 sigma fix
changed every ``policy_target`` while leaving old records perfectly loadable.
These tests pin the boundary: reading stays permissive so stale buffers can be
replayed and re-derived, training refuses them.
"""

from __future__ import annotations

import json

import pytest

from .buffer import (
    SCHEMA_VERSION,
    SPEC_VERSION,
    TARGET_VERSION,
    GameRecord,
    StaleTargetsError,
    check_target_versions,
    from_json_line,
    target_version_census,
    to_json_line,
)


def _record(target_version: int = TARGET_VERSION) -> GameRecord:
    return GameRecord(
        seed=1,
        first_player=0,
        agents={"0": "a", "1": "b"},
        iteration=0,
        winner=0,
        victory_type="civilian",
        scores=(10, 5),
        chance_log=(),
        moves=(),
        final_digest="sha256:0",
        trajectory_digest="sha256:0",
        target_version=target_version,
    )


def test_new_records_carry_the_current_target_version():
    assert _record().target_version == TARGET_VERSION
    assert json.loads(to_json_line(_record()))["target_version"] == TARGET_VERSION


def test_roundtrip_preserves_target_version():
    assert from_json_line(to_json_line(_record(1))).target_version == 1


def test_records_predating_the_field_read_as_version_one():
    """Every buffer written before this guard holds definition-1 targets."""

    payload = json.loads(to_json_line(_record()))
    del payload["target_version"]
    assert from_json_line(json.dumps(payload)).target_version == 1


def test_reading_a_stale_record_does_not_raise():
    """Replay and reanalyze must still be able to open old buffers.

    The whole point of the replay invariant is that stale targets can be
    RE-derived rather than thrown away, which is impossible if the reader
    rejects them.
    """

    payload = json.loads(to_json_line(_record()))
    payload["target_version"] = 1
    record = from_json_line(json.dumps(payload))
    assert record.schema == SCHEMA_VERSION
    assert record.spec_version == SPEC_VERSION


def test_current_records_pass_the_check():
    census = check_target_versions([_record(), _record()], source="test")
    assert census == {TARGET_VERSION: 2}


def test_stale_records_are_refused():
    with pytest.raises(StaleTargetsError) as excinfo:
        check_target_versions([_record(1)], source="buffer_final.jsonl")
    message = str(excinfo.value)
    assert "buffer_final.jsonl" in message
    assert "target_version=1" in message


def test_a_single_stale_record_poisons_a_mostly_current_buffer():
    """The mix is the hazard, not the proportion.

    One stale record among thousands still trains the head against two
    objectives, and a proportion-based rule would invite a threshold nobody
    can justify.
    """

    records = [_record() for _ in range(999)] + [_record(1)]
    with pytest.raises(StaleTargetsError):
        check_target_versions(records, source="test")


def test_stale_records_pass_when_explicitly_allowed(capsys):
    census = check_target_versions([_record(1)], source="test", allow_stale=True)
    assert census == {1: 1}
    assert "WARNING" in capsys.readouterr().out


def test_census_reports_every_definition_present():
    records = [_record(), _record(1), _record(1)]
    assert target_version_census(records) == {TARGET_VERSION: 1, 1: 2}
