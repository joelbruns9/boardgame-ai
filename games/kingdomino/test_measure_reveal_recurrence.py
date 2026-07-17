import json

import pytest

from games.kingdomino import measure_reveal_recurrence as measure


def test_measurement_counts_only_consecutive_own_decisions(tmp_path):
    path = tmp_path / "table_42.jsonl"
    records = []
    for second, deck, active in (
        (0, [1, 2], "me"),
        (1, [1, 2], "other"),
        (2, [1, 2], "me"),
        (4, [3], "me"),
        (7, [3], "me"),
    ):
        records.append({
            "kind": "decision",
            "table_id": "42",
            "viewer_id": "me",
            "active_player": active,
            "captured_at": f"2026-07-17T00:00:{second:02d}Z",
            "state": {"debug": {"deck": deck}, "deck_count": len(deck)},
        })
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    games, invalid = measure.load_own_decisions([path])
    result = measure.summarize_game("42", games["42"])

    assert invalid == 0
    assert result["decisions"] == 4
    assert result["windows"] == 2
    assert result["recurring_pairs"] == 2
    assert result["pairs"] == 3
    assert result["recurrence_rate"] == pytest.approx(2 / 3)
    assert result["decisions_per_window"] == 2
    assert result["all_gaps"] == [2.0, 2.0, 3.0]
