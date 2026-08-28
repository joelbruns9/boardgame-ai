"""Durable S2 opponent-league selection and integrity gates."""

from __future__ import annotations

import json

import pytest

from games.welcome_to import s2_league


def _checkpoint(path, value: int):
    path.write_bytes(bytes([value]) * (value + 1))
    return path


def test_league_keeps_all_entries_but_selects_a_bounded_pool(tmp_path):
    league = s2_league.S2League(tmp_path / "league.json")
    entries = [
        league.register(
            _checkpoint(tmp_path / f"archive_{iteration}.pt", iteration),
            archived_at_iteration=iteration,
        )
        for iteration in range(1, 7)
    ]
    assert len(league.entries()) == 6
    assert league.register(
        tmp_path / "archive_6.pt", archived_at_iteration=99
    ) == entries[-1]
    assert len(league.entries()) == 6

    current = _checkpoint(tmp_path / "current_best.pt", 20)
    config = s2_league.LeagueConfig(
        current_best_weight=0.6,
        recent_weight=0.3,
        hall_of_fame_weight=0.1,
        recent_count=2,
        hall_of_fame_count=1,
        seed=17,
    )
    first = league.select(current, iteration=8, config=config)
    second = league.select(current, iteration=8, config=config)
    assert first == second
    assert len(first.opponents) == 4  # current + two recent + one older HOF
    assert first.opponents[0].kind == "current_best"
    recent = [item for item in first.opponents if item.kind == "recent"]
    assert [item.archived_at_iteration for item in recent] == [5, 6]
    assert recent[1].weight > recent[0].weight
    assert len([item for item in first.opponents if item.kind == "hall_of_fame"]) == 1
    assert sum(item.weight for item in first.opponents) == pytest.approx(1.0)


def test_empty_league_is_current_best_only(tmp_path):
    current = _checkpoint(tmp_path / "current.pt", 9)
    selection = s2_league.S2League(tmp_path / "missing.json").select(
        current, iteration=0
    )
    assert len(selection.opponents) == 1
    assert selection.opponents[0].kind == "current_best"
    assert selection.metrics()["opponents"][0]["normalized_weight"] == 1.0


def test_history_ramps_in_across_early_promotions(tmp_path):
    current = _checkpoint(tmp_path / "current.pt", 12)
    league = s2_league.S2League(tmp_path / "league.json")
    league.register(
        _checkpoint(tmp_path / "first_best.pt", 2), archived_at_iteration=1
    )
    selection = league.select(current, iteration=2)
    by_kind = {item.kind: item.weight for item in selection.opponents}
    assert by_kind["current_best"] == pytest.approx(0.9)
    assert by_kind["recent"] == pytest.approx(0.1)


def test_league_refuses_a_changed_archive(tmp_path):
    archive = _checkpoint(tmp_path / "archive.pt", 3)
    league = s2_league.S2League(tmp_path / "league.json")
    league.register(archive, archived_at_iteration=1)
    archive.write_bytes(b"changed")
    current = _checkpoint(tmp_path / "current.pt", 8)
    with pytest.raises(ValueError, match="missing or changed"):
        league.select(current, iteration=2)


def test_manifest_is_explicit_and_strict(tmp_path):
    path = tmp_path / "league.json"
    path.write_text(json.dumps({"format": "wrong", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        s2_league.S2League(path).entries()
