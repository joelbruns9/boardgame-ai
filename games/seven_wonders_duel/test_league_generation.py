"""HOF league play in generation (7WD cloud plan W1.3).

`test_league_routing.py` proves the Rust routing is searcher-owned. This file
covers the loop's side: when league play starts, which games it touches, that the
draw is resume-stable, that the archive's policy targets are withheld, and that
the opponent is identified per game so W3 can split outcomes by it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from games.seven_wonders_duel.phase_d import (
    LeagueAssignment,
    PhaseDConfig,
    PhaseDLoop,
    _tag_league_opponents,
)
from games.seven_wonders_duel.train import build_model, make_checkpoint


def make_loop(tmp_path: Path, **overrides) -> PhaseDLoop:
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        seed_games=0,
        d_model=32,
        layers=1,
        device="cpu",
        **overrides,
    )
    return PhaseDLoop(config)


def write_iterations(loop: PhaseDLoop, count: int, games: int) -> None:
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(count):
        path = loop.buffer_dir / f"iter_{iteration:04d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(games):
                handle.write(json.dumps({"game": index}) + "\n")


def add_archive(loop: PhaseDLoop, iteration: int) -> None:
    """Put one real checkpoint in the HOF so sampling has something to draw."""

    loop.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = loop.checkpoint_dir / f"archive_{iteration:04d}.pt"
    torch.manual_seed(iteration + 1)
    model = build_model("transformer", 32, 1, None)
    torch.save(
        make_checkpoint(
            model,
            {
                "model": "transformer",
                "d_model": 32,
                "layers": 1,
                "heads": 4,
                "iteration": iteration,
            },
        ),
        path,
    )
    loop.hof.add(path, iteration=iteration, tag="promoted")


# -- when league play starts ----------------------------------------------


def test_off_by_default(tmp_path: Path):
    loop = make_loop(tmp_path)
    write_iterations(loop, 40, 500)
    add_archive(loop, 5)
    assert loop.config.hof_opponent_fraction == 0.0
    assert loop.league_assignment(40, 500) is None


def test_nothing_before_the_start_games_threshold(tmp_path: Path):
    loop = make_loop(
        tmp_path,
        games_per_iteration=500,
        hof_opponent_fraction=0.15,
        hof_start_games=10_000,
    )
    write_iterations(loop, 10, 500)  # 5,000 games so far
    add_archive(loop, 5)
    assert loop.generation_clock(10) == 5_000
    assert loop.league_assignment(10, 500) is None


def test_league_play_begins_once_the_threshold_is_passed(tmp_path: Path):
    loop = make_loop(
        tmp_path,
        games_per_iteration=500,
        hof_opponent_fraction=0.15,
        hof_start_games=10_000,
    )
    write_iterations(loop, 20, 500)  # 10,000 games
    add_archive(loop, 5)
    assignment = loop.league_assignment(20, 500)
    assert assignment is not None
    assert assignment.games == 75  # 15% of 500


def test_an_empty_archive_is_not_an_error(tmp_path: Path):
    """A run can pass the threshold before anything has been promoted."""

    loop = make_loop(
        tmp_path, games_per_iteration=500, hof_opponent_fraction=0.15,
        hof_start_games=0,
    )
    write_iterations(loop, 20, 500)
    assert loop.league_assignment(20, 500) is None


def test_a_fraction_too_small_to_fill_one_game_is_not_league_play(tmp_path: Path):
    loop = make_loop(
        tmp_path, games_per_iteration=8, hof_opponent_fraction=0.01,
        hof_start_games=0,
    )
    write_iterations(loop, 2, 8)
    add_archive(loop, 1)
    assert loop.league_assignment(2, 8) is None


# -- the assignment itself -------------------------------------------------


def test_the_archive_plays_both_seats_within_one_iteration(tmp_path: Path):
    """7WD has a first-player asymmetry, so a fixed seat would bias the games."""

    loop = make_loop(
        tmp_path, games_per_iteration=100, hof_opponent_fraction=0.20,
        hof_start_games=0,
    )
    write_iterations(loop, 2, 100)
    add_archive(loop, 1)
    assignment = loop.league_assignment(2, 100)
    assert assignment.games == 20
    on_p0 = sum(assignment.nets_p0)
    on_p1 = sum(assignment.nets_p1)
    assert on_p0 + on_p1 == 20
    assert abs(on_p0 - on_p1) <= 1


def test_league_games_are_spread_not_a_prefix(tmp_path: Path):
    """`first_player` is `(index // 2) % 2`, so a prefix correlates with seating."""

    loop = make_loop(
        tmp_path, games_per_iteration=100, hof_opponent_fraction=0.10,
        hof_start_games=0,
    )
    write_iterations(loop, 2, 100)
    add_archive(loop, 1)
    assignment = loop.league_assignment(2, 100)
    chosen = [
        index
        for index, (a, b) in enumerate(zip(assignment.nets_p0, assignment.nets_p1))
        if a or b
    ]
    assert len(chosen) == 10
    assert max(chosen) > 50, "league games must reach the back of the iteration"
    assert min(chosen) < 50


def test_no_game_has_the_archive_on_both_seats(tmp_path: Path):
    loop = make_loop(
        tmp_path, games_per_iteration=50, hof_opponent_fraction=1.0,
        hof_start_games=0,
    )
    write_iterations(loop, 2, 50)
    add_archive(loop, 1)
    assignment = loop.league_assignment(2, 50)
    assert assignment.games == 50
    assert not any(
        a and b for a, b in zip(assignment.nets_p0, assignment.nets_p1)
    )


def test_the_draw_is_deterministic_and_resume_stable(tmp_path: Path):
    """Re-running an iteration must draw the same opponent for the same games."""

    def assignment_for(path: Path):
        loop = make_loop(
            path, games_per_iteration=100, hof_opponent_fraction=0.20,
            hof_start_games=0, hof_sampling_mode="uniform",
        )
        write_iterations(loop, 3, 100)
        for iteration in (1, 2, 3, 4, 5):
            add_archive(loop, iteration)
        return loop.league_assignment(3, 100)

    first = assignment_for(tmp_path / "a")
    second = assignment_for(tmp_path / "b")
    assert first.sha256 == second.sha256
    assert first.nets_p0 == second.nets_p0
    assert first.nets_p1 == second.nets_p1


def test_different_iterations_can_draw_different_opponents(tmp_path: Path):
    loop = make_loop(
        tmp_path, games_per_iteration=100, hof_opponent_fraction=0.20,
        hof_start_games=0, hof_sampling_mode="uniform",
    )
    write_iterations(loop, 30, 100)
    for iteration in range(1, 9):
        add_archive(loop, iteration)
    drawn = {
        loop.league_assignment(iteration, 100).sha256
        for iteration in range(5, 30)
    }
    assert len(drawn) > 1, "every iteration drew the same archive"


# -- record tagging --------------------------------------------------------


def test_tagging_names_the_opponent_only_in_league_games():
    class Record:
        """Minimal stand-in: tagging touches `agents` and nothing else."""

        def __init__(self):
            self.agents = {"p0": "network", "p1": "network", "kind": "self_play"}

    from dataclasses import dataclass, field, replace as dc_replace

    @dataclass(frozen=True)
    class FakeRecord:
        agents: dict

    league = LeagueAssignment(
        checkpoint="runs/x/hof/iter_0007_promoted_abc.pt",
        sha256="abc123def456789",
        iteration_added=7,
        nets_p0=(0, 1, 0),
        nets_p1=(0, 0, 1),
    )
    records = [
        FakeRecord(agents={"p0": "network", "p1": "network", "kind": "self_play"})
        for _ in range(3)
    ]
    tagged = _tag_league_opponents(records, league)

    # Game 0 is pure self-play and untouched.
    assert tagged[0].agents["kind"] == "self_play"
    assert "opponent_source" not in tagged[0].agents
    # Game 1 has the archive on seat 0, game 2 on seat 1.
    assert tagged[1].agents["p0"] == league.name
    assert tagged[1].agents["p1"] == "network"
    assert tagged[2].agents["p1"] == league.name
    assert tagged[2].agents["p0"] == "network"
    for index in (1, 2):
        assert tagged[index].agents["kind"] == "league"
        assert tagged[index].agents["opponent_source"] == league.checkpoint


def test_tagging_refuses_a_length_mismatch():
    """A silent misalignment would attribute games to the wrong opponent."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FakeRecord:
        agents: dict

    league = LeagueAssignment(
        checkpoint="x.pt",
        sha256="abc",
        iteration_added=1,
        nets_p0=(0, 1),
        nets_p1=(0, 0),
    )
    with pytest.raises(ValueError, match="covers 2 games"):
        _tag_league_opponents([FakeRecord(agents={})], league)


def test_the_opponent_name_carries_iteration_and_hash():
    league = LeagueAssignment(
        checkpoint="x.pt",
        sha256="0123456789abcdef",
        iteration_added=42,
        nets_p0=(1,),
        nets_p1=(0,),
    )
    assert league.name == "hof_iter_0042_0123456789ab"
    assert league.archive_seat(0) == 0


# -- validation ------------------------------------------------------------


def test_negative_start_games_is_rejected():
    with pytest.raises(ValueError, match="hof_start_games"):
        PhaseDConfig(run_dir="x", hof_start_games=-1).validate()
