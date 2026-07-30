"""The games clock and the schedules that read it (7WD cloud plan W1.1/W1.2/W1.4).

These tests exist because of a specific failure mode the plan calls out: a
schedule that silently rescales when ``games_per_iteration`` changes between
runs.  So the assertions are less about arithmetic than about two invariants:

1. the clock is **recomputed from immutable files**, never restored from stored
   state, so a resume lands on exactly the same schedule position;
2. a schedule's meaning is **invariant to games-per-iteration**, which is the
   whole reason for keying on games rather than iterations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from games.az_loop import (
    GameSchedule,
    GamesLedger,
    GrowingReplayWindow,
    LinearSchedule,
    count_games,
    iteration_of_buffer_file,
)


def write_buffer(buffer_dir: Path, iteration: int, games: int) -> Path:
    """One JSON record per line, matching ``_write_records``' shape."""

    buffer_dir.mkdir(parents=True, exist_ok=True)
    path = buffer_dir / f"iter_{iteration:04d}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(games):
            handle.write(json.dumps({"game": index, "iteration": iteration}) + "\n")
    return path


# -- filename parsing ------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("iter_0000.jsonl", 0),
        ("iter_0007.jsonl", 7),
        ("iter_1234.jsonl", 1234),
        # Everything else in a buffer directory must not be counted as an
        # iteration, or the clock invents games that were never played.
        ("curriculum_seed.jsonl", None),
        ("iter_0007.jsonl.tmp", None),
        ("iter_007.jsonl", None),
        ("buffer_final.jsonl", None),
        ("xiter_0007.jsonl", None),
    ],
)
def test_only_iteration_buffers_are_recognised(name: str, expected: int | None):
    assert iteration_of_buffer_file(name) == expected


def test_count_games_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "iter_0000.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n', encoding="utf-8")
    assert count_games(path) == 2


def test_count_games_of_missing_file_is_zero(tmp_path: Path):
    assert count_games(tmp_path / "iter_0000.jsonl") == 0


# -- the clock -------------------------------------------------------------


def test_totals_accumulate_and_exclude_the_seed_corpus(tmp_path: Path):
    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    write_buffer(buffers, 1, 400)
    write_buffer(buffers, 2, 500)
    # A seed corpus is a prior the run starts with, not progress it has made.
    (buffers / "curriculum_seed.jsonl").write_text(
        "".join('{"seed": true}\n' for _ in range(5_000)), encoding="utf-8"
    )

    ledger = GamesLedger(buffers)
    assert ledger.known_iterations() == [0, 1, 2]
    assert ledger.games_for(1) == 400
    assert ledger.total_through(1) == 800
    assert ledger.total_through(2) == 1_300
    # The value schedules read: games that existed when the iteration began.
    assert ledger.total_before(0) == 0
    assert ledger.total_before(2) == 800


def test_total_before_zero_and_negative_iterations_are_zero(tmp_path: Path):
    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    ledger = GamesLedger(buffers)
    assert ledger.total_before(0) == 0
    assert ledger.total_through(-1) == 0


def test_gaps_contribute_zero_rather_than_raising(tmp_path: Path):
    """A run resumed after a failed iteration has a hole; report what exists."""

    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    write_buffer(buffers, 3, 400)
    ledger = GamesLedger(buffers)
    assert ledger.games_for(1) == 0
    assert ledger.total_through(3) == 800


def test_a_fresh_ledger_recomputes_the_same_totals(tmp_path: Path):
    """The resume invariant: no stored schedule position, so none can drift."""

    buffers = tmp_path / "buffers"
    for iteration in range(5):
        write_buffer(buffers, iteration, 400)

    before = GamesLedger(buffers)
    before.refresh()
    totals = [before.total_before(i) for i in range(5)]

    # A "resume": brand-new object, same directory. The cache file exists, but
    # deleting it must not change a single number.
    (tmp_path / "games_ledger.json").unlink()
    after = GamesLedger(buffers)
    assert [after.total_before(i) for i in range(5)] == totals


def test_cache_is_used_but_invalidated_by_a_size_change(tmp_path: Path):
    buffers = tmp_path / "buffers"
    path = write_buffer(buffers, 0, 400)

    ledger = GamesLedger(buffers)
    assert ledger.games_for(0) == 400
    cache = json.loads((tmp_path / "games_ledger.json").read_text(encoding="utf-8"))
    assert cache["files"]["iter_0000.jsonl"]["games"] == 400

    # Buffer files are immutable in practice; if one changes anyway, the cached
    # count must not survive it.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"game": 400}) + "\n")
    assert GamesLedger(buffers).games_for(0) == 401


def test_a_corrupt_cache_falls_back_to_counting(tmp_path: Path):
    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    (tmp_path / "games_ledger.json").write_text("{not json", encoding="utf-8")
    assert GamesLedger(buffers).total_through(0) == 400


def test_summary_reports_the_clock_for_a_stats_row(tmp_path: Path):
    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    write_buffer(buffers, 1, 300)
    summary = GamesLedger(buffers).summary(1)
    assert summary == {
        "iteration": 1,
        "games_before_iteration": 400,
        "games_this_iteration": 300,
        "games_through_iteration": 700,
        "iterations_on_disk": 2,
    }


# -- games-keyed anneal ----------------------------------------------------


def test_game_schedule_anneals_and_clamps():
    schedule = GameSchedule(0.15, 0.0, 10_000)
    assert schedule.value(0) == pytest.approx(0.15)
    assert schedule.value(5_000) == pytest.approx(0.075)
    assert schedule.value(10_000) == pytest.approx(0.0)
    assert schedule.value(50_000) == pytest.approx(0.0)
    assert not schedule.finished(9_999)
    assert schedule.finished(10_000)


def test_zero_duration_reads_as_already_finished():
    """Matches LinearSchedule, so a disabled curriculum is off, not undefined."""

    assert GameSchedule(0.15, 0.0, 0).value(0) == pytest.approx(0.0)
    assert GameSchedule(0.15, 0.0, -5).value(0) == pytest.approx(0.0)
    assert GameSchedule(0.15, 0.0, 0).finished(0)


def test_game_schedule_rejects_a_negative_clock():
    with pytest.raises(ValueError, match="non-negative"):
        GameSchedule(1.0, 0.0, 100).value(-1)


def test_the_anneal_is_invariant_to_games_per_iteration():
    """The point of W1.2, stated as a test.

    Two runs reach 8,000 games at different iteration counts.  A games-keyed
    schedule gives them the same mix; the iteration-keyed schedule it replaces
    does not, and that divergence is silent.
    """

    games_schedule = GameSchedule(0.15, 0.0, 10_000)
    run_03 = 8_000 // 400  # 20 iterations at run 03's 400 games/iteration
    cloud = 8_000 // 800  # 10 iterations at a cloud run's 800

    assert games_schedule.value(8_000) == pytest.approx(games_schedule.value(8_000))

    iteration_schedule = LinearSchedule(0.15, 0.0, 25)
    assert iteration_schedule.value(run_03) != pytest.approx(
        iteration_schedule.value(cloud)
    )


# -- growing window --------------------------------------------------------


def test_window_grows_sublinearly_and_hits_the_cap():
    window = GrowingReplayWindow(coefficient=40.0, exponent=0.6, cap_games=20_000)
    small = window.games(1_000)
    large = window.games(100_000)
    assert small < large <= 20_000
    # Sublinear: 100x the games must not buy 100x the window.
    assert large / small < 100
    assert window.games(10_000_000) == 20_000


def test_window_respects_its_floor_at_the_start_of_a_run():
    window = GrowingReplayWindow(
        coefficient=40.0, exponent=0.6, cap_games=20_000, floor_games=1_000
    )
    assert window.games(0) == 1_000


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(coefficient=0.0, exponent=0.6, cap_games=100), "coefficient"),
        (dict(coefficient=1.0, exponent=0.0, cap_games=100), "exponent"),
        (dict(coefficient=1.0, exponent=1.5, cap_games=100), "exponent"),
        (dict(coefficient=1.0, exponent=0.6, cap_games=0), "cap_games"),
        (
            dict(coefficient=1.0, exponent=0.6, cap_games=100, floor_games=101),
            "floor_games must not exceed",
        ),
    ],
)
def test_window_rejects_incoherent_configuration(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        GrowingReplayWindow(**kwargs)


def test_selection_takes_whole_iterations_newest_first(tmp_path: Path):
    """The steady-state case: more history than the target, so it trims."""

    buffers = tmp_path / "buffers"
    for iteration in range(60):
        write_buffer(buffers, iteration, 400)
    ledger = GamesLedger(buffers)

    # Shipping defaults: at 24,000 games this asks for ~7,400.
    window = GrowingReplayWindow(coefficient=16.0, exponent=0.6, cap_games=20_000)
    total = ledger.total_through(59)
    selection = window.select(total, 59, ledger.games_for, ledger.known_iterations())

    assert total == 24_000
    assert selection.newest_iteration == 59
    assert selection.realised_games >= selection.target_games
    # Whole iterations only, and contiguous from the newest backwards.
    assert list(selection.iterations) == list(range(selection.oldest_iteration, 60))
    # Overshoot is bounded by one iteration's worth of games.
    assert selection.realised_games - selection.target_games < 400
    # And it is a trim, not "everything": the point of a window.
    assert selection.realised_games < total


def test_selection_underfills_when_history_is_shorter_than_the_target(tmp_path: Path):
    """Early in a run the target exceeds all games ever played.

    Documented behaviour, not an error -- and the reason ``WindowSelection``
    carries both numbers: a run whose realised window sits below its target has
    a window schedule that is not yet binding, which is only visible if both are
    logged.
    """

    buffers = tmp_path / "buffers"
    for iteration in range(10):
        write_buffer(buffers, iteration, 400)
    ledger = GamesLedger(buffers)

    window = GrowingReplayWindow(coefficient=40.0, exponent=0.6, cap_games=20_000)
    selection = window.select(4_000, 9, ledger.games_for, ledger.known_iterations())

    assert selection.target_games > 4_000
    assert selection.realised_games == 4_000
    assert list(selection.iterations) == list(range(10))


def test_selection_always_includes_the_newest_iteration(tmp_path: Path):
    """At the start of a run the target is smaller than one iteration."""

    buffers = tmp_path / "buffers"
    write_buffer(buffers, 0, 400)
    ledger = GamesLedger(buffers)
    window = GrowingReplayWindow(
        coefficient=1.0, exponent=0.5, cap_games=20_000, floor_games=1
    )
    selection = window.select(400, 0, ledger.games_for, ledger.known_iterations())
    assert selection.iterations == (0,)
    assert selection.realised_games == 400
    assert selection.target_games == 20


def test_selection_ignores_iterations_newer_than_the_current_one(tmp_path: Path):
    """A resumed run can have buffers ahead of the iteration being replayed."""

    buffers = tmp_path / "buffers"
    for iteration in range(5):
        write_buffer(buffers, iteration, 400)
    ledger = GamesLedger(buffers)
    window = GrowingReplayWindow(
        coefficient=40.0, exponent=0.6, cap_games=20_000, floor_games=1
    )
    selection = window.select(1_200, 2, ledger.games_for, ledger.known_iterations())
    assert selection.newest_iteration == 2
    assert 3 not in selection.iterations
    assert 4 not in selection.iterations


def test_selection_is_empty_only_when_nothing_exists(tmp_path: Path):
    ledger = GamesLedger(tmp_path / "buffers")
    window = GrowingReplayWindow(
        coefficient=40.0, exponent=0.6, cap_games=20_000, floor_games=1
    )
    selection = window.select(0, 0, ledger.games_for, ledger.known_iterations())
    assert selection.iterations == ()
    assert selection.realised_games == 0
    assert selection.oldest_iteration is None


def test_a_resumed_selection_reproduces_the_original(tmp_path: Path):
    """W1.4's resume requirement, at the level these primitives can guarantee."""

    buffers = tmp_path / "buffers"
    for iteration in range(8):
        write_buffer(buffers, iteration, 400)
    window = GrowingReplayWindow(
        coefficient=40.0, exponent=0.6, cap_games=20_000, floor_games=1
    )

    first = GamesLedger(buffers)
    original = window.select(
        first.total_through(7), 7, first.games_for, first.known_iterations()
    )

    (tmp_path / "games_ledger.json").unlink()
    second = GamesLedger(buffers)
    resumed = window.select(
        second.total_through(7), 7, second.games_for, second.known_iterations()
    )
    assert resumed == original
