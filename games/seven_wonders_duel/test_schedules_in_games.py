"""Schedules read the games clock, and a resume reproduces them (W1.1/W1.2/W1.4).

``test_games_clock.py`` covers the primitives in isolation.  These tests cover
the wiring: that ``PhaseDLoop`` reads the clock rather than the iteration index,
that both bases stay available and distinct, and that the specific failure the
plan names -- a schedule silently rescaling when ``games_per_iteration`` changes
-- cannot happen on the games basis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from games.seven_wonders_duel.phase_d import PhaseDConfig, PhaseDLoop


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


def write_iterations(loop: PhaseDLoop, games_per_iteration: dict[int, int]) -> None:
    """Fake generated buffers, one JSON line per game.

    The clock counts lines, so these need not be valid records -- and using
    stubs keeps the test about scheduling rather than about self-play.
    """

    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    for iteration, games in games_per_iteration.items():
        path = loop.buffer_dir / f"iter_{iteration:04d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(games):
                handle.write(json.dumps({"game": index}) + "\n")


# -- the clock the loop reads ----------------------------------------------


def test_generation_clock_excludes_the_iteration_being_generated(tmp_path: Path):
    """Schedules must not depend on the games they are deciding how to make."""

    loop = make_loop(tmp_path)
    write_iterations(loop, {0: 400, 1: 400, 2: 400})
    assert loop.generation_clock(0) == 0
    assert loop.generation_clock(2) == 800
    # The window is applied after generation, so it sees this iteration's games.
    assert loop.training_clock(2) == 1_200


def test_curriculum_and_draft_prior_follow_the_games_clock(tmp_path: Path):
    loop = make_loop(
        tmp_path,
        games_per_iteration=500,
        opponent_fraction=0.15,
        curriculum_anneal_games=10_000,
        draft_prior_games=10_000,
    )
    write_iterations(loop, {iteration: 500 for iteration in range(10)})

    # Iteration 10 begins at 5,000 games: exactly half way through the anneal.
    assert loop.generation_clock(10) == 5_000
    assert loop.curriculum_mix_fraction(10) == pytest.approx(0.075)
    assert loop.draft_prior_amount(10) == pytest.approx(0.5)


def test_curriculum_reaches_zero_at_the_measured_exhaustion_point(tmp_path: Path):
    loop = make_loop(tmp_path, games_per_iteration=500, curriculum_anneal_games=10_000)
    write_iterations(loop, {iteration: 500 for iteration in range(20)})
    assert loop.generation_clock(20) == 10_000
    assert loop.curriculum_mix_fraction(20) == pytest.approx(0.0)


def test_the_schedule_is_invariant_to_games_per_iteration(tmp_path: Path):
    """W1.2's whole purpose, at the loop level.

    Two runs reach 8,000 games with different iteration counts.  On the games
    basis they see the same curriculum mix; on the iterations basis they do not,
    and nothing warns.
    """

    slow = make_loop(
        tmp_path / "slow", games_per_iteration=400, curriculum_anneal_games=10_000
    )
    write_iterations(slow, {iteration: 400 for iteration in range(20)})
    fast = make_loop(
        tmp_path / "fast", games_per_iteration=800, curriculum_anneal_games=10_000
    )
    write_iterations(fast, {iteration: 800 for iteration in range(10)})

    assert slow.generation_clock(20) == fast.generation_clock(10) == 8_000
    assert slow.curriculum_mix_fraction(20) == pytest.approx(
        fast.curriculum_mix_fraction(10)
    )

    legacy_slow = make_loop(
        tmp_path / "ls",
        schedule_basis="iterations",
        games_per_iteration=400,
        curriculum_anneal_iterations=25,
    )
    legacy_fast = make_loop(
        tmp_path / "lf",
        schedule_basis="iterations",
        games_per_iteration=800,
        curriculum_anneal_iterations=25,
    )
    assert legacy_slow.curriculum_mix_fraction(20) != pytest.approx(
        legacy_fast.curriculum_mix_fraction(10)
    )


# -- growing window --------------------------------------------------------


def test_window_grows_with_the_run_and_trims_the_oldest(tmp_path: Path):
    loop = make_loop(
        tmp_path,
        games_per_iteration=400,
        replay_window_coefficient=16.0,
        replay_window_exponent=0.6,
        replay_window_cap_games=20_000,
    )
    write_iterations(loop, {iteration: 400 for iteration in range(60)})

    early = loop.window_selection(4)
    late = loop.window_selection(59)
    assert early is not None and late is not None
    # Grows in absolute terms...
    assert late.target_games > early.target_games
    # ...and shrinks as a share of all games played: sublinear growth.
    assert late.realised_games / loop.training_clock(59) < (
        early.realised_games / loop.training_clock(4)
    )
    # Late in the run the window is a trim, and it is contiguous and recent.
    assert late.newest_iteration == 59
    assert late.oldest_iteration > 0


def test_window_is_capped(tmp_path: Path):
    loop = make_loop(
        tmp_path, games_per_iteration=1_000, replay_window_cap_games=3_000
    )
    write_iterations(loop, {iteration: 1_000 for iteration in range(40)})
    selection = loop.window_selection(39)
    assert selection.target_games == 3_000
    assert selection.iterations == (37, 38, 39)


def test_training_records_are_drawn_from_the_selected_window(tmp_path: Path):
    """The selection is what actually reaches training, not just a report."""

    loop = make_loop(
        tmp_path, games_per_iteration=1_000, replay_window_cap_games=2_000
    )
    write_iterations(loop, {iteration: 1_000 for iteration in range(5)})
    selection = loop.window_selection(4)
    assert selection.iterations == (3, 4)
    # `training_records` parses the buffers, so point it at the same files and
    # check the paths agree rather than re-parsing stub records here.
    assert [loop.games_ledger.path_for(i) for i in selection.iterations] == [
        loop.buffer_dir / "iter_0003.jsonl",
        loop.buffer_dir / "iter_0004.jsonl",
    ]


def test_legacy_basis_keeps_the_fixed_iteration_window(tmp_path: Path):
    loop = make_loop(tmp_path, schedule_basis="iterations", replay_window=3)
    write_iterations(loop, {iteration: 400 for iteration in range(10)})
    assert loop.window_selection(9) is None
    assert loop.window_iterations(9) == 3


# -- resume ----------------------------------------------------------------


def test_a_resumed_loop_reproduces_every_schedule_value(tmp_path: Path):
    """W1.4: the clock is recomputed from buffers, so a resume cannot drift."""

    first = make_loop(tmp_path, games_per_iteration=400)
    write_iterations(first, {iteration: 400 for iteration in range(12)})
    original = first.schedule_state(12)

    # Discard the memo pad and rebuild the loop from scratch.
    (Path(first.run_dir) / "games_ledger.json").unlink(missing_ok=True)
    second = make_loop(tmp_path, games_per_iteration=400)
    assert second.schedule_state(12) == original


def test_a_resume_that_changes_games_per_iteration_keeps_its_place(tmp_path: Path):
    """The cloud case: run 03's 400 resumed at 800 must not jump the schedule."""

    original = make_loop(tmp_path, games_per_iteration=400)
    write_iterations(original, {iteration: 400 for iteration in range(20)})
    before = original.curriculum_mix_fraction(20)

    resumed = make_loop(tmp_path, games_per_iteration=800)
    assert resumed.generation_clock(20) == 8_000
    assert resumed.curriculum_mix_fraction(20) == pytest.approx(before)


def test_schedule_identity_omits_games_per_iteration_on_the_games_basis():
    """Because changing it is exactly what the games basis makes safe."""

    identity = PhaseDConfig(run_dir="x", games_per_iteration=400).schedule_identity()
    assert "games_per_iteration" not in identity
    assert identity["schedule_basis"] == "games"
    assert identity["curriculum_anneal_games"] == 10_000

    legacy = PhaseDConfig(
        run_dir="x", schedule_basis="iterations", replay_window=7
    ).schedule_identity()
    assert legacy["replay_window"] == 7
    assert "replay_window_cap_games" not in legacy


# -- resume refusal --------------------------------------------------------


def test_an_unchanged_resume_is_allowed(tmp_path: Path):
    make_loop(tmp_path).initialize()
    make_loop(tmp_path).initialize()


@pytest.mark.parametrize(
    "overrides",
    [
        dict(curriculum_anneal_games=5_000),
        dict(draft_prior_games=5_000),
        dict(replay_window_exponent=0.7),
        dict(replay_window_coefficient=32.0),
        dict(replay_window_cap_games=10_000),
        dict(opponent_fraction=0.30),
        dict(hof_opponent_fraction=0.15),
        dict(hof_sampling_mode="uniform"),
        dict(schedule_basis="iterations"),
    ],
)
def test_a_resume_that_moves_a_schedule_is_refused(tmp_path: Path, overrides: dict):
    make_loop(tmp_path).initialize()
    with pytest.raises(ValueError, match="changed training schedules"):
        make_loop(tmp_path, **overrides).initialize()


def test_changing_games_per_iteration_across_a_resume_is_allowed(tmp_path: Path):
    """The freedom the games basis exists to provide, asserted explicitly."""

    make_loop(tmp_path, games_per_iteration=400).initialize()
    make_loop(tmp_path, games_per_iteration=800).initialize()


def test_a_pre_games_clock_run_is_refused_and_told_what_to_pass(tmp_path: Path):
    """A manifest with no basis describes an iteration-keyed run.

    Defaulting it to the *config* default would silently rescale every schedule
    on the first resume after this change shipped -- which is the exact failure
    the guard exists to prevent, so the default is what the old run really used.
    """

    loop = make_loop(tmp_path)
    loop.initialize()
    manifest = Path(loop.manifest.path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("schedule_basis", "curriculum_anneal_games", "draft_prior_games"):
        payload["config"].pop(key, None)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="--schedule-basis iterations"):
        make_loop(tmp_path).initialize()


# -- stats -----------------------------------------------------------------


def test_schedule_state_reports_target_and_realised_window(tmp_path: Path):
    loop = make_loop(
        tmp_path, games_per_iteration=400, replay_window_cap_games=1_200
    )
    write_iterations(loop, {iteration: 400 for iteration in range(10)})
    state = loop.schedule_state(9)
    assert state["basis"] == "games"
    assert state["games_before_iteration"] == 3_600
    assert state["games_through_iteration"] == 4_000
    assert state["replay_window_target_games"] == 1_200
    assert state["replay_window_realised_games"] == 1_200
    assert state["replay_window_iterations"] == 3
    assert state["learning_rate"] == loop.config.learning_rate
    assert "hof_opponent_fraction" in state


# -- configuration validation ---------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(schedule_basis="epochs"), "schedule_basis"),
        (dict(curriculum_anneal_games=-1), "non-negative"),
        (dict(draft_prior_games=-1), "non-negative"),
        (dict(hof_opponent_fraction=1.5), "hof_opponent_fraction"),
        (dict(hof_sampling_mode="best"), "hof_sampling_mode"),
        (dict(replay_window_exponent=0.0), "exponent"),
        (dict(replay_window_exponent=1.5), "exponent"),
        (dict(replay_window_coefficient=0.0), "coefficient"),
        (dict(replay_window_cap_games=0), "cap_games"),
    ],
)
def test_incoherent_schedule_config_is_rejected_at_config_time(
    overrides: dict, message: str
):
    with pytest.raises(ValueError, match=message):
        PhaseDConfig(run_dir="x", **overrides).validate()


def test_a_window_cap_below_one_iteration_is_rejected(tmp_path: Path):
    """The floor is one iteration, so a cap below it is incoherent, not clamped."""

    with pytest.raises(ValueError, match="floor_games must not exceed"):
        PhaseDConfig(
            run_dir="x", games_per_iteration=500, replay_window_cap_games=100
        ).validate()
