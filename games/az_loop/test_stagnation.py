"""W7a/W7b: the detector's null must be self-calibrating, and never a point."""

from __future__ import annotations

import pytest

from .stagnation import (
    DEFAULT_LADDER,
    EXHAUSTED,
    INTERVENING,
    NORMAL,
    STAGNANT,
    AnchorMeasurement,
    InterventionLadder,
    LadderState,
    StagnationDetector,
)


def _series(scores, *, lower_offset=-0.06, start_games=20_000, step=10_000):
    """Measurements at a fixed cadence, with intervals around each score."""

    return [
        AnchorMeasurement(
            games=start_games + index * step,
            score_rate=score,
            lower=score + lower_offset,
            upper=score - lower_offset,
            anchor_games=start_games + index * step - 20_000,
        )
        for index, score in enumerate(scores)
    ]


DETECTOR = StagnationDetector()


def test_a_single_flat_measurement_is_never_stagnation():
    verdict = DETECTOR.verdict(_series([0.50]))
    assert verdict.stagnant is False
    assert "insufficient" in verdict.reasons[0]


def test_two_measurements_are_still_not_enough():
    assert DETECTOR.verdict(_series([0.50, 0.50])).stagnant is False


def test_a_learning_run_is_not_stagnant():
    # The anchor lags 20k games, so a learning run beats it and the score rises.
    verdict = DETECTOR.verdict(_series([0.60, 0.65, 0.70]))
    assert verdict.stagnant is False
    assert verdict.slope_per_10k_games > 0


def test_a_run_whose_anchor_caught_up_is_stagnant():
    # Learning stopped: the anchor advances to the frozen best and the score
    # converges to a net against itself.
    verdict = DETECTOR.verdict(_series([0.52, 0.51, 0.50]))
    assert verdict.stagnant is True
    assert any("clears 0.50" in reason for reason in verdict.reasons)


def test_a_steady_learning_rate_is_not_stagnant():
    # A constant fixed-lag advantage is steady learning, not stagnation. Its
    # near-zero slope measures zero acceleration and remains telemetry only.
    verdict = DETECTOR.verdict(
        _series([0.70, 0.70, 0.70], lower_offset=-0.05)
    )
    assert verdict.stagnant is False
    assert verdict.reasons == ()
    assert verdict.slope_per_10k_games == pytest.approx(0.0)


def test_the_interval_trigger_can_fire_alone_while_the_score_creeps_up():
    # Improving too slowly to ever clear its own past self.
    verdict = DETECTOR.verdict(_series([0.50, 0.55, 0.60], lower_offset=-0.15))
    assert verdict.stagnant is True
    assert any("lower bounds" in reason for reason in verdict.reasons)


def test_a_rising_run_whose_latest_interval_clears_the_null_is_healthy():
    # Wide intervals early, but the newest measurement is confidently ahead.
    verdict = DETECTOR.verdict(_series([0.50, 0.60, 0.72], lower_offset=-0.20))
    assert verdict.stagnant is False


def test_the_slope_is_per_games_not_per_measurement():
    """Cadence changes mid-run when a rung lengthens the window."""

    dense = _series([0.50, 0.55, 0.60], step=5_000)
    sparse = _series([0.50, 0.55, 0.60], step=20_000)
    dense_slope = DETECTOR.verdict(dense).slope_per_10k_games
    sparse_slope = DETECTOR.verdict(sparse).slope_per_10k_games
    assert dense_slope > sparse_slope
    assert dense_slope == pytest.approx(sparse_slope * 4, rel=1e-6)


def test_measurements_survive_a_round_trip():
    measurement = _series([0.61])[0]
    assert AnchorMeasurement.from_dict(measurement.as_dict()) == measurement


# -- W7b: the ladder --------------------------------------------------------


STAGNANT_VERDICT = DETECTOR.verdict(_series([0.50, 0.50, 0.50]))
HEALTHY_VERDICT = DETECTOR.verdict(_series([0.60, 0.66, 0.72]))


def test_the_ladder_is_off_by_default_but_detection_still_reports():
    ladder = InterventionLadder()
    assert ladder.enabled is False
    state = ladder.advance(LadderState(), STAGNANT_VERDICT, games=50_000)
    assert state.state == STAGNANT
    assert ladder.active(state) is None, "no rung is applied while disabled"


def test_the_first_rung_is_search_budget():
    ladder = InterventionLadder(enabled=True)
    state = ladder.advance(LadderState(), STAGNANT_VERDICT, games=50_000)
    assert state.state == INTERVENING
    assert ladder.active(state).name == "raise_search_budget"
    assert ladder.active(state).sims_multiplier > 1.0


def test_a_rung_is_held_for_its_measurement_window():
    ladder = InterventionLadder(enabled=True, measurement_window_games=20_000)
    state = ladder.advance(LadderState(), STAGNANT_VERDICT, games=50_000)
    # Still stagnant, but the rung has not had its window yet.
    held = ladder.advance(state, STAGNANT_VERDICT, games=60_000)
    assert held == state, "escalating early makes the effect unattributable"
    escalated = ladder.advance(state, STAGNANT_VERDICT, games=70_000)
    assert escalated.rung == state.rung + 1


def test_recovery_returns_to_normal_and_drops_the_rung():
    ladder = InterventionLadder(enabled=True, measurement_window_games=20_000)
    state = ladder.advance(LadderState(), STAGNANT_VERDICT, games=50_000)
    recovered = ladder.advance(state, HEALTHY_VERDICT, games=80_000)
    assert recovered.state == NORMAL
    assert recovered.rung == -1
    assert ladder.active(recovered) is None, "a recovered run keeps no rung"


def test_the_ladder_escalates_one_rung_at_a_time_then_reports_exhausted():
    ladder = InterventionLadder(enabled=True, measurement_window_games=10_000)
    state = LadderState()
    games = 50_000
    seen = []
    for _ in range(len(DEFAULT_LADDER)):
        state = ladder.advance(state, STAGNANT_VERDICT, games=games)
        seen.append(ladder.active(state).name)
        games += 10_000
    assert seen == [rung.name for rung in DEFAULT_LADDER]

    state = ladder.advance(state, STAGNANT_VERDICT, games=games)
    assert state.state == EXHAUSTED
    assert ladder.active(state).name == DEFAULT_LADDER[-1].name, (
        "the last rung stays applied; dropping everything at exhaustion would "
        "be a fifth uncontrolled change"
    )


def test_model_growth_is_not_on_the_ladder():
    names = {rung.name for rung in DEFAULT_LADDER}
    assert not any("model" in name or "width" in name for name in names)
    assert all(
        not hasattr(rung, "d_model_multiplier") for rung in DEFAULT_LADDER
    )


def test_ladder_state_survives_a_resume():
    ladder = InterventionLadder(enabled=True)
    state = ladder.advance(LadderState(), STAGNANT_VERDICT, games=50_000)
    restored = LadderState.from_dict(state.as_dict())
    assert restored == state
    assert ladder.active(restored).name == ladder.active(state).name


def test_an_absent_ladder_block_resumes_at_normal():
    assert LadderState.from_dict(None) == LadderState()
    assert LadderState.from_dict({}) == LadderState()
