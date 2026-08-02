"""W5.8: the gate size is scheduled, and the schedule is resume-safe."""

from __future__ import annotations

import pytest

from .training_control import (
    ACCEPT,
    CONTINUE,
    REJECT,
    GateLadder,
    GeneratorMode,
    GeneratorState,
    gate_transition,
    initial_state,
)


LADDER = GateLadder(rungs=(100, 200, 400, 800), step_up_after=2)


def _step(state: GeneratorState, decision: str, **kwargs) -> GeneratorState:
    return gate_transition(
        state,
        decision,
        revert_reset_after=2,
        iteration=state.last_iteration + 1,
        ladder=LADDER,
        **kwargs,
    ).next_state


def test_two_probations_step_up_and_a_promotion_steps_back_down():
    state = initial_state(GeneratorMode.SOFT_GATE)
    assert LADDER.games(state.gate_rung) == 100

    state = _step(state, CONTINUE)
    assert (state.gate_rung, state.consecutive_probations) == (0, 1)

    state = _step(state, CONTINUE)
    assert (state.gate_rung, state.consecutive_probations) == (1, 0)
    assert LADDER.games(state.gate_rung) == 200

    state = _step(state, CONTINUE)
    state = _step(state, CONTINUE)
    assert LADDER.games(state.gate_rung) == 400

    state = _step(state, ACCEPT)
    assert LADDER.games(state.gate_rung) == 200
    assert state.consecutive_probations == 0


def test_a_single_probation_between_decisive_gates_never_steps_up():
    state = initial_state(GeneratorMode.SOFT_GATE)
    for decision in (CONTINUE, ACCEPT, CONTINUE, REJECT, CONTINUE):
        state = _step(state, decision)
    assert state.gate_rung == 0
    assert state.consecutive_probations == 1


def test_revert_holds_the_rung_so_the_confirming_gate_has_equal_resolution():
    # A revert is decisive evidence, and `revert_reset_after` needs a second
    # gate to confirm it. Shrinking the sample for that gate would make the
    # confirmation weaker than the accusation.
    state = initial_state(GeneratorMode.SOFT_GATE)
    state = _step(state, CONTINUE)
    state = _step(state, CONTINUE)
    rung_before = state.gate_rung
    state = _step(state, REJECT)
    assert state.gate_rung == rung_before
    assert state.consecutive_probations == 0


def test_the_ladder_stops_at_the_top_rung():
    state = initial_state(GeneratorMode.SOFT_GATE)
    for _ in range(40):
        state = _step(state, CONTINUE)
    assert state.gate_rung == LADDER.top
    assert LADDER.games(state.gate_rung) == 800


def test_the_floor_holds_the_bottom_rung_through_bootstrap_probations():
    state = initial_state(GeneratorMode.SOFT_GATE)
    for _ in range(6):
        state = _step(state, CONTINUE, allow_step_up=False)
    assert state.gate_rung == 0, "bootstrap probation is not evidence of stagnation"
    assert state.consecutive_probations == 0, (
        "blocked probations must not accumulate into a debt that steps the "
        "ladder up the moment the floor clears"
    )
    # Clearing the floor starts the count fresh: one probation is not enough.
    state = _step(state, CONTINUE)
    assert state.gate_rung == 0
    state = _step(state, CONTINUE)
    assert state.gate_rung == 1


def test_ladder_position_survives_a_resume():
    state = initial_state(GeneratorMode.SOFT_GATE)
    state = _step(state, CONTINUE)
    state = _step(state, CONTINUE)
    state = _step(state, CONTINUE)
    restored = GeneratorState.from_row(state.as_row())
    assert restored == state
    assert (restored.gate_rung, restored.consecutive_probations) == (1, 1)


def test_a_pre_ladder_row_resumes_at_the_bottom_rung():
    row = initial_state(GeneratorMode.SOFT_GATE).as_row()
    del row["gate_rung"]
    del row["consecutive_probations"]
    restored = GeneratorState.from_row(row)
    assert (restored.gate_rung, restored.consecutive_probations) == (0, 0)


def test_rungs_must_be_ascending_positive_and_even():
    GateLadder(rungs=(100, 200)).validate()
    GateLadder.fixed(400).validate()
    with pytest.raises(ValueError, match="ascending"):
        GateLadder(rungs=(400, 200)).validate()
    with pytest.raises(ValueError, match="even"):
        GateLadder(rungs=(101,)).validate()
    with pytest.raises(ValueError, match="at least one rung"):
        GateLadder(rungs=()).validate()


# -- W6.6: run-health heartbeat --------------------------------------------


def test_heartbeat_reports_health_from_the_committed_row_alone():
    from .run_controller import RunController

    row = {
        "iteration": 42,
        "current_best_iteration": 40,
        "promotion_action": "probation",
        "stats": {
            "generation": {"games_per_second": 1.2345},
            "resources": {
                "peak_rss_bytes": 6 * 1024**3,
                "vram_peak_physical_bytes": 9 * 1024**3,
            },
            # Every gate is fixed-N since W5.5; `stop_reason` is what
            # separates a promotion decision from an anchor measurement.
            "gates": [
                {
                    "fixed_n": True,
                    "stop_reason": "probation",
                    "score_rate": 0.53,
                    "wilson_lcb": 0.47,
                    "wilson_ucb": 0.59,
                    "games": 400,
                },
                {"fixed_n": True, "stop_reason": "fixed_n", "score_rate": 0.82},
                {"fixed_n": True, "stop_reason": "fixed_n", "score_rate": 0.66},
            ],
        },
    }
    line = RunController.heartbeat_line(row)
    assert "iter=0042" in line
    assert "games/s=1.234" in line or "games/s=1.235" in line
    assert "rss=6.00GiB" in line
    assert "vram=9.00GiB" in line
    assert "best_iter=40" in line
    assert "action=probation" in line
    assert "gate=0.530[0.470,0.590]n=400" in line
    assert "bots=0.740(min 0.660)" in line


def test_heartbeat_carries_the_w7a_signals_when_a_measurement_ran():
    from .run_controller import RunController

    row = {
        "iteration": 7,
        "current_best_iteration": 5,
        "promotion_action": "probation",
        "promotions_in_window": "2/20000g",
        "stats": {
            "outcomes": {
                "terminal_reason": {
                    "civilian": 70,
                    "scientific": 13,
                    "military": 17,
                }
            },
            "training": {"accuracies": {"value_acc": 0.612}},
            "game_specific": {
                "stagnation": {
                    "anchor": {
                        "score_rate": 0.58,
                        "wilson_lcb": 0.52,
                        "wilson_ucb": 0.64,
                        "lag_games": 20_000,
                    },
                    "verdict": {"stagnant": True, "slope_per_10k_games": 0.001},
                    "ladder": {"active": "raise_search_budget"},
                }
            },
        },
    }
    line = RunController.heartbeat_line(row)
    assert "self=0.580[0.520,0.640]lag=20000" in line
    assert "slope=+0.0010/10k" in line
    assert "STAGNANT" in line
    assert "rung=raise_search_budget" in line
    assert "value_acc=0.612" in line
    assert "mix=0.70/0.13/0.17" in line
    assert "promotions=2/20000g" in line


def test_heartbeat_survives_a_row_with_no_stats_block():
    from .run_controller import RunController

    line = RunController.heartbeat_line({"iteration": 1})
    assert "iter=0001" in line
    assert "rss=" not in line, "absent telemetry should be absent, not zero"


# -- W7b: inconclusive gates accumulate ------------------------------------
#
# Run 03 degraded for 45 iterations without `revert_reset_after` firing. Both
# rules below are about how a *probation* -- "at this many games we could not
# tell" -- is allowed to affect the counters that arrest a bad learner.


def test_probation_no_longer_clears_the_revert_tally():
    # Run 03's actual sequence: a revert at iteration 105, then a 0.465 gate at
    # 110 that was too close to call. The near-miss used to wipe the revert, so
    # the three consecutive reverts a reset needs could never accumulate.
    state = initial_state(GeneratorMode.SOFT_GATE)
    state = _step(state, REJECT)
    assert state.consecutive_reverts == 1

    state = _step(state, CONTINUE)
    assert state.consecutive_reverts == 1, "a probation is not evidence of innocence"

    result = gate_transition(
        state,
        REJECT,
        revert_reset_after=2,
        iteration=state.last_iteration + 1,
        ladder=LADDER,
    )
    assert result.reset_learner, "the second revert should now reach the reset"


def test_a_promotion_clears_the_revert_tally():
    state = initial_state(GeneratorMode.SOFT_GATE)
    state = _step(state, REJECT)
    state = _step(state, ACCEPT)
    assert state.consecutive_reverts == 0


def test_sustained_probation_reaches_the_reset_on_its_own():
    state = initial_state(GeneratorMode.SOFT_GATE)
    for expected in (1, 2):
        result = gate_transition(
            state,
            CONTINUE,
            revert_reset_after=0,
            probation_reset_after=3,
            iteration=state.last_iteration + 1,
            ladder=LADDER,
        )
        assert not result.reset_learner
        state = result.next_state
        assert state.probations_since_decisive == expected

    result = gate_transition(
        state,
        CONTINUE,
        revert_reset_after=0,
        probation_reset_after=3,
        iteration=state.last_iteration + 1,
        ladder=LADDER,
    )
    assert result.reset_learner
    assert result.action.value == "revert_reset"
    assert result.next_state.probations_since_decisive == 0


def test_the_probation_counter_survives_a_ladder_step_up():
    # The ladder zeroes `consecutive_probations` every `step_up_after`, so the
    # reset needs its own counter or it could never exceed step_up_after - 1.
    state = initial_state(GeneratorMode.SOFT_GATE)
    for _ in range(3):
        state = _step(state, CONTINUE, probation_reset_after=0)
    assert state.gate_rung == 1, "the ladder stepped up and reset its own counter"
    assert state.consecutive_probations == 1
    assert state.probations_since_decisive == 3, "this counter is not the ladder's"


def test_a_decisive_gate_clears_the_probation_counter():
    for decisive in (ACCEPT, REJECT):
        state = initial_state(GeneratorMode.SOFT_GATE)
        state = _step(state, CONTINUE, probation_reset_after=0)
        state = _step(state, CONTINUE, probation_reset_after=0)
        assert state.probations_since_decisive == 2
        state = _step(state, decisive, probation_reset_after=0)
        assert state.probations_since_decisive == 0


def test_probation_reset_is_off_by_default():
    state = initial_state(GeneratorMode.SOFT_GATE)
    for _ in range(10):
        result = gate_transition(
            state,
            CONTINUE,
            revert_reset_after=0,
            iteration=state.last_iteration + 1,
            ladder=LADDER,
        )
        assert not result.reset_learner
        state = result.next_state


def test_probation_reset_after_rejects_a_negative_value():
    state = initial_state(GeneratorMode.SOFT_GATE)
    with pytest.raises(ValueError, match="probation_reset_after"):
        gate_transition(
            state,
            CONTINUE,
            revert_reset_after=0,
            probation_reset_after=-1,
            iteration=0,
            ladder=LADDER,
        )


def test_both_counters_round_trip_through_a_committed_row():
    state = initial_state(GeneratorMode.SOFT_GATE)
    state = _step(state, REJECT)
    state = _step(state, CONTINUE, probation_reset_after=0)
    restored = GeneratorState.from_row(state.as_row())
    assert restored.consecutive_reverts == state.consecutive_reverts
    assert restored.probations_since_decisive == state.probations_since_decisive


def test_a_pre_w7b_row_resumes_with_a_zero_probation_counter():
    state = initial_state(GeneratorMode.SOFT_GATE)
    row = state.as_row()
    del row["probations_since_decisive"]
    assert GeneratorState.from_row(row).probations_since_decisive == 0


def test_a_suppressed_revert_does_not_advance_or_trigger_any_counter():
    state = GeneratorState(
        mode=GeneratorMode.SOFT_GATE,
        consecutive_reverts=1,
        consecutive_probations=1,
        probations_since_decisive=2,
    )
    result = gate_transition(
        state,
        CONTINUE,
        revert_reset_after=2,
        probation_reset_after=3,
        iteration=4,
        ladder=LADDER,
        gate_stop_reason="revert_suppressed_knot",
    )
    assert result.action.value == "probation"
    assert result.reset_learner is False
    assert result.next_state.consecutive_reverts == 1
    assert result.next_state.consecutive_probations == 1
    assert result.next_state.probations_since_decisive == 2
