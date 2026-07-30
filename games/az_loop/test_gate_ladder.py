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
    # Once the floor is cleared the same evidence ladders normally.
    state = _step(state, CONTINUE)
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
