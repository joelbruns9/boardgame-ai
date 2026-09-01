from __future__ import annotations

import pytest

from games.kingdomino.action_codec import encode_action
from games.kingdomino.deep_target_stage2 import (
    CohortRules,
    aggregate_restricted_search,
    select_cohort,
)
from games.kingdomino.game import GameState, Phase, PickAction


def _row(
    position_id: str,
    *,
    deck: int,
    roots: tuple[float, float] = (0.0, 0.0),
    gaps: tuple[float | None, float | None] = (0.1, 0.1),
    unstable: bool = False,
    control: bool = False,
    starved: bool = False,
    split: str = "development",
):
    return {
        "position_id": position_id,
        "state_sha256": position_id * 2,
        "table_id": "1",
        "source_decision_index": 1,
        "deck_count": deck,
        "phase": "PLACE_AND_SELECT",
        "split": split,
        "searches": [
            {"root_value_actor": root, "top2_pick_q_gap": gap}
            for root, gap in zip(roots, gaps)
        ],
        "screen_flags": {
            "seed_pick_disagreement": unstable,
            "random_control": control,
            "pick_group_starved": starved,
        },
    }


def test_cohort_selection_uses_both_close_repeats_and_one_starved_per_deck():
    rows = [
        _row("primary", deck=40, gaps=(0.02, 0.03)),
        _row("one-close", deck=40, gaps=(0.02, 0.04)),
        _row("unstable", deck=36, roots=(0.9, 0.9), unstable=True),
        _row("control", deck=32, roots=(0.9, 0.9), control=True),
        _row("starved-a", deck=28, starved=True),
        _row("starved-b", deck=28, starved=True),
        _row("starved-c", deck=24, starved=True),
        _row("confirmation", deck=20, unstable=True, split="confirmation"),
    ]
    entries = select_cohort(rows, CohortRules())
    by_id = {entry["position_id"]: entry for entry in entries}

    assert "primary" in by_id
    assert "one-close" not in by_id
    assert by_id["unstable"]["reasons"] == ["stage1_pick_unstable"]
    assert by_id["control"]["reasons"] == ["stage1_easy_control"]
    assert len({"starved-a", "starved-b"} & set(by_id)) == 1
    assert "starved-c" in by_id
    assert "confirmation" not in by_id

    confirmation = select_cohort(rows, CohortRules(), split="confirmation")
    assert [entry["position_id"] for entry in confirmation] == ["confirmation"]


def test_restricted_aggregation_accepts_exact_forced_pick_action_set():
    state = GameState.new(seed=91, start_player=0)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(PickAction(state.current_row[0]))
    pick_id = state.current_row[0]
    allowed = [
        int(encode_action(action, state))
        for action in state.legal_actions()
        if action.pick_domino_id == pick_id
    ]
    actor_sign = 1 if state.current_actor == 0 else -1
    children = [
        (action_idx, 10 + index, actor_sign * (10 + index) * 0.2, 0.1)
        for index, action_idx in enumerate(allowed)
    ]
    result = aggregate_restricted_search(
        state,
        children,
        allowed_actions=allowed,
        root_value_p0=actor_sign * 0.2,
        elapsed_seconds=0.1,
        seed=3,
    )

    assert result["allowed_action_count"] == len(allowed)
    assert result["root_total_visits"] == sum(child[1] for child in children)
    assert result["root_value_actor"] == 0.2
    assert result["selected_action"]["pick_domino_id"] == pick_id


def test_restricted_aggregation_rejects_extra_actions_and_wrong_visit_count():
    state = GameState.new(seed=91, start_player=0)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(PickAction(state.current_row[0]))
    pick_id = state.current_row[0]
    legal = [int(encode_action(action, state)) for action in state.legal_actions()]
    allowed = [
        int(encode_action(action, state))
        for action in state.legal_actions()
        if action.pick_domino_id == pick_id
    ]
    extra = next(action_idx for action_idx in legal if action_idx not in allowed)
    children = [(action_idx, 10, 2.0, 0.1) for action_idx in allowed]

    with pytest.raises(ValueError, match="extra="):
        aggregate_restricted_search(
            state,
            children + [(extra, 1, 0.0, 0.1)],
            allowed_actions=allowed,
            root_value_p0=0.2,
            elapsed_seconds=0.1,
            seed=3,
        )

    with pytest.raises(ValueError, match="restricted visit mismatch"):
        aggregate_restricted_search(
            state,
            children,
            allowed_actions=allowed,
            root_value_p0=0.2,
            elapsed_seconds=0.1,
            seed=3,
            expected_visits=sum(child[1] for child in children) + 1,
        )
