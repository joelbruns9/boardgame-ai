from __future__ import annotations

from games.kingdomino.action_codec import encode_action
from games.kingdomino.deep_target_screen import aggregate_search, screen_flags
from games.kingdomino.game import GameState, Phase, PickAction, TurnAction


def _first_place_state(seed: int = 91) -> GameState:
    state = GameState.new(seed=seed, start_player=0)
    while state.phase == Phase.INITIAL_SELECTION:
        state = state.step(PickAction(state.current_row[0]))
    return state


def test_search_aggregation_preserves_pick_groups_and_actor_frame():
    state = _first_place_state()
    actions = state.legal_actions()
    children = []
    for index, action in enumerate(actions):
        assert isinstance(action, TurnAction)
        visits = 10 + (int(action.pick_domino_id) % 3)
        q_actor = 0.1 + int(action.pick_domino_id) / 100.0
        value_sum_p0 = q_actor * visits * (1 if state.current_actor == 0 else -1)
        children.append(
            (int(encode_action(action, state)), visits, value_sum_p0, 1.0 / len(actions))
        )

    result = aggregate_search(
        state,
        children,
        root_value_p0=0.25 if state.current_actor == 0 else -0.25,
        elapsed_seconds=0.1,
        seed=7,
    )

    assert len(result["pick_groups"]) == len(state.current_row)
    assert {g["pick_domino_id"] for g in result["pick_groups"]} == set(state.current_row)
    assert sum(g["visits"] for g in result["pick_groups"]) == result["root_total_visits"]
    assert result["root_value_actor"] == 0.25
    assert len(result["top_joint_actions"]) == 3


def test_screen_flags_escalate_instability_close_q_and_exact():
    base_search = {
        "selected_action": {"action_idx": 10},
        "selected_pick_domino_id": 20,
        "top2_pick_q_gap": 0.02,
        "top_pick_visit_share": 0.7,
        "starved_pick_group_count": 0,
    }
    other_search = dict(base_search)
    other_search["selected_action"] = {"action_idx": 11}
    position = {
        "position_id": "synthetic",
        "human_action": {"pick_domino_id": 20},
        "audit_tags": {"exact_candidate": True},
    }

    flags = screen_flags(
        position,
        [base_search, other_search],
        q_gap_threshold=0.05,
        visit_share_threshold=0.6,
        control_fraction=0.0,
        base_seed=1,
    )

    assert flags["seed_action_disagreement"]
    assert not flags["seed_pick_disagreement"]
    assert flags["top2_q_close"]
    assert flags["exact_candidate"]
    assert not flags["random_control"]
