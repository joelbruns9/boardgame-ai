from games.kingdomino.absorption_gap import analyze


def test_absorption_gap_compares_prior_to_both_search_seeds():
    stage1 = [
        {
            "position_id": "p1",
            "raw_policy": {"selected_action": {"action_idx": 1}, "selected_pick_domino_id": 10},
        },
        {
            "position_id": "p2",
            "raw_policy": {"selected_action": {"action_idx": 2}, "selected_pick_domino_id": 20},
        },
    ]
    stage2 = [
        {
            "position_id": "p1",
            "table_id": "g1",
            "stage2_searches": [
                {"selected_action": {"action_idx": 1}, "selected_pick_domino_id": 10},
                {"selected_action": {"action_idx": 1}, "selected_pick_domino_id": 10},
            ],
        },
        {
            "position_id": "p2",
            "table_id": "g2",
            "stage2_searches": [
                {"selected_action": {"action_idx": 3}, "selected_pick_domino_id": 21},
                {"selected_action": {"action_idx": 4}, "selected_pick_domino_id": 22},
            ],
        },
    ]
    result = analyze(stage1, stage2, bootstrap_samples=100, bootstrap_seed=3)

    assert result["joint_action"]["prior_vs_4800_agreement"] == 0.5
    assert result["joint_action"]["4800_cross_seed_self_agreement"] == 0.5
    assert result["joint_action"]["absorption_gap_self_minus_prior"] == 0.0
    assert result["tile_group"]["prior_vs_4800_agreement"] == 0.5
