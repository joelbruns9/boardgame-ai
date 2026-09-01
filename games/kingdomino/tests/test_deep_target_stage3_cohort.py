from games.kingdomino.deep_target_stage3_cohort import select_stage3


def _stage2(position_id, stage1_picks, stage2_picks, split="development"):
    def searches(picks):
        return [{"selected_pick_domino_id": pick} for pick in picks]

    return {
        "position_id": position_id,
        "table_id": "1",
        "source_decision_index": 1,
        "deck_count": 20,
        "phase": "PLACE_AND_SELECT",
        "state_sha256": position_id * 2,
        "split": split,
        "stage1_searches": searches(stage1_picks),
        "stage2_searches": searches(stage2_picks),
    }


def test_stage3_selection_has_three_predeclared_routes_and_no_confirmation():
    stage2 = [
        _stage2("changed", (1, 1), (2, 2)),
        _stage2("unstable", (1, 1), (1, 2)),
        _stage2("forced", (1, 1), (1, 1)),
        _stage2("easy", (1, 1), (1, 1)),
        _stage2("confirmation", (1, 1), (2, 2), split="confirmation"),
    ]
    forced = [
        {
            "position_id": "forced",
            "pick_groups": [
                {"was_starved_at_4800": True, "regret_vs_best_forced_q": 0.02}
            ],
        },
        {
            "position_id": "easy",
            "pick_groups": [
                {"was_starved_at_4800": True, "regret_vs_best_forced_q": 0.04}
            ],
        },
    ]
    entries = select_stage3(stage2, forced)
    by_id = {entry["position_id"]: entry for entry in entries}

    assert set(by_id) == {"changed", "unstable", "forced"}
    assert by_id["changed"]["reasons"] == ["stable_consensus_pick_changed"]
    assert by_id["unstable"]["reasons"] == ["stage2_pick_unstable"]
    assert by_id["forced"]["reasons"] == ["starved_forced_q_within_003"]

    confirmation = select_stage3(stage2, forced, split="confirmation")
    assert [entry["position_id"] for entry in confirmation] == ["confirmation"]
