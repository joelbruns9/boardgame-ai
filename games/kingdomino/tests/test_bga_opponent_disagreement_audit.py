import pytest

from games.kingdomino.bga_opponent_disagreement_audit import (
    claim_round,
    game_outcome,
    rank_pick_policy,
)


def test_claim_round_maps_mighty_duel_deck_counts():
    assert claim_round(44) == 1
    assert claim_round(40) == 2
    assert claim_round(0) == 12
    assert claim_round(41) is None


def test_pick_policy_records_human_and_model_tiles_with_stable_ranks():
    policy, result = rank_pick_policy({8: 0.10, 14: 0.70, 21: 0.20}, 8)

    assert [entry["domino_id"] for entry in policy] == [14, 21, 8]
    assert result["human_pick_domino_id"] == 8
    assert result["model_pick_domino_id"] == 14
    assert result["human_pick_rank"] == 3
    assert result["model_to_human_probability_ratio"] == pytest.approx(7.0)
    assert result["disagreement"]


def test_game_outcome_is_opponent_framed():
    players = {
        "hero": {"name": "Hero", "score": 110},
        "opp": {"name": "StrongHuman", "score": 115},
    }
    outcome = game_outcome(players, "hero", "opp")

    assert outcome["opponent_result"] == "win"
    assert outcome["opponent_score_margin"] == 5
    assert outcome["winner_name"] == "StrongHuman"
