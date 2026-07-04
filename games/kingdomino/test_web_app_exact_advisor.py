from __future__ import annotations

import pytest

from fastapi import HTTPException

import games.kingdomino.web_app as web_app
from games.kingdomino.board import Board
from games.kingdomino.dominoes import Terrain
from games.kingdomino.game import Claim, GameConfig, GameState, Phase


def _full_board_with_no_legal_placements() -> Board:
    board = Board()
    cx, cy = board.castle_pos
    board.terrain.fill(int(Terrain.EMPTY))
    board.crowns.fill(0)
    board.domino_id.fill(0)
    board._occupied = set()
    board._cell = {}

    for y in range(cy - 3, cy + 4):
        for x in range(cx - 3, cx + 4):
            terrain = int(Terrain.CASTLE if (x, y) == (cx, cy) else Terrain.WHEAT)
            board.terrain[y, x] = terrain
            board.domino_id[y, x] = -1 if terrain == int(Terrain.CASTLE) else 99
            board._occupied.add((x, y))
            board._cell[(x, y)] = terrain

    board._min_x = cx - 3
    board._max_x = cx + 3
    board._min_y = cy - 3
    board._max_y = cy + 3
    return board


def _forced_discard_final_state() -> GameState:
    return GameState(
        config=GameConfig(),
        boards=[_full_board_with_no_legal_placements(), _full_board_with_no_legal_placements()],
        deck=[],
        current_row=[],
        pending_claims=[Claim(0, 5), Claim(1, 6)],
        next_claims=[],
        phase=Phase.FINAL_PLACEMENT,
    )


def _forced_discard_deck4_state() -> GameState:
    return GameState(
        config=GameConfig(),
        boards=[_full_board_with_no_legal_placements(), _full_board_with_no_legal_placements()],
        deck=[21, 22, 23, 24],
        current_row=[1, 2, 3, 4],
        pending_claims=[Claim(0, 5), Claim(1, 6), Claim(1, 7), Claim(0, 8)],
        next_claims=[],
        phase=Phase.PLACE_AND_SELECT,
    )


def setup_function() -> None:
    web_app._EXACT_ADVISOR_VALUE_CACHE.clear()


def test_exact_advisor_returns_solved_recommendations_for_deck0_final():
    pytest.importorskip("kingdomino_rust")
    state = _forced_discard_final_state()
    req = web_app.RecommendRequest(
        engine="exact",
        state=web_app.state_to_debug_json(state),
        top_k=4,
        exact_max_secs=3.0,
    )

    response = web_app.recommend(req)

    assert response["engine"] == "exact"
    assert response["exact"]["solved"] is True
    assert response["exact"]["deck_count"] == 0
    assert response["num_simulations"] == 0
    assert response["recommendations"]
    rec = response["recommendations"][0]
    assert rec["visit_frac"] is None
    assert 0.0 <= rec["q_win_prob"] <= 1.0
    assert rec["debug"]["label"] == "exact"


def test_exact_advisor_caches_deck4_child_values():
    pytest.importorskip("kingdomino_rust")
    state = _forced_discard_deck4_state()
    req = web_app.RecommendRequest(
        engine="exact",
        state=web_app.state_to_debug_json(state),
        top_k=4,
        exact_max_secs=3.0,
    )

    first = web_app.recommend(req)
    second = web_app.recommend(req)

    assert first["exact"]["solved"] is True
    assert first["exact"]["deck_count"] == 4
    assert second["exact"]["cache_hits"] > first["exact"]["cache_hits"]
    assert second["recommendations"][0]["debug"]["cache_hit"] is True


def test_auto_advisor_uses_exact_for_eligible_deck4_state():
    pytest.importorskip("kingdomino_rust")
    state = _forced_discard_deck4_state()
    req = web_app.RecommendRequest(
        engine="auto",
        state=web_app.state_to_debug_json(state),
        top_k=4,
        exact_max_secs=3.0,
    )

    response = web_app.recommend(req)

    assert response["engine"] == "exact"
    assert response["exact"]["solved"] is True
    assert response["exact"]["deck_count"] == 4


def test_exact_advisor_rejects_non_endgame_state():
    state = GameState.new(seed=1)
    req = web_app.RecommendRequest(engine="exact", state=web_app.state_to_debug_json(state))

    with pytest.raises(HTTPException) as excinfo:
        web_app.recommend(req)

    assert excinfo.value.status_code == 400
    assert "only available" in str(excinfo.value.detail)


def test_exact_advisor_requires_hidden_deck_for_deck4_capture():
    state = _forced_discard_deck4_state()
    public_state = web_app.state_to_public_json(state, include_debug=False)
    req = web_app.RecommendRequest(engine="exact", state=public_state)

    with pytest.raises(HTTPException) as excinfo:
        web_app.recommend(req)

    assert excinfo.value.status_code == 400
    assert "debug.deck" in str(excinfo.value.detail)
