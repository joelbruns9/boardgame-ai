from __future__ import annotations

from types import SimpleNamespace

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
    web_app._EXACT_ADVISOR_MARGIN_CACHE.clear()


def test_exact_advisor_interactive_budget_defaults_to_30_seconds():
    assert web_app.RecommendRequest().exact_max_secs == 30.0


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


def test_exact_advisor_solves_each_node_once_and_derives_all_outputs(monkeypatch):
    rust = pytest.importorskip("kingdomino_rust")
    import games.kingdomino.endgame_solver as endgame_solver

    state = _forced_discard_deck4_state()
    original = endgame_solver.exact_endgame_value
    calls = []

    def counted(state_arg, **kwargs):
        calls.append((web_app._exact_state_key(state_arg), kwargs.copy()))
        return original(state_arg, **kwargs)

    monkeypatch.setattr(endgame_solver, "exact_endgame_value", counted)
    req = web_app.RecommendRequest(
        engine="exact",
        state=web_app.state_to_debug_json(state),
        top_k=100,
        exact_max_secs=3.0,
        swindle=False,
    )

    response = web_app.recommend(req)

    expected_states = {web_app._exact_state_key(state)}
    expected_states.update(web_app._exact_state_key(state.step(a)) for a in state.legal_actions())
    assert len(calls) == len(expected_states)
    assert {key for key, _kwargs in calls} == expected_states
    assert all(kwargs["alpha"] == 1.0 and kwargs["margin_gain"] == 1.0
               for _key, kwargs in calls)

    root_margin = int(response["root_margin_pts"])
    assert response["root_margin_pts"] == float(root_margin)
    assert response["root_value_player0"] == pytest.approx(
        rust.margin_to_training_value(root_margin, 160.0, 2.0, 0.0), abs=0.0
    )
    assert response["root_rank_value"] == pytest.approx(
        rust.margin_to_training_value(root_margin, 160.0, 2.0, 0.5), abs=0.0
    )
    assert all(rec["exact_margin_pts"].is_integer() for rec in response["recommendations"])


@pytest.mark.parametrize("engine", ["auto", "exact"])
def test_exact_timeout_falls_back_to_nn_mcts(monkeypatch, engine):
    from games.kingdomino.action_codec import encode_action
    import games.kingdomino.mcts_az as mcts_az

    state = _forced_discard_deck4_state()
    action = state.legal_actions()[0]
    child = SimpleNamespace(prior=0.6, visit_count=1, value=0.2)
    root = SimpleNamespace(children={encode_action(action, state): child})
    mcts_kwargs = {}

    def force_timeout(*_args, **_kwargs):
        raise web_app.ExactTimeout("Exact solver did not finish the root within the 30 s budget.")

    class FakeMCTS:
        def __init__(self, *_args, **kwargs):
            mcts_kwargs.update(kwargs)

    monkeypatch.setattr(web_app, "_recommend_exact", force_timeout)
    monkeypatch.setattr(web_app, "_load_nn_evaluator", lambda _req: (object(), object(), "fake.pt"))
    monkeypatch.setattr(web_app, "_root_trajectory", lambda *_args: {})
    monkeypatch.setattr(mcts_az, "OpenLoopMCTS", FakeMCTS)
    monkeypatch.setattr(
        mcts_az,
        "run_pimc_open_loop",
        lambda *_args, **_kwargs: ({action: 1.0}, 0.1, root),
    )
    req = web_app.RecommendRequest(
        engine=engine,
        state=web_app.state_to_debug_json(state),
        nn_sims=1,
        swindle=False,
        draft_matrix=False,
    )

    response = web_app.recommend(req)

    assert response["engine"] == "nn-mcts"
    assert response["exact_fallback"] is True
    assert "30 s budget" in response["reason"]
    assert response["recommendations"]
    assert mcts_kwargs["exact_endgame_enabled"] is False
    assert mcts_kwargs["exact_endgame_max_secs"] == 0.0


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
