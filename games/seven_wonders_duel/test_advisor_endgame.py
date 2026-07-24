"""Exact endgame annotator: correctness, regimes, gating, host integration.

The headline correctness bar is the optimal-line check: for a deterministic
(chance-free) endgame, playing the solver's best move for both players to
terminal must realize exactly the predicted outcome.
"""

from __future__ import annotations

import threading
import time

import pytest

from games.advisor import JobManager, RecommendRequest

from .advisor_adapter import SevenWondersAdvisor, _Position
from .advisor_endgame import ExactEndgameAnnotator, solve_position
from .codec import decode_action, legal_action_indices
from .engine import apply_action
from .game import Phase, new_game
from .search import state_actor


def _deadline():
    return time.perf_counter() + 10.0


def _present(game):
    return sum(1 for card in game.tableau.cards.values() if card.present)


def _facedown(game):
    return any(c.present and not c.revealed for c in game.tableau.cards.values())


@pytest.fixture(scope="module")
def positions():
    """One deterministic small Age-3 position, one small expectimax one, and a
    mid-game Age-2 position, harvested from random playouts."""

    det = exp = mid = None
    for seed in range(120):
        game = new_game(seed, first_player=seed % 2)
        rng = __import__("random").Random(500 + seed)
        while game.phase is not Phase.COMPLETE:
            if game.phase is Phase.PLAY_AGE and game.age == 2 and mid is None:
                mid = game.clone()
            if game.phase is Phase.PLAY_AGE and game.age == 3:
                if det is None and not _facedown(game) and _present(game) <= 6:
                    det = game.clone()
                if exp is None and _facedown(game) and _present(game) <= 5:
                    exp = game.clone()
            apply_action(game, decode_action(game, rng.choice(legal_action_indices(game))))
        if det is not None and exp is not None and mid is not None:
            break
    assert det is not None and mid is not None
    return {"deterministic": det, "expectimax": exp, "midgame": mid}


def _shim(game):
    return _Position(game=game, key="test")


def test_deterministic_optimal_line_realizes_prediction(positions):
    game = positions["deterministic"]
    root_actor = state_actor(game)
    solved = solve_position(game, deadline=_deadline(), max_nodes=300_000)
    assert solved is not None and solved["regime"] == "exact"
    assert abs(abs(solved["root_value"]) - round(abs(solved["root_value"]))) < 1e-6

    line = game.clone()
    guard = 0
    while line.phase is not Phase.COMPLETE and guard < 40:
        step = solve_position(line, deadline=_deadline(), max_nodes=300_000)
        apply_action(line, decode_action(line, step["best_index"]))
        guard += 1
    realized = 0.0 if line.winner is None else (1.0 if line.winner == root_actor else -1.0)
    predicted = solved["root_value"]
    assert (predicted > 1e-9) == (realized > 1e-9)
    assert (predicted < -1e-9) == (realized < -1e-9)


def test_deterministic_regime_value_is_integral(positions):
    solved = solve_position(positions["deterministic"], deadline=_deadline(), max_nodes=300_000)
    assert solved["regime"] == "exact"
    for value in solved["per_action_value"].values():
        assert value in (-1.0, 0.0, 1.0)


def test_expectimax_regime_when_facedown(positions):
    if positions["expectimax"] is None:
        pytest.skip("no small expectimax position harvested")
    solved = solve_position(positions["expectimax"], deadline=_deadline(), max_nodes=300_000)
    if solved is None:
        pytest.skip("expectimax position exceeded budget")
    assert solved["regime"] == "exact_expectimax"
    assert -1.0 <= solved["root_value"] <= 1.0


def test_midgame_is_gated(positions):
    result = ExactEndgameAnnotator().annotate(
        _shim(positions["midgame"]),
        [],
        RecommendRequest(),
        deadline=_deadline(),
        stop_event=threading.Event(),
    )
    assert result is None


def test_node_budget_gates(positions):
    result = ExactEndgameAnnotator(max_nodes=5).annotate(
        _shim(positions["deterministic"]),
        [],
        RecommendRequest(),
        deadline=_deadline(),
        stop_event=threading.Event(),
    )
    assert result is None


def test_cancellation_gates(positions):
    stop = threading.Event()
    stop.set()
    result = ExactEndgameAnnotator().annotate(
        _shim(positions["deterministic"]),
        [],
        RecommendRequest(),
        deadline=_deadline(),
        stop_event=stop,
    )
    assert result is None


def test_deterministic_annotation_shape(positions):
    ann = ExactEndgameAnnotator()
    result = ann.annotate(
        _shim(positions["deterministic"]),
        [],
        RecommendRequest(),
        deadline=_deadline(),
        stop_event=threading.Event(),
    )
    assert result is not None and result.name == "exact_endgame"
    assert result.summary["regime"] == "exact"
    assert result.summary["outcome"] in ("win", "loss", "draw")
    exactly_one_best = sum(1 for b in result.per_action.values() if b["is_best"])
    assert exactly_one_best == 1


def test_host_attaches_endgame_annotation(positions):
    from .inference import Evaluator
    from .train import build_model

    adapter = SevenWondersAdvisor(evaluator=Evaluator(build_model("transformer", 32, 1), "cpu"))
    resp = JobManager(adapter).run_blocking(
        _shim(positions["deterministic"]),
        RecommendRequest(engine="auto", max_sims=40, chunk_sims=20, top_k=4, seed=1),
    )
    assert resp.ok
    assert "exact_endgame" in resp.summary
    assert any("exact_endgame" in r.annotations for r in resp.recommendations)
