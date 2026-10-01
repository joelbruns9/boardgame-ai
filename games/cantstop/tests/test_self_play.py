"""Tests for solver-driven self-play and the training rows it produces.

These are slow by nature: a real turn solve is ~1 s of Python enumeration, so
the full-game tests use one game on the cheapest rule set and the per-rule-set
coverage tests a single turn rather than a whole game.

Run: python -m pytest games/cantstop/tests/test_self_play.py -q
"""

import random

import numpy as np
import pytest
import torch

from games.cantstop.encoder import FEATURE_SIZE, encode_board, seat_order
from games.cantstop.engine import ALL_RULESETS, GameState, Phase, RuleSet
from games.cantstop.model import CantStopNet, NetEvaluator
from games.cantstop.self_play import (
    GameResult, TurnLimitExceeded, play_game, play_games, play_turn,
    stack_rows, summarize,
)
from games.cantstop.solver import ProgressHeuristic

CHEAP = RuleSet.make(2, extended=False, blocking=False)


def one_game(seed=3, rules=CHEAP, **kw):
    return play_game(rules, ProgressHeuristic(), random.Random(seed), **kw)


# ---- what a row is ----

def test_every_turn_end_is_a_row_except_the_one_that_wins():
    """A winning stop is scored exactly by the solver and never reaches the
    net, so it must not become a training row."""
    r = one_game()
    assert r.turns > 1
    assert len(r) == r.turns - 1
    assert r.features.shape == (len(r), FEATURE_SIZE)


def test_rows_carry_no_terminal_board():
    r = one_game(keep_boards=True)
    assert all(not b.game_over for b in r.boards)
    assert all(b.phase == Phase.AWAIT_ROLL for b in r.boards)
    assert all(not b.runners for b in r.boards)


def test_features_are_exactly_the_encoder_on_the_recorded_boards():
    r = one_game(keep_boards=True)
    for i, board in enumerate(r.boards):
        assert np.array_equal(r.features[i], encode_board(board))


def test_winner_slot_resolves_back_to_the_game_winner():
    """The label is an encoding slot, not an absolute seat. Getting this
    wrong mislabels every row of every game."""
    r = one_game(keep_boards=True)
    assert r.winner is not None
    for board, slot in zip(r.boards, r.winner_slots):
        assert seat_order(board)[slot] == r.winner


def test_empty_row_set_still_has_the_right_feature_width():
    r = GameResult(rules=CHEAP, winner=0,
                   features=np.zeros((0, FEATURE_SIZE), dtype=np.float32),
                   winner_slots=np.zeros(0, dtype=np.int64),
                   turns=0, solves=0, evaluator_rows=0)
    assert len(r) == 0
    assert r.features.shape[1] == FEATURE_SIZE


# ---- cost ----

def test_a_turn_costs_at_most_one_solve():
    r = one_game()
    assert r.solves <= r.turns
    assert r.solves > 0


def test_each_solve_is_exactly_one_evaluator_call():
    """The measurement the Rust-port ordering rests on, end to end over a
    whole game rather than a single solve."""
    torch.manual_seed(0)
    ev = NetEvaluator(CantStopNet(hidden=(16, 16)), device="cpu")
    r = play_game(CHEAP, ev, random.Random(5))
    assert ev.calls == r.solves
    assert r.evaluator_rows == ev.rows > r.solves


# ---- coverage across rule sets ----

def test_every_ruleset_plays_a_turn_and_passes_it_on():
    rng = random.Random(17)
    for rules in ALL_RULESETS:
        s = GameState(rules)
        solves, decisions = play_turn(s, ProgressHeuristic(), rng)
        assert solves in (0, 1)
        assert s.phase in (Phase.AWAIT_ROLL, Phase.GAME_OVER)
        assert not s.runners
        if not s.game_over:
            assert s.active_player != 0 or rules.num_players == 1
        if solves:
            assert decisions >= 1


# ---- guards ----

def test_a_game_that_never_ends_is_an_error_not_a_draw():
    with pytest.raises(TurnLimitExceeded, match="never banking"):
        play_game(CHEAP, ProgressHeuristic(), random.Random(1), max_turns=2)


def test_the_same_seed_replays_the_same_game():
    a = one_game(seed=9)
    b = one_game(seed=9)
    assert a.winner == b.winner
    assert a.turns == b.turns
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.winner_slots, b.winner_slots)


# ---- aggregation ----

def test_stack_rows_concatenates_and_rejects_an_empty_set():
    results = play_games(CHEAP, ProgressHeuristic(), random.Random(11), 2)
    x, y = stack_rows(results)
    assert x.shape == (sum(len(r) for r in results), FEATURE_SIZE)
    assert y.shape == (x.shape[0],)
    with pytest.raises(ValueError, match="no rows"):
        stack_rows([])


def test_summarize_reports_the_collapse_watch():
    results = play_games(CHEAP, ProgressHeuristic(), random.Random(13), 2)
    out = summarize(results)
    assert out["games"] == 2
    assert out["rows"] == sum(len(r) for r in results)
    assert out["turns"] == sum(r.turns for r in results)
    # Turn length is what tells always-stop and always-roll collapse apart.
    assert out["mean_turn_length"] > 0
    assert out["max_turn_length"] >= out["mean_turn_length"]
    assert sum(out["seat_wins"].values()) == 2
