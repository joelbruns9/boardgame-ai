"""Rule of 28 opponents and the solver's stop_bias hook."""

import statistics

import numpy as np
import pytest

from games.cantstop.engine import (GameState, Phase, RuleSet, apply_move,
                                   roll)
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rule28 import (CONSTANT_CHAMPION, LINEAR_CHAMPION,
                                   RULE_OF_28, Rule28Player, choose_move,
                                   move_value, progress_value, should_stop,
                                   solitaire_turns)
from games.cantstop.rule28_sweep import play_game, summarize
from games.cantstop.rust_solver import RustTurnSolver, hashed_evaluator, sample_positions
from games.cantstop.solver import ProgressHeuristic, TurnSolver


def _state(runners, saved=None, claimed=(), dice=None):
    s = GameState(RuleSet.make(2))
    for c, p in (saved or {}).items():
        s.progress[0][c] = p
    for c in claimed:
        s.claimed_by[c] = 1
    s.runners = dict(runners)
    s.phase = Phase.AWAIT_DECISION
    if dice is not None:
        s.phase, s.dice = Phase.AWAIT_MOVE, tuple(sorted(dice))
    return s


def test_paper_figure1_move_choice():
    # Runners on 5 and 8, column 4 won. Dice 1-3-4-4 pair as 4+8 (only the
    # 8 is usable) or 5+7. Paper: using the 8 scores 5, the 5 and 7 score
    # 4 + 6 - 6 = 4, so the rule takes the 8.
    s = _state({5: 2, 8: 3}, saved={5: 1, 8: 2}, claimed=(4,), dice=(1, 3, 4, 4))
    assert move_value(s, (8,), RULE_OF_28) == 5
    assert move_value(s, (5, 7), RULE_OF_28) == 4
    assert choose_move(s, RULE_OF_28) == (8,)


def test_paper_figure2_progress_value():
    # Advanced 2 on column 4, 3 on column 6, 1 on column 10: 12 + 8 + 8,
    # all even -2 = 26 < 28, so roll again.
    s = _state({4: 2, 6: 3, 10: 1})
    assert progress_value(s, RULE_OF_28) == 26
    assert not should_stop(s, RULE_OF_28)
    s.runners[6] = 4                      # one more space on 6: 28
    assert progress_value(s, RULE_OF_28) == 28 and should_stop(s, RULE_OF_28)


def test_difficulty_needs_three_runners():
    assert progress_value(_state({4: 1, 6: 1}), RULE_OF_28) == 2 * 4 + 2 * 2
    # all odd and all high: 3,1 -> +2 odd, +4 high
    three = _state({7: 1, 9: 1, 11: 1})
    assert progress_value(three, RULE_OF_28) == 2 * 1 + 2 * 3 + 2 * 5 + 2 + 4


def test_linear_space_weights():
    # Column 2 (length 3): floor(64 * j / 3 + 7) for j = 1, 2, 3.
    assert [LINEAR_CHAMPION.space_weight(2, j) for j in (1, 2, 3)] == [28, 49, 71]
    assert [LINEAR_CHAMPION.space_weight(12, j) for j in (1, 2, 3)] == [28, 49, 71]
    assert CONSTANT_CHAMPION.space_weight(6, 5) == 4


def test_bank_wins_and_blocked_stop():
    s = _state({12: 3}, claimed=())
    s.claimed_by[2] = s.claimed_by[3] = 0
    assert should_stop(s, RULE_OF_28)                       # stopping wins
    assert not should_stop(s, RULE_OF_28, bank_wins=False)  # 4 * 6 = 24 < 28
    b = GameState(RuleSet.make(2, blocking=True))
    b.progress[1][7] = 5
    b.runners = {7: 5, 6: 10}      # 6 + 22 = 28
    b.phase = Phase.AWAIT_DECISION
    assert progress_value(b, RULE_OF_28) >= 28
    assert not should_stop(b, RULE_OF_28)                   # blocked: must roll


def test_solitaire_rule_of_28_matches_paper():
    # Paper: 10.74 turns, sd ~3.25. 3000 games: se ~0.06.
    rng = PortableRng(11)
    turns = [solitaire_turns(RULE_OF_28, rng, bank_wins=True) for _ in range(3000)]
    assert abs(statistics.mean(turns) - 10.74) < 0.25


def test_rule28_players_finish_games():
    for rules in (RuleSet.make(2), RuleSet.make(4, blocking=True)):
        s = GameState(rules)
        rng = PortableRng(3)
        players = [Rule28Player(p) for p in (RULE_OF_28, CONSTANT_CHAMPION,
                                             LINEAR_CHAMPION, RULE_OF_28)]
        for _ in range(2000):
            if s.game_over:
                break
            players[s.active_player].play_turn(s, rng)
        assert s.game_over


def test_rust_stop_bias_matches_python_solver():
    for rules in (RuleSet.make(2), RuleSet.make(3, blocking=True)):
        for s in sample_positions(rules, 5, 8):
            for bias in (-0.04, 0.03):
                py = TurnSolver(s, hashed_evaluator, stop_bias=bias)
                rs = RustTurnSolver(s, hashed_evaluator, stop_bias=bias)
                assert np.array_equal(py.value(s), rs.value(s))
                if s.phase == Phase.AWAIT_MOVE:
                    assert py.choose_move(s) == rs.choose_move(s)
                if s.phase == Phase.AWAIT_DECISION:
                    assert py.should_stop(s) == rs.should_stop(s)


def test_stop_bias_changes_play_and_refuses_after_solve():
    s = next(p for p in sample_positions(RuleSet.make(2), 1, 20)
             if p.phase == Phase.AWAIT_MOVE)
    plain = RustTurnSolver(s, ProgressHeuristic())
    with pytest.raises(ValueError):
        plain._s.stop_bias = 0.1
    assert plain._s.stop_bias == 0.0
    assert RustTurnSolver(s, ProgressHeuristic(), stop_bias=-0.5)._s.stop_bias == -0.5


def test_sweep_game_and_pairing_summary():
    r = play_game(RuleSet.make(2), ProgressHeuristic(), 7, 1, 0.0, "rule_of_28")
    assert r["turns"] > 0 and isinstance(r["won"], bool)
    results = {0.0: [(0, 0, True, 10), (1, 1, False, 12), (2, 0, True, 9), (3, 1, False, 9)],
               0.02: [(0, 0, True, 10), (1, 1, True, 12), (2, 0, False, 9), (3, 1, True, 9)]}
    arms = summarize(results, [0.0, 0.02], 2)["arms"]
    assert arms[0]["win_rate"] == 0.5 and "paired_vs_bias0" not in arms[0]
    assert arms[1]["paired_vs_bias0"]["difference"] == pytest.approx(0.25)
    assert arms[1]["by_seat"][1] == {"seat": 1, "games": 2, "wins": 2}
