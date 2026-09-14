"""Rules tests for the variant-parameterized Can't Stop engine.

Run: python -m pytest games/cantstop/tests -q
"""

import random
from itertools import product

import pytest

from games.cantstop.engine import (
    ALL_RULESETS, COLUMN_HEIGHTS, COLUMNS, MAX_RUNNERS, GameState, Phase,
    RuleSet, apply_move, bust, can_stop, dice_pairings, legal_moves,
    random_dice, roll, stop, stop_blocked,
)

BASE2 = RuleSet.make(2)


def state_with(rules=BASE2, active=0, runners=None, progress=None,
               claimed=None, phase=Phase.AWAIT_DECISION):
    s = GameState(rules)
    s.active_player = active
    for p, cols in (progress or {}).items():
        s.progress[p].update(cols)
    for col, p in (claimed or {}).items():
        s.claimed_by[col] = p
    s.runners = dict(runners or {})
    s.phase = phase
    return s


# ---- Rule sets ----

class TestRuleSets:
    def test_ten_distinct_rule_sets(self):
        assert len(ALL_RULESETS) == 10
        assert len(set(ALL_RULESETS)) == 10

    @pytest.mark.parametrize("players, extended, expected", [
        (2, False, 3), (2, True, 5),
        (3, False, 3), (3, True, 4),
        (4, False, 3), (4, True, 3),
    ])
    def test_columns_to_win(self, players, extended, expected):
        assert RuleSet.make(players, extended).columns_to_win == expected

    def test_four_player_extended_equals_base(self):
        assert RuleSet.make(4, True, True) == RuleSet.make(4, False, True)

    @pytest.mark.parametrize("players, cols", [(1, 3), (5, 3), (2, 4), (3, 5)])
    def test_invalid_rule_sets_rejected(self, players, cols):
        with pytest.raises(ValueError):
            RuleSet(players, cols, False)


# ---- Dice ----

class TestDicePairings:
    def test_three_distinct_pairings(self):
        assert sorted(dice_pairings((1, 2, 3, 4))) == [(3, 7), (4, 6), (5, 5)]

    def test_all_same_face(self):
        assert dice_pairings((3, 3, 3, 3)) == [(6, 6)]

    def test_order_invariant(self):
        for dice in product(range(1, 7), repeat=4):
            assert (sorted(dice_pairings(dice))
                    == sorted(dice_pairings(tuple(sorted(dice)))))

    def test_random_dice_in_range(self):
        rng = random.Random(0)
        for _ in range(200):
            d = random_dice(rng)
            assert len(d) == 4 and all(1 <= x <= 6 for x in d)


# ---- Legal moves ----

class TestLegalMoves:
    def test_empty_board(self):
        s = state_with(phase=Phase.AWAIT_ROLL)
        assert legal_moves(s, (1, 2, 3, 4)) == [(3, 7), (4, 6), (5, 5)]

    def test_double_advances_two(self):
        s = state_with(phase=Phase.AWAIT_ROLL)
        assert (5, 5) in legal_moves(s, (1, 4, 2, 3))

    def test_double_with_one_space_left_is_single_step(self):
        s = state_with(progress={0: {2: 2}}, phase=Phase.AWAIT_ROLL)
        assert legal_moves(s, (1, 1, 1, 1)) == [(2,)]

    def test_claimed_column_unusable(self):
        s = state_with(claimed={7: 1}, phase=Phase.AWAIT_ROLL)
        assert legal_moves(s, (3, 4, 3, 4)) == [(6, 8)]

    def test_partial_when_one_column_claimed(self):
        s = state_with(claimed={7: 1}, phase=Phase.AWAIT_ROLL)
        # (1,6)+(2,5)=(7,7); (1,2)+(5,6)=(3,11); (1,5)+(2,6)=(6,8)
        assert legal_moves(s, (1, 2, 5, 6)) == [(3, 11), (6, 8)]
        s2 = state_with(claimed={3: 1}, phase=Phase.AWAIT_ROLL)
        assert (11,) in legal_moves(s2, (1, 2, 5, 6))

    def test_runner_at_top_blocks_column(self):
        s = state_with(runners={2: 3}, phase=Phase.AWAIT_DECISION)
        assert legal_moves(s, (1, 1, 1, 1)) == []

    def test_cap_blocks_new_column(self):
        s = state_with(runners={5: 1, 6: 1, 8: 1})
        assert legal_moves(s, (1, 1, 1, 1)) == []

    def test_cap_allows_existing_runner(self):
        s = state_with(runners={5: 1, 6: 1, 8: 1})
        assert (6, 8) in legal_moves(s, (3, 3, 4, 4))

    def test_cap_splits_two_new_columns_into_partials(self):
        # Two runners placed; (3,4) would need two new runners.
        s = state_with(runners={5: 1, 9: 1})
        moves = legal_moves(s, (1, 2, 1, 3))
        assert (3, 4) not in moves
        assert (3,) in moves and (4,) in moves

    def test_one_existing_one_new_at_two_runners(self):
        s = state_with(runners={3: 1, 9: 1})
        assert (3, 4) in legal_moves(s, (1, 2, 1, 3))

    def test_existing_runner_partial_when_other_is_new_at_cap(self):
        s = state_with(runners={3: 1, 9: 1, 10: 1})
        moves = legal_moves(s, (1, 2, 1, 3))
        assert (3,) in moves and (4,) not in moves

    def test_opponent_marker_does_not_restrict_moves(self):
        rules = RuleSet.make(2, blocking=True)
        s = state_with(rules, progress={1: {7: 1, 6: 1}}, phase=Phase.AWAIT_ROLL)
        assert (6, 7) in legal_moves(s, (3, 3, 3, 4))

    def test_bust_when_all_columns_blocked(self):
        s = state_with(runners={5: 1, 6: 1, 8: 1}, claimed={7: 1})
        assert legal_moves(s, (1, 1, 1, 1)) == []


# ---- Transitions ----

class TestTransitions:
    def test_roll_then_move_then_decision(self):
        s = GameState(BASE2)
        moves = roll(s, (4, 3, 2, 1))
        assert s.phase == Phase.AWAIT_MOVE and s.dice == (1, 2, 3, 4)
        apply_move(s, moves[0])
        assert s.phase == Phase.AWAIT_DECISION and s.dice is None

    def test_move_builds_on_saved_progress(self):
        s = state_with(progress={0: {7: 4}}, phase=Phase.AWAIT_ROLL)
        roll(s, (3, 4, 3, 4))
        apply_move(s, (7, 7))
        assert s.runners == {7: 6}
        assert s.progress[0][7] == 4

    def test_illegal_move_rejected(self):
        s = GameState(BASE2)
        roll(s, (1, 2, 3, 4))
        with pytest.raises(ValueError):
            apply_move(s, (2, 12))

    def test_phase_guards(self):
        s = GameState(BASE2)
        with pytest.raises(ValueError):
            stop(s)
        with pytest.raises(ValueError):
            apply_move(s, (7,))

    def test_busting_roll_passes_turn(self):
        s = state_with(runners={5: 3, 6: 1, 8: 1}, progress={0: {5: 1}})
        assert roll(s, (1, 1, 1, 1)) == []
        assert s.runners == {} and s.active_player == 1
        assert s.phase == Phase.AWAIT_ROLL
        assert s.progress[0][5] == 1

    def test_stop_saves_and_passes(self):
        s = state_with(runners={6: 2, 8: 1})
        stop(s)
        assert s.progress[0][6] == 2 and s.progress[0][8] == 1
        assert s.runners == {} and s.active_player == 1
        assert s.phase == Phase.AWAIT_ROLL

    def test_stop_claims_and_clears_other_markers(self):
        rules = RuleSet.make(3)
        s = state_with(rules, runners={2: 3}, progress={1: {2: 2}, 2: {2: 1}})
        stop(s)
        assert s.claimed_by[2] == 0
        assert all(prog[2] == 0 for prog in s.progress)

    def test_bust_keeps_saved_progress(self):
        s = state_with(runners={6: 5}, progress={0: {6: 3}})
        bust(s)
        assert s.progress[0][6] == 3 and s.runners == {}

    def test_clone_independent(self):
        s = state_with(runners={6: 2}, progress={0: {6: 1}})
        c = s.clone()
        c.runners[6] = 5
        c.progress[0][6] = 4
        c.claimed_by[7] = 1
        assert s.runners == {6: 2} and s.progress[0][6] == 1
        assert s.claimed_by[7] is None


# ---- Winning ----

class TestWinning:
    @pytest.mark.parametrize("rules", ALL_RULESETS, ids=repr)
    def test_win_exactly_at_threshold(self, rules):
        need = rules.columns_to_win
        tops = [2, 12, 3, 11, 4]
        claimed = {c: 0 for c in tops[:need - 1]}
        s = state_with(rules, claimed=claimed, runners={tops[need - 1]: COLUMN_HEIGHTS[tops[need - 1]]})
        stop(s)
        assert s.game_over and s.winner == 0

    @pytest.mark.parametrize("rules", ALL_RULESETS, ids=repr)
    def test_no_win_below_threshold(self, rules):
        need = rules.columns_to_win
        tops = [2, 12, 3, 11, 4]
        claimed = {c: 0 for c in tops[:need - 2]}
        s = state_with(rules, claimed=claimed, runners={tops[need - 2]: COLUMN_HEIGHTS[tops[need - 2]]})
        stop(s)
        assert not s.game_over and s.active_player == 1

    def test_opponent_claims_do_not_count(self):
        rules = RuleSet.make(2)
        s = state_with(rules, claimed={2: 1, 12: 1}, runners={3: 5})
        stop(s)
        assert not s.game_over


# ---- Turn order ----

class TestTurnOrder:
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_rotation(self, n):
        s = GameState(RuleSet.make(n))
        seen = []
        for _ in range(2 * n):
            seen.append(s.active_player)
            bust(s)
        assert seen == list(range(n)) * 2


# ---- Blocking ----

class TestBlocking:
    RULES = RuleSet.make(3, extended=True, blocking=True)

    def test_stop_illegal_on_opponent_marker(self):
        s = state_with(self.RULES, runners={7: 4}, progress={2: {7: 4}})
        assert stop_blocked(s) and not can_stop(s)
        with pytest.raises(ValueError):
            stop(s)

    def test_any_runner_blocks(self):
        s = state_with(self.RULES, runners={6: 1, 7: 4, 8: 2},
                       progress={1: {7: 4}})
        assert not can_stop(s)

    def test_own_saved_marker_does_not_block(self):
        s = state_with(self.RULES, runners={7: 4}, progress={0: {7: 4}})
        assert can_stop(s)

    def test_passing_marker_then_stop_is_legal(self):
        s = state_with(self.RULES, runners={7: 3}, progress={1: {7: 4}},
                       phase=Phase.AWAIT_ROLL)
        roll(s, (3, 4, 3, 4))
        apply_move(s, (7, 7))  # 3 -> 5, passing the marker on 4
        assert can_stop(s)

    def test_landing_on_marker_then_rolling_on_is_legal(self):
        s = state_with(self.RULES, runners={7: 3}, progress={1: {7: 4}},
                       phase=Phase.AWAIT_ROLL)
        roll(s, (1, 6, 2, 3))
        apply_move(s, (5, 7))  # land on the marker
        assert not can_stop(s)
        roll(s, (1, 6, 1, 1))  # (2, 7): advance off it
        apply_move(s, (2, 7))
        assert can_stop(s)

    def test_blocked_player_can_still_bust(self):
        s = state_with(self.RULES, runners={5: 2, 6: 1, 7: 4},
                       progress={1: {7: 4}})
        assert roll(s, (1, 1, 1, 1)) == []
        assert s.active_player == 1

    def test_no_block_without_variant(self):
        rules = RuleSet.make(3, extended=True, blocking=False)
        s = state_with(rules, runners={7: 4}, progress={1: {7: 4}})
        assert can_stop(s)
        stop(s)
        assert s.progress[0][7] == 4 == s.progress[1][7]

    def test_top_of_column_never_blocked(self):
        s = state_with(self.RULES, runners={2: 3}, progress={1: {2: 2}})
        assert can_stop(s)


# ---- Random play across every rule set ----

def play_random_game(rules, rng, stop_prob=0.3, max_steps=100_000):
    """Random legal play. Checks invariants after every transition."""
    s = GameState(rules)
    steps = 0
    while not s.game_over:
        steps += 1
        assert steps < max_steps, "game did not terminate"
        if s.phase == Phase.AWAIT_ROLL:
            roll(s, random_dice(rng))
        elif s.phase == Phase.AWAIT_MOVE:
            apply_move(s, rng.choice(legal_moves(s, s.dice)))
        elif can_stop(s) and rng.random() < stop_prob:
            stop(s)
        else:
            roll(s, random_dice(rng))
        check_invariants(s)
    return s


def check_invariants(s):
    rules = s.rules
    assert len(s.runners) <= MAX_RUNNERS
    for col, pos in s.runners.items():
        assert s.claimed_by[col] is None
        assert s.progress[s.active_player][col] < pos <= COLUMN_HEIGHTS[col]
    for col in COLUMNS:
        saved = [prog[col] for prog in s.progress]
        assert all(0 <= v < COLUMN_HEIGHTS[col] for v in saved)
        if s.claimed_by[col] is not None:
            assert not any(saved)
        if rules.blocking:
            nonzero = [v for v in saved if v]
            assert len(nonzero) == len(set(nonzero)), \
                f"saved markers share a space on column {col}: {saved}"
    if s.game_over:
        assert len(s.claimed_columns(s.winner)) >= rules.columns_to_win
        assert not s.runners
    else:
        for p in range(rules.num_players):
            assert len(s.claimed_columns(p)) < rules.columns_to_win


@pytest.mark.parametrize("rules", ALL_RULESETS, ids=repr)
def test_random_games_complete_with_invariants(rules):
    rng = random.Random(hash((rules.num_players, rules.columns_to_win,
                              rules.blocking)) & 0xFFFF)
    for _ in range(40):
        play_random_game(rules, rng)


def test_blocking_actually_forces_rolls():
    """Random play under blocking reaches blocked decisions (the rule is
    exercised, not vacuous)."""
    rng = random.Random(7)
    rules = RuleSet.make(3, extended=True, blocking=True)
    blocked = 0
    for _ in range(30):
        s = GameState(rules)
        while not s.game_over:
            if s.phase == Phase.AWAIT_ROLL:
                roll(s, random_dice(rng))
            elif s.phase == Phase.AWAIT_MOVE:
                apply_move(s, rng.choice(legal_moves(s, s.dice)))
            else:
                blocked += stop_blocked(s)
                if can_stop(s) and rng.random() < 0.3:
                    stop(s)
                else:
                    roll(s, random_dice(rng))
    assert blocked > 0
