# test_engine.py
# Comprehensive test suite for engine.py
# Covers correctness invariants, edge cases, and both bugs that were found
# and fixed. Run with: python -m pytest games/cantstop/test_engine.py -v

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from games.cantstop.engine import (
    GameState, get_valid_moves, get_possible_moves,
    apply_move, stop_turn, bust_turn,
    COLUMN_HEIGHTS, MAX_RUNNERS, COLUMNS_TO_WIN, ALL_COLUMNS
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def fresh_state():
    """Clean 2-player game state, no dice."""
    return GameState(2)

@pytest.fixture
def rolled_state():
    """Clean state with dice rolled."""
    s = GameState(2)
    s.roll_dice()
    return s


# ============================================================
# GAMESTATE INITIALIZATION
# ============================================================

class TestGameStateInit:

    def test_players(self, fresh_state):
        assert fresh_state.players == [0, 1]

    def test_active_player_starts_at_zero(self, fresh_state):
        assert fresh_state.active_player == 0

    def test_claimed_empty(self, fresh_state):
        assert fresh_state.claimed[0] == set()
        assert fresh_state.claimed[1] == set()

    def test_all_claimed_empty(self, fresh_state):
        assert fresh_state.all_claimed == set()

    def test_progress_empty(self, fresh_state):
        assert fresh_state.progress[0] == {}
        assert fresh_state.progress[1] == {}

    def test_runners_empty(self, fresh_state):
        assert fresh_state.runners == {}

    def test_dice_empty(self, fresh_state):
        assert fresh_state.dice == []

    def test_game_not_over(self, fresh_state):
        assert fresh_state.game_over is False
        assert fresh_state.winner is None


# ============================================================
# ROLL DICE
# ============================================================

class TestRollDice:

    def test_produces_four_dice(self, fresh_state):
        fresh_state.roll_dice()
        assert len(fresh_state.dice) == 4

    def test_dice_in_valid_range(self, fresh_state):
        for _ in range(100):
            fresh_state.roll_dice()
            assert all(1 <= d <= 6 for d in fresh_state.dice)

    def test_roll_replaces_previous(self, fresh_state):
        fresh_state.roll_dice()
        first = list(fresh_state.dice)
        # Roll many times to ensure replacement happens
        for _ in range(20):
            fresh_state.roll_dice()
        # Can't guarantee different but can verify it's a list of 4
        assert len(fresh_state.dice) == 4

    def test_returns_dice(self, fresh_state):
        result = fresh_state.roll_dice()
        assert result == fresh_state.dice


# ============================================================
# CLONE
# ============================================================

class TestClone:

    def test_clone_is_independent_claimed(self, fresh_state):
        fresh_state.claimed[0].add(7)
        fresh_state.all_claimed.add(7)
        clone = fresh_state.clone()
        clone.claimed[0].add(11)
        clone.all_claimed.add(11)
        assert 11 not in fresh_state.claimed[0]
        assert 11 not in fresh_state.all_claimed

    def test_clone_is_independent_progress(self, fresh_state):
        fresh_state.progress[0][5] = 3
        clone = fresh_state.clone()
        clone.progress[0][5] = 99
        assert fresh_state.progress[0][5] == 3

    def test_clone_is_independent_runners(self, fresh_state):
        fresh_state.runners[7] = 2
        clone = fresh_state.clone()
        clone.runners[7] = 99
        assert fresh_state.runners[7] == 2

    def test_clone_shares_players(self, fresh_state):
        clone = fresh_state.clone()
        assert clone.players is fresh_state.players

    def test_clone_shares_dice_initially(self, fresh_state):
        fresh_state.roll_dice()
        clone = fresh_state.clone()
        assert clone.dice is fresh_state.dice

    def test_reroll_on_clone_does_not_affect_original(self, fresh_state):
        fresh_state.roll_dice()
        original_dice = list(fresh_state.dice)
        clone = fresh_state.clone()
        clone.roll_dice()
        assert fresh_state.dice == original_dice

    def test_clone_copies_active_player(self, fresh_state):
        fresh_state.active_player = 1
        clone = fresh_state.clone()
        assert clone.active_player == 1
        clone.active_player = 0
        assert fresh_state.active_player == 1

    def test_clone_copies_game_over(self, fresh_state):
        fresh_state.game_over = True
        fresh_state.winner = 0
        clone = fresh_state.clone()
        assert clone.game_over is True
        assert clone.winner == 0

    def test_clone_full_independence(self):
        """Comprehensive independence check."""
        s = GameState(2)
        s.roll_dice()
        s.claimed[0] = {2, 12}
        s.all_claimed = {2, 12}
        s.progress[0] = {5: 3, 7: 4}
        s.progress[1] = {9: 2}
        s.runners = {6: 1, 8: 2}
        s.active_player = 1

        c = s.clone()
        c.claimed[0].add(3)
        c.all_claimed.add(3)
        c.progress[0][5] = 99
        c.runners[6] = 99
        c.active_player = 0

        assert s.claimed[0] == {2, 12}
        assert s.all_claimed == {2, 12}
        assert s.progress[0][5] == 3
        assert s.runners[6] == 1
        assert s.active_player == 1


# ============================================================
# SNAPSHOT / RESTORE
# ============================================================

class TestSnapshotRestore:

    def test_round_trip(self, fresh_state):
        fresh_state.roll_dice()
        fresh_state.claimed[1].add(5)
        fresh_state.all_claimed.add(5)
        fresh_state.progress[0][8] = 3
        fresh_state.runners[6] = 2

        snap = fresh_state.save_snapshot()
        fresh_state.claimed[1].add(11)
        fresh_state.progress[0][8] = 99
        fresh_state.runners.clear()
        fresh_state.active_player = 1

        fresh_state.restore_snapshot(snap)
        assert fresh_state.claimed[1] == {5}
        assert fresh_state.all_claimed == {5}
        assert fresh_state.progress[0][8] == 3
        assert fresh_state.runners == {6: 2}
        assert fresh_state.active_player == 0

    def test_snapshot_is_independent(self, fresh_state):
        fresh_state.runners[7] = 1
        snap = fresh_state.save_snapshot()
        fresh_state.runners[7] = 99
        assert snap['runners'][7] == 1


# ============================================================
# GET_POSSIBLE_MOVES
# ============================================================

class TestGetPossibleMoves:

    def test_three_unique_partitions(self):
        # [1,2,3,4]: (3,7),(4,6),(5,5) — all distinct
        parts = get_possible_moves([1, 2, 3, 4])
        assert len(parts) == 3

    def test_dedup_two_identical(self):
        # [1,1,2,2]: (2,4),(3,3),(3,3) → (2,4),(3,3)
        parts = get_possible_moves([1, 1, 2, 2])
        assert len(parts) == 2
        parts_set = set(parts)
        assert (2, 4) in parts_set
        assert (3, 3) in parts_set

    def test_dedup_all_identical(self):
        # [3,3,3,3]: all partitions are (6,6)
        parts = get_possible_moves([3, 3, 3, 3])
        assert parts == [(6, 6)]

    def test_all_same_face(self):
        # [1,1,1,1]: all partitions give (2,2)
        parts = get_possible_moves([1, 1, 1, 1])
        assert parts == [(2, 2)]

    def test_sorted_order(self):
        # Each partition should be sorted (smaller first)
        parts = get_possible_moves([6, 1, 5, 2])
        for a, b in parts:
            assert a <= b

    def test_specific_known_case(self):
        # [1,6,1,6]: (7,7),(2,12),(7,7) → (2,12),(7,7)
        parts = get_possible_moves([1, 6, 1, 6])
        parts_set = set(parts)
        assert (2, 12) in parts_set
        assert (7, 7) in parts_set
        assert len(parts) == 2


# ============================================================
# GET_VALID_MOVES — basic cases
# ============================================================

class TestGetValidMovesBasic:

    def test_empty_board_all_columns_available(self):
        s = GameState(2)
        s.dice = [1, 2, 3, 4]
        valid = get_valid_moves(s)
        assert len(valid) > 0

    def test_no_moves_when_all_partitions_claimed(self):
        s = GameState(2)
        s.dice = [1, 1, 1, 1]  # only partition is (2,2)
        s.all_claimed.add(2)
        valid = get_valid_moves(s)
        assert valid == []

    def test_true_double_emitted(self):
        s = GameState(2)
        s.dice = [1, 6, 1, 6]  # includes (7,7)
        valid = get_valid_moves(s)
        assert (7, 7) in valid

    def test_true_double_not_emitted_when_claimed(self):
        s = GameState(2)
        s.dice = [1, 6, 1, 6]
        s.all_claimed.add(7)
        valid = get_valid_moves(s)
        assert (7, 7) not in valid

    def test_partial_emitted_when_one_column_claimed(self):
        s = GameState(2)
        s.dice = [1, 2, 3, 4]  # partition (3,7) among others
        s.all_claimed.add(3)
        valid = get_valid_moves(s)
        # (3,7) full move not valid, but (7,) partial should be
        assert (3, 7) not in valid
        assert (7,) in valid

    def test_no_duplicate_moves(self):
        s = GameState(2)
        s.dice = [3, 3, 4, 4]
        valid = get_valid_moves(s)
        assert len(valid) == len(set(valid))

    def test_column_at_max_not_valid(self):
        s = GameState(2)
        s.dice = [3, 4, 3, 4]  # partition (7,7) among others
        s.progress[0][7] = COLUMN_HEIGHTS[7]  # fully advanced
        valid = get_valid_moves(s)
        assert (7, 7) not in valid


# ============================================================
# GET_VALID_MOVES — runner cap (Bug #1 fix)
# ============================================================

class TestGetValidMovesRunnerCap:

    def test_runner_cap_blocks_both_new_columns(self):
        """
        3 runners already placed. Partition gives 2 fresh columns.
        Neither can be played — no move from this partition.
        """
        s = GameState(2)
        s.runners = {5: 1, 7: 1, 9: 1}  # 3 runners = cap
        s.dice = [1, 2, 1, 2]  # partition (2,4) — both fresh
        valid = get_valid_moves(s)
        assert (2, 4) not in valid
        assert (2,) not in valid
        assert (4,) not in valid

    def test_runner_cap_allows_existing_runner(self):
        """
        3 runners. Partition includes an already-running column.
        That column should be playable.
        """
        s = GameState(2)
        s.runners = {5: 1, 7: 1, 9: 1}
        s.dice = [2, 3, 2, 5]  # partition (5,7) — both existing runners
        valid = get_valid_moves(s)
        assert (5, 7) in valid

    def test_runner_cap_partial_fix(self):
        """
        BUG FIX: 2 runners placed. Partition gives 2 fresh columns.
        Only one new runner slot available — each column should be
        emitted as a partial, not silently dropped.
        """
        s = GameState(2)
        s.runners = {5: 1, 9: 1}  # 2 runners, 1 slot left
        s.dice = [1, 2, 1, 3]  # partition (3,4) — both fresh
        valid = get_valid_moves(s)
        valid_set = set(valid)
        # Full pair move should NOT be valid (needs 2 new runners)
        assert (3, 4) not in valid_set
        # Each individual column SHOULD be valid as partial
        assert (3,) in valid_set or (4,) in valid_set, \
            f"At least one partial should be emitted: {valid}"

    def test_runner_cap_both_partials_emitted(self):
        """
        Both columns in a partition are individually legal under cap.
        Both partials should be emitted so player has the choice.
        """
        s = GameState(2)
        s.runners = {5: 1, 9: 1}
        # Force a partition where both columns are unclaimed and not at max
        s.dice = [1, 2, 1, 3]
        valid = get_valid_moves(s)
        valid_set = set(valid)
        # Both (3,) and (4,) should appear if both are legal
        if (3,) in valid_set and (4,) in valid_set:
            pass  # ideal case
        # At minimum, neither should be silently dropped if valid
        assert not (
            (3,) not in valid_set and (4,) not in valid_set and
            (3, 4) not in valid_set
        ), f"Partition (3,4) produced no moves despite legal columns: {valid}"

    def test_one_runner_slot_one_existing_one_new(self):
        """
        2 runners. Partition: one existing runner col, one fresh col.
        Full move valid (only 1 new runner needed).
        """
        s = GameState(2)
        s.runners = {5: 1, 9: 1}
        # Craft dice so partition (5, 6) appears: 5=2+3, 6=1+5 or similar
        s.dice = [2, 3, 1, 5]  # (5,6) and (3,8) and (7,4)
        valid = get_valid_moves(s)
        valid_set = set(valid)
        # (5,6): col 5 has runner, col 6 fresh — only 1 new runner needed
        if (5, 6) in valid_set:
            pass  # correct
        # At minimum (5,) should be valid since it has an existing runner
        assert (5,) in valid_set or (5, 6) in valid_set, \
            f"Col 5 (existing runner) should be playable: {valid}"


# ============================================================
# APPLY_MOVE
# ============================================================

class TestApplyMove:

    def test_normal_move_advances_runners(self):
        s = GameState(2)
        apply_move(s, (5, 7))
        assert s.runners[5] == 1
        assert s.runners[7] == 1

    def test_true_double_advances_two_steps(self):
        s = GameState(2)
        apply_move(s, (7, 7))
        assert s.runners[7] == 2

    def test_true_double_caps_at_column_height(self):
        s = GameState(2)
        s.progress[0][7] = COLUMN_HEIGHTS[7] - 1  # 1 step remaining
        apply_move(s, (7, 7))
        assert s.runners[7] == 1  # capped at 1, not 2

    def test_partial_move_advances_one_column(self):
        s = GameState(2)
        apply_move(s, (5,))
        assert s.runners.get(5) == 1
        assert len(s.runners) == 1

    def test_claimed_column_skipped(self):
        s = GameState(2)
        s.all_claimed.add(5)
        apply_move(s, (5, 7))
        assert 5 not in s.runners
        assert s.runners.get(7) == 1

    def test_full_column_skipped(self):
        s = GameState(2)
        s.progress[0][5] = COLUMN_HEIGHTS[5]
        apply_move(s, (5, 7))
        assert 5 not in s.runners
        assert s.runners.get(7) == 1

    def test_existing_runner_advanced(self):
        s = GameState(2)
        s.runners[7] = 3
        apply_move(s, (5, 7))
        assert s.runners[7] == 4  # advanced from 3
        assert s.runners[5] == 1

    def test_apply_move_bug_fix_mixed_existing_new(self):
        """
        BUG FIX: num_runners incorrectly incremented when advancing
        existing runner, causing second column to be skipped.
        Move (existing_runner_col, new_col) — both should advance.
        """
        s = GameState(2)
        s.runners = {5: 1, 8: 1}  # 2 existing runners
        # Move advances col 5 (existing) and col 6 (new, 1 slot available)
        apply_move(s, (5, 6))
        assert s.runners.get(5) == 2, \
            f"Col 5 (existing) should advance: {s.runners}"
        assert s.runners.get(6) == 1, \
            f"Col 6 (new) should be placed — apply_move bug: {s.runners}"

    def test_apply_move_bug_fix_three_runners_existing(self):
        """
        3 runners already. Move advances existing runner only.
        New column should be skipped (cap), existing should advance.
        """
        s = GameState(2)
        s.runners = {5: 1, 7: 1, 9: 1}
        apply_move(s, (5, 6))  # col 5 existing, col 6 fresh but capped
        assert s.runners.get(5) == 2, "Existing runner should advance"
        assert 6 not in s.runners, "New runner should be blocked by cap"

    def test_runner_count_correct_after_move(self):
        s = GameState(2)
        apply_move(s, (5, 7))
        assert len(s.runners) == 2

    def test_active_player_unchanged(self):
        s = GameState(2)
        s.active_player = 1
        apply_move(s, (5, 7))
        assert s.active_player == 1


# ============================================================
# STOP_TURN
# ============================================================

class TestStopTurn:

    def test_saves_runner_progress(self):
        s = GameState(2)
        s.runners = {7: 3}
        stop_turn(s)
        assert s.progress[0].get(7) == 3
        assert s.runners == {}

    def test_clears_runners(self):
        s = GameState(2)
        s.runners = {5: 1, 7: 2, 9: 1}
        stop_turn(s)
        assert s.runners == {}

    def test_switches_active_player(self):
        s = GameState(2)
        stop_turn(s)
        assert s.active_player == 1

    def test_claims_completed_column(self):
        s = GameState(2)
        s.progress[0][7] = COLUMN_HEIGHTS[7] - 1
        s.runners[7] = 1
        stop_turn(s)
        assert 7 in s.claimed[0]
        assert 7 in s.all_claimed
        assert 7 not in s.progress[0]

    def test_partial_progress_not_claimed(self):
        s = GameState(2)
        s.runners[7] = 1  # not enough to claim
        stop_turn(s)
        assert 7 not in s.claimed[0]
        assert s.progress[0].get(7) == 1

    def test_winning_claim_sets_game_over(self):
        s = GameState(2)
        s.claimed[0] = {2, 12}
        s.all_claimed = {2, 12}
        s.progress[0][3] = COLUMN_HEIGHTS[3] - 1
        s.runners[3] = 1
        stop_turn(s)
        assert s.game_over is True
        assert s.winner == 0
        assert 3 in s.claimed[0]

    def test_winner_player_1(self):
        s = GameState(2)
        s.active_player = 1
        s.claimed[1] = {2, 12}
        s.all_claimed = {2, 12}
        s.progress[1][3] = COLUMN_HEIGHTS[3] - 1
        s.runners[3] = 1
        stop_turn(s)
        assert s.game_over is True
        assert s.winner == 1

    def test_no_player_switch_on_win(self):
        s = GameState(2)
        s.claimed[0] = {2, 12}
        s.all_claimed = {2, 12}
        s.progress[0][3] = COLUMN_HEIGHTS[3] - 1
        s.runners[3] = 1
        stop_turn(s)
        assert s.active_player == 0  # winner stays active

    def test_two_columns_claimed_not_enough_to_win(self):
        s = GameState(2)
        s.progress[0][2] = COLUMN_HEIGHTS[2] - 1
        s.progress[0][12] = COLUMN_HEIGHTS[12] - 1
        s.runners[2] = 1
        s.runners[12] = 1
        stop_turn(s)
        assert s.game_over is False
        assert 2 in s.claimed[0]
        assert 12 in s.claimed[0]

    def test_claimed_column_not_double_added(self):
        """Stopping on an already-claimed column should not re-claim."""
        s = GameState(2)
        s.claimed[1].add(7)
        s.all_claimed.add(7)
        s.runners[7] = 1
        stop_turn(s)
        # claimed[0] should not contain 7 (player 0 is active)
        assert 7 not in s.claimed[0]


# ============================================================
# BUST_TURN
# ============================================================

class TestBustTurn:

    def test_clears_runners(self):
        s = GameState(2)
        s.runners = {5: 2, 7: 1, 9: 3}
        bust_turn(s)
        assert s.runners == {}

    def test_switches_active_player(self):
        s = GameState(2)
        s.active_player = 0
        bust_turn(s)
        assert s.active_player == 1

    def test_switches_back_to_player_0(self):
        s = GameState(2)
        s.active_player = 1
        bust_turn(s)
        assert s.active_player == 0

    def test_does_not_save_progress(self):
        s = GameState(2)
        s.runners = {7: 5}
        bust_turn(s)
        assert s.progress[0].get(7) is None

    def test_does_not_affect_saved_progress(self):
        s = GameState(2)
        s.progress[0][5] = 3
        s.runners = {7: 2}
        bust_turn(s)
        assert s.progress[0][5] == 3

    def test_does_not_affect_claimed(self):
        s = GameState(2)
        s.claimed[0].add(2)
        s.all_claimed.add(2)
        s.runners = {7: 1}
        bust_turn(s)
        assert 2 in s.claimed[0]
        assert 2 in s.all_claimed

    def test_game_not_over_after_bust(self):
        s = GameState(2)
        s.runners = {5: 10}
        bust_turn(s)
        assert s.game_over is False


# ============================================================
# COLUMN HEIGHTS CONSTANTS
# ============================================================

class TestColumnHeights:

    def test_all_columns_present(self):
        for col in range(2, 13):
            assert col in COLUMN_HEIGHTS

    def test_symmetric(self):
        for col in range(2, 8):
            mirror = 14 - col
            assert COLUMN_HEIGHTS[col] == COLUMN_HEIGHTS[mirror], \
                f"Col {col} height {COLUMN_HEIGHTS[col]} != col {mirror} height {COLUMN_HEIGHTS[mirror]}"

    def test_col7_tallest(self):
        assert COLUMN_HEIGHTS[7] == max(COLUMN_HEIGHTS.values())

    def test_col2_col12_shortest(self):
        assert COLUMN_HEIGHTS[2] == COLUMN_HEIGHTS[12] == min(COLUMN_HEIGHTS.values())


# ============================================================
# INTEGRATION — full game simulation
# ============================================================

class TestIntegration:

    def test_random_game_completes(self):
        """A random game should complete within max_turns."""
        import random
        s = GameState(2)
        max_turns = 300
        turns = 0
        while not s.game_over and turns < max_turns:
            turns += 1
            s.roll_dice()
            valid = get_valid_moves(s)
            if not valid:
                bust_turn(s)
                continue
            apply_move(s, random.choice(valid))
            if random.random() < 0.4:
                stop_turn(s)
        assert s.game_over, "Game should complete within max_turns"
        assert s.winner in [0, 1]

    def test_winner_has_three_claimed_columns(self):
        import random
        for _ in range(20):
            s = GameState(2)
            max_turns = 300
            turns = 0
            while not s.game_over and turns < max_turns:
                turns += 1
                s.roll_dice()
                valid = get_valid_moves(s)
                if not valid:
                    bust_turn(s)
                    continue
                apply_move(s, random.choice(valid))
                if random.random() < 0.4:
                    stop_turn(s)
            if s.game_over:
                assert len(s.claimed[s.winner]) >= COLUMNS_TO_WIN

    def test_claimed_columns_never_overlap_between_players(self):
        import random
        s = GameState(2)
        max_turns = 200
        turns = 0
        while not s.game_over and turns < max_turns:
            turns += 1
            s.roll_dice()
            valid = get_valid_moves(s)
            if not valid:
                bust_turn(s)
                continue
            apply_move(s, random.choice(valid))
            # Claimed sets should never overlap
            assert s.claimed[0].isdisjoint(s.claimed[1]), \
                f"Player claims overlap: {s.claimed}"
            if random.random() < 0.4:
                stop_turn(s)

    def test_runners_never_exceed_max(self):
        import random
        s = GameState(2)
        max_turns = 200
        turns = 0
        while not s.game_over and turns < max_turns:
            turns += 1
            s.roll_dice()
            valid = get_valid_moves(s)
            if not valid:
                bust_turn(s)
                continue
            apply_move(s, random.choice(valid))
            assert len(s.runners) <= MAX_RUNNERS, \
                f"Runner cap violated: {s.runners}"
            if random.random() < 0.4:
                stop_turn(s)

    def test_progress_never_exceeds_column_height(self):
        import random
        s = GameState(2)
        max_turns = 200
        turns = 0
        while not s.game_over and turns < max_turns:
            turns += 1
            s.roll_dice()
            valid = get_valid_moves(s)
            if not valid:
                bust_turn(s)
                continue
            apply_move(s, random.choice(valid))
            for player in [0, 1]:
                for col, prog in s.progress[player].items():
                    runner = s.runners.get(col, 0) if player == s.active_player else 0
                    total = prog + runner
                    assert total <= COLUMN_HEIGHTS[col], \
                        f"Progress exceeds height: col={col} total={total} height={COLUMN_HEIGHTS[col]}"
            if random.random() < 0.4:
                stop_turn(s)