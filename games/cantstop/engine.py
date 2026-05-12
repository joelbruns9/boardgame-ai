# engine.py
# Can't Stop game engine — performance optimized.
#
# Hot-path optimizations applied vs prior version:
# - clone() bypasses __init__ via __new__, avoiding wasted allocation.
# - clone() uses .copy() (set/dict native) instead of set(...)/dict(...)
#   constructors which are slower for known-type inputs.
# - clone() specializes the 2-player case: dict literal instead of
#   dict comprehension for `claimed` and `progress`.
# - clone() shares `players` and `dice` lists across clones — both are
#   effectively immutable (always reassigned wholesale, never mutated
#   in place). This eliminates 2 list allocations per clone.
# - roll_dice() uses random.choices on a precomputed tuple — single
#   C-level call instead of 4 separate randint() invocations.
# - get_valid_moves() local-binds hot dict/set attributes once.
# - get_valid_moves() builds results into a set directly, avoiding the
#   list(set(...)) round-trip at the end.
#
# Correctness fix:
# - get_valid_moves() now correctly enumerates partials when BOTH
#   partition columns are legal but the runner cap prevents playing
#   both as new runners. Previously the engine silently emitted no
#   move for this case, which is reachable in normal play (2 runners
#   already placed, partition gives 2 fresh columns). Each playable
#   column is now emitted as a partial.

import random

# ---- BOARD CONSTANTS ----
COLUMN_HEIGHTS = {
    2: 3,  3: 5,  4: 7,  5: 9,  6: 11,
    7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
}

COLUMNS_TO_WIN = 3
MAX_RUNNERS = 3
ALL_COLUMNS = list(COLUMN_HEIGHTS.keys())

# Precomputed for roll_dice — random.choices avoids the per-die call
# overhead of [random.randint(1,6) for _ in range(4)].
_DICE_FACES = (1, 2, 3, 4, 5, 6)


# ---- GAME STATE ----
class GameState:
    """
    Holds complete game state at one moment in time.
    Uses __slots__ for memory efficiency.
    Players are integers: 0, 1.
    """
    __slots__ = [
        'players',
        'active_player',
        'claimed',
        'all_claimed',
        'progress',
        'runners',
        'dice',
        'game_over',
        'winner',
    ]

    def __init__(self, num_players=2):
        self.players = list(range(num_players))
        self.active_player = 0
        self.claimed = {p: set() for p in self.players}
        self.all_claimed = set()
        self.progress = {p: {} for p in self.players}
        self.runners = {}
        self.dice = []
        self.game_over = False
        self.winner = None

    def roll_dice(self):
        """Roll 4 fresh dice. Only call at start of a new turn."""
        # random.choices is a single C-level call; ~2-3x faster than
        # [random.randint(1,6) for _ in range(4)] which has per-die
        # Python overhead.
        self.dice = random.choices(_DICE_FACES, k=4)
        return self.dice

    def get_current_progress(self, player, column):
        """
        Total progress on a column including active runner.
        Only adds runner if player is the active player.
        """
        saved = self.progress[player].get(column, 0)
        runner = self.runners.get(column, 0) if player == self.active_player else 0
        return saved + runner

    def save_snapshot(self):
        """Lightweight snapshot of mutable state."""
        return {
            'active_player': self.active_player,
            'claimed':       {p: cols.copy() for p, cols in self.claimed.items()},
            'all_claimed':   self.all_claimed.copy(),
            'progress':      {p: prog.copy() for p, prog in self.progress.items()},
            'runners':       self.runners.copy(),
            'dice':          self.dice,           # shared — never mutated in place
            'game_over':     self.game_over,
            'winner':        self.winner,
        }

    def restore_snapshot(self, snap):
        """Restore state from a snapshot."""
        self.active_player = snap['active_player']
        # Copies again on restore so the snapshot remains usable for
        # repeated restores (an idiom we rely on if MCTS ever uses
        # snapshot/restore in a try/finally pattern).
        self.claimed =     {p: cols.copy() for p, cols in snap['claimed'].items()}
        self.all_claimed = snap['all_claimed'].copy()
        self.progress =    {p: prog.copy() for p, prog in snap['progress'].items()}
        self.runners =     snap['runners'].copy()
        self.dice =        snap['dice']           # shared
        self.game_over =   snap['game_over']
        self.winner =      snap['winner']

    def clone(self):
        """
        Full independent copy. Used by MCTS on every chance-node
        traversal and decision-node expansion, so it's a hot path.

        Optimizations vs naive clone:
          - Bypass __init__ (skips the empty dict/set/list allocations
            that would immediately be overwritten anyway).
          - Use .copy() on sets and dicts (faster than set(s)/dict(d)
            constructors for known-type inputs).
          - For the 2-player common case, use a dict literal instead
            of a comprehension over self.players.
          - Share `players` and `dice` lists (both effectively
            immutable — always reassigned, never mutated in place).
        """
        new = GameState.__new__(GameState)
        new.players = self.players              # shared (immutable)
        new.active_player = self.active_player

        if len(self.players) == 2:
            # Fast path: dict literal beats comprehension for tiny dicts.
            c = self.claimed
            p = self.progress
            new.claimed = {0: c[0].copy(), 1: c[1].copy()}
            new.progress = {0: p[0].copy(), 1: p[1].copy()}
        else:
            new.claimed  = {p: cols.copy() for p, cols in self.claimed.items()}
            new.progress = {p: prog.copy() for p, prog in self.progress.items()}

        new.all_claimed = self.all_claimed.copy()
        new.runners = self.runners.copy()
        new.dice = self.dice                    # shared (immutable in practice)
        new.game_over = self.game_over
        new.winner = self.winner
        return new

    def heuristic_value(self, player):
        """
        Estimate win probability when game doesn't finish
        within max_turns. Returns float 0.0 to 1.0.
        Uses relative advantage — empty board returns 0.5.
        """
        opponent = 1 - player

        my_claimed  = len(self.claimed[player])
        opp_claimed = len(self.claimed[opponent])
        total_claimed = my_claimed + opp_claimed

        claimed_score = 0.5 if total_claimed == 0 else my_claimed / total_claimed

        my_progress = sum(
            steps / COLUMN_HEIGHTS[col]
            for col, steps in self.progress[player].items()
        )
        opp_progress = sum(
            steps / COLUMN_HEIGHTS[col]
            for col, steps in self.progress[opponent].items()
        )
        total_progress = my_progress + opp_progress

        progress_score = 0.5 if total_progress == 0 else my_progress / total_progress

        return 0.8 * claimed_score + 0.2 * progress_score

    def __repr__(self):
        lines = [
            f"\n=== Can't Stop ===",
            f"Active player: {self.active_player}",
            f"Dice: {self.dice}",
            f"Runners: {self.runners}",
        ]
        for p in self.players:
            lines.append(
                f"Player {p} - claimed: {sorted(self.claimed[p])} "
                f"| progress: {self.progress[p]}"
            )
        if self.game_over:
            lines.append(f"GAME OVER - Winner: Player {self.winner}")
        return "\n".join(lines)


# ---- MOVE CALCULATION ----
def get_possible_moves(dice):
    """All unique normalized column pairs from 4 dice."""
    d0, d1, d2, d3 = dice[0], dice[1], dice[2], dice[3]
    # Three pairings of 4 dice into 2 pairs each.
    # Sort each pair so we can dedup as sorted tuples.
    s1a, s1b = d0 + d1, d2 + d3
    s2a, s2b = d0 + d2, d1 + d3
    s3a, s3b = d0 + d3, d1 + d2
    p1 = (s1a, s1b) if s1a <= s1b else (s1b, s1a)
    p2 = (s2a, s2b) if s2a <= s2b else (s2b, s2a)
    p3 = (s3a, s3b) if s3a <= s3b else (s3b, s3a)
    # Manual dedup is faster than list(set([...])) for 3 elements.
    if p1 == p2:
        return [p1] if p1 == p3 else [p1, p3]
    if p1 == p3:
        return [p1, p2]
    if p2 == p3:
        return [p1, p2]
    return [p1, p2, p3]


def get_valid_moves(state):
    """
    Filter possible moves to legal ones.

    Move formats returned:
        (a, b)  — normal move, two different columns
        (a, a)  — true double from dice, same column twice
        (a,)    — partial move, one column blocked/full

    Correctness note: when a partition gives two distinct columns that
    are both legal individually but the runner cap prevents using both
    as fresh new columns, each column is emitted as a separate partial.
    The player chooses which to play.
    """
    # Local-bind hot lookups.
    possible = get_possible_moves(state.dice)
    player = state.active_player
    all_claimed = state.all_claimed
    runners = state.runners
    progress_p = state.progress[player]
    num_runners = len(runners)
    heights = COLUMN_HEIGHTS

    # Build into a set so we never emit duplicates. (Concrete instances
    # of duplicates from distinct partitions seem unreachable in practice
    # but the set guarantees the invariant cheaply.)
    valid = set()
    valid_add = valid.add  # micro-opt: local-bind .add for tight loop

    for col_a, col_b in possible:

        if col_a == col_b:
            # True double from dice.
            col = col_a
            if col in all_claimed:
                continue
            current = progress_p.get(col, 0) + runners.get(col, 0)
            if current >= heights[col]:
                continue
            if col not in runners and num_runners >= MAX_RUNNERS:
                continue
            valid_add((col, col))
            continue

        # Two different columns in this partition.
        a_curr = progress_p.get(col_a, 0) + runners.get(col_a, 0)
        b_curr = progress_p.get(col_b, 0) + runners.get(col_b, 0)
        a_in_runners = col_a in runners
        b_in_runners = col_b in runners

        can_a = (col_a not in all_claimed and a_curr < heights[col_a])
        can_b = (col_b not in all_claimed and b_curr < heights[col_b])

        # Apply runner-cap to each column individually.
        if can_a and not a_in_runners and num_runners >= MAX_RUNNERS:
            can_a = False
        if can_b and not b_in_runners and num_runners >= MAX_RUNNERS:
            can_b = False

        if not can_a and not can_b:
            continue

        # Determine whether we can play BOTH as a full pair-move.
        # New runners needed for the pair = number of these columns
        # that aren't already in runners.
        new_needed_for_both = (
            (0 if a_in_runners else 1) + (0 if b_in_runners else 1)
        )
        can_play_both = (
            can_a and can_b and
            num_runners + new_needed_for_both <= MAX_RUNNERS
        )

        if can_play_both:
            valid_add((col_a, col_b))
        else:
            # Emit each individually-legal column as a partial.
            # CORRECTNESS FIX: previously, if both columns were
            # individually legal but the cap blocked using BOTH as new
            # runners, nothing was emitted — the player was incorrectly
            # given no move for the partition.
            if can_a:
                valid_add((col_a,))
            if can_b:
                valid_add((col_b,))

    return list(valid)


def apply_move(state, move):
    """
    Advance runners based on chosen move.

    Move formats:
        (6, 8)  — normal: advance columns 6 and 8 by 1 each
        (7, 7)  — true double: advance column 7 by 2 steps
        (6,)    — partial: advance column 6 by 1 step only

    Defensive: if the caller somehow passes an illegal move (blocked
    column, full column, or runner-cap violation), the move is
    silently no-oped. MCTS only applies moves from get_valid_moves so
    this branch shouldn't fire in normal use.
    """
    player = state.active_player
    all_claimed = state.all_claimed
    runners = state.runners
    progress_p = state.progress[player]
    num_runners = len(runners)
    heights = COLUMN_HEIGHTS

    # ---- PARTIAL MOVE ----
    if len(move) == 1:
        col = move[0]
        if col in all_claimed:
            return state
        current = progress_p.get(col, 0) + runners.get(col, 0)
        if current >= heights[col]:
            return state
        if col not in runners and num_runners >= MAX_RUNNERS:
            return state
        runners[col] = runners.get(col, 0) + 1
        return state

    col_a, col_b = move

    # ---- TRUE DOUBLE MOVE ----
    if col_a == col_b:
        col = col_a
        if col in all_claimed:
            return state
        current = progress_p.get(col, 0) + runners.get(col, 0)
        if current >= heights[col]:
            return state
        if col not in runners and num_runners >= MAX_RUNNERS:
            return state
        steps = min(2, heights[col] - current)
        runners[col] = runners.get(col, 0) + steps

    # ---- NORMAL MOVE ----
    else:
        for col in (col_a, col_b):
            if col in all_claimed:
                continue
            current = progress_p.get(col, 0) + runners.get(col, 0)
            if current >= heights[col]:
                continue
            if col not in runners and num_runners >= MAX_RUNNERS:
                continue
            runners[col] = runners.get(col, 0) + 1
            num_runners += 1

    return state


def stop_turn(state):
    """
    Save runner progress. Claim completed columns.
    Update all_claimed incrementally. Pass to next player.
    """
    player = state.active_player
    heights = COLUMN_HEIGHTS
    all_claimed = state.all_claimed
    claimed_p = state.claimed[player]
    progress_p = state.progress[player]

    for col, runner_steps in state.runners.items():
        saved = progress_p.get(col, 0)
        new_pos = saved + runner_steps

        if new_pos >= heights[col] and col not in all_claimed:
            claimed_p.add(col)
            all_claimed.add(col)
            progress_p.pop(col, None)

            if len(claimed_p) >= COLUMNS_TO_WIN:
                state.game_over = True
                state.winner = player
        else:
            progress_p[col] = new_pos

    state.runners = {}

    if not state.game_over:
        state.active_player = (state.active_player + 1) % len(state.players)

    return state


def bust_turn(state):
    """Lose runners. Pass to next player."""
    state.runners = {}
    state.active_player = (state.active_player + 1) % len(state.players)
    return state


def _next_player(state):
    """Rotate to next player. Kept for backward compatibility."""
    state.active_player = (state.active_player + 1) % len(state.players)


# ---- SELF-TEST ----
# Complementary to the project's existing 39-test suite — focuses on
# correctness invariants and the bug fix, plus a perf benchmark.

if __name__ == "__main__":
    import time
    import copy

    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    # ---- Test 1: clone() is fully independent ----
    s = GameState(2)
    s.roll_dice()
    s.claimed[0].add(7)
    s.all_claimed.add(7)
    s.progress[0][5] = 2
    s.runners[3] = 1
    s.active_player = 1

    c = s.clone()
    # Mutate clone — original must be unchanged.
    c.claimed[0].add(11)
    c.all_claimed.add(11)
    c.progress[0][5] = 99
    c.runners[3] = 99
    c.runners[6] = 5
    c.active_player = 0

    assert s.claimed[0] == {7}, f"clone leaked claimed: {s.claimed[0]}"
    assert s.all_claimed == {7}, f"clone leaked all_claimed: {s.all_claimed}"
    assert s.progress[0][5] == 2, f"clone leaked progress: {s.progress[0]}"
    assert s.runners == {3: 1}, f"clone leaked runners: {s.runners}"
    assert s.active_player == 1
    print("  [1] clone() produces fully independent copy: PASS")

    # ---- Test 2: players list is shared (safe — never mutated) ----
    s2 = GameState(2)
    c2 = s2.clone()
    assert c2.players is s2.players, "players should be shared"
    print("  [2] clone() shares players list (safe, immutable): PASS")

    # ---- Test 3: dice list is shared (safe — never mutated in place) ----
    s3 = GameState(2)
    s3.roll_dice()
    c3 = s3.clone()
    assert c3.dice is s3.dice, "dice should be shared"
    # Both should still see the same dice
    assert c3.dice == s3.dice
    # Re-roll: should not affect parent
    c3.roll_dice()
    assert c3.dice is not s3.dice, "re-roll should swap, not mutate"
    print("  [3] clone() shares dice safely (re-roll swaps, doesn't mutate): PASS")

    # ---- Test 4: roll_dice produces 4 dice in 1..6 ----
    s4 = GameState(2)
    for _ in range(100):
        s4.roll_dice()
        assert len(s4.dice) == 4
        assert all(1 <= d <= 6 for d in s4.dice)
    print("  [4] roll_dice() always produces 4 dice in 1..6: PASS")

    # ---- Test 5: get_valid_moves — runner-cap correctness fix ----
    # 2 runners already on cols 5, 9. Dice produce partition (3,4) where
    # both columns are legal but cap = 3 prevents using both as new.
    # Both partials (3,) and (4,) should be emitted.
    s5 = GameState(2)
    s5.runners = {5: 1, 9: 1}
    s5.dice = [1, 2, 1, 3]   # sums in partitions:
    # (a+b, c+d) = (3, 4)
    # (a+c, b+d) = (2, 5)
    # (a+d, b+c) = (4, 3) → (3, 4)
    # So partition (3, 4) appears — both new columns, runner cap = 3.
    valid = get_valid_moves(s5)
    valid_set = set(valid)
    print(f"  [5] State: 2 runners {dict(s5.runners)}, dice {s5.dice}")
    print(f"      Valid moves: {sorted(valid)}")
    # Should NOT contain (3, 4) — would need 4 runners.
    assert (3, 4) not in valid_set, \
        f"Should not allow full move (3,4) with cap: {valid}"
    # SHOULD contain (3,) and (4,) — the correctness fix.
    assert (3,) in valid_set, \
        f"Missing partial (3,) — the correctness bug: {valid}"
    assert (4,) in valid_set, \
        f"Missing partial (4,) — the correctness bug: {valid}"
    print("  [5] Partial emission under runner cap (correctness fix): PASS")

    # ---- Test 6: get_valid_moves — basic cases still correct ----
    s6 = GameState(2)
    s6.dice = [1, 1, 6, 6]   # partitions: (2,12), (7,7), (7,7) → dedup (2,12), (7,7)
    valid = get_valid_moves(s6)
    valid_set = set(valid)
    print(f"  [6] Dice (1,1,6,6) → partitions (2,12), (7,7)")
    print(f"      Valid: {sorted(valid)}")
    assert (2, 12) in valid_set
    assert (7, 7) in valid_set
    print("  [6] Basic get_valid_moves output: PASS")

    # ---- Test 7: True double respects column height ----
    s7 = GameState(2)
    s7.progress[0][7] = 12  # height 13; only 1 step remaining
    s7.dice = [1, 6, 1, 6]  # partition (7, 7)
    valid = get_valid_moves(s7)
    assert (7, 7) in valid, f"Should be valid even with 1 step left: {valid}"
    # Apply and verify it advances by 1 (capped) not 2.
    apply_move(s7, (7, 7))
    assert s7.runners[7] == 1, \
        f"Double should cap at remaining height: runner={s7.runners[7]}"
    print("  [7] True double caps at remaining column height: PASS")

    # ---- Test 8: stop_turn claims columns and triggers winner ----
    s8 = GameState(2)
    s8.claimed[0] = {2, 12}
    s8.all_claimed = {2, 12}
    s8.progress[0][3] = 4
    s8.runners[3] = 1
    s8.active_player = 0
    stop_turn(s8)
    assert 3 in s8.claimed[0], f"col 3 should be claimed: {s8.claimed[0]}"
    assert s8.game_over, "Game should be over"
    assert s8.winner == 0, f"Winner should be 0: {s8.winner}"
    print("  [8] stop_turn detects winning claim: PASS")

    # ---- Test 9: bust_turn clears runners and switches player ----
    s9 = GameState(2)
    s9.runners = {5: 2, 7: 1}
    s9.active_player = 0
    bust_turn(s9)
    assert s9.runners == {}, "Runners should be cleared"
    assert s9.active_player == 1, "Player should rotate"
    print("  [9] bust_turn clears runners and rotates player: PASS")

    # ---- Test 10: get_possible_moves dedups partitions ----
    parts = get_possible_moves([3, 3, 4, 4])
    # Pairings: (6, 8), (7, 7), (7, 7) → dedup (6,8) and (7,7)
    assert sorted(parts) == [(6, 8), (7, 7)], f"Bad dedup: {parts}"
    print(" [10] get_possible_moves dedups correctly: PASS")

    # ---- Test 11: Snapshot/restore round-trip ----
    s11 = GameState(2)
    s11.roll_dice()
    s11.claimed[1].add(5)
    s11.all_claimed.add(5)
    s11.progress[0][8] = 3
    s11.runners[6] = 2
    snap = s11.save_snapshot()

    s11.claimed[1].add(11)
    s11.all_claimed.add(11)
    s11.progress[0][8] = 99
    s11.runners.clear()
    s11.active_player = 1

    s11.restore_snapshot(snap)
    assert s11.claimed[1] == {5}, f"snap restore claimed: {s11.claimed[1]}"
    assert s11.all_claimed == {5}, f"snap restore all_claimed: {s11.all_claimed}"
    assert s11.progress[0][8] == 3
    assert s11.runners == {6: 2}
    assert s11.active_player == 0
    print(" [11] snapshot/restore round-trip: PASS")

    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    state = GameState(2)
    state.roll_dice()

    N = 10000
    start = time.time()
    for _ in range(N):
        cloned = state.clone()
    clone_time = time.time() - start

    start = time.time()
    for _ in range(N):
        snap = state.save_snapshot()
        state.restore_snapshot(snap)
    snap_time = time.time() - start

    start = time.time()
    for _ in range(N):
        copied = copy.deepcopy(state)
    deep_time = time.time() - start

    print(f"  {N:,} iterations:")
    print(f"    deepcopy:         {deep_time*1000:7.1f} ms")
    print(f"    clone():          {clone_time*1000:7.1f} ms "
          f"({deep_time/clone_time:.1f}x faster)")
    print(f"    snapshot/restore: {snap_time*1000:7.1f} ms "
          f"({deep_time/snap_time:.1f}x faster)")

    # roll_dice benchmark
    N_DICE = 200_000
    start = time.time()
    for _ in range(N_DICE):
        state.roll_dice()
    roll_time = time.time() - start
    print(f"\n  roll_dice × {N_DICE:,}: {roll_time*1000:.1f} ms "
          f"({N_DICE / roll_time / 1000:.0f}k rolls/s)")

    # get_valid_moves benchmark — typical mid-game state
    state.runners = {5: 1, 8: 2}
    state.progress[0] = {3: 2, 11: 1}
    state.all_claimed = {2, 12}
    state.claimed[0] = {2}
    state.claimed[1] = {12}
    state.roll_dice()
    N_VALID = 50_000
    start = time.time()
    for _ in range(N_VALID):
        v = get_valid_moves(state)
    valid_time = time.time() - start
    print(f"  get_valid_moves × {N_VALID:,}: {valid_time*1000:.1f} ms "
          f"({N_VALID / valid_time / 1000:.0f}k calls/s)")

    print("\n  Simulating 1000 random games...")
    start = time.time()
    winners = {0: 0, 1: 0}
    for _ in range(1000):
        state = GameState(2)
        turns = 0
        while not state.game_over and turns < 150:
            turns += 1
            state.roll_dice()
            valid = get_valid_moves(state)
            if not valid:
                bust_turn(state)
                continue
            apply_move(state, random.choice(valid))
            if random.random() < 0.5:
                stop_turn(state)
        if state.winner is not None:
            winners[state.winner] += 1

    elapsed = time.time() - start
    print(f"    Time: {elapsed:.2f}s")
    print(f"    Games/second: {1000/elapsed:.0f}")
    print(f"    Winners: {winners}")