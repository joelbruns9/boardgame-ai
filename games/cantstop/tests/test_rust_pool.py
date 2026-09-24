"""M3 gate: the Rust self-play pool plays exactly the Python games.

Every field of every ``GameResult`` -- training rows byte for byte -- must
match ``self_play.play_game`` with the same per-game seed, and must not
depend on thread count or on how many games share a pool. The full-size
gate and the benchmark are ``python -m games.cantstop.rust_pool_equiv``.
Skipped, not failed, when the extension is not built.

Run: python -m pytest games/cantstop/tests/test_rust_pool.py -q
"""

import numpy as np
import pytest

from games.cantstop import arena, train
from games.cantstop.encoder import decode_features, encode_batch
from games.cantstop.engine import (ALL_RULESETS, GameState, RuleSet,
                                   apply_move, can_stop, random_dice, roll,
                                   stop)
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool import (game_seeds, play_match, run_pool,
                                      values_for)
from games.cantstop.rust_pool_equiv import (HashedMock, MoverWinsMock,
                                            compare_results, gate)
from games.cantstop.self_play import TurnLimitExceeded
from games.cantstop.snapshot import snapshot
from games.cantstop.solver import ProgressHeuristic

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")


def seeds(n, s=3):
    return game_seeds(PortableRng(s), n)


# ---- the pool is the Python game ----

@pytest.mark.parametrize("rules", [RuleSet.make(2),
                                   RuleSet.make(3, extended=True,
                                                blocking=True),
                                   RuleSet.make(4, blocking=True)], ids=str)
def test_pool_plays_the_python_games(rules):
    gate([rules, rules], seeds(2), "hashed")


def test_pool_plays_the_python_games_on_a_rounding_error():
    """mover_wins puts every stop-or-roll choice within ~1e-15 of a tie, so
    a single differently-rounded backup changes the game."""
    gate([RuleSet.make(2, blocking=True)], seeds(1, 9), "mover_wins")


# ---- results do not depend on how the pool runs ----

@pytest.fixture(scope="module")
def mixed_schedule():
    rules_list = [r for r in ALL_RULESETS for _ in range(3)]
    return rules_list, seeds(len(rules_list), 17)


@pytest.fixture(scope="module")
def all_at_once(mixed_schedule):
    rules_list, s = mixed_schedule
    return run_pool(rules_list, s, [HashedMock()], threads=0, in_flight=0)


def test_thread_count_does_not_change_any_game(mixed_schedule, all_at_once):
    rules_list, s = mixed_schedule
    one = run_pool(rules_list, s, [HashedMock()], threads=1)
    for i, (a, b) in enumerate(zip(one, all_at_once)):
        compare_results(a, b, f"game {i}")


def test_games_in_flight_do_not_change_any_game(mixed_schedule, all_at_once):
    """Each game alone in its own pool == all 30 sharing one: batching mixes
    rule sets and seats in one buffer and must not leak between games."""
    rules_list, s = mixed_schedule
    for i in range(0, len(rules_list), 4):
        alone = run_pool([rules_list[i]], [s[i]], [HashedMock()])[0]
        compare_results(alone, all_at_once[i], f"game {i}")


@pytest.mark.parametrize("in_flight", [1, 4, 7])
def test_refilling_a_small_window_does_not_change_any_game(
        mixed_schedule, all_at_once, in_flight):
    """Games queue behind a small in-flight window and start as others
    finish; each must still be exactly its own game."""
    rules_list, s = mixed_schedule
    got = run_pool(rules_list, s, [HashedMock()], in_flight=in_flight)
    for i, (a, b) in enumerate(zip(got, all_at_once)):
        compare_results(a, b, f"game {i} in_flight={in_flight}")


def test_mixed_rule_sets_all_finish(all_at_once):
    assert all(r.winner >= 0 for r in all_at_once)
    assert all(len(r) == r.turns - 1 for r in all_at_once)


# ---- backends agree through the public entry points ----

def test_generate_backends_agree():
    rule_sets = [RuleSet.make(2)]
    py = train.generate(rule_sets, 1, HashedMock(), PortableRng(4),
                        backend="python")
    rs = train.generate(rule_sets, 1, HashedMock(), PortableRng(4),
                        backend="rust")
    compare_results(py[0], rs[0], "generate")


def test_arena_backends_agree():
    """Two different evaluators, seats rotated: the pool must score each
    seat's turns with that seat's evaluator and credit the player."""
    rules = RuleSet.make(2)
    players = [HashedMock(), MoverWinsMock()]
    py = arena.play_match(rules, players, 2, PortableRng(8), backend="python")
    rs = arena.play_match(rules, players, 2, PortableRng(8), backend="rust")
    assert py == rs


def test_arena_shares_one_slot_per_distinct_evaluator():
    """compare() seats n-1 copies of the incumbent; they must share one
    forward per round, not get one each."""
    calls = []

    class Counting(HashedMock):
        def relative_probs(self, features):
            calls.append(len(features))
            return np.full((len(features), 4), 0.25, dtype=np.float32)

    incumbent = Counting()
    from games.cantstop.rust_pool import PoolStats
    stats = PoolStats()
    play_match(RuleSet.make(4), [HashedMock(), incumbent, incumbent,
                                 incumbent], 4, PortableRng(1), stats=stats)
    assert len(calls) <= stats.rounds


# ---- the heuristic scores features exactly as it scores boards ----

def _boards_from_games():
    boards = []
    for i, rules in enumerate(ALL_RULESETS):
        rng = PortableRng(100 + i)
        state = GameState(rules)
        while not state.game_over:
            moves = roll(state, random_dice(rng))
            if moves:
                apply_move(state, moves[rng.randrange(len(moves))])
                if can_stop(state) and rng.next_float() < 0.35:
                    stop(state)
            if state.phase == 0 and not state.game_over:
                boards.append(state.clone())
    return boards


def test_decode_inverts_encode_exactly():
    boards = _boards_from_games()
    feats = encode_batch(boards)
    for b, f in zip(boards, feats):
        back = decode_features(f, b.active_player)
        assert snapshot(back) == snapshot(b)


def test_heuristic_scores_features_exactly_as_boards():
    h = ProgressHeuristic()
    boards = [b for b in _boards_from_games() if b.rules == ALL_RULESETS[5]]
    by_active = {}
    for b in boards:
        by_active.setdefault(b.active_player, []).append(b)
    for active, group in by_active.items():
        ref = group[0]
        assert np.array_equal(h.evaluate_features(encode_batch(group), ref),
                              h(group))


# ---- failure modes ----

def test_turn_limit_raises_like_python():
    with pytest.raises(TurnLimitExceeded):
        run_pool([RuleSet.make(2)], seeds(1), [HashedMock()], max_turns=3)


def test_resume_checks_the_value_count():
    pool = rust.SelfPlayPool([(snapshot(GameState(RuleSet.make(2))), 1,
                               [0, 0])], 400)
    with pytest.raises(RuntimeError, match="pending"):
        pool.resume(b"")
    pool.advance()
    pool.pending()
    with pytest.raises(ValueError, match="expected"):
        pool.resume(np.zeros(3, dtype="<f8").tobytes())


def test_seat_count_must_match_players():
    with pytest.raises(ValueError, match="evaluator seats"):
        rust.SelfPlayPool([(snapshot(GameState(RuleSet.make(3))), 1,
                            [0, 0])], 400)


def test_values_for_rejects_a_wrong_shape():
    rules = RuleSet.make(2)

    class Bad:
        def evaluate_features(self, features, reference):
            return np.zeros((len(features), 3))

    feats = np.zeros((2, 98), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        values_for([Bad()], feats, [(0, 2, 0, 2, 1)], [rules])


# ---- the relative-output fast path ----

class RelativeMock(HashedMock):
    """A 'net': per-row hashed seat-relative probabilities, float32, masked
    to live seats. Has both entry points, so the same games can be played
    through the batched relative path and the absolute path. ``salt`` makes
    distinct 'nets'."""

    def __init__(self, salt=b""):
        self.salt = salt

    def relative_probs(self, features):
        from games.cantstop.encoder import seat_mask
        import zlib
        out = np.zeros((len(features), 4), dtype=np.float32)
        live = seat_mask(features)
        for i, f in enumerate(features):
            rng = PortableRng(zlib.crc32(self.salt + f.tobytes()))
            v = np.array([rng.next_float() + 1e-3 for _ in range(4)])
            v = v * live[i]
            out[i] = v / v.sum()
        return out

    def evaluate_features(self, features, reference):
        from games.cantstop.encoder import to_absolute
        return to_absolute(self.relative_probs(features), reference)


class AbsoluteOnly:
    """The same evaluator with the batched entry hidden."""

    def __init__(self, inner):
        self.inner = inner

    def evaluate_features(self, features, reference):
        return self.inner.evaluate_features(features, reference)


def test_rust_rotation_is_to_absolute_bit_for_bit():
    from games.cantstop.encoder import FEATURE_SIZE, to_absolute
    rules_list = [r for r in ALL_RULESETS for _ in range(2)]
    pool = rust.SelfPlayPool(
        [(snapshot(GameState(r)), s, [0] * r.num_players)
         for r, s in zip(rules_list, seeds(len(rules_list), 5))], 400)
    pool.advance()
    raw, blocks = pool.pending()
    feats = np.frombuffer(raw, dtype="<f4").reshape(-1, FEATURE_SIZE)
    rel = RelativeMock().relative_probs(feats)
    got = np.frombuffer(pool.absolute_from_relative(rel.tobytes()), "<f8")
    starts = np.cumsum([0] + [b[1] for b in blocks])
    want = []
    for i, (game, rows, _ev, n, active) in enumerate(blocks):
        ref = GameState(rules_list[game])
        ref.active_player = active
        want.append(to_absolute(rel[starts[i]:starts[i + 1]], ref).ravel())
    assert got.tobytes() == np.concatenate(want).astype("<f8").tobytes()


def test_relative_and_absolute_paths_play_the_same_games():
    rules_list = [RuleSet.make(3, blocking=True), RuleSet.make(2),
                  RuleSet.make(4)]
    s = seeds(3, 12)
    ev = RelativeMock()
    fast = run_pool(rules_list, s, [ev])
    slow = run_pool(rules_list, s, [AbsoluteOnly(ev)])
    for i, (a, b) in enumerate(zip(fast, slow)):
        compare_results(a, b, f"game {i}")


def test_arena_relative_path_regroups_two_nets():
    """Two batched evaluators in one pool: rows are split by evaluator for
    the forwards and must land back in block order."""
    rules = RuleSet.make(3)
    a, b = RelativeMock(b"a"), RelativeMock(b"b")
    fast = play_match(rules, [a, b, b], 3, PortableRng(2))
    slow = play_match(rules, [AbsoluteOnly(a), AbsoluteOnly(b), AbsoluteOnly(b)],
                      3, PortableRng(2))
    assert fast == slow
