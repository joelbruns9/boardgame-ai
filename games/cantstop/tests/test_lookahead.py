"""Exact-root turn values and the selective 2-turn lookahead.

The gate is per turn: the Rust pool is started from mid-game positions for
one or two turns and compared with Python's ``play_turn`` -- turn value,
turn length, and the resulting board byte for byte. Whole games with
lookahead take minutes each in Python, so they are a manual smoke
(``rust_pool_equiv``), not part of the suite.

Run: python -m pytest games/cantstop/tests/test_lookahead.py -q
"""

import numpy as np
import pytest

from games.cantstop.encoder import FEATURE_SIZE, encode_board
from games.cantstop.engine import ALL_RULESETS, GameState, Phase, RuleSet
from games.cantstop.lookahead import BUST, choose, leaf_reach, refine
from games.cantstop.portable_rng import PortableRng
from games.cantstop.rust_pool import game_seeds, run_pool
from games.cantstop.rust_pool_equiv import MOCKS, compare_results
from games.cantstop.rust_solver import sample_positions
from games.cantstop.self_play import PLAIN, Search, play_turn
from games.cantstop.snapshot import snapshot
from games.cantstop.solver import TurnSolver

rust = pytest.importorskip("cantstop_rust", reason="run maturin develop first")

K2 = Search(lookahead_k=2)
K3_RAW = Search(lookahead_k=3, lookahead_offset=False)
K3_EXACT = Search(exact_root=True, lookahead_k=3)
EXACT = Search(exact_root=True)


def _key(s):
    return (s.exact_root, s.lookahead_k, s.lookahead_offset)


def turn_starts(rules, count, seed=4):
    """The ``count`` turn-start positions with the smallest turn-start
    tables among a sample, so the Python reference stays quick.

    Loud rather than empty: the first version filtered on a fixed size
    (3,000) that NO rule set met -- turn-start tables run 3k-44k -- and the
    gate silently checked nothing."""
    starts = [p for p in sample_positions(rules, seed, 60)
              if p.phase == Phase.AWAIT_ROLL]
    starts.sort(key=lambda p: rust.TurnSolver(snapshot(p)).num_positions)
    assert len(starts) >= count, f"only {len(starts)} turn starts for {rules}"
    return starts[:count]


def rust_turns(state, seed, evaluator, searches_by_seat, turns):
    """Play ``turns`` turns from ``state`` on the pool; returns the game's
    raw result tuple and its recorded rows."""
    n = state.rules.num_players
    slots = list(dict.fromkeys(searches_by_seat))
    pool = rust.SelfPlayPool(
        [(snapshot(state), seed, [0] * n)], turns, 1, 0,
        [_key(s) for s in slots],
        [[slots.index(s) for s in searches_by_seat]])
    pool.advance()
    while pool.running:
        raw, blocks = pool.pending()
        feats = np.frombuffer(raw, dtype="<f4").reshape(-1, FEATURE_SIZE)
        vals = []
        for b in blocks:
            game, rows, _ev, n_, active, _sub = b
            start = sum(x[1] for x in blocks[:blocks.index(b)])
            ref = GameState(state.rules)
            ref.active_player = active
            vals.append(evaluator.evaluate_features(
                feats[start:start + rows], ref).astype("<f8").ravel())
        pool.resume(np.concatenate(vals).tobytes())
    result = pool.results()[0]
    rows = np.frombuffer(pool.features(), dtype="<f4").reshape(-1, FEATURE_SIZE)
    return result, rows


def python_turns(state, seed, evaluator, searches_by_seat, turns):
    s = state.clone()
    rng = PortableRng(seed)
    values, lengths, rows = [], [], []
    for _ in range(turns):
        if s.game_over:
            break
        _solves, decisions = play_turn(s, evaluator, rng, values,
                                       searches_by_seat[s.active_player])
        lengths.append(decisions)
        if not s.game_over:
            rows.append(encode_board(s))
    return values, lengths, rows, s


def check_turns(state, seed, mock, searches_by_seat, turns=1):
    ev = MOCKS[mock]
    values, lengths, rows, final = python_turns(state, seed, ev,
                                                searches_by_seat, turns)
    result, rs_rows = rust_turns(state, seed, ev, searches_by_seat, turns)
    (_id, winner, _rows, _slots, _turns, _solves, _ev_rows, rs_lengths,
     rs_values) = result
    where = f"{state!r} seed={seed} {mock} {searches_by_seat}"
    assert len(values) == turns or final.game_over, where   # it really played
    assert [None if v is None else list(v) for v in rs_values] == values, where
    assert list(rs_lengths) == lengths, where
    assert rs_rows.tobytes() == b"".join(r.tobytes() for r in rows), where
    if final.game_over:
        assert winner == final.winner


# ---- the gate ----

@pytest.mark.parametrize("search", [EXACT, K2, K3_RAW, K3_EXACT],
                         ids=["exact", "k2", "k3-raw", "k3-exact"])
@pytest.mark.parametrize("mock", ["hashed", "mover_wins"])
def test_turns_match_python(search, mock):
    for rules in (RuleSet.make(2), RuleSet.make(3, blocking=True),
                  RuleSet.make(4, extended=True)):
        for i, state in enumerate(turn_starts(rules, 1)):
            check_turns(state, 100 + i, mock, [search] * rules.num_players)


def test_each_seat_uses_its_own_search():
    """Arena games give seats different searches: two consecutive turns,
    the first mover with lookahead, the next without (and the reverse)."""
    rules = RuleSet.make(2)
    for state in turn_starts(rules, 2):
        for pair in ([K2, PLAIN], [PLAIN, K2]):
            check_turns(state, 7, "hashed", pair, turns=2)


# ---- exact root ----

def test_exact_root_plays_exactly_the_plain_games():
    """Solving before the opening roll changes what is RECORDED, never what
    is PLAYED: one table answers every decision either way."""
    rules_list = [r for r in ALL_RULESETS for _ in range(2)]
    seeds = game_seeds(PortableRng(21), len(rules_list))
    plain = run_pool(rules_list, seeds, [MOCKS["hashed"]])
    exact = run_pool(rules_list, seeds, [MOCKS["hashed"]], search=EXACT)
    for p, e in zip(plain, exact):
        assert (p.winner, p.turns, p.turn_lengths) == \
            (e.winner, e.turns, e.turn_lengths)
        assert p.features.tobytes() == e.features.tobytes()
        assert all(v is not None for v in e.turn_values)
        assert e.solves == e.turns >= p.solves
        assert e.evaluator_rows > p.evaluator_rows


def test_exact_value_is_the_expectation_over_opening_rolls():
    """The exact root value equals the probability-weighted average, over
    every opening roll, of the value after that roll (bust included)."""
    from games.cantstop.engine import roll
    from games.cantstop.solver import ROLL_CLASSES
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(2), 1)[0]
    exact = TurnSolver(state, ev).value(state)
    total = np.zeros(2)
    for dice, p in ROLL_CLASSES:
        s = state.clone()
        if roll(s, dice):
            total += p * TurnSolver(s, ev).value(s)
        else:
            total += p * ev([s])[0]          # the bust board, as is
    assert np.allclose(total, exact, atol=1e-12)


# ---- the reference itself ----

def test_choose_takes_the_bust_board_and_the_most_reached_stops():
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(3), 1)[0]
    ps = TurnSolver(state, ev)
    bust, stops = leaf_reach(ps, state)
    leaves, weights = choose(ps, state, 4)
    assert leaves[0] is BUST and weights[0] == bust
    ranked = sorted((k for k, r in stops.items() if r > 0),
                    key=lambda k: (-stops[k], k))
    assert leaves[1:] == ranked[:3]
    assert 0 < bust < 1
    assert choose(ps, state, 0) == ([], [])


def test_reach_is_a_probability_split_between_busting_and_stopping():
    """Under the policy every turn ends exactly once: in a bust, or in a
    stop (a chosen stop leaf, or a winning stop). So bust reach + the reach
    of stops the policy takes + winning reach = 1."""
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(2), 1)[0]
    ps = TurnSolver(state, ev)
    bust, stops = leaf_reach(ps, state)
    taken = sum(r for k, r in stops.items()
                if ps.decision_values[k] is ps.stop_values[k])
    # Reach into winning configurations, recomputed the same way.
    a = ps.active
    reach = {(): 1.0}
    win = 0.0
    for key in sorted(ps.keys, key=lambda k: (sum(p for _, p in k), k)):
        r = reach.get(key, 0.0)
        if r == 0.0:
            continue
        if ps.winning[key]:
            win += r
            continue
        if ps.stoppable[key] and ps.decision_values[key] is ps.stop_values[key]:
            continue
        for children, p in ps.menus[key]:
            best = max(children, key=lambda c: ps.decision_values[c][a])
            reach[best] = reach.get(best, 0.0) + r * p
    assert abs(bust + taken + win - 1.0) < 1e-9


def test_refine_replaces_chosen_leaves_and_shifts_the_rest():
    ev = MOCKS["hashed"]
    state = turn_starts(RuleSet.make(2), 1)[0]
    base = TurnSolver(state, ev)
    v1_stop = {k: v.copy() for k, v in base.stop_values.items()}
    ps = TurnSolver(state, ev)
    leaves = refine(ps, state, ev, 3, offset=True)
    chosen = [k for k in leaves if k is not BUST]
    for k in chosen:
        board = ps._board_after(k)
        assert np.array_equal(ps.stop_values[k],
                              TurnSolver(board, ev).value(board))
    others = [k for k in v1_stop if k not in chosen and not ps.winning[k]]
    shifts = {tuple(np.round(ps.stop_values[k] - v1_stop[k], 12))
              for k in others}
    assert len(shifts) == 1              # one common offset
    raw = TurnSolver(state, ev)
    refine(raw, state, ev, 3, offset=False)
    for k in others:
        assert np.array_equal(raw.stop_values[k], v1_stop[k])
