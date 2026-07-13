"""Phase-0 expectiminimax harness correctness tests.

The load-bearing property is information-set safety: the search value must not
depend on the hidden deck ORDER, only on the public bag composition. If it did,
the searcher would be clairvoyant and (as the project learned early) breed false
confidence. We check that directly on the chance-expansion primitive and on a
full search value.
"""
from __future__ import annotations

import random

from games.kingdomino.game import GameState, Phase, determine_winner
from games.kingdomino.bots import RandomBot
from games.kingdomino.bot_match import play_bot_game
import pytest

from games.kingdomino.expectiminimax import (
    ExpectiminimaxBot,
    terminal_value_p0,
    outcome_p0,
    margin_p0,
)


def _advance(seed: int, plies: int) -> GameState:
    st = GameState.new(seed=seed)
    rng = random.Random(seed)
    for _ in range(plies):
        if st.phase == Phase.GAME_OVER:
            break
        st = st.step(rng.choice(st.legal_actions()))
    return st


def _round_boundary_state(seed: int, max_deck: int | None = None) -> GameState:
    """A PLACE_AND_SELECT state whose next action draws a new row (deck shrinks).

    If `max_deck` is set, only return boundaries with a bag small enough that the
    chance node is ENUMERATED exactly (deck<=8 -> C(8,4)=70 <= default enum_cap)."""
    st = GameState.new(seed=seed)
    rng = random.Random(seed)
    for _ in range(400):
        if st.phase != Phase.PLACE_AND_SELECT or not st.deck:
            st = st.step(rng.choice(st.legal_actions()))
            continue
        boundary = st.actor_index == len(st.pending_claims) - 1
        small = max_deck is None or len(st.deck) <= max_deck
        if boundary and small:
            return st
        st = st.step(rng.choice(st.legal_actions()))
    raise AssertionError("no matching round-boundary state found")


def _reference_value(bot: ExpectiminimaxBot, state: GameState, depth: int) -> float:
    """Unpruned expectiminimax: the correctness oracle for `_value` + alpha-beta.

    Identical recursion to `_value` with NO pruning. Since `_expand_chance` is
    deterministic (stable-seeded), the pruned and unpruned searches see the same
    tree, so alpha-beta must return exactly the same value."""
    if state.phase == Phase.GAME_OVER:
        return terminal_value_p0(state, bot.margin_weight)
    if depth <= 0:
        return bot.eval_fn(state)
    vals = []
    for a in state.legal_actions():
        ev = 0.0
        for child, p in bot._expand_chance(state, a):
            ev += p * _reference_value(bot, child, depth - 1)
        vals.append(ev)
    return max(vals) if state.current_actor == 0 else min(vals)


def test_chance_weights_sum_to_one():
    bot = ExpectiminimaxBot(enum_cap=16, chance_samples=8)
    st = _round_boundary_state(7)
    action = st.legal_actions()[0]
    children = bot._expand_chance(st, action)
    assert len(children) > 1, "expected a genuine chance node at a round boundary"
    assert abs(sum(p for _, p in children) - 1.0) < 1e-9


def test_chance_expansion_invariant_to_deck_order():
    """Permuting the hidden deck must not change the chance children or weights."""
    bot = ExpectiminimaxBot(enum_cap=16, chance_samples=8)
    st = _round_boundary_state(11)
    action = st.legal_actions()[0]

    def signature(state):
        kids = bot._expand_chance(state, action)
        return sorted(
            (tuple(c.current_row), tuple(c.deck), round(p, 12)) for c, p in kids
        )

    base = signature(st)
    for perm_seed in range(5):
        shuffled = st.copy()
        random.Random(perm_seed).shuffle(shuffled.deck)
        assert signature(shuffled) == base, "chance expansion leaked deck order"


def test_search_value_invariant_to_deck_order():
    """Full search value is identical across hidden deck permutations."""
    bot = ExpectiminimaxBot(depth=2, enum_cap=16, chance_samples=8)
    st = _round_boundary_state(3)
    action = st.legal_actions()[0]
    from math import inf

    base = bot._action_value(st, action, bot.depth, -inf, inf)
    for perm_seed in range(4):
        shuffled = st.copy()
        random.Random(perm_seed).shuffle(shuffled.deck)
        v = bot._action_value(shuffled, action, bot.depth, -inf, inf)
        assert abs(v - base) < 1e-9, f"search value leaked deck order: {v} vs {base}"


def test_choose_action_deterministic():
    st = _advance(5, 6)
    acts = st.legal_actions()
    a1 = ExpectiminimaxBot(depth=2).choose_action(st, acts, rng=random.Random(0))
    a2 = ExpectiminimaxBot(depth=2).choose_action(st, acts, rng=random.Random(0))
    assert a1 == a2


def test_terminal_value_sign_matches_official_winner():
    """At real terminals, the value sign must agree with the official cascade."""
    for seed in range(12):
        _, state = play_bot_game(seed=seed, bot0=RandomBot(), bot1=RandomBot())
        assert state.phase == Phase.GAME_OVER
        v = terminal_value_p0(state)
        winner = determine_winner(state)
        if winner == 0:
            assert v > 0
        elif winner == 1:
            assert v < 0
        else:
            # true draw: no win bonus, value is the (zero-ish) margin
            assert abs(v) < 1.0


def test_alphabeta_equals_unpruned_reference():
    """Central correctness test (review finding 4): alpha-beta must return exactly
    the full unpruned expectiminimax value, across sampled AND enumerated chance,
    player-0 and player-1 decisions, and depths that cross chance layers."""
    from math import inf

    bot = ExpectiminimaxBot(depth=2, chance_samples=6, enum_cap=128)

    cases = [
        _advance(1, 5),                       # mid state, player-0-ish to move
        _advance(2, 6),                       # different actor / continuation
        _round_boundary_state(4),             # WIDE (sampled) chance node
        _round_boundary_state(8, max_deck=8), # NARROW (enumerated) chance node
    ]
    for st in cases:
        if st.phase == Phase.GAME_OVER:
            continue
        got = bot._value(st, bot.depth, -inf, inf)
        want = _reference_value(bot, st, bot.depth)
        assert abs(got - want) < 1e-9, f"alpha-beta != reference: {got} vs {want}"

    # A deeper (depth-3) check on a late, low-branching state.
    late = _round_boundary_state(13, max_deck=8)
    deep = ExpectiminimaxBot(depth=3, enum_cap=128)
    got = deep._value(late, deep.depth, -inf, inf)
    want = _reference_value(deep, late, deep.depth)
    assert abs(got - want) < 1e-9, f"depth-3 alpha-beta != reference: {got} vs {want}"


def test_terminal_delegates_to_official_winner():
    """terminal_value_p0 (margin_weight=0) must be exactly the official outcome —
    there is no independent margin-sign path that could disagree on a tiebreak."""
    tie_seen = False
    for seed in range(48):
        _, state = play_bot_game(seed=seed, bot0=RandomBot(), bot1=RandomBot())
        winner = determine_winner(state)
        expect = 0.0 if winner is None else (1.0 if winner == 0 else -1.0)
        assert outcome_p0(state) == expect
        assert terminal_value_p0(state, margin_weight=0.0) == expect
        s = state.scores()
        if s[0] == s[1] and winner is not None:  # equal total, decided by tiebreak
            tie_seen = True
            assert (expect > 0) == (winner == 0)
    # Not asserted as required (ties are rare), but recorded when it happens.
    _ = tie_seen


def test_constructor_rejects_bad_params():
    for kwargs in ({"depth": 0}, {"chance_samples": 0}, {"enum_cap": 0}):
        with pytest.raises(ValueError):
            ExpectiminimaxBot(**kwargs)


def test_depth1_beats_random():
    bot = ExpectiminimaxBot(depth=1)
    wins = 0
    for seed in range(12):
        scores, state = play_bot_game(seed=seed, bot0=bot, bot1=RandomBot())
        if determine_winner(state) == 0:
            wins += 1
    assert wins >= 9, f"depth-1 expectiminimax should dominate random, got {wins}/12"
