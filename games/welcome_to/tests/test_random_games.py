"""Whole-game properties, checked by playing a lot of them."""
from __future__ import annotations

import random

import numpy as np
import pytest

from games.welcome_to import action_codec as codec
from games.welcome_to.constants import (
    NUM_BASE_CARDS,
    PERMIT_BOXES,
    ROUNDABOUT,
    SOLO_CARD_ID,
)
from games.welcome_to.bots import GreedyBot, RandomBot, play_match
from games.welcome_to.game import GameConfig, GameState
from games.welcome_to.random_play import check_terminal_invariants

CONFIGS = [
    GameConfig(players=2),
    GameConfig(players=3),
    GameConfig(players=4),
    GameConfig(players=2, advanced=True),
    GameConfig(players=4, advanced=True),
    GameConfig(players=2, expert=True),
    GameConfig(players=1),
    GameConfig(players=1, advanced=True),
]


def _live_cards(state: GameState) -> list[int]:
    """Every card the engine still accounts for, wherever it is."""
    ids = list(state.deck[state.deck_pos:]) + list(state.discard)
    for group in state.stack_new:
        ids += [c for c in group if c is not None]
    for group in state.stack_old:
        ids += [c for c in group if c is not None]
    ids += [c for c in state.expert_pending if c is not None]
    return ids


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"p{c.players}"
                         f"{'a' if c.advanced else ''}{'x' if c.expert else ''}")
def test_random_games_finish_and_stay_legal(config):
    for seed in range(6):
        rng = random.Random(1000 + seed)
        state = GameState.new(seed=seed, config=config)
        steps = 0
        while not state.is_terminal:
            legal = state.legal_actions()
            assert legal, f"stuck in {state.phase.name}"
            assert len(set(legal)) == len(legal), "duplicate action offered"
            assert all(0 <= a < codec.NUM_ACTIONS for a in legal)
            state.apply(rng.choice(legal))
            steps += 1
            assert steps < 5000, "game is not converging"
        check_terminal_invariants(state)
        assert len(state.scores()) == config.players


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"p{c.players}"
                         f"{'a' if c.advanced else ''}{'x' if c.expert else ''}")
def test_cards_are_conserved_all_game(config):
    rng = random.Random(7)
    state = GameState.new(seed=99, config=config)
    while not state.is_terminal:
        cards = _live_cards(state)
        assert len(cards) == len(set(cards)), "a card was duplicated"
        expected = set(range(NUM_BASE_CARDS))
        if config.solo:
            # the solo card is removed from play the moment it is drawn
            assert set(cards) <= expected | {SOLO_CARD_ID}
            assert expected - set(cards) == set(), "a construction card vanished"
        else:
            assert set(cards) == expected, "a construction card vanished"
        state.apply(rng.choice(state.legal_actions()))


def test_the_legal_mask_agrees_with_the_legal_list():
    rng = random.Random(3)
    state = GameState.new(seed=5, config=GameConfig(players=3))
    for _ in range(60):
        if state.is_terminal:
            break
        mask = state.legal_mask()
        assert mask.dtype == np.bool_
        assert mask.shape == (codec.NUM_ACTIONS,)
        assert set(np.flatnonzero(mask).tolist()) == set(state.legal_actions())
        state.apply(rng.choice(state.legal_actions()))


def test_a_finished_game_reports_why_it_ended():
    for seed in range(15):
        state = GameState.new(seed=seed, config=GameConfig(players=3))
        rng = random.Random(seed)
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        reason = state.end_of_game_reason()
        assert reason is not None
        triggered = any(
            not sheet.has_free_box()
            or sheet.permits >= PERMIT_BOXES
            or all(p in state.plan_turns[slot] for slot in range(3))
            for p, sheet in enumerate(state.sheets)
        )
        assert triggered


def test_no_action_is_ever_taken_after_the_game_ends():
    state = GameState.new(seed=2, config=GameConfig(players=2))
    rng = random.Random(2)
    while not state.is_terminal:
        state.apply(rng.choice(state.legal_actions()))
    assert state.legal_actions() == []


def test_bis_houses_always_match_a_neighbour():
    rng = random.Random(4)
    for seed in range(8):
        state = GameState.new(seed=seed, config=GameConfig(players=2))
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        for sheet in state.sheets:
            for x, row in enumerate(sheet.numbers):
                for y, n in enumerate(row):
                    if not sheet.is_bis[x][y]:
                        continue
                    neighbours = []
                    if y > 0:
                        neighbours.append(row[y - 1])
                    if y + 1 < len(row):
                        neighbours.append(row[y + 1])
                    assert n in neighbours, "a bis with nothing to duplicate"
                    assert n != ROUNDABOUT


def test_streets_stay_in_ascending_order_apart_from_bis_and_roundabouts():
    rng = random.Random(6)
    for seed in range(8):
        state = GameState.new(seed=seed, config=GameConfig(players=2, advanced=True))
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        for sheet in state.sheets:
            for x, row in enumerate(sheet.numbers):
                previous = -1
                for y, n in enumerate(row):
                    if n is None:
                        continue
                    if n == ROUNDABOUT:
                        previous = -1
                        continue
                    if sheet.is_bis[x][y]:
                        # a bis copies a neighbour, so it is allowed to tie with
                        # it in either direction; it never advances the chain
                        continue
                    assert n > previous, f"street {x} box {y} breaks the order"
                    previous = n


def test_permit_refusals_never_exceed_three():
    rng = random.Random(8)
    for seed in range(10):
        state = GameState.new(seed=seed, config=GameConfig(players=4))
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        for sheet in state.sheets:
            assert sheet.permits <= PERMIT_BOXES


def test_greedy_outscores_random():
    """A one-ply score-greedy player should be clearly ahead of a coin flip.

    Compared on mean score rather than win rate: both seats see the same cards
    in standard mode, so the score gap is the low-variance signal and a handful
    of games is enough to see it.
    """
    greedy_total = 0
    random_total = 0
    games = 12
    for seed in range(games):
        bots = [GreedyBot(random.Random(seed)), RandomBot(random.Random(seed + 500))]
        scores = play_match(bots, seed=seed, config=GameConfig(players=2)).scores()
        greedy_total += scores[0]
        random_total += scores[1]
    assert greedy_total > random_total, (
        f"greedy {greedy_total / games:.1f} vs random {random_total / games:.1f}"
    )


def test_returns_are_consistent_with_the_scores():
    for seed in range(6):
        state = GameState.new(seed=seed, config=GameConfig(players=3))
        rng = random.Random(seed)
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        returns = state.returns()
        winners = state.winners()
        assert len(returns) == 3
        for p in range(3):
            if p in winners:
                assert returns[p] > -1.0
            else:
                assert returns[p] == -1.0
        assert state.ranking()[0] in winners
