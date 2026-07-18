"""Phase C gates: chance-signature exactness, determinism, closed-mode
expectimax equivalence, open-mode agreement, self-play smoke (plan §5)."""

import math
import random

import pytest

torch = pytest.importorskip("torch")

from games.seven_wonders_duel.codec import decode_action, legal_action_indices
from games.seven_wonders_duel.engine import apply_action
from games.seven_wonders_duel.game import ChanceKind, Phase, new_game
from games.seven_wonders_duel.inference import Evaluator
from games.seven_wonders_duel.net import SWDNet
from games.seven_wonders_duel.search import (
    GumbelMCTS,
    SearchConfig,
    chance_signature,
    closed_root_exact_value,
    enumerate_chains,
    expand_exhaustive,
    state_actor,
)


@pytest.fixture(scope="module")
def evaluator():
    torch.manual_seed(7)
    return Evaluator(SWDNet(32, 1, 2))


def _play_random(seed, until=None, rng_seed=None):
    game = new_game(seed, first_player=seed % 2)
    rng = random.Random(rng_seed if rng_seed is not None else seed * 13 + 5)
    while game.phase is not Phase.COMPLETE:
        if until is not None and until(game):
            return game
        indices = legal_action_indices(game)
        apply_action(game, decode_action(game, rng.choice(indices)))
    return game


def _present_count(game):
    return sum(1 for card in game.tableau.cards.values() if card.present)


# --- chance signature is exact vs engine events -----------------------------


def test_chance_signature_matches_engine_events_over_full_games():
    checked = 0
    for seed in range(6):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(seed)
        while game.phase is not Phase.COMPLETE:
            index = rng.choice(legal_action_indices(game))
            action = decode_action(game, index)
            specs = chance_signature(game, action)
            result = apply_action(game, action)
            assert len(result.events) == len(specs), (specs, result.events)
            for spec, event in zip(specs, result.events):
                assert spec.kind is event.kind
                if spec.kind is ChanceKind.CARD_REVEAL:
                    assert spec.context == event.context
                elif spec.kind is ChanceKind.AGE_DEAL:
                    assert spec.context == event.context
                checked += 1
    assert checked > 100


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_search_is_deterministic_given_seed(evaluator, mode):
    game = _play_random(3, until=lambda g: g.phase is Phase.PLAY_AGE)
    config = SearchConfig(sims=16, top_k=4, mode=mode, seed=11)
    first = GumbelMCTS(evaluator, config).search(game.clone())
    second = GumbelMCTS(evaluator, SearchConfig(sims=16, top_k=4, mode=mode, seed=11)).search(
        game.clone()
    )
    assert first.action_index == second.action_index
    assert first.visits == second.visits
    assert first.root_value == pytest.approx(second.root_value)
    assert first.policy_target == pytest.approx(second.policy_target)


# --- closed mode == brute-force expectimax on small positions (1e-6) --------


def _expectimax(game, evaluator, memo=None):
    """Independent reference: exact expectation over enumerated chance,
    minimax over decisions, terminal values only (positions chosen so every
    line ends). Player-0 perspective."""

    if game.phase is Phase.COMPLETE:
        if game.winner is None:
            return 0.0
        return 1.0 if game.winner == 0 else -1.0
    actor = state_actor(game)
    sign = 1.0 if actor == 0 else -1.0
    best = -math.inf
    for index in legal_action_indices(game):
        action = decode_action(game, index)
        specs = chance_signature(game, action)
        expected = 0.0
        for outcomes, probability, _ in enumerate_chains(game, specs):
            clone = game.clone()
            clone.search_barrier = True
            apply_action(clone, decode_action(clone, index), chance_outcomes=outcomes or None)
            expected += probability * _expectimax(clone, evaluator)
        best = max(best, sign * expected)
    return sign * best


def _near_terminal_position(max_cards=3, need_hidden=True):
    for seed in range(40):
        game = _play_random(
            seed,
            until=lambda g: (
                g.age == 3
                and g.pending_choice is None
                and g.phase is Phase.PLAY_AGE
                and _present_count(g) <= max_cards
            ),
        )
        if game.phase is Phase.COMPLETE:
            continue
        hidden = any(
            c.present and not c.revealed for c in game.tableau.cards.values()
        )
        if hidden or not need_hidden:
            return game
    raise AssertionError("no suitable near-terminal position found")


def test_closed_exact_value_matches_expectimax_to_1e6(evaluator):
    game = _near_terminal_position()
    mcts = GumbelMCTS(evaluator, SearchConfig(mode="closed", seed=5))
    root_state = game.clone()
    root_state.search_barrier = True
    root = mcts._make_closed_node(root_state)
    expand_exhaustive(mcts, root)
    exact = closed_root_exact_value(root)
    reference = _expectimax(game, evaluator)
    assert exact == pytest.approx(reference, abs=1e-6)


def test_closed_search_converges_to_exact_on_small_position(evaluator):
    game = _near_terminal_position()
    reference = _expectimax(game, evaluator)
    config = SearchConfig(sims=400, top_k=8, mode="closed", seed=2)
    result = GumbelMCTS(evaluator, config).search(game)
    assert result.root_value == pytest.approx(reference, abs=0.2)
    # The chosen action must be an exact-optimal one.
    actor = state_actor(game)
    sign = 1.0 if actor == 0 else -1.0
    action_values = {}
    for index in legal_action_indices(game):
        specs = chance_signature(game, decode_action(game, index))
        expected = 0.0
        for outcomes, probability, _ in enumerate_chains(game, specs):
            clone = game.clone()
            clone.search_barrier = True
            apply_action(
                clone, decode_action(clone, index), chance_outcomes=outcomes or None
            )
            expected += probability * _expectimax(clone, evaluator)
        action_values[index] = sign * expected
    best_value = max(action_values.values())
    assert action_values[result.action_index] == pytest.approx(best_value, abs=1e-9)


def test_open_mode_agrees_on_small_position(evaluator):
    game = _near_terminal_position()
    reference = _expectimax(game, evaluator)
    config = SearchConfig(sims=400, top_k=8, mode="open", seed=3)
    result = GumbelMCTS(evaluator, config).search(game)
    assert result.root_value == pytest.approx(reference, abs=0.35)


# --- Gumbel root contract ---------------------------------------------------


@pytest.mark.parametrize("mode", ["closed", "open"])
def test_gumbel_policy_target_is_a_distribution(evaluator, mode):
    game = _play_random(9, until=lambda g: g.phase is Phase.PLAY_AGE)
    result = GumbelMCTS(evaluator, SearchConfig(sims=24, top_k=8, mode=mode, seed=1)).search(
        game
    )
    legal = set(legal_action_indices(game))
    assert result.action_index in legal
    assert set(result.policy_target) == legal
    assert sum(result.policy_target.values()) == pytest.approx(1.0)
    assert all(p >= 0 for p in result.policy_target.values())
    assert sum(result.visits.values()) > 0


# --- self-play smoke (barrier holds end to end) -----------------------------


def test_self_play_smoke_both_modes(evaluator):
    game = new_game(21)
    move = 0
    while game.phase is not Phase.COMPLETE and move < 90:
        mode = "closed" if move % 2 == 0 else "open"
        config = SearchConfig(sims=8, top_k=4, mode=mode, seed=move)
        result = GumbelMCTS(evaluator, config).search(game)
        assert result.action_index in set(legal_action_indices(game))
        apply_action(game, decode_action(game, result.action_index))
        move += 1
    assert game.phase is Phase.COMPLETE
    assert game.search_barrier is False  # the real game was never barred
