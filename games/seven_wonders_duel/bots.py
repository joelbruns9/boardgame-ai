"""Baseline players and reproducible match runners."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from .data import CARDS_BY_NAME, PROGRESS_BY_NAME, WONDERS_BY_NAME, CardColor
from .engine import Action, ActionUse, apply_action, legal_actions, score_player
from .game import GameState, Phase, VictoryType, new_game
from .rules import Resource


class Bot(Protocol):
    name: str

    def select_action(self, game: GameState) -> Action: ...


def _require_actions(game: GameState) -> tuple[Action, ...]:
    actions = legal_actions(game)
    if not actions:
        raise ValueError("no legal action is available")
    return actions


class RandomBot:
    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def select_action(self, game: GameState) -> Action:
        return self._rng.choice(_require_actions(game))


def _science_count(game: GameState, player: int) -> int:
    return len(_science_symbols(game, player))


def _science_symbols(game: GameState, player: int) -> frozenset:
    symbols = {
        CARDS_BY_NAME[name].science
        for name in game.cities[player].buildings
        if CARDS_BY_NAME[name].science is not None
    }
    symbols.update(
        PROGRESS_BY_NAME[name].science
        for name in game.cities[player].progress_tokens
        if PROGRESS_BY_NAME[name].science is not None
    )
    return frozenset(symbols)


def _actor(game: GameState) -> int:
    return (
        game.pending_choice.player
        if game.pending_choice is not None
        else game.active_player
    )


def _economy_value(game: GameState, player: int) -> float:
    value = 0.0
    for name in game.cities[player].buildings:
        card = CARDS_BY_NAME[name]
        value += 0.7 * len(card.fixed_production)
        if card.choice_production:
            value += 0.8 + 0.2 * (len(card.choice_production) - 1)
        value += 0.7 * len(card.trade_discount)
    for name in game.cities[player].built_wonders:
        choices = WONDERS_BY_NAME[name].choice_production
        if choices:
            value += 0.8 + 0.2 * (len(choices) - 1)
    return value


def _unbuilt_wonder_value(game: GameState, player: int) -> float:
    city = game.cities[player]
    value = 0.0
    for name in city.wonders:
        if name in city.built_wonders or name in game.retired_wonders:
            continue
        wonder = WONDERS_BY_NAME[name]
        value += 0.25 * wonder.victory_points + 0.75 * wonder.shields
        if wonder.choice_production:
            value += 0.6
        for effect in wonder.effects:
            if effect.kind.value == "play_again":
                value += 1.2
            elif effect.kind.value in {
                "destroy_opponent_brown",
                "destroy_opponent_grey",
                "build_from_discard_free",
                "choose_unused_progress",
            }:
                value += 0.8
            elif effect.kind.value == "immediate_coins":
                value += 0.06 * effect.amount
    return value


def evaluate_state(game: GameState, player: int) -> float:
    """Public-feature heuristic from ``player``'s perspective."""

    opponent = 1 - player
    if game.phase is Phase.COMPLETE:
        if game.winner == player:
            return 1_000_000.0
        if game.winner == opponent:
            return -1_000_000.0
        return 0.0

    scores = (score_player(game, player), score_player(game, opponent))
    military = game.conflict_position if player == 0 else -game.conflict_position
    value = 8.0 * (scores[0].total - scores[1].total)
    value += 0.2 * (game.cities[player].coins - game.cities[opponent].coins)
    value += 2.0 * military
    value += 4.0 * (_science_count(game, player) - _science_count(game, opponent))
    value += 1.2 * (_economy_value(game, player) - _economy_value(game, opponent))
    value += 0.5 * (
        _unbuilt_wonder_value(game, player) - _unbuilt_wonder_value(game, opponent)
    )
    if game.active_player == player:
        value += 0.1
    return value


def _action_tiebreak(action: Action) -> tuple:
    return (
        action.use.value,
        action.slot_id or (-1, -1),
        action.wonder_name or "",
        action.choice or "",
        -1 if action.starting_player is None else action.starting_player,
    )


class GreedyBot:
    """Deterministic one-ply baseline using only public state features."""

    name = "greedy"

    def select_action(self, game: GameState) -> Action:
        player = _actor(game)
        actions = _require_actions(game)
        scored: list[tuple[float, tuple, Action]] = []
        for action in actions:
            child = game.clone()
            apply_action(child, action)
            scored.append(
                (evaluate_state(child, player), _action_tiebreak(action), action)
            )
        return max(scored, key=lambda entry: (entry[0], entry[1]))[2]


# Rush bots are an early-training curriculum, not general-purpose evaluators.
# They intentionally overvalue one supremacy route so successful examples make
# science and military pressure visible to the policy/value heads immediately.

_SCIENCE_RESOURCE_WEIGHTS = {
    Resource.WOOD: 1.1,
    Resource.CLAY: 0.8,
    Resource.STONE: 1.0,
    Resource.GLASS: 1.5,
    Resource.PAPYRUS: 1.5,
}
_MILITARY_RESOURCE_WEIGHTS = {
    Resource.WOOD: 1.4,
    Resource.CLAY: 1.4,
    Resource.STONE: 1.4,
    Resource.GLASS: 0.7,
    Resource.PAPYRUS: 0.8,
}

_SCIENCE_WONDER_VALUES = {
    "The Great Library": 420.0,
    "The Mausoleum": 300.0,
    "Piraeus": 170.0,
    "The Great Lighthouse": 150.0,
    "The Hanging Gardens": 130.0,
    "The Sphinx": 130.0,
    "The Temple of Artemis": 120.0,
    "The Appian Way": 100.0,
}
_MILITARY_WONDER_VALUES = {
    "The Colossus": 440.0,
    "The Statue of Zeus": 360.0,
    "Circus Maximus": 340.0,
    "The Appian Way": 160.0,
    "The Hanging Gardens": 130.0,
    "The Sphinx": 130.0,
    "Piraeus": 120.0,
    "The Temple of Artemis": 110.0,
}


def _card_for_action(game: GameState, action: Action):
    if action.slot_id is None:
        return None
    card = game.tableau.cards.get(action.slot_id)
    if card is None or not card.present or not card.revealed:
        return None
    return CARDS_BY_NAME[card.card_name]


def _resource_support_value(
    game: GameState, player: int, weights: dict[Resource, float]
) -> float:
    """Public economic support for a rush route.

    Flexible producers count as one unit of their best supported resource.
    This deliberately values capability rather than peeking at future cards.
    """

    value = 0.0
    for name in game.cities[player].buildings:
        card = CARDS_BY_NAME[name]
        value += sum(weights[resource] for resource in card.fixed_production)
        if card.choice_production:
            value += max(weights[resource] for resource in card.choice_production)
        value += 0.8 * sum(weights[resource] for resource in card.trade_discount)
    for name in game.cities[player].built_wonders:
        choices = WONDERS_BY_NAME[name].choice_production
        if choices:
            value += max(weights[resource] for resource in choices)
    return value


def _progress_is_off_board(game: GameState, token_name: str) -> bool:
    if token_name in game.available_progress_tokens:
        return False
    return not any(
        token_name in city.progress_tokens for city in game.cities
    )


def _relative_military(game: GameState, player: int) -> int:
    return game.conflict_position if player == 0 else -game.conflict_position


def _military_pressure(position: int) -> float:
    # Odd, monotonic, and convex toward either capital.
    return 25.0 * position + 8.0 * position * abs(position)


class _RushBot:
    """Shared reproducible one-ply shell for the four curriculum bots.

    Action scoring reads the action identity and public city/track state only.
    Applying an action to a clone may resolve a reveal, but no score reads the
    revealed tableau identity, so the policy is invariant to hidden deals.
    """

    name = "rush"
    target_victory: VictoryType

    def __init__(
        self,
        seed: int = 0,
        *,
        exploration: float = 0.0,
        top_choices: int = 3,
    ):
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be in [0, 1]")
        if top_choices <= 0:
            raise ValueError("top_choices must be positive")
        self._rng = random.Random(seed)
        self.exploration = exploration
        self.top_choices = top_choices

    def select_action(self, game: GameState) -> Action:
        player = _actor(game)
        scored: list[tuple[float, tuple, Action]] = []
        for action in _require_actions(game):
            score = self._score_action(game, action, player)
            scored.append((score, _action_tiebreak(action), action))
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        if self.exploration and self._rng.random() < self.exploration:
            choices = scored[: min(self.top_choices, len(scored))]
            return self._rng.choice(choices)[2]
        return scored[0][2]

    def _score_action(self, game: GameState, action: Action, player: int) -> float:
        card = _card_for_action(game, action)
        child = game.clone()
        apply_action(child, action)

        if child.phase is Phase.COMPLETE:
            if child.winner == player:
                return (
                    3_000_000.0
                    if child.victory_type is self.target_victory
                    else 2_000_000.0
                )
            if child.winner is None:
                return 0.0
            return -3_000_000.0

        score = 0.02 * evaluate_state(child, player)
        score += self._focus_score(game, child, action, player, card)
        if _actor(child) == player:
            score += 25.0  # replay Wonders and pending choices are real tempo
        if (
            action.use is ActionUse.CHOOSE_NEXT_START_PLAYER
            and action.starting_player == player
        ):
            score += 15.0
        return score

    def _focus_score(self, game, child, action, player, card) -> float:
        raise NotImplementedError


class _ScienceRushBot(_RushBot):
    target_victory = VictoryType.SCIENTIFIC
    economy_weight = 8.0
    coin_weight = 0.4
    economy_first = False

    def _focus_score(self, game, child, action, player, card) -> float:
        before_symbols = _science_symbols(game, player)
        after_symbols = _science_symbols(child, player)
        new_symbols = len(after_symbols - before_symbols)
        before_pairs = len(game.cities[player].claimed_science_pairs)
        after_pairs = len(child.cities[player].claimed_science_pairs)
        pair_delta = after_pairs - before_pairs

        score = 0.0
        if new_symbols:
            score += new_symbols * (700.0 + 140.0 * len(before_symbols))
        if pair_delta:
            law_live = (
                "Law" in game.available_progress_tokens
                and "Law" not in game.cities[player].progress_tokens
            )
            score += pair_delta * (900.0 if law_live else 260.0)

        if card is not None and card.color is CardColor.GREEN:
            if action.use is ActionUse.CONSTRUCT_BUILDING:
                score += 100.0
                if card.age == 1 and card.science not in before_symbols:
                    score += 1_100.0  # hard Age-I symbol-family override
            elif action.use in (
                ActionUse.DISCARD_FOR_COINS,
                ActionUse.CONSTRUCT_WONDER,
            ):
                score -= 900.0 if card.age == 1 else 350.0

        wonder_name = action.wonder_name
        if wonder_name is not None:
            score += _SCIENCE_WONDER_VALUES.get(wonder_name, 0.0)
            if wonder_name == "The Great Library" and _progress_is_off_board(
                game, "Law"
            ):
                score += 280.0
            if wonder_name == "The Mausoleum":
                score += 80.0 * sum(
                    CARDS_BY_NAME[name].color is CardColor.GREEN
                    for name in game.discard_pile
                )
            if self.economy_first and wonder_name in {
                "Piraeus",
                "The Great Lighthouse",
            }:
                score += 130.0

        if action.use is ActionUse.RESOLVE_PENDING_CHOICE and action.choice:
            if action.choice == "Law":
                score += 1_200.0
            elif action.choice in {"Urbanism", "Agriculture"}:
                score += 180.0
            elif action.choice == "Economy":
                score += 100.0

        support_delta = _resource_support_value(
            child, player, _SCIENCE_RESOURCE_WEIGHTS
        ) - _resource_support_value(game, player, _SCIENCE_RESOURCE_WEIGHTS)
        coin_delta = child.cities[player].coins - game.cities[player].coins
        score += self.economy_weight * support_delta + self.coin_weight * coin_delta
        return score


class ScienceAggressiveBot(_ScienceRushBot):
    """Takes missing science symbols immediately and forces the Law/Library route."""

    name = "science_aggressive/v1"


class ScienceEconomyBot(_ScienceRushBot):
    """Builds science-enabling production between immediate symbol opportunities."""

    name = "science_economy/v1"
    economy_weight = 55.0
    coin_weight = 1.8
    economy_first = True


class _MilitaryRushBot(_RushBot):
    target_victory = VictoryType.MILITARY
    economy_weight = 8.0
    coin_weight = 0.3
    economy_first = False

    def _focus_score(self, game, child, action, player, card) -> float:
        before_position = _relative_military(game, player)
        after_position = _relative_military(child, player)
        score = 3.0 * (
            _military_pressure(after_position) - _military_pressure(before_position)
        )

        if card is not None and card.color is CardColor.RED:
            if action.use is ActionUse.CONSTRUCT_BUILDING:
                score += 650.0 if card.age == 1 else 300.0
            elif action.use in (
                ActionUse.DISCARD_FOR_COINS,
                ActionUse.CONSTRUCT_WONDER,
            ):
                score -= 650.0 if card.age == 1 else 300.0

        wonder_name = action.wonder_name
        if wonder_name is not None:
            score += _MILITARY_WONDER_VALUES.get(wonder_name, 0.0)
            if self.economy_first and wonder_name in {
                "Piraeus",
                "The Great Lighthouse",
            }:
                score += 90.0

        if action.use is ActionUse.RESOLVE_PENDING_CHOICE and action.choice:
            if action.choice == "Strategy":
                score += 1_200.0
            elif action.choice in {"Urbanism", "Agriculture"}:
                score += 170.0
            elif action.choice == "Economy":
                score += 100.0

        opponent = 1 - player
        coin_damage = game.cities[opponent].coins - child.cities[opponent].coins
        score += 8.0 * max(0, coin_damage)
        support_delta = _resource_support_value(
            child, player, _MILITARY_RESOURCE_WEIGHTS
        ) - _resource_support_value(game, player, _MILITARY_RESOURCE_WEIGHTS)
        coin_delta = child.cities[player].coins - game.cities[player].coins
        score += self.economy_weight * support_delta + self.coin_weight * coin_delta
        return score


class MilitaryAggressiveBot(_MilitaryRushBot):
    """Builds shields immediately and values track pressure increasingly near 9."""

    name = "military_aggressive/v1"


class MilitaryEconomyBot(_MilitaryRushBot):
    """Builds military-enabling production between immediate red-card chances."""

    name = "military_economy/v1"
    economy_weight = 55.0
    coin_weight = 1.6
    economy_first = True


@dataclass(frozen=True, slots=True)
class MatchResult:
    seed: int
    winner: int | None
    victory_type: VictoryType
    final_scores: tuple[int, int] | None
    actions: int


@dataclass(frozen=True, slots=True)
class SeriesResult:
    games: int
    bot_a_wins: int
    bot_b_wins: int
    draws: int
    military: int
    scientific: int
    civilian: int
    average_actions: float


def play_game(
    player_zero: Bot,
    player_one: Bot,
    *,
    seed: int,
    first_player: int = 0,
    max_actions: int = 256,
) -> MatchResult:
    game = new_game(seed=seed, first_player=first_player)
    bots = (player_zero, player_one)
    action_count = 0
    while game.phase is not Phase.COMPLETE:
        if action_count >= max_actions:
            raise RuntimeError(f"game exceeded {max_actions} actions")
        player = _actor(game)
        action = bots[player].select_action(game)
        if action not in legal_actions(game):
            raise ValueError(f"{bots[player].name} returned illegal action: {action}")
        apply_action(game, action)
        action_count += 1
    if game.victory_type is None:
        raise AssertionError("terminal game is missing a victory type")
    return MatchResult(
        seed=seed,
        winner=game.winner,
        victory_type=game.victory_type,
        final_scores=game.final_scores,
        actions=action_count,
    )


def play_series(
    bot_a: Bot,
    bot_b: Bot,
    *,
    games: int,
    seed: int = 0,
) -> SeriesResult:
    if games <= 0:
        raise ValueError("games must be positive")
    a_wins = b_wins = draws = 0
    victory_counts = {kind: 0 for kind in VictoryType}
    total_actions = 0
    for index in range(games):
        a_is_zero = index % 2 == 0
        result = play_game(
            bot_a if a_is_zero else bot_b,
            bot_b if a_is_zero else bot_a,
            seed=seed + index,
            first_player=0,
        )
        total_actions += result.actions
        victory_counts[result.victory_type] += 1
        if result.winner is None:
            draws += 1
        elif (result.winner == 0) == a_is_zero:
            a_wins += 1
        else:
            b_wins += 1
    return SeriesResult(
        games=games,
        bot_a_wins=a_wins,
        bot_b_wins=b_wins,
        draws=draws,
        military=victory_counts[VictoryType.MILITARY],
        scientific=victory_counts[VictoryType.SCIENTIFIC],
        civilian=(
            victory_counts[VictoryType.CIVILIAN]
            + victory_counts[VictoryType.SHARED_CIVILIAN]
        ),
        average_actions=total_actions / games,
    )
