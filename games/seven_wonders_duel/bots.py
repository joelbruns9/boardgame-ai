"""Baseline players and reproducible match runners."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from .data import CARDS_BY_NAME, PROGRESS_BY_NAME, WONDERS_BY_NAME, CardColor
from .engine import Action, ActionUse, apply_action, legal_actions, score_player
from .game import GameState, Phase, VictoryType, new_game


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
    return len(symbols)


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
        player = game.active_player
        actions = _require_actions(game)
        scored: list[tuple[float, tuple, Action]] = []
        for action in actions:
            child = game.clone()
            apply_action(child, action)
            scored.append((evaluate_state(child, player), _action_tiebreak(action), action))
        return max(scored)[2]


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
        action = bots[game.active_player].select_action(game)
        if action not in legal_actions(game):
            raise ValueError(f"{bots[game.active_player].name} returned illegal action: {action}")
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
