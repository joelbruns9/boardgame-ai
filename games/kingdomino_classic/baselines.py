"""Stable handcrafted baselines for Classic Kingdomino.

The heuristic bots score complete actions.  During ordinary rounds that means
the current placement *and* the next-row claim; during setup it means only the
claim.  Each stronger tier retains the preceding tier's terms so ablations have
a clear interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol, Sequence

from .board import ClassicBoard
from .dominoes import DOMINOES, Domino
from .game import Action, ClassicGameState, PickAction, TurnAction


class Bot(Protocol):
    name: str

    def choose_action(
        self,
        state: ClassicGameState,
        actions: Sequence[Action] | None = None,
        rng: random.Random | None = None,
    ) -> Action: ...


def _rng_or_default(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random.Random()


def _tile_crowns(domino: Domino) -> int:
    return domino.a.crowns + domino.b.crowns


def _tile_quality(domino_id: int) -> float:
    """Cheap public material value used before a board-specific evaluation."""

    domino = DOMINOES[domino_id]
    # Number is the official draft-order cost: stronger/crown-heavy dominoes
    # tend to act later next round.  Keep it a small tiebreak relative to crowns.
    return 3.0 * _tile_crowns(domino) + domino_id / 48.0


def _configured_score(state: ClassicGameState, board: ClassicBoard) -> int:
    return board.score(
        harmony=state.config.harmony,
        middle_kingdom=state.config.middle_kingdom,
    ).total


def _after_placement(
    state: ClassicGameState,
    action: Action,
    cache: dict[object, object],
) -> ClassicBoard | None:
    if not isinstance(action, TurnAction):
        return None
    actor = state.current_actor
    if action.placement is None:
        return state.boards[actor]
    key = ("after", action.placement)
    cached = cache.get(key)
    if isinstance(cached, ClassicBoard):
        return cached
    board = state.boards[actor].copy()
    claim = state.pending_claims[0]
    board.place(DOMINOES[claim.domino_id], action.placement)
    cache[key] = board
    return board


def _select_best(
    scored: Sequence[tuple[float, Action]], rng: random.Random
) -> Action:
    if not scored:
        raise ValueError("A bot cannot choose from an empty action list.")
    best = max(score for score, _action in scored)
    tied = [action for score, action in scored if score == best]
    return rng.choice(tied)


@dataclass(frozen=True, slots=True)
class RandomBot:
    name: str = "random"

    def choose_action(
        self,
        state: ClassicGameState,
        actions: Sequence[Action] | None = None,
        rng: random.Random | None = None,
    ) -> Action:
        legal = list(actions if actions is not None else state.legal_actions())
        if not legal:
            raise ValueError("RandomBot received no legal actions.")
        return _rng_or_default(rng).choice(legal)


@dataclass(frozen=True, slots=True)
class ImmediateScoreBot:
    """Greedy placement score plus a simple value for the next domino."""

    name: str = "immediate_score"
    placement_gain_weight: float = 8.0
    forced_discard_penalty: float = 24.0
    pick_quality_weight: float = 1.0

    def evaluate_action(
        self,
        state: ClassicGameState,
        action: Action,
        cache: dict[object, object] | None = None,
    ) -> float:
        cache = {} if cache is None else cache
        score = 0.0
        if isinstance(action, TurnAction):
            actor = state.current_actor
            if action.placement is None:
                score -= self.forced_discard_penalty
            else:
                before_key = ("territory", actor, "before")
                before = cache.get(before_key)
                if not isinstance(before, int):
                    before = _configured_score(state, state.boards[actor])
                    cache[before_key] = before
                after_board = _after_placement(state, action, cache)
                assert after_board is not None
                score += self.placement_gain_weight * (
                    _configured_score(state, after_board) - before
                )

        pick_id = (
            action.domino_id
            if isinstance(action, PickAction)
            else action.pick_domino_id
        )
        if pick_id is not None:
            score += self.pick_quality_weight * _tile_quality(pick_id)
        return score

    def choose_action(
        self,
        state: ClassicGameState,
        actions: Sequence[Action] | None = None,
        rng: random.Random | None = None,
    ) -> Action:
        legal = list(actions if actions is not None else state.legal_actions())
        if len(legal) == 1:
            return legal[0]
        cache: dict[object, object] = {}
        scored = [
            (self.evaluate_action(state, action, cache), action) for action in legal
        ]
        return _select_best(scored, _rng_or_default(rng))


@dataclass(frozen=True, slots=True)
class FlexibilityBot(ImmediateScoreBot):
    """Greedy bot that values whether its claimed domino remains placeable."""

    name: str = "flexibility"
    future_placement_weight: float = 0.45
    unplaceable_pick_penalty: float = 5.0
    compactness_weight: float = 0.05

    def evaluate_action(
        self,
        state: ClassicGameState,
        action: Action,
        cache: dict[object, object] | None = None,
    ) -> float:
        cache = {} if cache is None else cache
        score = ImmediateScoreBot.evaluate_action(self, state, action, cache)
        actor = state.current_actor
        board = _after_placement(state, action, cache) or state.boards[actor]

        if isinstance(action, TurnAction) and action.placement is not None:
            min_x, min_y, max_x, max_y = board.occupied_bbox()
            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            holes = bbox_area - board.occupied_count
            # Internal holes preserve ways to grow without expanding the 5x5
            # bounding box.  This is a small shape-health term, not a score proxy.
            score += self.compactness_weight * holes

        pick_id = (
            action.domino_id
            if isinstance(action, PickAction)
            else action.pick_domino_id
        )
        if pick_id is not None:
            key = ("future_placements", board.state_key(), pick_id)
            count = cache.get(key)
            if not isinstance(count, int):
                count = len(board.legal_placements(DOMINOES[pick_id]))
                cache[key] = count
            if count == 0:
                score -= self.unplaceable_pick_penalty
            else:
                score += self.future_placement_weight * math.log1p(count)
        return score


@dataclass(frozen=True, slots=True)
class DenialAwareBot(FlexibilityBot):
    """Flexibility bot that also prices a claim's value to rival kingdoms."""

    name: str = "denial_aware"
    denial_weight: float = 0.5
    three_player_denial_scale: float = 0.4
    opponent_score_pressure: float = 0.015

    @staticmethod
    def _opponent_tile_value(
        state: ClassicGameState,
        board: ClassicBoard,
        domino_id: int,
        cache: dict[object, object],
    ) -> float:
        key = (
            "opponent_tile",
            state.config.harmony,
            state.config.middle_kingdom,
            board.state_key(),
            domino_id,
        )
        cached = cache.get(key)
        if isinstance(cached, float):
            return cached
        domino = DOMINOES[domino_id]
        placements = board.legal_placements(domino)
        if not placements:
            value = -12.0
        else:
            before = _configured_score(state, board)
            best_gain = 0
            for placement in placements:
                child = board.copy()
                child.place(domino, placement)
                best_gain = max(best_gain, _configured_score(state, child) - before)
            value = (
                5.0 * best_gain
                + 1.5 * math.log1p(len(placements))
                + _tile_quality(domino_id)
            )
        cache[key] = float(value)
        return float(value)

    def evaluate_action(
        self,
        state: ClassicGameState,
        action: Action,
        cache: dict[object, object] | None = None,
    ) -> float:
        cache = {} if cache is None else cache
        score = FlexibilityBot.evaluate_action(self, state, action, cache)
        pick_id = (
            action.domino_id
            if isinstance(action, PickAction)
            else action.pick_domino_id
        )
        if pick_id is None:
            return score

        actor = state.current_actor
        configured_scores = tuple(
            _configured_score(state, board) for board in state.boards
        )
        actor_score = configured_scores[actor]
        denial_pressure = float("-inf")
        for opponent, board in enumerate(state.boards):
            if opponent == actor:
                continue
            leader_multiplier = 1.0 + self.opponent_score_pressure * max(
                0, configured_scores[opponent] - actor_score
            )
            value = self._opponent_tile_value(state, board, pick_id, cache)
            denial_pressure = max(denial_pressure, leader_multiplier * value)
        if denial_pressure != float("-inf"):
            # In 3p, one of the four offered dominoes is discarded.  A rival-
            # friendly tile therefore needs less active denial than in 4p,
            # where every tile must be claimed.
            player_count_scale = (
                self.three_player_denial_scale if state.config.players == 3 else 1.0
            )
            score += self.denial_weight * player_count_scale * denial_pressure
        return score


BASELINE_FACTORIES = {
    "random": RandomBot,
    "immediate_score": ImmediateScoreBot,
    "flexibility": FlexibilityBot,
    "denial_aware": DenialAwareBot,
}
