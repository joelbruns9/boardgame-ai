import random
from dataclasses import dataclass

from games.kingdomino.dominoes import DOMINOES
from games.kingdomino.game import Phase


class RandomBot:
    def choose_action(self, state, actions, rng=None):
        rng = rng or random
        return rng.choice(actions)


def current_player(state):
    current_actor = getattr(state, "current_actor", None)

    if callable(current_actor):
        actor = current_actor()
        return actor.player

    if isinstance(current_actor, int):
        return current_actor

    if state.phase == Phase.INITIAL_SELECTION:
        if state.initial_pick_count in (0, 3):
            return state.start_player
        return 1 - state.start_player

    claim = state.pending_claims[state.actor_index]
    return claim.player


def current_domino(state):
    if state.phase == Phase.INITIAL_SELECTION:
        return None

    if hasattr(state, "current_actor_domino"):
        value = state.current_actor_domino
        return value() if callable(value) else value

    claim = state.pending_claims[state.actor_index]
    return DOMINOES[claim.domino_id]


def total_score(score):
    return score.territory_score + score.harmony_bonus + score.middle_kingdom_bonus


@dataclass
class GreedyBot:
    placement_weight: float = 1.0
    pick_crown_weight: float = 8.0
    pick_id_weight: float = 0.05
    middle_kingdom_weight: float = 1.0
    harmony_weight: float = 1.0

    def choose_action(self, state, actions, rng=None):
        rng = rng or random

        best_score = None
        best_actions = []

        for action in actions:
            score = self.evaluate_action(state, action)

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return rng.choice(best_actions)

    def evaluate_action(self, state, action):
        score = 0.0

        placement = getattr(action, "placement", None)

        if placement is not None:
            player = current_player(state)
            domino = current_domino(state)

            before_score = state.boards[player].score()
            before = total_score(before_score)

            board_copy = state.boards[player].copy()
            board_copy.place(domino, placement)

            after_score = board_copy.score()
            after = total_score(after_score)

            score += self.placement_weight * (after - before)

            # Lightly reward bonuses directly so they are not ignored.
            score += self.middle_kingdom_weight * (
                after_score.middle_kingdom_bonus - before_score.middle_kingdom_bonus
            )
            score += self.harmony_weight * (
                after_score.harmony_bonus - before_score.harmony_bonus
            )

        pick_id = getattr(action, "pick_domino_id", None)

        if pick_id is not None:
            domino = DOMINOES[pick_id]
            crowns = domino.a.crowns + domino.b.crowns

            score += self.pick_crown_weight * crowns
            score += self.pick_id_weight * pick_id

        return score