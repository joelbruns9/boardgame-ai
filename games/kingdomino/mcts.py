"""
LEGACY — Pre-AlphaZero MCTS implementation using random rollouts.

This is the original MCTS bot used before the neural network pipeline
was built. It uses UCT with random rollouts rather than a learned value
function. Superseded by mcts_az.py (AlphaZero-style MCTS with neural
network evaluation).

Kept for reference and as a baseline opponent. GreedyBot in bots.py
is a faster baseline for most evaluation purposes.
"""
import math
import random
from dataclasses import dataclass, field

from games.kingdomino.evaluation import evaluate_state, score_action_prior
from games.kingdomino.bots import RandomBot, GreedyBot
from games.kingdomino.bot_match import (
    legal_actions,
    apply_action,
    is_terminal,
    current_player,
)


def total_score(score):
    return score.territory_score + score.harmony_bonus + score.middle_kingdom_bonus


def terminal_value(state, root_player):
    scores = [total_score(board.score()) for board in state.boards]
    margin = scores[root_player] - scores[1 - root_player]
    # Keep value bounded for UCB stability; margins can be large.
    return math.tanh(margin / 50.0)


@dataclass
class MCTSNode:
    parent: object = None
    action: object = None
    player_to_act: int = None
    children: list = field(default_factory=list)
    # untried_actions is kept sorted best-prior-first so widening expands the
    # most promising move next.
    untried_actions: list = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def best_child_ucb(self, exploration=1.4, maximize=True):
        # child.q is always from the root player's perspective:
        # root-controlled nodes maximize q, opponent nodes minimize q.
        best_score = None
        best_children = []
        log_n = math.log(max(1, self.visits))

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.q if maximize else -child.q
                explore = exploration * math.sqrt(log_n / child.visits)
                score = exploit + explore

            if best_score is None or score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)


class MCTSBot:
    def __init__(
        self,
        simulations=200,
        exploration=1.4,
        rollout_policy="random",
        rollout_depth_limit=8,
        pw_c=1.5,
        pw_alpha=0.5,
        seed=None,
        expand_top_k=1,
    ):
        self.simulations = simulations
        self.exploration = exploration
        self.rollout_depth_limit = rollout_depth_limit
        # Progressive widening parameters: a node with v visits may hold up to
        # ceil(pw_c * v**pw_alpha) children. Smaller values => deeper, narrower
        # trees; larger => wider, shallower.
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.rng = random.Random(seed)
        self.expand_top_k = expand_top_k

        if rollout_policy == "random":
            self.rollout_bot = RandomBot()
        elif rollout_policy == "greedy":
            self.rollout_bot = GreedyBot()
        elif rollout_policy == "evaluate":
            # Skip rollouts entirely and use the static evaluator at the leaf.
            self.rollout_bot = None
        else:
            raise ValueError(f"Unknown rollout_policy: {rollout_policy}")

    def _pw_limit(self, visits):
        return max(1, math.ceil(self.pw_c * (visits ** self.pw_alpha)))

    def _ranked(self, state, actions):
        player = current_player(state)
        return sorted(
            actions,
            key=lambda a: score_action_prior(state, a, player=player),
            reverse=True,
        )

    def choose_action(self, state, actions, rng=None):
        rng = rng or self.rng

        if len(actions) == 1:
            return actions[0]

        root_player = current_player(state)
        root = MCTSNode(
            player_to_act=root_player,
            untried_actions=self._ranked(state, actions),
        )

        for _ in range(self.simulations):
            sim_state = state
            node = root
            path = [node]

            # Combined selection + progressive-widening expansion.
            while not is_terminal(sim_state):
                can_widen = (
                    node.untried_actions
                    and len(node.children) < self._pw_limit(node.visits)
                )

                if can_widen:
                    top_k = min(self.expand_top_k, len(node.untried_actions))
                    idx = rng.randrange(top_k)
                    action = node.untried_actions.pop(idx)
                    
                    sim_state = apply_action(sim_state, action)
                    terminal = is_terminal(sim_state)
                    child = MCTSNode(
                        parent=node,
                        action=action,
                        player_to_act=None if terminal else current_player(sim_state),
                        untried_actions=(
                            [] if terminal
                            else self._ranked(sim_state, legal_actions(sim_state))
                        ),
                    )
                    node.children.append(child)
                    node = child
                    path.append(node)
                    break  # evaluate the freshly added leaf

                acting_player = current_player(sim_state)
                maximize = acting_player == root_player
                node = node.best_child_ucb(self.exploration, maximize)
                sim_state = apply_action(sim_state, node.action)
                path.append(node)

            value = self.evaluate_leaf(sim_state, root_player, rng)

            for n in path:
                n.visits += 1
                n.value_sum += value

        # Expose the search tree for debugging/analysis (no effect on play).
        self.last_root = root

        # Root choice: highest mean value among visited children, tie-break by visits.
        visited_children = [c for c in root.children if c.visits > 0]
        best_q = max(c.q for c in visited_children)
        best_children = [c for c in visited_children if c.q == best_q]
        best_visits = max(c.visits for c in best_children)
        best_children = [c for c in best_children if c.visits == best_visits]
        return rng.choice(best_children).action

    def evaluate_leaf(self, state, root_player, rng):
        if is_terminal(state):
            return terminal_value(state, root_player)
        if self.rollout_bot is None:
            return evaluate_state(state, root_player)
        return self.rollout(state, root_player, rng)

    def rollout(self, state, root_player, rng):
        depth = 0
        while not is_terminal(state):
            if self.rollout_depth_limit is not None and depth >= self.rollout_depth_limit:
                return evaluate_state(state, root_player)
            actions = legal_actions(state)
            action = self.rollout_bot.choose_action(state, actions, rng=rng)
            state = apply_action(state, action)
            depth += 1
        return terminal_value(state, root_player)