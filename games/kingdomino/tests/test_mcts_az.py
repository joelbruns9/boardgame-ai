"""
mcts_az.py — AlphaZero-style PUCT search for Kingdomino.

This is the self-play / evaluation search.  It is deliberately a SEPARATE
class from the legacy UCB+heuristic MCTS:
  - No import of evaluation.py.  The heuristic never touches training data.
  - Leaf value comes ONLY from the network's value head.
  - Action priors come ONLY from the network's policy head.
  - No rollout policy, no rollouts.
A grep for "evaluation" in this file should find nothing.

PUCT SELECTION
At each node, pick a = argmax_a [ Q(a) + c_puct · P(a) · √N(s) / (1 + N(a)) ]
where P(a) is the network policy prior, N(s) the node visit count, N(a) the
edge visit count, and Q(a) the mean action value.

VALUE PERSPECTIVE — fixed player-0 frame
Every value stored in the tree is expressed from player 0's perspective.
  - Network leaf value is from state.current_actor's view → negate if actor==1.
  - Terminal value = compute_target_z(state, player=0).
  - Backup adds the player-0 value to every node on the path; NO sign flips.
  - At selection, a node maximises from ITS actor's view: q = child.Q if
    actor==0 else -child.Q.
This avoids per-node negation bugs during backup (the classic 2-player MCTS
footgun) and is correct because the value target is zero-sum: tanh(margin)
for player 0 is exactly -tanh(margin) for player 1.

IMPERFECT INFORMATION — determinization is the CALLER'S job
`search(root_state)` treats root_state's deck as a fixed, known world (closed
-loop PUCT, perfect information from here).  For information-set-safe play the
caller must pass a *determinized* root — sample the hidden deck with
`encoder.redeterminize` first.  `run_pimc` does this and aggregates visit
counts over several determinizations (Perfect Information Monte Carlo).

Why this is safe: the encoder computes features from public information only
(it never reads deck order), so the network sees identical inputs regardless
of which determinization the engine is stepping through.  The determinization
only decides which tiles get revealed in future rounds during simulation; it
never leaks into what the network evaluates.

NOTE: leaf evaluations are single (one network call per expansion).  Batching
leaf evaluations across parallel simulations/games is the main throughput
optimisation for later; the algorithm here is correctness-first.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from games.kingdomino.game import GameState, Phase
from games.kingdomino.encoder import encode_state, compute_target_z, redeterminize
from games.kingdomino.action_codec import (
    NUM_JOINT_ACTIONS, encode_action,
)


class Node:
    """A search-tree node. Edge statistics are stored in the child nodes."""
    __slots__ = ("state", "prior", "visit_count", "value_sum",
                 "children", "is_expanded")

    def __init__(self, prior: float):
        self.state: Optional[GameState] = None  # set lazily when descended into
        self.prior: float = prior               # P(parent → this), from policy head
        self.visit_count: int = 0
        self.value_sum: float = 0.0             # PLAYER-0 frame
        self.children: Dict[object, "Node"] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        """Mean value in player-0 frame (0.0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


# An Evaluator isolates the ONLY part of MCTS that touches the network: a
# single forward pass.  It maps one encoded position to (value, policy_logits):
#     (my_board (9,13,13), opp_board (9,13,13), flat (261,))
#       -> (value  in [-1,1] from the ENCODED player's perspective,
#           logits (NUM_JOINT_ACTIONS,) over the full joint action space)
# MCTS owns encoding and legal-action prior extraction; the evaluator owns the
# network and device.  This is the seam that lets a serial (batch-1) evaluator
# be swapped for a batched inference-server evaluator with no MCTS changes.
Evaluator = Callable[
    [np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]
]


def make_serial_evaluator(network: torch.nn.Module, device: str = "cpu") -> Evaluator:
    """Wrap a network into a batch-1 Evaluator (eval mode, no_grad, on device)."""
    network = network.to(device).eval()

    def evaluator(mb: np.ndarray, ob: np.ndarray, flat: np.ndarray):
        with torch.no_grad():
            mb_t = torch.from_numpy(mb).unsqueeze(0).to(device)
            ob_t = torch.from_numpy(ob).unsqueeze(0).to(device)
            flat_t = torch.from_numpy(flat).unsqueeze(0).to(device)
            v, logits = network(mb_t, ob_t, flat_t)
        return float(v.item()), logits[0].detach().cpu().numpy()

    return evaluator


class AlphaZeroMCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        *,
        c_puct: float = 1.5,
        n_simulations: int = 100,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu: float = 0.0,
    ):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu = fpu  # first-play-urgency: Q assumed for unvisited edges

    # ── public ──
    def search(
        self,
        root_state: GameState,
        *,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[Dict[object, int], Node]:
        """Run n_simulations of PUCT from `root_state`.

        Returns (visit_counts, root):
            visit_counts : {action: edge visit count} at the root
            root         : the root Node (for value / diagnostics)

        IMPORTANT: `root_state` is treated as a fully-known world.  For
        information-set-safe play, pass a determinized state (see run_pimc).
        """
        if root_state.phase == Phase.GAME_OVER:
            raise ValueError("Cannot search from a terminal state.")

        root = Node(prior=1.0)
        root.state = root_state
        root_value0 = self._expand(root)
        root.visit_count = 1
        root.value_sum = root_value0

        if add_noise:
            self._add_dirichlet_noise(root, rng)

        for _ in range(self.n_simulations):
            self._simulate(root)

        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        return visit_counts, root

    # ── internals ──
    def _simulate(self, root: Node) -> None:
        path = [root]
        node = root
        # Descend until an unexpanded node or a terminal state.
        while node.is_expanded and not (node.state.phase == Phase.GAME_OVER):
            action, child = self._select_child(node)
            if child.state is None:
                child.state = node.state.step(action)
            path.append(child)
            node = child

        if (node.state.phase == Phase.GAME_OVER):
            v0 = compute_target_z(node.state, player=0)
        else:
            v0 = self._expand(node)

        # Backup — all values in player-0 frame, no sign flips.
        for n in path:
            n.visit_count += 1
            n.value_sum += v0

    def _select_child(self, node: Node) -> Tuple[object, Node]:
        actor = node.state.current_actor
        sqrt_n = math.sqrt(node.visit_count)
        best_score = -math.inf
        best = None
        for action, child in node.children.items():
            if child.visit_count > 0:
                q0 = child.value_sum / child.visit_count   # player-0 frame
                q = q0 if actor == 0 else -q0               # → actor's frame
            else:
                q = self.fpu
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = (action, child)
        return best

    def _expand(self, node: Node) -> float:
        """Evaluate the network at node.state, create child edges with priors.

        Returns the leaf value in PLAYER-0 frame.
        """
        state = node.state
        value0, priors = self._evaluate(state)
        for action, p in priors.items():
            node.children[action] = Node(prior=p)
        node.is_expanded = True
        return value0

    def _evaluate(self, state: GameState) -> Tuple[float, Dict[object, float]]:
        """Single network evaluation. Returns (value_player0, {action: prior})."""
        legal = state.legal_actions()
        if not legal:
            raise ValueError(
                f"_evaluate received a non-terminal state with no legal "
                f"actions (phase={state.phase.name})."
            )
        mb, ob, flat = encode_state(state, state.current_actor)
        v, logits = self.evaluator(mb, ob, flat)

        if not np.isfinite(logits).all() or not math.isfinite(v):
            raise FloatingPointError(
                "Evaluator returned NaN/Inf in MCTS evaluation."
            )

        # Priors over legal actions only.  Indices come from the engine's
        # legal_actions via encode_action, so they're guaranteed legal.
        idxs = np.fromiter((encode_action(a, state) for a in legal),
                           dtype=np.int64, count=len(legal))
        legal_logits = logits[idxs]
        legal_logits = legal_logits - legal_logits.max()  # stable softmax
        exp = np.exp(legal_logits)
        probs = exp / exp.sum()
        priors = {a: float(p) for a, p in zip(legal, probs)}

        value0 = v if state.current_actor == 0 else -v
        return value0, priors

    def _add_dirichlet_noise(self, root: Node, rng: Optional[np.random.Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        actions = list(root.children.keys())
        if not actions:
            return
        noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - eps) * child.prior + eps * float(n)


# ─── helpers ──────────────────────────────────────────────────────────────
def root_value_for_actor(root: Node, state: GameState) -> float:
    """Root value in the current actor's frame (player-0 stored frame → actor)."""
    v0 = root.value
    return v0 if state.current_actor == 0 else -v0


def visit_counts_to_policy(
    visit_counts: Dict[object, int],
    state: GameState,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert visit counts to a (NUM_JOINT_ACTIONS,) policy target.

    temperature=1.0 → proportional to visits (the standard AlphaZero policy
    target).  temperature→0 → all mass on the most-visited action.
    """
    policy = np.zeros(NUM_JOINT_ACTIONS, dtype=np.float32)
    if not visit_counts:
        raise ValueError("Cannot build policy target from empty visit_counts.")
    actions = list(visit_counts.keys())
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if counts.sum() <= 0:
        raise ValueError(
            "All root visit counts are zero; increase n_simulations."
        )

    if temperature <= 1e-6:
        best = actions[int(counts.argmax())]
        policy[encode_action(best, state)] = 1.0
        return policy

    weights = counts ** (1.0 / temperature)
    weights /= weights.sum()
    for a, w in zip(actions, weights):
        policy[encode_action(a, state)] = w
    return policy


def select_move(
    visit_counts: Dict[object, int],
    temperature: float,
    rng: np.random.Generator,
) -> object:
    """Pick a move from visit counts. temperature=0 → greedy (argmax visits)."""
    actions = list(visit_counts.keys())
    if not actions:
        raise ValueError("Cannot select a move from empty visit_counts.")
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if counts.sum() <= 0:
        raise ValueError(
            "All root visit counts are zero; increase n_simulations."
        )
    if temperature <= 1e-6:
        return actions[int(counts.argmax())]
    weights = counts ** (1.0 / temperature)
    weights /= weights.sum()
    return actions[int(rng.choice(len(actions), p=weights))]


def run_pimc(
    mcts: AlphaZeroMCTS,
    public_state: GameState,
    py_rng: random.Random,
    *,
    n_determinizations: int = 1,
    add_noise: bool = False,
    np_rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[object, int], float]:
    """Perfect Information Monte Carlo over several determinizations.

    Samples `n_determinizations` worlds consistent with the public information
    set (via redeterminize), runs a full PUCT search on each, and aggregates
    root visit counts.  Returns (aggregated_visit_counts, mean_root_value in
    player-0 frame).

    The root's legal actions are determinization-independent (they depend only
    on public info — current_row and the pending claim), so aggregating root
    visit counts across determinizations is well-defined.
    """
    agg: Dict[object, float] = defaultdict(float)
    total_value0 = 0.0
    for _ in range(n_determinizations):
        det_state = redeterminize(public_state, py_rng)
        visit_counts, root = mcts.search(det_state, add_noise=add_noise, rng=np_rng)
        for a, c in visit_counts.items():
            agg[a] += c
        total_value0 += root.value
    return dict(agg), total_value0 / n_determinizations


if __name__ == "__main__":
    from games.kingdomino.network import KingdominoNet
    net = KingdominoNet(channels=48, blocks=4, bilinear_dim=32)
    mcts = AlphaZeroMCTS(make_serial_evaluator(net), n_simulations=50)
    state = GameState.new(seed=0)
    rng = random.Random(0)
    while state.phase != Phase.PLACE_AND_SELECT:
        state = state.step(rng.choice(state.legal_actions()))
    vc, root = mcts.search(state)
    print(f"phase={state.phase.name} legal={len(state.legal_actions())} "
          f"visited={len(vc)} total_visits={sum(vc.values())} "
          f"root_value0={root.value:+.3f}")