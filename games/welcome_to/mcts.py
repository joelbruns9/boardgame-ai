"""
Search — a learner-only semi-MDP over the macro vocabulary.

THE ROOT-PLAYER CONTRACT — binding, all four clauses
────────────────────────────────────────────────────
An earlier wording ("no minimax negation; back up the learner's own score") was
underspecified in a way that produces a real bug: ``encode_state`` defaults to
``state.actor``, so a leaf reached while seat 1 is to move silently yields *seat
1's* value, and backing that up as seat 0's is simply wrong.  So, explicitly:

1. **Tree nodes belong only to the root player ``r``.**  No node in this tree is
   ever an opponent's decision.
2. **Every other seat is sampled forward** — one policy sample per
   determinization, no nested search — until ``r`` is to act again.  An
   opponent's turn is a *transition*, not a node.
3. **Leaves are evaluated as ``encode_state(state, r)``**, never
   ``state.actor``.
4. **The backed-up scalar stays in ``r``'s frame** for the whole path.  No
   negation, no reframing.

This is legitimate because Welcome To is near-solitaire: the stacks are shared
and never consumed, so an opponent's concurrent choice cannot change your legal
moves or your immediate scoring.  Sampling them approximates an expectation, and
no equilibrium reasoning is required.

⚠ **S3 needs more than this.** Once several seats are learners, a single scalar
tree that changes viewpoint is not a fixable variant — it is the wrong object.
Run a separate root-player search per acting seat, or move to a vector-valued
Maxⁿ design.  Budget it as real work, not as a flag.

CHANCE IS KEYED, NOT MERGED
───────────────────────────
Open loop: every simulation starts from its own ``redeterminize(search_rng)``.
Keying nodes on the action sequence alone — Kingdomino's ``OpenLoopMCTS`` shape —
would let one node aggregate simulations in which **different cards were
revealed**, mixing distinct observable states into one set of priors, children
and Q.  The sharp consequence is a *selection bias*: an action that is only
**legal** under favourable reveals accumulates Q solely from the simulations
where it was legal, so it is systematically overvalued.

So a child is keyed on ``(action, observed cards)``.  Chance enters only at turn
boundaries, so the branching this adds is confined and shallow.

**Opponent samples are deliberately *not* in the key.**  They are not chance —
they are the opponent model, and averaging over them is exactly the expectation
wanted.  They also cannot create the legality-driven bias above, because nothing
an opponent does changes what ``r`` may play.

THE LEAF VALUE
──────────────
``AUX_TARGETS_SPEC.md`` §5.  The rank head predicts a distribution over finishing
positions; the utility is applied *afterwards*, so the objective can be changed
on a frozen checkpoint instead of in a second training run::

    rank_value   = 2 · Σ p_r·u_r − 1
    margin_value = tanh((score_me − best opponent) · MARGIN_GAIN)
    spread       = 1 − 4·Var[u]
    floor_n      = 1 − (n+1) / (3·(n−1))          0, 1/3, 4/9 for n = 2, 3, 4
    confidence   = max(0, (spread − floor_n) / (1 − floor_n)) ** k
    leaf_value   = (1 − α)·rank_value + α·confidence·margin_value

The gate is the **variance** of the utility, not ``win_value ** 4``.  With three
or more outcomes an expectation cannot carry its own variance — a certain second
place and a coin flip between first and third have the same mean — so a
mean-based gate is a category error, and it goes gradient-dead exactly where it
matters: a 4p seat locked into second has a flat rank *and* margin suppressed by
98%, which is a common position in the last third of every game.

**No auxiliary head enters the leaf value.**  Deciding a permit is worth 0.3
points would be hand-tuning a valuation, and it is redundant: if predicted
permits are informative about score, the score head already uses them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from games.welcome_to import encoder as enc
from games.welcome_to import macro_codec as mc
from games.welcome_to import network as nw
from games.welcome_to import training
from games.welcome_to.game import GameState

#: What the caller observes between its own decisions: every card on the table.
#: All of them are fully identified, so this is public, and it is exactly the
#: chance outcome a turn boundary reveals.
Observation = tuple


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 128
    c_puct: float = 1.5
    #: Leaf-value blend, AUX_TARGETS_SPEC §5.2.  ``k = 1`` is the starting
    #: default; ``k = 2`` is KD parity and suppresses margin to ~2% of leaf
    #: value at turn 12, which discards the better signal for most of the game.
    alpha: float = 0.5
    margin_gain: float = 2.0
    confidence_power: float = 1.0
    #: Root exploration noise.  Off by default; S2 turns it on.
    dirichlet_alpha: Optional[float] = None
    dirichlet_weight: float = 0.25
    #: Visit-count temperature for :func:`play`.  0 plays the argmax.
    temperature: float = 0.0


class Node:
    """One decision by the root player ``r``."""

    __slots__ = ("actions", "prior", "visits", "total", "children", "expanded")

    def __init__(self, actions: np.ndarray, prior: np.ndarray) -> None:
        self.actions = actions
        self.prior = prior
        self.visits = np.zeros(len(actions), dtype=np.float64)
        self.total = np.zeros(len(actions), dtype=np.float64)
        self.children: dict[tuple[int, Observation], "Node"] = {}
        self.expanded = True

    def select(self, c_puct: float) -> int:
        """PUCT.  Returns an index into :attr:`actions`, not a macro index."""
        total_visits = self.visits.sum()
        q = np.divide(
            self.total, self.visits, out=np.zeros_like(self.total), where=self.visits > 0
        )
        exploration = c_puct * self.prior * math.sqrt(max(total_visits, 1.0)) / (
            1.0 + self.visits
        )
        return int(np.argmax(q + exploration))


# ──────────────────────────────────────────────────────────────────────────
# The leaf value
# ──────────────────────────────────────────────────────────────────────────
def confidence_floor(num_seats: int) -> float:
    """``1 − (n+1) / (3·(n−1))`` — the seat-count correction of §5.1a.

    ``1 − 4·Var[u]`` is not comparable across table sizes: a *uniform*
    distribution over three finishing positions already scores higher than one
    over two, so an uncorrected gate would call a 4p position confident merely
    for having four outcomes.  This is that uniform baseline, subtracted off.
    """
    if num_seats < 2:
        return 0.0
    return 1.0 - (num_seats + 1) / (3.0 * (num_seats - 1))


def blend_value(
    rank_probs: Sequence[float],
    scores: Sequence[float],
    num_seats: int,
    config: SearchConfig,
) -> tuple[float, dict[str, float]]:
    """The leaf scalar, in the root player's frame, plus its parts for logging.

    ``rank_probs`` and ``scores`` are the network's outputs **on the seat axis**,
    so index 0 is the root player by construction of ``encode_state(state, r)``.
    ``scores`` are the head's own units (points ÷ 80), which is why the margin
    needs no second division.
    """
    utility = training.rank_utility(num_seats)
    probs = list(rank_probs[:num_seats])
    mean_u = sum(p * u for p, u in zip(probs, utility))
    var_u = sum(p * u * u for p, u in zip(probs, utility)) - mean_u * mean_u
    rank_value = 2.0 * mean_u - 1.0

    if num_seats < 2:
        margin = 0.0
    else:
        margin = math.tanh(
            (scores[0] - max(scores[1:num_seats])) * config.margin_gain
        )

    spread = 1.0 - 4.0 * var_u
    floor = confidence_floor(num_seats)
    confidence = max(0.0, (spread - floor) / (1.0 - floor)) ** config.confidence_power
    value = (1.0 - config.alpha) * rank_value + config.alpha * confidence * margin
    return value, {
        "rank_value": rank_value,
        "margin": margin,
        "var_u": var_u,
        "confidence": confidence,
    }


def terminal_value(state: GameState, root: int, config: SearchConfig) -> float:
    """The same blend, on a finished game, where the distribution is a point mass.

    Computed rather than looked up so that a terminal leaf and an evaluated leaf
    are on one scale — a terminal node valued differently from its own predicted
    value would put a step discontinuity at the end of every line.
    """
    seats = state.config.players
    order = enc.seat_order(state, root)
    distribution = training.rank_distributions(state)[root]
    scores = state.scores()
    ordered = [scores[s] / training.SCORE_SCALE for s in order]
    return blend_value(distribution, ordered, min(seats, enc.MAX_SEATS), config)[0]


# ──────────────────────────────────────────────────────────────────────────
# The evaluator
# ──────────────────────────────────────────────────────────────────────────
class NetEvaluator:
    """Wraps the network: one forward gives priors, the leaf value and a policy.

    Every call is ``encode_state(state, viewer)`` with an explicit viewer —
    clause 3 of the contract.  The default-to-``state.actor`` behaviour is never
    used here, and that is the whole point of passing it.
    """

    def __init__(
        self,
        net: nw.WelcomeToNet,
        device: Optional[torch.device] = None,
        config: Optional[SearchConfig] = None,
    ) -> None:
        self.net = net
        self.device = device or next(net.parameters()).device
        self.config = config or SearchConfig()
        self.calls = 0

    @torch.no_grad()
    def _forward(self, state: GameState, viewer: int) -> dict[str, torch.Tensor]:
        self.net.eval()
        self.calls += 1
        arrays = enc.encode_state(state, viewer)
        tensors = [
            torch.as_tensor(a).unsqueeze(0).float().to(self.device) for a in arrays
        ]
        return self.net(*tensors)

    def evaluate(self, state: GameState, viewer: int) -> tuple[np.ndarray, float]:
        """``(priors over the legal macros, leaf value in ``viewer``'s frame)``."""
        out = self._forward(state, viewer)
        seats = min(state.config.players, enc.MAX_SEATS)

        mask = np.zeros(training.MAX_RANKS, dtype=np.float32)
        mask[:seats] = 1.0
        rank_probs = nw.rank_probabilities(
            out["rank_logits"], torch.as_tensor(mask).to(self.device).unsqueeze(0)
        )[0].cpu().numpy()
        scores = out["score"][0].cpu().numpy()
        value, _ = blend_value(rank_probs, scores, seats, self.config)

        legal = mc.legal_mask(state)
        logits = out["policy_logits"][0].cpu().numpy()
        priors = _masked_softmax(logits, legal)
        return priors, value

    def policy(self, state: GameState, viewer: int) -> np.ndarray:
        """The bare policy, for sampling an opponent forward.  No value, no tree."""
        out = self._forward(state, viewer)
        logits = out["policy_logits"][0].cpu().numpy()
        return _masked_softmax(logits, mc.legal_mask(state))


def _masked_softmax(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, logits, -np.inf)
    shifted = masked - masked.max()
    exp = np.exp(shifted, where=legal, out=np.zeros_like(shifted))
    total = exp.sum()
    if total <= 0.0:  # every legal logit underflowed
        out = legal.astype(np.float32)
        return out / max(out.sum(), 1.0)
    return (exp / total).astype(np.float32)


#: An opponent model: given a state and the seat to move, apply one move.
OpponentPolicy = Callable[[GameState, int, random.Random], None]


def sampling_opponent(evaluator: NetEvaluator) -> OpponentPolicy:
    """Sample one move from the network's own policy — clause 2, no nested search."""

    def move(state: GameState, seat: int, rng: random.Random) -> None:
        probs = evaluator.policy(state, seat)
        index = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        mc.apply_macro(state, index)

    return move


# ──────────────────────────────────────────────────────────────────────────
# Search
# ──────────────────────────────────────────────────────────────────────────
class MCTS:
    def __init__(
        self,
        evaluator: NetEvaluator,
        config: Optional[SearchConfig] = None,
        opponent: Optional[OpponentPolicy] = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or evaluator.config
        self.opponent = opponent or sampling_opponent(evaluator)

    # ── the transition: everything between two of r's decisions ──────────
    def _advance(
        self, state: GameState, root: int, rng: random.Random
    ) -> Observation:
        """Sample every other seat forward until ``root`` is to act, or the end.

        Clause 2.  An opponent's turn is a transition, not a node, so nothing
        here is stored and nothing here is searched.
        """
        guard = 0
        while not state.is_terminal and state.actor != root:
            self.opponent(state, state.actor, rng)
            guard += 1
            if guard > 5000:  # pragma: no cover - a stuck engine, not a rules case
                raise RuntimeError("opponents did not yield the turn")
        return tuple(state.table_cards(root))

    def _leaf(self, state: GameState, root: int) -> tuple[Optional[Node], float]:
        if state.is_terminal:
            return None, terminal_value(state, root, self.config)
        priors, value = self.evaluator.evaluate(state, root)
        actions = np.asarray(mc.legal_macros(state), dtype=np.int64)
        prior = priors[actions]
        total = prior.sum()
        prior = prior / total if total > 0 else np.full(len(actions), 1.0 / len(actions))
        return Node(actions, prior.astype(np.float64)), value

    def _simulate(self, state: GameState, node: Node, root: int, rng: random.Random) -> float:
        path: list[tuple[Node, int]] = []
        while True:
            index = node.select(self.config.c_puct)
            path.append((node, index))
            mc.apply_macro(state, int(node.actions[index]))
            observation = self._advance(state, root, rng)

            key = (int(node.actions[index]), observation)
            child = node.children.get(key)
            if child is None:
                child, value = self._leaf(state, root)
                if child is not None:
                    node.children[key] = child
                break
            node = child

        for parent, index in path:
            parent.visits[index] += 1.0
            parent.total[index] += value  # clause 4: r's frame, never negated
        return value

    def search(
        self,
        state: GameState,
        root: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> tuple[np.ndarray, np.ndarray, Node]:
        """``(macro indices, visit counts, root node)``.

        ``state`` is not modified.  Each simulation gets its own
        ``redeterminize(rng)``, and ``rng`` is advanced by the call — passing the
        state's own generator would return the identical shuffle every time,
        because ``copy`` clones the RNG state exactly.
        """
        root = state.actor if root is None else root
        rng = rng or random.Random()
        if state.actor != root:
            raise ValueError("search starts at a state the root player is to act in")

        node, _ = self._leaf(state, root)
        if node is None:
            raise ValueError("cannot search a finished game")
        self._apply_root_noise(node, rng)

        for _ in range(self.config.simulations):
            self._simulate(state.redeterminize(rng), node, root, rng)
        return node.actions, node.visits, node

    def _apply_root_noise(self, node: Node, rng: random.Random) -> None:
        if self.config.dirichlet_alpha is None or len(node.actions) < 2:
            return
        noise = np.random.default_rng(rng.randrange(1 << 32)).dirichlet(
            [self.config.dirichlet_alpha] * len(node.actions)
        )
        weight = self.config.dirichlet_weight
        node.prior = (1.0 - weight) * node.prior + weight * noise

    # ── playing ──────────────────────────────────────────────────────────
    def play(
        self,
        state: GameState,
        root: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Choose one macro for the acting seat, and return it.

        A single legal move skips the search entirely.  Half of all decisions in
        this game have two options or fewer, so this is not a micro-optimisation
        — it is most of the budget.
        """
        root = state.actor if root is None else root
        legal = mc.legal_macros(state)
        if len(legal) == 1:
            return legal[0]

        actions, visits, _ = self.search(state, root, rng)
        if self.config.temperature <= 0.0:
            return int(actions[int(np.argmax(visits))])
        weights = visits ** (1.0 / self.config.temperature)
        if weights.sum() <= 0.0:  # pragma: no cover - every simulation failed
            return int(actions[int(np.argmax(visits))])
        rng = rng or random.Random()
        return int(actions[rng.choices(range(len(actions)), weights=weights, k=1)[0]])


def visit_policy(actions: np.ndarray, visits: np.ndarray) -> np.ndarray:
    """The visit distribution as a full-width vector — S2's policy target."""
    policy = np.zeros(mc.NUM_MACRO_ACTIONS, dtype=np.float32)
    total = visits.sum()
    if total > 0:
        policy[actions] = visits / total
    return policy


def arena_player(search: MCTS):
    """Adapt a search to :mod:`games.welcome_to.arena`'s ``Player`` protocol.

    The seat is passed explicitly and used as the search root -- the arena
    substitutes one seat, so the root is that seat and never ``state.actor`` by
    default.  Clause 3 again, one layer up.
    """

    def move(state: GameState, seat: int, rng: random.Random) -> None:
        mc.apply_macro(state, search.play(state, root=seat, rng=rng))

    return move
