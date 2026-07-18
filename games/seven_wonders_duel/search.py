"""Dual-mode MCTS with a Gumbel root (plan §5, Phase C Python reference).

Structure mirrors the proven Kingdomino `mcts_az.py` dual-tree design
(closed `AlphaZeroMCTS` + `OpenLoopMCTS`), rewritten for what 7WD needs that
Kingdomino's tree does not have:

- **Gumbel root**: top-k by Gumbel + log-prior, sequential halving over the
  sims budget, completed-Q improved policy target (the §2 lever vs ZeusAI).
- **Explicit chance layer** (closed mode): the searcher PREDICTS each action's
  chance events from public information (`chance_signature`), samples outcomes
  from `UnseenPool` enumerations, and steps barred clones with explicit
  `chance_outcomes` — the locked deal is never read (HiddenInformationError
  otherwise). Chance edges use exact probability-weighted expectation over
  expanded children (Star-style), so a fully expanded tree reproduces
  expectimax to float precision (`closed_root_exact_value`, the §5 gate).
- **Open mode**: nodes keyed by action path; each descent re-determinizes the
  root clone via `resample_hidden` and walks it in simulator mode; legality is
  re-masked per world; priors cached at first expansion (the known weakness
  the Phase E A/B measures).
- **Per-node actors**: extra turns and pending choices break strict
  alternation, so values are stored player-0-relative and converted per node.

The actor sequence along a path is deterministic given the actions (reveals
change identities, never who acts next), which is what makes path-keyed open
nodes and closed-tree actors well-defined.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .codec import decode_action, legal_action_indices
from .data import TABLEAU_LAYOUTS, BackType, back_type_of, covering_slots
from .encoder import encode
from .engine import Action, ActionUse, apply_action
from .game import ChanceKind, GameState, Phase
from .inference import Evaluator
from .pool import (
    enumerate_card_reveal,
    enumerate_great_library,
    enumerate_wonder_flip,
    unseen_pool,
)

AGE_BACKS = {1: BackType.AGE_I, 2: BackType.AGE_II, 3: BackType.AGE_III}


# --------------------------------------------------------------------------
# Chance signature: public prediction of the events an action will fire.
# Gated against actual engine StepResult events in test_search.py.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChanceSpec:
    kind: ChanceKind
    context: tuple = ()


def _newly_accessible_after_take(state: GameState, taken) -> list[tuple]:
    """(slot_id, back) of face-down cards a take would expose — public info:
    layout coverage is printed on the board and backs are visible."""

    tableau = state.tableau
    layout = TABLEAU_LAYOUTS[tableau.age]
    present = {
        slot_id
        for slot_id, card in tableau.cards.items()
        if card.present and slot_id != taken
    }
    exposed = []
    for slot_id, card in tableau.cards.items():
        if slot_id == taken or not card.present or card.revealed:
            continue
        coverers = covering_slots(layout, card.slot)
        if not any((c.row, c.x) in present for c in coverers):
            exposed.append((slot_id, back_type_of(card.card_name)))
    return sorted(exposed)


def chance_signature(state: GameState, action: Action) -> tuple[ChanceSpec, ...]:
    if action.use is ActionUse.DRAFT_WONDER:
        specs = []
        if state.wonder_round == 0 and state.wonder_pick_index == 3:
            specs.append(ChanceSpec(ChanceKind.WONDER_GROUP_REVEAL))
        if state.wonder_round == 1 and state.wonder_pick_index == 3:
            specs.append(ChanceSpec(ChanceKind.AGE_DEAL, (1,)))
        return tuple(specs)
    if action.use is ActionUse.CHOOSE_NEXT_START_PLAYER:
        return (ChanceSpec(ChanceKind.AGE_DEAL, (state.age + 1,)),)
    if action.use is ActionUse.RESOLVE_PENDING_CHOICE:
        return ()
    specs = [
        ChanceSpec(ChanceKind.CARD_REVEAL, (slot_id, back))
        for slot_id, back in _newly_accessible_after_take(state, action.slot_id)
    ]
    if (
        action.use is ActionUse.CONSTRUCT_WONDER
        and action.wonder_name == "The Great Library"
        and state.unused_progress_tokens
    ):
        specs.append(ChanceSpec(ChanceKind.GREAT_LIBRARY_DRAW))
    return tuple(specs)


def sample_outcomes(
    state: GameState, specs, rng: random.Random
) -> tuple[list, float | None, tuple]:
    """Sample one outcome per spec. Returns (outcomes, joint probability or
    None when any event is sample-only, hashable child key). Sequential
    CARD_REVEALs condition on earlier outcomes (same-back pools shrink)."""

    pool = unseen_pool(state.observation(state.active_player))
    used: set[str] = set()
    outcomes: list = []
    probability: float | None = 1.0
    for spec in specs:
        if spec.kind is ChanceKind.CARD_REVEAL:
            back = spec.context[1]
            names = [
                name
                for name, _ in enumerate_card_reveal(pool, back)
                if name not in used
            ]
            choice = names[rng.randrange(len(names))]
            used.add(choice)
            outcomes.append(choice)
            if probability is not None:
                probability *= 1.0 / len(names)
        elif spec.kind is ChanceKind.GREAT_LIBRARY_DRAW:
            subsets = enumerate_great_library(pool)
            subset, p = subsets[rng.randrange(len(subsets))]
            outcomes.append(subset)
            if probability is not None:
                probability *= p
        elif spec.kind is ChanceKind.WONDER_GROUP_REVEAL:
            flips = enumerate_wonder_flip(pool)
            subset, p = flips[rng.randrange(len(flips))]
            outcomes.append(subset)
            if probability is not None:
                probability *= p
        elif spec.kind is ChanceKind.AGE_DEAL:
            age = spec.context[0]
            names = sorted(pool.cards[AGE_BACKS[age]])
            rng.shuffle(names)
            if age == 3:
                guilds = sorted(pool.cards[BackType.GUILD])
                rng.shuffle(guilds)
                deal = names[:17] + guilds[:3]
                rng.shuffle(deal)
            else:
                deal = names[: len(TABLEAU_LAYOUTS[age])]
            outcomes.append(tuple(deal))
            probability = None  # sample-only event (spec §4.2)
        else:  # pragma: no cover
            raise AssertionError(spec.kind)
    key = tuple(
        outcome if isinstance(outcome, (str, tuple)) else tuple(outcome)
        for outcome in outcomes
    )
    return outcomes, probability, key


# --------------------------------------------------------------------------
# Tree structures (values stored player-0-relative; converted per node actor)
# --------------------------------------------------------------------------


def state_actor(state: GameState) -> int:
    return (
        state.pending_choice.player
        if state.pending_choice is not None
        else state.active_player
    )


def _terminal_value_p0(state: GameState) -> float:
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner == 0 else -1.0


@dataclass(slots=True)
class _Child:
    probability: float | None  # None for sample-only chance
    node: "ClosedNode"
    samples: int = 0  # descent count (Monte Carlo weight for sample-only)


@dataclass(slots=True)
class _Edge:
    action_index: int
    prior: float
    specs: tuple
    children: dict = field(default_factory=dict)  # key -> _Child
    visits: int = 0
    value_sum_p0: float = 0.0  # visit-weighted running mean (selection Q)

    @property
    def q_p0(self) -> float:
        return self.value_sum_p0 / self.visits if self.visits else 0.0


@dataclass(slots=True)
class ClosedNode:
    state: GameState  # barred clone
    actor: int
    terminal: bool
    edges: list = field(default_factory=list)  # [_Edge] aligned to legal
    legal: tuple = ()
    visits: int = 0
    value_sum_p0: float = 0.0

    @property
    def value_p0(self) -> float:
        return self.value_sum_p0 / self.visits if self.visits else 0.0


@dataclass(slots=True)
class OpenNode:
    actor: int | None = None
    priors: dict | None = None  # action_index -> prior, cached at 1st expansion
    children: dict = field(default_factory=dict)  # action_index -> OpenNode
    visits: int = 0
    value_sum_p0: float = 0.0
    edge_visits: dict = field(default_factory=dict)  # action_index -> int
    edge_value_p0: dict = field(default_factory=dict)

    @property
    def value_p0(self) -> float:
        return self.value_sum_p0 / self.visits if self.visits else 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_index: int
    root_value: float  # root-actor perspective
    visits: dict  # action_index -> visit count
    policy_target: dict  # action_index -> improved-policy probability
    sims: int
    mode: str


@dataclass(slots=True)
class SearchConfig:
    sims: int = 64
    top_k: int = 16
    mode: str = "closed"  # or "open"
    c_puct: float = 1.5
    c_visit: float = 50.0
    c_scale: float = 1.0
    seed: int = 0


class GumbelMCTS:
    """One search instance per move. `search(state)` returns the chosen action
    plus the improved policy target (buffer §6 fields)."""

    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None):
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = random.Random(self.config.seed)

    # ---- shared -----------------------------------------------------------

    def _evaluate(self, state: GameState) -> tuple[float, dict]:
        """(value_p0, priors dict over legal indices) from the net."""

        if state.phase is Phase.COMPLETE:
            return _terminal_value_p0(state), {}
        actor = state_actor(state)
        legal = legal_action_indices(state)
        evaluation = self.evaluator.evaluate(
            [encode(state.observation(actor))], [legal]
        )[0]
        value_actor = float(evaluation.wdl[0] - evaluation.wdl[2])
        value_p0 = value_actor if actor == 0 else -value_actor
        priors = {index: float(p) for index, p in zip(legal, evaluation.policy)}
        return value_p0, priors

    def _sigma(self, q: float, max_visits: int) -> float:
        return (self.config.c_visit + max_visits) * self.config.c_scale * q

    def search(self, state: GameState) -> SearchResult:
        if self.config.mode == "closed":
            return self._search_closed(state)
        if self.config.mode == "open":
            return self._search_open(state)
        raise ValueError(f"unknown mode: {self.config.mode}")

    def _gumbel_root(self, legal, priors, simulate, root_value, root_actor):
        """Gumbel top-k + sequential halving. `simulate(action_index)` runs one
        descent through that root action and returns its running Q (root-actor
        perspective) and visit count."""

        config = self.config
        log_prior = {a: math.log(max(priors.get(a, 1e-12), 1e-12)) for a in legal}
        gumbel = {a: self.rng.gammavariate(1.0, 1.0) for a in legal}
        gumbel = {a: -math.log(max(g, 1e-12)) for a, g in gumbel.items()}
        candidates = sorted(
            legal, key=lambda a: gumbel[a] + log_prior[a], reverse=True
        )[: min(config.top_k, len(legal))]

        sims_used = 0
        rounds = max(1, math.ceil(math.log2(max(len(candidates), 2))))
        q_hat: dict = {}
        visits: dict = {a: 0 for a in legal}
        while True:
            per_action = max(
                1, config.sims // max(1, rounds * len(candidates))
            )
            for action in candidates:
                for _ in range(per_action):
                    if sims_used >= config.sims and len(candidates) == 1:
                        break
                    q, n = simulate(action)
                    q_hat[action] = q
                    visits[action] = n
                    sims_used += 1
            if len(candidates) == 1 or sims_used >= config.sims:
                break
            max_visits = max(visits.values()) if visits else 0
            candidates = sorted(
                candidates,
                key=lambda a: gumbel[a]
                + log_prior[a]
                + self._sigma(q_hat.get(a, root_value), max_visits),
                reverse=True,
            )[: max(1, len(candidates) // 2)]

        max_visits = max(visits.values()) if visits else 0
        best = max(
            candidates,
            key=lambda a: gumbel[a]
            + log_prior[a]
            + self._sigma(q_hat.get(a, root_value), max_visits),
        )
        # Improved policy over ALL legal actions: completed Q (root value for
        # unvisited actions) — the Gumbel policy target.
        completed = {
            a: q_hat.get(a, root_value) for a in legal
        }
        logits = {
            a: log_prior[a] + self._sigma(completed[a], max_visits) for a in legal
        }
        peak = max(logits.values())
        weights = {a: math.exp(v - peak) for a, v in logits.items()}
        total = sum(weights.values())
        policy_target = {a: w / total for a, w in weights.items()}
        return best, visits, policy_target, sims_used

    # ---- closed mode ------------------------------------------------------

    def _make_closed_node(self, state: GameState) -> ClosedNode:
        terminal = state.phase is Phase.COMPLETE
        node = ClosedNode(
            state=state,
            actor=state_actor(state) if not terminal else 0,
            terminal=terminal,
        )
        if not terminal:
            node.legal = legal_action_indices(state)
        return node

    def _closed_child(self, node: ClosedNode, edge: _Edge) -> ClosedNode:
        """Descend one edge: sample the chance chain, materialize/reuse the
        child. Never touches the locked deal (barred clones + explicit
        outcomes)."""

        if edge.specs:
            outcomes, probability, key = sample_outcomes(
                node.state, edge.specs, self.rng
            )
        else:
            outcomes, probability, key = None, 1.0, ()
        child = edge.children.get(key)
        if child is None:
            clone = node.state.clone()
            clone.search_barrier = True
            apply_action(
                clone,
                decode_action(clone, edge.action_index),
                chance_outcomes=outcomes,
            )
            child = _Child(probability=probability, node=self._make_closed_node(clone))
            edge.children[key] = child
        child.samples += 1
        return child.node

    def _expand_closed(self, node: ClosedNode) -> float:
        value_p0, priors = self._evaluate(node.state)
        if not node.terminal:
            node.edges = [
                _Edge(
                    action_index=index,
                    prior=priors.get(index, 0.0),
                    specs=chance_signature(
                        node.state, decode_action(node.state, index)
                    ),
                )
                for index in node.legal
            ]
        return value_p0

    def _select_closed(self, node: ClosedNode) -> _Edge:
        sign = 1.0 if node.actor == 0 else -1.0
        total = math.sqrt(max(1, node.visits))
        best, best_score = None, -math.inf
        for edge in node.edges:
            q = sign * edge.q_p0
            score = q + self.config.c_puct * edge.prior * total / (1 + edge.visits)
            if score > best_score:
                best, best_score = edge, score
        return best

    def _descend_closed(self, node: ClosedNode, forced_edge: _Edge | None) -> float:
        """One simulation from `node`; returns the leaf value (p0 terms)."""

        if node.terminal:
            value = _terminal_value_p0(node.state)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        if not node.edges:  # unexpanded leaf
            value = self._expand_closed(node)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        edge = forced_edge if forced_edge is not None else self._select_closed(node)
        child = self._closed_child(node, edge)
        value = self._descend_closed(child, None)
        edge.visits += 1
        edge.value_sum_p0 += value
        node.visits += 1
        node.value_sum_p0 += value
        return value

    def _search_closed(self, state: GameState) -> SearchResult:
        root_state = state.clone()
        root_state.search_barrier = True
        root = self._make_closed_node(root_state)
        root_value_p0 = self._expand_closed(root)
        root.visits += 1
        root.value_sum_p0 += root_value_p0
        sign = 1.0 if root.actor == 0 else -1.0
        edges_by_action = {edge.action_index: edge for edge in root.edges}
        priors = {edge.action_index: edge.prior for edge in root.edges}

        def simulate(action_index: int):
            edge = edges_by_action[action_index]
            self._descend_closed(root, forced_edge=edge)
            return sign * edge.q_p0, edge.visits

        best, visits, policy_target, sims = self._gumbel_root(
            root.legal, priors, simulate, sign * root_value_p0, root.actor
        )
        self._closed_root = root  # exposed for gates/inspection
        return SearchResult(
            action_index=best,
            root_value=sign * root.value_p0,
            visits=visits,
            policy_target=policy_target,
            sims=sims,
            mode="closed",
        )

    # ---- open mode --------------------------------------------------------

    def _descend_open(
        self, node: OpenNode, world: GameState, forced_action: int | None
    ) -> float:
        if world.phase is Phase.COMPLETE:
            value = _terminal_value_p0(world)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        actor = state_actor(world)
        if node.actor is None:
            node.actor = actor
        legal = legal_action_indices(world)  # per-world masking
        if node.priors is None:
            value, priors = self._evaluate(world)
            node.priors = priors  # cached at first expansion (open-loop flaw)
            node.visits += 1
            node.value_sum_p0 += value
            return value
        if forced_action is not None:
            action = forced_action
        else:
            sign = 1.0 if actor == 0 else -1.0
            total = math.sqrt(max(1, node.visits))
            prior_sum = sum(node.priors.get(a, 0.0) for a in legal) or 1.0

            def score(a):
                q = sign * (
                    node.edge_value_p0.get(a, 0.0) / node.edge_visits[a]
                    if node.edge_visits.get(a)
                    else 0.0
                )
                prior = node.priors.get(a, 0.0) / prior_sum
                return q + self.config.c_puct * prior * total / (
                    1 + node.edge_visits.get(a, 0)
                )

            action = max(legal, key=score)
        child = node.children.get(action)
        if child is None:
            child = node.children[action] = OpenNode()
        apply_action(world, decode_action(world, action))
        value = self._descend_open(child, world, None)
        node.edge_visits[action] = node.edge_visits.get(action, 0) + 1
        node.edge_value_p0[action] = node.edge_value_p0.get(action, 0.0) + value
        node.visits += 1
        node.value_sum_p0 += value
        return value

    def _search_open(self, state: GameState) -> SearchResult:
        from .pool import resample_hidden

        root = OpenNode()
        root_value_p0, priors = self._evaluate(state)
        root.priors = priors
        root.actor = state_actor(state)
        root.visits += 1
        root.value_sum_p0 += root_value_p0
        legal = legal_action_indices(state)
        sign = 1.0 if root.actor == 0 else -1.0

        def simulate(action_index: int):
            world = state.clone()
            world.search_barrier = False
            resample_hidden(world, self.rng)
            self._descend_open(root, world, forced_action=action_index)
            n = root.edge_visits.get(action_index, 0)
            q = sign * (root.edge_value_p0.get(action_index, 0.0) / n) if n else 0.0
            return q, n

        best, visits, policy_target, sims = self._gumbel_root(
            legal, priors, simulate, sign * root_value_p0, root.actor
        )
        self._open_root = root
        return SearchResult(
            action_index=best,
            root_value=sign * root.value_p0,
            visits=visits,
            policy_target=policy_target,
            sims=sims,
            mode="open",
        )


def enumerate_chains(state: GameState, specs) -> list[tuple[list, float, tuple]]:
    """All (outcomes, joint probability, key) chains for enumerable specs —
    sequential CARD_REVEALs condition later pools on earlier outcomes. Used by
    exhaustive expansion (gates) and root force-expansion. AGE_DEAL is
    sample-only and unsupported here by design."""

    pool = unseen_pool(state.observation(state.active_player))

    def expand(index: int, used: frozenset):
        if index == len(specs):
            return [([], 1.0)]
        spec = specs[index]
        results = []
        if spec.kind is ChanceKind.CARD_REVEAL:
            back = spec.context[1]
            names = [
                name for name, _ in enumerate_card_reveal(pool, back) if name not in used
            ]
            for name in names:
                for tail, p in expand(index + 1, used | {name}):
                    results.append(([name, *tail], p / len(names)))
        elif spec.kind is ChanceKind.GREAT_LIBRARY_DRAW:
            for subset, p in enumerate_great_library(pool):
                for tail, tail_p in expand(index + 1, used):
                    results.append(([subset, *tail], p * tail_p))
        elif spec.kind is ChanceKind.WONDER_GROUP_REVEAL:
            for subset, p in enumerate_wonder_flip(pool):
                for tail, tail_p in expand(index + 1, used):
                    results.append(([subset, *tail], p * tail_p))
        else:
            raise ValueError(f"cannot enumerate {spec.kind}")
        return results

    chains = expand(0, frozenset())
    return [
        (outcomes, probability, tuple(outcomes)) for outcomes, probability in chains
    ]


def expand_exhaustive(mcts: GumbelMCTS, node: ClosedNode) -> None:
    """Fully expand a closed subtree to terminal, materializing every chance
    outcome with exact probabilities. Gate/verifier utility for small
    positions — raises on AGE_DEAL (sample-only)."""

    if node.terminal:
        return
    if not node.edges:
        mcts._expand_closed(node)
    for edge in node.edges:
        for outcomes, probability, key in enumerate_chains(node.state, edge.specs):
            if key not in edge.children:
                clone = node.state.clone()
                clone.search_barrier = True
                apply_action(
                    clone,
                    decode_action(clone, edge.action_index),
                    chance_outcomes=outcomes or None,
                )
                edge.children[key] = _Child(
                    probability=probability, node=mcts._make_closed_node(clone)
                )
        for child in edge.children.values():
            expand_exhaustive(mcts, child.node)


# --------------------------------------------------------------------------
# Exact recomputation over a (fully expanded) closed tree — the §5 gate hook.
# --------------------------------------------------------------------------


def closed_root_exact_value(node: ClosedNode) -> float:
    """Recursive exact value (p0 terms): max over edges at decision nodes,
    probability-weighted expectation at chance edges. Requires every reachable
    child to be expanded (true on small gate positions searched to
    exhaustion); unexpanded regions raise."""

    if node.terminal:
        return _terminal_value_p0(node.state)
    if not node.edges:
        raise ValueError("exact value requires a fully expanded tree")
    sign = 1.0 if node.actor == 0 else -1.0
    best = -math.inf
    for edge in node.edges:
        if not edge.children:
            raise ValueError("exact value requires every edge expanded")
        if any(child.probability is None for child in edge.children.values()):
            # Sample-only chance (AGE_DEAL): Monte Carlo mean over samples.
            weight = sum(child.samples for child in edge.children.values())
            value = (
                sum(
                    child.samples * closed_root_exact_value(child.node)
                    for child in edge.children.values()
                )
                / weight
            )
        else:
            mass = sum(child.probability for child in edge.children.values())
            value = (
                sum(
                    child.probability * closed_root_exact_value(child.node)
                    for child in edge.children.values()
                )
                / mass
            )
        best = max(best, sign * value)
    return sign * best
