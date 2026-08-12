"""Vector-valued multiplayer MCTS with public-state chance widening."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Generic, Hashable, Protocol, Sequence, TypeVar

import numpy as np
import torch

from .action_codec import (
    decode_action,
    legal_action_indices,
    legal_action_mask,
    transform_action_index_d4,
)
from .baselines import DenialAwareBot, FlexibilityBot, ImmediateScoreBot
from .encoder import (
    MAX_PLAYERS,
    NUM_D4_TRANSFORMS,
    encode_state,
    transform_coord_d4,
)
from .game import Action, ClassicGameState, Phase
from .network import KingdominoNetwork


StateT = TypeVar("StateT")
OutcomeT = TypeVar("OutcomeT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    priors: dict[int, float]
    value: np.ndarray


class SearchEvaluator(Protocol[StateT]):
    def evaluate(
        self, state: StateT, legal_actions: Sequence[int]
    ) -> SearchEvaluation: ...


class SearchAdapter(Protocol[StateT, OutcomeT]):
    max_players: int

    def is_terminal(self, state: StateT) -> bool: ...

    def actor(self, state: StateT) -> int: ...

    def legal_actions(self, state: StateT) -> tuple[int, ...]: ...

    def step(self, state: StateT, action: int) -> StateT: ...

    def terminal_values(self, state: StateT) -> np.ndarray: ...

    def chance_schedule(
        self, state: StateT, action: int, *, limit: int, seed: int
    ) -> tuple[OutcomeT, ...]: ...

    def step_chance(
        self, state: StateT, action: int, outcome: OutcomeT
    ) -> StateT: ...


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 200
    c_puct: float = 1.5
    chance_admission_visits: int = 2
    chance_initial_width: int = 4
    chance_min_visits: int = 4
    chance_width_cap: int = 16
    max_depth: int = 256

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive.")
        if self.chance_admission_visits < 1:
            raise ValueError("chance_admission_visits must be positive.")
        if self.chance_initial_width < 1:
            raise ValueError("chance_initial_width must be positive.")
        if self.chance_min_visits < 1:
            raise ValueError("chance_min_visits must be positive.")
        if self.chance_width_cap < self.chance_initial_width:
            raise ValueError("chance_width_cap must cover chance_initial_width.")


@dataclass(slots=True)
class DecisionNode(Generic[StateT, OutcomeT]):
    state: StateT
    visits: int = 0
    expanded: bool = False
    edges: dict[int, "EdgeStats[StateT, OutcomeT]"] = field(default_factory=dict)


@dataclass(slots=True)
class ChanceOutcome(Generic[StateT, OutcomeT]):
    outcome: OutcomeT
    child: DecisionNode[StateT, OutcomeT]
    bootstrap_value: np.ndarray
    visits: int = 0
    value_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_PLAYERS, dtype=np.float64)
    )

    def current_value(self) -> np.ndarray:
        if self.visits:
            return self.value_sum / self.visits
        return self.bootstrap_value


@dataclass(slots=True)
class ChanceNode(Generic[StateT, OutcomeT]):
    state: StateT
    action: int
    schedule: tuple[OutcomeT, ...]
    outcomes: list[ChanceOutcome[StateT, OutcomeT]] = field(default_factory=list)
    crossings: int = 0
    admissions: int = 0
    widenings: int = 0

    def current_mean(self) -> np.ndarray:
        if not self.outcomes:
            raise AssertionError("A chance mean requires an active outcome.")
        return np.stack([outcome.current_value() for outcome in self.outcomes]).mean(
            axis=0
        )


@dataclass(slots=True)
class EdgeStats(Generic[StateT, OutcomeT]):
    action: int
    prior: float
    visits: int = 0
    value_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(MAX_PLAYERS, dtype=np.float64)
    )
    child: DecisionNode[StateT, OutcomeT] | None = None
    chance: ChanceNode[StateT, OutcomeT] | None = None

    def value(self) -> np.ndarray:
        if self.chance is not None and self.chance.outcomes:
            return self.chance.current_mean()
        if self.visits:
            return self.value_sum / self.visits
        return np.zeros(MAX_PLAYERS, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: int
    visits: dict[int, int]
    values: dict[int, tuple[float, ...]]
    root_value: tuple[float, ...]
    chance_nodes: int
    chance_admissions: int
    chance_widenings: int
    maximum_chance_width: int


class VectorMCTS(Generic[StateT, OutcomeT]):
    """PUCT search that always backs up a full player-value vector."""

    def __init__(
        self,
        adapter: SearchAdapter[StateT, OutcomeT],
        evaluator: SearchEvaluator[StateT],
        config: SearchConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.adapter = adapter
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.seed = seed
        self._chance_nodes: list[ChanceNode[StateT, OutcomeT]] = []

    def _terminal_value(self, state: StateT) -> np.ndarray:
        value = np.asarray(self.adapter.terminal_values(state), dtype=np.float64)
        if value.shape != (self.adapter.max_players,):
            raise ValueError("Terminal values must use the padded player axis.")
        return value

    def _expand(self, node: DecisionNode[StateT, OutcomeT]) -> np.ndarray:
        if self.adapter.is_terminal(node.state):
            return self._terminal_value(node.state)
        legal = self.adapter.legal_actions(node.state)
        if not legal:
            raise AssertionError("A nonterminal search node has no legal actions.")
        evaluation = self.evaluator.evaluate(node.state, legal)
        value = np.asarray(evaluation.value, dtype=np.float64)
        if value.shape != (self.adapter.max_players,):
            raise ValueError("Evaluator values must use the padded player axis.")
        priors = np.asarray(
            [max(0.0, evaluation.priors.get(action, 0.0)) for action in legal],
            dtype=np.float64,
        )
        if not np.isfinite(priors).all() or priors.sum() <= 0:
            priors = np.ones(len(legal), dtype=np.float64)
        priors /= priors.sum()
        node.edges = {
            action: EdgeStats(action=action, prior=float(prior))
            for action, prior in zip(legal, priors)
        }
        node.expanded = True
        return value

    def _select_edge(
        self, node: DecisionNode[StateT, OutcomeT]
    ) -> EdgeStats[StateT, OutcomeT]:
        actor = self.adapter.actor(node.state)
        exploration_scale = math.sqrt(max(1, node.visits))

        def key(edge: EdgeStats[StateT, OutcomeT]) -> tuple[float, float, int]:
            q_value = float(edge.value()[actor])
            exploration = (
                self.config.c_puct
                * edge.prior
                * exploration_scale
                / (1 + edge.visits)
            )
            return q_value + exploration, edge.prior, -edge.action

        return max(node.edges.values(), key=key)

    def _activate_outcomes(
        self, chance: ChanceNode[StateT, OutcomeT], target: int
    ) -> None:
        target = min(target, len(chance.schedule))
        while len(chance.outcomes) < target:
            outcome = chance.schedule[len(chance.outcomes)]
            child_state = self.adapter.step_chance(
                chance.state, chance.action, outcome
            )
            child = DecisionNode[StateT, OutcomeT](child_state)
            bootstrap = self._expand(child)
            chance.outcomes.append(
                ChanceOutcome(
                    outcome=outcome,
                    child=child,
                    bootstrap_value=bootstrap,
                    value_sum=np.zeros(
                        self.adapter.max_players, dtype=np.float64
                    ),
                )
            )

    def _simulate_chance(
        self, chance: ChanceNode[StateT, OutcomeT], depth: int
    ) -> np.ndarray:
        if not chance.outcomes:
            self._activate_outcomes(chance, 1)
        selected = min(
            enumerate(chance.outcomes),
            key=lambda item: (item[1].visits, item[0]),
        )[1]
        value = self._simulate(selected.child, depth + 1)
        selected.visits += 1
        selected.value_sum += value
        chance.crossings += 1

        if (
            len(chance.outcomes) == 1
            and chance.crossings >= self.config.chance_admission_visits
            and len(chance.schedule) > 1
        ):
            self._activate_outcomes(chance, self.config.chance_initial_width)
            chance.admissions += 1
        elif (
            len(chance.outcomes) >= self.config.chance_initial_width
            and len(chance.outcomes) < len(chance.schedule)
            and min(outcome.visits for outcome in chance.outcomes)
            >= self.config.chance_min_visits
        ):
            self._activate_outcomes(chance, 2 * len(chance.outcomes))
            chance.widenings += 1
        return chance.current_mean()

    def _simulate(
        self, node: DecisionNode[StateT, OutcomeT], depth: int = 0
    ) -> np.ndarray:
        if depth >= self.config.max_depth:
            if self.adapter.is_terminal(node.state):
                return self._terminal_value(node.state)
            return self.evaluator.evaluate(
                node.state, self.adapter.legal_actions(node.state)
            ).value
        if self.adapter.is_terminal(node.state):
            node.visits += 1
            return self._terminal_value(node.state)
        if not node.expanded:
            node.visits += 1
            return self._expand(node)

        edge = self._select_edge(node)
        if edge.child is None and edge.chance is None:
            schedule = self.adapter.chance_schedule(
                node.state,
                edge.action,
                limit=self.config.chance_width_cap,
                seed=self.seed,
            )
            if schedule:
                edge.chance = ChanceNode(
                    state=node.state,
                    action=edge.action,
                    schedule=schedule,
                )
                self._chance_nodes.append(edge.chance)
            else:
                edge.child = DecisionNode(
                    self.adapter.step(node.state, edge.action)
                )
        if edge.chance is not None:
            value = self._simulate_chance(edge.chance, depth)
        else:
            assert edge.child is not None
            value = self._simulate(edge.child, depth + 1)
        edge.visits += 1
        edge.value_sum += value
        node.visits += 1
        return value

    def search(self, state: StateT) -> SearchResult:
        if self.adapter.is_terminal(state):
            raise ValueError("Cannot search a terminal state.")
        begin_search = getattr(self.evaluator, "begin_search", None)
        if begin_search is not None:
            begin_search(state)
        self._chance_nodes = []
        root = DecisionNode[StateT, OutcomeT](state)
        root_bootstrap = self._expand(root)
        for _simulation in range(self.config.simulations):
            self._simulate(root)
        chosen = max(
            root.edges.values(),
            key=lambda edge: (
                edge.visits,
                edge.value()[self.adapter.actor(state)],
                -edge.action,
            ),
        )
        visits = {action: edge.visits for action, edge in root.edges.items()}
        values = {
            action: tuple(float(component) for component in edge.value())
            for action, edge in root.edges.items()
        }
        visited = [edge for edge in root.edges.values() if edge.visits]
        if visited:
            total_visits = sum(edge.visits for edge in visited)
            root_value = sum(
                edge.visits * edge.value() for edge in visited
            ) / total_visits
        else:
            root_value = root_bootstrap
        return SearchResult(
            action=chosen.action,
            visits=visits,
            values=values,
            root_value=tuple(float(component) for component in root_value),
            chance_nodes=len(self._chance_nodes),
            chance_admissions=sum(node.admissions for node in self._chance_nodes),
            chance_widenings=sum(node.widenings for node in self._chance_nodes),
            maximum_chance_width=max(
                (len(node.outcomes) for node in self._chance_nodes), default=0
            ),
        )


def _softmax_priors(
    actions: Sequence[int], logits: Sequence[float]
) -> dict[int, float]:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(initial=0.0)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return {
        action: float(probability)
        for action, probability in zip(actions, probabilities)
    }


class ClassicNetworkEvaluator(SearchEvaluator[ClassicGameState]):
    def __init__(
        self,
        network: KingdominoNetwork,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.network = network.eval()
        self.device = device

    def evaluate(
        self, state: ClassicGameState, legal_actions: Sequence[int]
    ) -> SearchEvaluation:
        mask = legal_action_mask(state)
        output = self.network.predict_encoded(
            encode_state(state), legal_mask=mask, device=self.device
        )
        logits = output.policy_logits[0].detach().cpu().numpy()
        value = output.win_probs[0].detach().cpu().numpy().astype(np.float64)
        return SearchEvaluation(
            priors=_softmax_priors(
                legal_actions, [float(logits[action]) for action in legal_actions]
            ),
            value=value,
        )


class ClassicHeuristicEvaluator(SearchEvaluator[ClassicGameState]):
    """Lightweight K1-derived leaf evaluator for search smoke and anchors."""

    def __init__(
        self,
        *,
        bot: ImmediateScoreBot | None = None,
        prior_temperature: float = 4.0,
    ) -> None:
        self.bot = bot if bot is not None else FlexibilityBot()
        self.prior_temperature = prior_temperature

    def evaluate(
        self, state: ClassicGameState, legal_actions: Sequence[int]
    ) -> SearchEvaluation:
        engine_actions = [decode_action(state, action) for action in legal_actions]
        cache: dict[object, object] = {}
        logits = [
            self.bot.evaluate_action(state, action, cache) / self.prior_temperature
            for action in engine_actions
        ]
        scores = np.asarray(state.scores(), dtype=np.float64)
        scores -= scores.max(initial=0.0)
        probabilities = np.exp(scores / 12.0)
        probabilities /= probabilities.sum()
        value = np.zeros(MAX_PLAYERS, dtype=np.float64)
        value[: state.config.players] = probabilities
        return SearchEvaluation(
            priors=_softmax_priors(legal_actions, logits), value=value
        )


class ClassicTieredHeuristicEvaluator(SearchEvaluator[ClassicGameState]):
    """Use denial-aware root ordering and cheaper flexibility at tree leaves."""

    def __init__(self, *, prior_temperature: float = 4.0) -> None:
        self.root = ClassicHeuristicEvaluator(
            bot=DenialAwareBot(), prior_temperature=prior_temperature
        )
        self.leaf = ClassicHeuristicEvaluator(
            bot=FlexibilityBot(), prior_temperature=prior_temperature
        )
        self._root_pending = True

    def begin_search(self, state: ClassicGameState) -> None:
        self._root_pending = True

    def evaluate(
        self, state: ClassicGameState, legal_actions: Sequence[int]
    ) -> SearchEvaluation:
        if self._root_pending:
            self._root_pending = False
            return self.root.evaluate(state, legal_actions)
        return self.leaf.evaluate(state, legal_actions)


def _unrank_combination(total: int, choose: int, rank: int) -> tuple[int, ...]:
    if not 0 <= rank < math.comb(total, choose):
        raise ValueError("Combination rank is outside the support.")
    result: list[int] = []
    next_value = 0
    for remaining in range(choose, 0, -1):
        while True:
            count = math.comb(total - next_value - 1, remaining - 1)
            if rank < count:
                result.append(next_value)
                next_value += 1
                break
            rank -= count
            next_value += 1
    return tuple(result)


class ClassicKingdominoAdapter(SearchAdapter[ClassicGameState, tuple[int, ...]]):
    max_players = MAX_PLAYERS

    def is_terminal(self, state: ClassicGameState) -> bool:
        return state.is_terminal

    def actor(self, state: ClassicGameState) -> int:
        return state.current_actor

    def legal_actions(self, state: ClassicGameState) -> tuple[int, ...]:
        return legal_action_indices(state)

    def step(self, state: ClassicGameState, action: int) -> ClassicGameState:
        return state.step(decode_action(state, action))

    def terminal_values(self, state: ClassicGameState) -> np.ndarray:
        value = np.zeros(MAX_PLAYERS, dtype=np.float64)
        value[: state.config.players] = state.returns()
        return value

    @staticmethod
    def _reveals_row(state: ClassicGameState) -> bool:
        if not state.deck:
            return False
        if state.phase == Phase.INITIAL_DRAFT:
            return (
                len(state.next_claims) + 1 == state.config.selections_per_round
            )
        if state.phase == Phase.PLACE_AND_DRAFT:
            return len(state.pending_claims) == 1
        return False

    @staticmethod
    def _canonical_board_key(state_key: tuple[tuple[int, ...], ...]) -> tuple:
        transformed_keys = []
        for transform_id in range(NUM_D4_TRANSFORMS):
            cells = []
            for x, y, terrain, crowns, domino_id in state_key:
                new_x, new_y = transform_coord_d4((x, y), transform_id)
                cells.append((new_x, new_y, terrain, crowns, domino_id))
            transformed_keys.append(tuple(sorted(cells)))
        return min(transformed_keys)

    @staticmethod
    def _public_schedule_seed(
        state: ClassicGameState, action: int, seed: int
    ) -> int:
        pending_by_player: list[list[tuple[int, int]]] = [
            [] for _ in range(state.config.players)
        ]
        for order, claim in enumerate(state.pending_claims):
            pending_by_player[claim.player].append((order, claim.domino_id))
        next_by_player: list[list[int]] = [
            [] for _ in range(state.config.players)
        ]
        for claim in state.next_claims:
            next_by_player[claim.player].append(claim.domino_id)
        actor = state.current_actor
        player_records = sorted(
            (
                ClassicKingdominoAdapter._canonical_board_key(
                    state.boards[player].state_key()
                ),
                tuple(state.forced_discards[player]),
                tuple(pending_by_player[player]),
                tuple(next_by_player[player]),
                player == actor,
                (
                    state.initial_player_order.index(player)
                    if state.phase == Phase.INITIAL_DRAFT
                    else -1
                ),
            )
            for player in range(state.config.players)
        )
        public_key = (
            state.config.configuration_key,
            int(state.phase),
            state.initial_pick_count,
            tuple(sorted(state.deck)),
            tuple(state.draft_row),
            tuple(state.unclaimed_discards),
            tuple(player_records),
            min(
                transform_action_index_d4(action, transform_id)
                for transform_id in range(NUM_D4_TRANSFORMS)
            ),
            seed,
        )
        digest = hashlib.sha256(repr(public_key).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")

    def chance_schedule(
        self,
        state: ClassicGameState,
        action: int,
        *,
        limit: int,
        seed: int,
    ) -> tuple[tuple[int, ...], ...]:
        if not self._reveals_row(state):
            return ()
        remaining = tuple(sorted(state.deck))
        support = math.comb(len(remaining), 4)
        width = min(limit, support)
        rng = random.Random(self._public_schedule_seed(state, action, seed))
        ranks = rng.sample(range(support), width)
        return tuple(
            tuple(
                remaining[index]
                for index in _unrank_combination(len(remaining), 4, rank)
            )
            for rank in ranks
        )

    def step_chance(
        self,
        state: ClassicGameState,
        action: int,
        outcome: tuple[int, ...],
    ) -> ClassicGameState:
        if len(outcome) != 4 or not set(outcome).issubset(state.deck):
            raise ValueError("A reveal outcome must contain four remaining dominoes.")
        prepared = state.copy()
        selected = set(outcome)
        prepared.deck = [
            *outcome,
            *(domino for domino in sorted(state.deck) if domino not in selected),
        ]
        return prepared.step(decode_action(prepared, action))


def search_action(
    state: ClassicGameState,
    evaluator: SearchEvaluator[ClassicGameState],
    *,
    config: SearchConfig | None = None,
    seed: int = 0,
) -> tuple[Action, SearchResult]:
    search = VectorMCTS(
        ClassicKingdominoAdapter(), evaluator, config=config, seed=seed
    )
    result = search.search(state)
    return decode_action(state, result.action), result


@dataclass(slots=True)
class MCTSBot:
    evaluator: SearchEvaluator[ClassicGameState] = field(
        default_factory=ClassicHeuristicEvaluator
    )
    config: SearchConfig = field(
        default_factory=lambda: SearchConfig(simulations=32)
    )
    name: str = "vector_mcts"

    def choose_action(
        self,
        state: ClassicGameState,
        actions: Sequence[Action] | None = None,
        rng: random.Random | None = None,
    ) -> Action:
        legal = list(actions if actions is not None else state.legal_actions())
        if not legal:
            raise ValueError("MCTSBot received no legal actions.")
        if len(legal) == 1:
            return legal[0]
        rng = rng if rng is not None else random.Random()
        action, _result = search_action(
            state,
            self.evaluator,
            config=self.config,
            seed=rng.randrange(2**63),
        )
        if action not in legal:
            raise AssertionError("MCTS selected an action outside the legal set.")
        return action
