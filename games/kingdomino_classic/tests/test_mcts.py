from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Hashable, Sequence

import numpy as np
import pytest

from games.kingdomino_classic import ClassicGameConfig, ClassicGameState, Phase
from games.kingdomino_classic.action_codec import (
    legal_action_indices,
    transform_action_index_d4,
)
from games.kingdomino_classic.mcts import (
    ClassicHeuristicEvaluator,
    ClassicKingdominoAdapter,
    ClassicNetworkEvaluator,
    SearchConfig,
    SearchEvaluation,
    VectorMCTS,
    search_action,
)
from games.kingdomino_classic.encoder import (
    permute_state_players,
    transform_state_d4,
)
from games.kingdomino_classic.network import KingdominoNetwork, NetworkConfig


@dataclass(frozen=True, slots=True)
class ToyState:
    name: str


class ToyAdapter:
    max_players = 4

    actors = {"root": 0, "rival": 1}
    transitions = {
        ("root", 0): "rival",
        ("root", 1): "safe",
        ("rival", 0): "generous",
        ("rival", 1): "selfish",
    }
    terminals = {
        "safe": np.array((0.6, 0.2, 0.2, 0.0)),
        "generous": np.array((0.8, 0.1, 0.1, 0.0)),
        "selfish": np.array((0.1, 0.8, 0.1, 0.0)),
    }

    def is_terminal(self, state: ToyState) -> bool:
        return state.name in self.terminals

    def actor(self, state: ToyState) -> int:
        return self.actors[state.name]

    def legal_actions(self, state: ToyState) -> tuple[int, ...]:
        return tuple(
            action
            for (name, action), _child in self.transitions.items()
            if name == state.name
        )

    def step(self, state: ToyState, action: int) -> ToyState:
        return ToyState(self.transitions[(state.name, action)])

    def terminal_values(self, state: ToyState) -> np.ndarray:
        return self.terminals[state.name]

    def chance_schedule(
        self, state: ToyState, action: int, *, limit: int, seed: int
    ) -> tuple[Hashable, ...]:
        return ()

    def step_chance(
        self, state: ToyState, action: int, outcome: Hashable
    ) -> ToyState:
        raise AssertionError("ToyAdapter has no chance transitions.")


class UniformToyEvaluator:
    def evaluate(
        self, state: ToyState, legal_actions: Sequence[int]
    ) -> SearchEvaluation:
        return SearchEvaluation(
            priors={action: 1.0 / len(legal_actions) for action in legal_actions},
            value=np.array((0.25, 0.25, 0.25, 0.0)),
        )


def test_each_node_selects_its_own_value_component_without_sign_flip() -> None:
    search = VectorMCTS(
        ToyAdapter(),
        UniformToyEvaluator(),
        SearchConfig(simulations=300, c_puct=1.2),
    )

    rival = search.search(ToyState("rival"))
    root = search.search(ToyState("root"))

    assert rival.action == 1  # player 1 chooses its own 0.8 outcome
    assert root.action == 1  # player 0 anticipates that and takes guaranteed 0.6
    assert root.values[0][0] < root.values[1][0]
    assert root.values[0][1] > root.values[1][1]


@dataclass(frozen=True, slots=True)
class ChanceToyState:
    terminal_outcome: int | None = None


class ChanceToyAdapter:
    max_players = 4

    def is_terminal(self, state: ChanceToyState) -> bool:
        return state.terminal_outcome is not None

    def actor(self, state: ChanceToyState) -> int:
        return 0

    def legal_actions(self, state: ChanceToyState) -> tuple[int, ...]:
        return (0,)

    def step(self, state: ChanceToyState, action: int) -> ChanceToyState:
        raise AssertionError("The only transition is stochastic.")

    def terminal_values(self, state: ChanceToyState) -> np.ndarray:
        assert state.terminal_outcome is not None
        own = state.terminal_outcome / 7.0
        return np.array((own, 1.0 - own, 0.0, 0.0))

    def chance_schedule(
        self, state: ChanceToyState, action: int, *, limit: int, seed: int
    ) -> tuple[int, ...]:
        return tuple(range(min(8, limit)))

    def step_chance(
        self, state: ChanceToyState, action: int, outcome: int
    ) -> ChanceToyState:
        return ChanceToyState(outcome)


def test_persistent_chance_node_admits_widens_and_uses_equal_mean() -> None:
    search = VectorMCTS(
        ChanceToyAdapter(),
        UniformToyEvaluator(),
        SearchConfig(
            simulations=24,
            chance_admission_visits=2,
            chance_initial_width=4,
            chance_min_visits=1,
            chance_width_cap=8,
        ),
    )

    result = search.search(ChanceToyState())

    assert result.chance_nodes == 1
    assert result.chance_admissions == 1
    assert result.chance_widenings == 1
    assert result.maximum_chance_width == 8
    assert result.root_value[0] == np.mean(np.arange(8) / 7.0)


def _before_three_player_reveal() -> ClassicGameState:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=3), seed=77, start_player=0
    )
    state = state.step(state.legal_actions()[0])
    state = state.step(state.legal_actions()[0])
    return state


def test_classic_chance_schedule_uses_public_set_not_hidden_deck_order() -> None:
    state = _before_three_player_reveal()
    shuffled = state.copy()
    random.Random(90210).shuffle(shuffled.deck)
    adapter = ClassicKingdominoAdapter()
    action = legal_action_indices(state)[0]

    first = adapter.chance_schedule(state, action, limit=16, seed=4)
    second = adapter.chance_schedule(shuffled, action, limit=16, seed=4)
    relabeled = permute_state_players(state, (2, 0, 1))
    third = adapter.chance_schedule(relabeled, action, limit=16, seed=4)
    rotated = transform_state_d4(state, 1)
    rotated_action = legal_action_indices(rotated)[0]
    fourth = adapter.chance_schedule(rotated, rotated_action, limit=16, seed=4)

    assert state.deck != shuffled.deck
    assert first == second
    assert first == third
    assert first == fourth
    assert len(first) == len(set(first)) == 16
    assert all(len(outcome) == 4 for outcome in first)
    child = adapter.step_chance(state, action, first[0])
    assert tuple(child.draft_row) == first[0]


def test_placement_reveal_schedule_is_d4_invariant() -> None:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=4), seed=81, start_player=0
    )
    while state.phase == Phase.INITIAL_DRAFT:
        state = state.step(state.legal_actions()[0])
    while len(state.pending_claims) > 1:
        state = state.step(state.legal_actions()[0])
    adapter = ClassicKingdominoAdapter()
    action = legal_action_indices(state)[0]
    transformed_state = transform_state_d4(state, 5)
    transformed_action = transform_action_index_d4(action, 5)

    assert transformed_action in legal_action_indices(transformed_state)
    assert adapter.chance_schedule(
        state, action, limit=16, seed=12
    ) == adapter.chance_schedule(
        transformed_state, transformed_action, limit=16, seed=12
    )


def test_classic_search_returns_a_legal_action_and_exercises_chance() -> None:
    state = _before_three_player_reveal()
    action, result = search_action(
        state,
        ClassicHeuristicEvaluator(),
        config=SearchConfig(
            simulations=12,
            chance_admission_visits=2,
            chance_initial_width=4,
            chance_min_visits=1,
            chance_width_cap=8,
        ),
        seed=11,
    )

    assert action in state.legal_actions()
    assert result.chance_nodes >= 1
    assert result.chance_admissions >= 1


def test_complete_classic_search_commutes_with_player_relabeling() -> None:
    state = _before_three_player_reveal()
    permutation = (2, 0, 1)
    relabeled = permute_state_players(state, permutation)
    config = SearchConfig(
        simulations=12,
        chance_admission_visits=2,
        chance_initial_width=4,
        chance_min_visits=1,
        chance_width_cap=8,
    )
    first = VectorMCTS(
        ClassicKingdominoAdapter(),
        ClassicHeuristicEvaluator(),
        config,
        seed=49,
    ).search(state)
    second = VectorMCTS(
        ClassicKingdominoAdapter(),
        ClassicHeuristicEvaluator(),
        config,
        seed=49,
    ).search(relabeled)

    assert first.action == second.action
    assert first.visits == second.visits
    for action in first.values:
        expected = np.asarray(first.values[action])[list(permutation) + [3]]
        np.testing.assert_allclose(second.values[action], expected)


def test_untrained_network_evaluator_runs_through_vector_search() -> None:
    import torch

    torch.manual_seed(8)
    network = KingdominoNetwork(
        NetworkConfig(
            board_channels=4,
            player_dim=16,
            domino_dim=8,
            global_dim=8,
            attention_heads=4,
            policy_hidden=16,
        )
    )
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=3), seed=5, start_player=0
    )
    action, result = search_action(
        state,
        ClassicNetworkEvaluator(network),
        config=SearchConfig(simulations=1),
        seed=2,
    )

    assert action in state.legal_actions()
    assert sum(result.root_value) == pytest.approx(1.0, abs=1e-6)
