"""BGA-scrape codec: reconstruct a searchable state from a public observation.

The correctness bar is public-equivalence (the reconstruction's observation
equals the original) plus search-runs (closed Gumbel search descends without a
HiddenInformationError), verified on real self-play positions across all ages.
"""

from __future__ import annotations

import json
import random

import pytest

from .advisor_scrape import (
    determinize_observation,
    observation_from_wire,
    observation_to_wire,
)
from .codec import decode_action, legal_action_indices
from .engine import apply_action, legal_actions
from .game import Phase, new_game


def _play_age_samples(per_age: int = 8):
    """Collect real PLAY_AGE observations across ages from random playouts."""

    samples: dict[int, list] = {1: [], 2: [], 3: []}
    for seed in range(60):
        game = new_game(seed, first_player=seed % 2)
        rng = random.Random(1000 + seed)
        for _ in range(400):
            if game.phase is Phase.COMPLETE:
                break
            if game.phase is Phase.PLAY_AGE and len(samples[game.age]) < per_age:
                samples[game.age].append(game.observation(0))
            legal = legal_action_indices(game)
            apply_action(game, decode_action(game, rng.choice(legal)))
        if all(len(v) >= per_age for v in samples.values()):
            break
    return samples


@pytest.fixture(scope="module")
def samples():
    got = _play_age_samples()
    assert all(len(v) > 0 for v in got.values()), got
    return got


@pytest.fixture(scope="module")
def evaluator():
    from .inference import Evaluator
    from .train import build_model

    return Evaluator(build_model("transformer", 32, 1), "cpu")


@pytest.mark.parametrize("age", [1, 2, 3])
def test_reconstruction_is_public_exact(samples, age):
    for obs in samples[age]:
        rebuilt = determinize_observation(obs, random.Random(7)).observation(0)
        assert rebuilt == obs


@pytest.mark.parametrize("age", [1, 2, 3])
def test_reconstructed_state_supports_search(samples, evaluator, age):
    from .search import GumbelMCTS, SearchConfig

    for obs in samples[age]:
        state = determinize_observation(obs, random.Random(7))
        mcts = GumbelMCTS(
            evaluator, SearchConfig(mode="closed", force_expand_root_chance=True, seed=1)
        )
        root = mcts.make_root(state)
        for _ in range(20):
            mcts.descend(root)  # raises HiddenInformationError if a leak occurred


def test_json_wire_round_trips(samples):
    for age in (1, 2, 3):
        obs = samples[age][0]
        wire = observation_to_wire(obs)
        restored = observation_from_wire(json.loads(json.dumps(wire)))
        assert restored == obs


def test_determinization_is_seed_varied_but_public_stable(samples):
    obs = samples[2][0]
    a = determinize_observation(obs, random.Random(1))
    b = determinize_observation(obs, random.Random(2))
    # same public projection, (very likely) different hidden fill
    assert a.observation(0) == obs and b.observation(0) == obs
    assert a.setup_fingerprint() != b.setup_fingerprint()


def test_draft_phase_is_supported():
    """The draft used to be rejected as "not reconstructable from a single public
    observation". It is: the hidden part is a uniform 4-of-8 partition, which is
    the same kind of thing the age-deck determinizer already samples. Covered in
    depth by test_advisor_draft.py."""
    obs = new_game(9).observation(0)
    state = determinize_observation(obs, random.Random(0))
    assert state.phase is Phase.WONDER_DRAFT
    assert state.observation(0) == obs


def test_between_age_start_player_position_reconstructs():
    """The position this codec used to refuse (advisor item F).

    It was refused because the engine dealt the next Age as a *consequence* of
    the choice, so there was no pyramid in the observation to reconstruct and a
    faithful model would have answered a different question from the one on
    screen. The engine now deals first, so the ordinary PLAY_AGE path covers it
    with no branch of its own.
    """

    game = new_game(9)
    rng = random.Random(0)
    while game.phase is not Phase.CHOOSE_NEXT_START_PLAYER:
        actions = legal_actions(game)
        if not actions:
            pytest.skip("no start-player choice reached")
        apply_action(game, rng.choice(actions))

    obs = game.observation(game.active_player)
    state = determinize_observation(obs, random.Random(0))
    assert state.phase is Phase.CHOOSE_NEXT_START_PLAYER
    assert state.observation(game.active_player) == obs  # public-exact
    # The new Age is on the table and is what the chooser is choosing about.
    assert state.age == game.age
    assert sum(1 for card in state.tableau.cards.values() if card.present) == 20
    assert {
        decode_action(state, index).starting_player
        for index in legal_action_indices(state)
    } == {0, 1}


def test_a_search_runs_on_a_reconstructed_start_player_position():
    """Wire-level round-tripping is not enough, and has missed a crash before.

    The mid-move CARD_REVEAL bug passed every wire test and died one ply into a
    real search, so this position gets searched rather than merely rebuilt.
    """

    from .inference import Evaluator
    from .search import GumbelMCTS, SearchConfig
    from .train import build_model

    game = new_game(9)
    rng = random.Random(0)
    while game.phase is not Phase.CHOOSE_NEXT_START_PLAYER:
        actions = legal_actions(game)
        if not actions:
            pytest.skip("no start-player choice reached")
        apply_action(game, rng.choice(actions))

    state = determinize_observation(game.observation(game.active_player), random.Random(0))
    state.search_barrier = True  # search must never read a hidden identity
    mcts = GumbelMCTS(
        Evaluator(build_model("transformer", 32, 1), "cpu"),
        SearchConfig(sims=120, top_k=2, mode="closed", seed=0),
    )
    result = mcts.search(state)
    assert result.action_index in set(legal_action_indices(state))
    assert sum(result.visits.values()) > 0


def test_adapter_scrape_path_recommends(samples):
    from games.advisor import JobManager, RecommendRequest

    from .advisor_adapter import SevenWondersAdvisor
    from .inference import Evaluator
    from .train import build_model

    adapter = SevenWondersAdvisor(evaluator=Evaluator(build_model("transformer", 32, 1), "cpu"))
    wire = observation_to_wire(samples[2][0])
    pos = adapter.state_from_wire({"observation": wire, "resample_seed": 0})
    public = adapter.state_to_public(pos)
    assert public["origin"] == "observation"
    assert adapter.state_key(pos).startswith("obs:")
    resp = JobManager(adapter).run_blocking(
        pos, RecommendRequest(engine="auto", max_sims=100, chunk_sims=50, top_k=4, seed=1)
    )
    assert resp.ok
    assert resp.recommendations
