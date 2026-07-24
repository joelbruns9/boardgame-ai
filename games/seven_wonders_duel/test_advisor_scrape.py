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
from .engine import apply_action
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


def test_draft_phase_is_rejected():
    obs = new_game(9).observation(0)
    with pytest.raises(ValueError, match="PLAY_AGE"):
        determinize_observation(obs, random.Random(0))


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
