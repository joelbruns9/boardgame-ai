from __future__ import annotations

from itertools import permutations
import random

import numpy as np
import pytest
import torch

from games.kingdomino_classic import ClassicGameConfig, ClassicGameState
from games.kingdomino_classic.action_codec import (
    legal_action_mask,
    transform_policy_d4,
)
from games.kingdomino_classic.encoder import (
    NUM_D4_TRANSFORMS,
    encode_state,
    padded_player_permutation,
    permute_encoded_players,
    transform_state_d4,
)
from games.kingdomino_classic.network import (
    KingdominoNetwork,
    NetworkConfig,
)


def _state(players: int, seed: int = 17) -> ClassicGameState:
    state = ClassicGameState.new(
        config=ClassicGameConfig(players=players),
        seed=seed,
        start_player=seed % players,
    )
    rng = random.Random(seed * 97 + players)
    for _ in range(19):
        state = state.step(rng.choice(state.legal_actions()))
    return state


@pytest.fixture(scope="module")
def network() -> KingdominoNetwork:
    torch.manual_seed(2026)
    model = KingdominoNetwork(
        NetworkConfig(
            board_channels=8,
            player_dim=24,
            domino_dim=12,
            global_dim=16,
            attention_heads=4,
            policy_hidden=32,
        )
    )
    return model.eval()


@pytest.mark.parametrize("players", [3, 4])
def test_output_shapes_masking_and_win_simplex(
    network: KingdominoNetwork, players: int
) -> None:
    state = _state(players)
    mask = legal_action_mask(state)
    output = network.predict_encoded(encode_state(state), legal_mask=mask)

    assert output.policy_logits.shape == (1, mask.size)
    assert output.score.shape == (1, 4)
    assert output.rank_logits.shape == (1, 4, 4)
    assert output.win_logits.shape == (1, 4)
    assert output.win_probs.shape == (1, 4)
    assert torch.isneginf(output.policy_logits[0, ~torch.from_numpy(mask)]).all()
    assert torch.isfinite(output.policy_logits[0, torch.from_numpy(mask)]).all()
    assert output.win_probs.sum().item() == pytest.approx(1.0, abs=1e-6)
    if players == 3:
        assert output.score[0, 3].item() == 0.0
        assert not output.rank_logits[0, 3].any()
        assert torch.isneginf(output.rank_logits[0, :3, 3]).all()
        assert torch.isfinite(output.rank_logits[0, :3, :3]).all()
        assert output.win_probs[0, 3].item() == 0.0
        assert torch.isneginf(output.win_logits[0, 3])


@pytest.mark.parametrize("players", [3, 4])
def test_all_player_permutations_preserve_policy_and_permute_values(
    network: KingdominoNetwork, players: int
) -> None:
    state = _state(players, seed=29)
    encoded = encode_state(state)
    base = network.predict_encoded(encoded)

    for real_permutation in permutations(range(players)):
        permutation = padded_player_permutation(real_permutation, players)
        permuted = network.predict_encoded(
            permute_encoded_players(encoded, permutation)
        )
        torch.testing.assert_close(permuted.policy_logits, base.policy_logits)
        torch.testing.assert_close(
            permuted.score, base.score[:, list(permutation)]
        )
        torch.testing.assert_close(
            permuted.rank_logits, base.rank_logits[:, list(permutation)]
        )
        torch.testing.assert_close(
            permuted.win_probs, base.win_probs[:, list(permutation)]
        )


@pytest.mark.parametrize("players", [3, 4])
def test_network_is_d4_equivariant_for_arbitrary_weights(
    network: KingdominoNetwork, players: int
) -> None:
    state = _state(players, seed=41)
    base = network.predict_encoded(encode_state(state))
    base_policy = base.policy_logits[0].detach().numpy()

    for transform_id in range(NUM_D4_TRANSFORMS):
        transformed = network.predict_encoded(
            encode_state(transform_state_d4(state, transform_id))
        )
        expected_policy = transform_policy_d4(base_policy, transform_id)
        np.testing.assert_allclose(
            transformed.policy_logits[0].detach().numpy(),
            expected_policy,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(transformed.score, base.score)
        torch.testing.assert_close(transformed.rank_logits, base.rank_logits)
        torch.testing.assert_close(transformed.win_probs, base.win_probs)
