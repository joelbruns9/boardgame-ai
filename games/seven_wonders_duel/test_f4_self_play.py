"""F4.3 gates for the complete Rust self-play hot path."""

from __future__ import annotations

import math

import pytest

from .buffer import replay, to_json_line, from_json_line
from .codec import decode_action
from .engine import apply_action
from .game import Phase, new_game
from .phase_d import blend_draft_priors, temperature_for_move
from .portable_rng import PortableRng
from .rust_bridge import phase_d_record_from_rust, rust_game_for_self_play
from .search import GumbelMCTS, SearchConfig
from .test_rust_engine_equiv import _mock_evaluate, logic_fingerprint


_GAME_RNG_XOR = 0xC6BC279692B5CC83


def _portable_sample(policy: dict[int, float], temperature: float, rng: PortableRng) -> int:
    legal = sorted(policy)
    power = 1.0 / temperature
    weights = [max(policy[action], 1e-12) ** power for action in legal]
    target = rng.next_float() * sum(weights)
    cumulative = 0.0
    for action, weight in zip(legal, weights):
        cumulative += weight
        if target < cumulative:
            return action
    return legal[-1]


def _python_full_game_oracle(
    seed: int,
    first_player: int,
    *,
    cheap: tuple[int, int],
    full: tuple[int, int],
    full_fraction: float,
    top_k: int,
    draft_prior: float,
):
    game = new_game(seed, first_player=first_player)
    rng = PortableRng(seed ^ _GAME_RNG_XOR)
    rows = []
    while game.phase is not Phase.COMPLETE:
        move_index = len(rows)
        full_search = rng.next_float() < full_fraction
        low, high = full if full_search else cheap
        sims = low + rng.randrange(high - low + 1)
        search_seed = rng.getrandbits(63)
        mcts = GumbelMCTS(
            None,
            SearchConfig(
                sims=sims,
                top_k=top_k,
                mode="closed",
                seed=search_seed,
            ),
        )

        def evaluate(state):
            value, priors = _mock_evaluate(state)
            return value, blend_draft_priors(state, priors, draft_prior)

        mcts._evaluate = evaluate  # type: ignore[method-assign]
        result = mcts.search(game)
        action = _portable_sample(
            result.policy_target, temperature_for_move(move_index), rng
        )
        rows.append((action, sims, search_seed, full_search, result))
        apply_action(game, decode_action(game, action))
    return game, rows


def test_f4_3_leaf_batch_one_full_game_oracle_and_replay():
    seed = 2026072103
    first_player = 1
    cheap = (1, 2)
    full = (2, 3)
    full_fraction = 0.35
    top_k = 4
    draft_prior = 0.65
    expected_game, expected_rows = _python_full_game_oracle(
        seed,
        first_player,
        cheap=cheap,
        full=full,
        full_fraction=full_fraction,
        top_k=top_k,
        draft_prior=draft_prior,
    )

    rust = rust_game_for_self_play(seed, first_player)
    raw = rust.self_play_mock(
        game_seed=seed,
        leaf_batch=1,
        cheap_sims_min=cheap[0],
        cheap_sims_max=cheap[1],
        full_sims_min=full[0],
        full_sims_max=full[1],
        full_search_fraction=full_fraction,
        top_k=top_k,
        draft_prior=draft_prior,
        iteration=7,
    )

    assert len(raw["moves"]) == len(expected_rows)
    for actual, (action, sims, search_seed, full_search, result) in zip(
        raw["moves"], expected_rows
    ):
        legal = sorted(result.visits)
        assert actual["action"] == action
        assert actual["sims"] == sims
        assert actual["search_seed"] == search_seed
        assert actual["full_search"] is full_search
        assert actual["legal"] == legal
        assert actual["visits"] == [result.visits[a] for a in legal]
        assert actual["gumbel_topk"] == list(result.gumbel_topk)
        assert actual["root_value"] == result.root_value
        assert actual["policy_target"] == pytest.approx(
            [result.policy_target[a] for a in legal], abs=1e-14, rel=1e-14
        )

    assert raw["final_fingerprint"] == logic_fingerprint(expected_game)
    record = phase_d_record_from_rust(raw)
    assert record.iteration == 7
    assert record.agents == {"p0": "network", "p1": "network", "kind": "self_play"}
    assert all(move.policy_target is not None for move in record.moves)
    assert all(math.isclose(sum(move.policy_target.values()), 1.0) for move in record.moves)
    assert logic_fingerprint(replay(record)) == logic_fingerprint(expected_game)

    # It is a normal schema-1 record, not a sidecar format.
    round_tripped = from_json_line(to_json_line(record))
    assert replay(round_tripped).final_scores == expected_game.final_scores


def test_f4_3_net_boundary_completes_without_python_move_control():
    seed = 2026072104
    calls = 0

    def uniform_adapter(tokens, actor, legal):
        nonlocal calls
        calls += 1
        return 0.0, [1.0 / len(legal)] * len(legal)

    rust = rust_game_for_self_play(seed, 0)
    raw = rust.self_play_net(
        uniform_adapter,
        game_seed=seed,
        leaf_batch=2,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=0.0,
        top_k=2,
        draft_prior=0.0,
    )
    record = phase_d_record_from_rust(raw)
    assert calls > 0
    assert record.moves
    assert replay(record).phase is Phase.COMPLETE


def test_f4_3_record_is_deterministic_and_captures_great_library_draw():
    seed = 2026072200
    kwargs = dict(
        game_seed=seed,
        leaf_batch=1,
        cheap_sims_min=1,
        cheap_sims_max=1,
        full_sims_min=1,
        full_sims_max=1,
        full_search_fraction=0.0,
        top_k=2,
        draft_prior=0.5,
    )
    raw = rust_game_for_self_play(seed, 0).self_play_mock(**kwargs)
    repeated = rust_game_for_self_play(seed, 0).self_play_mock(**kwargs)
    assert raw == repeated
    assert any(event["kind_id"] == 1 for event in raw["chance_log"])
    record = phase_d_record_from_rust(raw)
    assert any(kind == "great_library_draw" for kind, _ in record.chance_log)
    assert replay(record).phase is Phase.COMPLETE


def test_f4_3_rejects_bad_config_and_evaluator_failure():
    rust = rust_game_for_self_play(99, 0)
    with pytest.raises(ValueError, match="simulation range"):
        rust.self_play_mock(
            game_seed=99,
            leaf_batch=1,
            cheap_sims_min=2,
            cheap_sims_max=1,
            full_sims_min=1,
            full_sims_max=1,
            full_search_fraction=0.0,
            top_k=1,
            draft_prior=0.0,
        )

    def broken_adapter(tokens, actor, legal):
        raise RuntimeError("f4.3 evaluator failed")

    with pytest.raises(RuntimeError, match="f4.3 evaluator failed"):
        rust.self_play_net(
            broken_adapter,
            game_seed=99,
            leaf_batch=2,
            cheap_sims_min=1,
            cheap_sims_max=1,
            full_sims_min=1,
            full_sims_max=1,
            full_search_fraction=0.0,
            top_k=1,
            draft_prior=0.0,
        )
