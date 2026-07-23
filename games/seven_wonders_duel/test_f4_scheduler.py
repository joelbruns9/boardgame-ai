"""F4.4 cooperative scheduler and global coalescer gates."""

from __future__ import annotations

import threading
import time

import pytest

from .buffer import replay
from .game import Phase
from .rust_bridge import (
    phase_d_records_from_rust,
    rust_game_for_self_play,
    rust_games_for_self_play,
)


def _common(*, leaf_batch=1, global_batch_cap=16):
    return dict(
        global_batch_cap=global_batch_cap,
        leaf_batch=leaf_batch,
        cheap_sims_min=1,
        cheap_sims_max=2,
        full_sims_min=2,
        full_sims_max=3,
        full_search_fraction=0.3,
        top_k=3,
        draft_prior=0.55,
        iteration=9,
    )


def _row_eval(tokens, actor, legal):
    """Deterministic, production-shaped evaluator independent of batch shape."""

    folded = 0x9E3779B97F4A7C15
    for type_id, entity_id, aux_id, features in tokens:
        folded ^= (type_id + 1) * 0x100000001B3
        folded ^= (entity_id + 3) * 0x9E3779B1
        folded ^= (aux_id + 5) * 0x85EBCA77
        folded ^= len(features) * 0xC2B2AE3D
        folded &= (1 << 64) - 1
    value_actor = ((folded >> 11) / float(1 << 53)) * 2.0 - 1.0
    weights = [1.0 + ((folded ^ (action * 0x9E3779B1)) & 0xFFFF) for action in legal]
    total = float(sum(weights))
    return value_actor, [weight / total for weight in weights]


def test_f4_4_mock_scheduler_matches_independent_slots_and_preserves_order():
    import seven_wonders_rust as swr

    seeds = [2026072304, 2026072301, 2026072303, 2026072302]
    first_players = [1, 0, 1, 0]
    kwargs = _common(leaf_batch=1, global_batch_cap=8)
    independent = [
        rust_game_for_self_play(seed, first).self_play_mock(
            game_seed=seed,
            **{key: value for key, value in kwargs.items() if key != "global_batch_cap"},
        )
        for seed, first in zip(seeds, first_players)
    ]
    records, metrics = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )

    assert records == independent
    assert [record["seed"] for record in records] == seeds
    assert metrics["games"] == len(seeds)
    assert metrics["moves"] == sum(len(record["moves"]) for record in records)
    assert metrics["global_rows"] == metrics["root_rows"] + metrics["leaf_rows"]
    assert metrics["max_batch_rows"] <= kwargs["global_batch_cap"]
    assert any(size > 1 for size in metrics["batch_rows"])
    assert all(replay(record).phase is Phase.COMPLETE for record in phase_d_records_from_rust(records))


def test_f4_4_batched_adapter_alignment_matches_scalar_games():
    import seven_wonders_rust as swr

    seeds = [2026072310, 2026072311, 2026072312]
    first_players = [0, 1, 0]
    kwargs = _common(leaf_batch=1, global_batch_cap=12)
    independent = [
        rust_game_for_self_play(seed, first).self_play_net(
            _row_eval,
            game_seed=seed,
            **{key: value for key, value in kwargs.items() if key != "global_batch_cap"},
        )
        for seed, first in zip(seeds, first_players)
    ]
    batch_shapes = []
    worker_threads = set()
    caller_thread = threading.get_ident()

    def batch_adapter(rows):
        worker_threads.add(threading.get_ident())
        batch_shapes.append([(len(tokens), len(legal)) for tokens, _, legal in rows])
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    records, metrics = swr.self_play_many_net(
        adapter=batch_adapter,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **kwargs,
    )
    assert records == independent
    assert len(batch_shapes) == metrics["global_batches"]
    assert any(len(batch) > 1 for batch in batch_shapes)
    assert len({shape for batch in batch_shapes for shape in batch}) > 3
    assert len(worker_threads) == 1
    assert caller_thread not in worker_threads


def test_f4_4_leaf_waves_coalesce_across_games_and_replay():
    import seven_wonders_rust as swr

    seeds = list(range(2026072320, 2026072326))
    first_players = [index % 2 for index in range(len(seeds))]
    observed_sizes = []

    def uniform_batch(rows):
        observed_sizes.append(len(rows))
        return [
            (0.0, [1.0 / len(legal)] * len(legal))
            for _, _, legal in rows
        ]

    records, metrics = swr.self_play_many_net(
        adapter=uniform_batch,
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **_common(leaf_batch=2, global_batch_cap=8),
    )
    assert [record["seed"] for record in records] == seeds
    assert metrics["max_batch_rows"] <= 8
    assert max(observed_sizes) > 2  # more than one intra-search wave
    assert metrics["max_inflight_batches"] == 2
    assert metrics["requested_nn_leaves"] >= metrics["unique_nn_leaves"]
    assert metrics["global_rows"] == sum(observed_sizes)
    converted = phase_d_records_from_rust(records)
    assert len(converted) == len(seeds)
    assert all(replay(record).phase is Phase.COMPLETE for record in converted)


def test_f4_4_failure_wakes_all_slots_and_preserves_original_error():
    import seven_wonders_rust as swr

    seeds = [2026072340, 2026072341, 2026072342]
    games = rust_games_for_self_play(seeds, [0, 1, 0])
    calls = 0

    def failing(rows):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("f4.4 global evaluator failed")
        return [_row_eval(tokens, actor, legal) for tokens, actor, legal in rows]

    with pytest.raises(RuntimeError, match="f4.4 global evaluator failed"):
        swr.self_play_many_net(
            adapter=failing,
            games=games,
            game_seeds=seeds,
            **_common(leaf_batch=2, global_batch_cap=8),
        )

    def missing_row(rows):
        return [_row_eval(*row) for row in rows[:-1]]

    with pytest.raises(ValueError, match="returned .* rows"):
        swr.self_play_many_net(
            adapter=missing_row,
            games=games,
            game_seeds=seeds,
            **_common(leaf_batch=1, global_batch_cap=8),
        )


def test_f4_4_contract_rejects_oversized_leaf_waves_before_generation():
    import seven_wonders_rust as swr

    seeds = [2026072350, 2026072351]
    games = rust_games_for_self_play(seeds, [0, 1])
    with pytest.raises(ValueError, match=r"leaf_batch=2 exceeds global_batch_cap=1"):
        swr.self_play_many_mock(
            games=games,
            game_seeds=seeds,
            **_common(leaf_batch=2, global_batch_cap=1),
        )

    adapter_called = False

    def must_not_evaluate(_rows):
        nonlocal adapter_called
        adapter_called = True
        raise AssertionError("invalid scheduler configuration reached inference")

    with pytest.raises(ValueError, match=r"leaf_batch_p0=3 exceeds global_batch_cap=2"):
        swr.self_play_many_net(
            adapter=must_not_evaluate,
            games=games,
            game_seeds=seeds,
            leaf_batch_p0=3,
            leaf_batch_p1=1,
            **_common(leaf_batch=2, global_batch_cap=2),
        )
    assert not adapter_called


def test_f4_4_timeout_wakes_scheduler_without_waiting_for_worker_shutdown():
    import seven_wonders_rust as swr

    seeds = [2026072355, 2026072356]

    def slow_batch(rows):
        time.sleep(0.08)
        return [_row_eval(*row) for row in rows]

    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="timed out"):
        swr.self_play_many_net(
            adapter=slow_batch,
            games=rust_games_for_self_play(seeds, [0, 1]),
            game_seeds=seeds,
            inference_timeout_ms=10.0,
            **_common(leaf_batch=2, global_batch_cap=8),
        )
    assert time.perf_counter() - started < 0.07


def test_f4_4_mock_stress_completes_twelve_slots_without_loss():
    import seven_wonders_rust as swr

    seeds = list(range(2026072360, 2026072372))
    records, metrics = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [index % 2 for index in range(12)]),
        game_seeds=seeds,
        **_common(leaf_batch=2, global_batch_cap=16),
    )
    assert len(records) == 12
    assert [record["seed"] for record in records] == seeds
    assert metrics["games"] == 12
    assert metrics["moves"] == sum(len(record["moves"]) for record in records)
    assert all(record["winner"] in (0, 1, None) for record in records)


def test_f4_r2_coarse_scheduler_shards_match_shape_invariant_eval_and_preserve_order():
    import seven_wonders_rust as swr

    seeds = list(range(2026072380, 2026072388))
    games = rust_games_for_self_play(seeds, [index % 2 for index in range(len(seeds))])
    kwargs = _common(leaf_batch=1, global_batch_cap=8)
    single, _ = swr.self_play_many_net(
        adapter=lambda rows: [_row_eval(*row) for row in rows],
        games=games,
        game_seeds=seeds,
        scheduler_workers=1,
        **kwargs,
    )
    sharded, metrics = swr.self_play_many_net(
        adapter=lambda rows: [_row_eval(*row) for row in rows],
        games=games,
        game_seeds=seeds,
        scheduler_workers=2,
        **kwargs,
    )
    # _row_eval is deliberately independent of batch shape. This equality is
    # not a CUDA bit-identity guarantee across scheduler shard counts.
    assert sharded == single
    assert [record["seed"] for record in sharded] == seeds
    assert metrics["scheduler_workers"] == 2
