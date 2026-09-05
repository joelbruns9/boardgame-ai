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

    # Ordering, not wall-clock. The claim is that the scheduler returns while
    # the worker is STILL BLOCKED, and an event proves that directly. The old
    # form asserted the whole call -- game setup included -- finished inside
    # 70 ms, which a loaded CI box or a parallel run can miss while the
    # scheduler is behaving perfectly.
    worker_returned = threading.Event()

    def slow_batch(rows):
        time.sleep(5.0)
        rows = [_row_eval(*row) for row in rows]
        worker_returned.set()
        return rows

    with pytest.raises(TimeoutError, match="timed out"):
        swr.self_play_many_net(
            adapter=slow_batch,
            games=rust_games_for_self_play(seeds, [0, 1]),
            game_seeds=seeds,
            inference_timeout_ms=10.0,
            **_common(leaf_batch=2, global_batch_cap=8),
        )
    assert not worker_returned.is_set(), (
        "the scheduler waited for its blocked inference worker instead of "
        "waking on the timeout"
    )


def test_f4_4_a_timed_out_worker_does_not_block_the_next_call():
    """The pooled inference thread must not be reissued while it is still stuck.

    Workers run on pooled threads now (`eval.rs`), because spawning one per call
    leaked Torch's per-thread state -- ~16 MB a call, which took a laptop down
    mid-match.  The hazard the pool introduces is exactly this case: a worker
    abandoned on timeout is still inside a Python call that may never return, so
    handing its thread to the next caller would stall a healthy search behind a
    dead one.  A thread is only offered back once its job has actually returned.
    """

    import seven_wonders_rust as swr

    seeds = [2026090401, 2026090402]

    stuck_returned = threading.Event()

    def slow_batch(rows):
        time.sleep(5.0)
        rows = [_row_eval(*row) for row in rows]
        stuck_returned.set()
        return rows

    def fast_batch(rows):
        return [_row_eval(*row) for row in rows]

    with pytest.raises(TimeoutError, match="timed out"):
        swr.self_play_many_net(
            adapter=slow_batch,
            games=rust_games_for_self_play(seeds, [0, 1]),
            game_seeds=seeds,
            inference_timeout_ms=20.0,
            **_common(leaf_batch=2, global_batch_cap=8),
        )

    # The abandoned worker is still sleeping. This call has to get a different
    # thread and finish on its own schedule rather than waiting out that sleep.
    records, _ = swr.self_play_many_net(
        adapter=fast_batch,
        games=rust_games_for_self_play(seeds, [0, 1]),
        game_seeds=seeds,
        **_common(leaf_batch=2, global_batch_cap=8),
    )
    assert len(records) == 2
    # Again ordering rather than a stopwatch: this call finished while the
    # abandoned worker was still inside its sleep, so it cannot have been
    # handed that thread or queued behind it.
    assert not stuck_returned.is_set()


def test_f4_4_concurrent_scheduler_calls_all_complete():
    """Pooled threads must not serialise independent searches into a deadlock.

    A single shared inference thread would be enough to stop the leak and would
    hang the advisor host the first time two of its jobs searched at once: a
    worker loop only ends when its `EvalWorker` drops, so a second call queued
    behind the first would never start.  The pool grows to meet concurrency
    instead, and this is the guard that it does.
    """

    import seven_wonders_rust as swr

    done: list[int] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def batch_eval(rows):
        return [_row_eval(*row) for row in rows]

    def play(index: int) -> None:
        seeds = [2026090500 + index * 2, 2026090501 + index * 2]
        try:
            records, _ = swr.self_play_many_net(
                adapter=batch_eval,
                games=rust_games_for_self_play(seeds, [0, 1]),
                game_seeds=seeds,
                **_common(leaf_batch=2, global_batch_cap=8),
            )
            assert len(records) == 2
            with lock:
                done.append(index)
        except BaseException as error:  # reported from the calling thread
            with lock:
                failures.append(error)

    workers = [
        threading.Thread(target=play, args=(index,), daemon=True)
        for index in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)

    assert not [worker for worker in workers if worker.is_alive()], "search deadlocked"
    assert not failures, failures[0]
    assert sorted(done) == list(range(8))


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
