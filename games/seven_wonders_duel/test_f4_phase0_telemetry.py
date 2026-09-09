"""Phase 0 instrumentation gates (THROUGHPUT_ACTION_PLAN.md).

Phase 0 buys knowledge, not throughput, so what has to be gated is that the
measurements mean what they claim: occupancy is integrated over time rather than
counted per loop iteration, the steady-state window is separated from the drain
tail, device time is distinguishable from host dispatch, and the cost-model
harness packs bytes the production boundary would accept.
"""

from __future__ import annotations

import pytest

from .f4_cost_model import (
    aggregate_passes,
    build_payload,
    fit_cost_model,
    token_length_buckets,
)
from .f4_throughput_bench import window_split
from .rust_bridge import rust_flat_batch_adapter, rust_games_for_self_play


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


def test_scheduler_reports_time_weighted_occupancy_not_loop_counts():
    import seven_wonders_rust as swr

    seeds = [2026072600, 2026072601, 2026072602, 2026072603]
    first_players = [0, 1, 0, 1]
    _, metrics = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **_common(leaf_batch=2, global_batch_cap=16),
    )

    assert metrics["scheduler_wall_ns"] > 0
    assert metrics["max_live_slots"] == len(seeds)
    # Slot-seconds are bounded by wall time times the slot count, and occupancy
    # cannot exceed the number of games.
    assert metrics["live_slot_ns"] <= metrics["scheduler_wall_ns"] * len(seeds)
    assert 0.0 < metrics["time_weighted_live_slots"] <= len(seeds)
    for key in ("ready_slot_ns", "waiting_slot_ns", "idle_slot_ns"):
        assert metrics[key] >= 0
    assert metrics["live_slot_ns"] == metrics["ready_slot_ns"] + metrics["waiting_slot_ns"]

    # Occupancy must fall over a run that has no refill: the drain is the whole
    # point of the measurement.
    assert metrics["time_weighted_live_slots"] < len(seeds)


def test_batch_series_is_row_aligned_and_monotone_in_time():
    import seven_wonders_rust as swr

    seeds = [2026072610, 2026072611, 2026072612]
    _, metrics = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [0, 1, 0]),
        game_seeds=seeds,
        **_common(leaf_batch=2, global_batch_cap=16),
    )
    rows = metrics["batch_rows"]
    live = metrics["batch_live_slots"]
    submitted = metrics["batch_submit_ns"]
    assert len(rows) == len(live) == len(submitted) > 1
    assert submitted == sorted(submitted)
    assert max(live) == len(seeds)
    assert min(live) >= 1
    assert submitted[-1] <= metrics["scheduler_wall_ns"]


def test_arena_telemetry_bounds_active_games():
    import seven_wonders_rust as swr

    seeds = [2026072620, 2026072621]
    _, metrics = swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, [0, 1]),
        game_seeds=seeds,
        **_common(leaf_batch=1, global_batch_cap=8),
    )
    assert metrics["arena_node_struct_bytes"] > 0
    assert metrics["arena_nodes_slot_peak"] >= 1
    assert metrics["arena_nodes_live_peak"] >= metrics["arena_nodes_slot_peak"]
    # The deep measurement includes each node's cloned GameState, so it must
    # exceed the bare struct-size estimate it replaces.
    assert (
        metrics["arena_deep_bytes_slot_peak"]
        > metrics["arena_nodes_slot_peak"] * metrics["arena_node_struct_bytes"]
    )


def test_window_split_separates_steady_state_from_drain():
    metrics = {
        "batch_rows": [20, 22, 18, 9, 4],
        "batch_live_slots": [8, 8, 8, 4, 1],
        "batch_submit_ns": [0, 1_000_000, 2_000_000, 3_000_000, 3_500_000],
        "scheduler_wall_ns": 5_000_000,
    }
    split = window_split(metrics)
    assert split["steady_batches"] == 3
    assert split["steady_rows"] == 60
    assert split["steady_seconds"] == pytest.approx(0.003)
    assert split["drain_batches"] == 2
    assert split["drain_rows"] == 13
    assert split["drain_seconds"] == pytest.approx(0.002)


def test_window_split_survives_transient_dips_in_a_pooled_run():
    """A pool dips for a cycle on every retirement; that is not the drain.

    Splitting at the *first* dip would call this run 20% steady when it is 80%
    steady, which is how a healthy pooled run gets misread as a starving one.
    """

    metrics = {
        "batch_rows": [20, 9, 21, 20, 22, 8, 3],
        "batch_live_slots": [8, 5, 8, 8, 8, 4, 1],
        "batch_submit_ns": [0, 1_000, 2_000, 3_000, 4_000_000, 4_500_000, 4_800_000],
        "scheduler_wall_ns": 5_000_000,
    }
    split = window_split(metrics)
    assert split["steady_batches"] == 5
    assert split["steady_rows"] == 92
    assert split["drain_batches"] == 2
    assert split["drain_rows"] == 11
    assert split["steady_seconds"] == pytest.approx(0.0045)


def test_window_split_handles_a_run_that_never_drains():
    metrics = {
        "batch_rows": [16, 16],
        "batch_live_slots": [4, 4],
        "batch_submit_ns": [0, 500_000],
        "scheduler_wall_ns": 1_000_000,
    }
    split = window_split(metrics)
    assert split["steady_batches"] == 2
    assert split["drain_batches"] == 0
    assert split["drain_seconds"] == 0.0
    assert split["steady_seconds"] == pytest.approx(0.001)


def test_window_split_falls_back_when_series_are_absent():
    split = window_split({"scheduler_wall_ns": 2_000_000})
    assert split["steady_batches"] == 0
    assert split["steady_seconds"] == pytest.approx(0.002)
    assert split["drain_seconds"] == 0.0


class _PayloadCapture:
    """Adapter that records the first packed payload, then evaluates uniformly."""

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        import numpy as np

        self.payloads.append(
            {
                key: bytes(value) if isinstance(value, (bytes, bytearray)) else value
                for key, value in payload.items()
            }
        )
        rows = int(payload["rows"])
        legal_offsets = np.frombuffer(payload["legal_offsets"], dtype=np.int32)
        return [
            (
                0.0,
                [1.0 / count] * count,
            )
            for count in (
                int(legal_offsets[row + 1]) - int(legal_offsets[row])
                for row in range(rows)
            )
        ]


def test_cost_model_payload_is_byte_identical_to_the_rust_packing():
    """The harness must measure the boundary self-play actually uses.

    One game means the first packed batch is exactly that game's root position,
    so Python's packing can be compared to Rust's byte for byte.
    """

    import seven_wonders_rust as swr

    seed = 2026072630
    capture = _PayloadCapture()
    swr.self_play_many_flat_net(
        adapter=capture,
        games=rust_games_for_self_play([seed], [0]),
        game_seeds=[seed],
        force=False,
        max_inflight_batches=2,
        **_common(leaf_batch=1, global_batch_cap=8),
    )
    assert capture.payloads
    rust_payload = capture.payloads[0]

    game = rust_games_for_self_play([seed], [0])[0]
    mine = build_payload(
        [
            {
                "tokens": game.encode(),
                "actor": game.actor,
                "legal": game.legal_action_indices(),
            }
        ]
    )
    assert mine["rows"] == rust_payload["rows"] == 1
    assert mine["tokens"] == rust_payload["tokens"]
    assert mine["max_tokens"] == rust_payload["max_tokens"]
    assert mine["feature_width"] == rust_payload["feature_width"]
    for key in (
        "token_offsets",
        "type_ids",
        "entity_ids",
        "aux_ids",
        "features",
        "actors",
        "legal_offsets",
        "legal_actions",
    ):
        assert bytes(mine[key]) == rust_payload[key], key


def test_token_buckets_partition_the_corpus_by_length():
    corpus = [{"tokens": [0] * length} for length in (5, 9, 1, 7, 3, 11)]
    buckets = token_length_buckets(corpus, 3)
    assert [len(bucket["rows"]) for bucket in buckets] == [2, 2, 2]
    assert [bucket["min_tokens"] for bucket in buckets] == [1, 5, 9]
    assert [bucket["max_tokens"] for bucket in buckets] == [3, 7, 11]


def test_cost_model_fit_recovers_a_known_linear_cost():
    pytest.importorskip("torch")
    cells = [
        {
            "rows": rows,
            "padded_tokens": rows * tokens,
            "host_total_ms_mean": 2.0 + 0.01 * rows + 0.001 * rows * tokens,
        }
        for rows in (1, 8, 32, 128)
        for tokens in (40, 80)
    ]
    fit = fit_cost_model(cells, "host_total_ms_mean")
    assert fit["fitted"]
    assert fit["fixed_ms"] == pytest.approx(2.0, abs=1e-6)
    assert fit["per_row_ms"] == pytest.approx(0.01, abs=1e-6)
    assert fit["per_padded_token_ms"] == pytest.approx(0.001, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_passes_are_medianed_per_cell_and_report_their_spread():
    passes = [
        [{"bucket": "a", "rows": 8, "host_total_ms_mean": 10.0}],
        [{"bucket": "a", "rows": 8, "host_total_ms_mean": 12.0}],
        [{"bucket": "a", "rows": 8, "host_total_ms_mean": 20.0}],
    ]
    merged = aggregate_passes(passes)
    assert len(merged) == 1
    cell = merged[0]
    assert cell["passes"] == 3
    # Median, not mean: one slow pass must not drag the model.
    assert cell["host_total_ms_mean"] == pytest.approx(12.0)
    assert cell["host_total_ms_mean_spread"] == pytest.approx(10.0 / 12.0)
    assert cell["bucket"] == "a"
    assert cell["rows"] == 8


def test_fit_refuses_to_extrapolate_from_too_few_points():
    fit = fit_cost_model([{"rows": 1, "padded_tokens": 40, "x": 1.0}], "x")
    assert fit["fitted"] is False


def test_adapter_separates_sync_and_device_timers_on_cpu():
    """On CPU there is no device timeline, so events must stay off cleanly."""

    torch = pytest.importorskip("torch")
    from .inference import Evaluator
    from .net import SWDNet

    import seven_wonders_rust as swr

    torch.manual_seed(7)
    evaluator = Evaluator(SWDNet(32, 1, 2), device="cpu", max_batch=16)
    adapter = rust_flat_batch_adapter(evaluator, cuda_events=True, diagnostic_sync=True)
    assert adapter.cuda_events is False

    seeds = [2026072640]
    swr.self_play_many_flat_net(
        adapter=adapter,
        games=rust_games_for_self_play(seeds, [0]),
        game_seeds=seeds,
        force=False,
        max_inflight_batches=2,
        **{
            **_common(leaf_batch=1, global_batch_cap=8),
            "cheap_sims_max": 1,
            "full_sims_min": 1,
            "full_sims_max": 1,
            "full_search_fraction": 0.0,
            "top_k": 2,
        },
    )
    adapter.drain_events()
    assert adapter.total_metrics["batches"] > 0
    assert adapter.total_metrics["sync_seconds"] == 0.0
    for key in (
        "device_h2d_seconds",
        "device_forward_seconds",
        "device_gather_seconds",
        "device_d2h_seconds",
    ):
        assert adapter.total_metrics[key] == 0.0
    assert adapter.batch_device_forward_ms == []


# ---------------------------------------------------------------------------
# Phase 0 of COALESCER_BUILD_PLAN.md: the counters the coalescer will be
# judged by, and the two defects that made the previous judgement wrong.
# ---------------------------------------------------------------------------


def _solver_run(workers, solver_threads, *, seeds=range(400, 408)):
    """One sharded self-play run with the endgame solver live.

    The solver's eligibility and thread count are process globals, so they are
    saved and restored: a leak would silently change every later test in the
    process.
    """

    import seven_wonders_rust as swr

    from .control_table import ensure_rust_table
    from .test_f4_scheduler import _row_eval

    ensure_rust_table()
    saved_solver = swr.endgame_solver()
    saved_threads = swr.solver_threads()
    swr.set_endgame_solver(2_000_000, 5.0, 8, True)
    swr.set_solver_threads(solver_threads)
    try:
        seeds = list(seeds)
        _records, metrics = swr.self_play_many_net(
            adapter=lambda rows: [_row_eval(t, a, l) for t, a, l in rows],
            games=rust_games_for_self_play(
                seeds, [index % 2 for index in range(len(seeds))]
            ),
            game_seeds=seeds,
            scheduler_workers=workers,
            max_active_slots=len(seeds),
            solve_endgames=True,
            **_common(leaf_batch=1, global_batch_cap=32),
        )
        return metrics
    finally:
        if saved_solver[0] == 0:
            swr.set_endgame_solver(0, 1.0, 0, False)
        else:
            swr.set_endgame_solver(*saved_solver)
        swr.set_solver_threads(saved_threads)


def test_solver_blocking_survives_multi_shard_aggregation():
    """`sched_solve_wait_ns` was omitted from `SchedulerMetrics::merge`.

    It was declared, incremented and exported, so every consumer looked correct
    -- including `generation_profile.py`, which reports `solve_wait_share` and
    therefore answered "does the solver block generation?" with a guaranteed no
    on every run with more than one shard. cloud2 ran four.

    `THROUGHPUT_LEVERS.md` §3.3 records the identical defect happening once
    before to a different counter, which is why the compile-time guard in
    `merge` matters more than this test does.
    """

    one = _solver_run(workers=1, solver_threads=2)
    many = _solver_run(workers=4, solver_threads=2)
    assert one["sched_solve_wait_ns"] > 0
    # The pre-fix code reported exactly zero here however long it blocked.
    assert many["sched_solve_wait_ns"] > 0


def test_solver_blocking_is_separated_from_the_work_around_it():
    """The span used to cover the whole pump -- harvesting outcomes, resuming
    slots, re-collecting ready groups, retiring them -- as well as the block.
    All but the block is work the scheduler owed anyway, and some of it was
    double-counted against `sched_collect_ns`."""

    metrics = _solver_run(workers=4, solver_threads=2)
    assert metrics["sched_solve_pump_ns"] >= metrics["sched_solve_wait_ns"]
    assert metrics["sched_solve_pump_ns"] > 0


def test_inline_solving_is_visible_rather_than_free():
    """With no pool, `finish_move` solves on the scheduler thread, so there is
    nothing to block on and no blocking counter can see the cost.

    Without `sched_solve_inline_ns`, an inline run and a perfectly-parallel run
    both report ~zero blocking, for opposite reasons -- and the inline one is
    the expensive case.
    """

    inline = _solver_run(workers=4, solver_threads=0)
    assert inline["sched_solve_inline_ns"] > 0
    assert inline["sched_solve_wait_ns"] == 0
    assert inline["sched_solve_pump_ns"] == 0

    pooled = _solver_run(workers=4, solver_threads=2)
    assert pooled["sched_solve_wait_ns"] > 0


def _flat_run(workers, *, seeds=range(900, 908)):
    """A run through the PRODUCTION flat worker, which is the only path whose
    boundary counters move."""

    import seven_wonders_rust as swr
    import torch

    from .control_table import ensure_rust_table
    from .inference import Evaluator
    from .net import SWDNet

    ensure_rust_table()
    torch.manual_seed(0)
    evaluator = Evaluator(
        SWDNet(d_model=32, layers=2, heads=4), "cpu", 64, fuse_embedder=False
    )
    seeds = list(seeds)
    _records, metrics = swr.self_play_many_flat_net(
        adapter=rust_flat_batch_adapter(evaluator),
        games=rust_games_for_self_play(
            seeds, [index % 2 for index in range(len(seeds))]
        ),
        game_seeds=seeds,
        global_batch_cap=64,
        leaf_batch=1,
        cheap_sims_min=2,
        cheap_sims_max=3,
        full_sims_min=6,
        full_sims_max=8,
        full_search_fraction=0.4,
        top_k=3,
        draft_prior=0.0,
        iteration=1,
        scheduler_workers=workers,
        max_active_slots=len(seeds),
        max_moves=256,
    )
    return metrics


def test_requests_and_forwards_are_counted_separately():
    """`global_batches` is incremented in the SHARD, once per submitted request,
    before anything could merge it.

    So `global_rows / global_batches` is rows per REQUEST and cannot move when a
    coalescer lands -- an earlier revision of `COALESCER_BUILD_PLAN.md` proposed
    it as the headline success metric, where it would have been blind to its own
    subject. `boundary_forwards` comes from the worker instead.

    The equality that used to hold here -- one forward per request -- was the
    pre-coalescer baseline, and the coalescer inverted it, as this docstring
    said it would. What survives is the part that was always the point: the two
    counters measure different things, and only one of them can see a merge.

    `test_coalescer.py` owns the repair; this owns the distinction.
    """

    metrics = _flat_run(workers=4)
    assert metrics["boundary_forwards"] > 0
    # Rows are conserved -- merging changes how they travel, not how many.
    assert metrics["boundary_forward_rows"] == metrics["global_rows"]
    # ... while forwards are strictly fewer than the requests that produced
    # them. `global_batches` is incremented in the shard and cannot show this.
    assert metrics["worker_requests"] == metrics["global_batches"]
    assert metrics["boundary_forwards"] < metrics["global_batches"]


def test_shards_fragment_batches_rather_than_pooling_them():
    """The mechanism behind cloud2's 46.9 rows per call.

    Nothing merges requests from concurrent shards, so each assembles only from
    its own ready leaves: the same total work arrives as more, narrower calls as
    shards rise. Measured here at 1 against 4 shards on identical seeds.

    This is the quantity the coalescer is meant to recover, so it is pinned
    before the change rather than argued about after it.
    """

    one = _flat_run(workers=1)
    many = _flat_run(workers=4)
    # Identical work: same games, same seeds, same search budget.
    assert one["global_rows"] == many["global_rows"]
    # ... arriving in materially more, materially narrower calls.
    assert many["global_batches"] > one["global_batches"] * 2
    width = lambda m: m["global_rows"] / m["global_batches"]
    assert width(many) < width(one) / 2
