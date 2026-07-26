"""Phase 1 gates: the bounded rolling active-game pool.

The pool changes *when* games run, never what they compute, so the primary gate
is identity. Under a batch-shape-independent evaluator that identity is exact
and is asserted byte for byte. On a real net it cannot be: `self_play.rs`
documents that batch shape can change search choices through CUDA float ties, so
the real-net gates are (a) row-wise output invariance across batch shapes and
(b) a *measured* divergence rate rather than an assumed zero.
"""

from __future__ import annotations

import pytest

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


def _row_eval(tokens, actor, legal):
    """Deterministic per-row evaluator, independent of batch shape."""

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


def _mock(seeds, first_players=None, **extra):
    import seven_wonders_rust as swr

    if first_players is None:
        first_players = [index % 2 for index in range(len(seeds))]
    return swr.self_play_many_mock(
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        **{**_common(leaf_batch=2, global_batch_cap=16), **extra},
    )


# --------------------------------------------------------------------------
# Gate 1 (mock): the pool is an exact refactor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("max_active_slots", [1, 2, 3, 5, 12, 64])
def test_pool_records_are_byte_identical_to_the_unpooled_path(max_active_slots):
    seeds = [2026072700 + index for index in range(12)]
    baseline, _ = _mock(seeds)
    pooled, metrics = _mock(seeds, max_active_slots=max_active_slots)

    assert pooled == baseline
    assert [record["seed"] for record in pooled] == seeds
    assert metrics["max_live_slots"] == min(max_active_slots, len(seeds))


def test_pool_records_match_the_chunked_path_it_replaces():
    """The pooled call must reproduce what per-chunk calls produced.

    This is the gate the plan names: `phase_d.py` used to submit ceil(N/S)
    separate calls of S games, and the pooled single call has to return the same
    records in the same order.
    """

    seeds = [2026072720 + index for index in range(9)]
    first_players = [index % 2 for index in range(9)]
    chunked = []
    for start in range(0, len(seeds), 3):
        window = slice(start, start + 3)
        # The same seat assignment the whole-run call would give these games,
        # so the only difference between the two paths is the schedule.
        records, _ = _mock(seeds[window], first_players[window])
        chunked.extend(records)
    pooled, _ = _mock(seeds, first_players, max_active_slots=3)
    assert pooled == chunked


def test_pool_holds_occupancy_at_the_ceiling_instead_of_draining():
    """The point of the pool: concurrency stops decaying to one.

    Without refill, 24 games at 4 slots means six independent drains and a mean
    occupancy near half the ceiling. With it, occupancy sits just under the
    ceiling until the queue empties.
    """

    seeds = [2026072740 + index for index in range(24)]
    _, pooled = _mock(seeds, max_active_slots=4)
    _, drained = _mock(seeds[:4])  # one un-refilled window of the same width

    assert pooled["max_live_slots"] == 4
    assert pooled["max_active_slots"] == 4
    # Time-weighted occupancy, not a loop count: with 24 games over 4 slots the
    # drain tail is a small fraction of the run.
    assert pooled["time_weighted_live_slots"] > 3.5
    assert pooled["time_weighted_live_slots"] > drained["time_weighted_live_slots"]
    # Refill must be visible as batches taken at full width late in the run.
    assert pooled["batch_live_slots"][-1] <= 4
    assert max(pooled["batch_live_slots"]) == 4
    assert pooled["batch_live_slots"].count(4) > len(pooled["batch_live_slots"]) // 2


def test_queued_games_hold_no_arena_until_activated():
    """Queued jobs must stay lightweight, or the pool bounds nothing.

    A pool of 2 over 16 games must peak at roughly the arena of 2 games, not 16.
    """

    seeds = [2026072760 + index for index in range(16)]
    _, small = _mock(seeds, max_active_slots=2)
    _, large = _mock(seeds, max_active_slots=16)
    assert small["arena_nodes_live_peak"] < large["arena_nodes_live_peak"]
    assert small["max_live_slots"] == 2


def test_pool_rejects_a_budget_that_cannot_feed_every_shard():
    import seven_wonders_rust as swr

    seeds = [2026072780 + index for index in range(6)]
    with pytest.raises(ValueError, match="below scheduler_workers"):
        swr.self_play_many_net(
            adapter=lambda rows: [_row_eval(*row) for row in rows],
            games=rust_games_for_self_play(seeds, [0, 1, 0, 1, 0, 1]),
            game_seeds=seeds,
            max_active_slots=2,
            scheduler_workers=4,
            max_inflight_batches=1,
            **_common(leaf_batch=1, global_batch_cap=8),
        )


def test_global_budget_is_not_multiplied_by_shards():
    """Sharding must not multiply resident games -- the budget is global."""

    import seven_wonders_rust as swr

    seeds = [2026072790 + index for index in range(8)]
    first_players = [index % 2 for index in range(8)]
    kwargs = {
        **_common(leaf_batch=1, global_batch_cap=8),
        "max_inflight_batches": 1,
        "max_active_slots": 4,
    }
    records, metrics = swr.self_play_many_net(
        adapter=lambda rows: [_row_eval(*row) for row in rows],
        games=rust_games_for_self_play(seeds, first_players),
        game_seeds=seeds,
        scheduler_workers=2,
        **kwargs,
    )
    assert [record["seed"] for record in records] == seeds
    assert metrics["max_active_slots"] == 4
    # Summed over shards, so it may not exceed the shared ceiling.
    assert metrics["max_live_slots"] <= 4


# --------------------------------------------------------------------------
# Gate 2 (real net): invariance first, then measured divergence
# --------------------------------------------------------------------------


def _evaluator():
    torch = pytest.importorskip("torch")
    from .inference import Evaluator
    from .net import SWDNet

    torch.manual_seed(11)
    return Evaluator(SWDNet(32, 1, 2), device="cpu", max_batch=64)


def test_real_net_rows_are_invariant_to_batch_shape():
    """Gate 2(a): the same position must evaluate the same at any batch width.

    If this fails, the pool cannot be gated as an exact refactor and has to be
    treated as a numerical change instead -- so it is checked before divergence
    is measured, not after.
    """

    torch = pytest.importorskip("torch")

    from .f4_cost_model import build_payload, collect_corpus

    evaluator = _evaluator()
    adapter = rust_flat_batch_adapter(evaluator)
    corpus = collect_corpus(2, 2026072800, stride=7)[:8]
    assert len(corpus) >= 4

    alone = [adapter(build_payload([row]))[0] for row in corpus]
    together = adapter(build_payload(corpus))
    padded = adapter(build_payload(corpus + corpus[:1]))[: len(corpus)]

    for index, (single, grouped, repeated) in enumerate(zip(alone, together, padded)):
        assert single[0] == pytest.approx(grouped[0], abs=1e-6), index
        assert single[0] == pytest.approx(repeated[0], abs=1e-6), index
        assert len(single[1]) == len(grouped[1])
        for one, many in zip(single[1], grouped[1]):
            assert one == pytest.approx(many, abs=1e-6), index


def test_real_net_pool_divergence_is_measured_not_assumed():
    """Gate 2(b): quantify how far pooled play drifts from unpooled play.

    Reported rather than asserted at zero, because CPU float determinism here
    does not license the same claim on CUDA. The assertion is the weaker,
    honest one: the games remain valid and complete, and any divergence is
    small enough to attribute to arithmetic rather than to a scheduling bug.
    """

    import seven_wonders_rust as swr

    from .buffer import replay
    from .game import Phase
    from .rust_bridge import phase_d_records_from_rust

    evaluator = _evaluator()
    seeds = [2026072810 + index for index in range(6)]
    first_players = [index % 2 for index in range(6)]
    kwargs = {
        **_common(leaf_batch=2, global_batch_cap=16),
        "cheap_sims_max": 1,
        "full_sims_min": 1,
        "full_sims_max": 1,
        "full_search_fraction": 0.0,
        "top_k": 2,
        "force": False,
        "max_inflight_batches": 2,
    }

    def play(max_active_slots):
        records, metrics = swr.self_play_many_flat_net(
            adapter=rust_flat_batch_adapter(evaluator),
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            max_active_slots=max_active_slots,
            **kwargs,
        )
        return records, metrics

    unpooled, _ = play(0)
    pooled, metrics = play(2)

    assert [record["seed"] for record in pooled] == seeds
    assert metrics["max_live_slots"] == 2
    # Records must be structurally valid whatever the schedule did.
    for record in phase_d_records_from_rust(pooled):
        assert replay(record).phase is Phase.COMPLETE

    total_moves = 0
    differing_moves = 0
    differing_games = 0
    for baseline, candidate in zip(unpooled, pooled):
        moves = min(len(baseline["moves"]), len(candidate["moves"]))
        total_moves += max(len(baseline["moves"]), len(candidate["moves"]))
        game_differs = len(baseline["moves"]) != len(candidate["moves"])
        for left, right in zip(baseline["moves"][:moves], candidate["moves"][:moves]):
            if left["action"] != right["action"]:
                differing_moves += 1
                game_differs = True
        if game_differs:
            differing_games += 1

    divergence = differing_moves / total_moves if total_moves else 0.0
    print(
        f"pool divergence: {differing_games}/{len(seeds)} games, "
        f"{differing_moves}/{total_moves} moves ({divergence:.4%})"
    )
    # On this deterministic CPU evaluator the schedule should change nothing.
    assert divergence == 0.0, "CPU-path divergence means a scheduling bug, not float ties"
