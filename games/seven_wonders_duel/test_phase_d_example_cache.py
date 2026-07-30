"""The in-memory example cache.

`examples_from_records` replays every game through the verified engine path, so
rebuilding the replay window each iteration re-verifies immutable data: measured
at 223 s over 1,633 games and 404 s over 4,800, against 11 s of actual training.
The cache replays each game once per process instead.

The gate that matters is equivalence: same examples, same order, same training.
Everything else here is about the cache not growing without bound and not
serving one game's examples for another.
"""

from __future__ import annotations

import numpy as np
import pytest

from .buffer import GameRecord
from .dataset import examples_from_records
from .phase_d import (
    DEFAULT_CACHE_CALIBRATION_FACTOR,
    LEGACY_EXAMPLE_BYTES,
    PhaseDConfig,
    PhaseDLoop,
)


def _same(left, right) -> bool:
    """Example equality: numpy fields compared by value, the rest by =="""

    arrays = ("type_ids", "entity_ids", "aux_ids", "features", "legal", "policy_target")
    for field in arrays:
        if not np.array_equal(getattr(left, field), getattr(right, field)):
            return False
    scalars = (
        "has_policy", "value_class", "joint7_class", "margin", "margin_valid",
        "military_final", "sci_final_my", "sci_final_opp", "game_key", "iteration",
    )
    return all(getattr(left, f) == getattr(right, f) for f in scalars)


def _loop(tmp_path, **overrides) -> PhaseDLoop:
    config = PhaseDConfig(
        run_dir=str(tmp_path / "run"),
        device="cpu",
        seed_games=0,
        games_per_iteration=2,
        **overrides,
    )
    loop = PhaseDLoop(config)
    loop.buffer_dir.mkdir(parents=True, exist_ok=True)
    return loop


@pytest.fixture(scope="module")
def records(tmp_path_factory) -> list[GameRecord]:
    """Real self-play games, generated here.

    They cannot be fabricated -- every example is produced by replaying the
    record through the verified engine path -- and they must contain *searched*
    moves, because a curriculum bot's moves record no simulations and so are
    never classified fast. Reading a previous run's buffer would make this file
    skip on a clean checkout.
    """

    directory = tmp_path_factory.mktemp("example_cache_records")
    config = PhaseDConfig(
        run_dir=str(directory / "run"),
        device="cpu",
        seed_games=0,
        games_per_iteration=6,
        d_model=32,
        layers=1,
        cheap_sims_min=1,
        cheap_sims_max=2,
        full_sims_min=2,
        full_sims_max=3,
        full_search_fraction=0.5,
        top_k=2,
        age_deal_samples=0,
    )
    loop = PhaseDLoop(config)
    loop.initialize()
    generated = loop.generate_iteration(loop.load_model(loop.current_best), 0)
    assert any(
        move.sims > 0 for record in generated for move in record.moves
    ), "fixture must contain searched moves"
    return generated


def test_cache_returns_exactly_what_the_uncached_path_returns(tmp_path, records):
    """The equivalence gate: same examples, same order, cold and warm."""

    loop = _loop(tmp_path)
    expected = examples_from_records(records, record_fast_moves=False)

    cold = loop._cached_examples(records)
    assert len(cold) == len(expected)
    assert all(_same(a, b) for a, b in zip(cold, expected))

    warm = loop._cached_examples(records)
    assert len(warm) == len(expected)
    assert all(_same(a, b) for a, b in zip(warm, expected))
    assert loop.last_example_cache_stats["replayed_games"] == 0, "second pass replayed"


def test_a_repeated_game_is_replayed_once_but_still_emitted_twice(tmp_path, records):
    """Duplicates share a cache entry without collapsing the training set."""

    loop = _loop(tmp_path)
    doubled = list(records) + list(records)
    examples = loop._cached_examples(doubled)

    single = examples_from_records(records, record_fast_moves=False)
    assert len(examples) == 2 * len(single)
    assert loop.last_example_cache_stats["replayed_games"] == len(records)


def test_record_fast_moves_is_part_of_the_key(tmp_path, records):
    """The two settings emit different example sets; one must not serve the other."""

    loop = _loop(tmp_path)
    slow = loop._cached_examples(records)
    loop.config.record_fast_moves = True
    fast = loop._cached_examples(records)

    assert len(fast) > len(slow), "fast moves must add examples"
    assert loop.last_example_cache_stats["replayed_games"] == len(records)


def test_same_trajectory_with_different_targets_does_not_share_an_entry(
    tmp_path, records
):
    """The collision the first key permitted.

    `trajectory_digest` chains the replayed *states*, so it is identical for two
    records of the same played game carrying different search targets -- exactly
    what reanalysis and warm-buffer imports produce. The policy label comes from
    `policy_target`, so sharing an entry would train on another record's labels.
    """

    from dataclasses import replace

    from .dataset import is_fast_search_move

    # The move must both carry a target and survive the fast-move filter, or
    # changing its label provably cannot change any emitted example.
    original, searched = next(
        (
            (record, index)
            for record in records
            for index, move in enumerate(record.moves)
            if move.policy_target and not is_fast_search_move(move)
        ),
        (None, None),
    )
    assert searched is not None, "fixture needs a recorded full-search move"

    move = original.moves[searched]
    flipped = dict(move.policy_target)
    keys = sorted(flipped)
    # Move all the mass onto a different legal action: same game, new label.
    retargeted = {key: (1.0 if key == keys[-1] else 0.0) for key in keys}
    altered = replace(
        original,
        moves=tuple(
            replace(m, policy_target=retargeted) if i == searched else m
            for i, m in enumerate(original.moves)
        ),
    )
    assert altered.trajectory_digest == original.trajectory_digest, (
        "precondition: the digest cannot see target changes"
    )

    loop = _loop(tmp_path)
    first = loop._cached_examples([original])
    second = loop._cached_examples([altered])
    assert loop.last_example_cache_stats["replayed_games"] == 1, "must not reuse"

    changed = [
        (a, b)
        for a, b in zip(first, second)
        if not np.array_equal(a.policy_target, b.policy_target)
    ]
    assert changed, "the retargeted move must produce a different label"


def test_altering_a_verified_field_misses_the_cache(tmp_path, records):
    """Keying on a stored digest would let tampering skip its own check.

    `trajectory_digest` is a field *of the record*, so a record whose actions
    were altered without updating it would have hit the cache and never been
    replayed -- bypassing the verification that exists to catch exactly that.
    Keying on the whole serialized record means any alteration misses.
    """

    from dataclasses import replace

    from .buffer import ReplayMismatchError

    original = records[0]
    tampered = replace(
        original,
        moves=tuple(
            replace(m, mask_hash="sha256:0" * 4) if i == 0 else m
            for i, m in enumerate(original.moves)
        ),
    )
    assert tampered.trajectory_digest == original.trajectory_digest

    loop = _loop(tmp_path)
    loop._cached_examples([original])
    with pytest.raises(ReplayMismatchError):
        loop._cached_examples([tampered])


def test_iteration_label_is_part_of_the_key(tmp_path, records):
    """Warm-buffer imports can carry the same game under a different iteration.

    The label lands in `Example.iteration`, which drives the temporal split, so
    serving one iteration's examples for another would silently mislabel them.
    """

    from dataclasses import replace

    loop = _loop(tmp_path)
    original = loop._cached_examples(records[:1])
    relabelled = loop._cached_examples([replace(records[0], iteration=999)])

    assert loop.last_example_cache_stats["replayed_games"] == 1, "must not reuse"
    assert {e.iteration for e in relabelled} == {999}
    assert {e.iteration for e in original} != {999}


def test_capacity_is_enforced_even_when_one_window_exceeds_it(tmp_path, records):
    """The cap must bound retention unconditionally.

    The first version refused to evict below the current window, so a window
    larger than the cap stayed cached in full and the cap silently did nothing --
    a memory bound that does not bound memory. `out` already holds every
    reference the training call needs, so evicting a current-window entry costs
    a replay next iteration and nothing else.
    """

    loop = _loop(tmp_path)
    per_game = len(loop._cached_examples(records[:1]))
    loop._example_cache.clear()
    # Room for roughly two games against a six-game window.
    loop.config.example_cache_examples = per_game * 2

    returned = loop._cached_examples(records)
    stats = loop.last_example_cache_stats
    assert stats["cached_examples"] <= loop.config.example_cache_examples, (
        "cache retained more than the configured cap"
    )
    assert stats["evicted_games"] > 0
    assert len(returned) == len(
        examples_from_records(records, record_fast_moves=False)
    ), "eviction must not change what is returned"


def test_an_oversized_window_warns_on_the_first_call(tmp_path, records, capsys):
    """The loud failure mode: a cap below one window re-replays forever."""

    loop = _loop(tmp_path, example_cache_examples=1)
    loop._cached_examples(records)
    printed = capsys.readouterr().out
    assert "below this window" in printed
    assert "--example-cache-examples" in printed


def test_a_cap_below_one_window_still_returns_every_example(tmp_path, records):
    """Thrashing is a performance failure, never a correctness one."""

    loop = _loop(tmp_path, example_cache_examples=1)
    examples = loop._cached_examples(records)
    expected = examples_from_records(records, record_fast_moves=False)
    assert len(examples) == len(expected)
    assert all(_same(a, b) for a, b in zip(examples, expected))


def test_zero_capacity_disables_the_cache(tmp_path, records):
    """The escape hatch must actually restore the old behaviour."""

    loop = _loop(tmp_path, example_cache_examples=0)
    first = loop._cached_examples(records)
    second = loop._cached_examples(records)

    assert loop._example_cache == {}
    assert loop.last_example_cache_stats["replayed_games"] == len(records)
    assert all(_same(a, b) for a, b in zip(first, second))


def test_cached_examples_cannot_be_mutated(tmp_path, records):
    """Aliasing is load-bearing, so immutability is enforced, not documented.

    The same object is handed to every iteration whose window contains the game,
    so an in-place write would propagate into every later iteration. An identity
    assertion cannot catch that -- a future writer would mutate the shared object
    and the assertion would still pass -- so the fields are frozen and the arrays
    are read-only.
    """

    loop = _loop(tmp_path)
    first = loop._cached_examples(records)
    second = loop._cached_examples(records)
    assert all(a is b for a, b in zip(first, second)), "sharing is the point"

    example = first[0]
    with pytest.raises(Exception):  # FrozenInstanceError
        example.value_class = 2
    with pytest.raises(ValueError):  # read-only array
        example.features[0, 0] = 1.0
    with pytest.raises(ValueError):
        example.policy_target[0] = 0.5
    with pytest.raises(ValueError):
        example.legal[0] = 0


def test_stats_report_what_was_replayed(tmp_path, records):
    loop = _loop(tmp_path)
    loop._cached_examples(records)
    cold = dict(loop.last_example_cache_stats)
    loop._cached_examples(records)
    warm = dict(loop.last_example_cache_stats)

    assert cold["games"] == warm["games"] == len(records)
    assert cold["replayed_games"] == len(records)
    assert warm["replayed_games"] == 0
    assert warm["cached_games"] == len(records)
    assert cold["examples"] == warm["examples"]


def test_legacy_count_converts_at_measured_rss_cost(tmp_path):
    loop = _loop(tmp_path, example_cache_examples=123)
    assert loop._cache_capacity_bytes() == 123 * LEGACY_EXAMPLE_BYTES


def test_cache_uses_calibrated_bytes_not_array_nbytes(tmp_path, records):
    loop = _loop(tmp_path)
    loop._cached_examples(records)
    stats = loop.last_example_cache_stats
    assert stats["calibration_factor"] >= DEFAULT_CACHE_CALIBRATION_FACTOR
    assert stats["estimated_bytes"] >= stats["raw_array_bytes"]
    assert stats["capacity_bytes"] == (
        loop.config.example_cache_examples * LEGACY_EXAMPLE_BYTES
    )


def test_gate_admission_evicts_cache_before_clean_memory_error(tmp_path, records):
    loop = _loop(tmp_path)
    loop._cached_examples(records)
    assert loop._example_cache
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.write_bytes(b"x" * 1024)
    loop.memory_budget_bytes = 1

    with pytest.raises(MemoryError, match="admission refused"):
        loop._admit_gate((checkpoint, checkpoint))

    assert not loop._example_cache
    assert any(
        event["phase"] == "pre_gate_admission"
        for event in loop.resource_monitor.memory_pressure
    )
