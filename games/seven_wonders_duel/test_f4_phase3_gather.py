"""Phase 3 gates: the vectorised legal-policy gather.

The boundary used to pay three costs that all scale with rows while the
per-batch cost is flat — one `torch.softmax` launch per row, two device-to-host
transfers, and a Python `float()` per legal action. Those are what make a wider
batch stop paying, so they had to go before any batch-widening work means
anything.

The plan sets two gates: row-wise outputs identical to the loop within tolerance,
and `gather_seconds` per row that no longer grows with batch width.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from .f4_cost_model import build_payload, collect_corpus
from .inference import Evaluator
from .net import SWDNet
from .rust_bridge import rust_flat_batch_adapter


TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def corpus():
    rows = collect_corpus(3, 2026073000, stride=5)
    assert len(rows) >= 24
    return rows


def _evaluator(seed=13):
    torch.manual_seed(seed)
    return Evaluator(SWDNet(32, 1, 2), device="cpu", max_batch=512)


def test_vectorised_gather_matches_the_per_row_loop(corpus):
    evaluator = _evaluator()
    loop = rust_flat_batch_adapter(evaluator, vectorized_gather=False)
    fast = rust_flat_batch_adapter(evaluator, vectorized_gather=True)
    for size in (1, 2, 7, 24):
        payload = build_payload(corpus[:size])
        expected = loop(payload)
        got = fast(payload)
        assert len(got) == len(expected) == size
        for index, (reference, candidate) in enumerate(zip(expected, got)):
            assert candidate[0] == pytest.approx(
                reference[0], rel=0, abs=TOLERANCE
            ), f"rows={size} row={index}: value"
            assert len(candidate[1]) == len(reference[1]), f"rows={size} row={index}"
            assert candidate[1] == pytest.approx(
                reference[1], rel=0, abs=TOLERANCE
            ), f"rows={size} row={index}: policy"


def test_gather_is_run_to_run_deterministic(corpus):
    """Repeated identical calls must return bit-identical policies.

    This gate exists because the first implementation failed it. A scatter-based
    segmented sum (`index_add_`) accumulates with atomics on CUDA, so its result
    varied by ~8e-7 relative between runs — which would have made generation
    irreproducible at a fixed seed, a worse property than any tolerance. The
    padded formulation writes each cell exactly once and reduces within a row, so
    it is deterministic; on CPU this passes either way, so it only bites on CUDA.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(13)
    evaluator = Evaluator(SWDNet(32, 1, 2), device=device, max_batch=512)
    adapter = rust_flat_batch_adapter(evaluator, vectorized_gather=True)
    payload = build_payload(corpus[:24])

    def flat(result):
        return [value for _value, policy in result for value in policy]

    first = flat(adapter(payload))
    for repeat in range(15):
        assert flat(adapter(payload)) == first, f"repeat {repeat} differed"


def test_each_rows_policy_is_a_distribution_over_its_own_legal_actions(corpus):
    """A segmented softmax's failure mode is leaking mass across segments.

    Row-wise agreement can hide it when rows are similar, so normalisation is
    checked per row directly.
    """

    evaluator = _evaluator()
    fast = rust_flat_batch_adapter(evaluator, vectorized_gather=True)
    payload = build_payload(corpus[:16])
    for index, (_value, policy) in enumerate(fast(payload)):
        assert len(policy) == len(corpus[index]["legal"])
        assert sum(policy) == pytest.approx(1.0, rel=0, abs=1e-6), index
        assert all(probability > 0.0 for probability in policy), index


def test_rows_with_no_legal_actions_do_not_poison_the_batch(corpus):
    """Terminal rows carry zero legal actions, so their segment sum is zero.

    The division must never read that zero — one NaN would propagate into every
    prior in the batch.
    """

    evaluator = _evaluator()
    fast = rust_flat_batch_adapter(evaluator, vectorized_gather=True)
    rows = [dict(corpus[0]), dict(corpus[1]), dict(corpus[2])]
    rows[1] = dict(rows[1], legal=[])  # a terminal row in the middle
    payload = build_payload(rows)
    result = fast(payload)
    assert result[1][1] == []
    for index in (0, 2):
        policy = result[index][1]
        assert policy, index
        assert all(probability == probability for probability in policy), index
        assert sum(policy) == pytest.approx(1.0, rel=0, abs=1e-6), index


def test_gather_cost_per_row_stops_scaling_with_batch_width(corpus):
    """The plan's second gate, as ratios rather than absolutes.

    Two claims, both controlled: the vectorised path's cost *per row* must fall as
    the batch widens (a fixed number of launches amortising), and at width it must
    beat the loop measured on the same machine in the same run.

    Deliberately not asserted: that the loop's own per-row cost stays flat. It
    does on a GPU, where the launch per row dominates, but on CPU the fixed
    per-call overhead amortises too — that is a property of the hardware, not of
    the change being gated.
    """

    evaluator = _evaluator()
    samples = [corpus[index % len(corpus)] for index in range(128)]

    def gather_per_row(adapter, size, repeats=12):
        payload = build_payload(samples[:size])
        for _ in range(3):
            adapter(payload)
        before = float(adapter.total_metrics["gather_seconds"])
        for _ in range(repeats):
            adapter(payload)
        spent = float(adapter.total_metrics["gather_seconds"]) - before
        return spent / repeats / size

    loop = rust_flat_batch_adapter(evaluator, vectorized_gather=False)
    fast = rust_flat_batch_adapter(evaluator, vectorized_gather=True)
    narrow_loop = gather_per_row(loop, 8)
    wide_loop = gather_per_row(loop, 128)
    narrow_fast = gather_per_row(fast, 8)
    wide_fast = gather_per_row(fast, 128)
    print(
        f"gather µs/row  loop: {narrow_loop * 1e6:.2f} (8 rows) -> "
        f"{wide_loop * 1e6:.2f} (128)   "
        f"vectorised: {narrow_fast * 1e6:.2f} -> {wide_fast * 1e6:.2f}"
    )
    # Amortises: per-row cost drops sharply as the batch widens.
    assert wide_fast < 0.5 * narrow_fast, (narrow_fast, wide_fast)
    # And beats the loop outright at width, measured in the same run.
    assert wide_fast < 0.5 * wide_loop, (wide_loop, wide_fast)


def test_vectorised_gather_does_the_same_scheduler_work():
    """End-to-end: the search must be untouched, not merely close per row."""

    import seven_wonders_rust as swr

    from .rust_bridge import rust_games_for_self_play

    evaluator = _evaluator(seed=21)
    seeds = [2026073010 + index for index in range(3)]
    first_players = [0, 1, 0]
    kwargs = dict(
        global_batch_cap=16,
        leaf_batch=4,
        cheap_sims_min=2,
        cheap_sims_max=2,
        full_sims_min=2,
        full_sims_max=2,
        full_search_fraction=0.0,
        top_k=4,
        draft_prior=0.55,
        iteration=3,
        force=False,
        max_inflight_batches=2,
        conflict_free_waves=True,
    )

    def play(vectorized):
        return swr.self_play_many_flat_net(
            adapter=rust_flat_batch_adapter(evaluator, vectorized_gather=vectorized),
            games=rust_games_for_self_play(seeds, first_players),
            game_seeds=seeds,
            **kwargs,
        )

    from .test_f4_phase3b_fused import assert_records_identical

    loop_records, loop_metrics = play(False)
    fast_records, fast_metrics = play(True)
    # Full trajectories, not just counts and fingerprints: two different move
    # sequences can reach the same final state, and counts stay equal when
    # actions change.
    drift = assert_records_identical(loop_records, fast_records)
    print(
        f"vectorised vs loop: every decision identical; "
        f"max policy-target drift {drift['max_policy_target_drift']:.2e}"
    )
    assert fast_metrics["simulations"] == loop_metrics["simulations"]
    assert fast_metrics["global_batches"] == loop_metrics["global_batches"]
    assert fast_metrics["moves"] == loop_metrics["moves"]
    assert list(fast_metrics["batch_rows"]) == list(loop_metrics["batch_rows"])


def test_vectorised_gather_is_on_by_default_and_can_be_turned_off():
    """Shipped as the default since Phase 3; the opt-out is the measurement path."""

    evaluator = _evaluator()
    assert rust_flat_batch_adapter(evaluator).vectorized_gather is True
    assert (
        rust_flat_batch_adapter(evaluator, vectorized_gather=False).vectorized_gather
        is False
    )


def test_evaluator_fuses_the_embedder_by_default_and_can_be_told_not_to():
    """Phase 3b ships through `Evaluator`, so every inference path inherits it."""

    from .inference import Evaluator

    from .net import fusion_is_profitable

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(31)
    evaluator = Evaluator(SWDNet(32, 1, 2), device=device, max_batch=8)
    # Fusing is enabled only where it is measured to pay -- CUDA. On CPU the
    # extra arithmetic is a ~10% loss at width, so the default declines.
    assert evaluator.fused_embedder is fusion_is_profitable(device)
    assert (evaluator.model.embedder._fused is not None) is evaluator.fused_embedder

    torch.manual_seed(31)
    plain = Evaluator(SWDNet(32, 1, 2), device=device, max_batch=8, fuse_embedder=False)
    assert plain.fused_embedder is False
    assert plain.model.embedder._fused is None
