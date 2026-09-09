"""Correctness gates for evaluator coalescing (COALESCER_BUILD_PLAN.md §4).

The coalescer merges requests from concurrent shards into one forward. Every
test here is written to fail two ways: on a WRONG merge, and on the
pre-coalescer path where nothing merges at all. A gate that only catches the
first would pass a silent revert to one-request-per-forward, which is the
failure this whole change set keeps finding elsewhere -- a subsystem that is
configured, reported, and not running.

The merges are FORCED rather than hoped for. `_coalescer_probe` submits an
exact list of requests and holds every ticket; an adapter that blocks on its
first call then guarantees the rest pile up behind it and drain together. A
test written through `self_play_many_flat_net` could not do this: the scheduler
decides its own request shapes from search state, so "two shards happened to
overlap" is a race, not an assertion.
"""

from __future__ import annotations

import struct
import threading

import pytest

from .rust_bridge import rust_games_for_self_play


def _games(count: int):
    seeds = [2026090900 + index for index in range(count)]
    return rust_games_for_self_play(seeds, [index % 2 for index in range(count)])


def _legal_lengths(payload) -> list[int]:
    offsets = struct.unpack(f"<{len(payload['legal_offsets']) // 4}I", bytes(payload["legal_offsets"]))
    return [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]


class _Adapter:
    """Records every call, and can be made to block on the first one.

    `value_of` decides what each row's value is, so a test can make the answer
    encode whatever it needs to trace -- the row's position in the forward, or
    the network it was routed to.
    """

    def __init__(self, *, block_first=False, fail_on_call=None, value_of=None):
        self.calls: list[dict] = []
        self.released = threading.Event()
        self.first_call_entered = threading.Event()
        self.block_first = block_first
        self.fail_on_call = fail_on_call
        self.value_of = value_of or (lambda call, row, net, rows: 0.25)

    def __call__(self, payload):
        rows = int(payload["rows"])
        nets = list(bytes(payload["net_ids"]))
        lengths = _legal_lengths(payload)
        index = len(self.calls)
        self.calls.append({"rows": rows, "net_ids": nets, "legal_lengths": lengths})
        if index == 0:
            self.first_call_entered.set()
            if self.block_first:
                # Held until the test has submitted everything else, so the
                # merge below is a fact rather than a race.
                assert self.released.wait(timeout=30.0), "probe never released"
        if self.fail_on_call is not None and index == self.fail_on_call:
            raise ValueError("adapter refused this batch")
        return [
            (
                self.value_of(index, row, nets[row], rows),
                [1.0 / lengths[row]] * lengths[row],
            )
            for row in range(rows)
        ]


def _probe(adapter, request_rows, *, request_nets=None, max_rows=64, wait_ms=0.0,
           timeout_ms=0.0, games=6, pause_after=0, gate=None, drop_tickets=()):
    import seven_wonders_rust as swr

    if request_nets is None:
        request_nets = [None] * len(request_rows)
    return swr._coalescer_probe(
        adapter,
        _games(games),
        request_rows,
        request_nets,
        max_rows,
        timeout_ms,
        wait_ms,
        pause_after,
        gate,
        list(drop_tickets),
    )


def _run_blocking(adapter, request_rows, **kwargs):
    """Submit request 0, wait for the adapter to be INSIDE its first call, then
    submit the rest and release it.

    Without the pause this is a race, and one that resolves the flattering way:
    submission is a few clones and a channel send while the worker has to
    acquire the GIL before it can call the adapter, so all the requests land
    first and leave as one enormous merge. That passes whatever the drain loop
    does -- including a broken carry-over or a cap that is never enforced. The
    gate makes the split between forwards a decision instead.
    """

    assert adapter.block_first, "_run_blocking needs an adapter that holds call 0"
    result: dict = {}

    def drive():
        try:
            result["value"] = _probe(
                adapter,
                request_rows,
                pause_after=1,
                # `Event.wait` releases the GIL, so the worker can take it and
                # enter the adapter while this blocks.
                gate=lambda: adapter.first_call_entered.wait(timeout=30.0),
                **kwargs,
            )
        except BaseException as error:  # surfaced by the caller
            result["error"] = error

    thread = threading.Thread(target=drive)
    thread.start()
    assert adapter.first_call_entered.wait(timeout=30.0), "adapter was never called"
    # The gate has now returned, so requests 1.. are being submitted. They pile
    # up behind the held call; release it once they are all in.
    thread.join(timeout=1.0)
    adapter.released.set()
    thread.join(timeout=60.0)
    assert not thread.is_alive(), "probe did not finish"
    if "error" in result:
        raise result["error"]
    return result["value"]


# ---------------------------------------------------------------------------
# The mechanism is engaged at all
# ---------------------------------------------------------------------------
def test_queued_requests_merge_into_one_forward():
    """Without this, everything below passes on the pre-coalescer path.

    Four requests, one held call: the three that queue behind it must leave as
    a single forward. `requests_per_forward` is 1.00 exactly when nothing
    merged, so this is the assertion a silent revert fails.
    """

    adapter = _Adapter(block_first=True)
    answers, metrics = _run_blocking(adapter, [2, 3, 4, 5])

    assert all(answer[0] == "ok" for answer in answers)
    assert metrics["worker_requests"] == 4
    # The held call is one forward; the three behind it are the second.
    assert metrics["forwards"] == 2
    assert metrics["worker_requests"] / metrics["forwards"] > 1.0
    assert adapter.calls[0]["rows"] == 2
    assert adapter.calls[1]["rows"] == 3 + 4 + 5
    assert metrics["forward_rows"] == 2 + 3 + 4 + 5


def test_a_zero_wait_still_drains_what_is_already_queued():
    """0 ms is the default and is not "coalescing off".

    The Python `CoalescingEvaluator` this shape was modelled on breaks its drain
    loop immediately at a zero deadline and therefore never merges anything. The
    Rust worker's zero-wait path is `try_recv`-only, which is a different
    contract: it takes everything already there, it just refuses to WAIT for
    more. At a queue measured ~3 deep that is the whole mechanism.
    """

    adapter = _Adapter(block_first=True)
    _answers, metrics = _run_blocking(adapter, [1, 1, 1, 1], wait_ms=0.0)

    assert metrics["forwards"] == 2
    # Nothing blocked to grow a batch: the width came from the standing queue.
    assert metrics["coalesce_wait_ns"] == 0


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------
def test_members_of_different_sizes_scatter_to_their_own_submitters():
    """Reply slicing by running offset, which is where an off-by-one hides.

    Each row's value encodes its position WITHIN THE FORWARD, so a request that
    receives its neighbour's rows returns the wrong numbers rather than the
    right count of wrong ones. Asserting row counts alone would pass a merge
    that shifted every request one slot along.
    """

    adapter = _Adapter(
        block_first=True,
        value_of=lambda call, row, net, rows: 0.001 * row,
    )
    answers, _metrics = _run_blocking(adapter, [1, 4, 2, 3])

    assert [answer[0] for answer in answers] == ["ok"] * 4
    # Request 0 was the held call: a forward of its own, so offset 0.
    assert answers[0][1] == pytest.approx([0.0])
    # Requests 1..3 shared the second forward, in receipt order.
    offset = 0
    for answer, count in zip(answers[1:], [4, 2, 3]):
        values, actors = answer[1], answer[2]
        assert len(values) == count
        expected = [
            0.001 * (offset + row) * (1 if actors[row] == 0 else -1)
            for row in range(count)
        ]
        assert values == pytest.approx(expected, abs=1e-9)
        offset += count


def test_replies_arrive_in_receipt_order():
    """A shard running `max_inflight_batches > 1` has several tickets open at
    once and reads them in submission order. Scattering out of receipt order
    would hand it another shard's answer with the right shape."""

    adapter = _Adapter(
        block_first=True,
        value_of=lambda call, row, net, rows: 0.001 * row,
    )
    answers, _metrics = _run_blocking(adapter, [1, 2, 2, 2])

    merged = [answer[1] for answer in answers[1:]]
    starts = []
    for values, answer in zip(merged, answers[1:]):
        actors = answer[2]
        starts.append(round(values[0] * (1 if actors[0] == 0 else -1) / 0.001))
    assert starts == sorted(starts)
    assert starts == [0, 2, 4]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def _nets_of(answer):
    """Recover the network each row was evaluated on from its returned value."""

    values, actors = answer[1], answer[2]
    return [
        round(value * (1 if actors[row] == 0 else -1) - 0.5)
        for row, value in enumerate(values)
    ]


@pytest.mark.parametrize(
    "request_nets,expected",
    [
        # Unrouted FIRST: routing is established by a later member, and the
        # rows already admitted have to be back-filled with zeros.
        ([None, None, [1, 1], [0, 1]], [0, 0, 1, 1, 0, 1]),
        # Unrouted LAST, which is the branch that actually extends an existing
        # id vector -- and the one a mutation test caught the first version of
        # this file failing to reach at all. Both orders, because they are
        # different code paths that fail the same silent way.
        ([None, [1, 1], None, [0, 1]], [1, 1, 0, 0, 0, 1]),
        ([None, [1, 0], [0, 1], None], [1, 0, 0, 1, 0, 0]),
    ],
)
def test_an_unrouted_request_merged_with_a_routed_one_keeps_network_zero(
    request_nets, expected
):
    """The trap the plan calls out: empty `net_ids` means "every row on network
    0", and concatenating a SHORT vector onto a long batch silently shifts rows
    onto the wrong network -- a league game evaluated by its opponent's net,
    which produces entirely plausible numbers.

    Asserts on WHICH network saw which row (each row's value is its own net id),
    never on row counts, which are identical under the bug.
    """

    adapter = _Adapter(
        block_first=True,
        value_of=lambda call, row, net, rows: 0.5 + net,
    )
    answers, _metrics = _run_blocking(
        adapter, [1, 2, 2, 2], request_nets=request_nets
    )

    # One forward carried all three merged requests, and the packer saw an id
    # for every row of it -- not just for the routed tail.
    merged = adapter.calls[1]
    assert merged["rows"] == 6
    assert merged["net_ids"] == expected

    # And each submitter got back the networks IT asked for.
    for answer, wanted in zip(answers[1:], request_nets[1:]):
        assert _nets_of(answer) == list(wanted or [0, 0])


def test_an_all_unrouted_merge_packs_no_routing_at_all():
    """The single-network case must not start paying for ids it does not need,
    and must stay byte-identical to the path W1's routing equivalence was
    verified against."""

    adapter = _Adapter(block_first=True)
    _answers, _metrics = _run_blocking(adapter, [2, 2, 2])

    # The packer writes zeros for an unrouted batch; what matters is that no
    # row acquired a non-zero id from a neighbour.
    assert set(adapter.calls[1]["net_ids"]) == {0}


# ---------------------------------------------------------------------------
# The cap
# ---------------------------------------------------------------------------
def test_the_row_cap_is_never_exceeded_and_the_held_request_is_served_next():
    """An `mpsc::Receiver` cannot un-receive. A request that would overflow
    `max_rows` has to be HELD and to head the next batch; dropping it would
    strand a submitter on a ticket nothing else completes."""

    adapter = _Adapter(block_first=True)
    answers, metrics = _run_blocking(adapter, [3, 4, 4, 4], max_rows=10)

    assert all(answer[0] == "ok" for answer in answers)
    assert all(call["rows"] <= 10 for call in adapter.calls)
    # 3 held | 4+4 merged (adding the third 4 would make 12) | 4 carried.
    assert [call["rows"] for call in adapter.calls] == [3, 8, 4]
    assert metrics["coalesce_carried"] == 1
    assert metrics["forward_rows"] == 15
    assert [len(answer[1]) for answer in answers] == [3, 4, 4, 4]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_an_evaluator_error_reaches_every_member_of_the_merged_batch():
    """One request's failure is all of their failures: they shared the forward
    that raised it. A member left waiting would hang its shard for the rest of
    the run."""

    adapter = _Adapter(block_first=True, fail_on_call=1)
    answers, _metrics = _run_blocking(adapter, [1, 2, 2, 2])

    assert answers[0][0] == "ok"
    assert [answer[0] for answer in answers[1:]] == ["err"] * 3
    for answer in answers[1:]:
        assert "adapter refused this batch" in answer[1]


def test_a_carried_over_request_is_answered_when_the_worker_fails():
    """The held-back request is not part of the failing batch and is the one
    thing an error path forgets: it is out of the channel, so the shutdown
    cannot re-drain it, and its submitter is already blocked on the ticket."""

    adapter = _Adapter(block_first=True, fail_on_call=1)
    answers, _metrics = _run_blocking(adapter, [3, 4, 4, 4], max_rows=10)

    assert answers[0][0] == "ok"
    # 4+4 merged and failed; the third 4 was being held for the next batch.
    assert [answer[0] for answer in answers[1:]] == ["err"] * 3
    assert "adapter refused this batch" in answers[3][1]


def test_a_request_over_the_cap_is_refused_at_submission():
    adapter = _Adapter()
    answers, _metrics = _probe(adapter, [12], max_rows=8)
    assert answers[0][0] == "err"


# ---------------------------------------------------------------------------
# Wait policy
# ---------------------------------------------------------------------------
def test_a_positive_wait_times_out_and_still_issues_its_partial_batch():
    """A wait must never turn into a hang when no more work is coming: the
    partial batch goes out at the deadline."""

    adapter = _Adapter()
    answers, metrics = _probe(adapter, [2], wait_ms=5.0)

    assert answers[0][0] == "ok"
    assert metrics["forwards"] == 1
    assert metrics["coalesce_wait_ns"] > 0


def test_a_wait_at_or_above_the_ticket_timeout_is_refused():
    """The wait is spent inside each ticket's deadline. A wait >= the timeout
    guarantees every ticket expires before its forward is issued -- a whole run
    failing slowly, with a message that points at the network."""

    import seven_wonders_rust as swr

    with pytest.raises(ValueError, match="inference_wait_ms"):
        swr._coalescer_probe(_Adapter(), _games(2), [1], [None], 8, 5.0, 5.0)


def test_a_negative_wait_is_refused():
    import seven_wonders_rust as swr

    with pytest.raises(ValueError, match="inference_wait_ms"):
        swr._coalescer_probe(_Adapter(), _games(2), [1], [None], 8, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------
def test_an_abandoned_ticket_does_not_wedge_the_loop():
    """Before coalescing, the worker broke its loop the moment a reply send
    failed -- and a failed send is an ABANDONED TICKET, whose owner timed out
    and walked away, not an error.

    That was survivable when one request meant one forward. Under merging it
    would strand every other member of the same batch, and then every request
    after it, on a worker that had quietly stopped. So the send failure is now
    ignored and the loop continues.

    The abandonment is modelled by dropping the ticket rather than by timing one
    out: `inference_timeout_ms` is shared, so a stopwatch version would expire
    the whole batch and prove nothing about the survivors.
    """

    adapter = _Adapter(block_first=True)
    answers, metrics = _run_blocking(adapter, [1, 1, 1], drop_tickets=[0])

    # Request 0 was walked away from; 1 and 2 merged behind it and must still be
    # served. On the pre-coalescer path they came back "worker dropped its
    # response" instead.
    assert answers[0] == ("err", "abandoned")
    assert [answer[0] for answer in answers[1:]] == ["ok", "ok"]
    assert metrics["worker_requests"] == 3
    assert metrics["forwards"] == 2


# ---------------------------------------------------------------------------
# End to end, through the real scheduler
# ---------------------------------------------------------------------------
def _deterministic_adapter(payload):
    """A pure function of each row.

    The point is that no row's answer can depend on its neighbours in a batch,
    so any difference in the records below is scheduling or slicing, never
    arithmetic. A real net on CUDA cannot give this guarantee -- batch shape
    moves float reductions -- which is why the strict-equality gate runs here
    and the real-net path gets a fingerprint gate instead.
    """

    import hashlib

    rows = int(payload["rows"])
    offsets = struct.unpack(
        f"<{len(payload['legal_offsets']) // 4}I", bytes(payload["legal_offsets"])
    )
    actions = struct.unpack(
        f"<{len(payload['legal_actions']) // 2}H", bytes(payload["legal_actions"])
    )
    token_offsets = struct.unpack(
        f"<{len(payload['token_offsets']) // 4}I", bytes(payload["token_offsets"])
    )
    type_ids = bytes(payload["type_ids"])
    out = []
    for row in range(rows):
        digest = hashlib.blake2b(
            type_ids[token_offsets[row] : token_offsets[row + 1]], digest_size=8
        ).digest()
        seed = int.from_bytes(digest, "little")
        legal = actions[offsets[row] : offsets[row + 1]]
        priors = [((seed >> (action % 40)) % 97 + 1) / 100.0 for action in legal]
        total = sum(priors)
        out.append((((seed % 2001) - 1000) / 1000.0, [p / total for p in priors]))
    return out


def _self_play(workers, wait_ms):
    import seven_wonders_rust as swr

    from .control_table import ensure_rust_table

    ensure_rust_table()
    seeds = [2026090950 + index for index in range(8)]
    return swr.self_play_many_flat_net(
        adapter=_deterministic_adapter,
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
        inference_wait_ms=wait_ms,
    )


def _trajectories(records):
    return [
        (
            move.get("state_digest"),
            move.get("action"),
            tuple(move.get("visits") or ()),
            tuple(move.get("policy") or ()),
            move.get("root_value"),
        )
        for record in records
        for move in record["moves"]
    ]


@pytest.mark.parametrize("workers,wait_ms", [(1, 2.0), (4, 0.0), (4, 2.0)])
def test_batch_composition_does_not_change_what_the_search_produces(workers, wait_ms):
    """Merging must be invisible to the trajectory.

    Coalescing changes which rows travel together, and nothing else. Under an
    evaluator whose answer depends only on the row, that has to mean identical
    digests, actions, visit counts AND float targets -- not "close".

    On the real net path this is a fingerprint contract instead (actions,
    digests, visits only): batch shape moves float reductions on CUDA by ~1e-5,
    the same contract `--rust-global-batch-cap` already lives under.
    """

    baseline, _ = _self_play(workers=1, wait_ms=0.0)
    records, _ = _self_play(workers=workers, wait_ms=wait_ms)
    assert _trajectories(records) == _trajectories(baseline)


def test_the_coalescer_recovers_the_width_that_sharding_fragments():
    """The measurement this whole build exists for.

    `test_shards_fragment_batches_rather_than_pooling_them` pins the damage:
    the same work arrives as more, narrower requests as shards rise. This pins
    the repair, on the same workload, at the only place that can show it --
    forwards the WORKER issued, not requests the scheduler submitted.
    """

    _one, one = _self_play(workers=1, wait_ms=0.0)
    _many, many = _self_play(workers=4, wait_ms=2.0)

    # Same games, same seeds, same search budget: the rows are the work.
    assert one["global_rows"] == many["global_rows"]
    # Sharding still fragments SUBMISSION -- that is upstream of the worker and
    # unchanged. If this stops holding, the test below proves nothing.
    assert many["global_batches"] > one["global_batches"] * 2
    # ... and the worker puts it back together.
    assert many["worker_requests"] > many["boundary_forwards"]
    assert many["worker_requests"] / many["boundary_forwards"] > 2.0
    # Width recovered to the unsharded baseline rather than merely improved.
    assert many["boundary_forwards"] <= one["boundary_forwards"]
    width = lambda m: m["boundary_forward_rows"] / m["boundary_forwards"]
    assert width(many) >= width(one) * 0.9
    # A positive wait is what bought the last of it, and it is not free.
    assert many["coalesce_wait_ns"] > 0


def test_a_single_shard_has_nothing_to_merge_and_says_so():
    """One shard means one submitter, so the ratio is 1.00 by construction. A
    coalescer that reported a merge here would be counting something else."""

    _records, metrics = _self_play(workers=1, wait_ms=0.0)
    assert metrics["worker_requests"] == metrics["boundary_forwards"]
    assert metrics["worker_requests"] == metrics["global_batches"]
