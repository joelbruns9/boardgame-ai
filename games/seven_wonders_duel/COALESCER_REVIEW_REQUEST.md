# Review request: evaluator coalescing

**What this is.** `COALESCER_BUILD_PLAN.md` §3–§5 implemented, on top of the
Phase 0 instrumentation repair that shipped separately. The plan is unchanged
and is still the specification; this says what was built, where it departs from
the plan, what is measured versus assumed, and where I think a reviewer will
find something.

**Status: built, gated, UNRUN ON A BOX.** The mechanism engages and is
bit-safe. Whether it buys throughput is not known and cannot be inferred from
anything here — see §2.

**Default behaviour changes.** This is not an opt-in feature. Coalescing is
always on; `--rust-inference-wait-ms` only controls whether the worker
additionally *blocks* to widen a batch. Any run on this branch gets merged
forwards whether or not it passes the flag. That is deliberate (§4.1) and is
the single most important thing to disagree with me about if you are going to.

---

## 1. What to read, in order

| File | What it holds |
|---|---|
| `seven_wonders_rust/src/eval.rs` | **The whole mechanism.** `CoalescedBatch` (merge, scatter, error fan-out) and the drain loop in `spawn_py_flat_worker` |
| `test_coalescer.py` | The gates. Read `_run_blocking` first — the synchronisation is the reason these assert anything |
| `seven_wonders_rust/src/self_play.rs` | Three counters, through the compile-time `merge` guard |
| `rust_bridge.py` | `model_forwards` — the caveat that keeps the headline number honest |
| `f4_phase_d_sweep.py`, `sweep_launch_env.py` | The wait as a swept axis, and the refusal to emit one that was not swept |
| `COALESCER_BUILD_PLAN.md` §9 | The measurements, written against the plan's own success criteria |

---

## 2. What is measured, what is assumed, what is not done

### Measured

8 games through the real scheduler, deterministic evaluator, identical seeds
and search budget, on the laptop:

| shards | wait | requests | forwards | req/forward |
|---|---|---|---|---|
| 1 | 0 ms | 401 | 401 | 1.00 |
| 4 | 0 ms | 1,521 | 785 | 1.94 |
| 4 | 2 ms | 1,521 | **401** | **3.79** |

At 4 shards and 2 ms the forward count lands *exactly* on the single-shard
count. The fragmentation `test_shards_fragment_batches_rather_than_pooling_them`
pins is fully recovered, not merely reduced.

Bit-identity across every composition above: digests, actions, visit counts and
float targets, by strict equality.

### Not measured, and not inferable from the above

**Games per hour.** Every number in this document is batch WIDTH. Width is the
mechanism; throughput is the prize, and the plan's §8 is explicit that a width
gain with no throughput gain means the bottleneck is not where the plan says.

The 91.8% `py_call_ns` share that motivates the whole build is **cloud2's**, from
the iter85 run. It is not a laptop number and I did not re-derive it here. The
laptop's evaluator is a 32-wide, 2-layer toy net on CPU; its cost curve is not
the box's.

### Assumed

That merging is worth it *because* the per-call cost is near-fixed at these
widths. That is cloud2's measurement (5.19 ms per forward carrying 46.9 rows
against a 2,048 cap), and it is the assumption the whole design rests on. If the
box's per-call cost turns out to scale with rows, coalescing trades latency for
nothing and should be reverted rather than tuned.

---

## 3. The mechanism

```
loop {
    first = carried.take() or request_rx.recv()   // shutdown when recv fails
    batch = [first]
    deadline = wait.map(|w| now + w)
    loop {
        if batch.rows() >= max_rows { break }
        next = match deadline {
            Some(d) => recv_timeout(d - now)  or, past d, try_recv
            None    => try_recv
        }
        if next is None { break }
        if batch.rows() + next.rows > max_rows { carried = next; break }
        batch.push(next)
    }
    one forward; scatter in receipt order; carried heads the next batch
}
```

Four things about it are load-bearing and are each gated:

**Reply slicing by running offset.** Each member's row count is recorded at
push; `scatter` consumes the forward's rows in receipt order. The gate asserts
on *values* that encode each row's position within the forward, not on row
counts — a merge that shifted every member one slot along returns the right
count of wrong numbers.

**`net_ids` expansion.** Empty means "every row on network 0". As soon as one
member routes, previously-admitted rows are back-filled with explicit zeros and
later unrouted members are extended with them. Concatenating a short vector
would put rows on the wrong network, and a league game evaluated by its
opponent's net produces entirely plausible numbers. The gate asserts on *which
network saw which row*.

**Carry-over.** An `mpsc::Receiver` cannot un-receive. An over-cap request is
held, and it is answered — not stranded — when the worker fails.

**Receipt order.** A shard at `max_inflight_batches > 1` holds several tickets
and reads them in submission order.

---

## 4. Where I departed from the plan

### 4.1 There is no off switch, and I think there should not be

The plan does not ask for one, but a reviewer reasonably might: an A/B against
the pre-coalescer path would be the cleanest way to attribute a throughput
change on the box.

I did not add one, for two reasons. Coalescing at zero wait has no cost to
switch off — it does not block, it drains a queue that already exists — so an
"off" mode would exist only to reproduce a defect. And a second code path
through the worker is a second path to keep correct; the value-leak contract in
W7 is a fresh reminder of what a rarely-taken branch costs.

**The consequence is real and I want it named:** there is no way to measure the
coalescer against itself on the box. The comparison available is against
cloud2's recorded numbers, at a different geometry, with the caveats in §2. If
you think that is not good enough, the cheap fix is a `--rust-inference-wait-ms
-1` sentinel meaning "one request per forward", and it is about ten lines.

### 4.2 `phase_d.py` gets a new flag rather than reusing `inference_wait_ms`

The plan's fold-in table says "pass `--inference-wait-ms` to the Rust path".
`inference_wait_ms` already exists, defaults to **2.0**, and belongs to the
Python `CoalescingEvaluator`, where it means something else: that one waits a
deadline out from an empty queue. Reusing it would have made 2 ms a live
default on the Rust path that nobody measured — precisely what §3.2 says not to
do. New field, new flag, default 0.

### 4.3 The gate path stays at 0

The wait is passed at the generation call site only. The gate runs a different
shard count and arrival pattern, and a wait swept on generation was not measured
on it. It still coalesces, at zero wait, like everything else.

### 4.4 A test-only pyfunction

`_coalescer_probe` in `lib.rs` is exported into the module and exists solely for
`test_coalescer.py`. I disliked adding it and could not do without it — see §5.

---

## 5. The gates, and why they are shaped this way

### The synchronisation is the point

The first version of `test_queued_requests_merge_into_one_forward` submitted
four requests and asserted two forwards. It failed with **one** forward: all
four requests had landed before the worker's first adapter call, because
submission is a few clones and a channel send while the worker must acquire the
GIL first.

That is a race, and it resolves the flattering way. A test written that way
passes whatever the drain loop does — including a broken carry-over, or a cap
that is never enforced, because everything merges into one batch anyway.

So `_coalescer_probe` takes a `gate` callback invoked after the first
submission, which blocks (on a `threading.Event`, which releases the GIL) until
the adapter has actually entered its first call. Which requests share which
forward is now a decision, not an observation.

**This is also why the gates could not go through `self_play_many_flat_net`.**
That path decides its own request shapes from search state; "two shards happened
to overlap" is not something a test can assert.

### The mutation that survived

Three mutations were introduced. Each was caught:

| mutation | caught by |
|---|---|
| over-cap request dropped instead of carried | the cap gate |
| replies scattered in reverse receipt order | the scatter and order gates |
| unrouted member not expanded to explicit zeros | the routing gates — **but only after I fixed them** |

The third initially **passed**. My routing test had the unrouted request
arriving *before* the routed one, which reaches the "back-fill what is already
admitted" branch and never the "extend with zeros" branch. Both orders are now
parametrised, plus a third where the unrouted request is last.

I record this because the trap is spelled out in the plan, in a section I wrote,
and the test I wrote against it still missed half of it. A gate written against
a named trap is not evidence it is covered.

---

## 6. The caveat that keeps the headline honest

`requests_per_forward` counts calls across the Rust boundary. A routed model
(`_SearcherRoutedModel`) runs **one forward per network present in the batch**,
so merging two shards that sit on different nets saves the boundary hop, the
pack and the H2D copy — and saves no GPU forward at all.

That is exactly the league configuration where the merge is widest. So
`rust_bridge.py` now counts `model_forwards` separately from adapter calls.

**Reporting `requests_per_forward` alone would overstate the win on precisely
the runs this project is about to do.** I have not measured how much: the
laptop's routed path was not exercised at width here. That is a gap.

---

## 7. Where I would attack this first

1. **§4.1.** The absence of an off switch is the decision I am least sure of and
   the one with the largest consequence for the box.

2. **The cap became reachable.** Batches previously ran ~47 rows; they can now
   reach `global_batch_cap` (2,048). Peak boundary memory rises to what that cap
   always implied but never spent — pack buffers, eleven `PyByteArray` copies,
   H2D, and the tensor build, all at up to 43× the rows. §5.5 stages a memory
   check on the box. I have not run one.

3. **Token padding is now worse, not better.** `padded_tokens = rows ×
   max_tokens`: one long row pads every row it travels with, so a wider batch
   pads harder. Cloud2 measured 26.6% of slots wasted, already above the 20%
   bucketing trigger. Coalescing raises the value of fixing that and I have not
   touched it.

4. **A failed reply send no longer stops the worker.** Before, it broke the
   loop. It means an abandoned ticket, and under merging that would strand every
   other member of the batch. But it also means a genuinely wedged reply channel
   is now silent. I think that is right — the ticket owner already got a timeout
   error — and I would like it checked.

5. **`queue_wait_ns` changed meaning slightly.** It was enqueue → worker pickup.
   It is now enqueue → admission into a batch, summed over members. Both are
   "queue wait"; the second is larger under merging, and anything comparing it
   across this change is comparing two things.

6. **`max_inflight_batches` must be retested and its existing null discarded.**
   A serial consumer could not use queue depth, so that measurement was
   structurally incapable of a non-null result. `training_parameters.md` now
   says so at the flag.

7. **The wait interacts with the ticket deadline.** It is spent *inside*
   `inference_timeout_ms`. `wait >= timeout` is refused at startup, but a wait
   close to it eats the margin, and I have not thought hard about what fraction
   is safe under a loaded box.

---

## 8. What did not change

* Single-shard behaviour: one submitter, ratio 1.00 by construction, and a gate
  asserts it rather than leaving it to be assumed.
* `global_batches` / `global_rows`, which still count what the scheduler
  submitted and are still structurally blind to the merge. That is the Phase 0
  distinction and it is now load-bearing.
* Trajectories, targets and digests, by strict equality on the deterministic
  path.
* `spawn_py_batch_worker` (the non-flat path) is untouched.
