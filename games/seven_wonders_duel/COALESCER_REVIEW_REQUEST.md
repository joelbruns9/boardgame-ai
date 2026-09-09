# Review request: evaluator coalescing

**What this is.** `COALESCER_BUILD_PLAN.md` §3–§5 implemented, on top of the
Phase 0 instrumentation repair that shipped separately. The plan is unchanged
and is still the specification; this says what was built, where it departs from
the plan, what is measured versus assumed, and where I think a reviewer will
find something.

**Reviewed 2026-09-09**: four findings, all reproduced, all fixed, plus two
pre-box items. **One of them invalidated a claim this document made** — the
identity gate was not testing the digests or the policy targets it said it was.
See §9 for the audit trail. Sections 2–8 describe the code as it now stands.

**Status: built, gated, UNRUN ON A BOX.** The mechanism engages and is
bit-safe. Whether it buys throughput is not known and cannot be inferred from
anything here — see §2.

**Default behaviour changes.** Coalescing is on by default;
`--rust-inference-wait-ms` only controls whether the worker additionally
*blocks* to widen a batch. `--no-rust-coalesce` restores one request per
forward, for a same-box A/B (§4.1).

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

Bit-identity across every composition above: trajectory and final digests,
actions, visit counts and float policy targets, by strict equality on
required-key access. `test_the_identity_comparison_can_actually_fail`
perturbs each of those fields in turn and requires the comparison to notice —
because the first version of this gate read two field names that do not exist
and passed on any input at all (§9, R1).

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

### 4.1 There is an off switch — I argued against one and was wrong

An earlier revision of this document defended having no off switch, on the
grounds that a second path through the worker is a second path to keep correct.

That was wrong on both halves. The comparison is **necessary**: without it the
only available baseline is cloud2's recorded numbers, at a different geometry
on different hardware, which cannot attribute a throughput change to this
implementation. And the cost I invoked was imaginary — `--no-rust-coalesce` is
one branch at the top of the drain loop, not a second path. The batch, the
scatter, the carry-over and the error fan-out are the same code, just never
given a second member.

`test_coalescing_off_restores_one_request_per_forward` pins the arm, and
`test_the_off_arm_produces_the_same_trajectories_as_the_on_arm` pins that the
two arms differ in timing and nothing else. A wait with coalescing off is
refused: latency on every forward, width on none.

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
the runs this project is about to do.**

The count now reaches production: `_generate_iteration_rust` publishes the
adapter's counters as `rust_boundary`, `GenerationStats` carries
`model_forwards` and `model_forwards_per_forward`, the sweep reports a median
per point, and the heartbeat appends `net=<split>` when a routed model split
merged batches apart. `test_a_routed_batch_costs_one_model_forward_per_network
_present` drives the real league adapter over two nets and requires
`forwards > calls` — it fails rather than passing vacuously if no batch mixed
networks.

It was previously computed and thrown away (§9, R3): the adapter was local to
`_generate_iteration_rust`, which kept only the Rust scheduler dict.

---

## 7. Where I would attack this first

1. **The cap became reachable.** Batches previously ran ~47 rows; they can now
   reach `global_batch_cap` (2,048). Peak boundary memory rises to what that cap
   always implied but never spent — pack buffers, eleven `PyByteArray` copies,
   H2D, and the tensor build, all at up to 43× the rows. §5.5 stages a memory
   check on the box. I have not run one.

2. **Token padding is now worse, not better.** `padded_tokens = rows ×
   max_tokens`: one long row pads every row it travels with, so a wider batch
   pads harder. Cloud2 measured 26.6% of slots wasted, already above the 20%
   bucketing trigger. Coalescing raises the value of fixing that and I have not
   touched it.

3. **A failed reply send no longer stops the worker.** Before, it broke the
   loop. It means an abandoned ticket, and under merging that would strand every
   other member of the batch. But it also means a genuinely wedged reply channel
   is now silent. I think that is right — the ticket owner already got a timeout
   error — and I would like it checked.

4. **`queue_wait_ns` changed meaning slightly.** It was enqueue → worker pickup.
   It is now enqueue → admission into a batch, summed over members. Both are
   "queue wait"; the second is larger under merging, and anything comparing it
   across this change is comparing two things.

5. **`max_inflight_batches` must be retested and its existing null discarded.**
   A serial consumer could not use queue depth, so that measurement was
   structurally incapable of a non-null result. `training_parameters.md` now
   says so at the flag.

6. **The wait interacts with the ticket deadline.** It is spent *inside*
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

---

## 9. Response to the 2026-09-09 review

Four findings. All reproduced against the code, all fixed. Two pre-box items,
both addressed. Nothing was disputed.

### R1 — the identity gate compared nothing *(P1, fixed)*

**Correct, and it invalidated a claim I made in this document and in the commit
message.** `_trajectories` read `move["state_digest"]` and `move["policy"]` with
`.get()`. Neither key exists: the digests are RECORD level
(`trajectory_digest`, `final_digest`) and the target is `policy_target`. So the
helper compared `None` against `None` and `()` against `()`.

Reproduced by dumping the real key names. The claim "bit-identical including
digests and float targets" was **not established** by that test — what it
actually compared was actions, visits and `root_value`.

Fixed: required-key access over named field tuples, record boundaries preserved
(a flattened sequence cannot see a move moved between games), and a new
`test_the_identity_comparison_can_actually_fail` that perturbs every covered
field and requires the comparison to notice.

**The claim itself survives** — the composition tests still pass against the
real fields — but it survived on evidence that did not exist until now, which
is not the same thing, and I should not have written it.

### R2 — the measured wait was dropped when held constant *(P2, fixed)*

Correct. I had copied the `len(splits) > 1` rule from `SOLVER_THREADS` without
noticing the rule depends on the **fallback**. An unset `SOLVER_THREADS` is
derived at stage 6b from the box's cores, which is better than a pin. An unset
`RUST_INFERENCE_WAIT_MS` is 0, or a stale environment value — neither of which
is what was measured. A confirmation sweep pinned at 2 ms therefore handed
production a 0 ms run while every other number described a 2 ms geometry.

Fixed: emitted whenever the winning row carries one. `render` says explicitly
whether the sweep varied it, so "this is a winner" and "this is what the
measured points ran at" are not confused.

### R3 — the model-forward counter never left the adapter *(P2, fixed)*

Correct, and the worst kind of gap: §6 of this document called the number
essential and it was unreachable. `_generate_iteration_rust` built the adapter
locally, kept `dict(metrics)` — the Rust scheduler only — and dropped it.

Fixed end to end: `rust_boundary` in the generation stats, two fields on
`GenerationStats`, a median per sweep point, `net=` in the heartbeat when the
split exceeds 1.0, and two tests — one driving the real routed league adapter,
one asserting the value reaches `_record_stats`.

### R4 — the baseline ignored the new axis *(P2, fixed)*

Correct. `key` covered four of six axes while `summary` is sorted fastest-first,
so `next()` returned the fastest row matching a partial key: **the baseline
became the winner**, and the axis the key forgot reported 1.00x. The wait
triggered it; the solver axis was already exposed.

Fixed: the key covers all six axes, `run_baseline` snapshots all six from the
run config (rather than re-parsing the manifest — the config already holds
`solver_threads` and `rust_inference_wait_ms`), and the label names each one.

### Pre-box: games against slots *(addressed)*

Correct and worse than a default problem. A point cannot hold more games live
than it is given, so `--games 200` against a 512-slot point measured **200 slots
wearing a 512 label**. The sweep now refuses `games < max_slots` outright and
warns below 3 games per slot; `setup_cloud_7wd.sh` derives the default from
`SWEEP_SLOTS_CSV` so raising the slot list cannot silently reintroduce it.

### Pre-box: a same-box baseline *(addressed)*

Taken, and see §4.1 — the off switch is in. Benchmarking the parent commit
would also work, but it needs a second `cargo build` and a `.pyd` swap mid-run
on rented time, which is a worse thing to be doing under a meter.

**Still true and still the point:** none of this measures games per hour.
