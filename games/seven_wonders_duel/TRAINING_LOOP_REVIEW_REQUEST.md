# Review request — 7WD training loop: pilot results and the example cache

Companion to `THROUGHPUT_ACTION_PLAN.md` (generation) and
`sevenwd_training_run_prep` notes. That work made generation 3.7× faster; this
document covers what the first real loop run then revealed, and the one change
made in response.

**Headline:** a 12-iteration pilot confirmed the training configuration learns —
policy top-1 0.356 → 0.577, both promotion gates accepted decisively — but
iterations ran 9.6–16.4 min against a predicted 7, and the excess was neither
generation nor training. It was **rebuilding the replay window's examples every
iteration**: 309 s of a 580 s iteration at the end of the pilot, growing with the
buffer. An in-memory per-game example cache takes that to 38 s.

**Rev 2 (2026-07-27), after review.** Two blocking defects in the first version —
a cache key that could serve one record's examples for another *and* let a
tampered record skip its own verification, and a capacity cap that did nothing in
the case it existed for. Both are fixed, both were reproduced first, and neither
was catchable by the rev-1 suite. See §10.

Review effort is best spent on §4 (the cache's correctness argument) and §6
(what I chose not to do).

## 1. Scope

| What | Commit | Files |
|---|---|---|
| Free-axis sweep + per-game bot routing | `8ce5e5e` | (already reviewed in `THROUGHPUT_REVIEW_REQUEST.md` §1) |
| 12-iteration pilot | — | `runs/laptop_pilot_20260727` (gitignored) |
| **In-memory example cache** | uncommitted | `phase_d.py`, `test_phase_d_example_cache.py` (new, 12 tests) |

Suite: 539 Python + 16 cargo, green.

## 2. What the pilot measured

Config: 400 games/iteration, `train_steps 76`, `warmup 25`, `rust_slots 48`,
`promotion_every 5`, `gate_max_games 200`, `gate_indifference 0.05`, soft gate,
1,000 seed games, anneal 6, d128 L4, RTX 3070 laptop.

**The thing under test — does ~5× sample reuse learn?** Yes.

| | iter 0 | iter 5 | iter 11 |
|---|---|---|---|
| policy top-1 | 0.356 | 0.577 | 0.571 |
| value acc | 0.598 | 0.681 | 0.673 |
| policy loss | 1.519 | 1.093 | 1.084 |

`samples_per_new_position` came in at 4.44 → 5.62, so the sizing rule
(`train_steps ≈ 0.19 × games_per_iteration`, from a measured 19.4 recorded
positions per game) held in a real loop.

**Both gates promoted**: iteration 5 accepted at 1.000 over `best_iter_0` (16
games), iteration 10 accepted at 0.944 over `best_iter_5` (18 games). Both
stopped early, so each cost ~30 s rather than the 6.2 min worst case budgeted.

Two observations worth recording:

* **Validation plateaus while strength does not.** Val metrics flatten after
  iteration 5, yet iteration 10's candidate beat a *freshly banked* best at
  0.944. The dissociation matches the comment in `train_steps` ("validation
  stays diagnostic until arena games show it predicts strength"). Consequence:
  validation curves cannot serve as the comparison signal between runs; the gate
  can.
* **Timing missed, and not where expected.** Generation 5.8 min (predicted 5.0),
  training **11 s** (predicted 48 s), gate ~0.1 min amortised (predicted 1.2).
  Everything budgeted came in at or under. The unbudgeted item grew 3.3 → 9.4
  min across the pilot.

## 3. The problem

`train_candidate` called `examples_from_records(records, ...)`, which replays
**every game in the replay window** through the verified engine path — mask
hashes, actors, chance log, trajectory and final digests — every iteration.
Buffer files are immutable once written, so this re-verifies data that cannot
have changed. Measured on the pilot's own buffers:

| window | JSONL read | example build |
|---|---|---|
| 1,633 games | 1.0 s | 223 s |
| 3,600 games | 2.8 s | 309 s |
| 4,800 games | 3.8 s | 404 s |

Reading is free; replay is the cost, at ~4 ms/example. Projected to the real
run's steady state (window 20 × 400 games ≈ 155k examples) that is **~12 min per
iteration**, roughly double generation.

Useful algebra: total reconstruction over a run is `total_games × replay_window`,
**independent of `games_per_iteration`**. No retuning of games or steps helps;
the only non-caching lever is the replay window, which is a learning decision.

**Kingdomino does not have this problem by construction.** Its `ReplayBuffer` is
a fixed-capacity ring of already-encoded `Example` objects held in RAM for the
whole run; self-play encodes once and `add()` appends. Examples are derived
exactly once, ever. 7WD instead stores compact, verifiable game *records* and
re-derives — a deliberate trade that bought resumability, human-readable
buffers, and tamper detection, and that costs O(window) replay per iteration.

## 4. What changed

`PhaseDLoop._cached_examples(records)` replaces the `examples_from_records` call
**inside `train_candidate`**, so both callers — the direct loop and the soft-gate
controller via `training_adapter` — are covered with no signature change.

* **Key**: `(sha256(to_json_line(record)), record_fast_moves)` — a hash of the
  record's own canonical serialization, the same function that wrote it, so the
  key covers every field the examples depend on and every field replay verifies.
  (Rev 2. The first version keyed on `(trajectory_digest, seed, iteration,
  record_fast_moves)` and was wrong twice over; see §10.1.)
* **Order**: examples are emitted per record in record order, so the flattened
  list is identical to `examples_from_records` — which matters because the
  training sampler draws from a seeded RNG over indices.
* **Eviction**: LRU bounded by `--example-cache-examples` (default 250,000
  ≈ 3.1 GB at a measured 12.5 KB/example), enforced unconditionally — including
  against games in the current window, since `out` already holds every reference
  the training call needs. A warning fires when the cap is below one window,
  which is the state where every iteration re-replays everything. (Rev 2; the
  first version refused to evict below the current window, which meant the cap
  did nothing at all in the case that mattered. See §10.2.)
* **Stats**: `example_cache` in the training row reports `replayed_games` per
  iteration — well above one iteration's worth means the cap is thrashing.

Verification is preserved, moved from once per iteration to once per process; a
restart re-derives from the records on disk.

**Measured effect** (pilot buffers, consecutive iterations on one loop):

| iteration | window | example build | replayed |
|---|---|---|---|
| 8 (cold) | 3,600 games | 321.1 s | 3,600 games |
| 9 | 4,000 | 37.9 s | 400 |
| 10 | 4,400 | 38.7 s | 400 |
| 11 | 4,800 | 37.9 s | 400 |

**8.2×** at steady state, and the residual 38 s is the new iteration's 400 games,
which genuinely must be replayed, plus ~4 s of whole-record hashing over the
window (measured 0.74 ms/game). The rev-1 key was 5 s faster and unsound; the
hashing is ~1% of an iteration and buys §10.1.

## 5. Gates (`test_phase_d_example_cache.py`, 12 tests)

The load-bearing one is **equivalence**: `_cached_examples` vs
`examples_from_records`, cold and warm, compared field by field with
`np.array_equal` on the arrays and `==` on the scalars. Then:

* a repeated game is replayed once but still emitted twice (dedup must not
  collapse the training set);
* **the same trajectory with different policy targets does not share an entry** —
  the rev-1 collision, with a precondition asserting the trajectory digest is
  blind to the change;
* **altering a verified field misses the cache and raises `ReplayMismatchError`**
  — tampering must not be able to skip its own check;
* `record_fast_moves` is part of the key (the two settings emit different sets);
* the `iteration` label is part of the key — warm-buffer imports can carry the
  same game under a different label, and it drives the temporal split;
* **capacity is enforced even when one window exceeds it**, and does not change
  what is returned;
* **an oversized window warns on the first call**;
* a cap below one window still returns every example — thrashing must be a
  performance failure, never a correctness one;
* `0` restores the previous behaviour exactly;
* cached objects are shared **and cannot be mutated** (see §6a).

Both rev-1 defects were reproduced directly before fixing: the old key collided
on a retargeted move *and* on a tampered `mask_hash`, and the old eviction loop
left a 600-example window fully cached under a 200-example cap, evicting nothing.

The fixture generates its own self-play games. Reading a previous run's buffer
would make the file silently skip on a clean checkout (`runs/` is gitignored),
and it needs *searched* moves, since a curriculum bot's moves record no
simulations and are never classified fast.

## 6. Focus areas — where I think the risk is

**(a) Aliasing — now enforced rather than checked** (rev 2, §10.3). `Example` is
`@dataclass(frozen=True, slots=True)` and `__post_init__` marks all six arrays
`write=False`, because freezing the fields alone would still leave
`example.features[0, 0] = 1.0` working. Deriving labels no longer mutates: the
inputs are staged and the Example is constructed once, complete, after the replay
finishes. One collate line took a no-copy view of `policy_target` and now
`.astype`-copies like its neighbours.

**(b) The default cap is a guess about the wrong machine.** 250,000 examples
≈ 3.1 GB suits this laptop (5–6 GB free) and the intended
`replay_window 20 × 400 games` + 1,000 seed games ≈ 212k. It is not derived from
config, so a larger window silently thrashes — detectable only by reading
`replayed_games` in the training row or noticing the printed warning. A derived
default (`replay_window × games_per_iteration × 20 + seed_games × 60`) would be
self-adjusting. I chose the fixed number for predictability; happy to be argued
out of it.

**(c) Memory reporting is by example count, not bytes.** 12.5 KB/example is a
measurement, not an invariant — it is dominated by `features [T, 130]` where T
averages 47, so a change to the encoder's token count or `MAX_FEATURES` moves it
without moving the cap. A byte-based cap would need per-example accounting;
I judged the count sufficient given the encoder is version-pinned, but this is
the assumption I would most like a second opinion on.

**(d) I did not use ragged feature storage.** Measured, only **12.8%** of the
`features` cells carry information — the rest is structural zero padding that the
fused embedder's single wide matmul relies on. Ragged storage would cut an
example from 12.5 KB to ~1.8 KB (the 3.1 GB cap would become ~450 MB), but
`collate` currently fills each example with one contiguous slice assignment, and
ragged storage turns that into a per-token scatter — roughly 24,000 small writes
per batch instead of 512. Against an 11 s training step that is a plausible
10–20% regression to save memory the machine has spare, and it would introduce a
second layout for the same tensor. Declined; recorded here because it is the
obvious optimisation and its absence should be a decision, not an oversight.

## 7. Known limitations / not done

* **The cache is per process.** A resumed run pays the cold cost once (~310 s at
  the pilot's window, ~690 s projected at the real run's). Persisting it to disk
  would remove that, at the price of a second on-disk representation of data the
  records already contain.
* **`training_records` still re-reads and re-parses every JSONL each iteration**
  (3.8 s at 4,800 games). Cheap relative to replay, and it is what feeds
  `summarize_records` and the warmup check, so it was left alone.
* **The pilot was not re-run with the cache.** The 33 s figure is measured on the
  pilot's real buffers through the real code path, but the end-to-end
  ~7 min/iteration claim is arithmetic over measured parts, not an observed run.
* **`--example-cache-examples 0`** is kept as an escape hatch and is gated, but
  nothing in the loop selects it automatically under memory pressure.

## 8. Sign-offs — status after review 1

| # | Request | Verdict | Now |
|---|---|---|---|
| 1 | Key cannot serve one game's examples for another | **No** | Rekeyed on the whole serialized record; both collisions reproduced then gated. **Re-requested.** |
| 2 | Equivalence genuinely established | Order yes; equivalence only for identical records | Collision cases added (§5). **Re-requested.** |
| 3 | Aliasing acceptable as a checked invariant | **No — enforce it** | Accepted: `Example` frozen, arrays read-only, derivation restructured. **Re-requested.** |
| 4 | Fixed default cap acceptable | **Yes**, prefer a fixed hard cap with a derived required-size warning | Kept fixed; the warning now reports the required size and its GB cost. Agreed. |
| 5 | Declining ragged storage | **Yes** | No change. |

## 9. Running the gates

```bash
python -m pytest games/seven_wonders_duel/test_phase_d_example_cache.py \
  games/seven_wonders_duel/test_phase_d.py -q            # ~2 min
python -m pytest games/seven_wonders_duel -q             # full, ~9 min
cd games/seven_wonders_duel/seven_wonders_rust && cargo test
```

## 10. Review response (2026-07-27)

All three findings accepted; both blocking defects were **reproduced before being
fixed**, and neither was catchable by the rev-1 suite — the reviewer was right
about that too.

### 10.1 [High] Key omitted example-determining fields — accepted, rekeyed

Confirmed on both counts. `trajectory_digest` chains replayed *states*, so it is
identical for two records of the same played game with different
`policy_target` / `visits` / `sims` / `policy_excluded` — precisely what
reanalysis and warm-buffer imports produce, and those fields decide both whether
a move becomes an example (`is_fast_search_move`) and what its label is.
Reproduced: a record with all policy mass moved to a different legal action
yielded an identical rev-1 key.

The second half was worse and I had claimed the opposite in §4. The digest is a
*stored field*, so a record whose `mask_hash` was altered without updating it
also produced an identical rev-1 key — a cache hit that skips the replay whose
whole purpose is catching that alteration. Reproduced likewise. The "tamper
detection is preserved" claim was false as written.

Now keyed on `sha256(to_json_line(record))`, reusing the canonical serializer
that wrote the file, so every field is covered. Cost measured at 0.74 ms/game,
~4 s over a 4,800-game window, ~1% of an iteration; steady-state build went
32.3 s → 37.9 s. Two tests added, one per collision.

### 10.2 [High] Capacity was not a cap — accepted, fixed

Confirmed exactly as described: the `len(cache) > len(used)` guard is false on a
cold call, so a window larger than the cap was retained in full and the cap did
nothing in the only case where it mattered. Reproduced in isolation: a
600-example window under a 200-example cap evicted zero. And the rev-1 test
encoded that state as acceptable — it was written to accommodate the bug.

Eviction is now unconditional, including against the current window, on the
reviewer's reasoning that `out` already holds the references the training call
needs. The warning also moved from "did we evict" to "is the cap below one
window", which is the condition that actually predicts permanent re-replaying,
and it now fires on the first such call and reports the required size in GB.

### 10.3 [Medium] Identity assertion was not a mutation guard — accepted, enforced

Correct, and the point about numpy is the one I would have missed: `frozen=True`
protects the attributes, not the arrays behind them. `Example` is now frozen
*and* `__post_init__` marks all six arrays `write=False`. The label back-fill
that previously mutated in place is gone — inputs are staged during replay and
the Example is constructed once, complete, afterwards. `collate` had one line
taking a no-copy `torch.from_numpy` view of `policy_target`; it now `.astype`
copies like the four lines around it.

The test asserts the mutations raise rather than asserting identity alone.

### Not changed

* **Fixed default cap** (sign-off 4) — the reviewer preferred a fixed hard cap
  over a config-derived allocation that could exceed physical RAM. Agreed; the
  derived figure is now surfaced in the warning instead of being allocated.
* **Ragged storage** (sign-off 5) — declined, confirmed.
