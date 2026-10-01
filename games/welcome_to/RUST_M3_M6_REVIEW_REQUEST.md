# Review request — Rust port implementation M3–M6

The Rust port plan was reviewed before implementation, and M1/M2 received a
recorded post-implementation review. This request covers the implementation that
came after that sign-off: the encoder, information key, MCTS with progressive
widening, and cooperative scheduler. There is extensive equivalence evidence,
but no review response for this code was found.

Review the committed code as landed. The current working tree contains a later
evaluator ABI v3; that delta belongs to `S2_GENERATION_REVIEW_REQUEST.md` and
must not be mistaken for part of these commits' ABI v2 review.

## 1. Scope

| Milestone | Commit | Principal implementation and gates |
|---|---|---|
| M3 — Rust encoder | `751ddd8` | `welcome_to_rust/src/encoder.rs`, `rust_encoder.py`, `rust_encode_equiv.py`, `tests/test_rust_encoder_equiv.py` |
| M4 — information key | `c6c9d6c` | `welcome_to_rust/src/information_key.rs`, `rust_key_equiv.py`, `tests/test_rust_information_key_equiv.py` |
| M5 — MCTS descent | `e0a79db` | `welcome_to_rust/src/search.rs`, `rust_search.py`, `rust_search_equiv.py`, `rust_search_real_equiv.py`, `tests/test_rust_search.py` |
| M5 — widening default | `9dd96c0` | Rust/Python widening logic, `SEARCH_SPEC.md`, search equivalence gates |
| M6 — global scheduler | `fa1f9c9` | `welcome_to_rust/src/scheduler.rs`, `rust_scheduler_bench.py`, scheduler tests |

`380ba46`, `dbe3eb9`, and `03ae7cf` are context only: M1/M2 are already signed
off. S2 generation commits beginning at `9e900ba` are out of scope.

## 2. What is already gated

These results are recorded in `RUST_PORT_PLAN.md`; they are evidence to audit,
not claims that code review is unnecessary.

* M3: 400,342 encodings across 124,741 primitive states, every supported
  seat/configuration/viewer cell, bit-exact on all four arrays; zero divergence.
* M4: 69 constructed states forming 48 observation groups, including 21
  required collisions and 43 visible separations; Python and Rust induce the
  same partition, and equal-key groups encode and act identically.
* M5 control arm: 9,216 simulations and 40,257 exact evaluator requests,
  including 30,407 opponent POLICY requests and terminal leaves; exact request,
  tree, visit, total, and action comparison.
* M5 widening arm: 6,144 simulations and 21,632 exact requests; traversal/fresh
  counts, exact edges, terminal values, particles, reservoirs, and tree shape
  compared.
* M5 real network: one complete two-seat advanced trajectory, 55 decisions and
  47 searched roots, exact at batch width one.
* M6: blocking and batched discrete fingerprints agree; the recorded CPU run
  improved 2,529 to 6,336 games/hour with 8/8 identical fingerprints.

## 3. Review focus

### 3.1 Encoder/key containment

For every pair of states merged by the Rust information key, the Rust encoder,
legal macro list, and terminal value must agree. Please compare the field lists
in `encoder.rs` and `information_key.rs`, not only their tests. The dangerous
failure is an encoder field added later without widening the key; natural random
positions almost never collide and would not expose it.

Requested sign-off: the key is at least as fine-grained as every representation
the child node consumes, including mid-turn public/live-sheet distinctions.

### 3.2 Viewer and root-player ownership

The root-player contract has no negamax sign flip: leaf encoding/value belongs
to the root player, opponent actions are sampled transitions, and the scalar is
backed up unchanged. Trace viewer selection through forced collapse, terminal
leaves, opponent POLICY requests, chance outcomes, and rerooting. A default to
`state.actor` is plausible and silently wrong.

### 3.3 Fixed random tapes and insertion order

Scheduler geometry must not consume randomness. Verify domain-separated search
seeds, stable request sorting, response routing by id, insertion-order outcome
sampling despite Rust `HashMap`, and the separate counts for edge traversals and
fresh chance draws. These are what make concurrency a throughput change rather
than a strength change.

### 3.4 Progressive widening lifecycle

Please audit all transitions among fresh outcome, merged outcome, exact edge,
particle retention, and terminal cached value. In particular:

* exact edges replay the deterministic transition and retain no particle;
* reuse increments traversal count but not empirical fresh-draw weight;
* reservoirs never exceed `max_particles`;
* the no-widening control allocates no particle state;
* terminal observations cannot merge values from distinct endings.

### 3.5 Scheduler failure and progress paths

Each search may have one outstanding request. Inspect malformed response,
unknown/duplicate id, Python exception, worker error, and slot reset paths for
lost work, stranded workers, or a capacity leak. Also verify that sorting by
stable input order happens before chunking and that a worker-count change cannot
change batches or request order.

### 3.6 Numerical contract

M5 uses f64 tree totals, but priors originate in f32 and batching may reorder
device arithmetic. Confirm that every exact claim is restricted to fixed tape
and fixed batch composition, and that the real-network gates assert discrete
equivalence where bit identity is not justified.

## 4. Known limitations

* Python remains the rules and encoder oracle; this review does not revalidate
  Python against BGA.
* Expert, solo, and expansions are outside the production matrix.
* M6 has no within-search leaf batching and intentionally has no virtual loss.
* The committed M6 ABI v2 is superseded in the working tree. Review ABI v3 in
  the generation request rather than asking this historical range to explain it.
* The large equivalence gates are expensive and are not ordinary unit-suite
  fixtures. Their recorded corpus construction and assertions deserve review.

## 5. Running the gates

Current focused result on 2026-08-27, as part of the combined review suite:
**138 Python tests passed with one test-only warning; 24 Rust tests passed.**

```powershell
# Fast implementation checks on the current tree
.\.venv\Scripts\python.exe -m pytest `
  games/welcome_to/tests/test_rust_encoder_equiv.py `
  games/welcome_to/tests/test_rust_information_key_equiv.py `
  games/welcome_to/tests/test_rust_search.py -q

# Rust unit tests
Push-Location games/welcome_to/welcome_to_rust
cargo test
Pop-Location

# Large recorded gates; see each module's --help before rerunning
.\.venv\Scripts\python.exe -m games.welcome_to.rust_encode_equiv --help
.\.venv\Scripts\python.exe -m games.welcome_to.rust_key_equiv --help
.\.venv\Scripts\python.exe -m games.welcome_to.rust_search_equiv --help
.\.venv\Scripts\python.exe -m games.welcome_to.rust_search_real_equiv --help
```

## 6. Sign-offs requested

1. Does the M4 key contain everything the M3 encoding and M5 child semantics
   consume?
2. Is viewer/root-player attribution correct on every request path?
3. Can scheduler geometry change any random draw or tie-break order?
4. Are widening traversal, fresh-draw, exact-edge, and particle counts updated
   on exactly the intended paths?
5. Does every scheduler failure wake workers and return reusable capacity?
6. Are the equivalence claims no stronger than their batch and float controls?

---

# Review response — 2026-08-27

Reviewed as landed, against the working tree. The findings below were applied;
each names the file it changed. The six sign-offs follow.

Gates rerun after the changes: M3 encoder **40,092 encodings / 12,599 states /
69 games, zero divergences**; M4 key **69 states, 48 groups, 21 collisions, 43
separations, partitions agree**; M5 widening **48 positions x 2 seeds, 4,608
simulations, 15,953 requests**; M5 control **32 x 2, 3,072 simulations, 12,079
requests**; M5 real network **44 decisions, 33 searched, batch width 1**; Rust
**25 tests**.

## 9.1 Findings and disposition

### F1 — `derive_search_seed` collides structurally *(fixed)*

`portable_rng.py`, `welcome_to_rust/src/rng.rs`.

The derivation was `game_seed ^ DOMAIN ^ search_index`. S2 hands out a
**contiguous** block of game seeds (`range(seed, seed + games)`) and indexes
searches by the learner's decision count, so the counter lands in the same low
bits as the seed: decision `i` of game `s` received *exactly* the tape decision
`0` of game `s ^ i` received. Over a run every search shared its tape with
roughly `min(games, decisions)` others.

Two searches on one tape draw the same determinization permutations, and —
because `rust_search.root_noise` seeds NumPy from this same value — the
**identical Dirichlet vector** whenever the root widths match. Root noise is the
mechanism by which self-play discovers lines its current policy does not already
favour; duplicating it across the corpus removes exploration the generation
budget was already paid for. This is the finding with the most direct bearing on
"the policy will not learn to finish plans".

Replaced with one draw of the stream itself: advance
`PortableRng(game_seed ^ DOMAIN)` by `search_index` states and take the next
output. SplitMix64's state step is `+GAMMA`, so the skip is O(1); the finalizer
is a bijection, so a collision now needs `Δseed == GAMMA · Δindex (mod 2**64)`.
Covered by a new collision test over a 64×64 seed/decision block in both
languages, alongside the existing cross-language agreement assertion.

Nothing already captured is invalidated: trajectories replay from recorded
actions and recorded search targets, never by re-deriving a search seed.

### F2 — the viewer plane reads `ctx` the key does not carry *(fixed)*

`encoder.py::_viewer_plane`, `welcome_to_rust/src/encoder.rs::write_viewer_plane`.

This is the §3.1 failure mode, present. The viewer plane consumed
`state.ctx.number` / `ctx.effect` **ungated by `viewer == actor`**, while
`information_key` writes `ctx` only on the viewer's own turn. Two states the key
merges could therefore encode differently. It is also an information leak in its
own right: for a non-actor viewer it answers "where could the viewer write the
number the *opponent* has just picked", which is the opponent's live mid-turn
choice — precisely what `public_sheets` and `plan_turns_for` exist to hide.

Not reachable today (search evaluates leaves only at macro roots, where the root
is the actor; `WRITE_NUMBER` is inside a macro; `self_play.replay` raises if the
recorded actor is not the learner), so gating it changes no emitted row — the M3
gate stayed bit-exact across all four arrays. Gated in both languages together so
the oracle relationship holds.

### F3 — `base_deck_matrix` rebuilt per call *(fixed, measured)*

`welcome_to_rust/src/encoder.rs`.

`deck_composition` runs once per `information_key` — once per search transition,
not per evaluated leaf — and rebuilt the 81-card base histogram every time (a
`Vec` allocation plus a full walk of the card table), then cloned `discard` again
to join it to the table cards. Memoized in a `OnceLock` and accumulated in place.
Counts are whole numbers held in `f32`, so summation order is exact and the
result is bit-identical.

Measured on `rust_key_bench --rows 40000`: **529,702 → 601,735–642,512 keys/s**,
about **+17%** on information-key construction.

### F4 — a per-row Python loop inside the inference barrier *(fixed)*

`rust_search.py::PackedNetEvaluator.forward`.

Legal-index validation ran `np.unique` per row. That loop executes on the one
thread holding the GIL while every Rust search worker is parked at the global
barrier, so at production width it put `rows` sorts directly in the critical path
of the whole pool. Replaced with one whole-batch check: tag each macro with its
row (`row * NUM_MACRO_ACTIONS + macro`) and take a single `np.unique`. The
row-locating sort now runs only on the failing batch, and the `row_of` vector it
builds is reused for the policy gather, removing a second `np.diff`.

### F5 — every LEAF enumerated its macros twice *(fixed)*

`welcome_to_rust/src/{search.rs,scheduler.rs,macro_codec.rs}`.

The evaluator row carries the **full** vocabulary while the tree searches the
**pruned** one, so `emit_cloud_event` called `legal_macros` to pack the request
and `expand_from_response` called `search_legal_macros` — which calls
`legal_macros` again — to expand the node that response created. At
`CHOOSE_CARDS` that enumeration clones the whole `Game` once per playable slot,
because legality is enumerated rather than intersected (deliberately; see the
`macro_codec` module docstring).

`EvalRequestEvent` now carries the enumeration, the session stashes the pruned
list for a response that cannot see a changed state, and `prune_search_macros` is
split out of `search_legal_macros` so the two remain one code path. Identical
output: same filter, same list, same order.

### F6 — a fifth seat panics inside a worker thread *(fixed)*

`welcome_to_rust/src/search.rs`.

`terminal_value` fills a four-wide rank distribution indexed by finishing
position, but `Game::new` accepts any seat count and the encoder's one-hot
reaches `MAX_PLAYERS = 6`. A five-seat table whose root sits in the last tie
group indexes `distribution[4..5]` and panics; in the cloud scheduler that
surfaces only as "cloud search worker panicked". `training.rank_distributions`
already refuses such a table outright, so `Search::search`,
`play_with_temperature` and `PlaySession::new` now do the same with a named
error.

### F7 — the widening-support test measured the cap, not the support *(fixed)*

`tests/test_mcts.py::test_a_chance_edge_has_more_outcomes_than_the_deck_has_reveals`.

Red on the current tree, green at `d8509b3`; it belongs to the in-scope `9dd96c0`
(M5 widening default). Its assertion is `max outcomes > 6`, but at the shipped
`C=1, alpha=0.5` the cap is `ceil(sqrt(n))`, so clearing six requires one boundary
edge to be traversed **37** times. Instrumented on that exact position: the widest
boundary edges hold 6 outcomes at 27–37 traversals — *sitting on their cap*. The
test was reading the widening schedule, not the transition support, and it flipped
to failing when an unrelated head change moved the prior.

Rewritten to run at `C=4`, where the cap cannot bind below the reveal support, and
to assert that the cap exceeds six before concluding anything from the count. The
docstring's recorded "20 outcomes" is inconsistent with `C=1` (which needs 400
traversals of one edge) and is replaced by the measurement below.

## 9.2 Measurements taken during this review

Cloud scheduler on CPU, 2p advanced, 64 simulations, random-init net
(`RustCloudScheduler`, inflight 64, 8 workers, `max_batch` 64):

| stage | ms |
|---|---:|
| `python_eval` | 540.2 |
| `search` (8 workers, summed) | 399.7 |
| `encode` (8 workers, summed) | 273.0 |
| `coordinator_wait` | 138.4 |
| `pack` / `decode` | 24.5 / 11.4 |
| wall | 726 |

**The network forward dominates**: 74% of wall, against roughly 12%
wall-equivalent for all Rust worker time. Coalescing is healthy — 71 of 164 waves
ran at the full width of 64. The Rust share is per-worker-per-wave and so scales
with `inflight / workers`, growing about 4× at the production 256/8; that is what
makes F3 and F5 worth having, and it is also why neither is where the headroom
is. **Batch width and the network are the throughput levers here, not Rust.**

⚠ Widening support, measured on one position (2,048 simulations, three-card deck,
`alpha=0.5`, `max_particles=4`):

| `chance_widening` | widest boundary edge |
|---|---:|
| 1.0 (shipped default) | 6 outcomes (at cap) |
| 4.0 | 30 outcomes |
| 8.0 | 80 outcomes |

At the shipped constant **every** boundary-crossing edge in that tree sat exactly
on its cap. The search's model of "what happens after my move" is therefore a
six-outcome, four-particle approximation of a distribution with tens of outcomes.
This is a strength question, not a defect — `SEARCH_SPEC.md` §12 already marks
`C=1, alpha=0.5, max_particles=4` as a default pending an equal-wall-clock bakeoff
— but it bears directly on plan racing, where what matters is whether an opponent
finishes a plan first and that depends on exactly the reveals and opponent replies
this edge compresses. **No default was changed here.** The bakeoff the spec already
calls for should sweep `C` and `max_particles` together and be read on
plan-completion rate, not only on games per second.

## 9.3 Sign-offs

1. **Does the M4 key contain everything M3 and M5 consume?** Yes, after F2. Field
   by field: `sheet_planes` reads `sheet_for` and the offered numbers, both in the
   key; `sheet_scalars` adds `score_breakdown(seat, Some(viewer))`, which routes
   through `sheet_view` / `plan_turns_view` and is viewer-scoped; `global_scalars`
   already gates `ctx` on `owns_ctx`, and `may_ask_reshuffle` depends only on
   completions with `t < turn`, every one of which `plan_turns_for` shows to every
   viewer. `aside_composition` reads `stack_old[0]`, which the key carries inside
   `table_cards` (`stack_new ++ stack_old`, in order, with both faces written).
   All twelve `Sheet` fields are written by `write_sheet`. Terminal leaves are safe
   by construction: `prepare_turn_boundary` increments `turn` and syncs
   `public_sheets` before setting `GameOver`, so at a terminal state nothing is
   hidden from any viewer and equal keys imply equal scores and equal tie-break
   keys — distinct endings cannot merge. The viewer plane was the one gap, and it
   is closed.
2. **Is viewer/root attribution correct on every request path?** Yes. LEAF uses
   `root` in both the blocking descent and `PlaySession`; opponent POLICY uses
   `state.actor`; `terminal_value` uses `root`; `blend_value` reads
   `scores[0] - max(scores[1..])`, which is viewer-relative because the encoder
   puts the viewer on seat axis 0 and the network's score head is emitted on that
   axis. Nothing defaults to `state.actor`. Note that at every `expand_leaf` the
   actor *is* the root, which is what made F2 latent rather than live.
3. **Can scheduler geometry change a random draw or tie-break?** No. Both
   schedulers sort `pending` by stable input order **before** chunking and assign
   `abi_id` after the sort; each search has one outstanding request, so `input` is
   unique within a wave. Every live session steps exactly once per wave whatever
   bucket owns it, so the wave's request set is worker-count-independent. Outcome
   reuse sorts by `ordinal`, not by `HashMap` order. `edge.visits` and
   `outcome.count` are separate and are incremented on the intended paths.
4. **Are widening counters updated on exactly the intended paths?** Yes. Exact
   edges replay the deterministic macros and retain no particle
   (`needs_particle_slot` requires `!exact`); reuse increments `edge.visits` only;
   fresh draws increment both; reservoirs are capped at `max_particles`; the
   control arm returns from `record_outcome` before allocating anything. One
   diagnostic trap worth recording: **in the control arm `edge_visits` is never
   incremented at all**, because `record_outcome` returns first — so
   `debug_tree`'s `edge_visits` reads as zeros there and must not be compared
   across arms.
5. **Does every scheduler failure wake workers and return capacity?** Yes. A
   Python error cancels every active worker and each drains its tasks into
   `Finished`; a worker panic is caught, reported as `Panicked`, and its slots are
   rebuilt with clean trees; a dropped channel sets an error and clears `active`.
   No worker can be in `active` while idle, so the "idle worker received a
   non-start command" panic is unreachable: a worker leaves the inner loop only by
   sending `Done`, and the coordinator removes it from `active` in the same wave
   it reads that message. Slots are restored from `Finished` before the rebuild
   pass, so a normal evaluator error keeps its tree.
6. **Are the equivalence claims no stronger than their controls?** Yes, and one
   caveat should be recorded. The M5 arms are exact under a fixed tape *and a
   fixed batch composition of one*: they run through `evaluate_request`, where
   **NumPy** performs the masked softmax over the dense 684-vector. The M6/cloud
   path performs that softmax **in Rust over the compact legal subset**. The two
   are equal in exact arithmetic and not bit-identical in `f32`, and a prior
   difference can flip PUCT's first-max tie-break. M6's recorded "8/8 identical
   fingerprints" is therefore an empirical discrete agreement, which is what it
   claims — not a bit-identity guarantee, and it must never be strengthened into
   one.

## 9.4 Not changed, recorded

* **Unvisited children take `Q = 0`** while values live in `[-1, 1]`, so an
  untried action looks optimistic when the root is losing and pessimistic when it
  is winning. This matches the Python oracle exactly (`np.argmax(q + exploration)`
  with `q = 0` where `visits == 0`), so changing it is a strength decision that
  would invalidate every M5 gate, not a port defect.
* **`noise_fresh_fraction`** is 1.0 by default and 0.25 in
  `self_play.default_search_config`, per `SEARCH_SPEC.md` §4. Correct as shipped;
  re-rooting buys nothing at 1.0 by design.
* **`RustScheduler`'s blocking adapter still enumerates macros twice** per leaf
  (F5 was applied to the cloud path only). It is the diagnostic arm, and threading
  the enumeration through its `FnMut` evaluator closure would change the M5 seam
  this review exists to hold still.
