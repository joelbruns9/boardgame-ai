# Review request — search steps 4–7 (boundary, key, batching, chance edge)

Companion to `SEARCH_SPEC.md`, which is the design and measurement **record** —
why each thing was built, what was measured, and which designs were withdrawn.
This is the **review brief**: what changed, which invariants must hold, and where
a second pair of eyes is worth most. Don't read the spec end to end; §6.3, §7.1a,
§7.9 and §12.1 are the parts these steps implement, and they are linked below.

**One of these changes is live and three are not.** Steps 4–6 are on by default;
step 7 is behind `SearchConfig.chance_widening`, which is `None`. So the search
today behaves as it did except that its child key is now the viewer information
state (measured to induce an identical partition — §3) and its leaf evaluations
can be pooled by a driver nobody has wired up yet.

**Related brief:** `SEARCH_STEPS_1_3_PROMPT.md` was the hand-off for steps 1–3,
whose external review found two defects (both fixed in `7ffebe1`). Those steps
are not in scope here except where 4–7 changed their behaviour.

---

## 1. Scope

| Step | What | Commit | Primary files |
|---|---|---|---|
| 4 | Engine-owned turn boundary: `prepare` / `sample` / `apply` triple | `e48948e` | `game.py`, `tests/test_game.py` |
| 5 | Viewer information-state key, and `_advance` keyed on it | `943f699` | `mcts.py`, `tests/test_mcts.py` |
| — | §7.1a resolution C and §7.8 noise decided | `943f699` | `SEARCH_SPEC.md` |
| 6 | Suspension seam + wave driver + batched evaluator | `40bf377` | `mcts.py`, `tests/test_mcts.py` |
| 7 | Chance edge as progressive widening, behind a flag | `07e63bf` | `mcts.py`, `tests/test_mcts.py` |
| — | Terminal-keying collapse found and fixed; this brief | *uncommitted* | `mcts.py`, `tests/test_mcts.py`, this document |

`game.py`: `prepare_turn_boundary`, `sample_boundary_outcome`,
`apply_boundary_outcome`, `BoundaryOutcome`, `_replay_draw`, `_reveal_step`,
`_open_turn`, `_is_boundary_afterstate`, `DrawFn` threading through
`_draw_playable` / `_draw_step` / `_reshuffle_decks` / `_begin_turn`.

`mcts.py`: `information_key`, `_printed`, `_composition_key`, `Ask`, `Request`,
`drive`, `run_searches`, `trajectory_fingerprint`, `NetEvaluator.answer_batch` /
`evaluate_batch` / `policy_batch` / `_forward_many`, `sampling_opponent` (now a
generator), `Outcome`, `Node.edge_visits` / `Node.outcomes`, `_edge_is_closed`,
`_pick_outcome`, `_record_outcome`, `_simulate`, `search_gen`, `_advance`,
`_collapse_forced`.

---

## 2. What is already gated (please don't re-verify by hand)

Full suite green; run it with §6. The things below are asserted, not assumed.

**Step 4 — the boundary triple**
* **Equivalence with the engine's own path**, parametrised over all three reveal
  cases (ordinary, exact-empty reform, queued reshuffle), comparing turn, actor,
  phase, both stacks, table, `reshuffle_next_turn`, `deck_remaining`,
  `solo_card_drawn`, the full card census and `legal_actions()`. It asserts its
  own comparison count (`compared == 12`) so it cannot pass vacuously.
* **One test per §6.3 case**, including terminal-before-reveal, plus the generic
  mid-draw case driven directly.
* **`deck_remaining % 3 == 0` at every `_draw_step`**, per seat count — which is
  why mid-draw exhaustion is not a standard-mode case.
* **`BoundaryOutcome.reformed`** checked against the real `_reform_deck` call,
  not inferred from cursor arithmetic (4,329/4,329 in the exploratory run).
* Card census preserved across reform and reshuffle; outcomes contain only cards
  the boundary makes public; `apply` is deterministic and rejects a foreign
  outcome; both entry points refuse an unprepared state.

**Step 5 — the information key**
* §12.1's "test that matters" verbatim: mutating seat 0's *live* sheet leaves
  seat 1's key unchanged, mutating `public_sheets[0]` changes it, and the
  viewer's own sheet is live.
* One test per exclusion: future deck order (`redeterminize` must not move the
  key), the table-wide `reshuffle_next_turn` against the viewer's own vote,
  printed types against physical ids, another seat's `ctx`.
* Deck **and discard composition** are in the key (the swap mutation).
* The other direction: 300+ distinct positions, no two sharing a key.

**Step 6 — the batching seam**
* **Wave equals blocking**, by `trajectory_fingerprint` over actions and visit
  counts — 32/32 identical.
* Both `Ask` kinds reach the seam and share a call; mean batch > 4.
* A mixed batch answers each kind exactly as its single form does.
* Counters kept apart (`calls` / `rows` / `batch_widths`); empty batch is not a
  call.

**Step 7 — progressive widening**
* **The control arm allocates nothing** — with the flag off, every node's
  `outcomes` and `edge_visits` are empty.
* The widening cap holds on every edge in the tree.
* Every outcome is resumable (particles or a terminal value) and particles are
  capped.
* **Reuse never moves a weight** (the Pólya-urn regression, §4.4).
* A deterministic edge re-merges onto its single outcome; reuse actually happens
  and skips the transition.
* Depth increases *and* answers to the budget (§7.9).
* At a three-card deck the edge finds its support and stops.
* **Terminal transitions are keyed apart**, and the key determines the terminal
  value (§4.5).

---

## 3. Focus areas

Ordered by what I think is most likely to be wrong, not by how interesting it is.

### 3.1 ⚠ MUST BE CORRECT — the particle/encoder coupling (step 7)

**This is the invariant the whole particle design rests on, and it is nowhere
stated in the spec.**

When a closed edge is reused, the descent resumes from `rng.choice(particles)`
and continues into `node.children[(action, observation)]` — a node whose priors
were computed by `encode_state` on **whichever particle arrived first**. That is
only sound if:

> every state sharing an `information_key` encodes identically.

i.e. `information_key` must be **at least as fine-grained as `encode_state`**. If
the encoder reads anything the key omits, the tree merges positions the network
can tell apart, and every prior below that node is the wrong one.

⚠ **ANSWERED, and the answer was no** — see §7. The key carried
`config.players` alone while the encoder also reads `advanced`, `expert` and
`solo`, so flipping `advanced` left the key identical and the encoding different.
My check could not have found it: 525 particle pairs across 6 searches with **0
encoding differences**, but every pair came from *within one search*, where the
configuration never changes. Fixed by putting the whole frozen `GameConfig` and
the raw `len(discard)` in the key. The two functions were written months apart against
`ENCODER_V2_SPEC.md` §9.3 and §12.1 respectively, and nothing makes them move
together.

**What I'd like checked:** read `encoder.encode_state` against
`mcts.information_key` field by field and tell me whether the containment holds
by construction. If it does not, the fix is to widen the key, not the encoder.

### 3.2 ⚠ MUST BE CORRECT — `_sheet_key` completeness (step 5, and step 1's guard)

`_sheet_key` enumerates a `Sheet`'s mutable fields by hand: `numbers`, `is_bis`,
`written_turn`, `fences`, `top_fences`, `parks`, `pools`, `estate_marks`,
`temps`, `bis_marks`, `permits`, `roundabouts`.

**Add a field to `Sheet` and both keys silently stop distinguishing on it.**
`_position_key` would then re-root onto a subtree from a different position, and
`information_key` would merge two positions under widening. Neither would fail a
test; both would be quietly wrong.

This is the same shape as the defect the steps 1–3 review found (`_position_key`
carrying `deck_pos` and `len(discard)` but neither composition). I fixed that
instance; I did not fix the *class*.

**Suggested guard, if you agree it is worth it:** assert
`set(_sheet_key_fields) == set(dataclasses.fields(Sheet))` so adding a field
breaks a test rather than a search.

### 3.3 ⚠ Assumption — "which boundary case fires is deterministic" (step 4)

`SEARCH_SPEC.md` §6.3 now claims, on my measurement, that the *case* is settled
by the afterstate and only the *cards* are chance: a queued reshuffle is
`reshuffle_next_turn`, an exact-empty reform is `deck_remaining == 0`. §7 leans
on this — a chance node's support has a known length (3, or 6 on a reshuffle).

Measured over **84,808 boundary crossings** at 2/3/4 seats with no exception.
But two caveats I want a second opinion on:

* **The solo card breaks it in principle.** `_draw_playable` consumes an extra
  card when it turns up, so a 3-card deck could reform *mid-triple*. Solo is out
  of training scope, and the measurement was on 2–4 seats where the solo card is
  not in the deck — but the claim in §6.3 is stated generally.
* **The count includes GreedyBot's one-ply lookahead** (~20× the real boundary
  count). They are ordinary engine transitions on legal states, and I say so in
  the spec, but if you think that inflates the evidence, say so.

### 3.4 ⚠ Complex logic — the two counters (step 7)

`Node.edge_visits[index]` counts **traversals** and drives the widening cap.
`Outcome.count` counts **fresh draws** and drives the weights. They must never be
confused, and I already got this wrong once:

> The first implementation used one counter for both. Sampling a child in
> proportion to a count that the sampling itself increments is a Pólya urn — it
> reinforces whichever outcome arrived first and converges to a random limit.
> Visible in the output: an edge whose outcomes were near-unique reveals, which
> *must* be uniform, read `[0.045, 0.045, 0.091, 0.091, 0.727]`.

Fixed, with a regression test, and the same edge now reads `[0.2] × 5`. **What I
want checked is whether any path still increments the wrong one.** The three
places that touch them are `_edge_is_closed` (reads visits), the reuse branch of
`_simulate` (visits only), and `_record_outcome` (visits **and** count). A fourth
site added later would be easy to get wrong and hard to notice — the symptom is a
slow drift in weights, not a failure.

### 3.5 ⚠ Complex logic — the generator seam and its two drivers (step 6)

`search_gen` suspends at every network request; `drive` answers them one at a
time through `evaluate`/`policy`, and `run_searches` pools them through
`answer_batch`. **Two code paths that must agree.**

* `evaluate` is `evaluate_batch([...])[0]` and `evaluate_batch` is `answer_batch`
  with `Ask.LEAF`, so the *interpretation* is shared by construction. Please
  check I have not left a way for them to diverge.
* `answer_batch` computes values only for `LEAF` rows, via fancy indexing:
  `out["rank_logits"][wants_value]` and `out["score"][wants_value]`, with a mask
  built in the *compressed* index space and results scattered back by
  `enumerate(wants_value)`. **This indexing is the fiddliest code in the change**
  and a transposition would produce plausible values attributed to the wrong
  rows. The mixed-batch test compares against singles, which should catch it —
  confirm that it would.
* `drive` catches `StopIteration` from `next(gen)` as well as from `send`, so a
  generator that finishes without yielding (a terminal root) returns correctly
  rather than raising. Worth a glance.

### 3.6 ⚠ Assumption — float drift never changes a decision (step 6)

A batched forward is not bit-identical to a single one; reductions reorder.
Measured **~1e-7** on priors and values, argmax stable across batch widths
2/4/8/16, and 32/32 trajectories identical.

That is evidence, not a proof. Under PUCT a tie to seven digits could in
principle flip, and if it does, the same game played at different wave
concurrency produces different trajectories — which would break
reproducibility-from-seed, a property `THROUGHPUT_LEVERS.md` §A treats as the
defining test of a class A lever.

**The question I cannot settle:** is "identical in 32/32 at one budget" enough to
call this class A, or should the wave carry a documented caveat that
reproducibility is up to float reordering? I have written it as the latter in
`run_searches`' docstring; tell me if that is too weak or too strong.

### 3.7 ⚠ Assumption — `_replay_draw` may leave the residual deck in any order (step 4)

`apply_boundary_outcome` swaps each named card to the top of the undrawn region
and draws it. Composition stays exact; the order of what remains is whatever the
swaps left.

I justified this as §7.3's non-anticipativity rule — the residual order is hidden
and re-determinized per simulation, so it must not be recorded. **But it assumes
every consumer re-determinizes.** A caller that applied a boundary outcome and
then read the deck order would be reading an artifact of the swap, not a deal.
Nothing does this today. Is it worth an assertion, or is the docstring enough?

### 3.8 ⚠ Complex logic — `_is_boundary_afterstate` is a heuristic (step 4)

Both `sample_boundary_outcome` and `apply_boundary_outcome` refuse a state that
is not "prepared", detected as *every `stack_new` slot is `None`*. That is the
signature `_discard_step` leaves, and I believe it is unreachable otherwise in
standard play — but it is a property of the data, not a flag, and expert mode
clears all groups' `stack_new` too.

A stronger guard would be an explicit marker set by `prepare_turn_boundary`.
I did not add one because it means a new `GameState` field threaded through
`copy()`. Say if you think that trade is wrong.

### 3.9 Behaviour change to review — reuse spends determinization diversity (step 7)

Resuming from a particle re-uses that transition's randomness, so a widened
search explores fewer distinct futures per simulation than the open-loop one. It
buys depth (1.42 → 2.14 at 256 sims) and spends diversity.

This is *intended* (§7.3) and the flag is off, so it is not a live regression.
But I want it said out loud that **the case for turning it on is not made by the
depth number** — it is a strength question for §11's bakeoff as equal-wall-clock
arms. If you think the depth measurement is being quoted as though it settled
something, say so.

### 3.10 What I found while writing this brief

Two invariants I decided to check rather than ask about. One held, one did not:

* **Particles encode identically** (§3.1): 525 pairs, 0 differences. Held.
* **Terminal transitions were collapsing.** `_advance` returned a bare `()` for
  every ending. Harmless in the open-loop arm — a terminal transition stores no
  child, so the key is never looked up — and silently wrong under widening, where
  the observation *is* the outcome key. Measured: **255 distinct endings of one
  edge merged into a single outcome**, carrying whichever final score was
  computed last, returned by every later reuse. Fixed by keying terminal states
  like any other; now 39 endings give 39 keys, and the key is verified to
  determine the value. Regression test added.

I mention this because it suggests the terminal path is thin elsewhere too — it
was reached by **zero** of the searches in my first invariant run, because
mid-game positions do not end. If you are looking for a place where coverage is
weakest, it is there.

---

## 4. Known limitations / out of scope

* **Nothing is wired into a self-play loop.** `run_searches` runs one wave to
  completion; there is no scheduler with games starting and finishing
  continuously. Deliberate — the loop comes first, then throughput.
* **S0 has never been run.** Every measurement here uses either an untrained net
  or a synthetic GreedyBot-shaped prior, and the spec says so at each site. No
  claim about *strength* is made anywhere in steps 4–7.
* **The encoder is now the top throughput lever and is untouched** — ~30% of
  search wall clock, per-row Python, does not batch away. `THROUGHPUT_LEVERS.md`
  §4.2. Out of scope here.
* **Expert and solo modes** are supported by the engine and out of training
  scope. Step 4 keeps them working (the mid-draw test exists for expert); step 7
  has not been exercised against them.
* `noise_fresh_fraction` and `chance_widening` are both unset pending strength
  measurement (§7.8, §7.9).
* **The equivalence gate for step 4 is 12 comparisons**, 3 cases × 4
  seed/seat combinations. Small. If you want it wider, that is cheap to do.
* **Two `test_arena.py` tests were seeded as part of this change**, and the
  reason is worth knowing. They built **unseeded** networks, so their weights
  came from wherever the global generator happened to be -- which makes them
  depend on how many `torch.manual_seed` calls ran earlier, i.e. on test
  ordering. Latent until somebody adds a seeded test; step 7's tests did, and
  `test_a_search_can_be_dropped_into_the_harness` failed in the full suite while
  passing alone. Verified pre-existing rather than caused: the same 20 net seeds
  give identical scores before and after the terminal-keying fix, and the
  control arm never looks up a terminal key. The assertion has a thin margin to
  spend on luck -- `subject_score > 0` from an untrained player at two
  simulations, mean 10.7 over 60 seeds but minimum 1.0.

  ⚠ **I did not audit the rest of the suite for the same shape.** If unseeded
  nets with thin assertions exist elsewhere, they are flakes waiting for the
  next seeded test.

---

## 5. Running the gates

```bash
# whole suite (~3.5 min)
python -m pytest games/welcome_to/tests -q

# step 4 only
python -m pytest games/welcome_to/tests/test_game.py -q -k "boundary or reform or reshuffle or triple or promotes or outcome"

# step 5 only
python -m pytest games/welcome_to/tests/test_mcts.py -q -k "key"

# step 6 only
python -m pytest games/welcome_to/tests/test_mcts.py -q -k "wave or mixed or counters or empty_batch"

# step 7 only
python -m pytest games/welcome_to/tests/test_mcts.py -q -k "widen or control_arm or outcome or three_card or deterministic_edge or deepens or reuse or terminal"
```

---

## 6. Sign-offs requested

1. **§3.1** — does `information_key` contain everything `encode_state` reads? If
   not, name the field. This is the one I most want an answer to.
2. **§3.2** — is the `_sheet_key`-completeness guard worth adding, or is it
   ceremony?
3. **§3.3** — is the "case is deterministic" claim safe to state generally, given
   solo, or should §6.3 scope it to standard 2–4 seats?
4. **§3.4** — is there a path that increments the wrong counter?
5. **§3.5** — does the mixed-batch test actually catch a transposition in
   `answer_batch`'s fancy indexing?
6. **§3.6** — class A, or class A *with a reproducibility caveat*?
7. **§3.8** — explicit prepared-marker on `GameState`, or leave the heuristic?
8. Anything in §3.9 that reads as though a throughput number settled a strength
   question.


---

## 7. Review response — 2026-08-23

**Every finding verified against the code and reproduced. None disputed.** The
verdict was "changes requested before enabling `chance_widening`"; both P1s and
all three P2s are now fixed, and three claims this document and `SEARCH_SPEC.md`
made are **retracted** rather than softened.

### The two P1s

**#1 — widening never stopped at finite support.** Reproduced exactly. The
closure predicate compared `len(outcomes)` against a target that keeps growing,
so once the target passes the real support it can never bind again.

| | before | after |
|---|---|---|
| support-one edges, reuse | **6 of 77 traversals (7.8%)** | **111 of 114 (97.4%)** |
| multi-outcome edges, reuse | 287 of 427 (67%) | 295 of 463 (64%) |

Fixed as recommended: determinism is now **proven, not inferred from
collisions**. A transition that changes neither the turn nor the game's end
consumed no randomness, so its support is exactly one and the edge is closed
permanently (`Node.edge_exact`). General finite support is *not* claimed —
ordinary progressive widening cannot detect exhaustion from collisions, and the
spec now says so instead of the opposite.

**The three-card test was worse than you said.** With the cap fixed, the same
edge shows **20** outcomes, not six — because the retained outcome is the whole
root-to-root transition, exactly the reveal/transition conflation §7.1a exists to
correct. The test is rewritten to assert that the transition support *exceeds*
the reveal support, and the six-reveal claim has moved to where it holds: the
boundary sampler, where 400 samples at `D = 3` give exactly **6** distinct
`BoundaryOutcome.draws`, near-uniformly.

**#2 — `information_key ⇒ identical encoding` was false.** Reproduced:
`same_key_config_flip True, same_encoding_config_flip False`. My own §3.1 spot
check could not have caught it — it compared particles *within one search*, where
the configuration never changes. The key now carries the whole frozen
`GameConfig` and the raw `len(discard)`, with a test for each.

### The three P2s

**#3 — rejection mutated the receiver.** Reproduced. `apply_boundary_outcome` is
now transactional: the replay runs on a copy and is adopted only after the
outcome validates whole. Tested against all three rejection modes, asserting the
receiver is byte-identical afterwards and still usable.

**#4 — particles froze on the first four.** Now reservoir replacement, so the
collection stays a uniform sample of *all* fresh draws for the same memory.
`max_particles = 4` is flagged in §7.9 as unmeasured and belongs in the bakeoff.

**#5 — the boundary lifecycle is now explicit.** `GameState.boundary_prepared`,
set by `prepare`, cleared by `_open_turn`, carried by `copy()`. It also refuses a
**second** `prepare`, which the old inference could not — the second call found
exactly the shape it was looking for and would have incremented the turn again.
The residual-order concern is enforced rather than documented:
`_scramble_undrawn` shuffles the undrawn tail, so its order carries no artefact
of which cards the outcome named.

### Sign-offs

| | Your answer | What was done |
|---|---|---|
| 1 | No — widen the key | Done: whole `GameConfig` + raw `len(discard)`, two tests |
| 2 | Add the guard | Done: `_SHEET_FIELDS` compared against `dataclasses.fields(Sheet)` |
| 3 | Scope to standard 2–4p | Done: §6.3 scoped, and the **rules argument** named as the justification with the lookahead count demoted to regression coverage |
| 4 | Counters are correct | Agreed — the defect was the predicate, not the counters. No change |
| 5 | Indexing correct; widen the test | Done: mixed 2/3/4-seat batch with row-distinct synthetic head outputs |
| 6 | Class A with a numerical caveat | Agreed; the docstring already says this. No change |
| 7 | Add the marker | Done: explicit flag, plus double-prepare refusal |
| 8 | Remove "fixes itself" / "stops" | Done: both **retracted in place** in §7.1a and §7.9 |

### What I did not do

* **No `BoundaryAfterstate` wrapper.** A flag on `GameState` gives the lifecycle
  guarantee and the double-prepare refusal without changing the shape of every
  call site or the `copy()` contract. If you want the wrapper for the type-level
  guarantee, say so and I will build it — but I did not want to change a public
  API shape on inference about future use.
* **No bakeoff.** `chance_widening` stays `None`. Enabling it needs the
  equal-wall-clock strength arms you describe, across widening constant,
  particle count and search seed — none of which exist until S0 has run.
* **No audit of the rest of the suite** for the unseeded-net flake shape noted
  in §4.
