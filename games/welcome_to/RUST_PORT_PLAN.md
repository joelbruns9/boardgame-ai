# Welcome To — Python → Rust port plan

The hot path is interpreted and it is now **93% of search wall clock**. This is
the plan to move it, in milestones that are each independently gated, so that a
laptop self-play shakedown is a lunch break rather than an overnight job.

**Read first:** `THROUGHPUT_LEVERS.md` §4.1 (the port is the largest lever in
this repository's history *and* the least reversible — treat it as a port with a
correctness obligation, not an optimisation), and the Kingdomino port's recorded
gotchas, which transferred to 7WD and mostly transfer again — §6 says which one
does not.

**Companion documents.** `SEARCH_SPEC.md` is what the search *is*; this is how it
gets compiled. `ENCODER_V2_SPEC.md` is the encoder contract the Rust encoder must
reproduce bit-for-bit. Neither is superseded by this.

---

## 0. What the 2026-08-23 review changed

Reviewed before implementation, which is the cheapest time. **Seven P1s and two
P2s, all accepted.** The direction survived; the gates did not. Recorded here
because a plan that quietly absorbs its review teaches nobody.

| finding | what was wrong | where it went |
|---|---|---|
| RNG contract absent | "same seed" does not give lockstep, **and Python cannot replay a Rust-generated trajectory at all** | §2 M0-B |
| M1 comparator incomplete | sheets + census cannot detect a card moved deck→discard, and omitted `public_sheets` would let an information leak pass a green rules gate | §4 M1 |
| comparator self-contradictory | said "sorted list" in one place and "raw order matters" in another | §4 M1 |
| ABI undefined | `answer_batch` takes **Python `GameState` rows and runs the Python encoder** — using it from Rust forfeits M3's entire gain | §2 M0-E, §3 |
| `leaf_batch > 1` mis-transferred | Welcome To has **no within-search leaf batch**; adding one needs virtual loss and is a class C strength change | §6 |
| widening excluded "permanently" | the no-widening arm cannot search past near-unique boundary outcomes | §8.1 — resolved: **built in M5 and promoted to the default** |
| M4 partition gate vacuous | information keys almost never collide naturally, so a random corpus proves only that both engines say "different" | §4 M4 |
| M5 fingerprint gate too strong | a mock evaluator removes torch drift but **not** redeterminization or opponent sampling | §4 M5 |
| checkpoints unmeasurable | M1 and M3 are not wired into the search, so "measure after M1" can only microbenchmark | §9 |

---

## 1. Why, with the measurement

Measured under the **wave** driver (`run_searches`, `max_batch=32`), 32 searches
× 64 simulations, two seats, CPU — i.e. *after* step 6 already made the network
cheap:

| component | share | ported? |
|---|---|---|
| `encode_state` | **46.6%** | yes — M3 |
| search + engine | **41.4%** | yes — M1, M2, M5 |
| `information_key` | 5.3% | yes — M4 |
| network forward | 6.2% | **no** — stays in torch |
| tensor build | 0.5% | no |

**93% of wall is Python that a port replaces. Ceiling ≈ 15×.**

⚠ **Step 6 raised this number, it did not lower it.** Before batching, the
forward was ~30% of wall and the ceiling was ~3×. Pooling leaves and opponent
policies into one call cut the forward to 6.2%. The two changes compound; the
port is worth *more* now.

Kingdomino's headline was **1,675 → 46,560 leaves/s ≈ 28×** on its ported part.
Do not promise that here: its profile was 51% board ops and 17% encoder, and ours
is the other way round.

---

## 2. M0 — contracts, before any implementation

**Nothing in M1–M6 is safe to build until these are fixed and written down.**
Each is a decision, not code; each takes an afternoon; each is expensive to
retrofit.

**✅ Signed off 2026-08-23.** B, C, D and F as written. A and E changed:
`dirichlet_alpha` is **supported**, not rejected — the draft contradicted §8 —
and the two rejection lists are now separated by *why*. E is cut back to the
structural minimum, because the rest of it was a throughput question answered
without a measurement.

### M0-A — the supported configuration matrix ✅ signed off 2026-08-23

Supported: **standard 2/3/4 seats, advanced on and off.** Everything else is
rejected **loudly, at construction, naming the reason** — a silently-ignored flag
is how a Rust self-play run stops being equivalent without anybody noticing.

⚠ **Two different lists, and an earlier draft conflated them.**

**Never supported** — out of training scope, refuse rather than half-implement:
expert mode, solo mode, expansions.

**M5 search modes:** progressive widening is supported with the same `C`,
`alpha`, and particle cap as Python. `chance_widening = 1.0` is the default;
`None` remains the explicit no-widening control. Invalid widening parameters are
refused at construction rather than silently reinterpreted.

✅ **`dirichlet_alpha` is SUPPORTED, and an earlier draft wrongly listed it as
rejected.** That contradicted §8, which already specifies the boundary — **Python
generates the noise vector, Rust applies it**. It is also not optional:
**S2 needs root noise**, so a backend that cannot apply it cannot run self-play,
which is the whole purpose of the port. What *is* forbidden is a hook that
silently omits noise; §8 says why.

### M0-B — the RNG contract ⚠ load-bearing, and it has two halves

`game.py` draws from `random.Random` for the deck shuffle, plan selection,
reshuffles and `redeterminize`; `mcts.py` draws for opponent sampling and
Dirichlet noise. **Rust cannot reproduce the Mersenne Twister**, so "same seed"
gives two different games.

⚠ **And this is worse than a testing problem.** `datagen.replay` rebuilds the
game with `GameState.new(seed=trajectory.seed)` and re-applies the recorded
actions. A trajectory generated by a Rust engine would replay against a
*different deck*, so recorded actions become illegal or scores diverge —
**Rust-generated training data would be unusable by the Python trainer.**

✅ **The answer already exists in this repository.**
`seven_wonders_duel/portable_rng.py` is a SplitMix64 stream written for exactly
this reason ("the Mersenne Twister and `gammavariate` cannot be reproduced in
Rust"), with a mirrored `rng.rs`. Port it, do not invent one.

Decisions to record in M0:

* **Engine RNG** becomes `PortableRng`, in both languages, for shuffle, plans,
  reshuffle and `redeterminize`.
* **`Trajectory` gains a version/`rng` field**, defaulting to `cpython` for
  everything already captured. The existing 5,000-game S0 corpus must stay
  replayable; a format that invalidates it silently is a data-loss bug.
* **Search RNG is domain-separated per search**, derived as
  `PortableRng(game_seed ^ constant ^ search_index)`. ⚠ It must not be drawn
  from a shared stream, or **scheduler concurrency and batch geometry would
  change the draws** — which would make the wave a class C lever instead of the
  class A one it is measured to be.

### M0-C — the state snapshot schema

One versioned, complete, order-preserving serialisation of `GameState` in both
directions. M1's comparator uses it; cross-engine hand-off needs it; a debugger
needs it. Derive the field list from `dataclasses.fields`, not by hand — see M1.

### M0-D — the static-table signature

Hash `CARD_TABLE`, `PLANS`, the scoring tables and the codec layout in both
engines and compare at load. A silent table divergence produces a legal-looking
game that is a different game, and it would pass every per-action gate.

### M0-E — the evaluator ABI

⚠ **The existing seam is the right *shape* and the wrong *interface*.**
`NetEvaluator.answer_batch` takes `(Ask, GameState, viewer)` rows and calls the
**Python** encoder. A Rust-owned state cannot use it without either marshalling a
`GameState` back into Python or re-encoding in Python — which forfeits M3, the
single largest win in the plan.

**✅ Frozen before M3 and corrected at M5: version 2, little-endian packed
buffers.** V1 specified `values` as f32 before the boundary had a consumer.
That would quantize Python's f64 `blend_value` before adding it to the f64 tree
total, defeating M5's exact-oracle gate for no throughput benefit. V2 changes
only that response column to f64; no V1 scheduler or corpus existed to migrate.
Every multi-byte field is little-endian; all float buffers are IEEE-754 `f32`.
Shapes are implicit in `(version, rows)` plus the encoder constants, never sent
as Python tuples per row. The single-state M3 diagnostic method returns four
`bytes` objects in this same row layout; M6 concatenates those rows without a
representation change.

```
RustScheduler.step() -> RequestBatchV2
    version            u16 = 2
    rows               u32
    sheet_planes, sheet_scalars, viewer_plane, global_scalars
                       four contiguous f32 buffers, batch-major, exact encoder
                       row shapes `(4,12,3,12)`, `(4,45)`, `(1,3,12)`, `(358,)`
    legal_indices      u16, concatenated in canonical macro order
    legal_offsets      u32[rows + 1], CSR-style
    kind               u8[rows]: 0 = LEAF, 1 = POLICY
    seats              u8[rows]
    request_id         u32[rows], stable and unique among outstanding requests

PythonEvaluator.forward(batch) -> ResponseBatchV2
    version            u16 = 2
    rows               u32, equal to request rows
    request_id         u32[rows], echoed in response order
    priors             f32[rows, 684], masked softmax; illegal entries are 0
    values             f64[rows], blended leaf value; ignored for POLICY rows

RustScheduler.update(results)
```

⚠ **The response is deliberately full-684 in V2.** An earlier block in this
plan said "gathered legal only" while the ownership table below said full 684;
that was an ABI contradiction. Full output is the simpler first implementation
and preserves today's Python `_masked_softmax` and `blend_value` exactly. The
request still carries CSR legal indices because Python needs the mask and Rust
needs the canonical list. A gathered response can become V2 only after
bytes-crossing-FFI is measured and with a parity gate; it is not a silent V2
layout optimization.

⚠ **That is the structural minimum, and M0 fixes nothing beyond it.** An
earlier draft also assigned ownership of the legal mask, the masked softmax,
rank masking, `blend_value` and the result scatter — **a priori, with no
measurement behind any of it.** Those are throughput questions, and this plan's
own §7 already lists the instrument that answers them (bytes crossing FFI per
batch, allocations per leaf).

**So: port the hot path first, then measure what is still in Python and decide
what else is worth moving.** Start each of these on the side it is on today and
move it only against a number:

| operation | starts | move it when |
|---|---|---|
| legal mask / gather | Rust — it reads state Rust owns, so it has nowhere else to be | — |
| priors: full 684 vs gathered-legal | **full 684 in ABI V2**, the simpler thing | bytes-crossing-FFI shows up in the profile. 684 × 32 rows ≈ 87 KB against ~6 KB gathered; changing it means ABI V3 |
| masked softmax | Python | it is measured to matter |
| rank masking, `blend_value` | Python — `training.rank_utility` lives there | ditto |
| LEAF/POLICY scatter | Rust — it owns `request_id` | — |

⚠ **Wherever each ends up, it needs a parity gate.** That requirement does not
move with the code: an operation that changes language silently changes results
unless something checks.

⚠ **"Only leaf evaluation crosses" was inaccurate** and is corrected in §3:
opponent `POLICY` requests cross too, and they were **97% of calls** before step
6 pooled them.

### M0-F — deterministic per-search seed derivation

Written down as a formula, tested in both engines, and **independent of how many
searches are in flight**. This is what keeps the wave class A.

---

## 3. The boundary — what crosses, and what does not

**Path B, as Kingdomino settled it:** Rust owns the search *and* the game state
behind an opaque handle. Python keeps the loop, the replay buffer and training.

**What crosses:** packed encoding batches out, packed priors and values back
(M0-E). Both `Ask` kinds — leaves **and** opponent policies.

Stays in Python:

* `network.py` and all of torch;
* `datagen.py` trajectories, `train.py`, the replay buffer;
* `arena.py`, the advisor, the BGA differential harness.

⚠ **Python stays the oracle, and the fidelity chain has a direction.** BGA's PHP
→ `game.py` → the Rust engine. `game.py` is validated against BGA by the
differential harness; the Rust engine is validated against `game.py`. **If they
disagree, Python wins** — including when Rust looks more sensible. A second
source of truth about the rules is the failure this project has most to lose
from.

---

## 4. Milestones

Each is a separate commit with its own gate. **Land single-threaded-correct
first; concurrency is M6.** That ordering is not taste — it is what worked on
Kingdomino and the note says to do it again.

### M1 — `RustGameState`: the engine ✅ BUILT, CORRECTED GATE GREEN 2026-08-23

**Historical gate result (2026-08-23): 8,000 games, 1,490,189 actions compared,
zero divergences, 37 minutes** (`--games 8000`, 3.6 games/s — the rate is the
*comparison's*, not the engine's: every action re-serialises both states in
Python). Plus 7 constructed positions, the read API and the boundary triple.

⚠ **Post-implementation review found two advertised checks were not actually in
that 8,000-game path.** `redeterminize` existed only in a small standalone test,
despite the harness saying "once per turn". And the information-set accessors
were checked only at turn boundaries, where live and public sheets are equal —
the exact symmetry trap that lets a live-opponent-sheet leak pass. The harness
now checks `redeterminize` once per turn and checks accessors again at the first
mid-turn seat hand-off, after seat 0's live sheet and any plan/vote can differ
from the public observation. **Corrected gate result: 8,000 games, 1,490,189
actions, zero divergences, 2,596.3s (43m16s, 3.1 games/s), with 444–445 games in
every configuration × driver cell.** M1 is signed off again. The historical run
remains strong evidence for primitive transition parity, but not for those two
claims; its 37-minute timing is retained above only as history.

`game.py` rules and `sheet.py` behind an opaque handle, including the boundary
triple, since `_end_turn` is built from it.

**What landed, and where.** M0's four load-bearing contracts were code, not just
decisions, so they landed with M1:

| piece | file |
|---|---|
| M0-A supported matrix, refused at construction | `welcome_to_rust/src/lib.rs::check_supported` |
| M0-B portable RNG, both languages | `portable_rng.py`, `src/rng.rs` |
| M0-B `Trajectory.rng`, defaulting to `cpython` | `datagen.py` |
| M0-C snapshot schema, **both directions** | `snapshot.py`, `src/snapshot.rs` |
| M0-D static-table signature | `tables.py`, `src/tables.rs` |
| the macro vocabulary (M2) | `src/macro_codec.rs` |
| the engine | `src/{game,sheet,plans,codec,constants}.rs` |
| the gate | `rust_equiv.py`, `tests/test_rust_engine_equiv.py` |
| §7 instrumentation | `rust_engine_bench.py` |

⚠ **M0-E and M0-F are still decisions, not code, and that is correct.** E is the
evaluator ABI, which nothing calls until M3; F is per-search seed derivation,
which nothing draws until M5. A, B, C and D became code here only because M1's
gate cannot run without them — a lockstep gate needs a shared generator, a
complete snapshot and an assurance that both engines hold the same tables.

**Four things the plan did not anticipate, all decided in its direction:**

* **The engine RNG had to change on the Python side too**, and `GameState.new`
  therefore takes `rng_kind`, defaulting to `portable`. `Trajectory.rng`
  defaults to `cpython`, which is what keeps the pre-port corpus replayable —
  a seed only reproduces a game together with the generator that consumed it.
* **`PortableRng.choice` is not `random.Random.choice`.** The latter draws
  through `_randbelow`'s rejection loop; this is one `next_u64 % n`, so Rust
  reproduces it in a line. Plan selection at setup is the only user.
* **The snapshot carries the generator state**, one u64, and the gate compares
  it. That is what makes a divergence in the *number of draws* fail on the step
  it happens rather than several boundaries later, when only the deal differs
  and the cause is gone.
* **Expert and solo are ported but refused.** They share the boundary code with
  the standard path, so omitting them meant writing a different engine, not a
  smaller one. M0-A refuses them as a *configuration*, by name, at construction.
* **M0-C/D are enforced when the native module imports**, not only by tests.
  A stale wheel whose table signature or snapshot version differs from Python
  raises `ImportError` before a self-play process can use it. This was tightened
  in post-implementation review; the original build compared both only inside
  tests/the equivalence harness, which did not satisfy "at load" literally.

**Gate:** lockstep equivalence over **8,000 games** at 2/3/4 seats, advanced on
and off, driven from a shared `PortableRng` seed (M0-B), comparing **after every
action** via the M0-C snapshot:

* `legal_actions()` — **in raw order**, and a set/multiplicity diff reported on
  mismatch as a diagnostic. ⚠ An earlier draft of this plan said "sorted list"
  here and "raw order matters" fifteen lines later. Raw order is what PUCT's
  first-max tie-break depends on, so raw order is what is compared;
* `phase`, `actor`, `turn`, `deck_pos`, `deck_remaining`;
* **the undrawn deck and the discard, as ordered lists** — a card census cannot
  see a card moved between them, and that move changes both the reveal
  distribution and the encoder;
* `stack_new`, `stack_old`, `expert_pending`;
* `plan_ids`, `plan_turns`;
* **`public_sheets` as well as `sheets`** — omitting the snapshot would let an
  information leak pass a green rules gate;
* every field of every `Sheet` (`_SHEET_FIELDS`);
* every field of `TurnCtx`;
* `turn_choice`, `reshuffle_next_turn`, `reshuffle_votes`, `solo_card_drawn`,
  `boundary_prepared`;
* `scores()` and `winners()` at the end.

**Three things beyond the list, added because the list is about *state* and a
port can get every field right and still answer a question differently:**

* **the read API**, at the boundary and at the first mid-turn seat hand-off —
  `visible_cards`, `next_effects` (§6.2's certainty), `table_cards`,
  `playable_slots`, `scorable_plan_slots`, and the information-set filters
  `plan_turns_for` / `reshuffle_vote_for`, plus viewer-scoped `scores` /
  `plan_scores` / `temp_scores` / every component of `score_breakdown`. The
  mid-turn checkpoint is load-bearing: only there can an opponent's live sheet
  differ from the public snapshot;
* **the boundary triple**, once per turn, on copies — `prepare` (including its
  `False`), then `sample` from the *same generator state*, then `apply`, with
  the afterstate compared before and after sampling (a chance node samples
  repeatedly, so sampling must not mutate) and the generator compared after;
* **`redeterminize`** — the primitive MCTS calls at every root, and the one
  place a port can cheat invisibly. Same permutation, same fresh child
  generator, same caller generator left behind.

⚠ **Completeness guards, not hand-written lists.** Assert the comparator's field
tuples against `dataclasses.fields(GameState)`, `fields(TurnCtx)` and
`fields(Sheet)`, excluding only the deliberately different RNG representation.
The `_SHEET_FIELDS` guard already in `mcts.py` exists because a hand-written list
of somebody else's fields rots silently; this is the same problem three times
over.

**Test:** `tests/test_rust_engine_equiv.py` — a *sample* of the gate (18 games,
one per configuration × driver, plus every constructed position). The corrected
M1 gate is `python -m games.welcome_to.rust_equiv --games 8000 --m1-only`; the
full M1+M2 gate omits `--m1-only`. The corrected M1-only run measured **43m16s**.
Keeping the switch matters: an M1 coverage rerun should not also pay for M2's
sampled apply-all-macros check. The combined corrected wall time has not been
measured, so do not reuse the historical 65-minute M2 number for scheduling. A
small 18-game all-cell sample measured **5.7s M1-only versus 10.4s with M2**;
that validates the switch, not an 8,000-game combined-wall-time extrapolation.

⚠ **A uniform-random driver is not a gate, and that was measured.** Over 60
uniform-random games: **60/60 ended on the third permit refusal**, City Plans
were validated 9 times *in total*, and the deck reformed **once**. A gate made
of those games would barely touch plan scoring, and would never see a reshuffle.
So the harness cycles three drivers — uniform, uniform-except-choosing-a-refusal,
and `GreedyBot` (~35× slower per action, 32 plan validations and 12 reforms in
12 games) — and the driver advances once per full pass over the configuration
matrix, so every configuration meets every driver.

⚠ **And played games still do not reach some rules at all.** Over 60 greedy
games, 56 ended on the third refusal, 4 filled a sheet, and **none** completed
three City Plans. Two of `isEndOfGame`'s three clauses, the queued reshuffle and
the exact-empty reform are therefore **constructed positions**, handed to Rust
through the M0-C snapshot — which is what the "both directions" in M0-C buys,
and the reason `RustGameState.from_snapshot` exists at M1 rather than M5. Each
case asserts what it is *supposed* to do at its boundary (`open`, `draws`,
`reformed`), so a case that quietly stopped being a queued reshuffle fails
instead of testing an ordinary boundary twice.

✅ **Canonical legal-action ordering — checked, and this trap does not apply.**
Kingdomino's Python order came from iterating a `set` and was unreplicable.
Verified here: `legal_actions()` and `legal_macros()` are order-stable across
repeated calls (0 instabilities over 15 games × 3 seats × 3 repeats), and the
only two set expressions in `game.py` are a one-member `.pop()` and a `sorted()`.
So Rust may reproduce list order directly — and M1's gate compares ordered lists
so that it fails loudly if that ever changes.

### M2 — `macro_codec` ✅ BUILT, GATE GREEN 2026-08-23

**Historical gate result: the same 8,000 games, 1,490,189 actions, zero
divergences, 65 minutes** — 18 configuration × driver cells at ~444 games each.
`legal_macros` and both settings of `search_legal_macros` compared at **every**
macro root; every macro offered at the turn-opening `CHOOSE_CARDS` root applied
end to end on one turn in three; all 684 primitive sequences compared
exhaustively in the test module. This remains the M2 result: the two M1 harness
omissions above do not weaken the macro-list or macro-application comparisons.

`src/macro_codec.rs`, exposed as `legal_macros`, `search_legal_macros`,
`is_macro_root`, `apply_macro`, `step_macro`, and the index arithmetic as module
functions.

**Gate:** over the same corpus, at every macro root, macro index lists identical
**in order**; every macro applies end to end identically (M0-C snapshot);
`search_legal_macros` agrees at both settings of `prune_roundabout_pass`.

⚠ **Legality is enumerated, never intersected** — `legal_macros` steps into each
playable slot and reads that child's own `legal_actions()`. Reconstructing it as
a mask intersection admits jointly-illegal pairs.

**Three notes from building it:**

* **The macro layout joined the table signature, and `SIGNATURE_VERSION` is now
  2** (`0xf584598b60c8e1f7`). The 684 indices are as much an ABI as the 357
  primitive ones — a checkpoint's policy head is indexed by them — and the 184
  surviving primitives are hashed **in order**, because the order *is* the
  mapping: a reordering that kept the same set would silently relabel 184
  logits.
* **`collapse` is deliberately not ported.** Reading a primitive trajectory as
  macro labels belongs to `datagen`, which §3 keeps in Python. Porting it would
  put a second implementation behind the training corpus for no gain.
* ⚠ **The compound-apply claim is sampled by root, never thinned by macro.**
  Multi-primitive macros exist only at the turn-opening `CHOOSE_CARDS` root. It
  offers up to ~100 of them and each comparison re-serialises two whole states
  in Python: checking every turn-opening root costs **5.4×** the M1 gate against
  **2.7×** at one turn in three (`--macro-apply-every`, default 3). Checking
  *all* the macros at fewer roots keeps the property whole; checking some macros
  everywhere would leave the hole exactly where a rare write lives. The
  remaining macros are one primitive each and ride M1's action parity. The list
  comparison, which is M2's actual subject, still runs at **every** root and
  costs almost nothing (+0.19s against a 0.49s baseline over six games).

### M3 — `encode_state` — the biggest win and the biggest risk

**✅ Built 2026-08-23.** Rust owns all four feature computations and exposes the
M0-E row layout as four packed little-endian `bytes` buffers; `rust_encoder.py`
adds read-only NumPy views without a NumPy dependency in the crate.  Python is
still the oracle, and the extension refuses to import if the encoder ABI version
or any frozen dimension differs.

**Gate:** **bit-exact** over ≥400,000 encodings from the M1 corpus, at 2/3/4
seats and every viewer, `np.array_equal` on all four arrays — not `allclose`.

**Gate result:** 400,342 encodings in 124,741 primitive states, 671 complete
games and 124,070 actions; all 18 configuration × driver cells had 37–38 games,
with **zero divergences** in 420.2s.  The focused suite also hands every M1
constructed rare position through the snapshot and separately verifies the
mid-turn live-sheet/public-snapshot leak invariant.  Full suite: 510 passed,
1 skipped; 20 Rust unit tests passed.

⚠ **Match numpy's cast chain exactly:** `int/int` divisions in **f64**, then cast
to f32, mirroring numpy's float64 → float32 array-assignment cast.

⚠ **Port the information-set safety, not just the arithmetic.**
`ENCODER_V2_SPEC` §9.3 is **two** tests: at a turn boundary the live sheet equals
the public snapshot, so a `state.sheets[p]` leak passes a boundary symmetry test
unnoticed — the leak is only visible **mid-turn**, where the snapshot legitimately
lags. Both, mutation-checked, in Rust as well as Python.

⚠ **Expect to find bugs, and treat them as training-data events.** Kingdomino's
port uncovered two real encoder defects. A fix here changes model inputs, and
checkpoints trained before it saw different data.

### M4 — `information_key`

**✅ Built 2026-08-23.** Rust emits a versioned, collision-free canonical byte
key directly from its owned state.  It is viewer-relative by construction and
is the representation M5's observation maps will own; the PyO3 method is only a
diagnostic view of those same bytes.

**Gate — constructed equivalence classes, not a random corpus.** ⚠ Information
keys almost never collide naturally, so comparing random pairs would
overwhelmingly prove that both engines say "different", and would pass even if
merging were wrong. It is also quadratic.

Build families that **must** collapse to one key:

* many `redeterminize`s of one observation;
* the two physical copies of a duplicated printed card face;
* an opponent's live-sheet mutation hidden behind the public snapshot;
* another seat's private reshuffle vote;
* another seat's hidden `ctx` mutation.

And separation mutations for **every** visible component of the key. Then compare
Python's and Rust's **group assignments** in linear time, and inside every
equal-key group assert identical encoding, identical `search_legal_macros`, and
identical terminal value where applicable.

**Gate result:** 69 deliberately constructed states formed 48 observation
groups: 21 required collisions and 43 visible-component separations.  Python's
tuple key and Rust's byte key induced exactly the same partition.  Every
equal-key group also had identical four-array encodings and
`search_legal_macros`; terminal groups had one value.  The cases include 12
redeterminizations, physical printed-card twins, the mid-turn live/public sheet
split, private votes and actor context, table order with an unchanged histogram,
and a discard-composition pair with equal raw count and deck composition.
Full suite: 513 passed, 1 skipped; 23 Rust unit tests passed.

### M5 — the search descent + progressive widening ✅ BUILT, GATES GREEN 2026-08-24

Rust owns the observation-keyed tree, transitions, M4 observation maps, f64 PUCT
statistics, progressive-widening particles and the within-turn retained subtree.
Python answers the version-1 packed
M0-E evaluator request one row at a time; M6 will coalesce those same requests.
The no-widening arm still allocates zero particle slots and remains available
with `chance_widening=None`, but it is no longer the default.

**Control Gate 1 result (before widening was promoted):** 256 matrix-balanced played-in positions × 3 independent M0-F
search tapes, 12 simulations each: **9,216 simulations, 40,257 evaluator
requests (30,407 POLICY), exact request sequence/viewers/packed encodings, visit
counts, f64 totals, observation-tree structure and chosen action; 134 terminal
leaves; zero particle slots.** The run took 59.5s. The corpus includes the last
root-player decision of a game in every 2/3/4-seat × standard/advanced cell, so
terminal handling is exercised rather than inferred from mid-game positions.
Separate fast cases cover forced-node collapse, exact within-turn re-rooting
with nonzero reused visits, Python-generated Dirichlet noise with the correctly
advanced random tape, and the zero-allocation control arm.

**Progressive-widening Gate 1 result:** 48 matrix-balanced played-in and terminal
positions × 2 independent tapes, 64 simulations each: **6,144 simulations,
21,632 exact evaluator requests (15,894 POLICY), 382 terminal leaves**, exact
request sequence/viewers/packed encodings, visits, f64 totals, full child tree,
edge traversal counts, fresh-draw counts, exact-edge flags, terminal values and
particle counts. The final run took 57.0s while the search suite ran concurrently,
so its wall time is not used as a throughput measurement. A separate 128-simulation played-in
regression requires at least one reused outcome, proves deterministic edges draw
exactly once, checks `len(outcomes) <= ceil(sqrt(traversals))`, and caps every
particle reservoir at four. Exact edges retain no particle: they replay their
cheap deterministic macros on the incoming state so a branch cannot replace each
simulation's fresh hidden deck with the first determinization it saw.

The §8.1 hedge's measured layout price on this target is **64 bytes per outcome
versus 24 bytes for the stripped open-loop record, plus one fixed 24-byte empty
particle-arena vector**. At 256 distinct outcomes the 40-byte delta is 10 KiB,
about 4.1% of M4's 249 KiB key payload before common HashMap/key overhead. The
runtime benchmark below includes it. That is a small bounded price; it does not
justify reopening the layout decision.

**Gate 2 result:** one complete two-seat advanced real-network trajectory with
default widening at identical batch width one: **55 decisions, 47 searched,
exact action/visit/total and
request fingerprints after every decision, and exact M0-C state after every
macro**, in 3.4s. This is deliberately the same batch composition; M6's wider
batches get the tolerance/reporting rule below rather than a false bit-identity
claim.

**Final package gate:** 525 Python tests passed, 1 skipped; 24 Rust unit tests
passed.

Two parity defects were found by scaling the original control gate. First,
caching terminal outcomes in the no-widening arm was value-equivalent but not
oracle-equivalent: Python stores terminal values only under widening. M5 now
populates them only in that arm. Second, `PortableRng.choices` inherited NumPy `float32`
accumulation when passed a policy array, while Rust widened each packed prior to
f64. On a wide uniform opponent policy one cumulative threshold straddled the
same draw and changed the opponent's placement. Python now casts every weight to
f64 before accumulation; a direct 331-action/256-draw cross-language regression
and the full fixed-tape gate cover it.

**Both arms are built.** With widening enabled, the edge admits
`ceil(C * traversals^alpha)` outcomes, uses a separate fresh-draw count for
empirical weights, merges by the viewer information key, reservoir-samples up to
`max_particles` concrete states per outcome, and permanently closes proven
deterministic within-turn edges. Outcome selection follows insertion order so the
portable RNG tape is identical to Python despite Rust's unordered `HashMap`.

**Gate 1 — strict, under full control.** A deterministic mock evaluator **and a
fixed random tape**. ⚠ A mock removes torch drift but *not* redeterminization or
opponent sampling; without a shared tape the trees diverge for a reason that has
nothing to do with the port. Compare visit counts, `total`, child keys and the
chosen action, over ≥256 positions × several seeds. Also assert explicitly: the
request sequence, viewer attribution per request, terminal handling, forced
collapse, and re-rooting.

**Gate 2 — real network.** Exact `trajectory_fingerprint` equality **only when
batch composition is held identical**. Otherwise report action agreement and
visit-distribution distance, with **zero tolerance** for a legal-action or viewer
mismatch. Measured, batching already moves priors by ~1e-7, which can flip a near
PUCT tie.

⚠ **f64 accumulators are necessary and not sufficient.** `Node.total` and `prior`
are f64 in Python; f32 diverges after tens of adds. But operation order, softmax
precision and tie-break rules must match too — f64 alone does not buy bit
identity.

⚠ **The root-player contract is four clauses and the port must carry all four.**
Leaves evaluated as the root player and never `state.actor`; opponents sampled as
transitions and never nodes; the backed-up scalar never negated. A rewrite is
exactly where clause 3 is lost, because `encode_state`'s viewer becomes an
argument somebody defaults.

### M6 — the global cooperative scheduler ✅ BUILT, GATE GREEN 2026-08-24

The eventual self-play CLI gets `--search-backend {python,rust}`, defaulting to
`python`. There is no self-play CLI yet; M6 exposes `native_scheduler(...)`
without inventing S2's trajectory/pool loop early.

**What M6 is:** many independent searches and games in flight, **one global
coalescer**, mixed `LEAF`/`POLICY` rows in a single call, no per-search callback,
and persistent replenishment as games finish.

⚠ **What M6 is NOT: within-search leaf batching.** Kingdomino's
`leaf_batch > 1` lesson does **not** transfer. `run_searches` advances each
generator exactly **once per round** — one suspended request per search — so
Welcome To has no within-search leaf batch at all. Adding one requires virtual
loss or collision handling, and under a root whose visit distribution *is* the
policy target that changes the target. It is a **class C strength change**
(`THROUGHPUT_LEVERS.md` §4.7) and needs its own equal-wall-clock bakeoff, not a
throughput A/B.

✅ What *does* transfer: **GIL release alone was insufficient** — Kingdomino
measured 418 evals/s at `leaf_batch=1` against 6,151 with coalescing. We already
have the coalescing evaluator; the Rust scheduler must feed it, not call it per
leaf.

⚠ **pyo3 0.28 renamed the API:** `Python::with_gil` → `Python::attach`,
`py.allow_threads` → `py.detach`. Pin the version.

**Gate:** whole suite green under both backends; `arena.paired` identical on
shared seeds by discrete fingerprint; throughput A/B on the real path in
**leaves/s and games/hour**, not batch width.

**Built.** `RustScheduler` owns fixed persistent search slots. Each independent
search runs on a Rust worker and may have exactly one outstanding request; the
coordinator releases the GIL at the wave barrier, sorts ready rows by stable
input order (never thread arrival order), packs the four batch-major encoder
buffers plus CSR legal indices, and calls `PackedNetEvaluator.forward` once per
chunk. LEAF and POLICY rows share the call. Responses are routed by globally
unique `u32` request id and may return in any order. A Python exception or bad
id wakes every worker, restores every search to its slot, and re-raises the
original error. There is no virtual loss and no within-search batch.

The fixed slots are the replenishment seam: a driver resets the seats belonging
to a completed game and immediately assigns that capacity to the next seed.
`rust_scheduler_bench.py` exercises that path over complete games. The missing
production selector is intentionally attached to S2's future loop rather than
to a benchmark-only command. The existing M5 Python/Rust exact trajectory gate
remains the backend oracle; M6 adds batch-width-1 versus batch-width-4 discrete
search fingerprints and full-game blocking-versus-scheduled fingerprints.

**Measured, production-sized network, CPU, 8 two-seat games × 16 simulations:**

| | games/hour | rows/s | Python calls | mean batch |
|---|---:|---:|---:|---:|
| M5 blocking | 2,529 | 920 | 10,478 | 1.0 |
| M6 scheduler | **6,336** | **2,305** | **2,443** | 4.3 |

That is **2.51×** on the real complete-game path with **8/8 identical discrete
fingerprints**. Batch width is reported only to explain the call reduction; the
gate is games/hour and rows/s. Full Welcome To suite: **531 passed, 1 skipped**;
Rust: **24 passed**.

#### Cloud extension — fixed workers and one deterministic broker

`RustCloudScheduler` is now the S2 production path. `PlaySession` makes the M5
descent resumable at both evaluator seams (root/leaf value requests and sampled
opponent policy requests), including a partially resolved root-to-root
transition. A persistent pool of `--scheduler-workers` OS threads advances many
sessions cooperatively; worker count and inflight-game count are independent.

All workers rendezvous at a deterministic barrier. The coordinator waits for
every active worker, sorts rows by stable game input, chunks only at
`max_batch`, calls the single Python evaluator, and routes responses by request
id. There is no timeout flush, arrival-order batch geometry, virtual loss, or
second outstanding request from one search. Consequently two workers can feed
a six-row batch (and eight workers can manage hundreds of inflight games)
without changing scalar MCTS semantics. The pool persists across `play` calls,
as do the per-game search slots and their within-turn retained subtrees.

The exactness gate compares blocking M6 with the resumable path at 1/2/4
workers: chosen action, action order, visits, totals, and final RNG state are
identical. Reordered evaluator responses, evaluator failure/slot recovery,
worker persistence, and retained-subtree rerooting are separately gated.
Current verification: **551 passed, 1 skipped** in the Python suite and **24
passed** in Rust.

`WelcomeToNet.forward_inference` evaluates only the three search outputs
(policy, rank, score) while retaining the exact checkpoint parameterization.
CUDA evaluation reuses four page-locked host staging tensors and enqueues
nonblocking H2D copies; FP32, head math, and value blending are unchanged.

Both halves expose cumulative stage profiles. Rust reports worker search,
encoding, coordinator wait, packing, Python evaluator, response decode, wall
time, wave count, calls, requests, and the batch-width histogram. Python reports
payload parsing, tensor/H2D preparation, network submission, and CPU
postprocess/synchronization. CUDA launches are asynchronous, so launch time is
reported separately and the unavoidable output synchronization is charged to
postprocess rather than hidden in a misleading "forward" timer.

Cloud calibration is a CUDA-only geometry matrix, not a CPU proxy:

```bash
python -m games.welcome_to.s2_throughput --checkpoint MODEL.pt \
  --games 2048 --inflight 128,256,512 --scheduler-workers 4,8,16 \
  --simulations 200 --out runs/welcome_to_s2/cloud_geometry.json
```

Use substantially more games than inflight slots so every arm reaches steady
state; choose by games/hour and stage counters, with discrete trajectory
agreement retained as the class-A gate.

---

## 5. Traps that transfer verbatim

* **`Vec<u8>` maps to Python `bytes`, not `list`.** `list_of_ints != bytes` is
  **always** True even element-wise equal, and *indexed* comparison falsely looks
  fine because `bytes[i]` yields an int. Wrap in `list(...)` before comparing.
  Only u8 vecs do this. It will bite `_composition_key`, which is deliberately
  int16 bytes.
* **f64, not f32, for anything accumulated** (M5).
* **numpy's cast chain** (M3).
* **pyo3 0.28 `attach`/`detach`** (M6).
* **Windows:** `PYTHONUTF8=1` for suites that print Unicode — these documents are
  full of `⚠` and box drawing, and cp1252 crashes on them. For background runs
  use `PYTHONUNBUFFERED=1` without piping through `tail`, which buffers to exit.

⚠ **One that does *not* transfer:** `leaf_batch > 1` — see M6.

---

## 6. Build and run

```bash
games/welcome_to/welcome_to_rust/{Cargo.toml,pyproject.toml,src/}

cd games/welcome_to/welcome_to_rust
maturin develop --release

# always as a module, from the repo root
python -m pytest games/welcome_to/tests -q
```

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

`Cargo.toml` starts with pyo3 only. rayon, mimalloc and sha2 arrive if profiling
asks, each recorded against the measurement that justified it — 7WD's dependency
list grew that way.

**Layout matters and should be decided at M1, not retrofitted:** fixed arrays and
bitsets for sheets and cards, an arena for nodes, reusable legal and encoding
buffers, no nested `Vec`/`HashMap` transliteration of the Python representation
unless profiling supports it.

---

## 7. Instrumentation — recorded at every milestone

⚠ **M1 and M3 are not wired into the search**, so "measure after M1" can only
microbenchmark until M6. Say which is which. Record from the start:

* state clones / make-unmake per simulation;
* allocations per leaf;
* **bytes crossing FFI per batch**;
* encoder rows/s;
* transitions/s;
* evaluator calls, rows, batch-width distribution;
* end-to-end leaves/s and games/hour, **once integrated**.

### 7.1 M1, measured — `rust_engine_bench.py`, 2026-08-23

Laptop, 2 seats, advanced, 150 uniform-random games, **engine only** — this is
the microbenchmark the section above says to label as one, not a claim about the
search:

| | Python | Rust | ratio |
|---|---|---|---|
| actions/s | 15,637 | 424,389 | **27×** |
| games/s | 141 | 3,825 | 27× |
| state clones/s | 78,176 | 2,241,650 | **29×** |

⚠ Two runs of the same benchmark gave 25.4× / 43× and 27× / 29×, on an
unquiesced laptop. The ratios are worth one significant figure and no more; the
clone rate in particular moved by a third between runs.

⚠ **Do not carry the 25.4× forward.** §1's ceiling is ~15× *including* the
encoder and the torch forward, which are untouched at M1; and 7WD's 1.99×
microbenchmark became 1.89× on the real path. The clone rate is the number worth
remembering, because it is what the fixed-array layout bought and what M5's tree
will spend.

### 7.2 M3, measured — `rust_encoder_bench.py`, 2026-08-23

Laptop, one played-in 4-seat advanced position at turn 10, 10,000 rows per
backend and viewers rotated across all four seats:

| Python | Rust | ratio |
|---:|---:|---:|
| 1,193 rows/s | 60,558 rows/s | **50.8×** |

This is still a microbenchmark: M3 is not wired into search.  The Rust number is
deliberately conservative for the then-future scheduler because it includes four
fresh byte-buffer allocations and four Python/NumPy views per row. M6 now uses
the frozen batch-major layout and retained packing buffers; that later benefit
is measured separately in §7.5 and is not retroactively claimed here.

### 7.3 M4, measured — `rust_key_bench.py`, 2026-08-23

Laptop, one played-in 4-seat advanced position at turn 8, 100,000 keys per
backend:

| Python | Rust | ratio | Rust key size |
|---:|---:|---:|---:|
| 9,465 keys/s | 550,863 keys/s | **58.2×** | 973 bytes |

The first collision-free fixed-width draft was slightly faster (598,246 keys/s)
but produced a 1,894-byte key.  Canonical varints cut retained-child key storage
by 49% for an 8% throughput cost, the right trade before M5 owns one key per
observation outcome.  At 256 simulations, the compact payload is roughly
250 KiB if every simulation creates a distinct 4-seat observation; allocator and
map overhead are not included.  This is still a microbenchmark—end-to-end search
is not measurable until M5/M6.

### 7.4 M5, measured — `rust_search_bench.py`, 2026-08-24

Laptop, 48 matrix-balanced positions × 32 simulations, blocking evaluator with
the same uniform policy and 0.25 value on both sides. Python and Rust both pay
for their own encoder and legal-policy construction; Torch is absent so this
isolates the integrated search/engine/encoder path M5 actually replaced:

| | Python | Rust | ratio |
|---|---:|---:|---:|
| simulations/s | 177.9 | 7,131.1 | **40.09×** |
| LEAF rows/s | 183.4 | 7,353.9 | **40.09×** |
| wall | 8.63s | 0.22s | **40.09×** |

Both sides issued exactly **5,922 evaluator requests** with the normal `C=1`
widening path. The retained control issued 6,958 on the same corpus, so reuse
removed 1,036 requests (**14.9%**) before M6 batching. In separate uncontended
runs, the control measured 153.8 / 5,823.3 simulations/s in Python/Rust versus
177.9 / 7,131.1 with widening. This is a blocking M5
hot-path measurement, not the self-play headline: a real network at batch width
one reintroduces fixed forward cost, while production's throughput depends on
M6 coalescing requests across games. The exact real-network trajectory gate is
the correctness measurement at this milestone; leaves/s and games/hour on the
real generation path remain M6's definition-of-done numbers.

### 7.5 M6, measured — `rust_scheduler_bench.py`, 2026-08-24

Laptop CPU, production-sized network, 8 complete two-seat games at 16
simulations per searched decision. The scheduled arm keeps eight games in
flight and replenishes completed slots; the control is the same Rust engine and
search calling the same network one row at a time.

| | games/hour | rows/s | Python calls | mean batch |
|---|---:|---:|---:|---:|
| M5 blocking | 2,529 | 920 | 10,478 | 1.0 |
| M6 scheduler | **6,336** | **2,305** | **2,443** | 4.3 |

**2.51×**, with all **8/8 discrete full-game fingerprints identical**. This is
the real-path number M5 deferred, not the M5 synthetic-evaluator microbenchmark.
It does not claim GPU geometry; rerun on the training device before choosing the
production in-flight count and `max_batch`.

**Profile consequence.** §1's shares were taken with an interpreted engine;
after M6 the remaining Python is a different mixture and must be re-profiled
before sizing another throughput lever.

**Production S2 CUDA selection (2026-08-24).** A subsequent 200-simulation
end-to-end sweep with the S0 checkpoint and 60/30/10 seat mixture scaled from
180.8 games/hour at 8 in-flight games to **961.0 games/hour at 256**. The 256
arm delivered 5,582 evaluator rows/s at mean batch 20.72 (p90 71), with only
36 MiB Torch peak allocation and no thermal issue. It is the selected default;
the fixed-size arm's drain tail makes that rate a conservative estimate for a
continuously replenished overnight run. No CPU arm was run. Batch width changed
some close discrete choices, so resumable generation pins scheduler width in
its manifest rather than treating concurrency as semantically invisible.
7WD's experience: **1.99× on a microbenchmark became 1.89× on the real path, and
+48% for a concurrency step became +21%** — earlier fixes removed the fixed cost
the later lever had been hiding.

---

## 8. Progressive widening — resolved in M5

⚠ **Correction.** An earlier draft said `chance_widening` stays in Python
permanently. That is incoherent with the objective: **if progressive widening
wins the equal-wall-clock bakeoff, a backend that cannot run it cannot be the
production backend.**

### 8.1 ✅ SUPERSEDED 2026-08-24 — build widening in M5 and make it the default

The first decision was narrower than "port widening or not": M5 could not assume
the tree never holds game states. That decision correctly reserved the particle
arena, but the follow-on choice to leave it dormant was superseded once the
no-widening mechanics were examined directly: near-unique boundary outcomes make
the search spend its budget on new reveal samples rather than depth.

The cap, the two counters and merging were additive bookkeeping. Particles were
the structural exception: the control stores **no** states in the tree, while
widening stores up to
`max_particles` concrete states per outcome. Measured: a `GameState` pickles to
**6,513 bytes** and copies in **28.7 µs** in Python, and in Rust the difference
is between a node arena of plain numbers and one that owns or indexes state
snapshots. That is a layout decision, and layout is what does not get
retrofitted.

The equal-wall-clock bakeoff remains useful for tuning `C` and
`max_particles`, but it no longer decides whether the production backend is
capable of widening. `C=1`, `alpha=0.5`, `max_particles=4` is the normal search;
`None` is retained as the measurement control.

The three layout constraints made the completed implementation additive rather
than structural:

1. **Edges are a table keyed by observation, with a fresh-draw count** — not a
   bare `(action, observation) → node` map.
2. **The transition step returns enough to key an outcome**, even when nothing
   stores one — the observation and whether the turn changed. `edge_exact` is a
   consequence of the second, and §7.1a's closure rule needs it.
3. **The node arena carries a side-table slot for particles.** It is allocated
   only with widening on; the explicit control has zero particle allocations.

The earlier hedge had a bounded layout price, measured in §7.4. Building the
particle path now avoided the later M7 rewrite that the reserved layout was
designed to prevent.

* Rust accepts both widening and the explicit control, and rejects invalid `C`,
  `alpha`, or particle caps loudly (M0-A).
* **Expert and solo** are out of training scope; reject them loudly rather than
  half-implement.
* **Dirichlet noise is supported, not deferred** (M0-A). The boundary is
  **Python generates the noise vector, Rust applies it** — cheap, keeps
  `numpy`'s Dirichlet as the single source, and needs only a parity gate on the
  application. ⚠ A "hook" that silently omits noise would make Rust self-play
  non-equivalent to Python self-play while every gate stayed green, which is why
  M0-A forbids the hook rather than the feature.

---

## 9. Order of operations

1. ✅ **M0.** Contracts first. Every one of them is cheap now and expensive
   later. **Built 2026-08-23** alongside M1, because M1's gate cannot run
   without B, C and D.
2. ✅ **M1**, and measure what is measurable (§7) — **built 2026-08-23**, §7.1.
3. ✅ **M2**, ✅ **M3** and ✅ **M4** (built 2026-08-23).
4. ✅ **M5**, both progressive widening and the explicit no-widening control,
   under §8.1's three design constraints — built 2026-08-24.
5. ✅ **M6** last — built 2026-08-24; 2.51× complete-game throughput with 8/8
   discrete fingerprints identical (§7.5).
6. **After S1**, when a checkpoint exists that search actually improves: tune
   `C`, `max_particles`, and §7.8's `noise_fresh_fraction` under equal wall
   clock. Keep `None` as a diagnostic control, not as the normal search.

**What would stop the port:** if M1's gate cannot be made to pass at 8,000 games,
do not proceed to M2 with a known divergence and a plan to fix it later. A rules
divergence that reaches training data is unrecoverable without a retrain, and
`game.py` is the only thing standing between this project and BGA infidelity.

---

## 10. Definition of done

* Every gate green, and the whole suite green under both backends.
* `--search-backend python` still works and gives the same discrete outputs — the
  Python path is not fast any more, it is the **oracle**, and it stays.
* Python can replay Rust-generated trajectories, and the pre-existing
  `cpython`-RNG corpus still replays (M0-B).
* A throughput A/B on the real generation path in leaves/s **and** games/hour,
  against §1's baseline of ~588 leaves/s (wave, 2 seats, 64 sims, CPU).
* §1's predicted gains replaced with measured ones — they are predictions, and
  are labelled as such.
