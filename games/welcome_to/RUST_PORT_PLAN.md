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
| widening excluded "permanently" | if it wins the bakeoff, the Rust backend cannot be the production backend | §8.1 — resolved: **design for it, do not build it** |
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

**Not implemented *yet*** — refused only until the milestone that adds it:
`chance_widening`, which §8.1 committed M5's layout to admitting and which
becomes M7 if it wins the bakeoff. The refusal is still required in the meantime:
a backend that silently ignored `chance_widening = 1.0` would run a different
search than the one it was asked for.

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

Specify, before M3:

```
RustScheduler.step()  -> a packed request batch
    sheet_planes, sheet_scalars, viewer_plane, global_scalars
                       four contiguous f32 buffers, batch-major
    legal              packed legal indices + CSR-style offsets, per row
    kind               u8 per row: LEAF | POLICY
    seats              u8 per row
    request_id         u32 per row, stable across the round trip

PythonEvaluator.forward(batch) -> packed results
    priors             gathered over each row's LEGAL indices only, never 684
    values             f32, LEAF rows only

RustScheduler.update(results)
```

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
| priors: full 684 vs gathered-legal | **full 684**, the simpler thing | bytes-crossing-FFI shows up in the profile. 684 × 32 rows ≈ 87 KB against ~6 KB gathered, so the *option* is worth keeping open in the layout |
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

### M1 — `RustGameState`: the engine ✅ BUILT, GATE GREEN 2026-08-23

**Gate result: 8,000 games, 1,490,189 actions compared, zero divergences, 37
minutes** (`--games 8000`, 3.6 games/s — the rate is the *comparison's*, not the
engine's: every action re-serialises both states in Python). Plus 7 constructed
positions and, once per turn, the read API, the boundary triple and
`redeterminize`.

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

* **the read API**, once per turn — `visible_cards`, `next_effects` (§6.2's
  certainty), `table_cards`, `playable_slots`, `scorable_plan_slots`, and the
  information-set filters `plan_turns_for` / `reshuffle_vote_for`, plus
  viewer-scoped `scores` / `plan_scores` / `temp_scores` / `score_breakdown`;
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
one per configuration × driver, plus every constructed position, ~5s). The gate
itself is `python -m games.welcome_to.rust_equiv --games 8000`, ~35 minutes.

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

**Gate result: the same 8,000 games, 1,490,189 actions, zero divergences, 65
minutes** — 18 configuration × driver cells at ~444 games each. `legal_macros`
and both settings of `search_legal_macros` compared at **every** macro root;
every macro applied end to end at one root in three; all 684 primitive
sequences compared exhaustively in the test module.

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
* ⚠ **The apply claim is sampled by root, never thinned by macro.** A root
  offers up to ~100 macros and each comparison re-serialises two whole states in
  Python: checking every root costs **5.4×** the M1 gate against **2.7×** at one
  root in three (`--macro-apply-every`, default 3). Checking *all* the macros at
  fewer roots keeps the property whole; checking some macros at every root would
  leave the hole exactly where a rare macro lives. The list comparison, which is
  M2's actual subject, runs at **every** root and costs almost nothing (+0.19s
  against a 0.49s baseline over six games).

### M3 — `encode_state` — the biggest win and the biggest risk

**Gate:** **bit-exact** over ≥400,000 encodings from the M1 corpus, at 2/3/4
seats and every viewer, `np.array_equal` on all four arrays — not `allclose`.

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

### M5 — the search descent

**Built open-loop, designed for widening — §8.1's three constraints are part of
this milestone, not a later concern:** an observation-keyed edge table with a
count field, a transition step that returns the observation and whether the turn
changed, and an unallocated particle slot in the node arena. Verify the cost is
small; if it is not, re-open §8.1 rather than paying it quietly.

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

### M6 — the global cooperative scheduler

`--search-backend {python,rust}`, defaulting to `python`.

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

**Re-measure the profile after M3.** §1's shares were taken with an interpreted
engine; once the encoder is compiled the remaining Python is a different mixture.
7WD's experience: **1.99× on a microbenchmark became 1.89× on the real path, and
+48% for a concurrency step became +21%** — earlier fixes removed the fixed cost
the later lever had been hiding.

---

## 8. Not in the first pass — and *not* "permanently"

⚠ **Correction.** An earlier draft said `chance_widening` stays in Python
permanently. That is incoherent with the objective: **if progressive widening
wins the equal-wall-clock bakeoff, a backend that cannot run it cannot be the
production backend.**

### 8.1 ✅ DECIDED 2026-08-23 — build open-loop, design for widening

**The decision that matters is narrower than "port widening or not". It is:
*may M5 assume the tree never holds game states?*** Answer: **no.**

Everything else about widening — the cap, the two counters, merging — is
bookkeeping that can be added later without disturbing anything. Particles are
not. Open-loop stores **no** states in the tree; widening stores up to
`max_particles` concrete states per outcome. Measured: a `GameState` pickles to
**6,513 bytes** and copies in **28.7 µs** in Python, and in Rust the difference
is between a node arena of plain numbers and one that owns or indexes state
snapshots. That is a layout decision, and layout is what does not get
retrofitted.

**Why this is not decided on evidence.** Deciding properly needs an
equal-wall-clock strength A/B, and that needs a checkpoint where **search
demonstrably beats no-search**. S0 is not it: its value head has score R² 0.200,
and widening's entire product is *depth* (1.42 → 2.14 leaves), which only pays if
the values at those leaves are informative. A bakeoff run now would most likely
return a null, and **a null from an uninformative arm is not evidence** —
`THROUGHPUT_LEVERS.md` §4.7. The real decision belongs after **S1**.

**So M5 builds open-loop only, under three design constraints** that make
widening additive rather than structural:

1. **Edges are a table keyed by observation, with room for a count** — not a
   bare `(action, observation) → node` map. `Outcome.count` is then a field that
   stays zero, not a new indirection.
2. **The transition step returns enough to key an outcome**, even when nothing
   stores one — the observation and whether the turn changed. `edge_exact` is a
   consequence of the second, and §7.1a's closure rule needs it.
3. **The node arena carries a side-table slot for particles that is never
   allocated** while `chance_widening` is `None`. Zero cost off, and no layout
   change to turn on.

⚠ **This is a hedge with a bounded price, and the price should be checked.** If
implementing those three costs more than a small fraction of M5, say so and
re-open the choice rather than paying it silently. The alternative on the table
was to build strictly open-loop and treat widening as a later M7 rewrite; that
bets widening loses a bakeoff nobody has run.

* Until implemented, **Rust rejects `chance_widening != None` loudly** (M0-A).
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
3. ✅ **M2** (built 2026-08-23), then M3, M4 in order.
4. M5, open-loop, under §8.1's three design constraints. **No longer blocked on
   the widening question** — that decision was taken as "design for it, do not
   build it", precisely so this step does not wait on a measurement that cannot
   be made yet.
5. M6 last — concurrency on top of an incorrect engine measures nothing.
6. **After S1**, when a checkpoint exists that search actually improves: run the
   widening bakeoff (and §7.8's `noise_fresh_fraction`, which waits on the same
   thing). If widening wins, it becomes M7 and the constraints above are what
   make that a feature rather than a rewrite.

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
