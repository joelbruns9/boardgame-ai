# 7WD advisor: unifying the advisor and training paths on Rust

**Goal: the advisor should search a real BGA position using the same engine,
encoder and searcher that self-play uses, instead of a parallel Python
implementation.**

Written to be picked up cold. Every performance number here was measured on this
machine (RTX 3070 Laptop, Windows) on 2026-08-02 and is reproducible from the
committed fixtures in `testdata/`. Where a number contradicts an earlier claim
elsewhere in the repo, this file is the later measurement.

Companion document: `ADVISOR_IMPLEMENTATION_PLAN.md` covers the BGA capture and
extension work (items A–E). This file covers only the Python↔Rust question.
**The two tracks are independent — neither blocks the other.**

**STATUS 2026-08-02: all five steps done.** The advisor now searches with the
same engine, encoder and searcher as self-play, 9.8× faster than the Python
path, agreeing with it on the answer. Remaining: CUDA for the larger model
(§5 step 5) and `force_expand_root_chance` on the Rust path (§5 step 4).

---

## 1. Why this track exists

Two reasons, and the first matters more than the second.

### 1.1 Divergence risk (the real driver)

Self-play runs on Rust. The advisor runs on Python. Any change made to one and
not the other is a silent split between "the search that produced the model" and
"the search that advises with it".

**This is not hypothetical — there is a live instance in the tree today.**
Commit `0aa0897` changed `pool.py::visible_card_names` to union
`observation.wonder_burials`, and did **not** port the same change to
`seven_wonders_rust/src/pool.rs` (which handles `buried_cards` only, line ~44).

It is currently harmless, and it is worth understanding exactly why, because the
reasoning is what makes it safe to leave for now:

* In an engine-built state, a taken tableau slot keeps its `card_name`, so
  `visible_card_names` already sees a buried card through `observation.tableau`.
  The union is a no-op there — **measured: 0 differences across 2,243 PLAY_AGE
  observations in 40 random games.**
* It only bites on a *reconstructed* state, where `bga_extract._tableau` sets
  `card_name: None` for an emptied slot. Rust cannot construct such a state
  today (see §3), so the divergence is unreachable.
* Therefore `test_rust_engine_equiv.py` still passes, and would keep passing
  even if the two implementations disagreed on this point.

That last bullet is the uncomfortable one: **the existing equivalence tests
cannot see this class of drift**, because they all start from a fresh deal and
replay a prefix.

### 1.2 Performance (secondary, and smaller than it first appears)

See §2. Short version: the Python tree walk is the dominant cost, so only moving
the *searcher* — not the encoder alone — is worth much.

---

## 2. Measurements

All on the committed Great Library capture
(`testdata/bga_892846644_greatlibrary.json`) unless stated.

### 2.1 Where advisor search time actually goes

Profiled by wrapping `search.encode` and the evaluator, 2,000 sims after a
200-sim warmup:

| model | sims/s | encode | net forward | tree walk | max gain from a *free* encoder |
|---|---|---|---|---|---|
| 128×4, 1.0M (current) | 621 | 25.6% | 11.4% | **63.0%** | 1.34× |
| 384×6, 11.4M (target) | 327 | 16.1% | **44.1%** | 39.8% | 1.19× |

**Read this before optimising anything.** The Python tree walk dominates today.
With the larger net, net forward becomes co-dominant. Encoding is never the
bottleneck.

### 2.2 Evaluations per simulation — why the above is not obvious

Not every simulation evaluates the net; many terminate or land on an expanded
node.

| position | sims/s | NN evals per sim |
|---|---|---|
| Age III `playerTurn`, 2 cards left | 2,782 | **0.02** |
| Great Library pending choice | 379 | **0.28** |
| Great Library, during the profile run above | 621 | 0.18 |

Consequences:

* **Throughput is position-dependent by ~7×.** A near-terminal position barely
  touches the net. Any single sims/s figure must name its position.
* **Throughput rises as a tree saturates.** Streaming the Great Library position
  averaged ~1,900 sims/s over 50k sims, versus 621 sims/s over the first 2k —
  later simulations increasingly re-walk existing nodes.
* Quoting a bare "sims/s" for this advisor without both caveats is misleading.

### 2.3 Encoder: Python vs Rust

Measured on a mid-Age-I position reachable by prefix replay (so Rust could
build it at all):

```
python encode():   2.4011 ms/state  ->    416 states/s
rust   encode():   0.0472 ms/state  -> 21,176 states/s      51x
```

The 51× is real **and mostly unusable on its own** — per §2.1 it buys 1.19–1.34×
end to end.

### 2.4 Net forward: CPU vs CUDA, by batch size

`Evaluator.evaluate` on pre-built encodings (so this excludes encoding):

| model | device | batch | ms/call | evals/s |
|---|---|---|---|---|
| 128×4 1.0M | cpu | 1 | 3.78 | 265 |
| 128×4 1.0M | cuda | 1 | 3.37 | 297 |
| 128×4 1.0M | cuda | 32 | 10.79 | 2,964 |
| 384×6 11.4M | cpu | 1 | 10.33 | 97 |
| 384×6 11.4M | **cuda** | **1** | 6.47 | **155** |
| 384×6 11.4M | cpu | 32 | 186.66 | 171 |
| 384×6 11.4M | **cuda** | **32** | 18.39 | **1,740** |

* At **1.0M**, CPU and CUDA tie at batch 1 — launch overhead swamps a tiny net.
  This is why `extension_7wd/README.md` specifies `SWD_ADVISOR_DEVICE=cpu`.
  **That recommendation is scoped to the current model and must be revisited for
  the target model.**
* At **11.4M**, CUDA wins even unbatched (1.6×) and by 10.2× at batch 32.
* Batching is worth ~10× on the net *if realized width holds* — see §5.3.

---

## 3. The blocker: Rust cannot load a scraped position

**There is no Python-facing way to inject an arbitrary mid-game state into the
Rust engine.** This gates the encoder, the searcher and batching equally; there
is no cheap subset.

What exists:

| entry point | takes | reaches |
|---|---|---|
| `RustGame.__new__` (`lib.rs:204`, the only `#[new]`) | a fresh **deal** (decks, wonder groups, guilds, removed cards) plus `library_draws` | the start of a game |
| `rust_bridge.rust_game_from_prefix` | `(seed, first_player, prefix)` | any position **reachable by replaying actions** |
| `rust_bridge.rust_setup` | a `GameState` | docstring: *"Constructor kwargs … from a **fresh** game"* |
| `state.rs:381 restore()` | a Rust `GameState` | Rust-internal (make/unmake); **not exposed to Python** |

The advisor's position comes from `advisor_scrape.determinize_observation`: a
public observation with hidden information invented by the determinizer and **no
action history**. There is no seed and no prefix, and none can be recovered.

So the advisor's replay wire (`{seed, first_player, prefix}`, used by the lab UI)
*can* reach Rust today. The BGA scrape wire cannot.

### 3.1 Second obstacle: Great Library chance outcomes

Rust requires Great Library draws to be supplied up front. Replaying random play
into Rust panics:

```
panicked at src\engine.rs:680:30: great library draw outcome missing from chance log
```

`rust_game_from_prefix` handles this with a Python prepass that discovers the
draws first. A searcher running from an injected live position needs the
equivalent, or a way to resolve those draws on demand.

### 3.2 Third obstacle: the Rust search entry points are one-shot

`closed_search_net`, `closed_search_resumable_net`, `closed_search_batched_net`
(`lib.rs:442–757`) take `sims` and return a result tuple. "Resumable" refers to
the internal arena implementation, **not** a Python-facing handle over a
persistent tree.

The advisor streams: `_ClosedHandle.advance(chunk)` deepens one tree across many
calls, which is what makes the panel's win probability refine while you think
(`ADVISOR_IMPLEMENTATION_PLAN.md`, "Continuous search"). Naively calling a
one-shot Rust search per chunk would **rebuild the tree every time** and destroy
that property.

---

## 4. What is and is not already equivalent

Established and tested — do not re-litigate:

| layer | status | evidence |
|---|---|---|
| Engine | byte-exact | F1: 10k games / 665k decisions |
| Encoder | bit-exact | F2 |
| Closed Gumbel searcher | bit-identical, incl. on a real net | F3.0–F3.4 |

**But the advisor does not use the Gumbel searcher.** Two different search
policies are in play *within Python*:

| | self-play | advisor |
|---|---|---|
| root selection | Gumbel top-k + sequential halving (`search.py:360` default, `:815`) | plain PUCT (`search.py:494 descend()`) |
| tree | fixed budget per search | persistent, deepened across `advance` calls |
| exploration noise | yes (training data) | none (best move) |

This is **deliberate and correct** — training needs exploration and policy
targets; analysis wants the strongest move and must stream. It is standard
AlphaZero practice to train on visit distributions and play with PUCT. It is
recorded here because it is a larger behavioural difference than anything
Python-vs-Rust, and a reader worried about divergence should know it is
intentional rather than an oversight.

Usefully, **Rust already supports both**: `tree::SearchConfig` carries a
`puct_root` field (`lib.rs:531` sets it `false`). So unifying does not require a
second searcher — only exposing the knob.

---

## 5. Plan

Ordered so that the thing which makes everything else trustworthy comes first.

### Step 1 — DONE 2026-08-02 (`8eada50`)

`pool.rs` now unions `wonder_burials` into `visible_cards`, matching `pool.py`.
Verified a no-op on the Rust side too: over **1,118 PLAY_AGE states — 998 of
them carrying burials — 0 unseen-pool mismatches**, and `test_rust_engine_equiv`
(32 tests) stayed green.

### Step 2 — DONE 2026-08-02 (`44624f5`)

`RustGame.from_state` + `rust_bridge.rust_state` inject a whole `GameState`.
`test_rust_state_injection.py` is the gate: an injected state fingerprints
byte-identically to the same position reached by replay, over **3,204 positions
across all four phases**.

**It caught a real bug on its first run — 2,635 of 3,204 mismatched.** Rust's
fingerprint pushes a tableau slot's `card_id` *unconditionally*, including slots
whose card has been taken, and Python's `TableauCard` likewise keeps
`card_name` after removal. The serializer was zeroing them. That retention is
the same fact behind step 1: it is precisely how a card buried under a wonder
still reads as visible to the unseen-card pool. Two independent bugs, one
underlying property.

The scraped `testdata` positions — which replay cannot construct at all — now
inject and agree with Python on `legal_action_indices` and on the unseen pool per
back type. That is the first time the Rust side has been checked against a state
it did not build itself.

Enum fields cross as declaration indices, with a test pinning both orders.

### Step 1 (original) — Port the `pool.rs` divergence (small)

Union `wonder_burials` into the Rust `visible_card_names` equivalent
(`seven_wonders_rust/src/pool.rs:~44`), matching `pool.py:62-79`.

Behaviour-neutral today (§1.1), so it cannot break the existing gates. Do it
first so the current drift does not outlive the memory of why it exists.

### Step 2 — State injection API + the test that makes it safe

A pyo3 constructor accepting a full mid-game `GameState`. Roughly 25 fields:
phase, active player, age, first player; per-city coins / buildings / wonders /
built wonders / progress tokens / claimed science pairs; the 20-slot tableau
(card, revealed, present); discard; `age_decks` and `removed_age_cards`;
available and unused progress tokens; wonder groups, offer, round, pick index,
unused wonders; selected and unused guilds; conflict position and remaining
military tokens; pending choice (kind, player, options, `consume_all_options`);
`pending_extra_turn`; `pending_shields`; `wonder_burials`; `buried_cards`.

**The test is the deliverable, not the API.** For every position reachable by
replay, both routes must agree:

```
inject(python_state).fingerprint() == replay(seed, prefix).fingerprint()
```

over thousands of positions spanning all three ages, every pending-choice kind,
wonder burials, and a completed draft. `RustGame.fingerprint()` (`lib.rs:264`)
already exists for exactly this kind of comparison.

Why this specific test matters: it is the **only** thing that would have caught
the `pool.rs` drift in §1.1, because it is the only test that exercises a state
Rust did not build itself.

Then extend the injected corpus to positions Rust *cannot* replay — the
determinized scrape states from `testdata/bga_892846644_*.json` — and assert
Python and Rust agree on `legal_action_indices`, `unseen_pool` and `encode`.

### Step 3 — DONE 2026-08-02: no prepass needed

Search on an injected position handles Great Library by itself. `apply_index`
panics without a pre-supplied draw, which is why the *replay* path does a Python
prepass -- but the **searcher** enumerates and samples those outcomes from
`pool.offboard_progress` (`chance.rs:207,389`) and feeds them in. An injected
live position, including one already sitting on a Great Library choice and
injected with `library_draws` empty, searches as-is.

Verified on both scraped fixtures. On the Age III position Rust's mock search
picks **action 61** -- the same "Build: Study" the Python advisor chose.

### Step 4 — DONE 2026-08-02 (`c155848`, `b87f054`)

`RustPuctSearch` is a pyclass holding a `SearchSession` across calls, so one tree
deepens over many `advance()` calls — the property the streaming panel depends
on. Fixed at `puct_root=true`.

Gated by `test_resumable_handle_matches_the_one_shot_puct_search`: visits per
legal action and total sims across 4 games × 2 seeds × 3 chunking patterns,
including `(1, 47)`, so **chunk boundaries provably do not change the tree**.
The narrower gate was the right one — `test_rust_puct_root_matches_python`
already gates Rust `puct_root` against Python, so equivalence follows by
transitivity without re-deriving it.

Needed `RustPuctSearch.open_mock`: the one-shot search runs Rust's internal
`MockEval`, which a Python adapter cannot reproduce byte-for-byte.

Still open: `force=False` only. The resumable path cannot force-expand the root
chance layer without the F4.5 forced-child cache, so that half of the one-shot
gate has no resumable counterpart, and an explicit `force_expand_root_chance`
request falls back to the Python searcher.

#### The two constraints, and what became of them

Feasible: `SearchSession` (`tree_resumable.rs:799`) is **owned** -- no lifetime
parameters, it owns its `Arena` -- so it can live in a `#[pyclass]` across calls.
`begin_search_from_root` touches `&GameState` only at construction. That is the
resumable handle; what is missing is the pyo3 wrapper and a snapshot readout.

But `begin_search_from_root` rejects two things the advisor uses today:

1. **`puct_root && leaf_batch > 1`** -- *"the root would select under WU virtual
   loss"*. The advisor wants PUCT root (§4). So **root-level leaf batching and
   PUCT root are mutually exclusive** in the current Rust searcher. This is a
   direct answer to "should we leaf batch": at the root, not without either
   accepting Gumbel root selection or teaching the root to select under virtual
   loss. Batching deeper in the tree is unaffected.
2. **`force_expand_root_chance`** -- *"requires the F4.5 forced-child cache"*.
   The advisor currently defaults this to `True`
   (`advisor_adapter.open_search`). Either that default changes for the Rust
   path, or F4.5 lands first.

Neither is fatal, both change the plan. Decide the root-selection question
before writing the wrapper, because it determines whether the advisor keeps
PUCT-root semantics (and gives up root batching) or moves to Gumbel root (and
changes what the advisor's numbers mean, §4).

### Step 3 (original) — Great Library prepass (§3.1)

Decide and document how an injected position resolves `GREAT_LIBRARY_DRAW`:
either supply the enumerated outcomes at injection (mirroring
`rust_game_from_prefix`), or let the Rust searcher sample them. The Python
searcher enumerates from `pool.offboard_progress`; whatever is chosen must match
it, or the two searchers explore different chance supports.

### Step 4 — Resumable Rust handle (§3.2)

A pyo3 object holding a tree across calls, with `advance(chunk_sims) -> snapshot`
and `puct_root: true`. This is what preserves streaming. Without it, do not
migrate the advisor — a non-streaming advisor is a worse product regardless of
throughput.

### Step 5 — DONE 2026-08-02 (`ca6f6b7`, `7e9c77a`)

**9.8× over the Python searcher.** Same net, same position, same process:

```
python searcher                980 sims/s   1.00x
rust  searcher leaf_batch=1   2732 sims/s   2.79x
rust  searcher leaf_batch=32  9610 sims/s   9.81x
```

Through the adapter fully wired: 1,829 → 18,513 sims/s.

**The scaling is all in the evaluation boundary, not the tree.** Two changes were
needed and only the pair works:

1. *Relaxing the PUCT-root batching guard* (`ca6f6b7`). It was our own guard, not
   a missing algorithm: `Arena::select` already folds in-flight counts into both
   PUCT terms, and it is the same function the root calls under `puct_root`.
   Gumbel is not entangled either — under `puct_root` the gumbel vector is
   hard-zeroed and sequential halving is bypassed. Opt-in via
   `begin_search_from_root_virtual_loss`, a separate entry point so all of
   self-play keeps the strict behaviour untouched.
2. *A batched evaluation boundary* (`7e9c77a`). `eval::PyBatchEval` already
   existed (F4.4) but was reachable only through a threaded worker.
   `rust_bridge.rust_batched_net_adapter` is the Python counterpart: one crossing
   and one forward pass per wave.

**On its own, (1) buys nothing** — `PyEval` does not override `evaluate_batch`,
so a wave was evaluated one state at a time. Measured across `leaf_batch` 1..16
on the scalar bridge: 1.00×–1.07×, pure noise. With the batched adapter the same
sweep gives 1.00 / 1.57 / 2.16 / 2.68 / 3.53 / 3.93× at 1/2/4/8/16/32.

**Quality held** — the evidence the guard was written without. Across
`leaf_batch` 1..32 the top action never changed and its visit share moved
0.941 → 0.930. Q on the top action agrees to three decimals across the Python
searcher, Rust unbatched and Rust batched (+0.9801 / +0.9810 / +0.9811).

Defaults: `leaf_batch=16`; `options={"leaf_batch": 1}` restores unperturbed root
selection, `search_impl="python"` the old searcher.

**Not done:** CUDA. At 1.0M parameters CPU and CUDA tie (§2.4); this must be
re-measured when the 11.4M model lands, where CUDA was 10.2× at batch 32 — and
batching now actually reaches the net, so that figure should finally transfer.

### Step 5 (original plan) — Performance, once correctness is nailed down

Only now: batched leaf evaluation (`closed_search_batched_net` already exists),
CUDA for the target model, and the encoder — which comes along for free rather
than being pursued for itself.

---

## 6. Traps, and mistakes already made

**Do not repeat these.**

1. **"The Python encoder is the bottleneck" — wrong.** An earlier revision of
   `ADVISOR_IMPLEMENTATION_PLAN.md` claimed the 2.39 ms encoder imposed a hard
   ~418 leaves/s ceiling and was "a bigger lever than CUDA or batching". That
   assumed ~1 evaluation per simulation. Measured, it is **0.18–0.28** (§2.2), so
   encoding is 16–26% of runtime and worth 1.19–1.34×. The claim was reasoned
   from an isolated microbenchmark instead of a profile.

2. **`evaluate_states()` re-encodes on every call.** Benchmarking with it
   measures the Python encoder, not inference. Use `Evaluator.evaluate()` with
   pre-built encodings.

3. **Never quote sims/s without the position.** §2.2: a 7× spread, and it drifts
   upward as the tree saturates.

4. **CPU-vs-CUDA conclusions do not transfer across model sizes.** They tie at
   1.0M and differ by 10× at 11.4M with batching.

5. **Existing equivalence tests cannot catch injection-path drift** (§1.1). Any
   Rust work here must extend the corpus, not lean on the current gates.

---

## 7. Open questions

* **Realized batch width.** `search.py:388-393` records that the Rust searcher's
  leaf waves held **1.19** realized width under blocked candidate ordering. A
  requested batch of 32 may not materialize, especially in narrow decisive
  positions — the forced-win position put 53,599 of 53,757 visits on one move.
  Measure realized width before assuming the 10× of §2.4 transfers.
* **Does PUCT-root Rust match PUCT-root Python bit-for-bit?** F3 verified the
  *Gumbel* path. The `puct_root: true` path has no equivalent gate.
* **Virtual loss.** If batching needs it, it perturbs selection. The distortion
  shrinks as visits accumulate, so it is safer at the advisor's sim counts than
  at self-play's — and the advisor is off the training path, making it a safe
  place to experiment. Not needed if wave batching across root candidates
  suffices.
* **Is unification worth it for the current 1.0M model?** Probably not on
  performance alone (1.34× ceiling from the encoder; the tree walk needs the full
  searcher). The argument is divergence risk plus readiness for the 11.4M model.

---

## 8. Decision log

* **2026-08-02** — Considered doing the Rust encoder alone. Rejected: blocked on
  the same state-injection work as everything else (§3), and worth only 1.19–1.34×
  (§2.1).
* **2026-08-02** — Considered CUDA + batching in Python as the cheap path.
  Rejected as the *primary* direction: it is the largest single lever for the
  11.4M model, but it deepens the Python/Rust split by building advisor-only
  performance machinery. Still the right fallback if unification is deferred.
* **2026-08-02** — Chose unification, sequenced correctness-first (§5), on the
  grounds that one implementation is worth more than the throughput.
