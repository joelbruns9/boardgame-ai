# Welcome To — Python → Rust port plan

The hot path is interpreted and it is now **93% of search wall clock**. This is
the plan to move it, in milestones that are each independently gated, so that a
laptop self-play shakedown is a lunch break rather than an overnight job.

**Read first:** `THROUGHPUT_LEVERS.md` §4.1 (the port is the largest lever in
this repository's history *and* the least reversible — treat it as a port with a
correctness obligation, not an optimisation), and the Kingdomino port's recorded
gotchas, which transferred verbatim to 7WD and will transfer again.

**Companion documents.** `SEARCH_SPEC.md` is what the search *is*; this is how it
gets compiled. `ENCODER_V2_SPEC.md` is the encoder contract the Rust encoder must
reproduce bit-for-bit. Neither is superseded by this.

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
policies into one call cut the forward to 6.2%, which is what leaves 93% on the
table. The two changes compound; the port is worth *more* now.

Realistic expectation, from Kingdomino's realised speedup on its ported part:

| speedup on the ported code | overall |
|---|---|
| 10× | 6.2× |
| 20× | 8.8× |
| 30× | 10.2× |

Kingdomino's headline was **1,675 → 46,560 leaves/s ≈ 28×** on the ported part.
Do not promise that here until it is measured: its profile was 51% board ops and
17% encoder, and ours is the other way round.

---

## 2. The boundary — what crosses, and what does not

**Path B, exactly as Kingdomino settled it:** Rust owns the search *and* the game
state behind an opaque handle. Python keeps the loop, the replay buffer and
training. **Only leaf evaluation crosses back**, in batches.

Stays in Python, permanently:

* `network.py` and all of torch;
* `datagen.py` trajectories, `train.py`, the replay buffer;
* `arena.py`, the advisor, the BGA differential harness;
* `chance_widening` and everything in `SEARCH_SPEC.md` §7.1a — see §6.

⚠ **The seam already exists and must not be redesigned.** `MCTS.search_gen`
suspends at every network request (`Ask.LEAF` / `Ask.POLICY`) and `run_searches`
decides when they are computed. A Rust scheduler occupies exactly that position.
Step 6 was built for this; the port does not get to change the shape of it.

⚠ **Python stays the oracle, and the fidelity chain has a direction.** BGA's PHP
→ `game.py` → the Rust engine. `game.py` is validated against BGA by the
differential harness; the Rust engine is validated against `game.py`. **If they
disagree, Python wins and the Rust side is wrong** — including when Rust looks
more sensible. A second source of truth about the rules is the failure this
project has most to lose from.

---

## 3. Milestones

Each is a separate commit with its own gate. **Land single-threaded-correct
first; concurrency is M6.** That ordering is not taste — it is what worked on
Kingdomino and the note says to do it again.

### M1 — `RustGameState`: the engine

`game.py` rules and `sheet.py`, behind an opaque handle. Includes the boundary
triple (`prepare_turn_boundary` / `sample_boundary_outcome` /
`apply_boundary_outcome`) since `_end_turn` is built from it.

**Gate:** lockstep equivalence over **8,000 games** at 2/3/4 seats, advanced on
and off. Drive both engines from the same seed with the same action sequence and
compare after *every* action:

* `legal_actions()` — as a **sorted list**, and the sets must be equal;
* `phase`, `actor`, `turn`, `deck_pos`, `deck_remaining`;
* every field of every `Sheet` (use `_SHEET_FIELDS`);
* `scores()` and `winners()` at the end;
* the full card census — nothing invented or destroyed.

**Test:** `tests/test_rust_engine_equiv.py`.

✅ **Canonical legal-action ordering — checked, and this trap does not apply.**
Kingdomino's Python order came from iterating a `set` and was unreplicable in
Rust; first-max tie-breaks in PUCT only agree if the child order agrees, so it
cost real time there. Verified here before writing this plan: `legal_actions()`
and `legal_macros()` are **order-stable across repeated calls** on the same state
(0 instabilities over 15 games × 3 seats × 3 repeats), and the only two set
expressions in `game.py` are safe — `({0,1,2} - {i,j}).pop()` in expert-only card
passing, where the set has exactly one member, and a `sorted({…})` in scoring.

So the Rust engine may reproduce Python's list order directly. **If a future
branch becomes set-derived, this stops being true silently**, which is why M1's
gate compares `legal_actions()` as an ordered list and not only as a set.

### M2 — `macro_codec`

`legal_macros`, `search_legal_macros`, `apply_macro`, `primitives_for`.

**Gate:** over the same 8,000-game corpus, at every macro root:

* the macro index **sets** are identical, and so is the order;
* every macro applies end to end in Rust exactly as in Python (compare the
  resulting state with M1's comparator);
* `search_legal_macros` agrees, both settings of `prune_roundabout_pass`.

⚠ **Legality is enumerated, never intersected** — `legal_macros` steps into each
playable slot and reads that child's own `legal_actions()`. The Rust version must
do the same walk. Reconstructing it as a mask intersection admits pairs that are
jointly illegal, which is the trap `macro_codec.py`'s header exists to warn about.

**Test:** `tests/test_rust_macro_codec_equiv.py`.

### M3 — `encode_state` — the biggest win and the biggest risk

46.6% of wall. Also the component carrying the information-set safety contract.

**Gate:** **bit-exact** over ≥400,000 encodings drawn from the M1 corpus, at
2/3/4 seats and every viewer. `np.array_equal` on every returned array, not
`allclose`.

⚠ **Match numpy's cast chain exactly.** Do `int/int` divisions in **f64** and
then cast to f32, mirroring numpy's float64 → float32 array-assignment cast.
Kingdomino needed this for bit-exactness and so will we.

⚠ **Port the information-set safety, not just the arithmetic.** `ENCODER_V2_SPEC`
§9.3 is **two** tests, not one: at a turn boundary the live sheet equals the
public snapshot, so a `state.sheets[p]` leak passes a boundary symmetry test
unnoticed — the leak is only visible **mid-turn**, where the snapshot legitimately
lags. Assert symmetry at a boundary and the absence of a leak mid-turn, both
mutation-checked, in Rust as well as Python.

⚠ **Expect to find bugs, and treat them as training-data events.** Kingdomino's
port uncovered two real encoder defects. If this one does the same, the fix
changes model inputs, and checkpoints trained before it saw different data. Say
so in the commit.

**Test:** `tests/test_rust_encoder_equiv.py`.

### M4 — `information_key`

5.3% of wall on its own, but it is the child key for every chance edge, so it is
on the descent's critical path and cannot stay in Python if the descent moves.

**Gate:** exact equality of the key over the M1 corpus at every viewer — and,
more importantly, **equality of the induced partition**: for every pair of states
in a sample, Python and Rust must agree on whether their keys match. A key that
differs in representation but partitions identically is acceptable; one that
partitions differently is not.

Also re-assert the invariants the steps 4–7 review pinned:

* every state sharing a key encodes identically (M3's encoder, both languages);
* the key moves when the whole `GameConfig` changes, and when the raw discard
  count changes;
* it does **not** move under `redeterminize`.

**Test:** `tests/test_rust_information_key_equiv.py`.

### M5 — the search descent

`Node`, PUCT `select`, `_simulate`, `_advance`, `_collapse_forced`, the
re-rooting guard, `search_gen`'s suspension points.

**Gate:** **bit-identical trees under a deterministic MOCK evaluator** — visit
counts, `total`, child keys and the chosen action, over ≥256 positions × several
seeds. The mock isolates tree logic from torch floating-point noise, which is the
only way to get a clean signal; Kingdomino used the same trick and it is the
reason its tree gate was 288/288 rather than "close enough".

Then a second pass with the real network at `simulations` ∈ {24, 128}, gated on
**discrete** outputs only — actions and visit counts via
`mcts.trajectory_fingerprint`. Never on float targets (`THROUGHPUT_LEVERS.md`
§A); they legitimately drift.

⚠ **Use f64 in Rust for anything accumulated.** `Node.total` and `prior` are f64
in Python (numpy default). f32 diverges after tens of adds and breaks
bit-identity. This was *required* to pass Kingdomino's gate and is the single
most likely way to lose a week here.

⚠ **The root-player contract is four clauses and the port must carry all four**
(`mcts.py` header). Leaves evaluated as the root player and never `state.actor`;
opponents sampled as transitions and never nodes; the backed-up scalar never
negated. A Rust rewrite is exactly where clause 3 gets lost, because
`encode_state`'s viewer becomes an argument someone defaults.

**Test:** `tests/test_rust_search_equiv.py`.

### M6 — wire in, release the GIL, coalesce

`--search-backend {python,rust}` on every entry point that searches, defaulting
to `python` until the bakeoff says otherwise.

**Gate:**

* the whole existing suite green with the Rust backend selected;
* `arena.paired` gives the same paired result under both backends on the same
  seeds (discrete fingerprints);
* a throughput A/B on the real path, reported as **leaves per second and games
  per hour**, not batch width (`THROUGHPUT_LEVERS.md` §4.7).

⚠ **GIL release alone is insufficient, measured.** Kingdomino's
`make_rust_evaluator` did one forward per call, so `mean_batch = 1` no matter
what: **418 evals/s at `leaf_batch=1` against 6,151 at 6.** A shared in-process
**coalescing** evaluator was also required.

✅ **We already have that half.** `NetEvaluator.answer_batch` takes mixed
`Ask` kinds in one forward, and `run_searches` pools across concurrent searches.
The Rust scheduler must hand batches to it, not call it per leaf.

⚠ **pyo3 0.28 renamed the API:** `Python::with_gil` → `Python::attach`,
`py.allow_threads` → `py.detach`. The old names do not exist. Pin the version.

---

## 4. Traps that transfer verbatim

Every one of these was paid for on Kingdomino and repeated on 7WD.

* **`Vec<u8>` maps to Python `bytes`, not `list`.** `list_of_ints != bytes` is
  **always** True even when element-wise equal — and *indexed* comparison looks
  fine, because `bytes[i]` yields an int. Wrap u8-vec accessors in `list(...)`
  before comparing. `Vec<u16>` and `Vec<(u8, u16)>` map to lists correctly; only
  u8 vecs become bytes. This will bite `_composition_key`, which is
  deliberately `int16` bytes.
* **f64, not f32, for anything accumulated** (M5).
* **numpy's cast chain** (M3).
* **pyo3 0.28 `attach`/`detach`** (M6).
* **`leaf_batch > 1` is essential** (M6).
* **Windows:** set `PYTHONUTF8=1` for suites that print Unicode — these
  documents and docstrings are full of `⚠` and box-drawing characters, and
  cp1252 will crash on them. For background runs use `PYTHONUNBUFFERED=1`
  without piping through `tail`, which buffers until exit.

---

## 5. Build and run

```bash
# crate lives beside the Python package, mirroring the other two games
games/welcome_to/welcome_to_rust/{Cargo.toml,pyproject.toml,src/}

cd games/welcome_to/welcome_to_rust
maturin develop --release          # into .venv

# from the repo root, always as a module
python -m games.welcome_to.train --games 5000 --epochs 4
python -m pytest games/welcome_to/tests -q
```

`Cargo.toml` starts minimal — pyo3 only. rayon, mimalloc and sha2 arrive with
M6 if profiling asks for them, not before; 7WD's dependency list grew that way
and each addition is recorded against the measurement that justified it.

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

---

## 6. What must NOT be ported yet

* **`chance_widening` and everything in `SEARCH_SPEC.md` §7.1a.** It is off by
  default, its constants (`C`, `alpha`, `max_particles`) are unmeasured, and the
  bakeoff may change its shape. `THROUGHPUT_LEVERS.md` §4.1's advice is to port a
  **settled** hot path; this one is not settled. The control-arm search is, and
  is twice-reviewed.
* **Expert and solo modes.** Out of training scope
  (`ENCODER_V2_SPEC.md` target configuration). The Rust engine may refuse them
  loudly, as `macro_codec` already does — but M1's equivalence gate should still
  cover them if the Rust engine claims to implement them at all. Refusing is
  cheaper than half-implementing.
* **The Dirichlet noise path.** `dirichlet_alpha` is `None` and §7.8's
  `noise_fresh_fraction` is unresolved pending a strength measurement. Port the
  hook, not a tuning.

---

## 7. Order of operations, and what would change it

1. **M1**, and measure. If the engine alone does not move the residual, the
   profile was wrong and this plan needs re-deriving before more is spent.
2. M2, M3, M4, M5 in order — each is a prerequisite for the next.
3. M6 last, because concurrency on top of an incorrect engine measures nothing.

**Re-measure the profile after M3.** §1's shares were taken with an interpreted
engine; once `encode_state` is compiled, the *remaining* Python is a different
mixture and the next target may not be the one this plan predicts. 7WD's
experience was that earlier fixes removed the fixed cost that later levers had
been hiding, so figures only partly transfer — **1.99× on a microbenchmark
became 1.89× on the real path, and +48% for a concurrency step became +21%**.

**What would stop the port:** if M1's equivalence gate cannot be made to pass at
8,000 games, do not proceed to M2 with a known divergence and a plan to fix it
later. A rules divergence that reaches training data is unrecoverable without a
retrain, and `game.py` is the only thing standing between this project and BGA
infidelity.

---

## 8. Definition of done

* Every milestone's gate green, and the whole suite green under both backends.
* `--search-backend python` still works and produces the same discrete outputs —
  the Python path is not fast any more, it is the **oracle**, and it stays.
* A throughput A/B on the real generation path, reported in leaves/s **and**
  games/hour, against the §1 baseline of ~588 leaves/s (wave, 2 seats, 64 sims,
  CPU).
* This document updated with the measured per-milestone gains, replacing the
  predictions in §1 — which are predictions, and are labelled as such.
