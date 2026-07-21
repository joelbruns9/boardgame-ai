# Review request — F3.4 real-net evaluator bridge

Companion to `PHASE_F.md` and `F3_SEARCHER_REVIEW_REQUEST.md` (which covered the
F3.0/F3.2/F3.3 searcher, already reviewed + hardened). This asks for a focused
review of the one remaining unreviewed piece: **F3.4**, the bridge that runs the
Rust searcher on the real neural net.

## 1. Scope

Commit `2200c8f`. Files:
- `seven_wonders_rust/src/eval.rs` — `PyEval` (real-net `Eval` impl).
- `seven_wonders_rust/src/lib.rs` — `RustGame.closed_search_net(adapter, …)`.
- `seven_wonders_rust/src/tree.rs` — `state_actor` made `pub(crate)` (reused).
- `test_rust_engine_equiv.py` — `_make_net_adapter`, `test_closed_search_net_matches_python`.

Reference: `search.py::GumbelMCTS._evaluate`; `inference.py::Evaluator`;
`encoder.py` (`Token`/`Encoding`/`TokenType`).

## 2. What it does

The F3 searcher is generic over an `Eval` trait. `MockEval` gated the search
logic bit-exactly (F3.2/F3.3). **F3.4 adds a real-net `Eval`** so the searcher can
run on the actual network and be validated against Python end-to-end.

- `PyEval` holds a Python callable `adapter`. On each leaf `evaluate(state)`:
  terminal → `(terminal_value_p0, [])`; else it encodes with the **F2 Rust
  encoder**, gets `legal`/`actor`, and calls `adapter(tokens, actor, legal)` under
  `Python::attach`, receiving `(value_actor, priors)`. It returns
  `(sign(actor)·value_actor, priors)`.
- The Python `adapter` rebuilds an `Encoding` from the Rust tokens, runs
  `Evaluator.evaluate`, and returns exactly what `_evaluate` computes:
  `value_actor = float(wdl[0] - wdl[2])`, `priors = [float(p) …]`.
- `closed_search_net` runs the same `search_closed` (Gumbel root, force-expansion)
  as `closed_search`, just with `PyEval` instead of `MockEval`.

**Design intent (and the reviewer's standing constraint):** this is a **scalar
per-leaf correctness bridge**. F4 replaces the boundary with leaf coalescing +
GIL release for the ≥20× throughput gate — and that batched path is deliberately
**not** bit-identical to the sequential reference (it evaluates several leaves
before backprop), so it is validated by throughput/agreement, not this gate.

## 3. What is already gated (please don't re-verify by hand)

`test_closed_search_net_matches_python`: with a real `SWDNet`, the Rust searcher
via `closed_search_net` **matches** Python's `GumbelMCTS` on the same `Evaluator`
— chosen action, `sims`, `gumbel_topk`, per-action visits, and state fingerprints
**exactly**; action/root value, policy target, and node values **within 1e-9** (a
comparison strong enough to catch the f32-vs-f64 subtraction subtlety) — across
sims/seeds × force-expansion off/on, over play-age positions in 3 games.

So the plumbing *is* proven equivalent for the tested positions. Focus elsewhere.

## 4. Focus areas

### Correctness (the crux is the float plumbing)
1. **`value_actor` computation.** The gate's bit-identity rests on: (a) the Rust
   encoder produces byte-identical tokens to Python (F2), so the reconstructed
   `Encoding` yields the same tensor and net output; (b) **all** value/prior
   arithmetic happens in the Python adapter (`float(wdl[0]-wdl[2])`, `float(p)`),
   mirroring `_evaluate`, and Rust only applies the integer `sign`. Please confirm
   there is no Rust-side f64 arithmetic on net outputs that could diverge from
   Python's f32-subtract-then-widen (`float(a-b)` ≠ `float(a)-float(b)`), and that
   the sign convention (`value_p0 = value_actor` for actor 0 else negated) matches
   `_evaluate`.
2. **`Encoding` reconstruction.** `Token(list(TokenType)[ti], eid, aid,
   tuple(feats))` and `Encoding(actor, tokens)` — is this faithful, including the
   `TokenType`-by-index mapping and `actor` (does the tensor build depend on
   `Encoding.actor`, and is `state_actor` the right value to pass)?
3. **Net determinism.** The gate assumes the `Evaluator` is deterministic (eval
   mode / no dropout) and that calling it twice (Python `_evaluate` + the adapter)
   on the same input gives identical outputs — no internal batching, caching,
   autocast, or nondeterministic kernels. Confirm, since the whole gate depends on
   it.

### Robustness / error handling
4. **Panics across FFI.** `PyEval::evaluate` uses `.expect()` on the adapter call
   and `extract`. A malformed adapter (wrong return shape, exception) **panics**
   through the pyo3 boundary rather than raising a `PyErr`. Acceptable for a
   gate-only bridge, or should it propagate (like `apply_with_chance` now does)?
5. **GIL / re-entrancy.** `evaluate` re-acquires via `Python::attach` while the
   search is already running under the GIL (called from Python). Confirm this is
   correct/cheap here, and — importantly — that it does not bake in an assumption
   that blocks F4's plan to **release** the GIL during tree work and re-acquire
   only for batched evals.

### Gate coverage (the recurring F1a/F2.3 lesson)
6. The real-net gate runs only on **play-age** positions (like F3.3's mock gate).
   It does not exercise draft / age-boundary roots (WONDER_GROUP_REVEAL,
   AGE_DEAL) **with the real net**, nor assert force-expansion actually engaged.
   The mock-oracle tests (`…force_expansion_coverage`, `…age_deal_coverage`) do
   cover those, but under `MockEval`, not `PyEval`. Is mock coverage of those
   branches + play-age coverage of `PyEval` sufficient, or should the net gate add
   a draft/force position?

### Throughput (F4 — flag, don't fix)
7. Every leaf does a Python round-trip (`Encoding` rebuild + net forward) with the
   GIL held throughout, and re-runs the Rust encoder per leaf (F2's per-encode
   cost, `PaymentContext` deferred). This is exactly what F4 must replace with
   batched coalescing. Confirm the `Eval` trait can grow a batch method and the
   searcher can collect leaves without disturbing the proven sequential path.

## 5. Known limitations / out of scope
- Scalar per-leaf boundary by design; batching + GIL release + ≥20× throughput is
  **F4** (Phase F exit).
- The batched fast path will not be bit-identical to the sequential reference (by
  design); it needs its own validation strategy (throughput + statistical
  agreement / blunder-rate), not this gate.
- Physical shared-crate extraction is **F3.5** (conditional), deferred.

## 6. Running the gate
```
cd games/seven_wonders_duel/seven_wonders_rust && cargo test          # 6 Rust unit tests
cd games/seven_wonders_duel && python -m pytest test_rust_engine_equiv.py -k closed_search_net
```
(The net gate needs torch; it `importorskip`s otherwise.)

## 7. Sign-offs requested
- Float-plumbing faithfulness and net-determinism assumption (items 1–3).
- Panic-vs-`PyErr` at the FFI boundary (item 4) — acceptable, or fix now.
- GIL re-entrancy does not preclude F4's GIL-release design (item 5).
- Net-gate coverage of draft/force branches sufficient via the mock tests, or add
  a `PyEval` draft/force position (item 6).

## 8. Resolution (2026-07-21 review completed)
Review approved float plumbing (f32 subtract then widen; Rust only negates),
`Encoding` reconstruction, net determinism (CPU/eval/no-grad/no-dropout, batch 1
— not to be extended to CUDA/F4 batch shapes), and GIL re-entrancy. Fixes applied
(see `PHASE_F.md` F3.4 review-hardening; commit follows):
- **P1 fallible evaluator** → `Eval::evaluate` returns `PyResult`; `PyEval`
  propagates the adapter's original `PyErr`; threaded through the search
  (`test_closed_search_net_propagates_adapter_error` — a raised `RuntimeError`
  surfaces as `RuntimeError`, not `PanicException`).
- **P2 contract validation** → `PyEval` verifies finite value, `priors.len() ==
  legal.len()`, finite/nonnegative priors, positive mass
  (`test_closed_search_net_validates_contract`).
- **P2 real-net force-expansion no-op** → `test_closed_search_net_force_expansion`
  finds a CARD_REVEAL root and asserts a weighted multi-outcome edge, matching
  Python to 1e-9.
- **P3 wording** → docs now say "matches within 1e-9" (exact for structure /
  visits / fingerprints), not "bit-identical".
- **F4 handoff** notes recorded in `PHASE_F.md` (split descent, precomputed
  actor/legal, flat encode buffers, batch forced children, `leaf_batch=1` oracle).
Gates: `cargo test` 6 / `pytest test_rust_engine_equiv.py` 24 green.
