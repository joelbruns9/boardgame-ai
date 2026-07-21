# Review request — F3 closed searcher (RNG, tree, Gumbel root)

Companion to `PHASE_F.md`. This asks for a focused review of the **unreviewed**
Phase F3 components: the portable RNG, the closed-node MCTS tree, and the Gumbel
root + force-expansion. F1 (engine), F2 (encoder), and F3.1 (chance engine) were
already reviewed and hardened; those are **out of scope** here except where the
searcher depends on them.

## 1. Scope

| Sub-step | What | Commits | Primary files |
|---|---|---|---|
| F3.0 | Portable SplitMix64 RNG (Python ref + Rust) | `ae39206`, `6793158` | `portable_rng.py`, `rng.rs`, `search.py` (RNG swap) |
| F3.2 (oracle) | `Eval` trait + deterministic `MockEval` | `27ba3a0` | `eval.rs` |
| F3.2 (tree) | `Node`/`Edge`/`Child`, PUCT descent, outcome-keyed materialization | `52c1f54` | `tree.rs` |
| F3.3 | Gumbel root, sequential halving, `force_expand_root_chance` | `087748a` | `tree.rs`, `lib.rs` |

Rust files to review: `seven_wonders_rust/src/{rng.rs, eval.rs, tree.rs}` and the
pyo3 methods in `lib.rs` (`mock_eval`, `gumbel_stream`, `closed_tree_digest`,
`closed_search`). Gate code: `test_rust_engine_equiv.py` (the `mock_eval`,
`gumbel`, `closed_tree`, `closed_search` tests).

**Python reference** (the ported source): `search.py` — `_gumbel_root`,
`_search_closed`, `_descend_closed`, `_closed_child`, `_select_closed`,
`_expand_closed`, `_force_expand_root`, `_make_closed_node`.

## 2. What it does (design)

Closed-loop Gumbel MCTS, written **fresh** in Rust (KD's search is an alpha-beta
solver — not reused). E-Tier-1 fixed the scope: **closed mode only**, with
`force_expand_root_chance` as a runtime toggle; open loop is not ported.

- **`Eval` trait** (`eval.rs`): `evaluate(state) -> (value_p0, priors)`. The tree
  is generic over it. `MockEval` is a deterministic fingerprint-derived oracle
  (splitmix fold → value in [-1,1) + **raw, unnormalized** per-action weight
  priors) used only by the gates. The real batched-NN evaluator is F3.4.
- **Tree** (`tree.rs`): `Node{state, actor, terminal, edges, legal, visits,
  value_sum_p0}`, `Edge{action_index, prior, specs, children, visits,
  value_sum_p0, probability_weighted}`, `Child{probability, node, samples}`.
  Children are a **`Vec` in insertion order** (not a map) so probability-weighted
  `q_p0` and value backprop fold in Python's dict-iteration order — cross-language
  f64 sums are order-sensitive. Values are stored player-0-relative; a `sign`
  flips per node actor.
- **Descent**: terminal → terminal value; unexpanded leaf → `expand` (eval +
  build edges); else PUCT `select` (or a forced edge at the root), `closed_child`
  (sample the chance chain via the portable `Rng`, dedup by the F3.1 observable
  key), recurse, backprop on edge and node.
- **Gumbel root** (`search_closed`): draw a Gumbel key per legal action (portable
  `Rng`, legal order), pick top-k by `gumbel + log_prior`, sequential halving
  over the sims budget (each visit is a forced-edge descent), completed-Q
  candidate reduction, first-argmax best, and a softmax policy target over all
  legal actions.
- **Force-expansion**: materialize + mock-evaluate every enumerable chance child
  of each root edge (AGE_DEAL stays sampled), mark those edges
  probability-weighted so their `q_p0` uses the exact chance expectation.

## 3. What is already gated (please don't re-verify by hand)

All under `MockEval` (a shared, bit-identical oracle), comparing Rust to the
**real Python searcher**:

- **RNG parity**: `next_u64`/`float`/`shuffle` golden; `gumbel` bulk-compared over
  500 draws × 5 seeds (cross-runtime `ln`).
- **MockEval parity**: value + aligned priors match at every state incl. terminal.
- **Tree** (`test_closed_tree_matches_python`): the full DFS digest (node/edge
  visits + values, child keys/samples/probabilities) is bit-identical under a
  fixed round-robin root schedule, sims {16,48} × seeds, play-age positions.
- **Full search** (`test_closed_search_matches_python`): chosen action, `sims`,
  `gumbel_topk`, per-action visits, action/root value, policy target, **and** the
  whole tree digest match, sims {16,64} × seeds × **force-expansion off and on**.

So the *algorithm* is proven equivalent under identical evaluations. Please focus
elsewhere.

## 4. Focus areas (highest value)

### Correctness
1. **Sequential-halving arithmetic** (`search_closed`): `rounds_total`,
   `rounds_remaining`, `per_action`, and the budget-exhaustion breaks vs Python's
   `_gumbel_root`. Off-by-one or a different integer-division rounding would only
   surface on budgets/`top_k` the gate doesn't hit.
2. **Tie-breaking**: the candidate sort (stable, descending) and the **first-max**
   `best` selection must match Python's `sorted(..., reverse=True)` and `max(...)`
   exactly. The gate covers specific seeds; a near-tie on an untested position
   could flip the chosen action. Is the stable-sort + strict-`>` argmax a faithful
   match?
3. **`ln` parity for `log_prior`**: only `gumbel`'s `ln` is bulk-gated. `log_prior
   = ln(max(prior, 1e-12))` uses `ln` on other values; a last-ULP cross-runtime
   difference could flip a selection. Worth a bulk `ln` parity check on arbitrary
   inputs, or an argument that it can't matter.
4. **Force-expansion mass**: Python asserts the chance mass sums to 1 (in
   `_force_expand_root` and in `q_p0` for weighted edges). The Rust port **omits**
   these checks. Is that acceptable (mass is implied by `enumerate_chains`), or
   should it validate?
5. **Sign / actor conventions**: `value_p0` sign flips, `sign * q_p0` in select and
   simulate, `root_value` = `sign * root.value_p0()` — confirm they mirror Python.

### Gate robustness (the recurring F1a/F2.3 lesson)
6. **Chance coverage inside the tree**: the tree/search gates run on **play-age**
   positions, so tree chance edges are reveals/GL only. **AGE_DEAL edges never
   appear** (they arise at draft / age boundaries and produce `probability=None`
   children). The `closed_child`/digest paths for `None`-probability children and
   the age-deal branch are therefore **untested in the tree**. Should the gate add
   a draft/boundary position, or is that adequately covered elsewhere?
7. **Force-expansion depth**: force-expansion is gated, but does the corpus
   actually produce root edges with enumerable chance (so `probability_weighted`
   edges and the exact-expectation `q_p0` are really exercised), and multi-outcome
   children? A counter/assertion would lock it (cf. F3.1 SWAP coverage).
8. **Digest completeness**: does the DFS digest capture everything that could
   diverge (e.g., `legal` ordering, edge `prior`), or could two different trees
   produce the same digest?

### Assumptions worth an explicit sign-off
9. **MockEval as proxy**: the gates prove the *search logic* matches given
   identical evaluations. They say nothing about the eventual **real NN** producing
   identical values in Python vs Rust — and it won't (f32 NN inference isn't
   bit-identical across paths). Is bit-exactness under a deterministic mock the
   right and sufficient bar for F3.2/F3.3, with NN-parity a separate F3.4 concern?
10. **Unnormalized mock priors**: `MockEval` returns raw weights (not a
    distribution) to dodge a cross-language normalization-sum ULP divergence,
    relying on PUCT/Gumbel being invariant to a common scale. Sound?

### Throughput (F4 is next; flag, don't fix)
11. `closed_child` **clones the full `GameState`** at every child materialization,
    and nodes are `Box`-allocated with **linear child lookup** (`Vec::position`).
    No arena. This is gate scaffolding; F4 needs the KD arena/coalescing. Confirm
    nothing here bakes in a design that blocks that.
12. The pyo3 return of a flat `Vec<f64>` digest and `closed_search` tuple is
    gate-only; the real loop drives the searcher in-Rust with the NN `Eval`.

## 5. Known limitations / out of scope
- Open loop not ported (E-Tier-1 decision); Python open kept as reference.
- Real NN `Eval` bridge is **F3.4** (not built); `MockEval` stands in.
- Physical shared-crate extraction is **F3.5** (conditional), deferred.
- Arena / journaled-undo / coalescing perf is **F4**.
- Gotcha already resolved: the tree gate first failed only under pytest — a
  duplicate-module import artifact in the *reference* builder, not the Rust code
  (see `PHASE_F.md`).

## 6. Running the gates
```
cd games/seven_wonders_duel/seven_wonders_rust && cargo test         # 6 Rust unit tests
cd games/seven_wonders_duel && python -m pytest test_rust_engine_equiv.py   # 16 gates
# searcher-only:
python -m pytest test_rust_engine_equiv.py -k "mock_eval or gumbel or closed_tree or closed_search"
```

## 7. Sign-offs requested
- Sequential-halving/tie-break/`ln` faithfulness (items 1–3).
- Force-expansion mass omission acceptable, or add the check (item 4).
- MockEval-as-proxy is the right bar for F3.2/F3.3 (item 9).
- Tree chance coverage gap (AGE_DEAL / `None`-probability children) — add a
  targeted position before F4, or confirm it's covered (item 6).

## 8. Resolution (2026-07-20 review completed)
Review approved sequential halving (checked 408 combos, budgets 1–17),
tie-breaking, and `ln` parity. Fixes applied (see `PHASE_F.md` F3 searcher
review-hardening; commit follows this doc):
- **P1 force-expansion no-op** → dedicated `WONDER_GROUP_REVEAL` and `AGE_DEAL`
  coverage tests that assert weighted/multi-child edges and `None`-probability
  children; still bit-identical to Python.
- **P2 config contract** → `search_closed` returns `Result`/`PyResult`, rejects
  `sims<1`/`top_k<1`/terminal-or-action-less root.
- **P2 mass check** → validated before an edge is marked probability-weighted.
- **P2 wrong prior justification** → comments corrected (raw priors give
  implementation parity, not PUCT scale-invariance); F3.4 must use normalized
  priors.
- **P2 AGE_DEAL / None coverage** → covered by the age-deal test above.
- **P3 digest** → now includes node actor/terminal + full state fingerprint and
  unambiguous key encoding (part counts + per-part lengths).
- **`ln` parity** → direct 100k-value gate added.
- **P1 throughput (double-eval)** → confirmed it matches Python (not a
  divergence); deferred to F3.4's batched evaluator, logged in `PHASE_F.md`.
Gates: `cargo test` 6 / `pytest test_rust_engine_equiv.py` 20 green.
