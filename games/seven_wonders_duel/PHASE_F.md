# Phase F Rust port — working doc

Operational companion to `AZ_PROJECT_PLAN.md` §8, in the `PHASE_D.md` mold: gate
definitions, crate-split decisions, and a running log. The plan stays the
charter; decisions and gotchas land here as they happen.

Method is the Kingdomino discipline (§2b "differential-gate test style"): every
step ships behind a bit-exact or 1e-6 equivalence gate against the Python
reference before the next step starts. Python keeps the dual-mode searcher as
the slow reference implementation permanently.

## Milestones

### F1 — Engine with make/unmake (the cost center)

One state core shared by MCTS and the full Rust self-play path; make/unmake from
day one, not retrofitted. Port effect-by-effect against `RULES_ORACLE.md`;
7WD resolution (chains, pendings, supremacies, extra turns) is meaningfully
more intricate than Kingdomino's.

- **Gate F1a:** replay ≥10k Python games byte-exactly from `(seed, actions)`
  (A4 buffers are replayable — use real run buffers, e.g. `runs/phase_d_toy`,
  plus fresh bot games for coverage).
- **Gate F1b:** make/unmake round-trip — the *complete* state (not just the
  cross-language fingerprint) is restored before/after apply+undo at every
  decision, plus an exhaustive sampled audit (`roundtrip_all_ok`): every legal
  action to depth 2 (nested LIFO) with full-state undo and apply determinism.
  Whole-state comparison via `GameState: PartialEq` so a future journaled undo
  cannot skip a non-fingerprinted field (e.g. `library_draws`).
- Status: **GREEN.** Crate `seven_wonders_rust/` (engine-only) ships the full
  base-game resolution ported effect-by-effect from `engine.py`, with codec
  index ↔ action and a language-neutral integer fingerprint. Gates live in
  `test_rust_engine_equiv.py` (`logic_fingerprint` mirrors
  `state.rs::fingerprint` byte-for-byte). Both gates assert agreement at
  *every* decision (fingerprint + legal-action mask), not just terminally,
  across real `runs/phase_d_toy` buffer games and seeded random-policy games.
  See the decision log for the RNG-boundary and fingerprint design.
  Gate run 2026-07-19: **10,000 buffer games / 664,991 decisions byte-exact**
  (victory mix civ 5667 / mil 2373 / sci 1956 / shared 4 — all endgame paths
  exercised), plus 60 random games incl. Great Library draws; F1b round-trip
  checked across the sample. `pytest test_rust_engine_equiv.py` green.
  Gate-hardening run 2026-07-20 (review follow-up, below): full **12,500-game**
  corpus across all buffer files byte-exact in 362s via `SWR_F1A_GAMES=0`;
  routine `pytest` runs a 400-game multi-file subset and *skips* (not passes)
  when buffers are absent. `cargo test` now runs 3 Rust unit tests (was 0).

### F2 — Encoder + codec

- **Gate F2:** bit-exact `encode_state` and exact `encode_action`/decode
  (order and value) vs Python on ≥100k sampled states across all game phases,
  per `CODEC_SPEC.md` (codec-1, encoder signature enforced).
- **Codec half already satisfied by F1a (2026-07-20 scoping).** `codec.rs`
  (`encode_action`/`decode_action`/`legal_action_indices`) shipped with F1, and
  F1a asserts the Rust legal-action indices equal Python's at *every* decision —
  i.e. `encode_action` order+value over every legal action across 664,991
  decisions / 12,500 games, far exceeding the ≥100k-state bar, with `decode`
  exercised by every apply. F2's remaining work is the **encoder**.
- **Encoder foundation:** `encode()` consumes `PlayerObservation` + `UnseenPool`
  (actor-relative, hidden-info-stripped), which the F1 crate does not yet model.
  For the F4 in-Rust self-play path the encoder must run from `GameState` with no
  Python hop, so `observation(viewer)` (game.py) and `UnseenPool` (pool.py) are
  ported to Rust first. Features computed in **f64** (KD f32-broke-bit-identity
  lesson) and compared bit-for-bit.
- **Sub-sequence (KD M2 discipline, each behind its own gate):**
  - **F2.1** — port `UnseenPool` + `observation()`; gate derived pool/obs fields
    vs Python over sampled states. **DONE 2026-07-20:** `pool.rs` (read-side
    `unseen_pool`/`visible_cards`) ships; `RustGame.unseen_pool()` matches Python
    at every decision across all phases (`test_unseen_pool_equivalent`, 40 random
    games draft→age III→endgame). No separate observation struct needed — the
    Rust `GameState` already holds every public field the encoder reads;
    phase-specific visibility (e.g. the draft-hidden tableau) is applied in the
    token builders. Search-side pool helpers (`resample_hidden`/`enumerate_*`)
    deferred to F3.
  - **F2.2** — port `encode()` token builders. **DONE 2026-07-20:** `encoder.rs`
    ships all nine token types in f64, feature-for-feature in `encoder.py` order,
    reusing `minimum_payment`/`fixed_production`/`choice_producers`/
    `opponent_trade_production`/`trade_discounts` (made `pub(crate)`) and a new
    `GameState::score_player` breakdown (`score_totals` now delegates to it). The
    Python "stub state" is unnecessary — the real `GameState`'s public fields
    equal the observation's. `RustGame.encode()` matches Python
    `encode(observation)` bit-for-bit (token type/entity_id/aux_id/features) at
    every decision across all phases (`test_encode_equivalent`, 40 random games).
  - **F2.3** — full `encode_state` bit-exact gate over ≥100k sampled states.
    **DONE 2026-07-20:** `test_encode_corpus_equivalent` (lean encode-only driver,
    env-sized `SWR_F2_GAMES` like F1a; skips when buffers absent). Acceptance run
    `SWR_F2_GAMES=2000`: **113,726 states bit-exact in 312s**; routine `pytest`
    runs a 60-game subset. `SWR_F2_GAMES=0` sweeps the whole corpus.
- Status: **GREEN 2026-07-20** (incl. review-hardening below). Codec covered by
  F1a; encoder ported and bit-exact (pool + all nine token types) over 139k+
  sampled states across all phases. `cargo test` 4 / `pytest
  test_rust_engine_equiv.py` 8 green. Next: F3 (closed searcher + Gumbel root;
  mode scope resolved by E-Tier-1 above).
- **F2 review-hardening (2026-07-20, external review — no logic divergence
  found; all sign-offs approved):**
  - **Acceptance gate now enforces its criteria** (was: a 1-game corpus would
    pass). `iter_buffer_records` samples **round-robin across all buffer files**
    (was a lexicographic prefix of `curriculum_seed.jsonl` — also fixes F1a
    sampling); the gate asserts `games == requested` and, at acceptance scale
    (`SWR_F2_GAMES` 0 or ≥2000), `states ≥ 100_000`. Acceptance run: **139,816
    states / 2000 games / 367s**, multi-file.
  - **Terminal + branch coverage** — the loop now also compares the terminal
    `COMPLETE` encoding (decision 8 was previously never gated), and the gate
    asserts all nine decision branches and all nine token types appear at
    acceptance scale (guards against silent corpus drift).
  - **Rust bound to `ENCODER_SIGNATURE`** — `encoder.rs` pins the signature +
    per-type `FEATURE_COUNTS`; `encoder_signature()` is exposed and
    `test_encoder_signature_matches` asserts equality with Python, so a schema
    change fails until Rust is updated in lockstep. A `debug_assert` + Rust unit
    test check per-token feature lengths.
  - **Deferred to F3/F4 (throughput, benchmark-driven):** rework encoder pricing
    to a per-seat `PaymentContext` (precompute fixed production / trade prices /
    discounts / chain / rebate / flexible assignments once per encode, instead of
    reconstructing them per `minimum_payment` call — `pool_tokens` alone does up
    to ~146 calls in draft), keeping exact pool cost aggregates; and add an
    `encode_into`-style reusable flat buffer feeding Rust-side batching (the
    current `Vec`-per-token + Python-object return is gate scaffolding). Enforce
    the encoder signature at the F4 checkpoint-load boundary.
  - **Sign-off follow-ups for F3:** when F3 adds Rust determinization, add a
    hidden-resampling invariance test (encoding unchanged under resampling of
    hidden cards) to preserve the "real GameState instead of stub" and
    "face-down back class from hidden id" guarantees.

### F3 — Searcher + Gumbel root (in-crate; extraction conditional)

Written fresh inside `seven_wonders_rust`, porting KD's
arena/coalescing/`allow_threads` scaffolding. A shared-crate extraction remains
an independently gated, conditional follow-up (see crate split below).

- **Mode scope — resolved by E-Tier-1 (2026-07-20 verdict below):** port the
  **closed searcher with `force_expand_root_chance` as a runtime toggle** (one
  searcher + a boolean, not two modes — `closed_forced` is just that flag set).
  **Open loop is not ported** — it trailed on trap coverage and showed the
  stale-prior signature; the Python open searcher stays as the permanent
  reference and open is revisited only if closed's Rust throughput disappoints.
  Closed vs closed_forced is left for E-Tier-2 to settle at equal wall-clock
  ON THE RUST SEARCHER, where the compute-normalized comparison is meaningful
  (force-expansion's extra reveal-layer evals only pay off if strength justifies
  the cost). Not porting open's determinizer is where the ~10–20% mode-delta
  saving lands.
- **Gate F3:** tree statistics (visit counts, values to 1e-6, chosen actions)
  match the Python reference searcher on fixed seeds/positions under a mock
  evaluator, per KD M4. One gate for the closed searcher, run with the flag both
  off and on.
- **Scope resolved 2026-07-20** (sources: `search.py` closed searcher +
  `kingdomino_rust/src/search.rs` reuse audit).
  - **Reuse map:** KD's `search.rs` is an *alpha-beta/expectiminimax solver*, not
    a Gumbel MCTS — the algorithm is written **fresh**. Reused: `splitmix64`, the
    arena pattern, leaf-coalescing, `allow_threads`/`py.detach`, and the `Game`
    trait *boundary shape* (`make`/`unmake`/`is_stochastic`/`chance_children`/
    `make_with_chance`). Engine `make`/`unmake` (F1) and encoder (F2) already
    exist.
  - **Two dominating prerequisites, both ahead of the searcher:**
    1. **Chance-node engine extension** (the F1-deferred item): add
       `make_with_chance` (action + resolved outcome) plus Rust
       `chance_signature`/`enumerate_chains`/`sample_outcomes` over the four
       chance kinds. Its own differential gate vs Python's chance layer.
    2. **RNG parity — the bit-exact crux.** Python uses Mersenne
       `gammavariate`/`shuffle`/`randrange`; Rust uses `splitmix64`. **Decision
       (2026-07-20): refactor the Python reference to a portable splitmix64
       stream** (Gumbel via `-log(-log(1-u))`, portable `randrange`/Fisher-Yates)
       so the tree gate is genuinely bit-exact (KD precedent, 288/288). Changes
       Python self-play noise going forward (harmless); re-verify Phase D/E after.
  - **Crate split — decision (2026-07-20): build the searcher *in-crate* first
    (`seven_wonders_rust`), extract the shared crate later** as a separate,
    KD-regression-gated step once the truly-shared surface is concrete (little
    shared *algorithm* code, so the trait boundary is clearer after the Gumbel
    searcher exists). Supersedes the "split at F3" default in §8 / the crate-split
    section below.
- **Sub-sequence (each behind a gate):**
  - **F3.0** — Python reference RNG → portable splitmix64 (Gumbel + chance
    sampling); Phase D/E re-verified. **DONE 2026-07-20:** `portable_rng.py`
    (`PortableRng`: `next_u64`/`next_float`/`gumbel`/`randrange`/Fisher-Yates
    `shuffle`/`getrandbits`, KD splitmix64 constants; seed-0 first output is the
    canonical `0xE220A8397B1DCDAF`). `search.py` now uses it for the Gumbel keys
    and `sample_outcomes`; open-mode `resample_hidden` works unchanged (portable
    `shuffle`/`getrandbits`). `test_portable_rng.py` pins the golden stream Rust
    must reproduce. `test_search.py` (23) + Phase D/E (20) green.
  - **F3.1** — Rust chance engine, in two halves:
    - **F3.1a DONE 2026-07-20:** `rng.rs` (SplitMix64 mirroring `portable_rng.py`;
      golden Rust unit test) + `chance.rs` (`ChanceKind`, `chance_signature`,
      `enumerate_chains` with a lexicographic `combinations` helper). Exposed as
      `RustGame.chance_signature`/`enumerate_chains`; `coverers` made
      `pub(crate)`. Gate `test_chance_signature_and_chains_equivalent`: specs +
      chain outcomes/probabilities match Python at every legal action across 25
      random games (all phases); AGE_DEAL refused by both.
    - **F3.1b-i DONE 2026-07-20:** `sample_outcomes` in `chance.rs` (portable
      `Rng`; AGE_DEAL uses `pool_by_name` alphabetical sort + triple shuffle to
      match Python's `sorted(names)`). Exposed as `RustGame.sample_outcomes(index,
      seed)`; gate `test_sample_outcomes_equivalent` reproduces Python's sampled
      chain under shared seeds (0/1/12345) at every chance-bearing action across
      25 random games, prob included.
    - **F3.1b-ii DONE 2026-07-20:** `apply_with_chance` (engine.rs) installs each
      supplied outcome into hidden state before the normal apply — `override_reveal`
      (swap the locked card into the outcome card's hidden location: sibling
      face-down slot → removed pile → unused guilds), `override_wonder_flip`
      (set group 1; `pick_wonder` copies it to the offer), Great-Library draw
      (`push_front` onto `library_draws`), and `validated_age_deal`. Pre-installing
      is provably equivalent to Python's mid-apply overrides for the searcher's
      distinct (used-deduplicated) outcomes. Gate `test_make_with_chance_equivalent`:
      resulting-state fingerprints match Python's `apply_action(chance_outcomes=…)`
      across all four chance kinds, 20 random games × multiple outcomes/action.
  - **F3.1 COMPLETE.** `cargo test` 5 / `pytest test_rust_engine_equiv.py` 11 green.
  - **F2 hidden-resampling follow-up — resolved:** the encoder-invariance-under-
    hidden-resampling property is established by F2.3 (the encoder consumes only
    the public projection, bit-exact vs Python's observation over 139k states). A
    dedicated Rust *determinizer*-invariance test is N/A for the closed-only port
    (no Rust determinizer is built — open is not ported); revisit only if open
    determinization is ever added.
  - **F3.1 review-hardening (2026-07-20, external review — no valid-path
    divergence found; four contract gaps closed before F3.2/F3.3):**
    - **Observable keys** now returned by `enumerate_chains`/`sample_outcomes`
      (`age_deal_key` coalesces hidden deals with the same public signature;
      off-AGE_DEAL the key equals the outcomes). Gate compares key parity,
      including the AGE_DEAL face-up/back-marker encoding — so F3.2 keys children
      correctly instead of by hidden deals.
    - **`apply_with_chance` is checked + atomic** — `validate_chance` runs a full
      pre-mutation pass (reveal back/pool membership + distinctness, wonder-flip
      length/uniqueness/pool, Great-Library subset, age-deal size/backs/visible/
      3-guilds) and returns `Result`; malformed input errors with the state
      untouched (`test_make_with_chance_rejects_malformed`).
    - **Gumbel parity gated in bulk** — `gumbel_stream` + a Rust golden test
      compare 500 draws across 5 seeds (cross-runtime `ln` parity), the F3.3
      root-selection prerequisite.
    - **SWAP branch coverage asserted** — the make_with_chance gate now requires
      sibling / removed-pile / unused-guild reveal sources, a sequential
      same-back reveal, and all three age-deal ages. `cargo test` 6 / `pytest
      test_rust_engine_equiv.py` 13 green.
  - **F3.2** — Rust closed-node tree + PUCT descent + outcome-keyed child
    materialization; matches Python to 1e-6 under a mock eval on deterministic
    positions (sampling off).
    - **Foundation DONE 2026-07-20:** `eval.rs` — an `Eval` trait + `MockEval`, a
      deterministic fingerprint-derived oracle (splitmix fold → value_p0 in
      [-1,1) + raw per-action weight priors). Priors are **unnormalized** on
      purpose: a cross-language normalization *sum* diverges in the last ULP, and
      PUCT/Gumbel are invariant to a common scale, so raw weights stay
      bit-identical. Gate `test_mock_eval_matches_python` matches at every state
      incl. terminal.
    - **Design note (surfaced here):** cross-language f64 **sum order** matters —
      the tree must iterate children in **insertion order** (like Python's dict)
      so probability-weighted `q_p0` and value backprop match bit-for-bit; use an
      insertion-ordered children map, not a `HashMap`.
    - **DONE 2026-07-20:** `tree.rs` — `Node`/`Edge`/`Child` with **insertion-
      ordered** children (a `Vec`, so probability-weighted `q_p0` and backprop
      fold in Python's dict order), `expand`/`select` (PUCT)/`closed_child`
      (sample via the portable `Rng`, dedup by observable key)/`descend`/a fixed
      round-robin root driver + a canonical DFS `digest`. Gate
      `test_closed_tree_matches_python`: the full tree digest (visits, values,
      edge/child stats, keys) is bit-identical to the Python reference searcher
      (real `GumbelMCTS` closed methods) under `MockEval`, across sims {16,48} and
      seeds, over play-age positions in 8 games. (`force_expand_root` /
      probability-weighted edges are wired but exercised in F3.3.)
    - **Gotcha logged:** the gate first failed only under pytest — a duplicate-
      module artifact (`_closed_tree_ref` used an absolute `from
      games.seven_wonders_duel.search import`, resolving to a second package copy
      whose codec saw the state's enums as foreign and returned 0 legal actions).
      Fix: import via the module-relative `.search` like the rest of the tests.
      The tree code was correct throughout (direct call always passed).
  - **F3.3** — Gumbel root (top-k + sequential halving + completed-Q policy
    target) + `force_expand_root_chance`; full `search()` matches Python
    (visits/values/chosen action) under mock across seeds, flag off and on.
    **DONE 2026-07-20:** `tree.rs::search_closed` + `force_expand_root` + a
    `SearchConfig`/`SearchResult`, ported from `_gumbel_root`/`_search_closed`.
    Gumbel keys drawn from the portable `Rng` in legal order; sequential halving,
    first-argmax best, and a legal-order left-folded policy normalizer match
    Python's sums. Gate `test_closed_search_matches_python`: chosen action,
    `sims`, `gumbel_topk`, per-action visits, action/root value, policy target,
    AND the full tree digest are bit-identical to the real Python searcher under
    `MockEval`, across sims {16,64} × seeds × **force-expansion off and on**.
  - **F3 searcher review-hardening (2026-07-20, external review — sequential
    halving verified over 408 combos, tie-breaking + ln approved; six fixes):**
    - **Force-expansion now actually exercised** (it was a no-op — the corpus had
      0 weighted edges): `test_closed_search_force_expansion_coverage` drives a
      draft `WONDER_GROUP_REVEAL` root and asserts a probability-weighted edge
      with many children; `test_closed_search_age_deal_coverage` drives an
      AGE_DEAL root and asserts sample-only (`None`-probability) children. Both
      still bit-identical to Python.
    - **Config contract enforced** — `search_closed` returns `Result`/`PyResult`
      and rejects `sims<1`, `top_k<1`, and a terminal/action-less root (was
      silently degrading); `test_closed_search_rejects_bad_config`.
    - **Force-expansion mass validated** before an edge is marked weighted
      (ported Python's tolerance check).
    - **Digest completed + disambiguated** — now includes node actor/terminal and
      the full **state fingerprint** (equal digests ⇒ equal states), and encodes
      child keys with explicit part counts + per-part lengths (`[[1],[2]]` vs
      `[[1,2]]` no longer collide).
    - **`ln` parity gated directly** (`ln_values` + `test_ln_parity`, 100k values
      over (0,1] plus edges) — closes the `log_prior` last-bit concern.
    - **Wrong comment fixed** — `MockEval` priors are raw/unnormalized for
      *implementation parity* (both sides consume identical priors), NOT because
      PUCT is scale-invariant (it isn't); noted that F3.4 must gate with
      production-shaped normalized priors.
    - **Deferred to F3.4/F4 (throughput):** forced children are evaluated then
      re-evaluated on first visit (matches Python's double-eval — not a
      divergence; the batched NN evaluator fixes it by preserving expansion
      state). `Box<Node>` + full-state clone per child + linear child lookup +
      scalar `Eval` remain gate scaffolding; the scalar evaluator boundary must
      NOT become the production batching interface.
  - **F3.4** — real-evaluator bridge across the pyo3 boundary (feeds F4).
    **DONE 2026-07-21:** `eval.rs::PyEval` — a real-net `Eval` that encodes with
    the F2 Rust encoder and calls a Python adapter `(tokens, actor, legal) ->
    (value_actor, priors)` running the net; `RustGame.closed_search_net(adapter,
    …)` runs the full Gumbel search on it. Gate `test_closed_search_net_matches_
    python`: with a real `SWDNet` the Rust searcher **matches** Python's
    `GumbelMCTS` on the same net — chosen action/top-k/visits/state-fingerprints
    **exactly**, values/policy/node-values **within 1e-9** — across sims/seeds and
    force-expansion off/on (1e-9, not exact, so it also catches the f32-vs-f64
    subtraction subtlety). Validates the entire F3 port (RNG + chance engine +
    tree + Gumbel root) against the actual network, not just the mock.
    **Scalar per-leaf bridge for correctness only** — F4 replaces this boundary
    with leaf coalescing + GIL release for the ≥20× throughput gate (that batched
    fast path is deliberately NOT bit-identical to the sequential reference —
    coalescing evaluates multiple leaves before backprop — so it is validated by
    throughput + agreement, not this gate).
  - **F3.4 review-hardening (2026-07-21, external review — float plumbing /
    encoding / determinism / GIL approved; three fixes):**
    - **Fallible evaluator** — `Eval::evaluate` now returns `PyResult`; `PyEval`
      propagates the adapter's original `PyErr` (CUDA OOM / bad checkpoint /
      Python bug) through the whole search instead of panicking, and
      `search_closed`/`closed_tree_fixed`/`closed_search[_net]` thread it. Gate
      `test_closed_search_net_propagates_adapter_error`.
    - **Contract validation** — `PyEval` rejects a non-finite value, a prior
      count ≠ legal count, non-finite/negative priors, and a zero-mass policy
      (`test_closed_search_net_validates_contract`).
    - **Real-net force-expansion actually exercised** — the earlier net gate's
      force cases were no-ops (0 weighted edges); `test_closed_search_net_force_
      expansion` finds a CARD_REVEAL root and asserts a weighted multi-outcome
      edge, matching Python to 1e-9. `cargo test` 6 / `pytest` 24 green.
    - **F4 handoff notes (from review):** F4 needs more than `evaluate_batch` —
      split descent into selection/materialization, pending-leaf batched eval,
      and backprop; pass precomputed actor/legal to the evaluator; encode into
      reusable flat buffers (not Python token objects); cache batched forced-child
      evaluations but consume them on the first ordinary visit so search depth and
      accounting stay unchanged; keep the scalar search as the oracle and require
      `leaf_batch=1` to match it exactly before testing larger batches
      statistically.
  - **F3.5** *(conditional)* — physical shared-crate extraction, KD gates intact.
  - Carries the F2 sign-off follow-up: hidden-resampling invariance test once
    determinization lands (F3.1).
- Status: **complete 2026-07-21.** F1–F3 green; conditional F3.5 extraction was
  not triggered.

### F4 — Batched inference bridge

**Design doc: `F4_THROUGHPUT_DESIGN.md`** (revised 2026-07-21). The complete
self-play hot path moves to Rust: game/chance advancement, resumable Gumbel search,
action sampling, recording, and a cooperative pool of game slots. Python remains
only the cold control plane and one global-batch Torch inference worker; the
Python/Torch boundary is explicitly profiled before any native-NN decision.

The fast search uses **WU-style incomplete visit counts** below the Gumbel root:
pending simulations contribute to effective PUCT visits but never inject a
fictional Q value. Leaf waves follow the sequential root schedule and fully drain
before every halving reduction. `leaf_batch` and the pending policy are selected
by the strengthened laptop quality suite and checked into
`f4_quality_lock.json`; cloud calibration may not retune them.

There is **no planned 7WD exact endgame solver and no reserved CPU budget**.
Scheduler threads/shards, concurrent slots, global batch cap, wait window,
buffering, and transformer packing are tuned for maximum games/s on the target
cloud GPU. The checked-in `run_f4_cloud_sweep.sh` runs only the Rust production
generator — no Python self-play baseline on rented time.

Sub-steps: F4.0 contracts/instrumentation; F4.1 arena + exact resumable phase
split; F4.2 WU leaf waves + laptop quality lock; F4.3 complete Rust game slot;
F4.4 cooperative scheduler/global coalescer; F4.5 transformer boundary +
semantic-safe forced cache; F4.6 local/cloud performance exit; F4.7 optional
native NN.

- **F4.0 DONE 2026-07-21 — preregistered contracts.**
  `f4_contract.json` freezes the quality sample sizes/margins before any
  `leaf_batch>1` observation, the laptop comparison, quality-lock fields, cloud
  override prohibitions, and the complete component-metric vocabulary. Headline
  quality bars: ≥240 stratified positions (≥50 consequential, ≥30 baseline-clean
  consequential), 32 paired seeds/position, zero new clean-fixture blunders,
  one-sided 95% upper trap-rate delta ≤+2 percentage points, regret/policy/value
  bounds, and playing-strength lower bound ≥-25 Elo over ≥1,200 seat-swapped
  pairs. The laptop speed gate remains a 95% lower speedup bound ≥20× over at
  least five 100-game repetitions after warmup.
- **F4.1 DONE 2026-07-21 — arena + exact resumable phase split.**
  `tree_resumable.rs` is deliberately separate from `tree.rs`, which remains the
  permanent F3.3 oracle. The new path stores stable arena node IDs and pending
  `(node_id, edge_id, chance_child_id)` steps, separates selection/materialization
  from aligned evaluation application and backup, bypasses pending-count behavior
  at `leaf_batch=1`, and preserves the scalar forced-child semantics pending the
  separately gated F4.5 cache. The exact gate compares action/top-k/visits/sims,
  policy/value to 1e-9, and the full canonical topology digest. Coverage includes
  mock + real-net adapters, force off/on, enumerable and sampled chance, draft,
  Ages I–III, between-age and pending choices, terminal leaves, varied seeds,
  top-k, and non-power-of-two sim budgets. Acceptance run: `cargo test` 6/6 and
  `pytest test_rust_engine_equiv.py` 26/26 green.
- **F4.2 mechanics + calibration harness DONE 2026-07-21; registered quality
  gate measured and failed.**
  The arena now carries WU incomplete node/edge counts used only in deeper PUCT
  exploration; completed Q is never perturbed. Scalar-order selection can fill a
  leaf wave across root-candidate boundaries but the session refuses to reduce a
  halving round until all requests/backups drain and every incomplete count is
  zero. Pending leaves are deduplicated in insertion order, expanded once, and
  backed up once per scheduled path; terminal leaves complete without an NN row.
  Batch response alignment is checked, and evaluator/shape failures clear the
  whole pending wave before surfacing the original error. Metrics cover scheduled,
  requested/unique/terminal leaves, collisions, waves, and maximum path/unique
  fill. `leaf_batch=1` still bypasses WU and remains full-digest exact.

  `f4_quality.py` consumes the preregistered contract and a Phase-E-compatible
  corpus, runs paired sequential-vs-WU searches, records completed-Q regret,
  gap-stratified agreement, policy JS divergence, root-value error, trap deltas,
  collision/fill metrics, natural sequential-seed variance, and the clustered
  one-sided confidence bound. It cannot write `f4_quality_lock.json`; the
  seat-swapped strength gate must also pass. `f4_corpus.py` deterministically
  preserves the complete consequential Phase-E subset, then fills the registered
  phase strata from replay buffers and records phase/chance coverage plus source
  hashes. Rejected Phase-E candidates are intentionally not multiplied across the
  32-seed sweep.

  Smoke results are honest rather than promoted: one existing buffer supplies
  307 positions with every phase quota ≥30 (card reveal 182, Great Library 16,
  wonder reveal 4, age deal 33), but the current Phase-E bank has only **11/50**
  required consequential positions. Therefore `structural_eligible=false`, no
  leaf-batch sweep is accepted, and no quality lock exists yet. Regression run:
  `cargo test` 6/6; Rust-equivalence + quality-contract pytest 32/32 green.

  Final registered calibration (2026-07-21): Phase-E expansion produced 650
  unique replayable candidates and 649 exact depth-2 labels (one frontier-cap
  skip). The assembled corpus is structurally eligible with 306 positions, 100
  consequential and 71 baseline-clean consequential fixtures, and phase counts
  40/54/122/30/30/30 for Ages I/II/III, between-ages, pending-choice, and draft.
  All 39,168 paired rows completed. No registered fast candidate passed: batch 2
  cleared large-gap agreement, trap-rate non-inferiority, and value-error bounds,
  but failed zero-new-clean-blunder, regret, policy-divergence, and medium-gap
  agreement; batches 4/6/8 failed additional gates. Therefore
  `largest_position_eligible_leaf_batch=null`, the strength runner was correctly
  not started, and no `f4_quality_lock.json` was written. Frozen thresholds were
  not changed after observing the result.

- **F4.3 DONE 2026-07-21 — complete Rust self-play slot.**
  `self_play.rs` owns a whole network-vs-network game after its locked setup:
  the portable game RNG, full/cheap simulation selection, inclusive simulation
  range, per-move search seed, Phase-D Wonder-tier prior blend, temperature
  action sampling, search/game advancement, actual chance logging, policy
  eligibility, move statistics, and final result. The fixed game RNG contract is
  `SplitMix64(game_seed ^ 0xC6BC279692B5CC83)`, consumed per move in the order
  full/cheap choice, simulation count, search seed, action sample. Both mock and
  Python-network evaluator entry points return only after the game completes;
  Python does not drive individual moves.

  `rust_bridge.phase_d_record_from_rust` is cold-path compatibility work: it
  converts the completed Rust record into the existing schema-1 `GameRecord`,
  recomputes Python's RNG-inclusive mask/state/trajectory digests by replay, and
  rejects any legal-mask, actor, chance-log, or final-result divergence. Great
  Library's sole play-time simulator RNG draw is pre-locked at setup, so it does
  not force control back across the language boundary. Targeted gate: a complete
  `leaf_batch=1` game matches the independent sequential Python oracle
  move-for-move (actions, schedule/seeds, visits/top-k/root values and policy
  within 1e-14), final logic fingerprint, JSONL schema round-trip and replay;
  the deterministic-record/Great-Library fixture, real evaluator-boundary full
  game, and error/config contracts also pass (`pytest
  test_f4_self_play.py`: 4/4 green). Full regression remains for the
  scheduled overnight run.

- **F4.4 DONE 2026-07-21 — cooperative scheduler + global coalescer.**
  `run_many`/`run_many_pipelined` maintain independent resumable game slots and
  advance each ready slot until it yields one indivisible root or WU leaf-wave
  request. Requests are packed in stable slot order up to `global_batch_cap`,
  never split across batches, and scattered by explicit slot/request ownership.
  Completion order cannot affect returned order: records always follow input job
  order. The network entry detaches the caller's GIL; one dedicated inference
  worker owns the Python adapter, and configurable one/two-plus batch buffering
  lets Rust scatter and advance batch N while the worker executes batch N+1.

  The F4.4 adapter contract is one Python call over ordered
  `(tokens, actor, legal)` rows and aligned `(value_actor, legal_priors)` rows.
  `rust_global_batch_adapter` remains the object-boundary correctness fallback;
  F4.5's production boundary below replaces it without changing scheduler
  ownership/scatter semantics.

  Every error path cancels outstanding search waves before returning. Adapter
  errors/OOMs retain their original Python exception; row/shape errors are
  localized before scatter; configurable inference timeout wakes all slots. A
  timed-out Python/Torch call cannot be killed safely, so its worker is detached
  with no slot/search ownership and exits when that call returns. The gate covers
  cooperative-vs-independent exact games at `leaf_batch=1`, mixed token/legal
  lengths, scalar-vs-global evaluator equivalence, cross-game leaf-wave fill,
  deterministic ordering, replay/schema validity, dedicated-thread ownership,
  adapter/shape/timeout cleanup, split-wave cap rejection, and a 12-slot stress
  run (`pytest test_f4_scheduler.py`: 7/7 green).

- **F4.5 DONE 2026-07-21 — flat transformer boundary + semantic-safe forced
  cache.** `self_play_many_flat_net` packs reusable Rust numeric scratch into
  aligned writable byte buffers (token offsets/type/entity/aux/features, actor,
  legal offsets/actions). Cached actor/legal metadata flows from arena nodes and
  root move metadata through global groups and the worker instead of being
  recomputed during packing. `_RustFlatBatchAdapter` builds tensors directly,
  runs the model once per global batch, gathers only legal logits, and transfers
  only actor-relative value plus ragged legal policy rows back to Rust. It does
  not reconstruct Python `Token`/`Encoding` objects or copy full policy and
  auxiliary heads to CPU.

  Boundary metrics add total/max tokens, padded tokens/padding ratio, Rust
  encode-pack, queue wait, Python call, and result extraction. The Python adapter
  separately records tensor construction, H2D, forward, legal gather, and D2H;
  diagnostic synchronization is opt-in so production timing remains
  asynchronous. Forced root children are materialized and evaluated in capped
  batches, retain value+priors while preserving the old seeded visit/value, then
  consume the cache on their first ordinary visit, expand, stop, and back up the
  same value—avoiding both a duplicate inference and an extra-ply semantic drift.
  Cooperative `force=true` is now supported.

  Gates cover exact object-vs-flat deterministic games, real SWDNet boundary
  agreement, replay/schema validity, buffer/cap contracts, exact forced-cache
  result and tree digest at `leaf_batch=1` with fewer evaluator calls, and exact
  independent-vs-cooperative forced games. Acceptance: `cargo test` 6/6 and 25
  targeted F4.1–F4.5 Python regressions green. Final verification after all F4
  implementation changes: `cargo fmt --check`, `cargo test` 6/6, and the complete
  Seven Wonders Duel Python suite 295/295 green (one existing PyTorch warning).

- **F4.6 implementation DONE 2026-07-21; registered measurements pending.**
  `f4_throughput_bench.py` provides the frozen laptop Python-vs-Rust comparison
  and the Rust-only cloud primitive. Runs are append-only/resumable, reject a
  checkpoint or algorithm setting that differs from `f4_quality_lock.json`, emit
  JSONL + CSV rows and exact manifests, and cover every registered scheduler,
  inference-boundary, utilization, memory, collision, padding, and throughput
  metric. The laptop decision uses a fixed-seed paired bootstrap over at least
  five 100-game repetitions and accepts only when its one-sided 95% speedup lower
  bound reaches 20x.

  `run_f4_cloud_sweep.sh` is a checked-in staged Rust-only sweep over batch/slot
  geometry, buffering, pinned transfer, and approved Torch compile modes. It
  records OOM/operational failures instead of selecting them, ranks only
  quality-locked eligible rows, repeats the winner, and runs a separate
  synchronized diagnostic. `f4_cloud_finalize.py` refuses underfilled
  confirmation, changed production geometry, non-synchronized diagnostics, or a
  checkpoint/software/target-box mismatch. The shell script passes `bash -n` on
  the development machine.

- **F4.7 conditional decision implemented 2026-07-21.** Native inference is
  required only when the synchronized target-box diagnostic attributes at least
  15% of wall time to removable Python/PyO3 tensor-boundary work and the resulting
  Amdahl upper bound is at least 1.10x. Otherwise the measured decision is
  `skip_native_inference`; F4.7 does not become an unconditional port.

- **Gate F4 (= Phase F exit, conjunctive):** `leaf_batch=1` equivalence;
  strengthened WU/leaf-batch quality and playing-strength non-inferiority;
  inference/scheduler and replay/schema correctness; complete Rust hot path;
  ≥20× vs Python at equal settings on the laptop; and a confirmed Rust-only
  production configuration from the cloud sweep.
- Status: **all planned F4 implementation and conditional-F4.7 decision tooling
  is present and the full regression suite is green. The registered fast-search
  position gate failed, so F4 has not exited: strength, the quality lock, the
  registered laptop comparison, and target-box cloud confirmation are correctly
  blocked. Implementation artifacts are not substituted for those measurements.**

## Crate split ("extract at two")

Potentially shared pieces include the generic `Game`/`Eval`/`SearchConfig`
boundary shape, `splitmix64`, arena/pending-path machinery, inference batching,
and instrumentation. Physical extraction remains conditional: 7WD is a fresh
Gumbel/chance MCTS rather than a second client of KD's expectiminimax algorithm,
so extract only a concrete surface that both production clients actually use.

**F1 resolution (2026-07-19):** the physical shared-search-crate split is
deferred to F3, as this section's open questions already anticipate — F1 has no
searcher, so `seven_wonders_rust` is a standalone engine-only crate (`data` +
`state` + `engine` + `codec` + pyo3 `lib`). Nothing KD-shaped is shared yet;
F3 revisits the split when the Gumbel searcher makes the genuinely shared trait
surface concrete (open questions 1–4 below).

**F3 scoping resolution (2026-07-20):** deferred *again* — build the 7WD Gumbel
searcher **inside `seven_wonders_rust`** first (adapting KD's `splitmix64`/arena/
coalescing patterns), then extract the shared crate as a separate step (F3.5,
conditional) once the truly-shared surface is concrete. KD's search is an
alpha-beta solver and 7WD's is a fresh Gumbel MCTS, so little *algorithm* code is
shared; designing the trait boundary before the searcher exists risks churn and
puts KD's regression gates at risk early. Open questions 1–4 are answered at
extraction time, not up front.

Open questions — resolve only if production reuse triggers extraction, then log
answers below:

1. Crate layout and name; what moves (search trait, arena, coalescing, RNG)
   vs. what stays game-side (encoders, codecs, pyo3 modules).
2. Trait surface for 7WD's chance layer: do shared interfaces grow first-class
   chance nodes (4 chance kinds, search barrier, `resample_hidden`/UnseenPool),
   or does the Gumbel MCTS keep a narrower game-local chance core?
3. pyo3 boundary: one shared extension module or per-game modules over shared
   rlib internals.
4. Where KD's regression gates run after the split (CI story for two clients).

## Inherited KD lessons (pinned so they're not relearned)

- f64 throughout the tree, not f32 — f32 broke bit-identity vs Python (M4).
- pyo3 `Vec<u8>` returns as Python `bytes`, not `list` — test accordingly (M1).
- Canonical ascending-index action order; watch symmetric/ambiguous action
  anchors (M3's symmetric-domino bug has a 7WD analog risk in the identity
  codec).
- Bit-exactness gates catch reference-side bugs too (M2 found Python's
  `_compute_bag` discard bug) — treat gate failures as two-sided evidence.
- Threaded generation saturates ~4× under the GIL; the throughput gate needs
  the F4 coalescing path, not more threads (M6, Phase D measurements).

## Decision & gotcha log

(append-only; date each entry)

- 2026-07-19: doc created. E-Tier-1 launched on the Phase D toy net; F1/F2
  cleared to start in parallel — they are mode-independent.
- 2026-07-19: E-Tier-1 harness shipped (`phase_e.py` + `test_phase_e.py`):
  mechanical trap detector (guaranteed-win predicate incl. pending chains;
  extra-turn wins excluded by documented approximation), depth-2 exact
  expectimax ground truth with batched net leaves (reuses
  `expand_exhaustive`/`closed_root_exact_value`), closed/open/closed_forced ×
  sims × seeds runner with paired Gumbel seeds, selected-action Q error, and a
  trap_gap-segmented report.
  All stages resumable. Smoke on real toy-run buffers: traps plentiful
  (~1 per 2 games), many in already-lost positions — read the
  "consequential" segment for the verdict.
- 2026-07-19: **F1 landed (engine + both gates green).** Key decisions:
  - **Crate:** new `seven_wonders_rust/` (`data`/`state`/`engine`/`codec`/`lib`),
    engine-only; shared-search split deferred to F3 (above).
  - **Data table integrity:** `export_rust_data.py` generates `src/data_gen.rs`
    from `data.py` (all 73 cards, 12 wonders, 10 tokens, 3 layouts) so component
    facts cannot drift by transcription. Regenerate on any `data.py` change.
  - **RNG boundary (the crux):** Python's `state_digest` hashes the
    `random.Random` internal state, which Rust cannot reproduce. Resolution: the
    Rust engine is built from a fully-locked setup (all decks/groups/guild
    selection/progress split, extracted from `GameState.new`) and replays via
    the same simulator path `buffer.replay` uses — reveals, age deals, and the
    wonder-group flip resolve from locked state; **the Great Library draw is the
    only play-time RNG event**, and its outcomes are supplied from the recorded
    `chance_log`. So Rust needs no RNG at all for F1. This also sidesteps the
    `selected_guilds` ordering divergence that a supplied-`AGE_DEAL` path would
    introduce (`_validated_age_deal` reorders guilds; the simulator path does
    not).
  - **Fingerprint:** a canonical `Vec<i32>` over all game-logic state (RNG
    excluded), numeric-id sorts throughout; `test_rust_engine_equiv.
    logic_fingerprint` mirrors it exactly. Compared at every decision plus final
    — strictly stronger than an end-state or trajectory hash, and it localizes
    the first divergent field on failure.
  - **make/unmake (F1b):** snapshot-based undo — provides the search/self-play
    mutation API "from day one" and passes the round-trip gate; a journaled-delta
    undo is a documented optimization if profiling wants cheaper undo. The gate
    validates either implementation unchanged.
  - **Effect-parity gotcha confirmed:** only `total_coins`/`trade_coins` of a
    payment affect state (Economy rebate, Urbanism chain bonus), so the Rust
    payment search minimizes trade and skips Python's purchased-tuple tiebreak
    — verified equivalent by the mask + fingerprint gates.
- 2026-07-20: **F1 review-hardening (gate robustness + boundary safety).**
  External review found no legal-trajectory divergence; the issues were gate
  durability and future-search safety. Changes (all gates re-green):
  - **F1a durable, multi-file, honest-skip:** `test_buffer_games_equivalent`
    now iterates *all* buffer `.jsonl` (was `iter_0000.jsonl` capped at 200),
    size via `SWR_F1A_GAMES` (default 400 for fast CI; **0 = full ≥10k gate**),
    and `pytest.skip`s when buffers are absent instead of returning a vacuous
    pass. Full-corpus run recorded above.
  - **F1b whole-state + exhaustive:** derived `PartialEq`/`Eq` on `GameState`
    (and `CityState`/`TableauState`/`TableauCard`/`PendingChoice`); `roundtrip_ok`
    now compares full state, and new `roundtrip_all_ok(depth=2)` audits every
    legal action with nested LIFO undo + apply determinism
    (`engine::make_unmake_audit`), sampled on a few games each in the gate. This
    closes the reviewer's gap: the fingerprint omits `library_draws` (Python has
    no equivalent remaining-draws queue), so a journaled undo forgetting it would
    have passed the old fingerprint-only round-trip; full-state `PartialEq` now
    catches it. Snapshot undo passes by construction — the audit is written to
    stay load-bearing when journaled undo lands.
  - **Public boundary guarded:** `apply_index` now rejects any non-legal index
    (the decoder alone does not verify wonder ownership/retirement/affordability,
    so an unchecked index could mutate state illegally). Regression test added.
  - **Generated-data drift enforced:** `test_generated_rust_data_matches_python`
    fails if `src/data_gen.rs` ≠ a fresh `export_rust_data.generate()`, making
    "cannot drift" a checked property. `cargo test` gained 3 Rust unit tests
    (action-space size, fingerprint determinism/clone-equality, the make/unmake
    audit through Age I) so it is no longer empty.
  - **Deferred to F3 by design (reviewer-concurred, benchmark-driven):**
    snapshot-clone undo allocation cost and the per-slot recompute in
    `legal_actions` (unbuilt-wonder list + wonder payment) are search-throughput
    optimisations — revisit with F3 profiling, validated by the strengthened F1b
    audit. Journaled-delta undo is the intended replacement then.
  - **F3 note — chance-outcome injection:** the current engine is intentionally
    replay-only (draws supplied from `chance_log`). Search-time chance handling
    (reveals, age deals, wonder flips, Great Library) is an explicit F3 engine
    extension (`make_with_chance`-style) requiring its own differential gate vs
    the Python searcher's chance layer; do not bolt it onto F1's replay path.
- 2026-07-20: **E-Tier-1 verdict (run `runs/phase_e_2026-07-19_fixed`).**
  Clean run: 120 unique harvested positions (all `curriculum_seed`), ground
  truth on all 120 with 0 skipped, 28,800 searches (120 × {closed, open,
  closed_forced} × {32,64,128,256} sims × 20 seeds). The earlier
  `_duplicate_quarantine` dir is the buggy predecessor (dup IDs); ignore/delete.
  - **Read the consequential segment only.** Median `trap_gap` is 0.002 — the
    mechanical detector flags many traps that cost ~nothing, so the `all`
    segment's ~40–48% trap rate is noise. Only **11/120** positions are
    consequential (`trap_gap ≥ 0.25`), and ~3 carry the signal.
  - **Closed-family wins trap coverage.** Consequential trap-pick rate averaged
    over sims: closed ~22%, closed_forced ~26%, open ~28%. Closed best at every
    budget (matches the on-record prediction).
  - **Open loop → not ported.** Worst coverage AND the stale-prior signature the
    plan feared: open uniquely traps on `21:62` (40%) and `153:64` (20%) where
    closed is clean — determinization masking a specific reveal. Caveat: Tier 1
    measures coverage, not open's hypothesized equal-wall-clock *strength* edge,
    which it can't see; "not worth the separate determinizer path" is the honest
    basis, not "disproven." Python open retained as reference.
  - **Closed vs closed_forced → unresolved, kept as a toggle.** Coverage tied
    within the 11-position noise (forced worse on `107:60`/`62:50`, better on
    `152:65`); the one robust fact is forced is **3–4× slower** (1088 vs 265 ms
    @32 sims). ZeusAI used force-expansion — a real prior in its favor, but our
    data currently disagrees on cost/benefit, so it's an explicit E-Tier-2 A/B,
    not a default. Nearly free to keep since it's a flag on the closed path.
  - **Bigger finding (net, not mode).** Trap rate is flat 32→256 sims — more
    search does not rescue these; they're value/prior blind spots. `35:63`
    (gap 1.46) is a **100% blunder for every variant at every budget** — a
    reproducible, mode-independent net failure. Banked as a named regression
    fixture; drives most consequential trap mass and the ~0.31–0.42 `|dQ(a)|`.
  - **Robustness caveat.** The mode decision is sound on this data, but 11
    consequential positions is thin for the "blunder rate ≈ 0" bar (plan §9).
    Raise consequential yield before treating robustness as cleared.
- 2026-07-21: **No 7WD endgame solver or reserved solver cores.** Hidden cards
  and three buried cards per round make a perfect-information tail solver a poor
  match for the information-set game. Phase F reserves no CPU cores for one and
  relies on search expansion plus forced-child retention.
- 2026-07-21: **Full CPU self-play hot path in Rust; Python remains the control
  plane and initial inference host.** Move generation, traversal, Gumbel
  scheduling, chance handling, backup, and game advancement stay in Rust. The
  Python/Torch boundary remains initially, but enqueue, transfer, forward,
  synchronization, and decode costs are measured explicitly.
- 2026-07-21: **WU-style incomplete visits are the default parallel-tree
  mechanism.** Pending simulations increment counts used in PUCT exploration
  while completed-value means remain unchanged. This preserves per-simulation
  Gumbel accounting better than BU-style aggregated backups; Kingdomino-style
  value-penalty virtual loss is only a gated fallback.
- 2026-07-21: **Laptop locks quality; cloud sweep tunes production throughput.**
  Laptop results retain the Rust-vs-Python comparison and lock `leaf_batch`.
  The cloud benchmark runs only the production Rust path and sweeps scheduler
  and batching settings for maximum self-play games/s on the rented GPU host.
- 2026-07-21: **F4 implementation started; F4.0/F4.1 green.** Preregistered the
  numerical quality/throughput/metrics contract in `f4_contract.json`, then added
  the independent arena-backed resumable search path in `tree_resumable.rs`.
  Keeping the recursive F3.3 tree untouched made the exact refactor gate a real
  oracle comparison rather than a self-comparison. Mock, real-net, all-phase,
  pending-choice, terminal-leaf, forced-enumeration, and sampled AGE_DEAL coverage
  pass to 1e-9/full-digest exactness; full Rust equivalence module 26/26 green.
- 2026-07-21: **F4.2 WU waves landed; calibration correctly remains open.** WU
  incomplete counts, round-boundary drains, stable aligned batch requests,
  pending-leaf dedup with per-path backup, terminal bypass, cleanup, and metrics
  are gated in Rust/Python. Added the contract-driven position-quality runner and
  deterministic stratified corpus builder. The all-phase corpus smoke cleared
  structural phase coverage from one replay buffer but confirmed the known
  consequential-fixture deficit (11 available vs 50 preregistered); no quality
  lock was emitted.
