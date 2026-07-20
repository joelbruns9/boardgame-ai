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

One state core shared by MCTS and the future Phase H solver; make/unmake from
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
  - **F2.2** — port `encode()` token builders simplest-first (global,
    draft_offer, city_card, wonder, progress, discard, pool, pool_wonder,
    tableau), reusing `minimum_payment`/scoring already in `engine.rs`; lock the
    per-type feature order to `_SCHEMA`.
  - **F2.3** — full `encode_state` bit-exact gate over ≥100k sampled states.
- Status: **in progress** — codec covered; encoder foundation (F2.1) starting.

### F3 — Searcher + Gumbel root (shared crate)

Written fresh in the shared search crate (see crate split below), porting KD's
arena/coalescing/`allow_threads` scaffolding.

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
- Status: mode scope resolved; still sequenced after F1/F2. F1 green, F2 next.

### F4 — Batched inference bridge

Leaf coalescing + GIL release (KD M6 design: `py.detach`, in-process
coalescing, `leaf_batch > 1`); ONNX/tch in-process only if profiles show the
Python hop dominating.

- **Gate F4 (= Phase F exit):** ≥20× self-play throughput vs the Python loop
  at equal settings (KD achieved ~28×).
- Status: not started.

## Crate split ("extract at two")

`kingdomino_rust/src/search.rs` (generic `Game`/`Eval`/`SearchConfig`,
`splitmix64`, expectiminimax, Star-ready chance children) splits into a shared
search/NN crate; 7WD is impl #2, KD is the regression client whose existing
equivalence gates must keep passing unchanged.

**F1 resolution (2026-07-19):** the physical shared-search-crate split is
deferred to F3, as this section's open questions already anticipate — F1 has no
searcher, so `seven_wonders_rust` is a standalone engine-only crate (`data` +
`state` + `engine` + `codec` + pyo3 `lib`). Nothing KD-shaped is shared yet;
F3 revisits the split when the Gumbel searcher and Star solver force the trait
surface (open questions 1–4 below).

Open questions — resolve when F3 forces them, log answers below:

1. Crate layout and name; what moves (search trait, arena, coalescing, RNG)
   vs. what stays game-side (encoders, codecs, pyo3 modules).
2. Trait surface for 7WD's chance layer: does `Game` grow first-class chance
   nodes (4 chance kinds, search barrier, `resample_hidden`/UnseenPool), or
   does the Gumbel MCTS sit beside the expectiminimax solver with a narrower
   shared core?
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
  - **make/unmake (F1b):** snapshot-based undo — provides the solver-facing API
    "from day one" and passes the round-trip gate; a journaled-delta undo is a
    documented F3 optimization if search profiling wants cheaper undo. The gate
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
