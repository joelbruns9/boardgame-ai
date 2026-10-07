# Review request: G4 layer 2b, G10a consequence channels (encoder-8), G12 restart archive, pretrain capacity/migration, G0 policy source

Branch `sevenwd-w9-prototype` (worktree `boardgame-ai-7wd`), commits
`87a6e0c..50e02c7`, on top of the fixes for the previous review (`1f9bc26`;
review `reviews/sevenwd-g2b-g82-pretrain-8014a6c.md`, response at the end of
`GROWTH_G2B_G82_PRETRAIN_REVIEW_REQUEST.md`). Design and measurements are in
`MODEL_GROWTH_PLAN.md`: start with **"Owner decisions 2026-10-06 (cloud run)"**,
then **"G4 layer 2b"**, **G10**, **G12**. This brief is the entry point.

**Why this matters now.** These are the last builds before the final cloud run.
2b and G12 change what self-play generates. G10a changes the network's inputs
(encoder-8) and every warm start goes through its migration. A defect here gets
baked into the run that is meant to be the last.

## Scope

| commit | item | files |
|---|---|---|
| `6d31bf5` | G0 `evaluate --policy-source combined|action` (the G6 W5-only check) | `tactical_suite.py`, `test_tactical_suite.py` |
| `4f3be9d` | `pretrain --grow-layers N`: capacity probe, no-op layers on a warm base | `pretrain.py` (`grow_layers`, `GROWN_ZERO_SUFFIXES`), `test_pretrain.py` |
| `eb893d6` | **G4 layer 2b**: exact proven worlds on interior (sampled) reveal edges | `seven_wonders_rust/src/tree_resumable.rs` (`Strata`, `reveal_strata`, `closed_child`, `backup`, `edge_exact`, `OutlookFate::Affine`), `tactics.rs` (`opponent_may_win_next`, switch), `lib.rs` (`set_exact_reveal_strata`, `RustGame.reveal_strata`), `test_exact_tactics.py` |
| `0b75b42` | **G10a**: 22 consequence channels on TABLEAU tokens (encoder-8) | `encoder.py` (`CONSEQUENCE_FEATURES`, `_consequence_values`, `_Derived.revives/consequence_base`), `seven_wonders_rust/src/encoder.rs` (`Consequences`), `pretrain.py` (additive migration; `build_model` renamed `base_model`), `test_encoder.py`, `test_control_encoder.py`, `test_g2_value_contract.py` |
| `50e02c7` | **G12**: restart archive with forced branch | `restart_archive.py`, `buffer.py` (`GameRecord.restart_from`, `record_digests`), `dataset.py` (`Example.outcome_free`, prefix skip), `train.py` (outcome-free targets, masked end-of-game heads), `phase_d.py` (`--restart-*`, `_restart_plan`, `_merge_restarts`, `_update_restart_archive`), `self_play.rs` + `lib.rs` (`first_move_excludes`), `test_restart_archive.py`, `training_parameters.md` |
| `87a6e0c`, `07532ae`, `68437b4` | plan: G14 result, G6 result, final pretrain + capacity probe | `MODEL_GROWTH_PLAN.md` |

## Already gated (no need to re-verify by hand)

- **Encoder-8 parity.** Rust == Python bit-for-bit over the buffer corpus and
  40 random games (`test_encode_corpus_equivalent`, `test_encode_equivalent`).
  Stripping the 22 columns reproduces the encoder-7 golden digests literally
  (`test_stripping_consequences_reproduces_the_encoder_7_digests`). The draft
  golden is unchanged.
- **Migration.** Loading candidate_0100 or the final pretrain into encoder-8
  grows exactly one tensor (`embedder.feature.tableau.weight`, 37 -> 59 columns,
  new columns zero). Every other tensor is equal
  (`test_a_base_from_the_previous_encoder_migrates_additively`).
- **Grown layers.** candidate_0100 grown 8 -> 12 layers gives max abs output
  difference 0.0 on run07 positions. The new layers receive gradient.
- **2b.** Rust unit tests cover the backup map, the affine outlook and the
  all-proven exact edge. On bot games, the proven worlds are a superset of the
  `losing_mass` reference worlds, with value bounds checked. On/off switch
  tested. With exact tactics off, search is unchanged.
- **G12.** Restarted positions match the ancestor (legal set). The first move
  avoids the tried set. Merged records pass full `replay` (masks, actors, chance
  log, both digests) and round-trip JSON. Python and Rust derivation agree, start
  at `restart_from`, and mark every row `outcome_free`. A two-iteration Phase D
  run restarts the expected share and writes replayable buffers.
- **Tests.** Full 7WD suite at `50e02c7`: 1938 passed, 6 skipped, 2 failed.
  Both failures are the known `test_async_solver` load flake: 11/11 pass when
  that file runs alone. The only Rust lib failure is the known
  `encoder_feature_counts_match_schema` (needs the control table; fails
  identically without these changes).

## Measurements you can rely on

- **G14 (iters 81-100 + G8.2 overlay).** Checkpoint arm vs value-reset arm tie
  on every G0 class and budget (held-out loss 2.335 vs 2.338). The random arm
  was dropped by the owner (it cannot answer the plasticity question at laptop
  scale).
- **G6.** W5 alone vs combined: no difference on G0.
- **Final pretrain (encoder-7, iters 41-100), vs candidate_0100.** Raw must_block
  blunders 17.7 -> 12.7% (19 fixed / 4 broken). Near-end must_block under search
  13% -> 0-1.9%.
- **12-layer probe.** Ties: held-out 2.3261 vs 2.3258. Equal on G0 decisive
  classes. Ordinary/quiet value error 0.004 better. Inference 1.26-1.39x slower.
- **2b cost.** -4.6 / -3.5 / -5.3% self-play moves/s (32 games x 200 sims,
  3 alternating pairs).
- **G10a dropped features 1-2.** `classify_actions` costs 77 us mean, p99 1.3 ms,
  max 51 ms per position, against 17 us for the per-node G4 check.
- **G10a live rate** on 5,674 accessible run07 card tokens:
  - burying kills a science route ~70x per side;
  - discarding keeps one alive via Mausoleum ~25-30x;
  - symbol-lost fires on 2-4% of tokens;
  - military reachability is ~90% on (the bound is loose).
- **G12 cost.** Harvest 23 ms/game, ~1.1 entries/game, restart plies median 64
  of ~70. Merge 8 ms per restart game.

## Focus areas (questions I could not settle myself)

1. **2b soundness and bias** (`tree_resumable.rs`).
   - Each backed-up value through a stratified edge is mapped
     `v -> sum_proven p_k v_k + (1 - mass) v`, and open worlds are sampled with
     probability `p_k / (1 - mass)`. Is the edge Q an unbiased estimate of the
     full expectation?
   - The map is applied at the EDGE, before the parent node's combine or
     proven replacement in `backup`. Is that order right, including when a
     stratified edge sits below a `LibraryOffer` or `Max` node, or above a node
     proven later?
   - A stratified edge is exact only when every world is proven. In the
     all-proven case, one proven world is materialized and its returned value is
     ignored (scale 0). Can `propagate_proofs` / `solve_node` read that child and
     mis-solve the parent?
   - Strata are built on the edge's FIRST visit, when `children.is_empty()`. Can
     an interior edge already have children by then (a forced or cached path),
     so it is never stratified, or stratified with stale children?
   - The `Affine` outlook composition across several stratified edges on one
     path.
2. **2b screen.** `opponent_may_win_next` gates enumeration, so worlds where the
   MOVER is proven to win are only found past it. Recall only, by design.
   Confirm it cannot cost soundness. Also: is the 64-world cap
   (two-card reveals stay sampled) a reasonable boundary?
3. **G10a definitions** (`encoder.py::_consequence_values`, `encoder.rs::Consequences`).
   - The card leaves the shared obtainable set for every use. A discard returns
     it for whichever seat `revives` (an unbuilt, unretired Mausoleum, mover
     included; a live revival choice counts as not reviving).
   - A build adds the symbol to the mover and moves the track by
     `effective_shields`. A burial ignores the Wonder's own effects, including a
     Mausoleum burial and the seventh-wonder retirement.
   - The military bound subtracts `card.shields + (Strategy obtainable and red)`,
     mirroring `military_bound`.

   Are these consistent with how `reachable_cards` / `military_bound` /
   `science_missing_obtainable` are defined? Any case where a channel says
   "reachable" for a route the existing features already rule out, or the
   reverse?
4. **G10a scope.** Channels are computed for every accessible revealed card,
   including in pending-choice states where no card is actionable. Harmless, or
   misleading to W5?
5. **Migration and pretrain** (`pretrain.base_model`).
   - Always `migrate=True`; refuses a migration that would ZERO anything, and
     allows `grown` / `initialized` / `neutral`. Is allowing `initialized` right
     for a warm start? It is how a W5/W2/W3 module absent from an old checkpoint
     arrives.
   - `--grow-layers` grows from the migrated state.
   - The cloud launcher's own `--init-checkpoint` path is NOT changed here. Does
     Phase D's warm start migrate the same way?
6. **G12 same-deal restart and the Great Library** (`restart_archive.restart_game`).
   The prefix's Library draws are taken from the replayed state. One
   continuation draw is sampled from the replayed engine's RNG, to match what
   `record_digests` / `replay` will draw. Is there any path (a Library built in
   the prefix, no unused tokens, the draw order) where the Rust game and the
   Python replay disagree? The tests only replay continuations that happen to
   occur in tiny games.
7. **G12 forced move** (`self_play.rs::exclude_played`).
   - The exclusion is applied to the PLAYED distribution after the endgame
     solver mask. If the solver's optimal set is exactly the historical move,
     the forced move is a proven loss. That is acceptable exploration in my
     view: the labels come from search, not from the result. Is it?
   - If the search put no mass outside the tried set, the remaining legal moves
     are taken uniformly.
8. **G12 targets** (`train.value_targets`, `compute_losses`).
   - Restart rows replace the outcome by the search value (`value_soft`) and
     W4's joint by the search outlook, where present.
   - joint7, margin, military and science skip those rows.
   - Proofs and certain wins still override.
   - The short-term TD(lambda) target is built from later root values. Does it
     reach the terminal outcome and so leak the deal-dependent result back in?
9. **G12 bookkeeping** (`phase_d._restart_plan/_merge_restarts/_update_restart_archive`).
   - Entries are charged a restart before generation. The archive is saved only
     after harvest. A crash mid-iteration repeats restarts on resume.
   - Restarts can land on league (archive / specialist) games.
   - The archive is dominated by late positions. Anything that makes the
     archive grow without bound, or starve?

## Known limitations (I'm not asking you to find these)

- G12 v1 reuses the ancestor's hidden deal (owner decision). Reshuffling
  (`restart_from` + a portable-RNG re-deal in both languages) is deferred.
- G12 v1 has no search-vs-network-correction or policy-surprise sources, no
  priority over entries, and no drifting curriculum.
- G10a's military channels carry little signal (the bound is loose). No channel
  models who reaches a card first.
- G9 is parked until G10a priors exist. The encoder-8 pretrain is running now
  and unscored.
- No game-level strength measurement of anything here.
- Phase D launcher wiring (warm start, `latest` mode, fixed anchor, G2b
  phase-out logging) is not done yet.

## How to run the gates

```
cd boardgame-ai-7wd/games/seven_wonders_duel/seven_wonders_rust
cargo test --release --lib stratified every_world_proven
maturin develop --release
cd ../../..
python -m pytest -n 4 games/seven_wonders_duel/test_exact_tactics.py \
  games/seven_wonders_duel/test_encoder.py games/seven_wonders_duel/test_control_encoder.py \
  games/seven_wonders_duel/test_rust_engine_equiv.py -k "encode or signature or strata or digest" \
  games/seven_wonders_duel/test_restart_archive.py games/seven_wonders_duel/test_pretrain.py
```

## Sign-offs requested

1. 2b leaves interior reveal-edge values unbiased and cannot mis-solve a node.
2. G10a's definitions are consistent with the existing reachability features,
   including the owner's Mausoleum rule.
3. The encoder-8 migration is additive and safe for the warm start.
4. G12 restarts produce replayable records and the right targets, with no
   result leakage via the deal.


## Response to the review of 50e02c7 (2026-10-06)

Review: `reviews/sevenwd-growth-50e02c7-review.md`. All six findings checked
against the code; all hold and are fixed. 2b, G10a and the encoder-8 migration
were signed off. Every regression test below FAILS against the pre-fix
`train.py` / `restart_archive.py` and passes after.

| # | finding | verdict | fix | regression test |
|---|---|---|---|---|
| 1 | short-term return carries the terminal result into restart targets | valid | restart rows get ONE contract in `value_targets`, built after the ordinary targets: no outcome, no outcome bootstrap, no short-term term | `test_finding_1_...` (both derivation backends; flipping the result and the short-term return leaves flat, W4 outcome and W4 type unchanged) |
| 2 | substitution overwrote the labels certain-win overrides use | valid | `hard` / `joint_hard` stay the realised labels; the restart base is a separate tensor, then `prove()` (solver, then certain) | `test_finding_2_...` (certain + agreeing solver +1 + search 0.2 -> (1,0,0); type = the known route) |
| 3 | no outlook -> W4 fell back to realised outcome/type | valid | W4 outcome from the outlook, else the scalar search value, else a proof; type only from the outlook or a certain win, else unsupervised; rows with nothing permitted are dropped (`value_rows` / `hier_rows` weights) | `test_finding_3_...` |
| 4 | zero bootstrap weights bypassed the restart target | valid | a flat target is always built when a batch has restart rows | `test_finding_4_...` (alone and mixed with ordinary rows; value loss invariant to the realised label) |
| 5 | restarts shared `(iteration, seed)` | valid | `seed` is the restart's own job seed (identity for G0, overlays, splits); the ancestor's deal replays from `deal_seed` (`GameRecord.replay_seed`, used by `replay` and `rust_game_for_record`); written to `setup.deal_seed` only when set | `test_finding_5_...` |
| 6 | restarts could cross the holdout from their ancestor | valid | `GameRecord.family` = root ancestor `(iteration, seed)`, carried through restarts of restarts; `Example.split_family`; `stable_game_split` keys on it | `test_finding_6_...` |

Also from the review: `w0_sizing._pack_examples` now REFUSES restart rows (its
packed schema has no `outcome_free` / `split_family`). Design answers accepted:
2b expectation/order, G10a loose-bound semantics, `initialized` not being an
output-equivalence guarantee in general, the forced proven-loss move as
exploration, temperature restarting at the restart ply, non-atomic archive
charging.

After the fixes: restart archive, G2 contract, buffer, w0 sizing, Rust
derivation, Phase D, tactical suite, targeted reanalysis, pretrain, tactic
labels, Rust engine equivalence -- 234 passed.
