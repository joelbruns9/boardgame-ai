# Review request: G2b tactic relabelling, proven-loss guard, G8.2 targeted reanalysis, windowed pretrain

Branch `sevenwd-w9-prototype` (worktree `boardgame-ai-7wd`), commits
`5638347..f28a78e`, on top of the fixes for the previous review (`eea095a`;
review `reviews/sevenwd-growth-g0-g4-6ab4342.md`, response table at the end of
`GROWTH_G0_G4_REVIEW_REQUEST.md`). Design context and measurements live in
`MODEL_GROWTH_PLAN.md`: start with **"Final run preparation"** and **G2b**. This brief
is the reviewer's entry point; the plan is the long record.

**Why this matters now.** This code produces the starting point of what should be
the last cloud run. The run07 buffers are corrected (G2b) and partly
re-searched (G8.2), and the base network is pretrained on them (`pretrain.py`).
The result then warm-starts self-play. A target bug here gets baked into the
pretrained weights before any self-play can wash it out.

## Scope

| commit | item | files |
|---|---|---|
| `5638347` | **G0 compare**: per-case readings, paired McNemar + game-clustered bootstrap between two checkpoints | `tactical_suite.py` (`evaluate --save-readings`, `compare`), `test_tactical_suite.py` |
| `242b7a3` | **G2b**: exact one-move tactics folded into training targets | `dataset.py` (`apply_tactic_labels`, `Example.tactic`), `seven_wonders_rust/src/derive.rs` + `lib.rs` (`derive_records(tactic_labels=)`), `phase_d.py` (`--tactic-labels`, example-cache key), `g3_offline_ab.py`, `test_tactic_labels.py` |
| `c1b5651` | G2b **on by default**; **proven-loss guard** at the search root; G3 dropped | `tree_resumable.rs` (`proven_losing_root_edges`, `guard_distribution`, `guard_action`, `proven_loss_guard_tests`), `phase_d.py`, `test_bga_extract.py`, `test_phase_d_example_cache.py` |
| `e389ac8`, `f28a78e` | **G8.2** targeted reanalysis: overlay of re-searched buffer positions | `targeted_reanalysis.py`, `dataset.py` (`apply_reanalysis`, both backends), `g3_offline_ab.py` (`--reanalysis-overlay`, `derive_window`), `test_targeted_reanalysis.py` |
| `8b3278f` | arena `--exact-tactics` (default on, both sides, restored on exit) | `arena.py` |
| `b03d9e7`, `f09ed29` | **windowed pretrain** with the G14 init axis; `--presentations-per-row` | `pretrain.py`, `test_pretrain.py` |
| `b65ef5c`, `bda6c6a` | plan: final-run sequence, owner decisions 11-13 | `MODEL_GROWTH_PLAN.md`, `training_parameters.md` |

## Already gated (no need to re-verify by hand)

- **Python and Rust derivation agree** with G2b labels and with a reanalysis
  overlay applied: move indices, policy targets, root values, `reanalysed` flag
  (`test_tactic_labels.py`, `test_derivation_applies_the_overlay_identically_in_both_backends`).
- **`classify_actions` (Rust) == Python reference**: the predicates were gated at
  16,676 positions in the previous review; G2b only consumes them.
- **G0-sealed games never enter training or the overlay**. `derive_window`
  drops them and asserts none got through; `select_record` returns nothing for
  a sealed game (tested).
- **The overlay resumes without duplicates**, and selection is reproducible per
  game across a resume (`test_the_overlay_is_well_formed_and_resumes`,
  `test_selection_is_random_within_reason_and_reproducible_per_game`).
- **Pretrain resume** restarts from the last finished window with model and
  optimizer state (`test_pretrain.py`).
- **Tests at `f28a78e`:** 87 passed over the affected files (tactic labels,
  reanalysis, pretrain, tactical suite, exact tactics, G2 contract, example cache,
  parameter docs, Rust derivation); Rust `proven_loss_guard` passes. Full suite
  at `c1b5651`: 1906 passed. The only failure there was the known
  `async_solver` flake, which happens under load and passes when run alone.

## Measurements you can rely on

All are on sealed G0, paired, with 300 cases per class.

**G2b three-arm offline A/B** (run07 iters 91-100, from candidate_0060):

| | uniform -> G2b-uniform |
|---|---|
| must_block blunders, 64 sims | 14.0 -> 7.7% (19 fixed / 0 broken) |
| must_block blunders, 800 sims | 12.0 -> 5.7% (19 / 0) |
| immediate wins taken | 89-92 -> 92-95% (0 broken) |
| reveal_trap, predecessor, solver, quiet, ordinary | unchanged |

Plain retraining (candidate_0060 -> uniform) **doubled** near-end must_block
blunders, 16.7 -> 31.5%. The cause is run07's move targets, which come from searches
without exact tactics. G2b more than undoes it.

**Base+** (candidate_0100 pretrained on corrected iters 61-100 at about 2
presentations per row), compared with candidate_0100: raw must_block blunders 17.7 -> 13.0% (p 0.004);
near-end at 64 sims 13.0 -> 1.9% (6 / 0); forced-loss value error 0.25 -> 0.16;
**raw reveal-trap picks worse**, 30.3 -> 34.7% (8 / 21, p 0.024). The reveal-trap
difference is not significant under search.

**G2b on run07 iter 100** (1,000 games): 6.4% of rows labelled. Must_block move
targets had more than 5% mass on a proven-losing move in 107 of 255 policy rows.

**G8.2 probe** (204 positions, re-searched at 1,600 sims): the move target
changed by total variation 0.30, the top move changed in 28-29%, and the value
moved by 0.16 (pre_decisive) / 0.07 (reveal). Cost is 0.42 s per position,
coalesced.

## Focus areas (questions I could not settle myself)

1. **G2b label semantics** (`apply_tactic_labels`).
   - Forced win available: the value is pinned to +1 and the move target is
     restricted to the winning moves.
   - Every move loses: the value is pinned to -1 and the move target is left alone.
   - must_block: losing moves are zeroed and the rest renormalised, uniform if
     no mass is left.

   Is anything wrong when the row is a cheap-search row (`has_policy`
   false), a retained proof row, or a row whose existing solver proof
   disagrees with the label? The code keeps a chance-free solver proof and replaces an
   expectimax one. Does the uniform fallback hand the network a bad target
   when search put all its mass on losing moves?
2. **Proven-loss guard** (`tree_resumable.rs`, `into_puct_result` / `into_result`).
   The guard zeroes proven-losing root moves in the target and the move
   choice, using `classify_actions` on the root plus `edge_exact`.
   - Does any root reader (root value, `root_completed_q`, the pruned PUCT target,
     metrics, the advisor's visit-based panel) now disagree with the guarded
     target in a way that matters?
   - Is the sign handling (`self.sign * value <= -1 + 1e-9`) right for both seats?
   - Is "all moves lost -> guard off" the right fallback?
3. **G8.2 target type and leakage** (`targeted_reanalysis.run`, `apply_reanalysis`).
   - Re-search runs with `puct_root=True, top_k=16, force=True`, with exact
     tactics forced on. The overlay replaces the row's move target and
     `root_value`. Is mixing these targets with run07's own targets coherent,
     i.e. the same target type and the same temperature?
   - `apply_reanalysis` sets `root_value_shaped=False` and
     `root_outlook=None`. Does any value head lose or mis-read its target on a
     re-searched row?
   - **Leakage:** the overlay will be produced by base+, which trained on these
     same games (value targets blended with the realised result). Does
     re-searching with it put the realised outcome back into the "search"
     value, making the labels overconfident? If so, is iteration-disjoint
     re-searching worth the cost?
4. **G8.2 selection** (`select_record`). The reasons are pre_decisive (within 4
   plies before a position where `classify_actions` has any nonzero label),
   reveal, and cheap. They are filled in that order, at random within each reason, up to
   a per-game cap. A planned cap-5 run on iters 81-100 gives about 87% pre_decisive and
   13% reveal. Is "any nonzero label" the right decisive trigger (it includes
   positions where the mover has a win)? Does pre_decisive-first starve the
   reveal class, which is the one base+ got worse on?
5. **Pretrain loop** (`pretrain.py`). The model and optimizer carry across
   10-iteration windows. Warmup runs on the first window only. The step count comes from
   `presentations_per_row`. Validation is a per-window game-hash split.
   - Are there drift or forgetting problems from training windows in
     sequence rather than shuffled?
   - Is the optimizer state carried correctly across windows, and is
     `--resume` exact?
   - Does `--init reset-value` reset every value-side module (`VALUE_MODULES`
     prefixes) and nothing else?
6. **G0 compare statistics** (`tactical_suite.compare`). McNemar on paired
   fixed/broken counts, plus a bootstrap clustered by game. Are the two networks'
   readings aligned case by case? Is the cluster bootstrap correct when classes
   share games?
7. **Arena default change** (`arena.py`). Exact tactics are now on by default for
   both sides and restored on exit. Any arena comparisons whose meaning silently
   changed?

## Known limitations (I'm not asking you to find these)

- G2b is one move deep. Predecessor positions (a loss two or more moves later)
  and partial-loss reveals keep run07's targets. G8.2 covers only a sample of them.
- G0 is biased towards any network trained on run07: the sealed games are
  withheld from training, but the same openings and policies made them.
- No game-level strength measurement (arena) of base+; candidate_0100 was
  chosen as the base without an arena match (owner decision 13).
- G3 code stays in the tree, off by default, with no plan to use it.
- The guard does not change the advisor's live panel, which reads visits.
- Python search has no exact-tactics mode; G2b labels come from Rust.

## How to run the gates

```
cd boardgame-ai-7wd/games/seven_wonders_duel/seven_wonders_rust
cargo test --release --lib proven_loss_guard
maturin develop --release
cd ../../..
python -m pytest -n 4 games/seven_wonders_duel/test_tactic_labels.py \
  games/seven_wonders_duel/test_targeted_reanalysis.py \
  games/seven_wonders_duel/test_pretrain.py \
  games/seven_wonders_duel/test_tactical_suite.py \
  games/seven_wonders_duel/test_exact_tactics.py \
  games/seven_wonders_duel/test_rust_derivation.py
```

## Sign-offs requested

1. G2b targets are correct and cannot teach a proven-losing move.
2. The proven-loss guard is sound and leaves no inconsistent root reader.
3. G8.2 overlay targets are safe to train on, including the base+ leakage question.
4. `pretrain.py` is fit to produce the warm-start checkpoint (and the three G14 arms).


## Response to the review of 8014a6c (2026-10-05)

Review: `reviews/sevenwd-g2b-g82-pretrain-8014a6c.md`. Every finding was checked
against the code; all five hold and are fixed. The overlay had not been
generated yet, so finding 3 is fixed at the source (the overlay now carries
the re-search's outlook) rather than at the loss boundary.

| # | finding | verdict | fix | regression test |
|---|---|---|---|---|
| 1 | random arm's W5 alpha frozen at 0 | valid | `pretrain.py` runs `refit_alpha` on each window's held-out rows for every arm, before the window checkpoint (so `--resume` starts from it); `--alpha-step` defaults to a jump to the fit | `test_every_arm_refits_w5_alpha_after_each_window` |
| 2 | reply targets keep proven-losing mass | valid | `dataset.sync_reply_targets`: after G8.2 and G2b, an eligible reply label is the following row's FINAL move target (eligibility rules unchanged; unchanged rows keep their bytes), both backends | `test_reply_targets_follow_the_corrected_move_targets` |
| 3 | W4 gets none of the re-searched value | valid | overlay schema 2 carries the re-search's `root_outlook`; `apply_reanalysis` sets it and `search_lambda=0` (the re-search is unbiased); schema-1 overlays are refused | `test_w4_trains_toward_the_re_searched_outlook`, `test_an_overlay_without_outlooks_is_refused` |
| 4 | specialist projection enables unfiltered policies | valid | `apply_tactic_labels` filters the stored policy regardless of `has_policy` (still the loss mask) | `test_a_cached_general_row_projects_to_the_directly_derived_specialist_row` |
| 5 | case-level McNemar on clustered cases | valid | `compare` drops `mcnemar_p`; adds `game_signflip_p` (game-level sign-flip test); fixed/broken counts kept as description | `test_fixes_concentrated_in_one_game_are_not_significant` |

**Measured after the fixes.** Run07 iter 100 (unsealed games): 2,290 reply
labels, 0 differ from the following row's final move target, 0 of the 139
following a G2b-corrected row keep mass on a removed move. Base+'s alpha
refitted on iters 91-100 held-out rows: 0.712 vs the 0.768 it carries
(held-out CE flat 1.32, W5 alone 0.93, mixed 0.91) -- base+ stays the
reanalysis teacher as is. The same numbers show finding 1 mattered: a W5-off
random arm would have served the flat head's 1.32.

**Earlier p-values withdrawn.** The case-level McNemar p-values quoted above
and in `MODEL_GROWTH_PLAN.md` (G2b A/B, base+) overstate significance; the
fixed/broken counts and the game-clustered intervals stand.

**Design answers accepted:** proof conflicts, guard reader qualification
(`root_completed_q` and `root_value` are not proof-authoritative), teacher
dependence (labels are bootstrapped training data, not ground truth), reveal
budget (measure overlap before reserving), sequential-window forgetting, and
the arena programmatic-vs-CLI default.

Tests after the fixes: 31 passed (pretrain, tactic labels, reanalysis,
tactical suite) + 156 passed (parameter docs, Rust derivation, example cache,
G2 contract, exact tactics, action alpha, specialist league).
