# Review request: growth plan G2, G4 / G4b / G8.0, G0 and G3

Branch `sevenwd-w9-prototype` (worktree `boardgame-ai-7wd`), commits
`343cd68..adc40bf` on top of G1 (`49cf9d4`). Design context and measurements
live in `MODEL_GROWTH_PLAN.md` (G0, G2, G3, G4, G4b, G8 sections) -- this brief
is the reviewer's entry point; the plan is the long record.

## Scope

| commit | item | files |
|---|---|---|
| `343cd68` | test fixes: per-table log skips, stubbed W9 checkpoint | `test_tableau_control.py`, `test_threat_corpus.py`, `test_w9_reference_case.py`, `test_training_parameters_doc.py` |
| `5d2b2c6` | **G2** value-target contract per head and proof type | `train.py` (`value_targets`, `_utility_loss`, `compute_losses`), `dataset.py` (`certain_win_moves`), `buffer.py` (`replay(on_events=)`), `derive.rs` (per-move chance counts), `w0_sizing.py`, `phase_d.py` |
| `62d0a68`, `4979084`, `c54c4e3`, `d90270d` | **G4** exact tactics in search: proven wins, proven losses, extra-turn and civilian wins, proof propagation (MCTS-Solver), default on | `tactics.py` (Python reference), `seven_wonders_rust/src/tactics.rs`, `tree_resumable.rs`, `phase_d.py`, `advisor_adapter.py`, `web_app.py`, `conftest.py` |
| `8bb906c`, `5807469` | **G4b + G8.0** option expansion of Mausoleum / Great Library choices; Library valued by the exact best-of-offer formula | `tree_resumable.rs` (`Combine`, `LibraryOffer`, `make_library_child`, `offer_weights`, option children) |
| `798dfdb` | **G0** tactical suite | `tactical_suite.py`, `tactics.py` (`classify_actions`, `losing_mass`), `tactics.rs` |
| `adc40bf` | **G3** priority sampling (off by default) + offline A/B script | `priority_sampling.py`, `train.py` (`sample_weights`), `phase_d.py`, `g3_offline_ab.py` |

## Already gated -- please do not re-verify by hand

- **Rust tactics == Python reference** on every position tested:
  `forced_win` / `forced_loss` 0 mismatches on 16,676 positions (bot games +
  run07 iter 60 self-play + their children; 438 wins, 223 losses);
  `classify_actions` and `losing_mass` in `test_tactical_suite.py`.
- **Python and Rust derivation agree** on `certain_win` (`test_rust_derivation.py`
  scalar fields, `test_g2_value_contract.py`).
- **Legacy contract reproduces the old W4 loss exactly**
  (`test_legacy_hierarchical_loss_is_the_old_joint_nll`).
- **`offer_weights`**: 0.6/0.3/0.1/0/0 for 3 of 5, sums to 1 for all n, d
  (Rust unit tests).
- **Exact-tactics off == historical search**: the Rust switch is off by default
  and `conftest.py` resets it before every test, so every Python/Rust
  equivalence gate still runs plain search. Full 7WD suite was green after G4
  layer 1b; later commits ran the affected test files (G3's tests: see
  Known limitations).

## Measurements you can rely on (laptop, run07 iter 60, `riccp_923216750_review/g4_native_tactics.py`)

| | baseline | full G4 + G4b |
|---|---|---|
| Great Library edge Q, ~1k sims | 7.5-7.7% | 50.9-52.0% |
| Great Library edge Q, 16k sims | 6.9% | 51.5% (verified route 51.43%) |
| losing University decision, top move at ~4k | Build University 3/3 | abandoned 3/3 |
| self-play sims/s vs off (3 x 32 games per arm) | -- | ~-2.5% |

G0 smoke (iters 99-100, 60 cases/class): must_block/deep blunders 23.5% raw,
11.8% at 64 sims; reveal traps picked 42% / 33%.

## Focus areas -- what I could not settle myself

1. **Proof soundness at the edges** (`tactics.rs`, `tactics.py`). Any position
   where the predicates claim a forced result that is not one. Particular
   worries: the reach screens (`within_reach`, `replay_reach`) as NECESSARY
   conditions -- especially the claim that play-again wonders add no shields
   or science, so without Theology only the civilian reach widens; the loss
   screen's "+1 card" civilian allowance; Theology-acquired-mid-turn; pending
   chains after an extra turn; anything `apply_with_chance(&[])` silently
   accepts on a pending option that Python would reject.
2. **Proof propagation and stat resets** (`propagate_proofs`, `solve_node`,
   `edge_exact` in `tree_resumable.rs`). Solving resets a node's and a
   deterministic edge's value sums to the exact value. Is any reader (root
   stats, completed-Q, Gumbel halving, outlook accumulation, policy targets)
   left inconsistent? Is skipping propagation under a specialist leaf bias the
   right boundary?
3. **The `LibraryOffer` node** (`make_library_child`). It is built through the
   engine with an arbitrary valid draw, then its pending options are widened to
   the whole pool. Any engine state that depends on WHICH tokens were drawn
   beyond `pending.options` (we believe none: draws are consumed from the
   queue, and `unused_progress_tokens` is only filtered at resolution) would
   corrupt the token afterstates. Also: the `select` rule that skips exact
   options inside an offer node, and the forced-root path that drops the
   node's cached network seed.
4. **G4b max backup** (`backup`, `combined_value`, `Combine::Max`). The plan's
   contract: one simulation per expansion, recompute the max from current
   child Q, never lock in cached values. Does the seeding (one visit per option
   at its network value, cached priors for the first real visit) honour it? Is
   the measured ~1 pt optimism at ~4k sims the expected winner's curse or a
   bookkeeping error?
5. **G2 targets** (`value_targets`). Owner decided NOT to mask W4's victory
   type on proof rows (the realised type given the realised outcome is still a
   valid sample of P(type | outcome)). Is that argument sound when the proof
   and the realised outcome disagree? Is the BCE-on-expected-utility form for
   expectimax proofs a proper scoring rule as claimed?
6. **`certain_win_moves`**. A win the mover reached alone with no chance event
   between is labelled exact, including the victory TYPE. The type is one the
   winner could force, not necessarily the only one -- is training W4 on it as
   a one-hot target acceptable?
7. **G0 labels and metrics** (`tactical_suite.py`). Class definitions, the
   near_end/deep split (own_win / forced_loss are dominated by game-ending
   moves), `found_win` being strict ("took the immediate win"), and whether
   the sealed split (20% of games by hash) is enough isolation given that the
   plan wanted family-level sealing.
8. **G3** (`priority_sampling.py`). The priority definition (mean of two
   mean-normalised signals, proof rows pinned at the cap), the cap semantics,
   and whether one no-gradient pass per training call with the
   about-to-train model is the right refresh. Is anything in Phase D's two
   training sites passing weights that no longer align with
   `train_examples`?

## Known limitations (not asking for these to be found)

- **G3's tests (`test_priority_sampling.py`) were written but not yet run**;
  G3 code has not been executed at all.
- No game-level strength measurement of any of this; G4 default-on is an
  owner decision on the RICCP evidence plus throughput.
- G0 lacks the plan's unplayed-alternatives class, sims-to-refute, burial
  checks and family-level sealing.
- G8.0 afterstate sharing is nearly idle once the Library is one offer node
  per reveal; kept, not removed.
- `--exact-tactics` is Rust-only; the Python searcher has no such mode.
- The off-distribution network read of a five-option Library choice is used
  for priors only (owner accepted).

## How to run the gates

```
cd boardgame-ai-7wd/games/seven_wonders_duel/seven_wonders_rust
cargo test --release --lib offer_weight
maturin develop --release
cd ../../..
python -m pytest games/seven_wonders_duel/test_exact_tactics.py \
  games/seven_wonders_duel/test_tactical_suite.py \
  games/seven_wonders_duel/test_g2_value_contract.py \
  games/seven_wonders_duel/test_priority_sampling.py -n 4
```

RICCP harness (GPU, minutes): `runs/seven_wonders_duel/riccp_923216750_review/g4_native_tactics.py`
(in the main checkout's gitignored `runs/`).

## Sign-offs requested

1. G4 proofs are sound (no false proven win / loss / solve).
2. G4b / LibraryOffer bookkeeping matches the plan's contract.
3. G2's target contract, including the owner's no-masking decision.
4. G0 is a fit instrument for judging G3.
5. G3 is ready for the offline A/B as written.
