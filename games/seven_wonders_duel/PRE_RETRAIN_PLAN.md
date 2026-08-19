# Everything that should land before the next 7WD training run

**Written to be picked up cold.** Every number here was measured on this machine
(2026-08-16/17) and is reproducible from the committed tools. Where a number
contradicts an earlier claim elsewhere in the repo, this file is the later
measurement and the earlier one has been corrected at its source.

The retrain is the gate on everything: **no checkpoint loads on the current
encoder** (all 580 carry `7d68ff20…`), and **no recorded self-play replays**
(0 of 100 cloud6 `iter_0010` games). So there is no strength number, no error
tail, and no on-distribution cost profile until a net exists. That makes the
retrain a prerequisite rather than housekeeping — and makes it worth landing
the changes below first, since each one costs a retrain to add later.

---

## 1. Where things stand

Already committed on `sevenwd-engine-correctness`:

| | |
|---|---|
| Both BGA-found engine defects | fixed (`ec08bc1`) — discard reachability, military band off-by-one |
| Exact endgame solver in self-play | committed, **off by default** (`2a509ab`, `a820979`) |
| Oracle probe (value head vs proven values) | `solver_oracle_probe.py` (`cb5cd2e`) |
| Trigger study + per-machine calibrator | `endgame_trigger_study.py` (`373ab15`) |
| Encoder tempo primitives | `7wd-encoder-5` (`28f1413`, reordered in `ec73d4b`) |
| Selection/training split + solver on every ply | `2c18415` |
| Policy-target pruning primitive | `de6b6b3` |
| Forced playouts, all three implementations | `473cf01` |
| Pruning wired to the forced visits | `bbcd137` |
| Hybrid root mode, Dirichlet/forcing CLI, `TARGET_VERSION` 3 | §A complete |
| Head-specific solved-row weighting + seven review fixes | `06986bf` |
| Mean/max readout and the opponent-reply head | §D complete |
| Solver made per-call so gates cannot inherit it | see below |

**§A and §D are done**, and the pre-retrain critical path with them: everything
still open in §B is efficiency or instrumentation that can land *after* a run,
and §C/§E need a net to measure against.

**One pattern accounts for four defects found today**, three of them by review
rather than by tests, and it is worth naming before the next setting is added:

| leaked | how |
|---|---|
| the `130` feature width | duplicated literal, two of four sites left behind |
| `forced_playout_k` | process-global, inherited by gates |
| `puct_root && full` | no-op in gates, where `full` is always false |
| the endgame solver | process-global, inherited by gates |

Every one is a value crossing an implementation boundary and being silently
inherited or ignored, invisible to type checking. **Three of the four are gates
inheriting a generation setting**, which suggests the structural answer is not
more per-call flags but making gate invocations categorically distinct from
generation ones. Until then: new generation settings default off, are passed
explicitly, and are gated by a test that asserts BEHAVIOUR (a gate-shaped call
producing no solver attempts, a record carrying no Gumbel candidate set) rather
than a config value.

Note also that the obvious gate for the solver -- `net_by_player[actor] == 0`,
the shape Dirichlet and forced playouts use -- would have been *worse* than none:
a gate runs a different net per seat, so it would mask one side's moves and not
the other's, which is a bias rather than a handicap.

**§A is done.** Every knob defaults to off, so behaviour is unchanged until a
run sets them; the gates went in before the switches so the failure modes are
covered rather than discovered. Nothing here is blocked except by the ordering
in §3.

---

## 2. Decisions already taken — do not relitigate

**Search: PUCT for recorded moves, Gumbel for cheap ones.** Three reasons, in
order of weight. The advisor is fixed PUCT (`RustPuctSearch`, `puct_root=true`),
so training and gating on Gumbel means the only surface a human judges is the
only one off-distribution. Depth is the lever left for strength, and PUCT gives
its best move a larger share of the budget. And the hybrid is free of the
target-mixing hazard it appears to have: **cheap moves emit no policy target at
all** (`policy_excluded = !full`, and `dataset.is_fast_search_move` drops them
from the example set), so every recorded target is a PUCT visit distribution and
the buffer stays homogeneous.

**Cheap moves are where diversity is cheapest — but it is NOT free.** They emit
no *policy* target, so varying sims, width, search type and chance capping there
costs no label consistency. It does not follow that it costs nothing: cheap
actions are ~75% of moves, so they determine every later recorded state and the
final outcome that every retained row's value label comes from. Weaker cheap
play means noisier value labels and an off-policy trajectory. KataGo disables
exploratory settings on fast searches precisely to keep them *strong*
(paper §3.1).

The tension is real rather than settled — diversity is a known limiting factor
here (KD's run6 was a diversity package; the temperature floor is the strongest
single lever) — and 7WD differs from Go in having chance and ~1 outcome label
per game. So it must be **measured, not assumed either way**: any cheap-move
diversity change needs a strength constraint alongside throughput (top-action
agreement against a strong reference, and head-to-head non-inferiority of the
cheap searcher). Games/hour alone will happily buy fast, weaker trajectories.
§B2 gives that constraint a sharper metric than throughput, and a second reason
to care: cheap moves may be breaking multi-move plans outright.

**Solve at every triggered ply; no solver cache.** A cache ceilings at 40% (the
deepest solve is 60% of per-game cost), but the 40% buys the thing that matters:
per-ply masking makes both sides play the endgame provably optimally, so the
game's realised result becomes the proven value of the first solved position —
which de-noises the outcome label for *every* row of the game, not just endgame
rows. With only ~1 independent outcome label per game, that is plausibly worth
more than the endgame labels themselves.

**This is not what the code does today, and the gap invalidates the argument.**
`endgame_overlay` is gated on `full` in both generation paths
(`self_play.rs` `run` and `Slot::finish_move`). At `full_search_fraction = 0.25`
the ply after a proof is cheap ~75% of the time, played by an unmasked search,
and can discard the proven result — so the outcome is *not* the proven value and
the de-noising does not happen. §3B carries the fix: **trigger the solve on
every eligible ply, mask always, and emit the label only on recorded rows.** The
mask changes play; the label is a separate concern.

**Value targets from chance-free proofs only.** A scalar expected utility is
`P(win) − P(loss)` and does not determine a three-class distribution when draws
exist. `exact_expectimax` rows keep the realised outcome. Recovering them needs
the solver to propagate a real (win, draw, loss) distribution under a stated tie
policy; the scalar cannot be post-processed into one.

**Keep the encoder feasibility flags.** `military_bound` /
`science_missing_obtainable` are thresholds on *sums over reachable sets*;
max-pooling computes *existentials*. Different operators, not duplicates.
Removing them would force the net to learn a sum over a variable-size masked set
under a capacity constraint we already suspect is binding. The bug risk is real
and is answered by the probe suite in §6, not by removal.

---

## 3. Work items, in dependency order

### A. The PUCT switch — do this first, everything composes against it

`selfplay_search_mode` and `eval_search_mode` already exist and already accept
`"puct"`; they default to `"gumbel"`. So the arm itself is a flag. What is not
a flag:

1. **`TARGET_VERSION` bump.** PUCT's target is visit counts (`_puct_root`);
   Gumbel's is completed-Q. This is exactly what the version exists for. Batch
   every other target change (§D) into the same bump.
2. ~~**Expose Dirichlet.**~~ **This was wrong.** `--dirichlet-epsilon` and
   `--dirichlet-alpha` already exist, validate, and reach the generator. It was
   inert only because it defaults to 0.0 and PUCT self-play had never been run.
   Nothing needed building. Alpha is **1.8**, not KD's 0.3 — the branching
   differs, and at 0.3 over ~5 actions almost all the noise mass lands on one
   arbitrary move. Applies to the PUCT root only; under PUCT it is the only root
   exploration there is, so `--selfplay-search-mode puct` without epsilon is a
   mistake.
3. **Forced playouts + policy target pruning.** Not optional under PUCT:
   Dirichlet noise pollutes a visit-count target directly, which is the problem
   KataGo invented PTP for (paper §3.2). Under Gumbel there was nothing to
   subtract. Three requirements that are easy to miss:

   * **Selection and training must read different distributions.** Both paths
     today sample the played move from the *same mutable* `policy_target` they
     then record (`self_play.rs` `run` ~line 817, `finish_move` ~line 1884).
     That is correct for the solver mask — a proof should change what is played —
     but wrong for PTP, which exists to clean the *label* while leaving the
     exploration in the trajectory. Applying PTP in place before `sample_policy`
     would delete the forced exploration from the games as well.
   * **PTP needs the NOISED priors**, and `SearchResult.prior` is deliberately
     the clean pre-noise snapshot (`tree.rs` ~line 551, kept clean so KL
     diagnostics mean something). Either expose the noised priors alongside it or
     carry explicit forced-playout counts out of the search.
   * **Use KataGo's counterfactual rule, not a flat `sqrt(kPN)` subtraction.**
     Subtract from each non-max child only down to the point where it would still
     have been PUCT-competitive with the most-visited child; an unconditional
     subtraction removes visits the search would have spent anyway.

4. **Composition order** — load-bearing, and easy to get silently wrong:

   * **played move** ← raw visit distribution, proof-masked when a solve
     succeeded (never PTP-pruned);
   * **training target** ← visits, PTP-pruned, then proof-masked, then
     renormalised.

   Renormalising before the mask normalises over the wrong support.

5. **Per-move mode switch — do NOT overload `full`.** Shipped as
   `--cheap-search-mode` / `SelfPlayConfig::cheap_puct_root`, an
   `Option<bool>` where `None` means "same as `puct_root`". The obvious
   `puct_root && full` is wrong: gate games set `full_search_fraction = 0.0` and
   deliver their strength through the cheap path with
   `cheap_sims = full_sims = gate_sims`, while enabling PUCT independently
   (`phase_d.py` ~line 3832). So `full` is *always* false in a gate, and that
   expression would silently run every promotion gate under Gumbel while
   `--eval-search-mode puct` reported success. Add a **generation-only** hybrid
   flag instead, and gate it behaviourally — assert PUCT-shaped output (no Gumbel
   candidate set in the record) rather than asserting a config value.
6. **Raise `full_sims` — and there is now a number.** At 7WD's budgets, `k = 2`
   spends most of the search on forcing. Measured with the mock evaluator over
   20 PlayAge positions (ε=0.25, α=1.8), the fraction of simulations that were
   forced:

   | sims | 64 | 128 | 400 |
   |---|---|---|---|
   | forced | 70.8% | 59.9% | 32.8% |

   Analytically the total quota is `sqrt(k·N)·Σ√Pᵢ`, so the forced fraction is
   `sqrt(k·m/N)` with `m = (Σ√P)²` the prior's effective support — and a flat
   prior over `n` actions gives `sqrt(k·n/N)`, which is what the measurement
   shows. Root legal-action counts over 2,460 PlayAge plies in 40 random games:
   mean 6.0, median 4, per-age p90 16/10/6. Even at n=4 that is 35% at N=64.

   **Rule of thumb: `full_sims ≳ 16·k·m` keeps forcing under 25%.** With m≈6 and
   k=2 that is ~190 sims, so the current 64–128 is too low for KataGo's constant
   — KataGo runs it at ~1600 playouts, where the same formula is negligible.
   Either raise sims or lower k; the sweep decides which, but "k=2 because
   KataGo" without checking N is wrong.

   Caveat stated plainly: the mock's priors are near-flat and a trained net
   concentrates them, lowering `m`. No net exists yet — that is what the retrain
   is for — so treat this as an upper bound on the forced fraction and re-measure
   on-distribution. Do **not** assume forcing and pruning cancel: under the mock,
   pruning took back almost none of it (total variation 0.058 at 64 sims),
   because the guard is self-calibrating and the mock's Q gaps are narrow. That
   number is not trustworthy with a real net either, but it is not evidence they
   cancel.

   64–128 was already flagged as low for PUCT on general AlphaZero grounds; the
   forced-playout arithmetic above turns that into a constraint rather than a
   preference. This is the sweep dimension that actually costs throughput.
7. **Update the stale comments** asserting "self-play must stay Gumbel"
   (`self_play.rs`, `SelfPlayConfig::puct_root`). They become wrong; leaving them
   misleads the next reader.

### B. Wire the solver in properly

* **Trigger on every eligible ply, not only `full` ones.** The mask is what makes
  the endgame provably-optimally played, and that property is what de-noises the
  outcome for the whole game (§2). Emit the *label* only on recorded rows; a
  cheap ply still produces no example. Without this the "no cache" decision
  loses its justification.
* **Head-specific weighting, NOT row oversampling.** KD's `endgame_oversample`
  duplicates rows, and `compute_losses` averages every head over the sampled
  batch — so duplicating a row multiplies its policy and all four auxiliary
  losses too, not just the value loss it was meant to emphasise. Worse, only
  `solver_exact` rows carry an exact W/D/L label; `exact_expectimax` rows keep
  the realised outcome and would be upweighted for a certainty they do not have.
  Instead:
  * a value weight applied only to `solver_exact` rows;
  * optionally a policy weight applied only to `solver_masked` rows;
  * **no** oversampling in validation, or the metric stops being comparable;
  * a cap per game, so a run of adjacent solved plies — which share one game and
    one proof — cannot masquerade as independent evidence.
* **Cost-predicted trigger**, replacing the card cap. Attempt iff predicted cost
  fits the budget, with margin for the residual.

  *Re-measured 2026-08-18 on production self-play* (`validate_cost_trigger.py`),
  because the original numbers came from 40 bot-driven games and an ad-hoc fit
  that was never persisted — `endgame_trigger_study.py` contains no regression at
  all, so the "held-out R² 0.904" could not be rerun. The shakedown's **18,031
  solve attempts across 1,933 games** are reconstructible through `buffer.replay`,
  and the model form holds up:

  | | 40-game claim | 18,031 solves, held out by game |
  |---|---|---|
  | R² on `log10(nodes)` | 0.904 | **0.935** |
  | residual | 0.51 decades | 0.22 median, **0.68 p90** |
  | value kept vs the card cap | ~100% | **98.9%** |
  | cost vs the card cap | 61% | **56.3%** (~1.78×) |

  Two limits on that, both material:

  * **Only half the claimed mechanism is tested.** The data was generated under
    `cards ≤ 10`, so every row has `cards_left ≤ 10` and the "takes the cheap
    11s" half is *untested* — what is demonstrated is that it drops the dear 9s
    and 10s while keeping 98.9% of the proofs. The 1.78× is therefore a
    lower bound on the gain and an unverified claim about its source.
  * **The model underestimates exactly the positions that matter.** 55.3% of
    censored solves are predicted *below* the node count they had already
    reached when cut off, and their true cost is higher still. Cost is
    log-normal-ish with a long tail (uncensored median 1.1k nodes, p90 301k),
    and the tail is where the budget is actually spent. Use a margin set from
    the p90 (0.68 decades, ~4.8×), not the median.

  Held out by **game**, never by row: adjacent solved plies share a position and
  a proof, so a row-wise split leaks across it.

  **The model transfers; the card cap does not.** Both were checked against
  cloud6 endgames — the only large corpus reached by a trained net rather than a
  bot (`--from-buffer`, 572 positions from 60 games of iteration 43):

  * *Transfers.* Fit on self-play endgames and scored on cloud strong-play ones,
    R² goes 0.947 → **0.936** (residual p90 0.59 → 0.77 decades). So the cost
    model is about board structure, not about which net produced the position.
    It can be fit once and shipped, which is what §B assumed.
  * *Does not transfer.* The **absolute cost** is far higher at strong play, and
    the gap grows with depth. Fraction of positions solvable within the
    production 4.5M budget:

    | cards left | self-play (128×4) | cloud6 strong play |
    |---|---|---|
    | 8 | ≥92% | 94% |
    | 9 | ≥79% | **68%** |
    | 10 | ≥44% | **29%** |
    | 11 | — | **18%** |

    The self-play figures are *lower bounds* (its own corpus is censored, below);
    the cloud figures are exact, since a censored cloud row provably exceeds the
    20M study budget. The comparison is therefore one-directional but sound: at
    9 and 10 cards, strong play is strictly harder.

  **Consequence for the shipped trigger.** `--endgame-solver-max-cards 10` at
  4.5M nodes was calibrated on weak play. Against a trained net it declines
  ~70% of its cards-10 attempts, burning the full budget on each. On the cloud
  corpus at a 5M budget the frontier is 1% declines at cap 8, 5% at 9, 12% at
  10, 19% at 11, and 73% of all nodes wasted at cap 10. replace the cap with the cost
  model below, and re-derive it from `--from-buffer` after the retrain rather
  than from bot games.

  **Refit on cloud data (1,955 positions, 220 games, iterations 37-43).** Asked
  because the cloud corpus is the right distribution; answered by scoring both
  models on it:

  | model | fit on | R² | p90 resid | underpredicts censored floors |
  |---|---|---|---|---|
  | 22 features | 8,242 self-play rows | 0.938 | 0.80 dec | 66% |
  | 22 features | 965 cloud rows | **0.939** | 0.83 dec | **51%** |
  | 4 features (plan's) | 965 cloud rows | 0.915 | 0.95 dec | 74% |

  So refitting buys **nothing in R²** — the model form is genuinely about board
  structure — but it meaningfully improves the *tail*, which is the half that
  decides affordability. Prefer the cloud refit for that reason alone, and note
  that even it underpredicts half the censored floors: **the safety margin must
  come from the p90 (~0.8 decades, ~6.3×), never the median.**

  Corpus size decides which feature set wins, and it flips: at 281 training rows
  the plan's four terms beat all 22 (0.903 vs 0.884, variance dominating); at 965
  the full set wins (0.939 vs 0.915). Fit whichever the available data supports —
  both are reported on every run so this need not be re-argued.

  **The shipped rule is the model, not a cap.** An earlier draft of this section
  recommended "ship cap 8", which was a mistake in kind: the entire purpose of a
  cost model is a trigger that *crosses* card counts, and a cap cannot. Measured
  on 990 held-out cloud positions at a 4.5M budget with a 0.4-decade margin:

  | cards left | attempted |
  |---|---|
  | 11 | **20%** |
  | 10 | 36% |
  | 9 | 82% |
  | 8 | **96%** |
  | ≤7 | 100% |

  It attempts a fifth of the 11-card positions and skips a twentieth of the
  8-card ones — exactly the behaviour a cap cannot express. Against fixed caps
  on the same holdout:

  | rule | solves | nodes |
  |---|---|---|
  | cap ≤ 10 | 805 | 564.6M |
  | **cost model, margin 0.4** | **805** | **249.1M (44%)** |
  | cap ≤ 9 | 762 | 265.6M |
  | cost model, margin 0.6 | 782 | 164.3M (62%) |

  Identical proof count for 44% of the compute, or more proofs for 62% of it.
  The margin is the dial: 0.0 → 875 attempts/50 declines, 0.8 → 757/6, 1.2 →
  689/3. **0.4 is the shipped default** — it matches the widest cap's value while
  halving the spend, and the leftover budget is better spent on depth than on
  insurance.

  **Wired in and shipping** behind `--endgame-cost-model` (bare flag uses the
  default file); without it the card cap still decides, so every earlier run
  reproduces. `seven_wonders_rust::cost_model` computes the twenty features and
  `solver_wants` consults the model instead of the cap.

  Three things made this safe to do:

  * The model is fit on exactly the twenty features Rust can compute
    (`RUST_FEATURES`). The two that need Python's `_Derived` reachability sets
    were dropped after measuring that they cost nothing — R² 0.939 either way,
    the same 805 solves — rather than after deciding they looked unimportant.
  * `test_cost_model_parity.py` compares all twenty against `position_features`
    on 100+ real Age III positions, and checks the end-to-end verdict agrees in
    both languages. The coefficients are applied positionally, so a divergence
    would not raise: it would price every position with the wrong weights. Rust
    also refuses a weight vector whose feature names do not match its order.
  * `endgame_overlay` had a **second copy** of the eligibility test that
    `solver_wants` already owned. Installing the model in one and not the other
    would have left the async path parking a slot for a solve the overlay then
    refused. There is one decision site now.

  Verified live: a smoke run attempted 11- and 12-card positions that
  `--endgame-solver-max-cards 10` refuses, and the manifest records which rule
  chose.

  **Refit recipe** (the shipped file must come out of this, never a hand
  transcription -- nothing downstream can check the file against the fit it
  claims to come from, because `test_cost_model_parity` compares Python to Rust
  and both read the same file):

  ```
  python -m games.seven_wonders_duel.endgame_trigger_study       --from-buffer "<cloud buffers>" --out cloud_endgames.json
  python -m games.seven_wonders_duel.validate_cost_trigger       --fit-on cloud_endgames.json --write-model
  ```

  `--fit-on` fits `RUST_FEATURES` -- the twenty columns the trigger can actually
  compute -- and reports the 22-column fit alongside as a comparison. It also
  prints the margin sweep per card count (`--margin-sweep`), which is section C's
  dimension, and refuses to write a model over the wrong feature set.

  ### Corrections from the 2026-08-19 review

  Four claims in this section were wrong or unreproducible. All four are fixed
  in code; the last needs a refit before the run.

  * **`chance_wonders` did not mean the same thing in both languages.** Python's
    `_unbuilt` counted a RETIRED Great Library or Mausoleum as unbuilt; Rust's
    `unbuilt_named` did not. Measured on the fit distribution: over 150 cloud6
    positions the two languages disagreed on `revive_wonders` and
    `discard_x_revive` 21 times and on `chance_wonders` 15 times -- 36 positions,
    24% -- and `chance_wonders` carries +0.43, larger than the whole 0.4-decade
    margin, in the direction of Rust pricing a position CHEAPER than the fitted
    model does. Honest qualifier: none of those 36 flipped the trigger's verdict,
    because the only strong-play corpus available for the check is the
    proven-endgame benchmark and every position in it has 1-5 cards left, far
    below the affordability boundary. The flips would happen at 8-11 cards, where
    the trigger actually decides and where no committed corpus reaches.
    Python now excludes retired wonders (Rust's definition was the correct one:
    a retired wonder adds no branch). **The shipped coefficients were fit
    against the old definition and are marked stale in the JSON; `phase_d`
    prints the warning at launch. Refit before the retrain.**

  * **Parity was tested on a corpus that could not fail.** The bot games in
    `test_cost_model_parity` never build a seventh wonder, so no retirement ever
    happens in them -- 0 of 146 positions -- and no number of them could have
    caught the above. The test now also runs over positions from
    `proven_endgames.jsonl`, which come from the cloud6 buffers the model is fit
    on and where 68% of positions have a retired wonder, plus an assertion that
    that corpus really does exercise retirement.

  * **The margin is not the residual p90, and should not be.** `cost_model.rs`
    said it was. The shipped 0.4 was chosen because it was measured to buy the
    same 805 proofs as `max_cards 10` for 44% of the nodes, which is better
    evidence than a residual quantile. That is defensible because the loss is
    bounded both ways: an underprediction costs at most `budget` nodes and then
    declines, an overprediction costs one proof. A p90 margin would be right
    only if an underprediction could run away -- which is exactly what a binding
    wall clock allowed, so section 7 is what makes 0.4 safe.

  * **Neither trigger comparison was scored honestly.** `best_cap` maximised
    solves-per-node, which picks the smallest cap; `compare_triggers` picked the
    cap with the most solves, which -- since cap sets are nested -- is always the
    largest and most expensive cap, flattering the model's node ratio for free.
    Both are replaced by `cap_frontier` plus `matched_cap`: the cheapest cap that
    buys at least as many proofs as the model did.

  Largest terms are `chance_fanout` (+1.04) and `chance_wonders` (+0.44), ahead
  of `cards_left` (+0.27) — cost is driven by how much of the board is still
  *unrevealed*, which is what turns a minimax subtree into an expectimax one.
  That is also why a card cap is the wrong shape: two 10-card positions can
  differ by an order of magnitude in cost depending on what is face down.

  *Second cost of the wall-clock bug.* The shakedown corpus cannot answer the
  budget question at all: the 3s clock cut solves below 4.5M, so "solvable
  within 4.5M" is bounded at 44%–100% for cards 10 — useless. A binding clock
  did not merely make training data load-dependent, it destroyed the corpus's
  value for calibration. The cloud runs, bounded by nodes, stayed usable.
* **Async solving.** A `SolvePending` slot stage plus a solver thread pool and a
  results channel, mirroring how a slot already parks on an NN batch and yields
  its thread. The mask must be applied *before* the move is chosen, so the
  **slot** waits while the **thread** does not. This is what lets the budget rise
  without stalling generation.
* **Statistics into the iteration log**: attempts, declines by reason
  (`unsolvable` / `budget`), nodes spent including on failures, solver seconds as
  a share of generation, solved-position count, and the
  `net_root_value` / `root_value` / `solver_value` triple per solved row — so the
  oracle diagnostic runs continuously instead of as a separate job.
* **`exact_fallback_positions` sidecar** — declined roots, kept out of the replay
  buffer. Now straightforward: `solver_attempted` and `solver_stop` identify them.

### B2. Do cheap moves break multi-move plans? — measure before spending

A concern that is **label contamination, not diversity**, and worth separating
from both.

A full-search move's row carries the game outcome as its value label. If the
search sees a three-move plan and plays move 1, but moves 2 and 3 are cheap and
do not see it, the plan dies and the outcome records that the plan was bad. The
row that did the right thing is punished for what the following cheap moves
failed to do.

The bias is selective in the worst direction. Cheap search is prior-guided, not
random, so it reliably executes plans the prior **already knows** and drops the
ones it does not — exactly the plans worth learning. The mechanism
preferentially suppresses novel strategy while leaving known strategy intact.

**The cheap-vs-full agreement probe.** No training run needed: play games
normally, and at every CHEAP move also run a full search, discard it, and record

* how often the two choose different actions;
* the same rate conditioned on the full search's choice being **low-prior** —
  the cases where search is discovering something the net does not know, and the
  only cases where a broken chain costs anything;
* the KL between the cheap and full policies at the same position;
* and, at solved endgame positions, how often each plays a **provably losing**
  move. That is ground truth rather than mutual agreement, and it puts a number
  on how much strategy the cheap path actually drops.

`search_gain_probe.py` measures something adjacent — search versus the bare net
— but not this. The oracle probe already generates solved positions, so the
last measurement is a filter over existing output.

**Read it as a gate on how much to spend.** High agreement (~95%), or
disagreement concentrated on moves the prior already liked, means plans rarely
break and neither fix below is worth buying. Disagreement concentrated on
low-prior moves confirms the mechanism and sizes it.

**If confirmed, prefer RUN-LENGTH full search over full-sims iterations.** Make
full moves arrive in runs of 3-4 rather than as independent coin flips, keeping
the same overall fraction: compute is identical, and it targets chain coherence
directly. The cost is that recorded positions become more correlated, which
slightly narrows positional diversity.

Full-sims iterations (every move full and recorded, every Nth iteration) are the
fallback. They are **not** equivalent to raising `--full-search-fraction`: a
three-move chain survives at roughly `f^3` — 1.6% at f=0.25, 12.5% at f=0.5 —
against ~100% within a full-sims iteration. Coherence is a per-GAME property, so
a raised fraction buys many partly-coherent games where periodic full iterations
buy a few wholly-coherent ones, and for multi-move plans those are not
substitutes. Cost is ~2.5x a mixed game (0.25*96 + 0.75*20 ~ 39 against 96), so
every tenth iteration is about +15% total compute. It yields ~70 recorded rows
per game instead of ~17 but still only ONE independent value label, so the gain
is policy labels plus coherence, not value signal.

This is the same argument that rejected the solver cache (§2): masking every
endgame ply rather than only the deepest was justified because both sides
playing coherently makes the realised result mean something. That reasoning
covers the last ~7 plies today; this is the midgame version of it.

### C. The sweep

**"Maximise proven labels per hour" is the WRONG objective** — it contradicts
§4's own finding. The cheapest positions to prove are the nearly-decided ones
(cheapest decile: |value| 1.00, 94% of moves already optimal), so a
labels-per-hour objective buys exactly the labels that teach nothing, and
combined with any endgame weighting it would flood training with them.

Report and constrain a vector instead, with **games/hour held fixed** as the
invariant:

* **unique solved games/hour**, not solved rows — adjacent solved plies in one
  game are one piece of evidence;
* **chance-free value labels/hour**, counted separately from expectimax masks,
  because only the former supply a value target;
* **losing policy mass eliminated per hour** — the mask's actual information
  content, `1 − optimal/legal` summed, which is the metric the trigger study
  already used and which correctly scores a fully-decided position at ~0;
* **policy targets/hour** and their improvement over the prior;
* **strength at a fixed deployment search budget** — the only number that
  answers the question the rest are proxies for.

The all-core-clock effect (solver scaling measured at 2.89×/4, 3.77×/8,
4.37×/16 threads, ceiling attributed to clock and SMT, *not* bandwidth) then
shows up as a measured drop in games/hour that the invariant rejects.

Dimensions: solver cores vs generation cores; node budget; **cost-model
margin**; `full_sims` and `full_search_fraction` (they interact — more sims per
full move means fewer full moves per hour); `endgame_oversample`; Gumbel `top_k`
on cheap moves.

**The margin is a first-order throughput dimension, not a tuning detail.** On
the 12-iteration shakedown the solver consumed 4,432–7,387 s of CPU against
4,947 s of generation wall — **22–37% of wall time across four solver threads** —
and that run used the card cap, which the cost model undercuts by more than half
again. The margin sets that spend directly, and the trade is steeply non-linear
(held-out cloud positions, 4.5M budget):

| margin | attempts | proofs | declines | nodes |
|---|---|---|---|---|
| 0.0 | 875 | 825 | 50 | 415.8M |
| 0.4 | 827 | 805 | 22 | 249.1M |
| 0.6 | 793 | 782 | 11 | 164.3M |
| 0.8 | 757 | 751 | 6 | 107.4M |
| 1.2 | 689 | 687 | 2 | 36.9M |

From 0.0 to 0.8, node spend falls **74%** while proofs fall **9%**. Anywhere on
that curve is defensible; the point of the sweep is to find where the released
compute buys more elsewhere than the lost proofs cost.

Two cautions for whoever runs it:

* **The margin and the node budget are the same knob at the trigger.**
  `affordable()` tests `predict + margin <= log10(budget)`, so only the
  *difference* matters and sweeping both independently wastes most of the grid.
  Sweep the margin at a fixed budget. The budget carries one *separate* effect
  worth its own small sweep: it also decides when an in-flight solve is
  abandoned, so it sets the cost of the model's underpredictions.
* **The shipped 0.4 was anchored to the wrong thing.** It was chosen because it
  matches `max_cards 10`'s proof count at 44% of its nodes — a legacy cap is a
  convenient reference, not an objective. The sweep should pick the margin
  against the vector above, and should be expected to move it.

`endgame_trigger_study.py --calibrate` already sizes node budget and depth for a
given box from a stored cost profile plus a one-minute rate measurement, because
**node counts are machine-independent**. The sweep is the throughput half that
the calibrator cannot answer.

### D. Network and targets

* **Mean/max readout.** The readout is `encoded[:, 0]`, a single CLS-style
  GLOBAL token. Concatenate masked mean- and max-pools over real tokens and
  project `3d → d` so `Heads` is unchanged:

  ```python
  h = self.final_norm(encoded)
  mask = ~batch["pad_mask"]
  counts = mask.sum(1, keepdim=True).clamp(min=1)
  mean = (h * mask.unsqueeze(-1)).sum(1) / counts
  maxed = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
  readout = self.readout_proj(torch.cat([h[:, 0], mean, maxed], -1))
  ```

  Masking is load-bearing in both. The point is **max**: attention is an
  averaging operator, and "is there *any* token with property X" is an
  existential it approximates poorly. The encoder hand-codes two such
  existentials (`sci_win_feasible`, `mil_win_feasible`); this generalises the
  pattern instead of adding a third bespoke flag.

* **Auxiliary reply head.** A second policy head predicting the opponent's
  improved policy at the next decision. Target for row *t* is the **next raw
  move's** `policy_target` over its legal set. No Rust: the buffer already holds
  it. Weight ~0.15 (KataGo §3.4 supports both the idea and the small weight; the
  risk here is the local plumbing).

  **Pair during raw replay, before filtering.** Fast decisions are dropped at the
  example boundary, so "row *t+1*" in the derived list can be several actual
  decisions later — pairing there would silently supervise a position against a
  reply two plies downstream, and every shape check would still pass. Skip when
  the immediate next raw move is cheap, policy-excluded, or has the same actor
  (an extra turn makes it "what do *I* do next", a different question). Carry a
  separate reply legality mask and a validity flag.

  **It adds no information** — Q already integrates the opponent's reply, which
  is why "don't take X, it uncovers Y" is already implicit in the recorded
  target. What it adds is supervision density and explicit pressure on the trunk
  to encode opponent intent, i.e. a better **prior** — which is where the oracle
  probe located the error (raw net mean |err| 0.221 vs search's 0.191; search
  improves on the net in only 71% of proven positions).

  Land it **after** the PUCT switch: under PUCT the row-*t+1* target is a visit
  distribution, so building it first supervises against a definition about to be
  replaced.

### E. Measurement

* **A frozen proven-position benchmark — BUILT 2026-08-18.**
  `proven_benchmark.py`, 1,000 positions committed at
  `testdata/proven_endgames.jsonl` (500 KB; stored as `(seed, first_player,
  action prefix)` and reconstructed, since the source buffers are gitignored and
  hundreds of megabytes). Score any checkpoint with

      python -m games.seven_wonders_duel.proven_benchmark score --checkpoint <path>

  Composition: 507 wins / 492 losses / 1 draw, 193 with an instant-win threat.
  Standard error **0.0158** on the mean absolute error — against 0.026 at 133
  positions — so differences around 0.03 are resolvable at 2σ.

  **Two properties are forced, not chosen, and both narrow what it measures.**
  Only an `exact` proof yields a value a three-class head can be scored against;
  a solve is exact exactly when it crosses no chance edge; and no chance edge
  means no card left face down. Measured over 77 solves the separation is total —
  26/26 fully-revealed positions returned `exact`, 0 of 51 others did. So every
  position is **fully revealed**, and because full revelation only happens late,
  every position has **1–5 cards left**. The benchmark therefore measures
  "can the prior see a forced result three or four plies out on a deterministic
  board". Positions whose outcome still turns on the deal are outside the reach
  of any proven-value instrument.

  **The filter's bias is measured, not assumed (2026-08-19).** The review was
  right that `unrevealed > 0 => not exact` is not guaranteed: `regime` is set
  from `ctx.saw_chance` as the search runs, not statically from the board, so a
  position with face-down cards whose value alpha-beta settles before any reveal
  is expanded would come back `exact` and be bankable. Those are precisely the
  forced/decided positions, i.e. the threat bucket, so a bias there would fall on
  the population the bucket exists to measure.

  `--unrevealed-sample` re-measures it on every build. Pooled over three samples
  at three budgets:

  | budget | sampled | came back exact | provably expectimax | node-capped |
  |---|---|---|---|---|
  | (original) | 51 | 0 | 51 | 0 |
  | 5M | 340 | **0** | 290 | 11 |
  | 20M | 261 | **0** | 249 | 4 |

  **0 of 652 confirmed misses.** The unresolved share falls as the budget rises
  (3.2% -> 1.5%) while the exact count stays at zero, which is the signature of
  the mechanism actually being what it looks like: a position that needs tens of
  millions of nodes to settle is expensive *because* it crosses chance edges, and
  crossing one is what makes it expectimax. The worst case -- every unresolved
  position proving exact at some larger budget -- is 2.3% of skipped positions,
  which would be ~111 positions against the shipped corpus's 1,000. Treat the
  threat bucket as usable and that 10% as the outer bound on what it is missing.

  That filter is also what made the build affordable: banking 26 proofs was
  costing 45 wasted expectimax solves and 6 declines, and predicting the regime
  from the board instead removed all of them. 1,000 positions took about an hour.

  Baseline (the 128×4 shakedown net, a plumbing net not a strong one): mean
  |error| **0.619**, sign agreement **76.2%**, threat bucket 0.671 vs quiet
  0.606. Recorded so the first cloud checkpoint has something to beat.

* *(original note)* **The missing instrument.** ~1,000 endgame positions with their solver truth, generated once
  and stored. Any net is then scored by a single forward per position: no search,
  no game generation, seconds per evaluation, and **paired** across arms because
  every net sees identical positions. Letting each net play its own games would
  destroy the pairing and most of the statistical power.

  Every value metric in this repo today is self-referential — search agreement,
  net-vs-prior KL, outcome accuracy. This is the only ground truth available, and
  it measures the raw prior, which §4 and §7 both identify as where the error
  lives. Size it properly: 133 positions with a p90 of 0.919 gives a standard
  error near 0.026 on the mean, which cannot separate 0.221 from 0.19.

  Bucket it by whether an instant-win threat exists, so mean/max can be tested
  against its actual claim (existentials) rather than on aggregate.

* **Gumbel vs PUCT against proven values** — useful, but it **cannot settle §A**,
  and an earlier draft of this file claimed it could. Three reasons:
  `root_value` is the visit-weighted *aggregate* over everything the search
  explored, so it penalises PUCT for deliberately sampling inferior actions;
  top-1 agreement is nearly uninformative when 78–82% of legal moves are tied
  optimal; and the only net that runs today is Gumbel-trained and migrated, so
  the comparison measures immediate search behaviour, not which target trains a
  stronger net. Gumbel's paper claims reliable improvement *at low simulation
  counts* specifically — deploying under PUCT does not by itself make PUCT the
  better training target.

  Use instead: **per-action solver regret** for the move actually selected,
  **probability mass placed on provably-losing moves after the selection
  temperature**, restricted to a **non-trivial subset** (drop positions where
  nearly every move is optimal). Then a **small closed-loop training comparison**
  — nothing short of that answers the target-quality question.

  *Re-run 2026-08-17 against the selected action's value, and the correction was
  large*: search error mean 0.191 → **0.096**, p90 0.662 → **0.248**, sign
  agreement 92% → **98%**, "moves toward truth" 71% → **86%**. See §4. The probe
  now records both fields so the bias stays visible.

  This makes the comparison **more** meaningful than the objection implied — the
  metric now measures what a searcher concludes rather than how widely it
  explored, which is exactly the bias that penalised PUCT. The other two
  objections stand unchanged: top-1 agreement is weak under 78–82% ties, and a
  Gumbel-trained net cannot settle which target trains a stronger net.
* **Gate under `--eval-search-mode puct`** so the number promoted on is the
  number the advisor delivers.
* **Error-tail analysis** (§6).
* **Regenerate the equivalence corpus** once a net loads — closes
  `test_buffer_games_equivalent` and `test_encode_corpus_equivalent`, red only
  because they replay games the corrected engine no longer produces.

---

## 4. Measured facts — do not re-derive

**Solver cost.** ~0.03 s/game at `max_cards 8` (55k–103k nodes/game, median
475–1,089 nodes/position). Cheap enough that async is about *raising* the budget,
not about affording the current one.

**Reach at a 3s budget** (= 4.5M nodes at this machine's measured 1.51M nodes/s):
97% of 8-card positions solve, 91% at 9, 73% at 10, 56% at 11, **9% at 12**.
Doubling machine speed buys ~2% more solves at cap 10 — the cost curve in card
count is far steeper than any plausible hardware difference.

**Declines at a 2M budget:** 3% at 8 cards, 18% at 9, 33% at 10, 67% at 11, 97%
at 12. **Marginal cost per extra solve:** 0.71M nodes for 8→9, 1.79M for 9→10,
4.91M for 10→11, **66.71M for 11→12**.

**Cost predictors** (Spearman vs log₁₀ nodes, 358 positions): `cards_left`
+0.938, `unrevealed` +0.925, `moves_from_end` +0.793, `accessible` +0.496,
`legal` +0.478. Within a fixed card count `legal` still carries +0.47…+0.73.
Variability within a card count is extreme: p90/median 6.7×–35.8×, and a 7-card
position ranges 1,148 to 6,020,300 nodes.

**Expensive positions are the valuable ones.** log₁₀(nodes) vs optimal-move
fraction **−0.346**; vs |root_value| **−0.499**. Cheapest decile: |value| 1.00
and 94% of moves already optimal (the mask does nothing). Dearest deciles:
|value| ~0.87 with 51–61% optimal. Most of this runs *through depth* — within a
fixed card count the effect weakens to −0.13…−0.27.

**Ties:** 78–82% of legal moves proven optimal, reproducing the plan's 77–88% on
self-play rather than the corpus.

**Gumbel's allocation** at 128 sims — the eventual winner's share is nearly flat
in `top_k`: `m=12` → 2, 5, 12, 38 per round, winner gets **57**; `m=4` → 48;
`m=2` → 64. Sequential halving is near-optimal in allocation, so **narrowing the
candidate set does not concentrate budget on the best line.** Low `top_k` is a
*decision-quality* lever (it raises the round-zero floor from 2 sims per
candidate to 16, and 2 is a noisy basis for eliminating half the field), not a
depth lever. Depth comes from `sims`.

**Oracle probe, migrated net, 133 proven positions.** Read against the value of
the **selected action** (`SearchResult.action_value`), not `root_value`:

| |value − truth| | mean | median | p90 |
|---|---|---|---|
| net, no search | 0.221 | 0.056 | 0.919 |
| **search (selected action)** | **0.096** | **0.005** | **0.248** |
| search (aggregate `root_value`) | 0.191 | 0.021 | 0.662 |

Knows who is winning: net 122/133 (92%), **search 131/133 (98%)**. Search moves
toward truth in **86%** of positions. By distance from the end, search error
rises monotonically 0.000 → 0.207 while the net's is flat-to-noisy 0.221 →
0.315 — the expected signature of search working: near-exact at a short horizon,
degrading as depth is needed. At `from_end = 0` search is *exact* (0.000, 100%
sign) against the net's 0.221.

**The aggregate row is a trap and this file previously fell into it.**
`root_value` is the visit-weighted mean over everything explored, so reading it
against a proof charges the searcher for the losing moves it correctly rejected.
That produced two claims now withdrawn: "search barely helps" (it halves the
error and cuts p90 nearly 4×) and "search inherits the net's error rather than
correcting it", which had been offered as independent support for the
optimization-or-capacity reading of the cloud6 stall. **That support is gone.**
The error concentrates in the raw prior, which strengthens the case for changes
that improve the prior (§D) and weakens the case that search quality is the
problem.

Read one-directionally: the net is off-distribution, so a *good* score is
trustworthy and a bad one is not. The net-vs-search delta is within-subject and
largely cancels the mismatch — and the delta is the part that moved.

**Civilian VP available:** Age I 11 total, Age II 28, Age III 61, wonders 42.

---

## 5. Rejected — do not retry without new information

* **Civilian VP reachability bound.** Age I holds 11 VP in total, and the Age III
  ceiling is ~31 VP at 10 cards left against typical gaps of 5–15, so it does not
  bind until 2–4 cards remain — where the solver already answers exactly in a
  median of 10–46 nodes. Its whole informative region is where it is redundant,
  and it carries the largest edge-case surface available (Mausoleum reachability,
  Great Library, Mathematics' `VP_PER_PROGRESS` nonlinearity, top-K truncation,
  accessibility). Not a modelling weakness: **nothing is civilian-decided in
  Age I**, so there is no fact to encode.
* **`chance_fanout`** (Σ log pool size over face-down slots). Correlates +0.916
  alone but *adds nothing* over the base model and slightly hurts (held-out R²
  0.8843 vs 0.8901). The unseen pool is nearly drained by late Age III, so the
  5–20 fan-out spread it was designed for does not exist there.
* **Decidedness features as cost predictors**: `vp_gap` −0.002, `science_threat`
  +0.006, `science_max` −0.018, `military_to_win` +0.076, `pending_options`
  −0.031. Closeness *does* drive cost (proven value −0.499), but civilian VP is
  not a proxy for it — a position level on points can be decided by tempo,
  military or science.
* **`i_take_last_card` / `i_choose_next_start` as encoder features.** They are
  projections assuming nobody repeats a turn, so they are false exactly when the
  age-transition fight is live — the same shape as `sci_win_feasible` being wrong
  exactly in the Mausoleum case. Replaced by unconditional primitives
  (`cards_remaining_odd`, `military_tied`) with the conjunction left to the net.
* **`opp_can_extra_turn` as a scalar.** Every wonder token already carries
  `mine`, `built`, `affordable` and `grants_extra_turn` (Theology-aware), so it
  is a conjunction of present flags that attention aggregates.
* **Per-action conditional reply target.** Q already integrates the opponent's
  reply, so a 1202×1202-shaped output re-expresses the same tree.
* **KataGo global pooling as such.** It exists because convolutions are local.
  Attention is all-to-all and the readout is already a dedicated GLOBAL token.
  Only the mean/max *readout* transfers (§D).
* Earlier solver rejections stand: star2 probing, lexicographic margin
  refinement, transposition tables, allocation tricks — see
  `SOLVER_SELF_PLAY_PLAN.md` §9.

---

## 6. Verification the retrain unblocks

**Science-threat probe suite** — the gate `sevenwd_science_blindspot` says the
Mausoleum fix should have had: families for a symbol face-up, under a face-down
card, in the discard with an unbuilt Mausoleum, and via the Law token. Cheap now
and it protects the feature we chose to keep in §2.

**Error-tail analysis, pre-registered.** Sort proven positions by `|net − truth|`,
take the top decile, and compare bucket membership against the body:

* unbuilt Mausoleum present **and** discard non-empty (the fixed bug's exact
  trigger);
* science threat live (either side ≥5 symbols);
* military within 2 of an instant win;
* parity and `military_tied` both live (tempo-decided);
* `exact` vs `exact_expectimax` regime.

The Mausoleum bucket doubles as end-to-end verification of `ec08bc1`: if the tail
is no longer enriched there, the fix worked. If it still is, `_revivable_cards`
is incomplete. This must run **after** the retrain — the migrated net was trained
with the buggy encoder and served with the fixed one, so its errors conflate the
value head with a train/serve mismatch concentrated in exactly those positions.

Any bucket that survives should be re-checked the way `ec08bc1` was: compute what
the encoder says, compute the truth, feed corrected features to the same weights,
and see whether the read moves. That is what turns "the net is bad here" into
"this feature is wrong."

---

## 7. Standing risks

**Capacity — and the arithmetic says the worry points the wrong way.** The cloud
config is **d_model 384, layers 8, heads 6, bf16 — 14.90M parameters** (cloud5
and cloud6 both). Against that:

| addition | params | share |
|---|---|---|
| reply head | 462.8k | 3.1% |
| mean/max readout | 442.8k | 3.0% |
| both | 905.6k | 6.1% |

Heads are only ~0.47M of 14.90M; the trunk is ~14.4M. So if capacity were
binding, the head count is the wrong term to economise on.

More to the point, cloud6 stalled after 38k games — **38k independent outcome
labels against 14.9M parameters**. That is not a capacity-starved profile. It is
the profile `train.py` already describes at `VALUE_WEIGHT_DEFAULT`: "~16,500
policy labels but only ~1,000 independent outcome labels … the side of the
objective best placed to memorise, and on a shared trunk memorising it drags the
representation the policy head depends on." So read
"optimization-or-capacity" as **memorisation** first.

That changes the reply head's status from a cost to tolerate into an on-mechanism
fix: it adds ~646k *distributional* labels per 38k games, dense supervision
exactly where the value head is starved. It also fits §4 — search reaches the
proven value at 0.096 mean error and 98% sign agreement while the raw net sits at
0.221 and 92%, so the net is not failing to search, it is failing to absorb what
search already knows.

**Falsifiable prediction:** if memorisation is the story, the reply head should
**narrow the train/val gap** — the symptom named in the project notes as "the gap
widening while `policy_top1` keeps improving". If it widens the gap instead, the
memorisation theory is wrong and the head is buying nothing.

The reply head is also **train-only** — nothing consumes an opponent-reply
prediction during generation — so it can be skipped in the inference path for
zero throughput cost (`fuse_for_inference` is precedent). Mean/max *is* on the
inference path, but it is two pooling ops plus one `3d → d` matmul on a `[B, d]`
vector, against a trunk over ~100 tokens.

Note the §4 correction narrows where the problem can be. Search reaches the
proven value closely (mean |err| 0.096, 98% sign agreement) while the raw net
does not (0.221, 92%) — so **the search is not the weak link**, and neither is
the target it produces. That points at the prior and at whatever stops the net
absorbing what the search already knows, which is a capacity/optimization
question rather than a search one. It is not proof — one migrated net, 133
positions — but it is the first evidence that distinguishes the two, and it
argues for §D (prior quality) over §A's sims increase as the strength lever.

**Feature selection by reasoning has a poor record here.** Three features argued
from mechanism did not survive: `chance_fanout`, the decidedness set, and the VP
bound — two refuted by measurement, one by argument. The one proposal that held
up (tempo) came from an engine rule plus an external observation, not a story.
Prefer the error tail (§6) as the selection method.

**Wall-clock budgets break reproducibility --- and this one bound.** A solve
bounded by seconds makes generation irreproducible from `(seed, net)`. Bound by
nodes; keep `max_secs` as a non-binding safety net.

*Measured 2026-08-18, and the warning was already being violated by this plan's
own launch command.* The 12-iteration shakedown ran `--endgame-solver-max-secs
3` alongside a 4.5M node cap. It declined **3,146 of 27,787 solves (11.3%)** and
**not one was node-capped**: `nodes_max` peaked at 3.71M. Every decline was the
clock. The per-iteration decline rate tracks generation throughput at
**r = -0.817** (25.9% in the slowest iteration at 0.413 games/s, 7.4% in the
fastest at 1.011) --- so which positions got a proof depended on what else was
using the machine. At the implied ~1.2M nodes/s the node cap needs ~3.75s to
bind, so 3s sat just underneath it. The launch command now passes 30.

The general lesson is sharper than "prefer nodes": a seconds limit set anywhere
near the node budget silently *becomes* the budget, and it fails asymmetrically
--- hardest positions on the busiest machines, which is exactly the subset the
solver exists to answer. The corpus gate has the same flaw — `BUILD_MAX_SECS` is
a deadline, so gate *coverage* varies with machine load (83, 85 and 86 of 86
positions across three runs with no code change). Worth pinning to the node
budget alone.

*The diagnosis above had to be INFERRED, and no longer does (2026-08-19).*
`SolveStop::Budget` was one value for "nodes ran out" and "clock ran out", so
"not one was node-capped" was read off `nodes_max` peaking below the cap rather
than off the buffer. The variant is now split into `Nodes` and `Deadline`, and a
record's `solver_stop` says which. Three things follow:

* `validate_cost_trigger` **drops deadline-censored rows from the fit** instead
  of imputing them at a floor. A node-capped decline is a right-censored
  observation of the position's cost; a clock-truncated one is an observation of
  how busy the box was, and `study_rows` used to floor both at the study budget
  --- so a solve cut off after 2M nodes entered the fit claiming 40M.
* `solve_endgame` **always returns a dict**, carrying the nodes spent and the
  reason, instead of returning `None` and discarding both. `solve_root_counted`
  already kept the count; only the Python boundary threw it away.
* `--study-secs` now defaults to 1200 (was 120), and both the study and the
  buffer analysis print a warning naming the count when any decline was a
  deadline.

*Record identity under async solving has a precondition, now stated.* The
`test_async_solver` gate proves the async and synchronous paths produce
byte-identical records, but only while the clock is slack: a deadline decline
depends on wall time, so under contention the two paths can decline different
positions. The gate now asserts no `deadline` stop occurred during the identity
run, and a companion test pins that a binding clock is visible and distinct in
the record.

*Solver threads are PER SHARD.* `run_many_pipelined` builds one `SolverPool` per
scheduler loop, so `--solver-threads 4 --rust-scheduler-workers 4` is sixteen
CPU-bound threads, not four. `configure_solver_threads` now prints the product
against the core count and warns on oversubscription --- which matters here
rather than elsewhere because slower solves in wall time are what push a run back
toward the deadline decline this section is about.

**`full` does not mean "recorded".** It means "this move drew the full simulation
budget". Gates set `full_search_fraction = 0.0` and get their strength from the
cheap path, so any rule written as `... && full` silently becomes a no-op in
evaluation. This already produced two P1 defects in the first draft of this plan
(the PUCT gate switch and the solver trigger). Prefer explicit
generation-only flags, and gate them on observed behaviour rather than on a
config value.

**Aggregate root values.** `SearchResult.root_value` is the visit-weighted mean
over everything the search explored, not the value of the move it chose. Any
comparison that reads it penalises a searcher for exploring — measured at ~2× the
error and 4× the p90 (§4), and it will bite PUCT harder than Gumbel because
Dirichlet noise puts visits on moves the net already believes are bad.
`action_value` is now recorded alongside it; prefer it for anything compared
against a proof.

**Schema position is load-bearing, not cosmetic.** Additive encoder features must
be **appended**. `migrate_state_dict` warm-starts an older checkpoint by
zero-padding grown projections, and that only aligns at the end of a vector;
inserted mid-vector the padding maps old weights onto different features. The
2026-08-17 tempo pair was inserted mid-vector, which cost the ability to serve
any existing checkpoint at all until it was moved. Two related gaps were fixed
with it: `migrate_state_dict` could not pad growth along the *input* dimension
(only dim 0), and `load_evaluator` refused **any** surgery rather than only the
unrecoverable kind — so the "additive-schema warm start" the code documents had
never actually been reachable. The line now is: `zeroed` is fatal (a parameter
has no counterpart, the net is partly random), `grown` is fine and loud (zero
weights on appended input columns contribute nothing, exactly).

**An architecture switch must travel with the weights, at every site.**
`--pooled-readout` and `--reply-head` change what parameters exist. The switch
therefore has to reach every place a model is rebuilt, and the first smoke run
found two places it did not:

1. `phase_d` wrote checkpoint configs at three sites (manifest contract,
   bootstrap, per-iteration), of which only `train.py`'s CLI copy had been
   updated — so a checkpoint was saved with `readout_proj` weights and a config
   that did not mention them. The strict reload failed on the *next* iteration.
2. `ModelAgentSpec` — the picklable recipe a gate rebuilds from — carried
   `heads` but not the two switches, and four call sites rebuilt from it.

Both are now single sites: `PhaseDLoop._model_contract()` and `_model_from_spec()`,
each reading the flags off the built model rather than off config, so a
checkpoint cannot record something the weights contradict. Note the failure was
loud but *late*: it surfaced at the first reload, which on an overnight run is
hours in. This is the fifth instance of the session's recurring shape — a value
crossing an implementation boundary and being silently defaulted — after the
duplicated feature width, the `forced_playout_k` global, `puct_root && full`, and
the solver global. The mitigation that keeps working is the same: **one
definition site, read from the object that was actually built, plus a test that
round-trips it.**

**Duplicated constants.** The padded feature width was a literal in four places
and adding two features left two behind, surfacing as a matmul shape error naming
nothing. All four now derive from the schema. The class of bug is worth watching
for elsewhere.

---

## 8. Suggested order

1. §E first bullet — Gumbel vs PUCT against proven values. Cheap, and it either
   confirms or overturns §A before anything is built on it.
2. §A — the PUCT switch, with Dirichlet, forced playouts + PTP, the per-move
   mode split, and one `TARGET_VERSION` bump covering §D as well.
3. §B — solver wiring: oversample, cost-predicted trigger, async, statistics.
4. §D — mean/max readout, then the reply head.
5. §C — the sweep, on the finished loop.
6. Retrain. Then §6, and regenerate the equivalence corpus.

Steps 2–4 all bump the same target version, so they should ship together or the
buffer fragments across definitions.

### On testing the network changes before the run

An earlier draft required a short identical-buffer ablation of mean/max and the
reply head. **That is now mostly not worth running**, and it is worth being
precise about why rather than quietly dropping it.

Its purpose was to catch capacity competition. At 3% each on a 14.9M model whose
problem is more likely memorisation than starvation (§7), that risk is small.
And the only buffer available to ablate on would be generated by the migrated
net — weak play, unrepresentative distribution — so a null result would mean
little in either direction.

What it cannot do at all is settle *benefit*: the reply head's claimed mechanism
is better prior → better search → better data, and a fixed buffer cuts that at
the first arrow. Any static ablation measures one turn of a flywheel.

Replace it with three cheaper things that answer real questions:

* **Unit tests on the reply-target pairing.** The failure mode that would
  genuinely poison training is the join being off by one — fast decisions are
  dropped at the example boundary, so pairing there silently supervises a
  position against a reply two plies downstream, and every shape check still
  passes. A test with cheap moves interspersed settles that definitively, and
  far more cheaply than a training run.
* **The train/val gap during the retrain**, against cloud6's logged trajectory
  as a reference. This is the §7 prediction and needs no separate run.
* **The frozen proven-position benchmark** (§E) — ~1,000 solved positions with
  their truth, stored once, scored by one forward per position. It makes
  prior quality measurable on *any* net in seconds, paired across arms, and it
  is the ground-truth evaluation set this repo does not otherwise have.

**Honest limit: attribution across five simultaneous changes is not cheaply
solvable.** PUCT, the solver, tempo features, mean/max and the reply head all
land together. Sequential runs would attribute but cost weeks. The pragmatic
choice is to ship together and *instrument* — benchmark for the prior, train/val
gap for memorisation, solver statistics for label yield — so that a failure is
**diagnosable even when it is not attributable**. That is a trade, not a
solution, and it should be named as one.

Cheap-move diversity is the exception: it still needs its strength constraint
measured rather than assumed, because a weaker cheap searcher degrades every
value label in the buffer (§2).
