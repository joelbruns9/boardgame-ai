Review of Can't Stop advisor and XG-inspired search, 2026-09-30

Reviewed `69323ea` in `C:/Users/joeld/projects/boardgame-ai-cantstop` against `games/cantstop/REVIEW_REQUEST.md` (`5dca6f9`). No source was changed. The installed `cantstop_rust` .pyd (built 2026-09-29 19:33) is newer than the last `lib.rs` edit (19:32), so the Python suites were run against a native binary that matches the reviewed source. Probe: `reviews/cantstop_xg_telescope_probe.py` (run from the worktree root with `PYTHONPATH=.`).

**Recommendation:** the implementation is sound where it matters most. The control variate has zero conditional mean. The Hoeffding algebra and union bound are correct. Seat and value conversions and the RNG separation hold up. I found no correctness defect in the rollout transitions. What should stop the experiments is fitness for purpose, not correctness. As configured, the adaptive backend cannot separate actions closer than about 14–17 win-percentage points. It will almost always return the baseline move, so an adaptive arena would produce a null result regardless of whether the search is any good. Separately, the default first progressive stage (H=0) recomputes, by sampling, a number the baseline table already holds exactly. Fix both before spending arena time. Do not use search output for training targets or the live advisor until an arena shows a gain.

**Findings, in priority order**

1. **[P1] The adaptive stopping rule cannot resolve realistic decisions, so the adaptive backend is the baseline in practice.** `adaptive_search.py:59`. The radius `sqrt(2 ln(2PJ/α)/n)` is a valid Hoeffding bound for differences in [-1,1]. It ignores variance, though: the bound assumes the worst-case variance for that range (proxy σ = 1). At the default budgets (32,128,512), α = .05 and n = 512, it is 0.137 for a stop/roll pair (P=1), 0.152 for three moves and 0.171 for six. The actual paired standard deviations in your own H=1 pilot (`rollout_variance_step3_final_cuda_20261001.json`, 48 pairs per mode, p4_pilot iter 80) have these medians:

   | mode | median paired SD |
   |---|---|
   | plain | 0.152 |
   | shared dice (raw) | 0.1075 |
   | dice luck (adjusted) | 0.009 |
   | both (adjusted) | 0.005 |

   A normal-theory 95% half-width at n=512 would be 0.0093 on raw shared-dice samples and 0.0004–0.0008 on adjusted samples. The Hoeffding radius is 15× the first and 170–320× the second. In sample terms that is roughly 200× and 30,000–100,000×. Reaching a radius of 0.02 for a single pair takes about 24,000 samples.

   Consequence: an action is eliminated only when it is worse by more than about 14 points under the baseline's own policy. The decision benchmark showed the net already agrees 95% on direction when the margin exceeds 5 points, so unresolved decisions fall back to the baseline choice (`:203`). The three step-5 positions all stayed unresolved, and budgets 4/8 had radii ≥ 1, so that was structurally guaranteed.

   The plan (§ adaptive, "Small budgets often leave close choices unresolved") acknowledges the conservatism but does not size it. Options, in the order I would try them:
   - (a) A paired t or batch-means interval on the adjusted differences, with Bonferroni correction over P·J, explicitly labeled asymptotic. With SD around 0.01 this resolves 1-point differences within a few hundred samples.
   - (b) An empirical-Bernstein bound on raw shared-dice differences. This keeps a finite-sample guarantee but gains only about 2×, because the range term dominates at n ≤ 512.
   - (c) For H=1, skip sampling entirely; see assessment point A2.

   The request's "no normal-distribution assumption" is a legitimate choice, but it costs four to five orders of magnitude in samples here. Make that choice deliberately.

2. **[P2] The H=0 stage reproduces the baseline table exactly. It is not an estimate.** `progressive_search.py:21` and `adaptive_arena.py:66` default to stages (0, 1). Under the baseline continuation policy, the dice-luck corrections telescope: each roll's expected value equals the previous realized post-roll value, and the last realized value is the stop leaf, bust leaf or win vector that forms the raw endpoint. Every corrected H=0 sample is therefore exactly `T0.value(force_action(state, a))`, which is what `TurnTableBackend` returns.

   Measured on 20 ProgressHeuristic positions (2p, all phases): adjusted SD ≤ 1.1e-16 and mean minus baseline ≤ 1.1e-16 for every candidate, while raw SD was up to 0.28. Without dice luck, H=0 is a noisy estimate of the same exact number. Either way, stage 0 adds cost and no information; filtering on it is filtering on baseline values plus noise.

   Fix: seed stage 0 from `TurnTableBackend` values at zero cost and start rollouts at H ≥ 1. Plan line 56 ("agrees statistically") and line 415 ("in the compact exact reference case") understate this: the identity holds for every position and every evaluator. This also explains `decision_step1_nn_final`'s 50/50 match. It also removes the stage-reuse selection bias. Stages share seed and offset, so the final stage's first samples replay the selection samples' root-turn dice. The plan documents this bias, but it only bites when stage 0 is noisy.

3. **[P2] Evaluation failures silently become baseline moves inside an arena that still reports a verdict.** `adaptive_search.py:197` catches `MemoryError`, `RuntimeError` and `ValueError`. That covers CUDA OOM (a `RuntimeError` subclass), `RolloutLimitExceeded`, "inference batcher is closed" and the evaluator's own contract check. It returns a fallback decision with status `interrupted`. `adaptive_arena.py:44` records `stop_reason` per decision, but `:117` computes `verdict()` without counting or refusing on them. A run with a broken evaluator, or full-game rollouts that hit `max_rolls`, would report a clean baseline-versus-baseline null.

   Fix: add per-reason fallback counts to the report, and withhold or flag the verdict when any `evaluation_failure` occurs. For future target generation and advisor use, raise on `evaluation_failure`; keep the fallback only for `time_budget` and `cancelled`.

4. **[P3] `MAX_UNFINISHED` is pooled across the ten variants.** `phase4.py:183`. One variant can stall in up to about 50% of its games while the iteration total stays under 5%, because each variant is roughly a tenth of the games. The stall that caused this was variant-specific (2p/5-col). Apply the threshold per variant. The selection bias from dropping stalled games is small at ≤ 5%.

   One side note: at `td_lambda=0` the rows from an unfinished game have valid bootstrapped targets, since they do not need the outcome, so dropping them discards exactly the stall states. That is acceptable, but it is a choice.

5. **[P3] `exploratory` candidate selection compares different players' values when depth ≥ 2.** `lib.rs:373` stores each node's own mover's component. `turn_search.py:136` then takes the max across nodes of different movers, and at an opponent's node "exploratory" favors leaves that are good for the opponent. At the default `depth=1`, only root candidates exist (`:130`), so current runs are unaffected. Use the root actor's component before enabling depth ≥ 2.

**Assessment requested by the brief**

A1. **XG transfer.** The mechanics transfer. The payoff does not transfer in the same way. XG's rollouts earn their keep because no exact solve exists within a backgammon ply sequence. Here the within-turn chance is already solved exactly, so rollouts can only add value across turn boundaries. Multiplayer max-n, where each seat maximizes its own component, is implemented consistently in the table tie-breaks, `choose_move` and the rollout policy. The documents do not overclaim expectiminimax: `turn_search.py` and `progressive_search.py` both state "not full N-turn expectiminimax".

A2. **What H=1 with dice luck actually estimates.** Apply the same telescoping per turn. A corrected H=1 sample equals `E1 + [T1.value(next player's post-opening state) − NN(root-turn end board)]`, because the next turn's opening roll is the only uncorrected roll. So the H=1 search:
   - (i) can change a decision only where the net disagrees with its own one-turn solve (a Bellman residual);
   - (ii) has randomness only from the root-turn end board and one opening roll, which is why the SD is 0.005–0.009.

   Correcting the opening roll with a turn-start solve (about 4.6× the leaves) would leave only end-board variance. The expectation over end boards is what reach-weighted expansion (`WholeTurnSearch`) computes deterministically. Rollouts are the right tool at H ≥ 2 or for full games. At H=1 they approximate a deterministic computation you already have.

A3. **Statistical meaning.** The control variate is zero-mean for the baseline table and for `WholeTurnSearch`, whose `bust_value` is the backed-up leaf, consistent with its `value()`. Future opening rolls get zero correction, as documented. The Hoeffding bound and the union over P·J are valid for a fixed rollout policy, including under adaptive removal. The fallback is surfaced (`fallback`, `stop_reason`). Decision `value` and `options[0]` can disagree under fallback; that is documented.

A4. **Concurrency and recovery.** These are safe enough for long offline runs.
   - Dropping `unsendable` is safe as written: no `PyTurnSolver` method calls `py.detach`, so the GIL serializes access and PyO3's borrow flag guards the `&mut` `leaf_reach` cache. Revisit this if any method is later detached.
   - The shared root is a plain `RustTurnSolver`, which is immutable after construction.
   - `BatchingEvaluator` has no deadlock: each requester holds at most one chunk and `max_pending ≥ max_rows`. `close()` orders `None` after every accepted request, and worker exceptions reach every future in the batch.
   - Stage commits are transactional. Resume is validated against the source, native binary and checkpoint hash.
   - Parallel candidates are slower (1.09 s vs 0.80 s) because rollouts are GIL-bound Python. Keep serial as the default.

A5. **Do the tests justify experiments?** On correctness, yes. Results are recorded below. On design, fix findings 1 and 2 first; otherwise S-series arenas on the adaptive backend measure the baseline.

A6. **Before training targets or live advisor use:** findings 1–3, an arena gain for the chosen search configuration, and a raise-not-fallback policy on evaluation failure. The advisor itself (turn cache, reachability check via `KeyError`, stale-capture rejection, blocking detection, win-on-stop) looks correct. It serves only the baseline turn solver, as stated.

**Test results**

- `pytest games/cantstop/tests` (main venv, worktree root): **753 passed in 33 min**, exit 0.
- `node extension_cantstop/test_*.cjs`: all five pass (capture, panel, requests, speed, turn identity).
- The step-5 artifacts were inspected, not rerun. Their budgets of 4/8 give Hoeffding radii of at least 1, so "unresolved" was guaranteed; they measure throughput only.
