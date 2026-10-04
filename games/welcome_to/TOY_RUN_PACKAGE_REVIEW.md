# Welcome To: toy-run package review

Reviewed 2026-10-03, against the Welcome To code at `5da865f`. HEAD advanced to `a735175` during review; that commit does not change Welcome To. This answers [the review request](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/TOY_RUN_PACKAGE_REVIEW_REQUEST.md), taking its reported large equivalence gates as established and incorporating the relaxed, temporary-helper policy.

**Keep the combined toy experiment, but fix the paired-root sampling and make the helper cutoff apply to training data before launching it. Keep encoder v4. Keep steering as an experiment whose success must include execution without steering.** The package is a credible way to investigate the missing skills; its current evaluation design does not yet justify choosing every component for cloud training.

| Requested sign-off | Verdict |
|---|---|
| Steered playouts | **Change, then proceed.** Repair the complete-street helper, measure the execution gap, and enforce an actual end to helper-data training. |
| Everything-on run | **Proceed after the targeted changes below.** Respect the owner's choice of a package experiment. Separate arms are needed for component attribution later, not before this feasibility run. |
| Encoder v4 | **Accept for the toy and as the cloud baseline candidate.** I found no representation or slicing defect requiring another ABI break. Final cloud acceptance should depend on unassisted strength and throughput. |

## Findings that affect the planned run

**F1 — P1: the paired sampler systematically removes four-player games.**

[paired_targets.py:108](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/paired_targets.py:108) shuffles the games, takes about 150, sorts those by seed, then takes the first 300 resulting roots. Player counts are assigned in contiguous seed blocks before job dispatch is shuffled. Truncating the sorted roots therefore preferentially removes higher player counts.

This is present in the saved experiment: **all 6,000 roots from iterations 36–55 contain 4,209 two-player roots, 1,791 three-player roots, and zero four-player roots.** Thus A's evidence and supervision do not cover the stated 60/30/10 population. This matters especially for margin, because the maximum over opponents behaves differently with more opponents.

Select roots in a seeded random order before the final cap, preferably stratifying by player count and turn bucket. Sort only the final selected set if stable rollout indices are desirable. Also record requested versus realized root counts: choosing `ceil(roots/2)` source games can underfill the quota when many sampled positions have no siblings.

**F2 — P1: iteration 12 stops new helpers, but most subsequent training still uses helper data.**

[s2_run.py:130](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_run.py:130) schedules generation; [paired_targets.py:142](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/paired_targets.py:142) and [s2_replay.py:325](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_replay.py:325) retain the earlier data without a helper filter.

For the proposed 500 games per iteration and default replay schedule:

| Training iteration | Ordinary replay includes | Consequence |
|---|---|---|
| 12 | 7–12 | Restart/forced-deal training persists; paired data also includes steered roots. |
| 15 | 9–15 | The four-iteration paired buffer has cleared pre-cutoff roots; ordinary replay has not. |
| 18 | 11–18 | The last helper-generation iteration is still in ordinary replay. |
| 19 | 12–19 | First iteration with both buffers clear of helper-generated data. |
| 20 | 12–20 | Only the second such training iteration. |

Unassisted evaluation is meaningful throughout; it is a different claim from continuing to learn with all helper data switched off. The plan currently conflates these claims. For its stated independence experiment, filter helper trajectories and steered/forced-deal paired roots at the cutoff while retaining older natural data. Log their actual sampled training shares. This need not erase the learned weights or optimizer history. If historical helper examples are intentionally retained, change the interpretation and allow a longer clean training tail.

**F3 — P2: the pool helper pursues complete-street progress in a street that cannot complete.**

The `COMPLETE_STREET` branch of [pool_rescue.py:89](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/pool_rescue.py:89) checks pool capacity but omits the roundabout requirement. The encoder's [pool_target_boxes](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/plans.py:794) correctly requires an existing roundabout or one still available.

Reproduction: deal `(0, 24, 14)`, place roundabouts at `(0,0)` and `(1,0)`. The helper reports streets `{0,2}` as needing both pools and parks; the encoder and `requirements().street_serves` correctly identify only street `0` as viable. Street `2` has no roundabout and neither remains available. The helper can therefore prefer irrelevant progress there even while preserving the real live alternative. Reuse the same viable-street predicate in both places and add this exhausted-roundabout regression.

**F4 — P2: A silently includes forced-deal games despite its ordinary-game contract.**

[paired_targets.py:101](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/paired_targets.py:101) excludes restarts and assisted games but does not exclude `game.plan_ids is not None`. Its compact `_KEEP` payload also omits forced-deal provenance. This is valid outcome data, but it makes A's distribution and held-out metrics partly curriculum-conditioned and complicates F2's cutoff.

Choose deliberately: either add the missing natural-game filter, matching the request, or retain and tag forced-deal roots and report/sample them separately. My default for this run is to match the stated ordinary-root contract.

**F5 — P2: paired caches do not have the replay system's semantic resume protection.**

[paired_targets.py:73](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/paired_targets.py:73) returns any existing `pairs.pt` before checking settings. The work manifest uses a checkpoint path without a content hash and omits encoder/table/target versions. [sibling_probe.py:245](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/sibling_probe.py:245) checks chunk root keys and future count, but not snapshot identity, steering mode, model identity, or rollout seed. `load_window` accepts unstamped afterstate arrays.

A fresh directory avoids this today. A restart after changing helper settings, replacing a checkpoint at the same path, or editing semantics can silently reuse incompatible labels. Stamp final files and chunks with a common recipe digest covering checkpoint content, source corpus, engine/encoder/target identity, continuation policy, seeds, candidate selection, steering and utility settings. Validate even when the final file already exists. The benchmark's explicit re-encoding from snapshots is a good separate compatibility mechanism.

**F6 — P2: the tiny-corpus fallback still splits families.**

The normal hash split is fixed, but [s2_train.py:168](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_train.py:168) moves one trajectory when a side is empty. With one source and one restart of source family `1234`, I reproduced family `1234` in both training and validation. Paired roots use the hash without this fallback, so their assignment can also disagree with a moved ordinary trajectory.

For small corpora, require two distinct families or use an explicit family assignment shared by ordinary and paired data. An independently recomputed fallback can still change membership as the corpus grows. This is unlikely at 500 games, but the blanket durable-family guarantee and smoke validation are currently false.

**F7 — P2 for experiment readiness: several promised measurements are absent.**

Per-plan counts/rates exist in generation JSON, but are not copied into `progress.jsonl`; completion turns are not implemented by `_plan_and_point_metrics`. The progress list also omits bis/refusal point costs. There is no paired-root breakdown by player count, source type, steering, actual interventions, or helper-data training share. These omissions obscure precisely the failure modes this experiment is meant to diagnose. Add those counters and the execution-gap measurement below before relying on the run for a cloud decision.

## Answers to the twelve questions

**Q1. Are steered continuations a sound bridge?**

Yes, as a temporary policy-improvement experiment. They estimate a different conditional quantity. If `pi` is the learner and `mu` applies the pool rule to it, the two targets are approximately `Q^pi(s,a)` and `Q^mu(s,a)`, with the specified opponent continuation held fixed. Both are real terminal outcomes. That does not make them interchangeable predictions of what the current player will achieve.

The critical missing link is execution. A currently labels one placement, without collecting policy targets for the later card, pool, park, roundabout and claim decisions that make the steered result possible. Its score regressor can learn the benefit before the deployed policy learns the necessary continuation. Mixing the two label types into one unconditioned head also produces a weighted compromise while both are present.

There is a relevant precedent in [AggreVaTe](https://arxiv.org/html/1406.5979v1): evaluate an action using an expert's subsequent cost-to-go, then improve the policy on learner-visited states. Its assumptions and policy-learning step matter. This package is a related experiment, not an implementation inheriting that paper's guarantees.

Use a fixed audit subset with the same roots, candidates and future seeds under both continuations. Measure:

1. **Execution gap:** `Q^mu - Q^pi`, in score, realized margin and pool-plan completion. Report its mean and distribution, including negative effects.
2. **Ranking transfer:** how often steering changes the preferred candidate, and how the steered-preferred candidate actually performs under `pi`.
3. **Calibration:** predicted values versus unsteered outcomes specifically on live-pool states, separated from the easy non-pool population.
4. **Follow-through:** override frequency by decision type and actual unassisted pool-plan completions. A low override rate alone is insufficient: the learner might simply stop visiting useful states.
5. **Deployment behavior:** natural-deal play with the actual 200-simulation learner, plus policy-only evaluation. Search may close an execution gap that the raw policy does not.

Success is rising unsteered return/completion with a shrinking need for intervention after the cutoff. Improved steered validation alone is insufficient. If value improves while execution stalls, extend outcome comparisons to card/effect choices and collect useful continuation states for searched policy training. Do this before increasing the steered loss weight. [Expert Iteration](https://arxiv.org/abs/1705.08439) is relevant to that planning-to-policy feedback loop; it does not establish that this particular helper will work.

**Q2. Can A be credited for the iteration-36–55 improvement?**

The timing, independent-future sibling result and promotions make A a plausible contributor. Ten preceding flat iterations do not supply the missing counterfactual. Continued optimization, changing replay, league composition and ordinary random variation remain confounded. The shuffled-label probe supports learning useful placement contrast in that experiment; it does not establish the amount of whole-game strength caused by A.

Honor the owner's from-scratch comparison decision. The proposed v4 package versus historical v3 measures the package, including encoder and helper changes. It cannot attribute the gain to A. Before allocating large cloud compute specifically to A, compare a current-code v4 run with A against the same recipe without A, preferably across multiple seeds. A continuation from iteration 35 is a cheaper alternative, not mandatory if proper from-scratch arms are used.

Same seed does not make historical aggregate metrics fully paired. Forced deals and restart states differ, and the opposing leagues evolve differently. Use the intersection of natural seeds for paired developmental summaries; use a fresh fixed evaluation league and deal bank for strength. Report both game/iteration exposure and wall time, because A adds substantial simulation work.

**Q3. Is the easy-plan deal curriculum reasonable, and is the floor enough?**

The 25% starting parameter is reasonable for a bounded experiment. Plan identity is observed, so a changed deal distribution does not intrinsically falsify the conditional targets. Finite network capacity and unequal exposure can still harm transfer. Natural-deal evaluation, per-plan denominators and player-count breakdowns are essential.

The `0.03` floor guarantees sampling support, not successful discovery. A never-completed plan has probability `0.03 / sum(rate + 0.03)` within its stack. For illustration, with eleven plans, one rate of 0.70 and ten zeros, each hard plan gets about 2.9% of forced deals in that stack. It still receives natural exposure, but exposure by itself has already failed to produce pool skills.

The schedule also supplies less help than the headline numbers imply: iteration 2 uses restart share 18.18%, and deals 22.73% of the remaining eligible games, about 18.6% of all games when all requested restarts are available. A's steering begins at iteration 3 at 24.55% of eligible live-pool roots, not 30% of all 300 roots. In the old 920-root dataset only 169 roots have a live pool plan; a similar distribution would steer about 14 roots at that iteration. That is an illustration, not a forecast for a new policy.

Completion-rate weighting reinforces existing successes. Independent per-stack weights also ignore joint difficulty: estate plans can compete for houses already consumed by another claim. Log triple-level and two-to-three completion rates. Once simple triples work, shift some curriculum mass toward intermediate success rates or measured learning progress, while preserving natural coverage. A persistent archive of legal successful triples/near-completion states would support this better than only the last iteration. The analogous idea of expanding backward from achievable goals is supported by [reverse curriculum learning](https://arxiv.org/abs/1707.05300); its robotics results are not evidence of a Welcome To gain.

**Q4. Is linear decay to iteration 12 sensible, and should A taper?**

The explicit zero deadline is sensible. The exact linear shape has no evidential privilege. Here it starts decaying before deals/restarts begin and before A begins; from-scratch learning may not discover the relevant behavior before the helper becomes rare. Record actual successful helper examples, not just nominal fractions. A short initial plateau followed by decay is a reasonable later alternative if the first run shows inadequate exposure.

Fix F2 so iterations 12–20 can actually test learning on natural, unsteered data. Otherwise only training iterations 19 and 20 satisfy that definition. Generation metrics for iteration `i` describe candidate `i-1`, so the apparent clean tail is shorter still when reading generation curves.

A itself need not switch off. It uses empirical outcomes and can remain a useful signal indefinitely. Do not taper solely because the repeatedly consulted 143-root benchmark reaches 80%. Maintain a current-policy, fresh-root audit alongside that historical benchmark. Reduce A's weight or rollout budget when marginal unassisted strength gain per compute falls, fresh decision regret stops improving, or paired gradients impair ordinary policy/rank/calibration. Retain a small fresh stream and monitor whether performance regresses after reducing it. No automatic benchmark-based taper currently exists in `s2_run`.

**Q5. Are the new features valuations? Should embeddings be pooled?**

The kind, street, estate multiset, point values, and requirements describe rules or state. They do not prescribe a tradeoff between plans and other points. They are a useful inductive bias within the revised constraints. The pool-target planes use conservative feasibility information; “potentially usable” is more accurate than an exact guarantee of eventual completion.

Keep the shared MLP and ordered concatenation. The three legal stacks have different distributions; claim actions and output heads refer to specific slots. An invariant pooled vector could discard slot correspondence unless another path restores it. With only three slots, attention is optional capacity to test later, not a needed correction. Shared parameters can transfer concepts across plan families even though a particular legal card always belongs to one stack.

**Q6. How should third-plan endings become learnable? Is the value design opposing them?**

Train and evaluate the actual decision to close a game, not completion rate alone. The useful sequence is: two plans done; third feasible; third claim available; choose claim or wait; handle the rest of the turn; finish ahead. Opponents may still act before the end is settled. Construct comparisons from legally reached positions and continue to the real terminal outcome.

The current restart generator rewinds any completed plan, so it can keep teaching easy first plans. Add archive strata for two-completed/third-live states and observed three-plan finishes, including descendants of restarts once their provenance is represented correctly. Grow rewind distance when unassisted completion from a stratum becomes reliable. Extend A's candidate coverage to claim/pass, necessary card/effect choices and roundabouts; same-card placement comparisons alone cannot teach all of that sequence.

The `end_trigger_*` heads are auxiliary predictions. They do not directly penalize or reward ending on plans in the search value. All-zero plan-ending targets explain why such a head learns near zero; promoting that auxiliary prediction to a reward would be the wrong repair. Keep the terminal objective tied to winning/competitive return and evaluate early endings while ahead versus while behind. Maximizing plan endings or pool-point share can produce a worse player if it ends losing games efficiently.

There is a real value concern: the blend gates score-margin influence by rank confidence. At a uniform two-player rank distribution, confidence is zero, and **all of A's direct score improvement contributes zero through the margin term at that leaf**. More generally, uncertainty is not proof that score information should be suppressed. This deserves Q7's isolated experiment.

Finally, “never” needs its stated ordinary-generation scope. Saved promotion reports contain plan-ending games at iterations 25 and 50 (1/300 each) and 55 (2/300). These are any-player ending counts, not proof that the learner executed each finish; inspect those rare trajectories if retained. They are potentially useful archive seeds, and they show why generation-only zeroes should not be generalized to every evaluation setting.

**Q7. Should margin replace the leaf blend?**

Not as another uncontrolled change inside this toy. It is a strong separate test candidate. Freeze a checkpoint, opponents, natural deals, search settings and budgets. Compare the existing blend with a blend without confidence gating and with a bounded margin-only utility. Run actual games, then compare normalized rank/win rate, realized margin, raw score, endings and time. Use paired intervals over deals and retain a confirmation deal bank.

Change leaf and terminal valuation consistently in Python and Rust. `alpha=1` alone does **not** create margin-only search: the confidence multiplier remains. Keep value ranges comparable so the same exploration coefficient remains interpretable; distinguish utility changes from scale changes. Test equal simulation budgets first and equal wall time for deployment relevance.

Also distinguish the metrics: `score_decisions` judges sign accuracy using **terminal blend differences**, even for the margin chooser. `margin_score_regret` is regret in the learner's own score; the prefix identifies the chooser, not a realized-margin target. Add realized-margin and rank/win decision measures explicitly.

At more than two seats, `own predicted mean - max(opponent predicted means)` is not the expected realized margin against the best opponent. Likewise, blending expected heads is not the expectation of the nonlinear terminal blend. These are approximation issues already present in v3; F1 currently removes the most relevant four-player A examples. If head-to-utility inconsistency remains important, test a direct empirical terminal-utility target separately.

**Q8. Is paired-loss normalization and weighting correct?**

Yes mathematically, for the intended per-root objective. Every root averages over candidates and real seats; the difference term divides by `C(C-1)N`. Both pair directions are counted, and the zero diagonal contributes nothing. Padded seats are excluded. All afterstates use viewer zero, and the corresponding target seat order matches that viewer.

For prediction errors `e_c = predicted_c - target_c`, the difference loss equals `2C/(C-1)` times the mean squared candidate-centered error, averaged over seats. With three candidates it is **three times** that centered error. Thus pair weight 25 means coefficient 75 on centered error within A, in addition to its absolute anchor. This emphasizes action differences as intended; it does not create three independent pieces of evidence from three correlated pairs.

The ordinary objective group averages rank and score, giving ordinary score MSE coefficient 0.5. At `pairs_weight=1`, A adds absolute-score coefficient 1 and difference coefficient 25, each on its own normalized batch. Sixteen paired roots versus 256 ordinary positions does **not** imply a 16/256 weight. Log gradient norms/cosines on the shared trunk and score head, not just scalar losses, before interpreting whether A dominates. Four hundred steps sample up to 6,400 root instances per iteration; their unique count is much smaller.

The family hash agrees between normal ordinary and paired splitting, since A excludes restart roots. F6 is the exception. Validation roots hold out source families, but repeated use for model/component selection turns the fixed benchmark into development data. Keep a final untouched test bank. Split or bootstrap by family/game, not by individual candidate pairs or roots from the same game.

**Q9. Are forced deals preserved across replay and resume?**

The principal wiring is correct. Both constructors consume the natural plan draws before substitution and validate stack membership. `engine_plan_ids` selects the restart source's forced plans; Python/Rust constructors share that value; JSON preserves it; the Rust capture receives `rust_plan_ids`; Rust `replay_history` reconstructs with `new_with_plans`. Curriculum candidates retain the forced source deal. Resume checks both the restart description and the current forced-deal assignment, and generation records a deal digest.

I found no new forced-deal replay divergence. The supplied large equivalence results and the focused tests are consistent with that reading. Natural twins share their initial random setup; changed play can still change reshuffle decisions and later observations, so this is not a promise of identical future visible cards under different actions.

One API boundary remains worth making explicit: `new_python_state`/`new_rust_state` construct the initial state, not a full restart-aware replay iterator. Consumers must apply the restart redeterminization at `restart.at`. Core training replay does; diagnostic loops such as `hygiene_rescue.game_metrics`, `pool_rescue.plan_rows`, and `sibling_probe.select_roots` do not. Their current natural-source callers avoid that case. Reject restart inputs there or use a common replay iterator before generalizing those tools to archives.

**Q10. Are the pool rule, resolved-state check and target planes correct?**

Fix F3. Otherwise the major safeguard is appropriate: `_resolved` takes an available pool/park action before judging feasibility, avoiding the false conclusion that a freshly written pool box was destroyed when its pool prompt has not yet been answered. `_kills` reads the original acting seat even if the copied state advances. Pool/park construction adds free track points, so overriding a pass at those prompts is coherent with this helper's rule.

Python and Rust `pool_target_boxes` implement the same candidate-street and box predicates. They exclude dead plans, filled boxes, complete pool tracks, and complete-street alternatives with no possible roundabout. The main Python/Rust equivalence gate is strong evidence for parity; the helper mismatch is a separate consumer bug.

Do not overread these planes as exact attainability. They inherit a deliberately conservative feasibility test, including roundabout/BIS reachability, rather than solving the future sequence of pool-effect draws and joint plan claims. A remaining target is an opportunity worth investigating, not a proof that the plan can finish. The crude helper also lacks the range/roundabout planning the request itself identifies; failure of this rule is not evidence that pool plans are unlearnable.

**Q11. Are shared-plan slices and padded seats correct?**

Yes. Measured zero-based half-open slices are:

| Slot | Global identity/characteristics | Per-seat progress |
|---|---|---|
| 0 | `[157:214]` | `[43:77]` |
| 1 | `[214:271]` | `[77:111]` |
| 2 | `[271:328]` | `[111:145]` |

Each embedding consumes `57 + 4*34 = 193` inputs through the shared 128→64 MLP. The three 64-vectors join the trunk. Actual padded progress blocks are zero; real-seat masks still govern target losses. The MLP is allowed to produce a nonzero embedding from a partially padded input: the slot is real even when some seats are absent. Global seat validity remains available to the trunk.

Passing validity directly into the plan MLP could be a future clarity improvement, but I found no indexing or masking defect requiring it now. The default network has **4,403,598 parameters**, consistent with the request. Removing the absolute viewer-seat one-hot addresses the identified distribution mismatch.

**Q12. Do the earlier review fixes do what they claim?**

The estate-aware normalized-rank fix is correct: it now uses the same rank distribution as training, including the engine's tiebreak. The gate condition was accurately renamed `rank_regression_not_established`; it still checks the upper confidence bound. It remains a permissive regression screen, not evidence of non-inferiority. That is a declared design choice, not an incorrectly implemented rename.

The family split is correct on normal-sized corpora, with F6's fallback exception. Re-splitting historical v3 replay cannot make a resumed checkpoint forget its earlier training exposure. The new from-scratch run avoids that historical holdout problem.

Brier/log-loss skill is the right improvement over majority-class accuracy. There is a small reporting defect at [s2_train.py:420](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_train.py:420): clipping an exactly zero/one empirical rate makes the denominator tiny instead of recognizing that skill relative to a perfect constant baseline is undefined. Report null/undefined skill there alongside raw loss and support counts. This matters particularly for the all-plans-ending target while there are no positive examples. Using the evaluated prevalence is acceptable as a descriptive baseline; label it that way rather than an independently fitted forecast.

## Throughput opportunities, in priority order

**1. Measure and optimize the helper's Python work before increasing its budget.** Thirty live-pool WRITE positions from the existing sibling dataset took median **10.1 ms**, mean **20.3 ms**, and maximum **193.5 ms** per `pool_choice` in this audit. Those are a small, non-random benchmark, not a whole-run forecast. The helper copies/resolves every legal WRITE, repeatedly recomputes feasibility, and copies states again for capacity tie-breaking. State conversion is only part of the cost.

Reuse the already resolved acting sheet for capacity calculations, cache state-wide facts during a choice, and avoid conversion at phases that cannot be changed. A Rust implementation of the same rule can then remove repeated cross-language state construction. Preserve tie-breaking and the resolved-prompt semantics with equivalence tests; early exits must not skip a preferred safe progress action.

A separate 36-terminal pilot, with identical roots/futures and a random v4 network, took **0.51 s unsteered versus 0.99 s steered**; **0.59 s** was inside the helper for 1,051 calls and 204 overrides. This is not a scalable 1.9× forecast: most of those calls were cheap, and a stronger policy reaches different, longer games. Time a representative mixed batch and log helper calls, WRITE candidates, overrides and CPU time by phase before committing to an overnight ETA.

**2. Batch the policy-only rollout boundary and avoid unnecessary probability work.** `PackedNetEvaluator.policy_states` still loops through Python to obtain encodings/legal moves, packages them, and reconstructs a dense 684-vector with softmax for every live state. `_finish_all` only needs a legal argmax. A specialized path can select the best compact legal logit with the same lowest-macro tie rule, return one action per state, and advance states in Rust. The existing policy inference path already skips value-head evaluation; retain that optimization. Batch terminal scores/ranks in Rust too, avoiding a full Python snapshot for every terminal.

**3. Vectorize paired training bookkeeping.** Precompute score/rank means once when loading the window. Pack fixed-three-candidate roots, mask the occasional smaller set, and compute the centered-error form in one tensor operation. Skip the rank calculation when its weight is zero. Avoid three `.item()`-equivalent GPU synchronizations inside each `paired_loss` call and keep logging reductions on device until the logging boundary. Add a value-only training path if profiling shows the unused policy/auxiliary forward work matters. These are mathematical-preserving optimizations to check against original losses and gradients.

**4. Batch diagnostic inference, but do not overstate its importance.** On this RTX 3070 Laptop GPU, 64 roots/307 candidates took **0.36 s** in the current two-pass decision-metrics routine; one batched forward plus output copy took **0.0067 s**, excluding final NumPy statistics. Compute both blend and margin from the same outputs. This is a clear local opportunity, but generation dominates run time, so it will not save hours by itself.

**5. Allocate futures where they resolve useful uncertainty.** Fixed 48-future labels are a reasonable baseline. A later 8→16→32→48 allocation can spend more on unresolved, high-regret comparisons and less on obvious ones. [Rollout Sampling Approximate Policy Iteration](https://arxiv.org/abs/0805.2027) motivates adaptive rollout allocation. Retain fixed independent evaluation futures and account for selection/stopping effects; adaptively stopping an empirical label can bias its mean. Preserve candidate coupling and measure the actual variance reduction from shared futures.

The recent v3 costs are a useful budget warning: iterations 46–55 averaged **17.49 minutes generation**, **6.05 minutes A**, and **0.99 minutes training**, with **29.85 minutes per gate**. Twenty generations/trainings, eighteen A collections and four gates already total about **9.96 hours**, before the new helper cost and unrecorded overhead. Early from-scratch games may be cheaper, but 20 iterations in ten hours is currently a target, not a validated ETA.

## Interpreting the evidence and making the cloud decision

The claim that one game outcome contains exactly “1/50 of the signal” is stronger than the probe supports. Averaging independent futures reduces noise variance approximately as `1/F`; it does not establish a fixed exchange rate between game trajectories and paired supervision. As an illustration, under a simple independent, equal-noise model, split-half reliability is `r(F)=signal_variance/(signal_variance+noise_variance/F)`. The reported correlations 0.42 at 12 and 0.75 at 48 are both compatible with noise variance about 16 times signal variance. The different root sets and heterogeneous decisions prevent treating that illustration as an exact estimate. The practical conclusion is that 48 futures made these particular comparisons substantially more reliable.

The decision statistics also need game-level uncertainty. Candidate pairs share afterstates/futures and roots share source games; the current pair-level standard errors treat them as independent. Bootstrap paired changes by game family. The empirical maximum used in regret is upward biased by outcome noise; compare model-to-played and model-to-model differences on the same evaluation futures, and regard the fit-selected “oracle” as a finite-sample comparator, not a true ceiling. A model that generalizes across roots can outperform a noisy fit-half chooser.

For this toy, predeclare a small set of tests:

- Natural-deal pool-plan completion, third-plan completion and normalized rank/realized margin, with denominators and intervals by player count and plan ID. A strategy metric is useful only alongside competitive outcomes.
- Actual zero helper-data sampling after the chosen cutoff, followed by stable or improving unassisted play over several checkpoints.
- Shrinking execution gap on the fixed steering audit and improvement on fresh, unsteered sibling roots. Keep the old 143-root bank as a historical diagnostic.
- A frozen opponent/deal evaluation set for comparisons between runs. Generation-versus-evolving-league curves alone are insufficient.
- Wall time and unique data counts for each component, including gate cost and actual steered-root counts.

If the package passes, the next cloud-relevant work is attribution of the expensive components, stronger searching opposition, broader decision coverage, and more than one training seed. If it fails, determine whether the first missing link is discovery, value calibration, policy execution, or insufficient clean training time. A single short combined run cannot establish that the architecture or learning approach has reached its ceiling.

## Verification and artifacts

I inspected the new code, relevant previous-review fixes, replay/engine/network consumers, saved paired datasets and run metrics, and the primary research linked above. I did not rerun the already established 60,000-encoding gate or start the overnight run. **33 focused existing tests passed** in 28.88 seconds across paired targets, helper scheduling, encoder v4, forced deals, pool rescue, training and promotion.

New audit outputs are [evidence.json](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/_toy_review_20261003/evidence.json) and [rollout_timing.json](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/_toy_review_20261003/rollout_timing.json). The reproducible audit programs are saved beside them. They reproduce the four-player exclusion, replay-buffer timeline, roundabout/needed-street disagreement and fallback family leakage, and record the local timing measurements. No production code or existing experiment artifact was changed by this review.
