# Welcome To: hygiene, City Plans, and the path to a world-class player

Review date: **2026-10-02**. Reviewed working tree: `a4498a5`; `git diff 65632c3 HEAD -- games/welcome_to` was empty. This answers [the review request](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/HYGIENE_AND_PLANS_REVIEW_REQUEST.md). The recommendations below are proposed experiments, not measured strength gains.

**Build A next, with terminal continuations and an explicit paired loss on the existing score/rank heads. Run B as a small, separately measured exploration experiment. Pilot C alongside, but do not assume that opponent search will reproduce the hygiene rescue.** Keep the current encoder for the first experiment. Before a large training run, tighten the validation split and test the absolute-seat input described below.

The central difficulty is learning *small action-dependent differences in long-term outcomes*. The evidence strongly supports that diagnosis. It does not yet distinguish insufficient contrast in the data from compression or optimization that discards contrast. A controlled supervised experiment can distinguish them much more cheaply than another long self-play run.

## 1. Direct answers to the six questions

| Question | Recommendation |
|---|---|
| 1. Is sibling value contrast the binding problem? | **Best-supported working diagnosis, not an exclusive explanation.** Run terminal rollout comparisons on ordinary, unfiltered placements; test whether the current architecture learns those differences. The 800-simulation result argues against simply increasing the present search budget, not against better search allocation or terminal rollouts. |
| 2. Change the encoder? | **No new ABI for A's first test.** Use the existing encoder on legal afterstates. If paired training fails, test a small shared action/afterstate scorer and exact local deltas before a larger trunk. Audit the absolute-seat one-hot now. |
| 3. Mid-game stochasticity? | **Yes, modest conditional placement exploration**, with printed slot and temp delta fixed in the cleanest ablation. Keep a smaller amount of card/effect exploration because plans require it. Root noise already makes training stochastic after turn 10, so T=0 is not literally zero exploration. |
| 4. Opponents search? | **Eventually yes for a substantial part of training and evaluation.** First measure 32–64 simulations against policy-only opponents at equal compute. Current search has not fixed hygiene, so this is an experiment, not an established cure. Avoid heuristic-assisted opponents as the default training route. |
| 5. Deferred City Plan value? | **Terminal counterfactual outcomes plus an adaptive, plan-stratified backward curriculum.** Add live-plan completion-time probabilities as a supporting target. Discovery of coordinated plan strategies is a separate problem that one changed placement may not solve. |
| 6. Ranking and missing ideas? | **A > B > revised H > conditional C > conditional E > D > F > G > I; retire J.** Ahead of all expensive runs: independent evaluation, representation probes, and a precise contract for the rollout targets. Missing items include direct action-gap supervision, coherent plan options, outcome-dependent stopping-time evaluation, and stronger opposition in the benchmark. |

## 2. What the repository adds to the diagnosis

### 2.1 The present training objective does not explicitly teach action differences

In [network.py](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/network.py:352), score and regression targets use per-row squared error; rank uses cross-entropy. There is no sibling-pair loss. Auxiliary losses share representations with value, but shared representations do not guarantee that the value readout uses every predictive feature. The comments claiming that score “already uses” informative auxiliary predictions describe an aspiration, not a mathematical property.

The actual model has **4,167,502 parameters**. Each sheet is compressed from **986 floats to 256 to 128** before the contextual trunk. That is a reasonable architecture, but a plausible place to lose a weak, local action distinction. Layer normalization and reasonable input scales make numerical underflow an implausible explanation; they do not guarantee retention of small predictive differences.

A one-point score difference is only `1/80 = 0.0125` in target units. Its squared magnitude is `0.00015625`, before masking and group averaging. This is not proof of a bad loss coefficient: gradients and correlations matter. It does explain why better overall score R² can coexist with poor one-point decisions. Measure gradients and decision error on sibling pairs, not just aggregate losses.

### 2.2 A specific correction to the “plan-death heads stayed at base rate” interpretation

I accept the gameplay measurements in §6. I challenge the inference from majority-class accuracy to “no predictive signal.” At iteration 8, the saved metrics for `plan_1_dies_soon` are:

| Quantity | Saved predictor | Constant predictor using the observed positive rate |
|---|---:|---:|
| Positive rate | 7.36% | 7.36% |
| Classification accuracy | 92.56% | 92.64% |
| Binary cross-entropy | **0.1662** | 0.2629 |
| Brier score | **0.05176** | 0.06821 |

That is **24.1% Brier skill** relative to the constant-rate predictor despite effectively base-rate accuracy. At iteration 35 the three plan-death heads have Brier skill of **+18.0%, +28.4%, and −12.5%**. The third remains problematic; the first two are not simply constant-rate predictors on this evaluation set.

These calculations use the saved [iteration-8 metrics](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/v3_curriculum_01/candidate_iter_0008.pt.metrics.json) and [iteration-35 metrics](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/v3_curriculum_01/candidate_iter_0035.pt.metrics.json), with `Brier skill = 1 − Brier/[p(1−p)]`. They do not establish that the predictions are useful for sibling action choice. Report log loss, Brier skill, precision–recall discrimination, and calibration on live plans, separated by turn, plan type, seat role, and ordinary/curriculum games. Compare against a turn/plan-type baseline as well as a constant baseline.

### 2.3 Curriculum families cross the validation boundary

[s2_train.split_trajectories](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_train.py:102) hashes `(iteration, game.seed)`. A restart has a new seed but shares a source trajectory and a reached board with its parent. The split does not group by `restart.source_seed`.

I inspected the iteration-35 replay window, iterations 24–35, resolving source lineage through iteration 23:

* 1,200 restart games.
* **211** have the opposite split assignment from their source.
* **88** training restarts come from validation sources; **123** validation restarts come from training sources.
* **186** source families present in that replay window have members in both splits.

This is dependence between the splits, not proof of identical target leakage: the continuations use fresh draws. The counts describe the full validation pool; the bounded 256-game evaluation contains a subset. It weakens claims about generalization from the held-out losses, particularly near completion. It does **not** invalidate independent paired gameplay gates or the rescue experiment.

For new experiments, assign the original game and all descendants to one durable split, using a source-family ID that includes run identity. Exclude validation families from training-curriculum construction. Group counterfactual siblings and their continuations the same way. Existing checkpoints have already seen the old data, so use fresh source families for the new benchmark rather than claiming a retroactive clean holdout.

### 2.4 Absolute seat identity creates an avoidable distribution shift

The sheets are viewer-relative, but [encoder.py](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/encoder.py:1385) also emits an **absolute viewer-seat one-hot**. Rust does the same. S2 records searched learner roots from seat 0; policy inference for actual and simulated opponents uses seats 1–3. Shared sheet weights do not remove this global difference.

A small sensitivity probe on **1,024 iteration-35 stored rows** changed only that one-hot from 0 to 1, holding legality and all other inputs fixed. Policy argmax changed on **31 rows (3.0%)**, with mean total-variation distance **0.0149**. This is an input-sensitivity measurement, not a measured loss of opponent strength, and the mutated rows are not a full seat-permutation experiment.

Test a properly canonicalized seat encoding or train on legitimately generated actor viewpoints. Preserve all information-set and tie/reshuffle semantics when doing so. Do not silently rewrite an input to an existing checkpoint and declare parity. This is an encoder item worth addressing before speculative new hygiene planes.

### 2.5 Search models different opponents from the ones actually playing

Actual opponents use league checkpoints with the turn-dependent temperature schedule. In ordinary generation, simulated opponents default to the learner network, and [Rust search](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/welcome_to_rust/src/search.rs:1538) samples their policy probabilities without the real-game late argmax schedule. The gate supplies the incumbent as the simulated policy, but simulated action sampling still differs from actual T=0 play.

This is an approximation, not necessarily a bug. It matters here because opponents determine the game horizon. Adding real opponent search while leaving the transition model unchanged increases that mismatch. Measure opponent refusal timing, plan timing, and action agreement under the model versus real play. Cheap searched-opponent policy distillation is a reasonable eventual solution; recursively nesting MCTS for every simulated opponent is unnecessary.

### 2.6 Smaller evaluation and documentation issues

The gate's [normalized_rank](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_promotion.py:130) treats equal scores as tied. The engine and [training rank targets](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/training.py:163) also use the estate tiebreak. Align these definitions. Also, `secondary_not_regressed` passes when the **upper** confidence bound is above `−tolerance`: that means “regression was not established,” not “non-inferiority was established.” A non-inferiority requirement uses the lower bound. Neither issue explains the hygiene plateau, but both matter for a serious strength claim.

The rescue helper preserves the printed slot but may change **both temp delta and box**, as [placement_assist.py](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/placement_assist.py:20) explicitly says. A box-only experiment must fix delta too. Its 95% intervention rate is strong evidence of divergence from this capacity/span objective, not evidence that 95% of moves are strategically wrong or that the clean-state support is literally zero.

The rulebook permits a roundabout **in addition to** numbering and an effect, including on a refusal turn. The request's “instead of a number” wording is misleading. Likewise, a zero numeric span does not prove a box can never be filled: bis and roundabouts matter. The engine/spec already account for these distinctions; preserve them in any new features. [Publisher rulebook, Advanced Mode](https://bluecocker.com/wp-content/uploads/2023/08/Rulebook-EN.pdf).

## 3. The decisive experiment: can the current model learn the contrast?

Freeze iteration 35. Collect an independent set of ordinary roots spanning early, middle, and late turns and all plan families. Do not select the whole set using the cleanup heuristic or value disagreement. Include a separately reported diagnostic slice of those positions if useful.

For each root, evaluate the incumbent placement and legal alternatives under multiple independently sampled futures, with paired randomness between alternatives. Use actual terminal scores and ranks. Split by source game before training, retaining all siblings and future tapes on the same side.

Use one dataset to compare:

1. The current checkpoint, with no fitting.
2. Frozen sheet/trunk features with trained value readouts.
3. The current full network, fine-tuned with absolute outcomes and paired differences.
4. Only if needed, a small shared candidate scorer using existing inputs plus exact action deltas.

First verify that a small training subset can be fit. Then evaluate on fresh source games and fresh continuation seeds. If readouts suffice, the features were already present. If full-network fitting works but frozen readouts do not, representation adaptation was needed, not necessarily a new architecture. If a delta scorer generalizes materially better on the same labels, E earns priority. If high-sample rollout rankings themselves are unstable, fix the evaluation budget or continuation policy before diagnosing the network.

Measure paired score/rank prediction error, sign accuracy with uncertainty bands, ranking correlation, and **decision regret**: the independent-rollout value lost by selecting the model's favorite among the sampled candidates. Include the incumbent and a shuffled-label/constant-difference control. Do not count every correlated pair from one root as an independent example or require “clean” to win every comparison.

This emphasis on decision-relevant error is consistent with research on action gaps: an evaluator's overall accuracy and its usefulness for selecting actions need not coincide. [Farahmand, *Action-Gap Phenomenon in Reinforcement Learning*](https://proceedings.neurips.cc/paper/2011/hash/013d407166ec4fa56eb1e1f8cbe183b9-Abstract.html).

## 4. A: the version I would build

### 4.1 Generate terminal comparisons, not short-horizon proxies

At a sampled learner WRITE root:

1. Record the information state, legal actions, selected slot/delta, turn, plan IDs, and frozen continuation-policy identities.
2. Include the incumbent action and a small stratified set of legal boxes: policy-supported alternatives plus uniform alternatives. No capacity ranking or GreedyBot labels. Start with **4–8 alternatives**, not exhaustive branching everywhere.
3. Apply each macro to a clone. Record the learner-view afterstate, including its decision phase. Finish any pending effect/plan/roundabout decisions legally under the continuation policy.
4. Continue to the **actual multiplayer terminal state**. Use the frozen learned policy initially, so the first comparison has no heuristic targets and no network-value cutoff.
5. Record all seats' scores, rank distribution, score components, plans and completion turns, refusals, terminal turn and triggers. Retain the per-replication vector, not just the winning action.

Use a fixed small pilot budget, such as 8–16 futures per candidate, then measure ranking stability on a subset with 32–64 fresh futures. These are starting budgets, not claims of adequate statistical power. Record evaluator rows and wall time before increasing the sample count. A few hundred roots can answer feasibility; it is not enough by itself to train a world-class player.

The estimand is `Q^π(s,a)` for a specified continuation policy and opponent mixture. A terminal rollout is unbiased for that policy's outcome expectation if its simulation is correct. It is not an oracle for optimal play. A weak continuation can fail to use a placement's potential. Refresh the rollout policy as the learner improves and verify a subset with modest continuation search or stronger learned policies.

This is a practical approximate-policy-iteration approach: rollout outcomes create policy/value supervision. The literature also supports allocating additional samples to unresolved action comparisons rather than equally sampling everything forever. [Dimitrakakis and Lagoudakis, *Rollout Sampling Approximate Policy Iteration*](https://arxiv.org/abs/0805.2027).

### 4.2 Pair the randomness correctly

For outcome `Y`, estimate the action difference as:

`Δ̂(a,b) = mean_m [Y(a, ω_m) − Y(b, ω_m)]`.

The variance is `(Var(Y_a) + Var(Y_b) − 2 Cov(Y_a,Y_b))/M`. Common draws help when the covariance is positive; do not assume they always do. A recent paper directly studies this issue and gives a counterexample to unconditional variance-reduction claims. [Yadav et al., *Using Common Random Numbers for Simulation-based Planning with Rollouts*, 2026](https://arxiv.org/html/2605.04732v1).

Use independent streams for deck randomness, policy sampling, and any search. Within a replication, couple environmental draws consistently between siblings. A single shared seed is insufficient if different numbers of effect decisions consume different portions of a common stream. Plan-triggered reshuffles require special care: couple the random permutations while allowing each branch's legal deck contents and timing to differ. Do not force identical revealed cards after the branches' deck processes legitimately diverge.

Redeterminize from public information before generating a future, and never let the policy read its hidden order. Simulated opponents must respond to their own visible observations; do not freeze their future action lists. A placement can change plan races and the terminal turn, so independent sheet-only rollouts are insufficient.

For an adaptive sampling implementation, use pilot draws to allocate budget and independent confirmation draws for winner assessment. Naively selecting and labeling the maximum noisy estimate creates winner's-curse bias; ordinary fixed-sample intervals are also not automatically valid after arbitrary repeated stopping decisions.

### 4.3 Teach the heads search actually reads

For seat `j`, let `sθ,j(x)` be the current normalized score output. Add:

`L_pair_score = mean_pairs,j [(sθ,j(x_a) − sθ,j(x_b)) − mean_m((S_a,m,j − S_b,m,j)/80)]²`.

Anchor each afterstate with its average absolute per-seat score target and its empirical terminal rank distribution, using the existing score MSE and rank cross-entropy. Add a paired rank-utility loss if useful. Keep ordinary replay mixed in and give the pair loss an explicit group weight. Normalize per root or a fixed number of pairs, so roots with many legal actions do not dominate quadratically. Fit pairs together in a minibatch.

This changes the objective supervision, not just an auxiliary head that the deployed value may ignore. Pairing cancels much of the common game difficulty and draw luck while the absolute targets prevent arbitrary offsets. Tune the new weight using held-out action regret and gameplay, not raw loss magnitude.

**Do not attach different action-conditioned outcomes to identical pre-action state-value inputs.** Attach them to afterstates, or create an explicit action-conditioned Q head. If an afterstate has a forced next action and is never normally a trained/search leaf, keep the new value-only sample valid and mask policy loss; also validate improvement at the actual leaf phases search reaches. If you advance through a stochastic transition before encoding, you have a sampled successor, not a single deterministic afterstate; average accordingly.

The shard format currently ties value rows to searched roots. Add a separate grouped counterfactual format with value/policy masks, lineage, rollout-policy version, and replication identities. Do not fake search visits for value-only rows. Both Python and Rust readers/targets need parity checks.

### 4.4 Resolve the nonlinear-value contract explicitly

The current `blend_value` is a nonlinear function of predicted rank probabilities and expected per-seat scores. In general:

`blend(E[rank distribution], E[scores]) ≠ E[terminal blend]`.

This remains true with perfect prediction: confidence depends on rank variance, `tanh` is nonlinear, and for 3–4 seats `max(E[opponent scores])` differs from `E[max(opponent scores)]`.

For the first A experiment, train the existing heads toward their own well-defined targets and paired differences, leaving the production blend unchanged. Report true rollout rank utility and margin as well as the resulting production-blend preferences. Do not simultaneously insist that the same blended scalar exactly equal the mean terminal blend.

If this mismatch limits decisions after the heads improve, test a direct expected-return head or an outcome/margin distribution as a **separate objective ablation**. It predicts actual game outcomes and therefore meets the outcome-only rule; it is not an auxiliary hygiene bonus. The prior confidence-power test makes changing that gate alone a low-priority hygiene fix.

### 4.5 Policy improvement should follow once labels are reliable

Value-only A is the cleanest diagnosis. For strength, then distill reliable rollout preferences into the policy too. With only same-slot/same-delta boxes evaluated, supervise the **conditional distribution over that evaluated set**. Do not zero every unevaluated macro and pretend all cards were compared. Preserve ordinary search targets for the remaining decisions or generate genuinely broader comparisons.

Use a conservative advantage-weighted update or soft preference targets, and keep unresolved comparisons weak. A hard one-hot winner from four noisy rollouts is a poor target. This learned-policy/stronger-planner loop is closely related to Expert Iteration; the expert can be your own outcome-based planner. [Anthony, Tian and Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*](https://arxiv.org/abs/1705.08439).

## 5. B, root noise, and search budget

**A before B alone:** the existing measurements already find a small outcome difference drowned in trajectory variation. A deliberately estimates that difference. B supplies broader coverage, but a new random placement followed by one terminal result still has a noisy label. Use B to discover states that A and subsequent self-play can exploit.

For a clean B pilot, retain the current card/delta selection and mix conditional visit sampling with a small uniform probability over its legal boxes. An initial epsilon sweep of **0.05 and 0.10** on turns roughly **8–20** is adequate to test the mechanism. Those settings are hypotheses. Track actual action-change rate, conditional entropy, zero-visit action coverage, plan destruction, and score. Allow exceptions for singleton decisions. Keep evaluation deterministic.

Temperature over visits alone never selects an action with zero visits. Uniform support or forced root evaluation is necessary if coverage is the issue. Preserve the action actually executed in the trajectory and label its reached states correctly; exploratory choices need not become one-hot policy targets. Broad placement randomness cannot discover a plan that requires selecting different effects and fences, so retain some card/effect and sequence-level exploration as well.

For root Dirichlet noise, use the number of **legal** actions `L`, not 684. With concentration 10, each legal action has expected noise mass `1/L`. If card group `c` has `L_c` legal macros, its expected group mass is `L_c/L`: cards with more placements get more total noise. Conditional within-group concentration is `10 L_c/L`. Thus the noise does explore placements, but not in a card-balanced way. It does not guarantee that every placement receives enough search, much less gets played.

A hierarchical experiment can choose noise over slots, then delta/box within slot, and preserve the flat macro vocabulary at the engine boundary. This is exploration structure, not a hand-coded valuation. Log its effects before adopting it.

At 200 simulations, inspect by phase and turn: legal count, fraction visited, visits per alternative within the selected card, prior/visit KL at the actual search root, realized card-versus-box changes, root Q gaps, and leaf depth measured in **completed game turns**. Training-set `policy_kl` is not the same thing as contemporaneous search improvement over its prior. The known −11.6 policy-only result already says search is useful overall; it does not establish useful placement supervision.

Gumbel root selection is worth testing after there is reliable action-value contrast. Its policy-improvement argument depends on sufficiently accurate action values; sequential halving cannot repair a systematically indifferent evaluator. Use the completed-Q policy-improvement target as well as the selection algorithm, not just Gumbel noise plus raw visits. [Danihelka et al., *Policy Improvement by Planning with Gumbel*](https://openreview.net/pdf?id=bERaNdoegnO); [DeepMind's implementation and stated value-accuracy condition](https://github.com/google-deepmind/mctx).

KataGo's forced playouts/target pruning and playout-cap randomization are relevant ways to separate exploration, policy-target quality, and value-data volume. They are efficiency tools, not evidence that a particular cap will solve Welcome To's plateau. [Wu, *Accelerating Self-Play Learning in Go*](https://arxiv.org/pdf/1902.10565).

## 6. C: stronger opponents and the game horizon

The all-seat rescue establishes that opponents' sheets constrain achievable plan outcomes. It does not establish that a little search produces clean opponents: even 800 simulations left hygiene almost unchanged. Small-search opponents may improve plans or scores while still ending the game early.

Pilot **policy-only versus 32/64-simulation opponents**, first with the same frozen checkpoint and then with an A-improved policy if available. Measure survival to turns 20/24/28, first/third refusal timing, plans, margin, and compute. Compare equal wall time or evaluator rows, not only equal games. Start with a minority mixture of searched tables if the pilot is promising, retaining historical-policy opponents for robustness.

For research attribution, run A-only and C-only controls before treating A+C as one improvement. If cheap search does not extend useful play, postpone C until the policy is better. A current-policy value function can legitimately assign little value to a plan that its opponents will always terminate before completion; more accurate prediction alone cannot manufacture a longer game.

Eventually train and evaluate tables where each seat has its own root-player search, and learn from those searched decisions. Do not back up a single scalar while switching viewpoints. Distill searched decisions into the cheap policy and preserve a mixed league. This reduces both the short-game bottleneck and the asymmetry between policy-only opponents and the searched learner.

Heuristic-assisted opponents are not my recommendation under this project's rule. Even without an imitation loss, those opponents alter the return-generating process and reintroduce the hand-designed scaffold. Learned policies/search give a cleaner experiment.

## 7. Deferred City Plan value needs discovery and credit assignment

### 7.1 Why the existing curriculum can stall

[curriculum.candidates](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/curriculum.py:100) uses only ordinary games in which seat 0 completed a plan; restart games cannot seed further restarts. [The driver](C:/Users/joeld/projects/boardgame-ai/games/welcome_to/s2_run.py:102) supplies only the previous iteration. The 1/2/4/8 distances are sampled rather than adapted to competence.

Consequences: absent plan families stay absent; rare successes can disappear from the next pool; the curriculum has no mechanism to propagate a successful restarted strategy progressively back to the opening. A 50% re-completion rate can mean useful training at the frontier, but it does not demonstrate transfer to ordinary starts. Also, 20% of *games* is not 20% of *training rows*: short restarts contain fewer decisions.

Replace the transient pool with a bounded persistent archive stratified by plan identity, turn, seat count, and source family. Track success by distance and plan type. Move starts earlier when the current frontier becomes reliably solvable, retain failures as well as successes, and preserve a majority of ordinary-start training. Include reachable states before early irreversible pool mistakes. Fresh continuations remain ordinary multiplayer games, with the true deck and stopping rules.

Reverse-curriculum research supports moving starting states outward as competence grows; it does not prove a particular restart schedule works here. [Florensa et al., *Reverse Curriculum Generation for Reinforcement Learning*](https://arxiv.org/abs/1707.05300).

### 7.2 Targets I would add, and their limits

For a live unclaimed plan, predict completion **before actual game end** in bins such as 1–2, 3–4, 5–8, 9+ turns, and never. An event-time distribution yields consistent cumulative probabilities for “within k” and retains late positives; a single short horizon turns many eventual successes into negatives. Already completed/dead states should have explicitly known status or appropriate masks, rather than making easy settled cases dominate evaluation.

The terminal game boundary is an observed failure to complete in that game, not missing follow-up to be censored away. For richer analysis, jointly track game-end timing and plan-completion timing; do not assume independent hazards. Existing `turns_to_plan` is conditional on eventual completion because failures are masked, so it cannot alone represent attainability.

Per-box future occupancy/pool/estate membership is another useful outcome target: it localizes the consequences that the current aggregate refusal target can obscure. Never treat predicted unfilled boxes or plan hazard as a manually priced leaf penalty. Longer refusal windows can help, but compare expected counts together with the remaining game horizon: raw refusal counts can rise when better play extends the game, as the rescue already shows.

Oversample informative *starting states* with fresh outcome sampling when possible. If oversampling positive outcomes or using class-weighted BCE, account for the changed class prior before interpreting outputs as probabilities. A head trained only on near-success boards needs calibration on ordinary games. A rare-event accuracy threshold of 0.5 is usually not the relevant diagnostic.

### 7.3 When one-step improvement cannot discover a plan

Suppose a pool-preserving placement pays only if several later card/effect choices also change. A rollout following the old policy may correctly report no plan gain. That is a limitation of the continuation policy, not necessarily label noise or a broken model.

After hygiene contrast works, broaden A to card choice, surveyor fences, plan claims, and roundabouts, and consider **learned plan-conditioned continuation policies/options**. Train them on actual completion outcomes, not progress bonuses. Compare their achieved *full-game rank and score* under common futures before distilling them into the main policy. Use fresh futures for verification so an option cannot win merely by fitting the sampled hidden deck.

Hindsight goal relabeling is relevant inspiration for learning what outcomes a trajectory achieved, but naively replacing the game's dealt plans is invalid: claims consume estates, alter race payouts, can trigger reshuffling, and can end the game. Use hindsight for separate, carefully defined achievement targets/options, not fabricated main-game returns. [Andrychowicz et al., *Hindsight Experience Replay*](https://arxiv.org/abs/1707.01495).

### 7.4 TD, reanalysis, and RUDDER

Do not replace terminal supervision wholesale with today's search value. Bootstrapping a blind evaluator can stabilize the same blindness. Once A improves the evaluator, test a lagged-network mixture of terminal Monte Carlo and multi-step/TD(λ) targets, with no double counting of already scored points. Use game turns, or explicit variable-duration transitions, rather than accidentally giving a long effect sequence a different time discount. The real objective has no reason to discount delayed points simply because they are delayed.

KataGo's documented short-horizon value targets average **future MCTS outcome estimates**; they are auxiliary heads, not “score over the next six turns” or a reason to replace final-game value. Its own documentation characterizes the supporting weighting experiment cautiously. [KataGo methods, Short-term Value and Score Targets](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md?plain=1).

RUDDER is relevant because it learns return redistribution for delayed rewards, but it still needs informative trajectories and adds another learning system. It is a lower-priority alternative to paired rollouts when an exact fast simulator is already available. Its results are not a guarantee for this multiplayer stopping-time problem. [Arjona-Medina et al., *RUDDER*](https://arxiv.org/abs/1806.07857).

## 8. Encoder changes worth testing if the supervised probe calls for them

Draw the boundary at **transition facts versus strategic preference**. Exact consequences of a legal action are acceptable inputs: left/right bounds, gap lengths, written number/delta, number-only capacity change, newly zero-span boxes, pool eligibility lost, and a sound detected plan death. “This costs 0.3 points,” a capacity-maximizing action label, or a heuristic rollout value crosses the project's boundary.

Expected future refusals are not an intrinsic property of a sheet plus deck: they depend on future actions, opponent behavior, and when the game ends. Predict them from actual rollouts under a specified policy. A static one-draw refusal probability has different semantics and must not be named as the former.

Useful later representations are a shared scorer of `(root context, candidate afterstate)` or a per-box/per-gap network with directional neighbors, explicit street/position identity, pool locations, fences, and plan requirements. Shared local operators do not require a perfectly translation-invariant game; ascending order rules are directional and local, while boundaries and street differences can be explicit. Avoid a naive 2-D grid convolution that treats vertical neighbors as numerically adjacent or assumes reflected boards are equivalent.

Try the smallest change that the probe justifies: a learned skip from existing hygiene/plan scalars into the value readout, a larger sheet embedding, or action-delta features. A new afterstate *training arrangement* does not itself require an encoder ABI change. New features do; follow the repository's explicit checkpoint/schema compatibility contract rather than silently loading old weights against a changed layout.

Another small ablation is residual score prediction: predict `final score − current exact score` and reconstruct the objective score by adding the known current score. This is an algebraic decomposition of the same outcome, not a progress bonus. Current scoring can decrease after later choices, so allow negative residuals. It may prevent learning already-known points from dominating the task, but the current score is already an input and the benefit must be measured. Do not combine this change with A's first diagnosis.

## 9. Ranked build sequence and stopping criteria

| Priority | Idea | Decision and evidence needed |
|---|---|---|
| 0 | Evaluation and data contracts | Fresh family-held-out sibling benchmark; fix future split lineage and tie semantics; quantify seat-input and opponent-model effects. Log phase/turn/plan strata. |
| 1 | **A** | Terminal paired outcomes, absolute anchors, explicit paired score/rank supervision. Continue only if held-out decision regret improves and the improvement reaches the deployed policy/search. |
| 2 | **B** | Cheap conditional placement exploration with nonzero support. Compare A, B, and A+B at equal compute. Keep it if it improves useful coverage and subsequent strength. |
| 3 | **H, revised** | Persistent plan-stratified adaptive curriculum plus event-time targets. Success means more ordinary-start plan completion and stronger play, not just easier restart success. |
| 4 | **C, conditional** | Small searched-opponent pilot alongside A; expand when useful game survival/plan play improves per unit compute. If not, revisit after A improves the policy. |
| 5 | **E, conditional** | Use the fixed paired dataset to justify deltas, local sharing, or a wider sheet embedding. Promote above H/C if the current network cannot learn reliable labels. |
| 6 | **D** | Gumbel/sequential halving or forced exploration with corrected targets. Evaluate search improvement at fixed budget after Q has a signal. |
| 7 | **F** | Multi-horizon and local outcome heads as supporting ablations. Avoid another large collection of heads without a test showing which decisions improve. |
| 8 | **G** | Reanalysis and lagged multi-step targets after evaluator improvements. Reanalyze on an independent future stream, not the recorded hidden deck. |
| 9 | **I** | Scale the pipeline that demonstrates the right learning curve. More compute remains necessary for world-class strength; §6.8 only argues against scaling the present failure unchanged. |
| Retire | **J** | The existing scaffold failed. Keeping old search policy targets while systematically changing executed choices did not provide explicit evidence comparing those choices. This does not refute outcome-supervised A or state-distribution curricula in general. |

Suggested experimental sequence:

1. **Frozen-checkpoint pilot:** measure terminal contrast reliability and compare ordinary loss against paired loss on identical data. No new architecture, no opponent-search change.
2. **Short learning runs:** baseline, A, B, A+B, with the same initialization and compute accounting. Use multiple continuation/training seeds; do not call single-seed game-level standard errors uncertainty across training runs.
3. **Independent C pilot:** policy-only and cheap searched opposition with the same checkpoint. Only then run the best learning treatment with C.
4. **Plan-frontier experiment:** improve the archive and temporal targets; if families remain absent, add learned multi-turn plan options and broaden counterfactual action coverage.
5. **Scale and search refinement:** larger data budgets, selective extra rollouts, Gumbel, and architecture changes justified by earlier results.

Track an explicit transfer chain: **more reliable outcome labels → lower sibling decision regret → better actual choices → stronger ordinary games → success against stronger opposition**. An improved auxiliary R² is insufficient. A failure at a particular link tells you what to change next.

## 10. What “strongest in the world” requires beyond the current gate

Keep the existing paired gate as a useful regression instrument. Add a separate strength ladder with 2- and 3-player advanced tables reported separately, seat-balanced evaluations, searched candidate-versus-searched incumbent matches, historical specialists, unseen deals, and unseen plan combinations. Track wins/rank using actual tiebreaks; keep score, plans, hygiene, and end reason as explanatory metrics. Longer games and more completed plans are not universally better decisions when ending while ahead wins.

The claim that 99% of high-level human games end through plans is a useful user-supplied observation, but I did not find a primary aggregate dataset establishing that percentage. Treat it as a hypothesis to validate from a representative advanced-game sample, not a training reward or a mandatory 99% target. Human play can supply evaluation cases without becoming a training signal.

Ultimately a world-best claim needs a documented benchmark against top human or independently strong play under the same rules and information access, with enough independent games to support it. The current margin gate against policy-only opposition cannot establish that claim. No paper or proposed experiment here guarantees it; the proposed sequence is designed to remove the demonstrated bottlenecks and expose the next ones.

## 11. Verification and reproducibility

This review inspected the encoder/network, target construction and loss, Rust sample capture, search and opponent sampling, curriculum generation, replay splitting, league, promotion gate, rescue helper, and saved run metadata/metrics. I did not rerun §6's large gameplay experiments or train a new agent.

**Executed:** 89 focused tests passed across `test_network.py`, `test_training.py`, `test_curriculum.py`, and `test_s2_promotion.py`. One existing tensor-to-scalar warning occurred in a test. The read-only audit confirmed encoder ABI 3, the model dimensions, metric calculations, curriculum split counts, and the absolute-seat sensitivity probe. Project Python was run with elevated permissions as required by `AGENTS.md`.

Audit outputs: [evidence.json](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/_hygiene_review_20261002/evidence.json). Reproducer: [inspect_evidence.py](C:/Users/joeld/projects/boardgame-ai/runs/welcome_to_s2/_hygiene_review_20261002/inspect_evidence.py). The audit writes only its own evidence output. No production training, engine, or search code was changed for this review.
