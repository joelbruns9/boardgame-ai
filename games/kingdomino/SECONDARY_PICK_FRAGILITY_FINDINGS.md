# Secondary-Pick Fragility: Experiments, Findings, and Next Steps

**Status:** Research and pilot summary, revised 2026-07-22  
**Scope:** Kingdomino AlphaZero/PUCT search, especially joint placement-and-pick actions at the start of a round

## Executive conclusion

The experiments established a real search weakness, but the training pilots have not yet converted it into better search behavior.

At the same root position, secondary tile picks are systematically valued too favorably by ordinary PUCT compared with a deeper, forced reference search. At 3,200 simulations, the median excess fragility of secondary picks over rank-1 picks was `0.1356`; at 10,000 simulations it was still `0.1248`. This is much larger than the rank-1 estimator offset and does not vanish with more ordinary simulations.

The latest value-training treatment then learned the held-out searched targets: value MAE fell `25.4%` and rank-conditioned value-gap MAE fell `16.5%`, while ordinary replay loss changed only `+0.07%`. It also passed the global-Q/rank-1 anti-deflation guards in actual root search. Nevertheless, it failed the behavioral gate. Absolute secondary fragility fell modestly, but rank-1 fragility improved too, secondary-specific excess barely improved at 10,000 simulations and worsened at 3,200, and stable decision flips did not decline.

The important distinction is:

> The network learned that states reached after many secondary picks are worse for the parent than it previously believed. It did **not** learn a sufficiently sharp policy for finding the particular opponent reply that proves why each pick is bad, and ordinary PUCT still allocated too little search inside those branches.

This is not evidence that the idea is exhausted. It says the next experiment should change **search allocation first**, at the natural tile-pick group level, before spending more compute on another scalar-value-only training run.

## 1. Problem definition

### 1.1 The action is factored, but the current search treats it as flat

A Kingdomino action combines:

1. placement of the previously selected domino; and
2. selection of one of the newly offered dominoes.

There may be several roughly equivalent placements for the same tile pick, while the difference between tile picks can be strategically large. A flat PUCT action space therefore spends visits distinguishing many placement variants while a low-prior tile-pick group can remain shallow.

For this investigation, joint actions are grouped by their selected tile. “Rank 1” is the tile-pick group with the highest network prior; “secondary” means policy ranks 2–4.

### 1.2 Fragility

For a root pick and random seed:

```text
fragility = ordinary root-search Q - deeper searched reference value
```

A positive value means ordinary PUCT evaluates the pick more favorably than the forced reference does. The reference explicitly searches all root picks and their opponent-reply continuations, so it is designed to expose a strong reply that ordinary PUCT may not find.

Root Q and the forced reference are not identical estimators. Root Q is a visit-weighted MCTS average; the forced tree uses a different, more exact backup structure. Rank-1 fragility is therefore the method-offset control. The principal behavioral quantity is:

```text
secondary-specific excess = secondary fragility - rank-1 fragility
```

### 1.3 The searched reference is not ground truth

The forced reference and ordinary PUCT use the same checkpoint to evaluate nonterminal leaves. The reference changes where compute is spent and how values are backed up; it does not remove neural leaf-evaluation bias. This matters because the held-out diagnostic below shows that the checkpoint undervalues opponent-to-move states after secondary picks much more than it undervalues states after rank-1 picks.

The working assumption is narrower than “the reference is correct”: forcing coverage of every parent pick and its opponent replies is expected to expose refutations that ordinary PUCT misses, making the searched reference a stronger teacher for this particular failure mode. Residual leaf bias can still affect its magnitude and even its ordering.

The rank-1 control removes a common root-Q-versus-reference offset, but it cannot remove rank-dependent differences in reference reliability. All results should therefore be described as disagreement with a **searched reference**, not error against ground truth. Before a final strength claim, audit the reference on a subset using higher forced budgets/depths, repeated seeds, and terminal or real-game outcomes where feasible.

## 2. Tests and results to date

### 2.1 Overnight search experiment

The overnight test used 50 frozen midgame positions, five root-search seeds, 59 rank-1 tile picks, and 141 secondary picks. Ordinary root search was evaluated at 800, 3,200, and 10,000 simulations against the already-computed forced references.

Runtime on the original RTX 3070 laptop was 7.52 hours.

| Root simulations | Secondary observations | Secondary median fragility | Secondary p90 fragility | Median root-Q seed SD | Positions with stable flips in at least 4/5 seeds |
|---:|---:|---:|---:|---:|---:|
| 800 | 630 | 0.1701 | 0.4245 | 0.00187 | 20 |
| 3,200 | 685 | 0.1514 | 0.4167 | 0.00266 | 18 |
| 10,000 | 695 | 0.1476 | 0.3616 | 0.00308 | 12 |

The rising seed-SD is largely compositional: higher simulation rungs add the hardest previously missing cells (`630 -> 685 -> 695`) rather than measuring exactly the same set at every rung. In every case the seed variation is tiny relative to the fragility effect.

#### Rank-1 baseline and secondary-specific excess

| Root simulations | Rank-1 observations | Rank-1 median fragility | Secondary median | Median excess | Rank-1 p90 | p90 excess |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | 295 | -0.0007 | 0.1701 | 0.1708 | 0.1884 | 0.2361 |
| 3,200 | 295 | 0.0158 | 0.1514 | 0.1356 | 0.1627 | 0.2540 |
| 10,000 | 295 | 0.0228 | 0.1476 | 0.1248 | 0.1547 | 0.2069 |

**Interpretation:** the rank-1 control rules out a generic estimator offset as the main explanation of the measured discrepancy. The disagreement with the searched reference is concentrated in secondary picks. More ordinary PUCT simulations help, but even 10,000 simulations leave a large discrepancy and persistent move-order flips. This does not prove that every forced-reference value is closer to the true game-theoretic value.

The original preregistered classification was technically `BETWEEN`, not a clean pass, because a median value-at-risk statistic of `0.09816` narrowly missed its `0.10` threshold. That conservative routing result should remain in the record. Substantively, however, the rank-1 reanalysis provides strong evidence of systematic secondary-specific overvaluation rather than pure sampling noise.

### 2.2 Cloud label-generation and pipeline validation

The cloud pilot froze 500 training roots and produced 2,000 raw reply candidates. Quality and ambiguity filters accepted 977 examples, split root-disjointly into:

| Split | Accepted examples |
|---|---:|
| Training | 776 |
| Validation | 201 |
| **Total** | **977** |

This confirmed the earlier estimate that a 2,000-accepted-example cap would not bind; 1,200–1,600 accepted examples was a realistic planning range for a larger generation run.

The setup and focused implementation suite completed with `45 passed, 1 skipped` before the later value-work changes. The final focused value-pilot suite completed with `10 passed`; the expanded focused/shared suite completed with `53 passed, 1 skipped`. The only reported warnings were pre-existing tests that returned booleans instead of using `assert`.

### 2.3 Perspective conversion: a critical target invariant

For a parent action that hands the turn to the opponent:

```text
Q_parent(action) = -V_child(child actor)
```

If the forced search stores a value from the parent actor's perspective but the example input is the child reply state, the target must be negated. Training a child state on the unconverted parent value teaches the opposite semantic. This invariant is now explicitly tested and was respected by the final value experiments.

This is separate from the later behavioral failure: the final treatment did learn correctly oriented held-out targets. Its problem was insufficient transfer from a scalar child-state value into better exploration of the decisive reply action.

### 2.4 First pilot: opponent-reply policy loss only

The original treatment added grouped cross-entropy on the opponent's searched reply tile, with `lambda_reply=0.15`, while the control performed the same ordinary replay training without that auxiliary loss.

On the held-out reply set:

| Metric | Control after training | Treatment after training | Result |
|---|---:|---:|---:|
| Reply cross-entropy | 1.2595 | 1.0123 | Treatment better by about 19.6% |
| Ordinary total loss | 1.5341 | 1.5371 | Treatment about 0.2% worse |
| Median reply-state KL to baseline | 0.0217 | 0.0247 | Small shift |

The within-tile placement distribution was not directly supervised by the grouped reply loss. Monitoring showed only a small distributional shift, but placement entropy/KL remains an appropriate safety diagnostic.

The frozen behavioral ladder failed:

| Metric | Control | Treatment | Outcome |
|---|---:|---:|---|
| 3,200 median excess fragility | 0.1136 | 0.1493 | Worse |
| 10,000 median excess fragility | 0.1300 | 0.1334 | Worse |
| 3,200 stable flips | 20 | 20 | No improvement |
| Persistent flips, 3,200 to 10,000 | 14 | 15 | Worse |

**Why this failed:** it successfully made the searched opponent reply more probable on the training metric, but the shared trunk/value head moved without any searched-value supervision. More importantly, improving grouped reply cross-entropy alone did not ensure that root PUCT would allocate enough visits to the relevant secondary branch and then enough visits to the killer reply within it.

### 2.5 Held-out value diagnostic

The 201 validation reply states were then evaluated directly against the searched value from the **child actor's** perspective.

| Parent pick rank | Mean prediction minus searched target | MAE |
|---:|---:|---:|
| 1 | -0.0095 | 0.1058 |
| 2 | -0.1775 | 0.1948 |
| 3 | -0.2333 | 0.2646 |
| 4 | -0.2680 | 0.2856 |

Prediction/target correlation was `0.8184`.

This provides a coherent mechanism for parent overvaluation. After a secondary parent pick, the network systematically undervalues the resulting position for the opponent. Negating that child value makes the parent pick look too good:

```text
opponent value predicted too low
    -> parent action value backed up too high
    -> secondary pick appears safer than forced search says it is
```

### 2.6 Value-loss calibration experiments

Several short treatments were necessary because “teach the searched value” is not by itself a safe loss design.

| Variant | What it tested | Result |
|---|---|---|
| Absolute searched-value loss | Directly regress every reply state to its searched scalar value | Value MAE improved, but rank-1 bias overcorrected and failed the guard |
| Symmetric secondary-vs-rank-1 gap loss | Learn relative gaps within a root instead of absolute level | Gap MAE improved, but the whole value level remained free to drift |
| Detached rank-1 predictions | Use the current rank-1 prediction as a no-gradient anchor | Catastrophic upward rank-1 drift; the loss remained translation-invariant |
| Moving control anchors | Anchor treatment to contemporaneous control predictions | Passed a smoke test, then failed the 1,000-step rank-1 guard |
| Fixed searched rank-1 anchor | Gap loss plus an immutable searched rank-1 target | Best training result and acceptable root-Q deflation behavior |

Deterministic training was also added. Two repeated 100-step runs produced identical control and treatment hashes, eliminating run-to-run nondeterminism as an explanation for the final result.

### 2.7 Final deterministic value pilot

The selected treatment used:

- 1,000 deterministic steps;
- symmetric searched gap loss, weight `1.0`;
- fixed searched rank-1 anchor, weight `8.0`;
- no reply-policy loss in this isolation test (`lambda_reply=0`);
- identical ordinary replay batches for control and treatment.

Training took 458.3 seconds locally.

#### Held-out learning metrics

| Metric | Control | Treatment | Relative change |
|---|---:|---:|---:|
| Ordinary total loss | 1.5352 | 1.5364 | +0.07% |
| Searched value MAE | 0.2327 | 0.1736 | **-25.4%** |
| Secondary-vs-rank-1 gap MAE | 0.1974 | 0.1649 | **-16.5%** |
| Rank-1 mean bias | -0.0237 | +0.0168 | Treatment closer to zero in absolute terms |
| Rank-2 mean bias | -0.2095 | -0.1362 | Improved |
| Rank-3 mean bias | -0.2774 | -0.1920 | Improved |
| Rank-4 mean bias | -0.3010 | -0.2077 | Improved |

This was a genuine supervised-learning success. It is not merely a global pessimistic shift.

#### Frozen root-search behavior

Each checkpoint was evaluated over 500 root/seed/simulation cells. The control and treatment ladders each took about 29 minutes locally. Fixed searched references were reused, so no expensive reference-tree rebuild was required.

| Simulations | Metric | Control | Treatment | Change |
|---:|---|---:|---:|---:|
| 3,200 | Rank-1 median fragility | 0.0564 | 0.0367 | -0.0197 |
| 3,200 | Secondary median fragility | 0.1889 | 0.1791 | -5.2% |
| 3,200 | Secondary-minus-rank-1 median | 0.1326 | 0.1425 | **worse** |
| 3,200 | Secondary-minus-rank-1 p90 | 0.1633 | 0.1386 | -15.2% |
| 3,200 | Missing secondary Q cells | 45 | 53 | **worse** |
| 3,200 | Stable flips | 20 | 21 | **worse** |
| 10,000 | Rank-1 median fragility | 0.0432 | 0.0332 | -0.0100 |
| 10,000 | Secondary median fragility | 0.1836 | 0.1704 | -7.2% |
| 10,000 | Secondary-minus-rank-1 median | 0.1404 | 0.1372 | -2.3% |
| 10,000 | Secondary-minus-rank-1 p90 | 0.2303 | 0.1535 | -33.4% |
| 10,000 | Stable flips | 19 | 18 | Slight improvement |
| Both | Persistent stable flips | 16 | 16 | No improvement |

The common-cell mean root-Q shift was `-0.0077`, and the rank-1 mean root-Q shift was only `-0.0032`. The treatment therefore passed the anti-deflation guard. It failed the overall behavior gate because:

- median excess reduction was below the required 20%;
- missing secondary Q increased at 3,200;
- stable flips remained above 14; and
- persistent flips remained above 8.

These pass/fail thresholds were preregistered engineering routing criteria derived from the baseline effect sizes. They are not independently justified statistical significance thresholds. The tabled treatment-control differences are descriptive: no confidence intervals or position-clustered significance tests were computed, so small “better” or “worse” changes should not be interpreted as resolved effects.

No checkpoint was promoted and `current_best` was not modified.

## 3. What the results mean

### 3.1 “Secondary Q fell” and “secondary search is still poor” are compatible

Root Q is an aggregate estimate produced by the search. Lowering the leaf value at states encountered inside a secondary branch can lower that branch's root Q even if the search never becomes good at locating its strongest opponent continuation.

The final treatment received a scalar lesson:

> “States after secondary picks of this type tend to be better for the opponent than you thought.”

It did not receive a policy lesson in the final isolation run:

> “At this exact opponent reply node, tile C is the refutation; search it before spending visits elsewhere.”

Because `lambda_reply=0`, the opponent-reply policy priors were intentionally left unchanged. Once a secondary root Q falls, ordinary PUCT may allocate **fewer** future visits to that root branch. That can reduce the opportunity to find the decisive reply and explains why missing secondary-Q cells rose from 45 to 53 at 3,200 simulations.

This is the central feedback loop:

```text
low prior or early mediocre samples
    -> few visits to secondary tile group
    -> shallow search of opponent replies
    -> killer reply remains undiscovered
    -> secondary root Q remains unreliable
```

The treatment improved one input to this loop—the scalar leaf evaluation—but did not change the allocation rule that caused the weak evidence.

### 3.2 A scalar value head cannot identify every action-specific refutation

The current value head outputs one value for the state. It can say the opponent-to-move position is favorable, but it cannot say which of the opponent's four tile picks makes it favorable. That identification must currently come from the policy prior plus online PUCT.

A per-tile action-value head is not required to test better search allocation. It is, however, the most direct longer-term way to amortize the expensive information we already computed for **all** searched tile groups.

Raw independent “slot 1/2/3/4” outputs and tile-identity-conditioned values are not interchangeable designs. The recommended architecture is a **shared tile-conditioned scorer**: encode the board and player state once, combine that representation with each offered tile's ID/features, and apply the same scorer to every offer. Operationally it emits up to four masked Q/advantage values, but shared weights make it permutation-equivariant rather than attaching permanent meaning to an offer index. A fixed all-tile-ID head is another viable design, but it computes many masked outputs and shares less structure unless designed carefully.

### 3.3 The nearest established problem is shallow-trap blindness, not a settled named theorem

Searches for the exact phrases **“PUCT's exploitation asymmetry”** and **“unexplored branch optimism”** did not locate a recognized paper or canonical definition. They are reasonable descriptions, but they mix several mechanisms:

- **Low-prior branch starvation:** the policy prior suppresses visits to a tile group.
- **Shallow-trap/refutation blindness:** selective sampling misses a short, strong opponent response. Ramanujan, Sabharwal, and Selman showed that UCT can fail to identify shallow adversarial traps and may search much deeper elsewhere instead ([ICAPS 2010](https://ojs.aaai.org/index.php/ICAPS/article/view/13437)).
- **First-play urgency or initialization bias:** the assumed value of a genuinely unvisited child can be optimistic or pessimistic depending on the FPU rule.
- **Factored-action dilution:** visits are divided among placement variants even though the tile pick is the strategically important component.

Our branches are often **underexplored**, not literally unexplored, and the configured FPU is pessimistic (`-0.2`). Therefore “unexplored branch optimism” is not a complete causal label. “Low-prior secondary-branch starvation with shallow-refutation blindness” is more precise for the evidence observed here.

## 4. What prior research offers

No paper proves a universal solution for this exact combination of a flat joint action, PUCT priors, adversarial replies, and a learned scalar value. Several established methods address parts of it.

### 4.1 Forced playouts plus policy-target pruning — KataGo

KataGo forces a minimum number of root visits to moves that have received any playouts, then subtracts as many of those forced exploratory visits as possible when constructing the policy training target. The key idea is that the distribution needed to **explore** is not necessarily the distribution the policy should imitate. These methods were part of a collection that made KataGo much more compute-efficient than earlier AlphaZero-style Go systems ([Wu, 2019](https://arxiv.org/abs/1902.10565)).

Relevance to Kingdomino:

- Strong fit for guaranteeing evidence on secondary tile groups.
- Policy-target pruning prevents forced exploration from teaching the policy that every forced action is good.
- The published method is principally a self-play/root-policy technique. Applying it hierarchically to opponent reply nodes and tile groups is our adaptation, not a published guarantee.

The current project configuration already contains forced-playout and policy-target-pruning options, but the fragility/reference pipeline did not use a tile-group-aware guarantee at both the parent and reply levels. A flat joint-action guarantee can still waste visits among equivalent placements.

### 4.2 Gumbel action sampling and Sequential Halving

Gumbel AlphaZero was motivated explicitly by AlphaZero failing to improve when not all root actions are visited. It samples actions without replacement and uses Sequential Halving to allocate a fixed simulation budget among candidates, with strong results at low simulation counts ([Danihelka et al., ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO)).

Relevance:

- A natural match for four tile-pick groups.
- Gives controlled initial coverage, then concentrates budget on survivors.
- More principled than simply raising temperature and hoping every group is explored.
- Its root guarantee does not automatically fix underexploration at the opponent reply node; we would need grouped sequential halving there as well, or a second verification pass.

### 4.3 Targeted state coverage — Go-Exploit

Go-Exploit starts some self-play trajectories from archived states of interest. This broadens deep-state coverage, produces more independent value targets, and improved AlphaZero sample efficiency in Connect Four and 9x9 Go ([Trudeau and Bowling, 2023](https://arxiv.org/abs/2302.12359)).

Relevance:

- Start training games or auxiliary rollouts from the child states created by vulnerable secondary picks.
- Obtain real game outcomes or stronger downstream searches instead of relying only on a fixed teacher.
- Addresses value generalization and state-distribution coverage, but does not alone guarantee that online PUCT finds a particular reply.

### 4.4 Amortizing searched action values — SAVE

Search with Amortized Value Estimates learns a prior over state-action Q values, improves those values with MCTS, and trains the prior using both search and real experience. It was designed to retain action-value information that ordinary search often discards and achieved strong results with small search budgets ([Hamrick et al., ICLR 2020](https://arxiv.org/abs/1912.02807)).

Relevance:

- Strong research precedent for preserving searched action values; in this game, the natural implementation is a shared tile-conditioned Q/advantage scorer evaluated on each offered tile.
- Lets every sufficiently searched Tile A/B/C/D group contribute a target even when only Tile A is actually played.
- Directly supplies PUCT with action-specific value priors, instead of expecting a scalar state value to identify the reply through repeated search.
- Requires architectural and search changes and careful mixing with real-outcome targets to avoid merely distilling teacher bias.

### 4.5 Action abstraction and policy pruning

Research on partial policies shows that restricting search to learned action subsets can improve anytime performance when branching is large, provided the subset retains good actions ([Pinto and Fern, JMLR 2017](https://jmlr.org/papers/v18/15-251.html)). More recent state-conditioned action abstraction specifically targets factored/combinatorial actions and discards redundant sub-actions during MCTS ([Kwak et al., UAI 2024](https://proceedings.mlr.press/v244/kwak24a.html)).

Relevance:

- Prune or merge redundant **placements within a tile-pick group**, then spend the saved budget comparing tile picks and opponent replies.
- Do not prune whole low-prior tile groups using the current policy: that risks deleting exactly the hidden refutation we are trying to discover.
- Re-evaluate every retained tile group after placement pruning.

### 4.6 Regularized policy optimization and adaptive exploration

AlphaZero's visit heuristic can be interpreted as an approximation to regularized policy optimization; solving the associated update more exactly outperformed the original heuristic in several domains ([Grill et al., ICML 2020](https://proceedings.mlr.press/v119/grill20a.html)). KataGo also scales cPUCT using empirical utility variance, exploring more when values are variable and concentrating when they are stable ([KataGo methods documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#dynamic-variance-scaled-cpuct)).

Relevance:

- These are principled alternatives to one global cPUCT and a high-temperature brute-force search.
- Variance-scaled exploration could target ambiguous reply nodes without uniformly increasing work.
- They are supporting options, not the first experiment: neither guarantees minimum tile-group coverage by itself.

### 4.7 “Move verification”

A search of Lc0's public site and GitHub materials did not verify **Move Verification** as the name of a documented Lc0 research algorithm. We should not attribute it to Leela Chess Zero without a specific source.

It remains a useful project term for a straightforward two-pass design:

1. run normal PUCT;
2. identify top, close, unstable, or underexplored tile groups;
3. independently re-search each selected group with a fixed budget;
4. search the opponent reply groups with a coverage guarantee; and
5. choose using the verified values.

This resembles selective extension or tactical verification in spirit, but the exact grouped algorithm below would be our own implementation and must be validated empirically.

## 5. Candidate fixes, ranked

| Candidate | Fixes root group coverage | Fixes reply coverage | Creates better training targets | Implementation risk | Recommendation |
|---|---:|---:|---:|---:|---|
| Grouped forced playouts + target pruning | Yes | Yes, if applied recursively | Yes | Medium | **Test first** |
| Grouped Gumbel + Sequential Halving | Yes | Yes, if applied at reply nodes | Yes | Medium/high | Test beside forced playouts |
| Two-pass grouped verification | Yes for selected groups | Yes | Yes | Medium | Strong diagnostic and fallback |
| Placement pruning + tile re-evaluation | Indirectly | Indirectly | Yes | Medium | Combine after safe placement grouping |
| Reply-policy + anchored value training | No by itself | Improves reply prior | Yes | Low/medium | Retry only after search target improves |
| Shared tile-conditioned Q/advantage head | Supplies direct prior | Supplies direct reply prior if used at both states | Yes | High | Best longer-term architecture |
| High temperature + many simulations | Weak/indirect | Weak/indirect | Possibly | Low | Diagnostic only; poor efficiency |

## 6. Recommended next experiment

### Phase A — Search-only allocation study

Do **not** train a new network first. Hold the checkpoint fixed and determine whether a different search allocator can expose the forced references at the same total simulation budget.

Implement search over the natural hierarchy:

```text
root state
  -> tile-pick group (4 or fewer)
      -> legal placement variants for that pick
          -> opponent reply state
              -> opponent tile-pick group
                  -> opponent placement variants
```

Use nested arms on a development copy of the frozen positions so the parent and reply allocation effects can be separated:

1. current flat PUCT baseline;
2. **parent-only tile-group visit floor**, with ordinary PUCT below the selected parent group;
3. parent and opponent-reply tile-group visit floors;
4. grouped Gumbel/Sequential Halving, first at the parent and then at both levels if the parent-only variant is insufficient; and
5. normal PUCT followed by grouped verification of the top, close, or underexplored picks.

The parent-only floor is the cheapest localization test. If it closes most of the gap, recursive reply grouping is unnecessary. If parent coverage improves while the reference discrepancy remains, the remaining failure is localized to search inside the opponent reply nodes.

Keep total simulations equal at 3,200 and 10,000. Track actual neural evaluations and wall time as well, because a nominal simulation can become more expensive if grouping reduces batching efficiency.

Retain the existing behavioral gates as engineering routing criteria:

- median secondary-minus-rank-1 fragility reduction at least 20%;
- p90 excess reduction at least 10%;
- no material rank-1 median-fragility or mean-root-Q shift;
- missing secondary Q must not increase;
- at most 14 stable flips at 3,200;
- at most 8 persistent flips from 3,200 to 10,000;
- no increased dependence on tie handling or seed instability.

In addition, report paired uncertainty by resampling **positions**, not individual picks or seeds. Picks and seeds from the same root are correlated. Use a position-clustered paired bootstrap for median and p90 excess fragility and bootstrap intervals for the flip-count differences. Designate median excess at 3,200 as the primary endpoint; treat the other gates as safety/diagnostic endpoints rather than multiple interchangeable ways to declare success.

The current frozen 50 has now influenced method design, so treat it as a development set. Before generating the untouched confirmation set:

1. choose the minimum worthwhile primary effect in advance—the existing 20% median-excess reduction is the current engineering target;
2. use the development-set position clusters in a simulation-based paired power analysis;
3. select the number of new positions needed for at least 80% power, preferably 90%, at a two-sided `alpha=0.05` for that primary effect; and
4. freeze the sample size, seeds, budgets, exclusion rules, and primary analysis before examining confirmation results.

The numerical confirmation-set size is intentionally not guessed here; it should come from the empirical position-level variance and missing-cell pattern. The old `≤14` and `≤8` flip thresholds remain useful operational targets, but they must not substitute for uncertainty estimates.

Before using the selected allocator as a teacher, audit the searched reference on a stratified subset. Re-run it at a larger forced budget/depth and additional seeds, compare rank-specific stability and Monte Carlo error, and use terminal continuations or real-game outcomes where affordable. Failure of the reference itself to stabilize blocks target generation even if an allocator appears to match it.

### Phase B — Generate improved multi-action targets

If a search-only arm passes, use it to write one record per root containing all sufficiently verified tile groups:

```text
state
offered-tile mask
for each offered tile group:
    offer slot
    tile ID and tile features
    searched parent Q
    visit count and uncertainty
    best/mixture placement target within the group
    child reply state
    opponent reply-group policy target
    searched child-actor value
played tile and eventual game outcome
```

Only the played action receives a real trajectory outcome in that game. Unplayed Tile B/C/D groups still receive **search-teacher** Q targets. This distinction must be stored so losses can weight real outcomes more strongly and avoid treating correlated search estimates as independent ground truth.

### Phase C — Combined policy/value pilot

The next small training pilot should combine, rather than isolate:

- grouped opponent-reply policy loss, to point search toward the killer response;
- searched relative-value gap loss, to encode how costly the secondary choice is;
- fixed searched rank-1 anchoring, to prevent translation/global deflation;
- ordinary replay and real-outcome value loss, to preserve calibrated play;
- within-group placement entropy/KL monitoring;
- rank-1/root-Q anti-deflation guards; and
- deterministic paired control/treatment training.

The search-only test should come first: training on targets produced by the same starving allocator risks distilling its blind spot.

### Phase D — Shared tile-conditioned action-value head if transfer remains weak

If improved search targets plus combined policy/value losses still do not transfer, add a shared action-value scorer `Q(s, tile)` or `A(s, tile)`. Evaluate the same scorer for each currently offered tile using its ID/features, return up to four masked values, train every verified tile group, and consume the result as a search prior—not as an unsearched replacement for PUCT. The target schema must preserve both tile identity and its temporary offer slot so alternative architectures remain testable.

This is the cleanest answer to “how can the network learn Tile B if Tile B is not played?” The deep search supplies a supervised Tile B Q target; the Q head amortizes it. Real outcomes from played actions and archived secondary-state trajectories keep the target grounded.

### Phase E — Strength confirmation

Only after the internal behavior gate passes:

1. evaluate on the untouched fragility set;
2. run the frozen BGA/fixed-suite regression checks;
3. run paired head-to-head games under the intended inference budget;
4. measure Elo/win-rate confidence intervals and throughput; and
5. promote only if behavior, calibration, and playing strength agree.

## 7. What not to conclude

- The overnight experiment does **not** prove that retraining will increase Elo. It establishes a repeatable search/value discrepancy.
- The final value pilot did **not** fail to learn. It learned its supervised targets but failed the root-search behavioral objective.
- Lower secondary Q is not sufficient. The relevant outcome is secondary-specific excess fragility, branch coverage, and decision stability relative to rank 1.
- High temperature is not equivalent to forced coverage. It redistributes probability but offers no minimum evidence guarantee.
- Policy pruning is not safe when applied to whole tile groups using the current policy. The hidden low-prior reply may be the important one.
- A per-action Q head need not be variable length, but raw slot identity should not carry the semantics. A shared tile-conditioned scorer can emit four masked values while preserving tile identity and permutation symmetry.
- None of the cited methods can be assumed to transfer unchanged. Kingdomino's joint placement/pick factorization and opponent-reply semantics require direct ablations.

## 8. Decision

The most informative next implementation is a **search-only, tile-group-aware allocation experiment**. Start with a parent-only tile-group visit floor, then add reply-level grouping only if needed; compare the result with grouped verification and Gumbel/Sequential Halving. This directly tests and localizes the remaining bottleneck exposed by the final pilot.

If one of those arms finds the reference replies reliably at the same budget—and the reference audit shows that those values stabilize—use its trees to train both the reply policy and anchored value gaps. If training still cannot amortize that improvement, proceed to a shared tile-conditioned Q/advantage scorer following the SAVE-style principle of preserving searched action values.

## 9. Local artifact index

- Original experiment and pilot plan: `games/kingdomino/AZ_SECONDARY_PICK_REPLY_PILOT_PLAN.md`
- Policy-only training report: `runs/kingdomino/reply_pilot/cloud/training/pilot_training_report.json`
- Policy-only behavior report: `runs/kingdomino/reply_pilot/cloud/evaluation/behavior_report.json`
- Final deterministic value training report: `runs/kingdomino/reply_pilot/cloud/value_pilot/det_gap_searched_rank1_anchor8_1000/pilot_training_report.json`
- Final deterministic behavior report: `runs/kingdomino/reply_pilot/cloud/value_pilot/det_gap_searched_rank1_anchor8_1000/evaluation/behavior_report.json`
- Final treatment root ladder: `runs/kingdomino/reply_pilot/cloud/value_pilot/det_gap_searched_rank1_anchor8_1000/evaluation/treatment_root_ladder.jsonl`
- Final control root ladder: `runs/kingdomino/reply_pilot/cloud/value_pilot/det_gap_searched_rank1_anchor8_1000/evaluation/control_root_ladder.jsonl`

## 10. Primary references

1. David J. Wu, [“Accelerating Self-Play Learning in Go”](https://arxiv.org/abs/1902.10565), 2019 — KataGo forced playouts and policy-target pruning.
2. Ivo Danihelka, Arthur Guez, Julian Schrittwieser, and David Silver, [“Policy improvement by planning with Gumbel”](https://openreview.net/forum?id=bERaNdoegnO), ICLR 2022 — sampling without replacement and Sequential Halving.
3. Raghuram Ramanujan, Ashish Sabharwal, and Bart Selman, [“On Adversarial Search Spaces and Sampling-Based Planning”](https://ojs.aaai.org/index.php/ICAPS/article/view/13437), ICAPS 2010 — shallow traps and UCT failure modes.
4. Alexandre Trudeau and Michael Bowling, [“Targeted Search Control in AlphaZero for Effective Policy Improvement”](https://arxiv.org/abs/2302.12359), 2023 — Go-Exploit and archived-state starts.
5. Jessica B. Hamrick et al., [“Combining Q-Learning and Search with Amortized Value Estimates”](https://arxiv.org/abs/1912.02807), ICLR 2020 — searched action-value amortization.
6. Jean-Bastien Grill et al., [“Monte-Carlo Tree Search as Regularized Policy Optimization”](https://proceedings.mlr.press/v119/grill20a.html), ICML 2020 — a principled view of AlphaZero search/policy updates.
7. Jervis Pinto and Alan Fern, [“Learning Partial Policies to Speedup MDP Tree Search via Reduction to I.I.D. Learning”](https://jmlr.org/papers/v18/15-251.html), JMLR 2017 — learned action subsets for time-bounded search.
8. Yunhyeok Kwak et al., [“Efficient Monte Carlo Tree Search via On-the-Fly State-Conditioned Action Abstraction”](https://proceedings.mlr.press/v244/kwak24a.html), UAI 2024 — abstraction for factored/combinatorial actions.
9. David J. Wu, [KataGo methods documentation](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — dynamic variance-scaled cPUCT and related post-paper methods.
