# Kingdomino deep-target audit: Stage 3 findings

## Decision

The corrected development audit **passes the predeclared qualification gate**.
On the frozen 24-root suspicious cohort, deeper tile choices produced a positive
cross-seed, game-clustered lower confidence bound over the 4,800-simulation
choices.

This does not yet authorize a training run. The signal is concentrated in two
positions, and the untouched 460-position confirmation split must now test the
frozen method. A positive confirmation would earn selective high-simulation
reanalysis; it would not justify 30,000 simulations on every self-play move or
put raw search Q values directly into the existing buffer.

## Correction to the first run

The original 32-root result is invalid and superseded. The Rust
`root_allowed_actions` mask was applied during initial expansion, but open-loop
missing-child recovery immediately reintroduced excluded legal actions. As a
result, no nominal 10,000-simulation matched tile search gave its requested tile
the full budget. The old negative conclusion was an artifact of comparing
partially restricted ordinary searches.

The mask now applies throughout root selection. Regression tests verify that
only requested actions are returned, all simulations remain inside the mask,
and mixed valid/invalid masks fail. Audit aggregation also rejects extra root
actions or a visit total different from the requested budget.

## Corrected frozen experiment

The corrected Stage-2 probe removed all eight starvation-derived roots from the
old cohort. The resulting cohort was frozen at 24 roots across 14 source games:

| Route | Roots |
|---|---:|
| Stable tile consensus changed from 800 to 4,800 | 11 |
| Two 4,800 searches still disagreed on the tile | 13 |

Every root used:

- the two existing ordinary 30,000-simulation searches, which never used the
  root restriction and remain valid; and
- two corrected 10,000-simulation searches for every available tile, with
  common seeds within each repeat.

All 136 corrected restricted searches received exactly 10,000 visits in their
requested tile group. The corrected restricted work took 17.36 minutes on the
RTX 3070 Laptop GPU. No confirmation root was searched.

## Corrected results

In-sample matched regret is the best restricted tile Q minus the restricted Q
of that repeat's 4,800 tile. It remains winner-biased because the same noisy
sample selects and scores the maximum.

| In-sample matched statistic | Result |
|---|---:|
| Mean regret | 0.00969 Q |
| Median | 0 |
| At most 0.01 | 44/48 |
| Greater than 0.03 | 3/48 |
| Greater than 0.05 | 3/48 |
| Maximum | 0.15134 Q |
| Teacher two-seed tile agreement | 17/24 (70.8%) |

The decisive analysis selects a teacher tile on one seed, scores it on the
other seed, reverses the direction, and bootstraps by source game.

| Other-seed evaluator | Decision-weighted mean uplift | Game-weighted mean | Median | Positive / negative | Game-clustered 95% interval |
|---|---:|---:|---:|---:|---:|
| Matched 10k-per-tile teacher | +0.00868 Q | +0.00741 Q | 0 | 9 / 10 | **[+0.00038, +0.01847]** |
| Ordinary 30k search | +0.00888 Q | +0.00746 Q | 0 | 8 / 3 | **[+0.00050, +0.01863]** |

The ordinary 30,000-simulation search changed 11/48 paired 4,800 choices. Both
the matched teacher and ordinary search agreed on the tile across their two
seeds on 17/24 roots.

## Concentration and interpretation

The mean is not a broad 0.009-Q improvement on typical positions. Three of the
48 cross-seed comparisons exceeded +0.03, none was below -0.01, and most were
zero or very small. Two BGA positions dominate:

- table `883162423`, source decision 11: approximately +0.135 and +0.148 Q;
- table `881199380`, source decision 32: approximately +0.111 Q in one
  cross-seed direction.

The next-largest gain was about +0.016 Q. This is exactly the pattern for which
selective reanalysis could be useful, but it also makes independent
confirmation essential: one or two source games can create a positive
development mean even with game-clustered resampling.

The actionable gate is therefore:

1. keep the selection rules, search budgets, cross-seed statistic, and
   game-level bootstrap frozen;
2. run the staged screen and deep qualification on the untouched confirmation
   games; and
3. authorize a selective label-generation pilot only if the confirmation lower
   bound is positive.

## Artifacts

- Runner: `games/kingdomino/deep_target_stage3.py`
- Corrected cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_v2.json`
- Corrected per-root output: `runs/kingdomino/placement_audit/deep_target_stage3_development_s30000_r3.jsonl`
- Corrected summary: `runs/kingdomino/placement_audit/deep_target_stage3_summary_development_s30000_r3.json`
- Tests: `games/kingdomino/tests/test_deep_target_stage3.py`

SHA-256:

- cohort: `e48e82eea239de1b276b3475a741eb0582c1a3b972c36be17494fa0291891c09`
- corrected per-root output: `0d01513ae0b8992eeff58b032fa41974faf1509b7d916d6234f6972eed98ca0e`
