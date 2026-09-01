# Kingdomino deep-target audit: Stage 3 findings

## Decision

The corrected development audit passed its qualification gate, but the frozen
confirmation pass **failed the predeclared hard gate** on 2026-08-13. The
matched-teacher cross-seed uplift had a game-clustered 95% interval of
[-0.00165, +0.00127] Q, so its lower bound was not above zero.

Selective high-simulation reanalysis is therefore not authorized as a training
component. The captured positions remain useful as diagnostics, but this audit
is closed rather than iterated or adaptively reselected.

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

## Frozen confirmation result

The untouched confirmation split was run with the same staged rules and no
adaptive changes:

- 460 roots received two 800-simulation searches;
- the frozen Stage-2 rules selected 84 roots for paired 4,800-simulation
  searches;
- the corrected starvation probe covered 8 roots and every tile group;
- the frozen suspicious cohort contained 11 roots across 7 source games: 7
  stable consensus changes, 3 unresolved 4,800 tile disagreements, and 1
  corrected starvation route; and
- every Stage-3 root received two ordinary 30,000-simulation searches and two
  corrected 10,000-simulation restricted searches per tile.

| Confirmation statistic | Result |
|---|---:|
| Matched-teacher decision-weighted cross-seed uplift | +0.00041 Q |
| Matched-teacher game-weighted uplift | -0.00006 Q |
| Matched-teacher game-clustered 95% interval | **[-0.00165, +0.00127]** |
| Ordinary-30k decision-weighted cross-seed uplift | -0.00006 Q |
| Ordinary-30k game-clustered 95% interval | [-0.00120, +0.00293] |

The development effect did not reproduce. The confirmation LCB is negative,
which mechanically fails the frozen `LCB > 0` gate.

The actionable outcome is therefore to drop selective reanalysis from the
final training attempt. Reopening it would require genuinely new evidence, not
another selection or budget iteration on these frozen splits.

## Artifacts

- Runner: `games/kingdomino/deep_target_stage3.py`
- Corrected cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_v2.json`
- Corrected per-root output: `runs/kingdomino/placement_audit/deep_target_stage3_development_s30000_r3.jsonl`
- Corrected summary: `runs/kingdomino/placement_audit/deep_target_stage3_summary_development_s30000_r3.json`
- Confirmation cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_confirmation_v1.json`
- Confirmation per-root output: `runs/kingdomino/placement_audit/deep_target_stage3_confirmation_s30000_r1.jsonl`
- Confirmation summary: `runs/kingdomino/placement_audit/deep_target_stage3_summary_confirmation_s30000_r1.json`
- Tests: `games/kingdomino/tests/test_deep_target_stage3.py`

SHA-256:

- cohort: `e48e82eea239de1b276b3475a741eb0582c1a3b972c36be17494fa0291891c09`
- corrected per-root output: `0d01513ae0b8992eeff58b032fa41974faf1509b7d916d6234f6972eed98ca0e`
- confirmation cohort: `22d661d22dec9ff05ab7cb0788c86d4594dab656fae31cec1941bf33d4407ab8`
- confirmation per-root output: `2624c34e0e791e25bae6530cd3904b6db9328c19512f0dcd485a1ba7b2ab1190`
- confirmation summary: `37d41a9ea14d39da17f4d24ff8169d733923176fd2c06facdaec72740fe6df59`
