# Kingdomino deep-target audit: Stage 3 findings

## Decision

Selective replay reanalysis **did not earn a training pilot**. Even on the 32
most suspicious development roots, deeper search did not show reproducible
value improvement over the existing 4,800-simulation tile choices.

Do not:

- build reconstructable self-play sidecars solely for target refresh;
- put raw 30,000-sim Q values into the current replay buffer;
- raise the general self-play search budget above 4,800 on this evidence; or
- spend the frozen 460-position confirmation split on this failed development
  gate.

This closes deep-target qualification unless new independent evidence identifies
a different, non-overlapping failure region.

## Frozen experiment

Stage 3 used the 32-root cohort frozen before any 30,000-sim result existed:

| Route | Roots |
|---|---:|
| Stable tile consensus changed from 800 to 4,800 | 11 |
| Two 4,800 searches still disagreed on the tile | 13 |
| Formerly starved tile within 0.03 Q under matched probing | 8 |

The routes were non-overlapping and covered 16 source games. Every root received:

- two ordinary 30,000-simulation searches using its two Stage-2 seeds; and
- two restricted 10,000-simulation searches for **every** available tile, with
  common seeds within each repeat.

The matched searches give each tile equal conditional search budget. They avoid
mistaking an ordinary MCTS zero-visit group for a low-value group. Runtime was
58.8 minutes on the RTX 3070 Laptop GPU. No confirmation root was searched.

## Apparent regret before noise correction

For each repeat, the in-sample matched regret was the best restricted tile Q
minus the restricted Q of that repeat's 4,800 tile:

| Statistic | Result |
|---|---:|
| Mean | 0.00651 Q |
| Median | 0.00145 Q |
| At most 0.01 | 52/64 |
| Greater than 0.03 | 5/64 |
| Greater than 0.05 | 1/64 |
| Maximum | 0.05648 Q |

This number is positively biased: it chooses the maximum from several noisy Q
estimates and scores that maximum on the same sample. Its nominal
game-clustered interval is therefore not evidence of real improvement. The
warning is visible directly in teacher stability: the equal-budget teacher chose
the same tile in both repeats on only 15/32 roots (46.9%).

Only one root had apparent regret above 0.03 in both repeats. This isolated tail
does not support a dataset-level relabeling mechanism.

## Cross-seed validation

The decisive analysis removes the same-sample winner advantage:

1. use repeat A to select the teacher tile;
2. use independent repeat B to score both that tile and repeat A's 4,800 tile;
3. reverse A and B; and
4. bootstrap by source game.

| Selector evaluated on the other seed | Mean uplift vs 4,800 tile | Median | Positive / negative | Game-clustered 95% interval |
|---|---:|---:|---:|---:|
| Matched 10k-per-tile teacher | -0.00002 Q | 0 | 15 / 20 | [-0.00562, +0.00371] |
| Ordinary 30k search | +0.00094 Q | 0 | 9 / 4 | [-0.000003, +0.00244] |

The matched teacher had one positive and one negative cross-seed result beyond
0.03. Ordinary 30k search had no cross-seed gain beyond 0.03. Thus the deeper
choices do not reproduce as better choices under independent determinization.

Ordinary 30k search changed 13/64 paired 4,800 choices and agreed between its
own two seeds on 25/32 roots (78.1%). Those action changes are real, but their
validated value was negligible. They are predominantly tie-breaking and search
noise, not demonstrated label headroom.

## Interpretation

This was an aggressively enriched cohort, only 32/940 development positions,
selected because earlier search looked least settled. Failure to find
reproducible regret here is stronger evidence against broad reanalysis than a
random-root null would have been.

The conclusion is limited to the tested purpose: refreshing current targets by
spending 20k–30k simulations on selected roots. The 1,400 BGA states can still
serve a different experiment, such as BGA-seeded self-play to broaden the state
distribution. That would test distribution coverage, not label depth, and
should not be described as reanalysis of the current buffer.

## Artifacts

- Runner: `games/kingdomino/deep_target_stage3.py`
- Frozen cohort: `runs/kingdomino/placement_audit/deep_target_stage3_cohort_v1.json`
- Per-root output: `runs/kingdomino/placement_audit/deep_target_stage3_development_s30000_r2.jsonl`
- Summary: `runs/kingdomino/placement_audit/deep_target_stage3_summary_development_s30000_r2.json`
- Tests: `games/kingdomino/tests/test_deep_target_stage3.py`

Per-root output SHA-256:
`8aed787aaf25904e7d39137b8ef420d4c71dac8ea6f35ff94428dfde704ae04d`.
