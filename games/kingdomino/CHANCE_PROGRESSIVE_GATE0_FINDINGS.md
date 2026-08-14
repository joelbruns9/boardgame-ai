# Kingdomino progressive-chance Gate-0 findings

- **Status:** Current progressive treatment closed
- **Date:** 2026-08-13
- **Checkpoint:** `runs/kingdomino/best_checkpoint/current_best.pt`
- **Checkpoint SHA-256:**
  `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`
- **Source branch:** `codex/kingdomino-chance-correct` at `ee3950685f07`

## Question

Does the implemented deck-8/deck-12 progressive-chance search beat ordinary
open-loop MCTS when both use the identical incumbent network and matched neural
evaluation budgets?

The treatment created observation-conditioned chance panels after eligible
reveals. It initialized and progressively widened balanced row panels up to a
cap of 16. The control was the production open-loop search. Both orientations
of each deck seed were played, so the uncertainty unit was the complete
same-seed seat pair.

## Results

| Budget | Games / pairs | Progressive score | One-sided 95% interval | Mean score margin | Verdict |
|---:|---:|---:|---:|---:|---|
| 800 NN rows per move | 2,048 / 1,024 | 48.24% | [47.34%, 49.15%] | -0.475 | Fail |
| 4,800 NN rows per move | 4,096 / 2,048 | 49.46% | [48.69%, 50.23%] | -0.059 | Inconclusive |

The higher-budget mechanism was not starved. At 4,800 rows, mean active width
was 15.98/16 and mean mature width was 15.93/16. Both eligible deck counts were
treated in every expected search. Bootstrap initialization consumed about 350
rows per treated search, or roughly 7.3% of the budget.

The 4,800-simulation result recovered most of the low-budget deficit but
converged toward parity, not superiority. The preregistered promotion condition
required a one-sided lower confidence bound above 50%; neither budget passed.

## Decision

Do not:

- enable this progressive treatment in advisor play;
- use it to generate another self-play training cycle;
- tune the width cap against the viewed match seeds; or
- extend the same topology to more reveal depths.

This closes the implemented cap-16 progressive topology and parameterization,
not every information-safe open-loop abstraction. A compact contextual scheme
may receive one offline qualification because it pools strategically similar
observations instead of paying the exact-row panel cost. It must pass the
shared-reference gate in `KINGDOMINO_CONTEXTUAL_OPEN_LOOP_PLAN.md` before any
engine implementation.

## Evidence

The downloaded result files are preserved in the experimental worktree cloud
bundle under:

`data/kingdomino/kingdomino_chance_progressive_cloud_v1_20260813/runs/kingdomino/chance_progressive_cloud_v1/`

- `gate0_incumbent_search_ab_800sims_2048/result.json`
  - SHA-256:
    `2e6ef19e76760a34ecc6b8b5b43141e1e60a4e7e49efb5098aaa6adfac4f324e`
- `gate0_teacher_search_ab_4800sims_4096/result.json`
  - SHA-256:
    `8afb18c599529feaac0f2cbab58e1a5c1595916d99a92870d0965eec18a03f68`

The implementation, full experiment history, and cloud-run instructions remain
on `codex/kingdomino-chance-correct`. This document is the canonical result
summary on `main`.
