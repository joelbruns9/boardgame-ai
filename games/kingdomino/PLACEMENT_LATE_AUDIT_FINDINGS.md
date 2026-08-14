# Kingdomino late-placement headroom audit

- **Date:** 2026-08-13
- **Checkpoint:** `runs/kingdomino/best_checkpoint/current_best.pt`
- **SHA-256:** `4bf07b0ca14e5452e6533a9232967e89bb0ab0df88c99e9928a65f402b1f04b3`

## Question and exact region

Given the same current board and ordered future claimed tiles, does the current
model lose more final score through placement than strong humans?

Whole-game exact optimization was infeasible. Freezing the actual board after
16 placement opportunities made the remaining eight-tile suffix exactly
enumerable in all feasibility games. The audit therefore covers placements
17–24 only. It cannot assess whether earlier placements created good future
flexibility.

For every reconstructable decision:

1. the logged next pick was held fixed;
2. every legal current placement was enumerated;
3. the remaining actual claimed-tile suffix was optimized exactly, with
   identical board states deduplicated at each layer;
4. human, raw-policy, and searched placements were scored against those same
   exact action values;
5. only the offline reference saw the future claimed-tile sequence.

Search used 4,800 simulations and was constrained at the root to placements
paired with the logged next pick. Deeper search was ordinary non-clairvoyant
open-loop search. Settings and the confirmation rule were frozen before the
confirmation split was opened.

## Results

| Split | Games | Decisions | Human regret | Raw-policy regret | Search regret | Raw − human 95% CI | Search − human 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 21 | 112 | 0.581 | 0.817 | 0.662 | [-0.182, 0.725] | [-0.181, 0.371] |
| Confirmation | 10 | 49 | 1.018 | 0.835 | 0.835 | [-0.713, 0.333] | [-0.711, 0.333] |

Regret values are game-weighted mean final-score points per audited decision.
Intervals are paired whole-game bootstrap intervals; decisions are not treated
as independent observations.

On confirmation, zero-regret fractions were 71.4% for humans and 77.6% for both
raw policy and search. Search chose the same exact-valued placements as raw
policy on every confirmation decision. The largest remaining human errors were
not restricted to the final one or two forced moves: confirmation placements
19–21 still contained meaningful human regret, while placements 22–24 were
zero-regret in the reconstructable sample.

## Decision

The pre-registered criterion required a positive lower confidence bound for
model regret minus human regret. It failed for both raw policy and search; the
confirmation point estimate favored the model.

**Conclusion:** placements 17–24 are not a demonstrated relative weakness of
the current-best network. This result does not support a board-auxiliary
supervision experiment on the basis of late placement. Search appeared to
repair part of a noisy raw-policy gap on development, but confirmation found no
gap to repair.

This is a regional conclusion, not a claim about placements 3–16. Those earlier
decisions are the flexibility-setting region the current exact solver cannot
cover. Investigating them would require a separately validated approximation
or a non-clairvoyant expectation audit; the exact late-game result alone does
not earn that larger build.

## Outputs

- `placement_late_audit_protocol_v1.json`
- `runs/kingdomino/placement_audit/late_human_regret_{development,confirmation}_p1.jsonl`
- `runs/kingdomino/placement_audit/late_raw_policy_{development,confirmation}_p1.jsonl`
- `runs/kingdomino/placement_audit/late_search_{development,confirmation}_p1_s4800.jsonl`
- corresponding `*_summary_*.json` files in the same output directory
