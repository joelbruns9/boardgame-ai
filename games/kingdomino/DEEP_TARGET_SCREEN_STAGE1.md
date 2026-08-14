# Kingdomino deep-target audit: Stage 1 screen

## Outcome

Stage 1 is complete on the frozen development split.
Stages 2 and 3 subsequently completed; the final cross-seed gate was negative.
See `DEEP_TARGET_STAGE3_FINDINGS.md` for the controlling decision.

| Quantity | Result |
|---|---:|
| Development positions searched | 940 |
| Confirmation positions searched | 0 |
| Searches | 1,880 (two independent runs per root) |
| Search budget | 800 simulations |
| Wall time on RTX 3070 Laptop | 20.4 minutes |
| Two-seed exact joint-action agreement | 90.9% |
| Two-seed tile-pick agreement | 99.0% |
| Exact action disagreements | 86 |
| Tile-pick disagreements | 9 |

The action/pick difference matters: the great majority of 800-sim instability
changes placement while preserving the claimed tile. Stage 1 does not establish
that any disagreement has material value.

## Artifacts

- Runner: `games/kingdomino/deep_target_screen.py`
- Results:
  `runs/kingdomino/placement_audit/deep_target_screen_development_s800_r2.jsonl`
- Summary and hashes:
  `runs/kingdomino/placement_audit/deep_target_screen_summary_development_s800_r2.json`
- Tests: `games/kingdomino/tests/test_deep_target_screen.py`

Every search record contains:

- the top three joint placement/pick actions;
- every pick group's total prior, visits, visit share, visit-weighted actor Q,
  and best joint action;
- the raw-network policy at both joint-action and pick-group level;
- the legally reconstructed human action when available; and
- independent deterministic search seeds, checkpoint/corpus hashes, and the
  Stage-1 flags.

## High-recall trigger results

The pre-run Stage-1 flags intentionally favored recall:

| Trigger | Positions |
|---|---:|
| Top-two pick Q within 0.05 in either repeat | 382 |
| At least one zero-visit pick group | 246 |
| Exact-candidate late root | 212 |
| Human pick differs from at least one 800 result | 124 |
| Top pick below 60% visit share | 105 |
| Exact joint action differs between repeats | 86 |
| Tile pick differs between repeats | 9 |
| Random easy control | 33 |
| Union of all triggers | 730 |

The union is not a recommended 4,800-sim workload. In particular, exact
eligibility and human disagreement are useful strata, not evidence that 800
search is uncertain. There are 105 roots triggered only by exact eligibility
and 12 triggered only by human-pick disagreement.

Q-gap sensitivity from the saved telemetry:

| Actor-Q threshold | Close in either repeat | Close in both repeats |
|---:|---:|---:|
| 0.01 | 160 | 112 |
| 0.02 | 239 | 206 |
| 0.03 | 306 | 272 |
| 0.04 | 348 | 324 |
| 0.05 | 382 | 359 |

The Q scale is the search's composite normalized utility, not literal win-rate
percentage points.

## Interpretation and frozen Stage-2 gate

The 800-sim tile decision was already highly repeatable, but repeatability is
not correctness. Stage 2 tested whether 4,800 simulations changed or resolved
the identified cases. Its narrower selection was frozen from these stored rows:

1. all 9 tile-pick disagreements;
2. a Q-gap stratum, preferably requiring the threshold in both repeats;
3. zero-visit pick groups routed to a fair forced-pick probe rather than assumed
   bad from ordinary MCTS alone;
4. a stratified sample of raw-policy/human disagreements; and
5. the 33 random easy controls.

Exact candidates continued through the exact-label path and were not added
wholesale merely to inflate the 4,800 cohort. No confirmation root was searched
before the Stage-2 selection and analysis rules were frozen.
