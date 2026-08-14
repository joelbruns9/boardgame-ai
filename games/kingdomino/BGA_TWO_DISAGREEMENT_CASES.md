# Two extreme BGA pick-disagreement case studies

## Main correction

Neither highlighted row is a simple case where a human chose one final tile and
rejected the model's tile. Mighty Duel gives each player two claims from the
four-tile row. In both games, the opponent took the model-disfavored tile first
and then received the model-favored tile with the second claim.

The original raw-policy disagreement remains correctly measured as a **first
claim** disagreement, but the 29x and 33x probability ratios overstate the
strategic difference between the completed two-tile allocations.

## Rabudipabudi: table 881648336, decision 42

Position:

- final claim round; no unrevealed tiles remained;
- current tile 2: wheat/wheat, no crowns;
- row: 6 (forest/forest), 22 (crowned wheat/swamp),
  27 (crowned forest/wheat), 39 (grass/crowned swamp);
- raw policy: 22 at 70.1%, 27 at 27.4%, 39 at 2.1%, 6 at 0.4%; and
- human first claim: 39.

Actual claim sequence:

1. Rabudipabudi took 39.
2. RollwJoel took 6 and 27.
3. Rabudipabudi received 22 as the remaining second tile.

Thus Rabudipabudi's completed pair was `{39, 22}`; the human did not forgo tile
22. Tile 39 was later impossible to place and was discarded. Final score was
117-116.

The complete remaining game solved exactly in 1.6 seconds:

| First claim, with its best current placement | Exact final margin for Rabudipabudi |
|---|---:|
| 6 | +5 |
| 27 | +5 |
| 22 | +1 |
| 39 (human) | +1 |

The human's actual placement/39 claim and the raw model's best placement/22
claim were exactly equivalent: both preserved a one-point win. The real error
was shared by both: first claiming 6 or 27 would have won by five. No subsequent
reveal luck can explain the result because the deck was already empty.

This is not evidence that the human saw value missing from the model. It is an
example where raw policy probability made two exactly equivalent winning claim
orders look radically different.

## wishiwas: table 883162423, decision 37

Position:

- penultimate claim round; the known remaining tile set was `{5, 17, 26, 36}`;
- current tile 15: wheat/grass, no crowns;
- row: 6 (forest/forest), 7 (water/water),
  20 (crowned wheat/water), 28 (crowned forest/water);
- raw policy: 20 at 91.7%, 6 at 4.4%, 28 at 3.1%, 7 at 0.7%; and
- human first claim: 28.

Actual claim sequence:

1. wishiwas took 28.
2. RollwJoel's exact advisor recommended 7; RollwJoel took 7.
3. wishiwas then took 20.
4. RollwJoel received 6.

Thus wishiwas deliberately or incidentally secured the pair `{28, 20}`—both
crowned tiles—while RollwJoel received `{7, 6}`. Again, the human did not reject
the raw model's tile 20; the disagreement was about which desired tile had to be
secured first across an intervening opponent response.

Whole-root exact solving and the counterfactual exact response after taking 20
first both timed out at 600 seconds. The non-exact comparison used two ordinary
30,000-simulation searches. A separately reported matched tile-group comparison
is withdrawn: the Rust root-action mask was bypassed during missing-child
recovery, so those searches did not give each requested group the claimed
isolated budget.

Both ordinary 30k searches preferred 20 first. The former -0.0154 Q matched
estimate is invalid and must not be cited. Ordinary search still establishes
the preferred action under its own allocation, but it does not independently
quantify the regret of taking 28 first. Final score was 111-106.

The evidence therefore does not show that 28 first was better. It does show why
the raw 29x ratio is misleading: the actual strategic object was an ordered
two-claim allocation, and wishiwas obtained both 28 and 20. There was no unknown
final-row composition after this point—the last four identities were already
known by elimination—although earlier reveals could still have affected the
game's overall luck.

## Consequence for the 606-pick audit

The pick-level artifact is accurate about individual claim order, but its 143
top-pick disagreements should not be interpreted as 143 disagreements about the
final tiles each player received. A better next descriptive statistic would
group each four-tile row into the player's completed two-tile bundle and compare
human versus model allocation/order. That directly handles denial and “take the
scarce tile first” strategy.

## Artifacts

- Exact Rabudipabudi case:
  `runs/kingdomino/placement_audit/bga_disagreement_case_studies_v1.json`
- Superseded deep wishiwas case (matched-Q fields invalid; ordinary searches
  remain valid):
  `runs/kingdomino/placement_audit/bga_wishiwas_disagreement_deep_v1.json`
- Exact case runner: `games/kingdomino/bga_disagreement_case_study.py`
- Deep case runner: `games/kingdomino/bga_disagreement_deep_case.py`
- Counterfactual exact attempt:
  `games/kingdomino/bga_wishiwas_counterfactual_response.py`
