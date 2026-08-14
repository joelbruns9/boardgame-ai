# BGA strong-human versus model pick disagreements

## Scope

This completes the first descriptive step of the strong-human disagreement
audit. It covers all 606 clean single-tile picks made by the opponents in the
36-game frozen BGA corpus.

The model side is `current_best`'s **raw policy**, grouped by claimed tile after
summing over every legal placement. This is the same quantity used by the prior
76.4% human/model top-pick agreement result. It is not an 800- or 4,800-sim
search recommendation and it is not a Q-regret estimate.

Each emitted row records:

- the human and model tiles;
- the complete available row and ranked tile-policy distribution;
- human rank, policy mass, probability gap, and model/human probability ratio;
- claim round, deck count, and currently placed domino;
- opponent/viewer identities; and
- recorded final scores, margin, winner, and development/confirmation split.

Game outcomes were joined only after the disagreement quantities were computed.
No luck attribution or causal claim is made here.

## Corpus totals

| Quantity | Result |
|---|---:|
| Frozen games | 36 |
| Clean opponent picks | 606 |
| Human matched model top tile | 463 (76.4%) |
| Human/model disagreements | 143 (23.6%) |
| Human tile ranked second | 116 |
| Human tile ranked third | 26 |
| Human tile ranked fourth | 1 |
| Human tile below 5% model probability | 22 |

The opponents won 7 games and lost 29. That strong baseline imbalance means the
outcome association below cannot establish that disagreement caused a loss.

## Most extreme individual disagreements

Rows are ranked by `model top-tile probability / human-tile probability`.
`R` is the claimed-tile round, not the placement number of the current domino.

| Game / opponent | R | Available row | Human pick | Model pick | Human mass / rank | Ratio | Opponent result |
|---|---:|---|---:|---:|---:|---:|---:|
| 882633263 / Vichi | 12 | 29, 46, 47 | 46 | 29 | 0.18%, #2 | 548.2x | L 122-141 |
| 881096144 / Trxior | 11 | 24, 40, 46, 47 | 46 | 24 | 0.55%, #3 | 175.5x | L 100-122 |
| 881647491 / kdweezer | 4 | 3, 32, 35, 44 | 44 | 32 | 0.54%, #4 | 133.8x | L 114-168 |
| 881629602 / StanleySu | 3 | 10, 35 | 10 | 35 | 1.01%, #2 | 98.1x | L 112-115 |
| 882560733 / LordRoland | 5 | 7, 28, 35, 44 | 7 | 28 | 0.95%, #3 | 97.7x | L 121-134 |
| 881647491 / kdweezer | 9 | 39, 42 | 39 | 42 | 1.18%, #2 | 84.0x | L 114-168 |
| 882560733 / LordRoland | 4 | 39, 41 | 39 | 41 | 1.38%, #2 | 71.3x | L 121-134 |
| 882132938 / ChronicMonkey24 | 6 | 19, 24, 32, 42 | 32 | 19 | 1.47%, #3 | 60.5x | L 105-154 |
| 881199380 / ginalucia2 | 3 | 39, 45 | 39 | 45 | 1.94%, #2 | 50.7x | L 113-114 |
| 883085653 / JohnPo | 9 | 4, 19, 42, 43 | 42 | 19 | 1.94%, #2 | 49.9x | L 111-138 |
| 883138189 / hclneo | 5 | 9, 31, 44 | 44 | 31 | 1.74%, #3 | 46.7x | L 103-150 |
| 882132814 / r1234 | 7 | 39, 43 | 39 | 43 | 2.74%, #2 | 35.5x | L 108-109 |
| 881648336 / Rabudipabudi | 12 | 6, 22, 27, 39 | 39 | 22 | 2.12%, #3 | 33.0x | W 117-116 |
| 882559864 / hclneo | 12 | 19, 32, 39, 40 | 32 | 19 | 3.15%, #2 | 30.7x | L 123-130 |
| 883162423 / wishiwas | 11 | 6, 7, 20, 28 | 28 | 20 | 3.11%, #3 | 29.5x | W 111-106 |

Thirteen of these fifteen extreme moves occurred in games the opponent lost.
The two exceptions—Rabudipabudi and wishiwas—are especially useful candidates
for a later question about whether the human saw strategic value missing from
the policy or instead benefited from subsequent reveals.

**Follow-up correction.** Exact/deep inspection showed that both opponents later
received the model-favored tile as their second claim from the same row. These
were claim-order disagreements, not rejections of the model tile. Rabudipabudi's
human and raw-model first claims were exactly equivalent at +1 final margin;
wishiwas's deep comparison slightly favored the model order. See
`BGA_TWO_DISAGREEMENT_CASES.md`.

## Games with the largest aggregate disagreement

Aggregate disagreement is the sum, over disagreement moves, of the natural log
of the model/human probability ratio. It is fixed without using final outcomes.

| Rank | Game / opponent | Disagreements | Rank 3/4 | Human mass below 5% | Opponent result |
|---:|---|---:|---:|---:|---:|
| 1 | 881647491 / kdweezer | 9 | 4 | 3 | L 114-168 |
| 2 | 881199380 / ginalucia2 | 7 | 1 | 1 | L 113-114 |
| 3 | 882132938 / ChronicMonkey24 | 7 | 1 | 2 | L 105-154 |
| 4 | 882560733 / LordRoland | 5 | 1 | 2 | L 121-134 |
| 5 | 883162423 / wishiwas | 6 | 1 | 2 | W 111-106 |
| 6 | 882132814 / r1234 | 7 | 0 | 1 | L 108-109 |
| 7 | 881096144 / Trxior | 6 | 2 | 1 | L 100-122 |
| 8 | 883159259 / wishiwas | 6 | 1 | 0 | L 120-121 |
| 9 | 881629602 / StanleySu | 5 | 1 | 1 | L 112-115 |
| 10 | 882633263 / Vichi | 3 | 0 | 1 | L 122-141 |
| 11 | 881648336 / Rabudipabudi | 5 | 1 | 1 | W 117-116 |
| 12 | 883145763 / Wonkey Kong | 5 | 2 | 0 | L 129-133 |

Only one of the top ten aggregate-disagreement games was an opponent win. Across
all games, losing opponents averaged 4.31 disagreement picks and winning
opponents 2.57. This is descriptive only: the viewer won 29/36 games, the number
of clean picks varies by game, raw policy disagreement is not value regret, and
final outcome includes every other decision plus reveal variance.

## Artifacts

- Extractor: `games/kingdomino/bga_opponent_disagreement_audit.py`
- All 606 pick rows: `runs/kingdomino/placement_audit/bga_opponent_pick_disagreements_v1.jsonl`
- Game rankings and top-30 moves: `runs/kingdomino/placement_audit/bga_opponent_pick_disagreements_summary_v1.json`
- Tests: `games/kingdomino/tests/test_bga_opponent_disagreement_audit.py`

The per-pick output SHA-256 is
`7de4eec1e8a4fbe0d86e9ada64f8d4b4168bd66c6ab058abbb288b2a8ad1ab2f`.
