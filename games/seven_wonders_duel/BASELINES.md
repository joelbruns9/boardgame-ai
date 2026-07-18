# Baseline bots

## Random

`RandomBot` chooses uniformly from the complete legal action list. Its private
PRNG is seeded independently from game setup, making both the game and policy
streams reproducible.

## Greedy

`GreedyBot` clones the state once per legal action, applies that action, and
selects the child with the highest deterministic public-feature evaluation. The
evaluation includes current score, coins, military position, distinct science,
resource production, trade discounts, drafted Wonder potential, and turn tempo.
Terminal wins and losses dominate every nonterminal feature.

The heuristic does not inspect face-down tableau identities, removed setup cards,
future Wonder offers, or future Age decks. Although simulation uses a complete
state clone, its evaluator reads only public city and board features and does not
value newly revealed tableau identity.

## Reproducible anchor check

Command:

```powershell
.\.venv\Scripts\python.exe -m games.seven_wonders_duel.baseline_eval --games 200 --seed 20260711
```

Result:

| Metric | Value |
| --- | ---: |
| Greedy wins | 181 |
| Random wins | 19 |
| Draws | 0 |
| Greedy win rate | 90.5% |
| Civilian endings | 190 |
| Military endings | 9 |
| Scientific endings | 1 |
| Average decisions per game | 70.855 |

Player seats alternate while player 0 remains the first Wonder drafter, so each
bot receives each seat and first-player status in half of the games. This is a
deterministic regression anchor, not a calibrated Elo estimate.

