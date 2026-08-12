# K1 Baseline Evaluation

K1 supplies four deterministic-in-interface evaluation anchors:

1. `random` samples uniformly from legal actions.
2. `immediate_score` values current score gain and next-domino material.
3. `flexibility` adds future placement count and a small board-shape term.
4. `denial_aware` adds the claimed domino's value to the strongest rival use.

All scoring heuristics honor the active Harmony and Middle Kingdom toggles.
Denial is scaled to 40% of its four-player weight in 3p because one offered
domino can go unclaimed; in 4p, every offered domino must be taken.

## Evaluation protocol

- Configuration: `harmony=true`, `middle_kingdom=true`.
- Held-out deck seeds: 20 through 39 inclusive.
- One challenger plays against copies of the preceding tier.
- Each seed is repeated with the challenger rotated through every seat.
- Seat zero starts each game, so the rotations also balance starting position.
- Tied wins are split equally.
- Equal-strength win share is `1 / players`, not 50%.
- Denial weights were selected on calibration seeds 0 through 9; no reported
  seed was used for that selection.

| Players | Challenger | Field | Games | Win share | Neutral | Lift | Mean rank | Last-place rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 3 | immediate_score | random | 60 | 91.67% | 33.33% | +58.33 pp | 1.117 | 3.33% |
| 3 | flexibility | immediate_score | 60 | 66.67% | 33.33% | +33.33 pp | 1.400 | 6.67% |
| 3 | denial_aware | flexibility | 60 | 38.33% | 33.33% | +5.00 pp | 1.967 | 35.00% |
| 4 | immediate_score | random | 80 | 76.25% | 25.00% | +51.25 pp | 1.375 | 2.50% |
| 4 | flexibility | immediate_score | 80 | 67.50% | 25.00% | +42.50 pp | 1.525 | 6.25% |
| 4 | denial_aware | flexibility | 80 | 32.50% | 25.00% | +7.50 pp | 2.425 | 26.25% |

Every intended tier has positive held-out win-share lift in both player counts.
This is the directional K1 gate, not a statistical promotion claim; future
model promotions should use larger suites and uncertainty estimates.

## Reproduction

Run one row with:

```powershell
.\.venv\Scripts\python.exe -m games.kingdomino_classic.baseline_eval `
  --challenger denial_aware --field flexibility --players 4 `
  --seed-start 20 --seeds 20
```

Use `--no-harmony` or `--no-middle-kingdom` to evaluate another scoring stratum,
and `--include-games` when per-game records are needed. Full eight-configuration
reporting remains part of K5.
