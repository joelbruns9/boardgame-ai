# Baseline headroom diagnostic

Run fresh ordinary baseline games, reservoir sample 10 eligible decisions and
200 turn-start occurrences per rule variant, then perform two diagnostics:

1. Compare the NN with its exact one-turn backup on the 2,000 turn-start boards.
   These boards have no runners. Residuals are in absolute seat coordinates.
2. Evaluate the baseline and its highest-valued rivals (up to three actions)
   using 256 full-game rollouts per action. Pick an action using samples 0–127,
   and score its paired gain over baseline on samples 128–255. Keep baseline
   selectable, prefer baseline on exact ties, and retain negative held-out gains.

The full-game continuation uses the baseline NN-guided turn solver. Payoffs come
from terminal winners, with shared simulated dice and zero-mean dice-luck
correction. The pooled backend starts after Python completes each forced root
turn. Future opening rolls remain uncorrected, as in the independent audit.

Sampling is uniform over eligible occurrences within each rule variant. Repeated
states are eligible; their frequency is part of the population. Both sampling
streams are independent of game dice. This is equal weighting of ten variants,
not a claim about their frequency in an arena. Source-game IDs are retained;
an exploratory stratified source-game cluster bootstrap accompanies the mean.
Conditional Monte Carlo uncertainty treats sampled roots and selected actions
as fixed. Few source games can make bootstrap intervals unstable.

```powershell
& C:\Users\joeld\projects\boardgame-ai\.venv\Scripts\python.exe -m games.cantstop.decision_headroom --checkpoint runs/p4_pilot/iter_0080.pt --device cuda --out runs/decision_headroom_20260930.json
```

Run from the Can’t Stop worktree. Expected full-game stage: roughly 30–50 minutes
on the measured RTX 3070 Laptop, plus collection and the consistency stage.
Progress reports actual timing; positions near the opening can take longer.
To resume, repeat the identical command with `--resume`. Sources, native binary,
checkpoint and configuration must match. Each successful batch is saved before
updating the report, so an interruption does not discard completed samples.
Any unfinished game fails its entire batch. No partial trajectories are scored.

Artifacts include a frozen `.positions.json` collection, per-root `.samples`
files with every raw/adjusted seat vector and batch cost, the main JSON report,
and a readable Markdown report. Candidates are frozen before rollout sampling.

One-turn consistency does not establish value accuracy. This held-out test
measures the specified candidate-search procedure with baseline continuation.
It is not an upper bound on root-search headroom: selection samples can miss a
better candidate and the top-three filter can omit better legal actions. Arena
strength requires independent games against an opponent.
