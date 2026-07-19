# Phase D loop infrastructure

`phase_d.py` is the executable toy-scale loop described in `AZ_PROJECT_PLAN.md`:

```text
current_best -> coalesced self-play workers -> replay window -> candidate train
      ^                                                        |
      |                                                        v
      +---- atomic promotion <- paired SPRT gates <- candidate checkpoint
```

The game-agnostic pieces live in `games.az_loop`: the `GameAdapter` seam,
deterministic job runner, paired-match runner, linear schedules, SPRT, HOF, Elo,
and run manifests. `games.kingdomino.loop_adapter` is the regression client;
`loop_adapter.py` is the 7WD client.

## Curriculum and labels

- `seed_games=5000` creates replayable Greedy-versus-rush games, cycling all four
  science/military aggressive/economy bots through both seats and starting roles.
- The entire seed buffer is present at iteration 0 and linearly retained to zero
  over ten iterations.
- Fifteen percent of generated games initially use a rush-bot opponent, also
  annealed to zero over ten iterations.
- Wonder-draft priors blend from tier-only to network-only over twenty iterations.
- Each network move independently chooses cheap (16-24) or full (64-128) search;
  only full-search moves carry policy loss. Cheap moves retain exact outcome and
  auxiliary labels. Bot labels disappear with the curriculum.
- Online validation holds out whole games within every self-play iteration, so
  fresh games enter training immediately without game leakage. Unlabeled
  curriculum games remain training-only. The current-best model is also measured
  on the entire newest iteration before training as a temporal diagnostic.

## Gates and persistence

Every candidate faces current-best at a 50% threshold; that SPRT alone controls
promotion, preserving a monotonic strength ratchet. Greedy at 65% and every rush
variant at 60% are Phase D exit criteria, not promotion requirements. These anchor
SPRTs run after every third promotion by default (`anchor_gate_every_promotions`),
or explicitly through `PhaseDLoop.phase_gate()`. Rejected/inconclusive candidates
never pay for anchor matches. Each SPRT decision is checked only after a paired
seed has put the candidate in both seats.

Each run directory contains:

```text
run_manifest.json
dirty_diff.patch
buffers/curriculum_seed.jsonl
buffers/iter_NNNN.jsonl
checkpoints/current_best.pt
checkpoints/candidate_NNNN.pt
hof/hof_index.jsonl
elo/elo.json
elo/elo_games.jsonl
```

Existing manifest iterations are detected on restart; `--iterations N` performs
N additional iterations. Existing iteration buffers are never overwritten after
an interrupted run, so recovery cannot silently mix two trajectories.

Each iteration manifest row reports victory and game-type mix, policy-eligible
move fraction, average search simulations, wall-clock games/second, inference
positions/batches and mean realized batch size. Candidate checkpoints retain the
training history and target base rates, so auxiliary-head progress and a collapsed
policy stream are diagnosable without reconstructing the run.

## Commands

Plumbing-only smoke mode uses two generated games, eight seed games, one-simulation
search, one training epoch, and two gate games per opponent:

```powershell
.\.venv\Scripts\python.exe -m games.seven_wonders_duel.phase_d `
  --run-dir runs/seven_wonders_duel/phase_d_smoke `
  --device cpu --plumbing-smoke
```

The default toy configuration is intentionally much larger (500 games/iteration,
5,000 seed games). Running it is the separate empirical Phase D gate, not part of
the plumbing smoke check.
