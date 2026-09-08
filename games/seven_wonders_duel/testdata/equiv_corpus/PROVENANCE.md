# Equivalence corpus — provenance

The 50 games the engine-equivalence gate runs on (W6.2). Committed rather than
generated on demand because `runs/` is gitignored: a corpus that only exists on
the gate box makes the cloud smoke pass vacuously, which is worse than no smoke.

Rebuild with `python -m games.seven_wonders_duel.build_equiv_corpus`; that
script is the authority on how these files are made and why each stratum exists.

## Current corpus

**Generated 2026-09-08 under `SPEC_VERSION` codec-2.**

| file | games | moves | states | victory mix |
|---|---:|---:|---:|---|
| `curriculum_seed.jsonl` | 20 | 1,146 | 1,166 | 11 scientific / 9 military |
| `selfplay_early.jsonl` | 15 | 991 | 1,006 | 11 civilian / 3 military / 1 scientific |
| `selfplay_late.jsonl` | 15 | 1,074 | 1,089 | 12 civilian / 2 scientific / 1 military |

3,261 encoded states, covering all 9 decision branches and all 9 token types.
Every record replays clean through `buffer.replay` on this engine.

Built with:

    python -m games.seven_wonders_duel.build_equiv_corpus \
        --late-checkpoint runs/seven_wonders_duel/cloud2/\
    7wd_cloud_20260825T005745Z/checkpoints/candidate_0085.pt \
        --late-iteration 85 --migrate

The late stratum came from that run's iteration 85 (384x8; `candidate_0085.pt`,
byte-identical to `learner_0085.pt`). Note it is NOT that run's `current_best.pt`,
which records iteration 70 -- 85 is the stronger net, not the last promoted one.
The early stratum is an untrained net seeded at 20260803 at Phase D's default
size, both at Phase D's default sims. The bot stratum is
`phase_d._bot_seed_game`, i.e. literally the seed step of a run.

`--migrate` was needed and is recorded here because it changes what played the
late games: the checkpoint predates the September encoder work (W3 control
channels, then the reveal-risk channels), so its signature no longer matches.
The migration is purely additive -- 141 tensors loaded, 2 GROWN
(`embedder.feature.global.weight`, `embedder.feature.tableau.weight`), 0
initialized -- and the grown columns are zero-filled, so the net played exactly
as it did when trained.

## Why it was regenerated

To refresh the late stratum. The previous one came from `laptop_training_03_w7`
at iteration 60, a 128x4 net; the corpus is meant to mirror the distribution a
real run puts in its buffers, and the cloud runs are 384x8. The records also
pick up the `root_outlook` field added since.

The engine itself did not change: every record of the previous corpus still
replayed clean on this engine, so this is a distribution refresh rather than a
correctness fix. The bot stratum's 20 games are the same 20 games as before
(identical actions, chance logs and digests) and differ only by that new field;
the two self-play strata are genuinely new games, since the net that played them
changed.

## History

* **2026-08-18** (`e39c7bd`) — rebuilt while fixing the checkpoint-rebuild path,
  50 games / 3,294 states. This file was not updated at the time and continued
  to describe the 2026-08-03 build until 2026-09-08.
* **2026-08-03** (`6ce6e2e`) — first codec-2 corpus, after the age-deal
  reordering (`ENGINE_AGE_DEAL_ORDERING.md`) moved the Age deal ahead of the
  start-player choice and so changed the chance stream of every game a seed
  produces.

  The corpus it replaced was codec-1. It kept passing after the reordering,
  because these tests drive *both* engines from the recorded action indices and
  so measure Rust-vs-Python parity rather than fidelity to the recorded game.
  But the trajectories it walked were ones the engine no longer produced, so the
  gate that runs on a rented box before training would have been checking parity
  over a distribution the cloud run will never generate. Records are cheap; a
  launch is not.
