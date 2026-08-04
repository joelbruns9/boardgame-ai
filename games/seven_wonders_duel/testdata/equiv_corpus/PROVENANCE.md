# Equivalence corpus — provenance

The 50 games the engine-equivalence gate runs on (W6.2). Committed rather than
generated on demand because `runs/` is gitignored: a corpus that only exists on
the gate box makes the cloud smoke pass vacuously, which is worse than no smoke.

Rebuild with `python -m games.seven_wonders_duel.build_equiv_corpus`; that
script is the authority on how these files are made and why each stratum exists.

## Current corpus

**Generated 2026-08-03 under `SPEC_VERSION` codec-2** — the first version after
the age-deal reordering (`ENGINE_AGE_DEAL_ORDERING.md`), which moved the Age deal
ahead of the start-player choice and so changed the chance stream of every game a
seed produces.

| file | games | moves | victory mix |
|---|---:|---:|---|
| `curriculum_seed.jsonl` | 20 | 1,146 | 12 scientific / 8 military |
| `selfplay_early.jsonl` | 15 | 1,024 | 13 civilian / 2 military |
| `selfplay_late.jsonl` | 15 | 1,053 | 11 civilian / 3 scientific / 1 military |

3,273 encoded states, covering all 9 decision branches and all 9 token types.
Every record replays clean through `buffer.replay` on this engine.

The late stratum came from `runs/laptop_training_03_w7/checkpoints/
current_best.pt` (128x4, run 03's promoted best), the early stratum from an
untrained net seeded at 20260803, both at Phase D's default sims. The bot
stratum is `phase_d._bot_seed_game`, i.e. literally the seed step of a run.

## Why it was regenerated

The previous corpus was codec-1. It kept passing after the reordering, because
these tests drive *both* engines from the recorded action indices and so measure
Rust-vs-Python parity rather than fidelity to the recorded game. But the
trajectories it walked were ones this engine no longer produces, so the gate that
runs on a rented box before training would have been checking parity over a
distribution the cloud run will never generate — including none of the new
start-of-Age ordering. Records are cheap; a launch is not.
