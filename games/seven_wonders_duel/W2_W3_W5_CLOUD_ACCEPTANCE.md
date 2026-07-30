# W5 cloud checks and W2/W3 operational telemetry

W2 is complete. The laptop established checked allocation failures, bounded
host-memory behavior, cleanup, and restart recovery. L paths also ran on the
laptop's 8 GB GPU, so the RTX 5090's 32 GB VRAM does not justify a separate
memory-hardening project.

The cloud host still records RSS and physical/allocated VRAM. Those readings
are operational telemetry and a launch sanity check, not deferred W2
acceptance. W3 is also complete; production rows validate its statistics on the
real workload. Production gate timing must be collected on the RTX 5090;
laptop measurements do not select production parameters.

## Before training

Copy one W0 L checkpoint to the cloud host. Its playing strength is irrelevant
to gate timing; its `384x8x6` tensor shapes are required.

Run:

```bash
./games/seven_wonders_duel/run_w2_w3_w5_cloud_acceptance.sh \
  /path/to/sweep_L_lr5e-05_seed0.pt \
  /path/to/production_run \
  /path/to/acceptance
```

This measures 200-, 400-, and 800-game L/bf16 gates on the target hardware and
writes the fixed-cost/per-game fit. Do not use laptop timing or an S checkpoint
to choose `--gate-max-games`. If a representative ungated iteration time is
known, pass it as the fourth argument; the report recommends the largest
measured cap whose projected gate share is at most 10%.

Before committing to an unattended run, execute one short iteration at the
exact L/bf16 launch geometry and inspect its RSS, physical VRAM, and allocated
VRAM fields. This is an operational smoke check for configuration mistakes,
not an additional W2 soak or acceptance gate.

The launch configuration is:

- `--d-model 384 --layers 8 --heads 6`
- `--precision bf16`
- `--learning-rate 5e-5`
- `--hof-opponent-fraction 0.15 --hof-start-games 10000`
- `--promotion-min-lcb 0.50 --revert-win-rate 0.48`
- `--anchor-games 200`
- the measured gate cap and box-specific batching/slot values
- explicit `--memory-budget-gb`, `--vram-budget-gb`, and
  `--memory-headroom-gb` resolved from this host

Record the resolved values in the run manifest. A production launch must not
inherit the provisional local defaults accidentally.

## During and after training

Run the same command at any time to regenerate `az_report.json` and
`az_report.md`. Once 60 iterations exist, it also validates that the fitted RSS
change across the final 30 iterations is at most 5%.

Required cloud evidence:

1. Every row is schema v2 and records model `384x8x6`, bf16, opponent mix,
   scheduler rows/batches, cache bytes, RSS, and physical versus allocated VRAM.
2. Realized HOF share, `(hof + hof_bot) / generated games`, converges to 0.15
   after `hof_start_games`. Bot-shadowed nominal assignments are recorded as
   unused and do not count as HOF traffic.
3. No gate reports a channel-disconnect surrogate for a Python/CUDA failure.
4. Gate timing uses at least three sizes and records moves per game.
5. Promotion decisions use independent seat pairs and are **fixed-N**: every
   gate row's `evaluated_games` equals the rung it was issued, `fixed_n` is
   true, and `stop_reason` is one of `promotion_lcb` / `revert_ucb` /
   `probation` / `revert_suppressed_knot`. A short row means the sequential
   rule has come back. Anchors stay fixed-N measurements.
6. Gate rows carry `pair_scores`, so any gate can be re-decided offline under a
   different threshold or rung without replaying games.
7. The ladder moves: `control_state.gate_rung` rises after two consecutive
   probations and falls after a promotion.
8. The last-30-iteration RSS fitted drift check remains green as an operational
   leak alert. A failure pauses the affected run for diagnosis; it does not
   retroactively reopen W2.

The separate `laptop_training_03` 70→90 continuation remains an S-model
regression test for resume and memory cleanup. It is not evidence for L sizing,
throughput, stability, or the production gate cap.
