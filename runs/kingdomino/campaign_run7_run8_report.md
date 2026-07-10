# Campaign report: run7 → run8/8b (2026-07-08 … 07-10)

Goal (FABLE_PROMPT_RUN7): squeeze remaining strength out of the 80x6 net.
Outcome: **+~55 Elo banked over the run5 net**, a training loop rebuilt from
silently-degrading to self-healing, and a clean localization of the remaining
ceiling. Canonical `best_checkpoint/current_best.pt` = run8's promoted
rolling average (sha `4bf07b…`).

## The measured chain (every link a 1500–2500-game paired match @300 sims)

| link | result |
|---|---|
| run8-avg vs run7-peak | 52.1% / LCB 50.1 (in-run gate, iter 25) |
| run7-peak vs run5 iter_0005 | 53.1% / LCB 51.2 (certification) |
| run7-peak vs run3_iter80 (off-lineage) | 54.9% / LCB 52.4 (certification) |
| run6 detour | best-of-run6 (iter_0025) chosen by 3-way round-robin; certification later showed the lineage had NOT confidently advanced past run5 until run7 |

## What was fixed, with evidence

1. **HOF label contamination (run7)** — run6 recorded HOF-game moves at 400
   sims (~21% of buffer). Fixed with the per-seat engine override (frontier
   at full sims + playout-cap, opponent shallow, opponent never recorded).
   run7 promoted at its first meaningful gate (55.1%/53.2).
2. **Replay-ratio overfit drift (run8)** — run7 gave back its gain
   (55→48→44 over 10 iters; brier_diag 0.10→0.25; ratio ~32 samples/example).
   train_steps 1000→300 (ratio ~10) slowed drift ~4x; rolling-average gating
   (k=8) removed snapshot noise/winner's curse (snapshots measured 43-44% vs
   the peak while their average measured 48.9%); learner reset after 2
   consecutive reverts made divergence recoverable. run8b executed the full
   self-healing cycle live: drift (50.5→47.4) → revert → reset → recovery
   to 51.6 next gate.
3. **Ruled out**: capacity (bake-off: 80x6 matched 96x6/80x10 on held-out
   fit — the smallest net won), gate-sims proxy (44% at 300 sims stayed 44%
   at 1600), search depth (earlier sims sweep).

## Verdict and run9 thesis

Three independent trajectories (run8, run8b, run8b-post-reset) equilibrated
at 49–51% vs the banked average across 8 gates. With loop, capacity, and
search ruled out, the ceiling is the **position distribution**: self-play
from one policy family vs near-clone opponents has finite signal. Run9
(launched 2026-07-10, seed 9) attacks exactly that: uniformly-random
unrecorded opening plies (k~U[2,8], half of games) + HOF value-personalities
(per-opponent alpha ∈ {0,1}) + recency-weighted pool. Pre-registered: resumed
promotions by gate ~55–65 confirm the thesis; parity again means 80x6
self-play is at its practical ceiling → BGA-seeded starts (run10) or ship.

## Ops learned the hard way

- SIGINT cannot reliably stop the trainer (main thread lives in GIL-released
  Rust; can block on the solver channel). Use `touch <run_dir>/STOP`;
  `--buffer_autosave_every 10` bounds buffer loss on hard kills.
- Rented-disk artifacts get downloaded the day they're minted (credit
  cutoff killed run8 mid-iteration 49; disk survived by luck).
- Kingdomino does not alternate turns (pick order) — bit both the swindle
  analyzer and any per-player accounting; now encoded in code comments.

## Artifact inventory (laptop)

- `best_checkpoint/current_best.pt` = run8 avg `4bf07b` (canonical best;
  advisor autodiscovers it)
- `cloud_80x6_run8b/` = full run8b checkpoint set + logs (incl. the harvest
  file it inherited)
- run7 peak `36abe…`: superseded, still box-side (`cloud_80x6_run7/`,
  `hof_run7/`) — bring over with the next sync
- Advisor gained along the way: exact margins in points, swindle mode,
  auto-refresh, passive BGA game logger, sims to 12800
