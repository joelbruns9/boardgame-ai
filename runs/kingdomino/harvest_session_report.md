# Kingdomino noise-ball harvest session — 2026-07-07

Local-laptop session against a snapshot of cloud run5 (`runs/kingdomino/cloud_80x6_run5/`,
checkpoints iter_0001–0096 + nohup.log). Nothing was written to the cloud box;
its gate still owns its own `current_best.pt`.

**TL;DR: the checkpoint average TIES current_best (49.2%, pre-registered tie band
48–52%). Hypothesis (A) weight noise-ball wander is rejected as the explanation for
the flat gate checks; hypothesis (B) self-play data exhaustion for this net is
confirmed. Recommended next step: run6 = diversity package (option b).
Sims sweep: no knee up to 12800 — residual flips are near-tie reshuffles; advisor
should default to a flat 3200 sims (current web default of 50 is far too low).
Final 5e-5 verdict (sync through iter 115): checks tighten (sd 2.7→1.0) but mean
stays ~48% and brier_diag keeps rising at the 1e-4 rate — variance reduction
only, no strength; run6 recommendation unchanged.**

---

## 1. Checkpoint average (SWA / model soup)

`avg_0006_0090.pt` = float64 running mean of 85 checkpoints, run5 iters 6–90
(all post-promotion, pre-lr-drop), BN stats averaged, built with
`games.kingdomino.average_checkpoints`. Config carried from iter_0090.

## 2. Head-to-head: avg_0006_0090 (candidate) vs iter_0005 (current_best)

Paired seat-swapped open-loop match via `promotion.evaluate_checkpoint_match`
(runner: `runs/kingdomino/harvest_h2h.py`), 516 games @ sims=100 — same protocol
as run5's soft_gate checks. Seed 20260707, laptop GPU, 10.5 min.

| metric | value |
|---|---|
| W-L-D | 245-253-18 |
| points | 254 / 516 |
| win rate | **49.22%** |
| 95% CI | [44.9%, 53.5%] |
| LCB (z=1.96) | 44.9% |
| mean margin | +0.48 pts |

**Pre-registered interpretation: 48–52% → TIE.** The average of 85 noise-ball
samples is not stronger than the promoted checkpoint, so wander around the basin
center is not masking hidden gains. Per the decision rules: no sims=400
confirmation match, and the laptop's `best_checkpoint/current_best.pt` stays as
is (verified byte-identical to run5/iter_0005 by MD5).

Supporting context: the 18 gate checks (iters 10–95) have mean 48.2%, sd 2.6 —
almost exactly the 2.2% sampling sd expected from 516-game checks of a fixed
~48% net. The learner has sat a hair *below* current_best all run (mean 48.2% is
~3.4 standard errors below 50%), and averaging recovers only ~1 point of that.
Everything is consistent with (B): the 2400-sim self-play data distribution has
nothing further to teach this 80x6 net beyond iter_0005.

## 3. Sims sweep (advisor decision stability) — current_best

`games.kingdomino.sims_sweep`, defaults: 120 positions (13–14 per deck-size
bucket 8–40), rungs 400→12800, common-random-number determinizations, knee
threshold 5% on contested (|v|<0.5) top-move change rate. Full log:
`runs/kingdomino/sims_sweep.out.log`; per-position detail in
`sims_sweep_results.jsonl`.

| rungs | chg all | chg contested | chg decided | mean \|dv\| | ms/search |
|---|---|---|---|---|---|
| 400→800 | 9.2% | 8.6% | 25.0% | 0.022 | 359 |
| 800→1600 | 10.8% | 11.4% | 0.0% | 0.021 | 729 |
| 1600→3200 | 11.7% | 12.5% | 0.0% | 0.020 | 1471 |
| 3200→6400 | 17.5% | 17.1% | 22.2% | 0.018 | 2946 |
| 6400→12800 | 13.3% | 12.6% | 22.2% | 0.019 | 5956 |

By deck size (pooled change rate): 8: 15.4% · 12: 15.4% · 16: 7.7% ·
20: 13.8% · 24: 13.8% · 28: 6.2% · 32: 4.3% · 36: 17.1% · 40: 18.6% — no
monotone deck trend; the early game (deck 36–40, most hidden-deck variance) is
the least stable.

**No knee below 12800 at the 5% threshold** — contested change rate never drops,
and agreement with the deepest rung climbs steadily without plateau
(400: 65.8% → 1600: 69.2% → 3200: 75.8% → 6400: 86.7%).

A re-cut of the per-position detail shows what the residual flips are made of:
positions that flip have a much smaller top-2 visit gap at the higher rung
(median 0.05–0.21) than positions that hold (median 0.23–0.64), and the value
drift accompanying a flip is tiny (median |dv| ≈ 0.01–0.02) — i.e. each
doubling mostly re-picks among moves the search rates as near-equal EV, driven
by freshly sampled deck determinizations, not by discovering better moves. The
31/120 positions still unstable across 3200–12800 are concentrated where deck
uncertainty dominates; they do not converge at any tested budget.

**Advisor recommendation — flat 3200 sims default** (~1.5 s/query on the laptop
GPU), no deck-size schedule:

- The current web-app default (`nn_sims=50`) is far below every rung tested and
  should be raised regardless of where the ceiling is set.
- 3200 agrees with 12800 on 75.8% of positions; the disagreements sit in
  near-tie moves (|dv| ≈ 0.02), so their expected cost is small. Beyond 3200
  each doubling doubles latency to reshuffle ~13–17% of contested calls among
  near-equal options.
- A deck-size schedule is not supported by the data: instability is highest
  early (deck 36–40) but is deck-variance-driven and does not converge even at
  12800, so spending extra sims there buys little.
- Optional "deep" mode at 6400 (~3 s, 86.7% agreement with 12800) for critical
  decisions; the `le=5000` cap on `nn_sims` in web_app.py would need raising
  to expose it.

## 4. Preliminary 5e-5 read from the snapshot (iters 91–96)

Snapshot freezes at ~iteration 96 — only **one** post-drop gate check exists
(iter 95). **The pre-registered verdict at checks 100–110 must wait for the next
sync**; everything below is preliminary.

Gate-check series (516 games @ 100 sims vs current_best; from nohup.log):

| iter | wr% | action | | iter | wr% | action |
|---|---|---|---|---|---|---|
| 5 | **56.1** | **promote** | | 55 | 46.7 | revert |
| 10 | 50.7 | probation | | 60 | 48.7 | probation |
| 15 | 47.0 | revert | | 65 | 51.5 | probation |
| 20 | 50.2 | probation | | 70 | 47.5 | revert |
| 25 | 47.2 | revert | | 75 | 50.9 | probation |
| 30 | 49.9 | probation | | 80 | 46.6 | revert |
| 35 | 48.5 | probation | | 85 | 52.9 | probation |
| 40 | 45.6 | revert | | 90 | 44.0 | revert |
| 45 | 50.3 | probation | | 95 | 46.6 | revert |
| 50 | 43.4 | revert | | | | |

- The iter 85→90 swing (52.9% → 44.0%) spans 8.9 points; with per-check sampling
  sd ≈ 2.2% a swing that size is ~2.9 sd of the *difference* — large but not
  outlandish for 18 draws, and the series shows no trend (first half mean 48.1%,
  second half 48.4%).
- Single post-drop check, iter 95: 46.6% — indistinguishable from the 1e-4-era
  distribution. Criterion (i) (checks tightening into a 47–51 band) cannot be
  assessed from one point.
- **brier_diag** (per-iteration diag from nohup.log) rose 0.119 (iters 6–15 mean)
  → 0.207 (iters 86–90 mean) under 1e-4. Window means:

  | iters | 6–15 | 16–30 | 31–45 | 46–60 | 61–75 | 76–85 | 86–90 | 91–96 |
  |---|---|---|---|---|---|---|---|---|
  | brier_diag | 0.119 | 0.147 | 0.179 | 0.184 | 0.198 | 0.203 | 0.207 | 0.207 |

  Late-1e-4 slope ≈ +0.0005/iter (76–85 → 86–90 window centers); post-drop slope
  ≈ +0.00005/iter (86–90 → 91–96) — a ~10× flattening, i.e. criterion (ii)
  (brier_diag stops rising) **tentatively holds**, but 6 iterations against
  ±0.01 per-iteration noise is weak evidence.
- Criterion (iii) (any check ≥ 53%): not met (iter 95 = 46.6%).

Preliminary read: 5e-5 has plausibly stopped the value-head drift (ii) but shows
no strength signal yet (i, iii). Re-run this section against nohup.log after the
next sync covering checks 100–110.

### UPDATE (sync through iter 115, received 2026-07-07) — final verdict

The next sync arrived same-day (iters 90–115 pasted from the box's log; checks
at 100/105/110/115). Pre-registered criteria, final scoring:

Post-drop checks: 95: 46.6% (revert) · 100: 47.9% (revert) · 105: 46.9%
(revert) · 110: 48.7% (probation) · 115: 48.9% (probation).

- **(i) checks tighten into 47–51 — MARGINAL.** The five post-drop checks span
  46.6–48.9 (sd 1.03) vs the 1e-4 era's 43.4–52.9 (sd 2.66). The tightening is
  real-looking but not conclusive (P(sd ≤ 1.03 from pure 2.2% sampling noise,
  n=5) ≈ 0.07), two of five checks sit below the band's 47 floor, and the mean
  did not move: 47.8% post-drop vs 48.3% before. Tighter, not stronger.
- **(ii) brier_diag stops rising — FAILS.** With all 25 post-drop iterations,
  the OLS slope over iters 91–115 is **+0.00053/iter — the same as the late
  1e-4 rate**. Window means: 0.207 (91–96) → 0.208 (97–102) → 0.216 (103–108)
  → 0.215 (109–115). The 10× flattening reported above from iters 91–96 alone
  was noise. Train-side value metrics agree: win loss drifts 0.244 → ~0.256
  and train brier 0.073 → 0.078 over the same span — the value head keeps
  churning at half the lr.
- **(iii) any check ≥ 53% — FAILS.** Max post-drop check is 48.9%.

**Verdict: the lr drop reduced the amplitude of the wander (smaller steps ⇒
smaller noise ball) but changed nothing else — drift continues at the same
rate and strength is flat at ~48%.** This closes the loop with the H2H tie:
neither averaging the noise ball (§2) nor shrinking it (this section) yields
strength, so the plateau is not an optimizer artifact at either scale. The
data-exhaustion diagnosis and the run6 recommendation (§5) stand unchanged,
including starting run6 at 1e-4: 5e-5 demonstrably buys variance reduction
only, which is not what a fresh data distribution needs early.

## 5. Recommended next step

**(b) run6 = diversity package on the 80x6.** The H2H tie rules out (a) —
averaging harvests nothing here because there is nothing hidden to harvest; the
bottleneck is the data, not the optimizer noise. Existing flags only:

```
--hof_fraction 0.2 --hof_start_iter 1 --temp_moves 30 --endgame_oversample 1.0
```

lr per the 5e-5 evidence: the evidence is preliminary (one post-drop check).
brier_diag flattening suggests 5e-5 curbs value-head drift, but there is no sign
it *adds* strength. Recommendation: start run6 at **1e-4** (fresh, more diverse
data distribution will need the larger step early) with the same decay to 5e-5
once its gate checks go flat — revisit if the next run5 sync shows checks
100–110 tightening into 47–51 at 5e-5 (criterion i), which would justify
dropping earlier.

Checkpoint averaging itself remains cheap and safe (it tied, cost ~1 min CPU +
one gate-length match); worth re-testing on run6 once *its* checks flatten, but
it is not standard practice yet on this evidence.

---

### Session artifacts

- `runs/kingdomino/cloud_80x6_run5/avg_0006_0090.pt` — 85-checkpoint average (not committed)
- `runs/kingdomino/harvest_h2h.py` + `harvest_h2h_result.json` — H2H runner/result
- `runs/kingdomino/sims_sweep_results.jsonl` + `sims_sweep.out.log` — sweep detail
